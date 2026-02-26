/* eslint-disable */
// @ts-nocheck — benchmark file uses random.split() indexing which returns jax-js Array, not TS array
/**
 * TTS (Pocket TTS) per-step benchmark.
 *
 * Replicates the model structure from website/src/routes/tts/pocket-tts.ts
 * with random weights to measure per-step inference latency.
 *
 * Model dimensions (Kyutai Pocket TTS):
 *   FlowLM: dim=1024, ldim=32, numHeads=16, headDim=64, ffnDim=4096, 6 layers
 *   FlowNet: 6 ResBlocks (256-wide MLP), timestep embedder
 *   Mimi decoder: dim=512, numHeads=8, headDim=64, 8 layers, SEANet decoder
 *
 * Real-time target: 1920 audio samples per step at 24kHz = 80ms/step.
 */
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  nn,
  numpy as np,
  random,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

// ── Helpers ──────────────────────────────────────────────────────────────────

type Linear = { weight: np.Array; bias?: np.Array };
type LayerNorm = { weight: np.Array; bias: np.Array };

function makeLinear(
  key: np.Array,
  inDim: number,
  outDim: number,
  bias = true,
  dtype = np.float16,
): Linear {
  const w = random.normal(key, [outDim, inDim]).astype(dtype);
  if (!bias) return { weight: w };
  const b = np.zeros([outDim], { dtype });
  return { weight: w, bias: b };
}

function makeLayerNorm(dim: number, dtype = np.float16): LayerNorm {
  return {
    weight: np.ones([dim], { dtype }),
    bias: np.zeros([dim], { dtype }),
  };
}

function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
}

function _runLayerNorm(
  { weight, bias }: Partial<LayerNorm>,
  x: np.Array,
  eps: number = 1e-5,
) {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const mean = x.mean(-1, { keepdims: true });
  const var_ = np.var_(x, -1, { mean, correction: 0, keepdims: true });
  x = x.sub(mean).div(np.sqrt(var_.add(eps)));
  if (weight) x = x.mul(weight).add(bias!);
  return x.astype(dtype);
}
let runLayerNorm: typeof _runLayerNorm;
function initJitFunctions() {
  runLayerNorm = jit(_runLayerNorm, { staticArgnums: [2] });
  runTransformerLayer = jit(_runTransformerLayer, { staticArgnums: [5, 6] });
  runFlowNet = jit(_runFlowNet);
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

function runRope(
  q: np.Array,
  k: np.Array,
  offset: np.Array,
  maxPeriod = 10000,
): [np.Array, np.Array] {
  const [T, H, D] = q.shape;
  const halfD = D / 2;
  const ds = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const freqs = np.exp(ds.mul((-Math.log(maxPeriod) * 2) / D));
  const ts = np.arange(T).add(offset).astype(np.float32).reshape([T, 1, 1]);
  const qReshaped = q.reshape([T, H, halfD, 2]);
  const kReshaped = k.reshape([T, H, halfD, 2]);
  let [qr, qi] = np.split(qReshaped, 2, -1);
  let [kr, ki] = np.split(kReshaped, 2, -1);
  qr = np.squeeze(qr, -1);
  qi = np.squeeze(qi, -1);
  kr = np.squeeze(kr, -1);
  ki = np.squeeze(ki, -1);
  const angles = freqs.mul(ts);
  const rotr = np.cos(angles).astype(qr.dtype);
  const roti = np.sin(angles).astype(qr.dtype);
  const qor = qr.mul(rotr).sub(qi.mul(roti));
  const qoi = qr.mul(roti).add(qi.mul(rotr));
  const kor = kr.mul(rotr).sub(ki.mul(roti));
  const koi = kr.mul(roti).add(ki.mul(rotr));
  const qo = np.stack([qor, qoi], -1).reshape([T, H, D]);
  const ko = np.stack([kor, koi], -1).reshape([T, H, D]);
  return [qo, ko];
}

// ── Streaming Multihead Attention (decode step only) ─────────────────────────

type KVCache = { key: np.Array; value: np.Array };
type MHA = { inProj: Linear; outProj: Linear };

function makeMHA(key: np.Array, dim: number, dtype = np.float16): MHA {
  const [k1, k2] = random.split(key) as [np.Array, np.Array];
  return {
    inProj: makeLinear(k1, dim, 3 * dim, false, dtype),
    outProj: makeLinear(k2, dim, dim, false, dtype),
  };
}

function runMHADecode(
  { inProj, outProj }: MHA,
  kvCache: KVCache,
  query: np.Array,
  offset: np.Array,
  kvCacheLen: np.Array,
  numHeads: number,
  context: number = 0,
): [np.Array, KVCache] {
  const [T, embedDim] = query.shape;
  const headDim = embedDim / numHeads;
  const projected = runLinear(inProj, query);
  const qkv = projected.reshape([T, 3 * numHeads, headDim]);
  const [q_, k_, v] = np.split(qkv, 3, 1);
  const [q, k] = runRope(q_, k_, offset);

  // Decode step — update KV cache and attend
  const capacity = kvCache.key.shape[0];
  const cacheMask = np.arange(capacity).reshape([-1, 1, 1]).less(kvCacheLen);
  kvCache.key = np.where(
    cacheMask,
    kvCache.key,
    np.tile(k, [capacity / T, 1, 1]),
  );
  kvCache.value = np.where(
    cacheMask,
    kvCache.value,
    np.tile(v, [capacity / T, 1, 1]),
  );
  const maskDelta = np
    .arange(capacity)
    .sub(np.arange(T).reshape([T, 1]))
    .sub(kvCacheLen);
  const mask = context
    ? maskDelta.lessEqual(0).mul(maskDelta.greater(-context))
    : maskDelta.lessEqual(0);
  let x = nn.dotProductAttention(q, kvCache.key, kvCache.value, { mask });
  x = x.reshape([T, embedDim]);
  x = runLinear(outProj, x);
  return [x, kvCache];
}

// ── Streaming Transformer Layer ──────────────────────────────────────────────

type TransformerLayer = {
  selfAttn: MHA;
  norm1: LayerNorm;
  norm2: LayerNorm;
  linear1: Linear;
  linear2: Linear;
  layerScale1?: np.Array;
  layerScale2?: np.Array;
};

function makeTransformerLayer(
  key: np.Array,
  dim: number,
  ffnDim: number,
  withLayerScale = false,
  dtype = np.float16,
): TransformerLayer {
  const keys = random.split(key, 4) as any;
  const result: TransformerLayer = {
    selfAttn: makeMHA(keys[0], dim, dtype),
    norm1: makeLayerNorm(dim, dtype),
    norm2: makeLayerNorm(dim, dtype),
    linear1: makeLinear(keys[1], dim, ffnDim, false, dtype),
    linear2: makeLinear(keys[2], ffnDim, dim, false, dtype),
  };
  if (withLayerScale) {
    result.layerScale1 = np.ones([dim], { dtype });
    result.layerScale2 = np.ones([dim], { dtype });
  }
  return result;
}

function _runTransformerLayer(
  layer: TransformerLayer,
  kvCache: KVCache,
  x: np.Array,
  offset: np.Array,
  kvCacheLen: np.Array,
  numHeads: number,
  context: number,
): [np.Array, KVCache] {
  const { selfAttn, norm1, norm2, linear1, linear2, layerScale1, layerScale2 } =
    layer;
  const xOrig = x;
  x = runLayerNorm(norm1, x);
  let update: np.Array;
  [update, kvCache] = runMHADecode(
    selfAttn,
    kvCache,
    x,
    offset,
    kvCacheLen,
    numHeads,
    context,
  );
  if (layerScale1) update = update.mul(layerScale1);
  x = xOrig.add(update);

  const xOrig2 = x;
  x = runLayerNorm(norm2, x);
  let ffnOut = runLinear(linear1, x);
  ffnOut = nn.gelu(ffnOut, { approximate: false });
  ffnOut = runLinear(linear2, ffnOut);
  if (layerScale2) ffnOut = ffnOut.mul(layerScale2);
  x = xOrig2.add(ffnOut);
  return [x, kvCache];
}
let runTransformerLayer: typeof _runTransformerLayer;

// ── FlowNet (SimpleMLPAdaLN) ─────────────────────────────────────────────────

type ResBlock = {
  inLN: LayerNorm;
  mlp: [Linear, undefined, Linear];
  adaLNModulation: [undefined, Linear];
};

type FlowNet = {
  inputProj: Linear;
  condEmbed: Linear;
  timeEmbed: {
    mlp: [Linear, undefined, Linear, { alpha: np.Array }];
    freqs: np.Array;
  }[];
  resBlocks: ResBlock[];
  finalLayer: { linear: Linear; adaLNModulation: [undefined, Linear] };
};

function makeFlowNet(
  key: np.Array,
  condDim: number,
  flowDim: number,
  dtype = np.float16,
): FlowNet {
  const keys = random.split(key, 20) as any;
  let ki = 0;

  const makeTimestepEmbedder = (k: np.Array) => ({
    mlp: [
      makeLinear(
        (random.split(k) as any as any[])[0],
        256,
        flowDim,
        true,
        dtype,
      ),
      undefined as undefined,
      makeLinear(
        (random.split(k) as any as any[])[1],
        flowDim,
        flowDim,
        true,
        dtype,
      ),
      { alpha: np.ones([flowDim], { dtype }) },
    ] as [Linear, undefined, Linear, { alpha: np.Array }],
    freqs: random
      .normal((random.split(k) as any as any[])[0], [128])
      .astype(dtype),
  });

  const makeResBlock = (k: np.Array): ResBlock => {
    const rkeys = random.split(k, 4) as any;
    return {
      inLN: makeLayerNorm(flowDim, dtype),
      mlp: [
        makeLinear(rkeys[0], flowDim, flowDim, true, dtype),
        undefined as undefined,
        makeLinear(rkeys[1], flowDim, flowDim, true, dtype),
      ],
      adaLNModulation: [
        undefined as undefined,
        makeLinear(rkeys[2], flowDim, 3 * flowDim, true, dtype),
      ],
    };
  };

  return {
    inputProj: makeLinear(keys[ki++], 32, flowDim, true, dtype),
    condEmbed: makeLinear(keys[ki++], condDim, flowDim, true, dtype),
    timeEmbed: [
      makeTimestepEmbedder(keys[ki++]),
      makeTimestepEmbedder(keys[ki++]),
    ],
    resBlocks: Array.from({ length: 6 }, () => makeResBlock(keys[ki++])),
    finalLayer: {
      linear: makeLinear(keys[ki++], flowDim, 32, true, dtype),
      adaLNModulation: [
        undefined as undefined,
        makeLinear(keys[ki++], flowDim, 2 * flowDim, true, dtype),
      ],
    },
  };
}

function runTimestepEmbedder(
  {
    mlp,
    freqs,
  }: { mlp: [Linear, undefined, Linear, { alpha: np.Array }]; freqs: np.Array },
  t: np.Array,
): np.Array {
  const [linear1, , linear2, rmsNorm] = mlp;
  const args = t.mul(freqs);
  const embedding = np.concatenate([np.cos(args), np.sin(args)], -1);
  let x = runLinear(linear1, embedding);
  x = nn.silu(x);
  x = runLinear(linear2, x);
  // RMSNorm
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const var_ = np.var_(x, -1, { correction: 0, keepdims: true });
  x = x.mul(rmsNorm.alpha).div(np.sqrt(var_.add(1e-5)));
  return x.astype(dtype);
}

function _runFlowNet(
  net: FlowNet,
  c: np.Array,
  s: np.Array,
  t: np.Array,
  x: np.Array,
): np.Array {
  x = runLinear(net.inputProj, x);
  const sEmb = runTimestepEmbedder(net.timeEmbed[0], s);
  const tEmb = runTimestepEmbedder(net.timeEmbed[1], t);
  const tCombined = sEmb.add(tEmb).div(2);
  const cEmb = runLinear(net.condEmbed, c);
  const y = tCombined.add(cEmb);

  for (const block of net.resBlocks) {
    const [, adaLNLinear] = block.adaLNModulation;
    const modulation = runLinear(adaLNLinear, nn.silu(y));
    const [shiftMlp, scaleMlp, gateMlp] = np.split(modulation, 3, -1);
    let h = runLayerNorm(block.inLN, x, 1e-6);
    h = h.mul(scaleMlp.add(1)).add(shiftMlp);
    const [mlpLinear1, , mlpLinear2] = block.mlp;
    h = runLinear(mlpLinear1, h);
    h = nn.silu(h);
    h = runLinear(mlpLinear2, h);
    x = x.add(gateMlp.mul(h));
  }

  const [, finalAdaLNLinear] = net.finalLayer.adaLNModulation;
  const finalMod = runLinear(finalAdaLNLinear, nn.silu(y));
  const [shift, scale] = np.split(finalMod, 2, -1);
  x = runLayerNorm({}, x, 1e-6);
  x = x.mul(scale.add(1)).add(shift);
  x = runLinear(net.finalLayer.linear, x);
  return x;
}
let runFlowNet: typeof _runFlowNet;

// ── Full FlowLM decode step ─────────────────────────────────────────────────

type FlowLMModel = {
  inputLinear: Linear;
  outNorm: LayerNorm;
  outEos: Linear;
  flowNet: FlowNet;
  transformer: TransformerLayer[];
};

function makeFlowLMModel(key: np.Array, dtype = np.float16): FlowLMModel {
  const keys = random.split(key, 10) as any;
  return {
    inputLinear: makeLinear(keys[0], 32, 1024, true, dtype),
    outNorm: makeLayerNorm(1024, dtype),
    outEos: makeLinear(keys[1], 1024, 1, true, dtype),
    flowNet: makeFlowNet(keys[2], 1024, 1024, dtype),
    transformer: Array.from({ length: 6 }, (_, i) =>
      makeTransformerLayer(keys[3 + i], 1024, 4096, false, dtype),
    ),
  };
}

// ── Mimi decoder (simplified) ────────────────────────────────────────────────

type MimiDecoder = {
  quantizerWeight: np.Array; // [512, 32, 1]
  upsampleWeight: np.Array; // conv transpose [512, 1, 33] depthwise
  decoderTransformer: TransformerLayer[];
  // SEANet decoder conv weights (simplified)
  seanetConv1: Linear; // initial 1x1-ish conv
  seanetConvFinal: Linear; // final conv
};

function makeMimiDecoder(key: np.Array, dtype = np.float16): MimiDecoder {
  const keys = random.split(key, 15) as any;
  return {
    quantizerWeight: random.normal(keys[0], [512, 32, 1]).astype(dtype),
    upsampleWeight: random.normal(keys[1], [512, 1, 33]).astype(dtype),
    decoderTransformer: Array.from({ length: 8 }, (_, i) =>
      makeTransformerLayer(keys[2 + i], 512, 2048, true, dtype),
    ),
    seanetConv1: makeLinear(keys[10], 512, 512, true, dtype),
    seanetConvFinal: makeLinear(keys[11], 512, 1, true, dtype),
  };
}

// ── Benchmark suite ──────────────────────────────────────────────────────────

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))("TTS per-step", async () => {
  defaultDevice("webgpu");
  initJitFunctions();

  const masterKey = random.key(42);
  const keys = random.split(masterKey, 10) as any;

  // Build random model
  const flowLM = makeFlowLMModel(keys[0]);
  const mimiDecoder = makeMimiDecoder(keys[1]);
  await blockUntilReady(tree.leaves(flowLM));
  await blockUntilReady(tree.leaves(mimiDecoder));

  // Pre-allocate KV caches for FlowLM (capacity 128, simulating mid-generation)
  const flowLMKVCaches: KVCache[] = flowLM.transformer.map(() => ({
    key: random.normal(random.key(0), [128, 16, 64]).astype(np.float16),
    value: random.normal(random.key(1), [128, 16, 64]).astype(np.float16),
  }));
  const flowLMKvCacheLen = 64; // simulate mid-generation

  // Pre-allocate KV caches for Mimi decoder
  const mimiKVCaches: KVCache[] = mimiDecoder.decoderTransformer.map(() => ({
    key: random.normal(random.key(2), [272, 8, 64]).astype(np.float16),
    value: random.normal(random.key(3), [272, 8, 64]).astype(np.float16),
  }));
  const mimiKvCacheLen = 128;

  await blockUntilReady([
    ...flowLMKVCaches.flatMap((c) => [c.key, c.value]),
    ...mimiKVCaches.flatMap((c) => [c.key, c.value]),
  ]);

  afterAll(() => {
    tree.dispose(flowLM);
    tree.dispose(mimiDecoder);
    tree.dispose(flowLMKVCaches);
    tree.dispose(mimiKVCaches);
  });

  // ── FlowLM transformer decode step (6 layers) ───────────────────────────
  bench(
    "FlowLM transformer (6 layers, T=1, dim=1024)",
    async () => {
      let x = random.normal(random.key(99), [1, 1024]).astype(np.float16);
      const offset = np.array(flowLMKvCacheLen, { dtype: np.int32 });
      const kvLen = np.array(flowLMKvCacheLen, { dtype: np.int32 });
      for (let i = 0; i < 6; i++) {
        const [newX, _kv] = runTransformerLayer(
          flowLM.transformer[i],
          flowLMKVCaches[i],
          x,
          offset,
          kvLen,
          16,
          0,
        );
        x.dispose();
        x = newX;
      }
      await blockUntilReady(x);
      x.dispose();
      offset.dispose();
      kvLen.dispose();
    },
    { iterations: 20 },
  );

  // ── FlowNet (SimpleMLPAdaLN) ────────────────────────────────────────────
  bench(
    "FlowNet (6 ResBlocks, dim=1024)",
    async () => {
      using c = random.normal(random.key(10), [1, 1024]).astype(np.float16);
      using s = np.full([1, 1], 0.0);
      using t = np.full([1, 1], 1.0);
      using x = random.normal(random.key(11), [1, 32]).astype(np.float16);
      const result = runFlowNet(flowLM.flowNet, c, s, t, x);
      await blockUntilReady(result);
      result.dispose();
    },
    { iterations: 20 },
  );

  // ── Mimi decoder transformer (8 layers, T=16) ──────────────────────────
  bench(
    "Mimi transformer (8 layers, T=16, dim=512)",
    async () => {
      let x = random.normal(random.key(20), [16, 512]).astype(np.float16);
      const offset = np.array(mimiKvCacheLen, { dtype: np.int32 });
      const kvLen = np.array(mimiKvCacheLen, { dtype: np.int32 });
      for (let i = 0; i < 8; i++) {
        const [newX, _kv] = runTransformerLayer(
          mimiDecoder.decoderTransformer[i],
          mimiKVCaches[i],
          x,
          offset,
          kvLen,
          8,
          250,
        );
        x.dispose();
        x = newX;
      }
      await blockUntilReady(x);
      x.dispose();
      offset.dispose();
      kvLen.dispose();
    },
    { iterations: 20 },
  );

  // ── Single attention layer (FlowLM-sized, decode) ──────────────────────
  bench(
    "Single attention (T=1, S=64, dim=1024, 16 heads)",
    async () => {
      using q = random.normal(random.key(30), [1, 1024]).astype(np.float16);
      const offset = np.array(flowLMKvCacheLen, { dtype: np.int32 });
      const kvLen = np.array(flowLMKvCacheLen, { dtype: np.int32 });
      const [result, _kv] = runMHADecode(
        flowLM.transformer[0].selfAttn,
        flowLMKVCaches[0],
        q,
        offset,
        kvLen,
        16,
        0,
      );
      await blockUntilReady(result);
      result.dispose();
      offset.dispose();
      kvLen.dispose();
    },
    { iterations: 30 },
  );

  // ── Single matmul operations (the core bottleneck) ─────────────────────
  bench(
    "matmul [1,1024] x [1024,4096] (FFN up)",
    async () => {
      using a = random.normal(random.key(40), [1, 1024]).astype(np.float16);
      using b = random.normal(random.key(41), [1024, 4096]).astype(np.float16);
      const c = np.dot(a, b);
      await blockUntilReady(c);
      c.dispose();
    },
    { iterations: 50 },
  );

  bench(
    "matmul [1,4096] x [4096,1024] (FFN down)",
    async () => {
      using a = random.normal(random.key(42), [1, 4096]).astype(np.float16);
      using b = random.normal(random.key(43), [4096, 1024]).astype(np.float16);
      const c = np.dot(a, b);
      await blockUntilReady(c);
      c.dispose();
    },
    { iterations: 50 },
  );

  // ── Attention einsum (the slow path) ───────────────────────────────────
  bench(
    "einsum BLNH,BSNH->BNLS (scores, B=1,L=1,S=64,N=16,H=64)",
    async () => {
      using q = random
        .normal(random.key(50), [1, 1, 16, 64])
        .astype(np.float16);
      using k = random
        .normal(random.key(51), [1, 64, 16, 64])
        .astype(np.float16);
      const scores = np.einsum("BLNH,BSNH->BNLS", q, k);
      await blockUntilReady(scores);
      scores.dispose();
    },
    { iterations: 50 },
  );

  bench(
    "einsum BNLS,BSNH->BLNH (weighted values, B=1,L=1,S=64,N=16,H=64)",
    async () => {
      using attn = random
        .normal(random.key(52), [1, 16, 1, 64])
        .astype(np.float16);
      using v = random
        .normal(random.key(53), [1, 64, 16, 64])
        .astype(np.float16);
      const out = np.einsum("BNLS,BSNH->BLNH", attn, v);
      await blockUntilReady(out);
      out.dispose();
    },
    { iterations: 50 },
  );

  // ── Full step estimate (FlowLM + Mimi) ────────────────────────────────
  bench(
    "Full TTS step (FlowLM + FlowNet only)",
    async () => {
      // FlowLM transformer
      let x = random.normal(random.key(99), [1, 1024]).astype(np.float16);
      const offset = np.array(flowLMKvCacheLen, { dtype: np.int32 });
      const kvLen = np.array(flowLMKvCacheLen, { dtype: np.int32 });
      for (let i = 0; i < 6; i++) {
        const [newX, _kv] = runTransformerLayer(
          flowLM.transformer[i],
          flowLMKVCaches[i],
          x,
          offset,
          kvLen,
          16,
          0,
        );
        x.dispose();
        x = newX;
      }
      // LayerNorm + EOS check
      using normed = runLayerNorm(flowLM.outNorm, x);
      x.dispose();
      using transformerOut = normed.slice([-1]);

      // FlowNet (LSD decode with 1 step)
      using s = np.full([1, 1], 0.0);
      using t = np.full([1, 1], 1.0);
      using noise = random.normal(random.key(100), [1, 32]).astype(np.float16);
      const latent = runFlowNet(flowLM.flowNet, transformerOut, s, t, noise);
      await blockUntilReady(latent);
      latent.dispose();
      offset.dispose();
      kvLen.dispose();
    },
    { iterations: 15 },
  );
});
