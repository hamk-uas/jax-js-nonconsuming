/* eslint-disable */
/**
 * TTS (Pocket TTS) per-step benchmark — Deno WebGPU
 *
 * Run: deno run --no-check --unstable-webgpu --allow-read --allow-env test/deno/tts.bench.ts
 *
 * Replicates the model structure from website/src/routes/tts/pocket-tts.ts
 * with random weights to measure per-step inference latency.
 *
 * Model dimensions (Kyutai Pocket TTS):
 *   FlowLM: dim=1024, ldim=32, numHeads=16, headDim=64, ffnDim=4096, 6 layers
 *   FlowNet: 6 ResBlocks (1024-wide MLP), timestep embedder
 *   Mimi decoder: dim=512, numHeads=8, headDim=64, 8 layers, SEANet decoder
 *
 * Real-time target: 1920 audio samples per step at 24kHz = 80ms/step.
 */
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  nn,
  numpy as np,
  random,
  tree,
} from "../../dist/index.js";

// ── Init ─────────────────────────────────────────────────────────────────────

const devices = await init();
if (!devices.includes("webgpu")) {
  console.log("WebGPU not available, skipping benchmarks");
  Deno.exit(0);
}
defaultDevice("webgpu");

// ── Types ────────────────────────────────────────────────────────────────────

type Linear = { weight: any; bias?: any };
type LayerNorm = { weight: any; bias: any };
type KVCache = { key: any; value: any };
type MHA = { inProj: Linear; outProj: Linear };

type TransformerLayer = {
  selfAttn: MHA;
  norm1: LayerNorm;
  norm2: LayerNorm;
  linear1: Linear;
  linear2: Linear;
  layerScale1?: any;
  layerScale2?: any;
};

type ResBlock = {
  inLN: LayerNorm;
  mlp: [Linear, undefined, Linear];
  adaLNModulation: [undefined, Linear];
};

type FlowNet = {
  inputProj: Linear;
  condEmbed: Linear;
  timeEmbed: { mlp: [Linear, undefined, Linear, { alpha: any }]; freqs: any }[];
  resBlocks: ResBlock[];
  finalLayer: { linear: Linear; adaLNModulation: [undefined, Linear] };
};

type FlowLMModel = {
  inputLinear: Linear;
  outNorm: LayerNorm;
  outEos: Linear;
  flowNet: FlowNet;
  transformer: TransformerLayer[];
};

type MimiDecoder = {
  decoderTransformer: TransformerLayer[];
};

// ── Helpers ──────────────────────────────────────────────────────────────────

let _seed = 0;
function nextKey() {
  return random.key(_seed++);
}

function makeLinear(
  inDim: number,
  outDim: number,
  bias = true,
  dtype = np.float16,
): Linear {
  const w = random.normal(nextKey(), [outDim, inDim]).astype(dtype);
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

function runLinear({ weight, bias }: Linear, x: any): any {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
}

const runLayerNorm = jit(
  function runLayerNorm({ weight, bias }: any, x: any, eps: number = 1e-5) {
    const dtype = x.dtype;
    x = x.astype(np.float32);
    const mean = x.mean(-1, { keepdims: true });
    const var_ = np.var_(x, -1, { mean, correction: 0, keepdims: true });
    x = x.sub(mean).div(np.sqrt(var_.add(eps)));
    if (weight) x = x.mul(weight).add(bias!);
    return x.astype(dtype);
  },
  { staticArgnums: [2] },
);

// ── RoPE ─────────────────────────────────────────────────────────────────────

function runRope(q: any, k: any, offset: any, maxPeriod = 10000): [any, any] {
  const [T, H, D] = q.shape;
  const halfD = D / 2;
  const ds = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const freqs = np.exp(ds.mul((-Math.log(maxPeriod) * 2) / D));
  const ts = np.arange(T).add(offset).astype(np.float32).reshape([T, 1, 1]);
  const qR = q.reshape([T, H, halfD, 2]);
  const kR = k.reshape([T, H, halfD, 2]);
  let [qr, qi] = np.split(qR, 2, -1);
  let [kr, ki] = np.split(kR, 2, -1);
  qr = np.squeeze(qr, -1);
  qi = np.squeeze(qi, -1);
  kr = np.squeeze(kr, -1);
  ki = np.squeeze(ki, -1);
  const angles = freqs.mul(ts);
  const rotr = np.cos(angles).astype(qr.dtype);
  const roti = np.sin(angles).astype(qr.dtype);
  const qo = np
    .stack([qr.mul(rotr).sub(qi.mul(roti)), qr.mul(roti).add(qi.mul(rotr))], -1)
    .reshape([T, H, D]);
  const ko = np
    .stack([kr.mul(rotr).sub(ki.mul(roti)), kr.mul(roti).add(ki.mul(rotr))], -1)
    .reshape([T, H, D]);
  return [qo, ko];
}

// ── Attention ────────────────────────────────────────────────────────────────

function runMHADecode(
  { inProj, outProj }: MHA,
  kvCache: KVCache,
  query: any,
  offset: any,
  kvCacheLen: any,
  numHeads: number,
  context = 0,
): [any, KVCache] {
  const [T, embedDim] = query.shape;
  const headDim = embedDim / numHeads;
  const projected = runLinear(inProj, query);
  const qkv = projected.reshape([T, 3 * numHeads, headDim]);
  const [q_, k_, v] = np.split(qkv, 3, 1);
  const [q, k] = runRope(q_, k_, offset);
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

// ── Transformer Layer ────────────────────────────────────────────────────────

function makeMHA(dim: number, dtype = np.float16): MHA {
  return {
    inProj: makeLinear(dim, 3 * dim, false, dtype),
    outProj: makeLinear(dim, dim, false, dtype),
  };
}

function makeTransformerLayer(
  dim: number,
  ffnDim: number,
  withLayerScale = false,
  dtype = np.float16,
): TransformerLayer {
  const r: TransformerLayer = {
    selfAttn: makeMHA(dim, dtype),
    norm1: makeLayerNorm(dim, dtype),
    norm2: makeLayerNorm(dim, dtype),
    linear1: makeLinear(dim, ffnDim, false, dtype),
    linear2: makeLinear(ffnDim, dim, false, dtype),
  };
  if (withLayerScale) {
    r.layerScale1 = np.ones([dim], { dtype });
    r.layerScale2 = np.ones([dim], { dtype });
  }
  return r;
}

const runTransformerLayer = jit(
  function runTransformerLayer(
    layer: TransformerLayer,
    kvCache: KVCache,
    x: any,
    offset: any,
    kvCacheLen: any,
    numHeads: number,
    context: number,
  ): [any, KVCache] {
    const {
      selfAttn,
      norm1,
      norm2,
      linear1,
      linear2,
      layerScale1,
      layerScale2,
    } = layer;
    const xOrig = x;
    x = runLayerNorm(norm1, x);
    let update: any;
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
  },
  { staticArgnums: [5, 6] },
);

// ── FlowNet ──────────────────────────────────────────────────────────────────

function makeFlowNet(
  condDim: number,
  flowDim: number,
  dtype = np.float16,
): FlowNet {
  const makeTE = () => ({
    mlp: [
      makeLinear(256, flowDim, true, dtype),
      undefined,
      makeLinear(flowDim, flowDim, true, dtype),
      { alpha: np.ones([flowDim], { dtype }) },
    ] as any,
    freqs: random.normal(nextKey(), [128]).astype(dtype),
  });
  const makeRB = (): ResBlock => ({
    inLN: makeLayerNorm(flowDim, dtype),
    mlp: [
      makeLinear(flowDim, flowDim, true, dtype),
      undefined as any,
      makeLinear(flowDim, flowDim, true, dtype),
    ],
    adaLNModulation: [
      undefined as any,
      makeLinear(flowDim, 3 * flowDim, true, dtype),
    ],
  });
  return {
    inputProj: makeLinear(32, flowDim, true, dtype),
    condEmbed: makeLinear(condDim, flowDim, true, dtype),
    timeEmbed: [makeTE(), makeTE()],
    resBlocks: Array.from({ length: 6 }, () => makeRB()),
    finalLayer: {
      linear: makeLinear(flowDim, 32, true, dtype),
      adaLNModulation: [
        undefined as any,
        makeLinear(flowDim, 2 * flowDim, true, dtype),
      ],
    },
  };
}

function runTimestepEmbedder(te: any, t: any): any {
  const [linear1, , linear2, rmsNorm] = te.mlp;
  const args = t.mul(te.freqs);
  const embedding = np.concatenate([np.cos(args), np.sin(args)], -1);
  let x = runLinear(linear1, embedding);
  x = nn.silu(x);
  x = runLinear(linear2, x);
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const var_ = np.var_(x, -1, { correction: 0, keepdims: true });
  x = x.mul(rmsNorm.alpha).div(np.sqrt(var_.add(1e-5)));
  return x.astype(dtype);
}

const runFlowNet = jit(function runFlowNet(
  net: FlowNet,
  c: any,
  s: any,
  t: any,
  x: any,
): any {
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
});

// ── Model construction ───────────────────────────────────────────────────────

function makeFlowLMModel(dtype = np.float16): FlowLMModel {
  return {
    inputLinear: makeLinear(32, 1024, true, dtype),
    outNorm: makeLayerNorm(1024, dtype),
    outEos: makeLinear(1024, 1, true, dtype),
    flowNet: makeFlowNet(1024, 1024, dtype),
    transformer: Array.from({ length: 6 }, () =>
      makeTransformerLayer(1024, 4096, false, dtype),
    ),
  };
}

function makeMimiDecoder(dtype = np.float16): MimiDecoder {
  return {
    decoderTransformer: Array.from({ length: 8 }, () =>
      makeTransformerLayer(512, 2048, true, dtype),
    ),
  };
}

// ── Benchmark harness ────────────────────────────────────────────────────────

async function bench(
  name: string,
  fn: () => Promise<void>,
  warmup = 3,
  iters = 15,
) {
  // Warmup
  for (let i = 0; i < warmup; i++) await fn();

  // Timed runs
  const times: number[] = [];
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const min = times[0];
  const max = times[times.length - 1];
  const p90 = times[Math.floor(times.length * 0.9)];
  console.log(
    `  ${name}: median=${median.toFixed(1)}ms mean=${mean.toFixed(1)}ms min=${min.toFixed(1)}ms p90=${p90.toFixed(1)}ms max=${max.toFixed(1)}ms`,
  );
  return { name, median, mean, min, max, p90 };
}

// ── Build model ──────────────────────────────────────────────────────────────

console.log("Building random TTS model...");
const flowLM = makeFlowLMModel();
const mimiDecoder = makeMimiDecoder();
await blockUntilReady(tree.leaves(flowLM));
await blockUntilReady(tree.leaves(mimiDecoder));

// KV caches for FlowLM (capacity 128, simulating mid-generation at position 64)
const flowLMKVCaches: KVCache[] = flowLM.transformer.map(() => ({
  key: random.normal(nextKey(), [128, 16, 64]).astype(np.float16),
  value: random.normal(nextKey(), [128, 16, 64]).astype(np.float16),
}));
const flowLMKvCacheLen = 64;

// KV caches for Mimi decoder (capacity 272 = 250 context + 16 + margin)
const mimiKVCaches: KVCache[] = mimiDecoder.decoderTransformer.map(() => ({
  key: random.normal(nextKey(), [272, 8, 64]).astype(np.float16),
  value: random.normal(nextKey(), [272, 8, 64]).astype(np.float16),
}));
const mimiKvCacheLen = 128;

await blockUntilReady([
  ...flowLMKVCaches.flatMap((c) => [c.key, c.value]),
  ...mimiKVCaches.flatMap((c) => [c.key, c.value]),
]);

console.log("Model ready. Running benchmarks...\n");
console.log("Real-time budget: 80ms per step (1920 samples @ 24kHz)\n");

// ── Pre-allocate benchmark inputs ────────────────────────────────────────────

const flowLMInput = random.normal(nextKey(), [1, 1024]).astype(np.float16);
const flowLMOffset = np.array(flowLMKvCacheLen, { dtype: np.int32 });
const flowLMKvLen = np.array(flowLMKvCacheLen, { dtype: np.int32 });

const mimiInput = random.normal(nextKey(), [16, 512]).astype(np.float16);
const mimiOffset = np.array(mimiKvCacheLen, { dtype: np.int32 });
const mimiKvLen = np.array(mimiKvCacheLen, { dtype: np.int32 });

const einsumQ = random.normal(nextKey(), [1, 1, 16, 64]).astype(np.float16);
const einsumK = random.normal(nextKey(), [1, 64, 16, 64]).astype(np.float16);
const einsumAttn = random.normal(nextKey(), [1, 16, 1, 64]).astype(np.float16);
const einsumV = random.normal(nextKey(), [1, 64, 16, 64]).astype(np.float16);

const matmulA_up = random.normal(nextKey(), [1, 1024]).astype(np.float16);
const matmulW_up = random.normal(nextKey(), [4096, 1024]).astype(np.float16);
const matmulA_down = random.normal(nextKey(), [1, 4096]).astype(np.float16);
const matmulW_down = random.normal(nextKey(), [1024, 4096]).astype(np.float16);

const flowNetC = random.normal(nextKey(), [1, 1024]).astype(np.float16);
const flowNetS = np.full([1, 1], 0.0);
const flowNetT = np.full([1, 1], 1.0);
const flowNetX = random.normal(nextKey(), [1, 32]).astype(np.float16);

await blockUntilReady([
  flowLMInput,
  flowLMOffset,
  flowLMKvLen,
  mimiInput,
  mimiOffset,
  mimiKvLen,
  einsumQ,
  einsumK,
  einsumAttn,
  einsumV,
  matmulA_up,
  matmulW_up,
  matmulA_down,
  matmulW_down,
  flowNetC,
  flowNetS,
  flowNetT,
  flowNetX,
]);

// ── Benchmarks ───────────────────────────────────────────────────────────────

const results: any[] = [];

// 1. Attention einsum (scores)
results.push(
  await bench(
    "einsum BLNH,BSNH->BNLS (scores B=1,L=1,S=64,N=16,H=64)",
    async () => {
      const scores = np.einsum("BLNH,BSNH->BNLS", einsumQ, einsumK);
      await blockUntilReady(scores);
      scores.dispose();
    },
    5,
    30,
  ),
);

// 2. Attention einsum (weighted values)
results.push(
  await bench(
    "einsum BNLS,BSNH->BLNH (values B=1,L=1,S=64,N=16,H=64)",
    async () => {
      const out = np.einsum("BNLS,BSNH->BLNH", einsumAttn, einsumV);
      await blockUntilReady(out);
      out.dispose();
    },
    5,
    30,
  ),
);

// 3. Small matmul (FFN up projection, single token)
results.push(
  await bench(
    "matmul [1,1024] x [4096,1024]^T (FFN up)",
    async () => {
      const c = np.dot(matmulA_up, matmulW_up.transpose());
      await blockUntilReady(c);
      c.dispose();
    },
    5,
    30,
  ),
);

// 4. Small matmul (FFN down projection, single token)
results.push(
  await bench(
    "matmul [1,4096] x [1024,4096]^T (FFN down)",
    async () => {
      const c = np.dot(matmulA_down, matmulW_down.transpose());
      await blockUntilReady(c);
      c.dispose();
    },
    5,
    30,
  ),
);

// 5. Single FlowLM transformer layer
results.push(
  await bench(
    "Single FlowLM transformer layer (T=1, dim=1024, 16 heads)",
    async () => {
      const [out, _kv] = runTransformerLayer(
        flowLM.transformer[0],
        flowLMKVCaches[0],
        flowLMInput,
        flowLMOffset,
        flowLMKvLen,
        16,
        0,
      );
      await blockUntilReady(out);
      out.dispose();
    },
    5,
    20,
  ),
);

// 6. Full FlowLM transformer (6 layers)
results.push(
  await bench(
    "FlowLM transformer (6 layers, T=1, dim=1024)",
    async () => {
      let x = flowLMInput;
      for (let i = 0; i < 6; i++) {
        const [newX, _kv] = runTransformerLayer(
          flowLM.transformer[i],
          flowLMKVCaches[i],
          x,
          flowLMOffset,
          flowLMKvLen,
          16,
          0,
        );
        if (x !== flowLMInput) x.dispose();
        x = newX;
      }
      await blockUntilReady(x);
      x.dispose();
    },
    3,
    15,
  ),
);

// 7. FlowNet (SimpleMLPAdaLN with 6 ResBlocks)
results.push(
  await bench(
    "FlowNet (6 ResBlocks, dim=1024)",
    async () => {
      const result = runFlowNet(
        flowLM.flowNet,
        flowNetC,
        flowNetS,
        flowNetT,
        flowNetX,
      );
      await blockUntilReady(result);
      result.dispose();
    },
    3,
    15,
  ),
);

// 8. Full FlowLM step (transformer + norm + FlowNet)
results.push(
  await bench(
    "Full FlowLM step (transformer + FlowNet)",
    async () => {
      let x: any = flowLMInput;
      for (let i = 0; i < 6; i++) {
        const [newX, _kv] = runTransformerLayer(
          flowLM.transformer[i],
          flowLMKVCaches[i],
          x,
          flowLMOffset,
          flowLMKvLen,
          16,
          0,
        );
        if (x !== flowLMInput) x.dispose();
        x = newX;
      }
      const normed = runLayerNorm(flowLM.outNorm, x);
      x.dispose();
      const transformerOut = normed.slice([-1]);
      normed.dispose();
      const noise = random.normal(nextKey(), [1, 32]).astype(np.float16);
      const latent = runFlowNet(
        flowLM.flowNet,
        transformerOut,
        flowNetS,
        flowNetT,
        noise,
      );
      await blockUntilReady(latent);
      latent.dispose();
      transformerOut.dispose();
      noise.dispose();
    },
    3,
    15,
  ),
);

// 9. Mimi decoder transformer (8 layers, T=16)
results.push(
  await bench(
    "Mimi transformer (8 layers, T=16, dim=512)",
    async () => {
      let x: any = mimiInput;
      for (let i = 0; i < 8; i++) {
        const [newX, _kv] = runTransformerLayer(
          mimiDecoder.decoderTransformer[i],
          mimiKVCaches[i],
          x,
          mimiOffset,
          mimiKvLen,
          8,
          250,
        );
        if (x !== mimiInput) x.dispose();
        x = newX;
      }
      await blockUntilReady(x);
      x.dispose();
    },
    3,
    15,
  ),
);

// ── Combined pipeline (one fence) ────────────────────────────────────────────

// 13. Full TTS step: FlowLM + Mimi with single blockUntilReady
results.push(
  await bench(
    "Full TTS step (FlowLM + Mimi, 1 fence)",
    async () => {
      // FlowLM transformer
      let x: any = flowLMInput;
      for (let i = 0; i < 6; i++) {
        const [newX, _kv] = runTransformerLayer(
          flowLM.transformer[i],
          flowLMKVCaches[i],
          x,
          flowLMOffset,
          flowLMKvLen,
          16,
          0,
        );
        if (x !== flowLMInput) x.dispose();
        x = newX;
      }
      // FlowLM output processing
      const normed = runLayerNorm(flowLM.outNorm, x);
      x.dispose();
      const transformerOut = normed.slice([-1]);
      normed.dispose();
      const noise = random.normal(nextKey(), [1, 32]).astype(np.float16);
      const latent = runFlowNet(
        flowLM.flowNet,
        transformerOut,
        flowNetS,
        flowNetT,
        noise,
      );
      latent.dispose();
      transformerOut.dispose();
      noise.dispose();

      // Mimi transformer
      let mx: any = mimiInput;
      for (let i = 0; i < 8; i++) {
        const [newX, _kv] = runTransformerLayer(
          mimiDecoder.decoderTransformer[i],
          mimiKVCaches[i],
          mx,
          mimiOffset,
          mimiKvLen,
          8,
          250,
        );
        if (mx !== mimiInput) mx.dispose();
        mx = newX;
      }
      await blockUntilReady(mx);
      mx.dispose();
    },
    3,
    15,
  ),
);

// ── Summary ──────────────────────────────────────────────────────────────────

console.log(
  "\n═══ Summary ═══════════════════════════════════════════════════",
);
const flowLMTime =
  results.find((r) => r.name.startsWith("Full FlowLM step ("))?.median ?? 0;
const mimiTime =
  results.find((r) => r.name === "Mimi transformer (8 layers, T=16, dim=512)")
    ?.median ?? 0;
const estTotal = flowLMTime + mimiTime;
console.log(`  Per-component (2 fences):`);
console.log(`    FlowLM step:       ${flowLMTime.toFixed(1)} ms`);
console.log(`    Mimi transformer:  ${mimiTime.toFixed(1)} ms`);
console.log(`    Sum:               ${estTotal.toFixed(1)} ms`);

const combinedTime =
  results.find((r) => r.name.startsWith("Full TTS step ("))?.median ?? 0;
console.log(`  Combined (1 fence):  ${combinedTime.toFixed(1)} ms`);

const bestTotal = Math.min(estTotal, combinedTime || Infinity);
console.log(`  Best total:          ${bestTotal.toFixed(1)} ms`);
console.log(`  Real-time budget:    80.0 ms`);
console.log(`  Ratio:               ${(bestTotal / 80).toFixed(2)}x real-time`);
if (bestTotal > 80) {
  console.log(
    `  ⚠ ${((bestTotal / 80 - 1) * 100).toFixed(0)}% over budget — need ${((1 - 80 / bestTotal) * 100).toFixed(0)}% speedup`,
  );
} else {
  console.log(`  ✓ Within real-time budget!`);
}
console.log(
  "═══════════════════════════════════════════════════════════════\n",
);

// Cleanup
tree.dispose(flowLM);
tree.dispose(mimiDecoder);
tree.dispose(flowLMKVCaches);
tree.dispose(mimiKVCaches);
