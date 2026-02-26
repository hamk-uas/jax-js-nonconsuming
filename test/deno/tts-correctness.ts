/* eslint-disable */
/**
 * TTS correctness check — captures output values with fixed seeds.
 * Run before and after optimizations to verify no regression.
 *
 * Usage: deno run --no-check --unstable-webgpu --allow-read --allow-env test/deno/tts-correctness.ts
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

const devices = await init();
if (!devices.includes("webgpu")) {
  console.log("WebGPU not available");
  Deno.exit(1);
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

// ── Deterministic key sequence ───────────────────────────────────────────────
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

// ── Fingerprint helper ───────────────────────────────────────────────────────

async function fingerprint(name: string, arr: any): Promise<string> {
  await blockUntilReady(arr);
  const data = await arr.data();
  const vals = [...data];
  const sum = vals.reduce((a: number, b: number) => a + b, 0);
  const absSum = vals.reduce((a: number, b: number) => a + Math.abs(b), 0);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  // Show first 8 values for visual comparison
  const first8 = vals
    .slice(0, 8)
    .map((v: number) => v.toFixed(4))
    .join(", ");
  const fp = `${name}: shape=${arr.shape} sum=${sum.toFixed(6)} absSum=${absSum.toFixed(6)} min=${min.toFixed(6)} max=${max.toFixed(6)} first8=[${first8}]`;
  console.log(fp);
  return fp;
}

// ── Build model with deterministic seeds ─────────────────────────────────────

console.log("=== TTS Correctness Check ===\n");
_seed = 0; // Reset seed

// Build a smaller model for faster correctness checking (1 layer each)
const layer1 = makeTransformerLayer(1024, 4096, false);
const layer2 = makeTransformerLayer(512, 2048, true);
const outNorm = makeLayerNorm(1024);

await blockUntilReady([...tree.leaves(layer1), ...tree.leaves(layer2)]);

// Fixed inputs
const x1024 = random.normal(nextKey(), [1, 1024]).astype(np.float16);
const x512 = random.normal(nextKey(), [16, 512]).astype(np.float16);
const offset64 = np.array(64, { dtype: np.int32 });
const kvLen64 = np.array(64, { dtype: np.int32 });
const offset128 = np.array(128, { dtype: np.int32 });
const kvLen128 = np.array(128, { dtype: np.int32 });

// KV caches
const kv1: KVCache = {
  key: random.normal(nextKey(), [128, 16, 64]).astype(np.float16),
  value: random.normal(nextKey(), [128, 16, 64]).astype(np.float16),
};
const kv2: KVCache = {
  key: random.normal(nextKey(), [272, 8, 64]).astype(np.float16),
  value: random.normal(nextKey(), [272, 8, 64]).astype(np.float16),
};

await blockUntilReady([x1024, x512, kv1.key, kv1.value, kv2.key, kv2.value]);

const results: string[] = [];

// ── Test 1: Raw einsum (attention scores) ────────────────────────────────────
console.log("\n--- Einsum correctness ---");
const eQ = random.normal(nextKey(), [1, 1, 16, 64]).astype(np.float16);
const eK = random.normal(nextKey(), [1, 64, 16, 64]).astype(np.float16);
const scores = np.einsum("BLNH,BSNH->BNLS", eQ, eK);
results.push(await fingerprint("einsum_scores", scores));
scores.dispose();

// ── Test 2: Raw einsum (weighted values) ─────────────────────────────────────
const eAttn = random.normal(nextKey(), [1, 16, 1, 64]).astype(np.float16);
const eV = random.normal(nextKey(), [1, 64, 16, 64]).astype(np.float16);
const wv = np.einsum("BNLS,BSNH->BLNH", eAttn, eV);
results.push(await fingerprint("einsum_values", wv));
wv.dispose();

// ── Test 3: dotProductAttention ──────────────────────────────────────────────
console.log("\n--- dotProductAttention ---");
const dpaQ = random.normal(nextKey(), [1, 16, 64]).astype(np.float16);
const dpaK = random.normal(nextKey(), [64, 16, 64]).astype(np.float16);
const dpaV = random.normal(nextKey(), [64, 16, 64]).astype(np.float16);
const dpaMask = np.arange(64).lessEqual(0);
const dpaOut = nn.dotProductAttention(dpaQ, dpaK, dpaV, { mask: dpaMask });
results.push(await fingerprint("dotProductAttention", dpaOut));
dpaOut.dispose();
dpaQ.dispose();
dpaK.dispose();
dpaV.dispose();
dpaMask.dispose();

// ── Test 4: Single FlowLM transformer layer ──────────────────────────────────
console.log("\n--- Transformer layer (dim=1024, 16 heads) ---");
const [tOut1, _kv1] = runTransformerLayer(
  layer1,
  kv1,
  x1024,
  offset64,
  kvLen64,
  16,
  0,
);
results.push(await fingerprint("transformer_1024", tOut1));
tOut1.dispose();

// ── Test 5: Single Mimi transformer layer ────────────────────────────────────
console.log("\n--- Transformer layer (dim=512, 8 heads, context=250) ---");
const [tOut2, _kv2] = runTransformerLayer(
  layer2,
  kv2,
  x512,
  offset128,
  kvLen128,
  8,
  250,
);
results.push(await fingerprint("transformer_512", tOut2));
tOut2.dispose();

// ── Test 6: LayerNorm ────────────────────────────────────────────────────────
console.log("\n--- LayerNorm ---");
const normOut = runLayerNorm(outNorm, x1024);
results.push(await fingerprint("layerNorm", normOut));
normOut.dispose();

// ── Test 7: Matmul ───────────────────────────────────────────────────────────
console.log("\n--- Matmul ---");
const mA = random.normal(nextKey(), [1, 1024]).astype(np.float16);
const mW = random.normal(nextKey(), [4096, 1024]).astype(np.float16);
const mOut = np.dot(mA, mW.transpose());
results.push(await fingerprint("matmul_1024x4096", mOut));
mOut.dispose();
mA.dispose();
mW.dispose();

// ── Summary ──────────────────────────────────────────────────────────────────
console.log("\n=== FINGERPRINT SUMMARY ===");
for (const r of results) console.log(r);
console.log("===========================\n");

// Cleanup
tree.dispose(layer1);
tree.dispose(layer2);
tree.dispose(outNorm);
x1024.dispose();
x512.dispose();
offset64.dispose();
kvLen64.dispose();
offset128.dispose();
kvLen128.dispose();
tree.dispose(kv1);
tree.dispose(kv2);
eQ.dispose();
eK.dispose();
eAttn.dispose();
eV.dispose();
