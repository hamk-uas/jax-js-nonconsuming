/**
 * Deno WebGPU benchmarks for buffer recycling & pool.
 *
 * Run with:
 *   pnpm build && deno bench --no-check --unstable-webgpu --allow-read --allow-env test/deno/recycle.bench.ts
 *
 * To compare WITH vs WITHOUT recycling:
 *   1. Run this file → record numbers
 *   2. Comment out `builder.recycleBuffers()` in src/frontend/jit.ts (~line 663)
 *      and the pool logic in src/backend/webgpu.ts (#poolPop/#poolPush in malloc/decRef)
 *   3. pnpm build
 *   4. Re-run this file → compare
 *
 * IMPORTANT: Imports from dist/ to share backend instances. Run `pnpm build` first.
 */

import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
} from "../../dist/index.js";

// Initialize WebGPU backend
const devices = await init();
if (!devices.includes("webgpu")) {
  console.log("WebGPU not available, skipping benchmarks");
  Deno.exit(0);
}
defaultDevice("webgpu");

// ============================================================================
// JIT benchmarks — recycling converts free+malloc pairs into recycle steps
// ============================================================================

// --- Fused chain (baseline: only 1 kernel, 1 malloc — recycling has no effect) ---

const chainX4k = np.ones([4096]);
await blockUntilReady(chainX4k);

const chainJit4k = jit((x: any) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
chainJit4k(chainX4k).dispose(); // warmup/compile

Deno.bench("jit chain x5 fused 4096 [baseline]", { group: "jit" }, () => {
  chainJit4k(chainX4k).dispose();
});

// --- Fused chain 65536 ---

const chainX64k = np.ones([65536]);
await blockUntilReady(chainX64k);

const chainJit64k = jit((x: any) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
chainJit64k(chainX64k).dispose();

Deno.bench("jit chain x5 fused 65536 [baseline]", { group: "jit" }, () => {
  chainJit64k(chainX64k).dispose();
});

// --- 2 outputs same size (recycling fires: free input → malloc output) ---

const twoOutX = np.ones([4096]);
await blockUntilReady(twoOutX);

const twoOutJit = jit((x: any) => {
  const a = x.add(1);
  const b = x.mul(2);
  return [a, b];
});
{
  const [a, b] = twoOutJit(twoOutX) as [any, any];
  a.dispose();
  b.dispose();
}

Deno.bench("jit 2-output same-size 4096 [recycle]", { group: "jit" }, () => {
  const [a, b] = twoOutJit(twoOutX) as [any, any];
  a.dispose();
  b.dispose();
});

// --- 3 outputs same size ---

const threeOutX = np.ones([4096]);
await blockUntilReady(threeOutX);

const threeOutJit = jit((x: any) => {
  const a = x.add(1);
  const b = x.mul(2);
  const c = x.sub(3);
  return [a, b, c];
});
{
  const [a, b, c] = threeOutJit(threeOutX) as [any, any, any];
  a.dispose();
  b.dispose();
  c.dispose();
}

Deno.bench("jit 3-output same-size 4096 [recycle]", { group: "jit" }, () => {
  const [a, b, c] = threeOutJit(threeOutX) as [any, any, any];
  a.dispose();
  b.dispose();
  c.dispose();
});

// --- Chain + reduce ---

const reduceX = np.ones([4096]);
await blockUntilReady(reduceX);

const reduceJit = jit((x: any) => x.add(1).sum().add(1));
reduceJit(reduceX).dispose();

Deno.bench("jit chain+reduce 4096", { group: "jit" }, () => {
  reduceJit(reduceX).dispose();
});

// --- 2x matmul 32x32 (unfused: two routines, free→malloc between them) ---

const matA32 = np.ones([32, 32]);
const matB32 = np.ones([32, 32]);
await blockUntilReady(matA32);
await blockUntilReady(matB32);

const matmul2Jit = jit((a: any, b: any) => {
  const c = np.matmul(a, b);
  const d = np.matmul(b, a);
  return [c, d];
});
{
  const [c, d] = matmul2Jit(matA32, matB32) as [any, any];
  c.dispose();
  d.dispose();
}

Deno.bench("jit 2x matmul 32x32 [recycle]", { group: "jit" }, () => {
  const [c, d] = matmul2Jit(matA32, matB32) as [any, any];
  c.dispose();
  d.dispose();
});

// --- Matmul 64x64 single ---

const matA64 = np.ones([64, 64]);
const matB64 = np.ones([64, 64]);
await blockUntilReady(matA64);
await blockUntilReady(matB64);

const matmulJit = jit((a: any, b: any) => np.matmul(a, b));
matmulJit(matA64, matB64).dispose();

Deno.bench("jit matmul 64x64", { group: "jit" }, () => {
  matmulJit(matA64, matB64).dispose();
});

// ============================================================================
// Scan benchmarks — mostly compiled-loop, recycling may help with intermediates
// ============================================================================

// --- Cumsum N=100 size=64 ---

const cumsumXs100 = np.ones([100, 64]);
await blockUntilReady(cumsumXs100);

const cumsumJit100 = jit((xs: any) => {
  return lax.scan(
    (carry: any, x: any) => {
      const c = carry.add(x);
      return [c, c];
    },
    np.zeros([64]),
    xs,
    { acceptPath: ["compiled-loop", "preencoded-routine"] },
  );
});
{
  const [c, y] = cumsumJit100(cumsumXs100) as [any, any];
  c.dispose();
  y.dispose();
}

Deno.bench("scan cumsum N=100 size=64", { group: "scan" }, () => {
  const [c, y] = cumsumJit100(cumsumXs100) as [any, any];
  c.dispose();
  y.dispose();
});

// --- Cumsum N=500 size=256 ---

const cumsumXs500 = np.ones([500, 256]);
await blockUntilReady(cumsumXs500);

const cumsumJit500 = jit((xs: any) => {
  return lax.scan(
    (carry: any, x: any) => {
      const c = carry.add(x);
      return [c, c];
    },
    np.zeros([256]),
    xs,
    { acceptPath: ["compiled-loop", "preencoded-routine"] },
  );
});
{
  const [c, y] = cumsumJit500(cumsumXs500) as [any, any];
  c.dispose();
  y.dispose();
}

Deno.bench("scan cumsum N=500 size=256", { group: "scan" }, () => {
  const [c, y] = cumsumJit500(cumsumXs500) as [any, any];
  c.dispose();
  y.dispose();
});

// ============================================================================
// Eager benchmarks — WebGPU buffer pool helps by reusing buffers across calls
// ============================================================================

// --- Eager chain 4096 (pool reuses freed buffers) ---

const eagerX4k = np.ones([4096]);
await blockUntilReady(eagerX4k);

Deno.bench("eager chain x5 4096 [pool]", { group: "eager" }, () => {
  eagerX4k.add(1).mul(2).sub(3).add(4).mul(0.5).dispose();
});

// --- Eager chain 65536 ---

const eagerX64k = np.ones([65536]);
await blockUntilReady(eagerX64k);

Deno.bench("eager chain x5 65536 [pool]", { group: "eager" }, () => {
  eagerX64k.add(1).mul(2).sub(3).add(4).mul(0.5).dispose();
});

// --- Alloc/free cycle (pure pool benchmark) ---

Deno.bench("eager alloc-free 16384 f32 [pool]", { group: "eager" }, () => {
  np.zeros([16384]).dispose();
});

// --- Alloc/free cycle large ---

Deno.bench("eager alloc-free 262144 f32 [pool]", { group: "eager" }, () => {
  np.zeros([262144]).dispose();
});
