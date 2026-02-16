/**
 * Deno WebGPU scan benchmarks for jax-js
 *
 * Run with: deno bench --no-check --unstable-webgpu --allow-read --allow-env test/deno/scan.bench.ts
 * Or:       pnpm run bench:deno
 *
 * Uses Deno's native wgpu-rs WebGPU implementation for hardware GPU benchmarking
 * without X11, enabling headless GPU benchmarks on servers.
 *
 * IMPORTANT: Imports from dist/ to share backend instances. Run `pnpm build` first.
 *
 * All benchmarks use acceptPath to force compiled-loop or preencoded-routine paths,
 * avoiding the fallback path which requires OffscreenCanvas (unavailable in Deno).
 * Patterns that require fallback (e.g. tree-structured carry like Kalman) are
 * excluded — use the Vitest benchmarks (bench/scan.bench.ts) for those.
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

const ACCEPT_PATH = ["compiled-loop", "preencoded-routine"];

// --- cumsum N=100 size=64 ---

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
    { acceptPath: ACCEPT_PATH },
  );
});

{
  const [c, y] = cumsumJit100(cumsumXs100);
  c.dispose();
  y.dispose();
}

Deno.bench("cumsum N=100 size=64", { group: "scan" }, () => {
  const [c, y] = cumsumJit100(cumsumXs100) as [any, any];
  c.dispose();
  y.dispose();
});

// --- cumsum N=500 size=256 ---

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
    { acceptPath: ACCEPT_PATH },
  );
});

{
  const [c, y] = cumsumJit500(cumsumXs500);
  c.dispose();
  y.dispose();
}

Deno.bench("cumsum N=500 size=256", { group: "scan" }, () => {
  const [c, y] = cumsumJit500(cumsumXs500) as [any, any];
  c.dispose();
  y.dispose();
});

// --- reverse N=200 size=64 ---

const reverseXs = np.ones([200, 64]);
await blockUntilReady(reverseXs);

const reverseJit = jit((xs: any) => {
  return lax.scan(
    (carry: any, x: any) => {
      const c = carry.add(x);
      return [c, c];
    },
    np.zeros([64]),
    xs,
    { reverse: true, acceptPath: ACCEPT_PATH },
  );
});

{
  const [c, y] = reverseJit(reverseXs);
  c.dispose();
  y.dispose();
}

Deno.bench("reverse N=200 size=64", { group: "scan" }, () => {
  const [c, y] = reverseJit(reverseXs) as [any, any];
  c.dispose();
  y.dispose();
});

// Note: Kalman benchmark (tree-structured carry) requires fallback path which
// uses readSync/OffscreenCanvas, unavailable in Deno. Run Kalman benchmarks
// via Vitest: pnpm vitest bench bench/scan.bench.ts
