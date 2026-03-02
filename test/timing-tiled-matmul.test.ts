/**
 * Quick Phase 3 timing test for tiledMatmul register tiling.
 * Run: pnpm build && pnpm vitest run tmp/timing-tiled-matmul.test.ts
 */
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { test } from "vitest";

async function timeIt(
  label: string,
  fn: () => np.Array,
  warmup = 5,
  iters = 20,
) {
  for (let i = 0; i < warmup; i++) {
    const r = fn();
    await blockUntilReady(r);
    r.dispose();
  }
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const r = fn();
    await blockUntilReady(r);
    r.dispose();
  }
  const avg = (performance.now() - t0) / iters;
  console.log(`  ${label}: ${avg.toFixed(3)} ms/iter`);
  return avg;
}

test("Phase 3 tiledMatmul benchmark", async () => {
  const devices = await init("webgpu");
  if (!devices.includes("webgpu")) {
    console.log("WebGPU not available");
    return;
  }
  defaultDevice("webgpu");

  for (const N of [256, 512, 1024]) {
    console.log(`\n=== ${N}x${N} ===`);
    const A = random.uniform(random.key(0), [N, N]);
    const B = random.uniform(random.key(1), [N, N]);
    await blockUntilReady([A, B]);

    const fStd = jit((a: np.Array, b: np.Array) => np.matmul(a, b));
    const stdMs = await timeIt("np.matmul (jit)", () => fStd(A, B));

    const fNoTT = jit((a: np.Array, b: np.Array) =>
      lax.tiledMatmul(a, b, { Br: 16, Bc: 16, Bk: 16 }),
    );
    const noTTms = await timeIt("tiledMatmul (no threadTile)", () =>
      fNoTT(A, B),
    );

    const fTT44 = jit((a: np.Array, b: np.Array) =>
      lax.tiledMatmul(a, b, { Br: 16, Bc: 16, Bk: 16, threadTile: [4, 4] }),
    );
    const tt44ms = await timeIt("tiledMatmul threadTile=[4,4]", () =>
      fTT44(A, B),
    );

    console.log(`  Speedup vs no-threadTile: ${(noTTms / tt44ms).toFixed(2)}x`);
    console.log(`  Speedup vs np.matmul: ${(stdMs / tt44ms).toFixed(2)}x`);

    fStd.dispose();
    fNoTT.dispose();
    fTT44.dispose();
    A.dispose();
    B.dispose();
  }
}, 300000);
