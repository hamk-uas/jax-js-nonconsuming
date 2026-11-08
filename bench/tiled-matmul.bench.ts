/**
 * Phase 3 register-tiling A/B benchmark for tiledMatmul.
 * Compares standard np.matmul, tiledMatmul (no threadTile), and tiledMatmul
 * with register tiling (threadTile=[4,4]).
 *
 * Run: pnpm vitest bench bench/tiled-matmul.bench.ts
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
import { describe, test } from "vitest";

const devices = await init("webgpu");

describe.skipIf(!devices.includes("webgpu"))(
  "tiledMatmul register tiling",
  () => {
    for (const N of [256, 512, 1024]) {
      test(`${N}x${N}`, async ({ bench }) => {
        defaultDevice("webgpu");
        using k0 = random.key(0);
        using k1 = random.key(1);
        using A = random.uniform(k0, [N, N]);
        using B = random.uniform(k1, [N, N]);
        await blockUntilReady([A, B]);

        // Standard np.matmul (baseline)
        using fStd = jit((a: np.Array, b: np.Array) => np.matmul(a, b));
        // tiledMatmul without threadTile (Phase 2 baseline)
        using fNoTT = jit((a: np.Array, b: np.Array) =>
          lax.tiledMatmul(a, b, { Br: 16, Bc: 16, Bk: 16 }),
        );
        // tiledMatmul with threadTile=[4,4] (Phase 3: register tiling)
        using fTT44 = jit((a: np.Array, b: np.Array) =>
          lax.tiledMatmul(a, b, {
            Br: 16,
            Bc: 16,
            Bk: 16,
            threadTile: [4, 4],
          }),
        );

        await bench.compare(
          bench("np.matmul", async () => {
            using c = fStd(A, B);
            await blockUntilReady(c);
          }),
          bench("tiledMatmul (no threadTile)", async () => {
            using c = fNoTT(A, B);
            await blockUntilReady(c);
          }),
          bench("tiledMatmul threadTile=[4,4]", async () => {
            using c = fTT44(A, B);
            await blockUntilReady(c);
          }),
        );
      });
    }
  },
);
