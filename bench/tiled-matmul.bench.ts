/**
 * Phase 3 register-tiling A/B benchmark for tiledMatmul.
 * Compares standard np.matmul, tiledMatmul (no threadTile), and tiledMatmul
 * with register tiling (threadTile=[4,4]).
 *
 * Run: pnpm build && pnpm vitest bench bench/tiled-matmul.bench.ts
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
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))(
  "tiledMatmul register tiling",
  async () => {
    defaultDevice("webgpu");

    for (const N of [256, 512, 1024]) {
      const A = random.uniform(random.key(0), [N, N]);
      const B = random.uniform(random.key(1), [N, N]);
      await blockUntilReady([A, B]);
      afterAll(() => {
        A.dispose();
        B.dispose();
      });

      // Standard np.matmul (baseline)
      const fStd = jit((a: np.Array, b: np.Array) => np.matmul(a, b));
      afterAll(() => fStd.dispose());
      bench(`${N}x${N} np.matmul`, async () => {
        const c = fStd(A, B);
        await blockUntilReady(c);
        c.dispose();
      });

      // tiledMatmul without threadTile (Phase 2 baseline)
      const fNoTT = jit((a: np.Array, b: np.Array) =>
        lax.tiledMatmul(a, b, { Br: 16, Bc: 16, Bk: 16 }),
      );
      afterAll(() => fNoTT.dispose());
      bench(`${N}x${N} tiledMatmul (no threadTile)`, async () => {
        const c = fNoTT(A, B);
        await blockUntilReady(c);
        c.dispose();
      });

      // tiledMatmul with threadTile=[4,4] (Phase 3: register tiling)
      const fTT44 = jit((a: np.Array, b: np.Array) =>
        lax.tiledMatmul(a, b, { Br: 16, Bc: 16, Bk: 16, threadTile: [4, 4] }),
      );
      afterAll(() => fTT44.dispose());
      bench(`${N}x${N} tiledMatmul threadTile=[4,4]`, async () => {
        const c = fTT44(A, B);
        await blockUntilReady(c);
        c.dispose();
      });
    }
  },
);
