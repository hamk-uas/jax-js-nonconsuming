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

// tinybench runs at least 64 measured + 16 warmup iterations by default; at
// ~0.4 s per eager 4096² matmul that exceeds the benchmark test timeout. The
// 1 s time budget still gives fast cases many samples.
const RUN_OPTIONS = { iterations: 10, warmupIterations: 2 };

describe.skipIf(!devices.includes("webgpu"))("gpu matmul", () => {
  for (const N of [2048, 4096] as const) {
    test(`${N}x${N}`, async ({ bench }) => {
      defaultDevice("webgpu");
      using k0 = random.key(0);
      using k1 = random.key(1);
      using a = random.uniform(k0, [N, N]);
      using b = random.uniform(k1, [N, N]);
      await blockUntilReady([a, b]);

      // np.matmul inside jit() → tiledMatmul with auto-selected tile config
      using fDef = jit((x: np.Array, y: np.Array) => np.matmul(x, y));
      // Explicit Br=Bc=16 no threadTile (pre-adaptive baseline)
      using f16 = jit((x: np.Array, y: np.Array) =>
        lax.tiledMatmul(x, y, { Br: 16, Bc: 16, Bk: 16 }),
      );

      await bench.compare(
        bench("eager", async () => {
          using c = np.matmul(a, b);
          await c.blockUntilReady();
        }),
        bench("jit(matmul)", async () => {
          using c = fDef(a, b);
          await c.blockUntilReady();
        }),
        bench("16×16 (baseline)", async () => {
          using c = f16(a, b);
          await c.blockUntilReady();
        }),
        RUN_OPTIONS,
      );
    });
  }
});
