/**
 * Tiled matmul benchmarks — performance gates.
 *
 * Compares lax.tiledMatmul (block_map) against np.matmul (naive) at
 * 256×256 through 4096×4096.
 *
 * Key results (RTX 4070 Ti SUPER, eGPU):
 *   4096×4096 tiledMatmul: ~12ms → 12,138 GFLOP/s → 53.7% of FP32 peak
 *   4096×4096 np.matmul: ~369ms → 372 GFLOP/s → 33× slower
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

// tinybench runs at least 64 measured + 16 warmup iterations by default; at
// ~0.4 s per 4096² np.matmul that exceeds the benchmark test timeout. The
// 1 s time budget still gives fast cases many samples.
const RUN_OPTIONS = { iterations: 10, warmupIterations: 2 };

describe.skipIf(!devices.includes("webgpu"))("tiled matmul", () => {
  for (const N of [256, 512, 1024, 2048, 4096]) {
    test(`${N}x${N}`, async ({ bench }) => {
      defaultDevice("webgpu");
      using k0 = random.key(0);
      using k1 = random.key(1);
      using a = random.uniform(k0, [N, N]);
      using b = random.uniform(k1, [N, N]);
      await blockUntilReady([a, b]);

      using tiledFn = jit((x: np.Array, y: np.Array) => lax.tiledMatmul(x, y));

      await bench.compare(
        bench("np.matmul", async () => {
          using c = np.matmul(a, b);
          await c.blockUntilReady();
        }),
        bench("tiledMatmul", async () => {
          using c = tiledFn(a, b);
          await c.blockUntilReady();
        }),
        RUN_OPTIONS,
      );
    });
  }
});
