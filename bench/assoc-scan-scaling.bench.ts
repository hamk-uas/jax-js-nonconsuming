/**
 * WebGPU N-scaling benchmark for associativeScan with pytree matmul compose.
 *
 * Tests that copy-batching keeps latency near-flat as N grows.
 * Runs on both NVIDIA RTX 4070 Ti SUPER and Intel Arc via gpu-test.sh.
 *
 * Run with:
 *   pnpm build && scripts/gpu-test.sh bench bench/assoc-scan-scaling.bench.ts
 *   GPU=nvidia scripts/gpu-test.sh bench bench/assoc-scan-scaling.bench.ts
 *   GPU=intel  scripts/gpu-test.sh bench bench/assoc-scan-scaling.bench.ts
 */
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))(
  "assocScan pytree N-scaling",
  async () => {
    defaultDevice("webgpu");

    for (const N of [100, 400, 800, 1600, 3200] as const) {
      const A = np.ones([N, 2, 2], { dtype: np.float32 }).mul(np.array([0.99]));
      const B = np.ones([N, 2, 1], { dtype: np.float32 }).mul(np.array([0.1]));
      await blockUntilReady([A, B]);

      afterAll(() => {
        A.dispose();
        B.dispose();
      });

      // Binary associative op: (A1,B1) * (A2,B2) = (A1@A2, A1@B2 + B1)
      const f = jit(([a, b]: [np.Array, np.Array]) => {
        return lax.associativeScan(
          ([a1, b1]: [np.Array, np.Array], [a2, b2]: [np.Array, np.Array]) => {
            const newA = np.matmul(a1, a2);
            const newB = np.matmul(a1, b2).add(b1);
            return [newA, newB] as [np.Array, np.Array];
          },
          [a, b] as [np.Array, np.Array],
        );
      });
      // Warmup — compile the JIT program
      const warmup = f([A, B]) as [np.Array, np.Array];
      await tree.consumeData(warmup);
      afterAll(() => f.dispose());

      bench(
        `N=${N} pytree matmul compose`,
        async () => {
          const r = f([A, B]) as [np.Array, np.Array];
          await tree.consumeData(r);
        },
        { warmupIterations: 0 },
      );
    }
  },
);
