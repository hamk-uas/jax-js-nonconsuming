/**
 * WebGPU N-scaling benchmark for associativeScan with pytree matmul compose.
 *
 * Tests that copy-batching keeps latency near-flat as N grows.
 * Runs on both NVIDIA RTX 4070 Ti SUPER and Intel Arc via gpu-test.sh.
 *
 * Run with:
 *   scripts/gpu-test.sh bench bench/assoc-scan-scaling.bench.ts
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
import { describe, test } from "vitest";

const devices = await init("webgpu");

describe.skipIf(!devices.includes("webgpu"))(
  "assocScan pytree N-scaling",
  () => {
    test("pytree matmul compose", async ({ bench }) => {
      defaultDevice("webgpu");
      using stack = new DisposableStack();

      // Binary associative op: (A1,B1) * (A2,B2) = (A1@A2, A1@B2 + B1)
      const f = stack.use(
        jit(([a, b]: [np.Array, np.Array]) =>
          lax.associativeScan(
            (
              [a1, b1]: [np.Array, np.Array],
              [a2, b2]: [np.Array, np.Array],
            ) => {
              const newA = np.matmul(a1, a2);
              const newB = np.matmul(a1, b2).add(b1);
              return [newA, newB] as [np.Array, np.Array];
            },
            [a, b] as [np.Array, np.Array],
          ),
        ),
      );

      const registrations = [];
      for (const N of [100, 400, 800, 1600, 3200] as const) {
        const A = stack.use(np.full([N, 2, 2], 0.99, { dtype: np.float32 }));
        const B = stack.use(np.full([N, 2, 1], 0.1, { dtype: np.float32 }));
        await blockUntilReady([A, B]);
        registrations.push(
          bench(`N=${N}`, async () => {
            const r = f([A, B]) as [np.Array, np.Array];
            await tree.consumeData(r);
          }),
        );
      }
      await bench.compare(...registrations);
    });
  },
);
