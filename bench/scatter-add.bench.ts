/**
 * M8.1 Benchmark: scatter_add (M2) performance.
 *
 * Measures scatter_add throughput on WASM (sequential) and comparison
 * with manual alternatives.
 *
 * Run with:
 *   pnpm vitest bench bench/scatter-add.bench.ts
 */
import {
  defaultDevice,
  DType,
  init,
  jit,
  numpy as np,
  scatterAdd,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

await init("wasm");

describe("scatter_add WASM", () => {
  test("updates → positions", async ({ bench }) => {
    defaultDevice("wasm");
    using stack = new DisposableStack();
    const registrations = [];
    for (const [updates, positions] of [
      [1000, 100],
      [10000, 100],
      [100000, 1000],
    ] as const) {
      const data = stack.use(np.ones([updates]));
      const indices = stack.use(
        np.array(
          Array.from({ length: updates }, (_, i) => i % positions),
          { dtype: DType.Int32 },
        ),
      );
      const scatter = stack.use(
        jit(
          (d: np.Array, idx: np.Array) =>
            scatterAdd(np.zeros([positions]), idx, d, 0) as np.Array,
        ),
      );
      registrations.push(
        bench(`${updates} updates → ${positions} positions`, () => {
          scatter(data, indices).dispose();
        }),
      );
    }
    await bench.compare(...registrations);
  });
});
