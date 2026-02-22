/**
 * M8.1 Benchmark: scatter_add (M2) performance.
 *
 * Measures scatter_add throughput on WASM (sequential) and comparison
 * with manual alternatives.
 *
 * Run with:
 *   pnpm build && pnpm vitest bench bench/scatter-add.bench.ts
 */
import { afterAll, bench, describe } from "vitest";

import {
  defaultDevice,
  DType,
  init,
  type Array as JaxArray,
  jit,
  numpy as np,
} from "../src";
import { scatterAdd } from "../src/frontend/core";

await init("wasm");
defaultDevice("wasm");

describe("scatter_add WASM", () => {
  // --- Small: 1K updates to 100 positions ---
  const data1k = np.ones([1000]);
  const indices1k = np.array(
    Array.from({ length: 1000 }, (_, i) => i % 100),
    { dtype: DType.Int32 },
  );
  afterAll(() => {
    data1k.dispose();
    indices1k.dispose();
  });

  const scatter1k = jit(
    (data: any, idx: any) =>
      scatterAdd(np.zeros([100]), idx, data, 0) as JaxArray,
  );
  (scatter1k(data1k, indices1k) as any).dispose(); // warmup
  afterAll(() => scatter1k.dispose());

  bench("1K updates → 100 positions", () => {
    (scatter1k(data1k, indices1k) as any).dispose();
  });

  // --- Medium: 10K updates to 100 positions ---
  const data10k = np.ones([10000]);
  const indices10k = np.array(
    Array.from({ length: 10000 }, (_, i) => i % 100),
    { dtype: DType.Int32 },
  );
  afterAll(() => {
    data10k.dispose();
    indices10k.dispose();
  });

  const scatter10k = jit(
    (data: any, idx: any) =>
      scatterAdd(np.zeros([100]), idx, data, 0) as JaxArray,
  );
  (scatter10k(data10k, indices10k) as any).dispose();
  afterAll(() => scatter10k.dispose());

  bench("10K updates → 100 positions", () => {
    (scatter10k(data10k, indices10k) as any).dispose();
  });

  // --- Large: 100K updates to 1000 positions ---
  const data100k = np.ones([100000]);
  const indices100k = np.array(
    Array.from({ length: 100000 }, (_, i) => i % 1000),
    { dtype: DType.Int32 },
  );
  afterAll(() => {
    data100k.dispose();
    indices100k.dispose();
  });

  const scatter100k = jit(
    (data: any, idx: any) =>
      scatterAdd(np.zeros([1000]), idx, data, 0) as JaxArray,
  );
  (scatter100k(data100k, indices100k) as any).dispose();
  afterAll(() => scatter100k.dispose());

  bench("100K updates → 1000 positions", () => {
    (scatter100k(data100k, indices100k) as any).dispose();
  });
});
