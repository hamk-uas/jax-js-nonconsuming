import { describe, expect, test } from "vitest";

import type { BackendCapabilities } from "../src/backend";
import { CostFeatures, evaluateTotalCost } from "../src/tuner";

// Fixed performance values, so the tests do not depend on calibration.
const caps: Partial<BackendCapabilities> = {
  dispatchOverheadUs: 25,
  bandwidthGBs: 30,
  tflops: 0.4,
  rOptWords: 48,
  barrierCostFactor: 1,
  adapterVendor: "intel",
  adapterArchitecture: "gen_12lp",
};

const base: CostFeatures = {
  nDispatch: 1,
  nBuffers: 3,
  countAlu: 1e6,
  countMem: 1e5,
  depthPriv: 20,
  sizeShmem: 4096,
  sizeWgsl: 8192,
  parallelism: 64,
  produceCount: 4096,
};

describe("evaluateTotalCost", () => {
  test("cost is normalized per produced element", () => {
    const one = evaluateTotalCost(base, caps);
    const two = evaluateTotalCost(
      { ...base, produceCount: 2 * base.produceCount! },
      caps,
    );
    expect(two).toBeCloseTo(one / 2, 12);
  });

  test("aggregate register pressure past the budget is a cliff", () => {
    // rOptWords 48 gives a budget of 48 × 64 = 3072 register words.
    const fits = evaluateTotalCost({ ...base, depthPriv: 40 }, caps);
    const overflows = evaluateTotalCost(
      { ...base, depthPriv: 40, parallelism: 256 },
      caps,
    );
    // Overflowing the budget 3.3× multiplies the cost by exp(3 × 2.3) ≈ 1000;
    // the latency term alone would make 256 threads cheaper.
    expect(overflows / fits).toBeGreaterThan(100);
  });

  test("per-thread registers past rOptWords cost more", () => {
    const fits = evaluateTotalCost({ ...base, depthPriv: 48 }, caps);
    const spills = evaluateTotalCost({ ...base, depthPriv: 96 }, caps);
    expect(spills).toBeGreaterThan(fits);
  });

  test("shared memory past 16 KiB costs more than below it", () => {
    const small = evaluateTotalCost({ ...base, sizeShmem: 8192 }, caps);
    const large = evaluateTotalCost({ ...base, sizeShmem: 65536 }, caps);
    expect(large).toBeGreaterThan(small);
  });

  test("more than 8 bound buffers costs more", () => {
    const few = evaluateTotalCost({ ...base, nBuffers: 8 }, caps);
    const many = evaluateTotalCost({ ...base, nBuffers: 16 }, caps);
    expect(many).toBeGreaterThan(few);
  });
});
