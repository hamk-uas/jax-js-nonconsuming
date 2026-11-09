/**
 * GPU-only correctness tests for O8c command tape ops (DUS, scatter_add, reverse).
 *
 * Verifies that real WebGPU tape activation occurs and produces correct results
 * for each of the three non-kernel step types added in O8c.
 *
 * Run with:
 *   GPU=nvidia scripts/gpu-test.sh run test/command-tape-gpu.test.ts
 *   pnpm vitest run test/command-tape-gpu.test.ts -c test/vitest.nvidia.config.ts
 */
import {
  clearCaches,
  defaultDevice,
  DType,
  init,
  jit,
  lax,
  numpy as np,
  profiler,
  reverse,
  scatterAdd,
  withBatch,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { _getCommandTapeStats } from "../src/frontend/jit";

await init("webgpu");

function vals(a: np.Array): number[] {
  return Array.from(a.dataSync() as Float32Array);
}

function webGpuTraces(): PerformanceMeasure[] {
  return (
    performance.getEntriesByType("measure") as PerformanceMeasure[]
  ).filter(
    (entry) =>
      entry.detail?.devtools?.trackGroup === "JAX Profiler" &&
      entry.detail.devtools.track === "webgpu",
  );
}

describe("O8c command tape GPU correctness", () => {
  beforeAll(() => {
    defaultDevice("webgpu");
  });

  beforeEach(() => {
    clearCaches();
  });

  it("scatter_add activates tape and produces correct results", () => {
    const f = jit(
      (target: np.Array, idx: np.Array, updates: np.Array) =>
        scatterAdd(target, idx, updates, 0) as np.Array,
    );

    using target = np.zeros([8]);
    using idx = np.array([1, 3, 5], { dtype: DType.Int32 });
    using updates = np.array([10, 20, 30]);
    using result = f(target, idx, updates);

    expect(vals(result)).toEqual([0, 10, 0, 20, 0, 30, 0, 0]);

    const stats = _getCommandTapeStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);

    f.dispose();
  });

  it("dynamic_update_slice activates tape and produces correct results", () => {
    const f = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, 2, 0),
    );

    using dst = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
    using src = np.array([90, 91, 92]);
    using result = f(dst, src) as np.Array;

    expect(vals(result)).toEqual([1, 2, 90, 91, 92, 6, 7, 8]);

    const stats = _getCommandTapeStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);

    f.dispose();
  });

  it("reverse activates tape and produces correct results", () => {
    const f = jit((x: np.Array) => reverse(x, 0) as np.Array);

    using x = np.array([10, 20, 30, 40]);
    using result = f(x);

    expect(vals(result)).toEqual([40, 30, 20, 10]);

    const stats = _getCommandTapeStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);

    f.dispose();
  });

  it("traces a warmed JIT command-tape dispatch", async () => {
    using f = jit((x: np.Array) => x.mul(2).add(1));
    using x = np.array([1, 2, 3, 4]);
    {
      using warmup = f(x);
      await warmup.blockUntilReady();
    }
    expect(_getCommandTapeStats().length).toBeGreaterThan(0);
    performance.clearMeasures();

    profiler.startTrace();
    try {
      using result = f(x);
      expect(Array.from(await result.data())).toEqual([3, 5, 7, 9]);
    } finally {
      profiler.stopTrace();
    }

    try {
      await expect.poll(() => webGpuTraces().length).toBe(1);
      const trace = webGpuTraces()[0];
      expect(trace.name).toMatch(/^Kernel\[/);
      expect(trace.duration).toBeGreaterThanOrEqual(0);
      expect(trace.detail.devtools.properties).toEqual(
        expect.arrayContaining([
          ["passes", "1"],
          ["source", expect.stringContaining("@compute")],
        ]),
      );
    } finally {
      performance.clearMeasures();
    }
  });

  it("traces all passes of a multi-pass routine on the tape", async () => {
    using f = jit((x: np.Array) => np.sort(x));
    using x = np.array(Array.from({ length: 4096 }, (_, i) => 4095 - i));
    {
      using warmup = f(x);
      await warmup.blockUntilReady();
    }
    expect(_getCommandTapeStats().length).toBeGreaterThan(0);
    performance.clearMeasures();

    profiler.startTrace();
    try {
      using result = f(x);
      expect(Array.from(await result.data())).toEqual(
        Array.from({ length: 4096 }, (_, i) => i),
      );
    } finally {
      profiler.stopTrace();
    }
    try {
      await expect.poll(() => webGpuTraces().length).toBe(1);
      const trace = webGpuTraces()[0];
      expect(trace.name).toMatch(/sort/i);
      expect(
        Number(
          trace.detail.devtools.properties.find(
            ([key]: [string, string]) => key === "passes",
          )[1],
        ),
      ).toBeGreaterThan(1);
    } finally {
      performance.clearMeasures();
    }
  });

  it("rolls timestamp batches over before stopping inside withBatch", async () => {
    using f = jit((x: np.Array) => x.add(1));
    using x = np.ones([8192]);
    {
      using warmup = f(x);
      await warmup.blockUntilReady();
    }
    performance.clearMeasures();
    try {
      // One more than a QuerySet's 2048 begin/end pairs, all in one encoder.
      withBatch(() => {
        profiler.startTrace();
        try {
          for (let i = 0; i < 2049; i++) {
            using _result = f(x);
          }
        } finally {
          profiler.stopTrace();
        }
      });
      await expect.poll(() => webGpuTraces().length).toBe(2049);
      expect(webGpuTraces().every((trace) => trace.duration >= 0)).toBe(true);
      expect(webGpuTraces().some((trace) => trace.duration > 0)).toBe(true);
    } finally {
      performance.clearMeasures();
    }
  });
});
