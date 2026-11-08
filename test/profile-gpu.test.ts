import {
  defaultDevice,
  init,
  jit,
  numpy as np,
  profileGpu,
  profiler,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

const devicesAvailable = await init("webgpu");
const hasWebGPU = devicesAvailable.includes("webgpu");

describe("profileGpu", () => {
  test("throws on non-WebGPU backend", async () => {
    defaultDevice("wasm");
    await expect(profileGpu(() => {})).rejects.toThrow(/WebGPU/);
  });

  describe.skipIf(!hasWebGPU)("WebGPU", () => {
    test("returns timing data for a jit call", async () => {
      defaultDevice("webgpu");
      using a = np.ones([64]);
      const fn = jit((x: np.Array) => x.mul(2).add(1));
      // warm up jit cache
      {
        using _ = fn(a);
      }
      const { result, timing } = await profileGpu(() => fn(a));
      (result as np.Array).dispose();
      fn.dispose();

      expect(timing.passes.length).toBeGreaterThan(0);
      expect(timing.totalMs).toBeGreaterThanOrEqual(0);
      expect(typeof timing.truncated).toBe("boolean");
      expect(timing.truncated).toBe(false);
      expect(timing.passes.some((pass) => pass.grid != null)).toBe(true);
      for (const pass of timing.passes) {
        expect(typeof pass.durationMs).toBe("number");
        expect(pass.durationMs).toBeGreaterThanOrEqual(0);
      }
    });

    test("throws when tracing is already active", async () => {
      defaultDevice("webgpu");
      profiler.startTrace();
      try {
        await expect(profileGpu(() => 42)).rejects.toThrow(
          /stopTrace|tracing/i,
        );
      } finally {
        profiler.stopTrace();
      }
    });

    test("kernels dispatched while profiling are not traced", async () => {
      defaultDevice("webgpu");
      using key = random.key(0);
      using a = random.uniform(key, [4096]);
      await a.blockUntilReady();
      const traced = () =>
        performance
          .getEntriesByType("measure")
          .filter(
            (m) =>
              (m as PerformanceMeasure).detail?.devtools?.trackGroup ===
              "JAX Profiler",
          );
      performance.clearMeasures();

      await profileGpu(async () => {
        profiler.startTrace();
        try {
          using y = a.mul(3);
          await y.data();
        } finally {
          profiler.stopTrace();
        }
      });
      // Trace entries are emitted when their timestamp readback completes;
      // a later readback lets any pending ones arrive.
      await a.data();

      // Profiling owns the passes' timestamp writes; a trace entry here
      // would show timestamps that were never written.
      expect(traced()).toEqual([]);
    });

    test("measures nonzero pass durations", async () => {
      defaultDevice("webgpu");
      using a = np.ones([2048, 2048]);
      const matmul = jit((x: np.Array, y: np.Array) => np.matmul(x, y));
      try {
        {
          using _ = matmul(a, a); // compile outside the profiled region
          await _.blockUntilReady();
        }
        const { timing } = await profileGpu(async () => {
          using c = matmul(a, a);
          await c.blockUntilReady();
        });
        // A 2048² matmul takes over a millisecond on a fast GPU. An invalid
        // QuerySet (too many queries) resolves nothing, and every duration
        // reads as 0.
        expect(timing.passes.length).toBeGreaterThan(0);
        expect(timing.passes[0].durationMs).toBeGreaterThan(0);
        expect(timing.totalMs).toBeGreaterThan(0);
      } finally {
        matmul.dispose();
      }
    });

    test("records at most 2048 passes, then reports truncation", async () => {
      defaultDevice("webgpu");
      using m = np.eye(8);
      // Eager matmuls do not fuse with each other: one pass each.
      const { timing } = await profileGpu(async () => {
        let y = np.ones([8, 8]);
        for (let i = 0; i < 2100; i++) {
          using previous = y;
          y = np.matmul(previous, m);
        }
        await y.data();
        y.dispose();
      });
      expect(timing.passes.length).toBe(2048);
      expect(timing.truncated).toBe(true);
    });

    test("empty body returns no passes", async () => {
      defaultDevice("webgpu");
      const { timing } = await profileGpu(() => 42);
      expect(timing.passes).toEqual([]);
      expect(timing.totalMs).toBe(0);
      expect(timing.truncated).toBe(false);
    });
  });
});
