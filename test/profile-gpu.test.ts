import {
  defaultDevice,
  init,
  jit,
  numpy as np,
  profileGpu,
  profiler,
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
        await expect(profileGpu(() => 42)).rejects.toThrow(/stopTrace|tracing/i);
      } finally {
        profiler.stopTrace();
      }
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
