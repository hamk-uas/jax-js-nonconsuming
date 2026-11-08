/**
 * Tests for opt-in GPU calibration (P10): `calibrateGpu()` and the calibrated
 * values' effect on JIT caching.
 *
 * Requires WebGPU — skipped when no WebGPU adapter is available.
 */
import {
  _applyCalibration,
  _resetCalibration,
  calibrateGpu,
  clearCaches,
  defaultDevice,
  getBackend,
  getCacheSizes,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterEach, beforeEach, describe, expect, test } from "vitest";

const devicesAvailable = await init("webgpu");
const hasWebGPU = devicesAvailable.includes("webgpu");

const performanceFields = [
  "dispatchOverheadUs",
  "bandwidthGBs",
  "tflops",
  "barrierCostFactor",
  "rOptWords",
] as const;

describe.skipIf(!hasWebGPU)("GPU calibration", () => {
  beforeEach(() => {
    defaultDevice("webgpu");
  });

  afterEach(() => {
    clearCaches();
    _resetCalibration();
  });

  test('init("webgpu") does not calibrate', async () => {
    await init("webgpu");
    const caps = getBackend("webgpu").capabilities;
    expect(caps.calibrated).toBeFalsy();
    for (const field of performanceFields) expect(caps[field]).toBeUndefined();
  });

  test("calibrateGpu measures plausible values and keeps device limits", async () => {
    const caps = getBackend("webgpu").capabilities;
    const limitsBefore = {
      maxComputeWorkgroupSizeX: caps.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize: caps.maxComputeWorkgroupStorageSize,
      shaderF16: caps.shaderF16,
      subgroups: caps.subgroups,
    };

    const run = calibrateGpu();
    expect(calibrateGpu()).toBe(run); // concurrent callers share one run
    await run;

    expect(caps.calibrated).toBe(true);
    expect(caps.dispatchOverheadUs!).toBeGreaterThanOrEqual(1);
    expect(caps.dispatchOverheadUs!).toBeLessThan(10_000);
    expect(caps.bandwidthGBs!).toBeGreaterThanOrEqual(1);
    expect(caps.bandwidthGBs!).toBeLessThan(5_000);
    expect(caps.tflops!).toBeGreaterThanOrEqual(0.01);
    expect(caps.tflops!).toBeLessThan(500);
    expect(caps.barrierCostFactor!).toBeGreaterThan(0);
    expect([16, 32, 48, 64, 96, 128, 192]).toContain(caps.rOptWords);
    expect(caps).toMatchObject(limitsBefore);

    // Once calibrated, further calls do not measure again; a new measurement
    // would almost surely give different values.
    const measured = { ...caps };
    await calibrateGpu();
    expect({ ...caps }).toEqual(measured);
  });

  test("calibrated values select their own JIT cache entries", () => {
    using x = np.ones([64]);
    const f = jit((a: np.Array) => a.mul(2));
    const compiledPrograms = () => getCacheSizes().jitCompile;
    try {
      f(x).dispose();
      const conservative = compiledPrograms();
      f(x).dispose();
      expect(compiledPrograms()).toBe(conservative);

      _applyCalibration({
        dispatchOverheadUs: 20,
        bandwidthGBs: 200,
        tflops: 10,
        barrierCostFactor: 2,
        rOptWords: 64,
      });
      f(x).dispose();
      expect(compiledPrograms()).toBe(conservative + 1);

      _resetCalibration();
      const caps = getBackend("webgpu").capabilities;
      expect(caps.calibrated).toBe(false);
      for (const field of performanceFields)
        expect(caps[field]).toBeUndefined();
      f(x).dispose();
      expect(compiledPrograms()).toBe(conservative + 1);
    } finally {
      f.dispose();
    }
  });
});
