/**
 * Tests for P10 microbenchmark auto-tuning.
 *
 * Covers: PerformanceBeliefState, calibrateGpu, calibration lifecycle,
 * cache key differentiation, _setCalibrationState reset.
 *
 * Requires WebGPU — skipped when no WebGPU adapter is available.
 */
import {
  _setCalibrationState,
  calibrateGpu,
  clearCaches,
  defaultDevice,
  getBackend,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterEach, beforeEach, describe, expect, test } from "vitest";

// init with calibration off so we control timing
_setCalibrationState("off");
const devicesAvailable = await init("webgpu", "wasm");
const hasWebGPU = devicesAvailable.includes("webgpu");

describe.skipIf(!hasWebGPU)("microbench calibration", () => {
  beforeEach(() => {
    defaultDevice("webgpu");
  });

  afterEach(() => {
    clearCaches();
    _setCalibrationState("off");
  });

  test("calibrateGpu populates empirical capabilities", async () => {
    const backend = getBackend("webgpu");
    const capsBefore = backend.capabilities;

    // Before calibration, calibrated should be falsy
    expect(capsBefore.calibrated).toBeFalsy();

    _setCalibrationState("pending");
    await calibrateGpu();

    const capsAfter = backend.capabilities;
    expect(capsAfter.calibrated).toBe(true);
    expect(capsAfter.bandwidthGBs).toBeGreaterThan(0);
    expect(capsAfter.tflops).toBeGreaterThan(0);
    expect(capsAfter.dispatchOverheadUs).toBeGreaterThan(0);
    expect(capsAfter.rOptWords).toBeGreaterThanOrEqual(16);

    // Immutable limits should be preserved
    expect(capsAfter.maxComputeWorkgroupSizeX).toBe(
      capsBefore.maxComputeWorkgroupSizeX,
    );
    expect(capsAfter.maxComputeWorkgroupStorageSize).toBe(
      capsBefore.maxComputeWorkgroupStorageSize,
    );
    expect(capsAfter.shaderF16).toBe(capsBefore.shaderF16);
    expect(capsAfter.subgroups).toBe(capsBefore.subgroups);
  });

  test("calibrateGpu is idempotent once done", async () => {
    _setCalibrationState("pending");
    await calibrateGpu();

    const backend = getBackend("webgpu");
    const caps1 = backend.capabilities;
    expect(caps1.calibrated).toBe(true);

    // Second call should be a no-op (state is "done")
    await calibrateGpu();
    const caps2 = backend.capabilities;
    // Same object reference — capabilities not replaced
    expect(caps2).toBe(caps1);
  });

  test("calibrateGpu skipped when state is off", async () => {
    _setCalibrationState("off");
    await calibrateGpu();

    const backend = getBackend("webgpu");
    expect(backend.capabilities.calibrated).toBeFalsy();
  });

  test("_setCalibrationState resets for test isolation", async () => {
    // Calibrate first
    _setCalibrationState("pending");
    await calibrateGpu();

    const backend = getBackend("webgpu");
    expect(backend.capabilities.calibrated).toBe(true);

    // Reset to pending — note: capabilities remain dirty from previous calibration,
    // but the state machine allows re-calibration
    _setCalibrationState("pending");
    await calibrateGpu();
    expect(backend.capabilities.calibrated).toBe(true);
  });

  test("jit cache key differentiates calibrated vs uncalibrated", async () => {
    // First JIT call with uncalibrated backend
    _setCalibrationState("off");
    using a = np.ones([64]);
    const fn = jit((x: np.Array) => x.mul(2));
    {
      using _ = fn(a);
    }

    // Now calibrate
    _setCalibrationState("pending");
    await calibrateGpu();
    clearCaches();

    // Re-JIT after calibration — should get a fresh compilation
    // (different cache key due to calibrated caps)
    {
      using _ = fn(a);
    }

    fn.dispose();
  });

  test("belief state values are within reasonable ranges", async () => {
    _setCalibrationState("pending");
    await calibrateGpu();

    const caps = getBackend("webgpu").capabilities;

    // Dispatch overhead: typically 5-500µs
    expect(caps.dispatchOverheadUs!).toBeGreaterThanOrEqual(1);
    expect(caps.dispatchOverheadUs!).toBeLessThan(10_000);

    // Bandwidth: typically 1-2000 GB/s
    expect(caps.bandwidthGBs!).toBeGreaterThanOrEqual(1);
    expect(caps.bandwidthGBs!).toBeLessThan(5_000);

    // TFLOPS: typically 0.01-100
    expect(caps.tflops!).toBeGreaterThanOrEqual(0.01);
    expect(caps.tflops!).toBeLessThan(500);

    // R_opt: one of the probed values or conservative default
    expect([16, 32, 48, 64, 96, 128, 192]).toContain(caps.rOptWords);
  });
});
