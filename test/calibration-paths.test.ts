/**
 * Calibration path-selection tests.
 *
 * Verifies that calibration-informed JIT decisions (evaluateTotalCost consumers)
 * produce performance at least as good as conservative-default decisions.
 *
 * Covers three calibration-sensitive paths:
 *   1. tuneWebgpu — cooperative group size + local tiling for reductions
 *   2. foriLoopToBlockMap — 1D/2D block shape selection
 *   3. associativeScan — block size selection (256/128/64/32)
 *
 * Requires WebGPU — run via gpu-test.sh or GPU-specific vitest configs:
 *   scripts/gpu-test.sh run test/calibration-paths.test.ts
 *   pnpm vitest run test/calibration-paths.test.ts -c test/vitest.nvidia.config.ts
 */
import {
  _lastForiRewritten,
  _setCalibrationState,
  blockUntilReady,
  calibrateGpu,
  clearCaches,
  defaultDevice,
  getBackend,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterEach, describe, expect, test } from "vitest";

// Start uncalibrated so we control calibration state per-test.
_setCalibrationState("off");
const devicesAvailable = await init("webgpu", "wasm");
const hasWebGPU = devicesAvailable.includes("webgpu");

/** Measure median wall-clock time over `runs` iterations (ms). */
async function measureMs(fn: () => Promise<void>, runs = 7): Promise<number> {
  // Warmup
  await fn();
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

describe.skipIf(!hasWebGPU)("calibration path selection", () => {
  afterEach(() => {
    clearCaches();
    _setCalibrationState("off");
  });

  /**
   * Test 1: Large reduction — tuneWebgpu cooperative groups + local tiling.
   *
   * sum(N) for large N exercises the cooperative-group selection path.
   * Calibrated rOptWords/tflops/bandwidth should pick group sizes ≥ conservative.
   */
  test("reduction: calibrated >= conservative", async () => {
    defaultDevice("webgpu");
    const N = 1_000_000;
    using x = np.ones([N]);

    // --- Uncalibrated (conservative defaults) ---
    _setCalibrationState("off");
    clearCaches();
    const fUnc = jit((a: np.Array) => np.sum(a));
    const msConservative = await measureMs(async () => {
      using r = fUnc(x);
      await r.blockUntilReady();
    });
    fUnc.dispose();

    // --- Calibrated ---
    _setCalibrationState("pending");
    await calibrateGpu();
    clearCaches();
    const fCal = jit((a: np.Array) => np.sum(a));
    const msCal = await measureMs(async () => {
      using r = fCal(x);
      await r.blockUntilReady();
    });
    fCal.dispose();

    console.log(
      `reduction: conservative=${msConservative.toFixed(2)}ms ` +
        `calibrated=${msCal.toFixed(2)}ms ` +
        `ratio=${(msCal / msConservative).toFixed(2)}`,
    );
    // Calibrated should not be pathologically slower (≤1.5× tolerance for
    // measurement noise on small workloads dominated by dispatch overhead).
    expect(msCal).toBeLessThan(msConservative * 1.5);
  });

  /**
   * Test 2: foriLoop rewrite — calibrated block shape selection.
   *
   * Pointwise foriLoop on 2D arrays: evaluateTotalCost picks among
   * [32×32, 16×16, 16×8, 8×8] block shapes.
   */
  test("foriLoop blockMap: calibrated >= conservative", async () => {
    defaultDevice("webgpu");
    const S = 128;

    const body = (a: np.Array, b: np.Array): [np.Array, np.Array] => {
      const result = lax.foriLoop(
        0,
        20,
        (_i: np.Array, carry: [np.Array, np.Array]) => {
          const [x, y] = carry;
          const nx = x.add(y.mul(0.01));
          const ny = y.sub(x.mul(0.01));
          return [nx, ny] as [np.Array, np.Array];
        },
        [a, b] as [np.Array, np.Array],
      );
      return result;
    };

    using a = np.ones([S, S]);
    using b = np.ones([S, S]);

    // --- Uncalibrated ---
    _setCalibrationState("off");
    clearCaches();
    const fUnc = jit(body);
    const msConservative = await measureMs(async () => {
      const [ra, rb] = fUnc(a, b);
      await blockUntilReady([ra, rb]);
      ra.dispose();
      rb.dispose();
    });
    const rewroteUnc = _lastForiRewritten();
    fUnc.dispose();

    // --- Calibrated ---
    _setCalibrationState("pending");
    await calibrateGpu();
    clearCaches();
    const fCal = jit(body);
    const msCal = await measureMs(async () => {
      const [ra, rb] = fCal(a, b);
      await blockUntilReady([ra, rb]);
      ra.dispose();
      rb.dispose();
    });
    const rewroteCal = _lastForiRewritten();
    fCal.dispose();

    console.log(
      `foriLoop: conservative=${msConservative.toFixed(2)}ms (rewrite=${rewroteUnc}) ` +
        `calibrated=${msCal.toFixed(2)}ms (rewrite=${rewroteCal}) ` +
        `ratio=${(msCal / msConservative).toFixed(2)}`,
    );
    // Both should rewrite (shape is eligible), calibrated should not regress.
    expect(rewroteUnc).toBe(true);
    expect(rewroteCal).toBe(true);
    expect(msCal).toBeLessThan(msConservative * 1.5);
  });

  /**
   * Test 3: Associative scan — calibrated block size selection.
   *
   * Cumulative sum via associativeScan: evaluateTotalCost picks among
   * [256, 128, 64, 32] block sizes for the Kogge-Stone WebGPU shader.
   */
  test("associativeScan: calibrated >= conservative", async () => {
    defaultDevice("webgpu");
    const N = 10_000;
    using x = np.ones([N]);

    const scanBody = (a: np.Array, b: np.Array) => a.add(b);

    // --- Uncalibrated ---
    _setCalibrationState("off");
    clearCaches();
    const fUnc = jit((arr: np.Array) => lax.associativeScan(scanBody, arr));
    const msConservative = await measureMs(async () => {
      using r = fUnc(x);
      await r.blockUntilReady();
    });
    fUnc.dispose();

    // --- Calibrated ---
    _setCalibrationState("pending");
    await calibrateGpu();
    clearCaches();
    const fCal = jit((arr: np.Array) => lax.associativeScan(scanBody, arr));
    const msCal = await measureMs(async () => {
      using r = fCal(x);
      await r.blockUntilReady();
    });
    fCal.dispose();

    console.log(
      `associativeScan: conservative=${msConservative.toFixed(2)}ms ` +
        `calibrated=${msCal.toFixed(2)}ms ` +
        `ratio=${(msCal / msConservative).toFixed(2)}`,
    );
    expect(msCal).toBeLessThan(msConservative * 1.5);
  });

  /**
   * Test 4: Calibrated values are physically plausible.
   *
   * After calibration, the measured hardware characteristics should be
   * within realistic bounds for any GPU.
   */
  test("calibrated values are plausible", async () => {
    defaultDevice("webgpu");
    _setCalibrationState("pending");
    await calibrateGpu();

    const caps = getBackend().capabilities;
    expect(caps.calibrated).toBe(true);

    // Dispatch overhead: 1–500 µs (covers native PCIe to eGPU)
    expect(caps.dispatchOverheadUs).toBeGreaterThan(0.5);
    expect(caps.dispatchOverheadUs).toBeLessThan(500);

    // Bandwidth: 1–2000 GB/s (covers mobile to HBM3e)
    expect(caps.bandwidthGBs).toBeGreaterThan(1);
    expect(caps.bandwidthGBs).toBeLessThan(2000);

    // TFLOPS: 0.1–200 (covers mobile iGPU to H100)
    expect(caps.tflops).toBeGreaterThan(0.1);
    expect(caps.tflops).toBeLessThan(200);

    // Register budget: 16–1024 words (covers gen-9 to modern discrete)
    expect(caps.rOptWords).toBeGreaterThanOrEqual(16);
    expect(caps.rOptWords).toBeLessThanOrEqual(1024);

    console.log(
      `calibrated: dispatch=${caps.dispatchOverheadUs?.toFixed(1)}µs ` +
        `BW=${caps.bandwidthGBs?.toFixed(1)}GB/s ` +
        `TFLOPS=${caps.tflops?.toFixed(2)} ` +
        `rOpt=${caps.rOptWords}`,
    );
  });
});
