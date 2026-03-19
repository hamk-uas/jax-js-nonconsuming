/**
 * Smoke test for the Mandelbrot demo variants.
 * Exercises all four calculateMandelbrot* functions under the checkLeaks
 * harness (setup.ts wraps every test with checkLeaks.start/stop).
 */
import { init } from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

import {
  calculateMandelbrot,
  calculateMandelbrotForiLoop,
  calculateMandelbrotJitLoop,
  calculateMandelbrotScan,
  height,
  width,
} from "../website/src/routes/mandelbrot/mandelbrot";

await init();

describe("mandelbrot demo — leak-free", () => {
  // Use a small iteration count; this is a leak test, not a correctness benchmark.
  const iters = 3;

  // meshgrid([x(width), y(height)]) produces [height, width] arrays
  const expectedShape = [height, width];

  it("calculateMandelbrot (eager jit loop)", () => {
    using result = calculateMandelbrot(iters);
    expect(result.shape).toEqual(expectedShape);
  });

  it("calculateMandelbrotJitLoop", () => {
    using result = calculateMandelbrotJitLoop(iters);
    expect(result.shape).toEqual(expectedShape);
  });

  it("calculateMandelbrotScan", () => {
    using result = calculateMandelbrotScan(iters);
    expect(result.shape).toEqual(expectedShape);
  });

  it("calculateMandelbrotForiLoop", () => {
    using result = calculateMandelbrotForiLoop(iters);
    expect(result.shape).toEqual(expectedShape);
  });
});
