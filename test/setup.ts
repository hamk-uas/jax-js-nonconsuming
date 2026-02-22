import {
  checkLeaks,
  devices,
  getBackend,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, afterEach, beforeEach, expect } from "vitest";

beforeEach(() => {
  checkLeaks.start();
});

afterEach(() => {
  const result = checkLeaks.stop();
  expect(result.leaked, result.summary).toBe(0);
});

// Tear down background workers so vitest/Playwright can exit cleanly.
// Without this, WasmWorkerPool and OrchestratorWorker threads keep the
// browser tab alive indefinitely after all tests complete.
afterAll(() => {
  for (const dev of devices) {
    try {
      const b = getBackend(dev) as any;
      if (typeof b.destroyWorkers === "function") {
        b.destroyWorkers();
      }
    } catch {
      // Backend may not be initialized; ignore.
    }
  }
});

expect.extend({
  toBeAllclose(
    actual: np.ArrayLike,
    expected: np.ArrayLike,
    options: { rtol?: number; atol?: number } = {},
  ) {
    const { isNot } = this;
    // Don't allocate arrays here — np.allclose handles conversion and disposal
    // of any copies it creates internally. Caller-owned Arrays are left alive.
    const pass = np.allclose(actual, expected, options);
    // Extract JS values for error display without allocating.
    const actualJs =
      actual != null && typeof (actual as np.Array).js === "function"
        ? (actual as np.Array).js()
        : actual;
    const expectedJs =
      expected != null && typeof (expected as np.Array).js === "function"
        ? (expected as np.Array).js()
        : expected;
    return {
      pass,
      message: () => `expected array to be${isNot ? " not" : ""} allclose`,
      actual: actualJs,
      expected: expectedJs,
    };
  },
  toBeWithinRange(actual: number, min: number, max: number) {
    const { isNot } = this;
    const pass = actual >= min && actual <= max;
    return {
      pass,
      message: () =>
        `expected ${actual} to be${isNot ? " not" : ""} within range [${min}, ${max}]`,
      actual,
      expected: `[${min}, ${max}]`,
    };
  },
});
