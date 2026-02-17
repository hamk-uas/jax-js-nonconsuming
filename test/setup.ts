import { checkLeaks, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { afterEach, beforeEach, expect } from "vitest";

beforeEach(() => {
  checkLeaks.start();
});

afterEach(() => {
  const result = checkLeaks.stop();
  expect(result.leaked, result.summary).toBe(0);
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
