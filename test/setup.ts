import {
  checkLeaks,
  type Device,
  devices,
  getBackend,
  init,
  numpy as np,
  setDebug,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, afterEach, beforeEach, expect, inject } from "vitest";

// Backends this run requires (see requiredDevices() in test/gpu-config.ts).
// A required backend that fails to initialize fails the file here, instead of
// turning every test for it into a skip.
const required = inject("requiredDevices") as Device[];
if (required.length > 0) {
  const initialized = await init(...required);
  const missing = required.filter((d) => !initialized.includes(d));
  if (missing.length > 0) {
    throw new Error(
      `Required backend(s) did not initialize: ${missing.join(", ")}. ` +
        "Set JAX_REQUIRE_DEVICES to change the requirement.",
    );
  }
}

beforeEach(() => {
  // Enable ownership diagnostics. JIT compile logs and [jax-js] warnings
  // are suppressed by vitest for passing tests; shown on failure for
  // debugging. Ownership warnings surface via the library regression test.
  setDebug(1);
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
    options: { rtol?: number; atol?: number; equalNaN?: boolean } = {},
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
