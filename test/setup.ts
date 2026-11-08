import {
  checkLeaks,
  type Device,
  devices,
  getBackend,
  getWebGPUDevice,
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

// WebGPU errors: an invalid call does not throw, it only produces an error
// event, which the backend merely logs. Each test runs inside error scopes so
// that errors from its own calls fail it; errors outside any test (module
// setup, work completing after a test) are collected by a listener and fail
// the next test, or the file.
const gpuErrorFilters: GPUErrorFilter[] = [
  "validation",
  "out-of-memory",
  "internal",
];
const uncapturedGpuErrors: string[] = [];
const listenedDevices = new WeakSet<GPUDevice>();
let scopedDevice: GPUDevice | undefined;

function webGpuDevice(): GPUDevice | undefined {
  // getWebGPUDevice() throws until a test file has initialized WebGPU.
  try {
    return getWebGPUDevice();
  } catch {
    return undefined;
  }
}

function beginGpuErrorScopes(): void {
  scopedDevice = webGpuDevice();
  if (!scopedDevice) return;
  if (!listenedDevices.has(scopedDevice)) {
    listenedDevices.add(scopedDevice);
    scopedDevice.addEventListener("uncapturederror", (event) => {
      uncapturedGpuErrors.push(event.error.message);
    });
  }
  for (const filter of gpuErrorFilters) scopedDevice.pushErrorScope(filter);
}

async function endGpuErrorScopes(): Promise<string[]> {
  const errors = uncapturedGpuErrors.splice(0);
  if (scopedDevice) {
    for (let i = 0; i < gpuErrorFilters.length; i++) {
      const error = await scopedDevice.popErrorScope();
      if (error) errors.push(error.message);
    }
    scopedDevice = undefined;
  }
  return errors;
}

beforeEach(() => {
  // Enable ownership diagnostics. JIT compile logs and [jax-js] warnings
  // are suppressed by vitest for passing tests; shown on failure for
  // debugging. Ownership warnings surface via the library regression test.
  setDebug(1);
  beginGpuErrorScopes();
  checkLeaks.start();
});

afterEach(async () => {
  // Both checks run before either assertion, so the scopes are always popped.
  const gpuErrors = await endGpuErrorScopes();
  const result = checkLeaks.stop();
  expect(gpuErrors, "WebGPU errors").toEqual([]);
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
  expect(
    uncapturedGpuErrors.splice(0),
    "WebGPU errors after the last test",
  ).toEqual([]);
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
