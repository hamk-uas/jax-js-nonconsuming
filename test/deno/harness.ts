/**
 * Deno test harness with memory leak detection for jax-js WebGPU tests.
 *
 * Usage:
 *   import { withLeakCheck, assertNoLeaks, getSlotCount } from "./harness.ts";
 *
 *   Deno.test("my test", withLeakCheck(async () => {
 *     // test code - all arrays should be disposed by end
 *   }));
 *
 * Or manually:
 *   Deno.test("my test", async () => {
 *     const before = getSlotCount();
 *     // test code
 *     assertNoLeaks(before, "test name");
 *   });
 */

import {
  checkLeaks,
  defaultDevice,
  getBackend,
  init,
  numpy as np,
} from "../../dist/index.js";

export { checkLeaks };

// Cached init result
let _initResult: string[] | null = null;

/**
 * Ensure jax-js is initialized with WebGPU and set as default device.
 * Caches the init result for performance across tests.
 */
export async function initWebGPU(): Promise<boolean> {
  if (!_initResult) _initResult = await init();
  if (!_initResult.includes("webgpu")) return false;
  defaultDevice("webgpu");
  return true;
}

/**
 * Check if WebGPU is available in this environment.
 */
export const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

/**
 * Assert that two arrays (or ArrayLike values) are element-wise close.
 * Mirrors the Vitest `toBeAllclose` custom matcher.
 *
 * NOTE: This calls np.allclose which uses dataSync() internally.
 * On Deno WebGPU, dataSync() requires OffscreenCanvas (not available).
 * Use assertAllcloseAsync() for Deno WebGPU tests instead.
 */
export function assertAllclose(
  actual: any,
  expected: any,
  options?: { rtol?: number; atol?: number },
): void {
  // eslint-disable-next-line jax-js/require-using -- np.allclose returns a plain boolean, not an array
  const pass = np.allclose(actual, expected, options);
  if (!pass) {
    const actualJs =
      actual != null && typeof actual.js === "function" ? actual.js() : actual;
    const expectedJs =
      expected != null && typeof expected.js === "function"
        ? expected.js()
        : expected;
    throw new Error(
      `Arrays not allclose:\n  actual:   ${JSON.stringify(actualJs)}\n  expected: ${JSON.stringify(expectedJs)}`,
    );
  }
}

/**
 * Async version of assertAllclose that uses .data() instead of dataSync().
 * Safe for Deno WebGPU where OffscreenCanvas is not available.
 */
export async function assertAllcloseAsync(
  actual: any,
  expected: any,
  options?: { rtol?: number; atol?: number },
): Promise<void> {
  const rtol = options?.rtol ?? 1e-5;
  const atol = options?.atol ?? 1e-6;

  // Convert actual to flat array
  let actualArr: number[];
  if (actual != null && typeof actual.data === "function") {
    const d = await actual.data();
    actualArr = Array.from(d);
  } else if (actual != null && typeof actual.js === "function") {
    actualArr = actual.js().flat(Infinity);
  } else if (Array.isArray(actual)) {
    actualArr = actual.flat(Infinity);
  } else {
    actualArr = [actual];
  }

  // Convert expected to flat array
  let expectedArr: number[];
  if (expected != null && typeof expected.data === "function") {
    const d = await expected.data();
    expectedArr = Array.from(d);
  } else if (expected != null && typeof expected.js === "function") {
    expectedArr = expected.js().flat(Infinity);
  } else if (Array.isArray(expected)) {
    expectedArr = expected.flat(Infinity);
  } else {
    expectedArr = [expected];
  }

  if (actualArr.length !== expectedArr.length) {
    throw new Error(
      `Arrays have different lengths: actual=${actualArr.length}, expected=${expectedArr.length}`,
    );
  }

  for (let i = 0; i < actualArr.length; i++) {
    const diff = Math.abs(actualArr[i] - expectedArr[i]);
    const tol = atol + rtol * Math.abs(expectedArr[i]);
    if (diff > tol) {
      throw new Error(
        `Arrays not allclose at index ${i}: actual=${actualArr[i]}, expected=${expectedArr[i]}, diff=${diff}, tol=${tol}\n  actual:   ${JSON.stringify(actualArr)}\n  expected: ${JSON.stringify(expectedArr)}`,
      );
    }
  }
}

/**
 * Get the current slot count from the active backend.
 * Returns 0 if no backend is initialized.
 */
export function getSlotCount(): number {
  try {
    const device = defaultDevice();
    if (!device) return 0;
    const backend = getBackend(device);
    return backend.slotCount();
  } catch {
    return 0;
  }
}

/**
 * Assert that no memory was leaked since the baseline.
 * @param baselineSlots - Slot count before the test
 * @param testName - Optional test name for error message
 */
export function assertNoLeaks(baselineSlots: number, testName?: string): void {
  const currentSlots = getSlotCount();
  const leaked = currentSlots - baselineSlots;
  if (leaked > 0) {
    const msg = testName
      ? `Memory leak in "${testName}": ${leaked} slot(s) leaked (${baselineSlots} -> ${currentSlots})`
      : `Memory leak: ${leaked} slot(s) leaked (${baselineSlots} -> ${currentSlots})`;
    throw new Error(msg);
  }
}

/**
 * Wrapper that adds memory leak checking to a test function.
 * Use with Deno.test:
 *
 *   Deno.test("my test", withLeakCheck(async () => { ... }));
 *
 * @param fn - The test function
 * @param options - Optional settings
 */
export function withLeakCheck(
  fn: () => void | Promise<void>,
  options?: { allowedLeaks?: number },
): () => Promise<void> {
  const allowedLeaks = options?.allowedLeaks ?? 0;

  return async () => {
    checkLeaks.start();
    let testError: unknown;
    try {
      await fn();
    } catch (e) {
      testError = e;
    }
    const report = checkLeaks.stop();
    if (testError) throw testError;
    if (report.leaked > allowedLeaks) {
      const details =
        report.details.length > 0
          ? `\n${report.details.map((d: string) => `  - ${d}`).join("\n")}`
          : "";
      throw new Error(
        `Memory leak: ${report.leaked} slot(s) leaked (allowed=${allowedLeaks})${details}`,
      );
    }
  };
}

/**
 * Test options with memory leak checking.
 */
export interface LeakCheckTestOptions extends Deno.TestDefinition {
  /** Number of leaked slots allowed (default: 0) */
  allowedLeaks?: number;
}

/**
 * Enhanced test registration with automatic leak checking.
 *
 * Usage:
 *   leakCheckTest({
 *     name: "my test",
 *     fn: async () => { ... },
 *   });
 */
export function leakCheckTest(options: LeakCheckTestOptions): void {
  const { allowedLeaks, fn, ...rest } = options;
  Deno.test({
    ...rest,
    fn: withLeakCheck(fn, { allowedLeaks }),
  });
}
