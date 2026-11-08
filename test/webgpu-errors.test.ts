/**
 * The test harness catches WebGPU errors (test/webgpu-errors.ts, R21).
 *
 * Each test declares the error it provokes with `expectWebGpuError()`, so it
 * passes only if the harness saw the error. Without the capture, the
 * declaration is unmet and the test fails.
 */
import { getWebGPUDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

import { expectWebGpuError } from "./webgpu-errors";

const hasWebGPU = (await init("webgpu")).includes("webgpu");

// WebGPU caps a QuerySet at 4096 queries; occlusion queries need no feature.
const invalidQuerySet = { type: "occlusion", count: 8192 } as const;
const invalidQuerySetError = /8192/;

// An invalid call while the module loads, before any test hook has run (the
// reproduction of phase 0 review round 2, F2.1). Wait until WebGPU has
// reported it, so the error arrives during module loading.
if (hasWebGPU) {
  const device = getWebGPUDevice();
  const reported = new Promise<void>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("WebGPU did not report the invalid QuerySet")),
      5000,
    );
    device.addEventListener(
      "uncapturederror",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
  device.createQuerySet(invalidQuerySet).destroy();
  // Dawn delivers errors when the device next processes work.
  device.queue.submit([]);
  await device.queue.onSubmittedWorkDone();
  await reported;
}

describe.skipIf(!hasWebGPU)("WebGPU errors fail tests", () => {
  test("an error raised while the module loads is attributed to the next test", () => {
    expectWebGpuError(invalidQuerySetError);
  });

  test("an error raised inside a test is attributed to it", () => {
    expectWebGpuError(invalidQuerySetError);
    getWebGPUDevice().createQuerySet(invalidQuerySet).destroy();
  });
});
