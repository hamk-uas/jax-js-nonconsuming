/**
 * WebGPU error capture for tests (R21).
 *
 * An invalid WebGPU call does not throw; it produces an error that the
 * backend merely logs. `test/setup.ts` uses this module so that such errors
 * fail tests:
 *
 * - Every test runs inside error scopes on the jax-js device; errors from its
 *   own calls fail it.
 * - Every GPUDevice created in the test file gets an `uncapturederror`
 *   listener when it is created. Errors outside any test, e.g. while the test
 *   module loads or from work finishing after a test, fail the next test, or
 *   the file after its last test. (The listener used to be attached at the
 *   first test, which missed errors raised while the module loaded: phase 0
 *   review round 2, F2.1.)
 *
 * A test that provokes an error on purpose declares it with
 * `expectWebGpuError()`.
 */

const FILTERS: GPUErrorFilter[] = ["validation", "out-of-memory", "internal"];

/** Errors reported outside error scopes and not yet attributed to a test. */
const uncaptured: string[] = [];
/** Of those, the ones that were already pending when the current test began. */
let pendingAtStart: string[] = [];
const expected: RegExp[] = [];
let scopedDevice: GPUDevice | undefined;
/** Every device created in this test file. */
const devices: GPUDevice[] = [];

/**
 * Attach the `uncapturederror` listener to every GPUDevice requested from now
 * on. Must run before anything in the test file requests a device.
 */
export function captureErrorsOfNewDevices(): void {
  if (typeof GPUAdapter === "undefined") return;
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (
    this: GPUAdapter,
    ...args: Parameters<GPUAdapter["requestDevice"]>
  ) {
    const device = await requestDevice.apply(this, args);
    devices.push(device);
    device.addEventListener("uncapturederror", (event) => {
      uncaptured.push(event.error.message);
    });
    return device;
  };
}

/** Start a test: open error scopes on `device`, if there is one. */
export function beginTestGpuErrors(device: GPUDevice | undefined): void {
  pendingAtStart = uncaptured.splice(0);
  scopedDevice = device;
  for (const filter of FILTERS) device?.pushErrorScope(filter);
}

/**
 * End a test: close its error scopes and return the WebGPU errors that fail
 * it, i.e. the errors not declared with `expectWebGpuError()`, and a message
 * for each declared error that did not occur.
 */
export async function endTestGpuErrors(): Promise<string[]> {
  const errors = pendingAtStart.map((m) => `(raised before this test) ${m}`);
  pendingAtStart = [];
  if (scopedDevice) {
    for (let i = 0; i < FILTERS.length; i++) {
      const error = await scopedDevice.popErrorScope();
      if (error) errors.push(error.message);
    }
    scopedDevice = undefined;
  }
  errors.push(...uncaptured.splice(0));

  const unexpected = errors.filter((e) => !expected.some((re) => re.test(e)));
  const missing = expected
    .filter((re) => !errors.some((e) => re.test(e)))
    .map((re) => `expected a WebGPU error matching ${re}, but none occurred`);
  expected.length = 0;
  return [...unexpected, ...missing];
}

/** WebGPU errors raised after the file's last test. */
export async function remainingGpuErrors(): Promise<string[]> {
  // WebGPU reports errors when a device next processes work; make every
  // device do so, so that errors from the last test's trailing work arrive.
  for (const device of devices) {
    device.queue.submit([]);
    await device.queue.onSubmittedWorkDone();
  }
  return uncaptured.splice(0);
}

/**
 * Declare that the current test expects a WebGPU error matching `pattern`
 * (the counterpart of `toThrow` for errors that WebGPU reports
 * asynchronously). Matching errors do not fail the test; the test fails if no
 * such error occurs. Errors pending from before the test count too.
 */
export function expectWebGpuError(pattern: RegExp): void {
  expected.push(pattern);
}
