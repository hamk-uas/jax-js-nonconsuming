import {
  defaultDevice,
  type Device,
  devices,
  init,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, suite } from "vitest";

let _initResult: Device[] | undefined;

/**
 * Run a test suite once per device. Automatically calls `init()`, skips
 * unavailable backends, and sets `defaultDevice` before each test.
 *
 * @param fn      Callback that registers tests (receives the current device)
 * @param devices Optional device list — defaults to the full `devices` array
 */
export async function deviceSuite(
  fn: (device: Device) => void,
  deviceList?: readonly Device[],
): Promise<void> {
  _initResult ??= await init();
  const available = _initResult;
  const list = deviceList ?? devices;
  suite.each(list)("device:%s", (device) => {
    const skipped = !available.includes(device);
    beforeEach(({ skip }) => {
      if (skipped) skip();
      defaultDevice(device);
    });
    fn(device);
  });
}
