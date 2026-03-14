// Platform-specific tests that are run on Deno.
//
// There is no WebGL2 support in Deno, but it does have support for WebGPU. It
// can run most jax-js APIs, with the exception of synchronous buffer mapping
// is missing due to reliance on DOM (`OffscreenCanvas`).
//
// Deno blog post on WebGPU: https://deno.com/blog/v1.39

import {
  defaultDevice,
  Device,
  init,
  jit,
  nn,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, expect, suite, test } from "vitest";

const denoDevices: Device[] = ["cpu", "wasm", "webgpu"];
const devicesAvailable = await init(...denoDevices);

suite.each(denoDevices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("basic operations work", async () => {
    using x = np.array([1, 2, 3]);
    using y = x.add(1);
    expect(await y.jsAsync()).toEqual([2, 3, 4]);
  });

  test("two-layer MLP JIT", async () => {
    using x = np.array([
      [1, 2],
      [3, 4],
    ]);
    using w1 = np.array([
      [1, 2],
      [3, 4],
    ]);
    using b1 = np.array([1, 2]);
    using w2 = np.array([
      [1, 2],
      [3, 4],
    ]);
    using b2 = np.array([1, 2]);

    using forward = jit(
      (x: np.Array, w1: np.Array, b1: np.Array, w2: np.Array, b2: np.Array) => {
        const h = nn.relu(np.dot(x, w1).add(b1));
        const y = nn.logSoftmax(np.dot(h, w2).add(b2));
        return y;
      },
    );

    using h = forward(x, w1, b1, w2, b2);
    expect(await h.jsAsync()).toEqual([
      [-21, 0],
      [-41, 0],
    ]);
  });

  test("numerics of isnan() comparisons", async () => {
    using x = np.array([NaN, 1, 2]);
    using y = np.isnan(x);
    expect(await y.jsAsync()).toEqual([true, false, false]);

    // This may fail on Nvidia GPUs due to compiler UB. See:
    // https://github.com/ekzhang/jax-js/issues/85
    using nanToZero = jit((x: np.Array) => np.where(np.isnan(x), 0, x));
    using a = np.array([NaN, 1.0, NaN, 2.0], { dtype: np.float32 });
    using result = nanToZero(a);
    expect(await result.jsAsync()).toEqual([0, 1.0, 0, 2.0]);
  });
});
