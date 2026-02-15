// Tests for the f16 data type.

import {
  defaultDevice,
  grad,
  init,
  jit,
  jvp,
  nn,
  numpy as np,
} from "@jax-js-nonconsuming/jax";
import { beforeEach, expect, suite, test } from "vitest";

// f16 is currently only supported on WebGPU.
const devices = ["cpu", "webgpu"] as const;

const devicesAvailable = await init(...devices);

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("create and access f16 array", async () => {
    using a = np.array([1.5, 2.5, 3.5], { dtype: np.float16 });
    expect(a.dtype).toBe(np.float16);
    expect(a.shape).toEqual([3]);
    expect(await a.data()).toEqual(new Float16Array([1.5, 2.5, 3.5]));
    expect(a.dataSync()).toEqual(new Float16Array([1.5, 2.5, 3.5]));
    expect(a.js()).toEqual([1.5, 2.5, 3.5]);
  });

  test("jit of f16 calculation", () => {
    using f = jit((x: np.Array) => np.sum(x.mul(x)));
    using arg = np.arange(10).astype(np.float16);
    using r = f(arg);
    expect(r).toBeAllclose(285);
  });

  test("jvp of f16 calculation", () => {
    const f = (x: np.Array) => x.mul(x);
    using primals = np.array([1.5, 2.5], { dtype: np.float16 });
    using tangents = np.array([1.0, 1.0], { dtype: np.float16 });
    const [y, dy] = jvp(f, [primals], [tangents]);
    using _y = y;
    using _dy = dy;
    expect(y.dtype).toBe(np.float16);
    expect(dy.dtype).toBe(np.float16);
    expect(y.dataSync()).toEqual(new Float16Array([2.25, 6.25]));
    expect(dy.dataSync()).toEqual(new Float16Array([3.0, 5.0]));
  });

  test("gradient of f16 calculation", () => {
    const f = (x: np.Array) => np.sum(x.mul(x));
    const g = grad(f);

    using x = np.array([1.5, 2.5], { dtype: np.float16 });
    using y = g(x);
    expect(y.dtype).toBe(np.float16);
    expect(y.dataSync()).toEqual(new Float16Array([3.0, 5.0]));
  });

  test("erfc() works for f16", () => {
    // nn.gelu() with approximate=false uses erfc().
    using x = np.array([-1.0, 0.0, 1.0], { dtype: np.float16 });
    using y = nn.gelu(x, { approximate: false });
    expect(y.dtype).toBe(np.float16);
    expect(y).toBeAllclose([-0.1587, 0.0, 0.8413], { rtol: 1e-3 });
  });

  test("f16 reductions are performed in f32", () => {
    // dot() / matmul() should be performed in f32 by default
    using x1 = np.array([100, 1, 1, 1, 1, 1, 1, 1, 1], { dtype: np.float16 });
    using y1 = np.dot(x1, x1);
    expect(y1.dtype).toBe(np.float16);
    expect(y1.dataSync()).toEqual(new Float16Array([10008]));

    // sum() should also be performed in f32
    using x2 = np.array([16000, 1, 1, 1, 1, 1, 1, 1, 1], { dtype: np.float16 });
    using y2 = np.sum(x2);
    expect(y2.dtype).toBe(np.float16);
    expect(y2.dataSync()).toEqual(new Float16Array([16008]));
  });
});
