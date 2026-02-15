// Tests for the f64 data type.

import {
  defaultDevice,
  grad,
  init,
  jit,
  jvp,
  nn,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, expect, suite, test } from "vitest";

// f64 is currently only supported on WebAssembly.
const devices = ["cpu", "wasm"] as const;

const devicesAvailable = await init(...devices);

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("create and access f64 array", async () => {
    using a = np.array([1.5, 2.5, 3.5], { dtype: np.float64 });
    expect(a.dtype).toBe(np.float64);
    expect(a.shape).toEqual([3]);
    expect(await a.data()).toEqual(new Float64Array([1.5, 2.5, 3.5]));
    expect(a.dataSync()).toEqual(new Float64Array([1.5, 2.5, 3.5]));
    expect(a.js()).toEqual([1.5, 2.5, 3.5]);
  });

  test("jit of f64 calculation", () => {
    using f = jit((x: np.Array) => np.sum(x.mul(x)));
    using arg = np.arange(10).astype(np.float64);
    using r = f(arg);
    expect(r).toBeAllclose(285);
  });

  test("jvp of f64 calculation", () => {
    const f = (x: np.Array) => x.mul(x);
    using primals = np.array([1.5, 2.5], { dtype: np.float64 });
    using tangents = np.array([1.0, 1.0], { dtype: np.float64 });
    const [y, dy] = jvp(f, [primals], [tangents]);
    using _y = y;
    using _dy = dy;
    expect(y.dtype).toBe(np.float64);
    expect(dy.dtype).toBe(np.float64);
    expect(y.dataSync()).toEqual(new Float64Array([2.25, 6.25]));
    expect(dy.dataSync()).toEqual(new Float64Array([3.0, 5.0]));
  });

  test("gradient of f64 calculation", () => {
    const f = (x: np.Array) => np.sum(x.mul(x));
    const g = grad(f);

    using x = np.array([1.5, 2.5], { dtype: np.float64 });
    using y = g(x);
    expect(y.dtype).toBe(np.float64);
    expect(y.dataSync()).toEqual(new Float64Array([3.0, 5.0]));
  });

  test("erfc() works for f64", () => {
    // nn.gelu() with approximate=false uses erfc().
    using x = np.array([-1.0, 0.0, 1.0], { dtype: np.float64 });
    using y = nn.gelu(x, { approximate: false });
    expect(y.dtype).toBe(np.float64);
    expect(y).toBeAllclose([-0.15865525, 0.0, 0.84134475]);
  });

  test("precision of f64 is high", async () => {
    using a = np.array([1 + 1e-15, 1 + 2e-15], { dtype: np.float64 });
    await a.blockUntilReady();
    using s1 = a.slice(1);
    using s0 = a.slice(0);
    using diff = s1.sub(s0);
    const b: number = await diff.jsAsync();
    expect(b).toBeCloseTo(1e-15, 15);
  });
});
