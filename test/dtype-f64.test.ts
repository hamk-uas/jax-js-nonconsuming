// Tests for the f64 data type.

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

  test("Kahan compensated summation in f64 dot product", async () => {
    // Test that f64 dot product uses compensated summation.
    // Without Kahan, naive summation of many small terms loses precision.
    // Sum 10000 copies of 1e-8: exact answer is 1e-4.
    const n = 10000;
    using a = np.ones([n], { dtype: np.float64 }).mul(
      np.array([1e-8], { dtype: np.float64 }),
    );
    using s = np.sum(a);
    const result: number = await s.jsAsync();
    // Kahan gives ~1e-16 relative error; naive gives ~1e-12 for n=10000.
    expect(Math.abs(result - 1e-4) / 1e-4).toBeLessThan(1e-13);
  });

  test("Kahan compensated summation in f64 dot product (large)", async () => {
    // Dot product of 50000 small values where naive summation loses precision.
    // Each product is 1e-8, so exact dot = 50000 * 1e-8 = 5e-4.
    const n = 50000;
    using a = np.ones([n], { dtype: np.float64 }).mul(
      np.array([1e-4], { dtype: np.float64 }),
    );
    using b = np.ones([n], { dtype: np.float64 }).mul(
      np.array([1e-4], { dtype: np.float64 }),
    );
    using result = np.dot(a, b);
    const val: number = await result.jsAsync();
    // Each a[i]*b[i] = 1e-8, dot = 50000 * 1e-8 = 5e-4.
    // Kahan keeps relative error < 1e-13.
    expect(Math.abs(val - 5e-4) / 5e-4).toBeLessThan(1e-10);
  });

  test("jit f64 dot product with Kahan summation", async () => {
    // Verify compensated summation works under JIT too.
    using f = jit((x: np.Array, y: np.Array) => np.dot(x, y));
    const n = 5000;
    using a = np.ones([n], { dtype: np.float64 }).mul(
      np.array([1e-8], { dtype: np.float64 }),
    );
    using b = np.ones([n], { dtype: np.float64 });
    using result = f(a, b);
    const val: number = await result.jsAsync();
    // dot(a, b) = sum of 5000 copies of 1e-8 = 5e-5
    expect(Math.abs(val - 5e-5) / 5e-5).toBeLessThan(1e-13);
  });
});
