// Make sure refcount and dispose mechanics work correctly.

import { checkLeaks, grad, numpy as np } from "@jax-js-nonconsuming/jax";
import { expect, suite, test } from "vitest";

suite("refcount through grad", () => {
  test("add and sum", () => {
    const f = (x: np.Array) => x.add(x).sum();
    const df = grad(f);

    using x = np.array([1, 2, 3, 4]);
    using r1 = df(x);
    expect(r1.js()).toEqual([2, 2, 2, 2]);
    // grad does NOT consume x — it's still alive
    using r2 = df(x);
    expect(r2.js()).toEqual([2, 2, 2, 2]);
  });

  test("multiply and sum", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    const df = grad(f);

    using x = np.array([1, 2, 3, 4]);
    using r1 = df(x);
    expect(r1.js()).toEqual([2, 4, 6, 8]);
    // grad does NOT consume x — it's still alive
    using r2 = df(x);
    expect(r2.js()).toEqual([2, 4, 6, 8]);
  });
});

suite("refCount property", () => {
  test("initial refCount is 1", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    expect(x.refCount).toBe(1);
    x.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);
  });

  test("refCount increments after .ref", () => {
    const x = np.array([1, 2, 3]);
    expect(x.refCount).toBe(1);
    const y = x.ref;
    expect(x.refCount).toBe(2);
    expect(y.refCount).toBe(2); // same array
    x.dispose();
    y.dispose();
  });

  test("refCount is 0 after final dispose", () => {
    const x = np.array([1, 2, 3]);
    expect(x.refCount).toBe(1);
    x.dispose();
    expect(x.refCount).toBe(0);
  });

  test("refCount is readable on disposed arrays", () => {
    const x = np.array([1, 2, 3]);
    x.dispose();
    // Should not throw - refCount is readable for debugging
    expect(x.refCount).toBe(0);
  });
});
