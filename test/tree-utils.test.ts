// Tests for tree.dispose, tree.makeDisposable, and Array.consumeData.

import { getBackend, numpy as np, tree } from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

suite("tree.dispose", () => {
  test("disposes all arrays in a flat object", () => {
    using a = np.array([1, 2]);
    using b = np.array([3, 4]);
    const obj = { a, b };
    tree.dispose(obj);
    expect(a.refCount).toBe(0);
    expect(b.refCount).toBe(0);
  });

  test("disposes nested tree of arrays", () => {
    using a = np.array([1]);
    using b = np.array([2]);
    using c = np.array([3]);
    const nested = { carry: { a, b }, output: [c] };
    tree.dispose(nested);
    expect(a.refCount).toBe(0);
    expect(b.refCount).toBe(0);
    expect(c.refCount).toBe(0);
  });

  test("handles null/undefined gracefully", () => {
    tree.dispose(null);
    tree.dispose(undefined);
  });

  test("handles aliased leaves without double-dispose", () => {
    using a = np.array([1.0]);
    using b = np.array([2.0]);
    using base = np.add(a, b);
    const aliased = { xf_0: base, yhat: base };

    expect(() => tree.dispose(aliased)).not.toThrow();
    expect(base.refCount).toBe(0);
  });
});

suite("tree.makeDisposable", () => {
  test("returns same object with Symbol.dispose", () => {
    using a = np.array([1, 2]);
    using b = np.array([3, 4]);
    const obj = tree.makeDisposable({ a, b });
    expect(obj.a).toBe(a);
    expect(obj.b).toBe(b);
    expect(Symbol.dispose in obj).toBe(true);
    obj[Symbol.dispose]();
    expect(a.refCount).toBe(0);
    expect(b.refCount).toBe(0);
  });

  test("works with using keyword", () => {
    const before = getBackend().slotCount();
    {
      using result = tree.makeDisposable({
        x: np.array([1, 2, 3]),
        y: np.array([4, 5, 6]),
      });
      expect(result.x.js()).toEqual([1, 2, 3]);
      expect(result.y.js()).toEqual([4, 5, 6]);
    }
    const after = getBackend().slotCount();
    expect(after).toBe(before);
  });

  test("works with scan-like tuple result", () => {
    using carry = np.array([10]);
    using ys = np.array([1, 2, 3]);
    {
      using result = tree.makeDisposable([carry, ys]);
      expect(result[0].js()).toEqual([10]);
      expect(result[1].js()).toEqual([1, 2, 3]);
    }
    expect(carry.refCount).toBe(0);
    expect(ys.refCount).toBe(0);
  });

  test("works with aliased object result", () => {
    using a = np.array([1.0]);
    using b = np.array([2.0]);
    using base = np.add(a, b);
    // eslint-disable-next-line jax-js/no-make-disposable-alias
    const result = tree.makeDisposable({ xf_0: base, yhat: base });

    expect(() => result[Symbol.dispose]()).not.toThrow();
    expect(base.refCount).toBe(0);
  });
});

suite("Array.consumeData", () => {
  test("consumeData returns data and disposes", async () => {
    using x = np.array([1, 2, 3]);
    const data = await x.consumeData();
    expect([...data]).toEqual([1, 2, 3]);
    expect(x.refCount).toBe(0);
  });

  test("consumeDataSync returns data and disposes", () => {
    using x = np.array([4, 5, 6]);
    const data = x.consumeDataSync();
    expect([...data]).toEqual([4, 5, 6]);
    expect(x.refCount).toBe(0);
  });
});

suite("tree.data", () => {
  test("reads all leaves in parallel", async () => {
    using a = np.array([1, 2, 3]);
    using b = np.array([4, 5]);
    const result = await tree.data({ a, b });
    expect([...result.a]).toEqual([1, 2, 3]);
    expect([...result.b]).toEqual([4, 5]);
    // Arrays should still be alive
    expect(a.refCount).toBeGreaterThan(0);
    expect(b.refCount).toBeGreaterThan(0);
  });

  test("handles nested pytree", async () => {
    using x = np.array([10]);
    using y = np.array([20, 30]);
    const result = await tree.data({ carry: x, ys: [y] });
    expect([...result.carry]).toEqual([10]);
    expect([...result.ys[0]]).toEqual([20, 30]);
  });

  test("handles tuple pytree", async () => {
    using a = np.array([1]);
    using b = np.array([2]);
    const result = await tree.data([a, b]);
    expect([...result[0]]).toEqual([1]);
    expect([...result[1]]).toEqual([2]);
  });
});

suite("tree.consumeData", () => {
  test("reads all leaves and disposes them", async () => {
    // eslint-disable-next-line jax-js/require-using -- testing consumeData disposal
    const a = np.array([1, 2, 3]);
    // eslint-disable-next-line jax-js/require-using -- testing consumeData disposal
    const b = np.array([4, 5]);
    const result = await tree.consumeData({ a, b });
    expect([...result.a]).toEqual([1, 2, 3]);
    expect([...result.b]).toEqual([4, 5]);
    // Arrays should be disposed
    expect(a.refCount).toBe(0);
    expect(b.refCount).toBe(0);
  });

  test("reads nested pytree and disposes", async () => {
    // eslint-disable-next-line jax-js/require-using -- testing consumeData disposal
    const x = np.array([10]);
    // eslint-disable-next-line jax-js/require-using -- testing consumeData disposal
    const y = np.array([20, 30]);
    const result = await tree.consumeData({ carry: x, ys: [y] });
    expect([...result.carry]).toEqual([10]);
    expect([...result.ys[0]]).toEqual([20, 30]);
    expect(x.refCount).toBe(0);
    expect(y.refCount).toBe(0);
  });
});
