// Tests for tree.dispose, tree.makeDisposable, and Array.consumeData.

import { getBackend, numpy as np, tree } from "@jax-js-nonconsuming/jax";
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
    using base = np.add(np.array([1.0]), np.array([2.0]));
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
    const before = (getBackend() as any).slotCount();
    {
      using result = tree.makeDisposable({
        x: np.array([1, 2, 3]),
        y: np.array([4, 5, 6]),
      });
      expect(result.x.js()).toEqual([1, 2, 3]);
      expect(result.y.js()).toEqual([4, 5, 6]);
    }
    const after = (getBackend() as any).slotCount();
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
    using base = np.add(np.array([1.0]), np.array([2.0]));
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
