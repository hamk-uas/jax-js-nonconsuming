// Tests for tree.dispose, tree.makeDisposable, and Array.consumeData.

import { getBackend, numpy as np, tree } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("tree.dispose", () => {
  test("disposes all arrays in a flat object", () => {
    const a = np.array([1, 2]);
    const b = np.array([3, 4]);
    const obj = { a, b };
    tree.dispose(obj);
    expect(a.refCount).toBe(0);
    expect(b.refCount).toBe(0);
  });

  test("disposes nested tree of arrays", () => {
    const a = np.array([1]);
    const b = np.array([2]);
    const c = np.array([3]);
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
    const base = np.add(np.array([1.0]), np.array([2.0]));
    const aliased = { xf_0: base, yhat: base };

    expect(() => tree.dispose(aliased)).not.toThrow();
    expect(base.refCount).toBe(0);
  });
});

suite("tree.makeDisposable", () => {
  test("returns same object with Symbol.dispose", () => {
    const a = np.array([1, 2]);
    const b = np.array([3, 4]);
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
    const carry = np.array([10]);
    const ys = np.array([1, 2, 3]);
    {
      using result = tree.makeDisposable([carry, ys]);
      expect(result[0].js()).toEqual([10]);
      expect(result[1].js()).toEqual([1, 2, 3]);
    }
    expect(carry.refCount).toBe(0);
    expect(ys.refCount).toBe(0);
  });

  test("works with aliased object result", () => {
    const base = np.add(np.array([1.0]), np.array([2.0]));
    const result = tree.makeDisposable({ xf_0: base, yhat: base });

    expect(() => result[Symbol.dispose]()).not.toThrow();
    expect(base.refCount).toBe(0);
  });
});

suite("Array.consumeData", () => {
  test("consumeData returns data and disposes", async () => {
    const x = np.array([1, 2, 3]);
    const data = await x.consumeData();
    expect([...data]).toEqual([1, 2, 3]);
    expect(x.refCount).toBe(0);
  });

  test("consumeDataSync returns data and disposes", () => {
    const x = np.array([4, 5, 6]);
    const data = x.consumeDataSync();
    expect([...data]).toEqual([4, 5, 6]);
    expect(x.refCount).toBe(0);
  });
});
