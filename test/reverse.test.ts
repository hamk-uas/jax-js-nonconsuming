/**
 * Dedicated tests for the `reverse` primitive (Primitive.Reverse).
 *
 * Covers: eager, JIT (concrete + polymorphic), transforms (jvp, grad, vmap).
 */
import {
  type Array,
  grad,
  init,
  jit,
  jvp,
  numpy as np,
  reverse,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

await init();

describe("reverse primitive", () => {
  // ---- Eager ----

  test("eager: 1-D", () => {
    using x = np.array([1, 2, 3, 4]);
    using r = reverse(x, 0) as Array;
    expect(r.js()).toEqual([4, 3, 2, 1]);
  });

  test("eager: 2-D axis=0", () => {
    using x = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    using r = reverse(x, 0) as Array;
    expect(r.js()).toEqual([
      [5, 6],
      [3, 4],
      [1, 2],
    ]);
  });

  test("eager: 2-D axis=1", () => {
    using x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using r = reverse(x, 1) as Array;
    expect(r.js()).toEqual([
      [3, 2, 1],
      [6, 5, 4],
    ]);
  });

  test("eager: 3-D axis=2", () => {
    using x = np.arange(24).reshape([2, 3, 4]);
    using r = reverse(x, 2) as Array;
    // Each inner row [0,1,2,3] → [3,2,1,0]
    const jsR = r.js() as number[][][];
    expect(jsR[0][0]).toEqual([3, 2, 1, 0]);
    expect(jsR[1][2]).toEqual([23, 22, 21, 20]);
  });

  test("eager: negative axis", () => {
    using x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using r = reverse(x, -1) as Array;
    expect(r.js()).toEqual([
      [3, 2, 1],
      [6, 5, 4],
    ]);
  });

  test("eager: size-1 axis is identity", () => {
    using x = np.array([[1, 2, 3]]);
    using r = reverse(x, 0) as Array;
    expect(r.js()).toEqual([[1, 2, 3]]);
  });

  test("eager: double reverse is identity", () => {
    using x = np.array([10, 20, 30, 40]);
    using r1 = reverse(x, 0) as Array;
    using r2 = reverse(r1, 0) as Array;
    expect(r2.js()).toEqual([10, 20, 30, 40]);
  });

  // ---- JIT ----

  test("jit: concrete shapes", () => {
    const f = jit((x: np.Array) => reverse(x, 0) as unknown as np.Array);
    using x = np.array([5, 4, 3, 2, 1]);
    using r = f(x);
    expect(r.js()).toEqual([1, 2, 3, 4, 5]);
    f.dispose();
  });

  test("jit: dynamic_axes", () => {
    const f = jit((x: np.Array) => reverse(x, 0) as unknown as np.Array, {
      dynamic_axes: { 0: "T" },
    });
    using x3 = np.array([1, 2, 3]);
    using r3 = f(x3);
    expect(r3.js()).toEqual([3, 2, 1]);

    using x5 = np.array([10, 20, 30, 40, 50]);
    using r5 = f(x5);
    expect(r5.js()).toEqual([50, 40, 30, 20, 10]);
    f.dispose();
  });

  // ---- Transforms ----

  test("jvp: tangent is reversed in the same way", () => {
    const f = (x: np.Array) => reverse(x, 0);
    using xIn = np.array([1, 2, 3]);
    using tIn = np.array([10, 20, 30]);
    const [primal, tangent] = jvp(f as any, [xIn], [tIn]);
    using p = primal as np.Array;
    using t = tangent as np.Array;
    expect(p.js()).toEqual([3, 2, 1]);
    expect(t.js()).toEqual([30, 20, 10]);
  });

  test("grad: sum(reverse(x)) = ones", () => {
    const g = grad((x: np.Array) => np.sum(reverse(x, 0) as any));
    using x = np.array([1, 2, 3, 4]);
    using gx = g(x);
    expect(gx).toBeAllclose([1, 1, 1, 1]);
  });

  test("grad: weighted sum through reverse", () => {
    // f(x) = sum(reverse(x) * [4, 3, 2, 1]) = sum([x3*4, x2*3, x1*2, x0*1])
    // grad = [1, 2, 3, 4]
    const w = np.array([4, 3, 2, 1]);
    const g = grad((x: np.Array) => {
      using rev = reverse(x, 0) as any;
      return np.sum(np.multiply(rev, w));
    });
    using x = np.array([10, 20, 30, 40]);
    using gx = g(x);
    expect(gx).toBeAllclose([1, 2, 3, 4]);
    w.dispose();
  });

  test("vmap: batched reverse", () => {
    const f = vmap((x: np.Array) => reverse(x, 0) as unknown as np.Array);
    using batch = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using r = f(batch);
    expect(r.js()).toEqual([
      [3, 2, 1],
      [6, 5, 4],
    ]);
  });
});
