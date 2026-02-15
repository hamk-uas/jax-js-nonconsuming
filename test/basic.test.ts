import {
  hessian,
  jacfwd,
  jacrev,
  jit,
  jvp,
  numpy as np,
  vmap,
} from "@jax-js-nonconsuming/jax";
import { expect, suite, test } from "vitest";

test("can create array", () => {
  using x = np.array([1, 2, 3]);
  expect(x.js()).toEqual([1, 2, 3]);
});

suite("jax.jvp()", () => {
  test("can take scalar derivatives", () => {
    /** Take the derivative of a simple function. */
    const deriv: (
      f: (x: np.Array) => np.Array,
    ) => (x: np.ArrayLike) => np.Array = (f) => (x) => {
      const [y, dy] = jvp(f, [x], [1.0]);
      y.dispose();
      return dy;
    };
    const x = 3.0;
    using s1 = np.sin(x);
    expect(s1).toBeAllclose(0.141120001);
    using d1 = deriv(np.sin)(x);
    expect(d1).toBeAllclose(-0.989992499);
    using d2 = deriv(deriv(np.sin))(x);
    expect(d2).toBeAllclose(-0.141120001);
    using d3 = deriv(deriv(deriv(np.sin)))(x);
    expect(d3).toBeAllclose(0.989992499);
  });

  test("can take jvp of pytrees", () => {
    const result = jvp(
      (x: { a: np.Array; b: np.Array }) => x.a.mul(x.a).add(x.b),
      [{ a: 1, b: 2 }],
      [{ a: 1, b: 0 }],
    );
    using r0 = result[0] as np.Array;
    using r1 = result[1] as np.Array;
    expect(r0).toBeAllclose(3);
    expect(r1).toBeAllclose(2);
  });

  test("works for vector to scalar functions", () => {
    const f = (x: np.Array) => np.sum(x);
    using x = np.array([1, 2, 3]);
    using fx = f(x);
    expect(fx).toBeAllclose(6);
    using dx = np.array([1, 1, 1]);
    const [y, dy] = jvp(f, [x], [dx]);
    using _y = y as np.Array;
    using _dy = dy as np.Array;
    expect(_dy).toBeAllclose(3);
  });

  test("can compute jvp of products", () => {
    using x = np.array([1, 2, 3, 4]);
    using dx = np.array([1, 10, 100, 1000]);

    const jvpProd = jvp((x: np.Array) => np.prod(x), [x], [dx]);
    using p = jvpProd[0] as np.Array;
    using dp = jvpProd[1] as np.Array;
    expect(p).toBeAllclose(24);
    // 1 * 2*3*4 + 10 * 1*3*4 + 100 * 1*2*4 + 1000 * 1*2*3
    expect(dp).toBeAllclose(6944);
  });

  test("can have jvp of min", () => {
    using x = np.array([1, 2, 1, 4]);
    using dx = np.array([0, 10, 5, 1000]);
    const [pmin, jmin] = jvp((x: np.Array) => np.min(x), [x], [dx]);
    using _pmin = pmin as np.Array;
    using _jmin = jmin as np.Array;
    expect(_pmin).toBeAllclose(1);
    expect(_jmin).toBeAllclose(2.5); // (0+5)/2
  });
});

suite("jax.vmap()", () => {
  test("vectorizes a function over a single axis", () => {
    const f = (x: np.Array) => x.mul(2);
    const batchedF = vmap(f, [0]);
    using x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using result = batchedF(x);
    expect(result.js()).toEqual([
      [2, 4, 6],
      [8, 10, 12],
    ]);
  });

  test("vectorizes a function with multiple arguments", () => {
    const f = (x: np.Array, y: np.Array) => x.add(y);
    const batchedF = vmap(f, [0, 1]);
    using x = np.array([
      [1, 2],
      [3, 4],
    ]);
    using y = np.array([
      [10, 20],
      [30, 40],
    ]);
    using result = batchedF(x, y);
    expect(result.js()).toEqual([
      [11, 32],
      [23, 44],
    ]);
  });

  test("vectorizes with a static argument", () => {
    const f = (x: np.Array, y: np.Array) => x.mul(y);
    const batchedF = vmap(f, [0, null]);
    using x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const y = 10;
    using result = batchedF(x, y);
    expect(result.js()).toEqual([
      [10, 20],
      [30, 40],
    ]);
  });

  test("vectorizes a function returning a pytree", () => {
    const f = (x: np.Array) => ({
      double: x.mul(2),
      square: x.mul(x),
      sum: x.sum(),
    });
    const batchedF = vmap(f, [0]);
    using x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = batchedF(x);
    using d = result.double;
    using s = result.square;
    using sm = result.sum;
    expect(d.js()).toEqual([
      [2, 4, 6],
      [8, 10, 12],
    ]);
    expect(s.js()).toEqual([
      [1, 4, 9],
      [16, 25, 36],
    ]);
    expect(sm.js()).toEqual([6, 15]);
  });

  test("vectorizes over pytrees inputs", () => {
    const f = (x: { a: np.Array; b: np.Array }) => x.a.add(x.b);
    const batchedF = vmap(f, [{ a: 0, b: 0 }]);
    using a = np.array([
      [1, 2],
      [3, 4],
    ]);
    using b = np.array([
      [10, 20],
      [30, 40],
    ]);
    using result = batchedF({ a, b });
    expect(result.js()).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });

  test("can take a single axis number", () => {
    using a = np.array([
      [1, 2],
      [3, 4],
    ]);
    using b = np.array([
      [5, 6],
      [7, 8],
    ]);
    using r1 = vmap(np.dot, 0)(a, b);
    using r2 = vmap(np.dot, 1)(a, b);
    expect(r1.js()).toEqual([17, 53]);
    expect(r2.js()).toEqual([26, 44]);
  });

  test("can vectorize over iteration and slice", () => {
    const f = (x: np.Array) => [...x];
    using batchedF = jit(vmap(f, 0));
    using x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const [a, b, c] = batchedF(x);
    using _a = a;
    using _b = b;
    using _c = c;
    expect(a.js()).toEqual([1, 4]);
    expect(b.js()).toEqual([2, 5]);
    expect(c.js()).toEqual([3, 6]);
  });

  test("can use axis number for entire pytree inputs", () => {
    type AB = { a: np.Array; b: np.Array };
    const f = (x: AB, y: AB) => x.a.add(x.b).add(y.a).add(y.b);
    using batchedF = jit(vmap(f, [null, 0]));
    using xa = np.array([1, 2]);
    using xb = np.array([3, 4]);
    const _ya = np.array([10, 20, 30, 40]);
    using ya = _ya.reshape([2, 2]);
    _ya.dispose();
    const _yb = np.array([100, 200, 300, 400]);
    using yb = _yb.reshape([2, 2]);
    _yb.dispose();
    using result = batchedF({ a: xa, b: xb }, { a: ya, b: yb });
    expect(result.js()).toEqual([
      [114, 226],
      [334, 446],
    ]);
  });
});

suite("jax.jacfwd()", () => {
  test("computes jacobian of 3d square", () => {
    const f = (x: np.Array) => x.mul(x);
    using x = np.array([1, 2, 3]);
    using jf = jit(jacfwd(f));
    using j = jf(x);
    using expected = np.array([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
    expect(j).toBeAllclose(expected);
  });

  test("equals jacrev() output", () => {
    const f = (x: np.Array) => np.sin(x).add(np.cos(x));
    using x = np.array([1, 2, 3]);
    using jfF = jit(jacfwd(f));
    using jFwd = jfF(x);
    using jfR = jit(jacrev(f));
    using jRev = jfR(x);
    expect(jFwd).toBeAllclose(jRev);
  });
});

suite("jax.hessian()", () => {
  test("computes hessian of sum of squares", () => {
    // f(x) = sum(x^2) => gradient = 2x, hessian = 2*I
    const f = (x: np.Array) => np.sum(x.mul(x));
    using x = np.array([1, 2, 3]);
    using hf = jit(hessian(f));
    using H = hf(x);
    using expected = np.array([
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]);
    expect(H).toBeAllclose(expected);
  });

  test("computes hessian of quadratic form", () => {
    using coeffs = np.array([1, 2, 3]);
    const f = (x: np.Array) => {
      return np.sum(coeffs.mul(x).mul(x));
    };
    using x = np.array([1, 1, 1]);
    using hf = jit(hessian(f));
    using H = hf(x);
    using expected = np.array([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
    expect(H).toBeAllclose(expected);
  });

  test("computes hessian with cross terms", () => {
    const f = (x: np.Array) => {
      const [x0, x1, x2] = x;
      return x0.mul(x1).add(x1.mul(x2));
    };
    using x = np.array([1, 2, 3]);
    using hf = jit(hessian(f));
    using H = hf(x);
    using expected = np.array([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ]);
    expect(H).toBeAllclose(expected);
  });
});
