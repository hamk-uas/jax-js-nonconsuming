/**
 * Tests for the non-consuming ownership model.
 *
 * Operations do NOT consume their inputs — no .ref needed for multi-use.
 * Memory is managed via explicit .dispose() or the `using` keyword.
 */
import {
  grad,
  jit,
  lax,
  numpy as np,
  valueAndGrad,
} from "@jax-js-nonconsuming/jax";
import { describe, expect, it } from "vitest";

describe("non-consuming ownership model", () => {
  describe("operations do not consume inputs", () => {
    it("input used twice without .ref", () => {
      using f = jit((x: np.Array) => x.mul(x));
      using x = np.array([2, 3]);
      using result = f(x);
      expect(result.js()).toEqual([4, 9]);
    });

    it("input used three times without .ref", () => {
      using f = jit((x: np.Array) => x.mul(x).add(x));
      using x = np.array([1, 2, 3]);
      using result = f(x);
      expect(result.js()).toEqual([2, 6, 12]);
    });

    it("intermediate used twice without .ref", () => {
      using f = jit((x: np.Array) => {
        const a = x.mul(2);
        return a.add(a);
      });
      using x = np.array([5, 10]);
      using result = f(x);
      expect(result.js()).toEqual([20, 40]);
    });

    it("multiple inputs used multiple times", () => {
      using f = jit((x: np.Array, y: np.Array) => {
        return x.mul(y).add(x.add(y));
      });
      using x = np.array([1, 2]);
      using y = np.array([3, 4]);
      using result = f(x, y);
      expect(result.js()).toEqual([7, 14]);
    });

    it("closures captured by jit", () => {
      using scale = np.array([10, 10, 10]);
      using f = jit((x: np.Array) => x.mul(scale));
      using x1 = np.array([1, 2, 3]);
      using r1 = f(x1);
      expect(r1.js()).toEqual([10, 20, 30]);
      using x2 = np.array([4, 5, 6]);
      using r2 = f(x2);
      expect(r2.js()).toEqual([40, 50, 60]);
    });

    it("works with reduction", () => {
      using f = jit((x: np.Array) => x.mul(x).sum());
      using x = np.array([1, 2, 3]);
      using result = f(x);
      expect(result.js()).toEqual(14);
    });

    it("works with 2D arrays", () => {
      using f = jit((x: np.Array) => {
        const t = x.add(x);
        return t.sum(-1);
      });
      using x = np.array([
        [1, 2],
        [3, 4],
      ]);
      using result = f(x);
      expect(result.js()).toEqual([6, 14]);
    });

    it("mandelbrot pattern without .ref", () => {
      using f = jit(
        (A: np.Array, B: np.Array, V: np.Array, X: np.Array, Y: np.Array) => {
          const Asq = A.mul(A);
          const Bsq = B.mul(B);
          V = V.add(Asq.add(Bsq).less(100).astype(np.float32));
          const A2 = np.clip(Asq.sub(Bsq).add(X), -50, 50);
          const B2 = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
          return [A2, B2, V];
        },
      );

      const size = 4;
      using a0 = np.zeros([size]);
      using b0 = np.zeros([size]);
      using v0 = np.zeros([size]);
      using x0 = np.linspace(-2, 1, size);
      using y0 = np.linspace(-1, 1, size);
      const [A2, B2, V2] = f(a0, b0, v0, x0, y0) as np.Array[];
      using _a2 = A2;
      using _b2 = B2;
      using _v2 = V2;
      expect(V2.js()).toEqual([1, 1, 1, 1]);
    });

    it("cached across calls with same shape", () => {
      using f = jit((x: np.Array) => x.mul(x));
      using x1 = np.array([1, 2]);
      using r1 = f(x1);
      expect(r1.js()).toEqual([1, 4]);
      using x2 = np.array([3, 4]);
      using r2 = f(x2);
      expect(r2.js()).toEqual([9, 16]);
    });
  });

  describe("does not leak memory", () => {
    it("multi-use input does not leak", () => {
      using f = jit((x: np.Array) => x.mul(x).add(1));
      using input = np.array([2, 3, 4]);
      using result = f(input);
      expect(result.js()).toEqual([5, 10, 17]);
    });

    it("unused input does not leak", () => {
      using f = jit((_x: np.Array, y: np.Array) => y.mul(2));
      using x = np.array([1]);
      using y = np.array([5]);
      using result = f(x, y);
      expect(result.js()).toEqual([10]);
    });
  });

  describe("with grad", () => {
    it("grad through jit without .ref", () => {
      using f = jit((x: np.Array) => x.mul(x).sum());
      const df = grad(f);
      using x = np.array([1, 2, 3]);
      using result = df(x);
      expect(result.js()).toEqual([2, 4, 6]);
    });

    it("grad with multi-use input", () => {
      // f(x) = (x*x + x).sum() => f'(x) = 2x + 1
      using f = jit((x: np.Array) => x.mul(x).add(x).sum());
      const df = grad(f);
      using x = np.array([1, 2, 3]);
      using result = df(x);
      expect(result.js()).toEqual([3, 5, 7]);
    });

    it("valueAndGrad without .ref", () => {
      using f = jit((x: np.Array) => x.mul(x).sum());
      const vg = valueAndGrad(f);
      using x = np.array([1, 2, 3]);
      const [val, dx] = vg(x);
      using _val = val;
      using _dx = dx;
      expect(val.js()).toEqual(14);
      expect(dx.js()).toEqual([2, 4, 6]);
    });
  });

  describe("with scan", () => {
    it("scan inside jit without .ref", () => {
      using init = np.array([0]);
      using f = jit((init: np.Array, xs: np.Array) => {
        return lax.scan(
          (carry: np.Array, x: np.Array) => {
            const s = carry.add(x);
            return [s, s];
          },
          init,
          xs.reshape([-1, 1]),
        );
      });
      using xs = np.array([1, 2, 3, 4, 5]);
      const [carry, ys] = f(init, xs) as [np.Array, np.Array];
      using _carry = carry;
      using _ys = ys;
      expect(carry.js()).toEqual([15]);
      expect(ys.js()).toEqual([[1], [3], [6], [10], [15]]);
    });

    it("scan body with multi-use carry", () => {
      using init = np.array([1]);
      using f = jit((init: np.Array, xs: np.Array) => {
        return lax.scan(
          (carry: np.Array, x: np.Array) => {
            const s = carry.mul(carry).add(x);
            return [s, s];
          },
          init,
          xs.reshape([-1, 1]),
        );
      });
      using xs = np.array([1, 2, 3]);
      const [carry, ys] = f(init, xs) as [np.Array, np.Array];
      using _carry = carry;
      using _ys = ys;
      expect(carry.js()).toEqual([39]);
      expect(ys.js()).toEqual([[2], [6], [39]]);
    });
  });

  describe("edge cases", () => {
    it("scalar inputs", () => {
      using f = jit((x: np.Array) => x.mul(x).add(x));
      using x = np.array(3);
      using result = f(x);
      expect(result.js()).toEqual(12);
    });

    it(".ref still works for backward compat", () => {
      // jax-js-lint: allow-ref — intentional test of backward-compat .ref
      using f = jit((x: np.Array) => x.ref.mul(x));
      using x = np.array([2, 3]);
      using result = f(x);
      expect(result.js()).toEqual([4, 9]);
    });
  });
});
