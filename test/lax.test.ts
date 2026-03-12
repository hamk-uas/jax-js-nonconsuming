import { grad, jit, lax, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

import { deviceSuite } from "./device-suite.js";

await deviceSuite((device) => {
  const deviceHasSort = ["cpu", "wasm", "webgpu"].includes(device);

  if (deviceHasSort) {
    suite("jax.lax.topK()", () => {
      test("returns top k values and indices", () => {
        using x = np.array([3, 1, 4, 1, 5, 9, 2, 6]);
        const [values, indices] = lax.topK(x, 3);
        using _v = values;
        using _i = indices;
        expect(values.js()).toEqual([9, 6, 5]);
        expect(indices.js()).toEqual([5, 7, 4]);
      });

      test("stability: equal values preserve original order", () => {
        using x = np.array([1, 3, 2, 3, 3, 0]);
        const [values, indices] = lax.topK(x, 4);
        using _v = values;
        using _i = indices;
        expect(values.js()).toEqual([3, 3, 3, 2]);
        expect(indices.js()).toEqual([1, 3, 4, 2]);
      });

      test("k equals array length", () => {
        using x = np.array([5, 2, 8]);
        const [values, indices] = lax.topK(x, 3);
        using _v = values;
        using _i = indices;
        expect(values.js()).toEqual([8, 5, 2]);
        expect(indices.js()).toEqual([2, 0, 1]);
      });

      test("k = 0 returns empty", () => {
        using x = np.array([1, 2, 3]);
        const [values, indices] = lax.topK(x, 0);
        using _v = values;
        using _i = indices;
        expect(values.shape).toEqual([0]);
        expect(indices.shape).toEqual([0]);
      });

      test("works with 2D along either axis", () => {
        using x = np.array([
          [3, 1, 4],
          [1, 5, 9],
        ]);
        const [values, indices] = lax.topK(x, 2);
        using _v = values;
        using _i = indices;
        expect(values.js()).toEqual([
          [4, 3],
          [9, 5],
        ]);
        expect(indices.js()).toEqual([
          [2, 0],
          [2, 1],
        ]);

        const [values2, indices2] = lax.topK(x, 1, 0);
        using _v2 = values2;
        using _i2 = indices2;
        expect(values2.js()).toEqual([[3, 5, 9]]);
        expect(indices2.js()).toEqual([[0, 1, 1]]);
      });

      test("works with floats and NaN is highest", () => {
        using x = np.array([1.5, NaN, 3.5, 2.1]);
        const [values, indices] = lax.topK(x, 2);
        using _v = values;
        using _i = indices;
        expect(values.js()).toEqual([NaN, 3.5]); // this is consistent with JAX
        expect(indices.js()).toEqual([1, 2]);
      });

      test("throws for invalid k", () => {
        using x = np.array([1, 2, 3]);
        expect(() => lax.topK(x, -1)).toThrow();
        expect(() => lax.topK(x, 4)).toThrow();
      });
    });
  }

  suite("lax.sliceInDim()", () => {
    test("basic slice", () => {
      using x = np.array([10, 20, 30, 40, 50]);
      using result = lax.sliceInDim(x, 1, 4);
      expect(result.js()).toEqual([20, 30, 40]);
    });

    test("negative start", () => {
      using x = np.array([10, 20, 30, 40, 50]);
      using result = lax.sliceInDim(x, -2);
      expect(result.js()).toEqual([40, 50]);
    });

    test("negative limit", () => {
      using x = np.array([10, 20, 30, 40, 50]);
      using result = lax.sliceInDim(x, 1, -1);
      expect(result.js()).toEqual([20, 30, 40]);
    });

    test("limit defaults to axis size", () => {
      using x = np.array([10, 20, 30]);
      using result = lax.sliceInDim(x, 1);
      expect(result.js()).toEqual([20, 30]);
    });

    test("axis parameter", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      using result = lax.sliceInDim(x, 1, 3, 1);
      expect(result.shape).toEqual([2, 2]);
      expect(result.js()).toEqual([
        [2, 3],
        [5, 6],
      ]);
    });

    test("empty slice when start >= limit", () => {
      using x = np.array([10, 20, 30]);
      using result = lax.sliceInDim(x, 2, 2);
      expect(result.shape).toEqual([0]);
    });
  });

  suite("lax.dynamicIndexInDim()", () => {
    test("indexes first element", () => {
      using x = np.array([10, 20, 30]);
      using result = lax.dynamicIndexInDim(x, 0);
      expect(result.shape).toEqual([]);
      expect(result.js()).toBe(10);
    });

    test("indexes last element with negative index", () => {
      using x = np.array([10, 20, 30]);
      using result = lax.dynamicIndexInDim(x, -1);
      expect(result.shape).toEqual([]);
      expect(result.js()).toBe(30);
    });

    test("keepdims=true preserves axis", () => {
      using x = np.array([10, 20, 30]);
      using result = lax.dynamicIndexInDim(x, 1, 0, true);
      expect(result.shape).toEqual([1]);
      expect(result.js()).toEqual([20]);
    });

    test("indexes along axis=1 in 2D", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      using result = lax.dynamicIndexInDim(x, 2, 1);
      expect(result.shape).toEqual([2]);
      expect(result.js()).toEqual([3, 6]);
    });

    test("grad through dynamicIndexInDim", () => {
      const f = (x: np.Array) => {
        using indexed = lax.dynamicIndexInDim(x, 1);
        return indexed.mul(indexed);
      };
      using x = np.array([3.0, 5.0, 7.0]);
      using g = grad(f)(x);
      // d/dx[1] of x[1]^2 = 2*5 = 10; other elements are 0
      expect(g.js()).toEqual([0, 10, 0]);
    });

    test("jit(dynamicIndexInDim) works", () => {
      using f = jit((x: np.Array) => lax.dynamicIndexInDim(x, 0));
      using x = np.array([42.0, 99.0]);
      using result = f(x);
      expect(result.js()).toBe(42);
    });
  });
});
