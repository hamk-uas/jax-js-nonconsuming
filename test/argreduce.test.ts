import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

import { deviceSuite } from "./device-suite.js";

// Tests run on all available backends: CPU, WASM, and WebGPU
await deviceSuite(() => {
  suite("argmax correctness", () => {
    test("argmax returns first index on ties", () => {
      using x = np.array([3, 5, 5, 2, 5, 1]);
      using result = np.argmax(x);
      expect(result.js()).toEqual(1); // First occurrence of maximum (5)
    });

    test("argmax works with negative values", () => {
      using x = np.array([-10, -5, -20, -5, -15]);
      using result = np.argmax(x);
      expect(result.js()).toEqual(1); // First occurrence of maximum (-5)
    });

    test("argmax handles all equal values", () => {
      using x = np.ones([5]);
      using result = np.argmax(x);
      expect(result.js()).toEqual(0); // Returns first index
    });

    test("argmax 2D various shapes", () => {
      using x = np.array([
        [1, 2, 3],
        [6, 5, 4],
        [7, 8, 9],
      ]);

      {
        using r = np.argmax(x);
        expect(r.js()).toEqual(8);
      } // Global argmax
      {
        using r = np.argmax(x, 0);
        expect(r.js()).toEqual([2, 2, 2]);
      } // Column-wise
      {
        using r = np.argmax(x, 1);
        expect(r.js()).toEqual([2, 0, 2]);
      } // Row-wise
    });
  });

  suite("argmin correctness", () => {
    test("argmin returns first index on ties", () => {
      using x = np.array([3, 1, 5, 1, 1, 2]);
      using result = np.argmin(x);
      expect(result.js()).toEqual(1); // First occurrence of minimum (1)
    });

    test("argmin handles all equal values", () => {
      using _x = np.ones([5]);
      using x = _x.mul(42);
      using result = np.argmin(x);
      expect(result.js()).toEqual(0); // Returns first index
    });

    test("argmin 2D various shapes", () => {
      using x = np.array([
        [9, 8, 7],
        [4, 5, 6],
        [3, 2, 1],
      ]);

      {
        using r = np.argmin(x);
        expect(r.js()).toEqual(8);
      } // Global argmin
      {
        using r = np.argmin(x, 0);
        expect(r.js()).toEqual([2, 2, 2]);
      } // Column-wise
      {
        using r = np.argmin(x, 1);
        expect(r.js()).toEqual([2, 0, 2]);
      } // Row-wise
    });
  });
});
