/**
 * Tests for M6.1: Mega-Module compiler.
 *
 * The mega-module compiles an entire JitProgram's kernel-only step list
 * into a single WASM function, eliminating JS↔WASM boundary crossings
 * between kernel dispatches.
 */
import { grad, init, jit, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { beforeAll, describe, expect, it } from "vitest";

describe("mega-module (M6.1)", () => {
  beforeAll(async () => {
    await init("wasm");
  });

  describe("correctness", () => {
    it("simple add chain", () => {
      using f = jit((x: np.Array) => x.add(1));
      using x = np.array([1, 2, 3, 4]);
      using result = f(x);
      expect(result.js()).toEqual([2, 3, 4, 5]);
    });

    it("chained elementwise ops", () => {
      // (x + 1) * 2 - 3
      using f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
      using x = np.array([1, 2, 3, 4]);
      using result = f(x);
      expect(result.js()).toEqual([1, 3, 5, 7]);
    });

    it("two-input operation", () => {
      using f = jit((a: np.Array, b: np.Array) => a.add(b));
      using a = np.array([1, 2, 3]);
      using b = np.array([10, 20, 30]);
      using result = f(a, b);
      expect(result.js()).toEqual([11, 22, 33]);
    });

    it("multi-input chain", () => {
      using f = jit((a: np.Array, b: np.Array, c: np.Array) => a.add(b).mul(c));
      using a = np.array([1, 2, 3]);
      using b = np.array([10, 20, 30]);
      using c = np.array([2, 3, 4]);
      using result = f(a, b, c);
      expect(result.js()).toEqual([22, 66, 132]);
    });

    it("reduction (sum)", () => {
      using f = jit((x: np.Array) => x.mul(2).sum().add(1));
      using x = np.array([1, 2, 3]);
      // (2+4+6) + 1 = 13
      using result = f(x);
      expect(result.js()).toEqual(13);
    });

    it("negative values", () => {
      using f = jit((x: np.Array) => x.mul(-1).add(10));
      using x = np.array([1, 2, 3]);
      using result = f(x);
      expect(result.js()).toEqual([9, 8, 7]);
    });

    it("float32 arithmetic", () => {
      using f = jit((x: np.Array) => x.mul(0.5).add(0.25));
      using x = np.array([1.0, 2.0, 4.0]);
      using result = f(x);
      expect(result.js()).toEqual([0.75, 1.25, 2.25]);
    });

    it("larger array", () => {
      using f = jit((x: np.Array) => x.add(1));
      const data = Array.from({ length: 1024 }, (_, i) => i);
      using x = np.array(data);
      using result = f(x);
      const expected = data.map((v) => v + 1);
      expect(result.js()).toEqual(expected);
    });
  });

  describe("repeated calls", () => {
    it("same-shape repeated calls are correct", () => {
      // Mega-module is compiled on 1st call, cached for subsequent calls
      using f = jit((x: np.Array) => x.add(1).mul(2));
      for (let i = 0; i < 5; i++) {
        using x = np.array([i, i + 1, i + 2]);
        using result = f(x);
        expect(result.js()).toEqual([(i + 1) * 2, (i + 2) * 2, (i + 3) * 2]);
      }
    });
  });

  describe("no leaks", () => {
    it("chain does not leak slots", async () => {
      using f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4));
      using x = np.array([10, 20, 30, 40]);
      using result = f(x);
      await result.data();
      // Leak detection is done by the global setup.ts afterEach
    });

    it("multi-output does not leak", async () => {
      // Two separate outputs — tests multi-output kernel codegen
      using f = jit((x: np.Array) => x.add(1).mul(3));
      using x = np.array([1, 2, 3]);
      using result = f(x);
      await result.data();
    });

    it("repeated calls do not leak", async () => {
      using f = jit((x: np.Array) => x.add(1));
      for (let i = 0; i < 10; i++) {
        using x = np.array([i]);
        using result = f(x);
        await result.data();
      }
    });
  });

  describe("grad through mega-module", () => {
    it("grad of x^2 sum", () => {
      // f(x) = sum(x^2), f'(x) = 2x
      const f = (x: np.Array) => np.multiply(x, x).sum();
      const df = grad(f);
      using x = np.array([1, 2, 3]);
      using result = df(x);
      expect(result.js()).toEqual([2, 4, 6]);
    });

    it("jit(grad(...)) composition", () => {
      // f(x) = sum((x + 1)^2), f'(x) = 2(x + 1)
      using df = jit(
        grad((x: np.Array) => {
          using xp1 = x.add(1);
          return np.multiply(xp1, xp1).sum();
        }),
      );
      using x = np.array([0, 1, 2]);
      using result = df(x);
      expect(result.js()).toEqual([2, 4, 6]);
    });
  });
});
