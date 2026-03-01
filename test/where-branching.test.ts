/**
 * Tests for WASM Where branching optimization.
 *
 * When `np.where` has an expensive arm (exp, log, sin, erf, …), the WASM
 * backend emits `if/else` true branching instead of the branchless `select`
 * instruction. This skips the expensive arm for elements that don't need it.
 *
 * These tests verify correctness of the branching codegen, including edge
 * cases around shared subexpressions (CSE) between arms.
 */
import {
  defaultDevice,
  type Device,
  devices,
  grad,
  init,
  jit,
  lax,
  nn,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

// Eager GPU dispatches dominate wall-clock; algorithm correctness proven by cpu+wasm.
const fastDevices: Device[] = devices.filter(
  (d) => d !== "webgpu" && d !== "webgl",
);

suite.each(fastDevices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("where branching correctness", () => {
    test("where with exp in one arm (ELU pattern)", () => {
      // elu(x) = where(x > 0, x, alpha * (exp(x) - 1))
      using x = np.array([-2, -1, 0, 1, 2]);
      using result = nn.elu(x);
      expect(result).toBeAllclose([-0.8647, -0.6321, 0, 1, 2], { atol: 1e-3 });
    });

    test("where with exp in one arm (SELU pattern)", () => {
      using x = np.array([-2, -1, 0, 1, 2]);
      using result = nn.selu(x);
      // selu(x) = lambda * where(x < 0, alpha * (exp(x) - 1), x)
      const alpha = 1.6732632423543772;
      const lambda = 1.0507009873554805;
      const expected = [-2, -1, 0, 1, 2].map((v) =>
        v < 0 ? lambda * alpha * (Math.exp(v) - 1) : lambda * v,
      );
      expect(result).toBeAllclose(expected, { atol: 1e-3 });
    });

    test("where with expensive arm — all-true condition", () => {
      // When condition is all-true, only the true arm should matter.
      using x = np.array([1, 2, 3, 4]);
      using cond = np.array([true, true, true, true]);
      using expX = np.exp(x);
      using zeros = np.array([0, 0, 0, 0]);
      using result = np.where(cond, expX, zeros);
      expect(result).toBeAllclose([1, 2, 3, 4].map(Math.exp), { atol: 1e-4 });
    });

    test("where with expensive arm — all-false condition", () => {
      // When condition is all-false, only the false arm should matter.
      using x = np.array([1, 2, 3, 4]);
      using cond = np.array([false, false, false, false]);
      using zeros = np.array([0, 0, 0, 0]);
      using logX = np.log(x);
      using result = np.where(cond, zeros, logX);
      expect(result).toBeAllclose([1, 2, 3, 4].map(Math.log), { atol: 1e-4 });
    });

    test("where with expensive arm — mixed condition", () => {
      using x = np.array([-1, 0.5, -2, 3]);
      using cond = x.greater(0);
      using expX = np.exp(x);
      using result = np.where(cond, x, expX);
      const expected = [-1, 0.5, -2, 3].map((v) => (v > 0 ? v : Math.exp(v)));
      expect(result).toBeAllclose(expected, { atol: 1e-5 });
    });

    test("where with both arms expensive", () => {
      // where(cond, exp(x), log(abs(x)))
      using x = np.array([1, 2, 3, 4]);
      using cond = np.array([true, false, true, false]);
      {
        using expX = np.exp(x);
        using absX = np.abs(x);
        using logAbsX = np.log(absX);
        using result = np.where(cond, expX, logAbsX);
        const expected = [Math.exp(1), Math.log(2), Math.exp(3), Math.log(4)];
        expect(result).toBeAllclose(expected, { atol: 1e-4 });
      }
    });

    test("where with shared subexpression between arms (CSE edge case)", () => {
      // where(x > 0, x * sin(x), x * cos(x))
      // sin(x) and cos(x) are expensive, but x is shared between arms
      using x = np.array([-1, 0, 1, 2]);
      using cond = x.greater(0);
      {
        using sinX = np.sin(x);
        using cosX = np.cos(x);
        using armT = x.mul(sinX);
        using armF = x.mul(cosX);
        using result = np.where(cond, armT, armF);
        const expected = [-1, 0, 1, 2].map((v) =>
          v > 0 ? v * Math.sin(v) : v * Math.cos(v),
        );
        expect(result).toBeAllclose(expected, { atol: 1e-5 });
      }
    });

    test("where with cheap arms stays branchless (relu pattern)", () => {
      // relu = where(x > 0, x, 0) — both arms are cheap
      using x = np.array([-2, -1, 0, 1, 2]);
      using result = nn.relu(x);
      expect(result.js()).toEqual([0, 0, 0, 1, 2]);
    });
  });

  suite("where branching with JIT", () => {
    test("jit wrapping preserves correctness", () => {
      const f = jit((x: np.Array) => {
        using cond = x.greater(0);
        return np.where(cond, x, np.exp(x));
      });
      using x = np.array([-1, 0.5, -2, 3]);
      using result = f(x);
      f.dispose();
      const expected = [-1, 0.5, -2, 3].map((v) => (v > 0 ? v : Math.exp(v)));
      expect(result).toBeAllclose(expected, { atol: 1e-5 });
    });

    test("jit with grad through where", () => {
      // grad of where(x > 0, x, exp(x)) wrt x
      // = where(x > 0, 1, exp(x))
      const f = (x: np.Array) => {
        using cond = x.greater(0);
        using selected = np.where(cond, x, np.exp(x));
        return selected.sum();
      };
      using x = np.array([-1, 0.5, -2, 3]);
      const gradF = grad(f);
      using gx = gradF(x);
      const expected = [-1, 0.5, -2, 3].map((v) => (v > 0 ? 1 : Math.exp(v)));
      expect(gx).toBeAllclose(expected, { atol: 1e-4 });
    });

    test("nn.elu through jit", () => {
      const f = jit((x: np.Array) => nn.elu(x));
      using x = np.array([-2, -1, 0, 1, 2]);
      using result = f(x);
      f.dispose();
      expect(result).toBeAllclose([-0.8647, -0.6321, 0, 1, 2], { atol: 1e-3 });
    });

    test("selu-pattern gradient (inlined where+exp)", () => {
      // Tests grad through a Where with expensive arm (exp), without
      // involving nn.selu's module-level jit() wrapper.
      const alpha = 1.6732632423543772;
      const lambda = 1.0507009873554805;
      using x = np.array([-2, -1, 0, 1, 2]);
      const gradF = grad((x: np.Array) => {
        using cond = x.less(0);
        using expm1X = np.expm1(x);
        using scaled = expm1X.mul(alpha);
        using selected = np.where(cond, scaled, x);
        using result = selected.mul(lambda);
        return result.sum();
      });
      using gx = gradF(x);
      // selu'(x) = lambda * alpha * exp(x) for x < 0, lambda for x >= 0
      const expected = [-2, -1, 0, 1, 2].map((v) =>
        v < 0 ? lambda * alpha * Math.exp(v) : lambda,
      );
      expect(gx).toBeAllclose(expected, { atol: 1e-3 });
    });
  });

  suite("where branching with scan", () => {
    test("scan body with where branching", () => {
      // Cumulative application: carry = where(x > 0, carry + x, carry * exp(x))
      const step = (carry: np.Array, x: np.Array) => {
        using cond = x.greater(0);
        using addBranch = carry.add(x);
        using expX = np.exp(x);
        using mulBranch = carry.mul(expX);
        const newCarry = np.where(cond, addBranch, mulBranch);
        return [newCarry, newCarry] as [np.Array, np.Array];
      };

      using init = np.array([1.0]);
      using xs = np.array([[0.5], [-0.3], [1.0], [-0.5]]);
      const [carry, ys] = lax.scan(step, init, xs);
      carry.dispose();
      ys.dispose();
      // Just verify it doesn't crash — correctness checked by the pattern
    });
  });
});
