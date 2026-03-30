/**
 * Tests for customJvp — user-defined forward-mode AD rules.
 *
 * Covers:
 *   - Basic custom JVP override
 *   - Numerically stable gradients (log1pexp via JVP)
 *   - Forward pass correctness (same result as calling fn directly)
 *   - Composition with jvp
 *   - Composition with grad (via transpose of custom JVP)
 *   - Composition with jit
 *   - Multiple arguments
 *   - Vector input/output
 *   - Error when customVjp function is used with jvp
 */
import {
  customJvp,
  customVjp,
  grad,
  jit,
  jvp,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

describe("customJvp", () => {
  test("basic: override tangent to return a constant", () => {
    // f(x) = x^2, but override tangent to always return 42 * tangent
    const f = customJvp(
      (x: np.Array) => x.mul(x),
      (primals, tangents) => {
        const [x] = primals as [np.Array];
        const [dx] = tangents as [np.Array];
        const xSq = x.mul(x);
        const tangentOut = np.multiply(42.0, dx);
        return [xSq, tangentOut];
      },
    );

    // Forward pass: should compute x^2
    using y = f(np.array(3.0));
    expect(y).toBeAllclose(9.0);

    // JVP: custom tangent = 42 * dx (not the true 2*x*dx = 6)
    using px = np.array(3.0);
    using tx = np.array(1.0);
    const [primal, tangent] = jvp(f, [px], [tx]);
    using p = primal;
    using t = tangent;
    expect(p).toBeAllclose(9.0);
    expect(t).toBeAllclose(42.0);
  });

  test("numerically stable log1pexp via custom JVP", () => {
    // log(1 + exp(x)) — unstable for large x via standard AD
    // Custom JVP: tangent = sigmoid(x) * dx
    const log1pexp = customJvp(
      (x: np.Array) => {
        using expX = np.exp(x);
        using onePlusExp = np.add(1.0, expX);
        return np.log(onePlusExp);
      },
      (primals, tangents) => {
        const [x] = primals as [np.Array];
        const [dx] = tangents as [np.Array];
        // Stable primal computation
        using expX = np.exp(x);
        using onePlusExp = np.add(1.0, expX);
        const primal = np.log(onePlusExp);
        // Stable tangent: sigmoid(x) * dx
        using negX = np.negative(x);
        using expNeg = np.exp(negX);
        using denom = np.add(1.0, expNeg);
        using sigmoid = np.reciprocal(denom);
        const tangent = np.multiply(sigmoid, dx);
        return [primal, tangent];
      },
    );

    // For large x, standard AD gives NaN; custom JVP gives sigmoid(100) ≈ 1
    using px = np.array(100.0);
    using tx = np.array(1.0);
    const [p_, t] = jvp(log1pexp, [px], [tx]);
    using _p = p_;
    using tang = t;
    expect(tang).toBeAllclose(1.0);
  });

  test("forward pass is correct without differentiation", () => {
    const f = customJvp(
      (x: np.Array) => {
        using ten = np.array(10.0);
        return np.add(x, ten);
      },
      (primals, tangents) => {
        const [x] = primals as [np.Array];
        const [dx] = tangents as [np.Array];
        using ten = np.array(10.0);
        const primal = np.add(x, ten);
        return [primal, dx];
      },
    );

    using result = f(np.array(5.0));
    expect(result).toBeAllclose(15.0);
  });

  test("composition with grad via JVP transpose", () => {
    // customJvp should support reverse-mode via linearize→transpose
    const f = customJvp(
      (x: np.Array) => x.mul(x),
      (primals, tangents) => {
        const [x] = primals as [np.Array];
        const [dx] = tangents as [np.Array];
        const xSq = x.mul(x);
        // Custom tangent: 7x * dx (instead of true 2x * dx)
        using sevenX = np.multiply(7.0, x);
        const tangent = np.multiply(sevenX, dx);
        return [xSq, tangent];
      },
    );

    // grad should use the custom JVP rule's tangent, giving 7x
    using dx = grad(f)(np.array(3.0));
    expect(dx).toBeAllclose(21.0); // 7 * 3
  });

  test("composition with jit", () => {
    const f = customJvp(
      (x: np.Array) => {
        using three = np.array(3.0);
        return x.mul(three);
      },
      (primals, tangents) => {
        const [x] = primals as [np.Array];
        const [dx] = tangents as [np.Array];
        using three = np.array(3.0);
        const primal = x.mul(three);
        // Custom tangent: 5 * dx instead of 3 * dx
        const tangent = np.multiply(5.0, dx);
        return [primal, tangent];
      },
    );

    // jit(f) should preserve custom JVP behavior
    using px = np.array(2.0);
    using tx = np.array(1.0);
    const [p, t] = jvp(jit(f), [px], [tx]);
    using primal = p;
    using tangent = t;
    expect(primal).toBeAllclose(6.0); // 2 * 3
    expect(tangent).toBeAllclose(5.0); // custom: 5 * 1
  });

  test("multiple arguments", () => {
    const f = customJvp(
      (x: np.Array, y: np.Array) => {
        using xy = x.mul(y);
        return np.add(xy, y);
      },
      (primals, tangents) => {
        const [x, y] = primals as [np.Array, np.Array];
        const [dx, dy] = tangents as [np.Array, np.Array];
        using xy = x.mul(y);
        const primal = np.add(xy, y);
        // Custom: tangent = 2*y*dx + x*dy (instead of y*dx + (x+1)*dy)
        using twoY = np.multiply(2.0, y);
        using term1 = np.multiply(twoY, dx);
        using term2 = np.multiply(x, dy);
        const tangent = np.add(term1, term2);
        return [primal, tangent];
      },
    );

    // JVP with tangents [1, 0] should give 2*y = 8
    using px1 = np.array(3.0);
    using py1 = np.array(4.0);
    using dx1 = np.array(1.0);
    using dy1 = np.array(0.0);
    const [p1, t1] = jvp(f, [px1, py1], [dx1, dy1]);
    using primal1 = p1;
    using tangent1 = t1;
    expect(primal1).toBeAllclose(16.0); // 3*4 + 4
    expect(tangent1).toBeAllclose(8.0); // 2*4*1 + 3*0

    // JVP with tangents [0, 1] should give x = 3
    using px2 = np.array(3.0);
    using py2 = np.array(4.0);
    using dx2 = np.array(0.0);
    using dy2 = np.array(1.0);
    const [p2, t2] = jvp(f, [px2, py2], [dx2, dy2]);
    using _primal2 = p2;
    using tangent2 = t2;
    expect(tangent2).toBeAllclose(3.0); // 2*4*0 + 3*1
  });

  test("vector input/output with custom JVP", () => {
    const doubleIt = customJvp(
      (x: np.Array) => {
        using two = np.array(2.0);
        return x.mul(two);
      },
      (primals: [np.Array], tangents: [np.Array]) => {
        const [x] = primals;
        const [dx] = tangents;
        using two = np.array(2.0);
        const primal = x.mul(two);
        // Custom: return dx + 1 instead of 2*dx
        using one = np.array(1.0);
        const tangent = np.add(dx, one);
        return [primal, tangent];
      },
    );

    using px = np.array([1.0, 2.0, 3.0]);
    using tx = np.array([1.0, 1.0, 1.0]);
    const [p, t] = jvp(doubleIt, [px], [tx]);
    using primal = p;
    using tangent = t;
    expect(primal).toBeAllclose([2.0, 4.0, 6.0]);
    expect(tangent).toBeAllclose([2.0, 2.0, 2.0]); // [1,1,1] + 1
  });

  test("error when customVjp function used with jvp", () => {
    const f = customVjp(
      (x: np.Array) => {
        const out = x.mul(x);
        return [out, null];
      },
      (_res: null, g: np.Array) => np.multiply(2.0, g),
    );

    using ex = np.array(1.0);
    using edx = np.array(1.0);
    expect(() => jvp(f, [ex], [edx])).toThrow(/customVjp.*forward-mode/);
  });

  test("jit(grad(customJvp))", () => {
    const f = customJvp(
      (x: np.Array) => {
        using xSq = x.mul(x);
        return xSq.mul(x); // x^3
      },
      (primals: [np.Array], tangents: [np.Array]) => {
        const [x] = primals;
        const [dx] = tangents;
        using xSq2 = x.mul(x);
        const xCubed = xSq2.mul(x);
        // Custom tangent: 10*x^2 * dx (instead of true 3*x^2 * dx)
        using xSq = x.mul(x);
        using tenXSq = np.multiply(10.0, xSq);
        const tangent = np.multiply(tenXSq, dx);
        return [xCubed, tangent];
      },
    );

    using jitGradF = jit(grad(f));
    using inp = np.array(2.0);
    using dx = jitGradF(inp);
    expect(dx).toBeAllclose(40.0); // 10 * 2^2
  });
});
