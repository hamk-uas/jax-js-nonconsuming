/**
 * Tests for customVjp — user-defined reverse-mode AD rules.
 *
 * Covers:
 *   - Basic custom gradient override
 *   - Numerically stable gradients (log1pexp)
 *   - Correct residual capture and usage
 *   - Composition with jit
 *   - Composition with grad (scalar output)
 *   - Multiple arguments
 *   - Pytree residuals
 *   - Forward pass correctness (same result as calling fwd directly)
 *   - Implicit differentiation pattern (the primary jaxopt use case)
 */
import {
  customVjp,
  grad,
  jit,
  numpy as np,
  valueAndGrad,
  vjp,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

describe("customVjp", () => {
  test("basic: override gradient to return a constant", () => {
    // f(x) = x^2, but override gradient to always return 42
    const f = customVjp(
      (x: np.Array) => {
        const out = x.mul(x);
        return [out, null]; // no residuals needed
      },
      (_res: null, g: np.Array) => {
        // Instead of 2x, always return 42 * g
        return np.multiply(42, g);
      },
    );

    // Forward pass: should compute x^2
    using y = f(np.array(3.0));
    expect(y).toBeAllclose(9.0);

    // Backward pass: should return 42 (not 2*3 = 6)
    using dx = grad(f)(np.array(3.0));
    expect(dx).toBeAllclose(42.0);
  });

  test("numerically stable log1pexp", () => {
    // log(1 + exp(x)) is numerically unstable for large x
    // (exp(100) overflows to Inf, log(Inf) = Inf, but the true value ≈ x)
    const log1pexp = customVjp(
      (x: np.Array) => {
        using exp_x = np.exp(x);
        using one = np.array(1.0);
        using one_plus_exp = np.add(one, exp_x);
        const out = np.log(one_plus_exp);
        return [out, x]; // save x for backward
      },
      (x: np.Array, g: np.Array) => {
        // d/dx log(1+exp(x)) = sigmoid(x) = 1/(1+exp(-x))
        using neg_x = np.negative(x);
        using exp_neg = np.exp(neg_x);
        using one = np.array(1.0);
        using denom = np.add(one, exp_neg);
        using sigmoid = np.reciprocal(denom);
        return np.multiply(g, sigmoid);
      },
    );

    // For large x, standard autodiff would give NaN
    // but our custom backward gives the correct sigmoid(x) ≈ 1
    using dx = grad(log1pexp)(np.array(100.0));
    expect(dx).toBeAllclose(1.0);

    // For moderate x, should be close to sigmoid(x)
    using dx2 = grad(log1pexp)(np.array(0.0));
    expect(dx2).toBeAllclose(0.5);
  });

  test("forward pass is correct", () => {
    const f = customVjp(
      (x: np.Array) => {
        using ten = np.array(10.0);
        const y = np.add(x, ten);
        return [y, null];
      },
      (_res: null, g: np.Array) => g,
    );

    using result = f(np.array(5.0));
    expect(result).toBeAllclose(15.0);
  });

  test("works with vjp directly", () => {
    const f = customVjp(
      (x: np.Array) => {
        using three = np.array(3.0);
        const y = x.mul(three);
        return [y, x]; // residual: x
      },
      (x: np.Array, g: np.Array) => {
        // Custom gradient: 3 * g + x (instead of just 3 * g)
        using scaled = np.multiply(3.0, g);
        return np.add(scaled, x);
      },
    );

    using x = np.array(2.0);
    const [y, vjpFn] = vjp(f, [x]);
    using yArr = y as np.Array;
    expect(yArr).toBeAllclose(6.0); // 2 * 3

    using ones = np.ones([]);
    const cotangents = vjpFn(ones);
    using dx = (cotangents as np.Array[])[0];
    expect(dx).toBeAllclose(5.0); // 3 * 1 + 2
    vjpFn.dispose();
  });

  test("multiple arguments", () => {
    const f = customVjp(
      (x: np.Array, y: np.Array) => {
        using xy = x.mul(y);
        const out = np.add(xy, y);
        return [out, { x, y }]; // save both
      },
      (res: { x: np.Array; y: np.Array }, g: np.Array) => {
        // Custom gradients: dx = 2*y*g, dy = x*g
        using twoY = np.multiply(2.0, res.y);
        const dx = np.multiply(twoY, g);
        const dy = np.multiply(res.x, g);
        return [dx, dy];
      },
    );

    // Differentiate w.r.t. first arg
    using dx = grad(f)(np.array(3.0), np.array(4.0));
    expect(dx).toBeAllclose(8.0); // 2 * 4 * 1
  });

  test("composition with jit", () => {
    const f = customVjp(
      (x: np.Array) => {
        const y = x.mul(x);
        return [y, x];
      },
      (x: np.Array, g: np.Array) => {
        // Custom: 3x * g instead of 2x * g
        using three_x = np.multiply(3.0, x);
        return np.multiply(three_x, g);
      },
    );

    using jitGrad = jit(grad(f));
    using x = np.array(5.0);
    using dx = jitGrad(x);
    expect(dx).toBeAllclose(15.0); // 3 * 5
  });

  test("valueAndGrad with customVjp", () => {
    const f = customVjp(
      (x: np.Array) => {
        const y = x.mul(x);
        return [y, x];
      },
      (x: np.Array, g: np.Array) => {
        return np.multiply(7.0, g); // always 7
      },
    );

    const [value, dx] = valueAndGrad(f)(np.array(4.0));
    using v = value as np.Array;
    using d = dx as np.Array;
    expect(v).toBeAllclose(16.0); // 4^2
    expect(d).toBeAllclose(7.0); // custom gradient
  });

  test("vector input/output", () => {
    const f = customVjp(
      (x: np.Array) => {
        using two = np.array(2.0);
        const y = x.mul(two);
        return [y, x];
      },
      (x: np.Array, g: np.Array) => {
        // Custom: return g + 1 instead of 2*g
        using one = np.array(1.0);
        return np.add(g, one);
      },
    );

    // grad(sum(f(x)))
    const loss = (x: np.Array) => {
      using y = f(x);
      return np.sum(y);
    };
    using x = np.array([1.0, 2.0, 3.0]);
    using dx = grad(loss)(x);
    // g = [1,1,1] from sum, custom backward: g + 1 = [2,2,2]
    expect(dx).toBeAllclose([2.0, 2.0, 2.0]);
  });

  test("implicit differentiation pattern", () => {
    // Simulate the core jaxopt implicit diff pattern:
    // Given a solver that finds x* such that ∇f(x*, θ) = 0,
    // compute dx*/dθ via the implicit function theorem.
    //
    // Example: f(x, θ) = (x - θ)^2, solution x* = θ, dx*/dθ = 1
    // Optimality: ∂f/∂x = 2(x - θ) = 0 → x* = θ
    // By IFT: dx*/dθ = -(∂²f/∂x²)⁻¹ · ∂²f/∂x∂θ = -(2)⁻¹ · (-2) = 1

    const solve = customVjp(
      (theta: np.Array) => {
        // "Solve" for x* = θ (in real code this would be an iterative solver)
        const xStar = theta;
        return [xStar, { xStar, theta }];
      },
      (res: { xStar: np.Array; theta: np.Array }, g: np.Array) => {
        // Implicit gradient: dx*/dθ = 1 for this simple case
        // In general: -(∂²f/∂x²)⁻¹ · (∂²f/∂x∂θ) · g
        // For f = (x-θ)², ∂²f/∂x² = 2, ∂²f/∂x∂θ = -2
        // So dx*/dθ = -2⁻¹ · (-2) = 1, and grad = 1 · g = g
        return g;
      },
    );

    // Downstream loss: L(θ) = (x*(θ))^2 = θ^2
    // dL/dθ = dL/dx* · dx*/dθ = 2θ · 1 = 2θ
    const loss = (theta: np.Array) => {
      using xStar = solve(theta);
      return np.sum(xStar.mul(xStar));
    };

    using dtheta = grad(loss)(np.array(3.0));
    expect(dtheta).toBeAllclose(6.0); // 2 * 3
  });

  test("residuals are disconnected from outer differentiation", () => {
    // Verify that residuals don't cause double-counting in gradients.
    // The forward pass uses x twice (once in output, once in residual),
    // but the residual path should be fully detached.
    let bwdCalled = false;
    const f = customVjp(
      (x: np.Array) => {
        using five = np.array(5.0);
        const y = x.mul(five);
        return [y, x];
      },
      (x: np.Array, g: np.Array) => {
        bwdCalled = true;
        return np.multiply(5.0, g); // correct gradient for 5x
      },
    );

    using dx = grad(f)(np.array(2.0));
    expect(bwdCalled).toBe(true);
    expect(dx).toBeAllclose(5.0);
  });

  test("pullback survives after output disposal (residual aliasing)", () => {
    // Residuals may alias the output. The pullback must independently retain
    // residual leaves so it works even after the caller disposes the output.
    const f = customVjp(
      (x: np.Array) => {
        const y = x.mul(x); // output
        return [y, y]; // residual aliases output
      },
      (res: np.Array, g: np.Array) => {
        // bwd: res * g  (res is the aliased output y = x^2)
        return np.multiply(res, g);
      },
    );

    using x = np.array(3.0);
    const [y, vjpFn] = vjp(f, [x]);
    // Dispose output BEFORE calling pullback — must not crash.
    (y as np.Array).dispose();
    using ones = np.ones([]);
    const cts = vjpFn(ones);
    using dx = (cts as np.Array[])[0];
    // y = x^2 = 9, custom bwd: res * g = 9 * 1 = 9
    expect(dx).toBeAllclose(9.0);
    vjpFn.dispose();
  });

  test("backward with wrong cotangent structure throws", () => {
    const f = customVjp(
      (x: np.Array) => [x, null],
      // Return an object instead of an array — wrong structure
      (_res: null, g: np.Array) => ({ wrong: g }),
    );

    using x = np.array([1.0, 2.0]);
    const [y, vjpFn] = vjp(f, [x]);
    using _yArr = y as np.Array;
    using ones = np.ones([2]);
    expect(() => vjpFn(ones)).toThrow(/customVjp backward/);
    vjpFn.dispose();
  });
});
