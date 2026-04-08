/**
 * Tests for customRoot — implicit differentiation for root-finding problems.
 *
 * Covers:
 *   - Basic root finding (linear system Ax = b)
 *   - Forward pass correctness
 *   - Gradient through closure parameters via IFT
 *   - Scalar root finding
 *   - Composition with jit
 *   - VJP returns correct cotangents
 *   - Non-linear root (square root via Newton's method)
 *   - Multi-parameter closure gradient
 */
import {
  customRoot,
  grad,
  jit,
  numpy as np,
  valueAndGrad,
  vjp,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

describe("customRoot", () => {
  test("basic root: solve Ax = b as root of f(x) = Ax - b", () => {
    // Find x such that f(x) = diag * x - b = 0, i.e., x = b / diag
    using diag = np.array([2.0, 3.0, 4.0]);
    using b = np.array([6.0, 9.0, 12.0]);

    // Optimality condition: f(x) = diag * x - b
    const f = (x: np.Array) => {
      using s = new DisposableStack();
      return np.subtract(s.use(np.multiply(diag, x)), b);
    };

    // Solver: x = b / diag
    const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) =>
      np.divide(b, diag);

    // Linear solver: g(x) = y => (diag) * x = y => x = y / diag
    const tangentSolve = (_g: (x: np.Array) => np.Array, y: np.Array) =>
      np.divide(y, diag);

    using x0 = np.zeros([3]);
    using result = customRoot(f, x0, solve, tangentSolve);
    expect(result).toBeAllclose([3.0, 3.0, 3.0]);
  });

  test("gradient flows through closure parameters via IFT", () => {
    // f(x, b) = diag * x - b = 0 => x*(b) = b / diag
    // loss(b) = sum(x*(b)^2) = sum((b/diag)^2)
    // d(loss)/db_i = 2 * b_i / diag_i^2
    //
    // For b=[6,9,12], diag=[2,3,4]:
    //   d(loss)/db = [2*6/4, 2*9/9, 2*12/16] = [3, 2, 1.5]
    using diag = np.array([2.0, 3.0, 4.0]);

    const loss = (b: np.Array) => {
      const f = (x: np.Array) => {
        using s = new DisposableStack();
        return np.subtract(s.use(np.multiply(diag, x)), b);
      };
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) =>
        np.divide(b, diag);
      const tangentSolve = (_g: (x: np.Array) => np.Array, y: np.Array) =>
        np.divide(y, diag);

      using init = np.zeros([3]);
      using x = customRoot(f, init, solve, tangentSolve);
      return np.sum(x.mul(x));
    };

    using gradInput = np.array([6.0, 9.0, 12.0]);
    using db = grad(loss)(gradInput);
    expect(db).toBeAllclose([3.0, 2.0, 1.5]);
  });

  test("scalar root: solve a*x - b = 0", () => {
    // f(x) = 5*x - 15 = 0 => x* = 3
    // loss(b) = x*(b)^2 = (b/5)^2
    // d(loss)/db = 2*b/25 = 2*15/25 = 1.2
    const loss = (b: np.Array) => {
      const f = (x: np.Array) => {
        using s = new DisposableStack();
        return np.subtract(s.use(np.multiply(5.0, x)), b);
      };
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) =>
        np.divide(b, np.array(5.0));
      const tangentSolve = (_g: (x: np.Array) => np.Array, y: np.Array) =>
        np.divide(y, np.array(5.0));

      using init = np.array(0.0);
      using x = customRoot(f, init, solve, tangentSolve);
      return np.sum(x.mul(x));
    };

    using gradInput = np.array(15.0);
    using db = grad(loss)(gradInput);
    expect(db).toBeAllclose(1.2);
  });

  test("composition with jit", () => {
    using diag = np.array([2.0, 4.0]);

    const loss = (b: np.Array) => {
      const f = (x: np.Array) => {
        using s = new DisposableStack();
        return np.subtract(s.use(np.multiply(diag, x)), b);
      };
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) =>
        np.divide(b, diag);
      const tangentSolve = (_g: (x: np.Array) => np.Array, y: np.Array) =>
        np.divide(y, diag);

      using init = np.zeros([2]);
      using x = customRoot(f, init, solve, tangentSolve);
      return np.sum(x.mul(x));
    };

    using jitGradLoss = jit(grad(loss));
    using jitInput = np.array([4.0, 8.0]);
    using db = jitGradLoss(jitInput);
    // x = [2, 2], g = 2*x = [4, 4]
    // dx*/db = diag(1/diag) = diag(0.5, 0.25)
    // d(loss)/db = dx*/db^T * g = [4*0.5, 4*0.25] = ... wait
    // Actually: x = b/diag, loss = sum(x^2) = sum((b/diag)^2)
    // d(loss)/db_i = 2*b_i/diag_i^2
    // = [2*4/4, 2*8/16] = [2, 1]
    expect(db).toBeAllclose([2.0, 1.0]);
  });

  test("vjp returns correct cotangents", () => {
    using diag = np.array([2.0, 3.0]);

    const rootFn = (b: np.Array) => {
      const f = (x: np.Array) => {
        using s = new DisposableStack();
        return np.subtract(s.use(np.multiply(diag, x)), b);
      };
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) =>
        np.divide(b, diag);
      const tangentSolve = (_g: (x: np.Array) => np.Array, y: np.Array) =>
        np.divide(y, diag);
      using init = np.zeros([2]);
      return customRoot(f, init, solve, tangentSolve);
    };

    using bArr = np.array([4.0, 9.0]);
    const [result, vjpFn] = vjp(rootFn, [bArr]);
    using x = result as np.Array;
    expect(x).toBeAllclose([2.0, 3.0]); // b / diag

    // VJP with cotangent [1, 1]:
    // dx*/db = diag(1/diag) = diag(0.5, 1/3)
    // cotangent of b = (dx*/db)^T * [1,1] = [0.5, 0.333...]
    using ones = np.ones([2]);
    const cts = vjpFn(ones);
    using db = (cts as np.Array[])[0];
    expect(db).toBeAllclose([0.5, 0.3333333], { atol: 1e-5 });
    vjpFn.dispose();
  });

  test("non-linear root: square root via Newton's method", () => {
    // Find x such that x^2 - a = 0, i.e., x = sqrt(a)
    // f(x) = x^2 - a
    // ∂f/∂x = 2x, ∂f/∂a = -1
    // dx*/da = -(2x*)^{-1} * (-1) = 1/(2*sqrt(a))
    //
    // loss(a) = x*(a)^2 = a
    // d(loss)/da = 1  (trivially, but via IFT!)
    //
    // More interesting: loss(a) = x*(a) = sqrt(a)
    // d(loss)/da = 1/(2*sqrt(a))

    const sqrtLoss = (a: np.Array) => {
      const f = (x: np.Array) => {
        using s = new DisposableStack();
        return np.subtract(s.use(np.multiply(x, x)), a);
      };

      // Newton solver: x_{k+1} = x_k - f(x_k) / f'(x_k) = (x_k + a/x_k) / 2
      const solve = (_f: (x: np.Array) => np.Array, x0: np.Array) => {
        let x: np.Array = np.add(x0, np.array(0.0)); // copy to own
        for (let i = 0; i < 20; i++) {
          using prev = x;
          x = np.divide(np.add(prev, np.divide(a, prev)), np.array(2.0));
        }
        return x;
      };

      // Tangent solve: 2x* * v = y => v = y / (2x*)
      // (This is exact since ∂f/∂x = 2x* at the root)
      const tangentSolve = (_g: (v: np.Array) => np.Array, y: np.Array) => {
        // We know the Jacobian is 2*sqrt(a), so solve directly.
        // In general, tangentSolve would use the provided g function.
        using xStar = np.sqrt(a);
        return np.divide(y, np.multiply(np.array(2.0), xStar));
      };

      using init = np.array(1.0);
      using x = customRoot(f, init, solve, tangentSolve);
      return np.sum(x);
    };

    // Forward: sqrt(4) = 2
    using input4 = np.array(4.0);
    const [val, gradVal] = valueAndGrad(sqrtLoss)(input4);
    using v = val;
    using g = gradVal;
    expect(v).toBeAllclose(2.0, { atol: 1e-6 });
    // Gradient: d(sqrt(a))/da = 1/(2*sqrt(4)) = 0.25
    expect(g).toBeAllclose(0.25, { atol: 1e-5 });
  });

  test("gradient through multiple closure parameters", () => {
    // f(x; a, b) = a * x - b = 0 => x* = b / a
    // loss(a, b) = x*(a, b) = b / a
    // d(loss)/da = -b/a^2
    // d(loss)/db = 1/a

    const loss = (a: np.Array, b: np.Array) => {
      const f = (x: np.Array) => {
        using s = new DisposableStack();
        return np.subtract(s.use(np.multiply(a, x)), b);
      };
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) =>
        np.divide(b, a);
      const tangentSolve = (_g: (x: np.Array) => np.Array, y: np.Array) =>
        np.divide(y, a);

      using init = np.array(0.0);
      using x = customRoot(f, init, solve, tangentSolve);
      return np.sum(x);
    };

    using aArr = np.array(3.0);
    using bArr = np.array(6.0);

    // x* = 6/3 = 2
    // d(loss)/da = -6/9 = -0.666...
    // d(loss)/db = 1/3 = 0.333...
    const [result, vjpFn] = vjp(loss, [aArr, bArr]);
    using val = result as np.Array;
    expect(val).toBeAllclose(2.0);

    using ones = np.array(1.0);
    const cts = vjpFn(ones);
    using da = (cts as np.Array[])[0];
    using db = (cts as np.Array[])[1];
    expect(da).toBeAllclose(-0.6666667, { atol: 1e-5 });
    expect(db).toBeAllclose(0.3333333, { atol: 1e-5 });
    vjpFn.dispose();
  });

  test("initialGuess cotangent is zero (solver is opaque)", () => {
    // The gradient w.r.t. initialGuess should be zero because
    // the solver is opaque — customRoot blocks gradient flow through
    // solve's iterations.
    using diag = np.array([2.0, 3.0]);
    using b = np.array([4.0, 9.0]);

    const rootFn = (guess: np.Array) => {
      const f = (x: np.Array) => {
        using s = new DisposableStack();
        return np.subtract(s.use(np.multiply(diag, x)), b);
      };
      // Solver ignores guess (direct formula), but customRoot still
      // blocks gradient flow through solve.
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) =>
        np.divide(b, diag);
      const tangentSolve = (_g: (x: np.Array) => np.Array, y: np.Array) =>
        np.divide(y, diag);
      return customRoot(f, guess, solve, tangentSolve);
    };

    // When differentiating rootFn w.r.t. guess, the answer is zero:
    // x* depends on b and diag (closure), not on the initial guess.
    using guess = np.array([100.0, 200.0]);
    using dGuess = grad((g: np.Array) => np.sum(rootFn(g)))(guess);
    expect(dGuess).toBeAllclose([0.0, 0.0], { atol: 1e-6 });
  });

  test("tangentSolve may call matvec without disposing aliased primals", async () => {
    const f = (x: np.Array) => x;
    const solve = (_f: (x: np.Array) => np.Array, x0: np.Array) => x0;
    const tangentSolve = (g: (x: np.Array) => np.Array, y: np.Array) => g(y);

    using init = np.array(0.0);
    using out = customRoot(f, init, solve, tangentSolve);
    expect(out).toBeAllclose(0.0);
  });

  test("gradient remains correct when tangentSolve evaluates aliased matvec", () => {
    const loss = (b: np.Array) => {
      const f = (x: np.Array) => np.subtract(x, b);
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) => b;
      const tangentSolve = (g: (x: np.Array) => np.Array, y: np.Array) => g(y);

      using init = np.array(0.0);
      using x = customRoot(f, init, solve, tangentSolve);
      return np.sum(x.mul(x));
    };

    using input = np.array(3.0);
    using db = grad(loss)(input);
    expect(db).toBeAllclose(6.0, { atol: 1e-6 });
  });

  test("jit(grad) works when tangentSolve evaluates aliased matvec", () => {
    const loss = (b: np.Array) => {
      const f = (x: np.Array) => np.subtract(x, b);
      const solve = (_f: (x: np.Array) => np.Array, _x0: np.Array) => b;
      const tangentSolve = (g: (x: np.Array) => np.Array, y: np.Array) => g(y);

      using init = np.array(0.0);
      using x = customRoot(f, init, solve, tangentSolve);
      return np.sum(x.mul(x));
    };

    using compiled = jit(grad(loss));
    using input = np.array(3.0);
    using db = compiled(input);
    expect(db).toBeAllclose(6.0, { atol: 1e-6 });
  });
});
