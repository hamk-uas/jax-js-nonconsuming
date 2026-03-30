/**
 * Tests for customLinearSolve — implicit differentiation for linear solves.
 *
 * Covers:
 *   - Basic symmetric linear solve (diagonal matrix)
 *   - Forward pass correctness
 *   - Gradient through symmetric solve
 *   - Non-symmetric solve with transposeSolve
 *   - Composition with jit
 *   - Matrix-vector linear system
 */
import {
  customLinearSolve,
  grad,
  jit,
  numpy as np,
  vjp,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

describe("customLinearSolve", () => {
  test("basic symmetric solve: diagonal system", () => {
    // A = diag([2, 3, 4]), b = [6, 9, 12]
    // Solution: x = [3, 3, 3]
    // Since A is diagonal, matvec(x) = [2x₁, 3x₂, 4x₃]
    // Solve: x_i = b_i / a_i
    using diag = np.array([2.0, 3.0, 4.0]);
    using b = np.array([6.0, 9.0, 12.0]);

    const x = customLinearSolve(
      (v: np.Array) => np.multiply(diag, v), // matvec: diag * v
      b,
      (_matvec: (x: np.Array) => np.Array, rhs: np.Array) =>
        np.divide(rhs, diag), // solve: b / diag
      { symmetric: true },
    );
    using result = x;
    expect(result).toBeAllclose([3.0, 3.0, 3.0]);
  });

  test("gradient through symmetric diagonal solve", () => {
    // A = diag([2, 3, 4]), minimize ||x(b)||^2 w.r.t. b
    // x(b) = A^{-1} b = b / diag
    // loss = sum(x^2) = sum((b/diag)^2)
    // d(loss)/db_i = 2 * b_i / diag_i^2
    //
    // But with implicit diff:
    //   VJP w.r.t. b: g → A^{-T} g = A^{-1} g (symmetric)
    //   g = d(loss)/dx = 2*x = 2*b/diag
    //   So d(loss)/db = A^{-1} (2*b/diag) = 2*b/diag^2
    //
    // For b=[6,9,12], diag=[2,3,4]:
    //   d(loss)/db = [2*6/4, 2*9/9, 2*12/16] = [3, 2, 1.5]

    using diag = np.array([2.0, 3.0, 4.0]);

    const loss = (b: np.Array) => {
      using x = customLinearSolve(
        (v: np.Array) => np.multiply(diag, v),
        b,
        (_matvec: (x: np.Array) => np.Array, rhs: np.Array) =>
          np.divide(rhs, diag),
        { symmetric: true },
      );
      return np.sum(x.mul(x));
    };

    using gradInput = np.array([6.0, 9.0, 12.0]);
    using db = grad(loss)(gradInput);
    expect(db).toBeAllclose([3.0, 2.0, 1.5]);
  });

  test("scalar symmetric solve", () => {
    // A = 5 (scalar), b = 15, x = 3
    // grad loss(b) = sum((b/5)^2) w.r.t. b
    // = 2*b/25 = 2*15/25 = 1.2
    const loss = (b: np.Array) => {
      using x = customLinearSolve(
        (v: np.Array) => np.multiply(5.0, v),
        b,
        (_mv: (x: np.Array) => np.Array, rhs: np.Array) =>
          np.divide(rhs, np.array(5.0)),
        { symmetric: true },
      );
      return np.sum(x.mul(x));
    };

    // Forward: x = 15/5 = 3, loss = 9
    using gradInput = np.array(15.0);
    using db = grad(loss)(gradInput);
    expect(db).toBeAllclose(1.2); // 2 * 15 / 25
  });

  test("composition with jit", () => {
    using diag = np.array([2.0, 4.0]);

    const loss = (b: np.Array) => {
      using x = customLinearSolve(
        (v: np.Array) => np.multiply(diag, v),
        b,
        (_mv: (x: np.Array) => np.Array, rhs: np.Array) => np.divide(rhs, diag),
        { symmetric: true },
      );
      return np.sum(x.mul(x));
    };

    using jitGradLoss = jit(grad(loss));
    using jitInput = np.array([4.0, 8.0]);
    using db = jitGradLoss(jitInput);
    // x = [2, 2], g = 2*x = [4, 4]
    // A^{-1} g = [4/2, 4/4] = [2, 1]
    expect(db).toBeAllclose([2.0, 1.0]);
  });

  test("non-symmetric solve with explicit transposeSolve", () => {
    // A = [[2, 1], [0, 3]]  (upper triangular, non-symmetric)
    // A^{-1} = [[0.5, -1/6], [0, 1/3]]
    // A^{-T} = [[0.5, 0], [-1/6, 1/3]]
    using A = np.array([
      [2, 1],
      [0, 3],
    ]);
    using Ainv = np.array([
      [0.5, -1.0 / 6.0],
      [0.0, 1.0 / 3.0],
    ]);
    using AinvT = np.array([
      [0.5, 0.0],
      [-1.0 / 6.0, 1.0 / 3.0],
    ]);

    const loss = (b: np.Array) => {
      using x = customLinearSolve(
        // matvec: A * v
        (v: np.Array) => np.matmul(A, v),
        b,
        // solve: A^{-1} * rhs
        (_mv: (x: np.Array) => np.Array, rhs: np.Array) => np.matmul(Ainv, rhs),
        {
          // Transpose solve: A^{-T} * rhs
          transposeSolve: (_vecmat: (x: np.Array) => np.Array, rhs: np.Array) =>
            np.matmul(AinvT, rhs),
        },
      );
      return np.sum(x.mul(x));
    };

    // b = [5, 6], x = A^{-1} b = [0.5*5 - 6/6, 6/3] = [1.5, 2]
    // loss = 1.5^2 + 2^2 = 6.25
    // g = 2*x = [3, 4]
    // db = A^{-T} g = [0.5*3, -3/6 + 4/3] = [1.5, 0.833...]
    using gradInput = np.array([5.0, 6.0]);
    using db = grad(loss)(gradInput);
    expect(db).toBeAllclose([1.5, 0.8333333], { atol: 1e-5 });
  });

  test("vjp returns correct cotangents", () => {
    using diag = np.array([2.0, 3.0]);

    const solveFn = (b: np.Array) => {
      return customLinearSolve(
        (v: np.Array) => np.multiply(diag, v),
        b,
        (_mv: (x: np.Array) => np.Array, rhs: np.Array) => np.divide(rhs, diag),
        { symmetric: true },
      );
    };

    using bArr = np.array([4.0, 9.0]);
    const [result, vjpFn] = vjp(solveFn, [bArr]);
    using x = result as np.Array;
    expect(x).toBeAllclose([2.0, 3.0]); // b / diag

    // cotangent g = [1, 1], VJP = A^{-1} g = [0.5, 0.333...]
    using ones = np.ones([2]);
    const cts = vjpFn(ones);
    using db = (cts as np.Array[])[0];
    expect(db).toBeAllclose([0.5, 0.3333333], { atol: 1e-5 });
    vjpFn.dispose();
  });
});
