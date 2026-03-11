import {
  type Device,
  grad,
  numpy as np,
  scipyLinalg,
} from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

import { deviceSuite } from "./device-suite.js";

const devicesWithLinalg: Device[] = ["cpu", "wasm", "webgpu"];

await deviceSuite(() => {
  suite("scipy.linalg.solveTriangular()", () => {
    test("solves lower triangular system", () => {
      using L = np.array([
        [2.0, 0.0],
        [1.0, 3.0],
      ]);
      using b = np.array([[4.0], [7.0]]);
      using x = scipyLinalg.solveTriangular(L, b, { lower: true });
      // L @ x = b → x = [2, 5/3]
      using Lx = np.matmul(L, x);
      expect(Lx).toBeAllclose(b, { atol: 1e-5 });
    });

    test("solves upper triangular system", () => {
      using U = np.array([
        [2.0, 1.0],
        [0.0, 3.0],
      ]);
      using b = np.array([[5.0], [6.0]]);
      using x = scipyLinalg.solveTriangular(U, b, { lower: false });
      // U @ x = b
      using Ux = np.matmul(U, x);
      expect(Ux).toBeAllclose(b, { atol: 1e-5 });
    });

    test("solves with trans=1 (transpose)", () => {
      using L = np.array([
        [2.0, 0.0],
        [1.0, 3.0],
      ]);
      using b = np.array([[5.0], [3.0]]);
      // trans=1: solve L^T @ x = b
      using x = scipyLinalg.solveTriangular(L, b, {
        lower: true,
        trans: 1,
      });
      using Lt = np.matrixTranspose(L);
      using Ltx = np.matmul(Lt, x);
      expect(Ltx).toBeAllclose(b, { atol: 1e-5 });
    });

    test("solves with unit diagonal", () => {
      using L = np.array([
        [999.0, 0.0],
        [2.0, 999.0],
      ]);
      using b = np.array([[3.0], [8.0]]);
      // unitDiagonal=true: treat diagonal as 1
      using x = scipyLinalg.solveTriangular(L, b, {
        lower: true,
        unitDiagonal: true,
      });
      // Effective L = [[1,0],[2,1]], solution: x1=3, x2=8-2*3=2
      expect(x).toBeAllclose([[3.0], [2.0]]);
    });

    test("solves 3x3 system", () => {
      using L = np.array([
        [3.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [4.0, 5.0, 6.0],
      ]);
      using b = np.array([[9.0], [5.0], [46.0]]);
      using x = scipyLinalg.solveTriangular(L, b, { lower: true });
      using Lx = np.matmul(L, x);
      expect(Lx).toBeAllclose(b, { atol: 1e-4 });
    });

    test("gradient through solveTriangular", () => {
      const f = (b: np.Array) => {
        using L = np.array([
          [2.0, 0.0],
          [1.0, 3.0],
        ]);
        using x = scipyLinalg.solveTriangular(L, b, { lower: true });
        return x.sum();
      };
      using b = np.array([[4.0], [7.0]]);
      using db = grad(f)(b);
      // gradient should be finite
      const vals = (db.js() as number[][]).flat();
      for (const v of vals) {
        expect(isFinite(v)).toBe(true);
      }
    });
  });

  suite("scipy.linalg.cholesky()", () => {
    test("SciPy convention: default returns upper Cholesky", () => {
      using A = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      // SciPy default is upper Cholesky
      using U = scipyLinalg.cholesky(A);
      // Verify: U^T @ U ≈ A
      using Ut = np.matrixTranspose(U);
      using reconstructed = np.matmul(Ut, U);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-5 });
    });

    test("lower=true returns lower Cholesky", () => {
      using A = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      using L = scipyLinalg.cholesky(A, { lower: true });
      // Verify: L @ L^T ≈ A
      using Lt = np.matrixTranspose(L);
      using reconstructed = np.matmul(L, Lt);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-5 });
    });
  });
}, devicesWithLinalg);
