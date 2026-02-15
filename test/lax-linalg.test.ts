import {
  defaultDevice,
  Device,
  grad,
  init,
  jvp,
  lax,
  numpy as np,
  random,
} from "@jax-js-nonconsuming/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();
const devicesWithLinalg: Device[] = ["cpu", "wasm", "webgpu"];

suite.each(devicesWithLinalg)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.lax.linalg.cholesky()", () => {
    test("computes lower Cholesky decomposition for 2x2 matrix", () => {
      using x = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
      ]);
      using L = lax.linalg.cholesky(x);

      // L should be lower triangular
      const LData = L.js();
      expect(LData[0][1]).toBeCloseTo(0);
      expect(LData[1][0]).not.toBe(0);

      // Verify: L @ L^T should equal x
      using Lt = L.transpose();
      using reconstructed = np.matmul(L, Lt);
      expect(reconstructed).toBeAllclose(x);
    });

    test("computes Cholesky decomposition for 3x3 matrix", () => {
      using x = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 3.0, 6.0],
      ]);
      using L = lax.linalg.cholesky(x);

      // Verify: L @ L^T should equal x
      using Lt = L.transpose();
      using reconstructed = np.matmul(L, Lt);
      expect(reconstructed).toBeAllclose(x);
    });

    test("throws on non-square matrix", () => {
      using x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      expect(() => lax.linalg.cholesky(x).js()).toThrow();
    });

    test("throws on non-2D array", () => {
      using x = np.array([1.0, 2.0, 3.0]);
      expect(() => lax.linalg.cholesky(x).js()).toThrow();
    });

    test("works with jvp", () => {
      using x = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      using dx = np.array([
        [0.1, 0.05],
        [0.05, 0.1],
      ]);
      const jvpResult = jvp(lax.linalg.cholesky, [x], [dx]);
      using L = jvpResult[0];
      using dL = jvpResult[1];

      // Verify L is correct
      using Lt = L.transpose();
      using LLt = np.matmul(L, Lt);
      expect(LLt).toBeAllclose(x);

      // Verify dL by finite differences: (cholesky(x + eps*dx) - L) / eps ≈ dL
      const eps = 1e-4;
      using dxe = dx.mul(eps);
      using xpe = x.add(dxe);
      using L2 = lax.linalg.cholesky(xpe);
      using L2subL = L2.sub(L);
      using dL_fd = L2subL.div(eps);
      expect(dL).toBeAllclose(dL_fd, { rtol: 1e-2, atol: 2e-3 });
    });

    test("works with grad", () => {
      using x = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      // Loss: sum of squared elements of L
      const f = (x: np.Array) => {
        using xt = x.transpose();
        using xPlusXt = x.add(xt);
        using sym = xPlusXt.mul(0.5); // Ensure symmetry
        using L = lax.linalg.cholesky(sym);
        using sq = np.square(L);
        return sq.sum();
      };
      using dx = grad(f)(x);

      // Verify gradient by finite differences
      const eps = 1e-4;
      const xData = x.js() as number[][];
      const expected: number[][] = [[], []];
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          const xp = xData.map((row) => [...row]);
          const xm = xData.map((row) => [...row]);
          xp[i][j] += eps;
          xm[i][j] -= eps;
          using arrP = np.array(xp);
          using fpArr = f(arrP);
          const fp = fpArr.js() as number;
          using arrM = np.array(xm);
          using fmArr = f(arrM);
          const fm = fmArr.js() as number;
          expected[i][j] = (fp - fm) / (2 * eps);
        }
      }
      expect(dx).toBeAllclose(expected, { rtol: 1e-2, atol: 1e-3 });
    });
  });

  suite("jax.lax.linalg.lu()", () => {
    test("example with partial pivoting", () => {
      using A = np.array([
        [4, 3],
        [6, 3],
      ]);
      const luResult = lax.linalg.lu(A);
      using lu = luResult[0];
      using pivots = luResult[1];
      using permutation = luResult[2];
      expect(lu).toBeAllclose([
        [6, 3],
        [0.6666667, 1.0],
      ]);
      expect(pivots.js()).toEqual([1, 1]);
      expect(permutation.js()).toEqual([1, 0]);
    });

    test("P @ A = L @ U holds", () => {
      const n = 30;
      using key = random.key(0);
      using A = random.uniform(key, [n, n]);
      const luResult = lax.linalg.lu(A);
      using lu = luResult[0];
      using _pivots = luResult[1];
      using permutation = luResult[2];
      using eye1 = np.eye(n);
      using P = eye1.slice(permutation);
      using trilLu = np.tril(lu, -1);
      using eye2 = np.eye(n);
      using L = trilLu.add(eye2);
      using U = np.triu(lu);

      using PA = np.matmul(P, A);
      using LU = np.matmul(L, U);
      expect(PA).toBeAllclose(LU, { rtol: 1e-5, atol: 1e-6 });
    });

    test("works with jvp", () => {
      using A = np.array([
        [4.0, 3.0, 6.3],
        [6.0, 3.0, -2.4],
      ]);
      using dA = np.array([
        [0.1, 0.2, -0.2],
        [0.3, 0.4, -0.1],
      ]);

      const luFn = (x: np.Array) => {
        const luResult = lax.linalg.lu(x);
        using _pivots = luResult[1];
        using _perm = luResult[2];
        return luResult[0];
      };
      const jvpResult = jvp(luFn, [A], [dA]);
      using lu = jvpResult[0];
      using dlu = jvpResult[1];

      // Verify dlu by finite differences.
      // Use larger eps (1e-3) and looser tolerance because f32 finite
      // differences are inherently noisy (the WASM LU routine uses native
      // f32 arithmetic, amplifying rounding in the FD quotient).
      const eps = 1e-3;
      using dAe = dA.mul(eps);
      using Ape = A.add(dAe);
      const lu2Result = lax.linalg.lu(Ape);
      using lu2 = lu2Result[0];
      using _lu2p = lu2Result[1];
      using _lu2perm = lu2Result[2];
      using lu2sublu = lu2.sub(lu);
      using dlu_fd = lu2sublu.div(eps);
      expect(dlu).toBeAllclose(dlu_fd, { rtol: 2e-2, atol: 2e-3 });
    });

    test("works with jvp (f64)", () => {
      if (device !== "wasm" && device !== "cpu") return; // f64 on CPU + WASM
      using A = np.array(
        [
          [4.0, 3.0, 6.3],
          [6.0, 3.0, -2.4],
        ],
        { dtype: np.float64 },
      );
      using dA = np.array(
        [
          [0.1, 0.2, -0.2],
          [0.3, 0.4, -0.1],
        ],
        { dtype: np.float64 },
      );

      const luFn = (x: np.Array) => {
        const luResult = lax.linalg.lu(x);
        using _pivots = luResult[1];
        using _perm = luResult[2];
        return luResult[0];
      };
      const jvpResult = jvp(luFn, [A], [dA]);
      using lu = jvpResult[0];
      using dlu = jvpResult[1];

      // f64 allows tighter eps and tolerance than f32
      const eps = 1e-6;
      using dAe = dA.mul(eps);
      using Ape = A.add(dAe);
      const lu2Result = lax.linalg.lu(Ape);
      using lu2 = lu2Result[0];
      using _lu2p = lu2Result[1];
      using _lu2perm = lu2Result[2];
      using lu2sublu = lu2.sub(lu);
      using dlu_fd = lu2sublu.div(eps);
      expect(dlu).toBeAllclose(dlu_fd, { rtol: 1e-4, atol: 1e-5 });
    });
  });

  suite("jax.lax.linalg.triangularSolve()", () => {
    test("solves lower-triangular system", () => {
      // Solve L @ x = b
      using L = np.array([
        [2, 0],
        [1, 3],
      ]);
      using b0 = np.array([4, 7]);
      using b = b0.reshape([2, 1]);
      using x = lax.linalg.triangularSolve(L, b, {
        leftSide: true,
        lower: true,
      });
      expect(x).toBeAllclose([[2], [5 / 3]]);
    });

    test("works with jvp on b", () => {
      using L = np.array([
        [2, 0],
        [1, 3],
      ]);
      using b = np.array([[4], [7]]);
      using db = np.array([[0.1], [0.2]]);

      const solve = (b: np.Array) =>
        lax.linalg.triangularSolve(L, b, { leftSide: true, lower: true });
      const jvpResult = jvp(solve, [b], [db]);
      using x = jvpResult[0];
      using dx = jvpResult[1];

      // Verify x is correct
      using Lx = np.matmul(L, x);
      expect(Lx).toBeAllclose(b);

      // Verify dx by finite differences
      const eps = 1e-4;
      using dbe = db.mul(eps);
      using bpe = b.add(dbe);
      using x2 = lax.linalg.triangularSolve(L, bpe, {
        leftSide: true,
        lower: true,
      });
      using x2subx = x2.sub(x);
      using dx_fd = x2subx.div(eps);
      expect(dx).toBeAllclose(dx_fd, { rtol: 1e-2, atol: 1e-3 });
    });

    test("works with grad on b", () => {
      using L = np.array([
        [2, 0],
        [1, 3],
      ]);
      using b = np.array([[4], [7]]);

      // Loss: sum of squared elements of solution
      const f = (b: np.Array) => {
        using sol = lax.linalg.triangularSolve(L, b, {
          leftSide: true,
          lower: true,
        });
        using sq = np.square(sol);
        return sq.sum();
      };
      using db = grad(f)(b);

      // Verify gradient by finite differences
      const eps = 1e-4;
      const bData = b.js() as number[][];
      const expected: number[][] = [[], []];
      for (let i = 0; i < 2; i++) {
        const bp = bData.map((row) => [...row]);
        const bm = bData.map((row) => [...row]);
        bp[i][0] += eps;
        bm[i][0] -= eps;
        using arrP = np.array(bp);
        using fpArr = f(arrP);
        const fp = fpArr.js() as number;
        using arrM = np.array(bm);
        using fmArr = f(arrM);
        const fm = fmArr.js() as number;
        expected[i][0] = (fp - fm) / (2 * eps);
      }
      expect(db).toBeAllclose(expected, { rtol: 1e-2, atol: 1e-3 });
    });

    test("behavior with transposed A", () => {
      // See: https://github.com/ekzhang/jax-js/issues/73
      using L = np.array([
        [1, 1000000],
        [1, 1],
      ]);
      using b = np.array([[1], [1]]);
      using x = lax.linalg.triangularSolve(L, b, {
        leftSide: true,
        lower: true,
        transposeA: true,
      });
      expect(x).toBeAllclose([[0], [1]]);
    });

    test("right-hand side triangular solve", () => {
      // Solve x @ U = b
      using U = np.array([
        [2, 1],
        [0, 3],
      ]);
      using b = np.array([[4, 7]]);
      using x = lax.linalg.triangularSolve(U, b, {
        leftSide: false,
        lower: false,
      });
      expect(x).toBeAllclose([[2, 5 / 3]]);
    });
  });
});
