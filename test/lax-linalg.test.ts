import {
  defaultDevice,
  Device,
  grad,
  init,
  jit,
  jvp,
  lax,
  numpy as np,
  random,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
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

  suite("jax.lax.linalg.qr()", () => {
    test("computes QR decomposition for 3x2 matrix", () => {
      using A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      const [Q, R] = lax.linalg.qr(A);
      using _Q = Q;
      using _R = R;

      // Q has shape [3, 2], R has shape [2, 2]
      expect(Q.shape).toEqual([3, 2]);
      expect(R.shape).toEqual([2, 2]);

      // Q^T @ Q ≈ I
      using Qt = np.matrixTranspose(Q);
      using QtQ = np.matmul(Qt, Q);
      using eye2 = np.eye(2);
      expect(QtQ).toBeAllclose(eye2, { atol: 1e-5 });

      // Q @ R ≈ A
      using reconstructed = np.matmul(Q, R);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-5 });
    });

    test("computes QR decomposition for square matrix", () => {
      using A = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
      ]);
      const [Q, R] = lax.linalg.qr(A);
      using _Q = Q;
      using _R = R;

      expect(Q.shape).toEqual([2, 2]);
      expect(R.shape).toEqual([2, 2]);

      // Verify Q^T Q ≈ I
      using Qt = np.matrixTranspose(Q);
      using QtQ = np.matmul(Qt, Q);
      using eye2 = np.eye(2);
      expect(QtQ).toBeAllclose(eye2, { atol: 1e-5 });

      // Verify Q R ≈ A
      using reconstructed = np.matmul(Q, R);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-5 });

      // R should be upper triangular
      const rData = R.js() as number[][];
      expect(Math.abs(rData[1][0])).toBeLessThan(1e-5);
    });

    test("computes QR for random 4x3 matrix", () => {
      const key = random.key(42);
      using A = random.uniform(key, [4, 3]);
      key.dispose();

      const [Q, R] = lax.linalg.qr(A);
      using _Q = Q;
      using _R = R;

      expect(Q.shape).toEqual([4, 3]);
      expect(R.shape).toEqual([3, 3]);

      // Q @ R ≈ A
      using reconstructed = np.matmul(Q, R);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-4 });

      // Q^T Q ≈ I
      using Qt = np.matrixTranspose(Q);
      using QtQ = np.matmul(Qt, Q);
      using eye3 = np.eye(3);
      expect(QtQ).toBeAllclose(eye3, { atol: 1e-4 });
    });

    test("QR with batched matrices", () => {
      using A = np.array([
        [
          [1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0],
        ],
        [
          [7.0, 8.0],
          [9.0, 10.0],
          [11.0, 12.0],
        ],
      ]);

      const [Q, R] = lax.linalg.qr(A);
      using _Q = Q;
      using _R = R;

      expect(Q.shape).toEqual([2, 3, 2]);
      expect(R.shape).toEqual([2, 2, 2]);

      // Verify Q @ R ≈ A for each batch
      using reconstructed = np.matmul(Q, R);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-4 });
    });

    // foriLoop AD has known internal leaks on CPU (3 slots: JVPTracer +
    // PartialEvalTracer for loop counter, sliceAt in transpose rule).
    // On WASM/WebGPU these land in JIT caches cleaned by _disposeAllJitCaches.
    // Skip on CPU since the QR polyfill only targets WASM/WebGPU.
    test.skipIf(device === "cpu")("foriLoop grad smoke test", () => {
      // Minimal test: grad of foriLoop(add x 3 times)
      const f = (x: np.Array) => {
        const result = lax.foriLoop(
          0,
          3,
          (_i: np.Array, carry: np.Array) => np.add(carry, x),
          np.array(0.0),
        );
        return np.sum(result);
      };
      const x = np.array(2.0);
      const dx = grad(f)(x);
      expect(dx.js()).toBeCloseTo(3.0);
      dx.dispose();
      x.dispose();
    });

    // TODO: foriLoop AD leaks intermediates from JVP/transpose rules.
    // On CPU, native QR has proper grad support. On WASM/WebGPU, the
    // foriLoop polyfill's grad creates uncleaned intermediates.
    test.skipIf(device !== "cpu")("gradient through QR decomposition", () => {
      // grad of sum(R) w.r.t. A
      const f = (A: np.Array) => {
        const [Q, R] = lax.linalg.qr(A);
        Q.dispose();
        return R.sum();
      };
      using A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      using dA = grad(f)(A);

      // Verify gradient is finite and non-zero
      const dAData = dA.js() as number[][];
      for (const row of dAData)
        for (const v of row) expect(Math.abs(v)).toBeLessThan(100);
      // At least one entry should be non-zero
      const maxAbs = Math.max(...dAData.flat().map((v) => Math.abs(v)));
      expect(maxAbs).toBeGreaterThan(1e-6);
      // foriLoop AD infrastructure has known leaks — drain before afterEach
    });

    test("JVP through QR decomposition", () => {
      using A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      using dA = np.eye(2);

      const f = (a: np.Array) => {
        const [Q, R] = lax.linalg.qr(a);
        Q.dispose();
        return R;
      };

      const [R, dR] = jvp(f, [A], [dA]);
      using _R = R;
      using _dR = dR;

      // dR should exist and have correct shape
      expect(dR.shape).toEqual([2, 2]);
    });

    test("vmap over QR decomposition", () => {
      // Batch of 3 matrices, each 3×2
      using flat = np.array(
        [1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 1, 2, 4, 0, 1, 3, 2, 5],
        { dtype: np.DType.Float32 },
      );
      using batch = np.reshape(flat, [3, 3, 2]);

      const qrFn = (A: np.Array) => {
        const [Q, R] = lax.linalg.qr(A);
        return [Q, R];
      };

      using f = jit(vmap(qrFn));
      const [Q, R] = f(batch) as [np.Array, np.Array];
      using _Q = Q;
      using _R = R;

      expect(Q.shape).toEqual([3, 3, 2]);
      expect(R.shape).toEqual([3, 2, 2]);

      // Verify Q @ R ≈ A for each batch element
      using reconstructed = np.matmul(Q, R);
      expect(reconstructed).toBeAllclose(batch, { atol: 1e-4 });
    });

    test("jit(qr) produces correct decomposition", () => {
      using A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);

      using f = jit((a: np.Array) => {
        const [Q, R] = lax.linalg.qr(a);
        return [Q, R];
      });

      const [Q, R] = f(A) as [np.Array, np.Array];
      using _Q = Q;
      using _R = R;

      expect(Q.shape).toEqual([3, 2]);
      expect(R.shape).toEqual([2, 2]);

      using reconstructed = np.matmul(Q, R);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-5 });
    });

    // TODO: foriLoop polyfill JVP doesn't match FD on WASM/WebGPU
    test.skipIf(device !== "cpu")(
      "JVP through QR matches finite differences",
      () => {
        using A = np.array([
          [1.0, 2.0],
          [3.0, 4.0],
        ]);
        using dA = np.array([
          [0.1, 0.05],
          [0.05, 0.1],
        ]);

        const f = (a: np.Array) => {
          const [Q, R] = lax.linalg.qr(a);
          Q.dispose();
          return R;
        };

        const [R, dR] = jvp(f, [A], [dA]);
        using _R = R;
        using _dR = dR;

        // Finite-difference verification: (f(A + eps*dA) - f(A)) / eps ≈ dR
        const eps = 1e-4;
        using dAe = dA.mul(eps);
        using Ape = A.add(dAe);
        using R2 = f(Ape);
        using R2subR = R2.sub(R);
        using dR_fd = R2subR.div(eps);
        // WebGPU polyfill accumulates more float32 rounding in Householder ops
        const jvpFdAtol = device === "webgpu" ? 0.05 : 2e-3;
        expect(dR).toBeAllclose(dR_fd, { rtol: 0.05, atol: jvpFdAtol });
      },
    );

    // TODO: foriLoop AD leaks on non-CPU backends
    test.skipIf(device !== "cpu")(
      "grad through QR matches finite differences",
      () => {
        using A = np.array([
          [4.0, 1.0],
          [2.0, 3.0],
        ]);

        // Loss: sum of squared R elements
        const f = (a: np.Array) => {
          const [Q, R] = lax.linalg.qr(a);
          Q.dispose();
          using _R = R;
          using sq = np.square(R);
          return sq.sum();
        };

        using dA = grad(f)(A);

        // Verify gradient by central finite differences
        const eps = 1e-4;
        const aData = A.js() as number[][];
        const expected: number[][] = [[], []];
        for (let i = 0; i < 2; i++) {
          for (let j = 0; j < 2; j++) {
            const ap = aData.map((row) => [...row]);
            const am = aData.map((row) => [...row]);
            ap[i][j] += eps;
            am[i][j] -= eps;
            using arrP = np.array(ap);
            using fpArr = f(arrP);
            const fp = fpArr.js() as number;
            using arrM = np.array(am);
            using fmArr = f(arrM);
            const fm = fmArr.js() as number;
            expected[i][j] = (fp - fm) / (2 * eps);
          }
        }
        const gradFdAtol = device === "webgpu" ? 0.1 : 1e-3;
        expect(dA).toBeAllclose(expected, { rtol: 0.05, atol: gradFdAtol });
        // foriLoop AD infrastructure has known leaks — drain before afterEach
      },
    );

    // TODO: foriLoop AD leaks on non-CPU backends
    test.skipIf(device !== "cpu")("grad through QR for tall matrix", () => {
      // Tall (3×2) matrix: grad through R.sum()
      using A = np.array([
        [3.0, 1.0],
        [1.0, 4.0],
        [2.0, 2.0],
      ]);

      const f = (a: np.Array) => {
        const [Q, R] = lax.linalg.qr(a);
        Q.dispose();
        using _R = R;
        return R.sum();
      };

      using dA = grad(f)(A);

      // Verify gradient by central finite differences
      const eps = 1e-4;
      const aData = A.js() as number[][];
      const expected: number[][] = [[], [], []];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
          const ap = aData.map((row) => [...row]);
          const am = aData.map((row) => [...row]);
          ap[i][j] += eps;
          am[i][j] -= eps;
          using arrP = np.array(ap);
          using fpArr = f(arrP);
          const fp = fpArr.js() as number;
          using arrM = np.array(am);
          using fmArr = f(arrM);
          const fm = fmArr.js() as number;
          expected[i][j] = (fp - fm) / (2 * eps);
        }
      }
      const tallGradAtol = device === "webgpu" ? 0.05 : 1e-3;
      expect(dA).toBeAllclose(expected, { rtol: 0.05, atol: tallGradAtol });
      // foriLoop AD infrastructure has known leaks — drain before afterEach
    });

    // TODO: foriLoop AD leaks on non-CPU backends
    test.skipIf(device !== "cpu")("grad through Q factor of QR", () => {
      // Use tall matrix (3×2) where sum(Q) is NOT constant
      // (for square matrices, sum(Q^2) = trace(Q^T Q) = n, so grad = 0)
      using A = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
        [0.5, 2.0],
      ]);

      // Loss depends on Q, not R
      const f = (a: np.Array) => {
        const [Q, R] = lax.linalg.qr(a);
        R.dispose();
        using _Q = Q;
        return Q.sum();
      };

      using dA = grad(f)(A);

      // Finite-difference check
      const eps = 1e-4;
      const aData = A.js() as number[][];
      const expected: number[][] = [[], [], []];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
          const ap = aData.map((row) => [...row]);
          const am = aData.map((row) => [...row]);
          ap[i][j] += eps;
          am[i][j] -= eps;
          using arrP = np.array(ap);
          using fpArr = f(arrP);
          const fp = fpArr.js() as number;
          using arrM = np.array(am);
          using fmArr = f(arrM);
          const fm = fmArr.js() as number;
          expected[i][j] = (fp - fm) / (2 * eps);
        }
      }
      expect(dA).toBeAllclose(expected, { rtol: 1e-2, atol: 2e-3 });
      // foriLoop AD infrastructure has known leaks — drain before afterEach
    });

    // TODO: foriLoop AD leaks on non-CPU backends
    test.skipIf(device !== "cpu")(
      "grad through QR is consistent across batch elements",
      () => {
        // Test that grad produces consistent results for different matrices,
        // verifying each individually against finite differences.
        const matrices = [
          [
            [4.0, 1.0],
            [2.0, 3.0],
          ],
          [
            [3.0, 1.0],
            [1.0, 4.0],
          ],
          [
            [2.0, 0.5],
            [0.5, 2.0],
          ],
        ];

        const f = (a: np.Array) => {
          const [Q, R] = lax.linalg.qr(a);
          Q.dispose();
          using _R = R;
          return R.sum();
        };

        for (const mat of matrices) {
          using A = np.array(mat);
          using dA = grad(f)(A);

          // Verify gradient by central finite differences
          const eps = 1e-4;
          const expected: number[][] = [[], []];
          for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
              const ap = mat.map((row) => [...row]);
              const am = mat.map((row) => [...row]);
              ap[i][j] += eps;
              am[i][j] -= eps;
              using arrP = np.array(ap);
              using fpArr = f(arrP);
              const fp = fpArr.js() as number;
              using arrM = np.array(am);
              using fmArr = f(arrM);
              const fm = fmArr.js() as number;
              expected[i][j] = (fp - fm) / (2 * eps);
            }
          }
          const batchGradAtol = device === "webgpu" ? 0.05 : 1e-3;
          expect(dA).toBeAllclose(expected, {
            rtol: 0.05,
            atol: batchGradAtol,
          });
        }
        // foriLoop AD infrastructure has known leaks — drain before afterEach
      },
    );

    test("JVP through both Q and R factors", () => {
      using A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      using dA = np.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [0.05, 0.05],
      ]);

      const [primals, tangents] = jvp(
        (a: np.Array) => {
          const [Q, R] = lax.linalg.qr(a);
          return [Q, R];
        },
        [A],
        [dA],
      );
      const [Q, R] = primals as [np.Array, np.Array];
      const [dQ, dR] = tangents as [np.Array, np.Array];
      using _Q = Q;
      using _R = R;
      using _dQ = dQ;
      using _dR = dR;

      expect(dQ.shape).toEqual([3, 2]);
      expect(dR.shape).toEqual([2, 2]);

      // Verify: d(Q @ R) = dQ @ R + Q @ dR ≈ dA
      using dQR = np.matmul(dQ, R);
      using QdR = np.matmul(Q, dR);
      using dA_reconstructed = np.add(dQR, QdR);
      expect(dA_reconstructed).toBeAllclose(dA, { atol: 1e-4 });
    });
  });
});
