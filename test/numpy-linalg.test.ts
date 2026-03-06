import {
  defaultDevice,
  Device,
  grad,
  init,
  jvp,
  numpy as np,
  random,
  valueAndGrad,
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

  suite("numpy.linalg.cholesky()", () => {
    test("symmetrizes input by default", () => {
      using x = np.array([
        [4.0, 2.01],
        [1.99, 5.0],
      ]);
      using L = np.linalg.cholesky(x);
      const LT = L.transpose();
      using reconstructed = np.matmul(L, LT);
      LT.dispose();
      const xT = x.transpose();
      const xPlusXT = x.add(xT);
      xT.dispose();
      using symmetrized = xPlusXT.mul(0.5);
      xPlusXT.dispose();
      expect(reconstructed).toBeAllclose(symmetrized);
    });
  });

  suite("numpy.linalg.det()", () => {
    test("computes determinant of simple matrix", () => {
      using a = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
      ]);
      using detA = np.linalg.det(a);
      expect(detA).toBeAllclose(10.0);
    });

    test("gradient of det is adjugate.mT", () => {
      const key0 = random.key(0);
      using a = random.uniform(key0, [15, 15]);
      key0.dispose();
      const g = valueAndGrad(np.linalg.det);
      const [detA, da] = g(a);
      using _detA = detA;
      using _da = da;
      const aInvDet = np.linalg.inv(a);
      using adjA = aInvDet.mul(detA);
      aInvDet.dispose();
      using adjAT = np.matrixTranspose(adjA);
      expect(da).toBeAllclose(adjAT, { rtol: 1e-3 });
    });
  });

  suite("numpy.linalg.inv()", () => {
    test("computes inverse of simple matrix", () => {
      using a = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
      ]);
      using aInv = np.linalg.inv(a);
      using identity = np.matmul(a, aInv);
      using expected = np.eye(2);
      // 2×2 uses analytical Cramer's rule; f32 rounding -> loosen atol.
      expect(identity).toBeAllclose(expected, { atol: 1e-6 });
    });

    test("computes inverse of batched matrices", () => {
      const key0b = random.key(0);
      using a = random.uniform(key0b, [2, 3, 4, 4]);
      key0b.dispose();
      using aInv = np.linalg.inv(a);
      using identity = np.matmul(a, aInv);
      const eye4 = np.eye(4);
      using expected = np.broadcastTo(eye4, [2, 3, 4, 4]);
      eye4.dispose();
      expect(identity).toBeAllclose(expected, {
        atol: 1e-4,
      });
    });

    test("gradient of sum(inv(A)) matches analytical formula", () => {
      using a = np.array([
        [2.0, 0.3],
        [0.1, 1.7],
      ]);

      const f = (x: np.Array) => {
        return np.linalg.inv(x).sum();
      };
      using da = grad(f)(a);

      using aInv = np.linalg.inv(a);
      using aInvT = np.matrixTranspose(aInv);
      using ones = np.ones([2, 2]);
      using left = np.matmul(aInvT, ones);
      using leftRight = np.matmul(left, aInvT);
      using expected = leftRight.mul(-1);

      expect(da).toBeAllclose(expected, { rtol: 1e-4, atol: 1e-4 });
    });

    test("1×1 inverse (reciprocal path)", () => {
      using a = np.array([[4.0]]);
      using aInv = np.linalg.inv(a);
      using expected = np.array([[0.25]]);
      expect(aInv).toBeAllclose(expected, { atol: 1e-7 });
    });

    test("3×3 inverse (cofactor/Cramer's rule path)", () => {
      using a = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [5.0, 6.0, 0.0],
      ]);
      using aInv = np.linalg.inv(a);
      using identity = np.matmul(a, aInv);
      using expected = np.eye(3);
      expect(identity).toBeAllclose(expected, { atol: 1e-5 });
    });

    test("gradient of sum(inv(A)) for 3×3", () => {
      using a = np.array([
        [2.0, 0.3, 0.1],
        [0.1, 1.7, 0.2],
        [0.4, 0.1, 1.5],
      ]);
      const f = (x: np.Array) => np.linalg.inv(x).sum();
      using da = grad(f)(a);
      using aInv = np.linalg.inv(a);
      using aInvT = np.matrixTranspose(aInv);
      using ones = np.ones([3, 3]);
      using left = np.matmul(aInvT, ones);
      using leftRight = np.matmul(left, aInvT);
      using expected = leftRight.mul(-1);
      expect(da).toBeAllclose(expected, { rtol: 1e-3, atol: 1e-3 });
    });

    test("gradient of sum(inv(A)) for 4×4", () => {
      using a = np.array([
        [3.0, 0.2, 0.1, 0.0],
        [0.1, 2.5, 0.3, 0.1],
        [0.2, 0.1, 2.0, 0.2],
        [0.0, 0.1, 0.1, 1.8],
      ]);
      const f = (x: np.Array) => np.linalg.inv(x).sum();
      using da = grad(f)(a);
      using aInv = np.linalg.inv(a);
      using aInvT = np.matrixTranspose(aInv);
      using ones = np.ones([4, 4]);
      using left = np.matmul(aInvT, ones);
      using leftRight = np.matmul(left, aInvT);
      using expected = leftRight.mul(-1);
      expect(da).toBeAllclose(expected, { rtol: 1e-3, atol: 1e-3 });
    });
  });

  suite("numpy.linalg.lstsq()", () => {
    test("solves overdetermined system (M > N)", () => {
      // 3x2 system: more equations than unknowns
      using a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      using b = np.array([[1.0], [2.0], [3.0]]);
      using x = np.linalg.lstsq(a, b);

      // Verify solution minimizes ||Ax - b||
      // The normal equations: A^T A x = A^T b
      const aT = a.transpose();
      using atA = np.matmul(aT, a);
      using atb = np.matmul(aT, b);
      aT.dispose();
      using lhs = np.matmul(atA, x);
      expect(lhs).toBeAllclose(atb, { rtol: 1e-4, atol: 1e-4 });
    });

    test("solves underdetermined system (M < N)", () => {
      // 2x3 system: fewer equations than unknowns
      using a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      using b = np.array([[1.0], [2.0]]);
      using x = np.linalg.lstsq(a, b);

      // Verify Ax = b (should be exact for underdetermined systems)
      using ax = np.matmul(a, x);
      expect(ax).toBeAllclose(b, { rtol: 1e-4, atol: 1e-4 });
    });

    test("solves square system exactly", () => {
      using a = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
      ]);
      using b = np.array([[5.0], [7.0]]);
      using x = np.linalg.lstsq(a, b);

      // Verify Ax = b
      using ax = np.matmul(a, x);
      expect(ax).toBeAllclose(b, { rtol: 1e-4, atol: 1e-4 });
    });

    test("handles multiple right-hand sides (M > N)", () => {
      using a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      using b = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      using x = np.linalg.lstsq(a, b);

      // x should have shape (2, 2)
      expect(x.shape).toEqual([2, 2]);

      // Verify normal equations for each column
      const aT = a.transpose();
      using atA = np.matmul(aT, a);
      using atb = np.matmul(aT, b);
      aT.dispose();
      using lhs = np.matmul(atA, x);
      expect(lhs).toBeAllclose(atb, { rtol: 1e-4, atol: 1e-4 });
    });

    test("handles multiple right-hand sides (M < N)", () => {
      using a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      using b = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      using x = np.linalg.lstsq(a, b);

      // x should have shape (3, 2)
      expect(x.shape).toEqual([3, 2]);

      // Verify Ax = b
      using ax = np.matmul(a, x);
      expect(ax).toBeAllclose(b, { rtol: 1e-4, atol: 1e-4 });
    });

    test("throws on non-2D coefficient matrix", () => {
      using a = np.array([1.0, 2.0, 3.0]);
      using b = np.array([1.0, 2.0, 3.0]);
      expect(() => np.linalg.lstsq(a, b).js()).toThrow();
    });

    test("throws on mismatched dimensions", () => {
      using a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      using b = np.array([1.0, 2.0, 3.0]); // Wrong size
      expect(() => np.linalg.lstsq(a, b).js()).toThrow();
    });

    test("works with jvp on b (underdetermined)", () => {
      using a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      using b = np.array([[1.0], [2.0]]);
      using db = np.array([[0.1], [0.1]]);

      const solve = (b: np.Array) => np.linalg.lstsq(a, b);
      const [x, dx] = jvp(solve, [b], [db]);
      using _x = x;
      using _dx = dx;

      // Verify dx by finite differences
      const eps = 1e-4;
      const dbScaled = db.mul(eps);
      const bPerturbed = b.add(dbScaled);
      dbScaled.dispose();
      const x2 = np.linalg.lstsq(a, bPerturbed);
      bPerturbed.dispose();
      const diff = x2.sub(x);
      x2.dispose();
      using dx_fd = diff.div(eps);
      diff.dispose();
      expect(dx).toBeAllclose(dx_fd, { rtol: 1e-2, atol: 1e-3 });
    });

    test("works with grad on b (underdetermined)", () => {
      using a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      using b = np.array([[1.0], [2.0]]);

      const f = (b: np.Array) => np.square(np.linalg.lstsq(a, b)).sum();
      using db = grad(f)(b);

      // Verify gradient by finite differences
      const eps = 1e-4;
      const bData = b.js() as number[][];
      const expected: number[][] = [];
      for (let i = 0; i < 2; i++) {
        const bp = bData.map((row) => [...row]);
        const bm = bData.map((row) => [...row]);
        bp[i][0] += eps;
        bm[i][0] -= eps;
        const bpArr = np.array(bp);
        const lstsqP = np.linalg.lstsq(a, bpArr);
        bpArr.dispose();
        const sqP = np.square(lstsqP);
        lstsqP.dispose();
        const fpArr = sqP.sum();
        sqP.dispose();
        const fp = fpArr.js() as number;
        fpArr.dispose();
        const bmArr = np.array(bm);
        const lstsqM = np.linalg.lstsq(a, bmArr);
        bmArr.dispose();
        const sqM = np.square(lstsqM);
        lstsqM.dispose();
        const fmArr = sqM.sum();
        sqM.dispose();
        const fm = fmArr.js() as number;
        fmArr.dispose();
        expected.push([(fp - fm) / (2 * eps)]);
      }
      expect(db).toBeAllclose(expected, { rtol: 1e-2, atol: 1e-3 });
    });
  });

  suite("numpy.linalg.slogdet()", () => {
    test("computes slogdet of simple matrix", () => {
      using a = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
      ]);
      const [sign, logdet] = np.linalg.slogdet(a);
      using _sign = sign;
      using _logdet = logdet;
      expect(sign).toBeAllclose(1);
      expect(logdet).toBeAllclose(Math.log(10));
    });
  });

  suite("numpy.linalg.solve()", () => {
    test("solves simple Ax = b", () => {
      using a = np.array([
        [3.0, 2.0],
        [1.0, 2.0],
      ]);
      using b = np.array([5.0, 4.0]);
      using x = np.linalg.solve(a, b);
      expect(x).toBeAllclose([0.5, 1.75]);
    });

    test("solves random batched AX = B", () => {
      using key = random.key(0);
      using splits = random.split(key, 2);
      const [k1, k2] = splits;
      using _k1 = k1;
      using _k2 = k2;
      using a = random.uniform(k1, [10, 15, 15]);
      using xTrue = random.uniform(k2, [10, 15, 5]);
      using b = np.matmul(a, xTrue); // B = A @ X_true
      expect(b.shape).toEqual(xTrue.shape);

      using xPred = np.linalg.solve(a, b);
      expect(xPred.shape).toEqual(xTrue.shape);
      expect(xPred).toBeAllclose(xTrue, { rtol: 1e-2, atol: 1e-4 });
    });
  });

  suite("numpy.linalg.qr()", () => {
    test("computes thin QR for 3x2 matrix", () => {
      using A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      const [Q, R] = np.linalg.qr(A);
      using _Q = Q;
      using _R = R;
      expect(Q.shape).toEqual([3, 2]);
      expect(R.shape).toEqual([2, 2]);

      using reconstructed = np.matmul(Q, R);
      expect(reconstructed).toBeAllclose(A, { atol: 1e-5 });
    });

    // foriLoop AD leaks on CPU; cleaned by JIT caches on WASM/WebGPU.
    test.skipIf(device === "cpu")("gradient through QR", () => {
      const f = (A: np.Array) => {
        const [Q, R] = np.linalg.qr(A);
        Q.dispose();
        return R.sum();
      };
      using A = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
      ]);
      using dA = grad(f)(A);
      // Gradient should be finite
      const vals = (dA.js() as number[][]).flat();
      for (const v of vals) {
        expect(isFinite(v)).toBe(true);
      }
    });
  });
});
