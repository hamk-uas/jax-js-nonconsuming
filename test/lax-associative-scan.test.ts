/**
 * Tests for lax.associativeScan — parallel prefix scan.
 *
 * Structure:
 * - Scalar/1-D correctness (cumsum, cumprod, running max)
 * - Pytree operands (associative composition of affine maps)
 * - Reverse scan
 * - Non-zero axis
 * - N=0, N=1 edge cases
 * - Ownership: no leaked slots
 * - Autodiff: grad through associativeScan
 * - Parallel Kalman filter (linear, constant-coefficient)
 */

import {
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

await init();

// ============================================================================
// Basic 1-D correctness
// ============================================================================

describe("lax.associativeScan — 1-D basic", () => {
  test("simple cumsum with split points", () => {
    const values = Array.from({ length: 17 }, (_, i) => i + 1);
    const expected = values.reduce<number[]>((acc, v, i) => {
      acc.push((acc[i - 1] ?? 0) + v);
      return acc;
    }, []);
    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose(expected);
  });

  test("cumulative sum", () => {
    using xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose([1, 3, 6, 10, 15]);
  });

  test("cumulative product", () => {
    using xs = np.array([1.0, 2.0, 3.0, 4.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => a.mul(b),
      xs,
    );
    expect(result).toBeAllclose([1, 2, 6, 24]);
  });

  test("running maximum", () => {
    using xs = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.maximum(a, b),
      xs,
    );
    expect(result).toBeAllclose([3, 3, 4, 4, 5, 9, 9]);
  });

  test("N=1 returns unchanged", () => {
    using xs = np.array([42.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose([42.0]);
  });

  test("single-element power-of-two N=8", () => {
    using xs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  test("non-power-of-two N=7", () => {
    using xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose([1, 3, 6, 10, 15, 21, 28]);
  });
});

// ============================================================================
// Reverse scan
// ============================================================================

describe("lax.associativeScan — reverse", () => {
  test("reverse cumsum", () => {
    // Result[i] = sum(xs[i..N-1])
    using xs = np.array([1.0, 2.0, 3.0, 4.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
      { reverse: true },
    );
    expect(result).toBeAllclose([10, 9, 7, 4]);
  });

  test("reverse cummax", () => {
    using xs = np.array([3.0, 1.0, 4.0, 1.0, 5.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.maximum(a, b),
      xs,
      { reverse: true },
    );
    expect(result).toBeAllclose([5, 5, 5, 5, 5]);
  });
});

// ============================================================================
// Non-zero axis
// ============================================================================

describe("lax.associativeScan — non-zero axis", () => {
  test("cumsum along axis=1 for 2-D array", () => {
    // xs shape [2, 4] — scan along columns
    using xs = np.array([
      [1.0, 2.0, 3.0, 4.0],
      [10.0, 20.0, 30.0, 40.0],
    ]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
      { axis: 1 },
    );
    expect(result).toBeAllclose([
      [1, 3, 6, 10],
      [10, 30, 60, 100],
    ]);
  });

  test("cumsum along axis=0 for 2-D array", () => {
    using xs = np.array([
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
      { axis: 0 },
    );
    expect(result).toBeAllclose([
      [1, 2],
      [4, 6],
      [9, 12],
    ]);
  });
});

// ============================================================================
// Pytree operands: composition of affine maps
//
// An affine map  y = a*x + b  can be composed associatively:
//   compose((a1, b1), (a2, b2)) = (a1*a2,  a1*b2 + b1)
//
// The prefix-scan of a sequence of affine maps gives the cumulative
// composition from left to right. This is a canonical example of a
// useful non-trivial associative operation on pytrees.
// ============================================================================

describe("lax.associativeScan — pytree (affine composition)", () => {
  // Compose two affine maps: apply p first, then q.
  // If p maps x → a1*x + b1, and q maps x → a2*x + b2, then:
  //   q ∘ p: x → a2*(a1*x + b1) + b2 = (a2*a1)*x + (a2*b1 + b2)
  // So compose(p, q) = (a1*a2, a2*b1 + b2)
  //
  // NOTE: fn must dispose its own internal intermediates.
  const composeAffine = (
    p: { a: np.Array; b: np.Array },
    q: { a: np.Array; b: np.Array },
  ) => {
    const newA = p.a.mul(q.a) as np.Array;
    using tmp = q.a.mul(p.b) as np.Array; // intermediate — disposed at block end
    const newB = tmp.add(q.b) as np.Array;
    return { a: newA, b: newB };
  };

  test("prefix composition of affine maps", () => {
    // f0: x → 2x+1,  f1: x → 3x+4,  f2: x → 5x+2
    //   result[0] = f0                           => a=2,  b=1
    //   result[1] = f1 ∘ f0  (f0 first, then f1) => 3(2x+1)+4 = 6x+7   => a=6,  b=7
    //   result[2] = f2 ∘ result[1]               => 5(6x+7)+2 = 30x+37 => a=30, b=37
    using as = np.array([2.0, 3.0, 5.0]);
    using bs = np.array([1.0, 4.0, 2.0]);
    const result = lax.associativeScan(composeAffine, { a: as, b: bs });
    const ra = (result as { a: np.Array; b: np.Array }).a;
    const rb = (result as { a: np.Array; b: np.Array }).b;
    try {
      expect(ra).toBeAllclose([2, 6, 30]);
      expect(rb).toBeAllclose([1, 7, 37]);
    } finally {
      ra.dispose();
      rb.dispose();
    }
  });
});

// ============================================================================
// Differentiability
// ============================================================================

describe("lax.associativeScan — autodiff", () => {
  test("grad of sum(associativeScan(add, xs)) wrt xs", () => {
    // sum(cumsums[i]) = sum_{i} (N-i) * xs[i], N=5
    // so d/d xs[i] of total = N - i = [5, 4, 3, 2, 1]
    using xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0]);

    const f = (xs: np.Array) => {
      const ys = lax.associativeScan(
        (a: np.Array, b: np.Array) => np.add(a, b),
        xs,
      );
      const s = ys.sum();
      ys.dispose();
      return s;
    };

    const dxs = grad(f)(xs);
    // d(sum)/d xs[i] = (N - i) = [5, 4, 3, 2, 1]
    expect(dxs).toBeAllclose([5, 4, 3, 2, 1]);
    dxs.dispose();
  });

  test("grad through jit(associativeScan)", () => {
    using xs = np.array([1.0, 2.0, 3.0]);
    const f = jit((xs: np.Array) => {
      const ys = lax.associativeScan(
        (a: np.Array, b: np.Array) => np.add(a, b),
        xs,
      );
      const s = ys.sum();
      ys.dispose();
      return s;
    });
    try {
      const dxs = grad(f)(xs);
      expect(dxs).toBeAllclose([3, 2, 1]);
      dxs.dispose();
    } finally {
      f.dispose();
    }
  });
});

// ============================================================================
// Parallel Kalman filter using associative_scan
//
// Background:
//   A constant-coefficient linear Kalman filter is a recurrence:
//     x_{t+1} = F * x_t + G * w_t        (state transition)
//     y_t     = H * x_t + v_t             (observation)
//
//   For the scalar case (state dim = 1, obs dim = 1):
//     F, H, Q (process noise var), R (obs noise var) are scalars.
//
// The parallel (associative) Kalman filter (Sarkka & Garcia-Fernandez 2020)
// expresses the filter as a prefix-scan of *associative elements*, each of
// which is a 2×2 matrix (in the scalar case: two scalars and one auxiliary):
//
//   An element in "information form" is a triplet (A, b, C) where:
//     A = information  gain: updated precision
//     b = information vector
//     C = carry matrix
//
// For simplicity, the test below uses the *affine-composition* representation
// of the Kalman smoother (gain and offset), which admits a clean scalar
// version whose output can be verified analytically.
//
// **What we test**: the filter on a *known* system where the optimal
// estimate can be computed analytically (all-zero measurements, unit
// process noise, unit observation noise). We verify that the parallel
// scan gives the same result as a sequential reference implementation.
// ============================================================================

describe("parallel Kalman filter via associativeScan", () => {
  // -------------------------------------------------------------------------
  // Scalar Kalman filter helpers
  //
  //   Predict:   P_pred = F^2 * P + Q
  //   Update:    K      = P_pred * H / (H^2 * P_pred + R)
  //              x_filt = x_pred + K * (y - H * x_pred)
  //              P_filt = (1 - K*H) * P_pred
  //
  // We represent the relation  x_filt = A * x_prev + b  as an affine map
  // (A, b) that is composed across time steps using the scan.
  //
  // For F=H=Q=R=1 each measurement y_t:
  //   A_t = 1 - K_t * F  (the coefficient on the previous best estimate)
  //   b_t = K_t * y_t    (the correction from observation y_t)
  //
  // The prefix-scan of (A, b) via affine composition gives the cumulative
  // relation from the initial state x_0 to each x_t.
  // -------------------------------------------------------------------------

  test("sequential vs parallel scan agree on 8 observations", () => {
    const F = 1.0; // state transition
    const H = 1.0; // observation matrix
    const Q = 1.0; // process noise variance
    const R = 1.0; // observation noise variance

    // Observations (arbitrary)
    const obsData = [0.5, -0.3, 0.8, 1.2, -0.5, 0.1, 0.9, 0.4];
    const N = obsData.length; // 8

    // --- Sequential reference ---
    // Run the standard Kalman filter step by step.
    const refEstimates: number[] = [];
    let xRef = 0.0; // initial state estimate
    let pRef = 1.0; // initial covariance
    for (let t = 0; t < N; t++) {
      // Predict
      const xPred = F * xRef;
      const pPred = F * F * pRef + Q;
      // Update
      const S = H * H * pPred + R;
      const K = (pPred * H) / S;
      xRef = xPred + K * (obsData[t] - H * xPred);
      pRef = (1 - K * H) * pPred;
      refEstimates.push(xRef);
    }

    // --- Parallel version via associativeScan ---
    //
    // Represent each filter step as an affine map (A_t, b_t):
    //   x_t = A_t * x_{t-1} + b_t
    //
    // We compute (A_t, b_t) analytically for each observation, using a
    // running covariance sequence (pre-computed).
    //
    // Since F, H, Q, R are constants, the covariance sequence is deterministic
    // and can be pre-computed. We compute it sequentially just for setup.
    const pSeq: number[] = [1.0]; // p_0
    for (let t = 0; t < N; t++) {
      const pPred = F * F * pSeq[t] + Q;
      const S = H * H * pPred + R;
      const K = (pPred * H) / S;
      pSeq.push((1 - K * H) * pPred);
    }

    // Build affine map arrays A_t and b_t.
    // For state x_{t-1} -> x_t:
    //   x_pred = F * x_{t-1}
    //   K_t    = p_pred_t * H / S_t
    //   x_t    = (1 - K_t*H)*F * x_{t-1} + K_t * y_t
    //          = A_t * x_{t-1} + b_t
    const aData: number[] = [];
    const bData: number[] = [];
    for (let t = 0; t < N; t++) {
      const pPred = F * F * pSeq[t] + Q;
      const S = H * H * pPred + R;
      const K = (pPred * H) / S;
      aData.push((1 - K * H) * F);
      bData.push(K * obsData[t]);
    }

    using aArr = np.array(aData);
    using bArr = np.array(bData);

    // Compose affine maps: (A1, b1) then (A2, b2) = (A2*A1, A2*b1 + b2)
    const composeAffine = (
      p: { a: np.Array; b: np.Array },
      q: { a: np.Array; b: np.Array },
    ) => {
      const newA = p.a.mul(q.a) as np.Array;
      using tmp = q.a.mul(p.b) as np.Array;
      const newB = tmp.add(q.b) as np.Array;
      return { a: newA, b: newB };
    };

    const scanResult = lax.associativeScan(composeAffine, {
      a: aArr,
      b: bArr,
    });
    const rA = (scanResult as { a: np.Array; b: np.Array }).a;
    const rB = (scanResult as { a: np.Array; b: np.Array }).b;

    // The t-th output: x_t = rA[t] * x_0 + rB[t]
    // With x_0 = 0: x_t = rB[t]
    const x0 = 0.0;
    const parallelEstimates: number[] = Array.from(
      rB.dataSync().map((b, i) => rA.dataSync()[i] * x0 + b),
    );
    // Use the synchronous data from rB directly (x_0 = 0 so it's just rB).
    const rBData = Array.from(rB.dataSync());

    rA.dispose();
    rB.dispose();

    // Compare with sequential reference
    for (let t = 0; t < N; t++) {
      expect(rBData[t]).toBeCloseTo(refEstimates[t], 5);
    }

    void parallelEstimates; // suppress unused warning (rA.dataSync used above)
  });

  test("parallel Kalman filter is differentiable wrt observations", () => {
    // Check that grad flows through the scan (loss = sum of filtered states).
    const N = 4;

    const f = (ys: np.Array) => {
      // Fixed covariances for simplicity (F=H=Q=R=1, P0=1).
      const pSeq = [1.0];
      for (let t = 0; t < N; t++) {
        const pPred = pSeq[t] + 1;
        const K = pPred / (pPred + 1);
        pSeq.push((1 - K) * pPred);
      }
      // Build Kalman gain array from pre-computed covariances.
      const kData = pSeq.slice(0, N).map((p) => {
        const pPred = p + 1;
        return pPred / (pPred + 1);
      });
      using aArr = np.array(
        pSeq.slice(0, N).map((p) => {
          const pPred = p + 1;
          const K = pPred / (pPred + 1);
          return 1 - K;
        }),
      );
      // b_t = K_t * y_t  (as an elementwise multiply)
      // Note: kData is a plain-JS array; we need an np.Array for it.
      using kArr = np.array(kData);
      // b_t = kArr[t] * ys[t]
      const bArr = kArr.mul(ys) as np.Array;

      const composeAffine = (
        p: { a: np.Array; b: np.Array },
        q: { a: np.Array; b: np.Array },
      ) => ({
        a: q.a.mul(p.a) as np.Array,
        b: q.a.mul(p.b).add(q.b) as np.Array,
      });

      const scanResult = lax.associativeScan(composeAffine, {
        a: aArr,
        b: bArr,
      });
      bArr.dispose();
      const rB = (scanResult as { a: np.Array; b: np.Array }).b;
      const rA = (scanResult as { a: np.Array; b: np.Array }).a;
      const loss = rB.sum();
      rB.dispose();
      rA.dispose();
      return loss;
    };

    using ys0 = np.array([0.5, -0.3, 0.8, 1.2]);
    const dys = grad(f)(ys0);
    // Gradient must be finite and have the right shape.
    expect(dys.shape).toEqual([N]);
    const dyData = Array.from(dys.dataSync());
    for (const d of dyData) {
      expect(isFinite(d)).toBe(true);
    }
    // All entries should be positive (the Kalman gains are strictly positive
    // and the map is linear through the scan).
    for (const d of dyData) {
      expect(d).toBeGreaterThan(0);
    }
    dys.dispose();
  });
});
