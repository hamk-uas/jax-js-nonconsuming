import { describe, expect, test } from "vitest";

// Import everything from src/ to avoid dual-module-graph issues
// (dist/ and src/ have separate class identities — instanceof Tracer fails)
import {
  _setVerifyEffects,
  type Array,
  DType,
  grad,
  jit,
  makeJaxpr,
  numpy as np,
  verifyJaxprEffects,
} from "../src";
import { scatterAdd } from "../src/frontend/core";

describe("scatter_add", () => {
  // ---------------------------------------------------------------------------
  // Basic correctness
  // ---------------------------------------------------------------------------

  test("basic 1-D unique indices", () => {
    using target = np.zeros([5]);
    using indices = np.array([1, 3], { dtype: DType.Int32 });
    using updates = np.array([10.0, 20.0]);
    using result = scatterAdd(target, indices, updates, 0) as Array;
    expect(result.js()).toEqual([0, 10, 0, 20, 0]);
  });

  test("duplicate indices are summed", () => {
    using target = np.zeros([4]);
    using indices = np.array([1, 1, 2], { dtype: DType.Int32 });
    using updates = np.array([10.0, 20.0, 5.0]);
    using result = scatterAdd(target, indices, updates, 0) as Array;
    expect(result.js()).toEqual([0, 30, 5, 0]);
  });

  test("adds to existing target values", () => {
    using target = np.array([100.0, 200.0, 300.0]);
    using indices = np.array([0, 2], { dtype: DType.Int32 });
    using updates = np.array([1.0, 2.0]);
    using result = scatterAdd(target, indices, updates, 0) as Array;
    expect(result.js()).toEqual([101, 200, 302]);
  });

  test("preserves target (non-consuming)", () => {
    using target = np.array([1.0, 2.0, 3.0]);
    using indices = np.array([0], { dtype: DType.Int32 });
    using updates = np.array([99.0]);
    using result = scatterAdd(target, indices, updates, 0) as Array;
    // target should be unchanged
    expect(target.js()).toEqual([1, 2, 3]);
    // but result should have the scatter applied
    expect(result.js()).toEqual([100, 2, 3]);
  });

  // ---------------------------------------------------------------------------
  // Multi-dimensional
  // ---------------------------------------------------------------------------

  test("2-D axis=0", () => {
    // target shape [3, 2], scatter along axis 0
    using target = np.zeros([3, 2]);
    using indices = np.array([0, 2], { dtype: DType.Int32 });
    // updates shape [2, 2] (updatesLen=2 along axis 0, inner=2)
    using updates = np.array([
      [1, 2],
      [3, 4],
    ]);
    using result = scatterAdd(target, indices, updates, 0) as Array;
    // Row 0 gets [1,2], row 1 unchanged, row 2 gets [3,4]
    expect(result.js()).toEqual([
      [1, 2],
      [0, 0],
      [3, 4],
    ]);
  });

  test("2-D axis=1", () => {
    // target shape [2, 4], scatter along axis 1
    using target = np.zeros([2, 4]);
    using indices = np.array([1, 3], { dtype: DType.Int32 });
    // updates shape [2, 2] (outer=2 from axis 0, updatesLen=2 along axis 1)
    using updates = np.array([
      [10, 20],
      [30, 40],
    ]);
    using result = scatterAdd(target, indices, updates, 1) as Array;
    // Row 0: [0, 10, 0, 20], Row 1: [0, 30, 0, 40]
    expect(result.js()).toEqual([
      [0, 10, 0, 20],
      [0, 30, 0, 40],
    ]);
  });

  // ---------------------------------------------------------------------------
  // Autodiff
  // ---------------------------------------------------------------------------

  test("grad(take) with unique indices", () => {
    // grad of sum(take(x, [1, 2])) wrt x should give [0, 1, 1, 0]
    const f = (x: Array) => {
      using taken = np.take(x, np.array([1, 2], { dtype: DType.Int32 }), 0);
      return np.sum(taken);
    };
    using x = np.array([1.0, 2.0, 3.0, 4.0]);
    using dx = grad(f)(x) as Array;
    expect(dx.js()).toEqual([0, 1, 1, 0]);
  });

  test("grad(take) with duplicate indices", () => {
    // grad of sum(take(x, [0, 1, 0])) wrt x should give [2, 1, 0]
    // Because x[0] is used twice and x[1] once
    const f = (x: Array) => {
      using taken = np.take(x, np.array([0, 1, 0], { dtype: DType.Int32 }), 0);
      return np.sum(taken);
    };
    using x = np.array([10.0, 20.0, 30.0]);
    using dx = grad(f)(x) as Array;
    expect(dx.js()).toEqual([2, 1, 0]);
  });

  test("grad(scatter_add) wrt target is identity", () => {
    // d/d(target) of sum(scatter_add(target, idx, updates))
    // = sum of cotangent = 1 for each position in target
    const f = (target: Array) => {
      using indices = np.array([1], { dtype: DType.Int32 });
      using updates = np.array([5.0]);
      using result = scatterAdd(target, indices, updates, 0) as Array;
      return np.sum(result);
    };
    using target = np.array([1.0, 2.0, 3.0]);
    using dt = grad(f)(target) as Array;
    // Identity: cotangent flows through target unchanged
    expect(dt.js()).toEqual([1, 1, 1]);
  });

  test("grad(scatter_add) wrt updates is gather", () => {
    // d/d(updates) of sum(scatter_add(zeros, idx, updates))
    // Since sum reduces everything, cotangent for updates[j] = 1 for all j
    // (gathered from the cotangent at the scattered positions)
    const f = (updates: Array) => {
      using target = np.zeros([4]);
      using indices = np.array([0, 2], { dtype: DType.Int32 });
      using result = scatterAdd(target, indices, updates, 0) as Array;
      return np.sum(result);
    };
    using updates = np.array([10.0, 20.0]);
    using du = grad(f)(updates) as Array;
    expect(du.js()).toEqual([1, 1]);
  });

  // ---------------------------------------------------------------------------
  // JIT compilation
  // ---------------------------------------------------------------------------

  test("jit(scatter_add)", () => {
    using f = jit((target: Array, updates: Array) => {
      using indices = np.array([0, 2, 0], { dtype: DType.Int32 });
      return scatterAdd(target, indices, updates, 0) as Array;
    });
    using target = np.zeros([4]);
    using updates = np.array([1.0, 2.0, 3.0]);
    using result = f(target, updates) as Array;
    // index 0 gets 1+3=4, index 2 gets 2
    expect(result.js()).toEqual([4, 0, 2, 0]);
  });

  test("jit(grad(take)) with duplicates", () => {
    using indices = np.array([0, 1, 0], { dtype: DType.Int32 });
    using f = jit(
      grad((x: Array) => {
        using taken = np.take(x, indices, 0);
        return np.sum(taken);
      }),
    );
    using x = np.array([10.0, 20.0, 30.0]);
    using dx = f(x) as Array;
    expect(dx.js()).toEqual([2, 1, 0]);
  });

  // ---------------------------------------------------------------------------
  // Effect checker
  // ---------------------------------------------------------------------------

  test("passes effect checker with Mutate on target", () => {
    _setVerifyEffects(true);
    try {
      using s0 = np.zeros([4]);
      using s1 = np.array([0, 1], { dtype: DType.Int32 });
      using s2 = np.array([1.0, 2.0]);
      const { jaxpr: closedJaxpr } = makeJaxpr(
        (target: Array, indices: Array, updates: Array) => {
          return scatterAdd(target, indices, updates, 0) as Array;
        },
      )(s0, s1, s2);
      const report = verifyJaxprEffects(closedJaxpr.jaxpr);
      expect(report.ok).toBe(true);
      closedJaxpr.dispose();
    } finally {
      _setVerifyEffects(false);
    }
  });
});
