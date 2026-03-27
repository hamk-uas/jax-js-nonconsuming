/**
 * Tests for N-dimensional DynamicUpdateSlice (DynamicUpdateSliceGeneral).
 *
 * Covers:
 *   - Eager: ND update at various offsets and ranks
 *   - JIT: 0-active, 1-active, and multi-active axis paths
 *   - AD: JVP (forward-mode), grad (reverse-mode via transpose)
 *   - vmap: batched ND updates
 *   - Regression: single-axis path via ND API matches original
 */
import {
  captureJitReport,
  clearCaches,
  DType,
  grad,
  jit,
  jvp,
  lax,
  makeJaxpr,
  numpy as np,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

// ============================================================
// Eager ND DynamicUpdateSlice
// ============================================================
describe("ND DynamicUpdateSlice (eager)", () => {
  test("2D block update at [1, 2]", () => {
    using dst = np.zeros([4, 6]);
    using src = np.ones([2, 3]);
    using result = lax.dynamicUpdateSlice(dst, src, [1, 2]);
    const js = result.js() as number[][];
    // Row 0: all zeros
    expect(js[0]).toEqual([0, 0, 0, 0, 0, 0]);
    // Row 1: zeros then 1s at cols 2..4
    expect(js[1]).toEqual([0, 0, 1, 1, 1, 0]);
    // Row 2: same
    expect(js[2]).toEqual([0, 0, 1, 1, 1, 0]);
    // Row 3: all zeros
    expect(js[3]).toEqual([0, 0, 0, 0, 0, 0]);
  });

  test("2D block update at origin [0, 0]", () => {
    using dst = np.arange(12).reshape([3, 4]);
    using src = np.array([
      [99, 98],
      [97, 96],
    ]);
    using result = lax.dynamicUpdateSlice(dst, src, [0, 0]);
    expect(result.js()).toEqual([
      [99, 98, 2, 3],
      [97, 96, 6, 7],
      [8, 9, 10, 11],
    ]);
  });

  test("2D full overwrite (src == dst shape)", () => {
    using dst = np.ones([2, 3]);
    using src = np.zeros([2, 3]);
    using result = lax.dynamicUpdateSlice(dst, src, [0, 0]);
    expect(result.js()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  test("3D block update", () => {
    using dst = np.zeros([3, 4, 5]);
    using src = np.ones([1, 2, 3]);
    using result = lax.dynamicUpdateSlice(dst, src, [1, 1, 2]);
    const js = result.js() as number[][][];
    // Only slice [1:2, 1:3, 2:5] should be 1s
    expect(js[0]).toEqual([
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
    ]);
    expect(js[1][0]).toEqual([0, 0, 0, 0, 0]);
    expect(js[1][1]).toEqual([0, 0, 1, 1, 1]);
    expect(js[1][2]).toEqual([0, 0, 1, 1, 1]);
    expect(js[1][3]).toEqual([0, 0, 0, 0, 0]);
    expect(js[2]).toEqual([
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
    ]);
  });

  test("1D update via ND API matches single-axis API", () => {
    using dst = np.array([1, 2, 3, 4, 5]);
    using src = np.array([99, 98]);
    using ndResult = lax.dynamicUpdateSlice(dst, src, [2]);
    using singleResult = lax.dynamicUpdateSlice(dst, src, 2);
    expect(ndResult.js()).toEqual(singleResult.js());
  });

  test("single-element update", () => {
    using dst = np.zeros([3, 3]);
    using src = np.array([[7]]);
    using result = lax.dynamicUpdateSlice(dst, src, [1, 1]);
    expect(result.js()).toEqual([
      [0, 0, 0],
      [0, 7, 0],
      [0, 0, 0],
    ]);
  });

  test("rejects out-of-bounds start indices", () => {
    using dst = np.zeros([3, 3]);
    using src = np.ones([2, 2]);
    // start [2, 2]: 2 + 2 = 4 > 3 on both axes
    expect(() => lax.dynamicUpdateSlice(dst, src, [2, 2])).toThrow(
      /out of bounds/,
    );
    // start [0, 2]: axis 1 overflows
    expect(() => lax.dynamicUpdateSlice(dst, src, [0, 2])).toThrow(
      /out of bounds/,
    );
    // start [2, 0]: axis 0 overflows
    expect(() => lax.dynamicUpdateSlice(dst, src, [2, 0])).toThrow(
      /out of bounds/,
    );
  });

  test("does not mutate caller's startIndices array", () => {
    using dst = np.zeros([3, 3]);
    using src = np.ones([2, 2]);
    const indices = [1.9, 0.7];
    using _result = lax.dynamicUpdateSlice(dst, src, indices);
    // Caller's array must be unchanged
    expect(indices).toEqual([1.9, 0.7]);
  });
});

// ============================================================
// JIT ND DynamicUpdateSlice
// ============================================================
describe("ND DynamicUpdateSlice (JIT)", () => {
  test("2D block update in jit", () => {
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [1, 1]),
    );
    using dst = np.zeros([4, 4]);
    using src = np.ones([2, 2]);
    using result = jitF(dst, src);
    expect(result.js()).toEqual([
      [0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0],
    ]);
  });

  test("JIT preserves single-axis fast path via ND API", () => {
    // Only axis 0 is active: startIndices = [2, 0] with matching columns
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [2, 0]),
    );
    using arange12 = np.arange(12);
    using reshaped = arange12.reshape([4, 3]);
    using dst = reshaped.astype(DType.Float32);
    using src = np.array([
      [99, 98, 97],
      [96, 95, 94],
    ]);
    using result = jitF(dst, src);
    expect(result.js()).toEqual([
      [0, 1, 2],
      [3, 4, 5],
      [99, 98, 97],
      [96, 95, 94],
    ]);
  });

  test("JIT 0-active-axes (full copy)", () => {
    // All start indices are 0 and src shape == dst shape
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [0, 0]),
    );
    using dst = np.ones([3, 3]);
    using src = np.zeros([3, 3]);
    using result = jitF(dst, src);
    expect(result.js()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  test("JIT 3D block update", () => {
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [0, 1, 1]),
    );
    using dst = np.zeros([2, 3, 3]);
    using src = np.ones([2, 2, 2]);
    using result = jitF(dst, src);
    const js = result.js() as number[][][];
    expect(js[0]).toEqual([
      [0, 0, 0],
      [0, 1, 1],
      [0, 1, 1],
    ]);
    expect(js[1]).toEqual([
      [0, 0, 0],
      [0, 1, 1],
      [0, 1, 1],
    ]);
  });

  test("JIT single active axis (only axis 1 offset)", () => {
    // startIndices = [0, 2], srcShape = [4, 2], dstShape = [4, 6]
    // Only axis 1 is active
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [0, 2]),
    );
    using dst = np.zeros([4, 6]);
    using src = np.ones([4, 2]);
    using result = jitF(dst, src);
    const js = result.js() as number[][];
    expect(js[0]).toEqual([0, 0, 1, 1, 0, 0]);
    expect(js[1]).toEqual([0, 0, 1, 1, 0, 0]);
    expect(js[2]).toEqual([0, 0, 1, 1, 0, 0]);
    expect(js[3]).toEqual([0, 0, 1, 1, 0, 0]);
  });

  test("JIT multi-axis ND stride path (srcShape != dstShape on outer axes)", () => {
    // src [2, 2, 2] into dst [4, 5, 6] at [1, 2, 3]: all three axes differ,
    // exercises the ndOuterExtents/ndDstStrides path.
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [1, 2, 3]),
    );
    using dst = np.zeros([4, 5, 6]);
    using src = np.ones([2, 2, 2]);
    using result = jitF(dst, src);
    const js = result.js() as number[][][];
    // Only dst[1:3, 2:4, 3:5] should be 1
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 5; j++) {
        for (let k = 0; k < 6; k++) {
          const expected =
            i >= 1 && i < 3 && j >= 2 && j < 4 && k >= 3 && k < 5 ? 1 : 0;
          expect(js[i][j][k]).toBe(expected);
        }
      }
    }
  });

  test("JIT 4D multi-axis ND stride path", () => {
    // src [1, 2, 2, 3] into dst [2, 4, 4, 6] at [1, 1, 2, 0]
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [1, 1, 2, 0]),
    );
    using dst = np.zeros([2, 4, 4, 6]);
    using src = np.ones([1, 2, 2, 3]);
    using result = jitF(dst, src);
    const js = result.js() as number[][][][];
    // Batch 0 should be all zeros
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++)
        for (let l = 0; l < 6; l++) expect(js[0][j][k][l]).toBe(0);
    // Batch 1: only [1:3, 2:4, 0:3] should be 1
    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) {
        for (let l = 0; l < 6; l++) {
          const expected =
            j >= 1 && j < 3 && k >= 2 && k < 4 && l >= 0 && l < 3 ? 1 : 0;
          expect(js[1][j][k][l]).toBe(expected);
        }
      }
    }
  });
});

// ============================================================
// ND DynamicUpdateSlice JVP (forward-mode AD)
// ============================================================
describe("ND DynamicUpdateSlice JVP", () => {
  test("jvp through ND DUS", () => {
    const f = (dst: np.Array) => {
      using src = np.array([[99, 98]]);
      return lax.dynamicUpdateSlice(dst, src, [1, 0]);
    };
    using x = np.arange(6).reshape([3, 2]);
    using dx = np.ones([3, 2]);
    const [y, dy] = jvp(f, [x], [dx]);
    // Primal: row 1 replaced by [99, 98]
    expect(y.js()).toEqual([
      [0, 1],
      [99, 98],
      [4, 5],
    ]);
    // Tangent: 1 everywhere except row 1 (src is constant → zero tangent there)
    expect(dy.js()).toEqual([
      [1, 1],
      [0, 0],
      [1, 1],
    ]);
    y.dispose();
    dy.dispose();
  });

  test("jvp through ND DUS wrt src", () => {
    const f = (src: np.Array) => {
      using dst = np.zeros([3, 4]);
      return lax.dynamicUpdateSlice(dst, src, [1, 1]);
    };
    using x = np.ones([1, 2]);
    using dx = np.array([[3, 7]]);
    const [y, dy] = jvp(f, [x], [dx]);
    // Primal: dst with src at [1, 1]
    expect(y.js()).toEqual([
      [0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0],
    ]);
    // Tangent: dx at [1, 1], zero elsewhere (dst is constant)
    expect(dy.js()).toEqual([
      [0, 0, 0, 0],
      [0, 3, 7, 0],
      [0, 0, 0, 0],
    ]);
    y.dispose();
    dy.dispose();
  });
});

// ============================================================
// ND DynamicUpdateSlice grad (reverse-mode AD)
// ============================================================
describe("ND DynamicUpdateSlice grad", () => {
  test("grad through ND DUS wrt dst", () => {
    const f = (dst: np.Array) => {
      using src = np.zeros([2, 2]);
      // Zero out the 2x2 block at [1, 1]
      return lax.dynamicUpdateSlice(dst, src, [1, 1]).sum();
    };
    using x = np.ones([4, 4]);
    using g = grad(f)(x);
    // Gradient: 1 everywhere except the 2x2 block at [1, 1] which is 0
    expect(g.js()).toEqual([
      [1, 1, 1, 1],
      [1, 0, 0, 1],
      [1, 0, 0, 1],
      [1, 1, 1, 1],
    ]);
  });

  test("grad through ND DUS wrt src", () => {
    const f = (src: np.Array) => {
      using dst = np.zeros([4, 4]);
      return lax.dynamicUpdateSlice(dst, src, [1, 1]).sum();
    };
    using x = np.ones([2, 2]);
    using g = grad(f)(x);
    // d/d(src) of sum(DUS(zeros, src, [1,1])) = all 1s (each src element
    // contributes exactly once to the sum)
    expect(g.js()).toEqual([
      [1, 1],
      [1, 1],
    ]);
  });

  test("jit(grad) through ND DUS", () => {
    using src = np.zeros([1, 2]);
    const f = (dst: np.Array, s: np.Array) =>
      lax.dynamicUpdateSlice(dst, s, [0, 1]).sum();
    using jitGrad = jit((dst: np.Array) => grad(f)(dst, src));
    using x = np.ones([3, 4]);
    using g = jitGrad(x);
    // Zeroing [0, 1:3] → gradient 0 at those positions, 1 elsewhere
    expect(g.js()).toEqual([
      [1, 0, 0, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ]);
  });
});

// ============================================================
// ND DynamicUpdateSlice vmap
// ============================================================
describe("ND DynamicUpdateSlice vmap", () => {
  test("vmap(ND DUS) batches along leading dim", () => {
    const f = (dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, [0, 1]);
    // Batch of 2, each 3x4 dst, 2x2 src
    using dsts = np.zeros([2, 3, 4]);
    using srcs = np.array([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    using jitF = jit(vmap(f));
    using result = jitF(dsts, srcs);
    const js = result.js() as number[][][];
    // Batch 0: [0,1:3] and [1,1:3] updated
    expect(js[0]).toEqual([
      [0, 1, 2, 0],
      [0, 3, 4, 0],
      [0, 0, 0, 0],
    ]);
    // Batch 1: same pattern
    expect(js[1]).toEqual([
      [0, 5, 6, 0],
      [0, 7, 8, 0],
      [0, 0, 0, 0],
    ]);
  });
});

// ============================================================
// Regression: original single-axis DUS unchanged
// ============================================================
describe("Single-axis DUS regression", () => {
  test("single-axis DUS still works (eager)", () => {
    using dst = np.array([1, 2, 3, 4, 5]);
    using src = np.array([99, 98]);
    using result = lax.dynamicUpdateSlice(dst, src, 2);
    expect(result.js()).toEqual([1, 2, 99, 98, 5]);
  });

  test("single-axis DUS still works (JIT)", () => {
    using jitF = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, 2),
    );
    using dst = np.array([1, 2, 3, 4, 5]);
    using src = np.array([99, 98]);
    using result = jitF(dst, src);
    expect(result.js()).toEqual([1, 2, 99, 98, 5]);
  });

  test("single-axis DUS with axis=1 still works (eager)", () => {
    using dst = np.arange(12).reshape([3, 4]);
    using src = np.array([[99], [98], [97]]);
    using result = lax.dynamicUpdateSlice(dst, src, 2, 1);
    expect(result.js()).toEqual([
      [0, 1, 99, 3],
      [4, 5, 98, 7],
      [8, 9, 97, 11],
    ]);
  });
});

// ============================================================
// Structural: IR stays a single primitive per DUS call
// ============================================================
describe("ND DynamicUpdateSlice (structural)", () => {
  test("multi-axis ND DUS traces to a single DynamicUpdateSliceGeneral equation", () => {
    using dst = np.zeros([4, 6]);
    using src = np.ones([2, 3]);
    const { jaxpr: cj } = makeJaxpr((d: any, s: any) =>
      lax.dynamicUpdateSlice(d, s, [1, 2]),
    )(dst, src);
    const dusEqns = cj.jaxpr.eqns.filter(
      (e: any) => e.primitive === "dynamic_update_slice_general",
    );
    expect(dusEqns.length).toBe(1);
    cj.dispose();
  });
});

// ============================================================
// JIT lowering: avoid per-axis DUS expansion
// ============================================================
describe("ND DynamicUpdateSlice (JIT lowering)", () => {
  test("multi-axis ND DUS lowers to two DUS steps: copy + patch", () => {
    clearCaches();
    using dst = np.zeros([3, 4, 5]);
    using src = np.ones([2, 2, 2]);
    const report = captureJitReport(
      (d: any, s: any) => lax.dynamicUpdateSlice(d, s, [1, 1, 1]),
      dst,
      src,
    );
    expect(report.program.stepCounts.dus).toBe(2);
  });

  test("single-active-axis ND DUS preserves the one-step fast path", () => {
    clearCaches();
    using dst = np.zeros([4, 6]);
    using src = np.ones([4, 2]);
    const report = captureJitReport(
      (d: any, s: any) => lax.dynamicUpdateSlice(d, s, [0, 2]),
      dst,
      src,
    );
    expect(report.program.stepCounts.dus).toBe(1);
  });
});
