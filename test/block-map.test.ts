/**
 * Phase 1 tests for lax.blockMap — Core IR + Eager Fallback.
 *
 * Structure:
 * - T2.1: Identity body (tiling/reassembly roundtrip)
 * - T2.2: Elementwise body (computation inside block)
 * - T2.3: Non-divisible N (padding + trimming)
 * - T2.4: 2D tiling (multi-axis slice/concat)
 * - T2.5: Pytree elems
 * - T2.6: Length-0 edge case
 * - T2.7: Single-element input
 * - T2.8: Leak regression: padded blocks
 * - T2.9: Leak regression: pytree intermediates
 * - T2.10: Leak regression: non-divisible N with using in body
 */

import {
  DType,
  grad,
  init,
  jit,
  jvp,
  lax,
  numpy as np,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

await init();

describe("lax.blockMap — Phase 1 eager", () => {
  // T2.1: Identity body — f(block) = block
  test("identity body (tiling/reassembly roundtrip)", () => {
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    using result = lax.blockMap((block: np.Array) => block, xs, {
      blockShape: [4],
    });
    expect(result).toBeAllclose([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  // T2.2: Elementwise body — f(block) = block * 2
  test("elementwise body (double)", () => {
    using xs = np.array([1, 2, 3, 4, 5, 6], { dtype: DType.Float32 });
    using result = lax.blockMap(
      (block: np.Array) => {
        using two = np.array(2, { dtype: DType.Float32 });
        return np.multiply(block, two);
      },
      xs,
      { blockShape: [3] },
    );
    expect(result).toBeAllclose([2, 4, 6, 8, 10, 12]);
  });

  // T2.3: Non-divisible N (N=10, blockShape=[4])
  test("non-divisible N pads and trims", () => {
    const values = Array.from({ length: 10 }, (_, i) => i + 1);
    using xs = np.array(values, { dtype: DType.Float32 });
    using result = lax.blockMap((block: np.Array) => block, xs, {
      blockShape: [4],
    });
    expect(result).toBeAllclose(values);
  });

  // T2.4: 2D tiling (blockShape=[2,2])
  test("2D tiling with identity body", () => {
    // 4x4 matrix tiled into 2x2 blocks
    using flat = np.arange(16).astype(DType.Float32);
    using xs = flat.reshape([4, 4]);
    using result = lax.blockMap((block: np.Array) => block, xs, {
      blockShape: [2, 2],
      inAxes: [0, 1],
      outAxes: [0, 1],
    });
    expect(result.shape).toEqual([4, 4]);
    expect(result).toBeAllclose(xs);
  });

  // T2.5: Pytree elems (object with two arrays)
  test("pytree elems", () => {
    using a = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using b = np.array([10, 20, 30, 40], { dtype: DType.Float32 });
    const result = lax.blockMap(
      ({ a: aBlock, b: bBlock }: { a: np.Array; b: np.Array }) => ({
        a: np.add(aBlock, bBlock),
        b: np.subtract(bBlock, aBlock),
      }),
      { a, b },
      { blockShape: [2], inAxes: [0], outAxes: [0] },
    );
    using ra = result.a;
    using rb = result.b;
    expect(ra).toBeAllclose([11, 22, 33, 44]);
    expect(rb).toBeAllclose([9, 18, 27, 36]);
  });

  // T2.6: Length-0 edge case
  test("length-0 input", () => {
    using xs = np.zeros([0], { dtype: DType.Float32 });
    using result = lax.blockMap((block: np.Array) => block, xs, {
      blockShape: [4],
    });
    expect(result.shape).toEqual([0]);
  });

  // T2.7: Single-element input
  test("single-element input", () => {
    using xs = np.array([42], { dtype: DType.Float32 });
    using result = lax.blockMap(
      (block: np.Array) => {
        using two = np.array(2, { dtype: DType.Float32 });
        return np.multiply(block, two);
      },
      xs,
      { blockShape: [1] },
    );
    expect(result).toBeAllclose([84]);
  });

  // T2.8: Leak regression: padded blocks
  // The checkLeaks harness in setup.ts catches any leaked slots.
  test("no leaks with padded blocks", () => {
    using xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using result = lax.blockMap((block: np.Array) => block, xs, {
      blockShape: [3],
    });
    expect(result).toBeAllclose([1, 2, 3, 4, 5]);
  });

  // T2.9: Leak regression: pytree intermediates
  test("no leaks with pytree intermediates", () => {
    using a = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using b = np.array([5, 6, 7, 8], { dtype: DType.Float32 });
    const result = lax.blockMap(
      ({ a: aBlock, b: bBlock }: { a: np.Array; b: np.Array }) => ({
        sum: np.add(aBlock, bBlock),
      }),
      { a, b },
      { blockShape: [2], inAxes: [0], outAxes: [0] },
    );
    using sum = result.sum;
    expect(sum).toBeAllclose([6, 8, 10, 12]);
  });

  // T2.10: Leak regression: non-divisible N with using in body
  test("no leaks with non-divisible N and body intermediates", () => {
    using xs = np.array([1, 2, 3, 4, 5, 6, 7], { dtype: DType.Float32 });
    using result = lax.blockMap(
      (block: np.Array) => {
        using two = np.array(2, { dtype: DType.Float32 });
        using doubled = np.multiply(block, two);
        using one = np.array(1, { dtype: DType.Float32 });
        return np.add(doubled, one);
      },
      xs,
      { blockShape: [3] },
    );
    expect(result).toBeAllclose([3, 5, 7, 9, 11, 13, 15]);
  });
});

// ==========================================================================
// Phase 1b: foriLoop and dynamicSlice
// ==========================================================================

describe("lax.foriLoop — Phase 1b", () => {
  // T3.1: Simple counter
  test("simple carry accumulation", () => {
    using init = np.array(0, { dtype: DType.Float32 });
    using result = lax.foriLoop(
      0,
      5,
      (i: np.Array, carry: np.Array) => {
        using iF = i.astype(DType.Float32);
        return np.add(carry, iF);
      },
      init,
    );
    // 0 + 1 + 2 + 3 + 4 = 10
    expect(result).toBeAllclose(10);
  });

  // T3.2: Matrix accumulation
  test("matrix carry accumulation", () => {
    using init = np.zeros([2, 2], { dtype: DType.Float32 });
    using result = lax.foriLoop(
      0,
      3,
      (_i: np.Array, carry: np.Array) => {
        using one = np.ones([2, 2], { dtype: DType.Float32 });
        return np.add(carry, one);
      },
      init,
    );
    expect(result).toBeAllclose([
      [3, 3],
      [3, 3],
    ]);
  });

  // T3.3: Zero iterations (lower === upper)
  test("zero iterations returns init unchanged", () => {
    using init = np.array([1, 2, 3], { dtype: DType.Float32 });
    using result = lax.foriLoop(
      5,
      5,
      (_i: np.Array, carry: np.Array) => {
        using big = np.array([100, 100, 100], { dtype: DType.Float32 });
        return np.add(carry, big);
      },
      init,
    );
    expect(result).toBeAllclose([1, 2, 3]);
  });

  // T3.4: Body uses loop index for data-dependent computation
  test("data-dependent loop body using index", () => {
    // Compute sum of squares: 0^2 + 1^2 + 2^2 + 3^2 = 14
    using init = np.array(0, { dtype: DType.Float32 });
    using result = lax.foriLoop(
      0,
      4,
      (i: np.Array, carry: np.Array) => {
        using iF = i.astype(DType.Float32);
        using sq = np.multiply(iF, iF);
        return np.add(carry, sq);
      },
      init,
    );
    expect(result).toBeAllclose(14);
  });

  // T3.5: grad through foriLoop
  test("grad through foriLoop", () => {
    // f(x) = fori_loop(0, 3, (i, carry) => carry * x, 1.0)
    // = x^3, grad = 3*x^2
    const f = (x: np.Array) => {
      using one = np.array(1, { dtype: DType.Float32 });
      return lax.foriLoop(
        0,
        3,
        (_i: np.Array, carry: np.Array) => np.multiply(carry, x),
        one,
      );
    };
    using x = np.array(2, { dtype: DType.Float32 });
    using g = grad(f)(x);
    // d/dx(x^3) = 3x^2 = 12
    expect(g).toBeAllclose(12);
  });

  // T3.6: Pytree carry
  test("pytree carry", () => {
    using a = np.array(0, { dtype: DType.Float32 });
    using b = np.array(1, { dtype: DType.Float32 });
    const result = lax.foriLoop(
      0,
      3,
      (_i: np.Array, carry: { sum: np.Array; prod: np.Array }) => {
        using two = np.array(2, { dtype: DType.Float32 });
        return {
          sum: np.add(carry.sum, two),
          prod: np.multiply(carry.prod, two),
        };
      },
      { sum: a, prod: b },
    );
    using sum = result.sum;
    using prod = result.prod;
    expect(sum).toBeAllclose(6); // 0 + 2 + 2 + 2
    expect(prod).toBeAllclose(8); // 1 * 2 * 2 * 2
  });
});

describe("lax.dynamicSlice — Phase 1b", () => {
  // T4.1: Basic 1D dynamic slice
  test("basic 1D slice", () => {
    using x = np.arange(10).astype(DType.Float32);
    using start = np.array(3, { dtype: DType.Int32 });
    using result = lax.dynamicSlice(x, [start], [4]);
    expect(result).toBeAllclose([3, 4, 5, 6]);
  });

  // T4.2: Out-of-bounds clamping
  test("out-of-bounds start is clamped", () => {
    using x = np.arange(8).astype(DType.Float32);
    using start = np.array(100, { dtype: DType.Int32 });
    using result = lax.dynamicSlice(x, [start], [3]);
    // Clamped to max valid start = 8 - 3 = 5, so [5, 6, 7]
    expect(result).toBeAllclose([5, 6, 7]);
  });

  // T4.3: Negative start is clamped to 0
  test("negative start is clamped to 0", () => {
    using x = np.arange(5).astype(DType.Float32);
    using start = np.array(-10, { dtype: DType.Int32 });
    using result = lax.dynamicSlice(x, [start], [3]);
    expect(result).toBeAllclose([0, 1, 2]);
  });

  // T4.4: 2D dynamic slice
  test("2D slice", () => {
    using xInt = np.arange(12);
    using xReshaped = xInt.reshape([3, 4]);
    using x = xReshaped.astype(DType.Float32);
    using s0 = np.array(1, { dtype: DType.Int32 });
    using s1 = np.array(2, { dtype: DType.Int32 });
    using result = lax.dynamicSlice(x, [s0, s1], [2, 2]);
    // Rows 1-2, cols 2-3: [[6, 7], [10, 11]]
    expect(result).toBeAllclose([
      [6, 7],
      [10, 11],
    ]);
  });

  // T4.5: Full-size slice (identity)
  test("full-size slice is identity", () => {
    using x = np.array([10, 20, 30], { dtype: DType.Float32 });
    using start = np.array(0, { dtype: DType.Int32 });
    using result = lax.dynamicSlice(x, [start], [3]);
    expect(result).toBeAllclose([10, 20, 30]);
  });

  // T4.6: JVP of dynamic_slice
  test("jvp through dynamic_slice", () => {
    using x = np.arange(8).astype(DType.Float32);
    using dx = np.ones([8], { dtype: DType.Float32 });
    using start = np.array(2, { dtype: DType.Int32 });
    const f = (operand: np.Array) => lax.dynamicSlice(operand, [start], [3]);
    const [primal, tangent] = jvp(f, [x], [dx]);
    using p = primal;
    using t = tangent;
    expect(p).toBeAllclose([2, 3, 4]);
    expect(t).toBeAllclose([1, 1, 1]); // ones sliced → ones
  });

  // T4.7: grad through dynamic_slice
  test("grad through dynamic_slice (sum of slice)", () => {
    // f(x) = sum(dynamic_slice(x, [2], [3]))
    // grad: 1s at positions 2,3,4 and 0s elsewhere
    using start = np.array(2, { dtype: DType.Int32 });
    const f = (x: np.Array) => {
      using sliced = lax.dynamicSlice(x, [start], [3]);
      return np.sum(sliced);
    };
    using x = np.arange(8).astype(DType.Float32);
    using g = grad(f)(x);
    expect(g).toBeAllclose([0, 0, 1, 1, 1, 0, 0, 0]);
  });

  // T4.8: foriLoop + dynamicSlice together (the key use case)
  test("foriLoop with dynamicSlice for K-accumulation", () => {
    // Simulate tiled matmul inner loop: accumulate slices
    using x = np.array([1, 2, 3, 4, 5, 6], { dtype: DType.Float32 });
    using init = np.array(0, { dtype: DType.Float32 });
    using result = lax.foriLoop(
      0,
      3,
      (i: np.Array, acc: np.Array) => {
        using two = np.array(2, { dtype: DType.Int32 });
        using start = np.multiply(i, two);
        using slice = lax.dynamicSlice(x, [start], [2]);
        using sliceSum = np.sum(slice);
        return np.add(acc, sliceSum);
      },
      init,
    );
    // slice[0:2] = [1,2] sum=3, slice[2:4] = [3,4] sum=7, slice[4:6] = [5,6] sum=11
    // total = 3 + 7 + 11 = 21
    expect(result).toBeAllclose(21);
  });

  // T4.9: vmap over batched operand, shared start indices
  test("vmap(dynamicSlice) batched operand, shared indices", () => {
    // Batch of 3 vectors, each length 6. Slice [2..4] from each.
    using xRange = np.arange(18);
    using xFloat = xRange.astype(DType.Float32);
    using x = xFloat.reshape([3, 6]);
    using start = np.array(2, { dtype: DType.Int32 });
    const f = jit((row: np.Array) => lax.dynamicSlice(row, [start], [3]));
    const vf = vmap(f);
    using result = vf(x);
    // Row 0: [0,1,2,3,4,5] → [2,3,4]
    // Row 1: [6,7,8,9,10,11] → [8,9,10]
    // Row 2: [12,13,14,15,16,17] → [14,15,16]
    expect(result.shape).toEqual([3, 3]);
    expect(result).toBeAllclose([
      [2, 3, 4],
      [8, 9, 10],
      [14, 15, 16],
    ]);
    f.dispose();
  });

  // T4.10: vmap over batched operand for uncheckedDynamicSlice
  test("vmap(uncheckedDynamicSlice) batched operand, shared indices", () => {
    using xRange = np.arange(18);
    using xFloat = xRange.astype(DType.Float32);
    using x = xFloat.reshape([3, 6]);
    using start = np.array(0, { dtype: DType.Int32 });
    const f = jit((row: np.Array) =>
      lax.uncheckedDynamicSlice(row, [start], [4]),
    );
    const vf = vmap(f);
    using result = vf(x);
    expect(result.shape).toEqual([3, 4]);
    expect(result).toBeAllclose([
      [0, 1, 2, 3],
      [6, 7, 8, 9],
      [12, 13, 14, 15],
    ]);
    f.dispose();
  });
});

// ============================================================================
// Phase 5 — Blocked AssociativeScan (T7)
// ============================================================================

describe("Phase 5 — blocked associativeScan", () => {
  // T7.1: Small N (single block, N < blockSize=256)
  test("T7.1: cumsum N=64 single block", () => {
    const f = jit((xs: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), xs),
    );
    using xs = np.arange(64).astype(DType.Float32);
    using result = f(xs);
    const vals = Array.from({ length: 64 }, (_, i) => (i * (i + 1)) / 2);
    using expected = np.array(vals, { dtype: DType.Float32 });
    expect(result).toBeAllclose(expected);
    f.dispose();
  });

  // T7.2: Multi-block N=1024
  test("T7.2: cumsum N=1024 multi-block", () => {
    const f = jit((xs: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), xs),
    );
    using xs = np.ones([1024]).astype(DType.Float32);
    using result = f(xs);
    const vals = Array.from({ length: 1024 }, (_, i) => i + 1);
    using expected = np.array(vals, { dtype: DType.Float32 });
    expect(result).toBeAllclose(expected);
    f.dispose();
  });

  // T7.3: Large N=4096
  test("T7.3: cumsum N=4096", () => {
    const f = jit((xs: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), xs),
    );
    using xs = np.ones([4096]).astype(DType.Float32);
    using result = f(xs);
    expect(result.dataSync()[4095]).toBe(4096);
    f.dispose();
  });

  // T7.4: Non-power-of-2 N=10 (irregular block)
  test("T7.4: cumsum N=10 non-power-of-2", () => {
    const f = jit((xs: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), xs),
    );
    using xs = np.arange(1, 11).astype(DType.Float32);
    using result = f(xs);
    using expected = np.array([1, 3, 6, 10, 15, 21, 28, 36, 45, 55], {
      dtype: DType.Float32,
    });
    expect(result).toBeAllclose(expected);
    f.dispose();
  });

  // T7.5: Non-add operator (multiply → cumprod)
  test("T7.5: cumprod via multiply", () => {
    const f = jit((xs: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => np.multiply(a, b), xs),
    );
    using xs = np.array([2, 3, 4, 5], { dtype: DType.Float32 });
    using result = f(xs);
    using expected = np.array([2, 6, 24, 120], { dtype: DType.Float32 });
    expect(result).toBeAllclose(expected);
    f.dispose();
  });

  // T7.6: Reverse scan
  test("T7.6: reverse cumsum N=300", () => {
    const f = jit((xs: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), xs, {
        reverse: true,
      }),
    );
    using xs = np.ones([300]).astype(DType.Float32);
    using result = f(xs);
    const vals = Array.from({ length: 300 }, (_, i) => 300 - i);
    using expected = np.array(vals, { dtype: DType.Float32 });
    expect(result).toBeAllclose(expected);
    f.dispose();
  });

  // T7.7: grad through blocked associativeScan
  // Note: jit(grad(f)) not grad(jit(f)) — see copilot-instructions
  test("T7.7: jit(grad(associativeScan)) cumsum", () => {
    const f = (xs: np.Array) => {
      using scanned = lax.associativeScan(
        (a: np.Array, b: np.Array) => np.add(a, b),
        xs,
      );
      return np.sum(scanned);
    };
    const gf = jit(grad(f));
    using xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using g = gf(xs);
    // d(sum(cumsum(x)))/dx_i = N - i
    using expected = np.array([5, 4, 3, 2, 1], { dtype: DType.Float32 });
    expect(g).toBeAllclose(expected);
    gf.dispose();
  });
});
