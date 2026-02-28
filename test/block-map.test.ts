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

import { DType, init, lax, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
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
