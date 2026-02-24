/**
 * Tests for AD/transform gaps fixed in Pool, PoolTranspose, DUS, ScatterAdd.
 *
 * Covers:
 *   - Pool / PoolTranspose vmap rules
 *   - DynamicUpdateSlice JVP, transpose (grad), vmap
 *   - ScatterAdd vmap
 */
import {
  DType,
  grad,
  jit,
  jvp,
  lax,
  numpy as np,
  scatterAdd,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

// ============================================================
// Pool / PoolTranspose vmap
// ============================================================
describe("Pool vmap", () => {
  test("vmap(maxpool) batches along leading dim", () => {
    // Batch of 2, each 4-element 1-D input, window=2, stride=2
    using x = np.array([
      [1, 5, 3, 4],
      [8, 2, 7, 1],
    ]);
    // vmap over axis 0: each row is independently max-pooled
    using jitF = jit(
      vmap((row: np.Array) => lax.reduceWindow(row, np.max, [2], [2])),
    );
    using result = jitF(x);
    // Row 0: max([1,5])=5, max([3,4])=4
    // Row 1: max([8,2])=8, max([7,1])=7
    expect(result.js()).toEqual([
      [5, 4],
      [8, 7],
    ]);
  });

  test("vmap(grad(maxpool)) batches gradients", () => {
    const maxPoolSum = (x: np.Array) =>
      lax.reduceWindow(x, np.max, [2], [2]).sum();

    using jitF = jit(vmap(grad(maxPoolSum)));
    using x = np.array([
      [1, 5, 3, 4],
      [8, 2, 7, 1],
    ]);
    using result = jitF(x);
    // Gradient: 1 at max positions, 0 elsewhere
    // Row 0: max([1,5])=5 -> [0,1], max([3,4])=4 -> [0,1]
    // Row 1: max([8,2])=8 -> [1,0], max([7,1])=7 -> [1,0]
    expect(result.js()).toEqual([
      [0, 1, 0, 1],
      [1, 0, 1, 0],
    ]);
  });
});

// ============================================================
// DynamicUpdateSlice JVP (forward-mode AD)
// ============================================================
describe("DynamicUpdateSlice JVP", () => {
  test("jvp through DUS", () => {
    const f = (dst: np.Array) => {
      using src = np.array([99]);
      return lax.dynamicUpdateSlice(dst, src, 1);
    };
    using x = np.array([1, 2, 3, 4]);
    using dx = np.ones([4]);
    const [y, dy] = jvp(f, [x], [dx]);
    // Primal: [1, 99, 3, 4]
    expect(y.js()).toEqual([1, 99, 3, 4]);
    // Tangent: [1, 0, 1, 1] — DUS zeroes tangent at the slice position
    // because src has zero tangent (constant), dst tangent passes through elsewhere
    expect(dy.js()).toEqual([1, 0, 1, 1]);
    y.dispose();
    dy.dispose();
  });
});

// ============================================================
// DynamicUpdateSlice grad (reverse-mode AD via transpose rule)
// ============================================================
describe("DynamicUpdateSlice grad", () => {
  test("grad through DUS wrt dst", () => {
    const f = (dst: np.Array) => {
      using src = np.array([0]);
      // Zero out position 2
      return lax.dynamicUpdateSlice(dst, src, 2).sum();
    };
    using x = np.array([1, 2, 3, 4]);
    using g = grad(f)(x);
    // d/d(dst) of sum(DUS(dst, [0], 2)) = [1, 1, 0, 1]
    // Position 2 zeroed by src, so gradient is 0 there
    expect(g.js()).toEqual([1, 1, 0, 1]);
  });

  test("grad through DUS wrt src", () => {
    // f(src) = sum(DUS([0,0,0,0], src, 1))
    const f = (src: np.Array) => {
      using dst = np.zeros([4]);
      return lax.dynamicUpdateSlice(dst, src, 1).sum();
    };
    using x = np.array([10, 20]);
    using g = grad(f)(x);
    // d/d(src) of sum(DUS([0,0,0,0], src, 1)) = [1, 1]
    // src contributes at positions 1,2 — gradient is 1 for each
    expect(g.js()).toEqual([1, 1]);
  });

  test("jit(grad) through DUS", () => {
    // Use jit-wrapping to avoid anonymous-const leaks
    using src = np.array([0, 0]);
    const f = (dst: np.Array, s: np.Array) =>
      lax.dynamicUpdateSlice(dst, s, 0).sum();
    using jitGrad = jit((dst: np.Array) => grad(f)(dst, src));
    using x = np.array([1, 2, 3, 4]);
    using g = jitGrad(x);
    // Zeroing positions 0,1 → gradient [0, 0, 1, 1]
    expect(g.js()).toEqual([0, 0, 1, 1]);
  });
});

// ============================================================
// DynamicUpdateSlice vmap
// ============================================================
describe("DynamicUpdateSlice vmap", () => {
  test("vmap(DUS) batches along leading dim", () => {
    const f = (dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, 1);
    // Batch of 2, each row is a 4-element array
    using dsts = np.array([
      [1, 2, 3, 4],
      [10, 20, 30, 40],
    ]);
    using srcs = np.array([
      [99, 98],
      [77, 66],
    ]);
    using jitF = jit(vmap(f));
    using result = jitF(dsts, srcs);
    // Row 0: DUS([1,2,3,4], [99,98], offset=1) = [1, 99, 98, 4]
    // Row 1: DUS([10,20,30,40], [77,66], offset=1) = [10, 77, 66, 40]
    expect(result.js()).toEqual([
      [1, 99, 98, 4],
      [10, 77, 66, 40],
    ]);
  });
});

// ============================================================
// ScatterAdd vmap
// ============================================================
describe("ScatterAdd vmap", () => {
  test("vmap(scatter_add) batches along leading dim with shared indices", () => {
    // Batch of 2, each with 3-element target, scatter at shared positions
    using targets = np.array([
      [0, 0, 0],
      [10, 20, 30],
    ]);
    // indices: 1-D, shared across batch (captured, not batched)
    using indices = np.array([0, 2], { dtype: DType.Int32 });
    using updates = np.array([
      [1, 2],
      [3, 4],
    ]);

    using jitF = jit(
      vmap(
        (t: np.Array, u: np.Array) => scatterAdd(t, indices, u, 0) as np.Array,
      ),
    );
    using result = jitF(targets, updates);
    // Batch 0: [0,0,0] + scatter([0,2], [1,2]) = [1, 0, 2]
    // Batch 1: [10,20,30] + scatter([0,2], [3,4]) = [13, 20, 34]
    expect(result.js()).toEqual([
      [1, 0, 2],
      [13, 20, 34],
    ]);
  });
});
