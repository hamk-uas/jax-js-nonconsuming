/**
 * Phase 3 tests for lax.blockMap — JIT compilation path.
 *
 * Verifies that jit(blockMap(...)) produces block_map JitSteps and
 * executes correctly via both fused WebGPU shader and fallback paths.
 */

import {
  defaultDevice,
  DType,
  grad,
  init,
  jit,
  lax,
  numpy as np,
  setDebug,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

await init();

describe("lax.blockMap — Phase 3 JIT", () => {
  // -------------------------------------------------------------------------
  // Basic JIT: identity body
  // -------------------------------------------------------------------------
  test("jit(blockMap(identity)) roundtrips correctly", () => {
    const f = (xs: np.Array) =>
      lax.blockMap((block: np.Array) => block, xs, { blockShape: [4] });

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([1, 2, 3, 4, 5, 6, 7, 8]);
    f_jit.dispose();
  });

  // -------------------------------------------------------------------------
  // Elementwise body under JIT
  // -------------------------------------------------------------------------
  test("jit(blockMap(double)) computes correctly", () => {
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          np.multiply(block, np.array(2, { dtype: DType.Float32 })),
        xs,
        { blockShape: [3] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5, 6], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([2, 4, 6, 8, 10, 12]);
    f_jit.dispose();
  });

  // -------------------------------------------------------------------------
  // Non-divisible N with padding/trimming
  // -------------------------------------------------------------------------
  test("jit(blockMap) with non-divisible N", () => {
    const f = (xs: np.Array) =>
      lax.blockMap((block: np.Array) => block, xs, { blockShape: [4] });

    const f_jit = jit(f);
    const values = Array.from({ length: 10 }, (_, i) => i + 1);
    using xs = np.array(values, { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose(values);
    f_jit.dispose();
  });

  // -------------------------------------------------------------------------
  // 2D tiling under JIT
  // -------------------------------------------------------------------------
  test("jit(blockMap) with 2D tiling", () => {
    const f = (xs: np.Array) =>
      lax.blockMap((block: np.Array) => block, xs, {
        blockShape: [2, 2],
        inAxes: [0, 1],
        outAxes: [0, 1],
      });

    const f_jit = jit(f);
    using flat = np.arange(16).astype(DType.Float32);
    using xs = flat.reshape([4, 4]);
    using result = f_jit(xs);
    expect(result.shape).toEqual([4, 4]);
    expect(result).toBeAllclose(xs);
    f_jit.dispose();
  });

  // -------------------------------------------------------------------------
  // JIT handles blockMap with computation
  // -------------------------------------------------------------------------
  test("jit(blockMap(triple)) computes correctly", () => {
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          np.multiply(block, np.array(3, { dtype: DType.Float32 })),
        xs,
        { blockShape: [4] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([3, 6, 9, 12, 15, 18, 21, 24]);
    f_jit.dispose();
  });

  // -------------------------------------------------------------------------
  // jit(grad(blockMap)) — end-to-end
  // -------------------------------------------------------------------------
  test("jit(grad(sum(blockMap(x^2))))", () => {
    const body = (x: np.Array) => np.multiply(x, x);
    const f = (xs: np.Array) => {
      using mapped = lax.blockMap(body, xs, { blockShape: [2] });
      return np.sum(mapped);
    };

    const f_jit = jit(grad(f));
    using x = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using g = f_jit(x);
    // grad of x^2 is 2x
    expect(g).toBeAllclose([2, 4, 6, 8]);
    f_jit.dispose();
  });

  // -------------------------------------------------------------------------
  // Multiple JIT calls reuse the compiled program
  // -------------------------------------------------------------------------
  test("repeated jit calls produce same results", () => {
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          np.add(block, np.array(10, { dtype: DType.Float32 })),
        xs,
        { blockShape: [2] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4], { dtype: DType.Float32 });

    using r1 = f_jit(xs);
    using r2 = f_jit(xs);
    expect(r1).toBeAllclose([11, 12, 13, 14]);
    expect(r2).toBeAllclose([11, 12, 13, 14]);
    f_jit.dispose();
  });

  // -------------------------------------------------------------------------
  // Single-element input under JIT
  // -------------------------------------------------------------------------
  test("jit(blockMap) with single element", () => {
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          np.multiply(block, np.array(2, { dtype: DType.Float32 })),
        xs,
        { blockShape: [1] },
      );

    const f_jit = jit(f);
    using xs = np.array([42], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([84]);
    f_jit.dispose();
  });
});

// ---------------------------------------------------------------------------
// Fused path verification
// ---------------------------------------------------------------------------
describe("lax.blockMap — fused shader path", () => {
  const isWebGPU = defaultDevice() === "webgpu";

  test("fused path is taken for divisible elementwise body", () => {
    if (!isWebGPU) return; // Only WebGPU has the fused path

    const logs: string[] = [];
    const origInfo = console.info;
    console.info = (...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    };

    try {
      setDebug(1);
      const f_jit = jit((xs: np.Array) =>
        lax.blockMap(
          (block: np.Array) =>
            np.multiply(block, np.array(2, { dtype: DType.Float32 })),
          xs,
          { blockShape: [4] },
        ),
      );
      using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
      using result = f_jit(xs);
      expect(result).toBeAllclose([2, 4, 6, 8, 10, 12, 14, 16]);

      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeDefined();

      f_jit.dispose();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
  });

  test("fused path handles identity pass-through body", () => {
    if (!isWebGPU) return;

    const logs: string[] = [];
    const origInfo = console.info;
    console.info = (...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    };

    try {
      setDebug(1);
      const f_jit = jit((xs: np.Array) =>
        lax.blockMap((block: np.Array) => block, xs, { blockShape: [4] }),
      );
      using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
      using result = f_jit(xs);
      expect(result).toBeAllclose([1, 2, 3, 4, 5, 6, 7, 8]);

      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeDefined();

      f_jit.dispose();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
  });

  test("fallback for non-divisible dimensions", () => {
    if (!isWebGPU) return;

    const logs: string[] = [];
    const origInfo = console.info;
    console.info = (...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    };

    try {
      setDebug(1);
      const f_jit = jit((xs: np.Array) =>
        lax.blockMap((block: np.Array) => block, xs, { blockShape: [4] }),
      );
      using xs = np.array(
        Array.from({ length: 10 }, (_, i) => i + 1),
        { dtype: DType.Float32 },
      );
      using result = f_jit(xs);
      expect(result).toBeAllclose(Array.from({ length: 10 }, (_, i) => i + 1));

      // Should NOT have taken the fused path
      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeUndefined();

      // Should have logged the fallback reason
      const fallbackLog = logs.find((l) => l.includes("not divisible"));
      expect(fallbackLog).toBeDefined();

      f_jit.dispose();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
  });
});
