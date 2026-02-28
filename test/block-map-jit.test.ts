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

  test("fused path handles non-divisible dimensions (boundary blocks)", () => {
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

      // Should have taken the fused path (boundary blocks now supported)
      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeDefined();

      f_jit.dispose();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
  });
});

// ---------------------------------------------------------------------------
// In-block reduction tests (Phase 3.1)
// ---------------------------------------------------------------------------
describe("lax.blockMap — in-block reductions", () => {
  const isWebGPU = defaultDevice() === "webgpu";

  test("block - max(block) centers correctly", () => {
    // Reduce-then-elementwise: max is intermediate, output same shape as input
    const body = (block: np.Array) => {
      using m = np.max(block);
      return np.subtract(block, m);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([1, 3, 2, 4, 10, 6, 8, 7], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // Block 0: [1,3,2,4] - 4 = [-3,-1,-2,0]
    // Block 1: [10,6,8,7] - 10 = [0,-4,-2,-3]
    expect(result).toBeAllclose([-3, -1, -2, 0, 0, -4, -2, -3]);
    f_jit.dispose();
  });

  test("block - mean(block) centers with epilogue", () => {
    // Mean reduction has epilogue: acc / size. Tests non-identity epilogue.
    const body = (block: np.Array) => {
      using m = np.mean(block);
      return np.subtract(block, m);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([2, 4, 6, 8, 1, 3, 5, 7], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // Block 0: mean([2,4,6,8]) = 5, [2,4,6,8] - 5 = [-3,-1,1,3]
    // Block 1: mean([1,3,5,7]) = 4, [1,3,5,7] - 4 = [-3,-1,1,3]
    expect(result).toBeAllclose([-3, -1, 1, 3, -3, -1, 1, 3]);
    f_jit.dispose();
  });

  test("exp(block - max(block)) softmax numerator", () => {
    // Two-step: reduce (max), elementwise (subtract + exp). Core of softmax.
    const body = (block: np.Array) => {
      using m = np.max(block);
      using shifted = np.subtract(block, m);
      return np.exp(shifted);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([0, 1, 2, 3, 1, 1, 1, 1], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // Block 0: max=3, exp([-3,-2,-1,0]) = [0.0498, 0.1353, 0.3679, 1.0]
    // Block 1: max=1, exp([0,0,0,0]) = [1,1,1,1]
    expect(result).toBeAllclose(
      [Math.exp(-3), Math.exp(-2), Math.exp(-1), Math.exp(0), 1, 1, 1, 1],
      { atol: 1e-5 },
    );
    f_jit.dispose();
  });

  test("sum reduction as intermediate followed by divide", () => {
    // block / sum(block) — normalize to sum=1
    const body = (block: np.Array) => {
      using s = np.sum(block);
      return np.divide(block, s);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([1, 2, 3, 4, 2, 2, 2, 2], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // Block 0: sum=10, [1/10, 2/10, 3/10, 4/10] = [0.1, 0.2, 0.3, 0.4]
    // Block 1: sum=8, [2/8, 2/8, 2/8, 2/8] = [0.25, 0.25, 0.25, 0.25]
    expect(result).toBeAllclose([0.1, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25]);
    f_jit.dispose();
  });

  test("fused path is taken for reduce-then-elementwise body", () => {
    if (!isWebGPU) return;

    const logs: string[] = [];
    const origInfo = console.info;
    console.info = (...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    };

    try {
      setDebug(1);
      const f_jit = jit((xs: np.Array) =>
        lax.blockMap(
          (block: np.Array) => {
            using m = np.max(block);
            return np.subtract(block, m);
          },
          xs,
          { blockShape: [4] },
        ),
      );
      using xs = np.array([1, 3, 2, 4, 10, 6, 8, 7], {
        dtype: DType.Float32,
      });
      using result = f_jit(xs);
      expect(result).toBeAllclose([-3, -1, -2, 0, 0, -4, -2, -3]);

      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeDefined();

      f_jit.dispose();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
  });

  test("block * 2 then sum as intermediate then broadcast", () => {
    // Multi-step: elementwise → reduce → elementwise
    // block * 2, then divide by sum(block*2)
    const body = (block: np.Array) => {
      using two = np.array(2, { dtype: DType.Float32 });
      using doubled = np.multiply(block, two);
      using s = np.sum(doubled);
      return np.divide(doubled, s);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // doubled = [2,4,6,8], sum = 20
    // [2/20, 4/20, 6/20, 8/20] = [0.1, 0.2, 0.3, 0.4]
    expect(result).toBeAllclose([0.1, 0.2, 0.3, 0.4]);
    f_jit.dispose();
  });

  test("non-power-of-2 block size reduction", () => {
    // blockSize=3 tests the non-power-of-2 tree reduction logic
    const body = (block: np.Array) => {
      using m = np.max(block);
      return np.subtract(block, m);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [3] }),
    );
    using xs = np.array([1, 5, 3, 7, 2, 9], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // Block 0: max([1,5,3])=5, [1-5, 5-5, 3-5] = [-4,0,-2]
    // Block 1: max([7,2,9])=9, [7-9, 2-9, 9-9] = [-2,-7,0]
    expect(result).toBeAllclose([-4, 0, -2, -2, -7, 0]);
    f_jit.dispose();
  });
});

// ---------------------------------------------------------------------------
// Boundary block tests (non-divisible dimensions)
// ---------------------------------------------------------------------------
describe("lax.blockMap — boundary blocks", () => {
  const isWebGPU = defaultDevice() === "webgpu";

  test("elementwise with non-divisible N uses fused path", () => {
    if (!isWebGPU) return;

    const logs: string[] = [];
    const origInfo = console.info;
    console.info = (...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    };

    try {
      setDebug(1);
      const f_jit = jit((xs: np.Array) =>
        lax.blockMap(
          (block: np.Array) => {
            using two = np.array(2, { dtype: DType.Float32 });
            return np.multiply(block, two);
          },
          xs,
          { blockShape: [4] },
        ),
      );
      // N=10, blockShape=4 → 3 blocks, last block has 2 valid elements
      using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], {
        dtype: DType.Float32,
      });
      using result = f_jit(xs);
      expect(result).toBeAllclose([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      // Should use fused path despite non-divisible dimensions
      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeDefined();

      f_jit.dispose();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
  });

  test("identity with non-divisible N", () => {
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap((block: np.Array) => block, xs, { blockShape: [4] }),
    );
    using xs = np.array([1, 2, 3, 4, 5, 6, 7], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([1, 2, 3, 4, 5, 6, 7]);
    f_jit.dispose();
  });

  test("elementwise double with boundary (N=5, block=3)", () => {
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) => {
          using two = np.array(2, { dtype: DType.Float32 });
          return np.multiply(block, two);
        },
        xs,
        { blockShape: [3] },
      ),
    );
    using xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([2, 4, 6, 8, 10]);
    f_jit.dispose();
  });

  test("reduce-then-elementwise with boundary", () => {
    // block - max(block) with N=10, block=4 → 3 blocks, last has 2 elements
    const body = (block: np.Array) => {
      using m = np.max(block);
      return np.subtract(block, m);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([1, 3, 2, 4, 10, 6, 8, 7, 5, 9], {
      dtype: DType.Float32,
    });
    using result = f_jit(xs);
    // Block 0: [1,3,2,4], max=4, result=[-3,-1,-2,0]
    // Block 1: [10,6,8,7], max=10, result=[0,-4,-2,-3]
    // Block 2: [5,9,0,0] (zero-padded), max=9, result=[-4,0,-9,-9] → only [-4,0] written
    expect(result).toBeAllclose([-3, -1, -2, 0, 0, -4, -2, -3, -4, 0]);
    f_jit.dispose();
  });

  test("mean reduction with boundary", () => {
    // block - mean(block) with N=6, block=4
    const body = (block: np.Array) => {
      using m = np.mean(block);
      return np.subtract(block, m);
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([2, 4, 6, 8, 3, 7], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // Block 0: [2,4,6,8], mean=5, result=[-3,-1,1,3]
    // Block 1: [3,7,0,0] (zero-padded), mean=2.5, result=[0.5,4.5,-2.5,-2.5] → only [0.5,4.5]
    expect(result).toBeAllclose([-3, -1, 1, 3, 0.5, 4.5]);
    f_jit.dispose();
  });

  test("single element in last block (N=5, block=4)", () => {
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) => {
          using ten = np.array(10, { dtype: DType.Float32 });
          return np.add(block, ten);
        },
        xs,
        { blockShape: [4] },
      ),
    );
    using xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([11, 12, 13, 14, 15]);
    f_jit.dispose();
  });

  test("grad through boundary block_map", () => {
    const body = (x: np.Array) => np.multiply(x, x);
    const f = (xs: np.Array) => {
      using mapped = lax.blockMap(body, xs, { blockShape: [3] });
      return np.sum(mapped);
    };
    // N=5, blockShape=3 → boundary
    const f_jit = jit(grad(f));
    using x = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using g = f_jit(x);
    // grad of x^2 is 2x
    expect(g).toBeAllclose([2, 4, 6, 8, 10]);
    f_jit.dispose();
  });
});
