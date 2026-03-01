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
// Guard / fallback tests (T5a.3, T5a.4)
// ---------------------------------------------------------------------------
describe("lax.blockMap — fused shader guards", () => {
  const isWebGPU = defaultDevice() === "webgpu";

  test("blockSize exceeding maxComputeInvocationsPerWorkgroup falls back correctly", () => {
    // T5a.4: blockShape=[512] exceeds typical maxInvocations=256.
    // The fused path should detect this and fall back to the JS loop,
    // still producing the correct result.
    using two = np.array(2, { dtype: DType.Float32 });
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap((block: np.Array) => np.multiply(block, two), xs, {
        blockShape: [512],
      }),
    );
    using xs = np.arange(1024).astype(DType.Float32);
    using expected = np.multiply(xs, two);
    using result = f_jit(xs);
    expect(result).toBeAllclose(expected);
    f_jit.dispose();
  });

  test("blockSize exceeding maxInvocations logs fallback on WebGPU", () => {
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
          (block: np.Array) =>
            np.multiply(block, np.array(2, { dtype: DType.Float32 })),
          xs,
          { blockShape: [512] },
        ),
      );
      using xs = np.arange(1024).astype(DType.Float32);
      using result = f_jit(xs);
      using two = np.array(2, { dtype: DType.Float32 });
      expect(result).toBeAllclose(np.multiply(xs, two));

      // Should NOT have taken the fused path
      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeUndefined();

      // Should log the fallback reason
      const fallbackLog = logs.find(
        (l) => l.includes("blockSize") && l.includes("maxInvocations"),
      );
      expect(fallbackLog).toBeDefined();

      f_jit.dispose();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
  });

  test("body exceeding shmem budget falls back correctly", () => {
    // T5a.3: Create a body with many sequential reductions. Each reduction
    // adds a blockSize*4 workspace to shmem. With blockSize=256 and 17
    // reductions: 17 * 1024 = 17408 bytes > 16384 byte budget.
    const body = (block: np.Array) => {
      let acc = block;
      for (let i = 0; i < 17; i++) {
        using s = np.sum(acc);
        acc = np.subtract(acc, s);
      }
      return acc;
    };
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [256] }),
    );
    // Compute expected result eagerly
    using xs = np.arange(256).astype(DType.Float32);
    using expected = lax.blockMap(body, xs, { blockShape: [256] });
    using result = f_jit(xs);
    expect(result).toBeAllclose(expected, { atol: 1e-3 });
    f_jit.dispose();
  });

  test("shmem budget exceeded logs fallback on WebGPU", () => {
    if (!isWebGPU) return;

    const logs: string[] = [];
    const origInfo = console.info;
    console.info = (...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    };

    try {
      setDebug(1);
      const body = (block: np.Array) => {
        let acc = block;
        for (let i = 0; i < 17; i++) {
          using s = np.sum(acc);
          acc = np.subtract(acc, s);
        }
        return acc;
      };
      const f_jit = jit((xs: np.Array) =>
        lax.blockMap(body, xs, { blockShape: [256] }),
      );
      using xs = np.arange(256).astype(DType.Float32);
      using result = f_jit(xs);
      void result; // consume

      // Should NOT have taken the fused path
      const fusedLog = logs.find((l) => l.includes("fused WebGPU shader path"));
      expect(fusedLog).toBeUndefined();

      // Should log the shmem fallback reason
      const shmemLog = logs.find(
        (l) => l.includes("shmem") && l.includes("fallback"),
      );
      expect(shmemLog).toBeDefined();

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

  // =========================================================================
  // fori_loop inside block_map
  // =========================================================================

  test("jit(blockMap(foriLoop)) simple accumulation", () => {
    // Each block accumulates block + block + block = 3 * block via 3 iterations
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.foriLoop(
            0,
            3,
            (_i: np.Array, acc: np.Array) => np.add(acc, block),
            np.zerosLike(block),
          ),
        xs,
        { blockShape: [4] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([3, 6, 9, 12, 15, 18, 21, 24]);
    f_jit.dispose();
  });

  test("jit(blockMap(foriLoop)) uses loop index", () => {
    // Computes block * sum(0..4) = block * (0+1+2+3) = block * 6
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.foriLoop(
            0,
            4,
            (i: np.Array, acc: np.Array) => np.add(acc, np.multiply(block, i)),
            np.zerosLike(block),
          ),
        xs,
        { blockShape: [4] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // [1,2,3,4] * 6 = [6,12,18,24]
    expect(result).toBeAllclose([6, 12, 18, 24]);
    f_jit.dispose();
  });

  test("jit(blockMap(foriLoop)) with boundary block", () => {
    // N=5, blockShape=4 → 1 full block + 1 boundary block
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.foriLoop(
            0,
            2,
            (_i: np.Array, acc: np.Array) => np.add(acc, block),
            np.zerosLike(block),
          ),
        xs,
        { blockShape: [4] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([2, 4, 6, 8, 10]);
    f_jit.dispose();
  });

  test("jit(blockMap) foriLoop + elementwise chain", () => {
    // foriLoop does accumulation, then multiply result by 2
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) => {
          using acc = lax.foriLoop(
            0,
            3,
            (_i: np.Array, acc: np.Array) => np.add(acc, block),
            np.zerosLike(block),
          );
          return np.multiply(acc, np.array(2, { dtype: DType.Float32 }));
        },
        xs,
        { blockShape: [4] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // [3,6,9,12] * 2 = [6,12,18,24]
    expect(result).toBeAllclose([6, 12, 18, 24]);
    f_jit.dispose();
  });

  // =========================================================================
  // dynamic_slice inside block_map
  // =========================================================================

  test("jit(blockMap(dynamicSlice)) extracts sub-block", () => {
    // dynamicSlice with start=0 and sliceSize=blockShape is identity + mul
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) => {
          using sliced = lax.dynamicSlice(
            block,
            [np.array(0, { dtype: DType.Int32 })],
            [4],
          );
          return np.multiply(sliced, np.array(3, { dtype: DType.Float32 }));
        },
        xs,
        { blockShape: [4] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([3, 6, 9, 12, 15, 18, 21, 24]);
    f_jit.dispose();
  });

  test("jit(blockMap(foriLoop+dynamicSlice)) tiled accumulation", () => {
    // fori_loop 2 iterations: acc += dynamicSlice(block, [0], [4])
    // Result: 2 * block
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.foriLoop(
            0,
            2,
            (_k: np.Array, acc: np.Array) => {
              using slice = lax.dynamicSlice(
                block,
                [np.array(0, { dtype: DType.Int32 })],
                [4],
              );
              return np.add(acc, slice);
            },
            np.zerosLike(block),
          ),
        xs,
        { blockShape: [4] },
      );

    const f_jit = jit(f);
    using xs = np.array([10, 20, 30, 40, 50, 60, 70, 80], {
      dtype: DType.Float32,
    });
    using result = f_jit(xs);
    expect(result).toBeAllclose([20, 40, 60, 80, 100, 120, 140, 160]);
    f_jit.dispose();
  });

  test("jit(blockMap(foriLoop+dynamicSlice)) with loop-index offset", () => {
    // blockShape=2. fori_loop 1 iteration: acc += dynamicSlice(block, [0], [2])
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.foriLoop(
            0,
            1,
            (_k: np.Array, acc: np.Array) => {
              using slice = lax.dynamicSlice(
                block,
                [np.array(0, { dtype: DType.Int32 })],
                [2],
              );
              return np.add(acc, slice);
            },
            np.zerosLike(block),
          ),
        xs,
        { blockShape: [2] },
      );

    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5, 6], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([1, 2, 3, 4, 5, 6]);
    f_jit.dispose();
  });

  // =========================================================================
  // standalone jit(dynamicSlice) tests
  // =========================================================================

  test("jit(dynamicSlice) standalone 1D", () => {
    const f = (x: np.Array, start: np.Array) =>
      lax.dynamicSlice(x, [start], [3]);

    const f_jit = jit(f);
    using x = np.array([10, 20, 30, 40, 50], { dtype: DType.Float32 });
    using start = np.array(1, { dtype: DType.Int32 });
    using result = f_jit(x, start);
    expect(result).toBeAllclose([20, 30, 40]);
    f_jit.dispose();
  });

  test("jit(dynamicSlice) clamps out-of-bounds start", () => {
    const f = (x: np.Array, start: np.Array) =>
      lax.dynamicSlice(x, [start], [3]);

    const f_jit = jit(f);
    using x = np.array([10, 20, 30, 40, 50], { dtype: DType.Float32 });
    using start = np.array(10, { dtype: DType.Int32 });
    using result = f_jit(x, start);
    expect(result).toBeAllclose([30, 40, 50]);
    f_jit.dispose();
  });

  test("jit(dynamicSlice) 2D slice", () => {
    const f = (x: np.Array, s0: np.Array, s1: np.Array) =>
      lax.dynamicSlice(x, [s0, s1], [2, 2]);

    const f_jit = jit(f);
    using x = np.array(
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ],
      { dtype: DType.Float32 },
    );
    using s0 = np.array(1, { dtype: DType.Int32 });
    using s1 = np.array(1, { dtype: DType.Int32 });
    using result = f_jit(x, s0, s1);
    expect(result).toBeAllclose([
      [6, 7],
      [10, 11],
    ]);
    f_jit.dispose();
  });

  // =========================================================================
  // Phase 4: Per-thread contraction (Dot in fori_loop body)
  // =========================================================================

  test("jit(blockMap(foriLoop+matmul)) per-thread contraction 4x4", () => {
    // Tiled matmul: C = A @ B with Br=Bc=Bk=2 for simplicity
    // A: [4,4], B: [4,4], blockShape: [2,2], K/Bk = 2 iterations
    const Br = 2,
      Bc = 2,
      Bk = 2;

    const tiledMatmul = (A: np.Array, B: np.Array) => {
      const K = (A.shape[1] as number) / Bk;
      return lax.blockMap(
        ({ A: aTile, B: bTile }: { A: np.Array; B: np.Array }) =>
          lax.foriLoop(
            0,
            K,
            (k: np.Array, acc: np.Array) => {
              using kIdx = np.multiply(k, np.array(Bk, { dtype: DType.Int32 }));
              using z0 = np.array(0, { dtype: DType.Int32 });
              using a = lax.dynamicSlice(aTile, [z0, kIdx], [Br, Bk]);
              using b = lax.dynamicSlice(bTile, [kIdx, z0], [Bk, Bc]);
              using prod = np.matmul(a, b);
              return np.add(acc, prod);
            },
            np.zeros([Br, Bc], { dtype: DType.Float32 }),
          ),
        { A, B },
        {
          blockShape: [Br, Bc],
          inAxes: [
            [0, null],
            [null, 1],
          ],
          outAxes: [[0, 1]],
        },
      );
    };

    const f_jit = jit(tiledMatmul);
    using A = np.array(
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ],
      { dtype: DType.Float32 },
    );
    using B = np.array(
      [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
      ],
      { dtype: DType.Float32 },
    );
    // Expected: A @ B
    // Hand-compute: row0 = [1*1+2*0+3*1+4*0, 1*0+2*1+3*1+4*0, 1*0+2*1+3*0+4*1, 1*1+2*0+3*0+4*1]
    //            = [4, 5, 6, 5]
    using expected = np.matmul(A, B);
    using result = f_jit(A, B);
    expect(result).toBeAllclose(expected);
    f_jit.dispose();
  });

  test("jit(blockMap(foriLoop+matmul)) tiled matmul 8x8", () => {
    const Br = 4,
      Bc = 4,
      Bk = 4;

    const tiledMatmul = (A: np.Array, B: np.Array) => {
      const K = (A.shape[1] as number) / Bk;
      return lax.blockMap(
        ({ A: aTile, B: bTile }: { A: np.Array; B: np.Array }) =>
          lax.foriLoop(
            0,
            K,
            (k: np.Array, acc: np.Array) => {
              using kIdx = np.multiply(k, np.array(Bk, { dtype: DType.Int32 }));
              using z0 = np.array(0, { dtype: DType.Int32 });
              using a = lax.dynamicSlice(aTile, [z0, kIdx], [Br, Bk]);
              using b = lax.dynamicSlice(bTile, [kIdx, z0], [Bk, Bc]);
              using prod = np.matmul(a, b);
              return np.add(acc, prod);
            },
            np.zeros([Br, Bc], { dtype: DType.Float32 }),
          ),
        { A, B },
        {
          blockShape: [Br, Bc],
          inAxes: [
            [0, null],
            [null, 1],
          ],
          outAxes: [[0, 1]],
        },
      );
    };

    const f_jit = jit(tiledMatmul);

    // Create 8x8 matrices with known values
    using A_flat = np.arange(64).astype(DType.Float32);
    using A = A_flat.reshape([8, 8]);
    using B = np.eye(8, { dtype: DType.Float32 }); // identity → result should be A
    using result = f_jit(A, B);
    expect(result).toBeAllclose(A);
    f_jit.dispose();
  });

  test("jit(blockMap(foriLoop+matmul)) tiled matmul 16x16 Br=Bc=Bk=4", () => {
    const Br = 4,
      Bc = 4,
      Bk = 4;

    const tiledMatmul = (A: np.Array, B: np.Array) => {
      const K = (A.shape[1] as number) / Bk;
      return lax.blockMap(
        ({ A: aTile, B: bTile }: { A: np.Array; B: np.Array }) =>
          lax.foriLoop(
            0,
            K,
            (k: np.Array, acc: np.Array) => {
              using kIdx = np.multiply(k, np.array(Bk, { dtype: DType.Int32 }));
              using z0 = np.array(0, { dtype: DType.Int32 });
              using a = lax.dynamicSlice(aTile, [z0, kIdx], [Br, Bk]);
              using b = lax.dynamicSlice(bTile, [kIdx, z0], [Bk, Bc]);
              using prod = np.matmul(a, b);
              return np.add(acc, prod);
            },
            np.zeros([Br, Bc], { dtype: DType.Float32 }),
          ),
        { A, B },
        {
          blockShape: [Br, Bc],
          inAxes: [
            [0, null],
            [null, 1],
          ],
          outAxes: [[0, 1]],
        },
      );
    };

    const f_jit = jit(tiledMatmul);
    using A_flat = np.arange(256).astype(DType.Float32);
    using A = A_flat.reshape([16, 16]);
    using B_flat = np.arange(256).astype(DType.Float32);
    using B_sq = B_flat.reshape([16, 16]);
    using B = np.transpose(B_sq);
    using expected = np.matmul(A, B);
    using result = f_jit(A, B);
    expect(result).toBeAllclose(expected, { atol: 1e-3 });
    f_jit.dispose();
  });
});

// ---------------------------------------------------------------------------
// Phase 4: lax.tiledMatmul library function
// ---------------------------------------------------------------------------
describe("lax.tiledMatmul", () => {
  test("64x64 matches np.matmul", () => {
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });
    using expected = np.matmul(A, B);
    using result = lax.tiledMatmul(A, B);
    expect(result).toBeAllclose(expected, { atol: 1e-3 });
  });

  test("64x64 jit(tiledMatmul) matches np.matmul", () => {
    const f = jit((A: np.Array, B: np.Array) => lax.tiledMatmul(A, B));
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });
    using expected = np.matmul(A, B);
    using result = f(A, B);
    expect(result).toBeAllclose(expected, { atol: 1e-3 });
    f.dispose();
  });

  test("non-square 128x64 @ 64x256 matches np.matmul", () => {
    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(128 * 64).astype(DType.Float32);
    using A = A_flat.reshape([128, 64]);
    using B_flat = np.arange(64 * 256).astype(DType.Float32);
    using B = B_flat.reshape([64, 256]);
    using expected = np.matmul(A, B);
    using result = f(A, B);
    expect(result).toBeAllclose(expected, { atol: 1 });
    f.dispose();
  });

  test("K not divisible by Bk — padding handles K=48 with Bk=16", () => {
    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(16 * 48).astype(DType.Float32);
    using A = A_flat.reshape([16, 48]);
    using B_flat = np.arange(48 * 16).astype(DType.Float32);
    using B = B_flat.reshape([48, 16]);
    using expected = np.matmul(A, B);
    using result = f(A, B);
    expect(result).toBeAllclose(expected, { atol: 1 });
    f.dispose();
  });

  test("K not divisible — K=50 with Bk=16 pads to K=64", () => {
    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(32 * 50).astype(DType.Float32);
    using A = A_flat.reshape([32, 50]);
    using B_flat = np.arange(50 * 32).astype(DType.Float32);
    using B = B_flat.reshape([50, 32]);
    using expected = np.matmul(A, B);
    using result = f(A, B);
    expect(result).toBeAllclose(expected, { atol: 1 });
    f.dispose();
  });

  test("256x256 matches np.matmul", () => {
    const f = jit((A: np.Array, B: np.Array) => lax.tiledMatmul(A, B));
    using A_flat = np.arange(256 * 256).astype(DType.Float32);
    using A = A_flat.reshape([256, 256]);
    using B = np.eye(256, { dtype: DType.Float32 });
    using expected = np.matmul(A, B);
    using result = f(A, B);
    expect(result).toBeAllclose(expected, { atol: 1e-2 });
    f.dispose();
  });

  test("grad(tiledMatmul) matches grad(np.matmul)", () => {
    using B = np.eye(16, { dtype: DType.Float32 });
    const f_tiled = (A: np.Array) => {
      using result = lax.tiledMatmul(A, B, { Br: 4, Bc: 4, Bk: 4 });
      return np.sum(result);
    };
    const f_ref = (A: np.Array) => {
      using result = np.matmul(A, B);
      return np.sum(result);
    };

    using A_flat = np.arange(256).astype(DType.Float32);
    using A = A_flat.reshape([16, 16]);
    using g_tiled = grad(f_tiled)(A);
    using g_ref = grad(f_ref)(A);
    expect(g_tiled).toBeAllclose(g_ref, { atol: 1e-3 });
  });
});

// ---------------------------------------------------------------------------
// T5b: WorkgroupAssociativeScan primitive
// ---------------------------------------------------------------------------
describe("lax.workgroupAssociativeScan", () => {
  // T5b.1: workgroupAssociativeScan(add, elems) inside block_map
  test("cumsum inside block_map via fused Kogge-Stone", () => {
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.workgroupAssociativeScan(
            (a: np.Array, b: np.Array) => np.add(a, b),
            block,
          ),
        xs,
        { blockShape: [8] },
      ),
    );
    // Input: [1,2,3,4, 5,6,7,8] → two blocks of 4
    using xs = np.array(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      { dtype: DType.Float32 },
    );
    // expected: cumsum within each block of size 8
    // block0: [1,3,6,10,15,21,28,36]
    // block1: [9,19,30,42,55,69,84,100]
    using result = f_jit(xs);
    expect(result).toBeAllclose([
      1, 3, 6, 10, 15, 21, 28, 36, 9, 19, 30, 42, 55, 69, 84, 100,
    ]);
    f_jit.dispose();
  });

  // T5b.2: workgroupAssociativeScan outside block_map — JIT fallback
  test("outside block_map runs via sequential fallback", () => {
    const f_jit = jit((xs: np.Array) =>
      lax.workgroupAssociativeScan(
        (a: np.Array, b: np.Array) => np.add(a, b),
        xs,
      ),
    );
    using xs = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // cumsum [1,2,3,4] → [1,3,6,10]
    expect(result).toBeAllclose([1, 3, 6, 10]);
    f_jit.dispose();
  });

  // T5b.3: eager impl matches associativeScan
  test("eager impl matches associativeScan", () => {
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    const fn = (a: np.Array, b: np.Array) => np.add(a, b);

    using expected = lax.associativeScan(fn, xs);
    using result = lax.workgroupAssociativeScan(fn, xs);
    expect(result).toBeAllclose(expected);
  });

  // T5b.4: non-power-of-2 block size
  test("non-power-of-2 block size", () => {
    // blockSize=6 (not power of 2) — tests masking in Kogge-Stone
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.workgroupAssociativeScan(
            (a: np.Array, b: np.Array) => np.add(a, b),
            block,
          ),
        xs,
        { blockShape: [6] },
      ),
    );
    // 12 elements → 2 blocks of 6
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], {
      dtype: DType.Float32,
    });
    using result = f_jit(xs);
    // block0: cumsum [1,2,3,4,5,6] → [1,3,6,10,15,21]
    // block1: cumsum [7,8,9,10,11,12] → [7,15,24,34,45,57]
    expect(result).toBeAllclose([1, 3, 6, 10, 15, 21, 7, 15, 24, 34, 45, 57]);
    f_jit.dispose();
  });

  // T5b.5: non-add operator (mul)
  test("cumulative product via mul operator", () => {
    const f_jit = jit((xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          lax.workgroupAssociativeScan(
            (a: np.Array, b: np.Array) => np.multiply(a, b),
            block,
          ),
        xs,
        { blockShape: [4] },
      ),
    );
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    using result = f_jit(xs);
    // block0: cumprod [1,2,3,4] → [1,2,6,24]
    // block1: cumprod [5,6,7,8] → [5,30,210,1680]
    expect(result).toBeAllclose([1, 2, 6, 24, 5, 30, 210, 1680]);
    f_jit.dispose();
  });
});

// ---------------------------------------------------------------------------
// T5c: WASM compiled block-loop tests
// ---------------------------------------------------------------------------

describe("lax.blockMap — WASM compiled block-loop", () => {
  // T5c.1: Elementwise body compiled into single WASM module
  test("T5c.1: elementwise body compiled into WASM loop", () => {
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

  // T5c.2: Reduction kernel inside body (sum + broadcast add)
  test("T5c.2: reduction body compiled in WASM loop", () => {
    // Body: sum(block) + block → tests reduction kernel + internal dependency.
    // Block0 [1,2,3,4] → sum=10, add(10,[1,2,3,4])=[11,12,13,14]
    // Block1 [5,6,7,8] → sum=26, add(26,[5,6,7,8])=[31,32,33,34]
    const f = (xs: np.Array) =>
      lax.blockMap((block: np.Array) => np.add(np.sum(block), block), xs, {
        blockShape: [4],
      });
    const f_jit = jit(f);
    using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([11, 12, 13, 14, 31, 32, 33, 34]);
    f_jit.dispose();
  });

  // T5c.3: jit(block_map(f, xs)) matches eager on WASM
  test("T5c.3: JIT matches eager on WASM", () => {
    const body = (block: np.Array) => {
      using two = np.array(2, { dtype: DType.Float32 });
      using one = np.array(1, { dtype: DType.Float32 });
      using scaled = np.multiply(block, two);
      return np.add(scaled, one);
    };
    const f = (xs: np.Array) => lax.blockMap(body, xs, { blockShape: [3] });

    // Eager
    using xs = np.array([10, 20, 30, 40, 50, 60], { dtype: DType.Float32 });
    using eager = f(xs);

    // JIT
    const f_jit = jit(f);
    using jitted = f_jit(xs);
    f_jit.dispose();

    // Both should produce [21, 41, 61, 81, 101, 121]
    expect(eager).toBeAllclose([21, 41, 61, 81, 101, 121]);
    expect(jitted).toBeAllclose([21, 41, 61, 81, 101, 121]);
  });

  // T5c.4: Block-loop with non-divisible N
  test("T5c.4: non-divisible N with boundary padding", () => {
    const f = (xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          np.multiply(block, np.array(2, { dtype: DType.Float32 })),
        xs,
        { blockShape: [4] },
      );
    const f_jit = jit(f);
    // N=5, blockShape=4 → 2 blocks. Last block has 1 real + 3 padded elements.
    using xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using result = f_jit(xs);
    expect(result).toBeAllclose([2, 4, 6, 8, 10]);
    f_jit.dispose();
  });
});

// ---------------------------------------------------------------------------
// Compiler Plan Phase 1A: Shader Quality Gates (C§8)
// ---------------------------------------------------------------------------
describe("shader quality gates", () => {
  const isWebGPU = defaultDevice() === "webgpu";

  /**
   * Capture all WGSL shader sources emitted during a callback.
   * Uses setDebug(2) which logs shaders via console.info.
   */
  function captureShaders(fn: () => void): string[] {
    const shaders: string[] = [];
    const origInfo = console.info;
    console.info = (...args: unknown[]) => {
      const msg = args.map(String).join(" ");
      if (msg.includes("=========== WebGPU shader ===========")) {
        // Extract the shader code after the header line
        const code = msg.replace("=========== WebGPU shader ===========\n", "");
        shaders.push(code);
      }
    };
    try {
      setDebug(2);
      fn();
    } finally {
      setDebug(0);
      console.info = origInfo;
    }
    return shaders;
  }

  test("P2a: tile-aligned tiledMatmul emits no select() in fused shader", () => {
    if (!isWebGPU) return;

    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    // 64x64 is perfectly divisible by 16 → no boundary masks → no select
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    // The block_map fused shader is the one with fori_loop.
    // It should contain no select() calls for tile-aligned inputs.
    const fusedShaders = shaders.filter(
      (s) => s.includes("workgroupBarrier") || s.includes("var<workgroup>"),
    );
    expect(fusedShaders.length).toBeGreaterThan(0);
    for (const shader of fusedShaders) {
      const selectCount = (shader.match(/\bselect\s*\(/g) || []).length;
      expect(selectCount, "fused shader should have no select() calls").toBe(0);
    }

    f.dispose();
  });

  test("P2a: non-aligned inputs still produce select() where needed", () => {
    if (!isWebGPU) return;

    // Non-divisible: 5 elements with blockShape=4 → boundary block needs select
    const f = jit((xs: np.Array) =>
      lax.blockMap(
        (block: np.Array) =>
          np.multiply(block, np.array(2, { dtype: DType.Float32 })),
        xs,
        { blockShape: [4] },
      ),
    );
    using xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(xs);
    });

    const fusedShaders = shaders.filter(
      (s) => s.includes("var<workgroup>") || s.includes("workgroupBarrier"),
    );
    // For non-aligned, the fused shader should still have select for boundary
    if (fusedShaders.length > 0) {
      const hasSelect = fusedShaders.some((s) => s.includes("select("));
      expect(hasSelect, "boundary blocks should still use select()").toBe(true);
    }

    f.dispose();
  });

  test("P0a: scalar intermediates promoted to let bindings (no tidx < 1u guard)", () => {
    if (!isWebGPU) return;

    // Two reductions whose results are combined by a size-1 non-reduction
    // kernel. That kernel cannot be a reduction epilogue (depends on both),
    // so without scalar promotion it would get an `if (tidx < 1u)` shmem
    // write + barrier. With promotion it becomes a `let` binding.
    const body = (block: np.Array) => {
      using m = np.max(block);
      using s = np.sum(block);
      using ratio = np.divide(m, s);
      return np.multiply(block, ratio);
    };
    const f = jit((xs: np.Array) =>
      lax.blockMap(body, xs, { blockShape: [4] }),
    );
    using xs = np.array([1, 2, 3, 4, 2, 2, 2, 2], { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(xs);
    });
    // Verify correctness
    using result = f(xs);
    // Block 0: max=4, sum=10, ratio=0.4, [0.4, 0.8, 1.2, 1.6]
    // Block 1: max=2, sum=8, ratio=0.25, [0.5, 0.5, 0.5, 0.5]
    expect(result).toBeAllclose([0.4, 0.8, 1.2, 1.6, 0.5, 0.5, 0.5, 0.5]);

    const fusedShaders = shaders.filter(
      (s) => s.includes("var<workgroup>") || s.includes("workgroupBarrier"),
    );
    expect(fusedShaders.length).toBeGreaterThan(0);
    for (const shader of fusedShaders) {
      // The size-1 kernel should be promoted to a let binding, not a guarded
      // shmem write. "tidx < 1u" is the signature of an un-promoted size-1 step.
      expect(
        shader,
        "promoted scalar should use let binding, not tidx < 1u guard",
      ).not.toContain("tidx < 1u");
    }

    f.dispose();
  });

  test("P1: tiledMatmul fori_loop has exactly 2 barriers per iteration", () => {
    if (!isWebGPU) return;

    // Tiled matmul K-loop: load A, load B, compute.
    // Optimal barriers: 1 after load phase, 1 wrap-around after compute.
    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    const fusedShaders = shaders.filter(
      (s) => s.includes("var<workgroup>") || s.includes("workgroupBarrier"),
    );
    expect(fusedShaders.length).toBeGreaterThan(0);

    for (const shader of fusedShaders) {
      // Find the for loop body and count barriers within it
      const forMatch = shader.match(
        /for\s*\(var\s+fl\d+_i.*?\{([\s\S]*?)\n\s*\}\s*\n/,
      );
      if (!forMatch) continue;
      const forBody = forMatch[1];
      const barrierCount = (forBody.match(/workgroupBarrier\(\)/g) || [])
        .length;
      // Exactly 2: after load phase + wrap-around after compute
      expect(
        barrierCount,
        `fori_loop should have exactly 2 barriers, got ${barrierCount}`,
      ).toBe(2);
    }

    f.dispose();
  });

  test("P2b: tiledMatmul uses unchecked dynamic slice (no min/max clamping)", () => {
    if (!isWebGPU) return;

    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    const fusedShaders = shaders.filter(
      (s) => s.includes("var<workgroup>") || s.includes("workgroupBarrier"),
    );
    expect(fusedShaders.length).toBeGreaterThan(0);

    for (const shader of fusedShaders) {
      // Extract the for loop body
      const forMatch = shader.match(
        /for\s*\(var\s+fl\d+_i.*?\{([\s\S]*?)\n\s*\}\s*\n/,
      );
      if (!forMatch) continue;
      const forBody = forMatch[1];
      // UncheckedDynamicSlice should produce no min() or max() clamping in
      // the index computation. Regular DynamicSlice would have both.
      const minCount = (forBody.match(/\bmin\s*\(/g) || []).length;
      const maxCount = (forBody.match(/\bmax\s*\(/g) || []).length;
      expect(
        minCount,
        "unchecked dynamic slice should produce no min() clamping",
      ).toBe(0);
      expect(
        maxCount,
        "unchecked dynamic slice should produce no max() clamping",
      ).toBe(0);
    }

    f.dispose();
  });

  test("P2c: non-aligned tiledMatmul produces no select() (padConcrete)", () => {
    if (!isWebGPU) return;

    // 64x60 @ 60x64: K=60 not divisible by 16, triggers padding to K=64
    // Ratio 4096/3840=1.07 < 1.25, elements 3840 > 1024 → padConcrete fires
    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(64 * 60).astype(DType.Float32);
    using A = A_flat.reshape([64, 60]);
    using B_flat = np.arange(60 * 64).astype(DType.Float32);
    using B = B_flat.reshape([60, 64]);

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    const fusedShaders = shaders.filter(
      (s) => s.includes("workgroupBarrier") || s.includes("var<workgroup>"),
    );
    expect(fusedShaders.length).toBeGreaterThan(0);
    for (const shader of fusedShaders) {
      const selectCount = (shader.match(/\bselect\s*\(/g) || []).length;
      expect(
        selectCount,
        "padConcrete should eliminate all select() in fused shader",
      ).toBe(0);
    }

    f.dispose();
  });

  test("P2c: non-aligned tiledMatmul correctness (64x60 @ 60x64)", () => {
    using A_flat = np.arange(64 * 60).astype(DType.Float32);
    using A = A_flat.reshape([64, 60]);
    using B_flat = np.arange(60 * 64).astype(DType.Float32);
    using B = B_flat.reshape([60, 64]);
    using expected = np.matmul(A, B);
    using result = lax.tiledMatmul(A, B);
    expect(result).toBeAllclose(expected, { atol: 1 });
  });

  test("P2c: tiny matrices fall back to mask-based pad (heuristic rejects)", () => {
    if (!isWebGPU) return;

    // 4x5 @ 5x4: small enough that heuristic rejects concrete padding
    // (4*5=20 elements < 1024 threshold)
    const f = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 4, Bc: 4, Bk: 4 }),
    );
    using aFlat = np.arange(20).astype(DType.Float32);
    using A = aFlat.reshape([4, 5]);
    using bFlat = np.arange(20).astype(DType.Float32);
    using B = bFlat.reshape([5, 4]);

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    // Small matrix → heuristic rejects padConcrete → mask-based pad → select present
    const fusedShaders = shaders.filter(
      (s) => s.includes("workgroupBarrier") || s.includes("var<workgroup>"),
    );
    if (fusedShaders.length > 0) {
      const hasSelect = fusedShaders.some((s) => s.includes("select("));
      expect(
        hasSelect,
        "tiny matrices should still use mask-based pad with select()",
      ).toBe(true);
    }

    // But correctness must still hold
    using expected = np.matmul(A, B);
    using result = f(A, B);
    expect(result).toBeAllclose(expected, { atol: 1e-3 });

    f.dispose();
  });

  test("P2c: padConcrete memory — padded buffers freed after matmul", () => {
    // Verify no leak: run non-aligned tiledMatmul inside jit and ensure
    // leak checker (from test setup) does not flag extra live arrays.
    const f = jit((A: np.Array, B: np.Array) => lax.tiledMatmul(A, B));
    using A_flat = np.arange(48 * 50).astype(DType.Float32);
    using A = A_flat.reshape([48, 50]);
    using B_flat = np.arange(50 * 48).astype(DType.Float32);
    using B = B_flat.reshape([50, 48]);
    using result = f(A, B);
    using expected = np.matmul(A, B);
    expect(result).toBeAllclose(expected, { atol: 1 });
    f.dispose();
  });
});
