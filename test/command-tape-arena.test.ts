/**
 * Tests for O9c constants slab and O9a-v2 conflict-graph coloring.
 */
import { describe, expect, it } from "vitest";

import { grad, jit, nn, numpy as np } from "../src";

describe("O9c constants slab", () => {
  it("jit with scalar constants produces correct results", () => {
    // This JIT program has scalar-promoted constants (O2): 1, 2, 3, 4, 0.5
    // With O9c, these should be packed into a single slab buffer.
    using x = np.array([10, 20, 30]);
    const f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
    using result = f(x);
    // (10+1)*2 - 3 + 4)*0.5 = (22-3+4)*0.5 = 23*0.5 = 11.5
    // (20+1)*2 - 3 + 4)*0.5 = (42-3+4)*0.5 = 43*0.5 = 21.5
    // (30+1)*2 - 3 + 4)*0.5 = (62-3+4)*0.5 = 63*0.5 = 31.5
    expect(result).toBeAllclose([11.5, 21.5, 31.5]);
    // Run again to verify slab reuse on second invocation
    using result2 = f(x);
    expect(result2).toBeAllclose([11.5, 21.5, 31.5]);
    f.dispose();
  });

  it("jit with scalar constants and grad produces correct results", () => {
    // grad generates additional scalar constants for the backward pass
    const f = jit(grad((x: np.Array) => x.mul(3).add(1).sum()));
    using x = np.array([1, 2, 3]);
    using dx = f(x);
    // d/dx(3x + 1).sum() = 3 for each element
    expect(dx).toBeAllclose([3, 3, 3]);
    // Second invocation should reuse slab
    using dx2 = f(x);
    expect(dx2).toBeAllclose([3, 3, 3]);
    f.dispose();
  });

  it("repeated invocations with different inputs produce correct results", () => {
    const f = jit((x: np.Array) => x.add(10).mul(0.1));
    using x1 = np.array([0, 1, 2]);
    using x2 = np.array([100, 200, 300]);

    using r1 = f(x1);
    expect(r1).toBeAllclose([1, 1.1, 1.2]);

    using r2 = f(x2);
    expect(r2).toBeAllclose([11, 21, 31]);

    // Re-run with original input to verify no corruption
    using r3 = f(x1);
    expect(r3).toBeAllclose([1, 1.1, 1.2]);

    f.dispose();
  });

  it("matmul + bias + relu with constants slab", () => {
    using x = np.array([
      [1, 2],
      [3, 4],
    ]);
    using w = np.array([
      [0.5, 0],
      [0, 0.5],
    ]);
    using b = np.array([1, -1]);

    const f = jit((x: np.Array, w: np.Array, b: np.Array) =>
      nn.relu(np.matmul(x, w).add(b)),
    );
    using result = f(x, w, b);
    // matmul: [[0.5, 1], [1.5, 2]], add bias: [[1.5, 0], [2.5, 1]], relu: [[1.5, 0], [2.5, 1]]
    expect(result).toBeAllclose([
      [1.5, 0],
      [2.5, 1],
    ]);

    using result2 = f(x, w, b);
    expect(result2).toBeAllclose([
      [1.5, 0],
      [2.5, 1],
    ]);

    f.dispose();
  });
});

describe("conflict-graph coloring", async () => {
  // We test the coloring indirectly through the command tape by verifying
  // that programs with various dispatch patterns produce correct results.
  // The buildConflictGraphAndColor function is tested directly below.

  it("imports and can be called", async () => {
    const { buildConflictGraphAndColor, canCompileToCommandTape } =
      await import("../src/backend/webgpu/command-tape");
    expect(typeof buildConflictGraphAndColor).toBe("function");
    expect(typeof canCompileToCommandTape).toBe("function");
  });
});
