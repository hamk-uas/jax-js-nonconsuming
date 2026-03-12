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
  const { buildConflictGraphAndColor } = await import(
    "../src/backend/webgpu/command-tape"
  );

  it("colors independent indices with same color", () => {
    // Two dispatches that don't share any indices → no conflicts → 1 color
    const tape = {
      ops: [
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 2,
            paddedSize: 16,
            originalSize: 12,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "dispatch" as const,
          dispatch: { inputIdxs: [0], outputIdxs: [2], pipeline: null! },
        },
        { type: "free" as const, tableIdx: 2 },
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 3,
            paddedSize: 16,
            originalSize: 12,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "dispatch" as const,
          dispatch: { inputIdxs: [0], outputIdxs: [3], pipeline: null! },
        },
        { type: "free" as const, tableIdx: 3 },
      ],
      tableSize: 4,
      inputTableIdxs: [0],
      outputTableIdxs: [1],
      allocatedIdxs: [2, 3],
      uniformBuffers: [],
      constSlab: null,
      arenaSlabs: null,
    };
    const result = buildConflictGraphAndColor(tape as any);
    // Indices 2 and 3 have no conflict → should share color 0
    expect(result.numColors).toBe(1);
    expect(result.colorGroups[0]).toEqual(expect.arrayContaining([2, 3]));
  });

  it("assigns different colors to conflicting indices", () => {
    // One dispatch reads idx=2 and writes idx=3 → they conflict
    const tape = {
      ops: [
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 2,
            paddedSize: 16,
            originalSize: 12,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 3,
            paddedSize: 16,
            originalSize: 12,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "dispatch" as const,
          dispatch: { inputIdxs: [2], outputIdxs: [3], pipeline: null! },
        },
      ],
      tableSize: 4,
      inputTableIdxs: [0],
      outputTableIdxs: [1],
      allocatedIdxs: [2, 3],
      uniformBuffers: [],
      constSlab: null,
      arenaSlabs: null,
    };
    const result = buildConflictGraphAndColor(tape as any);
    expect(result.numColors).toBe(2);
    expect(result.colors[2]).not.toBe(result.colors[3]);
  });

  it("excludes recycle participants from coloring", () => {
    const tape = {
      ops: [
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 2,
            paddedSize: 16,
            originalSize: 12,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "dispatch" as const,
          dispatch: { inputIdxs: [0], outputIdxs: [2], pipeline: null! },
        },
        { type: "recycle" as const, fromIdx: 2, toIdx: 3 },
      ],
      tableSize: 4,
      inputTableIdxs: [0],
      outputTableIdxs: [1],
      allocatedIdxs: [2, 3],
      uniformBuffers: [],
      constSlab: null,
      arenaSlabs: null,
    };
    const result = buildConflictGraphAndColor(tape as any);
    // Both 2 and 3 are recycle participants → excluded → 0 colors
    expect(result.colors[2]).toBe(-1);
    expect(result.colors[3]).toBe(-1);
  });
});
