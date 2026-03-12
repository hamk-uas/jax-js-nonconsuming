/**
 * Tests for O9c constants slab, O9a-v2 conflict-graph coloring,
 * and O8c DUS/scatter_add/reverse tape eligibility.
 */
import { describe, expect, it } from "vitest";

import { DType, grad, jit, nn, numpy as np } from "../src";
import {
  buildConflictGraphAndColor,
  canCompileToCommandTape,
} from "../src/backend/webgpu/command-tape";
import { SymbolicSize, SymDim } from "../src/dim";
import type { JitStep } from "../src/frontend/jit";

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

// ---------------------------------------------------------------------------
// O8c: DUS / scatter_add / reverse tape eligibility
// ---------------------------------------------------------------------------

describe("O8c canCompileToCommandTape", () => {
  it("accepts DUS with concrete 4-byte-aligned sizes", () => {
    const steps: JitStep[] = [
      {
        type: "malloc",
        size: 16,
        output: 2,
      },
      {
        type: "dus",
        dst: 0,
        src: 1,
        output: 2,
        dstSizeBytes: 16,
        offsetBytes: 4,
        sliceBytes: 4,
        outerFibers: 1,
        srcFiberBytes: 4,
        dstFiberBytes: 16,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(true);
  });

  it("rejects DUS with symbolic sliceBytes", () => {
    const steps: JitStep[] = [
      {
        type: "dus",
        dst: 0,
        src: 1,
        output: 2,
        dstSizeBytes: 16,
        offsetBytes: 0,
        sliceBytes: new SymbolicSize(4, ["T"]),
        outerFibers: 1,
        srcFiberBytes: 4,
        dstFiberBytes: 16,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(false);
  });

  it("rejects DUS with non-4-byte-aligned offsetBytes", () => {
    const steps: JitStep[] = [
      {
        type: "dus",
        dst: 0,
        src: 1,
        output: 2,
        dstSizeBytes: 16,
        offsetBytes: 2, // f16: 2-byte aligned
        sliceBytes: 2,
        outerFibers: 1,
        srcFiberBytes: 2,
        dstFiberBytes: 16,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(false);
  });

  it("accepts scatter_add with f32 dtype", () => {
    const steps: JitStep[] = [
      {
        type: "malloc",
        size: 20,
        output: 3,
      },
      {
        type: "scatter_add",
        target: 0,
        indices: 1,
        updates: 2,
        output: 3,
        axis: 0,
        targetShape: [5],
        updatesLen: 2,
        dtype: DType.Float32,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(true);
  });

  it("rejects scatter_add with f64 dtype", () => {
    const steps: JitStep[] = [
      {
        type: "scatter_add",
        target: 0,
        indices: 1,
        updates: 2,
        output: 3,
        axis: 0,
        targetShape: [5],
        updatesLen: 2,
        dtype: DType.Float64,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(false);
  });

  it("accepts reverse with concrete axis size and 4-byte aligned inner", () => {
    const steps: JitStep[] = [
      {
        type: "malloc",
        size: 16,
        output: 1,
      },
      {
        type: "reverse",
        input: 0,
        output: 1,
        axis: 0,
        axisSize: 4,
        innerBytes: 4,
        totalBytes: 16,
        dtype: DType.Float32,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(true);
  });

  it("rejects reverse with symbolic axisSize", () => {
    const steps: JitStep[] = [
      {
        type: "reverse",
        input: 0,
        output: 1,
        axis: 0,
        axisSize: new SymDim("T"),
        innerBytes: 4,
        totalBytes: new SymbolicSize(4, ["T"]),
        dtype: DType.Float32,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(false);
  });

  it("rejects reverse with non-4-byte-aligned innerBytes", () => {
    const steps: JitStep[] = [
      {
        type: "reverse",
        input: 0,
        output: 1,
        axis: 0,
        axisSize: 8,
        innerBytes: 2, // f16 elements
        totalBytes: 16,
        dtype: DType.Float16,
      },
    ];
    expect(canCompileToCommandTape(steps)).toBe(false);
  });
});

describe("O8c conflict graph with scatter_add", () => {
  it("scatter_add output conflicts with indices and updates", () => {
    const tape = {
      ops: [
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 2,
            paddedSize: 20,
            originalSize: 20,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 3,
            paddedSize: 8,
            originalSize: 8,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "malloc" as const,
          malloc: {
            tableIdx: 4,
            paddedSize: 8,
            originalSize: 8,
            slabAllocated: false,
            arenaAllocated: false,
          },
        },
        {
          type: "scatter_add" as const,
          scatterAdd: {
            targetIdx: 0,
            indicesIdx: 3,
            updatesIdx: 4,
            outIdx: 2,
            targetBytes: 20,
            pipeline: null!,
            bindGroupLayout: null!,
            grid: [1, 1] as [number, number],
          },
        },
      ],
      tableSize: 5,
      inputTableIdxs: [0],
      outputTableIdxs: [1],
      allocatedIdxs: [2, 3, 4],
      uniformBuffers: [],
      constSlab: null,
      arenaSlabs: null,
    };
    const result = buildConflictGraphAndColor(tape as any);
    // outIdx=2 conflicts with indicesIdx=3 and updatesIdx=4
    // → 2, 3, 4 should NOT all share the same color
    expect(result.colors[2]).not.toBe(-1);
    expect(result.colors[3]).not.toBe(-1);
    expect(result.colors[4]).not.toBe(-1);
    // 2 should differ from 3 and 4 (it conflicts with both)
    expect(result.colors[2]).not.toBe(result.colors[3]);
    expect(result.colors[2]).not.toBe(result.colors[4]);
    // 3 and 4 can share (they don't conflict with each other — both read-only)
    expect(result.colors[3]).toBe(result.colors[4]);
  });
});
