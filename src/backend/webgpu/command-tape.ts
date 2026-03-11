// Command tape types and eligibility check for WebGPU dispatch acceleration.
//
// A command tape pre-compiles a JitProgram's dispatch sequence into a flat
// representation with pre-resolved pipelines, pre-computed buffer indices,
// and pre-built uniform bind groups. This eliminates per-step JS overhead
// (scope lookups, array allocation, refcounting, pipeline cache lookups)
// while producing a tight command-encoding loop over a flat GPUBuffer[] table.
//
// Ops are stored in original step order so that frees return buffers to the
// pool *before* subsequent mallocs, reducing peak VRAM compared to the
// bulk-malloc-then-bulk-free approach.

import { Kernel } from "../../alu";
import type { JitStep } from "../../frontend/jit";
import { isSymbolicSize } from "../../shape";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Pre-resolved representation of a single shader dispatch (one per ShaderDispatch). */
export interface TapeDispatch {
  /** Pre-compiled GPU pipeline. */
  pipeline: GPUComputePipeline;
  /** Bind group layout for storage bindings (@group(0)). */
  bindGroupLayout: GPUBindGroupLayout;
  /** Indices into the flat buffer table for inputs. */
  inputIdxs: number[];
  /** Indices into the flat buffer table for outputs. */
  outputIdxs: number[];
  /** Per-pass grid dimensions. */
  passes: { grid: [number, number] }[];
  /** Pre-built uniform bind group (@group(1)), null if no uniforms. */
  uniformBindGroup: GPUBindGroup | null;
  /** Uniform buffer alignment (for dynamic offset calculation). */
  uniformAlignment: number;
}

/** Pre-resolved buffer allocation. */
export interface TapeMalloc {
  /** Index in the flat buffer table. */
  tableIdx: number;
  /** Buffer size in bytes (padded to 4-byte alignment). */
  paddedSize: number;
  /** Original (unpadded) byte size for output slot creation. */
  originalSize: number;
  /** Pre-filled constant data (for O2 scalar promotion). Null if no data. */
  initialData: Uint8Array<ArrayBuffer> | null;
}

/** A single tape operation in execution order. */
export type TapeOp =
  | { type: "malloc"; malloc: TapeMalloc }
  | { type: "free"; tableIdx: number }
  | { type: "recycle"; fromIdx: number; toIdx: number }
  | { type: "dispatch"; dispatch: TapeDispatch };

// ---------------------------------------------------------------------------
// Arena layout (O9a: slab sub-allocation)
// ---------------------------------------------------------------------------

/** Per-entry arena placement: offset within the slab + padded size. */
export interface ArenaEntry {
  /** 256-byte-aligned offset within the slab buffer. */
  offset: number;
  /** 4-byte-aligned allocation size (for bind group `size` parameter). */
  paddedSize: number;
}

/** Pre-computed slab layout for a tape's intermediates. */
export interface ArenaLayout {
  /** Total slab size in bytes (all entries 256-byte-aligned). */
  slabSize: number;
  /** Table index → arena entry for each eligible malloc. */
  entries: Map<number, ArenaEntry>;
  /** Initial data writes to the slab (for O2 scalar promotion constants). */
  initialDataWrites: { offset: number; data: Uint8Array<ArrayBuffer> }[];
}

/**
 * Compute offline arena layout by simulating the malloc/free pattern.
 *
 * Every non-zero-size malloc gets assigned a 256-byte-aligned offset within a
 * single slab buffer.  Freed regions are reclaimed by later mallocs (best-fit).
 * Recycles inherit their source entry's offset.  Returns null when there are
 * no arena-eligible allocations.
 */
export function computeArenaLayout(ops: TapeOp[]): ArenaLayout | null {
  const ALIGNMENT = 256;
  const entries = new Map<number, ArenaEntry>();
  const initialDataWrites: { offset: number; data: Uint8Array<ArrayBuffer> }[] =
    [];

  // Free list for region reuse (best-fit allocation).
  const freeList: { offset: number; size: number }[] = [];
  let slabSize = 0;

  for (const op of ops) {
    if (op.type === "malloc") {
      const m = op.malloc;
      if (m.paddedSize === 0) continue; // handled by reusable ZSB

      const alignedSize = Math.ceil(m.paddedSize / ALIGNMENT) * ALIGNMENT;

      // Best-fit search
      let bestIdx = -1;
      let bestWaste = Infinity;
      for (let i = 0; i < freeList.length; i++) {
        const waste = freeList[i].size - alignedSize;
        if (waste >= 0 && waste < bestWaste) {
          bestIdx = i;
          bestWaste = waste;
          if (waste === 0) break;
        }
      }

      let offset: number;
      if (bestIdx >= 0) {
        const region = freeList[bestIdx];
        offset = region.offset;
        if (region.size === alignedSize) {
          freeList.splice(bestIdx, 1);
        } else {
          region.offset += alignedSize;
          region.size -= alignedSize;
        }
      } else {
        offset = slabSize;
        slabSize += alignedSize;
      }

      entries.set(m.tableIdx, { offset, paddedSize: m.paddedSize });
      if (m.initialData) {
        initialDataWrites.push({ offset, data: m.initialData });
      }
    } else if (op.type === "free") {
      const entry = entries.get(op.tableIdx);
      if (entry) {
        const alignedSize = Math.ceil(entry.paddedSize / ALIGNMENT) * ALIGNMENT;
        freeList.push({ offset: entry.offset, size: alignedSize });
      }
    } else if (op.type === "recycle") {
      const fromEntry = entries.get(op.fromIdx);
      if (fromEntry) {
        entries.set(op.toIdx, fromEntry);
      }
    }
  }

  return slabSize > 0 ? { slabSize, entries, initialDataWrites } : null;
}

/** A pre-compiled dispatch sequence for a kernel-only JitProgram. */
export interface WebGPUCommandTape {
  /** Operations in original step order (malloc/free/recycle/dispatch). */
  ops: TapeOp[];
  /** Number of entries in the flat buffer table. */
  tableSize: number;
  /** Mapping: external input position → table index. */
  inputTableIdxs: number[];
  /** Mapping: external output position → table index. */
  outputTableIdxs: number[];
  /** Table indices that own allocated buffers (for cleanup on error). */
  allocatedIdxs: number[];
  /** Uniform buffers owned by this tape (kept alive for bind group references). */
  uniformBuffers: GPUBuffer[];
  /** Slab sub-allocation layout, or null when no intermediates exist. */
  arena: ArenaLayout | null;
}

// ---------------------------------------------------------------------------
// Eligibility check
// ---------------------------------------------------------------------------

/**
 * Check whether a JitProgram's steps can be compiled to a WebGPU command tape.
 *
 * Returns true if all steps are supported: execute (Kernel or Routine),
 * malloc (concrete size), free, and recycle. Programs with scan, DUS,
 * scatter_add, block_map, or other complex steps fall back to step-by-step.
 */
export function canCompileToCommandTape(steps: JitStep[]): boolean {
  for (const step of steps) {
    switch (step.type) {
      case "malloc":
        if (isSymbolicSize(step.size)) return false;
        break;
      case "free":
      case "recycle":
        break;
      case "execute":
        // Reject symbolic kernels — need runtime-resolved grid + uniform
        if (step.source instanceof Kernel && step.source.needsDynamicParams)
          return false;
        break;
      case "incref":
      case "scan":
      case "dus":
      case "scatter_add":
      case "reverse":
      case "assoc_scan":
      case "block_map":
      case "fori_loop":
      case "workgroup_assoc_scan":
        return false;
      default:
        return false;
    }
  }
  return true;
}
