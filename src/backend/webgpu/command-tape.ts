// Command tape types and eligibility check for WebGPU dispatch acceleration.
//
// A command tape pre-compiles a JitProgram's dispatch sequence into a flat
// representation with pre-resolved pipelines, pre-computed buffer indices,
// and pre-built uniform bind groups. This eliminates per-step JS overhead
// (scope lookups, array allocation, refcounting, pipeline cache lookups)
// while producing a tight command-encoding loop over a flat GPUBuffer[] table.

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

/** A pre-compiled dispatch sequence for a kernel-only JitProgram. */
export interface WebGPUCommandTape {
  /** One entry per shader dispatch (may be >1 per execute step for routines). */
  dispatches: TapeDispatch[];
  /** Bulk malloc plan in step order. */
  mallocs: TapeMalloc[];
  /** Recycle plan: [fromIdx, toIdx][] — pointer copy in flat table. */
  recycles: [number, number][];
  /** Table indices to free after all dispatches complete. */
  frees: number[];
  /** Number of entries in the flat buffer table. */
  tableSize: number;
  /** Mapping: external input position → table index. */
  inputTableIdxs: number[];
  /** Mapping: external output position → table index. */
  outputTableIdxs: number[];
  /** Uniform buffers owned by this tape (kept alive for bind group references). */
  uniformBuffers: GPUBuffer[];
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
