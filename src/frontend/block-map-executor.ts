/**
 * @file BlockMap JIT executor — runs the block_map body program over a grid
 * of blocks, slicing inputs and assembling outputs.
 *
 * The fallback path iterates in JS, calling bodyProgram.execute() per block.
 * The fused path (WebGPU only) compiles the body into a single WGSL shader
 * where each workgroup processes one block using shared memory.
 */

import { byteWidth } from "../alu";
import type { Backend, Slot } from "../backend";
import { DEBUG } from "../utils";
import type { PendingExecute } from "./array";
import type { Jaxpr } from "./jaxpr";
import type { JitProgram } from "./jit";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface ExecuteBlockMapParams {
  backend: Backend;
  bodyProgram: JitProgram;
  bodyJaxpr: Jaxpr;
  blockShape: number[];
  inAxes: (number | null)[][];
  outAxes: (number | null)[][];
  numConsts: number;
  numInputs: number;
  gridShape: number[];
  /** Original (untrimmed) input shapes. */
  inputShapes: number[][];
  /** Original (untrimmed) output shapes. */
  outputShapes: number[][];
  constSlots: Slot[];
  inputSlots: Slot[];
  /** Preallocated output slots — sized to match outputShapes. */
  outputSlots: Slot[];
}

export interface ExecuteBlockMapResult {
  outputs: Slot[];
  pending: PendingExecute[];
}

/**
 * Execute a block_map. Tries the fused WebGPU path first (single dispatch,
 * shared-memory intermediates), then falls back to the JS loop executor.
 */
export function executeBlockMap(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult {
  // Try fused WebGPU shader path
  if (params.backend.type === "webgpu") {
    const result = tryExecuteBlockMapFused(params);
    if (result) return result;
  }
  return executeBlockMapFallback(params);
}

// ---------------------------------------------------------------------------
// Fused WebGPU path: single-dispatch shared-memory shader
// ---------------------------------------------------------------------------

function tryExecuteBlockMapFused(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult | null {
  // Lazy import to avoid circular dependency at module load time.
  // The scan-executor uses the same pattern (type import + cast).
  const webgpuBackend =
    params.backend as import("../backend/webgpu").WebGPUBackend;

  const exe = webgpuBackend.prepareBlockMapFused({
    bodyProgram: params.bodyProgram,
    blockShape: params.blockShape,
    gridShape: params.gridShape,
    inAxes: params.inAxes,
    outAxes: params.outAxes,
    numConsts: params.numConsts,
    numInputs: params.numInputs,
    inputShapes: params.inputShapes,
    outputShapes: params.outputShapes,
  });

  if (!exe) return null;

  if (DEBUG >= 1) {
    console.info("block_map: using fused WebGPU shader path");
  }

  // Dispatch: inputs = [consts, blockInputs], outputs = [outputSlots]
  const inputs = [...params.constSlots, ...params.inputSlots];
  webgpuBackend.dispatchBlockMapFused(exe, inputs, params.outputSlots);

  return { outputs: params.outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Fallback: JS loop calling bodyProgram.execute() per block
// ---------------------------------------------------------------------------

function executeBlockMapFallback(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult {
  const {
    backend,
    bodyProgram,
    bodyJaxpr,
    blockShape,
    inAxes,
    outAxes,
    numConsts: _numConsts,
    numInputs,
    gridShape,
    inputShapes,
    outputShapes,
    constSlots,
    inputSlots,
    outputSlots,
  } = params;

  const gridRank = blockShape.length;
  const numBlocks = gridShape.reduce((a, b) => a * b, 1);
  const numOutputs = bodyJaxpr.outs.length;
  const pending: PendingExecute[] = [];

  const bodyInputAvals = bodyJaxpr.inBinders
    .slice(_numConsts)
    .map((v) => v.aval);
  const bodyOutAvals = bodyJaxpr.outs.map((v) => v.aval);

  const BATCH_SIZE = 64;
  const useBatching = backend.beginBatch != null && numBlocks > 1;
  if (useBatching) backend.beginBatch!();

  try {
    for (let flatIdx = 0; flatIdx < numBlocks; flatIdx++) {
      // Convert flat index to grid coordinates (row-major)
      const blockIdx: number[] = new Array(gridRank);
      let remaining = flatIdx;
      for (let g = gridRank - 1; g >= 0; g--) {
        blockIdx[g] = remaining % gridShape[g];
        remaining = Math.floor(remaining / gridShape[g]);
      }

      // Flush pending before each body invocation
      flushPending(pending, backend);

      // Determine if this is a boundary block (last block on any axis where
      // the input dimension is not evenly divisible by blockShape).
      const isBoundary = blockIdx.some((bi, g) => bi === gridShape[g] - 1);

      // Create block input slots by slicing from full inputs
      const blockInputSlots: Slot[] = [];
      const ownedSlices: Slot[] = [];

      for (let i = 0; i < numInputs; i++) {
        const axes = inAxes[i];
        const blockAval = bodyInputAvals[i];
        const nd = blockAval.shape.length;
        const elemBytes = byteWidth(blockAval.dtype);
        const blockBytes = blockAval.size * elemBytes;

        // Compute the slice region in the original input
        const sliceStarts: number[] = new Array(nd).fill(0);
        const sliceSizes: number[] = [...blockAval.shape] as number[];
        let needsPad = false;

        for (let g = 0; g < gridRank; g++) {
          if (axes[g] !== null) {
            const ax = axes[g]!;
            sliceStarts[ax] = blockIdx[g] * blockShape[g];
            // Clamp to available input data
            const available = inputShapes[i][ax] - sliceStarts[ax];
            if (available < blockShape[g]) {
              sliceSizes[ax] = Math.max(0, available);
              needsPad = true;
            }
          }
        }

        if (needsPad && isBoundary) {
          // For boundary blocks, allocate zero-filled and copy valid portion
          const zeros = new Uint8Array(blockBytes);
          const blockSlot = backend.malloc(blockBytes, zeros);

          // Copy valid portion of input into the block
          const validBytes = sliceSizes.reduce((a, b) => a * b, 1) * elemBytes;
          if (validBytes > 0) {
            copyBlock(
              backend,
              inputSlots[i],
              inputShapes[i],
              sliceStarts,
              sliceSizes,
              blockSlot,
              blockAval.shape as number[],
              new Array(nd).fill(0) as number[],
              elemBytes,
            );
          }

          blockInputSlots.push(blockSlot);
          ownedSlices.push(blockSlot);
        } else {
          // Full block — allocate and copy
          const blockSlot = backend.malloc(blockBytes);
          const contiguous = isSliceContiguous(
            sliceStarts,
            sliceSizes,
            inputShapes[i],
          );
          if (contiguous) {
            const byteOffset = computeLinearOffset(
              sliceStarts,
              inputShapes[i],
              elemBytes,
            );
            backend.copyBufferToBuffer(
              inputSlots[i],
              byteOffset,
              blockSlot,
              0,
              blockBytes,
            );
          } else {
            copyBlock(
              backend,
              inputSlots[i],
              inputShapes[i],
              sliceStarts,
              sliceSizes,
              blockSlot,
              blockAval.shape as number[],
              new Array(nd).fill(0) as number[],
              elemBytes,
            );
          }
          blockInputSlots.push(blockSlot);
          ownedSlices.push(blockSlot);
        }
      }

      // IncRef consts (body borrows them)
      for (const slot of constSlots) backend.incRef(slot);

      // Build body inputs: [consts, blockInputs]
      const bodyInputs = [...constSlots, ...blockInputSlots];

      // Execute body program
      const bodyResult = bodyProgram.execute(bodyInputs);
      pending.push(...bodyResult.pending);

      // Flush pending from body execution
      flushPending(pending, backend);

      // Release consts and block input slices
      for (const slot of constSlots) backend.decRef(slot);
      for (const slot of ownedSlices) backend.decRef(slot);

      // Copy body outputs into the preallocated output buffers
      for (let o = 0; o < numOutputs; o++) {
        const axes = outAxes[o];
        const blockAval = bodyOutAvals[o];
        const nd = blockAval.shape.length;
        const elemBytes = byteWidth(blockAval.dtype);

        // Compute the destination region in the output buffer
        const dstStarts: number[] = new Array(nd).fill(0);
        const copySizes: number[] = [...blockAval.shape] as number[];
        for (let g = 0; g < gridRank; g++) {
          if (axes[g] !== null) {
            const ax = axes[g]!;
            dstStarts[ax] = blockIdx[g] * blockShape[g];
            // Clamp copy size to output extent
            const remaining = outputShapes[o][ax] - dstStarts[ax];
            if (remaining < blockShape[g]) {
              copySizes[ax] = Math.max(0, remaining);
            }
          }
        }

        const totalCopyElems = copySizes.reduce((a, b) => a * b, 1);
        if (totalCopyElems > 0) {
          copyBlock(
            backend,
            bodyResult.outputs[o],
            blockAval.shape as number[],
            new Array(nd).fill(0) as number[], // source starts at 0
            copySizes,
            outputSlots[o],
            outputShapes[o],
            dstStarts,
            elemBytes,
          );
        }

        backend.decRef(bodyResult.outputs[o]);
      }

      // Periodic batch flush
      if (
        useBatching &&
        flatIdx < numBlocks - 1 &&
        (flatIdx + 1) % BATCH_SIZE === 0
      ) {
        backend.endBatch!();
        backend.beginBatch!();
      }
    }
  } finally {
    if (useBatching) backend.endBatch!();
  }

  flushPending(pending, backend);
  return { outputs: outputSlots, pending };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Check if a slice is contiguous in memory (can be copied in one go). */
function isSliceContiguous(
  starts: number[],
  sizes: number[],
  fullShape: number[],
): boolean {
  // A slice is contiguous if, from the first axis that doesn't take the
  // full extent, all subsequent axes take the full extent from offset 0.
  let foundPartial = false;
  for (let d = 0; d < starts.length; d++) {
    if (sizes[d] !== fullShape[d] || starts[d] !== 0) {
      if (foundPartial) return false;
      foundPartial = true;
    }
  }
  return true;
}

/** Compute linear byte offset for a multi-index into a row-major shape. */
function computeLinearOffset(
  starts: number[],
  fullShape: number[],
  elemBytes: number,
): number {
  const nd = starts.length;
  let offset = 0;
  let stride = elemBytes;
  for (let d = nd - 1; d >= 0; d--) {
    offset += starts[d] * stride;
    stride *= fullShape[d];
  }
  return offset;
}

/** Flush pending operations. */
function flushPending(pending: PendingExecute[], backend: Backend): void {
  if (pending.length === 0) return;
  backend.beginBatch?.();
  try {
    for (const p of pending) {
      p.prepareSync();
      p.submit();
    }
  } finally {
    backend.endBatch?.();
  }
  pending.length = 0;
}

/**
 * Copy a block between buffers with arbitrary source/dest offsets and shapes.
 * Handles both contiguous (single copy) and strided (row-by-row) cases.
 */
function copyBlock(
  backend: Backend,
  srcSlot: Slot,
  srcFullShape: number[],
  srcStarts: number[],
  copySizes: number[],
  dstSlot: Slot,
  dstFullShape: number[],
  dstStarts: number[],
  elemBytes: number,
): void {
  const nd = copySizes.length;

  // Check contiguity on both sides
  const srcContiguous = isSliceContiguous(srcStarts, copySizes, srcFullShape);
  const dstContiguous = isSliceContiguous(dstStarts, copySizes, dstFullShape);

  if (srcContiguous && dstContiguous) {
    // Single copy
    const srcOffset = computeLinearOffset(srcStarts, srcFullShape, elemBytes);
    const dstOffset = computeLinearOffset(dstStarts, dstFullShape, elemBytes);
    const totalBytes = copySizes.reduce((a, b) => a * b, 1) * elemBytes;
    backend.copyBufferToBuffer(
      srcSlot,
      srcOffset,
      dstSlot,
      dstOffset,
      totalBytes,
    );
    return;
  }

  // Strided: copy innermost-dimension rows
  const innermostBytes = copySizes[nd - 1] * elemBytes;
  const numRows = copySizes.slice(0, -1).reduce((a, b) => a * b, 1) || 1;

  const srcStrides = computeStrides(srcFullShape, elemBytes);
  const dstStrides = computeStrides(dstFullShape, elemBytes);

  const idx = new Array(Math.max(nd - 1, 0)).fill(0);
  for (let row = 0; row < numRows; row++) {
    let srcOffset = 0;
    let dstOffset = 0;
    for (let d = 0; d < nd - 1; d++) {
      srcOffset += (srcStarts[d] + idx[d]) * srcStrides[d];
      dstOffset += (dstStarts[d] + idx[d]) * dstStrides[d];
    }
    srcOffset += srcStarts[nd - 1] * elemBytes;
    dstOffset += dstStarts[nd - 1] * elemBytes;

    backend.copyBufferToBuffer(
      srcSlot,
      srcOffset,
      dstSlot,
      dstOffset,
      innermostBytes,
    );

    // Increment multi-index (row-major)
    for (let d = nd - 2; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < copySizes[d]) break;
      idx[d] = 0;
    }
  }
}

/** Compute byte strides for a shape (row-major, last dim is innermost). */
function computeStrides(shape: number[], elemBytes: number): number[] {
  const nd = shape.length;
  const strides = new Array(nd);
  strides[nd - 1] = elemBytes;
  for (let d = nd - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * shape[d + 1];
  }
  return strides;
}
