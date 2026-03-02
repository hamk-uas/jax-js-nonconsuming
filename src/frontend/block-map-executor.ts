/**
 * @file BlockMap JIT executor — runs the block_map body program over a grid
 * of blocks, slicing inputs and assembling outputs.
 *
 * The fallback path iterates in JS, calling bodyProgram.execute() per block.
 * The fused path (WebGPU only) compiles the body into a single WGSL shader
 * where each workgroup processes one block using shared memory.
 */

import { byteWidth, Kernel, Reduction } from "../alu";
import { type Backend, Executable, type Slot } from "../backend";
import type { BlockMapWasmParams, GeneralScanStep } from "../backend/wasm";
import { DEBUG } from "../utils";
import type { PendingExecute } from "./array";
import type { Jaxpr } from "./jaxpr";
import type { JitProgram, JitStep } from "./jit";

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
  /** Register tiling dimensions. */
  threadTile?: number[];
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
  // Try compiled WASM loop path
  if (params.backend.type === "wasm") {
    const result = tryExecuteBlockMapWasm(params);
    if (result) return result;
  }
  return executeBlockMapFallback(params);
}

// ---------------------------------------------------------------------------
// Fused WebGPU path: single-dispatch shared-memory shader
// ---------------------------------------------------------------------------

// Cache fused Executable by bodyProgram identity. The JitProgram is
// identity-stable (cached by jitCompile) and encodes the backend, so
// this avoids re-running the full WGSL codegen on every dispatch.
// WeakMap: when the JitProgram is GC'd (after clearCaches()), the entry dies.
const blockMapFusedCache = new WeakMap<JitProgram, Executable | null>();

function tryExecuteBlockMapFused(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult | null {
  // Lazy import to avoid circular dependency at module load time.
  // The scan-executor uses the same pattern (type import + cast).
  const webgpuBackend =
    params.backend as import("../backend/webgpu").WebGPUBackend;

  let exe = blockMapFusedCache.get(params.bodyProgram);
  if (exe === undefined) {
    exe =
      webgpuBackend.prepareBlockMapFused({
        bodyProgram: params.bodyProgram,
        blockShape: params.blockShape,
        gridShape: params.gridShape,
        inAxes: params.inAxes,
        outAxes: params.outAxes,
        numConsts: params.numConsts,
        numInputs: params.numInputs,
        inputShapes: params.inputShapes,
        outputShapes: params.outputShapes,
        threadTile: params.threadTile,
      }) ?? null;
    blockMapFusedCache.set(params.bodyProgram, exe);
  }

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
// Compiled WASM block-loop path
// ---------------------------------------------------------------------------

function tryExecuteBlockMapWasm(
  params: ExecuteBlockMapParams,
): ExecuteBlockMapResult | null {
  const wasmBackend = params.backend as import("../backend/wasm").WasmBackend;

  const wasmParams = buildBlockMapWasmParams(params);
  if (!wasmParams) return null;

  const exe = wasmBackend.prepareBlockMapWasm(wasmParams);
  if (!exe) return null;

  if (DEBUG >= 1) {
    console.info("block_map: using compiled WASM loop path");
  }

  wasmBackend.dispatchBlockMapWasm(
    exe,
    wasmParams,
    params.constSlots,
    params.inputSlots,
    params.outputSlots,
  );

  return { outputs: params.outputSlots, pending: [] };
}

/**
 * Build BlockMapWasmParams from the body program, or return null if the body
 * can't be compiled to WASM (routines, loops, non-contiguous slices, etc.).
 */
function buildBlockMapWasmParams(
  params: ExecuteBlockMapParams,
): BlockMapWasmParams | null {
  const {
    bodyProgram,
    bodyJaxpr,
    blockShape,
    inAxes,
    outAxes,
    numConsts,
    numInputs,
    gridShape,
    inputShapes,
    outputShapes,
  } = params;
  const gridRank = blockShape.length;
  const numOutputs = bodyJaxpr.outs.length;

  // Only support kernel-only bodies (no routines, nested scans, etc.)
  type ExecuteStep = Extract<JitStep, { type: "execute" }>;
  const executeSteps: ExecuteStep[] = [];
  for (const step of bodyProgram.steps) {
    if (step.type === "execute") {
      if (!(step.source instanceof Kernel)) {
        if (DEBUG >= 2)
          console.log("[wasm-block-map] rejected: non-kernel execute step");
        return null;
      }
      executeSteps.push(step as ExecuteStep);
    } else if (
      step.type === "malloc" ||
      step.type === "free" ||
      step.type === "recycle" ||
      step.type === "incref"
    ) {
      // These are fine — they're resource management
    } else {
      // scan, assoc_scan, block_map, fori_loop, etc. — can't inline
      if (DEBUG >= 2)
        console.log(
          `[wasm-block-map] rejected: unsupported step type ${step.type}`,
        );
      return null;
    }
  }

  if (executeSteps.length === 0) {
    if (DEBUG >= 2) console.log("[wasm-block-map] rejected: no execute steps");
    return null;
  }

  // Check contiguity: for each input with axis mapping, the slice must be contiguous.
  // A slice is contiguous if all mapped axes are "leading" — i.e., for mapped axis `ax`,
  // all dimensions d > ax match the full input shape (not sub-sliced by a different grid dim).
  for (let i = 0; i < numInputs; i++) {
    const axes = inAxes[i];
    const shape = inputShapes[i];
    // Collect mapped dims
    const mappedDims = new Set<number>();
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) mappedDims.add(axes[g]!);
    }
    // For contiguity: if dim `d` is mapped, then all dims d+1..nd-1 must also
    // be mapped (so the block is a contiguous chunk of memory).
    // Actually, for a single mapped dim, the slice is contiguous as long as
    // it slices a contiguous inner extent. This is true when the mapped dim
    // is the outermost non-trivial dim: all inner dims are fully included.
    // For simplicity: require that mapped dims form a prefix {0} or {0,1} etc.
    // OR that there's only one mapped dim and blockShape matches inner dims
    // after that dim.

    // Simple check: for each mapped dim ax, all dims (ax+1..nd-1) must NOT
    // be differently sliced. Since the body expects blockShape dimensions on
    // mapped axes and full input dims on unmapped axes, this is satisfied when
    // the inner dimensions match.
    for (const ax of mappedDims) {
      for (let d = ax + 1; d < shape.length; d++) {
        if (!mappedDims.has(d)) {
          // Inner dim `d` is unmapped — the body reads it at full size.
          // The slice includes the full extent in this dim — contiguous.
          continue;
        }
        // If inner dim is also mapped, then the slice is a 2D sub-block
        // of a 2D array. This is NOT contiguous in general for row-major.
        // Exception: if it's the last (innermost) dim, the contiguous
        // row covers blockShape[g] elements, and we'd need row-by-row copy.
        // For now, reject multi-dim sub-blocking.
        if (DEBUG >= 2)
          console.log(
            "[wasm-block-map] rejected: non-contiguous multi-dim slicing",
          );
        return null;
      }
    }
  }

  // Same check for outputs
  for (let o = 0; o < numOutputs; o++) {
    const axes = outAxes[o];
    const shape = outputShapes[o];
    const mappedDims = new Set<number>();
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) mappedDims.add(axes[g]!);
    }
    for (const ax of mappedDims) {
      for (let d = ax + 1; d < shape.length; d++) {
        if (mappedDims.has(d)) {
          if (DEBUG >= 2)
            console.log(
              "[wasm-block-map] rejected: non-contiguous multi-dim output slicing",
            );
          return null;
        }
      }
    }
  }

  // Build slot-to-internal mapping
  const numBodyInputs = numConsts + numInputs;
  const slotToInternal = new Map<number, number>();
  const internalSizes: number[] = [];

  for (const step of executeSteps) {
    const kernel = step.source as Kernel;
    const internalIdx = internalSizes.length;
    slotToInternal.set(step.outputs[0], internalIdx);
    internalSizes.push(
      (kernel.size as number) * byteWidth(kernel.outputs[0].dtype),
    );
  }

  // Build reindexed kernel steps
  const reindexedSteps: GeneralScanStep[] = [];
  for (const step of executeSteps) {
    const kernel = step.source as Kernel;

    // Map each input JitId to a body-local index
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numBodyInputs) {
        inputSlots.push(inputId);
      } else {
        const intIdx = slotToInternal.get(inputId);
        if (intIdx === undefined) {
          if (DEBUG >= 2)
            console.log(`[wasm-block-map] rejected: unmapped input ${inputId}`);
          return null;
        }
        inputSlots.push(numBodyInputs + intIdx);
      }
    }

    // Reindex kernel expressions
    const reindexMap = inputSlots;
    const reindexedExp = kernel.outputs[0].exp.reindexGids(reindexMap);
    const reindexedReduction = kernel.outputs[0].reduction
      ? new Reduction(
          kernel.outputs[0].reduction.dtype,
          kernel.outputs[0].reduction.op,
          kernel.outputs[0].reduction.size,
          kernel.outputs[0].reduction.epilogue.reindexGids(reindexMap),
        )
      : undefined;
    const reindexedKernel = Kernel.single(
      numBodyInputs + internalSizes.length,
      kernel.size,
      reindexedExp,
      reindexedReduction,
    );

    reindexedSteps.push({
      source: reindexedKernel,
      inputSlots,
      outputInternalIdx: slotToInternal.get(step.outputs[0])!,
    });
  }

  // Determine which internal buffer each output comes from
  const outputSources: number[] = [];
  for (const outSlot of bodyProgram.outputs) {
    const intIdx = slotToInternal.get(outSlot);
    if (intIdx !== undefined) {
      outputSources.push(intIdx);
    } else {
      // Output is a passthrough from input — not supported in compiled path
      if (DEBUG >= 2)
        console.log("[wasm-block-map] rejected: passthrough output");
      return null;
    }
  }

  // Compute byte sizes and strides
  const bodyInputAvals = bodyJaxpr.inBinders
    .slice(numConsts)
    .map((v) => v.aval);
  const bodyOutAvals = bodyJaxpr.outs.map((v) => v.aval);

  const constSizes: number[] = [];
  for (let c = 0; c < numConsts; c++) {
    const aval = bodyJaxpr.inBinders[c].aval;
    constSizes.push((aval.size as number) * byteWidth(aval.dtype));
  }

  const blockInputSizes: number[] = [];
  for (let i = 0; i < numInputs; i++) {
    blockInputSizes.push(
      (bodyInputAvals[i].size as number) * byteWidth(bodyInputAvals[i].dtype),
    );
  }

  const blockOutputSizes: number[] = [];
  for (let o = 0; o < numOutputs; o++) {
    blockOutputSizes.push(
      (bodyOutAvals[o].size as number) * byteWidth(bodyOutAvals[o].dtype),
    );
  }

  const inputStridesArr: number[][] = [];
  for (let i = 0; i < numInputs; i++) {
    const shape = inputShapes[i];
    const elemBytes = byteWidth(bodyInputAvals[i].dtype);
    inputStridesArr.push(computeStrides(shape, elemBytes));
  }

  const outputStridesArr: number[][] = [];
  for (let o = 0; o < numOutputs; o++) {
    const shape = outputShapes[o];
    const elemBytes = byteWidth(bodyOutAvals[o].dtype);
    outputStridesArr.push(computeStrides(shape, elemBytes));
  }

  return {
    numConsts,
    numInputs,
    numOutputs,
    gridRank,
    gridShape,
    blockShape,
    inAxes,
    outAxes,
    inputShapes,
    outputShapes,
    constSizes,
    blockInputSizes,
    blockOutputSizes,
    internalSizes,
    steps: reindexedSteps,
    outputSources,
    inputStrides: inputStridesArr,
    outputStrides: outputStridesArr,
  };
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
  // In row-major layout, a slice is contiguous iff for every dimension d
  // where sizes[d] > 1, ALL inner dimensions d' > d have
  // sizes[d'] == fullShape[d']. This ensures the stride matches.
  const nd = starts.length;
  for (let d = 0; d < nd; d++) {
    if (sizes[d] <= 1) continue;
    for (let d2 = d + 1; d2 < nd; d2++) {
      if (sizes[d2] !== fullShape[d2]) return false;
    }
    break; // Only need to check the outermost non-trivial dim
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
