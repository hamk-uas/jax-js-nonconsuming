/**
 * @file Unified scan executor — single execution path for all scan strategies.
 *
 * This replaces v1's dual loop (eager Primitive.Scan + JIT scanRunner) with one
 * function that handles ownership, flush, and dispatch for all backends and all
 * plan paths (fallback, compiled-loop, preencoded-routine).
 */

import { byteWidth } from "../alu";
import type { Backend, Slot } from "../backend";
import type { PendingExecute } from "./array";
import { Array as JaxArray } from "./array";
import { _associativeScanCoreImpl, ShapedArray } from "./core";
import type { Jaxpr } from "./jaxpr";
import type { JitProgram } from "./jit";
import type { AssocScanPlan, ScanPlan } from "./scan-plan";
import type { WasmBackend } from "../backend/wasm";
import type { NativeScanMultiParams, WebGPUBackend } from "../backend/webgpu";
import {
  concreteDim,
  concreteShape,
  type Dim,
  hasSymbolicDims,
  isSymbolicDim,
  resolveShape,
  ShapeTracker,
} from "../shape";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface ExecuteScanParams {
  backend: Backend;
  plan: ScanPlan;
  bodyProgram: JitProgram;
  bodyJaxpr: Jaxpr;
  length: number;
  numCarry: number;
  numConsts: number;
  numX: number;
  numY: number;
  reverse: boolean;
  constSlots: Slot[];
  initCarrySlots: Slot[];
  xsSlots: Slot[];
  xsAvals: ShapedArray[];
  /** Preallocated output slots: [carry_out..., stacked_ys...] */
  outputSlots: Slot[];
}

export interface ExecuteScanResult {
  outputs: Slot[];
  pending: PendingExecute[];
}

/**
 * Execute a scan loop. Dispatches to the appropriate strategy based on the plan.
 *
 * Ownership contract:
 * - constSlots: borrowed (incRef'd before each body call, not consumed)
 * - initCarrySlots: consumed (absorbed into first iteration)
 * - xsSlots: borrowed (sliced per iteration via createView or readSync)
 * - outputSlots: filled in-place (carry outputs + stacked Y outputs)
 */
export function executeScan(params: ExecuteScanParams): ExecuteScanResult {
  switch (params.plan.path) {
    case "fallback":
      return executeScanFallback(params);
    case "compiled-loop":
      return executeScanCompiledLoop(params);
    case "preencoded-routine":
      return executeScanPreencodedRoutine(params);
    case "preencoded-multi-step":
      return executeScanPreencodedMultiStep(params);
  }
}

// ---------------------------------------------------------------------------
// Fallback: JS loop calling bodyProgram.execute() per iteration
// ---------------------------------------------------------------------------

function executeScanFallback(params: ExecuteScanParams): ExecuteScanResult {
  const {
    backend,
    bodyProgram,
    bodyJaxpr,
    length,
    numCarry,
    numConsts: _numConsts,
    numX: _numX,
    numY,
    reverse,
    constSlots,
    initCarrySlots,
    xsSlots,
    xsAvals,
    outputSlots,
  } = params;

  // Compute per-xs byte strides (size of one iteration's slice)
  const xsStrides = xsAvals.map((aval) => aval.size * byteWidth(aval.dtype));

  // Compute per-y byte strides from body jaxpr outputs
  const yOutAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
  const ysStrides = yOutAvals.map((aval) => aval.size * byteWidth(aval.dtype));

  // Current carry slots — start with initCarry.
  // IncRef so the loop can uniformly decRef old carry each iteration
  // (initCarrySlots are borrowed from the caller who frees them separately).
  let carry = initCarrySlots.slice();
  for (const slot of carry) backend.incRef(slot);

  // Y output slots from the preallocated outputs
  const ysOutputSlots = outputSlots.slice(numCarry);

  // Track pending operations
  const pending: PendingExecute[] = [];

  // Sub-batch GPU commands: instead of one queue.submit() per iteration,
  // batch SCAN_BATCH_SIZE iterations into one submit. This reduces O(2N)
  // submits to O(N/SCAN_BATCH_SIZE) while limiting deferred buffer
  // accumulation. The batch encoder accumulates dispatches and copies;
  // nested flushPending calls become no-ops (depth tracking).
  const SCAN_BATCH_SIZE = 256;
  const useBatching = backend.beginBatch != null && length > 1;

  if (useBatching) backend.beginBatch!();

  try {
    for (let step = 0; step < length; step++) {
      const i = reverse ? length - 1 - step : step;

      // Invariant 1: Flush pending ops before each body invocation
      flushPending(pending);

      // Slice xs for this iteration
      const xSlices = sliceXsAtIteration(
        backend,
        xsSlots,
        xsStrides,
        xsAvals,
        i,
      );

      // IncRef consts so body can consume them
      for (const slot of constSlots) backend.incRef(slot);

      // Build body inputs: [consts, carry, xSlices]
      // carry is consumed (body takes ownership)
      const bodyInputs = [...constSlots, ...carry, ...xSlices];

      // Execute body
      const bodyResult = bodyProgram.execute(bodyInputs);
      pending.push(...bodyResult.pending);

      // Flush pending ops from body execution before reading output slots
      // (the body's kernels must be dispatched before we can copy from them)
      flushPending(pending);

      const newCarry = bodyResult.outputs.slice(0, numCarry);
      const ySlices = bodyResult.outputs.slice(numCarry);

      // Release borrowed consts and created x slice slots.
      // Note: JitProgram.execute() already inserts incref steps for any output
      // that is a passthrough from an input or appears multiple times in the
      // output list, so each output position has its own reference. No extra
      // alias-protection incRef is needed here — the JIT's refs protect outputs
      // from being prematurely freed by these input decRefs.
      for (const slot of constSlots) backend.decRef(slot);
      for (const slot of xSlices) backend.decRef(slot);

      // Invariant 3: Y stacking — copy y slices into preallocated output buffers
      for (let yi = 0; yi < numY; yi++) {
        if (ysStrides[yi] > 0) {
          copySliceToBuffer(
            backend,
            ysOutputSlots[yi],
            ySlices[yi],
            i,
            ysStrides[yi],
            ysStrides[yi],
          );
        }
        // Free the y slice (it's been copied into the output buffer)
        backend.decRef(ySlices[yi]);
      }

      // Invariant 2: Carry lifecycle — body.execute() borrows inputs (does not
      // consume them). We must explicitly release old carry slots. The JIT's
      // incref for passthrough/duplicate outputs ensures that any carry slot
      // reappearing in newCarry has an extra ref, so this decRef is safe.
      for (const slot of carry) backend.decRef(slot);
      carry = newCarry;

      // Periodic flush: end current batch and start a new one to release
      // deferred buffers and limit GPU memory accumulation.
      if (
        useBatching &&
        step < length - 1 &&
        (step + 1) % SCAN_BATCH_SIZE === 0
      ) {
        backend.endBatch!();
        backend.beginBatch!();
      }
    }
  } finally {
    if (useBatching) backend.endBatch!();
  }

  // Flush any remaining pending ops before writing final carry
  flushPending(pending);

  // Write final carry to output slots
  const carryOutputSlots = outputSlots.slice(0, numCarry);
  for (let ci = 0; ci < numCarry; ci++) {
    if (carry[ci] === carryOutputSlots[ci]) {
      // Slot is already the output (shouldn't normally happen in fallback)
      continue;
    }
    // Copy carry data into the preallocated output slot
    const carrySize =
      bodyJaxpr.outs[ci].aval.size * byteWidth(bodyJaxpr.outs[ci].aval.dtype);
    copySliceToBuffer(
      backend,
      carryOutputSlots[ci],
      carry[ci],
      0,
      0,
      carrySize,
    );
    backend.decRef(carry[ci]);
  }

  return { outputs: outputSlots, pending };
}

// ---------------------------------------------------------------------------
// Compiled-loop: WASM or WebGPU native scan
// ---------------------------------------------------------------------------

function executeScanCompiledLoop(params: ExecuteScanParams): ExecuteScanResult {
  const {
    backend,
    plan,
    numCarry,
    numY,
    constSlots,
    initCarrySlots,
    xsSlots,
    outputSlots,
  } = params;

  if (plan.path !== "compiled-loop") throw new Error("unreachable");
  if (!plan.params) throw new Error("compiled-loop plan missing params");

  const carryOutSlots = outputSlots.slice(0, numCarry);
  const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);

  if (backend.type === "webgpu") {
    // WebGPU native scan — no internal slots needed
    const webgpuBackend = backend as WebGPUBackend;
    webgpuBackend.dispatchNativeScanGeneral(
      plan.executable,
      plan.params as NativeScanMultiParams,
      constSlots,
      initCarrySlots,
      xsSlots,
      carryOutSlots,
      ysStackedSlots,
    );
  } else {
    // WASM native scan
    const wasmBackend = backend as WasmBackend;
    wasmBackend.dispatchNativeScanGeneral(
      plan.executable,
      plan.params as import("../backend/wasm").NativeScanGeneralParams,
      params.length,
      constSlots,
      initCarrySlots,
      xsSlots,
      carryOutSlots,
      ysStackedSlots,
    );
  }

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Preencoded-routine: WebGPU routine scan with uniform offsets (P4)
// ---------------------------------------------------------------------------

function executeScanPreencodedRoutine(
  params: ExecuteScanParams,
): ExecuteScanResult {
  const {
    backend,
    plan,
    numCarry,
    numY,
    constSlots,
    initCarrySlots,
    xsSlots,
    outputSlots,
  } = params;

  if (plan.path !== "preencoded-routine") throw new Error("unreachable");

  const carryOutSlots = outputSlots.slice(0, numCarry);
  const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);

  const webgpuBackend = backend as WebGPUBackend;
  webgpuBackend.dispatchPreencodedScan(
    plan.preencodedParams,
    constSlots,
    initCarrySlots,
    xsSlots,
    carryOutSlots,
    ysStackedSlots,
  );

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Preencoded multi-step: WebGPU multi-kernel scan with per-step offsets (P2c)
// ---------------------------------------------------------------------------

function executeScanPreencodedMultiStep(
  params: ExecuteScanParams,
): ExecuteScanResult {
  const {
    backend,
    plan,
    numCarry,
    numY,
    constSlots,
    initCarrySlots,
    xsSlots,
    outputSlots,
  } = params;

  if (plan.path !== "preencoded-multi-step") throw new Error("unreachable");

  const carryOutSlots = outputSlots.slice(0, numCarry);
  const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);

  const webgpuBackend = backend as WebGPUBackend;
  webgpuBackend.dispatchPreencodedMultiStepScan(
    plan.prepared,
    constSlots,
    initCarrySlots,
    xsSlots,
    carryOutSlots,
    ysStackedSlots,
  );

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Associative scan executor (Kogge-Stone)
// ---------------------------------------------------------------------------

export interface ExecuteAssocScanParams {
  backend: Backend;
  plan: AssocScanPlan;
  bodyJaxpr: Jaxpr;
  numLeaves: number;
  numConsts: number;
  axis: number;
  reverse: boolean;
  constSlots: Slot[];
  elemSlots: Slot[];
  constAvals: ShapedArray[];
  elemAvals: ShapedArray[];
  /** Preallocated output slots (may be replaced in fallback path). */
  outputSlots: Slot[];
  /** Dimension bindings for symbolic shape resolution. */
  dimBindings?: ReadonlyMap<string, number>;
}

export interface ExecuteAssocScanResult {
  /** Final output slots — same as input for native paths, replaced for fallback. */
  outputs: Slot[];
  pending: PendingExecute[];
}

/**
 * Execute an associative scan. Dispatches based on plan path:
 * - compiled-loop: WASM-compiled Kogge-Stone ladder
 * - webgpu-fused: fused WGSL shader per round
 * - fallback: JS Kogge-Stone loop via vmap + evalJaxpr
 */
export function executeAssociativeScan(
  params: ExecuteAssocScanParams,
): ExecuteAssocScanResult {
  const {
    backend,
    plan,
    bodyJaxpr,
    numLeaves,
    numConsts: _numConsts,
    axis,
    reverse,
    constSlots,
    elemSlots,
    constAvals,
    elemAvals,
    outputSlots,
    dimBindings,
  } = params;

  if (plan.path === "compiled-loop") {
    const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
    (backend as WasmBackend).dispatchNativeAssociativeScan(
      plan.executable,
      plan.params,
      N,
      constSlots,
      elemSlots,
      outputSlots,
    );
    return { outputs: outputSlots, pending: [] };
  }

  if (plan.path === "compiled-loop-blocked") {
    const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
    (backend as WasmBackend).dispatchBlockedAssociativeScan(
      plan.executable,
      plan.params,
      N,
      constSlots,
      elemSlots,
      outputSlots,
    );
    return { outputs: outputSlots, pending: [] };
  }

  if (plan.path === "webgpu-fused") {
    const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
    (backend as WebGPUBackend).dispatchAssocScan(
      plan.prepared,
      plan.params,
      constSlots,
      elemSlots,
      outputSlots,
      N,
      reverse,
    );
    return { outputs: outputSlots, pending: [] };
  }

  // Fallback: JS Kogge-Stone loop
  const resolveAvalShape = (shape: Dim[]): number[] =>
    hasSymbolicDims(shape)
      ? concreteShape(resolveShape(shape, dimBindings!))
      : (shape as number[]);

  const constArrays = constSlots.map((slot, i) => {
    backend.incRef(slot);
    const aval = constAvals[i];
    return new JaxArray({
      source: slot,
      st: ShapeTracker.fromShape(resolveAvalShape(aval.shape)),
      dtype: aval.dtype,
      weakType: aval.weakType,
      backend,
      committed: false,
    });
  });

  const elemArrays = elemSlots.map((slot, i) => {
    backend.incRef(slot);
    const aval = elemAvals[i];
    return new JaxArray({
      source: slot,
      st: ShapeTracker.fromShape(resolveAvalShape(aval.shape)),
      dtype: aval.dtype,
      weakType: aval.weakType,
      backend,
      committed: false,
    });
  });

  const results = _associativeScanCoreImpl!(
    bodyJaxpr,
    constArrays,
    elemArrays,
    numLeaves,
    axis,
    reverse,
  );

  // Extract slots from results, replacing pre-allocated output buffers.
  const finalOutputs: Slot[] = [];
  for (let i = 0; i < numLeaves; i++) {
    const resultArr = results[i] as JaxArray;
    resultArr._flushPendingSync();
    const slot = resultArr._realizeSource();
    backend.incRef(slot);
    // Free the pre-allocated output buffer
    backend.decRef(outputSlots[i]);
    finalOutputs.push(slot);
    resultArr.dispose();
  }

  // Dispose input Array wrappers
  for (const a of constArrays) a.dispose();
  for (const a of elemArrays) a.dispose();

  return { outputs: finalOutputs, pending: [] };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resolve the scan axis dimension N, handling symbolic dims. */
function resolveAxisN(
  shape: Dim[],
  axis: number,
  dimBindings?: ReadonlyMap<string, number>,
): number {
  const nDim = shape[axis];
  return isSymbolicDim(nDim)
    ? concreteDim(resolveShape([nDim], dimBindings!)[0], "assoc_scan N")
    : (nDim as number);
}

/** Flush all pending GPU/WASM operations, batching dispatches when possible. */
function flushPending(pending: PendingExecute[]): void {
  if (pending.length === 0) return;
  const backend = pending[0].backend;
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
 * Slice xs buffers at a given iteration index.
 * Returns new slots (views or copies) for each xs input.
 */
function sliceXsAtIteration(
  backend: Backend,
  xsSlots: Slot[],
  xsStrides: number[],
  xsAvals: ShapedArray[],
  iterIdx: number,
): Slot[] {
  const slices: Slot[] = [];
  for (let j = 0; j < xsSlots.length; j++) {
    const srcOffset = iterIdx * xsStrides[j];
    const sliceSize = xsAvals[j].size * byteWidth(xsAvals[j].dtype);

    // Copy the xs slice into a new buffer. Prefer copyBufferToBuffer
    // (keeps data on-device, avoids readSync which needs OffscreenCanvas
    // on WebGPU and is unavailable in Deno). Fall back to readSync + malloc
    // for backends that don't implement copyBufferToBuffer (CPU).
    if (backend.copyBufferToBuffer) {
      const slot = backend.malloc(sliceSize);
      backend.copyBufferToBuffer(xsSlots[j], srcOffset, slot, 0, sliceSize);
      slices.push(slot);
    } else {
      const data = backend.readSync(xsSlots[j], srcOffset, sliceSize);
      const slot = backend.malloc(sliceSize, data);
      slices.push(slot);
    }
  }
  return slices;
}

/**
 * Copy a slice from src slot into dst slot at a given iteration offset.
 */
function copySliceToBuffer(
  backend: Backend,
  dst: Slot,
  src: Slot,
  iterIdx: number,
  strideBytes: number,
  sliceBytes: number,
): void {
  const dstOffset = iterIdx * strideBytes;
  backend.copyBufferToBuffer(src, 0, dst, dstOffset, sliceBytes);
}
