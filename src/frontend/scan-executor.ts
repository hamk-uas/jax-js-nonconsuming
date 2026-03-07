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
import {
  executeBlockMap,
  type ExecuteBlockMapParams,
} from "./block-map-executor";

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
  /** Dim bindings for resolving symbolic sizes in body program execution */
  dimBindings?: ReadonlyMap<string, number>;
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
    dimBindings,
  } = params;

  // Helper: resolve size of an aval, handling symbolic shapes via dimBindings
  const resolveAvalSize = (aval: ShapedArray): number => {
    if (hasSymbolicDims(aval.shape)) {
      if (!dimBindings) throw new Error("Symbolic shape but no dimBindings");
      const shape = resolveShape(aval.shape, dimBindings);
      return shape.reduce((a, b) => a * b, 1);
    }
    return aval.size as number;
  };

  // Compute per-xs byte strides (size of one iteration's slice)
  const xsStrides = xsAvals.map(
    (aval) => resolveAvalSize(aval) * byteWidth(aval.dtype),
  );

  // Compute per-y byte strides from body jaxpr outputs
  const yOutAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
  const ysStrides = yOutAvals.map(
    (aval) => resolveAvalSize(aval) * byteWidth(aval.dtype),
  );

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
      const bodyResult = bodyProgram.execute(bodyInputs, dimBindings);
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
    const carryAval = bodyJaxpr.outs[ci].aval;
    const carrySize = resolveAvalSize(carryAval) * byteWidth(carryAval.dtype);
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

  if (plan.path === "webgpu-fused-blocked") {
    const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
    (backend as WebGPUBackend).dispatchBlockedAssocScan(
      plan.prepared,
      plan.params,
      constSlots,
      elemSlots,
      outputSlots,
      N,
      plan.blockSize,
      reverse,
    );
    return { outputs: outputSlots, pending: [] };
  }

  if (plan.path === "webgpu-block-map") {
    return executeAssocScanBlockMap(params);
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
// Block-map–based associative scan execution
// ---------------------------------------------------------------------------

/**
 * Execute an associative scan using the block-map decomposition:
 * 1. Local scan: block_map with WorkgroupAssociativeScan body
 * 2. If M > 1: gather block summaries → recursive scan → apply
 *
 * This path reuses block-map.ts's fused shader infrastructure for the
 * local scan, eliminating the standalone WebGPU assocScan shaders.
 */
function executeAssocScanBlockMap(
  params: ExecuteAssocScanParams,
): ExecuteAssocScanResult {
  const {
    backend,
    plan,
    numLeaves,
    axis,
    reverse,
    constSlots,
    elemSlots,
    elemAvals,
    constAvals,
    outputSlots,
    dimBindings,
  } = params;

  if (plan.path !== "webgpu-block-map") {
    throw new Error("executeAssocScanBlockMap: expected webgpu-block-map plan");
  }

  const {
    localScanBodyProgram,
    localScanBodyJaxpr,
    scanBodyJaxpr: _scanBodyJaxpr,
    scanBodyProgram,
    blockSize: B,
    numConsts,
  } = plan;

  // Resolve N at runtime
  const N = resolveAxisN(elemAvals[0].shape, axis, dimBindings);
  const M = Math.ceil(N / B);

  // Handle reverse: reverse input elements before local scan, reverse output after apply.
  // For simplicity, we reverse in-place in the output buffers.
  let inputSlots = elemSlots;
  if (reverse) {
    // Allocate reversed copies of input elements.
    inputSlots = [];
    for (let k = 0; k < numLeaves; k++) {
      const elemShape = elemAvals[k].shape as number[];
      const elemBw = byteWidth(elemAvals[k].dtype);
      const perElemBytes =
        elemShape.slice(1).reduce((a, b) => a * b, 1) * elemBw;
      const totalBytes = N * perElemBytes;
      const revSlot = backend.malloc(totalBytes);
      // Reverse: copy element i → position N-1-i
      for (let i = 0; i < N; i++) {
        backend.copyBufferToBuffer(
          elemSlots[k],
          i * perElemBytes,
          revSlot,
          (N - 1 - i) * perElemBytes,
          perElemBytes,
        );
      }
      inputSlots.push(revSlot);
    }
  }

  // --- Phase 1: Local scan via block_map ---
  const gridShape = [M];
  const blockShape = [B];

  // inAxes: constants broadcast (null), elements sliced along axis 0
  const inAxes: (number | null)[][] = [
    ...constAvals.map(() => [null as number | null]),
    ...elemAvals.map(() => [0 as number | null]),
  ];
  const outAxes: (number | null)[][] = elemAvals.map(() => [0]);

  const inputShapes = elemAvals.map((a) => a.shape as number[]);
  const outputShapes = elemAvals.map((a) => a.shape as number[]);

  // Allocate local scan output buffers
  const localScanSlots: Slot[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const totalBytes =
      (elemAvals[k].shape as number[]).reduce((a, b) => a * b, 1) *
      byteWidth(elemAvals[k].dtype);
    localScanSlots.push(backend.malloc(totalBytes));
  }

  const blockMapParams: ExecuteBlockMapParams = {
    backend,
    bodyProgram: localScanBodyProgram,
    bodyJaxpr: localScanBodyJaxpr,
    blockShape,
    inAxes,
    outAxes,
    numConsts,
    numInputs: numLeaves,
    gridShape,
    inputShapes,
    outputShapes,
    constSlots,
    inputSlots: reverse ? inputSlots : elemSlots,
    outputSlots: localScanSlots,
  };

  const bmResult = executeBlockMap(blockMapParams);
  // Flush any pending block map operations
  if (bmResult.pending.length > 0) {
    flushPending(bmResult.pending);
  }

  if (M === 1) {
    // Single block: local scan output IS the final output.
    // Copy to pre-allocated output slots.
    for (let k = 0; k < numLeaves; k++) {
      const totalBytes =
        (elemAvals[k].shape as number[]).reduce((a, b) => a * b, 1) *
        byteWidth(elemAvals[k].dtype);
      if (reverse) {
        // Reverse the output
        const perElemBytes =
          (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
          byteWidth(elemAvals[k].dtype);
        for (let i = 0; i < N; i++) {
          backend.copyBufferToBuffer(
            localScanSlots[k],
            i * perElemBytes,
            outputSlots[k],
            (N - 1 - i) * perElemBytes,
            perElemBytes,
          );
        }
      } else {
        backend.copyBufferToBuffer(
          localScanSlots[k],
          0,
          outputSlots[k],
          0,
          totalBytes,
        );
      }
      backend.decRef(localScanSlots[k]);
    }
    if (reverse) {
      for (const s of inputSlots) backend.decRef(s);
    }
    return { outputs: outputSlots, pending: [] };
  }

  // --- Phase 2: Gather block summaries ---
  // For each leaf, extract the last element of each block.
  // summary_k[i] = localScan_k[min((i+1)*B - 1, N-1)]
  const summarySlots: Slot[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const elemShape = elemAvals[k].shape as number[];
    const perElemBytes =
      elemShape.slice(1).reduce((a, b) => a * b, 1) *
      byteWidth(elemAvals[k].dtype);
    const summaryBytes = M * perElemBytes;
    const summarySlot = backend.malloc(summaryBytes);
    for (let i = 0; i < M; i++) {
      const srcIdx = Math.min((i + 1) * B - 1, N - 1);
      backend.copyBufferToBuffer(
        localScanSlots[k],
        srcIdx * perElemBytes,
        summarySlot,
        i * perElemBytes,
        perElemBytes,
      );
    }
    summarySlots.push(summarySlot);
  }

  // --- Phase 3: Prefix scan the summaries ---
  // Use the per-element body program to scan M summary values.
  // For small M (≤ B), this runs as a sequential scan.
  // The body takes [consts, a, b] and returns [result].
  const scannedSummarySlots: Slot[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const elemShape = elemAvals[k].shape as number[];
    const perElemBytes =
      elemShape.slice(1).reduce((a, b) => a * b, 1) *
      byteWidth(elemAvals[k].dtype);
    const scannedSlot = backend.malloc(M * perElemBytes);
    scannedSummarySlots.push(scannedSlot);
  }

  // Sequential prefix scan on summaries: scanned[0] = summary[0],
  // scanned[i] = body(scanned[i-1], summary[i])
  {
    // Initialize: scanned[0] = summary[0]
    for (let k = 0; k < numLeaves; k++) {
      const perElemBytes =
        (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
        byteWidth(elemAvals[k].dtype);
      backend.copyBufferToBuffer(
        summarySlots[k],
        0,
        scannedSummarySlots[k],
        0,
        perElemBytes,
      );
    }

    // Prefix scan: for i = 1..M-1
    for (let i = 1; i < M; i++) {
      // Extract a = scanned[i-1], b = summary[i]
      const aSlots: Slot[] = [];
      const bSlots: Slot[] = [];
      for (let k = 0; k < numLeaves; k++) {
        const perElemBytes =
          (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
          byteWidth(elemAvals[k].dtype);
        const a = backend.malloc(perElemBytes);
        backend.copyBufferToBuffer(
          scannedSummarySlots[k],
          (i - 1) * perElemBytes,
          a,
          0,
          perElemBytes,
        );
        aSlots.push(a);

        const b = backend.malloc(perElemBytes);
        backend.copyBufferToBuffer(
          summarySlots[k],
          i * perElemBytes,
          b,
          0,
          perElemBytes,
        );
        bSlots.push(b);
      }

      // IncRef consts for body
      for (const s of constSlots) backend.incRef(s);

      const bodyInputs = [...constSlots, ...aSlots, ...bSlots];
      const bodyResult = scanBodyProgram.execute(bodyInputs, dimBindings);
      if (bodyResult.pending.length > 0) flushPending(bodyResult.pending);

      // DecRef consts
      for (const s of constSlots) backend.decRef(s);
      // DecRef a and b
      for (const s of aSlots) backend.decRef(s);
      for (const s of bSlots) backend.decRef(s);

      // Write result to scannedSummary[i]
      for (let k = 0; k < numLeaves; k++) {
        const perElemBytes =
          (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
          byteWidth(elemAvals[k].dtype);
        backend.copyBufferToBuffer(
          bodyResult.outputs[k],
          0,
          scannedSummarySlots[k],
          i * perElemBytes,
          perElemBytes,
        );
        backend.decRef(bodyResult.outputs[k]);
      }
    }
  }

  // Free raw summaries
  for (const s of summarySlots) backend.decRef(s);

  // --- Phase 4: Apply scanned summaries to blocks 1..M-1 ---
  // For each element in block i > 0: output[i*B+j] = body(scannedSummary[i-1], localScan[i*B+j])
  for (let blockIdx = 1; blockIdx < M; blockIdx++) {
    const blockStart = blockIdx * B;
    const blockEnd = Math.min(blockStart + B, N);
    const blockLen = blockEnd - blockStart;

    for (let j = 0; j < blockLen; j++) {
      const globalIdx = blockStart + j;
      const aSlots: Slot[] = []; // scannedSummary[blockIdx - 1]
      const bSlots: Slot[] = []; // localScan[globalIdx]

      for (let k = 0; k < numLeaves; k++) {
        const perElemBytes =
          (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
          byteWidth(elemAvals[k].dtype);

        const a = backend.malloc(perElemBytes);
        backend.copyBufferToBuffer(
          scannedSummarySlots[k],
          (blockIdx - 1) * perElemBytes,
          a,
          0,
          perElemBytes,
        );
        aSlots.push(a);

        const b = backend.malloc(perElemBytes);
        backend.copyBufferToBuffer(
          localScanSlots[k],
          globalIdx * perElemBytes,
          b,
          0,
          perElemBytes,
        );
        bSlots.push(b);
      }

      // IncRef consts
      for (const s of constSlots) backend.incRef(s);

      const bodyInputs = [...constSlots, ...aSlots, ...bSlots];
      const bodyResult = scanBodyProgram.execute(bodyInputs, dimBindings);
      if (bodyResult.pending.length > 0) flushPending(bodyResult.pending);

      // DecRef consts, a, b
      for (const s of constSlots) backend.decRef(s);
      for (const s of aSlots) backend.decRef(s);
      for (const s of bSlots) backend.decRef(s);

      // Write result to output at globalIdx
      for (let k = 0; k < numLeaves; k++) {
        const perElemBytes =
          (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
          byteWidth(elemAvals[k].dtype);
        backend.copyBufferToBuffer(
          bodyResult.outputs[k],
          0,
          outputSlots[k],
          globalIdx * perElemBytes,
          perElemBytes,
        );
        backend.decRef(bodyResult.outputs[k]);
      }
    }
  }

  // Copy block 0 from local scan to output (unchanged)
  {
    const block0End = Math.min(B, N);
    for (let k = 0; k < numLeaves; k++) {
      const perElemBytes =
        (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
        byteWidth(elemAvals[k].dtype);
      backend.copyBufferToBuffer(
        localScanSlots[k],
        0,
        outputSlots[k],
        0,
        block0End * perElemBytes,
      );
    }
  }

  // If reverse, reverse the output
  if (reverse) {
    for (let k = 0; k < numLeaves; k++) {
      const perElemBytes =
        (elemAvals[k].shape as number[]).slice(1).reduce((a, b) => a * b, 1) *
        byteWidth(elemAvals[k].dtype);
      const totalBytes = N * perElemBytes;
      const tmpSlot = backend.malloc(totalBytes);
      backend.copyBufferToBuffer(outputSlots[k], 0, tmpSlot, 0, totalBytes);
      for (let i = 0; i < N; i++) {
        backend.copyBufferToBuffer(
          tmpSlot,
          i * perElemBytes,
          outputSlots[k],
          (N - 1 - i) * perElemBytes,
          perElemBytes,
        );
      }
      backend.decRef(tmpSlot);
    }
    // Free reversed input copies
    for (const s of inputSlots) backend.decRef(s);
  }

  // Cleanup
  for (const s of localScanSlots) backend.decRef(s);
  for (const s of scannedSummarySlots) backend.decRef(s);

  return { outputs: outputSlots, pending: [] };
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
    const sliceSize = xsStrides[j]; // stride = per-element bytes

    // Copy the xs slice into a new buffer. Prefer copyBufferToBuffer
    // (keeps data on-device, avoids readSync which needs OffscreenCanvas
    // on WebGPU). Fall back to readSync + malloc
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
