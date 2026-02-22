/**
 * @file Scan plan construction — determines the execution strategy for a scan.
 */

import { AluExp, AluOp, byteWidth, Kernel, Reduction } from "../alu";
import type { Backend, Executable } from "../backend";
import {
  getScanRoutineInfo,
  NativeAssocScanParams,
  NativeScanGeneralParams,
  ScanRoutineInfo,
  WasmBackend,
} from "../backend/wasm";
import type {
  NativeScanMultiParams,
  NativeScanMultiStep,
  PreparedPreencodedScan,
} from "../backend/webgpu";
import type { WebGPUBackend } from "../backend/webgpu";
import { Routine, Routines } from "../routine";
import { type Dim, isSymbolicDim } from "../shape";
import { DEBUG } from "../utils";
import type { ScanPath } from "../utils";
import type { Jaxpr } from "./jaxpr";
import type { JitId, JitProgram, JitStep } from "./jit";

// ---------------------------------------------------------------------------
// ScanPlan: a discriminated union of execution strategies
// ---------------------------------------------------------------------------

export type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | {
      path: "compiled-loop";
      executable: Executable;
      params?: NativeScanGeneralParams | NativeScanMultiParams;
      internalSizes?: number[];
    }
  | { path: "preencoded-routine"; preencodedParams: PreparedPreencodedScan };

/**
 * Execution plan for `lax.associativeScan` (Kogge-Stone parallel prefix scan).
 *
 * - `compiled-loop`: Entire Kogge-Stone ladder compiled to a single WASM module.
 *   N is a runtime parameter — the same compiled module supports any input length.
 * - `fallback`: JS-driven Kogge-Stone loop calling body program per round.
 */
export type AssocScanPlan =
  | {
      path: "compiled-loop";
      executable: Executable;
      params: NativeAssocScanParams;
    }
  | { path: "fallback" };

type ExecuteStep = Extract<JitStep, { type: "execute" }>;

// ---------------------------------------------------------------------------
// Path acceptance checking (for testing / debugging)
// ---------------------------------------------------------------------------

/**
 * Check if a chosen scan path satisfies the acceptPath constraint.
 * Returns an error message if the path is not allowed, or null if OK.
 *
 * Special case: an empty array `[]` always rejects, showing the chosen path.
 */
export function checkAcceptedPath(
  chosenPath: ScanPath,
  acceptPath: string | string[] | undefined,
  extraInfo?: string,
): string | null {
  if (!acceptPath) return null;

  const allowedPaths = Array.isArray(acceptPath) ? acceptPath : [acceptPath];
  const suffix = extraInfo ? ` (${extraInfo})` : "";

  if (allowedPaths.length === 0) {
    return `Scan path debug: chose "${chosenPath}"${suffix}`;
  }

  if (!allowedPaths.includes(chosenPath)) {
    return (
      `Scan acceptPath constraint not satisfied: ` +
      `got "${chosenPath}" but accepted paths are [${allowedPaths.map((p) => `"${p}"`).join(", ")}]${suffix}`
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// Buffer size helpers (shared by backends in later phases)
// ---------------------------------------------------------------------------

/**
 * Extract buffer sizes and strides from body jaxpr for native scan codegen.
 * Shared by WebGPU and WASM native scan implementations.
 */
export function getScanBufferSizes(
  bodyJaxpr: Jaxpr,
  numConsts: number,
  numCarry: number,
  numX: number,
) {
  const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry, numConsts + numCarry + numX)
    .map((v) => v.aval);
  const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);

  return {
    constSizes: constAvals.map((a) => a.size * byteWidth(a.dtype)),
    carrySizes: carryAvals.map((a) => a.size * byteWidth(a.dtype)),
    xsStrides: xAvals.map((a) => a.size * byteWidth(a.dtype)),
    ysStrides: yAvals.map((a) => a.size * byteWidth(a.dtype)),
  };
}

// ---------------------------------------------------------------------------
// planScan: decide which execution strategy to use
// ---------------------------------------------------------------------------

/**
 * Try to prepare a WASM native scan executable.
 *
 * Builds GeneralScanStep[] from the body program's execute steps, maps
 * slot IDs to internal buffer indices, determines CarryOutputSource and
 * YOutputSource for each output, and calls backend.prepareNativeScanGeneral().
 *
 * Returns null if the body can't be compiled (e.g. unsupported step types,
 * unmapped slots).
 */
function tryPrepareWasmNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  executeSteps: ExecuteStep[],
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): {
  executable: Executable;
  internalSizes: number[];
  params?: NativeScanGeneralParams;
} | null {
  if (DEBUG >= 2) {
    console.log(
      `[wasm-scan] trying with numCarry=${numCarry}, numY=${numY}, steps=${executeSteps.length}`,
    );
  }

  // Check for unsupported routines.
  // We delegate the check to getScanRoutineInfo in the WASM backend, which
  // knows which routines it supports bindings for.
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      if (!getScanRoutineInfo(step.source)) {
        if (DEBUG >= 1) {
          console.log(
            `[wasm-scan] skipped, unsupported routine: ${step.source.name}`,
          );
        }
        return null;
      }
    }
  }

  const numInputs = numConsts + numCarry + numX;

  const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(
    bodyJaxpr,
    numConsts,
    numCarry,
    numX,
  );

  // Build mapping from JitId (output slot) to internal buffer index
  // Multi-output routines need multiple internal buffers
  const slotToInternal = new Map<JitId, number>();
  const stepToInternalBase = new Map<number, number>();
  const internalSizes: number[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;
    stepToInternalBase.set(i, internalSizes.length);

    if (source instanceof Kernel) {
      const internalIdx = internalSizes.length;
      slotToInternal.set(step.outputs[0], internalIdx);
      internalSizes.push(
        (source.size as number) * byteWidth(source.outputs[0].dtype),
      );
    } else if (source instanceof Routine) {
      // Routine: may have multiple outputs
      for (let outIdx = 0; outIdx < step.outputs.length; outIdx++) {
        const internalIdx = internalSizes.length;
        slotToInternal.set(step.outputs[outIdx], internalIdx);
        const outShape = source.type.outputShapes[outIdx];
        const outDtype = source.type.outputDtypes[outIdx];
        internalSizes.push(
          outShape.reduce((a, b) => a * b, 1) * byteWidth(outDtype),
        );
      }
    }
  }

  // Calculate aux buffer size for routines that need it
  let auxBufferSize = 0;
  let elementSize: 4 | 8 = 4;
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      const routine = step.source;
      const dtype = routine.type.inputDtypes[0];
      elementSize = byteWidth(dtype) as 4 | 8;
      if (routine.name === Routines.Sort) {
        const inputShape = routine.type.inputShapes[0];
        const sortDim = inputShape[inputShape.length - 1];
        auxBufferSize = Math.max(auxBufferSize, sortDim * elementSize);
      } else if (routine.name === Routines.Argsort) {
        const inputShape = routine.type.inputShapes[0];
        const sortDim = inputShape[inputShape.length - 1];
        auxBufferSize = Math.max(auxBufferSize, sortDim * 4);
      }
    }
  }

  // Build routineInfos array for WASM imports (size-specialized)
  const routineInfos: ScanRoutineInfo[] = [];
  const stepToRoutineInfoIdx = new Map<number, number>();

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    if (step.source instanceof Routine) {
      const routine = step.source;
      const info = getScanRoutineInfo(routine);
      if (!info) {
        throw new Error(
          `[wasm-scan] Unexpected unsupported routine: ${routine.name}`,
        );
      }

      const routineInfoIdx = routineInfos.length;
      stepToRoutineInfoIdx.set(i, routineInfoIdx);
      routineInfos.push(info);
    }
  }

  // Build steps with reindexed inputs
  type LocalStep = import("../backend/wasm").GeneralScanStep;
  const steps: LocalStep[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;

    // Map each input JitId to either a jaxpr input index or an internal buffer
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        inputSlots.push(inputId);
      } else {
        const internalIdx = slotToInternal.get(inputId);
        if (internalIdx === undefined) {
          if (DEBUG >= 1)
            console.log(
              `[wasm-scan] skipped, input ${inputId} not found in slot mapping`,
            );
          return null;
        }
        inputSlots.push(numInputs + internalIdx);
      }
    }

    if (source instanceof Kernel) {
      // Reindex kernel expressions to use our inputSlots mapping
      const reindexMap = inputSlots;
      const reindexedExp = source.outputs[0].exp.reindexGids(reindexMap);
      const reindexedReduction = source.outputs[0].reduction
        ? new Reduction(
            source.outputs[0].reduction.dtype,
            source.outputs[0].reduction.op,
            source.outputs[0].reduction.size,
            source.outputs[0].reduction.epilogue.reindexGids(reindexMap),
          )
        : undefined;
      const reindexedKernel = Kernel.single(
        numInputs + internalSizes.length,
        source.size,
        reindexedExp,
        reindexedReduction,
      );

      const internalBase = stepToInternalBase.get(i)!;
      steps.push({
        source: reindexedKernel,
        inputSlots,
        outputInternalIdx: internalBase,
      });
    } else if (source instanceof Routine) {
      // Routine step: build routineCallInfo with static params
      const routine = source;
      const routineName = routine.name as Routines;
      const routineInfoIdx = stepToRoutineInfoIdx.get(i)!;
      const internalBase = stepToInternalBase.get(i)!;
      const numOutputs = routine.type.outputShapes.length;
      const outputInternalIndices: number[] = [];
      for (let outIdx = 0; outIdx < numOutputs; outIdx++) {
        outputInternalIndices.push(internalBase + outIdx);
      }

      // Build static params based on routine type
      let staticParams: number[] = [];
      if (routineName === Routines.Cholesky) {
        const inputShape = routine.type.inputShapes[0];
        staticParams = [inputShape[inputShape.length - 1]];
      } else if (routineName === Routines.Sort) {
        const inputShape = routine.type.inputShapes[0];
        staticParams = [inputShape[inputShape.length - 1]];
      } else if (routineName === Routines.TriangularSolve) {
        const aShape = routine.type.inputShapes[0];
        const bShape = routine.type.inputShapes[1];
        const n = aShape[aShape.length - 1];
        const batchRows = bShape[bShape.length - 1];
        const numBatches = 1;
        const unitDiagonal = routine.params?.unitDiagonal ? 1 : 0;
        const lower = 0;
        staticParams = [n, batchRows, numBatches, unitDiagonal, lower];
      } else if (routineName === Routines.LU) {
        const inputShape = routine.type.inputShapes[0];
        const m = inputShape[inputShape.length - 2];
        const n = inputShape[inputShape.length - 1];
        staticParams = [m, n];
      } else if (routineName === Routines.Argsort) {
        const inputShape = routine.type.inputShapes[0];
        staticParams = [inputShape[inputShape.length - 1]];
      }

      steps.push({
        source,
        inputSlots,
        outputInternalIdx: internalBase,
        outputInternalIndices,
        routineCallInfo: {
          routineInfoIdx,
          staticParams,
        },
      });
    }
  }

  // Determine carry output sources
  const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
  const carryInputSlots = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  type LocalCarrySource = import("../backend/wasm").CarryOutputSource;
  const carryOutSources: LocalCarrySource[] = [];
  for (const slot of carryOutSlots) {
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      carryOutSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(
          `[wasm-scan] skipped, carry output slot ${slot} not produced by any execute step`,
        );
      return null;
    }
    carryOutSources.push({ type: "internal", internalIdx });
  }

  // Determine Y output sources
  const xsInputSlots = bodyProgram.inputs.slice(
    numConsts + numCarry,
    numConsts + numCarry + numX,
  );
  const yOutputSlots = bodyProgram.outputs.slice(numCarry);

  type LocalYSource = import("../backend/wasm").YOutputSource;
  const yOutputSources: LocalYSource[] = [];
  for (const slot of yOutputSlots) {
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      yOutputSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    const xsIdx = xsInputSlots.indexOf(slot);
    if (xsIdx !== -1) {
      yOutputSources.push({ type: "xs-passthrough", xsIdx });
      continue;
    }
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(`[wasm-scan] skipped, Y output slot ${slot} not found`);
      return null;
    }
    yOutputSources.push({ type: "internal", internalIdx });
  }

  // Build params and prepare the native scan
  const wasmBackend = backend as WasmBackend;
  if (!wasmBackend.prepareNativeScanGeneral) {
    if (DEBUG >= 2)
      console.log("[wasm-scan] backend has no prepareNativeScanGeneral");
    return null;
  }

  const params: NativeScanGeneralParams = {
    numConsts,
    constSizes,
    numCarry,
    carrySizes,
    numX,
    xsStrides,
    numY,
    ysStrides,
    internalSizes,
    steps,
    carryOutSources,
    yOutputSources,
    reverse,
    auxBufferSize: auxBufferSize > 0 ? auxBufferSize : undefined,
    elementSize: auxBufferSize > 0 ? elementSize : undefined,
    routineInfos: routineInfos.length > 0 ? routineInfos : undefined,
  };

  try {
    const exe = wasmBackend.prepareNativeScanGeneral(params);
    if (exe) {
      if (DEBUG >= 1) {
        const hasRoutines = steps.some((s) => s.source instanceof Routine);
        console.log(
          `[wasm-scan] SUCCESS! Using WASM native scan with ${steps.length} steps` +
            (hasRoutines ? " (includes routines)" : ""),
        );
      }
      return { executable: exe, internalSizes, params };
    }
    return null;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("[wasm-scan] preparation failed:", e);
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// WebGPU multi-kernel native scan (P3)
// ---------------------------------------------------------------------------

/**
 * Try to prepare a WebGPU native scan.
 *
 * Constraints:
 * - numCarry === numY or numY === 0 (each carry maps 1:1 to a Y output)
 * - No routine steps (only kernel steps)
 * - No passthrough carry (each carry output must come from a kernel step)
 * - No internal buffer dependencies (step inputs reference jaxpr inputs only)
 */
function tryPrepareWebGPUNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  executeSteps: ExecuteStep[],
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): {
  executable: Executable;
  params: NativeScanMultiParams;
} | null {
  // Constraint: numCarry === numY or numY === 0
  if (numY !== 0 && numCarry !== numY) {
    if (DEBUG >= 1)
      console.log(
        `[webgpu-scan] skipped, numCarry=${numCarry} !== numY=${numY}`,
      );
    return null;
  }

  // No routine steps
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      if (DEBUG >= 1)
        console.log(`[webgpu-scan] skipped, routine in scan body`);
      return null;
    }
  }

  const numInputs = numConsts + numCarry + numX;
  const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(
    bodyJaxpr,
    numConsts,
    numCarry,
    numX,
  );

  // Map step output JitIds to the execute step that produced them
  const outputToStep = new Map<JitId, ExecuteStep>();
  for (const step of executeSteps) {
    outputToStep.set(step.outputs[0], step);
  }

  // Check carry outputs: each must be produced by an execute step (no passthrough)
  const carryOutIds = bodyProgram.outputs.slice(0, numCarry);
  const carryInputIds = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  const stepToCarryIdx = new Map<ExecuteStep, number>();
  for (let ci = 0; ci < numCarry; ci++) {
    const outId = carryOutIds[ci];
    // Check passthrough
    if (carryInputIds.includes(outId)) {
      if (DEBUG >= 1)
        console.log(`[webgpu-scan] skipped, carry ${ci} is passthrough`);
      return null;
    }
    const step = outputToStep.get(outId);
    if (!step) {
      if (DEBUG >= 1)
        console.log(
          `[webgpu-scan] skipped, carry ${ci} not produced by execute step`,
        );
      return null;
    }
    stepToCarryIdx.set(step, ci);
  }

  // Check Y outputs: each must match the corresponding carry output (since numCarry === numY)
  if (numY > 0) {
    const yOutIds = bodyProgram.outputs.slice(numCarry);
    for (let yi = 0; yi < numY; yi++) {
      if (yOutIds[yi] !== carryOutIds[yi]) {
        if (DEBUG >= 1)
          console.log(
            `[webgpu-scan] skipped, y${yi} output slot differs from carry${yi}`,
          );
        return null;
      }
    }
  }

  // Check all step inputs reference only body jaxpr inputs (no internal buffer deps)
  for (const step of executeSteps) {
    for (const inputId of step.inputs) {
      if (inputId >= numInputs) {
        if (DEBUG >= 1)
          console.log(
            `[webgpu-scan] skipped, step has internal dep (input ${inputId} >= ${numInputs})`,
          );
        return null;
      }
    }
  }

  // Build NativeScanMultiStep[] — ordered by carry output index
  const orderedSteps: { carryIdx: number; step: ExecuteStep }[] = [];
  for (const [step, carryIdx] of stepToCarryIdx) {
    orderedSteps.push({ carryIdx, step });
  }
  orderedSteps.sort((a, b) => a.carryIdx - b.carryIdx);

  // Check for cross-step carry read-after-write hazards.
  // If step i writes to carry[ci], no later step j may read carry[ci],
  // because the WGSL shader writes carry[ci] immediately after step i
  // (before step j runs), so step j would see NEW values instead of OLD.
  // This catches JVP'd bilinear bodies where the tangent step reads the
  // primal carry. WASM handles this via shadow buffers; WebGPU falls back.
  for (let i = 0; i < orderedSteps.length; i++) {
    const writtenCarryGid = numConsts + orderedSteps[i].carryIdx;
    for (let j = i + 1; j < orderedSteps.length; j++) {
      const laterStep = orderedSteps[j].step;
      const laterSource = laterStep.source as Kernel;
      const laterReindexMap = laterStep.inputs;
      const laterExp = laterSource.outputs[0].exp.reindexGids(laterReindexMap);
      const readsWrittenCarry = laterExp.some(
        (e: AluExp) =>
          (e.op === AluOp.GlobalIndex || e.op === AluOp.GlobalView) &&
          e.arg[0] === writtenCarryGid,
      );
      if (readsWrittenCarry) {
        if (DEBUG >= 1)
          console.log(
            `[webgpu-scan] skipped, step ${j} reads carry[${orderedSteps[i].carryIdx}] written by step ${i}`,
          );
        return null;
      }
      // Also check reduction epilogue if present
      if (laterSource.outputs[0].reduction) {
        const laterReductionExp =
          laterSource.outputs[0].reduction.epilogue.reindexGids(
            laterReindexMap,
          );
        const readsInReduction = laterReductionExp.some(
          (e: AluExp) =>
            (e.op === AluOp.GlobalIndex || e.op === AluOp.GlobalView) &&
            e.arg[0] === writtenCarryGid,
        );
        if (readsInReduction) {
          if (DEBUG >= 1)
            console.log(
              `[webgpu-scan] skipped, step ${j} reduction reads carry[${orderedSteps[i].carryIdx}] written by step ${i}`,
            );
          return null;
        }
      }
    }
  }

  const multiSteps: NativeScanMultiStep[] = orderedSteps.map(
    ({ carryIdx, step }) => {
      const source = step.source as Kernel;
      // Reindex kernel expression gids from local args to scan buffer layout
      const reindexMap = step.inputs;
      const reindexedExp = source.outputs[0].exp.reindexGids(reindexMap);
      const reindexedReduction = source.outputs[0].reduction
        ? new Reduction(
            source.outputs[0].reduction.dtype,
            source.outputs[0].reduction.op,
            source.outputs[0].reduction.size,
            source.outputs[0].reduction.epilogue.reindexGids(reindexMap),
          )
        : undefined;
      const reindexedKernel = Kernel.single(
        numInputs,
        source.size,
        reindexedExp,
        reindexedReduction,
      );

      return {
        kernel: reindexedKernel,
        inputs: reindexMap.slice(),
        outputCarryIdx: carryIdx,
        outputSize: source.size as number,
      };
    },
  );

  const params: NativeScanMultiParams = {
    length,
    numConsts,
    constSizes,
    numCarry,
    carrySizes,
    numX,
    xsStrides,
    numY,
    ysStrides,
    steps: multiSteps,
    reverse,
  };

  // Call backend
  const webgpuBackend = backend as WebGPUBackend;
  const exe = webgpuBackend.prepareNativeScanMulti(params);
  if (!exe) return null;

  if (DEBUG >= 1) {
    console.log(
      `[webgpu-scan] SUCCESS! Using WebGPU native scan with ${multiSteps.length} steps`,
    );
  }
  return { executable: exe, params };
}

/**
 * Try to prepare a native scan executable (any backend).
 */
function tryPrepareNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number | Dim,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): {
  executable: Executable;
  internalSizes?: number[];
  params?: NativeScanGeneralParams | NativeScanMultiParams;
} | null {
  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as ExecuteStep[];
  if (executeSteps.length === 0) {
    if (DEBUG >= 1) console.log("[compiled-loop] skipped, no execute steps");
    return null;
  }

  if (backend.type === "wasm") {
    return tryPrepareWasmNativeScan(
      backend,
      bodyProgram,
      bodyJaxpr,
      executeSteps,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
    );
  }

  if (backend.type === "webgpu") {
    if (isSymbolicDim(length)) {
      if (DEBUG >= 1)
        console.log(
          "[compiled-loop] skipped, symbolic length not supported on WebGPU yet",
        );
      return null;
    }
    return tryPrepareWebGPUNativeScan(
      backend,
      bodyProgram,
      bodyJaxpr,
      executeSteps,
      length,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
    );
  }

  if (DEBUG >= 1)
    console.log(
      `[compiled-loop] skipped, backend=${backend.type} not supported yet`,
    );
  return null;
}

// ---------------------------------------------------------------------------
// Preencoded-routine scan (P4: WebGPU routine bodies)
// ---------------------------------------------------------------------------

/**
 * Try to prepare a preencoded scan for routine bodies (matmul, cholesky, etc.).
 *
 * Requirements:
 * - WebGPU backend
 * - Exactly 1 execute step in body that is a Routine
 * - numCarry === numY (passthrough pattern)
 * - Routine shader must not already use uniforms (e.g. Sort excluded)
 */
function tryPreparePreencodedScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): PreparedPreencodedScan | null {
  if (backend.type !== "webgpu") {
    if (DEBUG >= 2)
      console.log("Preencoded scan: skipped, unsupported backend");
    return null;
  }

  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as ExecuteStep[];
  if (executeSteps.length !== 1) {
    if (DEBUG >= 2)
      console.log(
        `Preencoded scan: skipped, ${executeSteps.length} execute steps (need exactly 1)`,
      );
    return null;
  }

  const execStep = executeSteps[0];
  if (!(execStep.source instanceof Routine)) {
    if (DEBUG >= 2) console.log("Preencoded scan: skipped, not a Routine");
    return null;
  }

  if (numCarry !== numY) {
    if (DEBUG >= 2)
      console.log(
        `Preencoded scan: skipped, numCarry=${numCarry} !== numY=${numY}`,
      );
    return null;
  }

  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry)
    .map((v) => v.aval);

  const carrySizes = carryAvals.map((a) => a.size * byteWidth(a.dtype));
  const xsElemStrides = xAvals.map((a) => a.size);
  const ysElemStrides = carryAvals.map((a) => a.size);

  if (!backend.prepareRoutineSync) {
    if (DEBUG >= 2)
      console.log(
        "Preencoded scan: skipped, backend has no prepareRoutineSync",
      );
    return null;
  }

  let bodyRoutineExe;
  try {
    bodyRoutineExe = backend.prepareRoutineSync(execStep.source);
  } catch (e) {
    if (DEBUG >= 2)
      console.warn("Preencoded scan: prepareRoutineSync failed:", e);
    return null;
  }

  const webgpuBackend = backend as WebGPUBackend;
  if (!webgpuBackend.preparePreencodedScan) {
    if (DEBUG >= 2)
      console.log(
        "Preencoded scan: skipped, backend has no preparePreencodedScan",
      );
    return null;
  }

  const preencodedScanParams = {
    length,
    carrySizes,
    xsElemStrides,
    ysElemStrides,
    bodyRoutine: bodyRoutineExe,
    numCarry,
    numX,
    numY,
    numConsts,
    reverse,
    routineInputJitIds: execStep.inputs,
    routineOutputJitIds: execStep.outputs,
  };

  try {
    const prepared = webgpuBackend.preparePreencodedScan(preencodedScanParams);
    if (prepared && DEBUG >= 1) {
      console.log(
        `Preencoded scan: SUCCESS! Using WebGPU preencoded scan for ${execStep.source.name}`,
      );
    }
    return prepared;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("Preencoded scan preparation failed:", e);
    }
    return null;
  }
}

/**
 * Choose a scan execution strategy.
 *
 * Priority: compiled-loop > preencoded-routine > fallback.
 */
export function planScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number | Dim,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
  acceptPath?: ScanPath | ScanPath[],
): ScanPlan {
  // Try compiled-loop (WASM native scan)
  const nativeScanResult = tryPrepareNativeScan(
    backend,
    bodyProgram,
    bodyJaxpr,
    length,
    numCarry,
    numConsts,
    numX,
    numY,
    reverse,
  );

  if (nativeScanResult) {
    const pathError = checkAcceptedPath("compiled-loop", acceptPath);
    if (pathError) throw new Error(pathError);
    return {
      path: "compiled-loop",
      executable: nativeScanResult.executable,
      internalSizes: nativeScanResult.internalSizes,
      params: nativeScanResult.params,
    };
  }

  // P4: preencoded-routine for WebGPU routine bodies
  // WebGPU preencoded-routine requires concrete length (uniform offsets)
  if (typeof length === "number") {
    const preencodedResult = tryPreparePreencodedScan(
      backend,
      bodyProgram,
      bodyJaxpr,
      length,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
    );

    if (preencodedResult) {
      const pathError = checkAcceptedPath("preencoded-routine", acceptPath);
      if (pathError) throw new Error(pathError);
      return {
        path: "preencoded-routine",
        preencodedParams: preencodedResult,
      };
    }
  }

  // Fallback: JS loop
  const dispatchCount = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ).length;
  const extraInfo = `${dispatchCount} dispatch${dispatchCount !== 1 ? "es" : ""} per iteration`;

  const pathError = checkAcceptedPath("fallback", acceptPath, extraInfo);
  if (pathError) throw new Error(pathError);

  return { path: "fallback", extraInfo };
}

// ---------------------------------------------------------------------------
// planAssociativeScan: decide execution strategy for associative scan (M7.2)
// ---------------------------------------------------------------------------

/**
 * Plan execution strategy for `lax.associativeScan`.
 *
 * Tries compiled-loop (WASM) first, falls back to JS Kogge-Stone.
 * The compiled WASM module takes N as a runtime parameter, enabling
 * polymorphic length: a single compilation can serve any input length.
 *
 * @param backend         Backend to compile for
 * @param bodyProgram     JIT-compiled body program (per-element, N-independent)
 * @param bodyJaxpr       Body jaxpr (per-element shapes, no scan axis)
 * @param numLeaves       Number of pytree leaves
 * @param numConsts       Number of constant inputs closed over by the body
 * @param reverse         Whether to reverse the scan direction
 */
export function planAssociativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  numLeaves: number,
  numConsts: number,
  reverse: boolean,
): AssocScanPlan {
  // Only WASM backend supports compiled-loop for assoc scan
  if (backend.type !== "wasm") {
    if (DEBUG >= 1) {
      console.log(
        `[assoc-scan] skipping compiled-loop: backend is ${backend.type}, not wasm`,
      );
    }
    return { path: "fallback" };
  }

  // Extract execute steps from body program
  const executeSteps = bodyProgram.steps.filter(
    (s): s is ExecuteStep => s.type === "execute",
  );

  if (executeSteps.length === 0) {
    if (DEBUG >= 1) {
      console.log("[assoc-scan] skipping compiled-loop: no execute steps");
    }
    return { path: "fallback" };
  }

  // Check that all steps are kernel-only (no routines for now)
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      if (DEBUG >= 1) {
        console.log(
          `[assoc-scan] skipping compiled-loop: routine ${step.source.name} in body`,
        );
      }
      return { path: "fallback" };
    }
  }

  // Check for non-execute steps that would make compilation impossible
  for (const step of bodyProgram.steps) {
    if (
      step.type !== "execute" &&
      step.type !== "malloc" &&
      step.type !== "free" &&
      step.type !== "recycle" &&
      step.type !== "incref"
    ) {
      if (DEBUG >= 1) {
        console.log(
          `[assoc-scan] skipping compiled-loop: unsupported step type "${step.type}"`,
        );
      }
      return { path: "fallback" };
    }
  }

  const numInputs = numConsts + 2 * numLeaves;

  // Build slot-to-internal mapping
  const slotToInternal = new Map<JitId, number>();
  const stepToInternalBase = new Map<number, number>();
  const internalSizes: number[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;
    if (!(source instanceof Kernel)) continue; // already checked above
    const internalIdx = internalSizes.length;
    stepToInternalBase.set(i, internalIdx);
    slotToInternal.set(step.outputs[0], internalIdx);
    internalSizes.push(
      (source.size as number) * byteWidth(source.outputs[0].dtype),
    );
  }

  // Determine leaf-to-internal mapping (which internal produced each output leaf)
  const leafToInternalIdx: number[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const outputId = bodyProgram.outputs[k];
    const internalIdx = slotToInternal.get(outputId);
    if (internalIdx === undefined) {
      // Output is a passthrough from input (not produced by a kernel).
      // This means fn(a,b) = a or fn(a,b) = b, which is unusual
      // but valid. Fall back for now.
      if (DEBUG >= 1) {
        console.log(
          `[assoc-scan] skipping compiled-loop: output leaf ${k} is passthrough`,
        );
      }
      return { path: "fallback" };
    }
    leafToInternalIdx.push(internalIdx);
  }

  // Build reindexed steps
  type LocalStep = import("../backend/wasm").GeneralScanStep;
  const steps: LocalStep[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source as Kernel;

    // Map each input JitId to a global input ID or internal buffer
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        inputSlots.push(inputId);
      } else {
        const intIdx = slotToInternal.get(inputId);
        if (intIdx === undefined) {
          if (DEBUG >= 1) {
            console.log(
              `[assoc-scan] skipping compiled-loop: unmapped slot ${inputId}`,
            );
          }
          return { path: "fallback" };
        }
        inputSlots.push(numInputs + intIdx);
      }
    }

    // Reindex kernel expression GlobalIndex IDs
    const reindexMap = inputSlots;
    const reindexedExp = source.outputs[0].exp.reindexGids(reindexMap);
    const reindexedReduction = source.outputs[0].reduction
      ? new Reduction(
          source.outputs[0].reduction.dtype,
          source.outputs[0].reduction.op,
          source.outputs[0].reduction.size,
          source.outputs[0].reduction.epilogue.reindexGids(reindexMap),
        )
      : undefined;
    const reindexedKernel = Kernel.single(
      numInputs + internalSizes.length,
      source.size,
      reindexedExp,
      reindexedReduction,
    );

    const internalBase = stepToInternalBase.get(i)!;
    steps.push({
      source: reindexedKernel,
      inputSlots,
      outputInternalIdx: internalBase,
    });
  }

  // Build per-element leaf sizes
  const leafElemSizes: number[] = [];
  for (let k = 0; k < numLeaves; k++) {
    const aval = bodyJaxpr.inBinders[numConsts + k].aval;
    leafElemSizes.push(aval.size * byteWidth(aval.dtype));
  }

  // Build const sizes
  const constSizes: number[] = [];
  for (let k = 0; k < numConsts; k++) {
    const aval = bodyJaxpr.inBinders[k].aval;
    constSizes.push(aval.size * byteWidth(aval.dtype));
  }

  const params: NativeAssocScanParams = {
    numConsts,
    constSizes,
    numLeaves,
    leafElemSizes,
    steps,
    internalSizes,
    reverse,
    leafToInternalIdx,
  };

  try {
    const wasmBackend = backend as WasmBackend;
    const exe = wasmBackend.prepareNativeAssociativeScan(params);
    if (DEBUG >= 1) {
      console.log(
        `[assoc-scan] SUCCESS! Using WASM compiled-loop with ${steps.length} step(s)`,
      );
    }
    return { path: "compiled-loop", executable: exe, params };
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("[assoc-scan] compilation failed:", e);
    }
    return { path: "fallback" };
  }
}
