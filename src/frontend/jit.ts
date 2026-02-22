// Handle Jit operations by translating Jaxprs into dispatched Kernels.

import {
  AluExp,
  AluOp,
  AluVar,
  byteWidth,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import { Backend, Slot } from "../backend";
import type { WasmBackend } from "../backend/wasm";
import {
  canCompileToMegaModule,
  compileToMegaModule,
  type WasmMegaModule,
} from "../backend/wasm/mega-module";
import { PPrint } from "../pprint";
import { Routine } from "../routine";
import {
  concreteDim,
  concreteShape,
  type Dim,
  dimProduct,
  hasSymbolicDims,
  isSymbolicDim,
  isSymbolicSize,
  Pair,
  resolveDim,
  resolveShape,
  resolveSizeExpr,
  ShapeTracker,
  type SizeExpr,
  sizeExprKey,
  sizeExprMul,
  unravelAlu,
} from "../shape";
import {
  DEBUG,
  deepEqual,
  FpHash,
  generalBroadcast,
  prod,
  range,
  rep,
} from "../utils";
import { aluCompare, Array as JaxArray, PendingExecute } from "./array";
import { _registerJitCacheDisposer } from "./check-leaks";
import { pool, poolTranspose, prepareConv } from "./convolution";
import {
  _associativeScanCoreImpl,
  Primitive,
  PrimitiveParams,
  promoteAvals,
  routinePrimitives,
  ShapedArray,
} from "./core";
import { Jaxpr, Lit, Var } from "./jaxpr";
import { executeScan } from "./scan-executor";
import type { AssocScanPlan, ScanPlan } from "./scan-plan";
import { planAssociativeScan, planScan } from "./scan-plan";
import type { ScanPath } from "../utils";

export type JitId = number;

export type JitStep =
  | {
      type: "execute";
      source: Kernel | Routine;
      inputs: JitId[]; // mapped to backend Slot
      outputs: JitId[]; // mapped to backend Slot
    }
  | {
      type: "malloc";
      size: SizeExpr;
      output: JitId;
    }
  | {
      type: "incref";
      input: JitId;
    }
  | {
      type: "free";
      input: JitId;
    }
  | {
      type: "recycle";
      input: JitId;
      output: JitId;
    }
  | {
      type: "scan";
      plan: ScanPlan;
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      length: number | Dim;
      numCarry: number;
      numConsts: number;
      numX: number;
      numY: number;
      reverse: boolean;
      consts: JitId[];
      initCarry: JitId[];
      xs: JitId[];
      xsAvals: ShapedArray[];
      outputs: JitId[];
    }
  | {
      type: "dus";
      dst: JitId; // the destination buffer (mutated via Mutate effect)
      src: JitId; // the source slice to copy
      output: JitId; // the result (same slot as dst if recycled, else separate)
      offsetBytes: number; // byte offset into dst where src is written
      sliceBytes: SizeExpr; // byte size of the src slice
      dstSizeBytes: SizeExpr; // total byte size of dst (= output size)
    }
  | {
      type: "scatter_add";
      target: JitId;
      indices: JitId;
      updates: JitId;
      output: JitId;
      axis: number;
      targetShape: number[];
      updatesLen: number; // number of updates along the scatter axis
      dtype: DType;
    }
  | {
      type: "assoc_scan";
      plan: AssocScanPlan;
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      numLeaves: number;
      numConsts: number;
      axis: number;
      reverse: boolean;
      consts: JitId[];
      elems: JitId[];
      constAvals: ShapedArray[];
      elemAvals: ShapedArray[];
      outputs: JitId[];
    };

/** Per-type step counts from {@link JitProgram.stepCounts}. */
export interface JitStepCounts {
  execute: number;
  malloc: number;
  free: number;
  recycle: number;
  incref: number;
  scan: number;
  dus: number;
  scatter_add: number;
  assoc_scan: number;
}

/**
 * Pool hints computed at JIT compile time. Tells the backend which buffer
 * sizes the program will allocate and how many bytes are live at peak, so the
 * pool can evict stale entries and cap retained memory.
 */
export interface PoolHints {
  /** Peak simultaneously-live bytes across all malloc/free/recycle steps. */
  readonly peakBytes: number;
  /** Set of every malloc'd byte size in the program (padded to 4-byte multiples). */
  readonly mallocSizes: ReadonlySet<number>;
}

/** Walk JitSteps to compute peak live bytes and the set of malloc sizes. */
function computePoolHints(steps: JitStep[]): PoolHints {
  const mallocSizes = new Set<number>();
  // Map JitId → padded byte size for live tracking.
  // WebGPU pads all allocations to 4-byte multiples; use the same padding here
  // so the budget and eviction sets match the actual GPUBuffer sizes.
  const sizeOf = new Map<JitId, number>();
  for (const s of steps) {
    if (s.type === "malloc") {
      // Resolve symbolic sizes using the current dim bindings (set during
      // jitCompile). If no bindings are available, skip — can't compute
      // concrete pool hints.
      let concreteSize: number;
      if (typeof s.size === "number") {
        concreteSize = s.size;
      } else if (_currentDimBindings) {
        concreteSize = resolveSizeExpr(s.size, _currentDimBindings);
      } else {
        continue;
      }
      const padded = Math.ceil(concreteSize / 4) * 4;
      sizeOf.set(s.output, padded);
      mallocSizes.add(padded);
    }
  }

  let liveBytes = 0;
  let peakBytes = 0;
  const live = new Map<JitId, number>(); // JitId → padded byte size (currently live)

  for (const s of steps) {
    switch (s.type) {
      case "malloc": {
        const padded = sizeOf.get(s.output)!;
        live.set(s.output, padded);
        liveBytes += padded;
        break;
      }
      case "free":
        liveBytes -= live.get(s.input) ?? 0;
        live.delete(s.input);
        break;
      case "recycle":
        // Size stays the same, just rename.
        live.set(s.output, live.get(s.input) ?? 0);
        live.delete(s.input);
        break;
      // execute, incref, scan don't change the set of live mallocs.
    }
    if (liveBytes > peakBytes) peakBytes = liveBytes;
  }

  return { peakBytes, mallocSizes };
}

/** Result of compiling a Jaxpr. Can be evaluated on a series of inputs. */
export class JitProgram {
  readonly poolHints: PoolHints;
  /** Cached mega-module: undefined = not attempted, null = unsupported. */
  private _megaModule?: WasmMegaModule | null;
  /** M6.2c: worker pool registration state for parallel mega-module dispatch.
   *  undefined = not attempted, false = registering, true = ready. */
  private _megaModulePoolReady?: boolean;

  constructor(
    readonly backend: Backend,
    readonly steps: JitStep[],
    readonly inputs: JitId[],
    readonly outputs: JitId[],
  ) {
    this.poolHints = computePoolHints(steps);
  }

  pprint(): PPrint {
    const steps: PPrint[] = this.steps.map((step) => {
      switch (step.type) {
        case "execute": {
          const inputsNice = step.inputs
            .map((id, i) => `${i}: %${id}`)
            .join(", ");
          const outputsNice = step.outputs.map((id) => `%${id}`).join(", ");
          const executeText = `execute (${inputsNice}) -> ${outputsNice}`;
          if (step.source instanceof Kernel) {
            return PPrint.pp(`${executeText}, kernel`).concat(
              step.source.pprint().indent(2),
            );
          } else if (step.source instanceof Routine) {
            return PPrint.pp(`${executeText}, routine ${step.source.name}`);
          } else {
            step.source satisfies never; // static check
            return PPrint.pp(executeText);
          }
        }
        case "malloc":
          return PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
        case "incref":
          return PPrint.pp(`incref ${step.input}`);
        case "free":
          return PPrint.pp(`free ${step.input}`);
        case "recycle":
          return PPrint.pp(`%${step.output} = recycle %${step.input}`);
        case "scan":
          return PPrint.pp(
            `scan [${step.plan.path}] length=${step.length} numCarry=${step.numCarry} ` +
              `numConsts=${step.numConsts} numX=${step.numX} numY=${step.numY}` +
              (step.reverse ? " reverse" : ""),
          );
        case "dus":
          return PPrint.pp(
            `%${step.output} = dus %${step.dst}[${step.offsetBytes}] <- %${step.src} (${step.sliceBytes} bytes)`,
          );
        case "scatter_add":
          return PPrint.pp(
            `%${step.output} = scatter_add %${step.target} %${step.indices} %${step.updates} axis=${step.axis}`,
          );
        case "assoc_scan":
          return PPrint.pp(
            `assoc_scan [${step.plan.path}] numLeaves=${step.numLeaves} numConsts=${step.numConsts} axis=${step.axis}` +
              (step.reverse ? " reverse" : ""),
          );
      }
    });
    const display = PPrint.prototype.concat(
      PPrint.pp(`device = ${this.backend.type}`),
      PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"),
      PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"),
      PPrint.pp("steps ="),
      PPrint.prototype.concat(...steps).indent(2),
    );
    return PPrint.pp("{ ").stack(display.stack(PPrint.pp(" }")));
  }

  toString(): string {
    return this.pprint().toString();
  }

  /**
   * Count steps by type. Useful for verifying fusion reduces dispatch count.
   *
   * @example
   * ```ts
   * const counts = program.stepCounts();
   * expect(counts.execute).toBe(1); // verify single dispatch after fusion
   * ```
   */
  stepCounts(): JitStepCounts {
    const counts: JitStepCounts = {
      execute: 0,
      malloc: 0,
      free: 0,
      recycle: 0,
      incref: 0,
      scan: 0,
      dus: 0,
      scatter_add: 0,
      assoc_scan: 0,
    };
    for (const step of this.steps) {
      counts[step.type]++;
    }
    return counts;
  }

  /**
   * Execute the JitProgram with the given inputs.
   * @param dimBindings Optional map of symbolic dimension names to concrete
   *   values. Required when the program contains symbolic sizes.
   */
  execute(
    inputs: Slot[],
    dimBindings?: ReadonlyMap<string, number>,
  ): { outputs: Slot[]; pending: PendingExecute[] } {
    // Tell the backend which buffer sizes we'll need and our peak memory,
    // so it can evict stale pool entries and cap retained bytes.
    this.backend.configurePool?.(this.poolHints);

    // Mega-module fast path (WASM only, kernel-only programs):
    // Compiles all steps into a single WASM function, eliminating
    // JS↔WASM boundary crossings between kernel dispatches.
    if (this.backend.type === "wasm") {
      if (this._megaModule === undefined) {
        this._megaModule = canCompileToMegaModule(this.steps)
          ? compileToMegaModule(this.steps, this.inputs, this.outputs)
          : null;
      }
      if (this._megaModule) {
        const wasmBackend = this.backend as WasmBackend;

        // M6.2c: parallel mega-module dispatch for programs with large kernels.
        // First call triggers async worker registration (falls through to
        // monolithic path). Once registered, subsequent calls use the parallel
        // path which fans out large kernels across workers.
        if (this._megaModulePoolReady === true) {
          // Worker pool ready — use parallel step-by-step dispatch
          const outputSlots = wasmBackend.executeMegaModuleParallelSync(
            this._megaModule,
            inputs,
          );
          return { outputs: outputSlots, pending: [] };
        }

        if (
          this._megaModulePoolReady === undefined &&
          wasmBackend.shouldUseParallelMegaModule(this._megaModule)
        ) {
          // First call: kick off async registration, fall through to
          // monolithic path for this invocation.
          this._megaModulePoolReady = false;
          const mm = this._megaModule;
          wasmBackend
            .registerMegaModuleOnPool(mm)
            .then(() => {
              this._megaModulePoolReady = true;
            })
            .catch(() => {
              // Registration failed — stay on monolithic path
              this._megaModulePoolReady = undefined;
            });
        }

        // Monolithic path: orchestrator (M6.2b) or direct execution
        const outputSlots = wasmBackend.executeMegaModule(
          this._megaModule,
          inputs,
        );
        return { outputs: outputSlots, pending: [] };
      }
    }

    const scope = new Map<JitId, Slot>();
    if (inputs.length !== this.inputs.length) {
      throw new TypeError(
        `Expected ${this.inputs.length} inputs, got ${inputs.length}`,
      );
    }
    for (const [i, id] of this.inputs.entries()) {
      scope.set(id, inputs[i]);
    }
    const pending: PendingExecute[] = [];
    for (const step of this.steps) {
      switch (step.type) {
        case "execute": {
          const inputs = step.inputs.map((id) => scope.get(id)!);
          const outputs = step.outputs.map((id) => scope.get(id)!);
          if (
            inputs.some((s) => s === undefined) ||
            outputs.some((s) => s === undefined)
          ) {
            throw new Error(`internal: JitProgram scope undefined`);
          }
          // Compute dynamicParams for kernels with symbolic dimensions
          let dynamicParams: number[] | undefined;
          if (
            step.source instanceof Kernel &&
            step.source.needsDynamicParams &&
            dimBindings
          ) {
            // dynamicParams[0]: resolved total size (concrete or symbolic)
            const resolvedSize = resolveSizeExpr(step.source.size, dimBindings);
            dynamicParams = [resolvedSize];
            // dynamicParams[1]: resolved reduction size (when symbolic)
            const re = step.source.outputs[0].reduction;
            if (re && isSymbolicSize(re.size)) {
              dynamicParams.push(resolveSizeExpr(re.size, dimBindings));
            }
          }
          pending.push(
            new PendingExecute(
              this.backend,
              step.source,
              inputs,
              outputs,
              dynamicParams,
            ),
          );
          break;
        }
        case "malloc": {
          const concreteSize =
            typeof step.size === "number"
              ? step.size
              : resolveSizeExpr(step.size, dimBindings!);
          const slot = this.backend.malloc(concreteSize);
          scope.set(step.output, slot);
          break;
        }
        case "incref": {
          const slot = scope.get(step.input)!;
          this.backend.incRef(slot);
          break;
        }
        case "free": {
          const slot = scope.get(step.input)!;
          this.backend.decRef(slot);
          scope.delete(step.input);
          break;
        }
        case "recycle": {
          // Reuse the same backend Slot for a new JitId — zero backend calls.
          const slot = scope.get(step.input)!;
          scope.delete(step.input);
          scope.set(step.output, slot);
          break;
        }
        case "scan": {
          // Flush pending ops before scan — scan needs materialized inputs
          for (const p of pending) {
            p.prepareSync();
            p.submit();
          }
          pending.length = 0;

          // Resolve slots from scope
          const constSlots = step.consts.map((id) => scope.get(id)!);
          const initCarrySlots = step.initCarry.map((id) => scope.get(id)!);
          const xsSlots = step.xs.map((id) => scope.get(id)!);
          const outputSlots = step.outputs.map((id) => scope.get(id)!);

          // IncRef consts and xs — executeScan borrows them
          for (const s of constSlots) this.backend.incRef(s);
          for (const s of xsSlots) this.backend.incRef(s);

          const result = executeScan({
            backend: this.backend,
            plan: step.plan,
            bodyProgram: step.bodyProgram,
            bodyJaxpr: step.bodyJaxpr,
            length:
              typeof step.length === "number"
                ? step.length
                : resolveDim(step.length, dimBindings!),
            numCarry: step.numCarry,
            numConsts: step.numConsts,
            numX: step.numX,
            numY: step.numY,
            reverse: step.reverse,
            constSlots,
            initCarrySlots,
            xsSlots,
            xsAvals: step.xsAvals,
            outputSlots,
          });

          // DecRef borrowed consts and xs
          for (const s of constSlots) this.backend.decRef(s);
          for (const s of xsSlots) this.backend.decRef(s);

          // Propagate scan pending ops
          pending.push(...result.pending);

          // Update scope with output slots
          for (let oi = 0; oi < step.outputs.length; oi++) {
            scope.set(step.outputs[oi], result.outputs[oi]);
          }
          break;
        }
        case "dus": {
          // Flush pending ops — DUS needs materialized inputs
          for (const p of pending) {
            p.prepareSync();
            p.submit();
          }
          pending.length = 0;

          const dstSlot = scope.get(step.dst)!;
          const srcSlot = scope.get(step.src)!;
          const outSlot = scope.get(step.output)!;

          // Zero-copy: if effectDrivenAllocate recycled dst → output,
          // skip the full copy (they share the same buffer).
          const concreteDstSize =
            typeof step.dstSizeBytes === "number"
              ? step.dstSizeBytes
              : resolveSizeExpr(step.dstSizeBytes, dimBindings!);
          const concreteSliceSize =
            typeof step.sliceBytes === "number"
              ? step.sliceBytes
              : resolveSizeExpr(step.sliceBytes, dimBindings!);

          if (dstSlot !== outSlot) {
            this.backend.copyBufferToBuffer!(
              dstSlot,
              0,
              outSlot,
              0,
              concreteDstSize,
            );
          }
          // Copy src slice into output at the byte offset
          this.backend.copyBufferToBuffer!(
            srcSlot,
            0,
            outSlot,
            step.offsetBytes,
            concreteSliceSize,
          );
          break;
        }
        case "scatter_add": {
          // Flush pending ops — scatter_add needs materialized inputs
          for (const p of pending) {
            p.prepareSync();
            p.submit();
          }
          pending.length = 0;

          const targetSlot = scope.get(step.target)!;
          const indicesSlot = scope.get(step.indices)!;
          const updatesSlot = scope.get(step.updates)!;
          const outSlot = scope.get(step.output)!;

          // Copy target to output if not already recycled
          const targetBytes = prod(step.targetShape) * byteWidth(step.dtype);
          if (targetSlot !== outSlot) {
            this.backend.copyBufferToBuffer!(
              targetSlot,
              0,
              outSlot,
              0,
              targetBytes,
            );
          }

          // Dispatch scatter_add kernel on the backend
          this.backend.dispatchScatterAdd!(
            outSlot,
            indicesSlot,
            updatesSlot,
            step.axis,
            step.targetShape,
            step.updatesLen,
            step.dtype,
          );
          break;
        }
        case "assoc_scan": {
          // Flush pending ops — assoc_scan needs materialized inputs
          for (const p of pending) {
            p.prepareSync();
            p.submit();
          }
          pending.length = 0;

          if (step.plan.path === "compiled-loop") {
            // Native WASM compiled-loop path: entire Kogge-Stone loop runs
            // in a single WASM invocation. N is a runtime parameter,
            // enabling polymorphic shapes (same compiled module for any N).
            const nDim = step.elemAvals[0].shape[step.axis];
            const N = isSymbolicDim(nDim)
              ? concreteDim(
                  resolveShape([nDim], dimBindings!)[0],
                  "assoc_scan N",
                )
              : (nDim as number);

            const constSlots = step.consts.map((id) => scope.get(id)!);
            const elemSlots = step.elems.map((id) => scope.get(id)!);
            const outputSlots = step.outputs.map((id) => scope.get(id)!);

            (this.backend as WasmBackend).dispatchNativeAssociativeScan(
              step.plan.executable,
              step.plan.params,
              N,
              constSlots,
              elemSlots,
              outputSlots,
            );
          } else {
            // Fallback: JS Kogge-Stone loop via vmap + evalJaxpr
            // Create Array wrappers from input slots.
            // Resolve symbolic shapes at execution time via dimBindings.
            const resolveAvalShape = (shape: Dim[]): number[] =>
              hasSymbolicDims(shape)
                ? concreteShape(resolveShape(shape, dimBindings!))
                : (shape as number[]);

            const constSlots = step.consts.map((id) => scope.get(id)!);
            const constArrays = constSlots.map((slot, i) => {
              this.backend.incRef(slot);
              const aval = step.constAvals[i];
              return new JaxArray({
                source: slot,
                st: ShapeTracker.fromShape(resolveAvalShape(aval.shape)),
                dtype: aval.dtype,
                weakType: aval.weakType,
                backend: this.backend,
                committed: false,
              });
            });

            const elemSlots = step.elems.map((id) => scope.get(id)!);
            const elemArrays = elemSlots.map((slot, i) => {
              this.backend.incRef(slot);
              const aval = step.elemAvals[i];
              return new JaxArray({
                source: slot,
                st: ShapeTracker.fromShape(resolveAvalShape(aval.shape)),
                dtype: aval.dtype,
                weakType: aval.weakType,
                backend: this.backend,
                committed: false,
              });
            });

            // Call Kogge-Stone core impl with body jaxpr + consts.
            // The core impl uses vmap internally to vectorize the element-level
            // body jaxpr over the batch dimension.
            const results = _associativeScanCoreImpl!(
              step.bodyJaxpr,
              constArrays,
              elemArrays,
              step.numLeaves,
              step.axis,
              step.reverse,
            );

            // Extract slots from results and store in output scope.
            // Must flush pending ops first — result arrays from associativeScanCore
            // have PendingExecute items (from concat/kernels) that haven't been
            // submitted yet. Without flushing, the slot buffer contains zeros.
            for (let i = 0; i < step.numLeaves; i++) {
              const resultArr = results[i] as JaxArray;
              resultArr._flushPendingSync();
              const slot = resultArr._realizeSource();
              this.backend.incRef(slot);
              // Free the pre-allocated output buffer before overwriting scope
              const oldSlot = scope.get(step.outputs[i]);
              if (oldSlot !== undefined) this.backend.decRef(oldSlot);
              scope.set(step.outputs[i], slot);
              resultArr.dispose();
            }

            // Dispose input Array wrappers
            for (const a of constArrays) a.dispose();
            for (const a of elemArrays) a.dispose();
          }
          break;
        }
        default:
          step satisfies never;
      }
    }
    return {
      outputs: this.outputs.map((id) => scope.get(id)!),
      pending,
    };
  }
}

/** Check whether a JitStep references the given JitId as input or output. */
function stepUsesId(step: JitStep, id: JitId): boolean {
  switch (step.type) {
    case "execute":
      return step.inputs.includes(id) || step.outputs.includes(id);
    case "malloc":
      return step.output === id;
    case "scan":
      return (
        step.outputs.includes(id) ||
        step.consts.includes(id) ||
        step.initCarry.includes(id) ||
        step.xs.includes(id)
      );
    case "dus":
      return step.dst === id || step.src === id || step.output === id;
    case "scatter_add":
      return (
        step.target === id ||
        step.indices === id ||
        step.updates === id ||
        step.output === id
      );
    case "assoc_scan":
      return (
        step.outputs.includes(id) ||
        step.consts.includes(id) ||
        step.elems.includes(id)
      );
    default:
      return false;
  }
}

class JitProgramBuilder {
  backend: Backend;
  #nextId: number;
  steps: JitStep[];

  constructor(backend: Backend, nargs: number) {
    this.backend = backend;
    this.#nextId = nargs;
    this.steps = [];
  }

  pushLit(lit: Lit): JitId {
    const kernel = Kernel.single(
      0,
      lit.aval.size,
      AluExp.const(lit.dtype, lit.value),
    );
    return this.pushKernel(kernel, []);
  }

  pushBuffer(size: SizeExpr): JitId {
    const id = this.#nextId++;
    this.steps.push({
      type: "malloc",
      size,
      output: id,
    });
    return id;
  }

  pushKernel(kernel: Kernel, inputs: JitId[]): JitId {
    const id = this.pushBuffer(kernel.outputs[0].bytes);
    this.steps.push({
      type: "execute",
      source: kernel,
      inputs,
      outputs: [id],
    });
    return id;
  }

  /**
   * Push a multi-output kernel. Allocates one buffer per output and emits a
   * single execute step. Returns array of JitIds, one per kernel output.
   */
  pushMultiKernel(kernel: Kernel, inputs: JitId[]): JitId[] {
    const ids: JitId[] = [];
    for (const o of kernel.outputs) {
      ids.push(this.pushBuffer(o.bytes));
    }
    this.steps.push({
      type: "execute",
      source: kernel,
      inputs,
      outputs: ids,
    });
    return ids;
  }

  pushRoutine(routine: Routine, inputs: JitId[], outputs: JitId[]): void {
    this.steps.push({
      type: "execute",
      source: routine,
      inputs,
      outputs,
    });
  }

  pushIncref(id: JitId): void {
    this.steps.push({
      type: "incref",
      input: id,
    });
  }

  pushFree(id: JitId): void {
    // Should be paired with the output of pushKernel() when last used.
    this.steps.push({
      type: "free",
      input: id,
    });
  }

  /**
   * Effect-driven buffer lifecycle allocation.
   *
   * Replaces the two-pass `insertFreeSteps` + `recycleBuffers` with a
   * single-pass algorithm that uses liveness analysis for optimal buffer
   * reuse. Key improvement: can recycle buffers across execute/scan step
   * boundaries, catching opportunities the old adjacent-pair scanner missed.
   *
   * Algorithm:
   * 1. Collect all malloc'd JitIds and their byte sizes.
   * 2. Compute last-use step index for each (excluding program outputs).
   * 3. Forward pass: when a JitId dies, add to a free pool (keyed by size).
   *    When a malloc is needed, check the pool → emit recycle or malloc.
   * 4. Emit free steps for any remaining pooled buffers.
   */
  effectDrivenAllocate(outputIds: JitId[]): void {
    // Phase 1: Collect all malloc'd JitIds and their sizes
    const mallocSizes = new Map<JitId, SizeExpr>();
    for (const s of this.steps) {
      if (s.type === "malloc") mallocSizes.set(s.output, s.size);
    }

    // Phase 2: Compute last-use step index for each non-output malloc'd JitId
    const outputSet = new Set(outputIds);
    const lastUse = new Map<JitId, number>();
    for (const [id] of mallocSizes) {
      if (outputSet.has(id)) continue;
      for (let j = this.steps.length - 1; j >= 0; j--) {
        if (stepUsesId(this.steps[j], id)) {
          lastUse.set(id, j);
          break;
        }
      }
    }

    // Pre-compute which JitIds die after each step index
    const pendingFree = new Map<number, JitId[]>();
    for (const [id, stepIdx] of lastUse) {
      let list = pendingFree.get(stepIdx);
      if (!list) {
        list = [];
        pendingFree.set(stepIdx, list);
      }
      list.push(id);
    }

    // Phase 3: Single forward pass with free pool for cross-boundary recycling
    const freePool = new Map<string | number, JitId[]>(); // sizeExprKey → available JitIds
    const newSteps: JitStep[] = [];
    let recycleCount = 0;

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];

      if (step.type === "malloc") {
        // Check free pool for a same-size buffer to recycle
        const key = sizeExprKey(step.size);
        const pool = freePool.get(key);
        if (pool && pool.length > 0) {
          const reusedId = pool.pop()!;
          newSteps.push({
            type: "recycle",
            input: reusedId,
            output: step.output,
          });
          recycleCount++;
        } else {
          newSteps.push(step);
        }
      } else {
        newSteps.push(step);
      }

      // After this step, release any JitIds whose lifetime ends here
      const dying = pendingFree.get(i);
      if (dying) {
        for (const id of dying) {
          const size = mallocSizes.get(id);
          if (size === undefined) continue;
          const key = sizeExprKey(size);
          let pool = freePool.get(key);
          if (!pool) {
            pool = [];
            freePool.set(key, pool);
          }
          pool.push(id);
        }
      }
    }

    // Phase 4: Emit free steps for any pooled buffers not recycled
    for (const [, pool] of freePool) {
      for (const id of pool) {
        newSteps.push({ type: "free", input: id });
      }
    }

    this.steps = newSteps;
    if (DEBUG >= 1 && recycleCount > 0) {
      console.info(`jit: effect-driven recycled ${recycleCount} buffer(s)`);
    }
  }
}

type JitValue =
  | { type: "imm"; arg: JitId } // Immediate
  | { type: "exp"; exp: AluExp; args: JitId[] } // Expression, lazily fused
  | { type: "red"; exp: AluExp; reduction: Reduction; args: JitId[] }; // Reduction + epilogue

const jitCompileCache = new Map<string, JitProgram>();

/**
 * Clear the internal JIT compilation cache. Called by `_disposeAllJitCaches()`
 * in jaxpr.ts during leak checking.
 * @internal
 */
export function _clearJitCompileCache(): void {
  jitCompileCache.clear();
}

// Register with jaxpr.ts so checkLeaks.stop() can flush this cache.
_registerJitCacheDisposer(_clearJitCompileCache);

/**
 * Module-level dim bindings for jitRules to use when resolving symbolic shapes
 * to concrete values for ShapeTracker operations. Set by jitCompile().
 * @internal
 */
let _currentDimBindings: ReadonlyMap<string, number> | undefined;

export function jitCompile(
  backend: Backend,
  jaxpr: Jaxpr,
  dimBindings?: ReadonlyMap<string, number>,
): JitProgram {
  const cacheKey = backend.type + "," + FpHash.hash(jaxpr);

  const cached = jitCompileCache.get(cacheKey);
  if (cached) return cached;

  // Set module-level dim bindings for jitRules to resolve symbolic shapes
  // to concrete values in ShapeTracker operations.
  _currentDimBindings = dimBindings;

  try {
    if (DEBUG >= 1) {
      console.info("=========== JIT Compile ===========\n" + jaxpr.toString());
    }

    jaxpr = jaxpr.flatten().simplify();
    const nargs = jaxpr.inBinders.length;
    const builder = new JitProgramBuilder(backend, nargs);

    const blackNodes = splitGraphDataflow(backend, jaxpr);

    /** Set concreteSizeHint on a kernel if dimBindings are available. */
    const setConcreteHint = (kernel: Kernel, size: SizeExpr): void => {
      if (_currentDimBindings && isSymbolicSize(size)) {
        kernel.concreteSizeHint = resolveSizeExpr(
          size,
          _currentDimBindings,
        ) as number;
      }
      // Also set concreteHint on any symbolic reductions.
      if (_currentDimBindings) {
        for (const o of kernel.outputs) {
          if (o.reduction && isSymbolicSize(o.reduction.size)) {
            o.reduction.concreteHint = resolveSizeExpr(
              o.reduction.size,
              _currentDimBindings,
            ) as number;
          }
        }
      }
    };

    // Initialize jaxpr inBinders.
    const ctx = new Map<Var, JitValue>();
    for (let i = 0; i < nargs; i++) {
      const v = jaxpr.inBinders[i];
      ctx.set(v, { type: "imm", arg: i }); // JitId i = input #i
    }

    // ---- Pending kernel batching ----
    // Defer black-node kernel dispatch to batch same-size non-reduction kernels
    // into multi-output Kernel steps, reducing dispatch overhead.
    interface PendingKernelEntry {
      outVar: Var;
      exp: AluExp;
      reduction: Reduction | undefined;
      inputArgs: JitId[];
      size: SizeExpr;
    }
    const pendingKernels: PendingKernelEntry[] = [];

    /** Flush all pending kernels into JitSteps. Groups same-size non-reduction
     *  kernels with identical inputArgs into multi-output Kernel steps. */
    const flushPendingKernels = (): void => {
      if (pendingKernels.length === 0) return;

      // If backend doesn't support multi-output, emit all as solo
      if (!backend.capabilities.multiOutputKernel) {
        for (const entry of pendingKernels) {
          const kernel = Kernel.single(
            entry.inputArgs.length,
            entry.size,
            entry.exp,
            entry.reduction,
          );
          setConcreteHint(kernel, entry.size);
          const outId = builder.pushKernel(kernel, entry.inputArgs);
          ctx.set(entry.outVar, { type: "imm", arg: outId });
        }
        pendingKernels.length = 0;
        return;
      }

      // Group by size + inputArgs identity (sorted JitIds as key).
      // Only non-reduction kernels with matching size AND inputArgs can merge.
      const groups = new Map<string, PendingKernelEntry[]>();
      const soloEntries: PendingKernelEntry[] = [];

      for (const entry of pendingKernels) {
        if (entry.reduction) {
          // Kernels with reductions always emit solo
          soloEntries.push(entry);
        } else {
          const key = `${sizeExprKey(entry.size)}:${entry.inputArgs.join(",")}`;
          const group = groups.get(key);
          if (group) group.push(entry);
          else groups.set(key, [entry]);
        }
      }

      // Emit multi-output kernels for groups with > 1 entries
      for (const [, group] of groups) {
        if (group.length === 1) {
          soloEntries.push(group[0]);
        } else {
          // Check maxArgs: nargs + numOutputs <= backend limit
          const maxArgs = backend.maxArgs;
          if (group[0].inputArgs.length + group.length > maxArgs) {
            // Too many buffers — fall back to individual kernels
            for (const e of group) soloEntries.push(e);
          } else {
            const nargs = group[0].inputArgs.length;
            const size = group[0].size;
            const outputDescs = group.map((e) => ({
              exp: e.exp,
              reduction: e.reduction,
            }));
            const kernel = Kernel.multi(nargs, size, outputDescs);
            setConcreteHint(kernel, size);
            const outIds = builder.pushMultiKernel(kernel, group[0].inputArgs);
            for (let j = 0; j < group.length; j++) {
              ctx.set(group[j].outVar, { type: "imm", arg: outIds[j] });
            }
          }
        }
      }

      // Emit solo kernels
      for (const entry of soloEntries) {
        const kernel = Kernel.single(
          entry.inputArgs.length,
          entry.size,
          entry.exp,
          entry.reduction,
        );
        setConcreteHint(kernel, entry.size);
        const outId = builder.pushKernel(kernel, entry.inputArgs);
        ctx.set(entry.outVar, { type: "imm", arg: outId });
      }

      pendingKernels.length = 0;
    };

    // Now run each primitive through a set of rules, mirroring implRules.
    for (let i = 0; i < jaxpr.eqns.length; i++) {
      const eqn = jaxpr.eqns[i];

      // Handle Primitive.Scan specially — it compiles the body jaxpr and
      // emits a single "scan" JitStep with a ScanPlan.
      if (eqn.primitive === Primitive.Scan) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<typeof Primitive.Scan>;
        const {
          jaxpr: bodyJaxpr,
          numCarry,
          numConsts,
          length,
          reverse,
          acceptPath,
        } = params;
        const numX = bodyJaxpr.inBinders.length - numConsts - numCarry;
        const numY = bodyJaxpr.outs.length - numCarry;

        // Resolve input JitIds (all must be "imm" — black nodes)
        const allInputIds: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error("jit: scan primitive input is not imm");
            }
            allInputIds.push(jv.arg);
          } else if (input instanceof Lit) {
            allInputIds.push(builder.pushLit(input));
          }
        }

        const constsIds = allInputIds.slice(0, numConsts);
        const initCarryIds = allInputIds.slice(numConsts, numConsts + numCarry);
        const xsIds = allInputIds.slice(numConsts + numCarry);

        // xs avals (actual shapes from the jaxpr, include leading length dim)
        const xsAvals: ShapedArray[] = [];
        const xsInputs = eqn.inputs.slice(numConsts + numCarry);
        for (const input of xsInputs) {
          xsAvals.push(input.aval);
        }

        // Allocate output buffers: [carry_out..., stacked_ys...]
        const outputIds: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputIds.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }

        // Compile body jaxpr
        const bodyProgram = jitCompile(backend, bodyJaxpr);

        // Determine scan plan
        const scanPlan = planScan(
          backend,
          bodyProgram,
          bodyJaxpr,
          length,
          numCarry,
          numConsts,
          numX,
          numY,
          reverse,
          acceptPath as ScanPath | ScanPath[] | undefined,
        );

        // Compute per-slice xsAvals (without leading length dimension)
        const xsSliceAvals = xsAvals.map(
          (a) => new ShapedArray(a.shape.slice(1), a.dtype, a.weakType),
        );

        builder.steps.push({
          type: "scan",
          plan: scanPlan,
          bodyProgram,
          bodyJaxpr,
          length,
          numCarry,
          numConsts,
          numX,
          numY,
          reverse,
          consts: constsIds,
          initCarry: initCarryIds,
          xs: xsIds,
          xsAvals: xsSliceAvals,
          outputs: outputIds,
        });
        continue;
      }

      // DynamicUpdateSlice: compile to a zero-copy DUS step.
      // The Mutate effect on dst allows effectDrivenAllocate to recycle
      // dst → output, avoiding the full buffer copy.
      if (eqn.primitive === Primitive.DynamicUpdateSlice) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.DynamicUpdateSlice
        >;
        const { offset, axis } = params;

        if (axis !== 0) {
          throw new Error(
            "DynamicUpdateSlice JIT: only axis=0 is currently supported",
          );
        }

        // Resolve input JitIds
        const dstInput = eqn.inputs[0];
        const srcInput = eqn.inputs[1];
        let dstId: JitId;
        let srcId: JitId;

        if (dstInput instanceof Var) {
          const jv = ctx.get(dstInput)!;
          if (jv.type !== "imm") {
            throw new Error("jit: DUS dst input is not imm");
          }
          dstId = jv.arg;
        } else {
          dstId = builder.pushLit(dstInput as Lit);
        }

        if (srcInput instanceof Var) {
          const jv = ctx.get(srcInput)!;
          if (jv.type !== "imm") {
            throw new Error("jit: DUS src input is not imm");
          }
          srcId = jv.arg;
        } else {
          srcId = builder.pushLit(srcInput as Lit);
        }

        const outVar = eqn.outBinders[0];
        const elemBytes = byteWidth(outVar.aval.dtype);
        const innerSize = (outVar.aval.shape as number[])
          .slice(1)
          .reduce((a, b) => a * b, 1);
        const offsetBytes = offset * innerSize * elemBytes;
        const sliceBytes = sizeExprMul(srcInput.aval.sizeExpr, elemBytes);
        const dstSizeBytes = sizeExprMul(outVar.aval.sizeExpr, elemBytes);

        // Allocate output buffer (same size as dst — recycling may reclaim dst)
        const outId = builder.pushBuffer(dstSizeBytes);
        ctx.set(outVar, { type: "imm", arg: outId });

        builder.steps.push({
          type: "dus",
          dst: dstId,
          src: srcId,
          output: outId,
          offsetBytes,
          sliceBytes,
          dstSizeBytes,
        });
        continue;
      }

      // Handle Primitive.AssociativeScan — creates an assoc_scan JitStep
      // that runs the Kogge-Stone loop at execution time.
      if (eqn.primitive === Primitive.AssociativeScan) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.AssociativeScan
        >;
        const { jaxpr: bodyJaxpr, numLeaves, axis, reverse } = params;
        const numConsts = eqn.inputs.length - numLeaves;

        // Resolve input JitIds
        const allInputIds: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error(
                "jit: AssociativeScan primitive input is not imm",
              );
            }
            allInputIds.push(jv.arg);
          } else if (input instanceof Lit) {
            allInputIds.push(builder.pushLit(input));
          }
        }

        const constsIds = allInputIds.slice(0, numConsts);
        const elemsIds = allInputIds.slice(numConsts);

        const constAvals = eqn.inputs
          .slice(0, numConsts)
          .map((v) => v.aval as ShapedArray);
        const elemAvals = eqn.inputs
          .slice(numConsts)
          .map((v) => v.aval as ShapedArray);

        // Allocate output buffers
        const outputIds: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputIds.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }

        // Compile the body jaxpr → JitProgram (for fallback path)
        const bodyProgram = jitCompile(backend, bodyJaxpr);

        // Plan the assoc scan — try compiled-loop (WASM), fallback otherwise
        const assocPlan = planAssociativeScan(
          backend,
          bodyProgram,
          bodyJaxpr,
          numLeaves,
          numConsts,
          reverse,
        );

        builder.steps.push({
          type: "assoc_scan",
          plan: assocPlan,
          bodyProgram,
          bodyJaxpr,
          numLeaves,
          numConsts,
          axis,
          reverse,
          consts: constsIds,
          elems: elemsIds,
          constAvals,
          elemAvals,
          outputs: outputIds,
        });
        continue;
      }

      // ScatterAdd: compile to a scatter_add JitStep.
      // The Mutate effect on target allows effectDrivenAllocate to recycle
      // target → output, avoiding a full buffer copy when sizes match.
      if (eqn.primitive === Primitive.ScatterAdd) {
        flushPendingKernels();
        const params = eqn.params as PrimitiveParams<
          typeof Primitive.ScatterAdd
        >;
        const { axis } = params;

        // Resolve input JitIds: target, indices, updates
        const resolveInput = (input: Var | Lit): JitId => {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error("jit: ScatterAdd input is not imm");
            }
            return jv.arg;
          }
          return builder.pushLit(input as Lit);
        };

        const targetId = resolveInput(eqn.inputs[0]);
        const indicesId = resolveInput(eqn.inputs[1]);
        const updatesId = resolveInput(eqn.inputs[2]);

        const outVar = eqn.outBinders[0];
        const targetShape = eqn.inputs[0].aval.shape as number[];
        const updatesLen = eqn.inputs[2].aval.shape[axis] as number;
        const dtype = outVar.aval.dtype;
        const outSizeBytes = sizeExprMul(
          outVar.aval.sizeExpr,
          byteWidth(dtype),
        );

        // Allocate output buffer (same size as target — recycling may reclaim target)
        const outId = builder.pushBuffer(outSizeBytes);
        ctx.set(outVar, { type: "imm", arg: outId });

        builder.steps.push({
          type: "scatter_add",
          target: targetId,
          indices: indicesId,
          updates: updatesId,
          output: outId,
          axis,
          targetShape,
          updatesLen,
          dtype,
        });
        continue;
      }

      // If this is a routine, construct and dispatch the routine.
      if (routinePrimitives.has(eqn.primitive)) {
        flushPendingKernels();
        // The rest of the code collaborates to make sure that all inputs to a
        // routine are "imm" (black node, dispatched) and so is itself.
        const routine = new Routine(
          routinePrimitives.get(eqn.primitive)!,
          {
            inputShapes: eqn.inputs.map((x) => x.aval.shape as number[]),
            inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
            outputShapes: eqn.outBinders.map((x) => x.aval.shape as number[]),
            outputDtypes: eqn.outBinders.map((x) => x.aval.dtype),
          },
          eqn.params as any,
        );
        const inputs: JitId[] = [];
        for (const input of eqn.inputs) {
          if (input instanceof Var) {
            const jv = ctx.get(input)!;
            if (jv.type !== "imm") {
              throw new Error(
                `jit: routine primitive ${eqn.primitive} input is not imm`,
              );
            }
            inputs.push(jv.arg);
          } else if (input instanceof Lit) {
            inputs.push(builder.pushLit(input));
          }
        }
        const outputs: JitId[] = [];
        for (const outVar of eqn.outBinders) {
          const outId = builder.pushBuffer(
            sizeExprMul(outVar.aval.sizeExpr, byteWidth(outVar.aval.dtype)),
          );
          outputs.push(outId);
          ctx.set(outVar, { type: "imm", arg: outId });
        }
        builder.pushRoutine(routine, inputs, outputs);
        continue;
      }

      // If any input references a pending kernel's output, flush first so the
      // output is materialized and available in ctx.
      if (pendingKernels.length > 0) {
        for (const input of eqn.inputs) {
          if (
            input instanceof Var &&
            !ctx.has(input) &&
            pendingKernels.some((pk) => pk.outVar === input)
          ) {
            flushPendingKernels();
            break;
          }
        }
      }

      // Transform each input into an AluExp to start, and normalize any arguments
      // as needed.
      const inputExps: AluExp[] = []; // len(inputs)
      const inputAvals: ShapedArray[] = []; // len(inputs)
      const inputArgs: JitId[] = [];

      let inputReduction: (JitValue & { type: "red" }) | null = null;

      // May need to reindex gids to match order, returns array of new gids.
      const addArgs = (args: JitId[]): number[] => {
        const newGids: number[] = [];
        for (const jitId of args) {
          let newGid = inputArgs.indexOf(jitId);
          if (newGid === -1) {
            newGid = inputArgs.length;
            inputArgs.push(jitId);
          }
          newGids.push(newGid);
        }
        return newGids;
      };

      for (const input of eqn.inputs) {
        if (input instanceof Var) {
          const jv = ctx.get(input)!;
          if (jv.type === "exp") {
            const newGids = addArgs(jv.args);
            inputExps.push(jv.exp.reindexGids(newGids));
          } else if (jv.type === "imm") {
            const [gid] = addArgs([jv.arg]);
            // For symbolic shapes, resolve to concrete using dimBindings for
            // ShapeTracker operations. The kernel.size stays symbolic (SizeExpr)
            // and is resolved at execution time via dynamicParams.
            const shape =
              hasSymbolicDims(input.aval.shape) && _currentDimBindings
                ? (resolveShape(
                    input.aval.shape,
                    _currentDimBindings,
                  ) as number[])
                : (input.aval.shape as number[]);
            const st = ShapeTracker.fromShape(shape);
            const indices = unravelAlu(st.shape, AluVar.gidx);
            inputExps.push(
              AluExp.globalView(input.aval.dtype, gid, st, indices),
            );
          } else if (jv.type === "red") {
            // Special case: We are consuming a 'red' JitValue, so we must be in the
            // fused epilogue of a reduction.
            if (inputReduction)
              throw new Error("jit: unexpected, multiple red inputs");
            const newGids = addArgs(jv.args);
            inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
            inputReduction = jv;
          } else {
            jv satisfies never; // static check
          }
          inputAvals.push(input.aval);
        } else if (input instanceof Lit) {
          inputExps.push(AluExp.const(input.dtype, input.value));
          inputAvals.push(input.aval);
        } else {
          throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
        }
      }

      // Produce a new expression and/or reduction for the operation based on the
      // jit() implementation of the primitive.
      const rule = jitRules[eqn.primitive];
      if (!rule)
        throw new TypeError(
          `JIT not implemented for primitive ${eqn.primitive}`,
        );

      let exp: AluExp[];
      let reduction: Reduction | undefined;

      if (inputReduction) {
        // Special case: we are in the fused epilogue of a reduction.
        const jv = inputReduction;
        const newEpilogue = rule(inputExps, inputAvals, eqn.params as any)
          .exp[0];
        exp = [jv.exp.reindexGids(addArgs(jv.args))];
        reduction = new Reduction(
          jv.reduction.dtype,
          jv.reduction.op,
          jv.reduction.size,
          newEpilogue,
        );
      } else {
        const ruleOutput = rule(inputExps, inputAvals, eqn.params as any);
        exp = ruleOutput.exp;
        reduction = ruleOutput.reduction;
      }

      // Then dispatch the kernel, if it is a "black" node as determined from
      // dataflow analysis above. Defer to pending kernel batching for potential
      // multi-output fusion.
      for (let i = 0; i < eqn.outBinders.length; i++) {
        const outVar = eqn.outBinders[i];
        if (blackNodes.has(outVar)) {
          const size = outVar.aval.sizeExpr;
          pendingKernels.push({
            outVar,
            exp: exp[i],
            reduction,
            inputArgs: [...inputArgs],
            size,
          });
        } else if (reduction) {
          // Reduction but not black, means it will have an epilogue.
          ctx.set(outVar, {
            type: "red",
            exp: exp[i],
            reduction,
            args: inputArgs,
          });
        } else {
          // Otherwise, fuse the kernel into the next expression.
          ctx.set(outVar, { type: "exp", exp: exp[i], args: inputArgs });
        }
      }
    }

    // Flush any remaining pending kernels before output collection.
    flushPendingKernels();

    // Finally, loop through the outputs.
    const outputIds: JitId[] = [];
    for (const out of jaxpr.outs) {
      if (out instanceof Var) {
        const jitValue = ctx.get(out)!;
        if (jitValue.type !== "imm")
          throw new Error("internal: Expected imm, since outs are black nodes");
        outputIds.push(jitValue.arg);
      } else if (out instanceof Lit) {
        outputIds.push(builder.pushLit(out));
      } else {
        out satisfies never; // static check
      }
    }

    // Each output should have its own backend reference. If an output slot is
    // returned twice, or if an input is returned directly, insert "incref" steps
    // to balance the books.
    const outputNeedsRef = new Set<JitId>(range(nargs)); // inputs
    for (const outputId of outputIds) {
      if (outputNeedsRef.has(outputId)) {
        builder.pushIncref(outputId);
      } else {
        // If this output is seen again, increment its ref at that point.
        outputNeedsRef.add(outputId);
      }
    }

    // Effect-driven buffer allocation: compute liveness, insert free steps,
    // and recycle same-size buffers across execute/scan boundaries.
    builder.effectDrivenAllocate(outputIds);

    const jp = new JitProgram(
      backend,
      builder.steps,
      range(0, nargs),
      outputIds,
    );
    if (DEBUG >= 4) console.info(jp.toString());
    jitCompileCache.set(cacheKey, jp);
    return jp;
  } finally {
    _currentDimBindings = undefined;
  }
}

/**
 * Rule for fusing the operation into a JIT expression to the backend.
 *
 * This takes in the expressions of the `src[]` inputs and produces a subsequent
 * expression, as well as optionally a reduction. The expressions use
 * AluVar.gidx (output index) and AluVar.ridx (reduction index).
 */
type JitRule<P extends Primitive> = (
  exps: AluExp[],
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => {
  exp: AluExp[]; // One expression for each output
  reduction?: Reduction;
};

function reshapeViews(
  exp: AluExp,
  mapping: (st: ShapeTracker) => ShapeTracker | undefined,
  reduceAxis: boolean = false,
): AluExp {
  return exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const [gid, st]: [number, ShapeTracker] = exp.arg;
      const newSt = mapping(st);
      if (newSt) {
        const indices = reduceAxis
          ? unravelAlu(newSt.shape.slice(0, -1), AluVar.gidx).concat(
              AluVar.ridx,
            )
          : unravelAlu(newSt.shape, AluVar.gidx);
        return AluExp.globalView(exp.dtype, gid, newSt, indices);
      }
    } else if (exp.op === AluOp.GlobalIndex) {
      throw new Error("internal: reshapeViews() called with GlobalIndex op");
    }
  });
}

// JIT handler for a broadcasted operation on at least 1 input.
function broadcastedJit<P extends Primitive>(
  fn: (exps: AluExp[], params: PrimitiveParams<P>) => AluExp,
  opts?: { skipCastIdx?: number[] },
): JitRule<P> {
  return (exps, avals, params) => {
    let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);

    const skipCastIdx = opts?.skipCastIdx ?? [];
    if (skipCastIdx.length) {
      // Skip casting some indices to a shared dtype.
      newDtype = avals
        .filter((_, i) => !skipCastIdx.includes(i))
        .reduce(promoteAvals).dtype;
    }

    // Perform a broadcast on each of the input expressions.
    //
    // Only GlobalView is affected. GlobalIndex is not used here, and neither is
    // AluVar.idx, since those are realized before jit().
    // For symbolic shapes, resolve to concrete using dimBindings for
    // ShapeTracker broadcast operations.
    const concreteNewShape =
      _currentDimBindings && hasSymbolicDims(newShape)
        ? (resolveShape(newShape, _currentDimBindings) as number[])
        : (newShape as number[]);
    exps = exps.map((exp, i) => {
      exp = reshapeViews(exp, (st) => {
        if (!deepEqual(st.shape, concreteNewShape))
          return st.broadcast(
            concreteNewShape,
            range(concreteNewShape.length - st.shape.length),
          );
      });
      if (exp.dtype !== newDtype && !skipCastIdx.includes(i)) {
        exp = AluExp.cast(newDtype, exp);
      }
      return exp;
    });

    // Then, we can call the function to produce a new expression.
    return { exp: [fn(exps, params)] };
  };
}

// Simpler JIT handler, equivalent to broadcastedJit for unary ops.
function unopJit<P extends Primitive>(
  fn: (exp: AluExp, params: PrimitiveParams<P>) => AluExp,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: [fn(a, params)] };
  };
}

function reshapeJit<P extends Primitive>(
  fn: (st: ShapeTracker, params: PrimitiveParams<P>) => ShapeTracker,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: [reshapeViews(a, (st) => fn(st, params))] };
  };
}

function routineNoJit<P extends Primitive>(): JitRule<P> {
  return () => {
    throw new Error("jit: rule is not implemented for routines");
  };
}

const jitRules: { [P in Primitive]: JitRule<P> } = {
  [Primitive.Add]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  [Primitive.Mul]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  [Primitive.Idiv]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  [Primitive.Mod]: broadcastedJit(([a, b]) => AluExp.mod(a, b)),
  [Primitive.Min]: broadcastedJit(([a, b]) => AluExp.min(a, b)),
  [Primitive.Max]: broadcastedJit(([a, b]) => AluExp.max(a, b)),
  [Primitive.Neg]: unopJit((a) => AluExp.sub(AluExp.const(a.dtype, 0), a)),
  [Primitive.Reciprocal]: unopJit(AluExp.reciprocal),
  [Primitive.Floor]: unopJit(AluExp.floor),
  [Primitive.Ceil]: unopJit(AluExp.ceil),
  [Primitive.StopGradient]: unopJit((a) => a), // No-op, just return the input.
  [Primitive.Cast]: unopJit((a, { dtype }) => AluExp.cast(dtype, a)),
  [Primitive.Bitcast]: unopJit((a, { dtype }) => AluExp.bitcast(dtype, a)),
  [Primitive.Sin]: unopJit(AluExp.sin),
  [Primitive.Cos]: unopJit(AluExp.cos),
  [Primitive.Asin]: unopJit(AluExp.asin),
  [Primitive.Atan]: unopJit(AluExp.atan),
  [Primitive.Exp]: unopJit(AluExp.exp),
  [Primitive.Log]: unopJit(AluExp.log),
  [Primitive.Erf]: unopJit(AluExp.erf),
  [Primitive.Erfc]: unopJit(AluExp.erfc),
  [Primitive.Sqrt]: unopJit(AluExp.sqrt),
  [Primitive.Reduce]([a], [as], { op, axis }) {
    // Resolve symbolic shapes to concrete for ShapeTracker operations.
    const shape =
      _currentDimBindings && hasSymbolicDims(as.shape)
        ? (resolveShape(as.shape, _currentDimBindings) as number[])
        : (as.shape as number[]);
    const keptAxes: number[] = [];
    const shiftedAxes: number[] = [];
    const newShape: number[] = [];
    for (let i = 0; i < shape.length; i++) {
      if (axis.includes(i)) shiftedAxes.push(i);
      else {
        keptAxes.push(i);
        newShape.push(shape[i]);
      }
    }
    const reductionSize = prod(shiftedAxes.map((ax) => shape[ax]));
    newShape.push(reductionSize);

    // Compute the reduction size as SizeExpr from the original (possibly
    // symbolic) shape so that the Reduction caches correctly under a
    // symbolic key and the dynamic loop bound is resolved at execute time.
    const reductionSizeExpr: SizeExpr = hasSymbolicDims(as.shape)
      ? dimProduct(shiftedAxes.map((ax) => as.shape[ax]))
      : reductionSize;

    const perm = keptAxes.concat(shiftedAxes);
    a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
    const reduction = new Reduction(a.dtype, op, reductionSizeExpr);
    return { exp: [a], reduction };
  },
  [Primitive.Pool]: reshapeJit((st, { window, strides }) =>
    pool(st, window, strides),
  ),
  [Primitive.PoolTranspose]([a], [as], { inShape, window, strides }) {
    let stX = poolTranspose(
      ShapeTracker.fromShape(as.shape as number[]),
      inShape,
      window,
      strides,
    );
    stX = stX.reshape([...inShape, prod(stX.shape.slice(inShape.length))]); // Combine all reduce axes.
    a = reshapeViews(a, (st) => st.compose(stX), true);
    const reduction = new Reduction(
      a.dtype,
      AluOp.Add,
      stX.shape[stX.shape.length - 1],
    );
    return { exp: [a], reduction };
  },
  [Primitive.Dot]([a, b], [as, bs]) {
    // Dot is just Mul->Reduce in sequence.
    const k1 = jitRules[Primitive.Mul]([a, b], [as, bs], {});
    const [c] = k1.exp;
    const cs = promoteAvals(as, bs);
    return jitRules[Primitive.Reduce]([c], [cs], {
      op: AluOp.Add,
      axis: [cs.ndim - 1],
    });
  },
  [Primitive.Conv]([a, b], [as, bs], params) {
    const [stX, stY] = prepareConv(
      ShapeTracker.fromShape(as.shape as number[]),
      ShapeTracker.fromShape(bs.shape as number[]),
      params,
    );
    a = reshapeViews(a, (st) => st.compose(stX));
    b = reshapeViews(b, (st) => st.compose(stY));
    as = new ShapedArray(stX.shape, as.dtype, as.weakType);
    bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
    return jitRules[Primitive.Dot]([a, b], [as, bs], {});
  },
  [Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
  [Primitive.Where]: broadcastedJit(
    ([cond, a, b]) => AluExp.where(cond, a, b),
    { skipCastIdx: [0] },
  ),
  [Primitive.Concatenate](exps, avals, { axis }) {
    const ndim = avals[0].ndim;
    const sizes = avals.map((x) => x.shape[axis] as number);
    const finalSize = sizes.reduce((a, b) => a + b, 0);
    const { dtype: dtypeOut } = avals
      .map((x) => x.scalar())
      .reduce(promoteAvals);
    const makePadAxis = (start: number, end: number): Pair[] =>
      range(ndim).map((i) => (i === axis ? [start, end] : [0, 0]));
    let cum = 0;
    const src: AluExp[] = [];
    for (let i = 0; i < exps.length; i++) {
      const padding = makePadAxis(cum, finalSize - cum - (sizes[i] as number));
      src.push(
        reshapeViews(AluExp.cast(dtypeOut, exps[i]), (st) => st.pad(padding)),
      );
      cum += sizes[i] as number;
    }
    return { exp: [src.reduce(AluExp.add)] };
  },
  [Primitive.Split]([a], [as], { axis, sizes }) {
    const exp: AluExp[] = [];
    let start = 0;
    for (const size of sizes) {
      const slice = range(as.ndim).map<Pair>((d) =>
        d === axis ? [start, start + size] : [0, as.shape[d] as number],
      );
      exp.push(reshapeViews(a, (st) => st.shrink(slice)));
      start += size;
    }
    return { exp };
  },
  [Primitive.RandomBits]: (keys, keyShapes, { shape, mode }) => {
    const keyShape = keyShapes[0].shape;
    const mapping = (st: ShapeTracker): ShapeTracker | undefined => {
      if (!deepEqual(st.shape, shape))
        return st.broadcast(shape, range(st.shape.length, shape.length));
    };
    const k0 = reshapeViews(keys[0], mapping);
    const k1 = reshapeViews(keys[1], mapping);
    const c0 = AluExp.u32(0);
    const c1 = AluExp.mod(
      AluExp.cast(DType.Uint32, AluVar.gidx),
      // max(..., 1) to avoid mod-by-zero compile error in degenerate case
      AluExp.u32(Math.max(prod(shape.slice(keyShape.length)), 1)),
    );
    const exp = AluExp.threefry2x32(k0, k1, c0, c1, mode);
    return { exp: [exp] };
  },
  [Primitive.Gather](
    [x, ...indices],
    [xs, ...indicesShapes],
    { axis, outDim },
  ) {
    const axisSet = new Set(axis);

    // First, broadcast each integer array in `indices`.
    const indexShape = indicesShapes
      .map((c) => c.shape)
      .reduce(generalBroadcast) as number[];
    const finalShape = (xs.shape as number[]).filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);

    // Make variables for expression indices for gathered axes, and non-axis.
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const _idxAxis = idxNonaxis.splice(outDim, indexShape.length);

    // Then, construct a kernel expression that gathers the data.
    const src: AluExp[] = [...idxNonaxis];
    for (let i = 0; i < xs.shape.length; i++) {
      // insert 'null' as axis placeholder, overwritten below as src[axis[i]].
      if (axisSet.has(i)) src.splice(i, 0, null as any);
    }

    for (const [i, iexp] of indices.entries()) {
      // Index iexp by the idxAxis variables, after broadcasting (via GlobalView).
      // [ ... | outDim | ... <iexp> | outDim + indexShape.length | ... ]
      src[axis[i]] = AluExp.cast(
        DType.Int32,
        reshapeViews(iexp, (st) =>
          st.broadcast(finalShape, [
            // Broadcast indices (aligned to the right), plus leading before outDim.
            ...range(outDim + indexShape.length - st.shape.length),
            // Indices to the right of outDim.
            ...range(outDim + indexShape.length, finalShape.length),
          ]),
        ),
      );
    }

    // Finally, index into "x" by replacing its gidx with a flat accessor into
    // the gathered indices.
    const [index, valid] = ShapeTracker.fromShape(
      xs.shape as number[],
    ).toAluExp(src);
    if (!valid.resolve())
      throw new Error("internal: expected full validity mask in Gather");
    return { exp: [x.substitute({ gidx: index })] };
  },
  [Primitive.Transpose]: reshapeJit((st, { perm }) => st.permute(perm)),
  [Primitive.Broadcast]: reshapeJit((st, { shape, axis }) =>
    st.broadcast(shape, axis),
  ),
  [Primitive.Reshape]: reshapeJit((st, { shape }) => st.reshape(shape)),
  [Primitive.Flip]: reshapeJit((st, { axis }) => {
    const arg = rep(st.shape.length, false);
    for (const ax of axis) arg[ax] = true;
    return st.flip(arg);
  }),
  [Primitive.Shrink]: reshapeJit((st, { slice }) => st.shrink(slice)),
  [Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),
  [Primitive.Sort]: routineNoJit(),
  [Primitive.Argsort]: routineNoJit(),
  [Primitive.TriangularSolve]: routineNoJit(),
  [Primitive.Cholesky]: routineNoJit(),
  [Primitive.LU]: routineNoJit(),
  [Primitive.Jit]() {
    throw new Error(
      "internal: Jit should have been flattened before JIT compilation",
    );
  },
  [Primitive.DynamicUpdateSlice]() {
    throw new Error(
      "internal: DynamicUpdateSlice is handled specially in jitCompile",
    );
  },
  [Primitive.Scan]() {
    throw new Error(
      "internal: Scan is handled specially in jitCompile, not via jitRules",
    );
  },
  [Primitive.ScatterAdd]() {
    throw new Error("internal: ScatterAdd is handled specially in jitCompile");
  },
  [Primitive.AssociativeScan]() {
    throw new Error(
      "internal: AssociativeScan is handled specially in jitCompile",
    );
  },
};

/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
function splitGraphDataflow(backend: Backend, jaxpr: Jaxpr): Set<Var> {
  const varToDefn = new Map<Var, number>(); // Var -> eqn index of definition
  const varToUsages: Map<Var, number[]> = new Map(); // Var -> eqn indices of usages
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    for (const v of eqn.outBinders) {
      if (v instanceof Var) varToDefn.set(v, i);
    }
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        const usages = varToUsages.get(input);
        if (usages) usages.push(i);
        else varToUsages.set(input, [i]);
      }
    }
  }

  // Calculate reduction epilogues.
  //
  // A reduction can be fused with one or more operations that use its output,
  // which are either 1) unary or 2) binary ops with a literal, or an array not
  // larger than it.
  //
  // We also need to make sure we don't fuse two reductions together.
  const reducePrimitives = [
    Primitive.Reduce,
    Primitive.Dot,
    Primitive.Conv,
    Primitive.PoolTranspose,
  ];
  const reductionEpilogueEqns = new Set<number>();
  const reductionEndpointEqns = new Set<number>();
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    if (reducePrimitives.includes(eqn.primitive)) {
      let head = i;
      while (true) {
        reductionEpilogueEqns.add(head);

        // Try moving outVar forward through the graph.
        const outVar = jaxpr.eqns[head].outBinders[0];
        const usages = varToUsages.get(outVar) ?? [];
        if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;

        // Next is already fused into a reduction epilogue, can't fuse again.
        if (reductionEpilogueEqns.has(usages[0])) break;

        const nextEqn = jaxpr.eqns[usages[0]];
        switch (nextEqn.primitive) {
          // We can always fuse unary operations.
          case Primitive.Neg:
          case Primitive.Reciprocal:
          case Primitive.Floor:
          case Primitive.Ceil:
          case Primitive.StopGradient:
          case Primitive.Cast:
          case Primitive.Bitcast:
          case Primitive.Sin:
          case Primitive.Cos:
          case Primitive.Asin:
          case Primitive.Atan:
          case Primitive.Exp:
          case Primitive.Log:
          case Primitive.Erf:
          case Primitive.Erfc:
          case Primitive.Sqrt:
            head = usages[0];
            continue;

          // We can fuse binary operations with a literal, or with an array such
          // that the array doesn't lead to broadcasting thus recomputation.
          case Primitive.Add:
          case Primitive.Mul:
          case Primitive.Idiv:
          case Primitive.Mod:
          case Primitive.Min:
          case Primitive.Max:
          case Primitive.Compare: {
            const otherInput = nextEqn.inputs.find((v) => v !== outVar)!;
            if (
              otherInput instanceof Lit ||
              deepEqual(
                generalBroadcast(otherInput.aval.shape, outVar.aval.shape),
                outVar.aval.shape,
              )
            ) {
              head = usages[0];
              continue;
            }
            break;
          }

          // Ternary Where: fusable if all non-chain inputs are literals or
          // same-shape arrays (no broadcasting that enlarges the result).
          case Primitive.Where: {
            const otherInputs = nextEqn.inputs.filter((v) => v !== outVar);
            if (
              otherInputs.every(
                (inp) =>
                  inp instanceof Lit ||
                  deepEqual(
                    generalBroadcast(inp.aval.shape, outVar.aval.shape),
                    outVar.aval.shape,
                  ),
              )
            ) {
              head = usages[0];
              continue;
            }
            break;
          }
        }
        break; // Can't move forward anymore.
      }
      reductionEndpointEqns.add(head);
    }
  }

  // Move backwards through the program and compute "black" endpoints.
  //
  // Black nodes are the endpoints where we dispatch a kernel to the backend
  // rather than producing intermediates. This includes:
  //
  // - Kernel outputs
  // - Reductions, except when fused with epilogue
  // - Gather/RandomBits operations (violates rule that kernels must have
  //   homogeneous GlobalView indices)
  // - Inputs to Pad operations, which need clean inputs
  //
  // Also, mark a node black if there are at least two black nodes that can be
  // reached from it, while only going through non-black nodes.
  //
  // TODO: Don't do the above for 'simple' nodes: reshape, cast, etc.
  const blackNodes = new Set<Var>();
  const p1NextBlack = new Map<Var, Var>();
  for (const v of jaxpr.outs) {
    if (v instanceof Var) {
      blackNodes.add(v);
      p1NextBlack.set(v, v);
    }
  }
  const heterogeneousViewPrimitives = [
    // These primitives generate heterogeneous GlobalView outputs, there are
    // multiple views in the expression with different indexing.
    Primitive.RandomBits,
    Primitive.Gather,
  ];
  const needsCleanShapePrimitives = [
    // Concatenate is based on Pad internally.
    Primitive.Concatenate,
    // If Pad is applied to a non-clean input, the reshaped padding would apply
    // to the view _inside_ of the expression. Imagine `GlobalView(...)+1`: if
    // you reshape each view, it adds zeros into the inner expression, so the
    // effect is to pad the intermediate with 1s instead of 0s!
    Primitive.Pad,
  ];
  const specialBlackPrimitives = [
    // These primitives are handled specially in jitCompile and must always
    // be black nodes with materialized inputs.
    Primitive.Scan,
    Primitive.DynamicUpdateSlice,
    Primitive.ScatterAdd,
    Primitive.AssociativeScan,
  ];
  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (
      reductionEndpointEqns.has(i) ||
      heterogeneousViewPrimitives.includes(eqn.primitive) ||
      routinePrimitives.has(eqn.primitive) ||
      specialBlackPrimitives.includes(eqn.primitive) ||
      eqn.outBinders.some((v) => blackNodes.has(v))
    ) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
      continue;
    }
    const reach = new Set<Var>();
    let needsCleanOutput = false;
    outer: for (const v of eqn.outBinders) {
      for (const j of varToUsages.get(v) ?? []) {
        if (
          needsCleanShapePrimitives.includes(jaxpr.eqns[j].primitive) ||
          routinePrimitives.has(jaxpr.eqns[j].primitive) ||
          specialBlackPrimitives.includes(jaxpr.eqns[j].primitive)
        ) {
          needsCleanOutput = true;
          break outer;
        }
        for (const o of jaxpr.eqns[j].outBinders) {
          const u = p1NextBlack.get(o);
          if (u) reach.add(u);
        }
      }
    }
    if (reach.size > 1 || needsCleanOutput) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
    } else if (reach.size === 1) {
      const b = reach.values().next().value!;
      for (const v of eqn.outBinders) p1NextBlack.set(v, b);
    }
  }

  // Also, mark nodes black if the maximum number of arguments per kernel is
  // exceeded (i.e., maxComputeBuffersPerShaderStage for WebGPU). This needs to
  // be done in a second forward pass over the equations list.
  const p2Deps = new Map<Var, Set<Var>>(); // -> members are Var (black) or inBinders.
  for (const v of jaxpr.inBinders) {
    p2Deps.set(v, new Set([v])); // Each input is a dependency of itself.
  }
  let p2idx = 0;
  while (p2idx < jaxpr.eqns.length) {
    const eqn = jaxpr.eqns[p2idx++];
    // Non-kernel black nodes (Scan, Routines) handle their own buffer bindings
    // internally and don't go through the WebGPU kernel compiler, so maxArgs
    // doesn't apply to them. Skip the dep check for these.
    //
    // NOTE: elementwise black nodes (e.g. Jaxpr outputs, multi-use nodes) DO
    // get compiled as WebGPU kernels and DO need the dep count check even when
    // all their outBinders are already in blackNodes. The former "skip all
    // all-black equations" early-continue was a bug: it allowed those kernel
    // endpoints to accumulate too many fused inputs without backtracking.
    const isNonKernelBlack =
      eqn.outBinders.every((v) => blackNodes.has(v)) &&
      (specialBlackPrimitives.includes(eqn.primitive) ||
        routinePrimitives.has(eqn.primitive));
    if (isNonKernelBlack) {
      for (const out of eqn.outBinders) p2Deps.set(out, new Set([out]));
      continue;
    }
    // For all-black kernel endpoints the output is already materialized, so
    // p2Deps should represent it as a single "base" dep rather than propagating
    // its transitive dep set to downstream equations.
    const isAllBlack = eqn.outBinders.every((v) => blackNodes.has(v));
    const deps: Set<Var>[] = [];
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        if (blackNodes.has(input)) deps.push(new Set([input]));
        else deps.push(p2Deps.get(input)!);
      } else {
        deps.push(new Set());
      }
    }
    const depCounter = new Map<Var, number>(); // includes counts
    for (const depSet of deps) {
      for (const dep of depSet) {
        depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
      }
    }
    if (depCounter.size > backend.maxArgs) {
      // We have too many dependencies, so we need to backtrack and mark one of
      // the inputs as black. By heuristic, we'll mark the one with the most
      // unique dependencies.
      let maxUniqueDeps = 0;
      let assocInput = -1;
      for (let i = 0; i < eqn.inputs.length; i++) {
        const input = eqn.inputs[i];
        if (input instanceof Var && varToDefn.has(input)) {
          let uniqueDeps = 0;
          for (const dep of deps[i]) {
            if (depCounter.get(dep) === 1) uniqueDeps++;
          }
          if (uniqueDeps > maxUniqueDeps) {
            maxUniqueDeps = uniqueDeps;
            assocInput = i;
          }
        }
      }
      if (assocInput === -1) {
        throw new Error(
          `internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`,
        );
      }
      const assocVar = eqn.inputs[assocInput] as Var;
      p2idx = varToDefn.get(assocVar)!; // backtrack to that equation
      for (const out of jaxpr.eqns[p2idx++].outBinders) {
        blackNodes.add(out);
      }
    } else {
      if (isAllBlack) {
        // Black kernel endpoint: output is materialized, treat as a single base dep.
        for (const out of eqn.outBinders) p2Deps.set(out, new Set([out]));
      } else {
        const s = new Set(depCounter.keys());
        for (const out of eqn.outBinders) p2Deps.set(out, s);
      }
    }
  }

  return blackNodes;
}
