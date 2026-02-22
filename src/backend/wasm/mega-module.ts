/**
 * Mega-Module compiler (M6.1): compiles an entire JitProgram's step list
 * into a single WASM function, eliminating JS↔WASM boundary crossings
 * between kernel dispatches.
 *
 * Architecture:
 * - Each JitId maps to a WASM local (i32 holding a memory pointer)
 * - Kernel bodies are inlined as gidx loops within the mega-function
 * - malloc/free are imported from the host WasmAllocator
 * - Routines are imported as separate WASM function calls
 * - incref is a no-op inside the module (handled at the JS Slot level)
 * - recycle is a zero-cost local rename
 *
 * The mega-module function signature:
 *   mega_execute(input0_ptr, ..., inputN_ptr, resultBufPtr) -> void
 *
 * Output pointers are written to the result buffer at known offsets.
 * The JS wrapper reads them and creates proper backend Slots.
 */

import { AluExp, AluOp, byteWidth, DType, Kernel } from "../../alu";
import type { JitId, JitStep } from "../../frontend/jit";
import { Routine } from "../../routine";
import { isSymbolicSize, type SizeExpr } from "../../shape";
import { tuneNullopt } from "../../tuner";
import { mapSetUnion, rep } from "../../utils";
import {
  codegenReductionAccumulate,
  configureMemoryImport,
  dty,
  importWasmHelperFuncs,
  translateExpCore,
} from "../wasm";
import { CodeGenerator } from "./wasmblr";

/** Result of compiling a JitProgram into a mega-module. */
export interface WasmMegaModule {
  /** The compiled WebAssembly module. */
  readonly module: WebAssembly.Module;
  /** Number of input parameters. */
  readonly numInputs: number;
  /** Number of output pointers written to the result buffer. */
  readonly numOutputs: number;
  /** Byte sizes of each output buffer (for creating Slots). */
  readonly outputSizes: number[];
  /**
   * Which step types are NOT supported (caused compilation to fail).
   * Empty if all steps were compiled.
   */
  readonly unsupportedSteps: string[];
}

/**
 * Check whether a JitProgram can be compiled to a mega-module.
 * Returns true if all steps are supported.
 */
export function canCompileToMegaModule(steps: JitStep[]): boolean {
  for (const step of steps) {
    switch (step.type) {
      case "malloc":
        // Symbolic malloc sizes require runtime resolution — not supported.
        if (isSymbolicSize(step.size)) return false;
        break;
      case "free":
      case "recycle":
        break;
      case "incref":
        // Incref requires refcount tracking inside WASM — not yet supported.
        // Without proper refcounting, a subsequent `free` would deallocate
        // memory that another local still references.
        return false;
      case "execute":
        // Support kernels (routines not yet supported)
        if (step.source instanceof Routine) return false;
        // Symbolic reduction sizes require runtime resolution via dynamicParams —
        // the mega-module inlines reduction loops with i32.const bounds, which
        // can't represent symbolic sizes. Reject and fall through to step-by-step.
        if (step.source instanceof Kernel && step.source.hasSymbolicReduction)
          return false;
        break;
      case "scan":
      case "dus":
      case "scatter_add":
      case "assoc_scan":
        return false;
      default:
        return false;
    }
  }
  return true;
}

/**
 * Compile a JitProgram's step list into a single WASM mega-module.
 *
 * The compiled function takes input pointers as i32 params and a
 * resultBufPtr as the last param. Output pointers are written to
 * the result buffer.
 *
 * @param steps       JitProgram steps
 * @param inputIds    JitIds of program inputs
 * @param outputIds   JitIds of program outputs
 * @returns Compiled mega-module or null if unsupported steps are present
 */
export function compileToMegaModule(
  steps: JitStep[],
  inputIds: JitId[],
  outputIds: JitId[],
): WasmMegaModule | null {
  // Check all steps are supported
  if (!canCompileToMegaModule(steps)) return null;

  // Collect all JitIds referenced in the program
  const allJitIds = new Set<JitId>();
  for (const id of inputIds) allJitIds.add(id);
  for (const id of outputIds) allJitIds.add(id);
  for (const step of steps) {
    switch (step.type) {
      case "malloc":
        allJitIds.add(step.output);
        break;
      case "free":
        allJitIds.add(step.input);
        break;
      case "recycle":
        allJitIds.add(step.input);
        allJitIds.add(step.output);
        break;
      case "incref":
        allJitIds.add(step.input);
        break;
      case "execute":
        for (const id of step.inputs) allJitIds.add(id);
        for (const id of step.outputs) allJitIds.add(id);
        break;
    }
  }

  // Track byte sizes of every buffer, including through recycle steps.
  // malloc creates a buffer; recycle transfers ownership (same size).
  const bufferSizes = new Map<JitId, SizeExpr>();
  for (const step of steps) {
    if (step.type === "malloc") {
      bufferSizes.set(step.output, step.size);
    } else if (step.type === "recycle") {
      const srcSize = bufferSizes.get(step.input);
      if (srcSize !== undefined) bufferSizes.set(step.output, srcSize);
    }
  }

  // Resolve output sizes (must be concrete for now)
  const inputIdSet = new Set(inputIds);
  const outputSizes: number[] = [];
  for (const id of outputIds) {
    const size = bufferSizes.get(id);
    if (size !== undefined) {
      if (isSymbolicSize(size)) return null; // Can't handle symbolic sizes yet
      outputSizes.push(size as number);
    } else if (inputIdSet.has(id)) {
      // Output is an input pass-through. Reject — pass-through programs are
      // trivial and don't benefit from mega-module compilation. The WASM
      // function's local for this JitId points to the input param, but other
      // steps (free, recycle) might overwrite it before the output pointer
      // is written to the result buffer. Rather than tracking which steps
      // modify which locals, we conservatively bail.
      return null;
    } else {
      // Output JitId is neither allocated nor an input — can't handle this.
      return null;
    }
  }

  // Collect all distinct AluOps across all kernels for importing helpers
  let allOps: Map<AluOp, Set<DType>> = new Map();

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    if (step.type === "execute" && step.source instanceof Kernel) {
      const kernel = step.source;
      for (const out of kernel.outputs) {
        const tune = tuneNullopt(
          Kernel.single(kernel.nargs, kernel.size, out.exp, out.reduction),
        );
        allOps = mapSetUnion(allOps, tune.exp.distinctOps());
        if (tune.epilogue)
          allOps = mapSetUnion(allOps, tune.epilogue.distinctOps());
      }
    }
  }

  // --- Generate WASM module ---
  const cg = new CodeGenerator();

  // Configure memory import (shared/non-shared)
  configureMemoryImport(cg);

  // Import allocator functions
  const allocFunc = cg.importFunction("env", "alloc", [cg.i32], [cg.i32]);
  const freeFunc = cg.importFunction("env", "free", [cg.i32], []);

  // Import math helper functions (reuse shared helper from wasm.ts)
  const funcs = importWasmHelperFuncs(cg, allOps);

  // Build function: mega_execute(input0..inputN, resultBufPtr) -> void
  const numInputParams = inputIds.length;
  const totalParams = numInputParams + 1; // +1 for resultBufPtr
  const paramTypes = rep(totalParams, cg.i32);

  const megaFunc = cg.function(paramTypes, [], () => {
    // Create a WASM local for each JitId.
    // Input JitIds are initialized from function params.
    // Others are initialized when created (malloc, recycle).
    const jitIdLocals = new Map<JitId, number>();

    // Map input JitIds to function params
    for (let i = 0; i < inputIds.length; i++) {
      jitIdLocals.set(inputIds[i], i); // param index
    }

    // Declare locals for non-input JitIds
    for (const id of allJitIds) {
      if (!jitIdLocals.has(id)) {
        const local = cg.local.declare(cg.i32);
        jitIdLocals.set(id, local);
      }
    }

    // Result buffer pointer is the last param
    const resultBufParam = numInputParams;

    // Emit code for each step
    for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
      const step = steps[stepIdx];

      switch (step.type) {
        case "malloc": {
          if (isSymbolicSize(step.size)) {
            throw new Error("Mega-module: symbolic malloc sizes not supported");
          }
          const size = step.size as number;
          // local.set(jitId, call $alloc(size))
          cg.i32.const(size);
          cg.call(allocFunc);
          cg.local.set(jitIdLocals.get(step.output)!);
          break;
        }

        case "free": {
          // call $free(local.get(jitId))
          cg.local.get(jitIdLocals.get(step.input)!);
          cg.call(freeFunc);
          break;
        }

        case "recycle": {
          // local.set(new, local.get(old)) — zero cost
          cg.local.get(jitIdLocals.get(step.input)!);
          cg.local.set(jitIdLocals.get(step.output)!);
          break;
        }

        case "execute": {
          if (step.source instanceof Kernel) {
            emitInlinedKernel(
              cg,
              funcs,
              step.source,
              step.inputs.map((id) => jitIdLocals.get(id)!),
              step.outputs.map((id) => jitIdLocals.get(id)!),
            );
          }
          break;
        }
      }
    }

    // Write output pointers to result buffer
    for (let i = 0; i < outputIds.length; i++) {
      const outputLocal = jitIdLocals.get(outputIds[i])!;
      // i32.store(resultBufPtr + i*4, local.get(outputJitId))
      cg.local.get(resultBufParam);
      if (i > 0) {
        cg.i32.const(i * 4);
        cg.i32.add();
      }
      cg.local.get(outputLocal);
      cg.i32.store(2); // align=4 (log2(4)=2)
    }
  });

  cg.export(megaFunc, "mega_execute");

  const bytes = cg.finish();
  const module = new WebAssembly.Module(bytes);

  return {
    module,
    numInputs: numInputParams,
    numOutputs: outputIds.length,
    outputSizes,
    unsupportedSteps: [],
  };
}

// ---------------------------------------------------------------------------
// Inline kernel emission
// ---------------------------------------------------------------------------

/**
 * Emit an inlined kernel body within the mega-module.
 *
 * This reuses the gidx loop structure but references JitId locals
 * instead of function parameters for input/output buffers.
 */
function emitInlinedKernel(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  kernel: Kernel,
  inputLocals: number[],
  outputLocals: number[],
): void {
  if (kernel.isMultiOutput) {
    emitInlinedMultiOutputKernel(cg, funcs, kernel, inputLocals, outputLocals);
    return;
  }

  const tune = tuneNullopt(kernel);
  const out = kernel.outputs[0];
  const re = out.reduction;
  const storeAlign = Math.log2(byteWidth(out.dtype));

  const gidx = cg.local.declare(cg.i32);
  const kernelSize = kernel.size as number;

  // gidx = 0
  cg.i32.const(0);
  cg.local.set(gidx);

  // loop
  cg.loop(cg.void);
  {
    // if (gidx >= size) break
    cg.block(cg.void);
    cg.local.get(gidx);
    cg.i32.const(kernelSize);
    cg.i32.ge_u();
    cg.br_if(0);

    // Output address: outputLocal + gidx * byteWidth
    cg.local.get(outputLocals[0]);
    cg.local.get(gidx);
    cg.i32.const(byteWidth(out.dtype));
    cg.i32.mul();
    cg.i32.add();

    if (re) {
      // Reduction: accumulator + inner ridx loop
      emitReductionBody(cg, funcs, tune, out, re, gidx, inputLocals);
    } else {
      // Non-reduction: translate expression directly
      translateExpMega(cg, funcs, tune.exp, gidx, inputLocals);
    }

    // Store result
    dty(cg, null, out.dtype).store(storeAlign);

    // gidx++
    cg.local.get(gidx);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(gidx);

    cg.br(1);
    cg.end();
  }
  cg.end();
}

/**
 * Emit an inlined multi-output kernel body.
 * One gidx loop evaluates and stores all output expressions.
 */
function emitInlinedMultiOutputKernel(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  kernel: Kernel,
  inputLocals: number[],
  outputLocals: number[],
): void {
  const numOutputs = kernel.numOutputs;
  const tunes = kernel.outputs.map((o) => {
    const tmpKernel = Kernel.single(
      kernel.nargs,
      kernel.size,
      o.exp,
      o.reduction,
    );
    return tuneNullopt(tmpKernel);
  });

  const gidx = cg.local.declare(cg.i32);
  const kernelSize = kernel.size as number;

  // gidx = 0
  cg.i32.const(0);
  cg.local.set(gidx);

  // loop
  cg.loop(cg.void);
  {
    // if (gidx >= size) break
    cg.block(cg.void);
    cg.local.get(gidx);
    cg.i32.const(kernelSize);
    cg.i32.ge_u();
    cg.br_if(0);

    // For each output: compute + store
    for (let oi = 0; oi < numOutputs; oi++) {
      const tune = tunes[oi];
      const out = kernel.outputs[oi];
      const storeAlign = Math.log2(byteWidth(out.dtype));

      // Output address: outputLocals[oi] + gidx * byteWidth
      cg.local.get(outputLocals[oi]);
      cg.local.get(gidx);
      cg.i32.const(byteWidth(out.dtype));
      cg.i32.mul();
      cg.i32.add();

      if (out.reduction) {
        emitReductionBody(
          cg,
          funcs,
          tune,
          out,
          out.reduction,
          gidx,
          inputLocals,
        );
      } else {
        translateExpMega(cg, funcs, tune.exp, gidx, inputLocals);
      }

      dty(cg, null, out.dtype).store(storeAlign);
    }

    // gidx++
    cg.local.get(gidx);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(gidx);

    cg.br(1);
    cg.end();
  }
  cg.end();
}

// ---------------------------------------------------------------------------
// Reduction body emission
// ---------------------------------------------------------------------------

/**
 * Emit reduction body: accumulator init, ridx loop, epilogue.
 * Shared between single-output and multi-output inlined kernels.
 */
function emitReductionBody(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  tune: ReturnType<typeof tuneNullopt>,
  out: {
    exp: AluExp;
    dtype: DType;
    reduction?: {
      op: AluOp;
      dtype: DType;
      size: SizeExpr;
      identity: number;
    } | null;
  },
  re: { op: AluOp; dtype: DType; size: SizeExpr; identity: number },
  gidx: number,
  inputLocals: number[],
): void {
  const acc = cg.local.declare(dty(cg, null, out.exp.dtype));
  dty(cg, null, out.exp.dtype).const(re.identity);
  cg.local.set(acc);

  // Kahan compensation for Float64 Add reductions
  const useKahan = re.dtype === DType.Float64 && re.op === AluOp.Add;
  let kahanComp: number | undefined;
  if (useKahan) {
    kahanComp = cg.local.declare(cg.f64);
    cg.f64.const(0);
    cg.local.set(kahanComp);
  }

  const ridx = cg.local.declare(cg.i32);
  cg.i32.const(0);
  cg.local.set(ridx);

  // Inner reduction loop
  // Note: mega-module rejects symbolic sizes in canCompileToMegaModule(),
  // so re.size is guaranteed to be concrete here.
  cg.loop(cg.void);
  {
    cg.block(cg.void);
    cg.local.get(ridx);
    cg.i32.const(re.size as number);
    cg.i32.ge_u();
    cg.br_if(0);

    translateExpMega(cg, funcs, tune.exp, gidx, inputLocals, ridx);
    codegenReductionAccumulate(cg, re, acc, kahanComp);

    cg.local.get(ridx);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(ridx);

    cg.br(1);
    cg.end();
  }
  cg.end();

  // Epilogue: apply post-reduction expression (e.g., divide by count for mean)
  translateExpMega(
    cg,
    funcs,
    tune.epilogue!,
    gidx,
    inputLocals,
    undefined,
    acc,
  );
}

// ---------------------------------------------------------------------------
// Expression translation for mega-module context
// ---------------------------------------------------------------------------

/**
 * Translate an AluExp within the mega-module context.
 * Delegates to the shared translateExpCore from wasm.ts, providing
 * mega-module-specific variable resolution and GlobalIndex handling.
 *
 * In the mega-module context, GlobalIndex buffer references use WASM locals
 * (inputLocals) directly, without the (start, end) parameter offset that
 * regular kernels have.
 */
function translateExpMega(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  gidx: number,
  inputLocals: number[],
  ridx?: number,
  acc?: number,
): void {
  translateExpCore(cg, funcs, exp, {
    getVariable: (name) => {
      if (name === "gidx") return gidx;
      if (name === "ridx") return ridx;
      if (name === "acc") return acc;
      return undefined;
    },
    handleGlobalIndex: (cg, gen, gid, len, indexExp, dtype) => {
      gen(indexExp);

      // Bounds check (same as regular kernel translation)
      const local = cg.local.declare(cg.i32);
      cg.local.tee(local);
      cg.i32.const(0);
      cg.local.get(local);
      cg.i32.const(len);
      cg.i32.lt_u();
      cg.select();

      // Compute byte offset
      cg.i32.const(byteWidth(dtype));
      cg.i32.mul();

      // Add base pointer from the mega-module's input local
      // (no RANGE_PARAMS offset — mega-module maps JitIds directly)
      cg.local.get(inputLocals[gid]);
      cg.i32.add();

      // Load value
      dty(cg, AluOp.GlobalIndex, dtype).load(Math.log2(byteWidth(dtype)));
    },
  });
}
