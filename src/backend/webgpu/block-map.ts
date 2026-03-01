/**
 * @file BlockMap fused shader compiler for WebGPU.
 *
 * Compiles kernel-only block_map bodies into single WGSL compute shaders
 * where each workgroup processes one block using `var<workgroup>` shared memory.
 *
 * Architecture:
 *   - Body JitProgram's malloc steps → var<workgroup> arrays
 *   - Body JitProgram's execute steps → inline WGSL via gen()
 *   - workgroupBarrier() between data-dependent steps
 *   - Thread i processes element i within the block
 *   - Block index from workgroup_id
 *
 * Returns null (→ fallback) if:
 *   - Body contains routine steps (Sort, Cholesky, etc.)
 *   - Shared memory exceeds device.limits.maxComputeWorkgroupStorageSize
 *   - prod(blockShape) exceeds device.limits.maxComputeInvocationsPerWorkgroup
 *   - Symbolic malloc sizes
 *   - More storage bindings than maxStorageBuffersPerShaderStage
 */

import { erfSrc, threefrySrc } from "./builtins";
import {
  calculateGrid,
  constToWgsl,
  dtypeToWgsl,
  headerWgsl,
  type ShaderInfo,
} from "./codegen";
import {
  accessorGlobal,
  AluExp,
  AluGroup,
  AluOp,
  byteWidth,
  DType,
  isFloatDtype,
  Kernel,
  Reduction,
} from "../../alu";
import type { JitId, JitProgram, JitStep } from "../../frontend/jit";
import { Routine } from "../../routine";
import { isSymbolicSize } from "../../shape";
import { DEBUG, mapSetUnion, prod, strip1 } from "../../utils";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface BlockMapShaderParams {
  /** The compiled body program (its steps are analyzed). */
  bodyProgram: JitProgram;
  /** Block dimensions (always concrete numbers). */
  blockShape: number[];
  /** Grid dimensions (number of blocks per axis). */
  gridShape: number[];
  /** How body inputs map to grid axes. Per-input: axis index per grid dim, or null. */
  inAxes: (number | null)[][];
  /** How body outputs map to grid axes. Per-output: axis index per grid dim, or null. */
  outAxes: (number | null)[][];
  /** Number of constant inputs (first numConsts body inputs are consts). */
  numConsts: number;
  /** Number of non-const inputs. */
  numInputs: number;
  /** Original (untrimmed) input shapes. */
  inputShapes: number[][];
  /** Original (untrimmed) output shapes. */
  outputShapes: number[][];
}

/**
 * Try to compile a block_map body into a fused WGSL compute shader.
 * Returns null if the body is not eligible for fusion.
 */
export function blockMapFusedShaderSource(
  device: GPUDevice,
  params: BlockMapShaderParams,
  _caps?: unknown,
): ShaderInfo | null {
  const { bodyProgram, blockShape, gridShape, numConsts, numInputs } = params;
  const steps = bodyProgram.steps;

  // --- Guard: prod(blockShape) ≤ maxComputeInvocationsPerWorkgroup ---
  const blockSize = prod(blockShape);
  const maxInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  if (blockSize > maxInvocations) {
    if (DEBUG >= 1)
      console.info(
        `block_map fused: blockSize ${blockSize} > maxInvocations ${maxInvocations}, fallback`,
      );
    return null;
  }

  // --- Detect boundary blocks (non-divisible dimensions) ---
  // When dimensions are not evenly divisible by blockShape, the last block
  // along each axis has some invalid (out-of-bounds) threads. We emit a
  // per-thread `valid` flag and guard all global reads/writes.
  const gridRank = blockShape.length;
  // Per-axis: the original dimension along each grid axis (for validity checks)
  const axisDims: (number | null)[] = new Array(gridRank).fill(null);
  let hasBoundary = false;
  for (let g = 0; g < gridRank; g++) {
    for (let i = 0; i < numInputs; i++) {
      const axes = params.inAxes[i];
      if (axes[g] !== null) {
        const dim = params.inputShapes[i][axes[g]!];
        axisDims[g] = dim;
        if (dim % blockShape[g] !== 0) hasBoundary = true;
      }
    }
  }

  // --- Guard: all execute steps must use Kernel (no Routines) ---
  // --- Classify steps and compute shared memory budget ---

  /** Map from JitId → shared memory info for intermediates. */
  const shmemMap = new Map<
    JitId,
    { sizeBytes: number; dtype: DType; elemCount: number }
  >();
  /** Steps that produce outputs (execute with Kernel source). */
  const kernelSteps: {
    step: Extract<JitStep, { type: "execute" }>;
    kernel: Kernel;
  }[] = [];
  /** Per-step reduction info (null if the step is elementwise). */
  const stepReductions: (Reduction | null)[] = [];

  /** Analyzed fori_loop steps. */
  interface ForiLoopInfo {
    foriStep: Extract<JitStep, { type: "fori_loop" }>;
    bodyKernels: {
      step: Extract<JitStep, { type: "execute" }>;
      kernel: Kernel;
    }[];
    bodyBarriers: Set<number>;
    bodyShmemMap: Map<JitId, { name: string; dtype: DType; elemCount: number }>;
    bodyShmemIds: Set<JitId>;
    bodyInputIds: JitId[];
    bodyOutputIds: JitId[];
    numConsts: number;
    loopVar: string;
  }
  const foriLoops: ForiLoopInfo[] = [];

  /** Analyzed workgroup_assoc_scan steps. */
  interface WorkgroupAssocScanInfo {
    wasStep: Extract<JitStep, { type: "workgroup_assoc_scan" }>;
    /** Body kernels (the binary operator fn). */
    bodyKernels: {
      step: Extract<JitStep, { type: "execute" }>;
      kernel: Kernel;
    }[];
    /** Body shmem intermediates. */
    bodyShmemMap: Map<JitId, { name: string; dtype: DType; elemCount: number }>;
    bodyShmemIds: Set<JitId>;
    bodyInputIds: JitId[];
    bodyOutputIds: JitId[];
    numConsts: number;
    numElems: number;
    /** Per-element dtype and count. */
    elemDtypes: DType[];
    elemCounts: number[];
    /** Ping/pong shmem names per elem: [pingName, pongName]. */
    pingPongNames: [string, string][];
    numRounds: number;
  }
  const workgroupAssocScans: WorkgroupAssocScanInfo[] = [];

  /** Combined codegen entries in step order. */
  type CodegenEntry =
    | { type: "kernel"; kernelIdx: number }
    | { type: "fori_loop"; flIdx: number }
    | { type: "workgroup_assoc_scan"; wasIdx: number };
  const codegenEntries: CodegenEntry[] = [];

  let totalShmemBytes = 0;

  for (const step of steps) {
    switch (step.type) {
      case "execute": {
        if (step.source instanceof Routine) {
          if (DEBUG >= 1)
            console.info("block_map fused: routine in body, fallback");
          return null;
        }
        const kernel = step.source as Kernel;
        // Reject symbolic kernel sizes
        if (kernel.isSymbolic) {
          if (DEBUG >= 1)
            console.info("block_map fused: symbolic kernel size, fallback");
          return null;
        }
        // Multi-output reductions not supported in fused shader
        if (kernel.hasReduction && kernel.numOutputs > 1) {
          if (DEBUG >= 1)
            console.info("block_map fused: multi-output reduction, fallback");
          return null;
        }
        const re = kernel.outputs[0]?.reduction ?? null;
        codegenEntries.push({ type: "kernel", kernelIdx: kernelSteps.length });
        kernelSteps.push({ step, kernel });
        stepReductions.push(re);
        break;
      }
      case "malloc": {
        if (isSymbolicSize(step.size)) {
          if (DEBUG >= 1)
            console.info("block_map fused: symbolic malloc size, fallback");
          return null;
        }
        const sizeBytes = step.size as number;
        shmemMap.set(step.output, {
          sizeBytes,
          dtype: DType.Float32,
          elemCount: sizeBytes / 4,
        });
        totalShmemBytes += sizeBytes;
        break;
      }
      case "free":
      case "recycle":
      case "incref":
        break;
      case "fori_loop": {
        // Analyze fori_loop body: must be kernel-only, elementwise-only
        const flIdx = foriLoops.length;
        const bodyProg = step.bodyProgram;
        const bodySteps = bodyProg.steps;
        const bodyKernels: ForiLoopInfo["bodyKernels"] = [];
        const bodyShmemMap = new Map<
          JitId,
          { name: string; dtype: DType; elemCount: number }
        >();
        const bodyShmemIds = new Set<JitId>();

        let valid = true;
        for (const bs of bodySteps) {
          switch (bs.type) {
            case "execute": {
              if (bs.source instanceof Routine) {
                if (DEBUG >= 1)
                  console.info(
                    "block_map fused: routine in fori_loop body, fallback",
                  );
                return null;
              }
              const bk = bs.source as Kernel;
              if (bk.isSymbolic) {
                valid = false;
                break;
              }
              // Reductions: allow per-thread contractions (kernel.size == blockSize)
              // but reject workgroup-level reductions (different thread counts).
              if (bk.hasReduction) {
                const re = bk.outputs[0]?.reduction;
                if (
                  !re ||
                  bk.numOutputs > 1 ||
                  (bk.size as number) !== blockSize
                ) {
                  if (DEBUG >= 1)
                    console.info(
                      "block_map fused: unsupported reduction in fori_loop body, fallback",
                    );
                  return null;
                }
                // Per-thread contraction: OK — each thread accumulates privately
              }
              // Body kernels larger than blockSize can't be processed 1:1
              if ((bk.size as number) > blockSize) {
                if (DEBUG >= 1)
                  console.info(
                    `block_map fused: fori_loop body kernel size ${bk.size} > blockSize ${blockSize}, fallback`,
                  );
                return null;
              }
              bodyKernels.push({
                step: bs as Extract<JitStep, { type: "execute" }>,
                kernel: bk,
              });
              break;
            }
            case "malloc": {
              if (isSymbolicSize(bs.size)) {
                valid = false;
                break;
              }
              const sz = bs.size as number;
              const sname = `fl${flIdx}_s${bs.output}`;
              bodyShmemMap.set(bs.output, {
                name: sname,
                dtype: DType.Float32,
                elemCount: sz / 4,
              });
              bodyShmemIds.add(bs.output);
              totalShmemBytes += sz;
              break;
            }
            case "free":
            case "recycle":
            case "incref":
              break;
            default:
              if (DEBUG >= 1)
                console.info(
                  `block_map fused: unsupported step in fori_loop body "${(bs as JitStep).type}", fallback`,
                );
              return null;
          }
          if (!valid) break;
        }
        if (!valid) {
          if (DEBUG >= 1)
            console.info("block_map fused: invalid fori_loop body, fallback");
          return null;
        }

        // Refine body shmem dtypes from body kernel outputs
        for (const { step: bs, kernel: bk } of bodyKernels) {
          for (let oi = 0; oi < bk.numOutputs; oi++) {
            const entry = bodyShmemMap.get(bs.outputs[oi]);
            if (entry) {
              entry.dtype = bk.outputs[oi].dtype;
              entry.elemCount = (entry.elemCount * 4) / byteWidth(entry.dtype);
            }
          }
        }

        // Barrier analysis for body kernel steps
        const bStepWrites = new Map<number, Set<JitId>>();
        const bStepReads = new Map<number, Set<JitId>>();
        for (let bsi = 0; bsi < bodyKernels.length; bsi++) {
          const { step: bs } = bodyKernels[bsi];
          const w = new Set<JitId>();
          const r = new Set<JitId>();
          for (const oid of bs.outputs) {
            if (bodyShmemIds.has(oid)) w.add(oid);
          }
          for (const iid of bs.inputs) {
            if (bodyShmemIds.has(iid)) r.add(iid);
          }
          bStepWrites.set(bsi, w);
          bStepReads.set(bsi, r);
        }
        const bodyBarriers = new Set<number>();
        for (let bsi = 1; bsi < bodyKernels.length; bsi++) {
          const reads = bStepReads.get(bsi)!;
          if (reads.size === 0) continue;
          for (let prev = 0; prev < bsi; prev++) {
            const writes = bStepWrites.get(prev)!;
            for (const id of reads) {
              if (writes.has(id)) {
                bodyBarriers.add(bsi);
                break;
              }
            }
            if (bodyBarriers.has(bsi)) break;
          }
        }

        foriLoops.push({
          foriStep: step as Extract<JitStep, { type: "fori_loop" }>,
          bodyKernels,
          bodyBarriers,
          bodyShmemMap,
          bodyShmemIds,
          bodyInputIds: bodyProg.inputs,
          bodyOutputIds: bodyProg.outputs,
          numConsts: step.numConsts,
          loopVar: `fl${flIdx}_i`,
        });
        codegenEntries.push({ type: "fori_loop", flIdx });
        break;
      }
      case "workgroup_assoc_scan": {
        // Analyze scan body: must be kernel-only, elementwise-only
        const wasIdx = workgroupAssocScans.length;
        const bodyProg = step.bodyProgram;
        const bodySteps = bodyProg.steps;
        const bodyKernels: WorkgroupAssocScanInfo["bodyKernels"] = [];
        const bodyShmemMap = new Map<
          JitId,
          { name: string; dtype: DType; elemCount: number }
        >();
        const bodyShmemIds = new Set<JitId>();

        let valid = true;
        for (const bs of bodySteps) {
          switch (bs.type) {
            case "execute": {
              if (bs.source instanceof Routine) {
                if (DEBUG >= 1)
                  console.info(
                    "block_map fused: routine in workgroup_assoc_scan body, fallback",
                  );
                return null;
              }
              const bk = bs.source as Kernel;
              if (bk.isSymbolic || bk.hasReduction) {
                valid = false;
                break;
              }
              bodyKernels.push({
                step: bs as Extract<JitStep, { type: "execute" }>,
                kernel: bk,
              });
              break;
            }
            case "malloc": {
              if (isSymbolicSize(bs.size)) {
                valid = false;
                break;
              }
              const sz = bs.size as number;
              const sname = `was${wasIdx}_s${bs.output}`;
              bodyShmemMap.set(bs.output, {
                name: sname,
                dtype: DType.Float32,
                elemCount: sz / 4,
              });
              bodyShmemIds.add(bs.output);
              totalShmemBytes += sz;
              break;
            }
            case "free":
            case "recycle":
            case "incref":
              break;
            default:
              valid = false;
              break;
          }
          if (!valid) break;
        }
        if (!valid) {
          if (DEBUG >= 1)
            console.info(
              "block_map fused: invalid workgroup_assoc_scan body, fallback",
            );
          return null;
        }

        // Refine body shmem dtypes from body kernel outputs
        for (const { step: bs, kernel: bk } of bodyKernels) {
          for (let oi = 0; oi < bk.numOutputs; oi++) {
            const entry = bodyShmemMap.get(bs.outputs[oi]);
            if (entry) {
              entry.dtype = bk.outputs[oi].dtype;
              entry.elemCount = (entry.elemCount * 4) / byteWidth(entry.dtype);
            }
          }
        }

        const numElems = step.numElems;
        const numConsts = step.numConsts;

        // Determine elem dtypes from elemAvals
        const elemDtypes: DType[] = step.elemAvals.map((a) => a.dtype);
        const elemCounts: number[] = step.elemAvals.map(
          (a) => a.size / (a.shape[0] as number),
        );

        // Allocate ping-pong shmem per elem (each: blockSize * elemCount elements)
        const pingPongNames: [string, string][] = [];
        for (let e = 0; e < numElems; e++) {
          const count = blockSize * elemCounts[e];
          const bytes = count * byteWidth(elemDtypes[e]);
          const pingName = `was${wasIdx}_ping_${e}`;
          const pongName = `was${wasIdx}_pong_${e}`;
          pingPongNames.push([pingName, pongName]);
          totalShmemBytes += bytes * 2; // ping + pong
        }

        const numRounds = Math.ceil(Math.log2(blockSize));

        workgroupAssocScans.push({
          wasStep: step as Extract<JitStep, { type: "workgroup_assoc_scan" }>,
          bodyKernels,
          bodyShmemMap,
          bodyShmemIds,
          bodyInputIds: bodyProg.inputs,
          bodyOutputIds: bodyProg.outputs,
          numConsts,
          numElems,
          elemDtypes,
          elemCounts,
          pingPongNames,
          numRounds,
        });
        codegenEntries.push({ type: "workgroup_assoc_scan", wasIdx });
        break;
      }
      default:
        // Unsupported step type (scan, dus, scatter_add, etc.)
        if (DEBUG >= 1)
          console.info(
            `block_map fused: unsupported step type "${(step as JitStep).type}", fallback`,
          );
        return null;
    }
  }

  // --- Account for reduction workspace shmem ---
  // Each reduction step needs a workspace array of blockSize elements for
  // the tree reduction. This is separate from the malloc-based shmem.
  const reduceWorkspaces: {
    stepIdx: number;
    dtype: DType;
    elemCount: number;
  }[] = [];
  for (let si = 0; si < kernelSteps.length; si++) {
    const re = stepReductions[si];
    if (re) {
      const wsBytes = blockSize * byteWidth(re.dtype);
      reduceWorkspaces.push({
        stepIdx: si,
        dtype: re.dtype,
        elemCount: blockSize,
      });
      totalShmemBytes += wsBytes;
    }
  }

  // --- Guard: shared memory budget ---
  const maxShmem = device.limits.maxComputeWorkgroupStorageSize;
  if (totalShmemBytes > maxShmem) {
    if (DEBUG >= 1)
      console.info(
        `block_map fused: shmem ${totalShmemBytes} > limit ${maxShmem}, fallback`,
      );
    return null;
  }

  // --- Determine body input/output buffer mapping ---
  // Body program inputs: [const0..constN, input0..inputM]
  // Body program outputs: [out0..outK]
  const bodyInputIds = bodyProgram.inputs; // JitId[]
  const bodyOutputIds = bodyProgram.outputs; // JitId[]
  const numBodyInputs = bodyInputIds.length;

  // Total global storage bindings needed:
  //   numConsts + numInputs (read) + numOutputs (read_write)
  const numOutputs = bodyOutputIds.length;
  const totalBindings = numConsts + numInputs + numOutputs;
  const maxBindings = device.limits.maxStorageBuffersPerShaderStage;
  if (totalBindings > maxBindings) {
    if (DEBUG >= 1)
      console.info(
        `block_map fused: ${totalBindings} bindings > max ${maxBindings}, fallback`,
      );
    return null;
  }

  // --- Refine shmem dtypes from kernel consumers ---
  // Walk kernel steps to determine what dtype each shmem buffer should use.
  // A kernel step's inputs map: step.inputs[j] → GlobalIndex arg[0] = j.
  // So step.inputs[bufferIndex] gives the JitId of the buffer.
  for (const { step, kernel } of kernelSteps) {
    // Check kernel outputs → determine dtype of their output JitIds
    for (let oi = 0; oi < kernel.numOutputs; oi++) {
      const outId = step.outputs[oi];
      const entry = shmemMap.get(outId);
      if (entry) {
        entry.dtype = kernel.outputs[oi].dtype;
        entry.elemCount = entry.sizeBytes / byteWidth(entry.dtype);
      }
    }
  }

  // --- Build JitId → buffer name mapping ---
  // Body inputs (consts + block inputs) → global storage buffer names
  // Intermediates (malloc outputs) → shmem array names
  // Body outputs → global storage result buffer names
  //
  // Note: pass-through outputs (output JitId === input JitId) are tracked
  // separately and handled as explicit copies in the shader.

  /** Map from JitId → input buffer name. */
  const inputIdToName = new Map<JitId, string>();
  /** Map from JitId → shmem or input name (for reading). */
  const idToReadName = new Map<JitId, string>();
  const idIsShmem = new Set<JitId>();
  /** Outputs that are pass-through copies of inputs. */
  const passThroughOutputs = new Map<
    number,
    { inputName: string; inputIdx: number }
  >(); // resultIdx → input info

  // Global inputs: in0..in{numBodyInputs-1}
  for (let i = 0; i < numBodyInputs; i++) {
    inputIdToName.set(bodyInputIds[i], `in${i}`);
    idToReadName.set(bodyInputIds[i], `in${i}`);
  }

  // Detect pass-through outputs
  for (let o = 0; o < numOutputs; o++) {
    const outId = bodyOutputIds[o];
    const bodyInputIdx = bodyInputIds.indexOf(outId);
    if (bodyInputIdx >= 0) {
      passThroughOutputs.set(o, {
        inputName: `in${bodyInputIdx}`,
        inputIdx: bodyInputIdx >= numConsts ? bodyInputIdx - numConsts : -1,
      });
    }
  }

  // Shared memory intermediates: shmem_{id}
  for (const [id] of shmemMap) {
    const name = `shmem_${id}`;
    idToReadName.set(id, name);
    idIsShmem.add(id);
  }

  // Determine which kernel outputs write directly to global result buffers
  // vs which write to shared memory.
  const outputWrittenByStep = new Map<
    JitId,
    { stepIdx: number; kernelOutputIdx: number }
  >();
  for (let si = 0; si < kernelSteps.length; si++) {
    const { step, kernel } = kernelSteps[si];
    for (let oi = 0; oi < kernel.numOutputs; oi++) {
      const outId = step.outputs[oi];
      if (bodyOutputIds.includes(outId)) {
        outputWrittenByStep.set(outId, { stepIdx: si, kernelOutputIdx: oi });
      }
    }
  }

  // --- Build data dependencies for barrier placement ---
  // Track which JitIds each step writes and reads for barrier insertion.
  // Conservative: barrier after every step that writes shmem, if a subsequent
  // step reads that shmem.
  const stepWrites = new Map<number, Set<JitId>>(); // stepIdx → written shmem ids
  const stepReads = new Map<number, Set<JitId>>(); // stepIdx → read shmem ids

  for (let si = 0; si < kernelSteps.length; si++) {
    const { step } = kernelSteps[si];
    const writes = new Set<JitId>();
    const reads = new Set<JitId>();
    for (const outId of step.outputs) {
      if (idIsShmem.has(outId)) writes.add(outId);
    }
    for (const inId of step.inputs) {
      if (idIsShmem.has(inId)) reads.add(inId);
    }
    stepWrites.set(si, writes);
    stepReads.set(si, reads);
  }

  // Determine where barriers are needed: before step si if it reads shmem
  // that was written by a previous step.
  const needsBarrierBefore = new Set<number>();
  for (let si = 1; si < kernelSteps.length; si++) {
    const reads = stepReads.get(si)!;
    if (reads.size === 0) continue;
    for (let prev = 0; prev < si; prev++) {
      const writes = stepWrites.get(prev)!;
      for (const id of reads) {
        if (writes.has(id)) {
          needsBarrierBefore.add(si);
          break;
        }
      }
      if (needsBarrierBefore.has(si)) break;
    }
  }

  // --- Codegen ---
  const shader: string[] = [];
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines: (string | symbol)[]) => {
    for (const line of lines) {
      if (line === pushIndent) indent += "  ";
      else if (line === popIndent) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : "");
    }
  };

  // Collect all distinct ops across all kernel steps for global functions
  let allOps: Map<AluOp, Set<DType>> = new Map();
  let needsF16 = false;
  const collectKernelOps = (kernel: Kernel) => {
    for (const output of kernel.outputs) {
      const ops = output.exp.distinctOps();
      allOps = mapSetUnion(allOps, ops);
      output.exp.fold((exp) => {
        if (exp.dtype === DType.Float16) needsF16 = true;
      });
    }
  };
  for (const { kernel } of kernelSteps) collectKernelOps(kernel);
  for (const fl of foriLoops) {
    for (const { kernel } of fl.bodyKernels) collectKernelOps(kernel);
  }
  for (const was of workgroupAssocScans) {
    for (const { kernel } of was.bodyKernels) collectKernelOps(kernel);
  }

  if (needsF16) {
    if (!device.features.has("shader-f16"))
      throw new Error("WebGPU device does not support shader-f16 feature");
    emit("enable f16;");
  }

  emit(headerWgsl);
  if (allOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allOps.has(AluOp.Erf) || allOps.has(AluOp.Erfc)) emit(erfSrc);
  emit("");

  // --- Global storage bindings ---
  // Determine dtype of each body input from kernel consumers
  const inputDtypes: (DType | null)[] = new Array(numBodyInputs).fill(null);
  for (const { step, kernel } of kernelSteps) {
    for (const output of kernel.outputs) {
      output.exp.fold((exp) => {
        if (exp.op === AluOp.GlobalIndex) {
          const bufIdx = exp.arg[0] as number;
          if (bufIdx < step.inputs.length) {
            // Map back to body input index
            const jitId = step.inputs[bufIdx];
            const bodyInputIdx = bodyInputIds.indexOf(jitId);
            if (bodyInputIdx >= 0) {
              inputDtypes[bodyInputIdx] = exp.dtype;
            }
          }
        }
      });
    }
  }

  for (let i = 0; i < numBodyInputs; i++) {
    const ty = dtypeToWgsl(inputDtypes[i] ?? DType.Float32, true);
    emit(`@group(0) @binding(${i}) var<storage, read> in${i} : array<${ty}>;`);
  }

  // Output bindings
  // Determine output dtypes from kernels that write to them
  const outputDtypes: DType[] = [];
  for (let o = 0; o < numOutputs; o++) {
    const outId = bodyOutputIds[o];
    let dtype = DType.Float32;
    const written = outputWrittenByStep.get(outId);
    if (written) {
      dtype =
        kernelSteps[written.stepIdx].kernel.outputs[written.kernelOutputIdx]
          .dtype;
    } else {
      // Pass-through output: use the input's dtype
      const ptInfo = passThroughOutputs.get(o);
      if (ptInfo) {
        const inDtype = inputDtypes[bodyInputIds.indexOf(outId)];
        if (inDtype) dtype = inDtype;
      }
    }
    outputDtypes.push(dtype);
    const ty = dtypeToWgsl(dtype, true);
    emit(
      `@group(0) @binding(${numBodyInputs + o}) var<storage, read_write> result${o} : array<${ty}>;`,
    );
  }

  // --- Shared memory declarations ---
  for (const [id, info] of shmemMap) {
    if (!idIsShmem.has(id)) continue;
    const ty = dtypeToWgsl(info.dtype, false);
    const name = idToReadName.get(id)!;
    emit(`var<workgroup> ${name}: array<${ty}, ${info.elemCount}>;`);
  }

  // Reduction workspace shmem arrays (one per reduction step)
  for (const ws of reduceWorkspaces) {
    const ty = dtypeToWgsl(ws.dtype, false);
    emit(
      `var<workgroup> reduce_ws_${ws.stepIdx}: array<${ty}, ${ws.elemCount}>;`,
    );
  }

  // Fori_loop body intermediate shmem arrays
  for (const fl of foriLoops) {
    for (const [, info] of fl.bodyShmemMap) {
      const ty = dtypeToWgsl(info.dtype, false);
      emit(`var<workgroup> ${info.name}: array<${ty}, ${info.elemCount}>;`);
    }
  }

  // WorkgroupAssociativeScan ping-pong shmem arrays + body intermediates
  for (const was of workgroupAssocScans) {
    for (let e = 0; e < was.numElems; e++) {
      const ty = dtypeToWgsl(was.elemDtypes[e], false);
      const count = blockSize * was.elemCounts[e];
      const [pingName, pongName] = was.pingPongNames[e];
      emit(`var<workgroup> ${pingName}: array<${ty}, ${count}>;`);
      emit(`var<workgroup> ${pongName}: array<${ty}, ${count}>;`);
    }
    for (const [, info] of was.bodyShmemMap) {
      const ty = dtypeToWgsl(info.dtype, false);
      emit(`var<workgroup> ${info.name}: array<${ty}, ${info.elemCount}>;`);
    }
  }

  // --- Workgroup size and grid ---
  // Each workgroup = 1 block. Thread i processes element i.
  // workgroup_id maps to block index in the grid.
  const wgSizeX = blockShape[0] ?? 1;
  const wgSizeY = blockShape.length > 1 ? blockShape[1] : 1;
  const wgSizeZ = blockShape.length > 2 ? blockShape[2] : 1;
  const totalBlocks = prod(gridShape);
  const [gridX, gridY] = calculateGrid(totalBlocks);

  // Build workgroup_size attribute
  let wgSizeStr = `${wgSizeX}`;
  if (wgSizeY > 1 || wgSizeZ > 1) wgSizeStr += `, ${wgSizeY}`;
  if (wgSizeZ > 1) wgSizeStr += `, ${wgSizeZ}`;

  emit(
    "",
    `@compute @workgroup_size(${wgSizeStr})`,
    "fn main(",
    `  @builtin(local_invocation_index) tidx: u32,`,
    `  @builtin(workgroup_id) wg_id: vec3<u32>,`,
    ") {",
    pushIndent,
  );

  // Compute flat block index from workgroup_id
  if (gridY === 1) {
    emit("let block_idx: u32 = wg_id.x;");
  } else {
    emit(`let block_idx: u32 = wg_id.y * ${gridX}u + wg_id.x;`);
  }

  // Guard: skip excess workgroups when grid is padded
  emit(`if (block_idx >= ${totalBlocks}u) { return; }`);

  // Compute per-axis block indices (row-major from block_idx)
  if (gridRank === 1) {
    emit("let block_i0: u32 = block_idx;");
  } else {
    emit("var _remaining: u32 = block_idx;");
    for (let g = gridRank - 1; g >= 0; g--) {
      if (g === 0) {
        emit(`let block_i${g}: u32 = _remaining;`);
      } else {
        emit(`let block_i${g}: u32 = _remaining % ${gridShape[g]}u;`);
        emit(`_remaining = _remaining / ${gridShape[g]}u;`);
      }
    }
  }

  // Compute per-axis thread indices within the block
  if (blockShape.length === 1) {
    emit("let tidx_0: u32 = tidx;");
  } else if (blockShape.length === 2) {
    // 2D: tidx = row * cols + col
    emit(`let tidx_0: u32 = tidx / ${blockShape[1]}u;`); // row
    emit(`let tidx_1: u32 = tidx % ${blockShape[1]}u;`); // col
  } else {
    // 3D: tidx = d0 * (d1*d2) + d1 * d2 + d2
    const d12 = blockShape[1] * blockShape[2];
    emit(`let tidx_0: u32 = tidx / ${d12}u;`);
    emit(`let tidx_1: u32 = (tidx % ${d12}u) / ${blockShape[2]}u;`);
    emit(`let tidx_2: u32 = tidx % ${blockShape[2]}u;`);
  }

  // --- Per-thread validity (boundary blocks) ---
  // For the last block along each axis, some threads may be out-of-bounds.
  // `valid` is true iff this thread maps to a real element for ALL axes.
  // Invalid threads read 0 from global inputs and don't write to global outputs,
  // matching the zero-pad semantics of the eagerness fallback.
  if (hasBoundary) {
    const terms: string[] = [];
    for (let g = 0; g < gridRank; g++) {
      if (axisDims[g] !== null) {
        terms.push(
          `(block_i${g} * ${blockShape[g]}u + tidx_${g} < ${axisDims[g]}u)`,
        );
      }
    }
    emit(`let valid: bool = ${terms.join(" && ")};`);
  }

  // Build global offset for each input and output.
  // For input i, axis g: globalOffset = block_i{g} * blockShape[g]
  // The kernel indexes with gidx (0..blockSize-1), but in the fused shader
  // we remap GlobalIndex reads to global buffer positions.

  // For each body input, precompute the global base offset
  for (let i = 0; i < numInputs; i++) {
    const axes = params.inAxes[i];
    const inShape = params.inputShapes[i];
    const nd = inShape.length;
    // Build stride array for the input
    const strides: number[] = new Array(nd);
    strides[nd - 1] = 1;
    for (let d = nd - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * inShape[d + 1];
    }
    // Compute base offset: sum of block_i{g} * blockShape[g] * stride[axes[g]]
    const terms: string[] = [];
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) {
        const ax = axes[g]!;
        const blockStride = blockShape[g] * strides[ax];
        terms.push(`block_i${g} * ${blockStride}u`);
      }
    }
    const baseExpr = terms.length > 0 ? terms.join(" + ") : "0u";
    emit(`let in_base_${i}: u32 = ${baseExpr};`);
  }

  // For each body output, precompute the global base offset
  for (let o = 0; o < numOutputs; o++) {
    const axes = params.outAxes[o];
    const outShape = params.outputShapes[o];
    const nd = outShape.length;
    const strides: number[] = new Array(nd);
    strides[nd - 1] = 1;
    for (let d = nd - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * outShape[d + 1];
    }
    const terms: string[] = [];
    for (let g = 0; g < gridRank; g++) {
      if (axes[g] !== null) {
        const ax = axes[g]!;
        const blockStride = blockShape[g] * strides[ax];
        terms.push(`block_i${g} * ${blockStride}u`);
      }
    }
    const baseExpr = terms.length > 0 ? terms.join(" + ") : "0u";
    emit(`let out_base_${o}: u32 = ${baseExpr};`);
  }

  // Declare gidx — the body kernel expressions reference this variable.
  // Each thread processes one element: gidx = tidx.
  emit("let gidx: i32 = i32(tidx);");

  // Phony assignments for unused inputs
  if (numBodyInputs > 0) {
    emit(
      Array.from({ length: numBodyInputs }, (_, i) => `_ = &in${i};`).join(" "),
    );
  }

  // --- Helper: create gen() function for a kernel step ---
  // The gen() function translates AluExp trees into WGSL expressions.
  // It is parameterized by a resolveGlobalIndex callback that maps
  // GlobalIndex reads to the correct WGSL (different for parent vs body steps).
  function createGen(
    kernel: Kernel,
    prefix: string,
    resolveGlobalIndex: (
      bufIdx: number,
      indexExpr: string,
      dtype: DType,
    ) => string,
    variableOverrides?: Map<string, string>,
  ): (exp: AluExp) => string {
    let gensymCount = 0;
    const gensym = () => `${prefix}_alu${gensymCount++}`;
    const isGensym = (text: string) =>
      text.startsWith(prefix + "_alu") &&
      /^\d+$/.test(text.slice(prefix.length + 4));

    const references = new Map<AluExp, number>();
    const seen = new Set<AluExp>();
    const countReferences = (exp: AluExp) => {
      references.set(exp, (references.get(exp) ?? 0) + 1);
      if (!seen.has(exp)) {
        seen.add(exp);
        for (const src of exp.src) countReferences(src);
      }
    };
    for (const output of kernel.outputs) countReferences(output.exp);

    const expContext = new Map<AluExp, string>();
    const gen = (exp: AluExp): string => {
      if (expContext.has(exp)) return expContext.get(exp)!;
      const { op, src, dtype, arg } = exp;

      let source = "";
      if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
        const a = gen(src[0]);
        const b = gen(src[1]);
        if (op === AluOp.Add) {
          if (dtype === DType.Bool) source = `(${a} || ${b})`;
          else source = `(${a} + ${b})`;
        } else if (op === AluOp.Sub) source = `(${a} - ${b})`;
        else if (op === AluOp.Mul) {
          if (dtype === DType.Bool) source = `(${a} && ${b})`;
          else source = `(${a} * ${b})`;
        } else if (op === AluOp.Idiv)
          source = isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
        else if (op === AluOp.Mod) source = `(${a} % ${b})`;
        else if (op === AluOp.Min) {
          if (dtype === DType.Bool) source = `(${a} && ${b})`;
          else source = `min(${strip1(a)}, ${strip1(b)})`;
        } else if (op === AluOp.Max) {
          if (dtype === DType.Bool) source = `(${a} || ${b})`;
          else source = `max(${strip1(a)}, ${strip1(b)})`;
        } else if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
        else if (op === AluOp.Cmpne) {
          if (isFloatDtype(src[0].dtype)) {
            const x = isGensym(a) ? a : gensym();
            if (x !== a) emit(`let ${x} = ${a};`);
            source = `(${x} != ${b} || min(${x}, ${dtypeToWgsl(src[0].dtype)}(inf())) != ${x})`;
          } else {
            source = `(${a} != ${b})`;
          }
        }
      } else if (AluGroup.Unary.has(op)) {
        if (op === AluOp.Reciprocal && src[0].op === AluOp.Sqrt) {
          const a = gen(src[0].src[0]);
          source = `inverseSqrt(${a})`;
        } else {
          const a = gen(src[0]);
          if (op === AluOp.Sin) source = `sin(${strip1(a)})`;
          else if (op === AluOp.Cos) source = `cos(${strip1(a)})`;
          else if (op === AluOp.Asin) source = `asin(${strip1(a)})`;
          else if (op === AluOp.Atan) source = `atan(${strip1(a)})`;
          else if (op === AluOp.Exp) source = `exp(${strip1(a)})`;
          else if (op === AluOp.Log) source = `log(${strip1(a)})`;
          else if (op === AluOp.Erf || op === AluOp.Erfc) {
            const funcName = op === AluOp.Erf ? "erf" : "erfc";
            if (dtype !== DType.Float32) {
              source = `${dtypeToWgsl(dtype)}(${funcName}(f32(${strip1(a)})))`;
            } else {
              source = `${funcName}(${strip1(a)})`;
            }
          } else if (op === AluOp.Sqrt) source = `sqrt(${strip1(a)})`;
          else if (op === AluOp.Reciprocal) source = `(1.0 / ${a})`;
          else if (op === AluOp.Floor) source = `floor(${strip1(a)})`;
          else if (op === AluOp.Ceil) source = `ceil(${strip1(a)})`;
          else if (op === AluOp.Cast)
            source = `${dtypeToWgsl(dtype)}(${strip1(a)})`;
          else if (op === AluOp.Bitcast)
            source = `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
        }
      } else if (op === AluOp.Where) {
        source = `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
      } else if (op === AluOp.Threefry2x32) {
        const x = gensym();
        const [k0, k1, c0, c1] = src.map((s) => strip1(gen(s)));
        emit(
          `let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`,
        );
        if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
        else if (arg === 0) source = `${x}.x`;
        else if (arg === 1) source = `${x}.y`;
      } else if (op === AluOp.Const) {
        return constToWgsl(dtype, arg);
      } else if (op === AluOp.Special) {
        return arg[0] as string;
      } else if (op === AluOp.Variable) {
        return variableOverrides?.get(arg as string) ?? (arg as string);
      } else if (op === AluOp.GlobalView) {
        // Rewrite to Where(valid, GlobalIndex(...), Const(0)) and recurse
        const [gid, st] = arg as [number, import("../../shape").ShapeTracker];
        const rewritten = accessorGlobal(dtype, gid, st, src);
        return gen(rewritten);
      } else if (op === AluOp.GlobalIndex) {
        const bufIdx = arg[0] as number;
        const indexExpr = strip1(gen(src[0]));
        source = resolveGlobalIndex(bufIdx, indexExpr, dtype);
        if (dtype === DType.Bool) source = `(${source} != 0)`;
      }

      if (!source) {
        throw new Error(`block_map fused: unsupported AluOp ${op}`);
      }

      const typeName = dtypeToWgsl(dtype);
      if ((references.get(exp) ?? 0) > 1) {
        const name = gensym();
        expContext.set(exp, name);
        emit(`let ${name}: ${typeName} = ${strip1(source)};`);
        return name;
      } else {
        expContext.set(exp, source);
        return source;
      }
    };
    return gen;
  }

  // --- Generate codegen entries ---
  for (const entry of codegenEntries) {
    if (entry.type === "kernel") {
      const si = entry.kernelIdx;
      if (needsBarrierBefore.has(si)) {
        emit("workgroupBarrier();");
      }

      const { step, kernel } = kernelSteps[si];

      // Build input name mapping for this step
      const stepInputNames: string[] = [];
      const stepInputIsGlobal: boolean[] = [];
      const stepInputBodyIdx: number[] = [];
      for (let j = 0; j < step.inputs.length; j++) {
        const jitId = step.inputs[j];
        const name = idToReadName.get(jitId) ?? inputIdToName.get(jitId);
        if (name) {
          stepInputNames.push(name);
          const isGlobalInput = inputIdToName.has(jitId);
          stepInputIsGlobal.push(isGlobalInput);
          const bodyIdx = bodyInputIds.indexOf(jitId);
          stepInputBodyIdx.push(
            bodyIdx >= numConsts ? bodyIdx - numConsts : -1,
          );
        } else {
          stepInputNames.push(`__unknown_${jitId}`);
          stepInputIsGlobal.push(false);
          stepInputBodyIdx.push(-1);
        }
      }

      const gen = createGen(kernel, `s${si}`, (bufIdx, indexExpr, dtype) => {
        if (stepInputIsGlobal[bufIdx]) {
          const inputIdx = stepInputBodyIdx[bufIdx];
          if (inputIdx >= 0) {
            const readExpr = `${stepInputNames[bufIdx]}[i32(in_base_${inputIdx}) + ${indexExpr}]`;
            return hasBoundary
              ? `select(${dtypeToWgsl(dtype)}(0), ${readExpr}, valid)`
              : readExpr;
          } else {
            return `${stepInputNames[bufIdx]}[${indexExpr}]`;
          }
        } else {
          return `${stepInputNames[bufIdx]}[${indexExpr}]`;
        }
      });

      // Generate WGSL for each kernel output
      const re = stepReductions[si];

      if (re) {
        // --- Reduction kernel: tree reduction in shared memory ---
        const outId = step.outputs[0];
        const rhs = strip1(gen(kernel.outputs[0].exp));
        const reDtype = re.dtype;
        const reTy = dtypeToWgsl(reDtype, false);
        const wsName = `reduce_ws_${si}`;

        emit(`${wsName}[tidx] = ${reTy}(${rhs});`);
        emit("workgroupBarrier();");

        let startStride = 1;
        while (startStride * 2 < blockSize) startStride *= 2;

        for (let stride = startStride; stride >= 1; stride >>= 1) {
          emit(
            `if (tidx < ${stride}u && tidx + ${stride}u < ${blockSize}u) {`,
            pushIndent,
          );
          const thisSlot = `${wsName}[tidx]`;
          const otherSlot = `${wsName}[tidx + ${stride}u]`;
          if (re.op === AluOp.Add) emit(`${thisSlot} += ${otherSlot};`);
          else if (re.op === AluOp.Mul) emit(`${thisSlot} *= ${otherSlot};`);
          else if (re.op === AluOp.Min) {
            if (reDtype === DType.Bool)
              emit(`${thisSlot} = ${thisSlot} && ${otherSlot};`);
            else emit(`${thisSlot} = min(${thisSlot}, ${otherSlot});`);
          } else if (re.op === AluOp.Max) {
            if (reDtype === DType.Bool)
              emit(`${thisSlot} = ${thisSlot} || ${otherSlot};`);
            else emit(`${thisSlot} = max(${thisSlot}, ${otherSlot});`);
          }
          emit(popIndent, "}");
          emit("workgroupBarrier();");
        }

        emit("if (tidx == 0u) {", pushIndent);

        const accVar = `${wsName}[0u]`;
        let finalValue: string;
        const isIdentityEpilogue =
          re.epilogue.op === AluOp.Variable && re.epilogue.arg === "acc";
        if (isIdentityEpilogue) {
          finalValue = accVar;
        } else {
          const epilogueGen = createGen(
            kernel,
            `s${si}_ep`,
            () => accVar,
            new Map([["acc", accVar]]),
          );
          finalValue = strip1(epilogueGen(re.epilogue));
        }

        const resultIdx = bodyOutputIds.indexOf(outId);
        if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
          const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
          const castFinal =
            resultTy !== reTy ? `${resultTy}(${finalValue})` : finalValue;
          emit(
            `result${resultIdx}[i32(out_base_${resultIdx})] = ${castFinal};`,
          );
        } else if (idIsShmem.has(outId)) {
          const shmemName = idToReadName.get(outId)!;
          emit(`${shmemName}[0u] = ${finalValue};`);
        }
        emit(popIndent, "}");
        emit("workgroupBarrier();");
      } else {
        // --- Elementwise kernel: each thread writes its own element ---
        for (let oi = 0; oi < kernel.numOutputs; oi++) {
          const outId = step.outputs[oi];
          const rhs = strip1(gen(kernel.outputs[oi].exp));

          const resultIdx = bodyOutputIds.indexOf(outId);
          if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
            const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
            const castRhs =
              resultTy !== dtypeToWgsl(kernel.outputs[oi].exp.dtype)
                ? `${resultTy}(${rhs})`
                : rhs;
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${castRhs};`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${castRhs};`,
              );
            }
          } else if (idIsShmem.has(outId)) {
            const shmemName = idToReadName.get(outId)!;
            emit(`${shmemName}[tidx] = ${rhs};`);
          }
        }
      }
    } else if (entry.type === "fori_loop") {
      // --- fori_loop step ---
      const fl = foriLoops[entry.flIdx];
      const fs = fl.foriStep;
      const numBodyConsts = fl.numConsts;
      const numCarries = fs.initCarries.length;

      // Build mapping from body input JitIds to WGSL names + properties
      // Body inputs: [const0..constN, idx, carry0..carryM]
      const bodyInputInfo: {
        name: string;
        isGlobal: boolean;
        parentInputIdx: number; // for block offset
        isIndex: boolean;
      }[] = [];

      for (let bi = 0; bi < fl.bodyInputIds.length; bi++) {
        if (bi < numBodyConsts) {
          // Const input — maps to a parent JitId
          const parentJitId = fs.consts[bi];
          const parentName =
            idToReadName.get(parentJitId) ??
            inputIdToName.get(parentJitId) ??
            `__fl_unknown_${parentJitId}`;
          const isGlobalInput = inputIdToName.has(parentJitId);
          const parentBodyIdx = bodyInputIds.indexOf(parentJitId);
          bodyInputInfo.push({
            name: parentName,
            isGlobal: isGlobalInput && !idIsShmem.has(parentJitId),
            parentInputIdx:
              parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1,
            isIndex: false,
          });
        } else if (bi === numBodyConsts) {
          // Loop index — scalar i32
          bodyInputInfo.push({
            name: fl.loopVar,
            isGlobal: false,
            parentInputIdx: -1,
            isIndex: true,
          });
        } else {
          // Carry input — maps to the parent output shmem
          const carryIdx = bi - numBodyConsts - 1;
          const parentOutId = fs.outputs[carryIdx];
          const carryShmemName = idToReadName.get(parentOutId)!;
          bodyInputInfo.push({
            name: carryShmemName,
            isGlobal: false,
            parentInputIdx: -1,
            isIndex: false,
          });
        }
      }

      // Build mapping from body JitIds to WGSL read names
      const bodyIdToName = new Map<JitId, string>();
      for (let bi = 0; bi < fl.bodyInputIds.length; bi++) {
        bodyIdToName.set(fl.bodyInputIds[bi], bodyInputInfo[bi].name);
      }
      for (const [bodyJitId, info] of fl.bodyShmemMap) {
        bodyIdToName.set(bodyJitId, info.name);
      }
      // Map body output JitIds to carry output shmem names
      // (body outputs that are also body shmem intermediates already mapped)
      // For outputs that map to parent output shmem, add the mapping
      for (let ci = 0; ci < numCarries; ci++) {
        const bodyOutId = fl.bodyOutputIds[ci];
        const parentOutId = fs.outputs[ci];
        const carryShmemName = idToReadName.get(parentOutId)!;
        // If the body output is a body shmem intermediate, it already has
        // a body shmem name. We want it to write to the carry shmem instead.
        // Override the mapping:
        bodyIdToName.set(bodyOutId, carryShmemName);
      }

      // Initialize carry shmem from init carry values
      for (let ci = 0; ci < numCarries; ci++) {
        const parentInitId = fs.initCarries[ci];
        const parentOutId = fs.outputs[ci];
        const initName =
          idToReadName.get(parentInitId) ?? inputIdToName.get(parentInitId);
        const carryName = idToReadName.get(parentOutId)!;
        if (initName && initName !== carryName) {
          // Copy init value to carry shmem
          if (inputIdToName.has(parentInitId) && !idIsShmem.has(parentInitId)) {
            // Init is a global input — apply block offset
            const parentBodyIdx = bodyInputIds.indexOf(parentInitId);
            const inIdx =
              parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1;
            if (inIdx >= 0) {
              const readExpr = `${initName}[i32(in_base_${inIdx}) + i32(tidx)]`;
              emit(
                `${carryName}[tidx] = ${hasBoundary ? `select(${dtypeToWgsl(shmemMap.get(parentOutId)?.dtype ?? DType.Float32)}(0), ${readExpr}, valid)` : readExpr};`,
              );
            } else {
              emit(`${carryName}[tidx] = ${initName}[i32(tidx)];`);
            }
          } else {
            // Init is shmem — direct copy
            emit(`${carryName}[tidx] = ${initName}[tidx];`);
          }
        }
        // If initName === carryName, they're the same shmem (recycled) — no copy needed
      }
      emit("workgroupBarrier();");

      // Emit WGSL for loop
      emit(
        `for (var ${fl.loopVar}: i32 = ${fs.lower}; ${fl.loopVar} < ${fs.upper}; ${fl.loopVar}++) {`,
        pushIndent,
      );

      // Generate body kernel steps
      for (let bsi = 0; bsi < fl.bodyKernels.length; bsi++) {
        if (fl.bodyBarriers.has(bsi)) {
          emit("workgroupBarrier();");
        }

        const { step: bStep, kernel: bKernel } = fl.bodyKernels[bsi];

        // Build per-step input mapping
        const bStepInputInfo: typeof bodyInputInfo = [];
        for (let j = 0; j < bStep.inputs.length; j++) {
          const jitId = bStep.inputs[j];
          const biIdx = fl.bodyInputIds.indexOf(jitId);
          if (biIdx >= 0) {
            bStepInputInfo.push(bodyInputInfo[biIdx]);
          } else {
            // Body shmem intermediate or carry output
            const sname = bodyIdToName.get(jitId);
            bStepInputInfo.push({
              name: sname ?? `__fl_unknown_${jitId}`,
              isGlobal: false,
              parentInputIdx: -1,
              isIndex: false,
            });
          }
        }

        const gen = createGen(
          bKernel,
          `fl${entry.flIdx}_s${bsi}`,
          (bufIdx, indexExpr, dtype) => {
            const info = bStepInputInfo[bufIdx];
            if (info.isIndex) {
              // Loop variable — scalar, cast to read dtype
              return `${dtypeToWgsl(dtype)}(${info.name})`;
            }
            if (info.isGlobal) {
              const inputIdx = info.parentInputIdx;
              if (inputIdx >= 0) {
                const readExpr = `${info.name}[i32(in_base_${inputIdx}) + ${indexExpr}]`;
                return hasBoundary
                  ? `select(${dtypeToWgsl(dtype)}(0), ${readExpr}, valid)`
                  : readExpr;
              } else {
                return `${info.name}[${indexExpr}]`;
              }
            }
            // Shmem (body intermediate or carry) — direct index
            return `${info.name}[${indexExpr}]`;
          },
        );

        // Check for per-thread contraction (reduction kernel)
        const bRe = bKernel.outputs[0]?.reduction ?? null;
        if (bRe && (bKernel.size as number) === blockSize) {
          // Per-thread accumulate loop: each thread computes its own
          // output by iterating over the reduction axis sequentially.
          const outId = bStep.outputs[0];
          const reDtype = bRe.dtype;
          const reTy = dtypeToWgsl(reDtype, false);
          const reSize = bRe.size as number;
          const prefix = `fl${entry.flIdx}_s${bsi}`;
          const accName = `${prefix}_acc`;
          emit(`{`);
          emit(
            `  var ${accName}: ${reTy} = ${constToWgsl(reDtype, bRe.identity)};`,
          );
          emit(
            `  for (var ridx: i32 = 0; ridx < ${reSize}; ridx++) {`,
            pushIndent,
          );
          // gen() references gidx (→ tidx) and ridx (→ loop var) in the expression
          const rhs = strip1(gen(bKernel.outputs[0].exp));
          if (bRe.op === AluOp.Add) emit(`  ${accName} += ${rhs};`);
          else if (bRe.op === AluOp.Mul) emit(`  ${accName} *= ${rhs};`);
          else if (bRe.op === AluOp.Min)
            emit(`  ${accName} = min(${accName}, ${rhs});`);
          else if (bRe.op === AluOp.Max)
            emit(`  ${accName} = max(${accName}, ${rhs});`);
          emit(popIndent, `  }`);
          // Apply epilogue if non-identity
          const isIdentityEpilogue =
            bRe.epilogue.op === AluOp.Variable && bRe.epilogue.arg === "acc";
          let finalValue = accName;
          if (!isIdentityEpilogue) {
            const epilogueGen = createGen(
              bKernel,
              `${prefix}_ep`,
              (bufIdx, indexExpr, dtype) => {
                const info = bStepInputInfo[bufIdx];
                if (info.isIndex) {
                  return `${dtypeToWgsl(dtype)}(${info.name})`;
                }
                if (info.isGlobal) {
                  const inputIdx = info.parentInputIdx;
                  if (inputIdx >= 0) {
                    const readExpr = `${info.name}[i32(in_base_${inputIdx}) + ${indexExpr}]`;
                    return hasBoundary
                      ? `select(${dtypeToWgsl(dtype)}(0), ${readExpr}, valid)`
                      : readExpr;
                  } else {
                    return `${info.name}[${indexExpr}]`;
                  }
                }
                return `${info.name}[${indexExpr}]`;
              },
              new Map([["acc", accName]]),
            );
            finalValue = strip1(epilogueGen(bRe.epilogue));
          }
          const targetName = bodyIdToName.get(outId);
          if (targetName) {
            emit(`  ${targetName}[tidx] = ${finalValue};`);
          }
          emit(`}`);
        } else {
          // Elementwise output writes — guard if kernel size < blockSize
          const bKernelSize = bKernel.size as number;
          const needSizeGuard = bKernelSize < blockSize;
          if (needSizeGuard) emit(`if (tidx < ${bKernelSize}u) {`, pushIndent);
          for (let oi = 0; oi < bKernel.numOutputs; oi++) {
            const outId = bStep.outputs[oi];
            const rhs = strip1(gen(bKernel.outputs[oi].exp));
            const targetName = bodyIdToName.get(outId);
            if (targetName) {
              emit(`${targetName}[tidx] = ${rhs};`);
            }
          }
          if (needSizeGuard) emit(popIndent, "}");
        }
      }

      emit(popIndent, "}"); // end for loop

      // Write final carries to parent output (global result buffers)
      for (let ci = 0; ci < numCarries; ci++) {
        const parentOutId = fs.outputs[ci];
        const resultIdx = bodyOutputIds.indexOf(parentOutId);
        if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
          const carryName = idToReadName.get(parentOutId)!;
          const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
          if (hasBoundary) {
            emit(`if (valid) {`, pushIndent);
            emit(
              `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${resultTy}(${carryName}[tidx]);`,
            );
            emit(popIndent, "}");
          } else {
            emit(
              `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${resultTy}(${carryName}[tidx]);`,
            );
          }
        }
      }
    } else if (entry.type === "workgroup_assoc_scan") {
      // --- Workgroup associative scan step (Kogge-Stone) ---
      const was = workgroupAssocScans[entry.wasIdx];
      const ws = was.wasStep;
      const numBodyConsts = was.numConsts;

      // Build mapping from body input JitIds to WGSL read info
      const bodyInputInfo: {
        name: string;
        isGlobal: boolean;
        parentInputIdx: number;
      }[] = [];

      for (let bi = 0; bi < was.bodyInputIds.length; bi++) {
        if (bi < numBodyConsts) {
          // Const input — maps to a parent JitId
          const parentJitId = ws.consts[bi];
          const parentName =
            idToReadName.get(parentJitId) ??
            inputIdToName.get(parentJitId) ??
            `__was_unknown_${parentJitId}`;
          const isGlobalInput =
            inputIdToName.has(parentJitId) && !idIsShmem.has(parentJitId);
          const parentBodyIdx = bodyInputIds.indexOf(parentJitId);
          bodyInputInfo.push({
            name: parentName,
            isGlobal: isGlobalInput,
            parentInputIdx:
              parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1,
          });
        } else {
          // a_elem or b_elem — will be resolved dynamically per round
          // placeholder (overridden per-kernel by custom resolveGlobalIndex)
          bodyInputInfo.push({
            name: "__placeholder",
            isGlobal: false,
            parentInputIdx: -1,
          });
        }
      }

      // Build mapping from body JitIds to WGSL read names (for intermediates)
      const bodyIdToName = new Map<JitId, string>();
      for (let bi = 0; bi < numBodyConsts; bi++) {
        bodyIdToName.set(was.bodyInputIds[bi], bodyInputInfo[bi].name);
      }
      for (const [bodyJitId, info] of was.bodyShmemMap) {
        bodyIdToName.set(bodyJitId, info.name);
      }

      // Load input elements into ping shmem arrays
      for (let e = 0; e < was.numElems; e++) {
        const parentElemId = ws.elems[e];
        const [pingName] = was.pingPongNames[e];
        const elemCount = was.elemCounts[e];
        const parentName =
          idToReadName.get(parentElemId) ?? inputIdToName.get(parentElemId);
        const isGlobalInput =
          inputIdToName.has(parentElemId) && !idIsShmem.has(parentElemId);
        const parentBodyIdx = bodyInputIds.indexOf(parentElemId);
        const inIdx =
          parentBodyIdx >= numConsts ? parentBodyIdx - numConsts : -1;

        if (elemCount > 1) {
          emit(
            `for (var _was_ei: u32 = 0u; _was_ei < ${elemCount}u; _was_ei++) {`,
            pushIndent,
          );
          const idx = `tidx * ${elemCount}u + _was_ei`;
          if (isGlobalInput && inIdx >= 0) {
            const readExpr = `${parentName}[i32(in_base_${inIdx}) + i32(${idx})]`;
            emit(
              `${pingName}[${idx}] = ${hasBoundary ? `select(${dtypeToWgsl(was.elemDtypes[e])}(0), ${readExpr}, valid)` : readExpr};`,
            );
          } else if (parentName) {
            emit(`${pingName}[${idx}] = ${parentName}[${idx}];`);
          }
          emit(popIndent, "}");
        } else {
          if (isGlobalInput && inIdx >= 0) {
            const readExpr = `${parentName}[i32(in_base_${inIdx}) + i32(tidx)]`;
            emit(
              `${pingName}[tidx] = ${hasBoundary ? `select(${dtypeToWgsl(was.elemDtypes[e])}(0), ${readExpr}, valid)` : readExpr};`,
            );
          } else if (parentName) {
            emit(`${pingName}[tidx] = ${parentName}[tidx];`);
          }
        }
      }
      emit("workgroupBarrier();");

      // Unrolled Kogge-Stone rounds
      // Even rounds: read from ping, write to pong
      // Odd rounds: read from pong, write to ping
      for (let r = 0; r < was.numRounds; r++) {
        const stride = 1 << r;
        const readNames = was.pingPongNames.map(([ping, pong]) =>
          r % 2 === 0 ? ping : pong,
        );
        const writeNames = was.pingPongNames.map(([ping, pong]) =>
          r % 2 === 0 ? pong : ping,
        );

        emit(`if (tidx >= ${stride}u) {`, pushIndent);

        // Evaluate body kernels: fn(a, b) where
        //   a = readBuf[(tidx - stride) * elemCount + ei]
        //   b = readBuf[tidx * elemCount + ei]
        // Map body output JitIds to writeBuf[tidx * elemCount + ei]
        for (let bsi = 0; bsi < was.bodyKernels.length; bsi++) {
          const { step: bStep, kernel: bKernel } = was.bodyKernels[bsi];

          // Build per-step input resolution
          const bStepInputNames: string[] = [];
          const bStepInputKind: ("const" | "a" | "b" | "shmem")[] = [];
          const bStepInputElemIdx: number[] = []; // which elem in the pytree
          for (let j = 0; j < bStep.inputs.length; j++) {
            const jitId = bStep.inputs[j];
            const biIdx = was.bodyInputIds.indexOf(jitId);
            if (biIdx >= 0 && biIdx < numBodyConsts) {
              // Const
              bStepInputNames.push(bodyInputInfo[biIdx].name);
              bStepInputKind.push("const");
              bStepInputElemIdx.push(-1);
            } else if (
              biIdx >= numBodyConsts &&
              biIdx < numBodyConsts + was.numElems
            ) {
              // a_elem
              const e = biIdx - numBodyConsts;
              bStepInputNames.push(readNames[e]);
              bStepInputKind.push("a");
              bStepInputElemIdx.push(e);
            } else if (biIdx >= numBodyConsts + was.numElems) {
              // b_elem
              const e = biIdx - numBodyConsts - was.numElems;
              bStepInputNames.push(readNames[e]);
              bStepInputKind.push("b");
              bStepInputElemIdx.push(e);
            } else {
              // Body shmem intermediate
              const sname = bodyIdToName.get(jitId);
              bStepInputNames.push(sname ?? `__was_unknown_${jitId}`);
              bStepInputKind.push("shmem");
              bStepInputElemIdx.push(-1);
            }
          }

          const gen = createGen(
            bKernel,
            `was${entry.wasIdx}_r${r}_s${bsi}`,
            (bufIdx, indexExpr, dtype) => {
              const kind = bStepInputKind[bufIdx];
              const name = bStepInputNames[bufIdx];
              if (kind === "const") {
                const inf =
                  bodyInputInfo[was.bodyInputIds.indexOf(bStep.inputs[bufIdx])];
                if (inf.isGlobal) {
                  if (inf.parentInputIdx >= 0) {
                    const readExpr = `${name}[i32(in_base_${inf.parentInputIdx}) + ${indexExpr}]`;
                    return hasBoundary
                      ? `select(${dtypeToWgsl(dtype)}(0), ${readExpr}, valid)`
                      : readExpr;
                  }
                  return `${name}[${indexExpr}]`;
                }
                return `${name}[${indexExpr}]`;
              } else if (kind === "a") {
                const e = bStepInputElemIdx[bufIdx];
                const ec = was.elemCounts[e];
                if (ec > 1) {
                  return `${name}[(tidx - ${stride}u) * ${ec}u + u32(${indexExpr})]`;
                }
                return `${name}[tidx - ${stride}u]`;
              } else if (kind === "b") {
                const e = bStepInputElemIdx[bufIdx];
                const ec = was.elemCounts[e];
                if (ec > 1) {
                  return `${name}[tidx * ${ec}u + u32(${indexExpr})]`;
                }
                return `${name}[tidx]`;
              }
              // shmem intermediate
              return `${name}[${indexExpr}]`;
            },
          );

          for (let oi = 0; oi < bKernel.numOutputs; oi++) {
            const outId = bStep.outputs[oi];
            const rhs = strip1(gen(bKernel.outputs[oi].exp));
            // Map body output to the correct write buffer
            const bodyOutIdx = was.bodyOutputIds.indexOf(outId);
            if (bodyOutIdx >= 0) {
              const e = bodyOutIdx;
              const ec = was.elemCounts[e];
              if (ec > 1) {
                // Need to emit for each sub-element — but gen already uses gidx
                // which maps to a single element. For multi-element outputs the
                // kernel iterates gidx over elemCount, so we wrap in a loop.
                emit(
                  `for (var _was_oi: u32 = 0u; _was_oi < ${ec}u; _was_oi++) {`,
                  pushIndent,
                );
                // Re-generate with explicit eidx
                const genEi = createGen(
                  bKernel,
                  `was${entry.wasIdx}_r${r}_s${bsi}_ei`,
                  (bufIdx2, _indexExpr, dtype2) => {
                    const kind2 = bStepInputKind[bufIdx2];
                    const name2 = bStepInputNames[bufIdx2];
                    if (kind2 === "const") {
                      const inf2 =
                        bodyInputInfo[
                          was.bodyInputIds.indexOf(bStep.inputs[bufIdx2])
                        ];
                      if (inf2.isGlobal && inf2.parentInputIdx >= 0) {
                        const readExpr2 = `${name2}[i32(in_base_${inf2.parentInputIdx}) + i32(_was_oi)]`;
                        return hasBoundary
                          ? `select(${dtypeToWgsl(dtype2)}(0), ${readExpr2}, valid)`
                          : readExpr2;
                      }
                      return `${name2}[i32(_was_oi)]`;
                    } else if (kind2 === "a") {
                      const e2 = bStepInputElemIdx[bufIdx2];
                      return `${name2}[(tidx - ${stride}u) * ${was.elemCounts[e2]}u + _was_oi]`;
                    } else if (kind2 === "b") {
                      const e2 = bStepInputElemIdx[bufIdx2];
                      return `${name2}[tidx * ${was.elemCounts[e2]}u + _was_oi]`;
                    }
                    return `${name2}[i32(_was_oi)]`;
                  },
                );
                const rhsEi = strip1(genEi(bKernel.outputs[oi].exp));
                emit(`${writeNames[e]}[tidx * ${ec}u + _was_oi] = ${rhsEi};`);
                emit(popIndent, "}");
              } else {
                emit(`${writeNames[e]}[tidx] = ${rhs};`);
              }
            } else {
              // Body intermediate → write to body shmem
              const sname = bodyIdToName.get(outId);
              if (sname) {
                emit(
                  `${sname}[${bKernel.numOutputs > 1 ? `i32(tidx) * ${bKernel.numOutputs} + ${oi}` : "tidx"}] = ${rhs};`,
                );
              }
            }
          }
        }

        emit(popIndent, "} else {", pushIndent);

        // Copy: writeBuf[tidx] = readBuf[tidx]
        for (let e = 0; e < was.numElems; e++) {
          const ec = was.elemCounts[e];
          if (ec > 1) {
            emit(
              `for (var _was_ci: u32 = 0u; _was_ci < ${ec}u; _was_ci++) {`,
              pushIndent,
            );
            emit(
              `${writeNames[e]}[tidx * ${ec}u + _was_ci] = ${readNames[e]}[tidx * ${ec}u + _was_ci];`,
            );
            emit(popIndent, "}");
          } else {
            emit(`${writeNames[e]}[tidx] = ${readNames[e]}[tidx];`);
          }
        }

        emit(popIndent, "}");
        emit("workgroupBarrier();");
      }

      // Write output from final buffer to parent output shmem / result buffers
      // After numRounds rounds, final data is in:
      //   numRounds even → ping; numRounds odd → pong
      const finalNames = was.pingPongNames.map(([ping, pong]) =>
        was.numRounds % 2 === 0 ? ping : pong,
      );

      for (let e = 0; e < was.numElems; e++) {
        const parentOutId = ws.outputs[e];
        const resultIdx = bodyOutputIds.indexOf(parentOutId);
        const ec = was.elemCounts[e];

        if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
          // Output goes to global result buffer
          const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
          if (ec > 1) {
            emit(
              `for (var _was_wi: u32 = 0u; _was_wi < ${ec}u; _was_wi++) {`,
              pushIndent,
            );
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx * ${ec}u + _was_wi)] = ${resultTy}(${finalNames[e]}[tidx * ${ec}u + _was_wi]);`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx * ${ec}u + _was_wi)] = ${resultTy}(${finalNames[e]}[tidx * ${ec}u + _was_wi]);`,
              );
            }
            emit(popIndent, "}");
          } else {
            if (hasBoundary) {
              emit(`if (valid) {`, pushIndent);
              emit(
                `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${resultTy}(${finalNames[e]}[tidx]);`,
              );
              emit(popIndent, "}");
            } else {
              emit(
                `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${resultTy}(${finalNames[e]}[tidx]);`,
              );
            }
          }
        } else {
          // Output goes to a parent shmem buffer (another step reads it)
          const shmemName = idToReadName.get(parentOutId);
          if (shmemName) {
            if (ec > 1) {
              emit(
                `for (var _was_wi: u32 = 0u; _was_wi < ${ec}u; _was_wi++) {`,
                pushIndent,
              );
              emit(
                `${shmemName}[tidx * ${ec}u + _was_wi] = ${finalNames[e]}[tidx * ${ec}u + _was_wi];`,
              );
              emit(popIndent, "}");
            } else {
              emit(`${shmemName}[tidx] = ${finalNames[e]}[tidx];`);
            }
          }
        }
      }
    }
  }

  // --- Pass-through outputs: copy input → output ---
  if (passThroughOutputs.size > 0 && hasBoundary) {
    emit("if (valid) {", pushIndent);
  }
  for (const [resultIdx, info] of passThroughOutputs) {
    if (info.inputIdx >= 0) {
      // Non-const input: apply block offset
      emit(
        `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${info.inputName}[i32(in_base_${info.inputIdx}) + i32(tidx)];`,
      );
    } else {
      // Const input: no block offset
      emit(
        `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${info.inputName}[i32(tidx)];`,
      );
    }
  }
  if (passThroughOutputs.size > 0 && hasBoundary) {
    emit(popIndent, "}");
  }

  emit(popIndent, "}");

  return {
    code: shader.join("\n"),
    numInputs: numBodyInputs,
    numOutputs,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }],
    sharedMemoryBytes: totalShmemBytes > 0 ? totalShmemBytes : undefined,
  };
}
