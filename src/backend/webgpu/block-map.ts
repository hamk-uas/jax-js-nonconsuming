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
 *   - Non-divisible dimensions (boundary blocks)
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

  // --- Guard: evenly divisible dimensions ---
  // Boundary blocks require per-thread validity guards on every read/write.
  // For now, fall back to the JS executor which handles boundary padding.
  // Future: emit per-thread `_valid` flag and guard all memory accesses.
  const gridRank = blockShape.length;
  for (let g = 0; g < gridRank; g++) {
    for (let i = 0; i < numInputs; i++) {
      const axes = params.inAxes[i];
      if (axes[g] !== null) {
        const ax = axes[g]!;
        const dim = params.inputShapes[i][ax];
        if (dim % blockShape[g] !== 0) {
          if (DEBUG >= 1)
            console.info(
              `block_map fused: input ${i} axis ${ax} dim ${dim} not divisible by block ${blockShape[g]}, fallback`,
            );
          return null;
        }
      }
    }
    for (let o = 0; o < params.outputShapes.length; o++) {
      const axes = params.outAxes[o];
      if (axes[g] !== null) {
        const ax = axes[g]!;
        const dim = params.outputShapes[o][ax];
        if (dim % blockShape[g] !== 0) {
          if (DEBUG >= 1)
            console.info(
              `block_map fused: output ${o} axis ${ax} dim ${dim} not divisible by block ${blockShape[g]}, fallback`,
            );
          return null;
        }
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
        // Determine dtype and element count from consumers.
        // We'll figure this out from the first kernel that reads this buffer.
        shmemMap.set(step.output, {
          sizeBytes,
          dtype: DType.Float32, // placeholder, refined below
          elemCount: sizeBytes / 4, // placeholder, refined below
        });
        totalShmemBytes += sizeBytes;
        break;
      }
      case "free":
      case "recycle":
      case "incref":
        // These don't affect codegen
        break;
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
  for (const { kernel } of kernelSteps) {
    for (const output of kernel.outputs) {
      const ops = output.exp.distinctOps();
      allOps = mapSetUnion(allOps, ops);
      output.exp.fold((exp) => {
        if (exp.dtype === DType.Float16) needsF16 = true;
      });
    }
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

  // --- Generate kernel step code ---
  // For each kernel step, inline its WGSL expression.
  // The gen() function is local to each step but shares CSE state for
  // multi-output steps.
  for (let si = 0; si < kernelSteps.length; si++) {
    if (needsBarrierBefore.has(si)) {
      emit("workgroupBarrier();");
    }

    const { step, kernel } = kernelSteps[si];

    // Build the input name mapping for this step.
    // step.inputs[j] → the JitId of the j-th buffer argument.
    // We need to translate GlobalIndex(arg=[j, len], src=[indexExpr]) into
    // the appropriate WGSL expression.
    const stepInputNames: string[] = [];
    const stepInputIsGlobal: boolean[] = [];
    const stepInputBodyIdx: number[] = []; // index into body inputs (for offset computation)

    for (let j = 0; j < step.inputs.length; j++) {
      const jitId = step.inputs[j];
      const name = idToReadName.get(jitId) ?? inputIdToName.get(jitId);
      if (name) {
        stepInputNames.push(name);
        const isGlobalInput = inputIdToName.has(jitId);
        stepInputIsGlobal.push(isGlobalInput);
        // Is it a non-const body input?
        const bodyIdx = bodyInputIds.indexOf(jitId);
        stepInputBodyIdx.push(bodyIdx >= numConsts ? bodyIdx - numConsts : -1);
      } else {
        // This shouldn't happen in a well-formed body program
        stepInputNames.push(`__unknown_${jitId}`);
        stepInputIsGlobal.push(false);
        stepInputBodyIdx.push(-1);
      }
    }

    // CSE infrastructure for this step
    let gensymCount = 0;
    const gensym = () => `s${si}_alu${gensymCount++}`;
    const isGensym = (text: string) => /^s\d+_alu\d+$/.test(text);

    const references = new Map<AluExp, number>();
    const seen = new Set<AluExp>();
    const countReferences = (exp: AluExp) => {
      references.set(exp, (references.get(exp) ?? 0) + 1);
      if (!seen.has(exp)) {
        seen.add(exp);
        for (const src of exp.src) countReferences(src);
      }
    };

    // Count references for all outputs of this kernel
    for (const output of kernel.outputs) {
      countReferences(output.exp);
    }

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
        return arg as string;
      } else if (op === AluOp.GlobalIndex) {
        // This is the critical remapping:
        // arg[0] = buffer index within this kernel step's inputs
        // src[0] = index expression (e.g., gidx)
        const bufIdx = arg[0] as number;
        const indexExpr = strip1(gen(src[0]));

        if (stepInputIsGlobal[bufIdx]) {
          // Global buffer read — add base offset for block position
          const inputIdx = stepInputBodyIdx[bufIdx];
          if (inputIdx >= 0) {
            // Non-const input: apply block offset
            source = `${stepInputNames[bufIdx]}[i32(in_base_${inputIdx}) + ${indexExpr}]`;
          } else {
            // Const input: no block offset (constants are shared)
            source = `${stepInputNames[bufIdx]}[${indexExpr}]`;
          }
        } else {
          // Shared memory read — direct indexing within block
          source = `${stepInputNames[bufIdx]}[${indexExpr}]`;
        }
        if (dtype === DType.Bool) source = `(${source} != 0)`;
      }

      if (!source) {
        if (DEBUG >= 1)
          console.info(`block_map fused: unsupported AluOp ${op}, fallback`);
        // Return a placeholder that will cause a compile error if used
        source = `/* unsupported op ${op} */`;
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

    // Generate the WGSL for each kernel output
    const re = stepReductions[si];

    if (re) {
      // --- Reduction kernel: tree reduction in shared memory ---
      // This kernel has exactly 1 output (multi-output reductions rejected above).
      const outId = step.outputs[0];
      const rhs = strip1(gen(kernel.outputs[0].exp));
      const reDtype = re.dtype;
      const reTy = dtypeToWgsl(reDtype, false);
      const wsName = `reduce_ws_${si}`;

      // 1. Each thread stores its per-element value into the workspace
      emit(`${wsName}[tidx] = ${reTy}(${rhs});`);
      emit("workgroupBarrier();");

      // 2. Tree reduction with halving stride (handles non-power-of-2)
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

      // 3. Thread 0 applies the epilogue and writes the result
      emit("if (tidx == 0u) {", pushIndent);

      // Apply reduction epilogue (e.g., acc/size for mean)
      // The epilogue is an AluExp that references the "acc" variable.
      const accVar = `${wsName}[0u]`;
      let finalValue: string;
      // Check if epilogue is identity (just acc variable)
      const isIdentityEpilogue =
        re.epilogue.op === AluOp.Variable && re.epilogue.arg === "acc";
      if (isIdentityEpilogue) {
        finalValue = accVar;
      } else {
        // Generate the epilogue expression with acc substituted
        // Use a fresh gensym context for the epilogue
        expContext.set(AluExp.variable(reDtype, "acc"), accVar);
        finalValue = strip1(gen(re.epilogue));
      }

      const resultIdx = bodyOutputIds.indexOf(outId);
      if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
        // Write scalar to global output — reduction produces 1 element per block
        const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
        const castFinal =
          resultTy !== reTy ? `${resultTy}(${finalValue})` : finalValue;
        emit(`result${resultIdx}[i32(out_base_${resultIdx})] = ${castFinal};`);
      } else if (idIsShmem.has(outId)) {
        // Write reduced scalar to shmem[0] for subsequent steps
        const shmemName = idToReadName.get(outId)!;
        emit(`${shmemName}[0u] = ${finalValue};`);
      }
      emit(popIndent, "}");
      // Barrier so all threads can read the reduced scalar in subsequent steps
      emit("workgroupBarrier();");
    } else {
      // --- Elementwise kernel: each thread writes its own element ---
      for (let oi = 0; oi < kernel.numOutputs; oi++) {
        const outId = step.outputs[oi];
        const rhs = strip1(gen(kernel.outputs[oi].exp));

        const resultIdx = bodyOutputIds.indexOf(outId);
        if (resultIdx >= 0 && !passThroughOutputs.has(resultIdx)) {
          // Write to global output buffer with block offset
          const resultTy = dtypeToWgsl(outputDtypes[resultIdx], true);
          const castRhs =
            resultTy !== dtypeToWgsl(kernel.outputs[oi].exp.dtype)
              ? `${resultTy}(${rhs})`
              : rhs;
          emit(
            `result${resultIdx}[i32(out_base_${resultIdx}) + i32(tidx)] = ${castRhs};`,
          );
        } else if (idIsShmem.has(outId)) {
          // Write to shared memory
          const shmemName = idToReadName.get(outId)!;
          emit(`${shmemName}[tidx] = ${rhs};`);
        }
      }
    }
  }

  // --- Pass-through outputs: copy input → output ---
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
