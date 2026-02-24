import {
  AluExp,
  AluGroup,
  AluOp,
  byteWidth,
  DType,
  isFloatDtype,
  Kernel,
} from "../alu";
import {
  Backend,
  type BackendCapabilities,
  Device,
  Executable,
  Slot,
  SlotError,
  UnsupportedOpError,
} from "../backend";
import { Routine } from "../routine";
import { isSymbolicSize } from "../shape";
import { tuneNullopt, tuneWebgpu, type WebGPUTuneResult } from "../tuner";
import {
  DEBUG,
  findPow2,
  FpHash,
  mapSetUnion,
  prod,
  range,
  strip1,
} from "../utils";
import { erfSrc, threefrySrc } from "./webgpu/builtins";
import {
  calculateGrid,
  constToWgsl,
  dtypeToWgsl,
  gridOffsetY,
  headerWgsl,
  ShaderInfo,
} from "./webgpu/codegen";
import { SyncReader } from "./webgpu/reader";
import { createRoutineShader } from "./webgpu/routines";
import {
  createAllIterationsOffsetsBuffer,
  type ScanBindingInfo,
  wrapRoutineForScan,
} from "./webgpu/scan-wrapper";

interface ShaderDispatch extends ShaderInfo {
  pipeline: GPUComputePipeline; // Compiled pipeline for the shader.
}

// ---------------------------------------------------------------------------
// Types for WebGPU native scan (multi-kernel shader)
// ---------------------------------------------------------------------------

export interface NativeScanMultiStep {
  /** The kernel to execute. */
  kernel: Kernel;
  /** Input mapping: indices into [consts, carry, xs] flattened. */
  inputs: number[];
  /** Which carry slot this kernel writes to (0..numCarry-1). */
  outputCarryIdx: number;
  /** Size of output in elements (not bytes). */
  outputSize: number;
}

/** Parameters for multi-kernel native scan execution on WebGPU. */
export interface NativeScanMultiParams {
  length: number;
  numConsts: number;
  constSizes: number[];
  numCarry: number;
  carrySizes: number[];
  numX: number;
  xsStrides: number[];
  numY: number;
  ysStrides: number[];
  steps: NativeScanMultiStep[];
  reverse?: boolean;
}

// ---------------------------------------------------------------------------
// Types for WebGPU native associative scan (fused Kogge-Stone)
// ---------------------------------------------------------------------------

/** A reindexed kernel step for the associative scan body. */
export interface AssocScanStep {
  /** Reindexed kernel (gids relative to [consts, a-leaves, b-leaves, internals]). */
  kernel: Kernel;
  /** Input mapping for this step. */
  inputSlots: number[];
  /** Which internal buffer this step writes to. */
  outputInternalIdx: number;
}

/**
 * Parameters for WebGPU fused associative scan.
 * The shader fuses all body kernel steps into a single dispatch per
 * Kogge-Stone round, reducing from ~20 dispatches/round to 1.
 */
export interface WebGPUAssocScanParams {
  /** Number of constant inputs. */
  numConsts: number;
  /** Number of pytree leaves. */
  numLeaves: number;
  /** Per-leaf element count in typed values (e.g. 16 for a 4×4 f32 matrix). */
  leafElemCounts: number[];
  /** Body kernel steps with reindexed gids. */
  steps: AssocScanStep[];
  /** Typed element counts for internal buffers. */
  internalElemCounts: number[];
  /** Whether to reverse the scan direction. */
  reverse: boolean;
  /**
   * Mapping from output leaf index to internal buffer index.
   * leafToInternalIdx[k] = the internal buffer that produces leaf k.
   */
  leafToInternalIdx: number[];
  /** dtype for all leaves (must be homogeneous). */
  dtype: DType;
}

/**
 * Prepared WebGPU fused associative scan — ready to dispatch.
 */
export interface PreparedWebGPUAssocScan {
  /** Compiled compute pipeline for one Kogge-Stone round. */
  pipeline: GPUComputePipeline;
  /** Bind group layout for storage bindings (ping, pong, consts). */
  storageLayout: GPUBindGroupLayout;
  /** Bind group layout for uniform bindings (stride, N). */
  uniformLayout: GPUBindGroupLayout;
  /** Workgroup size for dispatches. */
  workgroupSize: number;
  /** Shader source code (for debugging). */
  code: string;
}

// ---------------------------------------------------------------------------
// Types for WebGPU preencoded-routine scan (P4)
// ---------------------------------------------------------------------------

/** Parameters for preencoded scan execution on WebGPU (routine body like matmul). */
export interface PreencodedScanParams {
  /** Number of scan iterations. */
  length: number;
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Strides (in ELEMENTS) along axis 0 for each xs input. */
  xsElemStrides: number[];
  /** Strides (in ELEMENTS) along axis 0 for each stacked y output. */
  ysElemStrides: number[];
  /** The prepared routine executable for the body. */
  bodyRoutine: Executable<ShaderDispatch[]>;
  /** Number of carry arrays. */
  numCarry: number;
  /** Number of xs inputs. */
  numX: number;
  /** Number of ys outputs. */
  numY: number;
  /** Number of const inputs (bound before carry). */
  numConsts: number;
  /** Whether to scan in reverse order. */
  reverse?: boolean;
  /** For each routine input binding i, the body jaxpr JitId, used to classify as const/carry/xs. */
  routineInputJitIds: number[];
  /** For each routine output binding i, the body output index. */
  routineOutputJitIds: number[];
}

/** Prepared preencoded scan with wrapped shaders and offset buffer. */
export interface PreparedPreencodedScan {
  params: PreencodedScanParams;
  /** Shaders with uniform offset support. */
  wrappedShaders: ShaderDispatch[];
  /** GPU buffer containing all iteration offsets. */
  offsetBuffer: GPUBuffer;
  /** Alignment of each iteration's offset data in the buffer. */
  offsetAlignment: number;
  /** Per-carry copy strategy for ys stacking (true = use shader copy). */
  copyUsesShader: boolean[];
  /** Bind group layout for the uniform offset group (with dynamic offset). */
  uniformLayout: GPUBindGroupLayout;
}

const COPY_WORKGROUP_SIZE = 64;

const COPY_SHADER_CODE = String.raw`
${headerWgsl}

struct CopyParams {
  srcOffset: u32,
  dstOffset: u32,
  size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(1) @binding(0) var<uniform> params: CopyParams;

fn byte_mask(n: u32) -> u32 {
  if (n >= 4u) { return 0xffffffffu; }
  return (1u << (n * 8u)) - 1u;
}

fn load_unaligned(offset: u32) -> u32 {
  let word = offset >> 2u;
  let shift = (offset & 3u) * 8u;
  if (shift == 0u) {
    return src[word];
  }
  let low = src[word];
  let high = src[word + 1u];
  return (low >> shift) | (high << (32u - shift));
}

@compute @workgroup_size(${COPY_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let gid = id.x + id.y * ${gridOffsetY}u;
  let firstDstWord = params.dstOffset >> 2u;
  let wordIdx = firstDstWord + gid;
  let lastDstWord = (params.dstOffset + params.size + 3u) >> 2u;
  if (wordIdx >= lastDstWord) { return; }

  let wordByteStart = wordIdx * 4u;
  let copyStart = max(params.dstOffset, wordByteStart);
  let copyEnd = min(params.dstOffset + params.size, wordByteStart + 4u);
  let nbytes = copyEnd - copyStart;
  let srcByteOff = params.srcOffset + (copyStart - params.dstOffset);
  let value = load_unaligned(srcByteOff);

  if (nbytes == 4u) {
    dst[wordIdx] = value;
  } else {
    let shift = (copyStart & 3u) * 8u;
    let mask = byte_mask(nbytes) << shift;
    let cur = dst[wordIdx];
    dst[wordIdx] = (cur & ~mask) | ((value << shift) & mask);
  }
}
`.trim();

/** Implementation of `Backend` that uses WebGPU in browsers. */
export class WebGPUBackend implements Backend {
  readonly type: Device = "webgpu";
  readonly maxArgs: number;
  readonly capabilities: BackendCapabilities;

  readonly pipelines: ShaderPipelineCache;
  readonly syncReader: SyncReader;
  readonly buffers: Map<
    Slot,
    {
      ref: number;
      size: number; // Refers to "true size" requested, less padding.
      buffer: GPUBuffer;
    }
  >;
  nextSlot: number;

  #cachedShaderMap = new Map<bigint, ShaderInfo>();
  #reusableZsb: GPUBuffer;
  #copyPipeline: GPUComputePipeline | null = null;

  /**
   * Buffer pool: recently-freed GPUBuffers indexed by padded byte size.
   * Avoids expensive device.createBuffer() / buffer.destroy() cycles for
   * same-size allocations, which are very common in JIT-compiled programs.
   *
   * **Peak-memory guarantee:** Before each JIT execution, `configurePool()`
   * evicts pool entries whose sizes won't be needed and caps total retained
   * bytes at the program's peak live bytes. This ensures physical peak memory
   * never exceeds what was already required during execution — the pool is
   * free from a peak-memory perspective.
   *
   * A per-size-class cap (MAX_POOL_PER_SIZE) limits redundant same-size
   * entries. MAX_POOL_BYTES_DEFAULT is a fallback byte budget used when
   * no JitProgram has configured the pool yet (e.g., in eager mode).
   */
  #bufferPool = new Map<number, GPUBuffer[]>();
  #poolBudgetBytes: number = 64 * 1024 * 1024; // default: 64 MB (eager mode)
  #poolCurrentBytes: number = 0;
  static readonly MAX_POOL_PER_SIZE = 4;
  /** Fallback budget for eager mode (no JitProgram to derive peak from). */
  static readonly MAX_POOL_BYTES_DEFAULT = 64 * 1024 * 1024; // 64 MB

  /**
   * Total bytes of GPU storage buffers currently allocated by jax-js
   * (live + pooled). Incremented on createBuffer, decremented on destroy.
   * Staging/read buffers are excluded since they're transient.
   */
  #gpuAllocatedBytes: number = 0;

  /**
   * High-water mark: the maximum value `#gpuAllocatedBytes` ever reached.
   * Useful for verifying that optimizations (fusion, recycling, pooling)
   * reduce peak GPU memory. Reset via `resetPeakGpuAllocatedBytes()`.
   */
  #gpuPeakBytes: number = 0;

  // --- Batch dispatch state ---
  // When non-null, dispatch() encodes into this shared encoder instead of
  // creating a new one per call. endBatch() submits and cleans up.
  #batchEncoder: GPUCommandEncoder | null = null;
  #batchDepth = 0;
  #batchUniformsToDestroy: GPUBuffer[] = [];
  #batchDeferredFreeBuffers: GPUBuffer[] = [];

  constructor(readonly device: GPUDevice) {
    if (DEBUG >= 3 && device.adapterInfo) {
      console.info(
        "webgpu adapter:",
        device.adapterInfo.vendor,
        device.adapterInfo.architecture,
      );
    }
    this.maxArgs = this.device.limits.maxStorageBuffersPerShaderStage - 1;
    this.capabilities = {
      atomicF32Add: device.features.has(
        "shader-f32-atomic-add" as GPUFeatureName,
      ),
      sharedMemory: false,
      multiOutputKernel: true,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    };
    this.pipelines = new ShaderPipelineCache(device);
    this.syncReader = new SyncReader(device);
    this.buffers = new Map();
    this.nextSlot = 1;

    // Special "zero-size buffer" that's reused across all allocations of size
    // zero, backing slots for those allocations.
    //
    // WebGPU allows creating buffers of size 0, but you cannot actually make
    // bindings of size 0 when calling `createBindGroup()`. The simplest way to
    // handle this is to just create a buffer of minimum size (4 bytes) and
    // reuse that across all zero-size allocations.
    this.#reusableZsb = this.#createBuffer(4);

    device.addEventListener("uncapturederror", (event) => {
      console.error("Uncaptured error in WebGPU backend:", event.error.message);
    });
  }

  /** Number of live backend slots (excludes pooled buffers). */
  slotCount(): number {
    return this.buffers.size;
  }

  /** Total GPU bytes held by jax-js: live buffer bytes + pooled buffer bytes. */
  gpuAllocatedBytes(): number {
    return this.#gpuAllocatedBytes;
  }

  /**
   * High-water mark: the maximum `gpuAllocatedBytes()` ever reached.
   * Useful for verifying that fusion, recycling, and pooling reduce peak memory.
   */
  peakGpuAllocatedBytes(): number {
    return this.#gpuPeakBytes;
  }

  /** Reset the peak memory watermark to the current allocation level. */
  resetPeakGpuAllocatedBytes(): void {
    this.#gpuPeakBytes = this.#gpuAllocatedBytes;
  }

  /** Buffer pool diagnostic: pooled buffer count, pooled bytes, and byte budget. */
  poolStats(): {
    pooledBuffers: number;
    pooledBytes: number;
    budgetBytes: number;
  } {
    let pooledBuffers = 0;
    for (const list of this.#bufferPool.values()) pooledBuffers += list.length;
    return {
      pooledBuffers,
      pooledBytes: this.#poolCurrentBytes,
      budgetBytes: this.#poolBudgetBytes,
    };
  }

  malloc(size: number, initialData?: Uint8Array<ArrayBuffer>): Slot {
    let buffer: GPUBuffer;
    // All GPUBuffer must be a multiple of 4 bytes in length, to support copy
    // operations. Pad it to a multiple of 4.
    const paddedSize = Math.ceil(size / 4) * 4;
    if (size === 0) {
      buffer = this.#reusableZsb;
    } else if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      // Try to reuse a pooled buffer for initial data too.
      const pooled = this.#poolPop(paddedSize);
      if (pooled) {
        buffer = pooled;
        if (initialData.byteLength % 4 === 0) {
          this.device.queue.writeBuffer(buffer, 0, initialData);
        } else {
          const aligned = initialData.byteLength - (initialData.byteLength % 4);
          this.device.queue.writeBuffer(buffer, 0, initialData, 0, aligned);
          const remainder = new Uint8Array(4);
          remainder.set(initialData.subarray(aligned));
          this.device.queue.writeBuffer(buffer, aligned, remainder);
        }
      } else if (initialData.byteLength < 4096) {
        buffer = this.#createBuffer(paddedSize, { mapped: true });
        new Uint8Array(buffer.getMappedRange(), 0, size).set(initialData);
        buffer.unmap();
      } else {
        // getMappedRange() seems slower for large buffers, use writeBuffer() instead.
        buffer = this.#createBuffer(paddedSize);
        if (initialData.byteLength % 4 === 0) {
          this.device.queue.writeBuffer(buffer, 0, initialData);
        } else {
          // Copy all but the last few bytes, then copy 4 bytes as remainder.
          const aligned = initialData.byteLength - (initialData.byteLength % 4);
          this.device.queue.writeBuffer(buffer, 0, initialData, 0, aligned);
          const remainder = new Uint8Array(4);
          remainder.set(initialData.subarray(aligned));
          this.device.queue.writeBuffer(buffer, aligned, remainder);
        }
      }
    } else {
      // No initial data — try the pool first.
      buffer = this.#poolPop(paddedSize) ?? this.#createBuffer(paddedSize);
    }

    const slot = this.nextSlot++;
    this.buffers.set(slot, { buffer, size, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.buffers.delete(slot);
      if (buffer.buffer !== this.#reusableZsb) {
        if (this.#batchEncoder) {
          // Defer pool-or-destroy: buffer may still be referenced by
          // commands in the batch encoder that haven't been submitted yet.
          this.#batchDeferredFreeBuffers.push(buffer.buffer);
        } else if (!this.#poolPush(buffer.buffer)) {
          this.#gpuAllocatedBytes -= buffer.buffer.size;
          buffer.buffer.destroy();
        }
      }
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    const { buffer, size } = this.#getBuffer(slot);
    if (buffer === this.#reusableZsb) return new Uint8Array();
    if (start === undefined) start = 0;
    if (count === undefined) count = size - start;

    // Need a GPUBuffer with MAP_READ usage when transfering data to host.
    const paddedSize = Math.ceil(count / 4) * 4;
    const staging = this.#createBuffer(paddedSize, { read: true });
    try {
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, paddedSize);
      this.device.queue.submit([commandEncoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const arrayBuffer = staging.getMappedRange();
      return new Uint8Array(arrayBuffer.slice(), 0, count);
    } finally {
      staging.destroy();
    }
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const { buffer, size } = this.#getBuffer(slot);
    if (buffer === this.#reusableZsb) return new Uint8Array();
    if (start === undefined) start = 0;
    if (count === undefined) count = size - start;
    return this.syncReader.read(buffer, start, count);
  }

  copyBufferToBuffer(
    src: Slot,
    srcOffset: number,
    dst: Slot,
    dstOffset: number,
    size: number,
  ): void {
    if (size === 0) return;
    const srcBuf = this.#getBuffer(src);
    const dstBuf = this.#getBuffer(dst);
    // Use batch encoder when a batch is active (e.g., fallback scan loop),
    // otherwise create a standalone command encoder.
    const encoder = this.#batchEncoder ?? this.device.createCommandEncoder();
    const ownEncoder = !this.#batchEncoder;
    // WebGPU copyBufferToBuffer requires 4-byte alignment on offsets and size.
    if (srcOffset % 4 === 0 && dstOffset % 4 === 0 && size % 4 === 0) {
      // Fast GPU copy path — all alignments satisfied.
      encoder.copyBufferToBuffer(
        srcBuf.buffer,
        srcOffset,
        dstBuf.buffer,
        dstOffset,
        size,
      );
      if (ownEncoder) this.device.queue.submit([encoder.finish()]);
    } else {
      // Unaligned fallback: use WGSL copy shader (stays on GPU).
      const uniformBuf = this.#encodeCopyWithShader(
        encoder,
        srcBuf.buffer,
        srcOffset,
        dstBuf.buffer,
        dstOffset,
        size,
      );
      if (ownEncoder) {
        this.device.queue.submit([encoder.finish()]);
        if (uniformBuf) uniformBuf.destroy();
      } else if (uniformBuf) {
        // Defer uniform buffer destruction until batch ends
        this.#batchUniformsToDestroy.push(uniformBuf);
      }
    }
  }

  #cachedShader(kernel: Kernel): ShaderInfo {
    const cacheKey = FpHash.hash(kernel);
    let result = this.#cachedShaderMap.get(cacheKey);
    if (!result) {
      result = pipelineSource(this.device, kernel, this.capabilities);
      this.#cachedShaderMap.set(cacheKey, result);
    }
    return result;
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<ShaderDispatch[]>> {
    const shader = this.#cachedShader(kernel);
    const pipeline = await this.pipelines.prepare(shader);
    return new Executable(kernel, [{ ...shader, pipeline }]);
  }

  prepareKernelSync(kernel: Kernel): Executable<ShaderDispatch[]> {
    const shader = this.#cachedShader(kernel);
    const pipeline = this.pipelines.prepareSync(shader);
    return new Executable(kernel, [{ ...shader, pipeline }]);
  }

  async prepareRoutine(
    routine: Routine,
  ): Promise<Executable<ShaderDispatch[]>> {
    const shaders = createRoutineShader(this.device, routine);
    const dispatches = await Promise.all(
      shaders.map(async (shader) => {
        const pipeline = await this.pipelines.prepare(shader);
        return { ...shader, pipeline };
      }),
    );
    return new Executable(routine, dispatches);
  }

  prepareRoutineSync(routine: Routine): Executable<ShaderDispatch[]> {
    const shaders = createRoutineShader(this.device, routine);
    const dispatches = shaders.map((shader) => {
      const pipeline = this.pipelines.prepareSync(shader);
      return { ...shader, pipeline };
    });
    return new Executable(routine, dispatches);
  }

  dispatch(
    exe: Executable<ShaderDispatch[]>,
    inputs: Slot[],
    outputs: Slot[],
    dynamicParams?: number[],
  ): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot).buffer);
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
    pipelineSubmit(
      this.device,
      exe.data,
      inputBuffers,
      outputBuffers,
      dynamicParams,
      this.#batchEncoder ?? undefined,
      this.#batchEncoder ? this.#batchUniformsToDestroy : undefined,
    );
  }

  // ---------------------------------------------------------------------------
  // Native scan methods (P3: WebGPU multi-kernel scan)
  // ---------------------------------------------------------------------------

  /**
   * Prepare a multi-kernel native scan shader.
   * Returns null if codegen fails.
   */
  prepareNativeScanMulti(
    params: NativeScanMultiParams,
  ): Executable<ShaderDispatch[]> | null {
    const { steps } = params;
    if (!steps || steps.length === 0) return null;

    try {
      const shader = nativeScanMultiShaderSource(this.device, params);
      const pipeline = this.pipelines.prepareSync(shader);
      const firstKernel = steps[0].kernel;
      return new Executable(firstKernel, [{ ...shader, pipeline }]);
    } catch (e) {
      if (DEBUG >= 2) {
        console.warn("WebGPU native scan multi codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Dispatch a native scan shader.
   *
   * Buffer binding order: [consts, initCarry, xs, carryOut, ysStacked]
   */
  dispatchNativeScanGeneral(
    exe: Executable<ShaderDispatch[]>,
    _params: NativeScanMultiParams,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void {
    const allBuffers = [
      ...consts.map((slot) => this.#getBuffer(slot).buffer),
      ...initCarry.map((slot) => this.#getBuffer(slot).buffer),
      ...xs.map((slot) => this.#getBuffer(slot).buffer),
      ...carryOut.map((slot) => this.#getBuffer(slot).buffer),
      ...ysStacked.map((slot) => this.#getBuffer(slot).buffer),
    ];

    const commandEncoder = this.device.createCommandEncoder();
    for (const { pipeline, ...shader } of exe.data) {
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: allBuffers.map((buffer, i) => ({
          binding: i,
          resource: { buffer },
        })),
      });

      for (const { grid } of shader.passes) {
        if (prod(grid) === 0) continue;
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(grid[0], grid[1]);
        passEncoder.end();
      }
    }
    this.device.queue.submit([commandEncoder.finish()]);
  }

  // ---------------------------------------------------------------------------
  // Fused associative scan methods (WebGPU Kogge-Stone)
  // ---------------------------------------------------------------------------

  /**
   * Compile the fused Kogge-Stone shader and create a reusable pipeline.
   *
   * Returns `PreparedWebGPUAssocScan` containing the pipeline plus the
   * precomputed bind group layout. The pipeline is cached by shader source.
   */
  prepareAssocScan(
    params: WebGPUAssocScanParams,
  ): PreparedWebGPUAssocScan | null {
    try {
      const { code, workgroupSize } = assocScanFusedShaderSource(
        this.device,
        params,
      );

      if (DEBUG >= 2) {
        console.info(
          "=========== WebGPU assocScan shader ===========\n" + code,
        );
      }

      // Build bind group layout:
      //   group(0): ping (read), pong (read_write), const0..constK (read)
      //   group(1): uniforms (uniform, hasDynamicOffset=false)
      const { numConsts } = params;
      const storageEntries: GPUBindGroupLayoutEntry[] = [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ];
      for (let i = 0; i < numConsts; i++) {
        storageEntries.push({
          binding: i + 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        });
      }
      const storageLayout = this.device.createBindGroupLayout({
        entries: storageEntries,
      });
      const uniformLayout = this.device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
        ],
      });
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [storageLayout, uniformLayout],
      });

      const shaderModule = this.device.createShaderModule({ code });
      const pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: "main" },
      });

      return {
        pipeline,
        storageLayout,
        uniformLayout,
        workgroupSize,
        code,
      };
    } catch (e) {
      if (DEBUG >= 1) {
        console.warn("WebGPU assocScan codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Execute the full Kogge-Stone scan: ceil(log₂ N) dispatches of the
   * fused shader, swapping ping/pong buffers each round.
   *
   * @param prepared  Compiled pipeline from `prepareAssocScan()`
   * @param params    Scan parameters (leaf info, steps, etc.)
   * @param constSlots  Constant input slots (backend Slots)
   * @param elemSlots   Input leaf element slots (one per leaf)
   * @param outputSlots Output slots (one per leaf, caller-allocated)
   * @param N           Number of positions (scan length)
   * @param reverse     Whether to process in reverse order
   */
  dispatchAssocScan(
    prepared: PreparedWebGPUAssocScan,
    params: WebGPUAssocScanParams,
    constSlots: Slot[],
    elemSlots: Slot[],
    outputSlots: Slot[],
    N: number,
    reverse: boolean,
  ): void {
    const { numLeaves, leafElemCounts, dtype } = params;
    const bytesPerElem = byteWidth(dtype);

    // Compute total interleaved buffer size:
    // sum(leafElemCounts) * N * bytesPerElem
    const totalElemsPerPos = leafElemCounts.reduce((a, b) => a + b, 0);
    const totalBytes = totalElemsPerPos * N * bytesPerElem;
    const paddedBytes = Math.max(Math.ceil(totalBytes / 4) * 4, 4);

    // Allocate transient ping/pong GPU buffers
    const pingBuf = this.#createBuffer(paddedBytes);
    const pongBuf = this.#createBuffer(paddedBytes);

    // Compute leaf start offsets (prefix sum of elemCounts)
    const leafStarts: number[] = [0];
    for (let k = 1; k < numLeaves; k++) {
      leafStarts[k] = leafStarts[k - 1] + leafElemCounts[k - 1];
    }

    // Create const GPU buffers array
    const constBuffers = constSlots.map((slot) => this.#getBuffer(slot).buffer);

    // Copy input elems into ping buffer (interleaved layout)
    // For each leaf k, copy elemSlots[k] → ping at offset leafStarts[k]*N*bytesPerElem
    // Each elemSlot[k] has shape [N, ...leafShape], stored contiguously as
    // N * leafElemCounts[k] typed elements — this matches our interleaved layout
    // [leaf0: N*ec0 | leaf1: N*ec1 | ...]
    const commandEncoder = this.device.createCommandEncoder();
    for (let k = 0; k < numLeaves; k++) {
      const srcBuf = this.#getBuffer(elemSlots[k]).buffer;
      const dstOffset = leafStarts[k] * N * bytesPerElem;
      const copySize = leafElemCounts[k] * N * bytesPerElem;
      if (copySize > 0 && copySize % 4 === 0) {
        commandEncoder.copyBufferToBuffer(
          srcBuf,
          0,
          pingBuf,
          dstOffset,
          copySize,
        );
      } else if (copySize > 0) {
        // Unaligned copy — use the WGSL copy shader
        this.#encodeCopyWithShader(
          commandEncoder,
          srcBuf,
          0,
          pingBuf,
          dstOffset,
          copySize,
        );
      }
    }

    // If reverse, we need to reverse the input data along the scan axis.
    // Approach: reverse at copy-in by writing position j as (N-1-j) in the output,
    // but that would require a shader. Simpler: reverse the output at the end.
    // For now we reverse at copy-in/copy-out with dedicated reversal dispatches.
    // Actually, since the scan body fn is associative, we can just reverse
    // input, scan forward, reverse output. But that adds 2 extra dispatches.
    // Better approach: modify the shader to swap a_pos logic.
    // For simple implementation, handle reverse by reversing copy-in and copy-out.
    // TODO: For better perf, generate a reverse-aware shader variant.

    this.device.queue.submit([commandEncoder.finish()]);

    // If reverse, we need to flip the data in the ping buffer
    if (reverse && N > 1) {
      this.#reverseAssocScanBuffer(
        pingBuf,
        N,
        numLeaves,
        leafElemCounts,
        leafStarts,
        bytesPerElem,
      );
    }

    // Kogge-Stone: ceil(log₂ N) rounds
    const numRounds = Math.ceil(Math.log2(N));

    // Alternate which buffer is ping (read) and which is pong (write)
    let curPing = pingBuf;
    let curPong = pongBuf;

    for (let round = 0; round < numRounds; round++) {
      const stride = 1 << round;

      // Create uniform buffer with stride and N
      const uniformData = new Uint32Array([stride, N]);
      const minAlign = this.device.limits.minUniformBufferOffsetAlignment;
      const uniformSize = Math.max(
        Math.ceil(uniformData.byteLength / minAlign) * minAlign,
        uniformData.byteLength,
      );
      const uniformBuf = this.device.createBuffer({
        size: uniformSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(
        uniformBuf,
        0,
        uniformData.buffer,
        0,
        uniformData.byteLength,
      );

      // Create bind groups
      const storageEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: curPing } },
        { binding: 1, resource: { buffer: curPong } },
      ];
      for (let i = 0; i < constBuffers.length; i++) {
        storageEntries.push({
          binding: i + 2,
          resource: { buffer: constBuffers[i] },
        });
      }
      const storageBindGroup = this.device.createBindGroup({
        layout: prepared.storageLayout,
        entries: storageEntries,
      });
      const uniformBindGroup = this.device.createBindGroup({
        layout: prepared.uniformLayout,
        entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
      });

      // Dispatch
      const wgSize = prepared.workgroupSize;
      const numWorkgroups = Math.ceil(N / wgSize);
      const [gridX, gridY] = calculateGrid(numWorkgroups);

      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(prepared.pipeline);
      pass.setBindGroup(0, storageBindGroup);
      pass.setBindGroup(1, uniformBindGroup);
      pass.dispatchWorkgroups(gridX, gridY);
      pass.end();
      this.device.queue.submit([enc.finish()]);

      uniformBuf.destroy();

      // Swap ping/pong
      const tmp = curPing;
      curPing = curPong;
      curPong = tmp;
    }

    // After all rounds, result is in curPing (last write target became ping after swap)

    // If reverse, flip the result data back
    if (reverse && N > 1) {
      this.#reverseAssocScanBuffer(
        curPing,
        N,
        numLeaves,
        leafElemCounts,
        leafStarts,
        bytesPerElem,
      );
    }

    // Copy results from ping buffer (interleaved) to output slots
    const outEncoder = this.device.createCommandEncoder();
    for (let k = 0; k < numLeaves; k++) {
      const dstBuf = this.#getBuffer(outputSlots[k]).buffer;
      const srcOffset = leafStarts[k] * N * bytesPerElem;
      const copySize = leafElemCounts[k] * N * bytesPerElem;
      if (copySize > 0 && copySize % 4 === 0) {
        outEncoder.copyBufferToBuffer(curPing, srcOffset, dstBuf, 0, copySize);
      } else if (copySize > 0) {
        this.#encodeCopyWithShader(
          outEncoder,
          curPing,
          srcOffset,
          dstBuf,
          0,
          copySize,
        );
      }
    }
    this.device.queue.submit([outEncoder.finish()]);

    // Destroy transient buffers
    pingBuf.destroy();
    pongBuf.destroy();
    this.#gpuAllocatedBytes -= paddedBytes * 2;
  }

  /**
   * Reverse elements along the scan axis (position 0..N-1) in an interleaved
   * buffer. Uses a simple GPU copy shader: position j ↔ position (N-1-j).
   */
  #reverseAssocScanBuffer(
    buffer: GPUBuffer,
    N: number,
    numLeaves: number,
    leafElemCounts: number[],
    leafStarts: number[],
    bytesPerElem: number,
  ): void {
    // Create a temp buffer, copy reversed, copy back
    const totalElemsPerPos = leafElemCounts.reduce((a, b) => a + b, 0);
    const totalBytes = totalElemsPerPos * N * bytesPerElem;
    const paddedBytes = Math.max(Math.ceil(totalBytes / 4) * 4, 4);
    const tempBuf = this.#createBuffer(paddedBytes);

    const enc = this.device.createCommandEncoder();
    for (let k = 0; k < numLeaves; k++) {
      const ec = leafElemCounts[k];
      const leafOffset = leafStarts[k] * N * bytesPerElem;
      const elemBytes = ec * bytesPerElem;
      for (let j = 0; j < N; j++) {
        const srcOff = leafOffset + j * elemBytes;
        const dstOff = leafOffset + (N - 1 - j) * elemBytes;
        if (elemBytes % 4 === 0) {
          enc.copyBufferToBuffer(buffer, srcOff, tempBuf, dstOff, elemBytes);
        }
      }
    }
    // Copy temp back to buffer
    enc.copyBufferToBuffer(tempBuf, 0, buffer, 0, paddedBytes);
    this.device.queue.submit([enc.finish()]);
    tempBuf.destroy();
    this.#gpuAllocatedBytes -= paddedBytes;
  }

  // ---------------------------------------------------------------------------
  // Preencoded-routine scan methods (P4)
  // ---------------------------------------------------------------------------

  #getCopyPipeline(): GPUComputePipeline {
    if (this.#copyPipeline) return this.#copyPipeline;
    const shader: ShaderInfo = {
      code: COPY_SHADER_CODE,
      numInputs: 1,
      numOutputs: 1,
      hasUniform: true,
      passes: [{ grid: [1, 1], uniform: new Uint8Array(16) }],
    };
    this.#copyPipeline = this.pipelines.prepareSync(shader);
    return this.#copyPipeline;
  }

  #encodeCopyWithShader(
    commandEncoder: GPUCommandEncoder,
    srcBuf: GPUBuffer,
    srcOffset: number,
    dstBuf: GPUBuffer,
    dstOffset: number,
    size: number,
  ): GPUBuffer | null {
    if (size <= 0) return null;
    const firstDstWord = dstOffset >>> 2;
    const lastDstWord = (dstOffset + size + 3) >>> 2;
    const words = lastDstWord - firstDstWord;
    const workgroups = Math.ceil(words / COPY_WORKGROUP_SIZE);
    if (workgroups === 0) return null;

    const [gridX, gridY] = calculateGrid(workgroups);
    const pipeline = this.#getCopyPipeline();

    const storageBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuf } },
        { binding: 1, resource: { buffer: dstBuf } },
      ],
    });

    const uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    new Uint32Array(uniformBuffer.getMappedRange()).set([
      srcOffset,
      dstOffset,
      size,
      0,
    ]);
    uniformBuffer.unmap();

    const uniformBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, storageBindGroup);
    passEncoder.setBindGroup(1, uniformBindGroup);
    passEncoder.dispatchWorkgroups(gridX, gridY);
    passEncoder.end();

    return uniformBuffer;
  }

  /**
   * Returns the minimum uniform buffer offset alignment for preencoded scan.
   */
  getPreencodedScanAlignment(): number {
    return this.device.limits.minUniformBufferOffsetAlignment ?? 256;
  }

  /**
   * Prepare a preencoded scan operation for routine bodies (matmul, etc.).
   *
   * Wraps routine shaders with uniform-based offset bindings for xs/ys,
   * creates the combined offsets buffer, and compiles pipelines.
   * Returns null if the routine can't be preencoded.
   */
  preparePreencodedScan(
    params: PreencodedScanParams,
  ): PreparedPreencodedScan | null {
    const {
      xsElemStrides,
      ysElemStrides,
      bodyRoutine,
      numConsts,
      numCarry,
      numX,
      numY,
      length,
      reverse,
      carrySizes,
      routineInputJitIds,
      routineOutputJitIds,
    } = params;

    if (!bodyRoutine || bodyRoutine.data.length === 0) {
      if (DEBUG >= 2) console.log("Preencoded scan: invalid routine");
      return null;
    }

    // Skip if no xs/ys need offsets
    if (numX === 0 && numY === 0) {
      if (DEBUG >= 2) console.log("Preencoded scan: no xs/ys, skipping");
      return null;
    }

    const scanInfo: ScanBindingInfo = {
      numConsts,
      numCarry,
      routineInputJitIds,
      routineOutputJitIds,
    };

    // Wrap each shader with scan offset support
    const wrappedShaders: ShaderDispatch[] = [];

    // Shared layout for group(1) with dynamic offset support — identical
    // for all shaders (single uniform binding for scan offsets).
    const uniformLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true },
        },
      ],
    });

    for (const shader of bodyRoutine.data) {
      // Shaders that already use uniforms (like Sort) conflict with our offset uniform
      if (shader.hasUniform) {
        if (DEBUG >= 2)
          console.log("Preencoded scan: shader already has uniform, skipping");
        return null;
      }

      const wrapped = wrapRoutineForScan(shader, scanInfo);
      if (!wrapped.hasUniform) {
        if (DEBUG >= 2)
          console.log("Preencoded scan: shader doesn't need offsets");
        return null;
      }

      const module = this.device.createShaderModule({ code: wrapped.code });

      // Create auto-layout pipeline to extract group(0)'s layout, then
      // rebuild with explicit group(1) that has hasDynamicOffset: true.
      const autoPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const group0Layout = autoPipeline.getBindGroupLayout(0);
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [group0Layout, uniformLayout],
      });
      const pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint: "main" },
      });

      wrappedShaders.push({
        ...shader,
        code: wrapped.code,
        hasUniform: true,
        pipeline,
      });
    }

    // Create combined uniform buffer with offsets for all iterations
    const alignment = this.getPreencodedScanAlignment();
    const { buffer: offsetData, alignment: offsetAlignment } =
      createAllIterationsOffsetsBuffer(
        numX,
        numY,
        length,
        xsElemStrides,
        ysElemStrides,
        alignment,
        reverse,
      );

    const offsetBuffer = this.device.createBuffer({
      size: offsetData.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(offsetBuffer.getMappedRange()).set(offsetData);
    offsetBuffer.unmap();

    if (DEBUG >= 1) {
      console.log(
        `Preencoded scan: prepared for ${length} iterations with uniform offsets`,
      );
    }

    const copyUsesShader = carrySizes.map((size) => size % 4 !== 0);

    return {
      params,
      wrappedShaders,
      offsetBuffer,
      offsetAlignment,
      copyUsesShader,
      uniformLayout,
    };
  }

  /**
   * Dispatch a preencoded scan with routine body.
   *
   * Uses ping-pong buffers for carry and uniform-based offsets for xs/ys.
   * All iteration dispatches are encoded in a single command buffer.
   */
  dispatchPreencodedScan(
    prepared: PreparedPreencodedScan,
    constSlots: Slot[],
    initCarrySlots: Slot[],
    xsSlots: Slot[],
    carryOutSlots: Slot[],
    ysStackedSlots: Slot[],
  ): void {
    const {
      params,
      wrappedShaders,
      offsetBuffer,
      offsetAlignment,
      copyUsesShader,
      uniformLayout,
    } = prepared;
    const {
      length,
      carrySizes,
      numCarry,
      numConsts,
      routineInputJitIds,
      routineOutputJitIds,
    } = params;

    const constBuffers = constSlots.map((slot) => this.#getBuffer(slot).buffer);
    const initCarryBuffers = initCarrySlots.map(
      (slot) => this.#getBuffer(slot).buffer,
    );
    const xsBuffers = xsSlots.map((slot) => this.#getBuffer(slot).buffer);
    const carryOutBuffers = carryOutSlots.map(
      (slot) => this.#getBuffer(slot).buffer,
    );
    const ysStackedBuffers = ysStackedSlots.map(
      (slot) => this.#getBuffer(slot).buffer,
    );

    // Create ping-pong buffers for carry state
    const carryPing = carrySizes.map((size) =>
      this.#createBuffer(Math.max(size, 4)),
    );
    const carryPong = carrySizes.map((size) =>
      this.#createBuffer(Math.max(size, 4)),
    );

    const commandEncoder = this.device.createCommandEncoder();
    const copyUniformBuffers: GPUBuffer[] = [];

    // Copy initCarry to carryPing
    for (let i = 0; i < numCarry; i++) {
      if (carrySizes[i] > 0) {
        commandEncoder.copyBufferToBuffer(
          initCarryBuffers[i],
          0,
          carryPing[i],
          0,
          carrySizes[i],
        );
      }
    }

    const xsStart = numConsts + numCarry;

    for (const shader of wrappedShaders) {
      const { pipeline, passes } = shader;

      // Helper to create storage bind group for a ping-pong configuration
      const createStorageBindGroup = (
        readCarry: GPUBuffer[],
        writeCarry: GPUBuffer[],
      ): GPUBindGroup => {
        const entries: GPUBindGroupEntry[] = [];
        let binding = 0;

        // Input bindings: classify by scan buffer role
        for (let i = 0; i < routineInputJitIds.length; i++) {
          const jitId = routineInputJitIds[i];
          let buffer: GPUBuffer;

          if (jitId < numConsts) {
            buffer = constBuffers[jitId];
          } else if (jitId < xsStart) {
            const carryIdx = jitId - numConsts;
            buffer = readCarry[carryIdx];
          } else {
            const xIdx = jitId - xsStart;
            buffer = xsBuffers[xIdx];
          }

          entries.push({ binding: binding++, resource: { buffer } });
        }

        // Output bindings: in passthrough pattern, all go to carry (write)
        for (let i = 0; i < routineOutputJitIds.length; i++) {
          const buffer = writeCarry[i];
          if (!buffer) {
            throw new Error(
              `Preencoded scan: routine output ${i} has no carry buffer ` +
                `(writeCarry.length=${writeCarry.length})`,
            );
          }
          entries.push({ binding: binding++, resource: { buffer } });
        }

        return this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });
      };

      const pingBindGroup = createStorageBindGroup(carryPing, carryPong);
      const pongBindGroup = createStorageBindGroup(carryPong, carryPing);

      // Create single uniform bind group with dynamic offset support
      const uniformBindGroup = this.device.createBindGroup({
        layout: uniformLayout,
        entries: [
          {
            binding: 0,
            resource: {
              buffer: offsetBuffer,
              offset: 0,
              size: offsetAlignment,
            },
          },
        ],
      });

      const filteredPasses = passes.filter(({ grid }) => prod(grid) > 0);

      for (let iter = 0; iter < length; iter++) {
        const storageBindGroup = iter % 2 === 0 ? pingBindGroup : pongBindGroup;

        for (const { grid } of filteredPasses) {
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(pipeline);
          passEncoder.setBindGroup(0, storageBindGroup);
          passEncoder.setBindGroup(1, uniformBindGroup, [
            iter * offsetAlignment,
          ]);
          passEncoder.dispatchWorkgroups(grid[0], grid[1]);
          passEncoder.end();
        }

        // Copy carry → ys for this iteration (passthrough pattern)
        const currentCarryBuffers = iter % 2 === 0 ? carryPong : carryPing;
        for (let c = 0; c < numCarry; c++) {
          const copySize = carrySizes[c];
          if (copySize <= 0) continue;
          const yOffset = iter * copySize;
          if (!copyUsesShader[c]) {
            commandEncoder.copyBufferToBuffer(
              currentCarryBuffers[c],
              0,
              ysStackedBuffers[c],
              yOffset,
              copySize,
            );
          } else {
            const uniformBuf = this.#encodeCopyWithShader(
              commandEncoder,
              currentCarryBuffers[c],
              0,
              ysStackedBuffers[c],
              yOffset,
              copySize,
            );
            if (uniformBuf) copyUniformBuffers.push(uniformBuf);
          }
        }
      }
    }

    // Copy final carry to carryOut
    const finalCarry = length % 2 === 0 ? carryPing : carryPong;
    for (let i = 0; i < numCarry; i++) {
      const copySize = carrySizes[i];
      if (copySize <= 0) continue;
      if (!copyUsesShader[i]) {
        commandEncoder.copyBufferToBuffer(
          finalCarry[i],
          0,
          carryOutBuffers[i],
          0,
          copySize,
        );
      } else {
        const uniformBuf = this.#encodeCopyWithShader(
          commandEncoder,
          finalCarry[i],
          0,
          carryOutBuffers[i],
          0,
          copySize,
        );
        if (uniformBuf) copyUniformBuffers.push(uniformBuf);
      }
    }

    this.device.queue.submit([commandEncoder.finish()]);

    // Clean up temporary buffers
    for (const buf of copyUniformBuffers) buf.destroy();
    for (const buf of [...carryPing, ...carryPong]) {
      this.#gpuAllocatedBytes -= buf.size;
      buf.destroy();
    }
    // offsetBuffer is NOT destroyed — owned by PreparedPreencodedScan for reuse
  }

  #getBuffer(slot: Slot): { buffer: GPUBuffer; size: number } {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return { buffer: buffer.buffer, size: buffer.size };
  }

  /**
   * Create a GPU buffer.
   *
   * By default, this creates a general-purpose buffer with the given size.
   *
   * - If `mapped` is true, initialize the buffer in mapped mode so that it can
   *   be populated with data from the CPU. (Call `.unmap()` later.)
   * - If `read` is true, create a staging buffer for returning data to CPU.
   *   (Call `.mapAsync()` later.)
   */
  /** Pop a buffer of the given padded byte size from the pool, or null. */
  #poolPop(paddedSize: number): GPUBuffer | null {
    const list = this.#bufferPool.get(paddedSize);
    if (list && list.length > 0) {
      const buf = list.pop()!;
      this.#poolCurrentBytes -= buf.size;
      return buf;
    }
    return null;
  }

  /** Push a freed buffer into the pool. Returns false if pool is full. */
  #poolPush(buffer: GPUBuffer): boolean {
    const paddedSize = buffer.size;
    // Per-size-class cap.
    let list = this.#bufferPool.get(paddedSize);
    if (!list) {
      list = [];
      this.#bufferPool.set(paddedSize, list);
    }
    if (list.length >= WebGPUBackend.MAX_POOL_PER_SIZE) return false;
    // Byte budget cap.
    if (this.#poolCurrentBytes + paddedSize > this.#poolBudgetBytes) {
      return false;
    }
    list.push(buffer);
    this.#poolCurrentBytes += paddedSize;
    return true;
  }

  /**
   * Prepare the pool for the next JitProgram execution:
   * 1. Evict entries whose sizes aren't needed (stale cross-program buffers).
   * 2. Set the byte budget to the program's peak live bytes.
   *
   * This guarantees physical peak memory ≤ peak live memory: the pool only
   * retains buffers that will be reused, and total retained bytes can't exceed
   * what the program already needs at its peak point.
   */
  configurePool(hints: {
    readonly peakBytes: number;
    readonly mallocSizes: ReadonlySet<number>;
  }): void {
    // Evict pool entries whose sizes won't be needed.
    for (const [size, list] of this.#bufferPool) {
      if (!hints.mallocSizes.has(size)) {
        for (const buf of list) {
          this.#poolCurrentBytes -= buf.size;
          this.#gpuAllocatedBytes -= buf.size;
          buf.destroy();
        }
        this.#bufferPool.delete(size);
      }
    }
    // Update byte budget to this program's peak.
    this.#poolBudgetBytes = hints.peakBytes;
    // If current pool exceeds new budget, evict excess (LIFO per size).
    while (this.#poolCurrentBytes > this.#poolBudgetBytes) {
      let evicted = false;
      for (const [_size, list] of this.#bufferPool) {
        if (list.length > 0) {
          const buf = list.pop()!;
          this.#poolCurrentBytes -= buf.size;
          this.#gpuAllocatedBytes -= buf.size;
          buf.destroy();
          evicted = true;
          break;
        }
      }
      if (!evicted) break;
    }
  }

  // ---------------------------------------------------------------------------
  // Batch dispatch: encode multiple dispatches into one command submission
  // ---------------------------------------------------------------------------

  beginBatch(): void {
    if (this.#batchDepth === 0) {
      this.#batchEncoder = this.device.createCommandEncoder();
      this.#batchUniformsToDestroy = [];
      this.#batchDeferredFreeBuffers = [];
    }
    this.#batchDepth++;
  }

  endBatch(): void {
    if (this.#batchDepth === 0) return;
    this.#batchDepth--;
    if (this.#batchDepth === 0) {
      const encoder = this.#batchEncoder;
      if (!encoder) return;
      this.#batchEncoder = null;
      this.device.queue.submit([encoder.finish()]);
      // Destroy uniform buffers created during batch encoding
      for (const buf of this.#batchUniformsToDestroy) buf.destroy();
      this.#batchUniformsToDestroy = [];
      // Process deferred buffer frees (delayed to avoid destroying buffers
      // that are referenced by commands in the batch encoder)
      for (const gpuBuf of this.#batchDeferredFreeBuffers) {
        if (!this.#poolPush(gpuBuf)) {
          this.#gpuAllocatedBytes -= gpuBuf.size;
          gpuBuf.destroy();
        }
      }
      this.#batchDeferredFreeBuffers = [];
    }
  }

  // ---------------------------------------------------------------------------
  // ScatterAdd dispatch (M2: scatter_add primitive)
  // ---------------------------------------------------------------------------
  #scatterAddPipelineCache = new Map<string, GPUComputePipeline>();

  dispatchScatterAdd(
    output: Slot,
    indices: Slot,
    updates: Slot,
    axis: number,
    targetShape: number[],
    updatesLen: number,
    dtype: DType,
  ): void {
    if (dtype === DType.Float64) {
      throw new Error("ScatterAdd: Float64 not supported on WebGPU");
    }

    // Compute strides for the scatter axis
    const ndim = targetShape.length;
    const innerSize =
      ndim > 0 ? targetShape.slice(axis + 1).reduce((a, b) => a * b, 1) : 1;
    const outerSize =
      ndim > 0 ? targetShape.slice(0, axis).reduce((a, b) => a * b, 1) : 1;
    const axisSize = ndim > 0 ? targetShape[axis] : 1;

    // Total number of update elements
    const totalUpdates = updatesLen * outerSize * innerSize;
    const [gridX, gridY] = calculateGrid(Math.ceil(totalUpdates / 64));

    // Use native atomicAdd for f32 when the device supports shader-f32-atomic-add
    const useNativeF32Atomic =
      dtype === DType.Float32 && this.capabilities.atomicF32Add;

    const cacheKey = `scatter_add_${dtype}_${ndim}_${axis}_${outerSize}_${innerSize}_${axisSize}_${updatesLen}_${useNativeF32Atomic ? "native" : "cas"}`;
    let pipeline = this.#scatterAddPipelineCache.get(cacheKey);

    if (!pipeline) {
      const wgslType = dtypeToWgsl(dtype);
      const isFloat = dtype === DType.Float32 || dtype === DType.Float16;
      const atomicType = useNativeF32Atomic
        ? "f32"
        : isFloat
          ? "u32"
          : wgslType === "i32"
            ? "i32"
            : "u32";

      // Build shader
      let code = useNativeF32Atomic ? "enable shader_f32_atomic_add;\n" : "";
      code += `
@group(0) @binding(0) var<storage, read_write> output: array<atomic<${atomicType}>>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> updates: array<${wgslType}>;

const INNER: u32 = ${innerSize}u;
const OUTER: u32 = ${outerSize}u;
const AXIS_SIZE: u32 = ${axisSize}u;
const UPDATES_LEN: u32 = ${updatesLen}u;
const TOTAL: u32 = ${totalUpdates}u;
const TARGET_INNER_STRIDE: u32 = ${axisSize * innerSize}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let flat = gid.x + gid.y * ${gridX}u;
  if (flat >= TOTAL) { return; }

  // Decompose flat index into (outer, updateIdx, inner)
  let inner = flat % INNER;
  let tmp = flat / INNER;
  let updateIdx = tmp % UPDATES_LEN;
  let outer = tmp / UPDATES_LEN;

  // Look up target axis index
  let targetAxisIdx = u32(indices[updateIdx]);
  if (targetAxisIdx >= AXIS_SIZE) { return; }

  // Compute flat output index
  let outFlat = outer * TARGET_INNER_STRIDE + targetAxisIdx * INNER + inner;

  let val = updates[flat];
`;

      if (useNativeF32Atomic) {
        // Native f32 atomicAdd via shader-f32-atomic-add extension
        code += `
  atomicAdd(&output[outFlat], val);
`;
      } else if (isFloat) {
        // CAS loop for f32/f16 atomics (bitcast through u32)
        code += `
  // CAS loop: atomically add via bitcast<u32>
  var old_bits = atomicLoad(&output[outFlat]);
  loop {
    let old_val = bitcast<${wgslType}>(old_bits);
    let new_val = old_val + val;
    let new_bits = bitcast<u32>(new_val);
    let result = atomicCompareExchangeWeak(&output[outFlat], old_bits, new_bits);
    if (result.exchanged) { break; }
    old_bits = result.old_value;
  }
`;
      } else {
        // Native atomicAdd for integer types
        code += `
  atomicAdd(&output[outFlat], ${atomicType === "u32" ? "bitcast<u32>(val)" : "val"});
`;
      }

      code += `}\n`;

      const shaderModule = this.device.createShaderModule({ code });
      const layout = this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.device.createBindGroupLayout({
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
              },
              {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
              },
            ],
          }),
        ],
      });
      pipeline = this.device.createComputePipeline({
        layout,
        compute: { module: shaderModule, entryPoint: "main" },
      });
      this.#scatterAddPipelineCache.set(cacheKey, pipeline);
    }

    const outBuf = this.#getBuffer(output).buffer;
    const idxBuf = this.#getBuffer(indices).buffer;
    const updBuf = this.#getBuffer(updates).buffer;

    const commandEncoder = this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outBuf } },
        { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: updBuf } },
      ],
    });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(gridX, gridY);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  #createBuffer(
    size: number,
    { mapped = false, read = false } = {},
  ): GPUBuffer {
    if (read && mapped) {
      throw new Error("mapped and read cannot both be true");
    }
    const buffer = this.device.createBuffer({
      size,
      usage: read
        ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      mappedAtCreation: mapped,
    });
    if (!read) {
      this.#gpuAllocatedBytes += size;
      if (this.#gpuAllocatedBytes > this.#gpuPeakBytes) {
        this.#gpuPeakBytes = this.#gpuAllocatedBytes;
      }
    }
    return buffer;
  }
}

/**
 * Compiles a multi-output kernel into a single WebGPU shader.
 * Multi-output kernels are always non-reduction (enforced by jitCompile batching).
 * Generates one result binding per output and one store per output per gidx.
 */
function pipelineSourceMulti(
  device: GPUDevice,
  kernel: Kernel,
  caps?: BackendCapabilities,
): ShaderInfo {
  const { nargs } = kernel;
  const numOutputs = kernel.numOutputs;
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  // Tune each output individually
  const tunes = kernel.outputs.map((o) => {
    const tmp = Kernel.single(nargs, kernel.size, o.exp, o.reduction);
    return tuneWebgpu(tmp, caps);
  });

  // All outputs share the same threadCount (same size, no reductions)
  const threadCount = tunes[0].threadCount;

  const shader: string[] = [];
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines: (string | symbol)[]) => {
    for (const line of lines) {
      if (line === pushIndent) indent += "  ";
      else if (line === popIndent) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : line);
    }
  };

  // Check for f16 across all outputs
  if (tunes.some((t) => t.exp.some((exp) => exp.dtype === DType.Float16))) {
    if (!device.features.has("shader-f16"))
      throw new Error("WebGPU device does not support shader-f16 feature");
    emit("enable f16;");
  }

  emit(headerWgsl);

  // Collect all distinct ops across all output expressions
  let allOps: Map<AluOp, Set<DType>> = new Map();
  for (const tune of tunes) {
    allOps = mapSetUnion(allOps, tune.exp.distinctOps());
  }
  if (allOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allOps.has(AluOp.Erf) || allOps.has(AluOp.Erfc)) emit(erfSrc);
  emit("");

  // Find used args across all outputs
  const usedArgs: (DType | null)[] = Array.from({ length: nargs }, () => null);
  for (const tune of tunes) {
    tune.exp.fold((exp) => {
      if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
    });
  }

  // Input bindings
  for (let i = 0; i < nargs; i++) {
    const ty = dtypeToWgsl(usedArgs[i] ?? DType.Float32, true);
    emit(
      `@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`,
    );
  }

  // Output bindings (one per output)
  for (let oi = 0; oi < numOutputs; oi++) {
    const resultTy = dtypeToWgsl(kernel.outputs[oi].dtype, true);
    emit(
      `@group(0) @binding(${nargs + oi}) var<storage, read_write> result${oi} : array<${resultTy}>;`,
    );
  }

  const symbolic = isSymbolicSize(threadCount);

  let workgroupSize: number;
  let gridX: number;
  let gridY: number;

  if (symbolic) {
    workgroupSize = 256;
    gridX = 1;
    gridY = 1;
    emit(
      "",
      "struct Dims { total_size: u32 }",
      "@group(1) @binding(0) var<uniform> dims: Dims;",
    );
  } else {
    const tc = threadCount as number;
    workgroupSize = findPow2(tc, 256);
    const gridSize = Math.ceil(tc / workgroupSize);
    [gridX, gridY] = calculateGrid(gridSize);
  }

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
    pushIndent,
  );

  if (symbolic) {
    const sizeX = gridOffsetY * workgroupSize;
    emit(
      `let linear_idx: u32 = ${sizeX}u * id.y + id.x;`,
      "if (linear_idx >= dims.total_size) { return; }",
      "let gidx: i32 = i32(linear_idx);",
    );
  } else {
    const tc = threadCount as number;
    if (gridY === 1) {
      emit(`if (id.x >= ${tc}) { return; }`, "let gidx: i32 = i32(id.x);");
    } else {
      const sizeX = gridX * workgroupSize;
      emit(
        `if (${sizeX} * id.y + id.x >= ${tc}) { return; }`,
        `let gidx: i32 = i32(${sizeX} * id.y + id.x);`,
      );
    }
  }

  // CSE infrastructure (shared across all outputs for subexpression sharing)
  let gensymCount = 0;
  const gensym = () => `alu${gensymCount++}`;
  const isGensym = (text: string) => text.match(/^alu[0-9]+$/);

  // Phony assignments for unused args
  if (args.length > 0) {
    emit(args.map((arg) => `_ = &${arg};`).join(" "));
  }

  // Count references across ALL output expressions for correct CSE
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };
  for (const tune of tunes) {
    countReferences(tune.exp);
  }

  // AluExp → WGSL translator with CSE
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
      const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
      emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
      if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
      else if (arg === 0) source = `${x}.x`;
      else if (arg === 1) source = `${x}.y`;
      else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return arg as string;
    } else if (op === AluOp.GlobalIndex) {
      source = `${args[arg[0]]}[${strip1(gen(src[0]))}]`;
      if (dtype === DType.Bool) source = `(${source} != 0)`;
    }

    if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
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

  // Generate stores for each output
  for (let oi = 0; oi < numOutputs; oi++) {
    const resultTy = dtypeToWgsl(kernel.outputs[oi].dtype, true);
    let rhs = strip1(gen(tunes[oi].exp));
    if (resultTy !== dtypeToWgsl(tunes[oi].exp.dtype))
      rhs = `${resultTy}(${rhs})`;
    emit(`result${oi}[gidx] = ${rhs};`);
  }

  emit(popIndent, "}");
  return {
    code: shader.join("\n"),
    numInputs: nargs,
    numOutputs: numOutputs,
    hasUniform: symbolic,
    passes: [{ grid: [gridX, gridY] }],
    isSymbolic: symbolic || undefined,
    workgroupSize: symbolic ? workgroupSize : undefined,
  };
}

/**
 * Compiles an expression into WebGPU shader source code.
 *
 * Returns the shader source and the number of workgroups to dispatch along x
 * and y axes, to run the kernel.
 */
function pipelineSource(
  device: GPUDevice,
  kernel: Kernel,
  caps?: BackendCapabilities,
): ShaderInfo {
  if (kernel.isMultiOutput) {
    return pipelineSourceMulti(device, kernel, caps);
  }

  const tune = tuneWebgpu(kernel, caps);
  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.outputs[0].exp}\ntune.exp: ${tune.exp}`);
  }

  const { nargs } = kernel;
  const re = kernel.outputs[0].reduction;
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  // binding(0..n-1): input buffers
  // binding(n): output buffer

  const shader: string[] = []; // line-separated
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines: (string | symbol)[]) => {
    for (const line of lines) {
      if (line === pushIndent) indent += "  ";
      else if (line === popIndent) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : line);
    }
  };

  if (
    tune.exp.some((exp) => exp.dtype === DType.Float16) ||
    tune.epilogue?.some((exp) => exp.dtype === DType.Float16)
  ) {
    if (!device.features.has("shader-f16"))
      throw new Error("WebGPU device does not support shader-f16 feature");
    emit("enable f16;");
  }

  emit(headerWgsl);

  // Global functions at the start of the shader.
  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  if (distinctOps.has(AluOp.Threefry2x32)) {
    emit(threefrySrc);
  }
  if (distinctOps.has(AluOp.Erf) || distinctOps.has(AluOp.Erfc)) {
    emit(erfSrc);
  }

  // End global function definitions.
  emit("");

  const usedArgs: (DType | null)[] = Array.from({ length: nargs }, () => null);
  tune.exp.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });
  tune.epilogue?.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });

  for (let i = 0; i < nargs; i++) {
    // If not used, just assume float32, all that matters is size / alignment.
    const ty = dtypeToWgsl(usedArgs[i] ?? DType.Float32, true);
    emit(
      `@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`,
    );
  }

  const resultTy = dtypeToWgsl(kernel.outputs[0].dtype, true);
  emit(
    `@group(0) @binding(${nargs}) var<storage, read_write> result : array<${resultTy}>;`,
  );

  const symbolic = isSymbolicSize(tune.threadCount);
  const symbolicReduce = isSymbolicSize(tune.size.reduce);

  // For symbolic kernels, use a uniform buffer to pass the total element count
  // at dispatch time. For concrete kernels, bake the count into the shader.
  // When the reduction axis is symbolic, also pass reduce_size via uniform.
  let workgroupSize: number;
  let gridX: number;
  let gridY: number;

  if (symbolic || symbolicReduce) {
    workgroupSize = symbolic ? 256 : findPow2(tune.threadCount as number, 256);
    if (symbolic) {
      // Grid resolved at dispatch time from dynamicParams.
      gridX = 1;
      gridY = 1;
    } else {
      const tc = tune.threadCount as number;
      const gridSize = Math.ceil(tc / workgroupSize);
      [gridX, gridY] = calculateGrid(gridSize);
    }
    // Build struct Dims with the fields we need
    const dimsFields: string[] = [];
    if (symbolic) dimsFields.push("total_size: u32");
    if (symbolicReduce) dimsFields.push("reduce_size: u32");
    emit(
      "",
      `struct Dims { ${dimsFields.join(", ")} }`,
      "@group(1) @binding(0) var<uniform> dims: Dims;",
    );
  } else {
    const threadCount = tune.threadCount as number;
    const tuneLocal = tune.size.local ?? 1;
    const tuneGroups = (tune as WebGPUTuneResult).size.groups ?? 1;
    if (tuneGroups > 1) {
      workgroupSize = tuneGroups;
    } else if (tuneLocal > 1) {
      workgroupSize = tuneLocal;
    } else {
      workgroupSize = findPow2(threadCount, 256);
    }
    const gridSize = Math.ceil(threadCount / workgroupSize);
    [gridX, gridY] = calculateGrid(gridSize);
  }

  // Shared memory for cooperative group reductions.
  const groupSize = re ? ((tune as WebGPUTuneResult).size.groups ?? 1) : 1;
  const useSharedMem = groupSize > 1;
  if (useSharedMem) {
    const upcast = (tune as WebGPUTuneResult).size.upcast ?? 1;
    const shmemTy = dtypeToWgsl(re!.dtype);
    emit(`var<workgroup> shmem: array<${shmemTy}, ${groupSize * upcast}>;`);
  }

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
    pushIndent,
  );

  if (symbolic) {
    // For symbolic: always use 2D formula (works for 1D when id.y=0).
    const sizeX = gridOffsetY * workgroupSize;
    emit(
      `let linear_idx: u32 = ${sizeX}u * id.y + id.x;`,
      "if (linear_idx >= dims.total_size) { return; }",
      "let gidx: i32 = i32(linear_idx);",
    );
  } else {
    const threadCount = tune.threadCount as number;
    if (gridY === 1) {
      emit(`if (id.x >= ${threadCount}u) { return; }`);
      if (useSharedMem) {
        emit(
          `let gidx: i32 = i32(id.x / ${groupSize}u);`,
          `let group: i32 = i32(id.x % ${groupSize}u);`,
        );
      } else {
        emit("let gidx: i32 = i32(id.x);");
      }
    } else {
      const sizeX = gridX * workgroupSize;
      if (useSharedMem) {
        emit(
          `let _tid: u32 = ${sizeX}u * id.y + id.x;`,
          `if (_tid >= ${threadCount}u) { return; }`,
          `let gidx: i32 = i32(_tid / ${groupSize}u);`,
          `let group: i32 = i32(_tid % ${groupSize}u);`,
        );
      } else {
        emit(
          `if (${sizeX} * id.y + id.x >= ${threadCount}) { return; }`,
          `let gidx: i32 = i32(${sizeX} * id.y + id.x);`,
        );
      }
    }
  }

  // Generate code for each AluExp operation.
  // Some expressions may be used twice, so we keep track of them.
  let gensymCount = 0;
  const gensym = () => `alu${gensymCount++}`;
  const isGensym = (text: string) => text.match(/^alu[0-9]+$/);

  // Insert phony assignments, in case some inputs are not in use.
  // https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686
  if (args.length > 0) {
    emit(args.map((arg) => `_ = &${arg};`).join(" "));
  }

  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, string>();
  const gen = (exp: AluExp): string => {
    if (expContext.has(exp)) return expContext.get(exp)!;
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them.
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
        // Edge case: WebGPU doesn't handle NaN correctly, it's unspecified.
        // Use bitcast to check the IEEE 754 bit pattern instead:
        // f32 NaN: exponent all-1s (0x7F800000) AND mantissa non-zero.
        // f16 NaN: exponent all-1s (0x7C00) AND mantissa non-zero.
        if (isFloatDtype(src[0].dtype)) {
          const x = isGensym(a) ? a : gensym();
          if (x !== a) emit(`let ${x} = ${a};`);
          const y = isGensym(b) ? b : gensym();
          if (y !== b) emit(`let ${y} = ${b};`);
          if (src[0].dtype === DType.Float16) {
            const bitsX = gensym();
            const bitsY = gensym();
            emit(
              `let ${bitsX} = bitcast<u32>(vec2<f16>(${x}, f16(0.0))) & 0xFFFFu;`,
            );
            emit(
              `let ${bitsY} = bitcast<u32>(vec2<f16>(${y}, f16(0.0))) & 0xFFFFu;`,
            );
            const isNanX = `((${bitsX} & 0x7C00u) == 0x7C00u && (${bitsX} & 0x03FFu) != 0u)`;
            const isNanY = `((${bitsY} & 0x7C00u) == 0x7C00u && (${bitsY} & 0x03FFu) != 0u)`;
            source = `(${x} != ${y} || ${isNanX} || ${isNanY})`;
          } else {
            const bitsX = gensym();
            const bitsY = gensym();
            emit(`let ${bitsX} = bitcast<u32>(${x});`);
            emit(`let ${bitsY} = bitcast<u32>(${y});`);
            const isNanX = `((${bitsX} & 0x7F800000u) == 0x7F800000u && (${bitsX} & 0x007FFFFFu) != 0u)`;
            const isNanY = `((${bitsY} & 0x7F800000u) == 0x7F800000u && (${bitsY} & 0x007FFFFFu) != 0u)`;
            source = `(${x} != ${y} || ${isNanX} || ${isNanY})`;
          }
        } else {
          source = `(${a} != ${b})`;
        }
      }
    } else if (AluGroup.Unary.has(op)) {
      if (op === AluOp.Reciprocal && src[0].op === AluOp.Sqrt) {
        // Special case: 1/sqrt(x) is optimized as rsqrt(x)
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
            // Always compute special functions in f32 for precision.
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
      // select(f, t, cond) -> cond ? t : f
      source = `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
    } else if (op === AluOp.Threefry2x32) {
      const x = gensym(); // temporary to hold the `vec2<u32>(x0, x1)`
      const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
      emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
      if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
      else if (arg === 0) source = `${x}.x`;
      else if (arg === 1) source = `${x}.y`;
      else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return arg as string;
    } else if (op === AluOp.GlobalIndex) {
      source = `${args[arg[0]]}[${strip1(gen(src[0]))}]`;
      if (dtype === DType.Bool) source = `(${source} != 0)`; // bool is represented as i32
    }

    if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
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

  if (!re) {
    countReferences(tune.exp);
    let rhs = strip1(gen(tune.exp));
    if (resultTy !== dtypeToWgsl(tune.exp.dtype)) rhs = `${resultTy}(${rhs})`;
    emit(`result[gidx] = ${rhs};`);
  } else {
    const unroll = tune.size.unroll ?? 1;
    const upcast = tune.size.upcast ?? 1;

    const acc = [...Array(upcast)].map((_, i) => `acc${i}`);
    for (let i = 0; i < upcast; i++) {
      emit(
        `var ${acc[i]}: ${dtypeToWgsl(re.dtype)} = ${constToWgsl(re.dtype, re.identity)};`,
      ); // Initialize accumulators.
    }

    const reduceBound = symbolicReduce
      ? "i32(dims.reduce_size)"
      : `${tune.size.reduce}`;
    emit(
      `for (var ridx: i32 = 0; ridx < ${reduceBound}; ridx++) {`,
      pushIndent,
    );

    // Now generate (shared) expressions for each accumulator and unroll value.
    const exps: AluExp[][] = [];
    const cache = new Map<bigint, AluExp>();
    for (let up = 0; up < upcast; up++) {
      exps.push([]);
      for (let un = 0; un < unroll; un++) {
        const exp = tune.exp.substitute({
          upcast: AluExp.i32(up),
          unroll: AluExp.i32(un),
        });
        exps[up].push(exp.simplify(cache));
        countReferences(exps[up][un]);
      }
    }

    // After references are counted, we can generate the code.
    const items = exps.map((ar) => ar.map(gen).map(strip1));
    for (let i = 0; i < upcast; i++) {
      let rhs = items[i][0];
      for (let j = 1; j < unroll; j++) {
        if (re.op === AluOp.Add) rhs = `${rhs} + ${items[i][j]}`;
        else if (re.op === AluOp.Mul) rhs = `${rhs} * ${items[i][j]}`;
        else if (re.op === AluOp.Min) {
          // For booleans, min is AND; for numerics, use min()
          rhs =
            re.dtype === DType.Bool
              ? `(${rhs} && ${items[i][j]})`
              : `min(${rhs}, ${items[i][j]})`;
        } else if (re.op === AluOp.Max) {
          // For booleans, max is OR; for numerics, use max()
          rhs =
            re.dtype === DType.Bool
              ? `(${rhs} || ${items[i][j]})`
              : `max(${rhs}, ${items[i][j]})`;
        } else throw new Error(`Unsupported reduction op: ${re.op}`);
      }
      if (re.op === AluOp.Add) emit(`${acc[i]} += ${rhs};`);
      else if (re.op === AluOp.Mul) emit(`${acc[i]} *= ${rhs};`);
      else if (re.op === AluOp.Min) {
        // For booleans, min is AND; for numerics, use min()
        if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} && ${rhs};`);
        else emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
      } else if (re.op === AluOp.Max) {
        // For booleans, max is OR; for numerics, use max()
        if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} || ${rhs};`);
        else emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
      } else throw new Error(`Unsupported reduction op: ${re.op}`);
    }
    emit(popIndent, "}");

    // Shared-memory tree reduction: each thread wrote its partial into acc[i],
    // now combine across workgroup threads via shmem.
    if (useSharedMem) {
      // Store each thread's partial into shared memory.
      for (let i = 0; i < upcast; i++) {
        emit(`shmem[${i * groupSize} + group] = ${acc[i]};`);
      }
      emit("workgroupBarrier();");

      // Tree reduction with halving stride.
      for (let stride = groupSize >> 1; stride >= 1; stride >>= 1) {
        emit(`if (group < ${stride}) {`, pushIndent);
        for (let i = 0; i < upcast; i++) {
          const thisSlot = `shmem[${i * groupSize} + group]`;
          const otherSlot = `shmem[${i * groupSize} + group + ${stride}]`;
          if (re.op === AluOp.Add) emit(`${thisSlot} += ${otherSlot};`);
          else if (re.op === AluOp.Mul) emit(`${thisSlot} *= ${otherSlot};`);
          else if (re.op === AluOp.Min) {
            if (re.dtype === DType.Bool)
              emit(`${thisSlot} = ${thisSlot} && ${otherSlot};`);
            else emit(`${thisSlot} = min(${thisSlot}, ${otherSlot});`);
          } else if (re.op === AluOp.Max) {
            if (re.dtype === DType.Bool)
              emit(`${thisSlot} = ${thisSlot} || ${otherSlot};`);
            else emit(`${thisSlot} = max(${thisSlot}, ${otherSlot});`);
          }
        }
        emit(popIndent, "}");
        emit("workgroupBarrier();");
      }

      // Thread 0 reads the final reduced value back into accumulators.
      emit("if (group == 0) {", pushIndent);
      for (let i = 0; i < upcast; i++) {
        emit(`${acc[i]} = shmem[${i * groupSize}];`);
      }
    }

    // Exited the reduction loop scope. Erase any local variables.
    expContext.clear();
    references.clear();
    seen.clear();

    const outputIdxExps: AluExp[] = [];
    const fusionExps: AluExp[] = [];
    for (let i = 0; i < upcast; i++) {
      const exp = tune.outputIdxExp.substitute({ upcast: AluExp.i32(i) });
      outputIdxExps.push(exp.simplify(cache));
      countReferences(outputIdxExps[i]);
      fusionExps.push(
        tune
          .epilogue!.substitute({
            acc: AluExp.variable(re.dtype, acc[i]),
            upcast: AluExp.i32(i),
          })
          .simplify(cache),
      );
      countReferences(fusionExps[i]);
    }
    for (let i = 0; i < upcast; i++) {
      const index = strip1(gen(outputIdxExps[i]));
      let rhs = strip1(gen(fusionExps[i]));
      if (resultTy !== dtypeToWgsl(fusionExps[i].dtype))
        rhs = `${resultTy}(${rhs})`;
      emit(`result[${index}] = ${rhs};`);
    }

    // Close the thread-0 guard for shared-memory reductions.
    if (useSharedMem) {
      emit(popIndent, "}");
    }
  }

  emit(popIndent, "}");

  const sharedBytes = useSharedMem
    ? groupSize *
      ((tune as WebGPUTuneResult).size.upcast ?? 1) *
      byteWidth(re!.dtype)
    : undefined;
  return {
    code: shader.join("\n"),
    numInputs: nargs,
    numOutputs: 1,
    hasUniform: symbolic || symbolicReduce,
    passes: [{ grid: [gridX, gridY] }],
    isSymbolic: symbolic || undefined,
    workgroupSize: symbolic ? workgroupSize : undefined,
    hasSymbolicReduction: symbolicReduce || undefined,
    sharedMemoryBytes: sharedBytes,
  };
}

// ---------------------------------------------------------------------------
// Scan shader codegen (P3: WebGPU multi-kernel scan)
// ---------------------------------------------------------------------------

type EmitFn = (...lines: (string | symbol)[]) => void;
const PUSH_INDENT = Symbol("pushIndent");
const POP_INDENT = Symbol("popIndent");

function createShaderEmitter(): {
  emit: EmitFn;
  pushIndent: symbol;
  popIndent: symbol;
  getCode: () => string;
} {
  const shader: string[] = [];
  let indent = "";
  const emit: EmitFn = (...lines) => {
    for (const line of lines) {
      if (line === PUSH_INDENT) indent += "  ";
      else if (line === POP_INDENT) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : line);
    }
  };
  return {
    emit,
    pushIndent: PUSH_INDENT,
    popIndent: POP_INDENT,
    getCode: () => shader.join("\n"),
  };
}

/**
 * Generate a WGSL expression for a scan body kernel.
 *
 * Maps GlobalIndex gids to scan-specific buffer names (const0, carry0, xs0).
 * xs inputs use `dataIdx` for iteration-aware addressing.
 */
function genScanExpressionWithRidx(
  exp: AluExp,
  _dtype: DType,
  numConsts: number,
  numCarry: number,
  xsElemStrides: number[],
): string {
  const gen = (e: AluExp): string => {
    const { op, src, dtype: eDtype, arg } = e;

    // Handle scan-specific GlobalIndex classification
    if (op === AluOp.GlobalIndex) {
      const gid = arg[0] as number;
      const idxCode = gen(src[0]);

      if (gid < numConsts) {
        const access = `const${gid}[${idxCode}]`;
        return eDtype === DType.Bool ? `(${access} != 0)` : access;
      } else if (gid < numConsts + numCarry) {
        const carryIdx = gid - numConsts;
        const access = `carry${carryIdx}[${idxCode}]`;
        return eDtype === DType.Bool ? `(${access} != 0)` : access;
      } else {
        const xIdx = gid - numConsts - numCarry;
        const stride = xsElemStrides[xIdx];
        const access = `xs${xIdx}[i32(dataIdx) * ${stride} + ${idxCode}]`;
        return eDtype === DType.Bool ? `(${access} != 0)` : access;
      }
    }

    if (op === AluOp.Const) return constToWgsl(eDtype, arg);

    if (op === AluOp.Special) {
      const name = Array.isArray(arg) ? arg[0] : arg;
      if (name === "gidx") return "gidx";
      if (name === "ridx") return "ridx";
      return name as string;
    }

    if (op === AluOp.Variable) {
      if (arg === "acc") return "acc";
      if (arg === "gidx") return "gidx";
      if (arg === "ridx") return "ridx";
      return arg as string;
    }

    // Erf/Erfc with f32 precision wrapper
    if (op === AluOp.Erf || op === AluOp.Erfc) {
      const funcName = op === AluOp.Erf ? "erf" : "erfc";
      const a = strip1(gen(src[0]));
      if (eDtype !== DType.Float32) {
        return `${dtypeToWgsl(eDtype)}(${funcName}(f32(${a})))`;
      }
      return `${funcName}(${a})`;
    }

    // Binary ops
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) {
        return eDtype === DType.Bool ? `(${a} || ${b})` : `(${a} + ${b})`;
      }
      if (op === AluOp.Sub) return `(${a} - ${b})`;
      if (op === AluOp.Mul) {
        return eDtype === DType.Bool ? `(${a} && ${b})` : `(${a} * ${b})`;
      }
      if (op === AluOp.Idiv) {
        return isFloatDtype(eDtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
      }
      if (op === AluOp.Mod) return `(${a} % ${b})`;
      if (op === AluOp.Min) {
        return eDtype === DType.Bool
          ? `(${a} && ${b})`
          : `min(${strip1(a)}, ${strip1(b)})`;
      }
      if (op === AluOp.Max) {
        return eDtype === DType.Bool
          ? `(${a} || ${b})`
          : `max(${strip1(a)}, ${strip1(b)})`;
      }
      if (op === AluOp.Cmplt) return `(${a} < ${b})`;
      if (op === AluOp.Cmpne) return `(${a} != ${b})`;
    }

    // Unary ops
    if (AluGroup.Unary.has(op)) {
      const a = gen(src[0]);
      if (op === AluOp.Sin) return `sin(${strip1(a)})`;
      if (op === AluOp.Cos) return `cos(${strip1(a)})`;
      if (op === AluOp.Asin) return `asin(${strip1(a)})`;
      if (op === AluOp.Atan) return `atan(${strip1(a)})`;
      if (op === AluOp.Exp) return `exp(${strip1(a)})`;
      if (op === AluOp.Log) return `log(${strip1(a)})`;
      if (op === AluOp.Sqrt) return `sqrt(${strip1(a)})`;
      if (op === AluOp.Reciprocal) return `(1.0 / ${a})`;
      if (op === AluOp.Floor) return `floor(${strip1(a)})`;
      if (op === AluOp.Ceil) return `ceil(${strip1(a)})`;
      if (op === AluOp.Cast) return `${dtypeToWgsl(eDtype)}(${strip1(a)})`;
      if (op === AluOp.Bitcast) {
        return `bitcast<${dtypeToWgsl(eDtype)}>(${strip1(a)})`;
      }
    }

    // Ternary
    if (op === AluOp.Where) {
      return `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
    }

    throw new Error(`genScanExpressionWithRidx: unsupported op ${AluOp[op]}`);
  };

  return strip1(gen(exp));
}

/**
 * Generate a WGSL shader for native scan with multiple kernel steps.
 *
 * Each step writes to a carry buffer and optionally to a stacked Y output.
 * The scan loop runs `length` iterations, executing all kernel steps per
 * iteration. Kernels may have reductions (inner loops).
 *
 * Buffer layout:
 *   - binding 0..numConsts-1: constants (read)
 *   - binding numConsts..numConsts+numCarry-1: initCarry (read)
 *   - binding numConsts+numCarry..+numX: xs (read)
 *   - binding +numX..+numCarry: carryOut (read_write)
 *   - binding +numCarry..+numY: ysStacked (read_write)
 */
function nativeScanMultiShaderSource(
  device: GPUDevice,
  params: NativeScanMultiParams,
): ShaderInfo {
  const {
    length,
    numConsts,
    carrySizes,
    xsStrides,
    ysStrides,
    steps,
    numCarry,
    numX,
    numY,
    reverse,
  } = params;

  // Determine dtype from first kernel step
  const dtype = steps[0]?.kernel.outputs[0].dtype ?? DType.Float32;
  const resultTy = dtypeToWgsl(dtype, true);
  const elemSize = byteWidth(dtype);

  // Compute element-level strides (byte strides / element size)
  const xsElemStrides = xsStrides.map((s) => s / elemSize);

  // Find the maximum kernel size across all steps
  const maxKernelSize = Math.max(
    ...steps.map((s) => s.kernel.size as number),
    1,
  );

  const { emit, pushIndent, popIndent, getCode } = createShaderEmitter();

  // Check for f16 requirement
  if (dtype === DType.Float16) {
    if (!device.features.has("shader-f16")) {
      throw new Error("WebGPU device does not support shader-f16 feature");
    }
    emit("enable f16;");
  }

  emit(headerWgsl);

  // Global function definitions needed by all kernels
  const allDistinctOps = new Set<AluOp>();
  for (const step of steps) {
    const tune = tuneNullopt(step.kernel);
    for (const [op] of tune.exp.distinctOps()) allDistinctOps.add(op);
    if (tune.epilogue) {
      for (const [op] of tune.epilogue.distinctOps()) allDistinctOps.add(op);
    }
  }
  if (allDistinctOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allDistinctOps.has(AluOp.Erf) || allDistinctOps.has(AluOp.Erfc)) {
    emit(erfSrc);
  }

  emit("");

  // Buffer declarations
  let bindingIdx = 0;

  for (let i = 0; i < numConsts; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> const${i}: array<${resultTy}>;`,
    );
  }
  for (let i = 0; i < numCarry; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> initCarry${i}: array<${resultTy}>;`,
    );
  }
  for (let i = 0; i < numX; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> xs${i}: array<${resultTy}>;`,
    );
  }
  for (let i = 0; i < numCarry; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> carry${i}: array<${resultTy}>;`,
    );
  }
  for (let i = 0; i < numY; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> ys${i}: array<${resultTy}>;`,
    );
  }

  // Compute shader entry point
  const workgroupSize = Math.min(Math.max(maxKernelSize, 1), 256);
  const [gridX, gridY] = calculateGrid(
    Math.ceil(Math.max(maxKernelSize, 1) / workgroupSize),
  );

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
    pushIndent,
  );

  emit(`let gidx = i32(id.x);`);
  emit("");

  // Step 1: Copy initCarry to carryOut (working buffer)
  emit("// Initialize carry from initCarry");
  for (let i = 0; i < numCarry; i++) {
    const carrySize = carrySizes[i] / elemSize;
    emit(`if (gidx < ${carrySize}) {`);
    emit(pushIndent);
    emit(`carry${i}[gidx] = initCarry${i}[gidx];`);
    emit(popIndent, "}");
  }
  emit("");

  // Step 2: Main scan loop
  emit(`// Main scan loop over ${length} iterations`);
  emit(`for (var iter: u32 = 0u; iter < ${length}u; iter++) {`, pushIndent);

  if (reverse) {
    emit(`let dataIdx = ${length - 1}u - iter;`);
  } else {
    emit(`let dataIdx = iter;`);
  }

  // Execute each kernel step
  for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
    const step = steps[stepIdx];
    const kernel = step.kernel;
    const tune = tuneNullopt(kernel);
    const carryIdx = step.outputCarryIdx;
    const kernelSize = kernel.size;
    const ysElemStride = ysStrides[carryIdx] / elemSize;

    emit("");
    emit(`// Step ${stepIdx}: kernel writes to carry${carryIdx}`);
    emit(`if (gidx < ${kernelSize}) {`);
    emit(pushIndent);

    const re = kernel.outputs[0].reduction;
    if (re) {
      // Reduction kernel: inner ridx loop + epilogue
      const accTy = dtypeToWgsl(re.dtype, true);
      emit(`var acc: ${accTy} = ${constToWgsl(re.dtype, re.identity)};`);
      emit(
        `for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`,
        pushIndent,
      );

      const expCode = genScanExpressionWithRidx(
        tune.exp,
        dtype,
        numConsts,
        numCarry,
        xsElemStrides,
      );
      emit(`let val = ${expCode};`);

      // Accumulate
      if (re.op === AluOp.Add) emit(`acc = acc + val;`);
      else if (re.op === AluOp.Mul) emit(`acc = acc * val;`);
      else if (re.op === AluOp.Min) emit(`acc = min(acc, val);`);
      else if (re.op === AluOp.Max) emit(`acc = max(acc, val);`);
      else throw new Error(`Unsupported reduction op: ${re.op}`);

      emit(popIndent, "}");

      // Apply epilogue
      const epilogueCode = genScanExpressionWithRidx(
        tune.epilogue!,
        dtype,
        numConsts,
        numCarry,
        xsElemStrides,
      );
      emit(`let result_val_${stepIdx}: ${resultTy} = ${epilogueCode};`);
    } else {
      // Elementwise kernel
      const expCode = genScanExpressionWithRidx(
        tune.exp,
        dtype,
        numConsts,
        numCarry,
        xsElemStrides,
      );
      emit(`let result_val_${stepIdx}: ${resultTy} = ${expCode};`);
    }

    // Write to ysStacked at dataIdx * stride + gidx
    if (numY > 0 && carryIdx < numY) {
      emit(
        `ys${carryIdx}[i32(dataIdx) * ${ysElemStride} + gidx] = result_val_${stepIdx};`,
      );
    }

    // Update carry for next iteration
    emit(`carry${carryIdx}[gidx] = result_val_${stepIdx};`);

    emit(popIndent, "}");
  }

  emit(popIndent, "}");
  emit(popIndent, "}");

  const numReadOnlyInputs = numConsts + numCarry + numX;
  const numReadWriteOutputs = numCarry + numY;

  return {
    code: getCode(),
    numInputs: numReadOnlyInputs,
    numOutputs: numReadWriteOutputs,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }],
  };
}

// ---------------------------------------------------------------------------
// Fused associative scan shader (WebGPU Kogge-Stone)
// ---------------------------------------------------------------------------

/**
 * Generate a WGSL expression for an associative scan body kernel step.
 *
 * Maps GlobalIndex gids to:
 *   - gid < numConsts         → const{gid}[idx]
 *   - gid ∈ [numConsts, numConsts+numLeaves)  → "a" leaf: ping at position (gidx - stride)
 *   - gid ∈ [numConsts+numLeaves, numConsts+2*numLeaves) → "b" leaf: ping at position gidx
 *   - gid ≥ numInputs         → internal_N[idx] (private array)
 *
 * Ping buffer layout: [leaf0: N * elemCount0][leaf1: N * elemCount1]...
 * Leaf k at position j, sub-index idx: ping[leafStart_k * uniforms.N + j * elemCount_k + idx]
 */
function genAssocScanExpression(
  exp: AluExp,
  _dtype: DType,
  numConsts: number,
  numLeaves: number,
  leafElemCounts: number[],
  _internalElemCounts: number[],
): string {
  const numInputs = numConsts + 2 * numLeaves;

  // Compute prefix sums for leaf start offsets (in typed elements relative to N=1).
  // leafStarts[k] = sum of leafElemCounts[0..k-1]
  const leafStarts: number[] = [0];
  for (let k = 1; k < numLeaves; k++) {
    leafStarts[k] = leafStarts[k - 1] + leafElemCounts[k - 1];
  }

  const gen = (e: AluExp): string => {
    const { op, src, dtype: eDtype, arg } = e;

    if (op === AluOp.GlobalIndex) {
      const gid = arg[0] as number;
      const idxCode = gen(src[0]);

      if (gid < numConsts) {
        // Constant buffer
        const access = `const${gid}[${idxCode}]`;
        return eDtype === DType.Bool ? `(${access} != 0)` : access;
      } else if (gid < numConsts + numLeaves) {
        // "a" leaf — read from ping at position (gidx - stride)
        const leafIdx = gid - numConsts;
        const elemCount = leafElemCounts[leafIdx];
        const start = leafStarts[leafIdx];
        const access = `ping[${start}u * uniforms.N + u32(a_pos) * ${elemCount}u + u32(${idxCode})]`;
        return eDtype === DType.Bool ? `(${access} != 0)` : access;
      } else if (gid < numInputs) {
        // "b" leaf — read from ping at position gidx
        const leafIdx = gid - numConsts - numLeaves;
        const elemCount = leafElemCounts[leafIdx];
        const start = leafStarts[leafIdx];
        const access = `ping[${start}u * uniforms.N + u32(gidx) * ${elemCount}u + u32(${idxCode})]`;
        return eDtype === DType.Bool ? `(${access} != 0)` : access;
      } else {
        // Internal buffer (var<private>)
        const intIdx = gid - numInputs;
        const access = `internal_${intIdx}[${idxCode}]`;
        return eDtype === DType.Bool ? `(${access} != 0)` : access;
      }
    }

    if (op === AluOp.Const) return constToWgsl(eDtype, arg);

    if (op === AluOp.Special) {
      const name = Array.isArray(arg) ? arg[0] : arg;
      if (name === "gidx") return "gidx";
      if (name === "ridx") return "ridx";
      return name as string;
    }

    if (op === AluOp.Variable) {
      if (arg === "acc") return "acc";
      if (arg === "gidx") return "gidx";
      if (arg === "ridx") return "ridx";
      return arg as string;
    }

    if (op === AluOp.Erf || op === AluOp.Erfc) {
      const funcName = op === AluOp.Erf ? "erf" : "erfc";
      const a = strip1(gen(src[0]));
      if (eDtype !== DType.Float32) {
        return `${dtypeToWgsl(eDtype)}(${funcName}(f32(${a})))`;
      }
      return `${funcName}(${a})`;
    }

    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) {
        return eDtype === DType.Bool ? `(${a} || ${b})` : `(${a} + ${b})`;
      }
      if (op === AluOp.Sub) return `(${a} - ${b})`;
      if (op === AluOp.Mul) {
        return eDtype === DType.Bool ? `(${a} && ${b})` : `(${a} * ${b})`;
      }
      if (op === AluOp.Idiv) {
        return isFloatDtype(eDtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
      }
      if (op === AluOp.Mod) return `(${a} % ${b})`;
      if (op === AluOp.Min) {
        return eDtype === DType.Bool
          ? `(${a} && ${b})`
          : `min(${strip1(a)}, ${strip1(b)})`;
      }
      if (op === AluOp.Max) {
        return eDtype === DType.Bool
          ? `(${a} || ${b})`
          : `max(${strip1(a)}, ${strip1(b)})`;
      }
      if (op === AluOp.Cmplt) return `(${a} < ${b})`;
      if (op === AluOp.Cmpne) return `(${a} != ${b})`;
    }

    if (AluGroup.Unary.has(op)) {
      const a = gen(src[0]);
      if (op === AluOp.Sin) return `sin(${strip1(a)})`;
      if (op === AluOp.Cos) return `cos(${strip1(a)})`;
      if (op === AluOp.Asin) return `asin(${strip1(a)})`;
      if (op === AluOp.Atan) return `atan(${strip1(a)})`;
      if (op === AluOp.Exp) return `exp(${strip1(a)})`;
      if (op === AluOp.Log) return `log(${strip1(a)})`;
      if (op === AluOp.Sqrt) return `sqrt(${strip1(a)})`;
      if (op === AluOp.Reciprocal) return `(1.0 / ${a})`;
      if (op === AluOp.Floor) return `floor(${strip1(a)})`;
      if (op === AluOp.Ceil) return `ceil(${strip1(a)})`;
      if (op === AluOp.Cast) return `${dtypeToWgsl(eDtype)}(${strip1(a)})`;
      if (op === AluOp.Bitcast) {
        return `bitcast<${dtypeToWgsl(eDtype)}>(${strip1(a)})`;
      }
    }

    if (op === AluOp.Where) {
      return `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
    }

    throw new Error(`genAssocScanExpression: unsupported op ${AluOp[op]}`);
  };

  return strip1(gen(exp));
}

/**
 * Generate WGSL shader source for one Kogge-Stone round of a fused
 * associative scan. Each GPU thread processes one element position:
 *
 *   if gidx >= stride:
 *     result = fn(ping[gidx - stride], ping[gidx])
 *     pong[gidx] = result
 *   else:
 *     pong[gidx] = ping[gidx]  (copy)
 *
 * Buffer layout (ping and pong):
 *   [leaf0: N * elemCount0 | leaf1: N * elemCount1 | ...]
 *
 * Bindings:
 *   group(0) binding(0):     ping (storage, read)
 *   group(0) binding(1):     pong (storage, read_write)
 *   group(0) binding(2..2+K): const0..constK (storage, read)
 *   group(1) binding(0):     uniforms { stride: u32, N: u32 }
 */
function assocScanFusedShaderSource(
  device: GPUDevice,
  params: WebGPUAssocScanParams,
): { code: string; workgroupSize: number } {
  const {
    numConsts,
    numLeaves,
    leafElemCounts,
    steps,
    internalElemCounts,
    leafToInternalIdx,
    dtype,
  } = params;

  const resultTy = dtypeToWgsl(dtype, true);

  // Compute max per-position elements across all leaves for thread count
  // Compute leaf start offsets (prefix sum of elemCounts)
  const leafStarts: number[] = [0];
  for (let k = 1; k < numLeaves; k++) {
    leafStarts[k] = leafStarts[k - 1] + leafElemCounts[k - 1];
  }

  const { emit, pushIndent, popIndent, getCode } = createShaderEmitter();

  if (dtype === DType.Float16) {
    if (!device.features.has("shader-f16")) {
      throw new Error("WebGPU device does not support shader-f16 feature");
    }
    emit("enable f16;");
  }

  emit(headerWgsl);

  // Collect ops that need global function definitions
  const allDistinctOps = new Set<AluOp>();
  for (const step of steps) {
    const tune = tuneNullopt(step.kernel);
    for (const [op] of tune.exp.distinctOps()) allDistinctOps.add(op);
    if (tune.epilogue) {
      for (const [op] of tune.epilogue.distinctOps()) allDistinctOps.add(op);
    }
  }
  if (allDistinctOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allDistinctOps.has(AluOp.Erf) || allDistinctOps.has(AluOp.Erfc)) {
    emit(erfSrc);
  }

  emit("");

  // Uniform struct
  emit("struct AssocScanUniforms {");
  emit("  stride: u32,");
  emit("  N: u32,");
  emit("}");
  emit("");

  // Buffer bindings
  emit(`@group(0) @binding(0) var<storage, read> ping: array<${resultTy}>;`);
  emit(
    `@group(0) @binding(1) var<storage, read_write> pong: array<${resultTy}>;`,
  );
  for (let i = 0; i < numConsts; i++) {
    emit(
      `@group(0) @binding(${i + 2}) var<storage, read> const${i}: array<${resultTy}>;`,
    );
  }
  emit("@group(1) @binding(0) var<uniform> uniforms: AssocScanUniforms;");
  emit("");

  // Compute shader
  // Each thread handles one "position" in the scan, but iterates over
  // all sub-elements within that position (multiple leaves, each with
  // multiple typed elements).
  const workgroupSize = 256;

  emit(`@compute @workgroup_size(${workgroupSize})`);
  emit("fn main(@builtin(global_invocation_id) id: vec3<u32>) {");
  emit(pushIndent);
  emit("let gidx = i32(id.x);");
  emit("if (u32(gidx) >= uniforms.N) { return; }");
  emit("");

  // Declare private internal buffers for intermediate step results
  for (let i = 0; i < internalElemCounts.length; i++) {
    const count = internalElemCounts[i];
    emit(`var internal_${i}: array<${resultTy}, ${count}>;`);
  }

  emit("");
  emit("let a_pos = gidx - i32(uniforms.stride);");
  emit("");

  emit("if (a_pos >= 0) {");
  emit(pushIndent);

  // Execute all body steps — this is the fused fn(a, b) computation
  for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
    const step = steps[stepIdx];
    const kernel = step.kernel;
    const tune = tuneNullopt(kernel);
    const kernelSize =
      typeof kernel.size === "number"
        ? kernel.size
        : (kernel.concreteSizeHint ?? 1);

    emit(`// Step ${stepIdx}`);

    // After tuneNullopt, the expression uses AluOp.Special("gidx") for the
    // output element index. In the fused scan shader, `gidx` is the scan
    // position (thread ID). We must rewrite Special("gidx") → Special("eidx")
    // so the expression indexes the per-position output element, not the scan
    // position. `substitute()` only matches AluOp.Variable — it won't touch
    // AluOp.Special nodes. Use `rewrite()` instead.
    const eidxVar = AluExp.special(DType.Int32, "eidx", kernelSize);
    const rewriteGidxToEidx = (exp: AluExp): AluExp =>
      exp.rewrite((node) => {
        if (node.op === AluOp.Special) {
          const name = Array.isArray(node.arg) ? node.arg[0] : node.arg;
          if (name === "gidx") return eidxVar;
        }
      });

    const re = kernel.outputs[0].reduction;
    if (re) {
      // Reduction kernel — must iterate over output elements (eidx)
      // just like elementwise kernels. Each output element has its own
      // reduction accumulator. For kernelSize=1 (scalar reduction) the
      // loop body runs once; for kernelSize>1 (e.g. matmul output) it
      // runs once per output element.
      const accTy = dtypeToWgsl(re.dtype, true);
      const redSize =
        typeof tune.size.reduce === "number"
          ? tune.size.reduce
          : (re.concreteHint ?? Number(tune.size.reduce));

      const substExp = rewriteGidxToEidx(tune.exp);
      const substEpilogue = rewriteGidxToEidx(tune.epilogue!);

      if (kernelSize > 1) {
        emit(
          `for (var eidx: i32 = 0; eidx < ${kernelSize}; eidx++) {`,
          pushIndent,
        );
      } else {
        emit(`{`);
        emit(pushIndent);
        emit(`let eidx: i32 = 0;`);
      }
      emit(`var acc: ${accTy} = ${constToWgsl(re.dtype, re.identity)};`);
      emit(`for (var ridx: i32 = 0; ridx < ${redSize}; ridx++) {`, pushIndent);

      const expCode = genAssocScanExpression(
        substExp,
        dtype,
        numConsts,
        numLeaves,
        leafElemCounts,
        internalElemCounts,
      );
      emit(`let val = ${expCode};`);

      if (re.op === AluOp.Add) emit(`acc = acc + val;`);
      else if (re.op === AluOp.Mul) emit(`acc = acc * val;`);
      else if (re.op === AluOp.Min) emit(`acc = min(acc, val);`);
      else if (re.op === AluOp.Max) emit(`acc = max(acc, val);`);
      else throw new Error(`Unsupported reduction op: ${re.op}`);

      emit(popIndent, "}");

      const epilogueCode = genAssocScanExpression(
        substEpilogue,
        dtype,
        numConsts,
        numLeaves,
        leafElemCounts,
        internalElemCounts,
      );
      emit(`internal_${step.outputInternalIdx}[eidx] = ${epilogueCode};`);
      emit(popIndent, "}");
    } else {
      // Elementwise kernel — iterate over sub-elements
      if (kernelSize > 1) {
        emit(`for (var eidx: i32 = 0; eidx < ${kernelSize}; eidx++) {`);
        emit(pushIndent);

        const substExp = rewriteGidxToEidx(tune.exp);
        const expCode = genAssocScanExpression(
          substExp,
          dtype,
          numConsts,
          numLeaves,
          leafElemCounts,
          internalElemCounts,
        );
        emit(`internal_${step.outputInternalIdx}[eidx] = ${expCode};`);
        emit(popIndent, "}");
      } else {
        const expCode = genAssocScanExpression(
          tune.exp,
          dtype,
          numConsts,
          numLeaves,
          leafElemCounts,
          internalElemCounts,
        );
        emit(`internal_${step.outputInternalIdx}[0] = ${expCode};`);
      }
    }
    emit("");
  }

  // Write results from internal buffers to pong
  for (let k = 0; k < numLeaves; k++) {
    const intIdx = leafToInternalIdx[k];
    const elemCount = leafElemCounts[k];
    const start = leafStarts[k];
    if (elemCount > 1) {
      emit(`for (var wi: u32 = 0u; wi < ${elemCount}u; wi++) {`);
      emit(pushIndent);
      emit(
        `pong[${start}u * uniforms.N + u32(gidx) * ${elemCount}u + wi] = internal_${intIdx}[wi];`,
      );
      emit(popIndent, "}");
    } else {
      emit(`pong[${start}u * uniforms.N + u32(gidx)] = internal_${intIdx}[0];`);
    }
  }

  emit(popIndent, "} else {");
  emit(pushIndent);

  // Copy: pong[gidx] = ping[gidx] for all leaf elements
  emit("// Copy: position before stride, no fn application");
  for (let k = 0; k < numLeaves; k++) {
    const elemCount = leafElemCounts[k];
    const start = leafStarts[k];
    if (elemCount > 1) {
      emit(`for (var ci: u32 = 0u; ci < ${elemCount}u; ci++) {`);
      emit(pushIndent);
      emit(
        `pong[${start}u * uniforms.N + u32(gidx) * ${elemCount}u + ci] = ping[${start}u * uniforms.N + u32(gidx) * ${elemCount}u + ci];`,
      );
      emit(popIndent, "}");
    } else {
      emit(
        `pong[${start}u * uniforms.N + u32(gidx)] = ping[${start}u * uniforms.N + u32(gidx)];`,
      );
    }
  }

  emit(popIndent, "}");
  emit(popIndent, "}");

  return { code: getCode(), workgroupSize };
}

function pipelineSubmit(
  device: GPUDevice,
  pipelines: ShaderDispatch[],
  inputs: GPUBuffer[],
  outputs: GPUBuffer[],
  dynamicParams?: number[],
  batchEncoder?: GPUCommandEncoder,
  batchUniformCollector?: GPUBuffer[],
) {
  const commandEncoder = batchEncoder ?? device.createCommandEncoder();
  const uniformBuffersToDestroy: GPUBuffer[] = [];
  for (const { pipeline, ...shader } of pipelines) {
    if (
      inputs.length !== shader.numInputs ||
      outputs.length !== shader.numOutputs
    ) {
      throw new Error(
        `webgpu: expected ${shader.numInputs} inputs and ${shader.numOutputs} outputs, ` +
          `got ${inputs.length} inputs and ${outputs.length} outputs`,
      );
    }

    const filteredPasses = shader.passes.filter(({ grid }) => prod(grid) > 0);
    if (filteredPasses.length === 0) continue; // No work to do.

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        ...inputs.map((buffer, i) => ({
          binding: i,
          resource: { buffer },
        })),
        ...outputs.map((buffer, i) => ({
          binding: inputs.length + i,
          resource: { buffer },
        })),
      ],
    });

    let uniformBindGroup: GPUBindGroup | null = null;
    let uniformAlignment = 0;

    // Symbolic shaders: compute grid dynamically from dynamicParams and
    // create a uniform buffer with resolved dimension values.
    // struct Dims layout matches the shader:
    //   isSymbolic + hasSymbolicReduction → { total_size: u32, reduce_size: u32 }
    //   isSymbolic only                   → { total_size: u32 }
    //   hasSymbolicReduction only         → { reduce_size: u32 }
    let symbolicGrid: [number, number] | null = null;
    const needsSymbolicUniform =
      (shader.isSymbolic || shader.hasSymbolicReduction) &&
      dynamicParams &&
      dynamicParams.length > 0;
    if (needsSymbolicUniform) {
      if (shader.isSymbolic) {
        const totalSize = dynamicParams[0];
        const wgSize =
          typeof shader.workgroupSize === "number"
            ? shader.workgroupSize
            : (shader.workgroupSize?.[0] ?? 256);
        const gridSize = Math.ceil(totalSize / wgSize);
        symbolicGrid = calculateGrid(gridSize);
      }

      // Build uniform data matching the shader's struct Dims field order
      const uniformFields: number[] = [];
      if (shader.isSymbolic) uniformFields.push(dynamicParams[0]); // total_size
      if (shader.hasSymbolicReduction) uniformFields.push(dynamicParams[1]); // reduce_size
      const dataSize = uniformFields.length * 4;
      const uniformData = new Uint8Array(dataSize);
      const dv = new DataView(uniformData.buffer);
      for (let fi = 0; fi < uniformFields.length; fi++) {
        dv.setUint32(fi * 4, uniformFields[fi], true);
      }

      const minAlign = device.limits.minUniformBufferOffsetAlignment;
      const alignment = Math.ceil(dataSize / minAlign) * minAlign;
      const uniformBuffer = device.createBuffer({
        size: alignment,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
      });
      new Uint8Array(uniformBuffer.getMappedRange()).set(uniformData);
      uniformBuffer.unmap();
      uniformBuffersToDestroy.push(uniformBuffer);
      uniformAlignment = alignment;
      uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer, size: alignment } },
        ],
      });
    } else if (shader.hasUniform) {
      // Non-symbolic shader with uniforms: create a shared buffer with uniform
      // values for each pass of the shader (use dynamic offsets).
      const uniforms = filteredPasses.map(({ uniform }) => uniform!);
      const [uniformBuffer, alignment] = combineUniforms(device, uniforms);
      uniformBuffersToDestroy.push(uniformBuffer);
      uniformAlignment = alignment;
      uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer, size: alignment } },
        ],
      });
    }

    for (let i = 0; i < filteredPasses.length; i++) {
      const grid = symbolicGrid ?? filteredPasses[i].grid;
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      if (uniformBindGroup)
        passEncoder.setBindGroup(1, uniformBindGroup, [i * uniformAlignment]);
      passEncoder.dispatchWorkgroups(grid[0], grid[1]);
      passEncoder.end();
    }
  }
  if (batchEncoder) {
    // In batch mode: collect uniforms for caller to destroy after submit
    batchUniformCollector!.push(...uniformBuffersToDestroy);
  } else {
    device.queue.submit([commandEncoder.finish()]);
    for (const buf of uniformBuffersToDestroy) buf.destroy();
  }
}

function combineUniforms(
  device: GPUDevice,
  uniforms: Uint8Array<ArrayBuffer>[],
): [GPUBuffer, number] {
  for (const buf of uniforms) {
    if (
      !buf ||
      buf.byteLength === 0 ||
      buf.byteLength !== uniforms[0].byteLength
    ) {
      throw new Error("webgpu: Uniform mismatch between shader passes");
    }
  }
  const minAlign = device.limits.minUniformBufferOffsetAlignment;
  const alignment = Math.ceil(uniforms[0].byteLength / minAlign) * minAlign;
  const buffer = device.createBuffer({
    size: alignment * uniforms.length,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const bufferMapped = new Uint8Array(buffer.getMappedRange());
  for (let i = 0; i < uniforms.length; i++)
    bufferMapped.set(uniforms[i], i * alignment);
  buffer.unmap();
  return [buffer, alignment];
}

/**
 * A cache for compiled GPU compute pipelines, keyed by the shader source.
 *
 * This supports both async compilation (recommended) and a synchronous variant.
 * If the pipeline is not in the cache, it will be compiled and added. For async
 * compilation, only one compilation will be in progress at a time for a given
 * shader source.
 */
class ShaderPipelineCache {
  cache: Map<string, GPUComputePipeline>;
  inProgress: Map<string, Promise<GPUComputePipeline>>;
  #layoutCache: Map<string, GPUPipelineLayout>;

  constructor(readonly device: GPUDevice) {
    this.cache = new Map();
    this.inProgress = new Map();
    this.#layoutCache = new Map();
  }

  #getLayout(shader: ShaderInfo): GPUPipelineLayout {
    if (
      shader.numInputs + shader.numOutputs >
      this.device.limits.maxStorageBuffersPerShaderStage
    ) {
      const actual = shader.numInputs + shader.numOutputs;
      const max = this.device.limits.maxStorageBuffersPerShaderStage;
      throw new Error(
        `Too many buffers (${actual}) for WebGPU pipeline (max: ${max})`,
      );
    }
    // Cache by signature: most JIT programs share the same layout shape.
    const key = `${shader.numInputs}:${shader.numOutputs}:${shader.hasUniform ? 1 : 0}`;
    const cached = this.#layoutCache.get(key);
    if (cached) return cached;

    const bindGroupLayouts: GPUBindGroupLayout[] = [
      this.device.createBindGroupLayout({
        entries: range(shader.numInputs + shader.numOutputs).map((i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: i < shader.numInputs ? "read-only-storage" : "storage",
          },
        })),
      }),
    ];
    if (shader.hasUniform) {
      bindGroupLayouts.push(
        this.device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "uniform", hasDynamicOffset: true },
            },
          ],
        }),
      );
    }
    const layout = this.device.createPipelineLayout({ bindGroupLayouts });
    this.#layoutCache.set(key, layout);
    return layout;
  }

  async prepare(shader: ShaderInfo): Promise<GPUComputePipeline> {
    // Deno's WebGPU (wgpu-rs) doesn't support createComputePipelineAsync
    // reliably. Fall back to synchronous pipeline creation.
    // @ts-expect-error Deno global
    if (typeof Deno !== "undefined") return this.prepareSync(shader);

    const existingPipeline = this.cache.get(shader.code);
    if (existingPipeline) return existingPipeline;

    const existingPromise = this.inProgress.get(shader.code);
    if (existingPromise) return await existingPromise;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + shader.code);
    }

    const shaderModule = this.device.createShaderModule({ code: shader.code });
    const promise = (async () => {
      this.device.pushErrorScope("validation");
      try {
        const pipeline = await this.device.createComputePipelineAsync({
          layout: this.#getLayout(shader),
          compute: {
            module: shaderModule,
            entryPoint: "main",
          },
        });
        await this.device.popErrorScope();
        return pipeline;
      } catch (_error: unknown) {
        // This can race with other compilations, but it shouldn't happen in
        // correct code. Any validation error here is a bug in `jax-js`.
        const scope = await this.device.popErrorScope();
        const emsg = await compileError(shaderModule, scope, shader.code);
        throw new Error(emsg);
      }
    })();
    this.inProgress.set(shader.code, promise);

    // This could race against getSync(), but it's okay since shader pipeline
    // creation is deterministic + idempotent.
    const pipeline = await promise;
    this.cache.set(shader.code, pipeline);
    return pipeline;
  }

  prepareSync(shader: ShaderInfo): GPUComputePipeline {
    const existingPipeline = this.cache.get(shader.code);
    if (existingPipeline) return existingPipeline;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + shader.code);
    }

    const shaderModule = this.device.createShaderModule({ code: shader.code });
    // Deno's wgpu-rs doesn't support pushErrorScope/popErrorScope reliably.
    // @ts-expect-error Deno global
    const hasScopeApi = typeof Deno === "undefined";
    if (hasScopeApi) this.device.pushErrorScope("validation");
    const pipeline = this.device.createComputePipeline({
      layout: this.#getLayout(shader),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    if (hasScopeApi) {
      this.device.popErrorScope().then(async (scope) => {
        // This happens asynchronously, so we can't throw here. But shader syntax
        // validation errors should never occur in correct code. Any issues here
        // reflect bugs in jax-js.
        if (scope !== null) {
          const emsg = await compileError(shaderModule, scope, shader.code);
          console.error(emsg);
        }
      });
    }
    this.cache.set(shader.code, pipeline);
    return pipeline;
  }
}

/** Gather information about a compilation error and format it. */
async function compileError(
  shaderModule: GPUShaderModule,
  scope: GPUError | null,
  code: string,
): Promise<string> {
  let message = `Failed to compile shader: ${scope ? scope.message : "(no error scope)"}`;
  const info = await shaderModule.getCompilationInfo();
  for (const msg of info.messages) {
    message += `\n  [${msg.type} at ${msg.lineNum}:${msg.linePos}] ${msg.message}`;
  }
  if (code) {
    message += `\n\n${code}`;
  }
  return message;
}
