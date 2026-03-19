These notes help AI coding agents be immediately productive.

1. **AEP** — The Agentic Evolution Protocol (mandatory methodology)
2. **Repository Overview** — Architecture, workflows, ownership, WebGPU/WASM features
3. **Block Map & Tiled Compute** — `block_map` primitive, shader optimization, tiled matmul
4. **Scan & Associative Scan** — `lax.scan` and `lax.associativeScan` implementation
5. **Buffer Recycling & Pool** — JIT recycling, WebGPU buffer pool
6. **Performance Benchmarks** — Measured results and future opportunities

---

# The Agentic Evolution Protocol (AEP)

## 1. The Core Mandate: Asymptotic Velocity & Debt Collection

Your success is measured by the **net reduction in total system complexity**.

- **Ruthless Refactoring:** Scale is an asset. Refactor 1,000 files to achieve an architecture that
  will be cleaner in the long term, rather than patching the edges. Workarounds are not solutions —
  they are debt that compounds.

## 2. Provider Mode: Writing for the "Downstream Agent"

When you modify our public API, your task is to provide a "Migration Bundle" that includes:

- **Deterministic Mapping:** Raw structural diffs (e.g.,
  `Class.oldMethod(x) -> NewModule.fetch(x)`).
- **Workaround Signatures (The "Debt Collector"):** Explicitly identify patterns that consumers
  likely used to bypass old limitations.
  - _Example:_ "If the consumer has a local utility that manually merges X and Y because v1 didn't
    support it, delete that utility and use the new `nativeMerge` endpoint."
- **Signal your compliance:** If you were brought here by a failed command call, make sure you
  actually have read and followed the protocol. Only then call the command again: AEP=AEP COMMAND.

## 3. Consumer Mode: The Upgrade & Optimization Loop

When an upstream dependency evolves, follow this loop:

- **Contract-First Verification:** Update all local mocks and test fixtures to the new upstream
  reality _before_ touching our code.
- **The Workaround Hunt:** Scan our codebase for the "Workaround Signatures" provided by the
  upstream agent. Prioritize deleting our internal "helper" functions, wrappers, and polyfills in
  favor of the new native API implementation.
- **Viral Propagation:** Follow the change through our entire call stack. Do not hide a new Async
  requirement behind a Sync wrapper.

## 4. Continuous Knowledge Capture (Decentralized)

- **Architecture Decision Records (ADRs):** Document the "Why" behind large restructurings in a way
  that future sessions benefit from it. Don't keep stale information.
- **Code Comments:** If you misunderstood something in the code, add a code comment that prevents
  you from doing the same mistake again. Likewise, proactively comment non-obvious code with the
  rationale behind it.

## 5. Methodology

1. **The Plan:** For large tasks, generate a "Refactor Plan." List the "Workaround Signatures" you
   intend to hunt.
2. **The Ground Truth:** Run the native test suite. **Test Integrity is Absolute.**
3. **The Cleanup:** Remove all deprecated code paths, dead variables, and legacy comments. If you
   find a "todo" that the new API fixes, resolve it.

---

# Part 1: Repository Overview

## What is jax-js?

A JavaScript/TypeScript port of [JAX](https://github.com/google/jax) for high-performance numerical
computing in the browser and Node.js. Provides NumPy-like array operations, automatic
differentiation (`grad`, `jvp`, `vjp`), JIT compilation, and composable transformations (`jit`,
`grad`, `vmap`).

### Key concepts

- **Tracing:** `jit`-wrapped functions execute with "tracer" objects to record operations into a
  Jaxpr IR, compiled to optimized native code.
- **Kernels:** Compiled computation units. Elementwise ops are fused into single GPU/WASM kernels.
- **Autodiff:** `grad(f)` traces `f` to build a graph, then applies the chain rule automatically.

### Design philosophy

- **Generative compiler over static kernel libraries** — generating kernels from IR at runtime beats
  shipping pre-compiled kernels (TF.js, ONNX RT approach). New ops don't need hand-written kernels.
- **"80% of XLA" philosophy** — target 3–5× of optimal, not peak. We don't know the hardware.
- **Lightweight over exhaustive** — prefer composable primitives over specialized ops.
- **Explicit disposal over GC** — GPU/WASM buffers freed via `.dispose()`. `using` declarations also
  work.
- **Non-consuming operations** — ops leave inputs alive. No `.ref` needed in user code. Silent leaks
  over crashes; `checkLeaks` + ESLint plugin compensate.
- **Compounding returns** — every compiler improvement makes all operations faster.

### Development priorities

1. **Correctness** — tests, reference-counting discipline, cross-backend consistency
2. **API breadth** — NumPy/JAX compatibility (see `FEATURES.md`)
3. **Performance** — significant headroom in WASM SIMD, transformer inference (~1/3 of raw matmul
   GFLOP/s), conv2d
4. **Demos** — fluid simulations, neural networks, audio processing

## Architecture

- **Core** (`src/`): `frontend/` (array, jit, jvp, linearize, vmap, convolution, scan-plan,
  scan-executor, block-map-executor, artifacts), `library/` (numpy, lax, lax-scan,
  lax-associative-scan, nn, random, scipy-special, numpy-linalg, numpy-fft, lax-linalg,
  scipy-linalg), `backend/` (cpu, wasm, webgl, webgpu)
- **Aux packages**: `packages/loaders` (safetensors, OPFS, BPE), `packages/onnx`, `packages/optax`,
  `packages/eslint-plugin`
- **Website**: `website/` — live demos that double as integration tests

## Developer workflows

```bash
pnpm install                       # requires pnpm ≥ 10
pnpm run build                     # tsdown → dist/*.js, dist/*.d.ts
pnpm test                          # Vitest + Playwright (browser + node)
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm vitest bench bench/<file>     # run benchmarks
pnpm -C website dev                # local dev server
```

**Dual-GPU testing** — this machine has both NVIDIA RTX 4070 Ti SUPER (TB3 eGPU) and Intel Arc
(iGPU). WebGPU tests and benchmarks should always run on both unless there's a specific reason not
to. GPU selection happens at Chromium launch time via Vulkan driver flags, so each GPU requires a
separate vitest config.

```bash
# Run tests on both GPUs (default):
scripts/gpu-test.sh run test/gpu-bench.test.ts
pnpm run test:gpu                  # all tests, both GPUs

# Run benchmarks on both GPUs:
scripts/gpu-test.sh bench bench/matmul.bench.ts
pnpm run bench:gpu                 # all benches, both GPUs

# Single GPU:
GPU=nvidia scripts/gpu-test.sh run test/gpu-bench.test.ts
GPU=intel  scripts/gpu-test.sh bench bench/sort.bench.ts
pnpm run test:gpu:nvidia           # all tests, NVIDIA only
pnpm run bench:gpu:intel           # all benches, Intel only

# Or use configs directly:
pnpm vitest run <file> -c test/vitest.nvidia.config.ts
pnpm vitest run <file> -c test/vitest.intel.config.ts
```

**GPU config architecture:** `test/gpu-config.ts` is the shared factory;
`test/vitest.nvidia.config.ts` and `test/vitest.intel.config.ts` are thin wrappers. Intel needs
`VK_DRIVER_FILES` to bypass NVIDIA's default Vulkan adapter priority. Intel OOMs on 4096×4096 —
bench files must handle this gracefully.

**DISPLAY for GPU tests:** Headless Chromium still needs a running X server to initialize the GPU
adapter. The GPU config auto-detects `DISPLAY` by scanning `/tmp/.X11-unix/` for X server sockets
(falls back to `:0`). If GPU tests silently skip all WebGPU cases, check `ls /tmp/.X11-unix/` and
verify the detected socket matches a running X server.

**Pre-commit CI checks**: Husky runs `lint-staged`, then build + check + lint + tests. The
pre-commit hook (`scripts/precommit.sh`) also runs WebGPU-only correctness tests via
`scripts/gpu-test.sh` (NVIDIA-only on feature branches, both GPUs on main). Before commit:

```bash
pnpm build && pnpm check && pnpm test
```

**Debug logging** — use `setDebug(level)` (not env vars):

| Level | Output                                |
| ----- | ------------------------------------- |
| 0     | None (default)                        |
| 1     | JIT compile logs, scan path selection |
| 2     | Shader/WASM code, detailed tracing    |
| 3     | Expressions and metadata              |
| 4     | JIT programs, tuning details          |

**Temporary files**: Use `tmp/` (gitignored) for scratch files.

**Editing Prettier-managed files**: Run `npx prettier --write <file>` before `read_file` to get
canonical text for edits.

## Memory management & ownership

Operations **do not consume** inputs. Arrays stay alive until `.dispose()`'d. `.data()` reads
without consuming.

```ts
using x = np.array([1, 2, 3]);
using y = x.mul(np.array([2, 2, 2]));
// x, y auto-disposed at block end
```

Inside `jit()`, the compiler manages intermediate lifetimes automatically. In eager mode,
intermediates live until `.dispose()` or GC. `jit()` is a **pure performance optimization** — code
must be ownership-correct in both modes.

**Key ownership rules:**

- Every array you create must eventually be `.dispose()`'d (or `using`'d)
- `vjpFn.dispose()` / `jitFn.dispose()` — free captured intermediates/constants
- `tree.dispose(obj)` or `tree.makeDisposable(obj)` for pytree cleanup
- `consumeData()` = read + dispose in one call
- `tree.data(pytree)` = read all leaves in parallel (overlapping `mapAsync` calls)
- `tree.consumeData(pytree)` = read all leaves in parallel + dispose

**GPU readback latency:** Each `GPUBuffer.mapAsync()` round-trip costs ~12ms on eGPU (TB4). When
reading multiple outputs, use `tree.data()` / `tree.consumeData()` or `Promise.all()` to overlap
readbacks. Sequential readback of N outputs costs ~12ms × N; parallel costs ~12ms total.

**Leak detection:** `checkLeaks.start()` / `checkLeaks.stop()` — snapshots slot counts, tracks
creations. Used in `test/setup.ts` to wrap every test.

**JIT cache hierarchy:**

| Cache                    | Freed by `.dispose()`? | Freed by `clearCaches()`? |
| ------------------------ | ---------------------- | ------------------------- |
| Per-function jaxpr cache | **Yes**                | **Yes**                   |
| `jitCompileCache`        | No                     | **Yes** (+ GPU teardown)  |
| `transposeJaxprCache`    | No                     | **Yes**                   |
| `ShaderPipelineCache`    | No                     | No (GPUDevice lifetime)   |
| WASM routine LRU cache   | No                     | No (Backend lifetime)     |

**`transposeJaxprCache` is cache-owned** — callers must NOT dispose returned `ClosedJaxpr`.

**`jit()` deduplicates by function identity** via `WeakMap<Function, Map<string, OwnedFunction>>`.
Same `(fn, opts)` → same cache. Inline arrow functions are NOT deduped.

## WebGPU backend

**No Float64 on WebGPU.** WGSL has no `f64` type. All f64 work runs on WASM/CPU.

**Critical limitation:** No global barrier in WebGPU — threads in different workgroups cannot sync
within a single dispatch.

### Hard limits

| Limit                              | Typical Value | Impact                                           |
| ---------------------------------- | ------------- | ------------------------------------------------ |
| `maxStorageBuffersPerShaderStage`  | 8-10          | Limits kernel inputs; excess triggers fallback   |
| `maxComputeWorkgroupsPerDimension` | 65535         | Large arrays need 2D grid splitting              |
| `maxComputeWorkgroupSizeX`         | 256           | Limits threads per workgroup                     |
| `minUniformBufferOffsetAlignment`  | 256 bytes     | Dynamic uniform offsets must be 256-byte aligned |
| `minStorageBufferOffsetAlignment`  | 256 bytes     | Can't use buffer offsets for arbitrary strides   |

`splitGraphDataflow()` P2 pass prevents exceeding `maxStorageBuffersPerShaderStage` by splitting
fusion boundaries. `calculateGrid()` handles >65535 via 2D grid. Scan uses uniform-based offsets to
work around 256-byte alignment.

### Features exploited

| Feature                     | Usage                                                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **shader-f16**              | Float16 dtype support                                                                                                                              |
| **Workgroup shared memory** | Sort, JIT cooperative reductions                                                                                                                   |
| **workgroupBarrier()**      | Sort, shared-memory reduction tree                                                                                                                 |
| **storageBarrier()**        | Memory fence in Sort, Cholesky, LU                                                                                                                 |
| **Pipeline caching**        | Compiled pipelines stored by shader hash                                                                                                           |
| **Pipeline layout caching** | Cached by `numInputs:numOutputs:hasUniform` signature                                                                                              |
| **Command batching**        | Multiple dispatches encoded before single `queue.submit()`                                                                                         |
| **WGSL copy shader**        | Byte-level buffer copy when alignment fails                                                                                                        |
| **shader-f32-atomic-add**   | Native f32 `atomicAdd` in scatter-add shader                                                                                                       |
| **Ping-pong buffers**       | `lax.scan` carry alternates between two buffers                                                                                                    |
| **Uniform buffers**         | Per-iteration offsets for `lax.scan`                                                                                                               |
| **Subgroups**               | `subgroupAdd`/`Mul`/`Min`/`Max` in JIT & block-map reductions, `subgroupShuffleUp` in assocScan, `subgroupInclusiveAdd`/`Mul` for scalar assocScan |
| **Atomic CAS + Store**      | Decoupled Fallback scan: inter-workgroup prefix propagation via bounded-spin + CAS fallback                                                        |
| **timestamp-query**         | `profileGpu()` API via `_beginComputePass()` wrapper (P9) ✅                                                                                       |

### Features NOT exploited (opportunities)

| Feature                       | What it enables                    | Why not used / status                                 |
| ----------------------------- | ---------------------------------- | ----------------------------------------------------- |
| **Indirect dispatch**         | GPU-driven workgroup counts        | No dynamic control flow needs it yet                  |
| **Texture sampling**          | Hardware-accelerated interpolation | All ops use storage buffers                           |
| **Atomic operations**         | Lock-free reductions, histograms   | Used by Decoupled Fallback scan (P10) and scatter-add |
| **Cooperative matrix (WMMA)** | Hardware tensor core matmul        | WGSL spec not stable; Dawn experimental; ~2026 (P7)   |

### WebGPU dispatch overhead model

For many-dispatch programs (~100+ kernel steps), per-invocation overhead breaks down as: ~76%
JS-side JIT loop (scope lookups, array allocation, refcounting), ~18% `queue.submit()` (GPU
scheduling cost, irreducible), ~5% `createBindGroup()`. The command tape eliminates JS-side
overhead. The colored arena + bind group cache eliminate `createBindGroup()` cost via stable buffer
identities (100% cache hits for internal-only dispatches). The `queue.submit()` cost is
hardware-determined and irreducible without reducing dispatch count.

### WASM feature opportunities

| Priority | Feature            | Browser risk       | Impact      | Notes                                              |
| -------- | ------------------ | ------------------ | ----------- | -------------------------------------------------- |
| Medium   | i64 in wasmblr     | None (MVP)         | Medium-High | Unlocks f64 builtins, simplifies Threefry PRNG     |
| Medium   | Relaxed SIMD (FMA) | Safari unsupported | High        | `f32x4.relaxed_madd` for 2× dot-product throughput |
| Low      | Sign extension ops | None               | Low         | Marginal for float-focused workloads               |

### SIMD speedup by matrix size (wasmblr routines)

| Matrix Size | f32x4 Speedup | f64x2 Speedup |
| ----------- | ------------- | ------------- |
| n < 32      | ~0.8x (skip)  | ~0.9x (skip)  |
| n = 32      | ~1.1x         | ~1.0x         |
| n = 64      | ~1.7x         | ~1.3x         |
| n = 128     | ~3.0x         | ~1.8x         |
| n = 256     | ~3.8x         | ~1.9x         |

## JIT compiler

**Pipeline:** `makeJaxpr(f)` → `jaxpr.flatten().simplify()` → `splitGraphDataflow()` →
`jitCompile()` → `JitProgram.execute()`

**JitStep types:** `execute` (kernel/routine dispatch), `malloc` (with optional `initialData` for
pre-filled constants), `incref`, `free`, `recycle` (buffer reuse), `scan`, `dus` (zero-copy
DynamicUpdateSlice), `scatter_add`, `assoc_scan`, `block_map`, `workgroup_assoc_scan`, `fori_loop`,
`reverse`

**Scalar promotion (O2):** `pushLit` encodes Lit scalar values to bytes at compile time and embeds
them as `initialData` on the `malloc` step. The backend fills buffers via `writeBuffer` (WebGPU) /
`memcpy` (WASM) instead of dispatching a zero-input kernel. Malloc steps with `initialData` are
exempt from buffer recycling (always fresh allocation, tiny 2–8 byte buffers). Mega-module emits
inline `i32.store` / `i32.store16` instructions for the data.

**Multi-output kernel fusion:** Independent elementwise outputs sharing inputs and size are fused
into a single dispatch. Reduction epilogue chains (e.g., `matmul(x,w).add(bias).relu()`) fuse into 1
dispatch.

**Mega-module (WASM):** Compiles a `JitProgram`'s entire step list into a single WASM function,
eliminating all JS↔WASM boundary crossings. Extracted kernel functions enable parallel dispatch via
`WasmWorkerPool`. Rejects: `incref`, `scan`, `dus`, `scatter_add`, `assoc_scan`, `block_map`,
`workgroup_assoc_scan`, `reverse`, Routine steps, symbolic sizes. `fori_loop` IS supported —
compiled to native WASM loop.

**Orchestrator worker:** Moves mega-module execution off main thread via dedicated Web Worker.
Shares `WebAssembly.Memory` via `SharedArrayBuffer` for zero-copy. Browser main threads can't
spin-wait (`Atomics.wait` blocked) — detected at construction, falls back to direct execution.

**Command tape (WebGPU):** Pre-compiles a `JitProgram`'s step list into a flattened dispatch
sequence with pre-resolved pipelines, pre-computed buffer indices, and pre-built uniform bind
groups. Eliminates per-step JS overhead (scope lookups, array allocation, refcounting, pipeline
cache lookups) by replacing the generic step loop with a tight command-encoding loop over a flat
`GPUBuffer[]` table. ~4× reduction in JS-side overhead. Supports kernels, routines,
malloc/free/recycle, DUS, scatter_add, and reverse (`#encodeCopyAuto` handles aligned and unaligned
copies transparently). Supports scan (non-fallback paths: compiled-loop, preencoded-routine,
preencoded-multi-step — scan dispatchers create their own encoder within the tape's command buffer,
with transient buffers pooled via `#poolPush`). Rejects assoc_scan, block_map, incref.

**Effect system:** `MemoryEffect` enum (`Alloc`, `Borrow`, `Consume`, `Mutate`) on Jaxpr equations.
`effectDrivenAllocate` uses annotations for sound buffer recycling including DUS/ScatterAdd
zero-copy.

## Routine system

Routines are backend-specific ops (sort, cholesky, triangular_solve, LU, QR) that can't be fused:

- **CPU**: JavaScript TypedArray (debug)
- **WASM**: wasmblr runtime JIT (size-specialized, LRU cached)
- **WebGPU**: Hand-written WGSL (parallel algorithms)

Routines are opaque primitives — JVP rules define derivatives in terms of other primitives. Scan
modules use WASM imports to call routines from separate wasmblr modules.

**Analytical small-matrix fast paths:** `np.linalg.inv` has non-Routine Cramer's rule
implementations for n ≤ 4 (`inv2x2`, `inv3x3`, `inv4x4` in `numpy-linalg.ts`). These trace to
fusable Kernel ops, enabling DLM compose bodies to fuse into block-map shaders for m ≤ 4. For n ≥ 5,
inv falls through to LU (Routine) → fusion blocked. Cholesky (n ≤ 4), TriangularSolve (n ≤ 8), and
QR (n ≤ 8) also have analytical (jaxpr-traceable) paths gated by `inMakeJaxprBody()`, enabling sqrt
DLM variant fusion. Sort/Argsort/LU are non-jaxprifiable (comparison-based or data-dependent control
flow).

## Codegen architecture

**WASM:** `translateExpCore()` (shared ALU cases) → `emitKernelBody()` (gidx loop + reduction +
store) → `codegenWasm()` / `codegenWasmMulti()` (kernel codegen with SIMD fast path). `AluOp.Where`
uses cost-based branching (branchless select for cheap arms, true `if/else` for expensive ones).

**WebGPU:** `translateAluOpToWgsl()` → `gen()` (CSE + special cases) → `pipelineSource()` /
`pipelineSourceMulti()` (shader generation with optional shared-memory reductions).

Scan adds `codegenNativeScanGeneral()` (WASM) and `nativeScanMultiShaderSource()` (WebGPU).

## Polymorphic shapes (dynamic_axes)

`SymDim("T")` propagates through shape inference during tracing. Programs compiled for concrete
values but cached under symbolic keys. WASM scan compiled-loop supports polymorphic length (N as
runtime i32 param).

## Common pitfalls

- Forgetting `.dispose()` → silent memory leak
- Exporting a symbol from library but not `src/index.ts` → missing from published types
- Changing WebGPU shaders without browser tests → silent breakage
- `splitGraphDataflow` P2: `isNonKernelBlack` distinction matters — non-kernel blacks (Scan,
  Routine, DUS) are exempt from `maxArgs` check; kernel-endpoint blacks are not
- `splitGraphDataflow` P1 diamond relaxation: unary ops and binary-with-literal ops are exempt from
  the diamond heuristic (like view ops). They get duplicated into downstream kernels instead of
  being materialized as separate dispatches. `setDebug(1)` reports `cheapDiamonds=N`.
- DUS JIT step uses fiber loop for axis > 0 (`outerFibers` separate `copyBufferToBuffer` calls).
  Axis=0 fast path is a single contiguous copy.
- ScatterAdd vmap: batched indices not supported (shared indices only)
- Cross-device copy must use `dataSync()`/`data()` (not raw `readSync()`/`read()`) for
  non-contiguous arrays
- Mega-module: symbolic sizes silently encode as 0 via NaN coercion — guards exist
- `no-unnecessary-ref` autofix is unsafe for internal tracer `.ref` propagation (e.g.,
  `BatchTracer.ref` must call `this.val.ref`)
- Sequential `.data()` calls on WebGPU: each `mapAsync` costs ~12ms on eGPU. Use `tree.data()` /
  `tree.consumeData()` / `Promise.all()` to overlap readbacks

## Adding new primitives (checklist)

1. Declare in `Primitive` enum (`src/frontend/core.ts`)
2. Add tracing rule in `implRules`
3. Add JVP rule (`jvp.ts`) and transpose rule (`linearize.ts`)
4. Add vmap rule (`vmap.ts`)
5. If fusable elementwise, add ALU lowering in `jit.ts`
6. If needs dedicated kernel, register in `routinePrimitives` and implement backends
7. If copy-like (DUS), emit dedicated JitStep
8. Export from `src/index.ts`
9. **Transform tests** (hand-written JVP/transpose rules): JVP vs finite differences, grad vs FD

## Adding new routines (checklist)

1. `src/backend/wasm/routines/<name>.ts` — size-specialized wasmblr module
2. `src/backend/wasm/routines/index.ts` — export
3. `src/backend/wasm/routine-provider.ts` — add builder to map
4. `src/routine.ts` — add to `Routines` enum
5. `src/frontend/core.ts` — add to `routinePrimitives` map
6. `src/backend/wasm.ts` — dispatch case
7. `src/frontend/scan-plan.ts` — add to `supportedRoutines` in `tryPrepareWasmNativeScan()`
8. `src/backend/wasm.ts` — scan codegen in `codegenNativeScanGeneral()`
9. Optional: CPU fallback, JVP rule, transpose rule, WebGPU WGSL implementation

## Exports

All public symbols must be exported from `src/index.ts`. Key exports: `jit`, `grad`, `valueAndGrad`,
`jvp`, `vjp`, `vmap`, `jacfwd`, `jacrev`, `hessian`, `linearize`, `makeJaxpr`, `init`,
`defaultDevice`, `devicePut`, `blockUntilReady`, `scatterAdd`, `clearCaches`, `checkLeaks`, `numpy`,
`lax`, `nn`, `random`, `scipySpecial`, `scipyLinalg`, `tree`, `ScanPath`, `profileGpu`,
`profileGpuDetailed`, `setCodeCapture`, `CodeCaptureEntry`, `getWebGPUDevice`.

Internal-only exports (underscore prefix, not public API): `_lastConvClass`, `_lastConvRewritten`,
`_lastForiRewritten`.

## Observability & code capture

**`setCodeCapture(cb)`** — public API that registers a callback invoked on every compiled code unit
(WGSL shader or WASM module) with a discriminated `CodeCaptureEntry`. Fires on cache miss only.
Disabled by default (zero overhead). `setCodeCapture(null)` disables.

**wasmblr trace mode:** When capture is enabled, `CodeGenerator` records WAT mnemonics alongside
binary emission. `cg.toWat()` produces human-readable WAT source text. Off by default (zero cost).

**`profileGpuDetailed(fn)`** — expert profiling API returning per-pass GPU duration, dispatch grid
dimensions, workgroup size, shader label/hash, and pipeline cache hit/miss flags.

**Hook points:** WebGPU: `ShaderPipelineCache.prepare()` on cache miss. WASM: 14 sites across
kernels, mega-module, scan, assoc-scan, block-map, and routines (all on cache miss).

**REPL integration:** The website REPL has a "Compiled Code" panel showing captured WGSL/WAT grouped
by backend, with kind/label badges and metadata summaries. Enabled via "Capture compiled code"
checkbox.

## Commit checklist

1. Run `pnpm build && pnpm check && pnpm test`
2. No new failures beyond known `KNOWN_BUG` tests (currently none active)
3. Update `FEATURES.md` for user-visible changes
4. Export new public symbols from `src/index.ts`

---

# Part 2: Block Map & Tiled Compute

## `lax.block_map` — Pallas for WebGPU

A general-purpose shared-memory compute primitive. Tiles data into blocks, applies a body sub-jaxpr
per block using `var<workgroup>` memory, and reassembles results. The WebGPU equivalent of JAX's
Pallas / Triton's `tl.program_id` model.

```ts
const C = lax.block_map(
  ({ A: aTile, B: bTile }) => tileMatmul(aTile, bTile),
  { A, B },
  { blockShape: [64, 64], inAxes: { A: [0, null], B: [null, 1] }, outAxes: [0, 1] },
);
```

**Why it exists:** The JIT fuses elementwise chains but had no mechanism for tiled, shared-memory
GPU compute. Without `block_map`: no tiled matmul, no flash attention, no fused normalization, and
every new routine required hand-written WGSL. With `block_map`: the same TypeScript body that runs
in eager mode compiles to a fused shared-memory shader automatically.

### Execution model

| Mode                           | Strategy                                             |
| ------------------------------ | ---------------------------------------------------- |
| Eager                          | JS loop: slice → f(block) → concat                   |
| JIT + WebGPU (body fits shmem) | Fused WGSL shader, 1 workgroup per block, 1 dispatch |
| JIT + WebGPU (exceeds shmem)   | Per-block dispatch (M × body steps)                  |
| JIT + WASM                     | Sequential block loop in compiled WASM               |

### Key primitives

| Primitive              | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| `Primitive.BlockMap`   | The block_map node in the IR                      |
| `Primitive.BlockIndex` | Expose current block index inside body (internal) |
| `Primitive.ForiLoop`   | Traced for-loop (K-tile iteration in matmul)      |
| `Primitive.Reverse`    | Materialized axis reverse (polymorphic-safe)      |

`BlockIndex` is body-local and unmapped by vmap. `ForiLoop` supports symbolic `Dim` bounds (resolved
at runtime via `dimBindings`) and compiles to native WASM loops in mega-module.

### Architecture decisions

| Decision                                  | Rationale                                                                                                                                                                                                                        |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `blockShape: number[]` + `inAxes/outAxes` | Supports 1D (`assocScan`, reduction) and 2D (matmul, attention) from day one                                                                                                                                                     |
| Optional `gridShape` on `BlockMap`        | Enables gather (M blocks from N elements) where no mapped input determines it                                                                                                                                                    |
| `threadTile` parameter                    | Register tiling: 4×4 to 8×8 outputs/thread in `var<private>`, not shmem                                                                                                                                                          |
| Phase-based barrier scheduling            | Merges independent shmem writers → 2 barriers/K-tile instead of 4                                                                                                                                                                |
| `UncheckedDynamicSlice` (internal only)   | Omits `min/max` clamping when caller proves in-bounds via padding + bounds                                                                                                                                                       |
| `padConcrete` for tile-aligned inputs     | Real zero buffer → no ShapeTracker mask → no `select` overhead in shaders                                                                                                                                                        |
| `accessorGlobal` skip-when-true           | When ShapeTracker has no mask, `valid === true` → skip `select(0, x, true)`                                                                                                                                                      |
| Bank padding (+1 inner dim)               | Eliminates shared-memory bank conflicts for column-wise reads                                                                                                                                                                    |
| Manual ridx unrolling (not `@unroll`)     | Chrome/Tint `@unroll` is buggy (produces zeros); manual unroll works                                                                                                                                                             |
| Two-level fused cache                     | `Map<JitProgram, Map<specKey, Executable>>` — correct under `dynamic_axes`                                                                                                                                                       |
| `halo` option + zero-copy backends        | Input tiles extend by `[lo,hi]` per axis for stencil overlap; WebGPU: signed bounds checks in fused shader; WASM: halo-shifted clamped copy with interior fast path (skip `memory.fill`); JS fallback: per-block clamped slicing |

### Tiled matmul performance (RTX 4070 Ti SUPER, TB3 eGPU)

| Size | GFLOP/s | % of Peak (22,577) | vs `np.matmul` |
| ---- | ------- | ------------------ | -------------- |
| 2048 | 5,601   | 24.8%              | 33×            |
| 4096 | 12,138  | **53.7%**          | **33×**        |

At 4096×4096 the workload is at ~100% memory bandwidth utilization. Further improvement requires
subgroup matrix ops (hardware tensor cores).

### Shader optimization pipeline (all implemented)

```
Tracing → JIT Compile → Block-Map Analysis →
  Scalar Promotion [P0]     → size-1 kernels become `let` bindings
  Check Elision [P2]        → skip Where(true), unchecked DS, concrete pad
  Phase Scheduling [P1]     → group shmem writes → minimize barriers
  Index Simplification [O2] → contiguous shmem → flat/minimal indices
  Thread Mapping [O4]       → threadTile → cooperative vs private split
  Bank Padding [O11]        → +1 inner dim for conflict-free reads
  Manual Unrolling [O5]     → ridx loop body emitted N times with literal indices
  → Emit optimized WGSL
```

### Scan reductions in WebGPU compiled-loop

`nativeScanMultiShaderSource` uses `var<private>` internal arrays (not `var<workgroup>`) for
reduction intermediates inside `lax.scan` bodies. Reuses the `createWgslGen` + `ResolveGlobalIndex`
infrastructure from associative scan. Enables patterns like matmul-inside-scan-body to compile to a
single fused GPU dispatch per iteration.

### Reduction kernels in WebGPU compiled paths

Both `workgroup_assoc_scan` (Phases 1-3) and `block_map` Phase 4 support reduction kernels (e.g.,
matmul/dot product) in the body. The analysis phase rejects only symbolic sizes and multi-output
reductions; single-output reductions with concrete sizes are allowed.

**workgroup_assoc_scan** codegen uses `var<private>` internal arrays for reduction intermediates.
**block_map Phase 4** codegen has three paths: (1) workgroup tree reduction for kernel.size == 1,
(2) per-element reduction for kernel.size > 1 (gidx loop + ridx accumulation per thread, unrolled
for reSize ≤ 8), (3) elementwise gidx loop for multi-element kernels. This enables DLM (Kalman)
patterns with pytree matmul compose to use 1 fused dispatch instead of O(M-1) dispatches.

### Halo overlap for stencil computations

`block_map` supports a `halo` option specifying per-input, per-grid-axis overlap. Each input tile
extends beyond the output block range by `[lo, hi]` elements, enabling stencil patterns (conv, blur,
PDE) without materializing overlapping patches.

```ts
lax.blockMap(
  body,
  { image, kernel },
  {
    blockShape: [16, 16],
    inAxes: { image: [2, 3], kernel: [null, null] },
    outAxes: [2, 3],
    halo: {
      image: [
        [1, 1],
        [1, 1],
      ],
    },
  },
);
// image body tiles: [18, 18]; output tiles: [16, 16]
```

**Zero-copy backends:** WebGPU uses signed bounds checks in the fused shader (no extra allocation).
WASM uses compiled clamped-copy with an interior fast path that skips `memory.fill`. JS fallback
uses per-block clamped slicing. No pre-padding on any backend.

**Halo VJP (reverse-mode differentiation):** Two paths:

- **Stencil body synthesis** (fast): For linear stencil bodies composed of
  `{Shrink, Mul(Lit), Add}`, `analyzeLinearStencil` in `stencil-analysis.ts` synthesizes a
  gather-based backward body with reversed spatial offsets. Shape contract: `(B+bwdHalo) → B`
  (halo-expanded input, disjoint output). No atomics or scatter-add needed.
- **Pad+add fallback** (general): `transposeJaxpr` produces overlapping gradient patches that are
  padded and accumulated. Works for all body types including non-linear.

JVP threads `originalJaxpr` + `originalNumConsts` through `BlockMap` params so the transpose rule
analyzes the clean forward body, not the JVP-doubled body.

### Conv2d lowering pipeline

Conv2d is lowered through multiple paths depending on kernel shape and backend:

| Classification       | Trigger                         | Lowering                                      |
| -------------------- | ------------------------------- | --------------------------------------------- |
| `generic-dot`        | Default                         | `prepareConv` → reshaped views + `Dot`        |
| `fast-1x1-dot`       | 1×1 kernel                      | `prepareConv1x1` → moveaxis + reshape + `Dot` |
| `fast-1x1-block-map` | 1×1 kernel + block_map eligible | Block-map tiled 1×1                           |
| `block-map-3x3`      | 3×3 stride-1 SAME, spatial ≥ 16 | Halo block_map with VALID body                |
| `block-map-5x5`      | 5×5 stride-1 SAME, spatial ≥ 16 | Halo block_map with VALID body                |

`_lastConvClass()` reports which path was selected. `_lastConvRewritten()` reports whether the
Conv→BlockMap jaxpr rewrite fired.

**Conv→BlockMap rewrite** (`rewriteConvToBlockMap()` in `jit.ts`): Rewrites eligible 3×3/5×5 Conv
equations to `BlockMap` nodes BEFORE `splitGraphDataflow`. Guards: stride=1, SAME-equivalent
padding, spatial ≥ 16, concrete shapes, `vmapDims=0`. The body uses VALID conv (SAME guard prevents
recursion). Currently gated off on WebGPU (block_map falls back to per-block dispatch for conv
bodies; generic-dot is faster). WASM: 1.02–1.36× speedup growing with spatial size.

**Key finding from benchmarking:** For single-batch inference shapes (3×3, batch=1, 32–128ch,
32²–128²), WebGPU is **dispatch-bound at ~2.5ms** with <1% GPU utilization. Kernel quality is
irrelevant until tensor sizes are large enough to dominate dispatch cost. Im2col materialization was
attempted and rejected (41% regression from 150MB VRAM bandwidth overhead for 3×3).

**ForiLoop→BlockMap rewrite** (`rewriteForiLoopToBlockMap()` in `jit.ts`): Rewrites eligible
`ForiLoop` equations to `BlockMap` wrapping `ForiLoop` BEFORE `splitGraphDataflow`. **WebGPU only**
— WASM mega-module already compiles foriLoop natively. Guards: concrete bounds, ≥2 iterations,
retilable body (strict `pointwisePrimitives` whitelist check to prevent shape-preserving but
non-pointwise ops), all carries same concrete shape (rank ≥ 1, each dim ≥ 4), consts either
same-shape or scalar, shmem budget ≤ `backend.capabilities.maxComputeWorkgroupStorageSize` (exact
`byteWidth` tally of all inner `outBinders`). The inner ForiLoop operates on tile-sized data; on
WebGPU the fused shader emits a native WGSL `for` loop — 1 dispatch regardless of iteration count.
Block size: static limits 1D → `min(256, N)`, 2D → `[16,16]`/`[16,8]`/`[8,8]`.
`_lastForiRewritten()` tracks actual rewrite. Recursion prevented by `_skipForiRewrite` flag
(block_map nodes not currently schedulable via WebGPU Command Tape in nested contexts).

---

# Part 3: Scan & Associative Scan

## `lax.scan`

```ts
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs, options);
// f: (carry, x) => [newCarry, y]
```

**Options:** `length?`, `reverse?`, `acceptPath?` (ScanPath | ScanPath[]), `checkpoint?` (boolean |
number for √N grad checkpointing)

**xs=null** — provide `length` option, body receives null. **Y=null** — `[newCarry, null]` skips
output stacking.

### Scan paths

| Path                      | Description                                              | Backend      |
| ------------------------- | -------------------------------------------------------- | ------------ |
| `"compiled-loop"`         | Entire scan loop in native code (WASM module/GPU shader) | WASM, WebGPU |
| `"preencoded-routine"`    | Pre-encoded GPU dispatches with uniform offsets          | WebGPU only  |
| `"preencoded-multi-step"` | N×S dispatches in 1 `queue.submit()` for multi-step      | WebGPU only  |
| `"fallback"`              | JS loop calling body program per iteration               | All          |

Use `acceptPath: ["compiled-loop", "preencoded-routine", "preencoded-multi-step"]` in tests.

### Backend support

**WASM compiled-loop** handles: all body step types (kernels + routines), constants, reductions, any
numCarry/numY combination, internal buffer deps, reverse, polymorphic length.

**WebGPU compiled-loop** handles: kernel-only bodies with same-gidx deps (Phase 1 extended fusion),
carry passthrough, constants, reverse. Declarative `WebGPUYOutputSource` descriptors support any
numCarry/numY combination (carry-snapshot, carry-live, xs passthrough, internal). Carry outputs must
be either carry-input passthroughs or produced by execute steps — const/xs carry outputs are
rejected (fall through to preencoded-multi-step).

**WebGPU preencoded-multi-step** handles: cross-element deps, mixed kernel+routine bodies (non-Sort
routines). Phase 3 added Cholesky/TriSolve/LU support. Declarative `WebGPUCarryOutputSource`
classifies carry outputs into four types (internal, carry-passthrough, const, xs).

**Limitations:** Sort in scan body on WebGPU → fallback (uniform conflict). Mixed-dtype carries on
WebGPU → fallback. Preencoded-routine requires all Ys to be carry-live (Y[i] = carry[i]).
Preencoded-multi-step accepts all four Y source classes and all four carry source classes via shared
classifiers.

### Autodiff

- **JVP:** Doubled `lax.scan` — primals + tangents flow together
- **VJP/Grad:** JVP-transpose pattern. Forward stores √N checkpoint carries by default. Backward
  iterates reverse, recomputing from checkpoints.
- **Vmap:** Each batch element runs independent `lax.scan`

**Transform compositions:** `jit(grad(lax.scan))` ✅, `vmap(grad(lax.scan))` ✅,
`grad(vmap(lax.scan))` ✅. `grad(jit(lax.scan))` ❌ — use `jit(grad(lax.scan))` instead.

### Debugging scan paths

```ts
await lax.scan(f, init, xs, { acceptPath: [] });
// Error: Scan path debug: chose "compiled-loop"
```

`setDebug(1)` shows scan path selection reason.

## `lax.associativeScan`

```ts
const result = lax.associativeScan(fn, elems, { axis?, reverse? });
// fn: (a: T, b: T) => T  — must be associative
// result[i] = fn(result[i-1], elems[i])
```

Uses Kogge-Stone doubling: O(N log N) work, O(log N) depth. ceil(log₂ N) parallel rounds.

**Decoupled Fallback (P10, Phase 1 done):** For scalar associative ops (add/mul/min/max) on f32,
replaces Kogge-Stone with O(N) single-dispatch scan via atomic inter-workgroup communication with
bounded spin + work-stealing fallback (FPG-safe). Descriptor: 2-bit flag + 30-bit value packed in
single `atomic<u32>` (~4 ULPs f32 precision loss per lookback step). u32 and i32 excluded (30-bit
packing truncates values exceeding ±2^29). Detected automatically in `buildNativeAssocScanPlan`.
Phase 2 (general bodies) deferred.

**Backend behavior:**

| Backend    | Strategy                                                                       |
| ---------- | ------------------------------------------------------------------------------ |
| **WebGPU** | Fused shader per round — 1 GPU dispatch per round regardless of body           |
| **WASM**   | Compiled Kogge-Stone ladder in single WASM module. M7.3: parallel for N ≥ 4096 |
| **CPU**    | JS TypedArray ops per round                                                    |

**Autodiff:** `Primitive.AssociativeScan` with dedicated JVP (doubled inputs), PE, transpose
(reverse sequential scan), and vmap rules. `grad(associativeScan)` maintains O(log N) depth.

**Pytrees:** Supported. All leaves must have same scan-axis size. `fn` must dispose own
intermediates.

**Storage binding limits:** When pytree `associativeScan` with many leaves exceeds
`maxStorageBuffersPerShaderStage` (typically 10), two mitigations activate automatically: (1)
constants move to `var<uniform>` (freeing storage bindings), (2) leaf packing concatenates all
same-dtype leaves into 1 packed input + 1 packed output buffer (binding count: 2 + uniform).
Detected by `needsLeafPacking` flag in `scan-plan.ts`. Enables 6+ tuple pytree scans on WebGPU.

**einsum in scan bodies:** `np.einsum` subscripts work inside `associativeScan` and `lax.scan`
bodies — `einsumFastPath` in `numpy.ts` catches all common batch-matmul patterns (`"nij,njk->nik"`,
`"nji,njk->nik"`, etc.) and lowers to `matmul()`/`swapaxes()` before the general parser. The general
`parseEinsumExpression` still fails with rank-reduced inputs from per-element tracing.

### Shared blocked-data-movement primitives

Associative scan and block_map share three primitives for moving data between blocks:

| Primitive          | Purpose                                                               |
| ------------------ | --------------------------------------------------------------------- |
| `gatherAxisPoints` | Gather specific indices along an axis (Kogge-Stone shift-and-combine) |
| `copyAxisRange`    | Copy a contiguous slice from one array to another (carry writeback)   |
| `mapOverBlocks`    | Apply a sub-jaxpr to axis-aligned blocks (the shared iteration model) |

All three helpers accept an `axis: number` parameter. For `axis === 0`, they use a single-copy fast
path. For `axis > 0`, they operate fiber-by-fiber using generic stride math derived from
`(shape, axis, dtype)`: `outerSize = prod(shape[0:axis])`,
`innerBytes = prod(shape[axis+1:]) * elemBytes`, looping over `outerSize` fibers.

`mapOverBlocks` is used by both `associativeScan` WebGPU and `block_map` WebGPU codepaths. This
eliminates the former plan-stage–specific types (`KoggeStoneRound`, `BlockWriteback`, etc.) in favor
of reusable building blocks (-1761 LOC net deletion).

**Architecture rationale:** Self-similar plan recursion — the orchestrator (`runFusedPlan`) calls
the same primitives regardless of whether it was invoked by assocScan or block_map. This makes the
system composable: new compute patterns only need to express themselves in terms of existing
primitives.

### Axis-aware native paths

`jit(vmap(assocScan))` — vmap shifts the scan axis from 0 to 1+. Both backends handle this natively:

| Backend    | Strategy for axis > 0                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------- |
| **WebGPU** | Block-map path uses `inAxes`/`outAxes` set to `[[axis]]` — no extra copies                     |
| **WASM**   | Boundary transpose: strided-gather → contiguous temp → WASM Kogge-Stone → strided-scatter back |

**Transform compositions:** `vmap(grad(assocScan))` ✅ correct gradients. `grad(vmap(assocScan))`
runs without crash but produces incorrect gradient values (known limitation of reverse-mode
transpose interacting with vmapped associative scan).

---

# Part 4: Buffer Recycling & Pool

## JIT buffer recycling

`recycleBuffers()` replaces adjacent `free(a) → malloc(b)` pairs of the same byte size with
zero-cost `recycle(a → b)` steps — reusing the backend Slot with no allocation calls.

## WebGPU buffer pool

`Map<paddedSize, GPUBuffer[]>` — recently-freed GPUBuffers indexed by size. `malloc` checks pool
first; `decRef` returns to pool instead of destroying.

**Peak-memory guarantee:** Before each JIT execution, `configurePool()` evicts stale entries and
caps retained bytes at the program's peak live bytes.

**16-byte malloc floor:** All `GPUBuffer` allocations are padded to ≥16 bytes (4-byte aligned).
Uniform bindings require ≥16 bytes (vec4), but the floor applies to all allocations because
threading an `isUniform` flag through `malloc` would add complexity for negligible gain — only
scalar-promotion mallocs (2–8 bytes) are affected. Removing the floor requires either (a) a surgical
approach adding `isUniform` params to `malloc`/pool/arena/recycle/tape paths (~60–120 lines, spreads
complexity) or (b) a proper allocator refactor splitting `mallocStorage` vs `mallocUniform`
(~half-day, cleaner but larger scope). Keep the floor unless measured fragmentation harm justifies
the change.

**WASM comparison:** `WasmAllocator` uses free-list + reset-on-empty + top-of-heap compaction.
Zeroes on free-list reuse. WebGPU pool does NOT zero pooled buffers.

---

# Part 5: Performance Benchmarks

## Buffer pool + recycling (WebGPU, Intel Core Ultra 5 125H)

| Benchmark                        | Without | With    | Speedup  |
| -------------------------------- | ------- | ------- | -------- |
| jit chain x5 fused (4096)        | 10.5 µs | 1.7 µs  | **6.2×** |
| jit 2-output same-size (4096)    | 17.0 µs | 2.1 µs  | **8.1×** |
| jit 3-output same-size (4096)    | 23.6 µs | 2.7 µs  | **8.7×** |
| jit 2× matmul 32×32              | 17.9 µs | 2.6 µs  | **6.9×** |
| `lax.scan` cumsum N=100 size=64  | 4.5 ms  | 77.6 µs | **58×**  |
| `lax.scan` cumsum N=500 size=256 | 4.4 ms  | 88.1 µs | **50×**  |
| eager chain x5 (4096)            | 90.1 µs | 90.0 µs | ~1×      |

## WASM `lax.scan` throughput (L=1000, compiled-loop)

| Body Pattern               | Throughput    | Notes                     |
| -------------------------- | ------------- | ------------------------- |
| Cumsum (scalar)            | ~62M iter/sec | direct-write active       |
| Carry-only (4×4, Y=null)   | ~50M iter/sec | no Y output               |
| Elementwise (n=4, Y=carry) | ~78M iter/sec | direct-write active       |
| Passthrough Y (4×4)        | ~35M iter/sec | direct-write not eligible |

**`lax.scan` vs jit(loop):** Compiled-loop is 84–98% faster at all tested sizes (16×16 to 128×128).

**Direct-write optimization:** 40–65% speedup for small `lax.scan` bodies by eliminating
`memory.copy`.

## Associative scan `grad` speedup

_WASM backend:_

| N    | `grad(assocScan)` | `grad(scan)` | Speedup    |
| ---- | ----------------- | ------------ | ---------- |
| 64   | 0.021 ms          | 0.097 ms     | 4.6×       |
| 256  | 0.035 ms          | 0.448 ms     | 12.9×      |
| 1024 | 0.027 ms          | 1.037 ms     | 38.6×      |
| 4096 | 0.025 ms          | 4.757 ms     | **187.7×** |

_WebGPU backend (headless Chromium, Intel Core Ultra 5 125H):_

| N    | `grad(assocScan)` | `grad(scan)` | Speedup   |
| ---- | ----------------- | ------------ | --------- |
| 64   | 0.058 ms          | 0.360 ms     | 6.2×      |
| 256  | 0.076 ms          | 1.020 ms     | 13.5×     |
| 1024 | 0.143 ms          | 3.831 ms     | 26.9×     |
| 4096 | 0.194 ms          | 15.148 ms    | **78.2×** |

## DLM `associativeScan` throughput (2-tuple matmul compose, 2×2 matrices)

| Benchmark                      | WASM (Hz) | WebGPU (Hz) | WASM/WebGPU |
| ------------------------------ | --------- | ----------- | ----------- |
| assocScan 2-tuple N=200        | 52K       | 9.6K        | 5×          |
| assocScan 2-tuple N=500        | 23K       | 10.1K       | 2×          |
| assocScan 3-tuple Särkkä N=200 | 29K       | 11.8K       | 2×          |
| assocScan 2-tuple 4×4 N=200    | 9.4K      | —           | —           |
| `lax.scan` 2-tuple N=200       | 129K      | 7.6K        | 17×         |
| grad(assocScan) 2-tuple N=200  | 29K       | 3.0K        | 10×         |
| grad(`lax.scan`) 2-tuple N=200 | 51K       | —           | —           |

**Key insight:** Small-matrix DLM is latency-bound on WebGPU. WASM is 2–17× faster due to zero GPU
dispatch overhead. WebGPU `assocScan` uses a fused shared-memory shader with per-element reduction
codegen — Phase 4 block_map emits per-thread gidx loops with independent ridx accumulation for
bodies containing matmul (kernel.size > blockSize). This enables 1 dispatch instead of O(M-1) for
the prefix-apply phase. WASM `assocScan` uses inline typed load/store (v128, f32) instead of
`memory.copy` for small leaf sizes (≤32 bytes).

## Benchmark suite

```bash
pnpm build && pnpm vitest bench bench/<file>.bench.ts
```

| File                                | What it measures                                                        |
| ----------------------------------- | ----------------------------------------------------------------------- |
| `bench/argreduce.bench.ts`          | Argmin/argmax reduction performance                                     |
| `bench/assoc-scan-scaling.bench.ts` | Associative scan scaling with N                                         |
| `bench/associative-scan.bench.ts`   | `associativeScan` vs sequential `scan` for cumsum/cumprod               |
| `bench/block-map-matmul.bench.ts`   | Tiled matmul via `block_map` at various sizes                           |
| `bench/command-tape.bench.ts`       | Command tape vs step-by-step dispatch overhead                          |
| `bench/conv2d.bench.ts`             | Conv2d throughput across backends, kernel sizes, batch sizes            |
| `bench/dlm-scan.bench.ts`           | DLM (Kalman) scan: assocScan/scan/grad with pytree matmul compose       |
| `bench/matmul.bench.ts`             | Matrix multiplication throughput at various sizes                       |
| `bench/mega-module.bench.ts`        | Mega-module vs step-by-step: chains, multi-output, reduce, grad, matmul |
| `bench/parallel-wasm.bench.ts`      | Large elementwise baseline + polymorphic JIT                            |
| `bench/scan.bench.ts`               | `lax.scan` throughput across backends and body types                    |
| `bench/scatter-add.bench.ts`        | `scatterAdd` throughput at 1K/10K/100K elements                         |
| `bench/sort.bench.ts`               | Sorting performance across backends                                     |
| `bench/tiled-matmul.bench.ts`       | Tiled matmul parameter sweeps (block size, thread tile)                 |
| `bench/tts.bench.ts`                | Text-to-speech inference pipeline                                       |
| `bench/where-branching.bench.ts`    | WASM `AluOp.Where` branching vs branchless select                       |

Bench files import from `@hamk-uas/jax-js-nonconsuming` (public API via `dist/`), use
`DType.Float32` / `DType.Int32` enums.

## Future performance work

| ID  | Title                      | Priority        | Description                                                                         |
| --- | -------------------------- | --------------- | ----------------------------------------------------------------------------------- |
| P2  | Relaxed SIMD FMA           | Medium          | `f32x4.relaxed_madd` for 2× WASM dot-product throughput. Safari unsupported         |
| P3  | i64 in wasmblr             | Medium          | Native i64 (WASM MVP). Simplifies Threefry PRNG, unlocks f64 builtins               |
| P4  | Conv2d WebGPU fused shader | Medium          | Fused-shader codegen for conv bodies in `block-map.ts`. WASM path done (1.02–1.36×) |
| P6  | Benchmark validation       | Medium          | Systematic benchmarks: matmul GFLOP/s, conv2d, SIMD chains, reductions              |
| P7  | Cooperative matrix (WMMA)  | Blocked (~2026) | Hardware tensor cores for 2–4× tiled matmul. WGSL spec not yet stable               |

See `PLAN.md` for deferred items (O6 multi-reduction, O8d fori_loop tape, Decoupled Fallback
Phase 2) and non-goals.

---

# Part 6: Session Continuity

## Before starting work

1. `pnpm build` before running benchmarks (bench files import from `dist/`). Tests resolve to source
   via `vitest.config.ts` aliases and do NOT require a prior build.
2. `git branch` to confirm you're on the right branch
3. `git log --oneline -10` for recent context

## Key implementation patterns

- **JIT flow:** `makeJaxpr` → `flatten().simplify()` → `splitGraphDataflow()` → `jitCompile()` →
  `JitProgram.execute()`
- **Ownership debugging:** Check artifact disposal timing, `transposeJaxprCache` (cache-owned),
  `getOrMakeConstTracer` `.ref` balance, `evalJaxprTransposed` `argPrimals` set
- **Multi-output kernel access:** `kernel.outputs[0].exp`, `.reduction`, `.dtype`, `.bytes` — no
  single-output shims

## ESLint config structure

All `jax-js/*` rules: `warn` globally. `configs.invariance` overlay upgrades to `error` on `src/**`,
`packages/**`, `test/**`. `no-array-chain` is in `strict` only, not `invariance`. Internal transform
rules (`require-retained-release`, `require-try-finally-symmetry`,
`require-wrapper-dispose-symmetry`) are in the main config.

## Key architecture decisions

| Decision                                                    | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Non-consuming ownership model                               | Eliminates `UseAfterFreeError`; trades for silent leaks + linting                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Concrete compilation + symbolic cache                       | Simpler than full symbolic IR; ShapeTracker needs concrete strides                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `effectDrivenAllocate` over two-pass                        | Single-pass liveness; DUS zero-copy from `Mutate` effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Direct LU→triSolve gradient path                            | Fixing TriSolve JVP `triu(dA)` mask made Newton refinement unnecessary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `transposeJaxprCache` is cache-owned                        | Prevents repeated transposition; callers must NOT dispose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| WASM `(start, end, ...ptrs)` kernel signature               | Enables `WasmWorkerPool` work-splitting                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Mega-module extracted-functions design                      | V8 inlines direct `call` → perf-neutral serial, enables parallel                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Module Workers (`type: "module"`)                           | Required for Vitest browser mode and worker pool                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| SAB constructability over `crossOriginIsolated`             | Works in browsers with COOP/COEP headers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `jit()` identity dedup via WeakMap                          | Prevents cache bloat from inline `jit(fn)(args)` patterns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| DUS vmap → shrink+concat decomposition                      | JIT `dus` step is axis=0 only; vmap shifts axis, so decompose instead                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Phase 1 carry snapshot fusion                               | Same-gidx deps in single fused shader; avoids N×S dispatch overhead                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Phase 2 preencoded-multi-step                               | Cross-element deps as N×S dispatches in 1 submit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Phase 3 preencoded routine support                          | Non-Sort routines in preencoded-multi-step                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| GPU config factory (`gpu-config.ts`)                        | DRY NVIDIA/Intel configs; thin wrappers over shared launch args                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Self-similar plan recursion                                 | `runFusedPlan` uses same primitives for assocScan and block_map                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Decoupled Fallback for scalar ops + Kogge-Stone for general | DF: O(N) single dispatch for f32 add/mul/min/max. Kogge-Stone: general bodies, pytree, u32/i32, axis > 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `Primitive.Reverse` over flip/view                          | Materialized reverse is polymorphic-safe; views need concrete strides                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Shared blocked-data-movement primitives                     | `gatherAxisPoints`/`copyAxisRange`/`mapOverBlocks` replace bespoke types (-1761 LOC)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Register tiling (`threadTile`) over scalar                  | 4×4–8×8 outputs/thread in `var<private>` → 4× fewer shmem reads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Two-lane IR for block-map codegen                           | Correctness-by-construction: shmem writes vs private reads cleanly separate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Reduction kernels in `workgroup_assoc_scan`                 | Allows DLM matmul compose in fused shmem path (25 Hz → 6 kHz, 245× speedup)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Per-element reduction codegen in Phase 4 block_map          | gidx loop + ridx accumulation per thread; 1 dispatch vs O(M-1) for matmul bodies                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Inline typed copy for small WASM assocScan leaves           | v128/i32 load/store instead of `memory.copy` for ≤32-byte leaves (~9% faster)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Axis-aware DUS fiber loop                                   | `outerFibers` separate `copyBufferToBuffer` calls for axis > 0; axis=0 fast path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Axis-aware blocked-data-movement helpers                    | `gatherAxisPoints`/`copyAxisRange`/`mapOverBlocks` accept `axis` param; generic stride math                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| WASM assocScan boundary transpose for axis > 0              | Strided gather/scatter around contiguous WASM core; avoids modifying codegen                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| WebGPU assocScan axis-aware via inAxes/outAxes              | Block-map body always sees B at block dim; `inAxes`/`outAxes` map to source axis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `tree.data()`/`tree.consumeData()` parallel read            | Overlap `mapAsync` calls via `Promise.all`; 13.2× faster for 15 outputs on eGPU                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Scalar promotion (`pushLit` → `initialData`)                | Lit scalars encoded to bytes at compile time; `writeBuffer`/`memcpy` instead of kernel dispatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Analytical inv for n ≤ 4 (Cramer's rule)                    | Jaxpr-traceable: fuses in block-map. Routines break fusion. Pattern extends to cholesky/QR/trisolve                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| WebGPU command tape over step-by-step                       | Pre-resolved pipelines + flat buffer table eliminates ~76% of JS-side JIT loop overhead                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Bind group cache + colored arena (O9a-v2)                   | Conflict-graph coloring partitions intermediates into role-compatible slabs. Stable buffer identities → 100% cache hits                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Colored arena recycle exclusion                             | Recycle participants excluded from arena: buffer-identity inheritance causes cross-slab aliasing conflicts                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Constants slab (O9c) for initialData                        | All constants read-only → single slab safe. Eliminates per-invocation mapped buffer creation + writeBuffer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Colored multi-slab arena (O9a-v2)                           | Conflict graph from command tape → greedy coloring → 1 GPUBuffer/color. 256-byte alignment. MAX_COLORS=4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Targeted jaxprification over general                        | Cholesky (n≤4), TriSolve/QR (n≤8) traced to fusable ops. Sort/Argsort/LU are non-jaxprifiable                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Command tape GPU teardown on `clearCaches()`                | `_clearJitCompileCache` iterates programs, destroys uniform/constSlab/arena GPUBuffers before dropping refs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| DUS/scatter_add/reverse in command tape                     | Pre-encoded copies + pre-resolved scatter pipeline. `#encodeCopyAuto` handles aligned (native `copyBufferToBuffer`) and unaligned (WGSL copy shader) transparently                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Leaf packing for 6+ tuple assocScan                         | `needsLeafPacking` flag → shader emits `in_packed`/`out_packed` with compile-time offsets. Executor pack/unpack around dispatch. Binding count reduced from 2×numLeaves to 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Halo zero-copy backends                                     | WebGPU: signed bounds checks in fused shader; WASM: compiled clamped-copy with interior fast path (skip `memory.fill`); JS fallback: per-block clamped slicing. No pre-pad allocation on any backend                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Stencil body synthesis for halo-VJP                         | Forward bodies use Shrink (from `sliceInDim`), not UDS. `analyzeLinearStencil` walks body for {Shrink, Add, Mul(Lit)} targeting single halo input; `Add(stencilTerms, Lit)` allowed (additive bias drops out in adjoint). Gather-based VJP synthesizes reversed-stencil backward body via `makeJaxpr`; non-linear/unrecognized bodies fall back to pad+add. Multi-input guard: if any non-haloed tangent input is differentiated, falls through to pad+add. JVP threads `originalJaxpr`+`originalNumConsts` through BlockMap params so transpose analyzes clean forward body, not JVP-doubled body                                                       |
| Conv→BlockMap jaxpr rewrite (C.3)                           | `rewriteConvToBlockMap()` rewrites eligible 3×3/5×5 Conv to BlockMap BEFORE `splitGraphDataflow`. Guards: stride=1, SAME-equiv padding, spatial≥16, v=0, backend≠webgpu. Body uses VALID conv (SAME guard prevents recursion). `_lastConvRewritten()` tracks actual rewrite. 1.02–1.36× WASM speedup (grows with spatial size)                                                                                                                                                                                                                                                                                                                           |
| ForiLoop→BlockMap jaxpr rewrite                             | `rewriteForiLoopToBlockMap()` rewrites eligible ForiLoop to BlockMap wrapping ForiLoop BEFORE `splitGraphDataflow`. WebGPU only (WASM mega-module already compiles natively). Guards: concrete bounds, ≥2 iters, retilable body (strict `pointwisePrimitives` whitelist check), same concrete shape (rank≥1, dim≥4), consts same-shape or scalar, shmem ≤ `backend.capabilities.maxComputeWorkgroupStorageSize` (exact `byteWidth` tally). `_lastForiRewritten()` tracks rewrite. `_skipForiRewrite` blocks nested rewrites (block_map rejected by tape).                                                                                                |
| WebGPU-only tests in GPU-enforced suite                     | Tests requiring WebGPU excluded from default vitest config; run under gpu-test.sh where adapter is guaranteed. Pre-commit runs them automatically (NVIDIA on feature, both GPUs on main)                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Scan-in-command-tape (`TapeScan`)                           | Non-fallback scan paths (compiled-loop, preencoded-routine, preencoded-multi-step) execute within the tape's command buffer. Scan dispatchers receive a `tapeCtx` with shared encoder, deferred destroys, and `Set<GPUBuffer>` for pooled transient buffers. Scan buffer indices excluded from arena coloring via `buildConflictGraphAndColor`                                                                                                                                                                                                                                                                                                           |
| Declarative `WebGPUYOutputSource` for compiled-loop scan    | Hybrid Y output model. Computed Ys (carry-live, internal) are written directly inline by the steps that produce them (via `outputYIdxs`). Passthrough Ys (carry-snapshot, xs) are written in a dedicated writeback phase driven by source descriptors. Enables any numCarry/numY combination.                                                                                                                                                                                                                                                                                                                                                            |
| Shared `classifyWebGPUYSources` for all WebGPU scan paths   | All three WebGPU scan paths classify Y outputs via `classifyWebGPUYSources()` into four categories (carry-snapshot, carry-live, xs, internal) and store the result as `ySources: WebGPUYOutputSource[]` in the prepared struct. Executors dispatch Y stacking from source type, not raw JitIds. Compiled-loop: carry-live/internal inline via `outputYIdxs`, passthroughs via writeback phase. Preencoded-routine: carry-live only (validated by planner). Preencoded-multi-step: all four types, source-based buffer resolution (carry-live → carryWrite, carry-snapshot → carryRead, xs → xsBuffers with iteration offset, internal → scratch buffer). |
| Declarative `WebGPUCarryOutputSource` for multi-step scan   | Carry outputs in `preencoded-multi-step` are classified via `classifyWebGPUCarrySources()` into four categories (internal, carry-passthrough, const, xs). The carry-copy loop dispatches cleanly from source type instead of conditionally resolving raw JitIds per iteration.                                                                                                                                                                                                                                                                                                                                                                           |
