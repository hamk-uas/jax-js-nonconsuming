These notes help AI coding agents be immediately productive. The document has nine parts:

1. **Repository Overview** — General jax-js knowledge for any development work
2. **Scan Feature Reference** — `lax.scan` implementation details and backend-specific behavior
3. **Buffer Recycling & WebGPU Buffer Pool** — JIT `recycle` step and pool architecture
4. **Ownership Friction Points, Debugging & Future Work** — edge cases, debugging strategies
5. **Associative Scan** — `lax.associativeScan` Kogge-Stone parallel prefix scan
6. **Linear Algebra Autodiff** — `solve`, `inv`, TriangularSolve JVP fix
7. **Polymorphic Shapes** — `SymDim`, `dynamic_axes`, symbolic caching
8. **Completed Architecture & Future Performance Work** — M0–M8 milestone status
9. **Session Continuity Notes** — build/test workflow, context preservation

---

# The Agentic Evolution Protocol (AEP)

## 1. The Core Mandate: Asymptotic Velocity & Debt Collection

Your success is measured by the **net reduction in total system complexity**.

- **Ruthless Refactoring:** Scale is an asset. Refactor 1,000 files to achieve a cleaner
  architecture rather than patching the edges. Workarounds are not solutions — they are debt that
  compounds.

## 2. Provider Mode: Writing for the "Downstream Agent"

When you modify our public API, your task is to provide a "Migration Bundle" that includes:

- **Deterministic Mapping:** Raw structural diffs (e.g.,
  `Class.oldMethod(x) -> NewModule.fetch(x)`).
- **Workaround Signatures (The "Debt Collector"):** Explicitly identify patterns that consumers
  likely used to bypass old limitations. Provide migration guidance in the commit message. If you
  create a new release version in GitHub, include these instructions in the version description.
  - _Example:_ "If the consumer has a local utility that manually merges X and Y because v1 didn't
    support it, delete that utility and use the new `nativeMerge` endpoint."
- **Signal your compliance** If you were brought here by a failed command call, make sure you
  actually have read and followed the protocol. Only then call the command again: `AEP=1 COMMAND`.

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

## 5. Methodology

1. **The Plan:** For large tasks, generate a "Refactor Plan." List the "Workaround Signatures" you
   intend to hunt.
2. **The Ground Truth:** Run the native test suite. **Test Integrity is Absolute.**
3. **The Cleanup:** Remove all deprecated code paths, dead variables, and legacy comments. If you
   find a "todo" that the new API fixes, resolve it.

---

# Part 1: Repository Overview

## What is jax-js?

A TypeScript port of [JAX](https://github.com/google/jax) bringing **numerical computing and ML** to
the web: array ops, automatic differentiation, JIT compilation, and composable transformations
(`grad`, `vmap`, `jit`). Target: browser and Node.js/Deno environments.

**Key concepts:** **Tracing** records ops with tracer objects → Jaxpr IR → compiled kernels.
**Kernels** are compiled WGSL shaders (GPU) or WASM modules, with automatic elementwise fusion.
**Autodiff** via `grad(f)` traces `f` to build computation graph, applies chain rule automatically.

### Design philosophy

- **Generative compiler** — generates kernels from IR at runtime (vs. pre-compiled kernel libraries
  like TF.js/ONNX). New ops don't need hand-written kernels; fusion is automatic.
- **"80% of XLA"** — targets 3–5× of optimal, not peak performance.
- **Lightweight** — small composable primitive set over large surface area.
- **Explicit disposal** — GPU/WASM buffers freed via `.dispose()` or `using`. Silent leaks over
  noisy crashes — move semantics crash immediately (`UseAfterFreeError`) when you forget `.ref`,
  pointing at the bug. The non-consuming model never crashes from reuse, but a forgotten
  `.dispose()` leaks GPU memory silently. `checkLeaks` and the ESLint plugin compensate, but they're
  opt-in. Method chains (`a.mul(b).add(c)`) are particularly dangerous — every intermediate
  allocates a GPU buffer nobody frees in eager mode. The bet: silent leaks + tooling >
  `UseAfterFreeError` + `.ref` boilerplate.
- **Ownership-correct in both modes** — code must work correctly in eager AND jit mode. `jit()` is
  purely a performance optimization. If code leaks or double-disposes in eager mode, that is a real
  bug — not something to paper over by wrapping in `jit()`. See Part 4: Ownership Correctness
  Principle.

### Ownership invariants (maintainer rubric)

1. **Conserve handles** — every handle has exactly one terminal path: transfer, dispose, or retained
   ownership.
2. **Make retention boundaries explicit** — if a value outlives scope, retain intentionally.
3. **Release retained handles symmetrically** — including error paths (`try/finally`).

### Development priorities

1. **Correctness** — tests, ref-counting, cross-backend consistency
2. **API breadth** — NumPy/JAX compatibility (see `FEATURES.md`)
3. **Performance** — WASM SIMD, transformer inference, conv2d tuning
4. **Demos** — fluid sims, neural nets, audio, embedding search, fractals

## Architecture

- **Core** (`src/`): `frontend/` (array, jit, jvp, linearize, vmap, convolution), `library/` (numpy,
  lax, nn, random, scipy-special, numpy-linalg, numpy-fft, lax-scan, lax-associative-scan)
- **Backends** (`src/backend/`): cpu, wasm (+`wasm/`), webgl (+`webgl/`), webgpu (+`webgpu/`)
- **Aux packages**: `packages/loaders` (safetensors, OPFS, tokenizers), `packages/onnx`,
  `packages/optax`
- **Website**: `website/` — live demos doubling as integration tests

## Developer workflows

```bash
pnpm install                       # pnpm ≥ 10
pnpm run build                     # tsdown → dist/
pnpm exec playwright install       # one-time: Chromium for WebGPU tests
pnpm test                          # Vitest + Playwright
pnpm run test:policy:strict        # strict: zero failures
pnpm run test:arch                 # architectural: failures must match .ci/expected-failures.json
pnpm run test:all                  # Vitest + Deno WebGPU
pnpm run test:deno                 # Deno WebGPU only
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm vitest bench bench/<file>     # benchmarks
pnpm -C website dev                # local dev server
```

**Pre-commit:** Husky runs lint-staged + full Vitest + Deno tests. Before commit, run:
`pnpm build && pnpm check && pnpm test && pnpm run test:deno`.

**Docs-only fast path:** When all staged files are `.md`/`.txt`/`docs/`, pre-commit skips
build/tests — runs only `format:check` + `lint`.

### Debug logging

Use `setDebug(level)` (NOT env vars): 0=off, 1=JIT logs, 2=shader code, 3=expressions, 4=programs,
5=verbose.

### Temporary files

Use `tmp/` (gitignored) for scratch files. Avoid `/tmp/`.

### Editing Prettier-managed files

Run `npx prettier --write <file>` before reading for edits to avoid stale match text.

## Memory management & ownership

Operations **do not consume** inputs. Arrays stay alive until `.dispose()`'d. No `.ref` needed in
user code (non-consuming fork).

> **Upstream note:** The upstream jax-js uses **move semantics** — every operation consumes its
> inputs (rc−1), reusing an array requires `.ref` (rc+1). This fork replaces that with a
> non-consuming model: operations leave inputs alive, `.ref` is never needed in user code. If you
> encounter `.ref` patterns in upstream code or git history, they are not needed here.

```ts
// Arrays reused freely — no .ref needed
const result = x.add(x.mul(y));
arr.dispose(); // explicit disposal when done
```

**`using` declarations** work via `[Symbol.dispose]()` on Arrays and jit functions:

```ts
{
  using x = np.array([1, 2, 3]); /* auto-disposed at block end */
}
{
  using f = jit((x) => x.mul(x).sum()); /* consts freed at block end */
}
```

Use `using` for short-lived locals; `.dispose()` for returned/stored values. `using` works inside
`jit()` bodies.

### Memory lifecycle

Slot = internal handle to backend memory (WASM pointer / GPU buffer). Array creation → Slot with
rc=1. `.dispose()` decrements rc; at 0, backend frees. `evalJaxpr` auto-disposes **intermediates**
(equation outputs) at last use but never consumes inputs — callers own their inputs and must dispose
temporaries they create (e.g., `zerosInternal` placeholders). Lit-created arrays inside `evalJaxpr`
are tracked and disposed automatically. Pass-through outputs (where an output IS an input array) get
`.ref`'d so callers can safely dispose inputs without killing outputs. `jitCompile` emits explicit
`malloc`/`free`/`recycle` steps.

### Ownership by layer

| Layer                         | Consumes?                          | How disposal works                                                                                                   |
| ----------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `EvalTrace.processPrimitive`  | No                                 | Delegates to impl rules                                                                                              |
| `JaxprTrace.processPrimitive` | No                                 | Builds graph, never disposes tracers                                                                                 |
| `evalJaxpr`                   | No (inputs) / Auto (intermediates) | Intermediates disposed at last use; inputs protected; Lit arrays tracked and disposed; pass-through outputs `.ref`'d |
| `jitCompile`/`JitProgram`     | Auto                               | Emits malloc/free/recycle steps                                                                                      |
| `.dispose()`                  | Manual                             | Decrements rc, frees at 0                                                                                            |

### Backend comparison

| Aspect     | WASM                                      | WebGPU                           |
| ---------- | ----------------------------------------- | -------------------------------- |
| Allocation | `WasmAllocator` over `WebAssembly.Memory` | `device.createBuffer()`          |
| Sync read  | Direct memory view                        | `SyncReader` with staging buffer |
| Dispatch   | Wasm module `kernel(start, end, ...ptrs)` | `dispatchWorkgroups()`           |
| Float64    | ✅ Full support (Kahan summation)         | ❌ No f64 in WGSL                |
| Parallel   | `WasmWorkerPool` (SharedArrayBuffer)      | Hardware workgroup parallelism   |

## WebGPU Backend

**No global barrier** — threads in different workgroups cannot sync within a dispatch. This shapes
all algorithm choices.

### Hard limits and how jax-js handles them

| Limit                              | Typical Value | Impact on jax-js                                                                |
| ---------------------------------- | ------------- | ------------------------------------------------------------------------------- |
| `maxStorageBuffersPerShaderStage`  | 8-10          | Limits kernel inputs; `splitGraphDataflow` P2 prevents overflow                 |
| `maxComputeWorkgroupsPerDimension` | 65535         | Large arrays need 2D grid splitting via `calculateGrid()`                       |
| `maxComputeWorkgroupSizeX`         | 256           | Limits threads per workgroup (Sort workgroup size)                              |
| `minUniformBufferOffsetAlignment`  | 256 bytes     | Dynamic uniform offsets must be 256-byte aligned                                |
| `minStorageBufferOffsetAlignment`  | 256 bytes     | Can't use buffer offsets for arbitrary strides → uniform-based offsets for scan |

**Storage buffer limit handling:** `splitGraphDataflow()` P2 pass counts transitive fused
dependencies for every kernel-dispatched equation and backtracks (splitting the fusion boundary)
when `depCounter.size > maxArgs`. This applies to all kernel-dispatched equations including
kernel-endpoint blacks (outputs, multi-use vars) — not just white (fusable) ops. Non-kernel blacks
(Scan, Routines, DUS) are exempt since they use dedicated JIT step types. A safety-net `throw` in
`webgpu.ts` catches any equation that slips through.

**Grid size handling:** `calculateGrid()` in `codegen.ts` splits into 2D grid when size > 65535.
Shader reconstructs linear index via `global_invocation_id.x + global_invocation_id.y * 65535u`.

### Features exploited

| Feature                     | How jax-js uses it                                               | Location                            |
| --------------------------- | ---------------------------------------------------------------- | ----------------------------------- |
| **shader-f16**              | Float16 dtype support                                            | `src/backend.ts` feature requests   |
| **Workgroup shared memory** | Sort local exchanges; JIT cooperative reductions                 | routines.ts, webgpu.ts              |
| **workgroupBarrier()**      | Sort workgroups; JIT shared-memory reduction tree                | routines.ts, webgpu.ts              |
| **storageBarrier()**        | Memory fence for shared variable consistency                     | Sort, Cholesky, LU in routines.ts   |
| **Pipeline caching**        | Compiled pipelines stored by shader hash                         | `pipelineCache` in webgpu.ts        |
| **Pipeline layout caching** | `GPUPipelineLayout` cached by `numInputs:numOutputs:hasUniform`  | `ShaderPipelineCache` in webgpu.ts  |
| **Command batching**        | Multiple dispatches encoded before single `queue.submit()`       | `PendingExecute` in webgpu.ts       |
| **WGSL copy shader**        | Byte-level buffer copy when `copyBufferToBuffer` alignment fails | `COPY_SHADER_CODE` in webgpu.ts     |
| **shader-f32-atomic-add**   | Native f32 `atomicAdd` in scatter-add shader (when available)    | `dispatchScatterAdd()` in webgpu.ts |

**Scan additionally uses:** Ping-pong buffers (carry alternates between two buffers) in
`dispatchPreencodedScan()`, uniform buffers for per-iteration offsets in `scan-wrapper.ts`.

**Pipeline caching detail:** Pipelines cached by shader source hash → avoids recompiling identical
shaders (common with JIT-generated kernels).

**Synchronous readback:** `SyncReader` (`src/backend/webgpu/reader.ts`) uses offscreen canvas with
webgpu context (borrowed from TensorFlow.js) for `.dataSync()`. Prefer `.data()` (async) for
performance.

### Features NOT exploited (opportunities)

| Feature                   | What it enables                                     | Why not used yet                         |
| ------------------------- | --------------------------------------------------- | ---------------------------------------- |
| **Subgroups**             | SIMD-width operations (shuffle, reduce within wave) | Requires `subgroups` feature; not stable |
| **Indirect dispatch**     | GPU-driven workgroup counts                         | No dynamic control flow needs it yet     |
| **Texture sampling**      | Hardware-accelerated interpolation                  | All ops use storage buffers currently    |
| **Tiled matrix multiply** | Shared memory blocking for large matmuls            | Matmul uses simple row×col accumulation  |
| **Atomic operations**     | Lock-free reductions, histograms                    | Reductions done via shader accumulation  |
| **timestamp-query**       | GPU-side profiling                                  | Requested but not wired up for profiling |
| **Render pipelines**      | Visualization without readback                      | Would need separate rendering path       |

**Subgroups opportunity:**

Subgroups enable operations like `subgroupAdd()` that sum across a SIMD lane (typically 32-64
threads) without explicit barriers. This would accelerate reductions significantly:

```wgsl
// Current (sequential accumulation):
var acc = 0.0;
for (var i = 0u; i < size; i++) { acc += data[i]; }

// With subgroups (parallel within wave):
let partial = subgroupAdd(data[local_id]);
if (subgroup_invocation_id == 0) { atomicAdd(&result, partial); }
```

**Tiled matmul opportunity:**

Current matmul computes `C[i,j] = sum(A[i,:] * B[:,j])` with each thread doing a full dot product.
Tiled matmul loads tiles of A and B into shared memory, enabling data reuse:

```wgsl
// Tiled approach (not implemented):
var<workgroup> tileA: array<f32, 16*16>;
var<workgroup> tileB: array<f32, 16*16>;
// Load tiles collaboratively, compute partial products, accumulate
```

This is a standard GPU optimization that could provide 5-10× speedup for large matrices.

### WebGPU-specific scan constraints

The "no global barrier" limitation creates scan-specific constraints:

| Constraint                        | Why it exists                                                | Consequence                   |
| --------------------------------- | ------------------------------------------------------------ | ----------------------------- |
| Per-element independence required | No cross-workgroup sync between iterations                   | Complex bodies → JS fallback  |
| numCarry ≠ numY unsupported       | compiled-loop shader assumes 1:1 carry↔output mapping       | Falls back to JS loop         |
| Internal buffer deps unsupported  | Shader can't allocate scratch temporaries between statements | Mandelbrot pattern → fallback |
| Sort in scan body                 | Sort already uses uniforms (conflict with scan offsets)      | Falls back to JS loop         |

WASM backend handles all these cases because it can allocate temporaries and has true sequential
control flow. WebGPU is more restricted but faster when patterns fit.

### Key WebGPU files

| File                                 | Purpose                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------- |
| `src/backend.ts`                     | WebGPU init, adapter/device creation, feature reqs                      |
| `src/backend/webgpu.ts`              | Main backend: kernels, scan, command encoding                           |
| `src/backend/webgpu/codegen.ts`      | `calculateGrid()`, WGSL helpers, `ShaderInfo`                           |
| `src/backend/webgpu/routines.ts`     | Bitonic sort, Cholesky, LU, TriangularSolve WGSL                        |
| `src/backend/webgpu/scan-wrapper.ts` | `wrapRoutineForScan` — transforms routine shaders for scan offsets      |
| `src/backend/webgpu/reader.ts`       | `SyncReader` for synchronous buffer readback                            |
| `src/backend/webgpu/builtins.ts`     | Shader snippets for special functions (erf, etc.)                       |
| `src/tuner.ts`                       | Kernel tuner: axis splitting, unrolling, SIMD, groups                   |
| `src/frontend/scan-plan.ts`          | `ScanPlan` type, `planScan()`, path selection, `tryPrepareNativeScan()` |
| `src/frontend/scan-executor.ts`      | `executeScan()`, `copySliceToBuffer`, fallback loop                     |
| `src/frontend/jit.ts`                | `"scan"` JitStep handling, `"assoc_scan"` inline execution (~120 LOC)   |

## Codegen architecture

**WASM:** `translateExpCore()` (shared ALU), `emitKernelBody()` (shared gidx loop),
`codegenWasm()`/`codegenWasmMulti()` (kernel codegen with SIMD fast path for f32). `AluOp.Where`
uses cost-based branching: branchless `select` for cheap arms, true `if/else` for expensive ones
(cost ≥ 15).

**WebGPU:** `translateAluOpToWgsl()`, `gen()` with CSE, `createShaderEmitter()`.

### Scan-specific codegen functions

**WASM scan codegen:**

| Function                               | Role                                                |
| -------------------------------------- | --------------------------------------------------- |
| `translateExpWithGeneralScanContext()` | Wrapper with const/carry/xs/internal classification |
| `codegenNativeScanGeneral()`           | Full scan loop codegen with direct-write analysis   |

**WebGPU scan codegen:**

| Function                        | Role                                          |
| ------------------------------- | --------------------------------------------- |
| `genScanExpressionWithRidx`     | Scan-specific GlobalIndex + inline generation |
| `nativeScanMultiShaderSource()` | Full scan shader (single and multi-kernel)    |

### Shared kernel body: `emitKernelBody()`

The inner per-element loop is shared between `codegenWasm()` and `codegenNativeScanGeneral()` via
`emitKernelBody()`. Callers inject behavior through three callbacks:

| Callback         | `codegenWasm` provides             | `codegenNativeScanGeneral` provides                    |
| ---------------- | ---------------------------------- | ------------------------------------------------------ |
| `emitOutputAddr` | `local.get(outputArg) + gidx * bw` | Direct-write: `carryOut[c]`; else: `internal[idx]`     |
| `emitExp`        | `translateExp(exp, {gidx, ridx})`  | `translateExpWithGeneralScanContext(exp, scanCtx)`     |
| `emitStore`      | Simple typed store                 | Dual-store: primary + ysStacked (for direct-write + Y) |

## Tuner system (`src/tuner.ts`)

Transforms `Kernel` → `TuneResult` controlling dispatch. Splits axes:
`[global | local | groups | reduce | unroll | upcast]`. WebGPU `groups > 1` emits shared-memory
cooperative reductions.

## Routine system

Backend-specific ops (sort, cholesky, LU, triangular solve) in 3 implementations: CPU (JS), WASM
(wasmblr runtime codegen), WebGPU (hand-written WGSL). Routines are opaque primitives — JVP rules
define derivatives in terms of other primitives.

**wasmblr:** Custom WASM bytecode assembler with high-level helpers (`forLoop`, `load`, `store`,
SIMD `f32x4`/`f64x2`). Size-specialized, LRU-cached (64 entries).

**Adding a new routine:** Add to `Primitive` enum → `routinePrimitives` → WASM builder → dispatch
case → scan support → optional JVP/transpose rules → tests (must verify JVP + grad vs finite
differences for hand-written rules).

## JIT compiler internals

**Pipeline:** Tracing → `makeJaxpr` → `simplify()` → `splitGraphDataflow()` → kernel fusion →
`jitCompile()` → `effectDrivenAllocate()` (single-pass liveness, recycle) → `JitProgram.execute()`.

**JitStep types:** `execute`, `malloc`, `incref`, `free`, `recycle`, `scan`, `dus`, `scatter_add`,
`assoc_scan`.

**Multi-output fusion:** Independent elementwise outputs sharing inputs+size → single
`Kernel.multi()` dispatch.

**Epilogue fusion:** Elementwise chains downstream of reductions fused into the reduction kernel
(e.g., `matmul(x,w).add(bias).relu()` → 1 dispatch).

**Effect system:** `MemoryEffect` enum (`Alloc`, `Borrow`, `Consume`, `Mutate`). DUS and ScatterAdd
use `Mutate` on first input → `effectDrivenAllocate` enables zero-copy recycling.

## Mega-Module (WASM JIT Fusion)

Compiles entire JitProgram step list into a single WASM function — zero JS↔WASM boundary crossings
between kernels. Rejects: `incref`, `scan`, `dus`, `scatter_add`, `assoc_scan`, Routine steps,
symbolic sizes, pass-through outputs.

**M6.2c:** All kernels (including reductions) extracted as separate `kernel_N` functions.
`MegaStepInfo[]` metadata enables JS-driven parallel execution via `WasmWorkerPool` (threshold ≥
4096 elements).

**Orchestrator (M6.2b):** Moves mega-module execution off main thread via dedicated Worker. Shares
`WebAssembly.Memory` (zero-copy). Spin-wait sync → deadlocks on browser main threads (detected via
`Atomics.wait` probe; falls back to direct execution).

## Scatter-add primitive

`scatterAdd(target, indices, updates, axis)` — accumulates updates at index positions. WebGPU uses
native `atomicAdd` or CAS fallback. WASM uses size-specialized wasmblr module. Linear in
target+updates (JVP rule). Transpose: ∂target=identity, ∂updates=gather. Vmap: shared indices only.

## Common pitfalls

- Forgetting `.dispose()` → silent GPU memory leak
- Not exporting from `src/index.ts` → missing from published types
- **Tests import from `dist/`** — source edits invisible until `pnpm build`
- CPU backend: collect both `AluOp.GlobalIndex` AND `AluOp.GlobalView`
- **DUS JIT axis=0 only** — vmap decomposes axis≠0 into shrink+concat
- **ScatterAdd vmap: batched indices not supported** — throws for `iBdim !== null`
- JIT: flush pending ops before scan step
- Cross-device copy must use `dataSync()`/`data()` (not raw `readSync`/`read`) — ShapeTracker
  ignored otherwise
- `splitGraphDataflow` P2: distinguish non-kernel blacks (Scan, Routine, DUS) from kernel-endpoint
  blacks (outputs, multi-use vars)
- Mega-module: `canCompileToMegaModule` must reject symbolic sizes; `i32.const()` has runtime guard
- `no-unnecessary-ref` autofix unsafe for internal tracer `.ref` propagation (BatchTracer incident)
  TODO: check if this information is stale.
- **`evalJaxpr` is non-consuming** — callers must dispose temporaries (zerosInternal, xSlices) after
  calling `evalJaxpr`. Do NOT `.ref` inputs before passing to `evalJaxpr`; that's a move-semantics
  pattern that causes leaks.

## Deno WebGPU test guidelines

- **Reuse jax-js's device** — never create a second `GPUDevice`. Never `device.destroy()`.
- **Import from `dist/`** not `src/`.
- Track and destroy `GPUBuffer`s in `finally` blocks.
- Use `withLeakCheck` from harness for leak detection.
- Run each file separately: `pnpm run test:deno` (not `deno test test/deno/`).

## Exports & public API

All public symbols exported from `src/index.ts`: `jit`, `grad`, `valueAndGrad`, `jvp`, `vjp`,
`vmap`, `jacfwd`, `jacrev`, `hessian`, `linearize`, `makeJaxpr`, `init`, `defaultDevice`,
`devicePut`, `blockUntilReady`, `scatterAdd`, `clearCaches`, `checkLeaks`, `numpy`, `lax`, `nn`,
`random`, `scipySpecial`, `scipyLinalg`, `tree`, `ScanPath`, `AssociativeScanOptions`.

## Commit checklist

1. Run pre-commit CI checks
2. Ensure pre-commit hook installed (`pnpm prepare`)
3. Run full test suite (`pnpm vitest run`)
4. Update docs for new features/APIs
5. Add `.dispose()` tests for new behavior
6. Export new public symbols from `src/index.ts`
7. Update `FEATURES.md` for user-visible changes

---

# Part 2: Scan Feature Reference

## Overview

`lax.scan` applies a function over the leading axis of arrays, threading carry state.

```ts
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs, options);
// f: (carry, x) => [newCarry, y]
```

**Options:** `length?`, `reverse?`, `acceptPath?: ScanPath | ScanPath[]`,
`checkpoint?: boolean | number`, `isJvpTransformed?` (internal).

**Scan paths (`ScanPath`):** `"compiled-loop"` (native code), `"preencoded-routine"` (WebGPU uniform
offsets), `"fallback"` (JS loop).

**Extensions:** `xs=null` (carry-only, requires `length`), `Y=null` (skip output stacking).

**Key files:** `src/library/lax-scan.ts` (API), `src/frontend/scan-plan.ts` (planning),
`src/frontend/scan-executor.ts` (execution), `src/backend/wasm.ts` (WASM compiled-loop),
`src/backend/webgpu.ts` (WebGPU paths), `src/backend/webgpu/scan-wrapper.ts` (routine scan).

### Scan API contract

Inputs NOT consumed. Caller owns outputs. Body function needs no `.ref`. Common patterns:
`[newCarry, y]`, `[newCarry, newCarry]` (passthrough), pytree carry, `xs=null`, `Y=null`.

### Backend status

| Backend | compiled-loop   | preencoded-routine | Notes                                                  |
| ------- | --------------- | ------------------ | ------------------------------------------------------ |
| CPU     | N/A             | N/A                | JS fallback, all tests pass                            |
| WASM    | ✅ All patterns | N/A                | Routines via imports, 50-80M iter/sec                  |
| WebGPU  | ✅ Kernel-only  | ✅ Single routine  | numCarry=numY required; mixed/multi-routine → fallback |
| WebGL   | N/A             | N/A                | Untested, uses fallback                                |

### Execution flow

```
lax.scan() → trace body → bodyJaxpr → planScan() → ScanPlan → executeScan()
  ├─ compiled-loop: entire loop compiled native
  ├─ preencoded-routine: pre-encoded GPU dispatches with uniform offsets
  └─ fallback: JS loop calling bodyProgram.execute()
```

Both JIT and non-JIT paths use the same `planScan()` + `executeScan()` flow. The body is always
JIT-compiled.

### Body composition types

Scan bodies are classified by what operations they contain:

| Body Type                | Description                        | Example                          |
| ------------------------ | ---------------------------------- | -------------------------------- |
| **kernel-only**          | Only elementwise/reduction kernels | `carry + x`, `sum(x)`            |
| **routine body**         | Single routine operation           | `cholesky(x)`, `sort(x)`         |
| **mixed kernel+routine** | Both kernels and routines          | `scale * x` then `cholesky(...)` |

**Execution path by body type and backend:**

| Body Type               | WASM          | WebGPU                           |
| ----------------------- | ------------- | -------------------------------- |
| kernel-only (simple)    | compiled-loop | compiled-loop                    |
| kernel-only (with deps) | compiled-loop | **fallback**                     |
| routine body (single)   | compiled-loop | preencoded-routine (or fallback) |
| mixed kernel+routine    | compiled-loop | **fallback**                     |
| multiple routines       | compiled-loop | **fallback**                     |

**Internal buffer dependencies:** When one kernel step reads from another step's output within the
same body. WASM handles this by allocating temporary buffers. WebGPU's shader codegen doesn't
support it yet — Phase 1 of the scan plan addresses this via deferred-write pattern.

```ts
// Body with internal deps (WebGPU falls back):
const body = (carry, x) => {
  const Asq = carry.A.mul(carry.A); // Step 1: produces Asq
  const newA = Asq.sub(carry.B); // Step 2: reads Asq (internal dep!)
  return [{ A: newA, B: carry.B }, newA];
};
```

**Carry passthrough:** When an output carry directly references the input carry without a kernel
producing it. WebGPU multi-kernel scan requires every carry to be produced by a kernel step.

```ts
// Carry passthrough (WebGPU multi-kernel falls back):
const body = (carry, x) => {
  const newA = carry.A.add(x);
  return [{ A: newA, B: carry.B }, newA]; // B is passthrough!
};
```

**Why `triangularSolve` creates mixed bodies:** The high-level API handles `leftSide`/`lower` via
transpose/flip operations (kernels) around the routine, creating 3+ steps. WebGPU falls back; WASM
handles it.

### Compiled-loop routing

`tryPrepareNativeScan()` routes to backend-specific implementations:

- **WebGPU kernel-only** → `tryPrepareWebGPUNativeScan()` → `prepareNativeScanMulti()`
- **WebGPU routine body** → `tryPreparePreencodedScan()` → `preparePreencodedScan()`
- **WASM (kernels + routines)** → `tryPrepareWasmNativeScan()` → `prepareNativeScanGeneral()`

**Dynamic routine planning:** `getScanRoutineInfo(routineName, routine)` queries backend
capabilities. Returns `ScanRoutineInfo` if eligible; `null` triggers fallback.

### Compiled-loop eligibility

**WASM:** All steps are Kernels or supported Routines (Cholesky, Sort, TriangularSolve, LU,
Argsort). Any numCarry/numY combination. Internal buffer deps supported.

**WebGPU single-kernel:** 1 Kernel, numCarry=1, numY=1. **WebGPU multi-kernel:** Multiple Kernels,
numCarry=numY, no internal buffer deps, no carry passthrough.

**WebGPU preencoded-routine:** Exactly 1 Routine, numCarry=numY, routine doesn't use uniforms.

### WASM compiled-loop details

Function signature:
`(length: i32, ...consts, ...carryIn, ...xs, ...carryOut, ...ysStacked, ...internals, aux?)`. Length
is first i32 param (polymorphic). Uses `memory.copy` for carry/output transfers.

**Direct-write optimization:** Pre-analysis builds `directWriteMap` — kernel stores directly to
carryOut/ysStacked instead of internals, skipping `memory.copy`. Requirements: Kernel step, no
reduction, no data deps, single carry target, no passthrough from target carry, no later step reads
target carry. Provides 40-65% speedup for small bodies.

### WebGPU compiled-loop shader structure

Current `nativeScanMultiShaderSource()` shader:

```wgsl
for (var iter: u32 = 0; iter < length; iter++) {
  var acc: f32 = 0.0;  // reduction identity
  for (var ridx: u32 = 0; ridx < reductionSize; ridx++) {
    acc = acc + /* expression using ridx */;
  }
  carry[gidx] = /* epilogue using acc */;
  ys[iter * carrySize + gidx] = carry[gidx];
}
```

Key insight: Thread `i` only reads/writes `carry[i]` and `xs[:,i]` — no `workgroupBarrier()` needed.
Each step's computation is inlined per-thread.

### WebGPU preencoded-routine architecture

For routine bodies that can't be inlined into a shader.

**Why uniform-based offsets (not buffer offsets):** `minStorageBufferOffsetAlignment` is 256 bytes;
typical strides (e.g., 120 bytes for a 5×6 f32 matrix) fail alignment. Solution: bind entire
buffers, pass offset as uniform variable.

**`wrapRoutineForScan` transforms routine shaders to:**

1. Parse buffer bindings from WGSL source
2. Identify which bindings need offsets via `ScanBindingInfo` (inputs with JitId ≥
   numConsts+numCarry are xs)
3. Generate `ScanOffsets` struct with offset fields for xs bindings
4. Transform array accesses to add offset (e.g., `x[idx]` → `x[x_offset + idx]`)

**Dispatch architecture:** Ping-pong carry buffers (transient backend allocations, not tracked by
pool). Stacked ys filled by `copyBufferToBuffer` per iteration. Separate uniform bind groups per
iteration.

### ScanPlan type

```typescript
type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | { path: "compiled-loop"; executable: Executable; params?: NativeScanGeneralParams }
  | { path: "preencoded-routine"; preencodedParams: PreparedPreencodedScan };
```

`executeScan()` in `scan-executor.ts` dispatches based on `plan.path`.

### Autodiff

**JVP:** Doubled scan — primals + tangents flow together. **VJP/Grad:** JVP-transpose pattern with
√N checkpoint carries by default (`checkpoint` option). **Vmap:** Independent scans per batch
element with axis permutation.

### Key limitations

- WebGPU: numCarry≠numY, internal buffer deps, mixed kernel+routine → fallback
- `grad(scan)` backward on WebGPU with linalg body → O(N) dispatches (reformulate as
  `associativeScan` or use WASM)
- Sort in scan body on WebGPU → fallback (uniform conflict)
- Mixed-dtype carries on WebGPU → fallback
- Nested loops (scan-inside-scan, assocScan-inside-scan) always fall back on both backends —
  planner's `bodyProgram.steps.filter(s => s.type === "execute")` excludes inner loop JitSteps

### Why grad(scan) backward systematically falls back on WebGPU

The chain rule for a multi-step forward body `B = f(A, x); C = g(B, x)` transposes to
`dA = f_T(dB); dB = dB + g_T(dC)` — a sequential dependency chain. **Any** forward body with 2+
composing operations produces internal deps in the transposed body. This is autodiff algebra, not a
fixable code path.

For a Kalman/DLM backward pass (RTS smoother), each of the N iterations requires multiple sequential
GPU dispatches orchestrated from JS, giving O(N) total dispatch calls. For N=1600 this causes ~1s
latency. **This is the dominant performance bottleneck** for `grad(scan)` over linalg-heavy bodies
on WebGPU and the primary motivation for the PREENCODED-MULTI-KERNEL-SCAN-PLAN.

**Workarounds:**

1. **Reformulate as `associativeScan`** — if the backward recursion is a parallel prefix over
   associative affine maps → O(log N) dispatches. Requires mathematical reformulation.
2. **Use WASM backend** — compiled-loop handles internal deps by allocating temporaries inside the
   module.
3. **Pre-encoded command buffer (planned)** — encode all N × S body steps into one
   `GPUCommandEncoder`, submit once → O(N×S) dispatches but only 1 submit.

**Fallback command batching (current mitigation):** `executeScanFallback()` wraps the iteration loop
in `backend.beginBatch()` / `backend.endBatch()`. Every `SCAN_BATCH_SIZE` (256) iterations, flush
and restart. Reduces O(N) `queue.submit()` to O(N/256) submissions.

### Debugging scan paths

```ts
await lax.scan(f, init, xs, { acceptPath: ["compiled-loop"] }); // throws if not matched
await lax.scan(f, init, xs, { acceptPath: [] }); // shows chosen path in error
setDebug(1); // scan path selection reason
```

---

# Part 3: Buffer Recycling & WebGPU Buffer Pool

Two complementary optimizations for allocation overhead:

1. **JIT recycling** — compiler replaces `free(a)→malloc(b)` pairs (same size) with zero-cost
   `recycle(a→b)` slot rename.
2. **WebGPU buffer pool** — backend pool of freed `GPUBuffer`s by padded size, avoids
   `createBuffer`/`destroy` cycles.

**Performance:** 6-9× for JIT multi-output programs, 50-58× for scan on WebGPU. Eager mode
unaffected.

### Peak-memory guarantee

Both preserve peak physical GPU memory. Pool budget set by `computePoolHints()` from JIT
compile-time analysis. `configurePool()` evicts stale entries and caps retained bytes at program's
peak before each execution.

### JIT recycling (`recycleBuffers()`)

Runs after `insertFreeSteps()`. Scans for `free(a)` followed by `malloc(b)` with
`size(a)===size(b)`, replacing with `recycle(a→b)`. Only skips past `incref`/`free` steps between
them.

### WebGPU pool

`Map<paddedSize, GPUBuffer[]>`, `MAX_POOL_PER_SIZE=4`. `decRef` pushes to pool instead of
`destroy()`. `malloc` checks pool before `createBuffer`. Pooled buffers NOT zeroed — all code paths
must fully overwrite output.

### WASM allocator

`WasmAllocator` with free-list + reset-on-empty + top-of-heap compaction. JIT recycling still helps
by skipping free-list search.

---

# Part 4: Ownership Friction Points & Debugging

### Key fixes (all resolved)

- **Anonymous constants in traced bodies:** `markAnonymousIfTracing(arr)` called by array factories
  during `makeJaxpr` body. Marks with `markAnonymous: true`. Flow: `inMakeJaxprBody` flag →
  `getOrMakeConstTracer` does `.ref` → `ClosedJaxpr.dispose()` does extra `.dispose()` for anonymous
  consts. `fullInternal`/`full()` are exempt (ESLint `require-mark-anonymous` enforces).
- **PETracer cascade:** Cascades to known values and Const recipe values but NOT
  `JaxprEqn.tracersIn`. Unreachable Const PETracers (from `hasAux` + `instantiateConst`) tracked in
  `PartialEvalTrace.allConstPETracers` (via `main.globalData`), disposed after
  `partialEvalGraphToJaxpr` returns. The `.ref` from `instantiateConst` is balanced by this cleanup.
- **User-disposed constants in grad bodies:** Constants with `rc≤1` protected before residual
  cleanup in `linearizeFlat`/`vjpFlat`. Without protection, residual disposal would free consts the
  user already disposed.
- **ClosedJaxpr.dispose() timing:** Too early → premature free of const buffers. Too late → leak.
  Called explicitly in `lax-scan.ts` at body jaxpr end-of-life points. Timing is critical in scan
  bodies where the jaxpr is reused across iterations.

### AOT linearization artifacts

| Type                   | Owns                                | Disposal                           |
| ---------------------- | ----------------------------------- | ---------------------------------- |
| `PrimalArtifact`       | Forward jaxpr consts, residuals     | `[Symbol.dispose]()`               |
| `PullbackArtifact`     | Backward jaxpr, residual refs       | `[Symbol.dispose]()`               |
| `ScanPullbackArtifact` | `primalForwardJaxpr`, `tangentBody` | NOT `transposedBody` (cache-owned) |

**Critical:** `transposeJaxprCache` is **cache-owned** — callers must NOT dispose returned
`ClosedJaxpr`.

### Debugging ownership issues

1. Identify the array (shape/dtype from error)
2. Check artifact ownership (disposed too early?)
3. Check cache-owned jaxprs (don't dispose `transposeJaxprCache` results)
4. Check disposal timing (`evalJaxprTransposed` `argPrimals`, PETracer cascade)
5. Check `getOrMakeConstTracer` ref balance
6. Add `slotCount()` checkpoints; test with CPU backend

**Critical transform compositions:** `grad(f)`, `jvp(grad(f))`, `hessian(f)`, `jit(grad(scan))`,
`vmap(grad(scan))`.

### Leak detection

```ts
const before = (getBackend() as any).slotCount();
// ...operations...
expect((getBackend() as any).slotCount() - before).toBe(0);
```

`checkLeaks.start()/stop()` wraps every test via `test/setup.ts`. Uses `slotCount()` across all
backends as ground truth. JIT caches cleared via `_disposeAllJitCaches()`.

### ESLint plugin (`@jax-js/eslint-plugin`)

In-repo at `packages/eslint-plugin`. Configs: `recommended`, `strict`, `internalTransforms`. Key
rules: `require-using`, `no-use-after-dispose`, `no-unnecessary-ref` (autofix), `no-array-chain`,
`no-dispose-then-reassign-param`, `require-mark-anonymous`, retention symmetry rules. `invariance`
overlay upgrades to `error` on `src/**`, `packages/**`, `test/**`.

**`no-unnecessary-ref` autofix caveat:** Safe for user code, **unsafe for internal tracer `.ref`
propagation** (BatchTracer incident). Skips `.ref` in `UpdateExpression`/`BinaryExpression`.
File-level `// jax-js-lint: allow-ref` suppresses all warnings.

### Disposal patterns guide

1. **`using` for intermediates** (80% case)
2. **Block scope + `using`** for early disposal
3. **Scan carry/output** — use `tree.dispose()` or `tree.makeDisposable()`
4. **Extract and dispose** — `await arr.consumeData()`
5. **`tree.makeDisposable`** for result structs
6. **JIT function disposal** — `f.dispose()` frees captured consts; `clearCaches()` for all metadata

**JIT function-identity dedup:** `_jitRegistry: WeakMap` prevents cache bloat from repeated
`jit(fn)` calls. Same `(fn, opts)` → same `OwnedFunction`. Inline arrows NOT deduped — hoist to
stable binding.

### Ownership Correctness Principle

> **Write code that is ownership-correct in both eager and JIT mode.**

`jit()` is a **pure performance optimization** (kernel fusion, buffer recycling, dispatch batching).
It must never change program semantics — including memory ownership. If code leaks or
double-disposes in eager mode, that is a real bug, not something to paper over by wrapping in
`jit()`.

| Aspect                    | Eager mode                         | JIT mode                              |
| ------------------------- | ---------------------------------- | ------------------------------------- |
| Intermediate lifetimes    | Live until `.dispose()` / GC       | Freed at exact last-use automatically |
| Peak memory (chains)      | O(all intermediates)               | O(max concurrent live)                |
| Buffer reuse              | Pool only (on dealloc→realloc)     | Compile-time recycling + pool         |
| Kernel fusion             | None (one dispatch per op)         | Fused into single kernels             |
| **Ownership correctness** | **Must be correct** (ground truth) | **Must also be correct**              |

**`using` inside `jit()` bodies:** Works correctly and is recommended. On tracers, `dispose()` is a
harmless no-op (`Tracer` base class defines `[Symbol.dispose]() {}`). The library uses `using`
extensively inside traced functions (`lax.ts`, `random.ts`, `numpy-fft.ts`, `lax-linalg.ts`).

### Migration guide from move semantics

For code ported from the upstream `.ref` / move-semantics model:

1. **Remove all `.ref` calls** — operations no longer consume inputs
2. **Replace `disposeAll(a, b, c)` with `using` / `.dispose()`** — or use `tree.dispose()`
3. **`using` for intermediates, `.dispose()` for object properties** — or wrap in
   `tree.makeDisposable()`
4. **`.data()` no longer auto-disposes** — add `.dispose()` after reading, or use `.consumeData()`
5. **Never use `using` on values that are returned** — `using` disposes at scope end
6. **Destructure `tree` module** to avoid `no-use-after-dispose` false positive:
   `const { dispose: disposeTree } = tree`

---

# Part 5: Associative Scan

## Overview

`lax.associativeScan(fn, elems, options)` — parallel prefix scan using Kogge-Stone doubling.
`fn: (a: T, b: T) => T` must be associative. O(N log N) work, O(log N) depth. Options: `axis?`,
`reverse?`.

**Key files:** `src/library/lax-associative-scan.ts`, `src/frontend/core.ts`
(`Primitive.AssociativeScan`), backends in `wasm.ts`/`webgpu.ts`, transform rules in
`jvp.ts`/`linearize.ts`/`vmap.ts`.

### Algorithm

Each round: `next = concat(current[0:stride], fn(current[0:N-stride], current[stride:N]))`. After
ceil(log₂ N) rounds, done. All slices use O(1) `core.shrink` views.

### Backend behavior

| Backend | Implementation                                                                     | Performance                               |
| ------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| WebGPU  | Fused shader (M7.4): all body kernels → 1 dispatch/round                           | ~5-8× vs scan at N=65536                  |
| WASM    | Compiled-loop (M7.2): full ladder in 1 WASM call; M7.3: parallel kernel for N≥4096 | Single invocation, zero boundary overhead |
| CPU     | JS TypedArray ops per round                                                        | Slower                                    |

### Autodiff

| Transform | Strategy                                                        |
| --------- | --------------------------------------------------------------- |
| JVP       | Doubled inputs, Kogge-Stone with JVP'd body via `jvpJaxprCache` |
| Transpose | Reverse sequential scan of N-1 iterations                       |
| Vmap      | Independent scans per batch axis                                |

`grad(associativeScan)` maintains O(log N) depth: 78-188× faster than `grad(scan)` at N=4096.

### API contract

Inputs NOT consumed. Caller owns result. `fn` must dispose its own intermediates (use `using`).
Supports pytrees (all leaves same size on scan axis). Reverse via flip→scan→flip.

### vs `lax.scan`

Use `scan` for non-associative recurrences or when carry≠output. Use `associativeScan` when fn is
associative and N is large (crossover ~N≈1024 on WebGPU). `grad` preserves O(log N) depth.

---

# Part 6: Linear Algebra Autodiff

## TriangularSolve JVP bug fix

The JVP rule for `Primitive.TriangularSolve` was missing a `triu(dA)` mask — it used all of `dA`
instead of only the upper triangle. This caused gradient leaks through `solve` and `inv` (errors
0.15-0.44, not precision issues).

**Fix:** `using maskedDa = (unitDiagonal ? triu(da, 1) : triu(da))` before
`batchMatmulT(maskedDa, x)`.

## Current design

`solve(A, b)` differentiates directly through `lu → triangularSolve` — no Newton refinement needed.
`stopGradient` only on `permRaw` (integer, no gradient). Do NOT stop gradient on `a` or `luRaw`.

`inv(A)`: n≤4 uses closed-form Cramer's rule; n≥5 delegates to `solve(A, I)`. Both paths fully
differentiable.

**Primitive convention:** `TriangularSolve` solves `A @ X^T = B^T` (A upper triangular), returns
`X = B @ A^{-T}`. `core.triangularSolve(..., {lower: true})` flips to upper.

---

# Part 7: Polymorphic Shapes

Single JIT-compiled program reused across different input sizes on specified axes.

**Key design:** Concrete compilation + symbolic caching. Compiled for first call's concrete sizes;
cached under symbolic key (`sizeExprKey()` e.g. `"T*4"`).

**Types (`src/dim.ts`):** `SymDim` (named dimension variable), `Dim` (`number | SymDim`),
`SymbolicSize` (factor × prod(syms)), `SizeExpr` (`number | SymbolicSize`).

**Flow:** `makeJaxpr` with `dynamic_axes` → `SymDim` propagates through shapes →
`_currentDimBindings` (module-level, set/cleared in `jitCompile()` via try/finally) resolves during
compilation → concrete values passed at execution.

**Polymorphic scan length:** WASM compiled-loop receives length as arg 0; `dimBindings` resolution
uses `args[numConsts]` (skips captured constants).

**Limitations:** Mega-module rejects symbolic sizes. WebGPU scan compiled-loop/preencoded-routine
don't support symbolic length.

---

# Part 8: Completed Architecture & Future Work

## Milestones (all complete)

M0: Baseline tests + `BackendCapabilities` | M1: Scan backward AOT | M2: ScatterAdd | M3:
Multi-output fusion + epilogue | M4: Polymorphic shapes | M5: WASM parallel dispatch | M6:
Mega-module + orchestrator + parallel kernels | M7: AssociativeScan + compiled Kogge-Stone +
multithreaded + WebGPU fused | M8: Benchmarks + dead code audit

## Future performance work

| ID  | Title                 | Priority | Notes                                                                                                                                                                                     |
| --- | --------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P6  | Benchmark validation  | Medium   | Systematic benchmarks: matmul GFLOP/s, conv2d, SIMD, reductions. Extend `bench/matmul.bench.ts`                                                                                           |
| P1  | Tiled matmul (WebGPU) | **High** | Shared-memory blocking (`var<workgroup>`) for 5-10× matmul speedup. `TILE=16`, 2 KB shared mem. Conv2d benefits automatically via Dot. Acceptance: 2048×2048 f32 ≥40% theoretical GFLOP/s |
| P3  | i64 in wasmblr        | Medium   | Native i64 support (WASM MVP). Simplifies Threefry PRNG, unlocks f64 builtins. Zero browser risk                                                                                          |
| P2  | Relaxed SIMD FMA      | Medium   | `f32x4.relaxed_madd` for 2× dot-product throughput. Peephole: `Add(Mul(a,b),c) → relaxed_madd` in `translateExpCoreSimd()`. Needs runtime detection — Safari unsupported                  |
| P4  | Conv2d tuning         | Medium   | Phase 1: benchmark after P1 (tiled matmul gives free improvement). Phase 2 (conditional): specialized WGSL for small kernels (3×3, 5×5)                                                   |
| P5  | WebGPU subgroups      | Low      | `subgroupAdd()` for 2-4× faster reductions. **Blocked:** spec not stable, Chrome Canary only (Feb 2026). Implement when ≥2 browsers ship                                                  |

---

# Part 9: Session Continuity Notes

## Before starting work

1. `pnpm build` before tests — Vitest imports from `dist/`, not `src/`
2. `git branch` to confirm branch
3. `git log --oneline -10` for context

## Key patterns

- **Adding primitives:** `Primitive` enum → impl/JVP/transpose/vmap rules → export from `index.ts`
- **JIT flow:** `makeJaxpr` → `simplify()` → `splitGraphDataflow()` → `jitCompile()` → `execute()`
- **Ownership debugging:** Check artifact disposal timing, `transposeJaxprCache` (cache-owned),
  `getOrMakeConstTracer` ref balance, `argPrimals` set

## What gets lost at summarization

1. **Tests import from `dist/`** — edits in `src/` invisible until `pnpm build`
2. **eslint.config.ts:** `warn` globally; `invariance` overlay → `error` on `src/**`, `packages/**`,
   `test/**`
3. **`_currentDimBindings`:** Module-level state in `jitCompile()` via try/finally
4. **Multi-output kernel:** Access `kernel.outputs[0].exp`/`.reduction`/`.dtype`/`.bytes` explicitly
5. **`no-array-chain`** only in `strict` config, not `invariance`

## Architecture decisions log

| Decision                                        | Rationale                                                                                                                                                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Non-consuming ownership model                   | Eliminates UAF from `.ref` mistakes; trades for silent leaks + linting                                                                                                                                      |
| Concrete compilation + symbolic cache           | Simpler than full symbolic IR; ShapeTracker needs concrete strides                                                                                                                                          |
| `effectDrivenAllocate` over two-pass            | Single-pass liveness; DUS zero-copy from `Mutate` effect                                                                                                                                                    |
| Direct LU→triSolve gradient path                | Fixed TriSolve JVP `triu(dA)` mask made Newton unnecessary                                                                                                                                                  |
| `transposeJaxprCache` is cache-owned            | Prevents repeated transposition; callers must NOT dispose                                                                                                                                                   |
| WASM `(start, end, ...ptrs)` signature          | Enables work-splitting for `WasmWorkerPool`                                                                                                                                                                 |
| Mega-module rejects pass-through                | Steps can overwrite input locals; conservative bail                                                                                                                                                         |
| Module Workers (`type: "module"`)               | Deno requires module workers for blob-URL workers                                                                                                                                                           |
| SAB constructability over `crossOriginIsolated` | Works in Deno (native SAB) and browsers (COOP/COEP)                                                                                                                                                         |
| Vitest SAB tests skipped, Deno covers           | Browser main threads can't spin-wait; 8 skip Vitest, 17 Deno cover                                                                                                                                          |
| M7.4 WebGPU fused assocScan                     | Interleaved ping/pong, custom bind group layout, homogeneous dtype                                                                                                                                          |
| DUS vmap shrink+concat decomposition            | JIT `dus` axis=0 only; vmap shifts axis → decompose to concat                                                                                                                                               |
| ScatterAdd vmap: shared indices only            | 1-D indices, same positions for all slices                                                                                                                                                                  |
| `jit()` function-identity dedup                 | WeakMap prevents cache bloat; inline arrows NOT deduped                                                                                                                                                     |
| `markAnonymousIfTracing` + `inMakeJaxprBody`    | Array factories mark anonymous consts; `fullInternal` exempt                                                                                                                                                |
| Non-consuming `evalJaxpr`                       | Inputs protected by `inputVars` set; callers dispose temporaries explicitly. `.ref` at call sites was move-semantics leftover causing leaks. Pass-through outputs `.ref`'d; Lit arrays tracked and disposed |
