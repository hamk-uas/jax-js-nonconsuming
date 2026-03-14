# Open Work

Items that are currently in progress, deferred, or blocked. Completed items have been consolidated
into `.github/copilot-instructions.md`.

## Open Performance Items

| ID  | Title                     | Priority        | Description                                                                 |
| --- | ------------------------- | --------------- | --------------------------------------------------------------------------- |
| P2  | Relaxed SIMD FMA          | Medium          | `f32x4.relaxed_madd` for 2× WASM dot-product throughput. Safari unsupported |
| P3  | i64 in wasmblr            | Medium          | Native i64 (WASM MVP). Simplifies Threefry PRNG, unlocks f64 builtins       |
| P4  | Conv2d tuning             | Medium          | Specialized WGSL for common kernel sizes (3×3, 5×5)                         |
| P6  | Benchmark validation      | Medium          | Systematic benchmarks: matmul GFLOP/s, conv2d, SIMD chains, reductions      |
| P7  | Cooperative matrix (WMMA) | Blocked (~2026) | Hardware tensor cores for 2–4× tiled matmul. WGSL spec not yet stable       |

## P4: Conv2d Tuning Plan

**Why this now:** Conv2d is still called out as a major source of performance headroom, and unlike
WMMA it is not blocked on browser standards. The current implementation lowers convolution through
`prepareConv()` + generic `Dot` machinery, which is correct and flexible but leaves performance on
the table for common inference shapes.

### Current implementation hooks

- `src/frontend/convolution.ts`
  - Shape validation and `prepareConv()` data/layout transformation
- `src/frontend/jit.ts`
  - `Primitive.Conv` lowering: convolution becomes reshaped views + `Dot`
- `src/library/lax.ts`
  - Public `convGeneralDilated`, `conv`, `convTranspose` wrappers
- `test/conv.test.ts`
  - Correctness, grad, vmap, grouped-conv coverage
- `website/src/routes/bench/conv2d/jaxStrategy.ts`
  - Existing browser benchmark entry point

### Goal

Improve **WebGPU conv2d throughput on common CNN inference cases** without regressing correctness,
autodiff, grouped convolution semantics, or fallback behavior.

### Non-goal for this phase

- Do not redesign the public convolution API
- Do not introduce a general new Routine unless the generic lowering clearly tops out
- Do not optimize transposed convolution first; prioritize forward conv2d hot paths
- Do not chase unusual dilations/groups/strides before standard inference cases are measured

### Target cases

Benchmark and optimize these first:

1. 3×3 stride-1 SAME, batch 1, channels 32/64/128
2. 5×5 stride-1 SAME, batch 1, channels 32/64
3. 1×1 pointwise conv (common in bottlenecks / projections)
4. 3×3 stride-2 downsampling conv

Use NCHW inputs and OIHW kernels, matching current `lax.convGeneralDilated()` conventions.

### Phase A0: Observability tooling

**Deliverable:** structured measurement infrastructure that Phase A benchmarks can use and Phase B/C
tests can assert against.

**Why this first:** The current tooling (`profileGpu()` returns per-pass timing; `setDebug(2)` dumps
raw WGSL to console) is adequate for ad-hoc debugging but insufficient for structured diagnosis.
Phase A needs to answer "is the bottleneck dispatch count, reduction codegen, or layout expansion?"
— answering that requires correlated data, not raw logs.

#### Work

1. **Conv lowering kind signal** — add an internal enum
   (`generic-dot | fast-1x1-dot | fast-1x1-block-map | block-map-3x3 | block-map-5x5`) set during
   `Primitive.Conv` JIT lowering. Expose via a `_lastConvLoweringKind()` internal API (underscore =
   not public). Tests can assert which path activated without parsing console output.

2. **`profileGpuDetailed(fn)`** — extend `profileGpu()` with a detailed variant that returns:
   - pass count
   - per-pass GPU duration (already available from timestamp-query)
   - per-pass dispatch grid dimensions and workgroup size
   - per-pass shader label / hash (from `ShaderPipelineCache` key)
   - pipeline cache hit/miss flag

   Keep `profileGpu()` as the simple user-facing API. `profileGpuDetailed()` is the expert API for
   optimization work and bench scripts.

3. **Code capture API (public, unified)** — `setCodeCapture(cb)` registers a callback invoked on
   every compiled code unit (WGSL shader or WASM module) with a discriminated `CodeCaptureEntry`.
   `setCodeCapture(null)` disables it. Disabled by default (zero overhead). This is a **public
   export** from `src/index.ts`.

   **Why unified:** A single API covers both backends with one callback. The REPL panel shows WGSL
   and WASM side-by-side when switching devices — no separate `setShaderCapture` + `setWasmCapture`.
   Tests and bench scripts get consistent capture regardless of backend.

   **Why public:** The website REPL runs on end-user hardware (diverse GPUs not available on the dev
   machine). Exposing code capture lets users and developers see exactly which WGSL/WASM is
   generated for their specific hardware, enabling remote diagnosis and bug reports with concrete
   compiler output.

   **API shape:**

   ```ts
   export type CodeCaptureEntry = {
     backend: "webgpu" | "wasm";
     kind: "kernel" | "mega-module" | "scan" | "assoc-scan" | "block-map" | "routine";
     label?: string;
     code?: string; // WGSL source (WebGPU) or WAT source (WASM)
     workgroupSize?: [number, number, number]; // WebGPU only
     // Structured metadata (both backends)
     metadata?: {
       size?: number; // kernel element count
       simd?: boolean; // WASM SIMD path used
       numInputs?: number;
       numOutputs?: number;
       dtype?: string;
       reduction?: boolean;
       numSteps?: number; // mega-module / scan step count
       numKernels?: number; // mega-module kernel count
       byteLength?: number; // WASM module size in bytes
       [key: string]: unknown; // kind-specific extras
     };
   };
   export function setCodeCapture(cb: ((entry: CodeCaptureEntry) => void) | null): void;
   ```

   **Hook points (WebGPU — 1 site):**
   - `ShaderPipelineCache.prepare()` / `prepareSync()` — invoke on cache miss (first compilation).
     `ShaderInfo` already has `.code` and workgroup-size data.

   **Hook points (WASM — 14 sites across 4 categories):**

   | Category    | Function                               | File                                   | Metadata available                                          |
   | ----------- | -------------------------------------- | -------------------------------------- | ----------------------------------------------------------- |
   | Kernel      | `codegenWasm()` / `codegenWasmMulti()` | `src/backend/wasm.ts`                  | nargs, size, dtype, reduction, SIMD flag, ALU ops           |
   | Mega-module | `compileToMegaModule()`                | `src/backend/wasm/mega-module.ts`      | steps, kernel exports (name/size/isReduction), bytes.length |
   | Scan        | `codegenNativeScanGeneral()`           | `src/backend/wasm.ts`                  | numCarry, numY, reverse, routine imports, step count        |
   | Assoc-scan  | `codegenBlockedAssociativeScan()`      | `src/backend/wasm.ts`                  | numLeaves, blockSize, leafElemSizes                         |
   | Block-map   | `codegenBlockMapLoop()`                | `src/backend/wasm.ts`                  | gridShape, blockShape, numBlocks, strides                   |
   | Routine (8) | `get*Module()` in routine-provider     | `src/backend/wasm/routine-provider.ts` | n, dtype, SIMD flag, algorithm-specific params              |

   **WASM readability via wasmblr trace mode:** wasmblr is in-tree (`src/backend/wasm/wasmblr.ts`),
   so we can generate WAT source text during emission at zero external dependency cost.
   `CodeGenerator` gains an optional trace mode: when enabled, each instruction method appends its
   WAT mnemonic to an internal `string[]` buffer alongside the normal binary emission. The four
   factory functions (`UNARY_OP`, `BINARY_OP`, `LOAD_OP`, `STORE_OP`) already receive the
   instruction name string (e.g., `"add"`, `"mul"`, `"load"`); trace mode simply records it.
   Manual-emit control flow methods (`block`, `loop`, `if`, `else`, `end`, `br`, `br_if`, `call`,
   `const`, `select`, `drop`, `return`) and `local.get`/`set`/`tee` append their own mnemonics with
   operands. Block nesting depth (tracked in `#blockFrames`) drives indentation. `cg.toWat()` joins
   the buffer into a human-readable WAT string.

   **Design:** Trace mode is off by default (zero overhead). Each codegen site enables it when the
   global `setCodeCapture` callback is installed: `cg.trace = true` before emission, then
   `entry.code = cg.toWat()` after `cg.finish()`. This gives WASM entries the same `code` field
   readability as WGSL entries — both backends produce source text that can be displayed, diffed,
   and searched.

   **Cache interaction:** Kernels are hash-cached in `prepareKernelSync()`, routines are LRU-cached
   in `routine-provider.ts`. Like WebGPU, the callback fires on cache miss only — repeated
   compilations of identical code units are not re-captured.

4. **REPL compiled-code panel** — add a "Compiled Code" tab/pane to the website REPL alongside the
   existing Console. Before each run, the runner installs a `setCodeCapture` callback that collects
   entries into a reactive `$state` array. After the run, the panel displays entries grouped by
   backend, with source code shown as syntax-highlighted text for both WGSL and WAT. The "Capture
   compiled code" checkbox sits next to the existing "Detailed leak diagnostics" toggle.

   **Implementation sketch:**
   - `runner.svelte.ts`: add `captureCode = $state(false)` and
     `capturedCode: CodeCaptureEntry[] = $state([])`. In `_runProgram`, if enabled, call
     `jax.setCodeCapture(entry => runner.capturedCode.push(entry))` before execution and
     `jax.setCodeCapture(null)` in the finally block.
   - `+page.svelte`: add tab toggle (Console / Compiled Code) in the bottom pane header.
   - `CodePanel.svelte` (new): iterates `capturedCode`, renders each in a collapsible `<details>`.
     Both backends display `entry.code` in `<pre><code>` with a kind/label badge and optional
     metadata summary (size, SIMD, kernel count). Syntax highlighting: WGSL uses existing wgsl
     grammar, WAT uses lisp-style S-expression highlighting (or plain monospace).
   - The capture is per-run: cleared at the start of each run, not accumulated across runs.

#### Files

- `src/frontend/jit.ts` — conv lowering kind signal
- `src/index.ts` — export `profileGpuDetailed`, `_lastConvLoweringKind`, `setCodeCapture`,
  `CodeCaptureEntry`
- `src/backend.ts` — extend `GpuTimingResult` with per-pass metadata; `CodeCaptureEntry` type;
  `setCodeCapture` callback storage + global variable
- `src/backend/webgpu.ts` — collect per-pass grid/workgroup/label during profiling; invoke code
  capture callback in `ShaderPipelineCache.prepare()`/`prepareSync()`
- `src/backend/wasm/wasmblr.ts` — trace mode: `trace` flag, `#traceLines` buffer, `toWat()` method;
  modify `UNARY_OP`/`BINARY_OP`/`LOAD_OP`/`STORE_OP` factories + manual-emit methods to record WAT
  mnemonics when tracing
- `src/backend/wasm.ts` — invoke code capture in `prepareKernelSync()` (kernel cache miss),
  `codegenNativeScanGeneral()`, `codegenBlockedAssociativeScan()`, `codegenBlockMapLoop()`; enable
  `cg.trace` when capture callback is installed
- `src/backend/wasm/mega-module.ts` — invoke code capture in `compileToMegaModule()`; enable
  `cg.trace` when capture callback is installed
- `src/backend/wasm/routine-provider.ts` — invoke code capture in each `get*Module()` on LRU miss;
  enable `cg.trace` when capture callback is installed
- `website/src/lib/repl/runner.svelte.ts` — code capture integration in `_runProgram`
- `website/src/lib/repl/CodePanel.svelte` (new) — compiled code display component
- `website/src/routes/repl/+page.svelte` — tab toggle for Console / Compiled Code

#### Scope guard

The minimum to unblock Phase A is item 1 (conv lowering kind signal, ~20 lines). Items 2–3 can land
in parallel with Phase A benchmarks. Item 4 (REPL panel) can land independently — it depends only on
item 3 (the `setCodeCapture` API). WASM hook points can land incrementally: start with kernel +
mega-module (highest value), add scan/assoc-scan/block-map/routine hooks later.

**Production build stripping:** The code capture machinery (wasmblr trace mode, `setCodeCapture`
callback, `CodeCaptureEntry` metadata collection) is development-only. The `setCodeCapture` callback
is `null` by default, and all hook-point code is gated behind `if (codeCaptureCallback)` checks — a
one-branch cost that the JIT eliminates after first megamorphic call. wasmblr trace mode is off by
default (`trace = false`); the factory functions and manual-emit methods skip the `string[]` append
entirely when tracing is disabled. No import-time cost, no allocation cost, no codegen cost when
capture is not enabled. Tree-shaking removes the `setCodeCapture` export itself if unused by the
consumer's bundle.

### Phase A: Benchmark and profile first

**Deliverable:** reproducible conv2d baseline numbers on CPU / WASM / WebGPU, both GPUs where
applicable. Use Phase A0 tooling to capture structured per-pass data alongside raw throughput.

#### Work

1. Add a repo-local benchmark file for conv2d in `bench/` rather than relying only on the website
   page.
2. Benchmark representative shapes across:

- 1×32×64×64 → 64 filters, 3×3 SAME
- 1×64×64×64 → 64 filters, 1×1 SAME
- 1×64×128×128 → 128 filters, 3×3 stride 2
- 1×32×128×128 → 32 filters, 5×5 SAME

3. Run WebGPU benches on both NVIDIA and Intel.
4. Use `profileGpuDetailed()` (or `setDebug(1)` as fallback) to capture per-shape:

- dispatch count and grid dimensions
- per-pass GPU time breakdown (which pass dominates?)
- shader count (how many distinct shaders per conv?)
- whether ShapeTracker complexity causes extra dispatches vs simpler shapes

#### Files

- `bench/conv2d.bench.ts` (new)
- `website/src/routes/bench/conv2d/jaxStrategy.ts` (optional alignment / shape parity)

### Phase B: Cheap wins in existing lowering

**Deliverable:** better performance without changing the primitive surface.

#### Candidate optimizations

1. **1×1 fast path**

- Detect: all kernel dims = 1, stride 1, no dilation, **no groups** (`vmapDims`-aware)
- Guard: `groups > 1` must fall through to generic path (grouped conv semantics differ from matmul)
- Lower to reshape + `Dot` directly, bypassing `pool()` entirely
- Currently `pool()` already takes its fast path for 1×1 (d=1, k=1 ≤ s=1), adding only two trailing
  dims of size 1. The actual win is reducing ShapeTracker view complexity, not eliminating pool.
  Realistic expectation: **20–50% improvement** unless we lower to `block_map` tiled matmul.
- To match matmul-class throughput, detect 1×1 conv with large channel counts and lower to
  `block_map` tiled matmul directly (reusing existing `tileMatmul` body).
- **Autodiff:** reshape + Dot both have JVP/transpose rules, so `jit(grad(conv1x1))` works
  automatically. No new transform rules needed.

2. **Kernel-shape heuristics**

- Detect 3×3 and 5×5 cases early in jit lowering so Phase C codegen can branch cleanly
- Tag the detection result on the JIT context, don't emit specialized code yet

#### Files

- `src/frontend/convolution.ts`
- `src/frontend/jit.ts`
- Possibly `src/alu.ts` if expression simplification around conv lowering proves limiting

### Phase C: Tiled conv2d via `block_map`

**Deliverable:** materially higher WebGPU throughput for 3×3 / 5×5 conv2d.

#### Mechanism: `block_map` (not hand-written WGSL)

The project's design philosophy is "generative compiler over static kernel libraries." The right
vehicle for tiled conv2d is `block_map`, which already provides shared-memory tiling, barrier
scheduling, register tiling (`threadTile`), and fused codegen. A new Routine with hand-written WGSL
would contradict the philosophy and duplicate infrastructure.

**Approach:**

1. Express conv2d as a `block_map` body: load input tile + halo into shared memory, apply kernel
   weights, accumulate output tile. Use `ForiLoop` to iterate over input-channel tiles (K-dim),
   mirroring the existing tiled matmul pattern.
2. Detect eligible conv shapes in `Primitive.Conv` JIT lowering (3×3/5×5, stride 1 or 2, no
   dilation, no groups) and emit `BlockMap` IR instead of generic `Dot`.
3. Generic conv lowering remains the fallback for all other shapes.

**Key differences from tiled matmul:**

- Input tile includes a halo region (kernel_size - 1 pixels on each spatial edge)
- Halo means input tile is larger than output tile (e.g., 66×66 input for 64×64 output with 3×3)
- Weight tile is small and fixed (e.g., 3×3×C_in per output channel group)
- Stride-2 halves the output spatial dims relative to input tile

**Autodiff:** `block_map` bodies are jaxpr-traceable, so `jit(grad(conv_blockmap))` works via
existing `BlockMap` JVP/transpose rules. No new transform rules needed.

#### Decision gate

Before implementing, compare Phase A numbers against the improved Phase B generic path. If Phase B
already achieves acceptable throughput on the target shapes, defer Phase C. Implement Phase C only
for cases where the generic Dot reduction is clearly memory-bandwidth-limited due to poor locality.

#### Files

- `src/frontend/jit.ts` — fast-path detection in `Primitive.Conv` rule, emit `BlockMap` IR
- `src/frontend/convolution.ts` — helper to compute halo sizes and tile parameters
- `src/library/lax.ts` — possibly a `tiledConv2d` internal helper (not public API)

### Correctness and regression coverage

Must preserve all existing behavior in `test/conv.test.ts`, plus add targeted cases for optimized
paths:

1. 1×1 conv fast path matches generic path
2. 3×3 SAME optimized path matches generic path
3. 5×5 SAME optimized path matches generic path
4. Grouped conv continues to use correct fallback path
5. `jit`, `grad`, and `vmap` behavior remain correct for optimized cases
6. **Path-selection tests** using `_lastConvLoweringKind()` (Phase A0): assert that eligible shapes
   take the intended fast path and ineligible shapes (groups, dilation, exotic padding) fall through
   to `generic-dot`. This replaces the fragile approach of inferring activation from performance
   deltas or console output.
7. WebGPU-only code capture tests confirm that `block_map` conv emits the expected fused shader
   (single dispatch, correct grid) rather than falling back to per-block dispatch
8. REPL compiled-code panel displays captured WGSL and WAT source when "Capture compiled code" is
   enabled

### Success criteria

1. Add benchmark coverage in `bench/conv2d.bench.ts`
2. No regressions in `test/conv.test.ts`
3. Clear WebGPU speedup on at least one common case:

- 1×1 conv: target **match raw matmul throughput** for equivalent matrix shapes (current generic
  path adds ~20–50% overhead from ShapeTracker complexity; 2× only achievable via `block_map`)
- 3×3/5×5 conv: target **2×+** on NVIDIA via `block_map` tiled path (shared-memory tiling is
  required to beat the generic Dot reduction's poor spatial locality)

4. Intel path remains correct; if a specialization regresses Intel badly, gate it by heuristic

### Recommended execution order

1. Phase A0 conv lowering kind signal (~20 lines, unblocks testable path selection)
2. Phase A0 `setCodeCapture` public API + wasmblr trace mode — WebGPU hook first, then WASM kernel +
   mega-module hooks (trace mode enables WAT source collection at all WASM sites)
3. Phase A benchmark file + baseline numbers (use `profileGpuDetailed` if ready, `setDebug(1)` as
   fallback)
4. Phase A0 items 2 + 4 (detailed profiling + REPL compiled-code panel) in parallel with Phase A
   analysis. WASM scan/assoc-scan/block-map/routine hooks land incrementally after initial panel
5. 1×1 fast path (highest expected ROI / lowest complexity)
6. Re-benchmark with structured profiling
7. 3×3 stride-1 `block_map` specialization if still clearly bottlenecked
8. 5×5 specialization only if benchmarks justify it

## Deferred Items

### O6: Multi-Reduction Kernels

**Status:** Deprioritized.

**What it would do:** Fuse independent same-size reductions sharing inputs into a single dispatch.
Currently, each matmul/dot creates a kernel dispatch boundary in standalone JitProgram execution.
Two strategies were analyzed:

- **Strategy A (multi-output reduction):** Fuse N independent reductions with same
  `(size, inputArgs, reductionOp, reductionSize)` into one multi-output dispatch. Limitation: the
  DLM compose body's dots read different second operands (`p.A`, `p.b`, `p.S`) — different
  `inputArgs` → no fusion without an input-arg union relaxation (requires gid reindexing in AluExp
  trees).
- **Strategy B (chained reduction):** Fuse dependent reductions via register-resident intermediates.
  Requires new AluExp nodes for "local array" and nested reduction codegen.

**Why deferred:** The block-map fused shader already compiles entire compose bodies (all matmul
reductions + elementwise ops) into 1–3 dispatches for scan/assocScan. Body complexity has zero
effect on dispatch count — 2-tuple and 3-tuple `associativeScan` both produce identical dispatch
counts. O6 only helps standalone JitPrograms with multiple independent same-input reductions — a
narrow use case.

### O8d: fori_loop Unrolling into Command Tape

**Status:** Not started. Low priority.

**What:** Unroll `fori_loop` iterations into the command tape dispatch sequence (each iteration
produces N dispatches). This would expand tape eligibility to programs containing `fori_loop` steps.
Currently, `fori_loop` with concrete bounds is supported by WASM mega-module (compiled to native
loop) but rejected by the WebGPU command tape.

### Decoupled Fallback Phase 2: General Bodies

**Status:** Deferred. Phase 1 (scalar f32 add/mul/min/max) is complete and ships.

**What:** Extend the single-dispatch O(N) prefix scan to arbitrary associative functions (pytree
bodies, matrix compose). The lookback subgroup must invoke the user's body `fn` on raw input tiles
during work-stealing fallback, and the atomic descriptor must represent the general reduction value.
For pytree bodies with multiple leaves, this requires a packed representation or a multi-field
descriptor scheme.

**Phase 1 remaining opportunities:**

- Subgroup-parallel lookback (first subgroup instead of thread-0 only)
- Raking pattern: multiple elements per thread for better bandwidth utilization

**References:**

- Smith, Levien, & Owens — "Decoupled Fallback" (2024):
  https://github.com/b0nes164/Decoupled-Fallback-Paper
- GPUPrefixSums — production WGSL implementation: https://github.com/b0nes164/GPUPrefixSums

## Non-Goals

- **General einsum rank-adaptation:** The general `parseEinsumExpression` fails with rank-reduced
  inputs from per-element tracing. Only matters for exotic subscript patterns not covered by
  `einsumFastPath`. All common batch-matmul patterns already work. Low priority.
- **Sort/Argsort/LU jaxprification:** Fundamentally data-dependent algorithms (comparison-based,
  data-dependent pivoting). Cannot be expressed as fixed arithmetic traces.
- **GPU/WASM ratio ≤ 2× for small-matrix DLM:** Not achievable at small N. Irreducible GPU API costs
  (`queue.submit`) prevent matching WASM mega-module latency. For large-matrix workloads (e.g.,
  matmul 4096×4096), WebGPU already achieves 53.7% peak.
- **Binding limit optimization on high-limit hardware:** On Deno/NVIDIA with `maxArgs = 1,048,575`,
  the P2 split pass has no effect. Browser deployments with Chrome's `maxArgs ≈ 9` see additional
  fragmentation — but the dominant dispatch overhead comes from P1 structural rules (reduction
  boundaries, diamond heuristic), not binding limits.
