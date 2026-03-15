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
   `Primitive.Conv` JIT lowering. Expose via a `_lastConvClass()` internal API (underscore = not
   public). Tests can assert which path activated without parsing console output.

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
- `src/index.ts` — export `profileGpuDetailed`, `_lastConvClass`, `setCodeCapture`,
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

#### Baseline results (bench/conv2d.bench.ts, March 2026)

All cases compile to exactly **1 kernel** (fused im2col + matmul + epilogue).

**WASM (same CPU for both GPU configs):**

| Case                | Eager (Hz) | Eager (ms) | JIT (Hz) | JIT (ms) |
| ------------------- | ---------- | ---------- | -------- | -------- |
| 3×3 s1 32ch 64×64   | 3.1        | 323        | 3.1      | 320      |
| 3×3 s1 64ch 64×64   | 1.6        | 642        | 1.6      | 636      |
| 3×3 s1 128ch 32×32  | 1.6        | 637        | 1.6      | 629      |
| 1×1 pw 64ch 64×64   | 29.2       | 34.2       | 29.6     | 33.7     |
| 1×1 pw 128ch 32×32  | 15.4       | 65.0       | 15.5     | 64.9     |
| 5×5 s1 32ch 64×64   | 2.2        | 453        | 2.2      | 452      |
| 3×3 s2 64ch 128×128 | 1.0        | 1,050      | 1.0      | 1,046    |

**WebGPU — NVIDIA RTX 4070 Ti SUPER (TB3 eGPU):**

| Case                | Eager (Hz) | Eager (ms) | JIT (Hz) | JIT (ms) |
| ------------------- | ---------- | ---------- | -------- | -------- |
| 3×3 s1 32ch 64×64   | 284        | 3.52       | 351      | 2.85     |
| 3×3 s1 64ch 64×64   | 331        | 3.02       | 365      | 2.74     |
| 3×3 s1 128ch 32×32  | 333        | 3.01       | 367      | 2.72     |
| 1×1 pw 64ch 64×64   | 1,453      | 0.69       | 377      | 2.65     |
| 1×1 pw 128ch 32×32  | 812        | 1.23       | 368      | 2.72     |
| 5×5 s1 32ch 64×64   | 302        | 3.32       | 366      | 2.74     |
| 3×3 s2 64ch 128×128 | 336        | 2.97       | —        | —        |

**WebGPU — Intel Arc iGPU:**

| Case                | Eager (Hz) | Eager (ms) | JIT (Hz) | JIT (ms) |
| ------------------- | ---------- | ---------- | -------- | -------- |
| 3×3 s1 32ch 64×64   | 363        | 2.76       | 402      | 2.49     |
| 3×3 s1 64ch 64×64   | 376        | 2.66       | 310      | 3.23     |
| 3×3 s1 128ch 32×32  | 260        | 3.85       | 278      | 3.60     |
| 1×1 pw 64ch 64×64   | 367        | 2.73       | 398      | 2.51     |
| 1×1 pw 128ch 32×32  | 380        | 2.63       | 385      | 2.60     |
| 5×5 s1 32ch 64×64   | 386        | 2.59       | 339      | 2.95     |
| 3×3 s2 64ch 128×128 | 262        | 3.82       | 234      | 4.27     |

**Key findings:**

1. **WASM eager ≈ JIT** — conv2d compiles to a single reduction kernel with no fusion benefit. JIT
   overhead is negligible; both paths execute the same scalar reduction. Original 1000× gap was a
   **benchmark bug** — eager bench was not materializing results (`.dispose()` cancelled pending
   execution without running the kernel). Fixed by adding `.dataSync()`.
2. **WebGPU is dispatch-bound at ~2.5–3ms** — all cases bottleneck on dispatch overhead, not
   compute. NVIDIA and Intel show similar timings → GPU compute is irrelevant at this scale.
3. **GPU utilization is <1%** — e.g. 3×3 32ch 64×64 = 0.151 GFLOP at 351 Hz ≈ 53 GFLOP/s vs RTX 4070
   Ti SUPER peak 22,577 GFLOP/s (0.2%).
4. **WebGPU eager faster for 1×1** — 1×1 conv eager bypasses command tape overhead, reaching 1,453
   Hz (0.69ms) vs JIT 377 Hz (2.65ms). For small ops, eager dispatch directly to GPU queue is faster
   than the command tape's pre-resolved pipeline machinery.
5. **Bottleneck is 100% dispatch overhead, not kernel quality** — Phase B/C kernel optimizations
   won't help until tensor sizes are large enough to dominate dispatch cost. Larger batch sizes or
   `lax.scan` over conv bodies would amortize dispatch cost.
6. **WASM conv is CPU-bound on reduction** — 3×3 64ch 64×64 takes 636ms (1.6 Hz). The reduction
   kernel uses scalar (non-SIMD) codegen. This is the main WASM optimization opportunity.

#### GPU timing data (NVIDIA RTX 4070 Ti SUPER)

Larger shapes measured with `blockUntilReady()` timing, bypassing the `vitest bench` API (which
hangs on WebGPU async). This data motivated the Phase C decision gate.

| Case              | GFLOP | Eager ms | JIT ms | GFLOP/s | % Peak |
| ----------------- | ----- | -------- | ------ | ------- | ------ |
| 3×3 1×32ch 64×64  | 0.151 | 2.82     | 2.64   | 57.3    | 0.3%   |
| 3×3 1×64ch 64×64  | 0.302 | 3.03     | 2.22   | 136.1   | 0.6%   |
| 3×3 1×128ch 32×32 | 0.302 | 3.04     | 2.53   | 119.4   | 0.5%   |
| 3×3 8×64ch 64×64  | 2.416 | 3.70     | 2.77   | 872.6   | 3.9%   |
| 3×3 8×128ch 64×64 | 9.664 | 9.40     | 7.20   | 1342.0  | 5.9%   |
| 5×5 1×32ch 64×64  | 0.210 | 3.22     | 2.62   | 80.1    | 0.4%   |
| 1×1 1×64ch 64×64  | 0.034 | 0.84     | 2.72   | 12.3    | 0.1%   |
| 1×1 1×256ch 64×64 | 0.537 | 3.33     | 2.61   | 206.0   | 0.9%   |
| 3×3 s2 128ch 128² | 0.604 | 2.92     | 2.60   | 232.7   | 1.0%   |

Note: peak (22,577 GFLOP/s) is theoretical FP32 throughput. Even the largest case (9.664 GFLOP)
achieves only 5.9% utilization, confirming conv2d remains dispatch-bound and memory-latency-bound
for single-batch shapes.

### Phase B: Cheap wins in existing lowering

**Deliverable:** better performance without changing the primitive surface.

#### Results

**1×1 fast path: implemented** (`prepareConv1x1` in `convolution.ts`, JIT branch in `jit.ts`).

The fast path bypasses `pool()` entirely, lowering 1×1 conv to `moveaxis + reshape + Dot` directly.
This reduces ShapeTracker view complexity (fewer compose operations during compilation) but produces
the **same kernel** — a single Dot reduction. Compilation time is negligibly faster.

**Benchmark delta: none measurable.** Both WASM and WebGPU produce identical throughput. This
confirms Phase A finding #5: the bottleneck is dispatch overhead (WebGPU ~2.5ms) and kernel
execution (WASM), not ShapeTracker complexity. The 1×1 fast path is an architectural simplification,
not a runtime optimization.

| Case      | Backend       | Baseline (Hz) | Post-1×1 (Hz) | Delta |
| --------- | ------------- | ------------- | ------------- | ----- |
| 1×1 64ch  | WebGPU NVIDIA | 371           | 374           | noise |
| 1×1 128ch | WebGPU NVIDIA | 355           | 355           | none  |
| 1×1 64ch  | WebGPU Intel  | 400           | 405           | noise |
| 1×1 128ch | WebGPU Intel  | 413           | 416           | noise |
| 1×1 64ch  | WASM JIT      | 29.5          | 29.5          | none  |
| 1×1 128ch | WASM JIT      | 15.4          | 15.5          | none  |

**Tests added:** 5 new tests in `test/conv.test.ts` — correctness (1d, 2d), grad, vmap, grouped-conv
fallback. All use `_lastConvClass()` path-selection assertions.

#### Candidate optimizations

1. **1×1 fast path** ✅ Done — see results above.

2. **Kernel-shape heuristics** ✅ Done — `classifyConv` in `jit.ts` detects 3×3/5×5 and returns
   `block-map-3x3`/`block-map-5x5`. These still fall through to generic Dot lowering. Path-selection
   tests in `test/conv.test.ts` assert the classification.

### Phase C: Tiled conv2d via `block_map`

**Status:** Im2col materialization approach **rejected** (41% regression on NVIDIA). Halo-aware
`block_map` extension **designed** — implementation pending.

#### Im2col materialization: attempted and rejected

The im2col + tiledMatmul approach was fully implemented and benchmarked:

- Extract kH×kW patches via `uncheckedDynamicSlice` (compiles to index math in JIT)
- Concatenate into [M, K] matrix → tiledMatmul via `block_map`

**Result:** 41% slower than generic Dot lowering on NVIDIA RTX 4070 Ti SUPER for the target case
(3×3, batch=8, 128ch, 64×64):

| Case              | Generic Dot  | Im2col + tiledMatmul | Delta |
| ----------------- | ------------ | -------------------- | ----- |
| 3×3 8×64ch 64×64  | 797 GFLOP/s  | 374 GFLOP/s          | −53%  |
| 3×3 8×128ch 64×64 | 1336 GFLOP/s | 787 GFLOP/s          | −41%  |

**Root cause:** The im2col materialization writes ~150 MB to VRAM (9× input for 3×3) then reads it
back in tiledMatmul. This extra bandwidth overwhelms the compute savings from shared-memory tiling.
The generic Dot path reads input directly with computed indices — higher arithmetic intensity per
byte fetched.

#### Halo-aware `block_map` design

**Status:** Design complete. Implementation pending.

##### Problem

`block_map` tiles inputs into **non-overlapping** blocks of `blockShape` elements per axis. Each
input tile is the same size as the output tile along mapped axes. Convolution has **overlapping
receptive fields**: a 3×3 conv needs 2 extra rows/cols of input beyond each output tile. Without
overlap, materialization (im2col) is the only option — and that was rejected above.

##### Solution: `halo` option with pre-pad strategy

Extend `BlockMapOptions` with a `halo` field specifying per-input, per-grid-axis overlap:

````ts
interface BlockMapOptions {
  blockShape: number[];
  inAxes?: (number | null)[] | (number | null)[][];
  outAxes?: (number | null)[] | (number | null)[][];
  gridShape?: number[];
  threadTile?: number[];
  /**
   * Per-input overlap along each mapped grid axis. Each input tile extends
   * beyond the output block range by `[lo, hi]` elements along the mapped
   * dimension. The body receives input shapes of `blockShape[g] + lo + hi`
   * (instead of `blockShape[g]`) along each halo-expanded axis.
   *
   * Format: `halo[i][g] = [lo, hi]` — extra elements before/after the
   * output block's range for input `i` along grid axis `g`.
   * `null` or `[0, 0]` means no halo.
   *
   * A single `([number, number] | null)[]` is broadcast to all inputs.
   * An array of `([number, number] | null)[][]` provides per-input specs.
   *
   * Elements outside the array boundary are zero-padded.
   *
   * @example 3×3 convolution halos
   * ```ts
   * lax.blockMap(stencilBody, { image, kernel }, {
   *   blockShape: [16, 16],
   *   inAxes: { image: [2, 3], kernel: [null, null] },
   *   outAxes: [2, 3],
   *   halo: { image: [[1, 1], [1, 1]], kernel: [null, null] },
   * });
   * // image body shape: [16+1+1, 16+1+1] = [18, 18] per tile
   * // kernel body shape: unchanged (broadcast, no halo)
   * ```
   */
  halo?: ([number, number] | null)[] | ([number, number] | null)[][];
}
````

After resolution: `flatHalo[inputIdx][gridAxis] = [lo, hi]`, defaulting to `[0, 0]`.

##### Core invariant: pre-pad makes halo transparent to backends

**Grid shape is computed from original (un-padded) input dimensions.** The grid covers the output
space; `blockShape` determines output tile size. Halo only expands input tiles.

The key insight is that **pre-padding** the input by `[lo, hi]` along each halo axis lets the
existing base-offset formula work unchanged:

```
Per block b, in the ORIGINAL tensor:
  output range:    [b·bs, (b+1)·bs)
  halo input range: [b·bs - lo, (b+1)·bs + hi)

Per block b, in the PADDED tensor (shifted by lo):
  halo input range: [b·bs, b·bs + (bs + lo + hi))
  base offset = b · bs · stride   ← SAME formula as without halo!
```

The padded input dimension is `ceil(originalDim / bs) · bs + lo + hi`, which ensures all halo reads
are in-bounds. No per-element OOB checks needed in shaders. No signed-arithmetic issues.

##### Changes by layer

| Layer                   | File(s)                 | What changes                                                                                                                |
| ----------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **API**                 | `lax-block-map.ts`      | Add `halo` option, resolve to `flatHalo`, expand block avals: `shape[axis] = blockShape[g] + lo + hi`                       |
| **Params type**         | `core.ts`               | Add `halo?: [number, number][][]` to `Primitive.BlockMap` params                                                            |
| **Eager execution**     | `array.ts`              | Expand slice range by `[lo, hi]` around output block, zero-pad edges. Grid from original dims                               |
| **JIT compilation**     | `jit.ts`                | Store `halo` in block_map step                                                                                              |
| **JIT execution**       | `jit.ts` (step loop)    | Pre-pad inputs with `lo`/`hi` zeros, pass padded `inputShapes`. Free padded buffers after dispatch                          |
| **JVP**                 | `jvp.ts`                | Forward `halo` in doubled BlockMap params (tangents have same spatial structure)                                            |
| **Transpose**           | `linearize.ts`          | Forward `halo` in transposed params. Overlapping grad accumulation **deferred** (see below)                                 |
| **Vmap**                | `vmap.ts`               | Forward `halo` unchanged (vmap adds batch dim; halo on spatial dims unaffected)                                             |
| **Partial eval**        | `linearize.ts`          | Forward `halo` unchanged                                                                                                    |
| **WebGPU fused shader** | `webgpu/block-map.ts`   | **No changes.** Pre-padded inputs + halo-expanded body avals flow through existing `inRemap`, `in_base`, stride math        |
| **WASM compiled loop**  | `wasm.ts`               | **No changes.** `blockInputSizes` from body avals already reflect halo; `memory.copy` from padded buffer works              |
| **Executor**            | `block-map-executor.ts` | **No changes.** Receives pre-padded inputs with correct `inputShapes`                                                       |
| **Spec key**            | `block-map-executor.ts` | Include `halo` in `blockMapSpecKey` (affects body program + shapes → implicitly covered since `inputShapes` already in key) |

**Why backends need no changes:** The body JitProgram is compiled from a jaxpr traced with
halo-expanded input avals. All kernel sizes, stride computations, and shared-memory allocations in
the body already reflect the expanded tile shape. The backends just see "larger input tiles" — the
`inRemap` decomposition, boundary guards, and cooperative loading all adapt automatically because
they're driven by `inputShapes` (padded) and body kernel dimensions (expanded).

##### Pre-pad mechanics (JIT execution)

```
For each input i with halo[i][g] = [lo, hi] on grid axis g → dim d = axes[g]:
  paddedDim = ceil(originalDim / bs) * bs + lo + hi
  Allocate padded buffer: same shape but dim d → paddedDim
  Copy original data into padded buffer at offset lo along dim d
  (remaining positions are zero — default GPUBuffer / WASM alloc behavior)
```

For the WebGPU path, this is a `copyBufferToBuffer` with appropriate offsets. For WASM, a
`memory.copy` into the scratch region. If no halo is active (`lo = hi = 0` for all inputs), no
padding occurs — zero overhead for existing code.

##### Body kernel size analysis

The body function receives halo-expanded inputs and produces `blockShape`-sized outputs. Inside the
body, stencil patterns trace like this:

```ts
// 3×3 stencil: 9 shifted views + weighted sum
// inputTile: [18, 18],  output: [16, 16]
for (ki of [-1, 0, 1])
  for (kj of [-1, 0, 1]) {
    const shifted = uncheckedDynamicSlice(inputTile, [1 + ki, 1 + kj], [16, 16]);
    // shifted: [16, 16] → kernel.size = 256 = prod(blockShape) ✓
    acc = acc.add(shifted.mul(weight[ki][kj]));
  }
```

All intermediate kernels are `prod(blockShape)`-sized (output tile). The only "large" entity is the
halo input's shared-memory allocation (`(blockShape[g] + lo + hi)` elements per mapped axis), but
that's a `malloc` step, not a kernel. The fused shader's `kernel.size % blockSize === 0` check
passes for all body steps.

##### Autodiff considerations

- **JVP (forward-mode):** Tangent inputs have the same shape as primals. JVP of halo block_map is
  another halo block_map with doubled inputs/outputs. Works automatically with forwarded `halo`.

- **VJP/Grad (reverse-mode):** Requires careful design. Two strategies exist with different
  trade-offs.

  **The overlap problem:** In the forward pass, each block reads a halo-expanded input tile
  (overlapping with neighbors) and writes a disjoint output tile. A naive transpose reverses this:
  each backward block reads a disjoint cotangent tile and writes to an overlapping input-gradient
  region. Adjacent backward blocks write to the same addresses — requiring scatter-add or atomics.

  **Why `transposeJaxpr` cannot produce a gather-based backward body:** The existing transpose of
  `UncheckedDynamicSlice(input[B+lo+hi], offset, [B])` is `scatterAdd(zeros[B+lo+hi], indices, ct)`
  (confirmed in `linearize.ts` line 3344). So `transposeJaxpr(forwardBody)` produces a body with
  shape contract `B → (B+lo+hi)` — the opposite of what gather-based VJP needs `(B+hi+lo) → B`. This
  is fundamental: the existing transpose machinery converts reads-with-halo into
  writes-with-overlap, not into reads-with-reversed-halo.

  **Strategy 1: Scatter-add (general, correct, slower)**

  Use `transposeJaxpr` as-is. Run the transposed block_map **without halo** — each backward block
  produces a `(B+lo+hi)`-sized gradient tile that overlaps with neighbors. Accumulate via
  `scatterAdd` post-pass. This works for ALL body types (linear, non-linear, data-dependent).

  Cost: extra dispatch + bandwidth for the scatter-add accumulation. On WebGPU, `atomicAdd` has ~4
  ULP precision loss per accumulation step. Breaks the "one workgroup owns one output tile"
  execution model.

  **Strategy 2: Stencil body synthesis (fast for linear stencils, requires compiler pass)**

  For bodies composed of `{UncheckedDynamicSlice, scalar Mul, Add}` — the standard linear stencil
  pattern — we can **synthesize** a new backward body that reads with halo and writes disjointly:

  | Pass     | Reads (with halo)     | Writes (disjoint) |
  | -------- | --------------------- | ----------------- |
  | Forward  | Input tensor          | Output tensor     |
  | Backward | Cotangent/output-grad | Input gradient    |

  The insight: the adjoint of a linear stencil is also a linear stencil with spatially reversed
  weights. For each `UDS(input, [k], [B])` with weight `w[k]` in the forward body, the synthesized
  backward body emits `UDS(ct_tile, [K-1-k], [B]) * w[K-1-k]`. The new body has shape contract
  `(B+hi+lo) → B` — halo-expanded input, disjoint output. No atomics, no scatter-add.

  **This is NOT `transposeJaxpr` reuse.** It is a domain-specific compiler pass that:
  1. Analyzes the forward body for `{UDS, Mul(scalar), Add}` patterns
  2. For each UDS at offset `[k]`, emits a reversed UDS at offset `[K-1-k]`
  3. Preserves the combining ops with spatially reversed weights
  4. Produces a new jaxpr with `(B+hi+lo) → B` shape contract

  **Halo derivation for backward body:**
  - Symmetric case (`lo === hi`): backward halo = forward halo (covers 3×3, 5×5, all odd kernels)
  - Asymmetric case: backward halo `[hi, lo]` (reversed), derivable from UDS offset analysis

  **For conv2d specifically:** `Primitive.Conv` has its own JVP/transpose rules — the block_map halo
  is an execution strategy invisible to autodiff. Training correctness does not depend on halo-VJP.
  `grad(conv2d)` traces conv transpose first, then JIT lowers both forward and backward convs to
  halo block_map independently. No block_map transpose rule involved.

  Supporting `grad(blockMap({halo}))` matters for user-authored stencils (custom blur, edge
  detection, PDE solvers) but is not needed for conv2d training.

  **Scope of stencil synthesis:** Mechanical for purely linear stencils. Undefined for non-linear
  operations (relu, max-pool, data-dependent indexing) — these always fall back to scatter-add.

  | Use case               | Linear body?       | Strategy 2 feasible?                  |
  | ---------------------- | ------------------ | ------------------------------------- |
  | 1D moving average      | Yes                | Yes                                   |
  | 3×3 / 5×5 conv stencil | Yes                | Yes                                   |
  | PDE Laplacian          | Yes (self-adjoint) | Yes (backward = forward)              |
  | Causal stencil [0,K-1] | Yes                | Yes (backward halo [K-1,0])           |
  | Non-linear stencil     | No                 | No → scatter-add                      |
  | Max-pool with halo     | No                 | No → scatter-add                      |
  | Halo + foriLoop body   | Depends            | Scope unclear → scatter-add initially |

##### Implementation phases

**Phase C.1: API + Tracing + Eager**

1. Add `halo` to `BlockMapOptions`, resolve to `flatHalo: [number, number][][]`
2. Modify block aval computation: `shape[axis] = blockShape[g] + lo + hi`
3. Add `halo` to `Primitive.BlockMap` params type in `core.ts`
4. Modify eager execution: expand slice range, zero-pad halo edges
5. Forward `halo` in JVP, vmap, PE rules
6. Test: 1D moving average (blockShape=[8], halo=[[1,1]]), verify eager output matches sequential

**Phase C.2: JIT Zero-Copy Halo** ✅ (WebGPU + JS fallback) / interim (WASM pre-pad)

_WebGPU fused shader:_ signed bounds-checked reads in `gen_resolve()` —
`clamp(rawExpr, packOff, packOff+total-1)` with `select(zero, read, valid)`. Leaf-packing-aware:
bounds use packed coordinates `[packOff, packOff+total)`. No pre-padding, no extra allocation.

_JS fallback:_ halo-aware per-block slicing with `rawStart = blockIdx * blockShape - lo`, clamped
ranges, hoisted `sharedZeros` buffer (reused across blocks, no GC pressure).

_WASM compiled loop (interim):_ `prePadForWasmHalo()` allocates zero-filled padded buffers in WASM
linear memory, copies original data at halo offset, passes to `buildBlockMapWasmParams` with
`halo: undefined`. Correct but wasteful — see Phase C.2b.

1. ~~Store `halo` in block_map JitStep~~ ✅
2. ~~WebGPU: zero-copy signed bounds checks in fused shader~~ ✅
3. ~~WebGPU: leaf-packing + halo correctness (packed coordinates)~~ ✅
4. ~~JS fallback: halo-aware slicing with shared zero buffer~~ ✅
5. ~~WASM: pre-pad interim (not rejected, falls through to compiled loop)~~ ✅
6. ~~Test: T8.1–T8.8 covering 1D/2D, symmetric/asymmetric, eager/jit~~ ✅

**Phase C.2b: WASM Compiled Halo (zero-copy) + Interior Fast Path** ✅

Replaced `prePadForWasmHalo()` with halo-aware codegen in `codegenBlockMapLoop()`. Interior blocks
skip `memory.fill` entirely (copy-only fast path). Boundary blocks use signed clamped copy with
`dstSkip` for halo offset. Non-halo inputs use unsigned boundary check. `prePadForWasmHalo()`
deleted (-80 LOC), `tryExecuteBlockMapWasm()` simplified (no alloc/free overhead).

1. ~~Add `halo?: [number, number][][]` to `BlockMapWasmParams`~~ ✅
2. ~~Pass `params.halo` through in `buildBlockMapWasmParams()`~~ ✅
3. ~~Rewrite `codegenBlockMapLoop()` Step 1: interior fast path + halo-aware boundary~~ ✅
4. ~~Remove `prePadForWasmHalo()` call from `tryExecuteBlockMapWasm()`~~ ✅
5. ~~Delete `prePadForWasmHalo()` function~~ ✅
6. ~~T8.1–T8.8 pass; all 1885 tests pass~~ ✅

**Phase C.3: Conv2d Body + Integration** ✅

`rewriteConvToBlockMap()` in `jit.ts` rewrites eligible Conv equations to BlockMap before dataflow
analysis. Guards: `block-map-3x3`/`block-map-5x5`, stride=1, SAME-equivalent padding
(padTop+padBottom = kH-1), spatial ≥ 16, concrete shapes, vmapDims=0, backend≠webgpu. Body jaxpr
uses VALID conv on halo-expanded tiles (recursion guard: VALID padding fails SAME-equivalence
check). Batch (N) and channel dims are untiled — each block handles the full N × C_out per tile.

Backend gate: WebGPU is excluded because `block_map` falls back to per-block dispatch for conv
bodies (no fused shader yet), which is slower than generic-dot. CPU and WASM have efficient
block_map execution paths. `_lastConvRewritten()` reports whether the rewrite actually fired
(distinct from `_lastConvClass()` which reports kernel-shape classification).

WASM A/B benchmark (Intel Core Ultra 5 125H, same workload, `_setConvRewriteEnabled` toggle):

| Size          | block_map | generic-dot | Speedup   |
| ------------- | --------- | ----------- | --------- |
| 3×3 4ch 16×16 | 162 µs    | 166 µs      | 1.02×     |
| 3×3 4ch 32×32 | 500 µs    | 643 µs      | **1.29×** |
| 3×3 8ch 64×64 | 7,504 µs  | 10,192 µs   | **1.36×** |

Speedup grows with spatial size as tiled memory access patterns dominate over dispatch overhead.

1. ~~Write conv body via `rewriteConvToBlockMap` jaxpr pass~~ ✅
2. ~~Wire into `classifyConv` dispatch for `block-map-3x3` / `block-map-5x5`~~ ✅
3. ~~Benchmark against generic Dot path~~ ✅ (WASM: 1.02–1.36× faster)
4. ~~Handle batch (N) and output channel (C_out) dimensions~~ ✅ (untiled, full per block)

**Phase C.4: Halo VJP**

_Prerequisite: C.1–C.2 landed (halo works in forward pass)._

**Step 1: Scatter-add fallback (correct baseline)**

The default path. Use `transposeJaxpr(forwardBody)` as-is — it produces a body with shape contract
`B → (B+lo+hi)` via the existing UDS→scatterAdd transpose. Run the transposed block_map WITHOUT
halo. Each backward block writes `(B+lo+hi)` elements that overlap neighbors. Post-pass scatter-add
accumulates the overlapping regions.

1. In `transposeBlockMap` (`linearize.ts`): when `params.halo` is present, emit the transposed
   block_map with NO halo and output shapes `(B+lo+hi)`. Follow with a `scatterAdd` to accumulate
   overlapping tiles into the `B`-sized input gradient.
2. Emit `setDebug(1)` diagnostic: "halo-VJP: using scatter-add accumulation".
3. Test: `grad(blockMap({halo: [[1,1]]}))` on 1D 3-point stencil matches finite differences.
4. Test: `grad(blockMap({halo: [[1,1],[1,1]]}))` on 2D 3×3 stencil matches finite differences.
5. Test: `jit(grad(blockMap({halo})))` on WASM and WebGPU.

**Step 2: Stencil body synthesis for linear stencils** (new `stencil-analysis.ts`)

A domain-specific compiler pass that synthesizes a gather-based backward body for linear stencil
bodies. NOT `transposeJaxpr` reuse — this constructs a new jaxpr.

1. Analyze the forward body jaxpr: walk equations for `{UncheckedDynamicSlice, Mul(scalar), Add}`.
   If the body contains only these ops (linear stencil), it is eligible.
2. For each `UDS(input, [offH, offW], [B, B])` with weight `w`, emit reversed
   `UDS(ct_tile, [K-1-offH, K-1-offW], [B, B]) * w` in the synthesized backward body.
3. Compute backward halo: symmetric → same as forward; asymmetric → `[hi, lo]` per axis.
4. Emit the synthesized backward body as a halo block_map with `(B+bwdHalo) → B` contract.
5. Test: symmetric halo `[[1,1]]` backward matches finite differences (no body analysis needed).
6. Test: asymmetric halo `[[0,2]]` (causal stencil) backward matches finite differences.
7. Test: non-linear body (relu in stencil) falls back to Step 1 scatter-add.

**Step 3: Detect and route** (`linearize.ts`)

Wire Steps 1 and 2 together in the BlockMap transpose rule:

1. If no `halo` → existing path unchanged.
2. If `halo` present → try stencil analysis (Step 2). If eligible → synthesized gather body.
3. If analysis fails → scatter-add fallback (Step 1) + diagnostic.

##### Affected files (complete list)

- `src/library/lax-block-map.ts` — API, resolution, block aval expansion _(C.1)_
- `src/frontend/core.ts` — `Primitive.BlockMap` params type _(C.1)_
- `src/frontend/array.ts` — eager block*map execution (halo slicing + padding) *(C.1)\_
- `src/frontend/jit.ts` — block*map step type, halo in JitStep *(C.2)\_
- `src/frontend/block-map-executor.ts` — WebGPU zero-copy, JS fallback halo-aware slicing _(C.2)_;
  delete `prePadForWasmHalo`, simplify `tryExecuteBlockMapWasm` _(C.2b)_
- `src/backend/webgpu/block-map.ts` — signed bounds-checked reads in `gen_resolve()` _(C.2)_
- `src/backend/wasm.ts` — `BlockMapWasmParams.halo` field, `codegenBlockMapLoop()` halo-aware input
  copy _(C.2b)_
- `src/frontend/jvp.ts` — forward `halo` in JVP rule _(C.1)_
- `src/frontend/linearize.ts` — forward `halo` in PE + transpose rules; scatter-add fallback +
  stencil synthesis routing _(C.4)_
- `src/frontend/stencil-analysis.ts` — (new, Phase C.4 Step 2) linear stencil body synthesis for
  gather-based halo VJP
- `src/frontend/vmap.ts` — forward `halo` in vmap rule _(C.1)_
- `src/index.ts` — no new exports needed (halo is part of existing `BlockMapOptions`)
- `test/block-map.test.ts` — halo-specific test cases T8.1–T8.8 _(C.1–C.2)_

### Correctness and regression coverage

Must preserve all existing behavior in `test/conv.test.ts`, plus add targeted cases for optimized
paths:

1. 1×1 conv fast path matches generic path
2. 3×3 SAME optimized path matches generic path
3. 5×5 SAME optimized path matches generic path
4. Grouped conv continues to use correct fallback path
5. `jit`, `grad`, and `vmap` behavior remain correct for optimized cases
6. **Path-selection tests** using `_lastConvClass()` (Phase A0): assert that eligible shapes take
   the intended fast path and ineligible shapes (groups, dilation, exotic padding) fall through to
   `generic-dot`. This replaces the fragile approach of inferring activation from performance deltas
   or console output.
7. WebGPU-only code capture tests confirm that `block_map` conv emits the expected fused shader
   (single dispatch, correct grid) rather than falling back to per-block dispatch
8. REPL compiled-code panel displays captured WGSL and WAT source when "Capture compiled code" is
   enabled

### Success criteria

1. Add benchmark coverage in `bench/conv2d.bench.ts` ✅
2. No regressions in `test/conv.test.ts` ✅
3. Path classification (block-map-3x3/5x5) in `classifyConv` ✅
4. 1×1 conv fast path ✅ (Phase B: `fast-1x1-dot` and `fast-1x1-block-map`)
5. Clear WebGPU speedup for 3×3/5×5: **partial** — C.3 WASM shows 1.02–1.36× speedup (grows with
   spatial size); WebGPU fused shader pending
6. Intel path remains correct ✅

### Recommended execution order

1. ~~Phase A0 conv lowering kind signal~~ ✅ Done (commit f1663c7)
2. Phase A0 `setCodeCapture` public API + wasmblr trace mode — WebGPU hook first, then WASM kernel +
   mega-module hooks (trace mode enables WAT source collection at all WASM sites)
3. ~~Phase A benchmark file + baseline numbers~~ ✅ Done (commit 83722e9)
4. Phase A0 items 2 + 4 (detailed profiling + REPL compiled-code panel) in parallel with Phase A
   analysis. WASM scan/assoc-scan/block-map/routine hooks land incrementally after initial panel
5. ~~1×1 fast path~~ ✅ Done (commit 83722e9, Phase B)
6. ~~3×3/5×5 classification~~ ✅ Done (commit 83722e9)
7. ~~3×3 im2col + tiledMatmul~~ ❌ Rejected (41% regression from materialization overhead)
8. ~~Halo-aware block_map design~~ ✅ Done (Phase C design: `halo` option + pre-pad strategy)
9. ~~Phase C.1: API + tracing + eager halo support~~ ✅ Done
10. ~~Phase C.2: JIT zero-copy halo (WebGPU + JS fallback + WASM pre-pad interim)~~ ✅ Done
11. ~~Phase C.2b: WASM compiled halo (zero-copy) — replace `prePadForWasmHalo` with compiled address
    generation~~ ✅ Done
12. ~~Phase C.3: Conv2d body + integration + benchmarks~~ ✅ Done
13. Phase C.4: Halo VJP (scatter-add fallback first → stencil body synthesis for linear bodies)

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
