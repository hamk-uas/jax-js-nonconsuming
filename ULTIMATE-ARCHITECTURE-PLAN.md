# The Ultimate Architecture: Fusion, Atomics, and Mega-Modules

**Branch:** `ultimate-architecture-plan` **Scope:** Transform `jax-js-nonconsuming` into a
fully-optimized, hardware-saturating compute engine. This plan eliminates the JS ↔ Native dispatch
boundary, minimizes VRAM bandwidth via aggressive fusion, introduces WebGPU atomics for
`scatter_add`, and parallelizes the Wasm backend via `SharedArrayBuffer`. **Execution model:**
Single autonomous, tireless coding agent.

---

## Motivation

With AOT Linearization (Option B) and Effect-Typed IR (Option A) complete, the library is
memory-safe, deterministic, and structurally sound. However, it currently leaves massive hardware
performance on the table due to four architectural bottlenecks:

1. **The Functional Blocker (`scatter_add`):** Gradients for `take`/`gather` with duplicate indices
   overwrite each other. Without `scatter_add` (which requires hardware atomics), embedding layers
   and Graph Neural Networks are mathematically impossible to train.
2. **The VRAM Bandwidth Bottleneck (Opaque Routines):** Routines like `matmul` cannot fuse with
   subsequent elementwise operations (e.g., `relu`). This forces the GPU to write the intermediate
   matrix to VRAM and immediately read it back, starving the compute units.
3. **The JS ↔ Native Boundary (Wasm Overhead):** A JIT program with 50 fused kernels crosses the JS
   ↔ Wasm boundary 50 times. For small/medium arrays, this dispatch overhead dominates execution
   time.
4. **Single-Threaded Wasm:** The Wasm backend currently runs on a single CPU thread, ignoring modern
   multi-core browser capabilities.
5. **JIT Compilation Overhead for Variable-Length Data:** Currently, array dimensions are baked into
   the JIT compilation. If a time series length changes, the entire program must be retraced and
   recompiled, causing severe latency spikes in dynamic workloads like trend analysis.

This plan resolves all five bottlenecks, culminating in a "Mega-Module" Wasm compiler, polymorphic
shapes, and a highly-fused WebGPU pipeline.

---

## Milestones

### M0 — Baseline Snapshot & Feature Detection (1–2 days)

Establish baselines and implement hardware feature detection for the advanced capabilities required
by this plan.

#### M0.1 — Record Baseline Test Results

**What:** Run the full test suite and Deno tests, record pass/fail counts and benchmark timings.
**Exit criteria:** Baseline files exist in `tmp/` with full logs, exit codes, and benchmark results.

#### M0.2 — Hardware Feature Detection

**What:** Implement robust feature detection for:

1. `crossOriginIsolated` (required for `SharedArrayBuffer` and Wasm threads).
2. WebGPU `shader-f32-atomic-add` extension (to determine if we need a software CAS loop for `f32`
   atomics). **Exit criteria:** `backend.ts` exposes these capabilities cleanly to the JIT compiler.

---

### M1 — The Missing Primitive: `scatter_add` & Atomics (4–6 days)

Leverage the `Mutate` effect to implement safe, in-place atomic additions, unlocking embedding
gradients.

#### M1.1 — `Primitive.ScatterAdd` & AD Rules

**What:** Define `Primitive.ScatterAdd`.

- **Effects:** Target buffer is `Mutate`, indices and updates are `Borrow`.
- **AD:** Implement the JVP and Transpose rules for `take` (which will now emit `scatter_add` in the
  backward pass) and `scatter_add` itself. **Exit criteria:** Jaxpr tracing for `scatter_add` passes
  the static Borrow Checker.

#### M1.2 — WebGPU Atomics (The CAS Loop)

**What:** Implement `scatter_add` in WGSL.

- If `shader-f32-atomic-add` is available, use it.
- Otherwise, implement a Compare-And-Swap (CAS) loop using `atomicCompareExchangeWeak` and
  `bitcast<u32>` to safely accumulate `f32` values across thousands of threads. **Exit criteria:**
  `scatter_add` produces correct results on WebGPU, even with highly duplicated indices.

#### M1.3 — Wasm `scatter_add`

**What:** Implement the sequential Wasm version of `scatter_add` using `wasmblr`. **Exit criteria:**
Wasm backend passes all `scatter_add` tests.

---

### M2 — Multi-Output Kernels & Epilogue Fusion (5–7 days)

Eliminate VRAM round-trips by blurring the line between Kernels and Routines.

#### M2.1 — Multi-Output `Kernel` Support

**What:** Upgrade the `Kernel` class and `splitGraphDataflow` to support fusing operations that
produce multiple outputs (e.g., complex AD backward passes). **Exit criteria:** The JIT compiler
successfully emits single WGSL/Wasm kernels that write to multiple output buffers simultaneously.

#### M2.2 — Routine Epilogue Fusion

**What:** Modify the `Routine` interface to accept an optional `epilogue: Jaxpr`.

- Update `matmul` and `cholesky` WGSL/Wasm generators to evaluate the epilogue Jaxpr _in registers_
  before writing the final value to VRAM. **Exit criteria:** `jit(x => relu(matmul(A, B) + bias))`
  compiles into a **single** WebGPU dispatch and a **single** Wasm call.

---

### M3 — Polymorphic Shapes (Dynamic Dimensions) (4–6 days)

Eliminate JIT recompilation for variable-length data by allowing users to specify which axes are
dynamic (symbolic) rather than constant-size.

#### M3.1 — Symbolic Shape IR & Tracing

**What:** Update `ShapeTracker` and `Jaxpr` to support symbolic dimensions (e.g., `["T", 64]`).

- Add an opt-in API to `jit`: `jit(f, { dynamic_axes: { 0: "time" } })`.
- Update `effectDrivenAllocate` to emit `malloc` steps with symbolic size formulas instead of static
  byte counts. **Exit criteria:** Jaxpr tracing successfully propagates symbolic dimensions through
  standard operations without throwing shape mismatch errors.

#### M3.2 — Parameterized Backend Codegen

**What:** Update Wasm and WebGPU codegen to accept dynamic dimensions as runtime arguments.

- **Wasm:** Pass symbols as Wasm function parameters. Disable loop unrolling for dynamic axes.
- **WebGPU:** Pass symbols via a dedicated Uniform buffer. Compute `dispatchWorkgroups` dynamically
  in JS before submission. **Exit criteria:** A JIT-compiled function can be called with `[100, 64]`
  and `[150, 64]` inputs without triggering a recompilation, producing correct results on both
  backends.

---

### M4 — Wasm Multithreading Foundation (5–7 days)

Saturate the CPU by parallelizing the Wasm backend.

#### M4.1 — `SharedArrayBuffer` Memory Pool

**What:** Upgrade `WasmAllocator` to use `WebAssembly.Memory({ shared: true })` when
`crossOriginIsolated` is true. **Exit criteria:** The Wasm backend functions correctly
(sequentially) using shared memory.

#### M4.2 — `WasmWorkerPool`

**What:** Implement a pool of Web Workers initialized at backend startup.

- Workers instantiate the same Wasm modules and share the same memory.
- Implement a lightweight synchronization primitive using `Atomics.wait` and `Atomics.notify`.
  **Exit criteria:** Worker pool initializes cleanly and can execute basic tasks.

#### M4.3 — Parallel `wasmblr` Loops

**What:** Add `parallelForLoop` to `wasmblr-hl.ts`.

- The main thread divides the loop range by the number of workers, writes the ranges to a shared
  control block, and wakes the workers. **Exit criteria:** Large elementwise operations and
  reductions show near-linear speedup on multi-core CPUs.

---

### M5 — Whole-Program Wasm Compilation (The "Mega-Module") (6–8 days)

Eliminate the JS ↔ Wasm boundary entirely for JIT programs.

#### M5.1 — `JitProgram` to Wasm Translator

**What:** Instead of executing `JitStep`s in a JS loop, write a compiler pass that translates the
entire `JitProgram` into a single `wasmblr` module.

- `malloc` and `recycle` steps become dynamic pointer arithmetic inside the Wasm module's local
  state, evaluating the polymorphic shape formulas (from M3) at runtime.
- `execute` steps (Kernels and Routines) are inlined as Wasm function calls within the module.
  **Exit criteria:** A JIT program with 50 operations compiles to 1 Wasm module and executes with 1
  JS call.

#### M5.2 — Mega-Module Multithreading

**What:** Ensure the Mega-Module correctly dispatches parallelizable loops to the `WasmWorkerPool`
without returning to JS. **Exit criteria:** Complex neural network forward passes execute entirely
in native Wasm, utilizing all CPU cores, with zero JS overhead.

---

### M6 — First-Class `associativeScan` (4–6 days)

Fix the O(N) WebGPU bottleneck and O(log N) Wasm overhead for parallel prefix scans.

#### M6.1 — `Primitive.AssociativeScan`

**What:** Promote `associativeScan` from a high-level unrolled function to a core `Primitive`.

- Implement JVP and Transpose rules (which will emit transposed `associativeScan` primitives rather
  than exploding into massive Jaxprs). **Exit criteria:** `grad(associativeScan)` produces a
  compact, O(1) depth Jaxpr.

#### M6.2 — Native `associativeScan` Compilers

**What:**

- **WebGPU:** Write a dedicated multi-dispatch orchestrator that executes the Kogge-Stone doubling
  loop without JS-side Jaxpr unrolling.
- **Wasm:** Write `codegenNativeAssociativeScan` that compiles the entire Kogge-Stone ladder
  (including ping-pong buffer swaps) into a single Wasm loop. **Exit criteria:** `associativeScan`
  performance matches or exceeds `lax.scan` on Wasm, and `grad(scan)` over linalg bodies can be
  cleanly reformulated by users to avoid WebGPU O(N) bottlenecks.

---

### M7 — Cleanup, Benchmarking & Documentation (2–3 days)

#### M7.1 — Benchmark Suite

**What:** Create a comprehensive benchmark suite comparing M0 baselines to M6.

- Focus on: Embedding layer backward pass (`scatter_add`), `matmul+relu` fusion, and Wasm
  Mega-Module latency. **Exit criteria:** Benchmark report generated and saved to
  `docs/ULTIMATE-BENCHMARKS.md`.

#### M7.2 — Documentation

**What:** Update `copilot-instructions.md` to document:

- How to write Epilogue-fused Routines.
- The Wasm Mega-Module architecture and Worker Pool.
- The `scatter_add` primitive and WebGPU CAS loops.
- Polymorphic shapes and dynamic dimensions. **Exit criteria:** Documentation reflects the new
  architecture.

#### M7.3 — Final Regression Run

**What:** Full CI-equivalent check. **Exit criteria:** All checks pass. Zero regressions from M0
baseline.

---

## Dependency Graph

```
M0.1 (baseline) ──→ M0.2 (feature detection)
  │
  ├─→ M1.1 (scatter_add IR) ──→ M1.2 (WebGPU CAS) ──→ M1.3 (Wasm scatter)
  │
  ├─→ M2.1 (Multi-output) ──→ M2.2 (Epilogue Fusion)
  │
  ├─→ M3.1 (Symbolic Shape IR) ──→ M3.2 (Parameterized Codegen)
  │                                      │
  ├─→ M4.1 (SharedArrayBuffer) ──→ M4.2 (Worker Pool) ──→ M4.3 (Parallel Wasm)
  │                                                              │
  │                                                              ↓
  └─→ M5.1 (Mega-Module Compiler) ─────────────────────────→ M5.2 (Mega-Module + Threads)
                                                                 │
  M6.1 (assocScan Primitive) ──→ M6.2 (Native assocScan) ────────┤
                                                                 │
                                                                 ↓
                                                          M7.1 - M7.3 (Cleanup)
```

## Risk Register

| Risk                                                    | Impact                             | Mitigation                                                                                                         |
| ------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `crossOriginIsolated` not available in user environment | Wasm falls back to single-thread   | Graceful degradation: `WasmAllocator` falls back to standard `ArrayBuffer`; Mega-Module executes sequentially.     |
| WebGPU CAS loop for `f32` is too slow                   | `scatter_add` bottlenecks training | Optimize CAS loop; rely on `shader-f32-atomic-add` where available; batch updates if possible.                     |
| Mega-Module compilation time is too high                | JIT latency spikes                 | Cache Mega-Modules aggressively; use `wasmblr`'s fast generation; defer heavy optimizations to background threads. |
| Epilogue fusion register pressure                       | WebGPU/Wasm register spilling      | Limit epilogue complexity; fallback to standard VRAM round-trip if epilogue exceeds heuristic cost.                |
| Polymorphic shapes break routine unrolling              | Wasm routines slow down            | Keep static sizes as the default; only use dynamic loops for explicitly marked dynamic axes.                       |

## Estimated Timeline

| Milestone | Effort   | Cumulative |
| --------- | -------- | ---------- |
| M0        | 1–2 days | 1–2 days   |
| M1        | 4–6 days | 5–8 days   |
| M2        | 5–7 days | 10–15 days |
| M3        | 4–6 days | 14–21 days |
| M4        | 5–7 days | 19–28 days |
| M5        | 6–8 days | 25–36 days |
| M6        | 4–6 days | 29–42 days |
| M7        | 2–3 days | 31–45 days |

Total: **5–7 weeks** of focused implementation by a tireless agent.

## Commit Strategy

- One commit per task (M0.1, M0.2, ..., M7.3).
- Commit message format: `ultimate M{n}.{m}: {short description}`
- Every commit must pass `pnpm vitest run`.
- Branch off `main` at start. Merge back after M7.3 passes full CI.
