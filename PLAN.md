# Open Work

Items that are currently in progress, deferred, or blocked. Completed items have been consolidated
into `.github/copilot-instructions.md`.

## Open Performance Items

| ID  | Title                      | Priority        | Description                                                                              |
| --- | -------------------------- | --------------- | ---------------------------------------------------------------------------------------- |
| P2  | Relaxed SIMD FMA           | Medium          | `f32x4.relaxed_madd` for 2× WASM dot-product throughput. Safari unsupported              |
| P3  | i64 in wasmblr             | Medium          | Native i64 (WASM MVP). Simplifies Threefry PRNG, unlocks f64 builtins                    |
| P4  | Conv2d WebGPU fused shader | Medium          | Fused-shader codegen for conv bodies in `block-map.ts`. WASM path done (1.02–1.36×)      |
| P6  | Benchmark validation       | Medium          | Systematic benchmarks: matmul GFLOP/s, conv2d, SIMD chains, reductions                   |
| P7  | Cooperative matrix (WMMA)  | Blocked (~2026) | Hardware tensor cores for 2–4× tiled matmul. WGSL spec not yet stable                    |
| P8  | Command tape block_map     | Low             | Add `block_map` to WebGPU command tape to unlock `foriLoop→blockMap` in `Scan`           |
| P9  | Continuous Cost Modeling   | High            | Migrate heuristic logic in `lax.ts`, `tuner.ts`, and `scan-plan.ts` (Phase 2 & 3)        |
| P10 | Microbenchmark Auto-Tuning | High            | Replace static profiles with Bayesian updates of hardware characteristics via microbench |

## P4: Conv2d WebGPU Fused Shader

The Conv→BlockMap rewrite (Phase C.3) works on WASM but is gated off on WebGPU because `block_map`
falls back to per-block dispatch for conv bodies (no fused shader yet), which is slower than
generic-dot. To unblock:

1. Extend fused-shader codegen in `src/backend/webgpu/block-map.ts` to handle conv body patterns
2. Verify single-dispatch conv via `_lastConvRewritten()` + `profileGpuDetailed()`
3. Benchmark against generic-dot on both NVIDIA and Intel
4. Remove the `backend ≠ webgpu` guard in `rewriteConvToBlockMap()`

### Conv2d baseline data (March 2026)

All cases compile to exactly **1 kernel** (fused im2col + matmul + epilogue).

**WebGPU — NVIDIA RTX 4070 Ti SUPER (TB3 eGPU):**

| Case              | GFLOP | Eager ms | JIT ms | GFLOP/s | % Peak |
| ----------------- | ----- | -------- | ------ | ------- | ------ |
| 3×3 1×32ch 64×64  | 0.151 | 2.82     | 2.64   | 57.3    | 0.3%   |
| 3×3 1×64ch 64×64  | 0.302 | 3.03     | 2.22   | 136.1   | 0.6%   |
| 3×3 1×128ch 32×32 | 0.302 | 3.04     | 2.53   | 119.4   | 0.5%   |
| 3×3 8×64ch 64×64  | 2.416 | 3.70     | 2.77   | 872.6   | 3.9%   |
| 3×3 8×128ch 64×64 | 9.664 | 9.40     | 7.20   | 1342.0  | 5.9%   |

**Key finding:** WebGPU conv2d is dispatch-bound at ~2.5ms for single-batch shapes (<1% GPU
utilization). Larger batches (8×128ch) reach 5.9% peak. Im2col materialization was attempted and
rejected (41% regression from bandwidth overhead). Fused-shader conv needs to eliminate dispatch
overhead to show speedup.

**WASM block_map speedup (vs generic-dot):**

| Size          | block_map | generic-dot | Speedup   |
| ------------- | --------- | ----------- | --------- |
| 3×3 4ch 16×16 | 162 µs    | 166 µs      | 1.02×     |
| 3×3 4ch 32×32 | 500 µs    | 643 µs      | **1.29×** |
| 3×3 8ch 64×64 | 7,504 µs  | 10,192 µs   | **1.36×** |

## P9: Continuous Cost Modeling

**Status:** Phase 2 complete. Phase 2b (aggregate register pressure model) addresses gen-9 igp
regression. The runtime model drives JIT path selection natively.

**Objective:** Replaced ad-hoc heuristic thresholds with continuous execution cost penalties using
the `evaluateTotalCost` logic. Legacy string heuristics across the JIT orchestrator have been
retired:

- **`lax.ts` (`chooseTileConfig`)**: Uses evaluated cost parameters with per-vendor `rOptWords` and
  aggregate workgroup register pressure to prevent catastrophic register spills on small-GRF GPUs
  (gen-9, mobile).
- **`jit.ts` (`foriLoopToBlockMap`)**: Selects `blockShape` bounds across search space by simulating
  dispatch constraints.
- **`tuner.ts`**: Cooperative group sizes and local tiles execute total cost calculations.
  `dangerAggregate` penalty models `threads × depthPriv / aggregateBudget` cliff.
- **`scan-plan.ts`**: Candidates dynamically scaled via associative cost array prior to probing
  runtime compilation.
- **`backend.ts` / `webgpu.ts`**: `rOptWords` plumbed from runtime model `R_opt_words` through
  `BackendCapabilities`.

## P10: Microbenchmark-Driven Auto-Tuning

**Status:** Phase 1 complete — core infrastructure implemented. Phase 2 (Async JIT & Tuning
Migration) in planning.

### Architectural Flaws to Address (Phase 2):

1. **Microbenchmark Scale**: Workloads are currently too small (e.g., ~32MB payload). Overhead from
   `queue.submit` dominates the timing, skewing the metrics (`BW_global`, `TFLOPS`) to a fraction of
   their correct values.
2. **Eager Initialization**: Calibration synchronously blocks the `init("webgpu")` pipeline (taking
   160-400ms), violating the intended lazy-trigger design.
3. **Ghost Metric - `barrierCostFactor`**: Implemented during benchmarking but silently ignored by
   the `evaluateTotalCost` JIT cost model.
4. **State Leakage in Tests**: `_setCalibrationState("off")` flips a boolean but leaves poisoned
   metrics in `BackendCapabilities`, breaking isolation for subsequent tests.

### Phase 2: Async JIT and Target Architecture

**Objective:** Stop trusting static, init-time GPU classification heuristics as the absolute truth.
Migrate to a three-layer profiling system, driven by an **Async JIT** architecture.

**The Case for Async JIT:** Currently, `jit` compiled executions and WebGPU pipeline creations
(`device.createComputePipeline`) operate synchronously. This blocks the main thread, causing severe
UI jank, and prevents integration of large-scale, asynchronous microbenchmark tuning directly into
the JIT lifecycle.

- **Goal**: Transition from synchronous pipeline compilation to asynchronous
  (`createComputePipelineAsync`). JIT calls should return immediately by leveraging lazy execution
  or returning Promise-wrapped execution handles.
- **Impact**: Calibration can correctly trigger "just-in-time" on the first async JIT pass without
  bottlenecking startup script execution.

1. **Refactor Workload Methodology**: Increase payload volume for `BW_global` and `Tflops`
   iteratively until the irreducible `queue.submit` overhead drops below a 5% margin of error, or
   subtract a baseline linear fit.
2. **Clarify Non-Calibrated Mode**: Keep pre-calibration behavior limited to real device constraints
   plus conservative built-in cost defaults unless a true static hardware prior is implemented and
   wired at runtime.
3. **Plumb the Ghost Metric**: Wire `barrierCostFactor` actively into `tuner.ts` `evaluateTotalCost`
   so that `dangerShmem` or shared-memory sync operations accurately model empirical reality.
4. **Fix Test Isolation**: Ensure `_setCalibrationState()` completely restores the
   conservative-default pre-calibration state, not just the `calibrated: false` boolean.

## Deferred Items

### O6: Multi-Reduction Kernels

**Status:** Deprioritized.

Fuse independent same-size reductions sharing inputs into a single dispatch. Two strategies analyzed
(multi-output + chained reduction), both require new AluExp nodes. Block-map fused shader already
handles compose bodies — O6 only helps standalone JitPrograms with multiple independent same-input
reductions (narrow use case).

### O8d: fori_loop Unrolling into Command Tape

**Status:** Not started. Low priority.

Unroll `fori_loop` iterations into the command tape dispatch sequence. Currently, `fori_loop` with
concrete bounds is supported by WASM mega-module but rejected by the WebGPU command tape.

### Decoupled Fallback Phase 2: General Bodies

**Status:** Deferred. Phase 1 (scalar f32 add/mul/min/max) is complete and ships.

Extend the single-dispatch O(N) prefix scan to arbitrary associative functions (pytree bodies,
matrix compose). Requires packed representation or multi-field descriptor for general reduction
values.

**Phase 1 remaining opportunities:** Subgroup-parallel lookback; raking pattern (multiple
elements/thread).

**References:**

- Smith, Levien, & Owens — "Decoupled Fallback" (2024):
  https://github.com/b0nes164/Decoupled-Fallback-Paper
- GPUPrefixSums — production WGSL implementation: https://github.com/b0nes164/GPUPrefixSums

## Non-Goals

- **General einsum rank-adaptation:** Only matters for exotic subscript patterns not covered by
  `einsumFastPath`. All common batch-matmul patterns already work.
- **Sort/Argsort/LU jaxprification:** Fundamentally data-dependent algorithms (comparison-based,
  data-dependent pivoting). Cannot be expressed as fixed arithmetic traces.
- **GPU/WASM ratio ≤ 2× for small-matrix DLM:** Irreducible GPU API costs (`queue.submit`) prevent
  matching WASM mega-module latency at small N.
- **Binding limit optimization on high-limit hardware:** On Deno/NVIDIA with `maxArgs = 1,048,575`,
  the P2 split pass has no effect. Browser `maxArgs ≈ 9` causes additional fragmentation, but
  dominant overhead is P1 structural rules.
