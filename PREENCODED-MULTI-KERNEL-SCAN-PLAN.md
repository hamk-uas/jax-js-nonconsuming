# Control-Flow Lowering & Scan Performance Plan

**Status:** Phase 0+1 complete, Phases 2–4 draft **Date:** 2026-02-26 **Supersedes:** v3 (review
feedback integrated), v2 (extended fusion + pre-encoded), v1 (pre-encoded multi-kernel only)
**Integrates:** Useful elements from `LOOP-PRIMITIVE-RESTRUCTURING-PLAN.md`, review feedback from
`UPDATED-PLAN-REVIEW.md` (formal fusion safety contract, nested-length compatibility matrix,
benchmark protocol, scope trimming)

---

## 1. JAX Control-Flow Primitive Architecture

JAX represents control flow as **Jaxpr-level primitives** with nested sub-jaxprs in their params —
not as a separate IR layer:

| JAX Primitive      | Params contain             | jax-js status                  | Notes                         |
| ------------------ | -------------------------- | ------------------------------ | ----------------------------- |
| `scan`             | `body_jaxpr`               | `Primitive.Scan` ✅            | Sequential carry threading    |
| `associative_scan` | `body_jaxpr`               | `Primitive.AssociativeScan` ✅ | Parallel prefix (Kogge-Stone) |
| `while_loop`       | `cond_jaxpr`, `body_jaxpr` | Not yet                        | Dynamic iteration count       |
| `cond` / `switch`  | branch jaxprs              | Not yet                        | Data-dependent branching      |

jax-js already follows this pattern. `Primitive.Scan` carries a `body_jaxpr` sub-jaxpr. The JIT
compiler lowers it to a `"scan"` JitStep containing `bodyProgram: JitProgram` (the compiled
sub-program). The backend executes this step based on a `ScanPlan`.

**Guiding principle:** The work in this plan is NOT about adding new IR constructs. It is about
improving how the backend **lowers** the existing `"scan"` JitStep for WebGPU, establishing
infrastructure that extends naturally to nested loops and future primitives (`while_loop`, `cond`),
and doing structural housekeeping to consolidate execution paths.

---

## 2. Problem Statement

The WebGPU compiled-loop scan (`nativeScanMultiShaderSource`) compiles the entire N-iteration ×
S-step body into a **single WGSL shader** dispatched once. Each thread handles one `gidx` across all
iterations with all S steps inlined per thread. This is dispatch-optimal: **1 dispatch, 1 submit**.

Previously the compiled-loop rejected bodies with internal buffer dependencies, cross-step carry RAW
hazards, carry passthrough, numCarry ≠ numY, routine steps, and inner loop steps. **Phase 1 (commit
74ddb7e) removed the first three restrictions** via carry snapshot locals + internal intermediate
locals. The remaining restrictions are:

- ~~Internal buffer dependencies between steps~~ ✅ Phase 1
- ~~Cross-step carry read-after-write hazards~~ ✅ Phase 1
- ~~Carry passthrough (carry unchanged)~~ ✅ Phase 1
- numCarry ≠ numY
- Routine steps (Sort, Cholesky, TriangularSolve)
- Inner loop steps (`scan`, `assoc_scan`)

`grad(scan)` backward passes systematically violate these constraints because the chain rule for
`f(g(x))` produces sequential dependency chains:

```
Forward body:  B = f(A, x);  C = g(B, x)
Transposed:    dA = f_T(dB);  dB = dB + g_T(dC)   ← sequential dependency chain
```

Additionally, nested loops (scan-inside-scan, assocScan-inside-scan) always fall back on both
backends. The planner filters to execute steps only — non-execute steps (including inner
`scan`/`assoc_scan` JitSteps) are invisible to the planner and cause fallback.

Result: the fallback JS loop with O(N/256) `queue.submit()` calls. For N=1600, this causes ~1 s
latency per call.

---

## 3. Design Principles

1. **No dispatch regression.** Bodies that currently compile to 1 dispatch stay at 1 dispatch.
2. **Reduce dispatches where possible.** Extended fusion reduces qualifying fallback bodies from N×S
   dispatches to **1 dispatch**.
3. **Reduce submits where fusion fails.** Pre-encoded dispatch reduces from O(N/256) submits to **1
   submit** for bodies that cannot be fused.
4. **Follow JAX.** Control flow stays at the Jaxpr primitive level with sub-jaxprs.
5. **Reusable infrastructure.** Backend encoding methods work for scan, nested loops, while_loop,
   and cond.
6. **Structural evolution by need.** Generalize planning/execution infrastructure when the third
   loop primitive arrives, not before. Design Phases 1–3 so generalisation is easy but don't pay for
   it yet.

---

## 4. Tiered Scan Lowering for WebGPU

### Tier 1: Fused Shader — 1 dispatch, 1 submit (DISPATCH OPTIMAL)

Existing `nativeScanMultiShaderSource()` inlines all body kernel steps into one WGSL shader: each
thread handles one `gidx`, loops N times, executes all S steps per iteration inline.

**Current eligibility:** kernel-only, no internal deps, no carry RAW hazards, no passthrough,
numCarry === numY.

**Extended eligibility (Phase 1, ✅ complete — commit 74ddb7e):** Carry snapshot locals + internal
intermediate locals. Bodies with same-gidx internal deps, carry RAW hazards, and carry passthrough
are now fusable — **1 dispatch** instead of N×S fallback. numCarry ≠ numY support deferred.

### Tier 2: Pre-encoded Command Buffer — N×S dispatches, 1 submit (GENERAL)

Encode all N iterations × S steps into a single `GPUCommandEncoder`. WebGPU guarantees sequential
execution within a command buffer. Zero JS involvement after submit.

For bodies that **cannot** be fused: routines, reductions with cross-element internal deps (e.g.,
step j reads step i's matmul output at different gidx), mixed kernel+routine, nested scans.

### Tier 3: JS Fallback — N×S dispatches, O(N/256) submits (LAST RESORT)

Existing path. JS loop calling `bodyProgram.execute()` per iteration with command batching. For
bodies with dynamic control flow or non-encodable patterns.

---

## 5. Dispatch Count Analysis

| Body type                    | Current path   | Dispatches | Submits  | New path        | Dispatches | Submits |
| ---------------------------- | -------------- | ---------- | -------- | --------------- | ---------- | ------- |
| Elementwise, no deps         | compiled-loop  | **1**      | 1        | Same            | **1**      | 1       |
| Elementwise, internal deps   | fallback       | N×S        | O(N/256) | Extended fusion | **1**      | **1**   |
| Elementwise, carry RAW       | fallback       | N×S        | O(N/256) | Extended fusion | **1**      | **1**   |
| Carry passthrough            | fallback       | N×S        | O(N/256) | Extended fusion | **1**      | **1**   |
| numCarry ≠ numY              | fallback       | N×S        | O(N/256) | Extended fusion | **1**      | **1**   |
| Reduction, same-gidx deps    | fallback       | N×S        | O(N/256) | Extended fusion | **1**      | **1**   |
| Reduction, cross-gidx deps   | fallback       | N×S        | O(N/256) | Pre-encoded     | N×S        | **1**   |
| Single routine               | preencoded-rtn | N×P        | 1        | Same            | N×P        | 1       |
| Mixed kernel+routine         | fallback       | N×S        | O(N/256) | Pre-encoded     | N×S        | **1**   |
| Nested scan (inner compiled) | fallback       | N×(S+M)    | O(N/256) | Pre-encoded     | N×(1+S')   | **1**   |
| Nested scan (inner fallback) | fallback       | N×M×S      | O(N/256) | Pre-encoded     | N×M×S      | **1**   |
| scan(assocScan(kernel))      | fallback       | N×…        | O(N/256) | Pre-encoded     | N×⌈log₂M⌉  | **1**   |

**Key improvements:**

- Extended fusion reduces qualifying bodies from N×S dispatches to **1 dispatch** (rows 2–6).
- Pre-encoded reduces remaining fallback bodies from O(N/256) submits to **1 submit** (rows 7–12).
- Nested loops benefit from pre-encoded: inner compiled-loop scans become single dispatches within
  the outer command buffer; assocScan rounds become ⌈log₂M⌉ fused dispatches.
- Zero regression for any currently compiled-loop body (row 1).

---

## 6. Implementation Phases

### Phase 0: Ownership-Correctness Gate (prerequisite, complete)

Before any scan-lowering work begins, the ownership / ref-balance layer must be stable for the
compositions that scan lowering exercises: `jit(valueAndGrad(scan(...)))`, `grad(scan(linalg))`,
nested `scan(assocScan(...))`, and cache cleanup paths (`clearCaches`, `checkLeaks.stop`).

Key fix landed: `partialEvalGraphToJaxpr` Const PETracer cleanup now drains inflated `#rc`
(`while (t.isAlive) t.dispose()`) so the cascade to `recipe.val.dispose()` fires exactly once.
Without this, multi-output equations (Scan with N carry+Y outputs) inflated PETracer `#rc` via
`processPrimitive`, leaving `instantiateConst`'s `.ref` unbalanced and blocking
`_anonymousExtraDispose` (rc > 1 when builderRefs reached 0). See copilot-instructions.md Contract 2
"Multi-output PETracer `.ref` inflation" and Contract 3.

**Gate criterion:** `pnpm vitest run` passes with zero leak failures across all 60 test files,
including `transform-compositions.test.ts`, `lax-scan.test.ts`, and `check-leaks.test.ts`.

**Status:** ✅ Complete.

### Phase 1: Extended Fused Shader — Dispatch Reduction ✅ COMPLETE

**Implemented in commit 74ddb7e** (2026-02-26, ~230 net LOC vs estimated ~320).

Extended `nativeScanMultiShaderSource()` to accept bodies with same-gidx internal deps, carry RAW
hazards, and carry passthrough. numCarry ≠ numY deferred (no current use case).

**Approach: carry snapshot locals + internal intermediate locals.**

The previous shader computed each step and immediately wrote to the `carry` storage buffer. This
caused carry RAW hazards and rejected internal deps. The fix (now implemented) restructures the
shader:

```wgsl
for (var iter: u32 = 0u; iter < N; iter++) {
  let dataIdx = select(iter, N - 1u - iter, REVERSE);  // reverse-aware

  // Snapshot OLD carry values into private vars
  var c_0: f32 = carry0[gidx];    // gidx < carrySize0
  var c_1: f32 = carry1[gidx];    // gidx < carrySize1

  // Compute ALL steps using snapshot carry + private intermediates
  var internal_0: f32 = 0.0;  // non-carry step output
  if (gidx < size0) {
    var result_val_0 = expr0(c_0, xs0[i32(dataIdx) * stride0 + gidx]);
    internal_0 = result_val_0;
  }
  if (gidx < size1) {
    let result_val_1 = expr1(internal_0, c_1, ...);  // reads step0's PRIVATE var ✓
    carry1[gidx] = result_val_1;
    ys1[i32(dataIdx) * ysStride1 + gidx] = result_val_1;
  }

  // Carry writes: carry steps write inline; passthrough carries untouched
  // Ys writes: inline with carry step computation
}
```

**Implementation notes (divergences from the original plan):**

- Plan proposed four separate phases (A: snapshot, B: compute, C: carry write, D: ys write). The
  actual implementation interleaves writes with computation — carry steps write carry+ys inline
  after their result computation, which is simpler and equivalent because snapshot locals eliminate
  RAW hazards regardless of write ordering.
- Plan proposed `var<private>` for all intermediates. Implementation uses `var` locals (function
  scope), which are equivalent in single-thread-per-invocation WGSL.
- Carry snapshot uses `var c_i` (mutable) for single-element carries where index is `"0"`, and
  buffer reads `carry_i[idxCode]` for multi-element carries accessed at non-gidx indices (e.g., dot
  products). This handles the matmul/reduction case correctly.
- Internal (non-carry) step outputs use `var internal_j` declared before the `if (gidx < size)`
  guard to ensure scope visibility for downstream steps.
- External `initCarry → carryOut` copy via `copyBufferToBuffer` before the compute pass, rather than
  a separate initCarry binding + Phase A read. This means the carry buffer is both read and written
  by the shader (read_write access).

**Why this is correct:**

- Carry snapshots read all carry values before any writes → carry RAW hazard eliminated.
- `var` intermediates → internal deps served from thread-local state.
- Each thread only reads its own `gidx` from carry and intermediates → no cross-thread dependency,
  no barrier needed.
- Passthrough carries are never written → they retain their value from the previous iteration.

**Expression rewriting for internal deps:**

When step j's expression references an internal buffer (step i's output), the expression generator
substitutes a private variable reference instead of a buffer read:

```typescript
// In genScanExpressionWithRidx, when encountering gid that maps to step i's output:
function mapGidToSource(gid: number, internalsMap: Map<number, string>): string {
  if (internalsMap.has(gid)) return internalsMap.get(gid)!; // "step0_val"
  // ... existing const/carry/xs mapping
}
```

**Carry passthrough handling:**

If `carry_out[i] = carry_in[i]` (passthrough), no step produces it. Phase C skips this carry — it
retains its OLD value, which is the desired behavior. No copy needed.

**numCarry ≠ numY support:**

With deferred writes, carry writes (Phase C) and ys writes (Phase D) are independent. Y outputs can
reference any step result or any old carry value. The shader emits the correct source for each Y
output based on `yOutputSources`.

**Eligibility check — `canFuseScanSteps()`:**

Uses `classifyBodySteps()` (Phase 4a) internally. For each internal dep:

1. Step i must be a Kernel (no Routine steps can be inlined).
2. The read must be at the **same flat gidx** as the write — guaranteed for elementwise kernels
   (output[gidx] read at [gidx]).
3. For reduction kernels: fusable only if the consumer reads the reduction output at the same `gidx`
   index (e.g., both produce M-element outputs, so thread `gidx` produced the value that thread
   `gidx` reads).
4. **Reject** if the producing step's output size < the consuming step's workload at the referenced
   index — this indicates a cross-thread dep (e.g., scalar reduction broadcast to all threads).
5. **Reject** if any step is a `scan`, `assoc_scan`, or other non-execute step.

Conservative policy: reject if uncertain. A false negative just means fallback (safe). A false
positive would produce wrong results.

**Phase 1 Fusion Safety Contract (formal admission predicate):**

A body step `j` that reads producer step `i`'s output is **same-gidx-safe** if and only if ALL of
the following structural conditions hold:

1. **Producer is a Kernel** — not a Routine, scan, or assoc_scan step.
2. **Size equality** — `producer.kernel.size === consumer.kernel.size`. If the producer's output
   count differs (e.g., scalar reduction), thread `gidx` in the consumer would read a value produced
   by a different thread → cross-thread dep → reject.
3. **Identity index mapping** — the consumer's `AluExp` tree references the producer's output via a
   `GlobalView` or `GlobalIndex` that resolves to the same flat `gidx` as the producer's store. In
   practice, this means no `ShapeTracker` transformations (transpose, reshape) between producer
   output and consumer input. Verified by `AluExp.fold()` — the referenced buffer's access pattern
   must be `buffer[gidx]` (stride-1, offset-0 on the fused axis).
4. **No broadcast** — the consumer does not broadcast the producer's output to a larger size. A
   broadcast means multiple consumer threads read the same producer cell → still safe for
   correctness, but the `var<private>` pattern assumes one-to-one → reject to avoid subtle bugs.
5. **Reduction producer constraint** — if step `i` has a reduction, the consumer must read the
   reduction output (not the pre-reduction intermediate). The reduction output has size =
   `kernel.size / reductionSize`. Same-gidx-safe requires the consumer operates at this reduced
   size. **Reject if the consumer's size ≠ the producer's reduction output size.**

**Proof obligation:** For any body admitted by `canFuseScanSteps()`, the deferred-write shader must
produce bit-identical results to sequential execution of the body steps. This is guaranteed when
each `var<private> step_i_val` holds the exact value that would be in `internal_buffer[gidx]` after
step `i` completes — which follows from conditions 1–5 above ensuring no cross-thread data flow.

**Reject-by-default:** Any pattern not explicitly matched by conditions 1–5 is rejected. The
function returns `false` and the body falls through to Tier 2 (pre-encoded) or Tier 3 (fallback).
New fusable patterns require adding explicit structural rules here, not relaxing the default.

**Files changed (actual):**

| File                        | Changes                                                                | LOC delta    |
| --------------------------- | ---------------------------------------------------------------------- | ------------ |
| `src/frontend/scan-plan.ts` | Rewrote `tryPrepareWebGPUNativeScan()` with internal dep + passthrough | +82 −68      |
| `src/backend/webgpu.ts`     | Carry snapshot, internal locals, `carryElemCounts`, initCarry copy     | +150 −82     |
| **Phase 1 actual**          |                                                                        | **~230 net** |

---

### Phase 2: Pre-encoded Multi-Step Scan — Submit Reduction (~680 LOC)

For bodies that can't be fused (routines, cross-element deps, nested loops), encode all N × S body
steps into one `GPUCommandEncoder` and submit once.

**Core primitive — `encodeBodyStep()`:**

The `"scan"` JitStep already contains `bodyProgram: JitProgram` — a complete list of JitSteps. The
key new method teaches the WebGPU backend to **encode** (not execute) these steps:

```typescript
// WebGPU backend method
encodeBodyStep(
  encoder: GPUCommandEncoder,
  step: JitStep,
  slotToBuffer: Map<JitId, GPUBuffer>,
): void
```

Step encoding by type:

| JitStep type        | Encoding                                                                   |
| ------------------- | -------------------------------------------------------------------------- |
| `execute` (Kernel)  | beginComputePass → setPipeline → setBindGroup → dispatch → end             |
| `execute` (Routine) | Same — routine compiles to pipeline(s) with multiple passes                |
| `malloc`            | No-op (pre-allocated before the loop)                                      |
| `free`              | No-op (deferred to after all iterations)                                   |
| `recycle`           | Remap in slotToBuffer (buffer alias, zero cost)                            |
| `scan`              | Encode inner scan iterations recursively (see **Nested loop encoding**)    |
| `assoc_scan`        | Encode inner Kogge-Stone rounds recursively (see **Nested loop encoding**) |

**Nested loop encoding:**

When `encodeBodyStep()` encounters a `scan` or `assoc_scan` JitStep, it recursively plans and
encodes the inner loop. This is where pre-encoded dispatch provides the most compositional benefit:

For inner `scan` steps:

```typescript
case "scan": {
  // Plan the inner scan (it gets its own compiled-loop/preencoded/fallback decision)
  const innerPlan = planScan(backend, step.bodyProgram, step.bodyJaxpr, ...);
  if (innerPlan.path === "compiled-loop") {
    // Inner scan compiles to a single fused shader — encode as ONE dispatch
    encodeCompiledLoopDispatch(encoder, innerPlan, iterSlotMapping);
  } else {
    // Inner scan can't fuse — encode its body steps for each inner iteration
    for (let innerIter = 0; innerIter < step.length; innerIter++) {
      for (const innerStep of step.bodyProgram.steps) {
        encodeBodyStep(encoder, innerStep, innerIterSlotMapping);
      }
    }
  }
}
```

For inner `assoc_scan` steps:

```typescript
case "assoc_scan": {
  const innerPlan = planAssociativeScan(backend, ...);
  if (innerPlan.path === "webgpu-fused") {
    // Kogge-Stone: ceil(log₂ M) rounds of fused shader dispatches
    for (let round = 0; round < Math.ceil(Math.log2(M)); round++) {
      encodeFusedRound(encoder, innerPlan, round, iterSlotMapping);
    }
  } else {
    // Fallback inner assocScan — encode body steps per round
    ...
  }
}
```

**Pre-encoded nested performance:**

| Nesting pattern               | Without pre-encoded           | With pre-encoded                           |
| ----------------------------- | ----------------------------- | ------------------------------------------ |
| scan(scan(kernel-only))       | O(N_outer/256) submits        | N_outer × 1 dispatches, **1 submit**       |
| scan(scan(matmul chain))      | O(N_outer/256) submits        | N_outer × N_inner × S, **1 submit**        |
| scan(assocScan(kernel))       | O(N_outer/256) submits        | N_outer × ⌈log₂M⌉ dispatches, **1 submit** |
| grad(scan(kernel)) — backward | O(N/256) submits              | N × S_transposed, **1 submit**             |
| Kalman smoother backward      | O(N/256) submits, ~1s @N=1600 | N×S, **1 submit**, ~20ms (hypothesis\*)    |

\* Performance estimates are hypotheses. Validate with:
`pnpm build && pnpm vitest bench bench/scan.bench.ts` on target hardware. Success criterion for
pre-encoded vs fallback: ≥5× speedup for N≥100 on a body with ≥2 sequential kernel steps.

**Scan dispatch method:**

```typescript
dispatchPreencodedMultiStepScan(
  bodyProgram: JitProgram,
  length: number,
  constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots,
): void {
  // 1. Pre-allocate: ping-pong carry [2 × numCarry] + internal scratch
  // 2. Pre-compile: pipeline per execute step (ShaderPipelineCache)
  // 3. Build bind groups: 2×S (ping/pong phases) + 1 uniform (xs offsets)
  // 4. Copy initCarry → carryPing
  // 5. Encode loop:
  //    for iter = 0..N:
  //      for step in bodyProgram.steps:
  //        encodeBodyStep(encoder, step, iterSlotMapping)
  //      encode ys copy (copyBuf2Buf or WGSL copy shader)
  // 6. queue.submit([encoder.finish()])
  // 7. Copy final carry → carryOut
  // 8. Destroy transient buffers
}
```

**Xs offset handling:**

Body kernel shaders read `input[gidx]`, but we need `input[gidx + iter*stride]` for xs. Each kernel
shader is wrapped with a scan offset uniform — the same pattern `scan-wrapper.ts` already uses for
routine shaders. Dynamic uniform offset selects the iteration. Generalize `wrapRoutineForScan` to
also wrap kernel shaders.

**Bind groups:** 2×S bind groups (ping/pong phases for each step) + 1 uniform bind group with
`hasDynamicOffset: true`. Total independent of N.

**Command buffer size limits:**

Encoding N_outer × N_inner × S dispatches for deep nesting could produce very large command buffers.
Mitigation: chunk encoding into blocks of ≤1024 iterations. Each block is one `encoder.finish()` +
`queue.submit()`. Even 10 submits is 25× fewer than fallback's O(N/256).

**Nested-length compatibility matrix:**

| Outer length | Inner length | Pre-encoded? | Fallback behavior                                          |
| ------------ | ------------ | ------------ | ---------------------------------------------------------- |
| Concrete     | Concrete     | ✅ Yes       | Full encoding: outer×inner×S dispatches, 1 submit          |
| Concrete     | Symbolic     | ❌ No        | Inner scan falls back to JS loop per outer iteration       |
| Symbolic     | Concrete     | ❌ No        | Outer scan falls back (preencoded rejects symbolic length) |
| Symbolic     | Symbolic     | ❌ No        | Both fall back to JS loops                                 |

**Policy:** Pre-encoded encoding requires **concrete lengths at all nesting levels**. Any symbolic
dimension at any level causes that loop (and all enclosing loops depending on it) to fall back to
the JS loop path. This is enforced at planning time — `tryPreparePreencodedMultiStep()` checks
`typeof length === "number"` for the outer scan length and recursively for any inner scan/assocScan
JitSteps encountered during `encodeBodyStep()`. The recursive check is a guard, not a planning step
— if an inner loop has symbolic length, `encodeBodyStep()` returns `null` and the outer scan falls
back to Tier 3.

**Why not encode outer-concrete + inner-symbolic?** The inner scan's length determines how many
dispatches to encode. A symbolic length can't be resolved at encoding time (it's resolved at
execution time from `dimBindings`). We'd need to encode for a maximum bound + predicated writes,
which adds complexity for an uncommon case. Deferred until a concrete use case motivates it.

**ScanPlan extension:**

```typescript
export type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | { path: "compiled-loop"; executable: Executable; params?: NativeScanGeneralParams }
  | { path: "preencoded-routine"; preencodedParams: PreparedPreencodedScan }
  | { path: "preencoded-multi-step"; prepared: PreparedPreencodedMultiStep };
```

**Storage buffer limit check:**

Each step's pipeline binds: consts + 1 carry-read + xs + internal reads + output. Must fit within
`maxStorageBuffersPerShaderStage - 1` (7 at minimum spec of 8; 9 on 99.6% of devices with limit 10;
see copilot-instructions.md WebGPU hard limits table for full coverage data from
[web3dsurvey.com](https://web3dsurvey.com/webgpu)). Checked at preparation time; fall back if
exceeded.

**Files changed:**

| File                                 | Changes                                                 | LOC      |
| ------------------------------------ | ------------------------------------------------------- | -------- |
| `src/frontend/scan-plan.ts`          | Add `tryPreparePreencodedMultiStep()`                   | +120     |
| `src/backend/webgpu.ts`              | `encodeBodyStep()`, `dispatchPreencodedMultiStepScan()` | +280     |
| `src/backend/webgpu/scan-wrapper.ts` | Generalize wrapping for kernel shaders                  | +50      |
| `src/frontend/scan-executor.ts`      | Add `"preencoded-multi-step"` case                      | +30      |
| `src/utils.ts`                       | Add to `ScanPath` type                                  | +1       |
| `test/lax-scan.test.ts`              | Pre-encoded tests (routines, mixed, nested, grad)       | +200     |
| **Phase 2 total**                    |                                                         | **~680** |

---

### Phase 3: Routine Support in Pre-encoded Path (~120 LOC)

Extend Phase 2 to encode routine steps within the command buffer. Routine shaders already compile to
GPU pipelines with passes (e.g., bitonic sort has log²(n) passes). Each pass is encoded as a compute
pass. Routine bind groups map scan buffers to routine bindings.

**Challenge:** Routine shaders have their own bind group layout (different from kernel shaders).
Each routine step needs its own pair of bind groups mapping scan carry/xs/const to the routine's
expected bindings.

**Estimated:** ~120 additional LOC (reuse Phase 2 infrastructure, add routine bind group
construction).

---

### Phase 4: Structural Housekeeping (~70 net LOC)

Two concrete cleanup items adopted from the `LOOP-PRIMITIVE-RESTRUCTURING-PLAN.md`:

#### 4a. Extract `classifyBodySteps()` utility

Both Phase 1 (`canFuseScanSteps`) and Phase 2 (`tryPreparePreencodedMultiStep`) need to analyze the
body program's step structure. Currently, `tryPrepareWasmNativeScan` and
`tryPrepareWebGPUNativeScan` each re-derive step classifications independently. Extract the shared
analysis into a utility:

```typescript
interface BodyStepClassification {
  kernelSteps: { index: number; step: ExecuteStep }[];
  routineSteps: { index: number; step: ExecuteStep }[];
  loopSteps: { index: number; step: JitStep; kind: "scan" | "assoc_scan" }[];
  internalDeps: { consumer: number; producer: number; slot: JitId }[];
  carryPassthroughs: number[];
  hasRoutines: boolean;
  hasLoops: boolean;
  hasInternalDeps: boolean;
  hasCrossGidxDeps: boolean;
}

function classifyBodySteps(
  bodyProgram: JitProgram,
  numCarry: number,
  numConsts: number,
  numX: number,
): BodyStepClassification;
```

This is a utility function, not a type hierarchy. It lives in `scan-plan.ts` and is called by
existing planner functions and by the new eligibility checks. Each backend's `tryPrepare*` function
receives the classification and applies backend-specific constraints on top.

**Note:** This classification also detects `scan` and `assoc_scan` steps in the body — making nested
loops visible to the planner. Phase 2's pre-encoded path can then decide to encode them recursively.

#### 4b. Move assocScan execution out of `jit.ts`

The `"assoc_scan"` case in `JitProgram.execute()` contains ~120 lines of loop orchestration
(Kogge-Stone round dispatch, buffer allocation, slot management). This belongs alongside
`executeScan()` in `scan-executor.ts`, not in the JIT execution engine.

```typescript
// scan-executor.ts (addition)
export function executeAssociativeScan(
  backend: Backend,
  plan: AssocScanPlan,
  ...params
): { outputs: Slot[]; pending: PendingOp[] };
```

The `jit.ts` `"assoc_scan"` case becomes a one-liner calling `executeAssociativeScan()`.

**Files changed:**

| File                            | Changes                                          | LOC         |
| ------------------------------- | ------------------------------------------------ | ----------- |
| `src/frontend/scan-plan.ts`     | Add `classifyBodySteps()`, refactor callers      | +60         |
| `src/frontend/scan-executor.ts` | Add `executeAssociativeScan()`                   | +130        |
| `src/frontend/jit.ts`           | Remove inline assocScan execution, call executor | −120        |
| **Phase 4 total**               |                                                  | **~70 net** |

---

## 7. Nested Loop Composition

### Current behaviour

Nested loops (scan-inside-scan, assocScan-inside-scan) always fall back to the JS loop on both WASM
and WebGPU backends. The planner's `bodyProgram.steps.filter(s => s.type === "execute")` excludes
inner `scan`/`assoc_scan` JitSteps entirely. The fallback `executeScan()` calls
`bodyProgram.execute()` per iteration, which internally handles inner loop JitSteps — correctness is
fine, but performance is not.

### How each tier handles nesting

**Tier 1 (fused shader):** Cannot handle inner loops. A fused WGSL shader inlines all body steps
into a single kernel — it cannot contain a sub-loop dispatching other shaders. Bodies containing
inner `scan`/`assoc_scan` steps skip Tier 1 and fall to Tier 2.

**Tier 2 (pre-encoded):** Handles nested loops naturally. When `encodeBodyStep()` encounters an
inner `scan` or `assoc_scan` JitStep, it recursively plans and encodes the inner loop's dispatches
into the same command buffer. Regardless of nesting depth, the entire outer×inner iteration space is
encoded before a single `queue.submit()`.

Inner loop optimisation is composable:

- Inner scan with a fuseable body → 1 compiled-loop dispatch per outer iteration
- Inner scan with routines → N_inner × S dispatches per outer iteration
- Inner assocScan with fused shader → ⌈log₂M⌉ dispatches per outer iteration
- Inner loop that itself has nested loops → recursion continues

**Tier 3 (fallback):** Handles nesting via `bodyProgram.execute()`, which dispatches inner JitSteps
including inner loops. JS orchestrates everything. O(N/256) submits.

### Why WASM compiled-loop doesn't help with nesting today

WASM `codegenNativeScanGeneral()` only processes Kernel and Routine steps. A `scan` JitStep in the
body is not a Kernel or Routine — it would need its own compiled module, allocated buffers, and loop
management. The routine-import pattern (where the outer module imports a separately compiled
function) could extend to inner scan modules:

```
outer_scan_module imports:
  inner_scan_fn(length, ...carry, ...xs, ...carryOut, ...ysStacked)

outer_loop:
  for iter = 0..N_outer:
    call kernel_0(...)         // kernel step
    call inner_scan_fn(...)    // inner scan via import
    call kernel_1(...)         // kernel step
```

This keeps the entire nested loop in native WASM with zero JS↔WASM boundary crossings per outer
iteration. Estimated: +150 LOC in `codegenNativeScanGeneral()`. Deferred until a concrete use case
motivates it. **Future work** (see Section 10).

### Practical nested loop patterns

| Pattern                                  | Use case                              | Phase  |
| ---------------------------------------- | ------------------------------------- | ------ |
| `scan(body_with_inner_scan)`             | Kalman filter+smoother pipeline       | 2      |
| `scan(body_with_assocScan)`              | Sequential scan using parallel prefix | 2      |
| `grad(scan(body_with_routine))`          | Backward through linalg-heavy scan    | 2+3    |
| `jit(grad(scan(assocScan(...))))`        | Full grad pipeline with nested prefix | 2      |
| `scan(scan(kernel))` on WASM via imports | Nested sequential loops in WASM       | Future |

---

## 8. Future Control-Flow Primitives

### 8a. `Primitive.WhileLoop`

**Semantics:** `while_loop(cond_fn, body_fn, init_carry)` — iterate body while cond returns true.
Equivalent to JAX's `lax.while_loop`.

**Interface constraints (for Phase 2 compatibility):**

- Jaxpr params: `{ cond_jaxpr: Jaxpr, body_jaxpr: Jaxpr }`
- JitStep type: `"while_loop"` with `condProgram`, `bodyProgram`, carry metadata
- `encodeBodyStep()` case: bounded pre-encoded (requires `max_iter` user hint) with predicated
  writes; JS loop fallback without `max_iter`
- WASM: compiled `loop`/`br_if` in single module, dynamic termination
- AD: JVP doubles carry (cond on primals only); transpose reuses `ScanPullbackArtifact` machinery
  (√N checkpointing); vmap pads to max iterations with masking (GPU) or per-element (WASM/CPU)

**Estimated implementation:** ~600 LOC total. Detailed lowering and AD design deferred to a separate
RFC when implementation begins.

### 8b. `Primitive.Cond`

**Semantics:** `cond(pred, true_fn, false_fn, *operands)` — execute one branch based on pred.
Equivalent to JAX's `lax.cond`.

**Interface constraints (for Phase 2 compatibility):**

- Jaxpr params: `{ branches: [true_jaxpr, false_jaxpr] }`
- JitStep type: `"cond"` with `branchPrograms: JitProgram[]`, output metadata
- `encodeBodyStep()` case: encode both branches + select dispatch (WebGPU can't branch per-element
  efficiently; `mapAsync` round trip ~50 µs > executing both for small branches)
- WASM: compile both branches, execute selected one (if/else)
- AD: JVP through both branches, select on primal pred; transpose through selected branch

**Estimated implementation:** ~350 LOC total. Detailed lowering and AD design deferred to a separate
RFC when implementation begins.

### 8c. `Primitive.Switch`

Generalisation of cond to N branches. Same lowering pattern. Deferred until cond is proven useful.

### How new primitives interact with the plan

Phase 2's `encodeBodyStep()` is deliberately designed as a general JitStep encoder:

| Future JitStep type  | `encodeBodyStep()` handling                                      |
| -------------------- | ---------------------------------------------------------------- |
| `while_loop`         | Encode bounded N iterations of body; predicated writes           |
| `cond`               | Encode both branches; select dispatch                            |
| `switch`             | Encode all branches; indexed select dispatch                     |
| `scan` (inner)       | Recurse: plan inner scan, encode its dispatches or compiled-loop |
| `assoc_scan` (inner) | Recurse: plan inner assocScan, encode fused rounds               |

This is the compositional payoff of Phase 2 — every new loop/branch primitive gets pre-encoded
support by adding a case to `encodeBodyStep()`, typically ~30 LOC.

---

## 9. Structural Evolution Path

The current planning infrastructure (`ScanPlan`, `AssocScanPlan`, `planScan()`,
`planAssociativeScan()`, `executeScan()`) serves 2 loop primitives adequately, especially after
Phase 4's housekeeping.

**When to generalise:** When the 3rd loop primitive is implemented (likely `while_loop`), the
following restructuring becomes warranted by real need:

1. Unify `ScanPlan` + `AssocScanPlan` + `WhileLoopPlan` into a `LoopPlan` discriminated union.
2. Separate "what loop" (`LoopSemantics`) from "how to execute" (`LoopPlan`).
3. Single `planLoop()` entry point dispatching to backend methods.
4. Optional `Backend` interface extensions: `tryCompileLoop()`, `tryPreencodedLoop()`,
   `tryFusedRounds()`.
5. Rename `scan-executor.ts` → `loop-executor.ts` with unified `executeLoop()`.

**Design signature for future `LoopPlan`:**

```typescript
type LoopPlan =
  | { strategy: "js-loop"; bodyProgram: JitProgram }
  | { strategy: "compiled-native"; executable: Executable; params: NativeLoopParams }
  | { strategy: "preencoded-gpu"; prepared: PreparedGPULoop }
  | { strategy: "fused-gpu-rounds"; prepared: PreparedFusedRounds; params: FusedRoundParams };

type LoopSemantics =
  | { kind: "scan"; numCarry; numConsts; numX; numY; length; reverse; checkpoint }
  | { kind: "assoc-scan"; numLeaves; numConsts; axis; reverse }
  | { kind: "while-loop"; maxIter?: number }
  | { kind: "cond"; numBranches: number };
```

This is recorded here for design continuity. It is **not implemented** until the trigger condition
is met. The current Phase 1–3 work is designed so the transition is mechanical:

| Current type/function                        | Maps to LoopPlan                              |
| -------------------------------------------- | --------------------------------------------- |
| `ScanPlan { path: "compiled-loop" }`         | `LoopPlan { strategy: "compiled-native" }`    |
| `ScanPlan { path: "preencoded-routine" }`    | `LoopPlan { strategy: "preencoded-gpu" }`     |
| `ScanPlan { path: "preencoded-multi-step" }` | `LoopPlan { strategy: "preencoded-gpu" }`     |
| `ScanPlan { path: "fallback" }`              | `LoopPlan { strategy: "js-loop" }`            |
| `AssocScanPlan { path: "compiled-loop" }`    | `LoopPlan { strategy: "compiled-native" }`    |
| `AssocScanPlan { path: "webgpu-fused" }`     | `LoopPlan { strategy: "fused-gpu-rounds" }`   |
| `planScan()`                                 | `planLoop(semantics: { kind: "scan", ... })`  |
| `planAssociativeScan()`                      | `planLoop(semantics: { kind: "assoc-scan" })` |

---

## 10. Scan Plan Priority Chain (Updated)

```
compiled-loop (Tier 1, extended in Phase 1 ✅)
  ← 1 dispatch, fused shader
  ← accepts: internal deps (same-gidx), carry RAW, passthrough [Phase 1 ✅]
  ← numCarry ≠ numY [deferred — no current use case]
  ↓ rejects: routine steps, cross-gidx deps, inner loops, symbolic sizes

preencoded-multi-step (Tier 2, Phase 2)
  ← N×S dispatches, 1 submit
  ← accepts: routines (Phase 3), mixed bodies, nested scans/assocScans,
     cross-gidx deps, future while_loop/cond
  ↓ rejects: non-encodable steps, symbolic length

preencoded-routine (existing, legacy)
  ← N×P dispatches, 1 submit (single routine only)
  ← subsumed by preencoded-multi-step once Phase 2+3 ship; deprecate then

fallback (Tier 3)
  ← N×S dispatches, O(N/256) submits
  ← accepts: everything
```

---

## 11. File Change Summary (All Phases)

| Phase | File                                 | Change                                      | LOC       |
| ----- | ------------------------------------ | ------------------------------------------- | --------- |
| 1 ✅  | `src/frontend/scan-plan.ts`          | Rewritten `tryPrepareWebGPUNativeScan()`    | +82 −68   |
| 1 ✅  | `src/backend/webgpu.ts`              | Carry snapshot + internal locals            | +150 −82  |
| 2     | `src/frontend/scan-plan.ts`          | `tryPreparePreencodedMultiStep()`           | +120      |
| 2     | `src/backend/webgpu.ts`              | `encodeBodyStep()`, dispatch method         | +280      |
| 2     | `src/backend/webgpu/scan-wrapper.ts` | Generalize for kernel shaders               | +50       |
| 2     | `src/frontend/scan-executor.ts`      | New executor case                           | +30       |
| 2     | `src/utils.ts`                       | Add to `ScanPath` type                      | +1        |
| 2     | `test/lax-scan.test.ts`              | Pre-encoded tests (routines, mixed, nested) | +200      |
| 3     | `src/backend/webgpu.ts`              | Routine step encoding                       | +120      |
| 4a    | `src/frontend/scan-plan.ts`          | `classifyBodySteps()`, refactor callers     | +60       |
| 4b    | `src/frontend/scan-executor.ts`      | `executeAssociativeScan()`                  | +130      |
| 4b    | `src/frontend/jit.ts`                | Remove inline assocScan, call executor      | −120      |
|       | **Total (Phases 1–4)**               |                                             | **~1190** |

Future (not in scope):

| Area        | File                                              | Change                           | LOC      |
| ----------- | ------------------------------------------------- | -------------------------------- | -------- |
| 8a          | Various                                           | `Primitive.WhileLoop` + lowering | ~600     |
| 8b          | Various                                           | `Primitive.Cond` + lowering      | ~350     |
| WASM        | `src/backend/wasm.ts`                             | Nested scan via module imports   | ~150     |
| Restructure | `src/frontend/scan-plan.ts` → `loop-plan.ts` etc. | Unified LoopPlan                 | ~200 net |

---

## 12. Risks & Mitigations

| Risk                                            | Impact           | Mitigation                                                                  |
| ----------------------------------------------- | ---------------- | --------------------------------------------------------------------------- |
| Same-gidx analysis false positive               | Wrong results    | Conservative reject; false negative = fallback                              |
| Command buffer size limits                      | Driver error     | Chunk into ≤1024-iteration blocks. 10 submits still 25× fewer than fallback |
| Nested loop command buffer explosion            | Driver error     | Product N_outer×N_inner×S guard; chunk or fall back above threshold         |
| Storage buffer limit (8 min, 100%; 10 on 99.6%) | Can't bind       | Admission control at preparation time. Fall back                            |
| Shader compilation cost (S pipelines)           | ~50–100 ms       | Pipeline cache amortizes. Bodies are deterministic                          |
| Routine bind group complexity                   | Impl effort      | Phase 3 deferred. Phase 2 covers kernel-only + nested loops                 |
| `var<private>` register pressure                | GPU occupancy    | Monitor. Fallback if step count > 16                                        |
| Deferred-write changes shader structure         | Regression       | Existing tests cover; Phase 1 extends, does not restructure working paths   |
| while_loop iteration bound unknown              | Can't pre-encode | Require `max_iter` hint for bounded path; JS loop otherwise                 |
| cond "execute both" waste                       | 2× compute       | Acceptable for small branches. Large branches get future optimisation       |
| Recursive encoding stack depth                  | Crash            | Cap recursion at 3 levels; deeper nesting uses fallback                     |
| Transform-level ref-balance bugs                | Silent leaks     | Ownership gate (Phase 0); `checkLeaks` in all scan tests; see Contract 2-3  |

---

## 13. Success Criteria

| Criterion                                                                       | Phase    | Validation                                                         |
| ------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------ |
| Elementwise grad(scan) backward fuses to 1 dispatch (WebGPU)                    | 1 ✅     | Mandelbrot: 14.9ms scan vs 55.6ms jit-for (3.7×)                   |
| `acceptPath: "compiled-loop"` works for bodies with internal deps (elementwise) | 1 ✅     | Scan tests pass with internal deps + passthrough                   |
| No dispatch regression for any currently compiled-loop body                     | 1 ✅     | 1635 Vitest + 102 Deno tests pass                                  |
| Bodies with carry passthrough use extended fusion                               | 1 ✅     | Passthrough carries skip writes, retain value                      |
| numCarry ≠ numY uses extended fusion                                            | deferred | No current use case; plan infrastructure supports it               |
| grad(scan) with linalg body uses pre-encoded (1 submit)                         | 2        | `acceptPath: "preencoded-multi-step"` in test                      |
| ≥5× speedup vs fallback for N≥100 on representative body                        | 2        | `pnpm vitest bench bench/scan.bench.ts` — add pre-encoded vs fall. |
| Nested scan(scan(kernel)) uses pre-encoded, 1 submit                            | 2        | New test case with nested scan bodies                              |
| Nested scan(assocScan(kernel)) correctly encodes ⌈log₂M⌉ rounds                 | 2        | New test case                                                      |
| Mixed kernel+routine body uses pre-encoded                                      | 3        | `acceptPath: "preencoded-multi-step"` in test                      |
| assocScan execution not inline in jit.ts                                        | 4        | Code review: `jit.ts` assoc_scan case is ≤5 lines                  |
| classifyBodySteps shared by ≥2 planning functions                               | 4        | grep: ≥2 call sites in `scan-plan.ts`                              |
| No GPU memory leaks (transient buffers destroyed)                               | 1,2      | `checkLeaks` in all new tests                                      |
| Ownership-correct under nested transforms (`jit(grad(scan(...)))`)              | 0,1,2    | `checkLeaks` + multi-output PETracer paths + cache cleanup         |
| All existing tests pass                                                         | 1,2,3,4  | `pnpm vitest run` green                                            |

---

## 14. Comparison with v1 and v2 Plans

| Aspect               | v1 (pre-encoded only)  | v2 (extended fusion + pre-encoded) | v3/v4 (this plan)                                    |
| -------------------- | ---------------------- | ---------------------------------- | ---------------------------------------------------- |
| Architecture         | New ScanPlan path only | Improve existing lowering          | Lowering + housekeeping + forward design             |
| Dispatch reduction   | None (N×S dispatches)  | **1 dispatch** for eligible        | Same + nested loop planning                          |
| Submit reduction     | 1 submit (kernel-only) | 1 submit for all encodable         | Same + nested scan encoding                          |
| Nested loops         | Not addressed          | Not addressed                      | **Pre-encoded recursive encoding**                   |
| Routine support      | Excluded               | Phase 3                            | Phase 3                                              |
| Body analysis        | Per-backend ad hoc     | Per-backend ad hoc                 | **Shared `classifyBodySteps()`**                     |
| assocScan execution  | Inline in jit.ts       | Inline in jit.ts                   | **Moved to scan-executor.ts**                        |
| Future primitives    | Not addressed          | Phase 4 sketch                     | **Interface sketches + separate RFC for detail**     |
| Structural evolution | Not addressed          | Not addressed                      | **Recorded LoopPlan design, clear trigger**          |
| Fusion safety        | Not formalized         | Not formalized                     | **Formal 5-condition admission predicate**           |
| Nested lengths       | Not addressed          | Not addressed                      | **Explicit compatibility matrix, concrete required** |
| Scope                | ~730 LOC kernel-only   | ~1560 LOC concrete                 | ~1190 LOC concrete + future roadmap                  |

---

## 15. Implementation Order

```
Phase 0: Ownership-correctness gate ✅ (complete)
  ↓
Phase 1: Extended fusion (carry snapshot + internal locals) ✅ (commit 74ddb7e)
  ↓ largest user-visible impact: internal deps + passthrough → 1 dispatch

Phase 4a: classifyBodySteps() extraction
  ↓ enables cleaner Phase 2 implementation (Phase 1 was done without it)

Phase 4b: Move assocScan execution to scan-executor.ts
  ↓ housekeeping, independent of Phase 2

Phase 2: Pre-encoded multi-step scan
  ↓ handles routines, nested loops, everything Phase 1 can't fuse

Phase 3: Routine bind groups in pre-encoded path
  ↓ completes pre-encoded coverage

── Future (triggered by 3rd primitive or concrete nested-loop demand) ──

Phase 8a: Primitive.WhileLoop
Phase 8b: Primitive.Cond
WASM nested scan imports
LoopPlan unification
```
