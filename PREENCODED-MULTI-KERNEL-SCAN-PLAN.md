# Control-Flow Lowering & Scan Performance Plan

**Status:** Draft v3 for review **Date:** 2026-02-25 **Supersedes:** v2 (extended fusion +
pre-encoded), v1 (pre-encoded multi-kernel only) **Integrates:** Useful elements from
`LOOP-PRIMITIVE-RESTRUCTURING-PLAN.md`

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

But the compiled-loop rejects bodies with:

- Internal buffer dependencies between steps
- Cross-step carry read-after-write hazards
- Carry passthrough (carry unchanged)
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

**Extended eligibility (Phase 1):** Relax constraints via deferred-write pattern + private WGSL
variables for intermediates. Bodies with same-gidx internal deps, carry RAW hazards, carry
passthrough, and numCarry ≠ numY become fusable — **1 dispatch** instead of N×S fallback.

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

### Phase 1: Extended Fused Shader — Dispatch Reduction (~320 LOC)

Extend `nativeScanMultiShaderSource()` to accept bodies with same-gidx internal deps, carry RAW
hazards, carry passthrough, and numCarry ≠ numY.

**Approach: deferred-write pattern + private WGSL variables.**

The current shader computes each step and immediately writes to the `carry` storage buffer. This
causes two problems:

1. **Carry RAW hazard:** Step j reads `carry[gidx]` after step i overwrote it — sees NEW value
   instead of the OLD carry from the iteration start.
2. **Internal deps rejected:** Step j reading step i's output buffer is rejected outright because
   the shader has no mechanism to pass intermediate results between step code blocks.

The fix restructures the shader into four phases:

```wgsl
for (var iter: u32 = 0u; iter < N; iter++) {
  let dataIdx = select(iter, N - 1u - iter, REVERSE);  // reverse-aware

  // Phase A: Snapshot OLD carry values into private vars
  let old_carry0: f32 = carry0[gidx];    // gidx < carrySize0
  let old_carry1: f32 = carry1[gidx];    // gidx < carrySize1

  // Phase B: Compute ALL steps using OLD carry + private intermediates
  var step0_val: f32 = 0.0;
  if (gidx < size0) {
    step0_val = expr0(old_carry0, xs0[i32(dataIdx) * stride0 + gidx]);
  }
  var step1_val: f32 = 0.0;
  if (gidx < size1) {
    step1_val = expr1(step0_val, old_carry1, ...);  // reads step0's PRIVATE var ✓
  }

  // Phase C: Deferred carry writes (all at once, after all computation)
  if (gidx < carrySize0) { carry0[gidx] = step0_val; }
  if (gidx < carrySize1) { carry1[gidx] = step1_val; }

  // Phase D: Write ys
  if (gidx < ysSize0) { ys0[i32(dataIdx) * ysStride0 + gidx] = step0_val; }
  if (gidx < ysSize1) { ys1[i32(dataIdx) * ysStride1 + gidx] = step1_val; }
}
```

**Why this is correct:**

- Phase A reads all carry values before any writes → carry RAW hazard eliminated.
- Phase B uses `var<private>` intermediates → internal deps served from thread-local state.
- Each thread only reads its own `gidx` from carry and intermediates → no cross-thread dependency,
  no barrier needed.
- Deferred writes (Phase C) ensure all steps see the OLD carry from the iteration start.

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

**Files changed:**

| File                        | Changes                                                                      | LOC      |
| --------------------------- | ---------------------------------------------------------------------------- | -------- |
| `src/frontend/scan-plan.ts` | Relax `tryPrepareWebGPUNativeScan()`, add `canFuseScanSteps()`               | +80      |
| `src/backend/webgpu.ts`     | Deferred-write shader gen, internal dep expression mapping                   | +120     |
| `test/lax-scan.test.ts`     | Extended fusion tests (internal deps, carry RAW, passthrough, numCarry≠numY) | +120     |
| **Phase 1 total**           |                                                                              | **~320** |

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
| Kalman smoother backward      | O(N/256) submits, ~1s @N=1600 | N×S, **1 submit**, ~20ms estimated         |

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
`maxStorageBuffersPerShaderStage - 1` (typically 7–9). Checked at preparation time; fall back if
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

**Jaxpr representation:**

```
Primitive.WhileLoop {
  cond_jaxpr: Jaxpr,    // (...carry) → bool
  body_jaxpr: Jaxpr,    // (...carry) → ...carry
}
```

**JitStep:** `"while_loop"` with `condProgram`, `bodyProgram`, carry metadata.

**Backend lowering:**

| Backend                   | Strategy            | Notes                                                                                                   |
| ------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------- |
| WASM                      | Compiled while loop | WASM `loop`/`br_if`. Compile cond+body into single module. Dynamic termination. Zero JS boundary.       |
| WebGPU (elementwise body) | Bounded GPU loop    | If user provides `max_iter`, WGSL: `for (var i=0u; i<MAX; i++) { if (!cond()) break; body(); }` .       |
| WebGPU (complex body)     | JS loop             | JS dispatches cond shader, reads bool, dispatches body if true. Like scan fallback with dynamic length. |
| WebGPU (with pre-encoded) | Bounded pre-encoded | Encode max_iter iterations. Predicated writes skip when cond is false. 1 submit.                        |

**AD rules:**

- **JVP:** Thread tangent carry alongside primal carry. Cond evaluated on primals only (bool is
  non-differentiable). Body JVP doubles the carry:
  `(primal_carry, tangent_carry) → (new_primal, new_tangent)`.
- **Transpose:** Requires iteration count from forward pass. Options: (a) store all N carries (O(N)
  memory), (b) √N checkpointing (same as scan). The scan backward machinery (`ScanPullbackArtifact`)
  generalises directly — the key method is `runOneForwardStep` / `runOneBackwardStep` which don't
  depend on scan-specific semantics.
- **Vmap:** Each batch element may iterate different number of times. Two approaches: (a) pad to max
  iterations with masking (JAX approach), (b) per-element sequential. Use (a) for GPU backends, (b)
  for WASM/CPU.

**Interaction with scan bodies:**

A `while_loop` step inside a scan body creates a body with a `while_loop` JitStep. Tier 2
(pre-encoded) handles this via bounded encoding: encode `max_iter` iterations of the inner
while_loop's body, with predicated writes. The outer scan's N iterations × inner while_loop's
max_iter iterations are all in one command buffer.

WASM compiled-loop handles it via compiled module import (analogous to nested scan import):

```
outer_scan_module imports:
  inner_while_fn(...carry, ...carryOut) → i32 (converged flag)
```

**Estimated implementation:** ~300 LOC (primitive + JitStep + WASM backend). WebGPU bounded loop:
+100 LOC. AD rules: +200 LOC. Pre-encoded support: reuses Phase 2 `encodeBodyStep()`.

### 8b. `Primitive.Cond`

**Semantics:** `cond(pred, true_fn, false_fn, *operands)` — execute one branch based on pred.
Equivalent to JAX's `lax.cond`.

**Jaxpr representation:**

```
Primitive.Cond {
  branches: [true_jaxpr, false_jaxpr],
}
```

**JitStep:** `"cond"` with `branchPrograms: JitProgram[]`, output metadata.

**Backend lowering:**

| Backend                   | Strategy                  | Notes                                                                     |
| ------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| WASM                      | if/else                   | Compile both branches. Execute selected one. Zero waste.                  |
| WebGPU                    | Execute both + select     | GPU can't branch efficiently. Execute both branches, `select` results.    |
| WebGPU (inside scan body) | Pre-encoded both + select | Both branches' dispatches encoded. Select dispatch copies correct result. |

**Why "execute both" on WebGPU:**

Reading `pred` from GPU to conditionally encode only one branch requires a `mapAsync` round trip
(~50 µs latency on typical hardware). For small branches this is far more expensive than executing
both and selecting. For very large branches (e.g., one does a 1024×1024 matmul, the other doesn't),
a future optimisation could batch cond evaluation or use indirect dispatch to skip work.

**AD rules:**

- **JVP:** JVP through both branches independently, select tangent output based on primal pred.
- **Transpose:** Transpose through selected branch (pred is non-differentiable).
- **Vmap:** Per-element cond → execute both branches on full batch, select per element.

**Cond inside scan body — pre-encoded handling:**

When a scan body contains a `cond` step, Phase 2's `encodeBodyStep()` encodes both branches:

```
for iter in 0..N:
  encode kernel step 1
  encode cond:
    encode true_branch dispatches
    encode false_branch dispatches
    encode select dispatch (conditional copy based on pred)
  encode kernel step 2
```

All in 1 submit. The select dispatch is a small shader that reads pred from a buffer and copies the
correct branch's output to the cond's output buffer.

**Estimated implementation:** ~200 LOC (primitive + JitStep + backend lowering). AD rules: +150 LOC.
Pre-encoded support: reuses Phase 2 `encodeBodyStep()`.

### 8c. `Primitive.Switch`

**Semantics:** `switch(index, branches, *operands)` — execute one of N branches based on index.
Generalisation of cond.

Same lowering pattern as cond: WASM does switch/if-chain, WebGPU executes all branches + select.
Deferred until cond is proven useful.

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
compiled-loop (Tier 1, extended in Phase 1)
  ← 1 dispatch, fused shader
  ← accepts: internal deps (same-gidx), carry RAW, passthrough, numCarry ≠ numY
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
| 1     | `src/frontend/scan-plan.ts`          | Relax constraints, `canFuseScanSteps()`     | +80       |
| 1     | `src/backend/webgpu.ts`              | Deferred-write shader gen                   | +120      |
| 1     | `test/lax-scan.test.ts`              | Extended fusion tests                       | +120      |
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

| Risk                                    | Impact           | Mitigation                                                                  |
| --------------------------------------- | ---------------- | --------------------------------------------------------------------------- |
| Same-gidx analysis false positive       | Wrong results    | Conservative reject; false negative = fallback                              |
| Command buffer size limits              | Driver error     | Chunk into ≤1024-iteration blocks. 10 submits still 25× fewer than fallback |
| Nested loop command buffer explosion    | Driver error     | Product N_outer×N_inner×S guard; chunk or fall back above threshold         |
| Storage buffer limit (8–10)             | Can't bind       | Admission control at preparation time. Fall back                            |
| Shader compilation cost (S pipelines)   | ~50–100 ms       | Pipeline cache amortizes. Bodies are deterministic                          |
| Routine bind group complexity           | Impl effort      | Phase 3 deferred. Phase 2 covers kernel-only + nested loops                 |
| `var<private>` register pressure        | GPU occupancy    | Monitor. Fallback if step count > 16                                        |
| Deferred-write changes shader structure | Regression       | Existing tests cover; Phase 1 extends, does not restructure working paths   |
| while_loop iteration bound unknown      | Can't pre-encode | Require `max_iter` hint for bounded path; JS loop otherwise                 |
| cond "execute both" waste               | 2× compute       | Acceptable for small branches. Large branches get future optimisation       |
| Recursive encoding stack depth          | Crash            | Cap recursion at 3 levels; deeper nesting uses fallback                     |

---

## 13. Success Criteria

| Criterion                                                                       | Phase   |
| ------------------------------------------------------------------------------- | ------- |
| Elementwise grad(scan) backward fuses to 1 dispatch (WebGPU)                    | 1       |
| `acceptPath: "compiled-loop"` works for bodies with internal deps (elementwise) | 1       |
| No dispatch regression for any currently compiled-loop body                     | 1       |
| Bodies with carry passthrough and numCarry ≠ numY use extended fusion           | 1       |
| grad(scan) with linalg body uses pre-encoded (1 submit)                         | 2       |
| ≥5× speedup vs fallback for N≥100 on representative body                        | 2       |
| Nested scan(scan(kernel)) uses pre-encoded, 1 submit                            | 2       |
| Nested scan(assocScan(kernel)) correctly encodes ⌈log₂M⌉ rounds                 | 2       |
| Mixed kernel+routine body uses pre-encoded                                      | 3       |
| assocScan execution not inline in jit.ts                                        | 4       |
| classifyBodySteps shared by ≥2 planning functions                               | 4       |
| No GPU memory leaks (transient buffers destroyed)                               | 1,2     |
| All existing tests pass                                                         | 1,2,3,4 |

---

## 14. Comparison with v1 and v2 Plans

| Aspect               | v1 (pre-encoded only)  | v2 (extended fusion + pre-encoded) | v3 (this plan)                               |
| -------------------- | ---------------------- | ---------------------------------- | -------------------------------------------- |
| Architecture         | New ScanPlan path only | Improve existing lowering          | Lowering + housekeeping + forward design     |
| Dispatch reduction   | None (N×S dispatches)  | **1 dispatch** for eligible        | Same + nested loop planning                  |
| Submit reduction     | 1 submit (kernel-only) | 1 submit for all encodable         | Same + nested scan encoding                  |
| Nested loops         | Not addressed          | Not addressed                      | **Pre-encoded recursive encoding**           |
| Routine support      | Excluded               | Phase 3                            | Phase 3                                      |
| Body analysis        | Per-backend ad hoc     | Per-backend ad hoc                 | **Shared `classifyBodySteps()`**             |
| assocScan execution  | Inline in jit.ts       | Inline in jit.ts                   | **Moved to scan-executor.ts**                |
| Future primitives    | Not addressed          | Phase 4 sketch                     | **Detailed while_loop, cond, switch design** |
| Structural evolution | Not addressed          | Not addressed                      | **Recorded LoopPlan design, clear trigger**  |
| Scope                | ~730 LOC kernel-only   | ~1560 LOC concrete                 | ~1190 LOC concrete + future roadmap          |

---

## 15. Implementation Order

```
Phase 4a: classifyBodySteps() extraction
  ↓ enables cleaner Phase 1 + Phase 2 implementation

Phase 1: Extended fusion (deferred-write shader)
  ↓ largest user-visible impact: grad(scan) → 1 dispatch

Phase 4b: Move assocScan execution to scan-executor.ts
  ↓ housekeeping, independent of Phase 1

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
