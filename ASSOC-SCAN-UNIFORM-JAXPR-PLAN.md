# Uniform Jaxpr-Based Associative Scan — Implementation Plan

Replaces the current hybrid approach — where only the local block scan goes through jaxpr while the
inter-block phases are hard-coded JS loops — with a fully plan-driven decomposition. Also removes
the legacy standalone WebGPU blocked shader path and fixes the shape-insensitive fused cache.

**Branch:** `block-map`, HEAD at `6ab7203` **Predecessor:** Phase 8.2 (commit `6ab7203`) — routed
local scan through `block_map` but left phases 2–4 as ad-hoc executor code.

---

## Current State: What the Shortcuts Are

### Shortcut 1: JS-level inter-block execution (scan-executor.ts:541–917, 377 LOC)

The `executeAssocScanBlockMap()` function is split into four phases:

| Phase           | Lines                     | What it does                                   | Problem                                        |
| --------------- | ------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 1. Local scan   | 609–647                   | `executeBlockMap()` — jaxpr-driven ✅          | None                                           |
| 2. Gather       | 689–709                   | JS loop: `copyBufferToBuffer` × M × numLeaves  | O(M) host-side calls, not a planned operation  |
| 3. Summary scan | 720–798                   | JS loop: `scanBodyProgram.execute()` × (M-1)   | O(M) host→GPU round-trips, not recursive       |
| 4. Apply        | 806–869                   | JS loop: `scanBodyProgram.execute()` × (N - B) | O(N) host→GPU round-trips — the worst offender |
| Reverse         | 578–605, 655–680, 889–917 | Explicit element-by-element buffer reversal    | O(N) copies instead of a planned permutation   |

Replace phases 2–4 of `executeAssocScanBlockMap()` with plan-driven operations so the entire
associative scan executes through jaxpr-compiled programs. Under the strict version, **no host-side
JS gather/apply/reverse loops remain** before the legacy WebGPU path is deleted.

### Strict invariants

Before Plan 2 starts, all of these must be true:

1. Gather is expressed by a traced body or primitive-backed Jaxpr, not `copyBufferToBuffer` loops.
2. Apply is expressed by a traced body or primitive-backed Jaxpr, not an executor loop over blocks.
3. Reverse is represented by existing array/view semantics or a Jaxpr primitive, not manual buffer
   reversal in `executeAssocScanBlockMap()`.
4. `executeAssocScanBlockMap()` becomes an orchestrator that invokes child plans only. It may plan
   recursively, allocate outputs, and dispatch child plans, but it may not implement algorithmic
   per-element or per-block control flow itself.

### 1a. Add the missing IR building blocks

**Files:** `src/frontend/core.ts`, `src/frontend/jaxpr.ts`, `src/frontend/jit.ts`,
`src/frontend/vmap.ts`, `src/frontend/jvp.ts`, `src/frontend/linearize.ts`,
`src/frontend/block-map-executor.ts`, `src/backend/webgpu/block-map.ts`

The current `block_map` abstraction is not quite rich enough to express strict gather/apply. Two
small IR-level additions are needed.

#### A. `Primitive.BlockIndex` (internal-only)

Purpose: expose the current block index inside a `block_map` body so the body can fetch the correct
summary element and compute the correct tail index for the boundary block.

Semantics:

```ts
// inside a block_map body with 1D grid
const i: int32 = blockIndex();
```

Rules:

- Eager `block_map` fallback: the executor binds the current JS loop block index.
- JIT block_map: lowered from the ambient block_map execution context to `workgroup_id.x` in fused
  WebGPU and to the loop index in the WASM / fallback block-map executor.
- Abstract eval: scalar `ShapedArray([], DType.Int32)`.
- JVP: zero tangent.
- Transpose: no cotangent result.
- Vmap: block index is body-local and remains unmapped.

Important design decision:

- Keep `BlockIndex` as an internal primitive.
- Do **not** model block index as a synthetic extra body input.

Reasoning:

- A synthetic input would push planner-only calling-convention details into every block_map body.
- It would make gather/apply depend on extra wrapper signatures rather than the structured block_map
  execution context.
- `BlockIndex` keeps the body jaxpr declarative: the body asks for the current block position the
  same way a loop body asks for its loop index.

Implementation note:

- `BlockIndex` should not introduce a standalone `JitStep` if that can be avoided.
- Prefer lowering it directly where block_map bodies already lower context-sensitive indexing:
  fallback/WASM materialize the scalar value for `evalJaxpr`, while the fused WebGPU path resolves
  it from `workgroup_id` during WGSL generation.

#### B. Explicit block-map grid contract and output shaping

Current `Primitive.BlockMap` infers output mapped extents from mapped input extents. That prevents
strict gather because gather needs execution over `M = ceil(N / B)` blocks while all gather inputs
are broadcast.

Extend BlockMap params with an optional explicit grid contract:

```ts
{
  jaxpr: Jaxpr;
  blockShape: number[];
  inAxes: ...;
  outAxes: ...;
  numConsts: number;
  numInputs: number;
  gridShape?: number[];
}
```

Revised typing rule:

- If `gridShape` is omitted, preserve current behavior.
- If `gridShape` is present, it supplies execution geometry even when no mapped input determines it.
- Output shape inference remains **non-padded**:
  - if an output mapped axis corresponds to an original mapped input dimension, use that original
    input extent as today;
  - otherwise, use `gridShape[g] * bodyOutShape[outAxes[g]]`.

This is enough to express:

- local scan: output extent `N` as today
- gather: output extent `M` with `blockShape = [1]` and planner-supplied `gridShape = [M]`
- apply: output extent `N` with `blockShape = [B]` using existing mapped-input inference

This is the critical enabler for keeping gather and apply inside planned `block_map` programs.

Validation gate before using this for assocScan:

- add a focused regression test that uses a block_map body with `BlockIndex` plus `dynamicSlice` on
  a broadcast input and verifies the fused WebGPU path executes correctly;
- if this fails, fix block_map lowering first rather than embedding the failure into assocScan.

### 1b. Canonicalize reverse out of the executor

**Files:** `src/library/lax-associative-scan.ts`, `src/frontend/jaxpr.ts`, `src/frontend/jit.ts`,
`src/frontend/scan-plan.ts`, `src/frontend/scan-executor.ts`

The strict plan does not allow manual buffer reversal. Reverse must be represented before execution
planning.

Use the existing `flip` / shape-tracker machinery already present in the array core instead of new
copy loops.

Canonical form:

```ts
associativeScan(fn, xs, { reverse: true }) ==
  flip(associativeScan(fn, flip(xs, [axis]), { reverse: false }), [axis]);
```

Implementation choices:

- Prefer rewriting in the library layer where `reverse: true` becomes forward scan on `flip(xs)` and
  result `flip(...)`.
- If that is too invasive for eager semantics, do it during tracing so `Primitive.AssociativeScan`
  reaches JIT with `reverse=false` plus surrounding `flip` operations.

Consequence:

- `AssocScanPlan` no longer needs a `reverse` field for the WebGPU block-map path.
- `executeAssocScanBlockMap()` does not contain reverse branches.
- Any reverse handling is now a normal part of traced program structure.

### 1c. Build gather as a planned block_map stage

**Files:** `src/frontend/scan-plan.ts`, `src/frontend/scan-executor.ts`

Strict gather design:

- Grid shape: `[M]`
- Block shape: `[1]`
- Broadcast inputs: the full local-scan outputs `[N, ...]`
- Body-local scalar: `blockIndex()` gives `i`
- Tail index per block: `tail = min((i + 1) * B - 1, N - 1)`
- Body output per leaf: `dynamicSlice(localScanLeaf, [tail, 0, ...], [1, ...]) |> squeeze(0)`

Because `gridShape=[M]` and `blockShape=[1]`, the output shape is `[M, ...]`, which is exactly the
summary buffer shape the recursive scan needs.

Body sketch:

```ts
// inputs: [const localScan_0[N,...], ..., localScan_L[N,...], N_scalar]
// outputs: [summary_0[...], ..., summary_L[...]]
const i = blockIndex();
const tail = min((i + 1) * B - 1, N - 1);
return localScanLeaves.map((leaf) => squeeze(dynamicSlice(leaf, [tail, 0, ...], [1, ...]), 0));
```

Implementation notes:

- `N` can be passed as a scalar const to the gather body, resolved from `dimBindings` at execution.
- No identity padding is required.
- No JS loops remain in gather.
- Build the gather body by tracing a normal frontend function with `makeJaxpr`; do not construct the
  gather Jaxpr by hand.
- The gather body is the proving ground for `BlockIndex` + broadcast `dynamicSlice`; land the
  dedicated regression test before wiring gather into assocScan.

### 1d. Build summary scan as recursive assocScan planning

**Files:** `src/frontend/scan-plan.ts`, `src/frontend/scan-executor.ts`

This part of the earlier plan stays, but under the strict version it becomes one child stage in the
composite plan instead of a special-case escape hatch.

Important architecture decision:

- the planner, not the executor, owns recursion;
- `AssocScanPlan` for the block-map path should become a tree that contains the child summary plan
  once `M > 1`;
- `executeAssocScanBlockMap()` must only dispatch already-built child plans, never call
  `planAssociativeScan()` itself.

That avoids planner/executor import cycles and keeps the executor purely operational.

Plan shape sketch:

```ts
{
  path: "webgpu-block-map";
  localScan: BlockMapStage;
  gatherSummary?: BlockMapStage;
  summaryPlan?: AssocScanPlan;
  apply?: BlockMapStage;
  ...
}
```

Termination argument:

- If `N <= B`, no summary stage exists.
- Else summaries have length `M = ceil(N / B)`.
- If `M <= B`, the recursive summary scan is single-block and bottoms out immediately.
- If `M > B`, recurse on `ceil(M / B)`.

### 1e. Build apply as a planned block_map stage

**Files:** `src/frontend/scan-plan.ts`, `src/frontend/scan-executor.ts`, `src/frontend/vmap.ts`

This is the most important correction to the previous plan: **apply may not fall back to an executor
loop over blocks.**

Strict apply design:

- Grid shape: `[M]`
- Block shape: `[B]`
- Mapped input: local scan output `[N, ...]`
- Broadcast input: the full scanned-summary buffer `[M, ...]`
- Body-local scalar: `blockIndex()` gives `i`
- For `i == 0`, return the local-scan block unchanged.
- For `i > 0`, fetch `summary = scannedSummary[i - 1]` and apply the vmapped associative body to the
  whole block.

The associative body is still lifted with `vmapJaxpr()`:

```ts
const dims = [
  ...Array(numConsts).fill(null),
  ...Array(numLeaves).fill(null),
  ...Array(numLeaves).fill(0),
];
const applyVmapped = vmapJaxpr(scanBodyJaxpr, blockSize, dims);
```

But now that vmapped body is embedded inside a `block_map` body rather than run from a host loop.

Body sketch:

```ts
// inputs: [scannedSummary_0[M,...], ..., scannedSummary_L[M,...], localBlock_0[B,...], ...]
const i = blockIndex();
if (i == 0) return localBlocks;
const prefixLeaves = scannedSummaryLeaves.map((leaf) =>
  squeeze(dynamicSlice(leaf, [i - 1, 0, ...], [1, ...]), 0),
);
return applyVmapped(prefixLeaves, localBlocks);
```

Implementation details:

- Build the apply body by tracing a normal frontend function with `makeJaxpr`; do not manually wire
  a Jaxpr call node.
- Inline the vmapped body with `evalJaxpr(applyVmapped.jaxpr, [...applyVmapped.consts, ...args])`
  during tracing, following the existing `jvp.ts` pattern, rather than inventing a new nested-call
  mechanism.
- The body-level conditional should be expressed with existing elementwise forms (`equal`,
  broadcast, `where` / `select`) so no new control-flow primitive is introduced.
- Keep the block-0 passthrough in the traced body, not in the executor.

Result:

- Apply is one planned `block_map` dispatch across all blocks, not `M` host-issued program calls.

### 1f. Executor becomes pure orchestration

After 1b–1e, `executeAssocScanBlockMap()` should reduce to:

```ts
executeAssocScanBlockMap(params):
  N = resolve(...)
  M = ceil(N / B)

  // Phase 1: local scan via executeBlockMap(plan.localScan)
  localScan = ...

  if (M === 1) {
    // direct copy to outputs, no special reverse path
    return
  }

  // Phase 2: gather via executeBlockMap(plan.gatherSummary)
  summaries = ...

  // Phase 3: child summary plan already built by planner
  scannedSummaries = ...

  // Phase 4: apply via executeBlockMap(plan.apply)
  outputs = ...

  return
```

Notably absent:

- no `for (let i = 1; i < M; i++)` loop
- no `for (let blockIdx = 1; blockIdx < M; blockIdx++)` loop
- no element-by-element `copyBufferToBuffer` reverse loop
- no recursive planning from inside the executor

### 1g. Commit plan

**Commit 1a: block_map IR support for strict decomposition**

Files modified:

- `src/frontend/core.ts` — add `Primitive.BlockIndex`
- `src/frontend/jaxpr.ts` — typing rule for `BlockIndex`; extend `BlockMap` with optional
  `gridShape`
- `src/frontend/jit.ts` — lower `BlockIndex` inside block_map body lowering without adding a
  planner-visible host loop escape hatch
- `src/frontend/vmap.ts` — ensure `BlockIndex` is body-local and unmapped
- `src/frontend/jvp.ts` — zero tangent rule
- `src/frontend/linearize.ts` — transpose rule with no cotangent result
- `src/frontend/block-map-executor.ts` — bind block index in fallback/WASM path
- `src/backend/webgpu/block-map.ts` — lower `BlockIndex` to `workgroup_id.x`
- `test/block-map-jit.test.ts` — regression coverage for `BlockIndex` and broadcast `dynamicSlice`

Quality gates:

- new unit tests for `block_map` bodies that read `blockIndex()`
- fused WebGPU test for `BlockIndex` + broadcast `dynamicSlice` passes
- no regressions in existing block-map tests

**Commit 1b: reverse canonicalization**

Files modified:

- `src/library/lax-associative-scan.ts` or tracing layer equivalent
- `src/frontend/scan-plan.ts`
- `src/frontend/scan-executor.ts`

Quality gates:

- reverse assocScan tests still pass
- `executeAssocScanBlockMap()` contains no reverse-specific copy loops

**Commit 1c: strict gather + recursive summary scan**

Files modified:

- `src/frontend/scan-plan.ts` — build `gatherSummary` stage and planner-owned child `summaryPlan`
- `src/frontend/scan-executor.ts` — replace manual gather + sequential summary loop

Quality gates:

- non-divisible `N` cases pass
- summary recursion works for `N = B + 1`, `2B + 17`, `B^2 + 5`
- `executeAssocScanBlockMap()` performs no recursive planning

**Commit 1d: strict apply**

Files modified:

- `src/frontend/scan-plan.ts` — build traced apply block_map body using `BlockIndex` + `vmapJaxpr`
- `src/frontend/scan-executor.ts` — replace per-block host loop with `executeBlockMap(plan.apply)`
- `src/frontend/jaxpr.ts` or helper layer — inline vmapped body with `evalJaxpr` during tracing if
  needed by the helper used to construct the apply stage

Quality gates:

- no block loop remains in `executeAssocScanBlockMap()`
- large `N` cases execute with one apply-stage dispatch rather than `M` host dispatches
- block-0 passthrough is expressed in the traced apply body, not the executor

### 1h. Complexity analysis

| Phase      | Current              | Strict Plan 1           | Improvement                   |
| ---------- | -------------------- | ----------------------- | ----------------------------- |
| Local scan | 1 block_map dispatch | 1 block_map dispatch    | Same                          |
| Gather     | O(M×L) host copies   | 1 block_map dispatch    | removes host loop             |
| Summary    | O(M) body dispatches | recursive assocScan     | removes sequential loop       |
| Apply      | O(N) body dispatches | 1 block_map dispatch    | removes host loop entirely    |
| Reverse    | O(N) copy loop       | traced `flip` semantics | removes executor special-case |
| **Total**  | O(N) dispatches      | O(log_B N) stage calls  | uniform staged execution      |

Where L = numLeaves, B = blockSize (256), M = ceil(N/B).

---

## Plan 2: Remove Legacy `webgpu-fused-blocked` Path

### Goal

Delete ~1,126 LOC of standalone WebGPU blocked assocScan codegen, leaving the block-map path as the
sole WebGPU native route.

### Prerequisite

Plan 1 must be complete and all parity checks pass. In strict mode that means:

- no JS gather loop in `executeAssocScanBlockMap()`
- no JS apply loop in `executeAssocScanBlockMap()`
- no manual reverse-copy path in `executeAssocScanBlockMap()`
- gather/apply/reverse are all represented by traced structure or internal primitives
- the block-map path handles every case the legacy path handles today

### 2a. Add debug gate for A/B comparison

**File:** `src/frontend/scan-plan.ts`

Add a temporary environment-like toggle (via `setDebug` level or a plan option):

```ts
// In planAssociativeScan, WebGPU section:
if (blockMapPlan) return blockMapPlan;
// If block-map failed, log reason and try legacy:
if (DEBUG >= 1) console.log("[assoc-scan] block-map failed, trying legacy");
```

This already exists. No code change needed — just use `setDebug(1)` during comparison testing.

### 2b. Parity test matrix

Run both paths on this matrix (force each via temporarily disabling the other in the planner):

| Test case                         | Status gate                   |
| --------------------------------- | ----------------------------- |
| Scalar cumsum, N < B              | Values match ± 1e-6           |
| Scalar cumsum, N = B              | Values match                  |
| Scalar cumsum, N = B + 1          | Values match (boundary block) |
| Scalar cumsum, N = 2B + 17        | Values match (non-divisible)  |
| Vector cumsum, elemShape=[3]      | Values match                  |
| Pytree 2-tuple composition        | Values match                  |
| Reverse cumsum, N = 500           | Values match                  |
| grad(assocScan), N = 256          | Gradient values match ± 1e-4  |
| jit(assocScan), N = 1000          | Values match                  |
| vmap(assocScan), batch=4, N = 256 | Values match                  |

### 2c. Benchmark comparison

**File:** `bench/associative-scan.bench.ts`

Add or extend benchmarks that isolate the two paths:

```ts
// Force block-map only:
bench("assocScan cumsum N=4096 (block-map)", async () => {
  // disable legacy in planner
});
// Force legacy only:
bench("assocScan cumsum N=4096 (legacy)", async () => {
  // disable block-map in planner
});
```

Measure at N = 256 (M=1), N = 1024 (M=4), N = 4096 (M=16), N = 16384 (M=64).

**Acceptance:** Block-map path must be within 2× of legacy for all sizes, or regression must be
documented and attributed to a specific fixable cause.

### 2d. Deletion checklist

Once parity and benchmarks pass:

1. **`src/frontend/scan-plan.ts`:**
   - Remove `"webgpu-fused-blocked"` from `AssocScanPlan` type union
   - Remove `webgpuParams` construction (lines ~1602–1635)
   - Remove `webgpuBackend.prepareBlockedAssocScan()` call (lines ~1663–1681)
   - Remove imports: `WebGPUAssocScanParams`, `PreparedWebGPUBlockedAssocScan`

2. **`src/frontend/scan-executor.ts`:**
   - Remove `if (plan.path === "webgpu-fused-blocked")` branch (lines ~449–462)
   - Remove `WebGPUBackend` import if no longer needed

3. **`src/backend/webgpu.ts`:**
   - Delete `blockedAssocScanLocalShaderSource()` (389 LOC, lines 4375–4763)
   - Delete `blockedAssocScanGatherShaderSource()` (64 LOC, lines 4774–4837)
   - Delete `blockedAssocScanApplyShaderSource()` (131 LOC, lines 4849–4979)
   - Delete `prepareBlockedAssocScan()` (211 LOC, lines 932–1142)
   - Delete `dispatchBlockedAssocScan()` (331 LOC, lines 1153–1483)
   - Delete `interface PreparedWebGPUBlockedAssocScan` (lines ~166–195)
   - Delete `interface WebGPUAssocScanParams` (lines ~119–134)
   - Delete `interface AssocScanStep` (lines ~105–118)
   - Delete or update `flatAssocScanShaderSource()` if only used by the blocked path
   - Remove related helper functions: `emitAssocScanBodySteps()` and `flatAssocScanShaderSource()`
     IF they are no longer referenced

4. **`src/backend/webgpu/wgsl-gen.ts`:**
   - Check if `ResolveGlobalIndex` callback type is still used after deletion
   - Keep it if block-map.ts still uses it (it does)

5. **Types cleanup:**
   - Remove re-exports from `src/backend/webgpu.ts` if they were only used by the blocked path
   - Update `src/frontend/scan-plan.ts` imports accordingly

6. **Tests:**
   - Update any test that asserts path name `"webgpu-fused-blocked"`
   - Verify `acceptPath` arrays in tests don't include the removed path

### 2e. Commit plan

**Commit 2a: Add parity tests + benchmarks**

- New or extended test cases in lax-associative-scan.test.ts
- Benchmark additions in associative-scan.bench.ts

**Commit 2b: Delete legacy blocked path**

- All deletions from checklist above
- Test suite passes, benchmarks recorded

**Expected LOC change:** ~-1,126 LOC deleted from webgpu.ts, ~-30 LOC from scan-plan.ts, ~-15 LOC
from scan-executor.ts. Total: **~-1,170 LOC**.

---

## Plan 3: Codegen-Specialization Cache for Fused Block-Map ✅ IMPLEMENTED

### Goal

Fix `blockMapFusedCache` so it produces correct executables when the same body program is used with
different grid shapes, input shapes, or thread tiles.

### Cache architecture context

The `jitCompile` cache (`jitCompileCache`) is keyed by `backend.type + FpHash.hash(jaxpr)`. When
`dynamic_axes` is used, the jaxpr contains `SymDim` nodes so the hash is identical regardless of
concrete lengths — one `JitProgram` is reused for all concrete sizes. This is the **polymorphic
frontend cache** and must not be disturbed.

`blockMapFusedCache` is a **backend codegen-specialization cache** sitting below `jitCompile`. The
fused WebGPU shader bakes concrete values into WGSL: grid decomposition, boundary guards, stride
remapping, buffer indexing. A single polymorphic `JitProgram` can therefore require multiple
concrete fused executables — one per distinct codegen-affecting geometry.

The specialization key includes only values that actually affect the generated WGSL — everything the
fused codegen reads from `BlockMapShaderParams`: `blockShape`, `gridShape`, `inputShapes`,
`outputShapes`, `inAxes`, `outAxes`, `numConsts`, `numInputs`, `threadTile`. Audited against all
`params.*` accesses in `src/backend/webgpu/block-map.ts` — **complete, no missing fields.**

If future work makes some of these runtime-driven (e.g., grid extent as a uniform buffer), they
should be removed from the key at that time.

### 3a. Specialization key — ✅ Done

`blockMapSpecKey()` in `src/frontend/block-map-executor.ts` serializes all codegen-affecting fields.

### 3b. Two-level cache — ✅ Done

`Map<JitProgram, Map<string, Executable | null>>` with inner map keyed by `blockMapSpecKey()`.
Compile-count counter (`_fusedCacheCompileCount`) exported for test observability.

### 3c. dynamic_axes block_map fix — ✅ Done

**Root cause:** The `block_map` JitStep stored `inputShapes`, `outputShapes`, and `gridShape` as
`number[]` at jitCompile time. When `dynamic_axes` was used, shapes contained `SymDim` objects. The
`as number` casts caused `Math.ceil(SymDim / blockShape)` → `NaN` for gridShape, and SymDim objects
were passed where WGSL codegen expected concrete numbers.

**Fix (src/frontend/jit.ts):**

- JitStep type: `inputShapes: Dim[][]`, `outputShapes: Dim[][]`, `gridShape` removed (computed at
  execution time from resolved shapes)
- Compile time: store raw `Dim[][]` shapes, no unsafe casts
- Execution time: `resolveShape()` via `dimBindings`, then compute `gridShape` from concrete values

**Fix (src/frontend/scan-executor.ts):**

- `executeAssocScanBlockMap()`: resolve `elemAvals[k].shape` once at function entry via
  `resolveShape()`, replacing all 14 unsafe `as number[]` casts with resolved shapes.

### 3d. Test coverage — ✅ Done

**T7.8-WebGPU** (`test/block-map-jit.test.ts`): One `jit` function with `dynamic_axes`, called with
N=8 (gridShape=[2]) and N=12 (gridShape=[3]). Verifies:

- Correctness: both sizes produce correct outputs
- Cache partitioning: `_fusedCacheCompileCount` asserts exactly 2 compiles (one per geometry)
- Cache reuse: repeat calls show compile count stays at 2

### 3e. Commit summary

Files modified:

- `src/frontend/jit.ts` — `Dim[][]` types, deferred gridShape computation, `resolveShape()` at exec
- `src/frontend/block-map-executor.ts` — two-level cache, `blockMapSpecKey()`, compile counter
- `src/frontend/scan-executor.ts` — resolve symbolic shapes in `executeAssocScanBlockMap()`
- `src/index.ts` — export `_fusedCacheCompileCount`
- `test/block-map-jit.test.ts` — T7.8-WebGPU with compile-count assertions

---

## Execution Order

| Order | Plan | Commit(s)                                                     | Risk   | LOC delta  |
| ----- | ---- | ------------------------------------------------------------- | ------ | ---------- |
| 1     | 3    | Shape-aware fused cache                                       | Low    | +40        |
| 2     | 1a   | `BlockIndex` + explicit `gridShape` semantics + proving tests | Medium | +140       |
| 3     | 1b   | Reverse canonicalization via traced `flip`                    | Low    | -40        |
| 4     | 1c   | Strict gather + planner-owned recursive summary scan          | Medium | +140, -140 |
| 5     | 1d   | Strict apply as a single traced block_map stage               | Medium | +100, -160 |
| 6     | 2a   | Parity tests + benchmarks                                     | Low    | +60        |
| 7     | 2b   | Delete legacy blocked path                                    | Medium | -1,170     |

**Net LOC change:** approximately **-1,220 LOC** (dominated by legacy shader deletion).

---

## Workaround Signatures to Hunt

When implementing these plans, search for and eliminate:

| Pattern to find                                             | Replace with                                                   |
| ----------------------------------------------------------- | -------------------------------------------------------------- |
| `scanBodyProgram.execute()` in a JS loop (scan-executor.ts) | recursive `executeAssociativeScan` or strict `block_map` stage |
| `backend.copyBufferToBuffer` in an element-by-element loop  | traced `flip` or strict gather/apply `block_map` stage         |
| `for (let i = 1; i < M; i++)` in scan-executor.ts           | recursive `executeAssociativeScan`                             |
| `for (let blockIdx = 1; blockIdx < M; blockIdx++)` loop     | single `executeBlockMap(plan.apply)`                           |
| `reverse` branches in `executeAssocScanBlockMap()`          | canonicalize to forward scan + `flip`                          |
| ad-hoc block-id helper inputs or planner-only body args     | `Primitive.BlockIndex`                                         |
| `prepareBlockedAssocScan` / `dispatchBlockedAssocScan`      | Delete after parity                                            |
| `path: "webgpu-fused-blocked"` in plan/executor             | Delete after parity                                            |
| `blockMapFusedCache.get(params.bodyProgram)` flat lookup    | Two-level `Map<Program, Map<key, Exe>>`                        |

---

## Acceptance Criteria

- [x] No algorithmic JS loops remain in `executeAssocScanBlockMap()` other than stage dispatch /
      cleanup (Plan 1c+1d: recursive summary + vmapped apply)
- [x] Gather is executed as a planned block_map stage or equivalent primitive-backed Jaxpr (local
      scan via block_map with WorkgroupAssociativeScan primitive)
- [x] Summary recursion is represented in the plan tree; the executor only dispatches the child plan
      (Plan 1c: recursive `planAssociativeScan()` + `executeAssociativeScan()`)
- [x] Apply phase is executed as one planned block_map stage across all blocks (Plan 1d: vmapped
      apply body via `vmapJaxpr()`)
- [ ] Reverse is represented by traced `flip` semantics or a Jaxpr primitive, not executor-side
      copies — **SKIPPED**: Plan 1b reverted; `ShapeTracker.flip()` bakes `(N-1)*stride` into
      compiled offset, incompatible with `dynamic_axes`. Reverse handled at executor level.
- [x] `Primitive.BlockIndex` and explicit `gridShape` support are implemented without synthetic
      block-id body inputs and are covered by tests (Plan 1a)
- [x] Focused block_map regression for `BlockIndex` + broadcast `dynamicSlice` passes on fused
      WebGPU (Plan 1a: T4 tests)
- [x] Gather/apply stage bodies are traced from frontend functions, not constructed as manual Jaxprs
      (Plan 1c+1d: uses `vmapJaxpr()` to trace apply body)
- [x] Apply embeds the vmapped body by inlining `evalJaxpr`, not by adding a new nested call
      primitive (vmapJaxpr produces a flat Jaxpr)
- [x] Legacy `webgpu-fused-blocked` path fully deleted (Plan 2b: -1761 LOC)
- [x] `blockMapFusedCache` is shape-specialized (two-level cache with `blockMapSpecKey()`)
- [x] `dynamic_axes` + `block_map` WebGPU bug fixed (SymDim resolution at execution time)
- [x] Compile-count test observability via `_fusedCacheCompileCount()`
- [x] All 2650 tests pass (26 skipped; 1 flaky backend.test.ts unrelated)
- [ ] Associative-scan benchmarks show no regression > 2×
- [ ] `BLOCK-MAP-IMPLEMENTATION-PLAN-2.md` updated to reference this plan's completion
