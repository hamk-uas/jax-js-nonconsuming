# Associative Scan Debt Retirement Plan

Finishes the architectural cleanup that the uniform jaxpr-based associative scan work started but
did not fully complete. The current implementation removed the legacy standalone WebGPU blocked
shader path and canonicalized reverse via `Primitive.Reverse`, but the WebGPU blocked assoc-scan
path still contains executor-owned gather/apply orchestration that should be dispatched through
shared blocked-data-movement infrastructure.

**Branch:** `block-map`

---

## Objective

Convert WebGPU blocked associative scan from a monolithic executor into a thin orchestrator that
dispatches pre-compiled stages through shared infrastructure:

```
localScan (BlockMapStage) → gatherAxisPoints → recurse → copyAxisRange + mapOverBlocks
```

The executor must stop owning algorithmic gather/apply loops and stop re-entering the planner.
Blocked-data-movement primitives live in a neutral shared module (`block-map-executor.ts`) beside
the existing copy/slice helpers they build on.

---

## Current Debt

The remaining debt is concentrated in `executeAssocScanBlockMap()` in
`src/frontend/scan-executor.ts`.

What still happens there today:

1. Manual gather of block summaries via `copyBufferToBuffer` loops (~20 lines).
2. Recursive summary scan planning inside the executor (`planAssociativeScan()` call).
3. Manual block-by-block apply orchestration (~60 lines, `for (let blockIdx = 1; ...)`).
4. Stale plan metadata (`reverse` field in plan shape that is semantically dead).

What this blocks:

1. Reuse of blocked-data-movement primitives for future structured primitives.
2. A clean planner/executor boundary.
3. Honest documentation claiming the blocked path is fully plan-driven.

---

## End State

### Plan shape

The `webgpu-block-map` plan carries pre-compiled programs. The plan is **self-similar**: the same
plan object works at every recursion level because `planAssociativeScan()` takes no N/M-dependent
arguments. The per-element body jaxpr, the block-map local-scan body, and the vmapped apply body are
all shape-independent.

Target shape:

```ts
{
  path: "webgpu-block-map";
  blockSize: number;
  numLeaves: number;
  numConsts: number;
  /** Local scan: block_map with WorkgroupAssociativeScan body. */
  localScan: BlockMapStage;
  /** Per-element body for recursive summary scan fallback / dispatch. */
  scanBodyJaxpr: Jaxpr;
  /** Vmapped apply body: [consts, prefix, block[B,…]] → [result[B,…]]. */
  applyVmapProgram: JitProgram;
}
```

Notes:

1. No `gather` stage — `BlockMapStage` cannot express asymmetric in/out blocking. Gather calls the
   generic `gatherAxisPoints()` primitive from `block-map-executor.ts`.
2. No `summaryPlan` / `DeferredAssocScanPlan` — the plan is self-similar. The executor recurses
   through `executeAssociativeScan()` with the same plan object, passing `scanBodyJaxpr` through the
   normal `ExecuteAssocScanParams` path for any fallback handling.
3. No `CopyStage` / `SliceMapStage` types — block-0 copy and prefix application use generic
   blocked-data-movement primitives from `block-map-executor.ts`. Promoting to formal plan types is
   deferred until a second consumer appears.
4. `reverse` is handled outside the blocked path by `Primitive.Reverse` canonicalization and is
   removed from the plan shape.

### Architecture: shared blocked-data-movement layer

`block-map-executor.ts` already owns the low-level copy helpers (`isSliceContiguous`,
`computeLinearOffset`, `copyBlock`) used by the block-map fallback executor. The new
blocked-data-movement primitives are built on top of these and exported from the same module:

| Primitive             | Semantics                                                    | Location                |
| --------------------- | ------------------------------------------------------------ | ----------------------- |
| `gatherAxisPoints(…)` | Gather one element per block along an axis                   | `block-map-executor.ts` |
| `copyAxisRange(…)`    | Copy a contiguous axis range across buffers                  | `block-map-executor.ts` |
| `mapOverBlocks(…)`    | Run a compiled body over blocks with gathered inputs         | `block-map-executor.ts` |
| `isSliceContiguous`   | Check if a slice is contiguous (existing, now exported)      | `block-map-executor.ts` |
| `computeLinearOffset` | Compute byte offset for multi-index (existing, now exported) | `block-map-executor.ts` |
| `copyBlock`           | Arbitrary src/dst offset copy (existing, now exported)       | `block-map-executor.ts` |

The assoc-scan executor imports these and calls them. No blocked-data-movement logic lives in
`scan-executor.ts`.

### Executor shape

`executeAssocScanBlockMap()` reduces to:

```
1. Execute localScan via executeBlockMap(plan.localScan, …)
2. If M === 1: copy to output, return
3. gatherAxisPoints(…)                          ← from block-map-executor.ts
4. executeAssociativeScan(…, same plan, …)      ← recurse through public dispatch
5. copyAxisRange(…)                             ← from block-map-executor.ts
6. mapOverBlocks(…)                             ← from block-map-executor.ts
7. Cleanup, return outputs
```

The executor allocates buffers, dispatches shared primitives, and manages lifetimes. It does not
implement algorithmic loops over blocks, does not re-enter the planner, and does not own
gather/apply semantics.

### Generic helper API design

The three new primitives use domain-neutral names and signatures:

**`gatherAxisPoints`** — Gathers one element per block from blocked arrays along axis 0.

```ts
export function gatherAxisPoints(
  backend: Backend,
  srcSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  blockSize: number,
  axisLen: number,
): Slot[];
```

For each source buffer, allocates an output of shape `[M, …rest]` where `M = ceil(axisLen / B)`, and
copies the element at index `min((i+1)*B - 1, axisLen-1)` to position `i`. "Gather points from
blocks along an axis" — not scan-specific. Any blocked reduction, blocked QR factorization, or
blocked prefix operation that needs block-tail extraction calls this.

**`copyAxisRange`** — Copies a contiguous range along axis 0 across multi-leaf slot arrays.

```ts
export function copyAxisRange(
  backend: Backend,
  srcSlots: Slot[],
  dstSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  axisStart: number,
  axisLen: number,
): void;
```

Copies elements `[axisStart, axisStart + axisLen)` from each source to the corresponding position in
each destination. Destination placement mirrors the source range, i.e.
`dst[axisStart .. axisStart + axisLen)`. "Copy a contiguous range of an axis" — not scan-specific.
The block-0 copy becomes `copyAxisRange(…, 0, min(B, N))`.

**`mapOverBlocks`** — Runs a compiled body program over a grid of blocks, materializing
point-gathered and block-gathered inputs per iteration.

```ts
export interface BlockInput {
  slots: Slot[];
  shapes: number[][];
  dtypes: DType[];
  mode: "point" | "block";
  /** For "point" mode: offset added to blockIdx to compute source index. */
  indexOffset?: number;
}

export function mapOverBlocks(
  backend: Backend,
  bodyProgram: JitProgram,
  constSlots: Slot[],
  inputs: BlockInput[],
  outputSlots: Slot[],
  outputShapes: number[][],
  outputDtypes: DType[],
  blockSize: number,
  axisLen: number,
  gridStart: number,
  gridEnd: number,
  numConsts: number,
  dimBindings?: ReadonlyMap<string, number>,
): void;
```

For each block in `[gridStart, gridEnd)`: materializes each input group according to its mode
(point-gather one element, or block-gather a `[B, …]` slice), runs the body, copies valid output
elements to the output buffer. "Apply a body over a block grid with declarative input modes" — not
scan-specific. Assoc-scan apply uses `gridStart=1, gridEnd=M`, with the prefix input as
`{ mode: "point", indexOffset: -1 }` and the local-scan input as `{ mode: "block" }`. `dimBindings`
is threaded through to `bodyProgram.execute()` so the primitive remains valid for generic callers
using symbolic shapes.

---

## Scope

### In scope

1. `BlockMapStage` type for planner-owned local-scan metadata.
2. Generic blocked-data-movement primitives in `block-map-executor.ts`.
3. Recursive summary scan through `executeAssociativeScan()` (same plan, no replanning).
4. Removal of `reverse` from the plan shape.
5. Export of existing low-level copy helpers from `block-map-executor.ts`.
6. Executor simplification.
7. Regression coverage and doc cleanup.

### Out of scope

1. Native backend acceleration for `reverseBuffer`.
2. Multi-axis `Primitive.Reverse`.
3. Mega-module support for `reverse`.
4. Unrelated scan or block-map optimizations.
5. Formal plan stage types for gather/apply (deferred until a second consumer appears).

---

## Design Constraints

1. Do not reintroduce reverse-specific execution branches into blocked assoc-scan.
2. Do not re-enter the planner from the executor (`planAssociativeScan()` must not be called).
3. Do not introduce plan types that have exactly one consumer today — use exported functions and
   promote to types when warranted by real demand.
4. Do not widen `ExecuteBlockMapParams` to encode semantics that are not block-map.
5. Preserve correctness under `dynamic_axes`.
6. Recursion through `executeAssociativeScan()` with the same plan — no N/M-dependent plan
   construction. This preserves the public dispatch point so future plan variants slot in cleanly.
7. All blocked-data-movement primitives live in `block-map-executor.ts`, not `scan-executor.ts`. The
   assoc-scan executor only imports and calls them.
8. Primitive names and signatures must not reference scan/prefix/assoc-scan concepts.

### Why self-recursion works

`planAssociativeScan()` arguments are
`(backend, bodyProgram, bodyJaxpr, numLeaves, numConsts, reverse, dimBindings)`. None depend on N or
M:

- `bodyJaxpr` operates on per-element shapes `[elemShape]` — same at every level.
- `localScanBodyProgram` processes `[B, elemShape]` blocks — B is fixed.
- `applyVmapProgram` takes `[consts, prefix, block[B,…]]` — B is fixed.

The plan is self-similar: level 0 uses it for N elements, level 1 for M=ceil(N/B) elements, etc.
Recursion terminates when M ≤ 1 (single-block fast path).

### Why recursion goes through executeAssociativeScan

The blocked executor self-recurses with the same pre-built plan for the summary scan. Two options:

- **Call `executeAssocScanBlockMap()` directly:** hard-codes that the summary always takes the
  blocked path. If a future plan variant (e.g., WASM-compiled blocked path) replaces the
  `webgpu-block-map` plan at the summary level, this direct call bypasses it.
- **Call `executeAssociativeScan()` with the same plan:** the public dispatch point routes to the
  correct handler for whatever plan path is in play. The blocked executor passes its own plan object
  unchanged; the dispatcher sees `path: "webgpu-block-map"` and routes back. This keeps the
  recursion loosely coupled to the specific executor implementation.

The second approach is correct because it preserves the public dispatch contract and makes no
assumption about which executor handles the recursive level.

### Why gather cannot be a BlockMapStage

`ExecuteBlockMapParams` uses a global `blockShape` for both input slicing and output placement. The
output placement code computes `dstStarts[ax] = blockIdx[g] * blockShape[g]`, meaning each body
output must fill `blockShape[g]` elements along the output axis. A gather body that reads `[B, …]`
input blocks but writes `[elemShape]` output elements (1 per block) has an asymmetric in/out
blocking that `ExecuteBlockMapParams` cannot represent.

Alternatives explored:

- `outAxes = [[null]]` (broadcast): all blocks write the same output — wrong.
- `blockShape=[1]` with broadcast inputs: DynamicSlice clamping at `jit.ts` L2617 bakes operand
  shape into compiled code — breaks shape-independence.
- `blockShape=[1]` with mapped inputs: 1-element blocks, not B-element blocks — wrong.

The correct approach is a generic axis-point-gather primitive in `block-map-executor.ts`.

### Why DeferredAssocScanPlan is unnecessary

The plan proposed a `DeferredAssocScanPlan` type with caching and ownership semantics. But since the
plan is self-similar (see above), the executor can simply recurse through `executeAssociativeScan`
with the same plan object. This eliminates:

- Caching logic (nothing to cache — one plan).
- Ownership questions (plan lifetime = parent plan lifetime, unchanged).
- Disposal concerns (no new allocations per recursion level).

Existing building blocks:

1. `Primitive.BlockIndex` already exists with abstract-eval / JVP / transpose / vmap support.
2. `Primitive.BlockMap` already supports explicit `gridShape`.
3. JIT block-map lowering already rewrites `BlockIndex` and threads it through fused WebGPU
   execution.
4. Block-map tests already cover `dynamicSlice`, `BlockIndex`, and explicit `gridShape`
   independently.

---

## Phase 1: Refactor Plan Shape and Export Helpers

**Files:** `src/frontend/scan-plan.ts`, `src/frontend/block-map-executor.ts`

Deliverables:

1. `BlockMapStage` type: holds plan-time metadata (body program, body jaxpr, inAxes, outAxes,
   blockShape, numConsts, numInputs) separately from runtime data (gridShape, slots, shapes).
2. Replace the `webgpu-block-map` plan payload with the target shape (above). Remove `reverse`,
   `localScanBodyProgram`/`localScanBodyJaxpr` become `localScan: BlockMapStage`, and drop dead
   executor-only-unused fields (`scanBodyProgram`, `applyVmapJaxpr`).
3. Export existing low-level copy helpers (`isSliceContiguous`, `computeLinearOffset`, `copyBlock`)
   from `block-map-executor.ts`.

Acceptance criteria:

1. `AssocScanPlan.webgpu-block-map` uses `BlockMapStage` for local scan and carries `scanBody*` /
   `applyVmapProgram` directly.
2. `reverse` is gone from the plan shape.
3. Copy helpers are exported and available to the new primitives in Phase 2.

---

## Phase 2: Add Generic Blocked-Data-Movement Primitives

**Files:** `src/frontend/block-map-executor.ts`, tests

Add the three new primitives to `block-map-executor.ts`, building on the now-exported low-level copy
helpers.

### `gatherAxisPoints`

```ts
export function gatherAxisPoints(
  backend: Backend,
  srcSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  blockSize: number,
  axisLen: number,
): Slot[];
```

For each source, allocates `[M, …rest]` output and copies element `min((i+1)*B - 1, axisLen-1)` to
position `i`, for `i = 0..M-1`. Uses `copyBufferToBuffer` with pre-computed byte offsets.

### `copyAxisRange`

```ts
export function copyAxisRange(
  backend: Backend,
  srcSlots: Slot[],
  dstSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  axisStart: number,
  axisLen: number,
): void;
```

Copies elements `[axisStart, axisStart + axisLen)` from each src to corresponding dst. Single
`copyBufferToBuffer` per leaf (axis-0 slices are always contiguous in row-major layout).

### `mapOverBlocks`

```ts
export function mapOverBlocks(
  backend: Backend,
  bodyProgram: JitProgram,
  constSlots: Slot[],
  inputs: BlockInput[],
  outputSlots: Slot[],
  outputShapes: number[][],
  outputDtypes: DType[],
  blockSize: number,
  axisLen: number,
  gridStart: number,
  gridEnd: number,
  numConsts: number,
  dimBindings?: ReadonlyMap<string, number>,
): void;
```

Iterates blocks `[gridStart, gridEnd)`. For each block, materializes inputs per `BlockInput.mode`:

- `"point"`: copies element at `blockIdx + indexOffset` from the source.
- `"block"`: allocates a full `[B, …]` temporary matching the vmapped body input shape, then copies
  `min(B, axisLen - blockIdx * B)` valid elements starting at `blockIdx * B` from the source into
  the front of that temporary. For a partial trailing block, the remaining elements in the temporary
  are padding / undefined data and must not be written back to the destination.

Runs body program on the full `[B, …]` temporaries, then copies only the valid output elements
(`min(B, axisLen - blockIdx * B)`) to the destination.

Requirements:

1. All three functions are exported from `block-map-executor.ts`.
2. None reference scan/prefix/assoc-scan in names, signatures, or implementation.
3. All three build on the existing copy helpers in the same module.
4. Each is independently testable.
5. `mapOverBlocks` forwards optional `dimBindings` to `bodyProgram.execute()`.
6. `mapOverBlocks` owns the per-iteration input lifetime pattern needed by `JitProgram.execute()`:
   temporary gathered inputs are disposed after each iteration, and shared `constSlots` are
   retained/released symmetrically around each body execution.
7. `mapOverBlocks` preserves the compiled body contract for partial trailing blocks: block-mode
   inputs are always materialized as full `[B, …]` temporaries even when fewer than `B` elements
   remain, and only the valid prefix of the body result is copied to the destination.

Acceptance criteria:

1. `gatherAxisPoints` works for any block-structured array along axis 0.
2. `copyAxisRange` handles arbitrary start/length ranges.
3. `mapOverBlocks` handles point and block input modes, including partial last blocks without
   changing the vmapped body input shape.
4. Unit tests cover boundary cases (N < B, N exactly divisible by B, partial last block).

---

## Phase 3: Replace Planner Re-entry with Public-Dispatch Recursion

**Files:** `src/frontend/scan-executor.ts`

Replace the `planAssociativeScan()` call at L676 with recursion through the public dispatch point:

```ts
// Before:
const summaryPlan = planAssociativeScan(backend, scanBodyProgram, ...);
const summaryResult = executeAssociativeScan({ plan: summaryPlan, ... });

// After:
const summaryResult = executeAssociativeScan({
   backend,
   plan,
   bodyJaxpr: plan.scanBodyJaxpr,
   ...summarySlots,
});
```

The same `plan` object is passed unchanged. `executeAssociativeScan()` sees
`path: "webgpu-block-map"` and routes to `executeAssocScanBlockMap()`, which terminates when M ≤ 1
(single-block fast path).

Requirements:

1. `planAssociativeScan` import is removed from `scan-executor.ts`.
2. The executor never re-enters the planner.
3. Recursion goes through the public `executeAssociativeScan()` dispatch point, not directly to the
   private blocked executor.
4. Uses exactly the same plan object.

Acceptance criteria:

1. No `planAssociativeScan()` call in `executeAssocScanBlockMap()`.
2. Recursive summary scan produces identical results (existing tests).
3. Multi-level recursion works for N > B² (e.g., N=70000 with B=256).

---

## Phase 4: Wire Assoc-Scan Executor to Shared Primitives

**Files:** `src/frontend/scan-executor.ts`

Replace the inline gather, block-0 copy, and apply loop with calls to the generic primitives:

```ts
// Phase 2 (gather):
const summarySlots = gatherAxisPoints(backend, localScanSlots, resolvedElemShapes, dtypes, B, N);

// Block-0 copy:
copyAxisRange(backend, localScanSlots, outputSlots, resolvedElemShapes, dtypes, 0, Math.min(B, N));

// Apply blocks 1..M-1:
mapOverBlocks(
  backend,
  plan.applyVmapProgram,
  constSlots,
  [
    { slots: scannedSummarySlots, shapes: summaryShapes, dtypes, mode: "point", indexOffset: -1 },
    { slots: localScanSlots, shapes: resolvedElemShapes, dtypes, mode: "block" },
  ],
  outputSlots,
  resolvedElemShapes,
  dtypes,
  B,
  N,
  1,
  M,
  plan.numConsts,
  dimBindings,
);
```

Requirements:

1. The executor's inline gather loop, block-0 copy, and apply loop are all replaced by calls to
   primitives from `block-map-executor.ts`.
2. No blocked-data-movement logic remains in `scan-executor.ts`.
3. The executor passes `dimBindings` through shared primitives rather than reintroducing local
   symbolic-shape handling.

Acceptance criteria:

1. No `for (let blockIdx = ...)` loop in `executeAssocScanBlockMap()`.
2. No inline `copyBufferToBuffer` gather or block-0 copy code.
3. All data-movement is done through `block-map-executor.ts` exports.

---

## Phase 5: Verify Executor is Thin Orchestration

**Files:** `src/frontend/scan-executor.ts`

After phases 1–4, `executeAssocScanBlockMap()` should read as:

```ts
function executeAssocScanBlockMap(params) {
  // 1. Resolve N, M
  // 2. Allocate localScan output buffers
  // 3. executeBlockMap(plan.localScan, …)            ← BlockMapStage dispatch
  // 4. If M === 1: copy to output, return
  // 5. gatherAxisPoints(…)                            ← from block-map-executor
  // 6. executeAssociativeScan(…, same plan, …)        ← public dispatch recursion
  // 7. copyAxisRange(…, 0, min(B, N))                 ← from block-map-executor
  // 8. mapOverBlocks(…, gridStart=1, gridEnd=M, …)   ← from block-map-executor
  // 9. Cleanup, return
}
```

Allowed responsibilities: allocate buffers, dispatch shared primitives, manage lifetimes,
single-block fast path, recurse through public dispatch.

Forbidden responsibilities: `planAssociativeScan()` calls, per-block algorithmic loops,
reverse-specific branches, any `copyBufferToBuffer` calls that are not inside shared primitives.

Acceptance criteria:

1. The function reads as stage orchestration, not algorithm implementation.
2. The only data-movement imports from `block-map-executor.ts` are generic primitives.
3. The planner/executor boundary is explicit and stable.

---

## Phase 6: Cleanup and Validation

**Files:** tests, docs

Cleanup:

1. Remove stale `reverse` field from plan shape.
2. Remove dead plan fields made obsolete by public-dispatch recursion (`scanBodyProgram`,
   `applyVmapJaxpr`).
3. Update documentation so "plan-driven" matches the code.
4. Remove stale comments describing JS-level gather/apply as part of the design.

Validation:

1. Focused regressions for:
   - `gatherAxisPoints` boundary behavior (partial trailing block, non-identity associative ops).
   - `copyAxisRange` correctness for arbitrary start/len combinations.
   - `mapOverBlocks` with point and block input modes, partial last block.
   - Public-dispatch recursion for N > B² (multi-level recursion).
   - Absence of reverse-specific blocked execution branches.
2. Existing associative scan parity tests remain green (1-D cumsum/cumprod/cummax, non-zero axis,
   pytree affine composition, autodiff, Kalman filter, DLM compose, `jit(grad(…))`).

Acceptance criteria:

1. WebGPU blocked assoc-scan no longer relies on executor-side gather/apply orchestration.
2. Generic primitives are independently tested.

---

## Execution Order

1. Add `BlockMapStage` type, restructure plan shape, export existing copy helpers.
2. Add `gatherAxisPoints`, `copyAxisRange`, `mapOverBlocks` to `block-map-executor.ts`.
3. Replace planner re-entry with public-dispatch recursion.
4. Wire assoc-scan executor to use shared primitives (remove inline loops).
5. Verify executor is thin orchestration.
6. Cleanup stale metadata, add regression tests, validate.

---

## Risks

1. Self-recursion depth for very large N could cause stack overflow. With B=256, N=2^32 requires
   only 4 levels. Not a practical risk.
2. Primitives are not fused — each is a separate JS-side loop over M blocks. This matches the
   current implementation's performance characteristics. Fusing into GPU dispatches is follow-up.
3. `mapOverBlocks` adds a `BlockInput` interface — minimal new surface area, but worth watching that
   it doesn't grow assoc-scan-specific fields over time.

Mitigations:

1. B=256 gives ceil(log₂₅₆(N)) recursion levels. Stack depth is bounded by ~5 for any realistic N.
2. Document that fused GPU paths are follow-up performance work, not architectural debt.
3. Code review `BlockInput` additions to ensure they remain generic.

---

## Definition of Done

1. `executeAssocScanBlockMap()` contains no inline data-movement loops.
2. The executor never calls `planAssociativeScan()`.
3. Recursive summary scan goes through `executeAssociativeScan()`, not through the private blocked
   executor directly.
4. Three generic primitives (`gatherAxisPoints`, `copyAxisRange`, `mapOverBlocks`) are exported from
   `block-map-executor.ts` with domain-neutral signatures.
5. Local scan uses `BlockMapStage` dispatched via `executeBlockMap()`.
6. `reverse` is removed from the plan shape.
7. Regression coverage exists for primitive boundary behavior and multi-level recursion.
8. No scan/prefix/assoc-scan references appear in the generic primitive names or signatures.
9. Documentation matches the code.

---

## When to Promote Primitives to Plan Types

The three primitives have generic signatures and live in a neutral module. If a second consumer
appears (e.g., blocked QR, blocked eigenvalue), promoting to formal plan stage types requires only:

1. Wrapping the function parameters in a stage struct (e.g., `GatherAxisPointsStage`).
2. Adding the struct to the composite plan type.
3. Adding a dispatch case in the consuming executor.

No behavioral change. The primitive implementation becomes the stage executor's body.

---

## Follow-Up Work (Not Part of This Plan)

1. Native backend acceleration for `reverseBuffer`.
2. Multi-axis `Primitive.Reverse`.
3. WASM mega-module support for `reverse`.
4. Fused GPU gather/apply (eliminate per-block JS dispatch overhead).
5. Further blocked assoc-scan performance tuning after the architecture is clean.

These should be treated as performance work, not as prerequisites for retiring the remaining
architectural debt.
