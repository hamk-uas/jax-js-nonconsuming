# Associative Scan Debt Retirement Plan

Two remaining debts:

1. **axis ≠ 0 fallback** — `jit(vmap(assocScan))` falls back to JS Kogge-Stone because vmap shifts
   the scan axis to 1+. The compiled WASM and WebGPU paths hard-code contiguous axis-0 layout.
2. **Remaining skipped tests** — 4 unconditional `test.skip` entries and ~21 conditional skips.

**Branch:** `block-map`

---

## Part A: Native axis ≠ 0 for compiled AssocScan

### Problem

The user-facing `associativeScan` normalizes axis via `moveaxis(src, normAxis, 0)` before calling
`bind(Primitive.AssociativeScan, …, { axis: 0 })`. This works in eager mode. But `vmap` inserts a
batch dimension at axis 0 and shifts the scan axis: `axis: axis + 1`. So
`jit(vmap(associativeScan))` always hits the `axis !== 0` guard in `planAssociativeScan` and falls
back to JS Kogge-Stone — a ~10–100× penalty for large N.

### Design options

| Option | Approach                                                                                                  | Pros                                                                                     | Cons                                                                                                                     |
| ------ | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **A1** | Rewrite the vmap rule to force scan axis back to 0                                                        | Small patch                                                                              | Shortcut: hides the layout problem, couples semantics to one transform, and does not build reusable axis-aware machinery |
| **A2** | Make native assoc-scan paths axis-aware via reusable generic helper generalization and strided addressing | Correct fix; reusable for future blocked primitives; no fallback; no extra JS dispatches | Deeper change across planning, helpers, and native codegen                                                               |
| **A3** | Special-case `jit(vmap(assocScan))` only                                                                  | Can be made to pass T7.13 quickly                                                        | Worst option: special-case debt, not reusable, breaks AEP                                                                |

**Decision: A2 — native axis-aware machinery.**

Rationale: A1 is a shortcut. It would make one composition work by reshuffling transforms rather
than teaching the native execution paths what layout they are actually operating on. That increases
technical debt and does not help future blocked JAX primitives. The correct fix is to make the
compiled assoc-scan implementations understand arbitrary scan-axis layout through reusable generic
helper logic and offset computation, then let `vmap`, `jit`, and future transforms ride on that
machinery.

### Implementation plan

**Step 1: Generalize the existing blocked-data-movement helpers, not the transform**

The reusable machinery already exists in the right place: `gatherAxisPoints`, `copyAxisRange`, and
`mapOverBlocks` in `src/frontend/block-map-executor.ts`, plus the already axis-agnostic low-level
helpers `copyBlock`, `computeLinearOffset`, and `computeStrides`. The problem is that the three
high-level helpers hard-code axis 0 through `.slice(1)` and contiguous offset formulas.

The fix should therefore be to extend those generic helpers with an `axis: number` parameter and
derive the needed layout facts from `(shape, axis, dtype)` locally:

```ts
export function gatherAxisPoints(
  backend: Backend,
  srcSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  axis: number,
  blockSize: number,
  axisLen: number,
): Slot[];

export function copyAxisRange(
  backend: Backend,
  srcSlots: Slot[],
  dstSlots: Slot[],
  srcShapes: number[][],
  dtypes: DType[],
  axis: number,
  axisStart: number,
  axisLen: number,
): void;

export function mapOverBlocks(
  backend: Backend,
  bodyProgram: JitProgram,
  constSlots: Slot[],
  inputs: BlockInput[],
  outputSlots: Slot[],
  outputShapes: number[][],
  outputDtypes: DType[],
  axis: number,
  blockSize: number,
  axisLen: number,
  gridStart: number,
  gridEnd: number,
  numConsts: number,
  dimBindings?: ReadonlyMap<string, number>,
): void;
```

Required behavior:

1. For `axis === 0`, keep the current single-copy fast path.
2. For `axis > 0`, operate fiber-by-fiber using generic stride math derived from the shape.
3. Reuse the same helper surface for future blocked primitives; do not add assoc-scan-specific
   helper variants or vmap-only shims.
4. Keep the executor thin: the planner/executor passes `axis`, the helper layer owns the copy
   strategy.

This preserves reusability without introducing an `AxisLayout` object that duplicates trivially
derivable stride information.

**Step 2: Make the WebGPU blocked path use the same generic axis-aware helper layer**

This is feasible without backend-specific special cases:

1. `tryBuildBlockMapAssocScanPlan` should accept `axis` and emit `inAxes` / `outAxes` using that
   axis instead of literal `[0]`.
2. The block-map WebGPU shader generation is already axis-aware through `inAxes` / `outAxes`; no new
   WGSL architecture is needed.
3. `executeAssocScanBlockMap` should thread `axis` through to `gatherAxisPoints`, `copyAxisRange`,
   and `mapOverBlocks` and stop reconstructing axis-0-only shapes with `shape.slice(1)`.

No transform rewrite, no pre-transpose shim, and no assoc-scan-only host path.

Notable details in `tryBuildBlockMapAssocScanPlan`:

- The block-shaped `elemAvals` are currently built as `[blockSize, ...perElem.shape]`, which
  prepends the block dimension at position 0. For `axis > 0` the block dimension must be inserted
  _at_ `axis` (e.g. `[B_batch, blockSize, ...]` for axis=1), not prepended.
- `applyVmapDims` currently maps block leaves on axis 0. It must map on the scan axis instead.
- The recursive summary scan in `executeAssocScanBlockMap` correctly uses `axis: 0` because summary
  slots are freshly allocated with the scan dimension first. However, the summary shape construction
  (`resolvedElemShapes[k].slice(1)`) must strip axis `axis`, not axis 0.

**Step 3: Teach WASM blocked assoc-scan boundary copies about non-zero scan axes**

The WASM codegen and its internal Kogge-Stone core should not change at all. Instead, the strided
copy-in/copy-out should live in the JS-level dispatcher (`dispatchBlockedAssociativeScan` in
`wasm.ts`):

1. For `axis === 0`: call the existing WASM module directly with the input/output slots (fast path,
   unchanged).
2. For `axis > 0`: a. Allocate contiguous temporary slots (same total bytes as the input/output
   leaves). b. Strided-gather from input slots into temporaries using fiber-by-fiber
   `copyBufferToBuffer` (one call per outer fiber per leaf). c. Call the existing WASM module with
   the contiguous temporaries as inputs and outputs. d. Strided-scatter from temporary outputs back
   to the real output slots. e. Free the temporaries.

This avoids modifying the generated WASM module entirely. The codegen, internal SoA layout, and SIMD
paths are completely untouched. The strided gather/scatter logic reuses the same
`(shape, axis, dtype)` → `(fiberBytes, outerElems, outerStride)` derivation as the generalized
helpers from Step 1, keeping the stride math in one place.

**Step 4: Remove the axis guard only after native parity exists**

`planAssociativeScan` should keep rejecting `axis !== 0` until both WASM and WebGPU paths are truly
axis-aware. The guard must be deleted only when:

1. `jit(vmap(assocScan))` takes the native path on WASM and WebGPU.
2. The address formulas and blocked-data-movement helpers are validated for axis 1+.
3. No transform-specific shims are required.

**Step 5: Benchmarks and path assertions**

Bench `jit(vmap(assocScan))` before/after on both WASM and WebGPU. Add explicit tests asserting the
native path is used, so we do not silently regress back to JS fallback.

### Architecture guardrails

The feasibility review changes the architecture in one important way: we should solve the actual
problem with reusable generic helper generalization, not invent a new policy or packed-region layer.

Guardrails:

1. No assoc-scan-only axis hacks in `vmap.ts`, `scan-executor.ts`, or JIT planning.
2. No pre-transpose / post-transpose shim that hides the layout issue behind extra copies.
3. No new speculative `layoutPolicy` API until there is a real second native strategy to choose
   between.
4. No duplicated helper families. The reusable surface remains the existing blocked-data-movement
   layer, extended to arbitrary axes.
5. Any new stride logic should live with the generic copy helpers, not get reimplemented in each
   caller.

This keeps debt flat while still producing reusable machinery for future blocked primitives.

### Files to modify

| File                                 | Change                                                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `src/frontend/scan-plan.ts`          | Thread `axis` through planning and delete the axis guard only after native parity                           |
| `src/frontend/block-map-executor.ts` | Make `gatherAxisPoints` / `copyAxisRange` / `mapOverBlocks` axis-aware and reusable via generic stride math |
| `src/frontend/scan-executor.ts`      | Pass `axis` into the generic blocked-data-movement layer and stop rebuilding axis-0-only shapes             |
| `src/backend/wasm.ts`                | Add strided copy-in / copy-out for non-zero scan axes while keeping the internal blocked core contiguous    |
| `src/frontend/vmap.ts`               | Keep transform semantics simple; no special assoc-scan-only axis hack                                       |
| `test/block-map-jit.test.ts`         | Add path assertions so `jit(vmap(assocScan))` native compilation is enforced                                |
| `bench/associative-scan.bench.ts`    | Add `jit(vmap(assocScan))` benchmark for WASM and WebGPU                                                    |
| `src/frontend/jit.ts` / plan caches  | No new policy/cache dimension required; only ensure axis-aware plans remain cache-correct                   |

### Risks

- **WASM codegen complexity**: Stride-aware addressing touches low-level pointer arithmetic.
  Mitigate with byte-for-byte address tests on synthetic shapes and axis positions.
- **WebGPU helper generalization**: `gatherAxisPoints` / `copyAxisRange` / `mapOverBlocks` must stay
  generic, not accrete assoc-scan-only flags. Mitigate with shared axis/stride derivation and tests
  that exercise non-assoc-scan-shaped buffers.
- **Nested transforms**: `vmap(vmap(assocScan))`, `grad(vmap(assocScan))`, and symbolic shapes all
  need coverage. Mitigate with explicit transform-composition tests rather than trusting the old
  axis-0 assumptions.
- **Command-encoder overhead on WebGPU**: non-zero-axis copies become fiber-wise buffer copies
  rather than one contiguous copy. Mitigate by keeping the axis-0 fast path, batching copies in one
  submit, and benchmarking realistic vmap batch sizes.
- **Over-generalizing too early**: speculative planner/policy layers would add debt without solving
  a present problem. Mitigate by keeping the change set focused on reusable helper generalization
  only.

---

## Part B: Remaining Skipped Tests

### Unconditional skips (4 tests)

| #   | Test                                                | File                       | Root Cause                                                                                                                                                                      | Action                                                                                                                                                                                                                                                          |
| --- | --------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `shmem budget exceeded logs fallback on WebGPU`     | block-map-jit.test.ts:367  | GPU-dependent device limits and current allocation shape make the hard-coded over-budget case brittle across GPUs and compiler changes                                          | **Fix root cause**: query the actual device/workgroup budget from the backend and derive a deterministic over-budget case from those limits and current allocation behavior. No skip, no hard-coded NVIDIA assumption.                                          |
| 2   | `P2c: non-aligned tiledMatmul produces no select()` | block-map-jit.test.ts:1670 | `padConcrete` calls DUS on axis 1 for K-dimension padding, but the DUS JIT step only supports axis=0 (the slice is non-contiguous for axis>0)                                   | **Implement axis-aware DUS as a fiber loop**: for axis>0, emit `product(shape[0:axis])` separate `copyBufferToBuffer` calls (one per outer fiber) instead of a single contiguous copy. Independent of the assocScan work — the assocScan path does not use DUS. |
| 3   | `P2c: tiny matrices fall back to mask-based pad`    | block-map-jit.test.ts:1715 | Assertion about shader contents may be wrong — needs actual shader dump to calibrate                                                                                            | **Fix test**: Run the body, capture shader, inspect for `select(`. If the shader doesn't use `select(` (because the body is too small for block-map and runs as plain kernel), update the assertion.                                                            |
| 4   | `vmap(grad(foriLoop))`                              | block-map.test.ts:293      | Leaks 1 int32[] from `zerosLike(i)` in ForiLoop JVP rule inside nested PE/JVP/vmap trace. Anonymous const tracking defers disposal but deferred cleanup doesn't fire correctly. | **Fix root cause**: retire the anonymous-const leak path in nested makeJaxpr/JVP/PE composition. Either eliminate anonymous const creation for traced zeros or fix deferred cleanup so nested traces discharge them correctly.                                  |

### Conditional skips (~21 tests)

| Category                   | Count | Condition                              | Status                                                                    |
| -------------------------- | ----- | -------------------------------------- | ------------------------------------------------------------------------- |
| `skipIf(device === "cpu")` | 7     | AD tests for QR/foriLoop that need JIT | **Correct**: CPU backend is debug-only, no JIT. Not bugs.                 |
| `skipIf(!hasWebGPU)`       | 12    | blocked-data-movement tests            | **Correct**: These test WebGPU-specific blocked assocScan infrastructure. |
| `skipIf(!canSpinWait)`     | 1     | orchestrator worker test               | **Correct**: `Atomics.wait` blocked on browser main thread.               |
| `skipIf` in numpy          | 2     | Backend-specific dtype tests           | **Correct**: f16/f64 availability varies.                                 |

**These 21 conditional skips are architecturally correct.** They skip on backends that fundamentally
cannot support the tested feature. No action needed.

---

## Execution Order

1. **B1: shmem budget test** — derive deterministic over-budget case from real device limits and
   current allocation behavior
2. **B3: tiny matrices test** — shader dump + assertion calibration
3. **B4: foriLoop anonymous-const leak** — root-cause fix in tracing/disposal machinery
4. **B2: axis-aware DUS/copy-like JIT support** — unblocks `padConcrete` test the right way
5. **A2.1: generic helper generalization** — add `axis` to reusable blocked-data-movement helpers
6. **A2.2: WebGPU blocked path consumes the generalized helpers** — no extra JS fallback path, no
   backend-specific shim
7. **A2.3: WASM strided boundary copies around the existing blocked core** — remove native axis=0
   assumption without rewriting the internal Kogge-Stone path
8. **A2.4: native-path benchmarks and path assertions** — lock in the regression coverage once both
   backends are working

---

## Success Criteria

- `jit(vmap(assocScan))` hits `compiled-loop-blocked` or `webgpu-block-map` path with **no JS
  fallback**
- T7.13 passes on the native path, and the test asserts the path choice
- `grad(vmap(assocScan))` and `vmap(grad(assocScan))` produce correct results
- Unconditional `test.skip` count drops from 4 to 0, with root-cause fixes rather than skips
- No new test failures (2677+ passing)
- The solution is implemented through reusable generic helpers, not assoc-scan-only or vmap-only
  special cases
- Future blocked primitives can reuse the same axis-aware helper layer without inventing another
  copy/update path
- Benchmark results show native-path usage and remain reproducible across the tested backends

---

## Resolved: `Primitive.Reverse` missing from `specialBlackPrimitives`

**Discovered via:** Downstream `dlm-js` regression — 69 tests failing with
`Error: jit: Reverse input is not imm` when running `jit(associativeScan({ reverse: true }))`.

**Root cause:** `Primitive.Reverse` was absent from the `specialBlackPrimitives` list in
`splitGraphDataflow` (jit.ts). This meant the dataflow analysis didn't force Reverse inputs to be
materialized as `"imm"` buffers. When the JIT compiled a reverse associative scan, the
`core.reverse()` call's input could remain a lazy `"exp"` expression, causing the Reverse handler to
throw.

**Fix:** Added `Primitive.Reverse` to `specialBlackPrimitives` (jit.ts ~L2834).

**Test added:** `jit(reverse cumsum)` in `test/lax-associative-scan.test.ts` — verifies
`jit(associativeScan({ reverse: true }))` produces correct results.

---

## Historical Context

The previous version of this plan covered the blocked-data-movement refactor (converting
`executeAssocScanBlockMap` from monolithic executor to thin orchestrator using shared
`gatherAxisPoints`/`copyAxisRange`/`mapOverBlocks` primitives). That work is **complete** — the
executor, plan shape, and shared primitives are all in place. The remaining debt is axis ≠ 0 support
(Part A above) and test cleanup (Part B).
