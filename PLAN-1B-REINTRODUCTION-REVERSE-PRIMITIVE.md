# Plan 1b Reintroduction — Reverse Primitive Canonicalization Plan

Reintroduces the intent of Plan 1b without reviving the failed `flip` / `ShapeTracker` rewrite. The
core idea is to canonicalize `associativeScan(..., { reverse: true })` into a forward scan plus a
new semantic `Reverse` primitive, rather than trying to encode reverse through view metadata.

**Branch:** `block-map`, HEAD after Plans 1a, 1c+1d, 2a, 2b, 3

---

## Current State: Why The Original Plan 1b Was Reverted

The original Plan 1b attempted to rewrite:

```ts
associativeScan(fn, xs, { reverse: true }) ==
  flip(associativeScan(fn, flip(xs, [axis]), { reverse: false }), [axis]);
```

This passed the dedicated assoc-scan tests, but failed the polymorphic-shape regression in
`test/polymorphic-shapes.test.ts`:

```ts
jit((data) => lax.associativeScan(add, data, { reverse: true }), {
  dynamic_axes: { 0: "T" },
});
```

### Root Cause

`View.flip()` in `src/shape.ts` bakes a concrete offset into the traced view:

```ts
offset += (s - 1) * this.strides[i];
```

When tracing happens at one concrete length and the resulting JIT program is reused at another
length via `dynamic_axes`, the baked offset is wrong. `flip` is currently a view transform, not a
runtime operation.

### Why This Matters

The original Plan 1b was trying to use `flip` as a semantic reverse operator. That is the wrong
abstraction boundary in a polymorphic-shape system:

- `flip` is a zero-copy view rewrite
- reverse canonicalization needs runtime-length-correct semantics
- `ShapeTracker` today assumes concrete offsets / strides

Making the original design work would require symbolic offsets in `View` / `ShapeTracker`, which is
far larger than a Plan 1b-sized change.

---

## Proposed Direction

Introduce a new first-class `Reverse` primitive and canonicalize reverse assoc-scan through it.

Canonical form:

```ts
associativeScan(fn, xs, { reverse: true }) ==
  reverse(associativeScan(fn, reverse(xs), { reverse: false }));
```

This preserves the architectural goal of Plan 1b:

- no reverse-specific branches in `executeAssocScanBlockMap()`
- reverse becomes normal traced program structure
- the WebGPU block-map assoc-scan plan no longer carries its own reverse behavior

But unlike the old plan, `Reverse` is materialized at execution time and therefore remains correct
under `dynamic_axes`.

---

## Design Principles

1. **Do not change `Flip` semantics.** `Flip` remains the existing view / reshape-style primitive.
2. **Do not teach `ShapeTracker` symbolic offsets as part of this work.** That is a separate
   architecture project.
3. **Make `Reverse` a general array primitive.** It should not be assoc-scan-specific.
4. **Prefer a dedicated JIT step over encoding reverse through `block_map`.** `block_map` is a good
   fallback implementation strategy, but not the cleanest long-term IR representation.
5. **Canonicalize reverse at the library layer first.** Only move it lower if eager semantics force
   it.
6. **Treat polymorphic length as a first-class requirement.** The generic IR and library
   canonicalization must support symbolic lengths even if some backend initially falls back to a
   less efficient implementation.
7. **Do not distort the architecture for one backend.** Backend-specific performance gaps are
   acceptable; backend-specific semantic forks in the IR or library layer are not.

---

## Backend Policy

`Reverse` is a semantic primitive with one canonical lowering path in the frontend and JIT IR.
Backends are allowed to differ in performance strategy, but not in semantics or IR shape.

This means:

- the library layer always canonicalizes reverse assoc-scan structurally through `Reverse`
- the JIT always models reverse as its own runtime operation, not as a hidden assoc-scan variant
- a backend may provide a fast native reverse path, or fall back to a generic implementation
- lack of an efficient reverse implementation on one backend must not force reverse-specific logic
  back into assoc-scan planning or execution on all backends

In practice, architectural duplication reduction wins over backend-specific special casing. A slower
but semantically clean fallback is preferable to reintroducing planner or executor duplication.

---

## Phase 1: Add `Primitive.Reverse`

**Files:**

- `src/frontend/core.ts`
- `src/frontend/array.ts`
- `src/frontend/jvp.ts`
- `src/frontend/linearize.ts`
- `src/frontend/vmap.ts`
- `src/index.ts`

### 1.1 Add the primitive

Add `Primitive.Reverse` near `Flip` in `src/frontend/core.ts`.

Primitive params:

```ts
[Primitive.Reverse]: { axis: number }
```

Semantics:

- same shape as input
- same dtype as input
- materialized value transform, not a view transform

Scope choice:

- phase 1 targets a single axis only
- assoc-scan only needs axis `0`, so this keeps the first implementation tight
- future multi-axis support can be added either by extending the primitive or by composing multiple
  `Reverse` nodes, without changing the canonical assoc-scan design

### 1.2 Add the public frontend helper

Add a `reverse(x, axis)` helper beside `flip(x, axis)` in `src/frontend/core.ts`.

Expected usage shape:

```ts
const y = reverse(x, 0);
```

### 1.3 Eager implementation

In `src/frontend/array.ts`, implement eager `Reverse` as a materialized operation over a concrete
shape.

Initial implementation target:

- correctness first
- materialization is acceptable
- polymorphic tracing correctness does not matter here because eager execution always has concrete
  shapes

Preferred eager strategy:

- reuse the existing concrete-shape `flip` view machinery only as an eager local implementation
- immediately materialize the result into contiguous storage

In other words, eager `Reverse` may be implemented as "flip view, then contiguize" as long as the
primitive's JIT lowering remains a dedicated runtime operation and does not route through
`ShapeTracker.flip()`.

### 1.4 Transform rules

Implement the same rule family that `Flip` has, but semantically rather than through reshape:

- **JVP:** linear, so `d(reverse(x)) = reverse(dx)`
- **Transpose / VJP:** self-inverse, so cotangent is `reverse(ct)`
- **Vmap:** axis bookkeeping analogous to `Flip`

### 1.5 Exports

Export `reverse` from `src/index.ts`.

### Deliverable

A new primitive exists, is callable from user code, and works in eager mode, JVP, transpose, and
vmap.

---

## Phase 2: Add JIT support for `Reverse`

**Files:**

- `src/frontend/jit.ts`
- optional: `src/frontend/jaxpr.ts` if any simplification hooks are desirable

### 2.1 Add a dedicated JIT step

Extend `JitStep` with a `reverse` variant.

Suggested structure:

```ts
{
  type: "reverse";
  input: JitId;
  output: JitId;
  axis: number;
  axisSize: Dim;
  innerBytes: number;
  totalBytes: SizeExpr;
  dtype: DType;
}
```

Notes:

- `axisSize` must remain potentially symbolic and be resolved through `dimBindings` at runtime
- `innerBytes` should remain concrete: bytes for one slice along the reversed axis
- `totalBytes` is the allocation size and should reuse the existing `SizeExpr` machinery
- this step should not go through `reshapeJit`
- this keeps the IR honest: `Reverse` is not a view op

Execution model:

- for phase 1, reverse a single axis by copying `axisSize` contiguous slices of size `innerBytes`
- this is enough for assoc-scan and matches the current executor's operational pattern
- the runtime path must use resolved symbolic lengths when `axisSize` is a `SymDim`

### 2.2 Add `jitCompile` handling

Handle `Primitive.Reverse` similarly to `dus`, `scatter_add`, `scan`, and `assoc_scan`:

- flush pending kernels if necessary
- allocate output buffer
- record a `reverse` step
- do not lower through `jitRules[Primitive.Flip]`

Compilation details:

- compute `innerBytes` from the concrete trailing dimensions and dtype byte width
- store the reversed axis length as `Dim`, not a baked concrete number
- store `totalBytes` as a `SizeExpr` so polymorphic allocation follows the same path as other
  dedicated steps

### 2.3 Add execution / pprint / liveness plumbing

Update:

- `JitProgram.pprint()`
- `JitProgram.execute()`
- `stepUsesId()`
- any step counters / debugging utilities that depend on exhaustive matching

Execution details:

- resolve symbolic `axisSize` through `dimBindings` exactly where execution needs a concrete loop
  bound
- keep the generic execute path authoritative; backend fast paths should be optional accelerators,
  not the only correct implementation

### 2.4 Mega-module rejection

Add `"reverse"` to the rejection list in `src/backend/wasm/mega-module.ts`
(`canCompileToMegaModule`), alongside `scan`, `dus`, `scatter_add`, `assoc_scan`, `block_map`, and
`workgroup_assoc_scan`.

### Deliverable

`jit(reverse(x))` compiles as a dedicated runtime operation and does not depend on ShapeTracker
offset rewrites.

---

## Phase 3: Canonicalize assoc-scan reverse through `Reverse`

**Files:**

- `src/library/lax-associative-scan.ts`
- maybe `src/frontend/core.ts` only if assoc-scan params are cleaned up in the same change

### 3.1 Rewrite library-level reverse handling

Current code in `src/library/lax-associative-scan.ts` uses `flip` before and after the scan. Replace
that with `reverse`.

```ts
if (reverse) {
  working = movedArrays.map((a) => core.reverse(a, 0) as Array);
}
```

and similarly in post-processing.

### 3.2 Normalize the primitive binding

The goal is that compiled assoc-scan sees only the forward case:

- bind `Primitive.AssociativeScan` with `reverse: false`
- represent reverse entirely through surrounding `Reverse` nodes

This is the architectural pivot of the plan. Once this lands, reverse semantics stop living inside
assoc-scan execution.

### 3.3 Keep scope tight

Do this in the library layer unless an eager/tracing mismatch forces moving it lower.

The library layer is preferred because:

- the canonical form is explicit
- the executor/planner no longer need reverse awareness
- no JAXPR mutation pass is needed just to normalize assoc-scan

### Deliverable

Assoc-scan reverse is represented structurally as `Reverse(AssocScan(Reverse(xs)))`.

---

## Phase 4: Backend execution strategy

**Files:**

- `src/backend.ts`
- `src/backend/webgpu.ts`
- `src/backend/wasm.ts`
- optionally `src/backend/cpu.ts`

### Preferred architecture

Keep one canonical `reverse` step in the JIT executor, with optional backend acceleration.

Add an optional backend hook for reverse execution, for example:

```ts
reverseBuffer?(
  input: Slot,
  output: Slot,
  axisSize: number,
  innerBytes: number,
  dtype: DType,
): void;
```

If a backend does not implement this hook, `JitProgram.execute()` uses the generic fallback.

The important contract is that the generic path remains correct for symbolic lengths and is the
semantic source of truth.

### 4.1 Generic fallback

Fallback should prioritize correctness over performance.

Preferred first fallback form:

- resolve `axisSize` at execution time
- perform an element-slice reversal with `copyBufferToBuffer`
- keep this implementation in the generic JIT execute path rather than pushing semantic logic back
  into assoc-scan code

This is acceptable for the first landing because the current assoc-scan reverse path is already
worse than this architecturally, and because it preserves one clean semantic path across backends.

### 4.2 WebGPU fast path

Add a single-dispatch reverse kernel in `src/backend/webgpu.ts`.

Design goal:

- replace O(N) JS command emission with O(1) dispatches
- use runtime-resolved axis length from `dimBindings`
- target the single-axis case used by assoc-scan first

Important: this remains a general backend utility, not assoc-scan-only infrastructure.

### 4.3 WASM fast path

Add a tight loop implementation in `src/backend/wasm.ts`.

Initial version can be simple:

- contiguous single-axis reversal for assoc-scan cases first
- preserve symbolic-length correctness by resolving the axis size at execution time
- extend to broader cases later only if there is a real consumer

### 4.4 Backend asymmetry policy

Some backends may never justify a highly optimized reverse kernel. That is acceptable.

The requirement is:

- semantic correctness under concrete and polymorphic lengths on every backend
- one canonical frontend and JIT design across all backends
- optimization optional per backend

This avoids backend-specific tech debt while still allowing WebGPU and WASM to optimize where the
payoff is real.

### Deliverable

`Reverse` has a backend execution story that is independent of ShapeTracker and suitable for JIT +
`dynamic_axes`.

---

## Phase 5: Remove reverse branches from assoc-scan execution

**Files:**

- `src/frontend/scan-executor.ts`
- `src/frontend/scan-plan.ts`
- potentially `src/frontend/core.ts`

### 5.1 Delete executor-side reverse handling

Delete the current manual reverse logic from `executeAssocScanBlockMap()`:

- reverse input copies before local scan
- reverse-on-M=1 fast path
- reverse final output after apply

These are the loops currently living in the assoc-scan executor and are the exact host-side logic
Plan 1b was meant to eliminate.

### 5.2 Simplify the plan type

After canonicalization, the WebGPU block-map assoc-scan path no longer needs `reverse` in its plan
payload.

Expected simplifications:

- forward local scan only
- forward recursive summary scan only
- forward vmapped apply only
- no reverse-specific resource management in executor cleanup

### 5.3 Decide whether assoc-scan primitive params still need `reverse`

Two acceptable end states:

- **Preferred:** `Primitive.AssociativeScan` keeps a `reverse` parameter for compatibility, but the
  library path always binds it as `false`
- **Stricter follow-up:** remove `reverse` from primitive params entirely after callers are cleaned
  up

For the first landing, keep the parameter if it reduces churn.

### Deliverable

`executeAssocScanBlockMap()` becomes reverse-agnostic.

---

## Optional Alternative: Implement `Reverse` Through `block_map`

This is architecturally acceptable but not preferred.

Potential shape:

- `gridShape=[N]`
- `blockShape=[1]`
- body reads `x[N - 1 - blockIndex()]`

Pros:

- reuses existing `block_map` machinery
- naturally handles WebGPU, WASM, fallback through one abstraction

Cons:

- `Reverse` is a general array op, not a block_map-specific concept
- less direct than a dedicated JIT step
- introduces more planner complexity for a primitive that should be simple

Recommendation:

- use this only as a fallback implementation strategy or prototype
- keep the public IR as `Primitive.Reverse`
- do not let a `block_map`-based backend convenience path leak back into the library or assoc-scan
  plan shape

---

## Tests

### Primitive correctness

Add tests for:

- eager `reverse([1,2,3,4]) -> [4,3,2,1]`
- `reverse` on non-scalar trailing shapes
- `reverse` along a non-zero axis if supported after the first landing
- `jit(reverse(x))`
- `jit(reverse(x), dynamic_axes)` across multiple concrete lengths

Suggested files:

- `test/basic.test.ts`
- `test/polymorphic-shapes.test.ts`

### Transform correctness

Add tests for:

- `jvp(reverse)`
- `grad(sum(reverse(x)))`
- `vmap(reverse)`

Suggested files:

- `test/transform-compositions.test.ts`
- `test/ad-gaps.test.ts` or another transform-focused test file if more appropriate

### Assoc-scan regression coverage

Must include:

- eager `associativeScan(reverse=true)`
- `jit(associativeScan(reverse=true))`
- polymorphic reverse assoc-scan regression from `test/polymorphic-shapes.test.ts`
- WebGPU block-map reverse parity
- parity on at least one non-WebGPU backend so WebGPU-specific optimization is not mistaken for the
  semantic fix

Suggested files:

- `test/lax-associative-scan.test.ts`
- `test/block-map-jit.test.ts`
- `test/polymorphic-shapes.test.ts`

### Known non-goal

Do not treat the existing `jit(vmap(assocScan))` large-N bug as part of this work. Keep that test
separate so Plan 1b reintroduction is not blocked by unrelated existing behavior.

---

## Benchmarks

**File:** `bench/associative-scan.bench.ts`

Add focused reverse benchmarks:

- forward assoc-scan baseline
- reverse assoc-scan current path vs new path
- N = 256, 1024, 4096
- scalar and small-vector cases

Acceptance target:

- WebGPU reverse path is materially better than the current executor copy loops
- WASM may be neutral in the first landing if the initial implementation is correctness-first
- generic fallback performance is not itself an acceptance blocker as long as the canonical design
  and polymorphic correctness are preserved

---

## Recommended Commit Sequence

### Commit A

`feat: add Reverse primitive and transform rules`

Includes:

- `Primitive.Reverse`
- eager implementation
- JVP / transpose / vmap rules
- exports

### Commit B

`feat: add reverse JIT step and generic execution`

Includes:

- new `JitStep`
- `jitCompile` support
- generic `JitProgram.execute` support
- symbolic axis-length resolution through runtime bindings

### Commit C

`refactor: canonicalize associativeScan reverse via Reverse`

Includes:

- library rewrite in `lax-associative-scan.ts`
- assoc-scan primitive bound in forward form

### Commit D

`feat: add backend reverse accelerators`

Includes:

- optional backend hooks
- WebGPU fast path
- WASM fast path if worthwhile

### Commit E

`refactor: remove assoc_scan executor reverse branches`

Includes:

- deletion of reverse loops from `scan-executor.ts`
- plan simplification in `scan-plan.ts`

### Commit F

`test: add reverse primitive and polymorphic assoc_scan coverage`

### Commit G

`bench: add reverse associative-scan benchmarks`

---

## Acceptance Criteria

- [ ] `Primitive.Reverse` exists and is exported
- [ ] eager `reverse` materializes a correct concrete result without depending on symbolic
      `ShapeTracker` behavior
- [ ] `jit(reverse(x))` works on concrete shapes
- [ ] `jit(reverse(x))` works under `dynamic_axes`
- [ ] `jvp(reverse)`, transpose, and `vmap(reverse)` are implemented
- [ ] `associativeScan(reverse=true)` is canonicalized into forward scan plus `Reverse`
- [ ] `executeAssocScanBlockMap()` contains no reverse-specific algorithmic branches
- [ ] WebGPU block-map assoc-scan plan no longer needs reverse-specific execution behavior
- [ ] polymorphic reverse assoc-scan regression passes
- [ ] generic reverse execution remains correct even when a backend-specific fast path is absent
- [ ] reverse benchmarks are recorded

---

## Explicit Non-Goal

This plan does **not** attempt to make `ShapeTracker.flip()` polymorphic by introducing symbolic
offsets or symbolic strides into `View`.

That remains a separate research / architecture effort.

---

## Recommendation

Proceed with the `Reverse` primitive design, not a second attempt at the original flip-based Plan
1b.

It is the smallest change that is:

- implementable
- correct under `dynamic_axes`
- architecturally honest in the IR
- aligned with the existing JIT step model used for `scan`, `assoc_scan`, `dus`, `scatter_add`, and
  `block_map`
- able to remove the remaining reverse-specific assoc-scan executor logic without dragging the shape
  system into a much larger redesign
