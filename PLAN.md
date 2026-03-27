# Matrix Exponential (`expm`) Plan

## Situation

We have two related but different needs:

1. A public matrix exponential in this repo with JAX/SciPy-style semantics.
2. A cheaper Taylor-based path that unblocks `diffSVMC`'s Yasso work without turning that
   approximation into part of the parity API.

The important current-state facts are:

- Public dense solve is already available, so the Padé `expm` path is not blocked on new solver
  machinery.
- Public `lax.foriLoop` is already available, so the downstream Taylor prototype can start
  immediately on top of existing public primitives.
- `foriLoop` already has JVP, transpose, and `vmap` support for the common case, but it still has
  real AD limits: `grad(jit(f))` is not supported for `foriLoop`, and reverse-mode with symbolic
  bounds is narrower than full JAX parity.
- Public `lax.whileLoop` is not the first gate. It is a possible follow-up only if real downstream
  measurements show that `foriLoop` is too awkward or too slow.
- `foriLoop` already gets us part of the way toward JAX-style dynamic control flow, but not full
  parity across eager mode and all optimized paths, so the plan should stay evidence-driven rather
  than assume new control-flow surface up front.

That means the immediate job is not to invent a merged API or add speculative primitives. It is to
keep the public `expm` surface clean, use the machinery we already expose, and only widen the
control-flow surface if the downstream prototype proves that the current path is insufficient.

## Goals

We need two things, and they should not be conflated into one public API:

1. A public `scipyLinalg.expm(a)` implementation that follows JAX/SciPy semantics and uses the same
   core algorithmic shape as JAX.
2. A cheaper Taylor-based scaling-and-squaring implementation for `diffSVMC`'s Yasso path, without
   polluting the parity API or forcing downstream users into masked-control-flow workarounds.

The earlier draft mixed these together behind a `{ method: "pade" | "taylor" }` option. That would
create avoidable API debt and incentivize performance shortcuts. This revised plan keeps the public
parity surface clean and treats the Taylor path as a separate concern.

---

## Design Decisions

### 1. Public API: keep `scipyLinalg.expm` parity-only

- Public API target: `scipyLinalg.expm(a)`.
- This API should mean "matrix exponential with JAX/SciPy-compatible semantics", not "pick one of
  our internal implementation modes".
- We should **not** add a public `{ method: "taylor" }` option to `scipyLinalg.expm`.
- If we later decide to ship the Taylor variant from this repo, it should live behind a separate
  name and namespace, for example an experimental or clearly domain-specific helper, not as an
  option on the parity API.

### 2. Taylor path: downstream-owned first

- The cheaper Taylor implementation is primarily needed by `diffSVMC` for bug-for-bug and
  performance-oriented compatibility with the original Fortran/Yasso path.
- The quickest downstream enabler is the public machinery that already exists today, especially
  `lax.foriLoop`.
- We should use that first, measure it on real downstream shapes, and only add more control-flow
  surface if the existing public path proves insufficient.

### 3. No masked-control-flow shortcuts

The implementation plan must explicitly avoid these shortcuts:

- No "compute several Padé candidates and select one with `where`" design.
- No fixed `MAX_SQUARINGS` loop with masked `matmul` in the body.
- No expm-local one-off dense solver hidden inside the implementation.

All three would add tech debt or bake permanent performance regressions into a hot linear-algebra
path.

---

## API Contract

### `scipyLinalg.expm(a)`

First implementation target:

- Input shape: square matrices and batched square matrices, `[..., n, n]`.
- First-release dtype scope: real floating dtypes only.
- Backend behavior should respect existing backend constraints rather than invent new implicit
  fallbacks.
  - In particular, WebGPU still has no `f64`, so `f64` behavior must remain aligned with existing
    backend policy.
- Autodiff is part of the contract, not a later bonus.
  - Forward-mode and reverse-mode behavior must be validated before considering the API complete.

Open item to resolve before implementation starts:

- Whether integer inputs are rejected or explicitly promoted before tracing. This should be decided
  deliberately up front, not left to accidental behavior.

### Taylor helper API

Default plan:

- Do **not** add a Taylor variant to `scipyLinalg.expm`.
- Prefer enabling downstream implementation through public control-flow primitives.

If we later decide the Taylor path belongs in this repo, the API should be separate and explicit,
for example:

- experimental helper
- domain-specific helper
- non-parity helper with naming that does not imply SciPy/JAX equivalence

The important constraint is that `scipyLinalg.expm` must remain the parity surface.

---

## Required Prerequisites

### Phase 0: Use existing public `lax.foriLoop` as the first downstream enabler

This is the first downstream-enablement step because it is already public, already traced, and
already tested in this repo. It requires no new core API surface to unblock the first `diffSVMC`
port.

### Why

Downstream already has enough public math primitives to write most of the Taylor path itself:

- identity construction
- scalar math
- reductions
- `matmul`
- public `foriLoop`

The immediate question is not whether `whileLoop` would be nicer. It would. The question is whether
the existing public `foriLoop` path is already good enough for the real `diffSVMC` shapes and
control-flow needs.

That should be answered empirically first. If `foriLoop` can express the Taylor-series accumulation
and the required scaling/squaring logic without duplicating matrix work or introducing measurable
downstream cost, then it is the right first enabler.

### Scope

Use the existing public `lax.foriLoop` surface as the first supported downstream implementation
target.

Concretely, Phase 0 should:

- prototype the Taylor path in `diffSVMC` using only existing public primitives
- use `lax.foriLoop` for the fixed 10-term Taylor accumulation immediately
- determine whether the scaling-exponent and squaring steps can also be expressed cleanly enough
  with `foriLoop` on real downstream workloads
- stay within the current `foriLoop` AD envelope instead of assuming full JAX parity for transformed
  loop programs
- reject any `foriLoop` formulation that relies on masked matrix-multiply work or other obviously
  permanent performance debt

This phase is intentionally benchmark-driven. The goal is to find out whether the current public
surface is already sufficient, not to assume that a new core primitive is required before trying the
simplest viable path.

### Optional follow-up: public `lax.whileLoop` only if Phase 0 proves insufficient

If the `foriLoop`-first downstream prototype shows that the remaining data-dependent control flow is
awkward, slow, or forces downstream into structural workarounds we do not want to bless, then we
should add a later phase for public `lax.whileLoop`.

That follow-up should be justified by concrete downstream evidence rather than speculation. It
remains a consumer-enablement track, not a gate in front of the public Padé `expm` implementation.

### Validation

- downstream smoke test showing the Taylor path can be written externally with existing public
  primitives
- benchmark on real `diffSVMC` shapes to decide whether `foriLoop` is sufficient as the first
  supported path
- explicit check that the downstream formulation does not depend on unsupported `foriLoop` AD cases
  such as `grad(jit(f))` or general symbolic-bound reverse-mode
- explicit review of whether any remaining control-flow workarounds duplicate matrix work or
  otherwise lock in debt

### Phase 1: Use the existing dense-solve machinery deliberately

This is not a missing prerequisite anymore. Public dense solve already exists as
`np.linalg.solve(a, b)`, including batched `[..., n, n]` support.

### Requirement

The `expm` implementation should reuse that existing solve path rather than introducing new
expm-local linear-solve logic.

### Optional follow-up

If we want a `scipyLinalg.solve(a, b)` alias for API completeness, that can be added as a thin
wrapper. It should not block `expm`.

### Non-goal

- Do not hide Gaussian elimination, matrix inversion, or an ad hoc solve path inside `expm`.

### Validation

- confirm `expm` uses the existing solve path rather than duplicating solver logic
- rely on the existing solve coverage for numerical correctness and AD

---

## Public `expm`: JAX-Parity Path

### Phase 2: Implement `scipyLinalg.expm` using JAX's algorithmic structure

### Target algorithm

Match the same overall algorithmic family as JAX:

- Higham-style scaling and squaring
- Padé approximants
- one selected degree
- one solve
- exact number of squarings

### Performance constraints

The implementation should preserve the cost model of the real algorithm:

- choose one Padé degree, do not evaluate several full candidates
- compute the squaring count exactly, do not approximate it with a masked fixed loop
- avoid extra matrix powers or matmuls that are not part of the chosen degree's evaluation

### Control flow requirement

Efficient Padé degree selection is a real design point.

If the norm is traced, then plain JavaScript branching is not available. The implementation phase
should therefore choose a scalar-only control-flow strategy before code is written.

Acceptable options include:

1. Introduce internal scalar control flow to choose the Padé degree.
2. Structure the evaluator so traced selection only affects scalar coefficient choice, not
   duplicated matrix-polynomial work.
3. Use bounded control flow for repeated squaring where the bound is part of the algorithm, not an
   arbitrary masked `MAX_SQUARINGS` escape hatch.

What we should **not** do is compute several matrix approximants and select between them with
`where`. Scalar-only selection is acceptable; duplicated matrix-polynomial work is not.

### Autodiff requirement

AD is release-blocking for the public API.

Before shipping `scipyLinalg.expm`, we must know whether:

- the dense-solve-based implementation differentiates correctly through existing primitives
- or `expm` needs a custom JVP/VJP rule for acceptable parity and numerical stability

This decision belongs in the implementation plan, not after the code lands.

### Validation

- parity tests against JAX/SciPy on representative real matrices
- batched tests on `[..., n, n]`
- forward-mode and reverse-mode checks against reference finite differences or JAX
- backend coverage consistent with existing linalg policy

### Implementation note

The public Padé `expm` path is now considered implementable without waiting for public
`lax.whileLoop`, because the solver prerequisite is already satisfied and the remaining control-flow
work is internal to `expm`.

---

## Taylor Path For `diffSVMC`

### Phase 3: Implement the cheaper Taylor algorithm in downstream

This path should initially live in `diffSVMC`, not in the public `scipyLinalg` namespace.

### Algorithm

Replicate the original Yasso/Fortran behavior:

1. Compute the Frobenius norm.
2. Determine the scaling exponent with real data-dependent control flow.
3. Scale the matrix.
4. Accumulate the fixed 10-term Taylor series.
5. Square exactly the required number of times.

The first downstream implementation target is the public `lax.foriLoop` surface that already exists.
That path should be attempted and benchmarked before we decide whether a public `lax.whileLoop` API
is actually required.

### Why downstream-first is the right default

- It keeps `scipyLinalg.expm` semantically clean.
- It avoids adding a downstream-specific approximation to the core public API before we know whether
  other consumers need it.
- It lets `diffSVMC` control its own compatibility/performance tradeoffs while using public
  primitives rather than internal hooks.

### Performance expectations

- The Taylor path is intended for small Yasso-sized matrices where avoiding the dense solve may be
  materially cheaper.
- It should prefer the existing `foriLoop` path first and only escalate to new control-flow surface
  if benchmarks justify it.
- It should be benchmarked on the actual downstream shapes rather than justified abstractly.

If `foriLoop`-based control flow turns out to be sufficient for the real downstream shapes without
introducing measurable overhead or awkward API usage, we should keep the plan there and avoid adding
`whileLoop` prematurely. If it does not, we can spin up a follow-up phase for public `lax.whileLoop`
with the downstream evidence already in hand.

### Optional future promotion

If multiple downstream consumers need the same helper, we can revisit whether to promote it into
this repo under a separate, non-parity API. That decision should be based on repeated demand, not
convenience during the first port.

---

## Benchmarks And Performance Gates

### Phase 4: Performance validation before feature completion

We should not accept an implementation that is only numerically correct but structurally expensive.

### Benchmarks to add

- small matrices representative of Yasso / `diffSVMC`
- small and medium dense matrices for Padé `expm`
- batched `[..., n, n]` cases
- eager vs `jit()`
- WASM vs WebGPU where applicable

### Things to watch for

- duplicated matrix-polynomial evaluation from masked degree selection
- excessive matmul count in the squaring phase
- accidental host-side loops over batch dimensions
- solver overhead dominating small-matrix cases

The goal is not just "works"; it is "works without baking in obvious permanent inefficiencies".

---

## Migration / Consumer Guidance

Once the prerequisites land, the intended split is:

- core repo owns `scipyLinalg.expm(a)` with JAX-parity semantics
- core repo keeps `lax.foriLoop` as the first downstream control-flow enabler and only adds
  `lax.whileLoop` later if measured need justifies it
- downstream `diffSVMC` owns the Taylor/Yasso compatibility implementation unless later demand
  justifies promotion

That split keeps the API honest and avoids teaching downstream users to rely on a parity API that
secretly contains domain-specific modes.

---

## Non-Blocking Follow-Up: Structural Update API (Priority 1 Complete)

The `expm` work does not depend on this, but the downstream ports have exposed a real adjacent
usability gap: JAX code that uses `.at[...]` for multi-axis structural updates does not always have
an equally clean public replacement here.

### What shipped in v0.11.0

**ND `dynamicUpdateSlice` with static offsets** — the Priority 1 item is complete.

`lax.dynamicUpdateSlice` now accepts two forms:

```typescript
// Single-axis form (unchanged):
lax.dynamicUpdateSlice(dst, src, offset, axis?): Array

// ND form (new):
lax.dynamicUpdateSlice(dst, src, startIndices: number[]): Array
```

The implementation adds a new `DynamicUpdateSliceGeneral` primitive (the original single-axis
`DynamicUpdateSlice` is preserved unchanged). TypeScript overloads dispatch at the public API
boundary.

**What it provides:**

- Multi-axis block updates with static JS-number per-axis offsets
- Full AD: JVP (linearTangentsJvp), reverse-mode (transpose via shrink), vmap
- JIT: single ND-aware DUS step with precomputed radix divisors; axes where `src` spans the full
  `dst` extent are optimized away; single-axis fast path activates automatically
- WebGPU command tape support with ND stride decomposition
- Eager mode: stride-based patching

**Design deviation from plan:** The plan proposed renaming the single-axis API to
`dynamicUpdateSliceInDim`. This rename was not taken. Instead, TypeScript overloads distinguish the
two forms by argument type (`number` vs `number[]`). This avoids a breaking change for all existing
callers while still providing the ND API.

### Migration bundle for downstream consumers (v0.11.0)

**Deterministic mapping — no breaking changes.** Existing single-axis calls continue to work
unchanged.

| Pattern                                                          | Action                                                                |
| ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| `lax.dynamicUpdateSlice(dst, src, offset, axis)`                 | No change needed. Works as before.                                    |
| Two chained single-axis `dynamicUpdateSlice` for 2D block update | Replace with `lax.dynamicUpdateSlice(dst, src, [rowStart, colStart])` |
| One-hot matmul/add to simulate column/row writes                 | Replace with `lax.dynamicUpdateSlice` (see patterns doc)              |
| Manual piecemeal construction with repeated single-axis patching | Evaluate whether a single ND call is cleaner                          |

**Workaround signatures (debt collector):**

- If downstream has a helper like `updateBlock2D(dst, src, row, col)` that chains two single-axis
  `dynamicUpdateSlice` calls, delete it and use the ND form directly.
- If downstream uses `np.matmul(newCol, oneHotRow)` + `np.add(A, patch)` to simulate a column write,
  replace with `lax.dynamicUpdateSlice(A, np.reshape(newCol, [m, 1]), [0, j])`.
- If downstream builds a 2D patch via `np.zeros` + repeated single-element writes, consider
  `np.array([[...]])` + single ND `dynamicUpdateSlice`.

**Full migration guidance and usage patterns:** See `docs/array-manipulation-patterns.md`.

### What the downstream evidence actually shows

Not all piecemeal construction is equally problematic:

- **Small fixed matrix assembly** from known scalar entries via `np.array([...])` or
  `np.stack([...])` is fine and does not need new primitives.
- **One-hot matrix algebra** (matmul/add with one-hot vectors to fake indexing) is the real
  anti-pattern. Downstream code uses this because the current update API is single-axis only.
- **Traced write offsets** are not yet a verified downstream need. Both `diffSVMC` and `dlm-js` use
  plain JS loop indices, not traced values from `lax.foriLoop`.

### Design constraints

- Public update helpers must continue to return new arrays (non-consuming semantics).
- Internal lowering may use mutate/recycle paths under `jit()`.
- No in-place mutation API or ownership-changing builder.

### Priority 1: ND `dynamicUpdateSlice` with static offsets ✅ Complete

Shipped in v0.11.0. See migration bundle above.

### Priority 2: Convenience helpers for common patterns

After the ND primitive ships, evaluate whether common downstream patterns justify thin helpers:

```typescript
// Set a single element at [i, j]:
lax.dynamicUpdateSlice(A, np.array([[value]]), [i, j]);

// This works but is verbose. A helper like:
lax.setElement(A, [i, j], value);
// could reduce boilerplate for scalar writes without new primitives.
```

Keep these as pure frontend sugar over `dynamicUpdateSlice`, not new primitives. Judge by clarity
and downstream adoption, not by literal JAX syntax parity.

### Future: Traced write offsets (only if downstream proves the need)

If a downstream use case demonstrates that traced loop indices need to flow into
`dynamicUpdateSlice`, the primitive can be further generalized to accept traced scalar start indices
as inputs (mirroring `DynamicSlice`). That change would require:

- Moving start indices from params to inputs
- Custom JVP rule (primal starts for tangent output)
- Transpose via `dynamicSlice` with traced starts
- Runtime-resolved DUS JIT step (similar to scan uniform offsets)
- Command tape and vmap updates for traced start inputs

This is well-understood architecturally but should only be built when downstream code actually needs
it. The current verified downstream patterns all use plain JS numbers.

### Future: Broader scatter coverage

- `scatterAdd` exists today.
- `scatterSet` and other scatter-style writes may be worth adding later.
- Evaluate demand after ND `dynamicUpdateSlice` ships and downstream code has been refactored.

---

## Summary Of Work Items

1. Prototype the Taylor path in `diffSVMC` using existing public primitives, with `lax.foriLoop` as
   the first implementation target.
2. Benchmark that downstream prototype on real shapes and decide whether public `lax.whileLoop` is
   actually needed.
3. Reuse the existing `np.linalg.solve` machinery for Padé `expm` rather than adding solver debt.
4. Implement public `scipyLinalg.expm(a)` as the JAX-parity path only.
5. Validate AD behavior as part of the public API contract.
6. ~~Add ND `dynamicUpdateSlice` with static JS-number offsets.~~ ✅ Done (v0.11.0). Shipped as
   TypeScript overload (no rename). See migration bundle in PLAN.md and usage patterns in
   `docs/array-manipulation-patterns.md`.
7. Evaluate convenience helpers and broader scatter coverage after downstream adoption.
8. Extend `dynamicUpdateSlice` to accept traced offsets only if downstream code demonstrates a real
   need for it.
9. Only consider promoting the Taylor path into this repo later if repeated downstream demand
   justifies a separate non-parity API.
