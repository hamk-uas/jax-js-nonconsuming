# Ownership Restructuring Plan

## Objective

Restructure tracing and autodiff ownership so that the runtime has one coherent model for lexical
`using` / `.dispose()`, retained internal handles, captured constants, and cache-owned artifacts.
The target state is an architecture with unambiguous ownership rules, no tracing-only disposal
escape hatches, and identical user-visible ownership semantics in eager and `jit()` execution. The
end state is **not** "everything is lexical." The end state is that every value is either: (a) owned
by its current lexical scope, or (b) explicitly retained by one named subsystem (builder, artifact,
executor, cache) with one clear release path.

## Ground Truth

_(Updated after Step 3d: phantom-ref machinery deleted, explicit lexical balancing active.)_

The remaining ownership friction is concentrated in the tracing pipeline:

- `src/frontend/array.ts` no longer carries a disposal suppression guard. `[Symbol.dispose]()`
  always runs normally. Two WeakSets provide creation-ref tracking: `_arraysCreatedInMakeJaxprBody`
  (marks arrays born during any `makeJaxpr` body) and `_liftedTangentZeros` (marks zero tangents
  from `JVPTrace.lift()`, whose creation ref is owned by `liftedTangents` cleanup).
- `src/frontend/linearize.ts` has an active `ResidualCollector` with `dispose(protectedVals)`. It
  cleans up dead PE temporaries that are not in the protection set. Forward jaxpr consts at rc≤1 are
  protected (their only remaining ref is the CJ's `.ref`); consts at rc>1 still carry a ref from
  `instantiateConst` that the collector balances.
- `src/frontend/jaxpr.ts` captures constants through `getOrMakeConstTracer` (`.ref`s the value). For
  raw values (`pureArray` results), the creation ref is immediately balanced after taking `.ref`.
  For non-raw arrays created during `makeJaxpr` body execution (tracked by
  `_arraysCreatedInMakeJaxprBody`), a per-builder `constsNeedingCreationRefBalance` Set tracks which
  consts need their creation ref balanced. The balancing happens in `makeJaxpr` after
  `builder.build()`, using a `refCount > 1` pre-build filter to distinguish stranded creation refs
  from explicitly user-balanced ones (e.g., `jacfwd`'s `eyeMatrix.dispose()`).
- `src/frontend/artifacts.ts` wrappers reach const lifecycle indirectly through
  `ClosedJaxpr.dispose()`. They do NOT reference `_peArrayCreationTracker` or creation-ref tracking
  directly.

The system works because:

1. `getOrMakeConstTracer`'s `.ref` keeps consts alive through `ClosedJaxpr` ownership.
2. Creation refs for body-created arrays are balanced by `makeJaxpr`'s post-build loop.
3. Creation refs for raw values are balanced immediately by `getOrMakeConstTracer`.
4. `ResidualCollector.dispose()` cleans up dead PE temporaries (reactivated in Step 3b).
5. `_liftedTangentZeros` prevents double-balancing of JVP lift zeros (their creation ref is owned by
   `liftedTangents` cleanup in `jvpFlat`).

## Target Invariants

1. Every concrete array starts with one lexical owner.
2. Any subsystem that needs a value beyond lexical scope takes an explicit independent retained
   handle.
3. Any value that outlives lexical scope must name its retained owner explicitly; there is no
   implicit global tracing owner or arena owner for escaping values.
4. Every retained handle has exactly one release path, including error paths.
5. Cache-owned objects remain cache-owned; callers do not opportunistically dispose them.
6. `jit()` is a performance optimization only. It must not rescue code that is ownership-incorrect
   in eager mode.
7. `using` and explicit `.dispose()` must have the same semantic meaning in eager and traced code.

## Canonical Ownership Matrix

To eliminate architectural ambiguity, every retained handle or array must map to exactly one of
these rows. There are no "multiple choice" fallback options.

| Value Class                   | Lexical Owner                        | Retained Owner                                                                               | Release Site                                                                                                                                         | Caller-Disposal Rules                                            |
| ----------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Normal Array**              | Function scope                       | None                                                                                         | `using` or explicit `.dispose()`                                                                                                                     | Fully caller-disposed                                            |
| **Captured Const**            | Function scope                       | The capturing object (e.g., `ClosedJaxpr`)                                                   | The capturing object's `.dispose()`                                                                                                                  | Caller safely disposes original; retained handle stays alive     |
| **PE Intermediate (Dead)**    | The PE lexical scope that created it | None (never promoted)                                                                        | Collector sweep before the PE function fully returns                                                                                                 | No caller references survive                                     |
| **PE Intermediate (Handoff)** | Artifact builder                     | Artifact (e.g., `PrimalArtifactImpl`)                                                        | Artifact's `.dispose()` (usually executed during backward pass)                                                                                      | Caller securely drops its handle; Artifact holds independent ref |
| **Inlined Captured Const**    | Function scope                       | The capturing object remains the sole retained owner even if the value is inlined as a `Lit` | The same capturing object's final cleanup path releases the last retained handle after `_inlineLiterals` has removed the builder-held representation | Callers never own the inlining cleanup path                      |
| **Cache Entry**               | The Cache                            | The Cache                                                                                    | `clearCaches()` or cache eviction                                                                                                                    | Callers MUST NOT dispose                                         |
| **Exec-body value**           | Loop/scan body                       | The executor structure                                                                       | Executor / context cleanup                                                                                                                           | Follows loop lifecycle                                           |

## Workaround Signatures To Hunt And Delete

These patterns are the architectural debt collectors for this refactor:

- ~~**`[Symbol.dispose]` guard** (`array.ts`):
  `if (inMakeJaxprBody() && anonymousConstArrays.has(this)) return;`~~ _(Deleted in Step 3d:
  explicit lexical balancing replaces disposal suppression.)_
- ~~**Dead `ResidualCollector.dispose()`** (`linearize.ts`): `return;` as first statement.~~
  _(Resolved in Step 3b: collector reactivated.)_
- **`_peArrayCreationTracker` guard in `JVPTracer`** (`jvp.ts:123`):
  `!_peArrayCreationTracker && ...`
- **`_peArrayCreationTracker` guard in Sort JVP** (`jvp.ts:419`):
  `if (!_peArrayCreationTracker) idx.dispose();`
- ~~**`_anonymousExtraDispose` phantom-ref balancing** (`jaxpr.ts`): fires or defers phantom
  creation ref disposal based on builder ref count and `refCount === 1`.~~ _(Deleted in Step 3d.)_
- ~~**`_deferredConstCreationDisposes` queue** (`jaxpr.ts`): processes phantom disposals after
  outermost `makeJaxpr` body completes.~~ _(Deleted in Step 3d.)_
- ~~**`#inlinedAnonymousConsts` on `ClosedJaxpr`** (`jaxpr.ts`): tracks literals whose phantom ref
  needs balancing after `_inlineLiterals` removes them from the consts list.~~ _(Deleted in Step
  3d.)_
- ~~**`evalJaxpr` Lit-array anonymous marking** (`jaxpr.ts`): materialized Lit arrays are added to
  `anonymousConstArrays`.~~ _(Deleted in Step 3d: `litArrays` cleanup is sufficient.)_

## Feasibility Assessment

### Risk Summary

| Area                                     | Risk             | Why                                                                                                                                                                                          |
| ---------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reactivate `ResidualCollector.dispose()` | **DONE** (3b)    | Reactivated with rc≤1 protection set. 42/42 characterization tests pass.                                                                                                                     |
| Delete phantom-ref machinery             | **DONE** (3d)    | Three-layer state machine replaced by explicit creation-ref balancing via `constsNeedingCreationRefBalance` + `_arraysCreatedInMakeJaxprBody` + `_liftedTangentZeros`. All 2,064 tests pass. |
| Evaluation-time Lit ownership            | **DONE** (3c/3d) | `evalJaxpr` Lit no longer marked anonymous — `litArrays` cleanup is sufficient. `evalJaxprTransposed` zeros/Lit tracked in `internalArrays` (independent lifecycle).                         |
| WS2–WS3 entanglement                     | **DONE** (3d)    | `anonymousConstArrays` deleted entirely. PE suppression guard deleted. Const lifecycle is now independent of PE tracking.                                                                    |
| Wrapper/cache boundaries                 | **LOW**          | Artifact wrappers don't directly reference suppression machinery. Mostly verification.                                                                                                       |
| Removing `[Symbol.dispose]` guard        | **DONE** (3d)    | Guard deleted. Creation refs balanced by makeJaxpr post-build loop.                                                                                                                          |

### Key Finding: PE Intermediates (Fixed in Step 3b)

`ResidualCollector.dispose()` was dead code (unconditional `return;`). Reactivated in Step 3b.
Non-anonymous PE intermediates are now cleaned up by the collector. Anonymous consts survive through
`getOrMakeConstTracer`’s `.ref` → `ClosedJaxpr` ownership. The rc≤1 protection guard ensures
ClosedJaxpr-owned consts are not prematurely disposed by the collector.

**Impact:** Silent memory leaks proportional to the number of JVP-rule evaluations during PE. Not
crashes, but unbounded memory growth in long-running `grad`/`vjp` workloads.

## Additional Finding: Anonymous-Const Tracking Also Serves Evaluation (RESOLVED)

_(Step 3d deleted `anonymousConstArrays` entirely. Evaluation-time Lit ownership is handled by
`litArrays` cleanup in `evalJaxpr` and `internalArrays` cleanup in `evalJaxprTransposed`. No unified
tracing+evaluation tracking mechanism is needed.)_

`anonymousConstArrays` was populated by evaluation paths that materialize `Lit` atoms into concrete
arrays. Step 3c removed the `evalJaxprTransposed` marking (tracked in `internalArrays` instead).
Step 3d confirmed that `evalJaxpr`'s `litArrays` cleanup is sufficient for Lit-created arrays
without anonymous marking.

## Workstreams (Revised)

### Phase 1: Document and prove the ownership contract (DONE)

- Maintainer-facing invariants written into `.github/copilot-instructions.md`. ✓
- This plan documents actual code state from audit. ✓
- `test/tracing.test.ts` and `test/refcount.test.ts` exist. ✓

### Phase 2: Build the regression test harness (DONE)

Before touching ownership machinery, write focused tests that pin the current behavior.
**Implemented in `test/ownership-characterization.test.ts` — 42 tests, all passing.**

#### Key Findings From Characterization

**Leak Boundary (refined):** The simple bare-path tests show `grad()` and `valueAndGrad()` leaking,
while simple `vjp()`, `jvp()`, and `linearize()` cases can run cleanly when properly disposed. The
suite also retains a smaller set of explicitly named `vjp` edge-case leak tests. Those are no longer
treated as evidence that bare `vjp()` generally leaks; they are scoped characterizations of specific
workaround-era paths that still need explanation during Phase 3.

**Bare grad(scan(f)) crashes:** `grad(f)` where `f` uses `lax.scan` with an anonymous const init
(`np.array(0)`) throws `UseAfterFreeError` — the anonymous const gets disposed during the backward
pass. `jit(grad(f))` works fine because JIT manages lifetimes.

**Known-leak taxonomy:** The harness now distinguishes three buckets:

- **Confirmed known leaks:** bare `grad`, `valueAndGrad`, and related grad-shaped variants.
- **Scoped edge-case leaks:** specifically named `vjp` paths that leak today but do not justify the
  broader claim that bare `vjp()` is generally leaky.
- **Provisional characterizations:** behaviors such as eager `scan` + inner `using` that are pinned
  for continuity without yet being elevated to a confirmed standalone leak class.

**Zero-leak tests (27 tests):** `jit(grad(f))`, `vmap(grad(f))`, `grad(grad(f))`, `jit(grad(scan))`,
`vjp`, `jvp`, `linearize`, `ClosedJaxpr.dispose()`, nested makeJaxpr, inlined literals, clearCaches,
jit function dispose — all pass with zero leaks.

#### Test Categories

| Category                           | Tests | Key Takeaway                                                               |
| ---------------------------------- | ----- | -------------------------------------------------------------------------- |
| 1. Transform composition leaks     | 10    | confirmed bare grad/valueAndGrad leak; simple vjp/jvp/linearize don't      |
| 2. `using` inside transform bodies | 4     | guard suppresses disposal; grad leaks and vjp has a named edge-case leak   |
| 3. ClosedJaxpr disposal            | 3     | builder refs correctly released; nested scopes safe                        |
| 4. transposeJaxprCache             | 2     | cache-owned contract verified                                              |
| 5. Nested-builder const sharing    | 3     | grad(grad(f)), triple nesting, shared consts safe                          |
| 6. Inlined literal ownership       | 4     | scalar consts cleaned up; bare grad still leaks                            |
| 7. Eval-time Lit materialization   | 4     | jit(grad) ok; bare grad leaks and a specific vjp Lit path leaks            |
| 8. ResidualCollector code paths    | 4     | grad leaks; one nonlinear vjp edge case also leaks; hasAux paths exercised |
| 9. Scan transform compositions     | 4     | bare grad(scan) crashes; jit(grad(scan)) ok                                |
| 10. Ownership edge cases           | 4     | clearCaches/jit.dispose cleanup verified                                   |

#### checkLeaks Scoping Pattern for KNOWN LEAK Tests

Manual `checkLeaks.stop()/start()` around leaky code. Critical constraint: do NOT use `using`
declarations inside the inner scope — `using` disposals happen at block exit (after
`checkLeaks.start()` reopening), causing negative leak counts in the afterEach check. Use explicit
`.dispose()` calls inside the inner scope instead.

### Phase 2.5: Reconcile the characterization baseline before Phase 3 (DONE)

Phase 2 introduced a documentation-level debt: the written summary initially said only bare
`grad()`/`valueAndGrad()` leak, while `test/ownership-characterization.test.ts` still included
multiple `vjp`-related tests marked as `KNOWN LEAK` with over-broad names.

This cleanup is now complete. The characterization file and this plan use the same taxonomy:
confirmed grad/valueAndGrad leaks, explicitly named `vjp` edge-case leaks, and one provisional eager
scan+using characterization.

Completed cleanup:

- Reclassified the checked-in manual-`checkLeaks` cases into confirmed leaks, scoped edge-case
  leaks, and provisional characterizations.
- Renamed over-broad `vjp` leak titles in the characterization file so they no longer imply that all
  bare `vjp()` paths leak.
- Updated this Phase 2 narrative so it matches the checked-in test taxonomy.
- Removed the contradictory Phase 2 wording that would have made Phase 3 start from an unstable
  baseline.

### Phase 3: Make PE retention and captured-const ownership explicit (MERGED WS2+WS3)

**STATUS: Steps 3a--3c DONE. Step 3c.5 DONE. Phase 3c.75 feasibility analysis DONE (Alternative A
selected: explicit lexical balancing). Step 3d ready to implement. Step 3e not started.**

Steps 3a–3b fixed 12 of 13 original KNOWN LEAK characterization tests. The remaining KNOWN LEAK
(`bare grad(scan(f))`) is a separate UseAfterFreeError from anonymous-const disposal during the
backward pass of scan — outside the scope of PE/collector ownership.

**Step 3a–3b changes landed:**

1. **Reactivated `ResidualCollector.dispose()`** (`linearize.ts`): removed the unconditional
   `return;` at the top of the method. The collector now disposes dead PE temporaries that are not
   in the protection set.

2. **Tightened protection set** (`linearize.ts` + `artifacts.ts`): forward jaxpr consts at rc≤1 are
   protected (their only remaining ref is the CJ's `.ref`). Consts at rc>1 still carry a ref from
   `instantiateConst` that the collector must clean up; protecting them would leak that ref.

3. **Marked PE-created arrays as anonymous** (`array.ts`): arrays created during the PE scope
   (`_peArrayCreationTracker` active) were also added to `anonymousConstArrays`. _(This temporary
   bridge was removed by Phase 3b.5.)_ Previously, PE intermediates created during
   `buildForwardJaxpr` were NOT anonymous (because `inMakeJaxprBody()` is false in that context), so
   when they became forward jaxpr consts via `getOrMakeConstTracer`, `ClosedJaxpr.dispose()` could
   not fire `_anonymousExtraDispose` to balance the phantom creation ref. This caused 1 leaked array
   for any function with non-trivial PE intermediates (e.g., `vjp(f)` where f has mul/reuse
   patterns).

**Step 3c changes landed:**

Step 3c's deliverable is a **scoping correction**, not a machinery replacement. The adopt model
(first builder adopts creation ref instead of `.ref`) was explored and rejected: inner CJ disposal
during nested tracing frees the adopted array, causing UseAfterFreeError when outer builders try to
capture it. The phantom-ref cleanup machinery is architecturally correct for nested builders and is
retained.

1. **Removed `markAnonymous` parameter from `evalJaxprTransposed`** (`linearize.ts`): zeros and Lit
   arrays created inside `evalJaxprTransposed` are tracked in `internalArrays` (lifecycle managed by
   cleanup at function exit). They do NOT need anonymous marking because their creation ref is
   balanced by the `internalArrays` cleanup, independent of the phantom-ref machinery.
   `transposeJaxpr` and `buildBackwardJaxpr` no longer pass `markAnonymous: true`.

2. **Removed PE const promotion tracking** (`jaxpr.ts`): deleted `_registerPEConstPromotion`,
   `_promotedPEConstArrays`, `_hasConstCreationDebt`. PE-promoted consts don't need special tracking
   because the PETracer disposal cascade consumes their creation ref, and the builder's `.ref` in
   `partialEvalGraphToJaxpr` provides independent ownership.

3. **Removed adopt model artifacts** (`jaxpr.ts`): deleted `_adoptOrRef`, `_claimedConsts`.

4. **Renamed internal state** (`jaxpr.ts`): `_anonymousBuilderRefs` → `_constCreationBuilderRefs`,
   `_deferredAnonymousDisposes` → `_deferredConstCreationDisposes`. Names now reflect their actual
   purpose (tracking builder refs for phantom-creation-ref cleanup) rather than the broader
   "anonymous" label.

5. **Improved `getOrMakeConstTracer`** (`jaxpr.ts`): always calls `(val as Tracer).ref;` then
   `_incrementBuilderRef(val)` for anonymous consts. Non-anonymous consts get `.ref` only. Debug
   logging moved inside the anonymous check (avoids noise for non-anonymous captures).

6. **Separated PE origin from anonymous-const lifecycle** (`array.ts`): introduced
   `_peTrackedArrays` WeakSet + `_wasCreatedInPEScope()` for future use. _(Deleted in Phase 3c.5:
   these were speculative scaffolding with no consumers. PE origin tracking is not needed for the
   explicit lexical balancing model selected in Phase 3c.75.)_

7. **Restored `protectedVals` rc≤1 guard** (`linearize.ts` + `artifacts.ts`): Step 3b had changed
   `vjpFlat` and `aotLinearize` to unconditionally protect all forward jaxpr consts. This was
   incorrect: consts at rc>1 carry a ref from `instantiateConst` that the collector must clean up.
   Restored the `if (c.refCount <= 1)` guard with clarifying comments.

8. **Confirmed JVP lift zeros need anonymous marking** (`jvp.ts`): `fullInternal` calls
   `markAnonymousIfTracing` only when `inMakeJaxprBody()` is true. `JVPTrace.lift` also runs inside
   `buildForwardJaxpr` (PE trace) where `inMakeJaxprBody()` is false. The explicit
   `anonymousConstArrays.add(zero)` covers that case and is load-bearing.

9. **Confirmed `evalJaxpr` Lit arrays need anonymous marking** (`jaxpr.ts`): `array()` calls
   `markAnonymousIfTracing`, but that only fires when `inMakeJaxprBody()` is true. The explicit
   `anonymousConstArrays.add(arr)` in `evalJaxpr` covers all evaluation contexts including PE-scope
   where `inMakeJaxprBody()` may be false.

**Key design finding from 3c:** The anonymous-const categories are:

- **Need anonymous marking (confirmed):** `pureArray`, `markAnonymousIfTracing` (fullInternal), JVP
  lift zeros (`JVPTrace.lift`), `evalJaxpr` Lit materialization — all have unowned phantom creation
  refs that need the phantom-ref cleanup machinery.
- **Don't need anonymous marking (confirmed):** `evalJaxprTransposed` zeros/Lit arrays (tracked in
  `internalArrays`, cleanup balances creation ref independently), PE-promoted consts (PETracer
  cascade consumes creation ref).

**Validation:** 42/42 characterization tests pass. Full suite: 2064 tests pass, 0 failures.

**Immediate cleanup required before Step 3c:** _(Resolved by Phase 3b.5, see status below.)_

The Phase 3b fix intentionally introduced a temporary bridge: arrays created while
`_peArrayCreationTracker` is active were also inserted into `anonymousConstArrays` so that the
existing phantom-ref cleanup machinery could release them if they later became captured consts.
Phase 3b.5 removed this bridge and restored `anonymousConstArrays` to a single meaning.

### Phase 3b.5: Remove the temporary PE-to-anonymous bridge before continuing to Step 3c

- Replace the temporary `anonymousConstArrays.add(this)` hook under `_peArrayCreationTracker` with a
  more explicit ownership handoff for PE-created arrays that later become captured consts.
- Restore `anonymousConstArrays` to a single meaning: values participating in anonymous-const
  lifecycle, not general PE-created arrays.
- Write down the exact transition rule for a PE-created array that is later promoted into a
  `ClosedJaxpr` const. The promotion path must name who takes the retained handle and who balances
  the creation-side obligation; do not leave that encoded implicitly through a shared WeakSet.
- Keep the current bridge only until the explicit promotion path lands and is re-validated by the
  ownership characterization suite and the full test suite.
- As part of this cleanup, remove stale comments/documentation introduced during the transition: any
  references to the old low-rc const heuristic, the old dead `ResidualCollector.dispose()` status,
  or outdated leak-count summaries must be brought back into sync before advancing.

**STATUS: DONE**

- PE origin is now tracked separately from anonymous-const lifecycle.
- Const promotion from partial evaluation explicitly registers creation-side cleanup before the
  ClosedJaxpr takes its retained `.ref`.
- `anonymousConstArrays` is again reserved for true anonymous-const lifecycle only.
- Transition comments/docs were brought back into sync.

**Step 3a: Triage PE intermediates.**

- Classify every array that enters `_peArrayCreationTracker` → `peCreatedArrays` →
  `ResidualCollector.peIntermediates`:
  - (i) Anonymous consts already owned explicitly (via `getOrMakeConstTracer` `.ref`)
  - (ii) Non-anonymous PE intermediates with no current owner (the leaking ones)
  - (iii) Values that flow into outputs or aux (already protected by `protectedVals`)
- For category (ii): Enforce strictly from the Ownership Matrix. Values needed only for handoff to
  the backward pass are strictly **Artifact-owned** (e.g., `PrimalArtifactImpl`). Pure dead
  intermediates are **Collector-owned** and must be disposed before the forward pass returns. Do not
  treat `ClosedJaxpr` as a dumping ground for intermediates.

**Step 3b: Reactivate `ResidualCollector.dispose()` (or replace it).**

- Remove the `return;` at `linearize.ts:287`.
- Default path: reactivate the existing `ResidualCollector.dispose()` implementation first, because
  its protection-set logic already handles wrapper aliasing and shared concrete arrays.
- Before reactivation, tighten the protection rule for captured consts: the current
  `forwardJaxpr.consts` low-rc heuristic (`refCount <= 1`) is insufficient on its own because it
  ignores anonymous-builder ownership tracked outside raw refcount. Do not land a version that
  treats builder-owned anonymous consts as dead collector targets.
- Treat the collector as a scoped cleanup mechanism for **Collector-owned dead temporaries only**.
  It may behave like a small arena for values proven not to escape PE, but it must not become a
  substitute for explicit retained ownership at builder/artifact/cache boundaries.
- Replacement is allowed only if the reactivated collector is proven unsound by focused tests or if
  the replacement demonstrably preserves the same protection-set guarantees with less machinery.
- If replacing `ResidualCollector` entirely, the replacement list must adhere strictly to the
  **Collector-owned** invariant: it must _only_ contain unpromoted dead PE intermediates and must be
  cleanly disposed before the PE pass exits.
- Run the Phase 2 test harness after each change to verify no new use-after-free bugs emerge.

**Step 3c: Design and land the anonymous-const replacement model before deletion.** (DONE)

- **Design finding:** The adopt model (first builder adopts creation ref) was explored and rejected
  because it fails for nested builders: inner CJ disposal frees the adopted array before outer
  builders can capture it. The phantom-ref cleanup machinery (`_constCreationBuilderRefs`,
  `_deferredConstCreationDisposes`, `_anonymousExtraDispose`) is architecturally correct for the
  nested-builder case and is the correct retained-owner model.
- **Scope correction delivered:** Instead of replacing the machinery, Step 3c scoped it correctly:
  removed `evalJaxprTransposed` markAnonymous parameter (arrays tracked in `internalArrays`
  instead), removed PE const promotion tracking (`_registerPEConstPromotion` et al.), renamed
  internal state for clarity, and confirmed which array categories need anonymous marking.
- **Preserved load-bearing behaviors:**
  1. **Nested-builder synchronization**: confirmed. Phantom-ref machinery handles this correctly.
  2. **Inlined-literal cleanup**: confirmed. `_inlineLiterals` → `_decrementBuilderRef` → deferred.
  3. **Evaluation-time Lit capture**: confirmed. `evalJaxpr` explicit `anonymousConstArrays.add()`
     retained. `evalJaxprTransposed` arrays don’t need it (tracked in `internalArrays`).
  4. **Outermost-boundary cleanup**: confirmed. `_processDeferredAnonymousDisposes()` at outermost.
- **For Step 3d:** The machinery is now correctly scoped and well-named, but deletion is no longer
  assumed to be the only valid next move. Before implementation, compare the current pivot against
  concrete alternatives and justify the chosen target state explicitly. The nested-builder case
  remains the key constraint.

### Phase 3c.5: Remove pivot scaffolding and restore source-of-truth alignment before Step 3d

Step 3c landed the correct ownership behavior, but it also left behind review debt from the design
pivot. This cleanup is mandatory before Step 3d so the next phase starts from a trustworthy code and
documentation baseline instead of carrying stale scaffolding.

- **Delete speculative scaffolding or wire it into a real invariant:** `_peTrackedArrays` /
  `_wasCreatedInPEScope()` were introduced as PE-origin bookkeeping, but they currently do not
  participate in any ownership decision. Before continuing, either remove them or make them part of
  one explicit invariant that Step 3d will actually rely on. Do not keep "for future use" ownership
  hooks in production code.
- **Remove commented-out ownership paths:** delete the commented `_peArrayCreationTracker` early
  return in `Array[Symbol.dispose]`. A dead commented branch in the disposal path is technical debt,
  not documentation.
- **Restore comment and test-header accuracy:** Step 3c reverted the protection rule to rc≤1, so any
  stale text claiming "all forwardJaxpr.consts are protected unconditionally" must be updated.
  Likewise, any stale references to "dead `ResidualCollector.dispose()`", the old anonymous-bridge
  fix, or the old `_anonymousBuilderRefs` / `_deferredAnonymousDisposes` names must be reconciled.
- **Re-audit the plan as code-adjacent documentation:** `PLAN.md` must describe the current code,
  not intermediate states. In particular, remove or rewrite any Step 3a–3b narrative that still
  describes PE-created arrays as being inserted into `anonymousConstArrays` now that 3b.5 is done.
- **Validation gate:** after this cleanup, rerun the ownership characterization suite and one full
  repository test pass before starting Step 3d. Documentation cleanup is only complete if it leaves
  the code and tests unchanged.

**Exit criterion for 3c.5:** no unused ownership scaffolding, no commented-out disposal branches,
and no plan/test prose that describes pre-pivot behavior as current reality.

### Phase 3c.75: Compare the current pivot against concrete alternatives before Step 3d

The current pivot is: keep the phantom-ref cleanup machinery because it correctly handles nested
builders after the adopt model failed. Before changing that machinery again, Step 3d must start from
an explicit comparison rather than an assumed destination.

- **Baseline (current pivot): retain the phantom-ref state machine** (`_constCreationBuilderRefs`,
  `_deferredConstCreationDisposes`, `_anonymousExtraDispose`) as the active retained-owner
  implementation for anonymous const lifecycle, with Step 3d limited to simplification and
  documentation hardening rather than replacement.
- **Alternative A: explicit lexical balancing** Replace phantom-ref cleanup with a pure `.ref` +
  `.dispose()` model where lexical code balances creation refs and each capture boundary takes one
  explicit retained handle. This is the intended end-state candidate only if it can satisfy
  nested-builder capture, inlined literals, and evaluation-time Lit materialization without
  introducing hybrid cleanup paths.
- **Alternative B: tracing-arena ownership** Introduce one named tracing-time retained owner for
  anonymous temporaries (arena/builder scope) that releases all such values at a single outer
  boundary. This may reduce state-machine complexity, but it is only acceptable if it preserves the
  Ownership Matrix vocabulary and does not reintroduce an implicit global tracing owner.
- **Rejected alternative (documented for comparison): adopt model**
  First-builder-adopts-creation-ref was explored and rejected because inner `ClosedJaxpr.dispose()`
  can free arrays still needed by outer builders. Keep this rejection in view so Step 3d does not
  drift back into the same failure mode under a different name.
- **Decision criteria:**
  1. Nested-builder safety: inner disposal must never invalidate outer capture.
  2. Single-owner clarity: every retained handle must have one named owner and one release path.
  3. No hybrid cleanup: the same value must not be eligible for both phantom cleanup and explicit
     retained-owner cleanup.
  4. Evaluation parity: `evalJaxpr`, `evalJaxprTransposed`, JVP lift zeros, and inlined literals
     must all fit the same chosen model or be explicitly carved out with one justified owner.
  5. Validation cost: the chosen model must be provable with the existing characterization suite
     plus focused nested-builder tests.
- **Required output before Step 3d implementation:** a short decision note in this plan stating
  which model is being pursued, why the other viable alternative was rejected, and which invariants
  will be used as the acceptance gate.

#### Feasibility Analysis: Explicit Lexical Balancing (Alternative A)

**Core insight.** The phantom-ref machinery exists because the `[Symbol.dispose]` guard blocks the
natural lexical disposal of anonymous arrays inside `makeJaxpr` bodies. The guard keeps the
"creation ref" (rc=1 from constructor) alive across the entire `makeJaxpr` body, and then the
phantom-ref state machine (`_constCreationBuilderRefs`, `_deferredConstCreationDisposes`,
`_anonymousExtraDispose`, `_processDeferredAnonymousDisposes`) fires that creation ref's disposal at
the outermost boundary when all builders have released.

But `getOrMakeConstTracer` already takes an independent `.ref` before the user's `using` can fire.
If the guard is removed, the user's `using` simply balances the creation ref (rc drops by 1), and
the builder's `.ref` remains alive independently. The builder's ref is then balanced by either
`_inlineLiterals` (for scalar consts) or `ClosedJaxpr.dispose()` (for non-scalar consts). This is
standard reference counting with no special machinery.

**Scenario traces (all verified against current code):**

| Scenario                                                  | Current System (phantom-ref)                                                                                                                                                                              | Explicit Lexical                                                                                                   | Result                                  |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| Simple scalar const: `using a = np.array(2); x.mul(a)`    | Guard blocks using. Builder takes .ref (rc=2). \_inlineLiterals disposes builder ref (rc=1). Phantom fires at boundary (rc=0).                                                                            | Using fires (rc 2→1). \_inlineLiterals disposes builder ref (rc 1→0).                                              | ✅ Same outcome                         |
| Non-scalar const: `using a = np.array([1,2,3]); x.add(a)` | Guard blocks. Builder .ref (rc=2). CJ.dispose: .dispose (rc=1), phantom fires (rc=0).                                                                                                                     | Using fires (rc 2→1). CJ.dispose releases builder ref (rc 1→0).                                                    | ✅ Same outcome                         |
| Nested builders, both capture same scalar                 | Guard blocks. Inner build: \_inlineLiterals (rc 2→1, builderRefs 1→0), deferred. Outer capture .ref (rc 2, builderRefs=1). Outer \_inlineLiterals (rc 1, builderRefs=0). Boundary phantom fires (rc 1→0). | Inner \_inlineLiterals (rc 2→1). Outer capture .ref (rc 2). Using fires (rc 1). Outer \_inlineLiterals (rc 1→0).   | ✅ Simpler                              |
| Nested builders, non-scalar                               | Guard blocks. builderRefs tracks both. CJ.dispose chain with builderRef decrement, extra dispose fires at last builder.                                                                                   | Inner CJ.dispose (rc 3→2). Outer CJ.dispose (rc 2→1). Using fires (rc 1→0).                                        | ✅ Much simpler, no builderRef tracking |
| `jit(grad(f))` — PE creates intermediates                 | PE tracker captures. Guard blocks. ResidualCollector disposes non-protected. Phantom handles builder-captured consts.                                                                                     | PE tracker captures. Using fires (if written). ResidualCollector disposes non-protected. Builder .ref independent. | ✅ Same for PE path                     |
| `evalJaxpr` Lit materialization                           | litArrays cleanup disposes non-result Lits. anonymousConstArrays.add for phantom-ref.                                                                                                                     | litArrays cleanup disposes non-result Lits. No phantom tracking needed.                                            | ✅ litArrays is sufficient              |
| JVP lift zeros                                            | liftedTangents cleanup disposes. anonymousConstArrays.add for phantom-ref.                                                                                                                                | liftedTangents cleanup disposes. No phantom tracking needed.                                                       | ✅ liftedTangents is sufficient         |
| Fire-and-forget: `x.mul(np.array(2))` (no `using`)        | Phantom-ref fires at boundary (rc 1→0 after builder releases).                                                                                                                                            | Builder .ref (rc=2). No using → creation ref leaks at rc=1 after CJ.dispose.                                       | ⚠️ Leak — but same as eager mode        |

**The fire-and-forget leak is by design.** The Ownership Matrix requires every array to have one
lexical owner who disposes it. Code that omits `using` leaks in eager mode, and explicit lexical
balancing makes `jit()` mode match. The current phantom-ref cleanup actually VIOLATES the stated
invariant that `jit()` must not rescue ownership-incorrect code:

> `jit()` is a performance optimization only. It must not rescue code that is ownership-incorrect in
> eager mode.

The phantom-ref machinery is a hidden arena that masks user-level leaks inside `jit()`. Removing it
makes jit and eager behavior identical for ownership.

**Infrastructure-level `pureArray` calls.** `getOrMakeConstTracer` calls `pureArray(val)` when `val`
is a raw value (number/boolean). This creates an anonymous array that nobody lexically owns. Fix:
after taking `.ref`, the `getOrMakeConstTracer` explicitly disposes the creation ref:

```ts
getOrMakeConstTracer(val: TracerValue): JaxprTracer {
  const wasRaw = !(val instanceof Tracer);
  if (wasRaw) val = pureArray(val);
  let tracer = this.builder.constTracers.get(val);
  if (tracer === undefined) {
    tracer = this.builder.newTracer(this, ...);
    (val as Tracer).ref;                         // builder takes retained handle
    this.builder.addConst(tracer, val);
    if (wasRaw) (val as Tracer).dispose();       // balance creation ref
  } else {
    tracer.trackLiftedConstant();
    if (wasRaw) (val as Tracer).dispose();       // pureArray was fresh, balance it
  }
  return tracer;
}
```

`pureArray` always creates a fresh Array (never hits the `constTracers` cache), so the `wasRaw`
disposal always fires on the `undefined` branch in practice.

`JVPTrace.pure` also calls `pureArray`, but feeds the result into `JVPTrace.lift`, which wraps it as
a JVPTracer primal. `JVPTracer.dispose()` cascades to `primal.dispose()`, balancing the creation ref
when the tracer is cleaned up. No separate fix needed.

**What gets deleted (complete list):**

| Mechanism                                         | File           | Why deletable                         |
| ------------------------------------------------- | -------------- | ------------------------------------- |
| `anonymousConstArrays` WeakSet                    | `array.ts`     | No phantom tracking needed            |
| `markAnonymousIfTracing`                          | `array.ts`     | No phantom tracking needed            |
| `pureArray`'s `anonymousConstArrays.add`          | `array.ts`     | No phantom tracking needed            |
| `[Symbol.dispose]` guard                          | `array.ts:342` | Builder .ref is independent           |
| `_constCreationBuilderRefs` WeakMap               | `jaxpr.ts`     | No builderRef counting                |
| `_deferredConstCreationDisposes` array            | `jaxpr.ts`     | No deferred queue                     |
| `_incrementBuilderRef`                            | `jaxpr.ts`     | No builderRef counting                |
| `_decrementBuilderRef`                            | `jaxpr.ts`     | No builderRef counting                |
| `_anonymousExtraDispose`                          | `jaxpr.ts`     | Creation ref balanced lexically       |
| `_processDeferredAnonymousDisposes`               | `jaxpr.ts`     | No deferred queue                     |
| `#inlinedAnonymousConsts` on ClosedJaxpr          | `jaxpr.ts`     | \_inlineLiterals just does .dispose() |
| `anonymousConstArrays` checks in CJ.dispose       | `jaxpr.ts`     | Just .dispose() on consts             |
| `anonymousConstArrays` checks in \_inlineLiterals | `jaxpr.ts`     | Just .dispose()                       |
| `anonymousConstArrays.add` in evalJaxpr           | `jaxpr.ts`     | litArrays cleanup is sufficient       |
| `anonymousConstArrays.add` in JVPTrace.lift       | `jvp.ts`       | liftedTangents cleanup is sufficient  |

**What remains unchanged:**

- `getOrMakeConstTracer` still calls `.ref` (essential — builder retained handle)
- `_inlineLiterals` still calls `.dispose()` on inlined consts (balances builder .ref)
- `CJ.dispose()` still calls `.dispose()` on consts (balances builder .ref)
- `liftedTangents` cleanup in `jvpFlat` (balances creation ref for lift zeros)
- `litArrays` cleanup in `evalJaxpr` (balances creation ref for Lit arrays)
- `ResidualCollector.dispose()` (handles PE dead intermediates — unchanged)
- `_peArrayCreationTracker` machinery (PE scope tracking — orthogonal)
- `JVPTracer[Symbol.dispose]` PE guard (orthogonal to phantom-ref)

**What changes:**

- `getOrMakeConstTracer`: dispose `pureArray` creation ref after taking `.ref`
- `_inlineLiterals`: remove `_decrementBuilderRef` + `inlinedAnonymous` tracking, keep `.dispose()`
- `ClosedJaxpr`: remove `#inlinedAnonymousConsts`, simplify `dispose()` to just `.dispose()` consts
- `ClosedJaxpr` constructor: remove `inlinedAnonymousConsts` parameter

**Risks:**

1. **Fire-and-forget leaks become visible.** Code that does `x.mul(np.array(2))` without `using`
   inside `jit()` will leak the `np.array(2)` at rc=1 after CJ.dispose. Currently the phantom
   machinery cleans this up silently. After the change, `checkLeaks` will flag these — which is the
   correct behavior per the design philosophy. ESLint + `checkLeaks` already catch these in eager
   mode.

2. **Characterization tests need updating.** Some "zero leak" tests that relied on phantom cleanup
   will show the same leak count as their bare (non-jit) counterparts. This is actually correct
   behavior — jit and eager should leak identically.

3. **Bare `grad(f)` is unaffected.** During bare `grad(f)`, `inMakeJaxprBody()` is false, so the
   `[Symbol.dispose]` guard was never active. The `ResidualCollector` and PE tracker handle cleanup.
   No change.

4. **Bare `grad(scan(f))` crash is unaffected.** This is a separate UAF from scan's anonymous init
   array being disposed during the backward pass. Orthogonal to phantom-ref machinery.

**Verdict: FEASIBLE.** Explicit lexical balancing is architecturally sound, consistent with stated
design principles, and substantially simpler than the current system. The only behavioral change is
that fire-and-forget anonymous arrays leak identically in jit and eager modes, which is the correct
behavior per the Ownership Matrix.

**Step 3d: Delete phantom-ref machinery (explicit lexical balancing).**

Phase 3c.75 analysis shows Alternative A (explicit lexical balancing) is feasible and consistent
with the Ownership Matrix. The adopt model (rejected) and tracing-arena (rejected — reintroduces
implicit global owner) are not viable. The current phantom-ref baseline works but violates the
`jit() == eager` ownership invariant by silently cleaning up fire-and-forget leaks.

**Decision: pursue Alternative A.** Reason: it makes jit and eager ownership semantics identical,
deletes ~150 lines of state-machine machinery, and satisfies all five decision criteria from Phase
3c.75 (nested-builder safety proved by scenario traces, single-owner clarity, no hybrid cleanup,
evaluation parity, provable with existing suite).

**Acceptance gate invariants:**

1. `getOrMakeConstTracer`'s `.ref` is the sole retained handle for every captured const.
2. Every creation ref is balanced by exactly one of: user `using`/`.dispose()`, `liftedTangents`
   cleanup, `litArrays` cleanup, `ResidualCollector`, or `getOrMakeConstTracer`'s explicit
   `pureArray` disposal. No phantom-ref pathway exists.
3. `_inlineLiterals`'s `.dispose()` and `CJ.dispose()`'s `.dispose()` each balance exactly one
   builder `.ref`. No extra phantom dispose fires.
4. `jit(f)` and bare `f` produce identical leak counts for identical code.

**Implementation sequence:**

1. **Fix `getOrMakeConstTracer`:** add explicit creation-ref disposal for `pureArray` results
   (`wasRaw` flag). This is the only new code.
2. **Delete phantom-ref state machine:** remove `anonymousConstArrays`, `_constCreationBuilderRefs`,
   `_deferredConstCreationDisposes`, `_incrementBuilderRef`, `_decrementBuilderRef`,
   `_anonymousExtraDispose`, `_processDeferredAnonymousDisposes`, `markAnonymousIfTracing`.
3. **Simplify `_inlineLiterals`:** remove `_decrementBuilderRef` + `inlinedAnonymous` tracking. Keep
   the `.dispose()` call (it balances the builder's `.ref`).
4. **Simplify `ClosedJaxpr`:** remove `#inlinedAnonymousConsts`, constructor parameter, `mapJaxpr`
   propagation. Simplify `dispose()` to just `.dispose()` on each const.
5. **Remove `[Symbol.dispose]` guard:** delete `if (inMakeJaxprBody() && ...)` in `array.ts`.
6. **Remove anonymous marking call sites:** `pureArray`'s `.add()`, `evalJaxpr`'s `.add()`,
   `JVPTrace.lift`'s `.add()`.
7. **Update characterization tests:** tests that verified phantom-ref cleanup (zero leaks under jit
   for fire-and-forget code) should now expect the same leak count as their bare counterpart.

Steps 1-6 can land as a single atomic commit because the phantom-ref machinery and explicit lexical
balancing do not conflict — removing the guard and the machinery simultaneously is safe because the
builder `.ref` provides independent ownership at every capture point. No hybrid intermediate state
is needed.

**Nested-builder verification (proved in feasibility analysis):** each builder takes `.ref`
independently. Inner `_inlineLiterals` or `CJ.dispose()` releases inner builder's ref. Outer
`_inlineLiterals` or `CJ.dispose()` releases outer builder's ref. The creation ref is a separate
counter balanced by the user's `using`. Total: +N (creation + N builders), -N (using + N builder
releases). All traces balance to zero.

**Inlined-literal verification:** `_inlineLiterals` calls `.dispose()` which releases the builder's
`.ref`. The creation ref was already balanced by the user's `using` (or leaked in both modes). No
`#inlinedAnonymousConsts` needed — the array is freed when rc reaches 0 from the `.dispose()` in
`_inlineLiterals` (if `using` already fired, rc goes from 1->0; if `using` fires later, it's a no-op
because the array's last ref was the builder's).

**Evaluation-time Lit verification:** `evalJaxpr`'s `litArrays` cleanup already balances the
creation ref for non-result Lit arrays. For result Lit arrays, the caller owns the creation ref
(same as any function return). The `anonymousConstArrays.add()` in `evalJaxpr` was purely for
phantom-ref tracking and is deletable.

**STATUS: DONE**

Steps 1-7 implemented. All phantom-ref machinery deleted (~150 lines). The explicit lexical
balancing replacement is simpler than the original plan anticipated because bare `using` was already
suppressed (guard deleted), so fire-and-forget arrays created inside `makeJaxpr` bodies needed an
explicit creation-ref balancing mechanism:

- `_arraysCreatedInMakeJaxprBody` WeakSet marks arrays born during any `makeJaxpr` body.
- `_liftedTangentZeros` WeakSet marks zero tangents from `JVPTrace.lift()` (creation ref owned by
  `liftedTangents` cleanup, not the builder).
- `constsNeedingCreationRefBalance` Set per `JaxprBuilder` tracks which captured consts have
  stranded creation refs.
- `makeJaxpr` snapshots consts with `refCount > 1` BEFORE `build()` (distinguishes stranded creation
  refs from user-balanced ones like `jacfwd`'s `eyeMatrix`).
- After `build()`, dispose snapshotted consts if `refCount > 0` and delete from
  `_arraysCreatedInMakeJaxprBody` (prevents double-balancing by outer builders).

**Acceptance gate results:**

1. ✅ `getOrMakeConstTracer`'s `.ref` is the sole retained handle for captured consts.
2. ✅ Creation refs balanced by: user `using`/`.dispose()`, `liftedTangents` cleanup, `litArrays`
   cleanup, `ResidualCollector`, `getOrMakeConstTracer`'s `wasRaw` disposal, or `makeJaxpr`'s
   post-build creation-ref balance loop. No phantom-ref pathway exists.
3. ✅ `_inlineLiterals`'s `.dispose()` and `CJ.dispose()`'s `.dispose()` each balance exactly one
   builder `.ref`.
4. ✅ `jit(f)` and bare `f` produce identical behavior — verified by 42/42 characterization tests
   and 2,064/2,064 full suite tests.

**Step 3e: Remove `_peArrayCreationTracker` guards from JVP paths.**

- `JVPTracer[Symbol.dispose]` guard (`jvp.ts:123`): replace with explicit disposal execution.
- Sort JVP `idx.dispose()` guard (`jvp.ts:419`): replace with explicit ownership transfer to
  Artifact.

### Phase 4: Verify wrapper and cache boundaries

- Verify artifact wrappers (`ResidualPackImpl`, `PrimalArtifactImpl`, `PullbackArtifactImpl`) own
  only their local conditionally retained state.
- Make boundary vocabulary explicit during the audit: values crossing wrapper/executor/cache
  boundaries should be describable as **borrowed**, **retained**, or **transferred**. Do not
  introduce a second consuming-semantics runtime to express this.
- Preserve `transposeJaxprCache` cache-owned contract.
- Audit `try/finally` parity around wrapper disposal and backward-pass cleanup.
- Can proceed in parallel with Phase 3.

### Phase 5: Remove remaining suppression paths and clean up

Step 3d deletes the phantom-ref state machine (`anonymousConstArrays`, `_constCreationBuilderRefs`,
`_deferredConstCreationDisposes`, `_anonymousExtraDispose`, `[Symbol.dispose]` guard). What remains
for Phase 5:

- Delete `_peArrayCreationTracker` infrastructure (definition in `core.ts:1589`, setter, all
  save/restore sites in `linearize.ts` and `jaxpr.ts`) — only after Step 3e replaces the JVP-path
  guards with explicit ownership transfers.
- Evaluate whether `ResidualCollector` can be simplified or removed after the PE-tracker guards are
  gone. It may still be needed for dead PE intermediate cleanup.
- Remove stale comments and debug logging for all deleted machinery.
- Every deletion above requires proving the replacement invariant in tests first.

## Key State Machine References

For implementors — exact locations of the machinery to modify or remove.

_(Updated after Step 3d. Struck-through items have been deleted.)_

| Mechanism                             | File           | Status              | Role                                                                       |
| ------------------------------------- | -------------- | ------------------- | -------------------------------------------------------------------------- |
| ~~`[Symbol.dispose]` guard~~          | `array.ts`     | **Deleted (3d)**    | Was: suppress disposal during tracing                                      |
| `_arraysCreatedInMakeJaxprBody`       | `array.ts`     | **New (3d)**        | Marks arrays born during makeJaxpr body for creation-ref balancing         |
| `_liftedTangentZeros`                 | `array.ts`     | **New (3d)**        | Marks JVP lift zeros — creation ref owned by liftedTangents cleanup        |
| `constsNeedingCreationRefBalance`     | `jaxpr.ts`     | **New (3d)**        | Per-builder Set tracking consts with stranded creation refs                |
| `_peArrayCreationTracker`             | `core.ts`      | Active              | Track PE-scope array creations                                             |
| `_setPACT`                            | `core.ts`      | Active              | Setter for tracker                                                         |
| PE tracker activation                 | `linearize.ts` | Active              | Install tracker in `partialEvalFlat`                                       |
| PE tracker save/restore               | `jaxpr.ts`     | Active              | Nested `makeJaxpr` isolation                                               |
| `ResidualCollector`                   | `linearize.ts` | Active              | Active disposal for dead PE intermediates                                  |
| `ResidualCollector.dispose()` call    | `linearize.ts` | Active              | Called in `vjpFlat`                                                        |
| ~~`anonymousConstArrays` definition~~ | `array.ts`     | **Deleted (3d)**    | Was: WeakSet identity tracking                                             |
| ~~`markAnonymousIfTracing`~~          | `array.ts`     | **Deleted (3d)**    | Was: `.add()` for arrays created in traced code                            |
| ~~`pureArray` anonymous mark~~        | `array.ts`     | **Deleted (3d)**    | Was: `.add()` for pureArray results                                        |
| `getOrMakeConstTracer`                | `jaxpr.ts`     | **Modified (3d)**   | `.ref` + `wasRaw` disposal + creation-ref tracking                         |
| ~~`evalJaxpr` Lit anonymous marking~~ | `jaxpr.ts`     | **Deleted (3d)**    | Was: enters anonymous-const state machine during evaluation                |
| ~~`_incrementBuilderRef`~~            | `jaxpr.ts`     | **Deleted (3d)**    | Was: builder ref count increment                                           |
| ~~`_decrementBuilderRef`~~            | `jaxpr.ts`     | **Deleted (3d)**    | Was: builder ref count decrement                                           |
| ~~`_anonymousExtraDispose`~~          | `jaxpr.ts`     | **Deleted (3d)**    | Was: phantom-ref fire/defer logic                                          |
| ~~`_deferredConstCreationDisposes`~~  | `jaxpr.ts`     | **Deleted (3d)**    | Was: deferred disposal queue processing                                    |
| `ClosedJaxpr.dispose()`               | `jaxpr.ts`     | **Simplified (3d)** | Just `.dispose()` on consts — no phantom-ref fire                          |
| ~~`#inlinedAnonymousConsts`~~         | `jaxpr.ts`     | **Deleted (3d)**    | Was: literals removed by `_inlineLiterals`                                 |
| `JVPTracer[Symbol.dispose]` guard     | `jvp.ts`       | Active (Step 3e)    | PE-scope suppress                                                          |
| Sort JVP idx guard                    | `jvp.ts`       | Active (Step 3e)    | PE-scope suppress                                                          |
| ~~JVP lift zero anonymous add~~       | `jvp.ts`       | **Replaced (3d)**   | Now uses `_liftedTangentZeros.add(zero)` instead of `anonymousConstArrays` |
| ~~ForiLoop JVP anonymous delete~~     | `jvp.ts`       | **Deleted (3d)**    | Was: `.delete()` from anonymousConstArrays                                 |
| `evalJaxprTransposed` zeros           | `linearize.ts` | Unchanged           | Tracked in `internalArrays` (no longer anonymous)                          |
| `evalJaxprTransposed` lit             | `linearize.ts` | Unchanged           | Tracked in `internalArrays` (no longer anonymous)                          |

## File Focus

- `src/frontend/array.ts` — `_arraysCreatedInMakeJaxprBody`, `_liftedTangentZeros`
- `src/frontend/linearize.ts` — `ResidualCollector`, `_peArrayCreationTracker` activation,
  transposition
- `src/frontend/jaxpr.ts` — `getOrMakeConstTracer`, `constsNeedingCreationRefBalance`,
  `ClosedJaxpr.dispose()`, makeJaxpr creation-ref balancing loop
- `src/frontend/jvp.ts` — PE-scope guards in `JVPTracer` and JVP rules, `_liftedTangentZeros` usage
- `src/frontend/core.ts` — `_peArrayCreationTracker` definition
- `src/frontend/artifacts.ts` — artifact wrapper ownership boundaries
- `test/tracing.test.ts`, `test/refcount.test.ts` — existing ownership tests
- `test/leak-repro.test.ts` — reproduction tests from prior investigation

## Validation Plan

Run focused checks during the refactor, then the broader suite before landing:

1. `pnpm vitest run test/tracing.test.ts`
2. `pnpm vitest run test/refcount.test.ts`
3. `pnpm vitest run test/leak-repro.test.ts`
4. `pnpm vitest run test/lax-scan.test.ts`
5. `pnpm vitest run test/transform-compositions.test.ts`
6. `pnpm build`
7. `pnpm check`
8. `pnpm test`

Add focused regression tests whenever a bug is reproduced through `using`, explicit `.dispose()`,
captured consts, or nested transform composition.

## Acceptance Criteria

- User code can use `using` and explicit `.dispose()` inside ownership-sensitive transform paths
  without triggering `UseAfterFreeError`.
- Retained internal handles are modeled explicitly rather than inferred from tracing context.
- `transposeJaxprCache` and other cache-owned objects retain their ownership contract.
- The final code has fewer special cases than the current implementation.
- PE intermediates do not leak (verified by `checkLeaks` tests).
- The focused ownership tests and full suite pass.

## Non-Goals

- Changing the public non-consuming array API.
- Reintroducing user-facing `.ref` requirements.
- Fixing unrelated ownership issues that are outside tracing, autodiff, captured constants, or
  artifact cleanup.

## Migration Bundle For Future Work

Deterministic mapping for the desired end state:

- Implicit tracing ownership → explicit retained handle.
- "Everything created during tracing dies together" → only dead PE temporaries may use scoped
  collector cleanup; any escaping value must name its retained owner.
- Context-based disposal suppression → normal `using` / `.dispose()` semantics.
- Builder/collector cleanup that guesses ownership → cleanup that releases only explicitly retained
  state.
- Three-layer phantom-ref state machine → `constsNeedingCreationRefBalance` per-builder Set +
  `makeJaxpr` post-build balancing loop. Single `.ref` in `getOrMakeConstTracer`, single
  `.dispose()` in `ClosedJaxpr.dispose()`. _(Done in Step 3d.)_

Workaround signatures for downstream cleanup:

- ~~Delete any local code that avoids `using` only because tracing used to special-case it.~~
  _(Done: `[Symbol.dispose]` guard deleted.)_
- Delete any helper that mirrors hidden tracing ownership instead of taking a real retained handle.
- ~~Delete any comments that justify disposal suppression once explicit ownership is in place.~~
  _(Done: phantom-ref comments removed with the code.)_
- Delete `_peArrayCreationTracker` checks in JVP rules once PE retention is explicit. _(Step 3e)_
- ~~Delete `anonymousConstArrays` checks in `[Symbol.dispose]` once const ownership is explicit.~~
  _(Done: entire `anonymousConstArrays` WeakSet deleted.)_
