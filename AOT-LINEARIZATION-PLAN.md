# AOT Linearization: Architectural Redesign Plan

**Branch:** `aot-linearization-plan`
**Scope:** Refactor the autodiff + JIT ownership model from ad-hoc PE intermediate
cleanup into explicit, artifact-scoped lifecycle management.
**Execution model:** Single autonomous agent. No human review gates.

---

## Motivation

The current autodiff pipeline in `src/frontend/linearize.ts` produces correct results
but suffers from ownership fragility:

1. **`disposePeIntermediates` (L213–260)** uses heuristic protected-set logic with
   `unwrapToConcreteArray` wrapper chasing — any new wrapper type or transform
   composition can break the identity checks.
2. **ClosedJaxpr const ownership** relies on timing: `getOrMakeConstTracer` calls
   `.ref`, `ClosedJaxpr.dispose()` drops it, but user disposal or nested transforms
   can interleave unpredictably. Three separate patches were needed (anonymous consts,
   unreachable Const PETracers, user-disposed consts with `refCount ≤ 1` protection).
3. **Scan backward pass** (L1870–2050) manually reconstructs primal residuals,
   tangent bodies, and transpose jaxprs per invocation — no caching, fragile slicing.
4. **`transposeJaxprCache` (L2152)** and `jitCompileCache` (jit.ts L475) are global
   mutable maps requiring coordinated cleanup via `_registerJitCacheDisposer`.

The redesign replaces this with three explicit artifact types that own their resources:

| Artifact          | Owns                                       | Created by           | Consumed by          |
|-------------------|--------------------------------------------|----------------------|----------------------|
| `PrimalArtifact`  | Forward jaxpr + compiled program            | `aotLinearize()`     | `aotVjp()`           |
| `ResidualPack`    | Concrete residual arrays from forward pass  | `PrimalArtifact.run()` | `PullbackArtifact.run()` |
| `PullbackArtifact`| Transposed jaxpr + compiled backward program| `aotLinearize()`     | User/grad            |

Each artifact implements `[Symbol.dispose]` and owns exactly the resources it needs.
No global heuristic disposal.  No protected-set chasing.

---

## Milestones

### M0 — Baseline snapshot & test harness (1–2 days)

Establish regression baselines before any refactoring.

#### M0.1 — Record baseline test results

**What:** Run the full test suite and Deno tests, record pass/fail counts.

**Files touched:** None (read-only).

**Commands:**
```bash
pnpm build
pnpm vitest run 2>&1 | tail -20 > tmp/m0-vitest-baseline.txt
pnpm run test:deno 2>&1 | tail -20 > tmp/m0-deno-baseline.txt
```

**Exit criteria:** Baseline files exist in `tmp/`. Zero unexpected failures.

---

#### M0.2 — Create artifact type stubs

**What:** Create `src/frontend/artifacts.ts` with empty type definitions and a
barrel re-export from `src/index.ts`. No logic yet — just types that compile.

**Types to define:**

```typescript
export interface PrimalArtifact extends Disposable {
  readonly forwardJaxpr: ClosedJaxpr;
  run(primals: Tracer[]): { primalsOut: Tracer[]; residuals: ResidualPack };
}

export interface ResidualPack extends Disposable {
  readonly arrays: Tracer[];
  readonly consumed: boolean;
}

export interface PullbackArtifact extends Disposable {
  readonly backwardJaxpr: ClosedJaxpr;
  run(residuals: ResidualPack, cotangents: Tracer[]): Tracer[];
}

export type AotLinearizeResult = {
  primal: PrimalArtifact;
  pullback: PullbackArtifact;
};

export function aotLinearize(
  f: (...args: Tracer[]) => Tracer[],
  exampleArgs: Tracer[],
): AotLinearizeResult;
```

**Files touched:**
- Create `src/frontend/artifacts.ts`
- Edit `src/index.ts` — add re-export

**Commands:**
```bash
pnpm run check   # must compile
pnpm vitest run   # no regressions
```

**Exit criteria:** Types compile. All existing tests pass.

---

#### M0.3 — Artifact-focused test file

**What:** Create `test/artifacts.test.ts` with placeholder tests that import the
stub types and verify they exist. Tests will initially be trivial (`expect(aotLinearize).toBeDefined()`).

**Files touched:**
- Create `test/artifacts.test.ts`

**Commands:**
```bash
pnpm vitest run test/artifacts.test.ts
```

**Exit criteria:** Test file runs. Placeholder tests pass.

---

### M1 — Extract forward/backward split (3–5 days)

Factor `partialEvalFlat` (linearize.ts L109) into a two-phase pipeline:
**phase 1** traces and produces a forward jaxpr + residual specification,
**phase 2** produces a backward jaxpr via `transposeJaxpr`.

This milestone does NOT change any public API or ownership semantics. It is a
pure internal refactor that makes the existing code more modular.

#### M1.1 — Extract `buildForwardJaxpr` from `partialEvalFlat`

**What:** Extract lines L109–200 of `partialEvalFlat` into a standalone function
`buildForwardJaxpr(f, pvalsIn)` that returns `{ jaxpr, pvalsOut, residualSpec }`.

The `residualSpec` is a new type describing which variables are residuals
(currently implicit in the "known" vs "unknown" PETracer partition).

**Key code to extract** (linearize.ts):
- PE trace setup (L122–130): `newMain(PartialEvalTrace)`, `trace.newArg(pval)`
- `_peArrayCreationTracker` scope (L135–155): save/restore + error cleanup
- Body execution (L148): `f(...tracersIn)`
- PETracer classification (L159–164): `unknownTracersIn/Out`
- `partialEvalGraphToJaxpr` call (L164)
- Unreachable Const PETracer cleanup (L173–185)
- Intermediate collection (L187–190)

**Refactoring strategy:** The extracted function keeps identical logic — just
relocated. No ownership changes yet. `partialEvalFlat` becomes a thin wrapper
calling `buildForwardJaxpr`.

**Files touched:**
- `src/frontend/linearize.ts` — extract function, keep old as wrapper

**Test commands:**
```bash
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run test/leak-diagnostic.test.ts
pnpm vitest run test/check-leaks.test.ts
pnpm vitest run test/lax-scan.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** All tests pass. `partialEvalFlat` delegates to `buildForwardJaxpr`.

---

#### M1.2 — Extract `buildBackwardJaxpr` wrapping `transposeJaxpr`

**What:** Create `buildBackwardJaxpr(forwardJaxpr, undefPrimals)` that wraps the
existing `transposeJaxpr` (L2165) call. This is a thin wrapper now, but will
become the `PullbackArtifact` constructor later.

**Key code to wrap** (linearize.ts):
- `transposeJaxpr(jaxpr, undefPrimals)` at L2165–2195
- Cache lookup/store logic at L2152–2194

**Files touched:**
- `src/frontend/linearize.ts` — new function, `transposeJaxpr` unchanged

**Test commands:**
```bash
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** All tests pass. New function exists and is called from at least
one site (can be a test).

---

#### M1.3 — Extract `disposePeIntermediates` into a `ResidualCollector` class

**What:** Replace the free function `disposePeIntermediates` (L213–260) with a
`ResidualCollector` class that:
1. Receives PE intermediates and literal intermediates from `buildForwardJaxpr`
2. Receives the protected-set from the caller (output arrays, aux captures, const-protection)
3. Has a `.dispose()` method that runs the current disposal logic
4. Has a `.getResiduals()` method returning the residual arrays

This class is the precursor to `ResidualPack`. For now it has identical behavior
to the free function, just encapsulated.

**Key code to encapsulate** (linearize.ts L213–260):
- `unwrapToConcreteArray` traversal (L201–210)
- `allProtected` set construction with concrete-array chasing (L227–240)
- Conditional disposal loop with `disposed` set dedup (L241–260)

**Files touched:**
- `src/frontend/linearize.ts` — replace function with class
- Call sites: `linearizeFlat` (L315), `vjpFlat` (L2198)

**Test commands:**
```bash
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run test/leak-diagnostic.test.ts
pnpm vitest run test/check-leaks.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** All tests pass. `disposePeIntermediates` replaced by
`ResidualCollector` with identical behavior.

---

#### M1.4 — Decouple `linearizeFlat` into `buildForwardJaxpr` + `buildBackwardJaxpr`

**What:** Rewrite `linearizeFlat` (L315–340) and `linearizeFlatUtil` (L286–313) to
use the extracted functions from M1.1 and M1.2 instead of inlining the logic.

`linearizeFlatUtil` currently:
1. Calls `jvp()` to get JVP'd function
2. Calls `partialEvalFlat()` to get forward jaxpr
3. Returns `{ primalsOut, jaxpr, peIntermediates, literalIntermediates }`

After this task, it returns a `ForwardResult` containing the forward jaxpr and
a `ResidualCollector` (from M1.3).

**Files touched:**
- `src/frontend/linearize.ts` — rewrite `linearizeFlatUtil` and `linearizeFlat`

**Test commands:**
```bash
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run test/lax-scan.test.ts
pnpm vitest run test/numpy-linalg.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** All tests pass. `linearizeFlatUtil` uses extracted functions.

---

### M2 — Implement artifact types (3–5 days)

Build real implementations of `PrimalArtifact`, `ResidualPack`, `PullbackArtifact`
on top of the M1 extractions.

#### M2.1 — Implement `ResidualPack`

**What:** Implement `ResidualPack` in `src/frontend/artifacts.ts`:

```typescript
class ResidualPackImpl implements ResidualPack {
  #arrays: Tracer[];
  #consumed = false;

  constructor(arrays: Tracer[]) {
    this.#arrays = arrays;
  }

  get arrays(): Tracer[] {
    if (this.#consumed) throw new UseAfterFreeError("ResidualPack already consumed");
    return this.#arrays;
  }

  get consumed(): boolean { return this.#consumed; }

  consume(): Tracer[] {
    if (this.#consumed) throw new UseAfterFreeError("ResidualPack already consumed");
    this.#consumed = true;
    return this.#arrays;
  }

  [Symbol.dispose](): void {
    if (!this.#consumed) {
      for (const a of this.#arrays) a.dispose();
      this.#consumed = true;
    }
  }
}
```

**Key design decision:** `consume()` transfers ownership — arrays are NOT disposed
by the pack, the consumer is responsible. `dispose()` frees unclaimed residuals
(e.g., if the backward pass is never run).

**Files touched:**
- `src/frontend/artifacts.ts`
- `test/artifacts.test.ts` — add tests

**Tests to add:**
- `ResidualPack.consume() transfers ownership`
- `ResidualPack.dispose() frees unclaimed arrays`
- `ResidualPack.consume() after dispose() throws UseAfterFreeError`
- `ResidualPack.consume() twice throws UseAfterFreeError`

**Exit criteria:** `ResidualPack` tests pass. Leak checks pass.

---

#### M2.2 — Implement `PullbackArtifact`

**What:** Implement `PullbackArtifact` wrapping a `ClosedJaxpr` (the backward jaxpr)
and a compiled `JitProgram`:

```typescript
class PullbackArtifactImpl implements PullbackArtifact {
  #backwardJaxpr: ClosedJaxpr;
  #disposed = false;

  constructor(backwardJaxpr: ClosedJaxpr) {
    this.#backwardJaxpr = backwardJaxpr;
  }

  get backwardJaxpr(): ClosedJaxpr { return this.#backwardJaxpr; }

  run(residuals: ResidualPack, cotangents: Tracer[]): Tracer[] {
    if (this.#disposed) throw new UseAfterFreeError("PullbackArtifact disposed");
    const resArrays = residuals.consume(); // transfers ownership
    const allInputs = [...resArrays, ...cotangents];
    return evalJaxpr(this.#backwardJaxpr.jaxpr, allInputs);
  }

  [Symbol.dispose](): void {
    if (!this.#disposed) {
      this.#backwardJaxpr.dispose();
      this.#disposed = true;
    }
  }
}
```

**Files touched:**
- `src/frontend/artifacts.ts`
- `test/artifacts.test.ts` — add tests

**Tests to add:**
- `PullbackArtifact.run() produces correct gradients` (compare with `grad`)
- `PullbackArtifact.dispose() frees backward jaxpr consts`
- `PullbackArtifact after dispose() throws`

**Exit criteria:** Pullback tests pass. Leak checks pass.

---

#### M2.3 — Implement `PrimalArtifact`

**What:** Implement `PrimalArtifact` wrapping the forward jaxpr. Its `run()` method
executes the forward pass and produces a `ResidualPack`:

```typescript
class PrimalArtifactImpl implements PrimalArtifact {
  #forwardJaxpr: ClosedJaxpr;
  #residualIndices: number[]; // which outputs are residuals vs user outputs
  #disposed = false;

  run(primals: Tracer[]): { primalsOut: Tracer[]; residuals: ResidualPack } {
    if (this.#disposed) throw new UseAfterFreeError("PrimalArtifact disposed");
    const allOut = evalJaxpr(this.#forwardJaxpr.jaxpr,
      [...this.#forwardJaxpr.consts.map(c => c.ref), ...primals]);
    const userOuts = allOut.filter((_, i) => !this.#residualIndices.includes(i));
    const resArrays = allOut.filter((_, i) => this.#residualIndices.includes(i));
    return { primalsOut: userOuts, residuals: new ResidualPackImpl(resArrays) };
  }

  [Symbol.dispose](): void {
    if (!this.#disposed) {
      this.#forwardJaxpr.dispose();
      this.#disposed = true;
    }
  }
}
```

**Files touched:**
- `src/frontend/artifacts.ts`
- `test/artifacts.test.ts` — add tests

**Tests to add:**
- `PrimalArtifact.run() produces correct outputs + residuals`
- `PrimalArtifact.dispose() frees forward jaxpr consts`
- `ResidualPack from PrimalArtifact feeds into PullbackArtifact correctly`

**Exit criteria:** End-to-end primal→residual→pullback chain works. Leak checks pass.

---

#### M2.4 — Implement `aotLinearize`

**What:** The top-level factory function that composes M1 extractions:

```typescript
export function aotLinearize(
  f: (...args: Tracer[]) => Tracer[],
  exampleArgs: Tracer[],
): AotLinearizeResult {
  // 1. JVP the function
  // 2. Partial-eval → forward jaxpr (via buildForwardJaxpr from M1.1)
  // 3. Transpose → backward jaxpr (via buildBackwardJaxpr from M1.2)
  // 4. Wrap in PrimalArtifact + PullbackArtifact
  // 5. Dispose PE intermediates (via ResidualCollector from M1.3)
  return { primal, pullback };
}
```

**Files touched:**
- `src/frontend/artifacts.ts`
- `src/index.ts` — export `aotLinearize`
- `test/artifacts.test.ts` — integration tests

**Tests to add:**
- `aotLinearize produces correct grad for sin`
- `aotLinearize produces correct grad for polynomial`
- `aotLinearize + dispose has zero leaks`
- `aotLinearize reuse: compile once, run many times`

**Exit criteria:** `aotLinearize` works for simple functions. All leak checks pass.

---

### M3 — Wire into existing transforms (3–5 days)

Replace the internals of `vjpFlat`, `linearizeFlat`, and `grad` to use artifacts
instead of the ad-hoc intermediate tracking.

#### M3.1 — Rewrite `vjpFlat` to use artifacts internally

**What:** Replace the body of `vjpFlat` (linearize.ts L2198–2250) to:
1. Call `aotLinearize` to get primal + pullback artifacts
2. Run the primal artifact to get outputs + residual pack
3. Return a `vjpFn` closure that calls `pullback.run(residuals, cotangents)`
4. Return a `dispose` function that disposes both artifacts

This eliminates:
- The `disposePeIntermediates` call at L2227
- The pending-flush block at L2233
- The manual `evalJaxprTransposed` call in the vjp closure
- The manual `ClosedJaxpr.dispose()` in the dispose function

**Current code structure** (linearize.ts L2198–2300):
```
vjpFlat(f, primalsIn, auxStore?)
  → linearizeFlatUtil(f, primalsIn)      // → { primalsOut, jaxpr, peIntermediates, literalIntermediates }
  → protectedVals construction           // L2206–2216
  → disposePeIntermediates(...)          // L2217  ← REPLACED
  → flush pending ops                   // L2233  ← REPLACED
  → return [primalsOut, fVjp, dispose]
```

**After rewrite:**
```
vjpFlat(f, primalsIn, auxStore?)
  → aotLinearize(f, primalsIn)           // → { primal, pullback }
  → primal.run(primalsIn)               // → { primalsOut, residuals }
  → return [primalsOut, fVjp, dispose]
    where fVjp calls pullback.run(residuals, cts)
    where dispose calls primal.dispose() + pullback.dispose()
```

**Files touched:**
- `src/frontend/linearize.ts` — rewrite `vjpFlat`

**Test commands:**
```bash
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run test/leak-diagnostic.test.ts
pnpm vitest run test/check-leaks.test.ts
pnpm vitest run test/numpy-linalg.test.ts
pnpm vitest run test/lax-scan.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** All tests pass. `vjpFlat` uses artifacts. Zero leaks.

---

#### M3.2 — Rewrite `linearizeFlat` to use artifacts internally

**What:** Replace the body of `linearizeFlat` (L315–340) analogously.

**Current structure** (linearize.ts L315):
```
linearizeFlat(f, primals)
  → linearizeFlatUtil(f, primals)
  → disposePeIntermediates(...)
  → flush pending
  → return [primalsOut, fLin, dispose]
    where fLin evaluates jaxpr with tangents
```

**After rewrite:** `linearizeFlat` calls `aotLinearize`, uses the forward jaxpr
for `fLin`. The closure captures residuals for later tangent evaluation.

**Files touched:**
- `src/frontend/linearize.ts` — rewrite `linearizeFlat`

**Test commands:**
```bash
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run test/leak-diagnostic.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** All tests pass. `linearizeFlat` uses artifacts.

---

#### M3.3 — Verify transform compositions

**What:** Run the systematic composition tests and verify all depths work:

The `test/transform-compositions.test.ts` file tests:
- Depth-1: `grad(f)`, `jit(f)`, `jvp(f)`, `vjp(f)`, `vmap(f)`, `linearize(f)`
- Depth-2: `grad(grad(f))`, `jit(grad(f))`, `grad(jit(f))`, `vmap(grad(f))`, etc.
- Depth-3: `jit(grad(grad(f)))`, `grad(jit(grad(f)))`, etc.
- Depth-4+: `jit(vmap(grad(f)))`, `vmap(jit(grad(f)))`, etc.

**Special compositions to verify:**
- `grad(scan)` — exercises scan backward pass (L1870–2050)
- `jit(grad(scan))` — JIT wrapping the backward pass
- `vmap(grad(scan))` — batched gradients
- `grad(associativeScan)` — trace-through (no primitive)

**Files touched:** None (verification only).

**Commands:**
```bash
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run test/lax-scan.test.ts
pnpm vitest run test/lax-associative-scan.test.ts
pnpm vitest run test/numpy-linalg.test.ts
pnpm vitest run test/lax-linalg.test.ts
pnpm vitest run   # full suite
pnpm run test:deno
```

**Exit criteria:** All tests pass including Deno WebGPU. Transform compositions
at all depths produce correct results with zero leaks.

---

### M4 — Scan backward pass artifact (3–5 days)

The scan backward pass (linearize.ts L1870–2050) is the most complex consumer
of the transpose machinery. This milestone wraps it in artifacts.

#### M4.1 — Extract scan backward step helpers

**What:** The scan backward pass currently has two inline helpers:
- `runOneForwardStep` — re-runs one forward iteration from a checkpoint
- `runOneBackwardStep` — transposes one backward iteration

Extract these into standalone functions with explicit input/output ownership.

**Current code** (linearize.ts):
- `runOneBackwardStep(iter, primalCarry)` at ~L1930
- `runOneForwardStep(iter, carry)` referenced in the checkpoint loop

**Files touched:**
- `src/frontend/linearize.ts` — extract helpers

**Test commands:**
```bash
pnpm vitest run test/lax-scan.test.ts
pnpm vitest run test/leak-diagnostic.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** All scan tests pass. Extracted helpers used.

---

#### M4.2 — Cache transposed scan body jaxpr

**What:** Currently, the scan backward pass calls `transposeJaxpr(tangentBody.jaxpr,
tangentBodyUndefPrimals)` (L1909) on every backward invocation. The result is
already cached by `transposeJaxprCache` (L2152), but the tangent body tracing
and undefPrimals construction is repeated.

Wrap this into a `ScanBackwardSpec` that is computed once and cached:

```typescript
interface ScanBackwardSpec {
  transposedBody: ClosedJaxpr;
  numPrimalCarry: number;
  numTangentCarry: number;
  numPrimalY: number;
  numTangentX: number;
  numTangentConsts: number;
  bodyUndefPrimals: boolean[];
}
```

**Files touched:**
- `src/frontend/linearize.ts` — create `ScanBackwardSpec`, cache it

**Test commands:**
```bash
pnpm vitest run test/lax-scan.test.ts
pnpm vitest run test/leak-diagnostic.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** Scan backward spec is computed once. Performance matches baseline.

---

#### M4.3 — Wrap scan backward pass in `ScanPullbackArtifact`

**What:** Create a `ScanPullbackArtifact` that owns:
- The `ScanBackwardSpec` from M4.2
- Checkpoint carries (the √N-spaced forward-pass state)
- xs residuals

Its `.run(cotangents)` method executes the checkpoint-based backward loop.
Its `[Symbol.dispose]()` frees checkpoint carries and xs residuals.

This replaces the 180-line inline backward pass in the Scan transpose rule.

**Files touched:**
- `src/frontend/artifacts.ts` or `src/frontend/scan-artifacts.ts`
- `src/frontend/linearize.ts` — Scan transpose rule delegates to artifact
- `test/artifacts.test.ts` — scan backward tests

**Tests to add:**
- `ScanPullbackArtifact produces correct grad(scan) for cumsum`
- `ScanPullbackArtifact with checkpoint: false matches grad(scan)`
- `ScanPullbackArtifact.dispose() frees checkpoint carries`

**Exit criteria:** `grad(scan)` works via `ScanPullbackArtifact`. All scan tests pass.

---

### M5 — Cleanup & documentation (2–3 days)

Remove dead code, update documentation, verify final state.

#### M5.1 — Remove `disposePeIntermediates` free function

**What:** After M3, the free function `disposePeIntermediates` (L213–260) should
have no call sites. Remove it and the `unwrapToConcreteArray` helper (L201–210).

**Precondition:** Verify no callers remain:
```bash
grep -rn 'disposePeIntermediates' src/
```

**Files touched:**
- `src/frontend/linearize.ts` — remove dead code

**Exit criteria:** Function removed. All tests pass.

---

#### M5.2 — Consolidate cache cleanup

**What:** Currently, cache cleanup is distributed:
- `transposeJaxprCache` cleanup registered via `_registerJitCacheDisposer` (L2155)
- `jitCompileCache` cleanup in `jit.ts` L475
- `_jitFunctionDisposers` set in `check-leaks.ts` L76

After M2–M4, artifact caches should be self-cleaning (artifacts own their
jaxprs, disposing the artifact frees the jaxpr). Audit remaining caches and
simplify the `_registerJitCacheDisposer` registration pattern.

**Files touched:**
- `src/frontend/linearize.ts`
- `src/frontend/check-leaks.ts`
- `src/frontend/jit.ts` (if applicable)

**Test commands:**
```bash
pnpm vitest run test/check-leaks.test.ts
pnpm vitest run   # full suite
```

**Exit criteria:** Cache cleanup is simpler. No orphaned registrations.

---

#### M5.3 — Update copilot-instructions.md

**What:** Update Part 4 of `.github/copilot-instructions.md` to document:
- The artifact types and their ownership contracts
- How `aotLinearize` replaces `partialEvalFlat` + `disposePeIntermediates`
- Updated "Debugging Ownership Issues" section
- Updated "Transform compositions to verify" table

**Files touched:**
- `.github/copilot-instructions.md`

**Exit criteria:** Documentation reflects new architecture.

---

#### M5.4 — Update FEATURES.md

**What:** Add `aotLinearize` to the API table in `FEATURES.md`.

**Files touched:**
- `FEATURES.md`

**Exit criteria:** Feature table updated.

---

#### M5.5 — Final regression run

**What:** Full CI-equivalent check.

**Commands:**
```bash
pnpm build
pnpm run check
pnpm vitest run
pnpm run test:deno
pnpm run lint
pnpm run format:check
```

**Exit criteria:** All checks pass. Zero regressions from M0 baseline.

---

## Dependency Graph

```
M0.1 (baseline)
  │
  ├─→ M0.2 (type stubs) ──→ M0.3 (test file)
  │
  ├─→ M1.1 (extract buildForwardJaxpr)
  │     │
  │     ├─→ M1.2 (extract buildBackwardJaxpr)
  │     │
  │     └─→ M1.3 (ResidualCollector class)
  │           │
  │           └─→ M1.4 (decouple linearizeFlatUtil)
  │                 │
  │                 └─→ M2.1 (ResidualPack) ──→ M2.2 (PullbackArtifact)
  │                       │                        │
  │                       └─→ M2.3 (PrimalArtifact)
  │                             │
  │                             └─→ M2.4 (aotLinearize) ──→ M3.1 (vjpFlat)
  │                                                          │
  │                                                          ├─→ M3.2 (linearizeFlat)
  │                                                          │
  │                                                          └─→ M3.3 (verify compositions)
  │                                                                │
  │                                                                └─→ M4.1 (scan helpers)
  │                                                                      │
  │                                                                      └─→ M4.2 (scan spec cache)
  │                                                                            │
  │                                                                            └─→ M4.3 (ScanPullbackArtifact)
  │                                                                                  │
  │                                                                                  └─→ M5.1–M5.5 (cleanup)
```

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| `disposePeIntermediates` has side effects not captured by new model | Leak or double-free | M1.3 preserves exact behavior; M3 validates via full suite |
| Scan backward pass has implicit state sharing with forward | Incorrect gradients | M4.3 test compares against current `grad(scan)` output |
| `transposeJaxprCache` invalidation breaks under artifacts | Stale backward programs | M5.2 audits cache lifecycle |
| `_peArrayCreationTracker` scope interacts with artifact construction | Missing intermediates | M1.1 preserves exact tracker scope |
| Nested transforms (grad(grad(f))) create recursive artifacts | Stack overflow or leak | M3.3 tests depth-4 compositions |
| `insideAbstractTrace()` guard in `disposePeIntermediates` (L214) | Skip disposal during tracing | M1.3 preserves guard in ResidualCollector |

## Invariants to Maintain

Throughout all milestones, these invariants must hold:

1. **Slot-count parity:** `slotCount()` before and after any operation matches
   (enforced by `test/setup.ts` global `checkLeaks`).
2. **Transform composition correctness:** All depth-1 through depth-4 compositions
   in `test/transform-compositions.test.ts` pass.
3. **Scan gradient correctness:** `grad(scan)` matches finite differences for
   cumsum, Cholesky-in-body, and pytree carry patterns.
4. **No new public API breaks:** `aotLinearize` is additive. Existing `grad`,
   `vjp`, `linearize`, `jit` APIs unchanged.
5. **Build succeeds:** `pnpm run check` (TypeScript) passes at every commit.

## Estimated Timeline

| Milestone | Effort   | Cumulative |
|-----------|----------|------------|
| M0        | 1–2 days | 1–2 days   |
| M1        | 3–5 days | 4–7 days   |
| M2        | 3–5 days | 7–12 days  |
| M3        | 3–5 days | 10–17 days |
| M4        | 3–5 days | 13–22 days |
| M5        | 2–3 days | 15–25 days |

Total: **3–5 weeks** of focused implementation.

## Commit Strategy

- One commit per task (M0.1, M0.2, ..., M5.5).
- Commit message format: `aot-linearize M{n}.{m}: {short description}`
- Every commit must pass `pnpm vitest run` (enforced by pre-commit hook).
- Branch off `main` at start. Merge back after M5.5 passes full CI.
