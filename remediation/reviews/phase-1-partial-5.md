# Re-review: phase 1 steady-state leak checks, round 5

- **Range:** `9a19712..ea88cbf` (one commit, `ea88cbf`).
- **Date:** 2026-11-09.
- **Scope:** open findings F1-F4 from [round 4](phase-1-partial-4.md), their fixes and new
  regression coverage. Previously accepted item 7 and item 10 work is not reopened.
- **Verdict:** **fail**: F1 is partially fixed but still admits an asynchronous result typed as a
  synchronous report. F2-F4 are verified fixed.
- **Reviewer changes:** no implementation changes; scratch experiments were confined to a fresh
  clone under `tmp/`. Only this report is added to the main worktree.
- **Exclusion carried forward:** homepage content.

## Independent validation

Fresh local clone at `ea88cbf`, with normal `pnpm install --frozen-lockfile` and its build/website
generators, not linked dependencies.

| Check                                         | Result                                                      |
| --------------------------------------------- | ----------------------------------------------------------- |
| `pnpm build`, `pnpm check`, `pnpm lint`       | Pass                                                        |
| Formatting of all four files in the range     | Pass                                                        |
| Full default suite                            | 92 files passed; 3406 tests passed / 18 skipped             |
| Two leak-check files, CPU/WASM                | 53 passed / 26 skipped                                      |
| Same two files, NVIDIA                        | 75 passed / 4 skipped                                       |
| Same two files, Intel                         | 75 passed / 4 skipped                                       |
| New structural-thenable scratch regression    | Type-check passes; runtime assertion fails with `undefined` |
| New tests against the previous checker source | Four type errors; four runtime failures                     |
| Restored `ea88cbf` checker                    | Clean clone status; type-check and all 27 API tests pass    |

Focused counts and the default-suite count match the implementation's claims. The per-area file
still has 48 passing cases and four skips, as corrected in the plan. Full GPU and full CPU-only
suites were not rerun: this re-review uses the full default suite and focused coverage on both GPUs;
no shader or worker-runtime code changed.

## Finding disposition

| Finding                                | Status and evidence                                                                                                                                                                            |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F1 Async callbacks typed synchronously | **Open, partially fixed.** Native promises with non-void results, explicit sync/async unions and unknown results are handled; the structural-thenable case below remains unsound               |
| F2 Cross-realm promises not awaited    | **Fixed.** Callable-`then` detection and `Promise.resolve` await foreign promises; the real iframe regression requires completion count 2, sequential starts `[0, 1]`, and one leaked CPU slot |
| F3 Opposite backend changes cancel     | **Fixed.** Absolute deltas produce `leaked: 2` for CPU -1 / WASM +1, with both signs preserved in the summary; a release-only case reports 1                                                   |
| F4 Coverage count                      | **Fixed.** The plan now records 48 passed / 4 skipped, consistent with the test file and observed runs                                                                                         |

### F1 — medium: ordinary structural thenables are still typed synchronously

[check-leaks.ts](../../src/frontend/check-leaks.ts#L748), lines 748-754.

Runtime detection accepts any value with a callable `then`. The conditional return type instead
tests TypeScript's stricter `PromiseLike<unknown>` interface. An ordinary thenable need not return
another promise from its `then` method, so it can fail both type tests while still being awaited at
runtime.

This exact scratch reproduction type-checks, including the explicit `LeakReport` assignment:

```ts
const body = () => ({
  then(resolve: (value: number) => void) {
    queueMicrotask(() => resolve(123));
  },
});
const result = checkLeaks.steadyState(body);
const report: LeakReport = result;
```

At runtime `result` is a promise, and `report.summary` is `undefined`. A normal
`if (report.leaked > 0)` check therefore still silently bypasses leak reporting. The independent
scratch test asserted that `report.summary` was `"No leaks detected."`; it failed with `undefined`,
then awaited the result in `finally`. There are no casts or erased callback types in this
reproduction.

This is the same failure class as round 4 F1, now at the newly advertised thenable boundary.
[FEATURES.md](../../FEATURES.md) and the API comment explicitly promise thenable support, and the
generic callback signature accepts this object. The committed thenable test annotates its callback
as `PromiseLike<number>`, which bypasses the mismatch.

**Required:** align the public result type with the callable-`then` values the runtime actually
awaits, including structural thenables and relevant unions. Add a type regression using an ordinary
thenable whose `then` returns `void`, plus a runtime assertion after awaiting it. No implementation
fix was made by the reviewer.

## Regression strength and unchanged behavior

In the scratch clone, I reverse-applied only the checker source changes, leaving the new tests
intact. On the old source:

- `pnpm check` exits 2 with four failed type assertions covering ignored async results, a
  possibly-async result, an unknown result and the explicitly typed thenable.
- The API test file exits 1 with four failures: thenable sequencing, cross-realm completion,
  opposite-backend cancellation and release-only reporting.

Restoring `ea88cbf` makes those checks pass. These tests therefore prove the corrected capabilities,
not merely successful execution. The mixed sync/async cases exercise asynchronous warm-up and
asynchronous measurement separately; rejection tests verify propagation from either invocation. The
existing warm-cache, cache-held leak and active-session tests remain green.

No skip, tolerance, timeout, production test hook, cache flush or tracing-dependent disposal path is
added. Migration notes correctly explain awaiting ignored async results and the absolute-delta
report semantics, subject to the remaining thenable typing defect. The commit subject accurately
describes the changes; no version bump or unrelated implementation work is included.

## Plan status

All unchanged phase 1 item dispositions from round 4 remain in force:

| Item                                      | Disposition                                                                                                |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1 CI and push gate                        | Prior local implementation reviewed; hosted CI remains open                                                |
| 2 Setup integrity                         | Previously reviewed; normal fresh-clone prepare succeeds                                                   |
| 3 Missing backends / 4 Hidden failures    | Previously completed in phase 0; enforcement retained and default suite green                              |
| 5 Backend-init isolation                  | Previously signed off; unchanged                                                                           |
| 6 GPU errors and tracing                  | Regular tape tracing previously signed off; specialized dispatch tracing remains open                      |
| 7 Test inventory                          | Implemented cleanup previously accepted; issue links, timeout reasons and benchmark relocation remain open |
| 8 Steady-state leaks                      | F2-F4 fixed, F1 remains open; existing per-area suite integration also remains open                        |
| 9 Native trig contract                    | Previously accepted; unchanged                                                                             |
| 10 FD tolerances                          | QR checks previously accepted; complete derivations and CPU/WASM f64 coverage remain open                  |
| 11 Website opt-in work / 12 TTS benchmark | Open, outside this range                                                                                   |

No new deferral is approved. No churn tooling or statistics were run or updated.

| Phase exit criterion                                                | Status                                                                                  |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Hosted CI green on `wasm-multithreading`                            | Not verified                                                                            |
| Default and both GPU suites green, required backends enforced       | Default verified here; focused GPU tests pass; full GPU suites last verified in round 4 |
| Every skip/tolerance/timeout has a reason and issue                 | Unmet; unchanged                                                                        |
| Steady-state mode enabled for linalg, scan, foriLoop and transforms | Partial new coverage; remaining API defect and integration stay open                    |

## Sign-off

F2, F3 and F4 are signed off for `9a19712..ea88cbf`. F1 remains open for structural thenable typing,
so **item 8 and phase 1 are not signed off**. Round 4's acceptance of the item 7 cleanup and item 10
QR checks is unchanged. This report is left uncommitted for the implementation session.

---

## Implementation response (2026-11-09)

**Implementation claims for re-review, not reviewer findings or sign-off.** Base: `0891af9`; the
response is in the working tree, not yet committed. The maintainer-requested plan revision is also
uncommitted. The original review above is unchanged.

### F1: one completion contract

`checkLeaks.steadyState(body)` now always returns `Promise<LeakReport>`. A single `async` function
awaits warm-up, snapshots backend slot counts, awaits measurement, and reports the delta. This
deletes both the conditional result type and manual callable-`then` detection. Native await
semantics handle structural thenables and foreign promises without equating their runtime behavior
with TypeScript's stricter `PromiseLike` interface.

All repository callers now await the result, including the per-area helper and its test callbacks.
Synchronous throws from either run now reject the promise rather than escaping the call.
[FEATURES.md](../../FEATURES.md) and API/agent documentation describe this intentional change:
`const report = checkLeaks.steadyState(body)` becomes
`const report = await checkLeaks.steadyState(body)`, helper result types become
`Promise<LeakReport>`, and error handlers await inside `try`. Async ownership propagates through
callers so shared resources and the enclosing leak session outlive both runs.

The migration also identifies obsolete sync/async discrimination, `Promise.resolve` wrappers and
callback-result adapters. Cache handling, active-session behavior and per-backend delta semantics
are unchanged.

### Regression evidence

Before changing the checker source, ran the updated API tests against the implementation at
`0891af9`:

- `pnpm check`: six type assertion failures, including the review's ordinary structural thenable, a
  union with that thenable, and a callback whose declared `void` return erases its promise.
- `pnpm run test:cpu test/check-leaks.test.ts --reporter=dot`: 30 passed / 3 failed. The failures
  are the newly required promise result for a synchronous body and rejected synchronous throws from
  runs 1 and 2. The structural-thenable regression is a type failure on the old source, not a claim
  that the old runtime failed to await it.

With the new source, the structural-thenable case verifies sequential starts `[0, 1]`, completion
count 2, and one leaked CPU slot before cleaning up its retained arrays. The union cases exercise
both asynchronous warm-up and asynchronous measurement. Existing cross-realm, cache-held leak,
opposite-backend delta and async-rejection regressions remain intact.

One test-local `no-misused-promises` suppression deliberately creates the erased-`void` callback; it
models an input TypeScript accepts and is not a production lint relaxation. No skip, tolerance,
timeout, cache flush or production test hook was added.

### Validation and remaining gates

Validation ran in the current working tree, not a fresh clone:

| Check                                         | Result                                                               |
| --------------------------------------------- | -------------------------------------------------------------------- |
| `pnpm build`, `pnpm check`, `pnpm lint`       | Pass                                                                 |
| Generated ESM/CJS declarations                | Both declare `steadyState(body: () => unknown): Promise<LeakReport>` |
| Focused leak-check files, CPU/WASM            | 59 passed / 26 skipped                                               |
| Focused leak-check files, NVIDIA and Intel    | Each 81 passed / 4 skipped                                           |
| Full `pnpm run test:cpu`                      | 89 files passed / 3 skipped; 2287 tests passed / 1143 skipped        |
| `pnpm run test:push`, default                 | 92 files passed; 3412 tests passed / 18 skipped                      |
| Same gate, NVIDIA and Intel                   | Each 101 files passed; 3462 tests passed / 18 skipped                |
| Touched-file formatting and whitespace checks | Pass                                                                 |

The real local push-gate command exited 0 with WebGPU/WebGL required; it does not push commits. No
commit, push or hosted CI run is claimed. F1 awaits independent re-review; neither item 8 nor the
new phase 1A readiness gate is signed off. The remaining readiness requirements and phase 1B work
are retained in the plan. No phase 2 ownership implementation has begun.
