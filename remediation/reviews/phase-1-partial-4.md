# Review: phase 1 test integrity and steady-state leak checks, round 4

- **Range:** `9b880b5..e42a033` (`a86be1d`, `dcda0df`, `e42a033`).
- **Date:** 2026-11-09.
- **Scope:** item 7's partial test cleanup, item 8's new leak-checking API and coverage, and item
  10's QR tolerance changes.
- **Verdict:** **fail** for item 8: three independently reproduced correctness issues in the new
  API. The test cleanup and tighter QR checks pass.
- **Phase status:** phase 1 remains incomplete.
- **Reviewer changes:** no implementation fix. At the human's explicit request during this review,
  removed `scripts/history-churn.py` and its future reporting requirements from the plan. Existing
  statistics, graph and prior reports are unchanged. No new churn statistics are recorded.
- **Exclusion carried forward:** homepage content.

## Independent validation

Used a fresh local clone at `e42a033`, with `pnpm install --frozen-lockfile` and its normal build
and website generators, not linked dependencies. The scratch reproductions were removed before the
final clean-snapshot runs.

| Check                                        | Result                                                        |
| -------------------------------------------- | ------------------------------------------------------------- |
| `pnpm build`, `pnpm check`, `pnpm lint`      | Pass                                                          |
| Full default suite                           | 92 files passed; 3395 tests passed / 18 skipped               |
| Full NVIDIA suite                            | 101 files passed; 3445 tests passed / 18 skipped              |
| Full Intel suite                             | 101 files passed; 3445 tests passed / 18 skipped              |
| Full CPU/WASM suite                          | 89 files passed / 3 skipped; 2270 tests passed / 1143 skipped |
| Six affected files, CPU/WASM                 | 307 tests passed / 191 skipped                                |
| Empty selection, default and Intel configs   | Both exit 1, as required                                      |
| Three added scratch regressions, CPU/WASM    | All three fail for the reasons below; cleanup completes       |
| Type-check including the scratch regressions | Pass, including F1's incorrectly synchronous assignment       |

Both GPU runs enforce WebGPU/WebGL initialization and GPU error checks. The complete QR file passes
all 111 tests on each GPU. The new steady-state coverage actually reports **48 passed / 4 skipped**.

## Findings

### F1 — medium: asynchronous bodies can receive a synchronous return type

[check-leaks.ts](../../src/frontend/check-leaks.ts#L733), lines 733-734.

The asynchronous overload accepts only `Promise<void>`, but TypeScript's `() => void` overload
accepts callbacks returning any value, including `Promise<number>`. This compiles:

```ts
const report = checkLeaks.steadyState(async () => 123);
if (report.leaked > 0) throw new Error(report.summary);
```

The inferred type is `LeakReport`, but the runtime value is a promise. `report.leaked` and
`report.summary` are `undefined`, so this ordinary failure check silently passes. The scratch
regression explicitly assigns the result to `LeakReport`; `pnpm check` passes, and its assertion on
`summary` fails with `undefined`.

**Required:** make the return type match all accepted asynchronous callbacks, including callbacks
whose results are ignored and callbacks typed as possibly asynchronous. Add a focused type
regression as well as a runtime check.

### F2 — medium: valid promises from another realm are not awaited

[check-leaks.ts](../../src/frontend/check-leaks.ts#L742), lines 742-746.

Both branches use `instanceof Promise`. A promise constructed with a same-origin iframe's `Promise`
constructor satisfies the declared `Promise<void>` contract but fails that check. Consequently,
warm-up and measurement run concurrently and the checker returns a clean report before either
callback completes.

Reproduced in Chromium by creating an iframe, obtaining its constructor with
`Reflect.get(frame.contentWindow, "Promise")`, and returning a promise that allocates an array
inside `setTimeout` before resolving. Even after `await checkLeaks.steadyState(body)`, the
completion counter is **0**, not 2, and the report has already been produced. The scratch test
awaits both pending callbacks and disposes their arrays afterward.

**Required:** recognize and await supported promises independently of their realm, preserving
sequential warm-up and measurement. Cover the cross-realm case.

### F3 — medium: opposite backend changes cancel into a false clean report

[check-leaks.ts](../../src/frontend/check-leaks.ts#L712), lines 712-724.

The implementation sums signed deltas and uses only that total as the verdict. During the measured
run, a released CPU slot can cancel a newly leaked WASM slot. The already-collected per-device
evidence is discarded by the `"No leaks detected."` branch.

Reproduced with two pre-existing CPU arrays and this identical operation on each invocation:

```ts
const report = checkLeaks.steadyState(() => {
  inputs.pop()!.dispose();
  kept.push(np.array([3], { device: "wasm" }));
});
```

The second call leaks one additional WASM array and releases one CPU array. The checker returns
`leaked: 0` and `"No leaks detected."`. All retained arrays were disposed after the assertion.

**Required:** evaluate backend changes independently so that a nonzero change cannot be hidden by
another backend's opposite delta. Cover this cancellation case.

### F4 — low: the new coverage count is incorrect

Phase 1 item 8 in [the plan](../../REMEDIATION-PLAN.md) claims 44 passed / 4 skipped for
[steady-state-leaks.test.ts](../../test/steady-state-leaks.test.ts). There are 13 cases across four
backends: **48 passed / 4 skipped**, independently observed in the default and both GPU suites.
Update the count when recording the implementation response.

## Claims and test integrity

- The 36 changed WebGPU guards now explicitly skip unavailable coverage instead of silently passing.
  CPU/WASM validation exercises these paths; GPU validation still executes the tests.
- Five cost-model tests replace print-only diagnostics. The assertion-free timing test is removed;
  the existing tiled-matmul benchmark remains. Both configurations reject an empty selection.
- No timeout or tolerance is loosened. The QR JVP check now uses central differences and
  `atol: 1e-3, rtol: 0`; three QR gradient checks lose their WebGPU-specific allowances.
- The new API does not flush caches or reset an active tracking session. Its tests detect
  per-invocation allocations and retained JIT constants. Its ordinary same-realm `Promise<void>`
  case works, but does not establish the broader async claim (F1/F2).
- The new per-area tests assert slot balance with caches intact. They do not establish that the
  existing linalg/scan/foriLoop/transform suites run in steady-state mode; the plan honestly says
  that integration remains open.
- Commit subjects match their implementation content. The new API's commit includes a migration
  mapping and workaround signatures. No version bump, production test hook, tracing-dependent
  disposal guard or new error-swallowing ownership path was added.

## Phase 1 item disposition

| Item                     | Disposition                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| 1 CI and push gate       | Prior local implementation reviewed; hosted CI remains open and was not checked here                              |
| 2 Setup integrity        | Previously reviewed; normal fresh-clone prepare succeeds here                                                     |
| 3 Missing backends       | Previously completed in phase 0; required-backend enforcement retained                                            |
| 4 Hidden failures        | Previously completed in phase 0; all native suites pass here                                                      |
| 5 Backend-init isolation | Previously signed off; unchanged                                                                                  |
| 6 GPU errors and tracing | Previously signed off for regular tape dispatches; dedicated scan/block-map/scatter-add/copy tracing remains open |
| 7 Test inventory         | `a86be1d` cleanup verified; issue links, timeout reasons and remaining benchmark relocation stay open             |
| 8 Steady-state leaks     | `dcda0df` blocked by F1-F3; existing per-area suite integration also remains open                                 |
| 9 Native trig contract   | Previously reviewed and unchanged                                                                                 |
| 10 FD tolerances         | `a86be1d` tighter QR checks verified on both GPUs; full derivations and CPU/WASM f64 coverage remain open         |
| 11 Website opt-in work   | Open, outside this range                                                                                          |
| 12 TTS benchmark         | Open, outside this range                                                                                          |

No new deferral is approved by this review. The phase 4 worker-runtime work remains where the plan
already assigns it.

## Phase exit criteria

| Criterion                                                           | Status                                                        |
| ------------------------------------------------------------------- | ------------------------------------------------------------- |
| Hosted CI green on `wasm-multithreading`                            | Not verified                                                  |
| Default and both GPU suites green, required backends enforced       | Verified at `e42a033`                                         |
| Every skip/tolerance/timeout has a reason and issue                 | Unmet; item 7 remains partial                                 |
| Steady-state mode enabled for linalg, scan, foriLoop and transforms | Partial new coverage; API defects and integration remain open |

## Sign-off

Signed off only for the implemented test cleanup under item 7 and the tightened QR checks under item
10 in `9b880b5..e42a033`. **Item 8 is not signed off** until F1-F3 are fixed and re-reviewed. F4 is
a documentation correction. No phase 1 sign-off is given. The report is left uncommitted for the
implementation session, as required by the review protocol.
