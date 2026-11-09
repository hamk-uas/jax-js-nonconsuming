# Review: phase 1 backend-init isolation (R20) and JIT tape tracing (R23), round 3

- **Range:** `c6d3b05..f532bc9` (one commit, `f532bc9`).
- **Date:** 2026-11-09.
- **Reviewer:** separate, non-implementing review session. It changed no tracked file; the only
  writes were scratch files under `tmp/` and this report.
- **Scope:** items 5 and 6 of phase 1 (R20 backend initialization, R23 command-tape tracing) and the
  "current continuation validation" paragraph of the plan.
- **Verdict:** **pass with findings.** Every completion claim in the range is verified, and both new
  regressions fail without the change. Two low findings; no blocking one.
- **Phase status:** phase 1 remains incomplete (hosted CI, items 7, 8, 10-12 and the exit criteria
  are outside this range). **This is not human sign-off.**
- **Exclusion:** homepage content remains excluded.

## Verification (fresh clone of `f532bc9`, `pnpm install --frozen-lockfile`)

| Check                                   | Result                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `pnpm run check`                        | pass                                                                                             |
| `pnpm run lint`                         | pass, no warnings                                                                                |
| CPU/WASM (`JAX_TEST_CPU_ONLY=1`)        | 2270 passed / 1082 skipped (claim: same)                                                         |
| Default suite (`vitest run`)            | 92 files, 3338 passed / 14 skipped (claim: same)                                                 |
| NVIDIA suite (`GPU=nvidia gpu-test.sh`) | 101 files, 3388 passed / 14 skipped (claim: same)                                                |
| Intel suite (`GPU=intel gpu-test.sh`)   | 101 files, 3388 passed / 14 skipped (claim: same)                                                |
| Focused "22 tests" claim                | `devices`, `command-tape-gpu`, `profile-gpu`, `with-batch-command-tape-gpu`: 22 passed on NVIDIA |

Not verified: hosted CI (outside the range).

## Plan items

| Item                                                       | Status                                                                                                                                                            |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5 Decouple backend init (R20)                              | Implemented and verified; correctly still "pending review" in the plan, not marked done                                                                           |
| 6 Trace the command tape (R23), test a `jit` yields traces | Implemented and verified for regular kernel/routine dispatches; scan, block-map, scatter-add and copy paths stay untraced, and both plan and `FEATURES.md` say so |

## Evidence that the tests prove the capability (checklist 4 and 5)

I restored `src/` to `c6d3b05` in the scratch clone and ran the new tests on NVIDIA and Intel:

- The three R20 cases fail: the interference is reproduced (`overlap: true`, WebGPU missing from
  `available`), and a throwing WebGL context rejects `init` with the injected error.
- The three tape-tracing cases fail with 0 entries (expected 1, 1 and 2049).
- `profile-gpu.test.ts`'s added tape call passes on the old source, which has no tape tracing, so it
  checks nothing there. It does check the new code: removing the `this.#profiling ?` guard in the
  tape dispatch makes "kernels dispatched while profiling are not traced" fail. Valid.

New hooks and catch blocks: the worker patches `navigator.gpu.requestAdapter`, `OffscreenCanvas`'s
`getContext` and `console.error` only inside its own worker, which `finally` terminates. The tests
assert the path that ran (adapter request count, overlap flag, trace names, pass counts, 2049
entries, nonzero duration), not merely that nothing threw. No skip, tolerance or timeout was added
or loosened; the three new skips are `skipIf(!webgpu)` and are reported, as the plan says. The
`catch` in `init` logs and continues by design (R20); see F2.

## Findings

**F1 (low, fixed at the user's request after review): dead code left behind by the tracing
rewrite.** `onFlushTrace`, `flushCallbacks` and the flush loop in `stopTrace()`
([tracing.ts](../../src/tracing.ts), lines 9, 28-30 and 37-40) have no caller any more:
`grep -rn onFlushTrace` finds only its definition. The comment "Establish baseline on the flush" in
`readTracingBatch` ([webgpu/tracing.ts](../../src/backend/webgpu/tracing.ts)) likewise describes a
removed flush. The AEP cleanup rule says to remove deprecated paths and legacy comments in the same
change. Not a correctness issue.

**F2 (low, observation; README now states the hang caveat): `init()` now cannot distinguish
"unavailable" from "broken".** An unexpected creation error, including a failed `requestDevice`,
only goes through `console.error` and the backend is omitted from the result, exactly like normal
unavailability. That is the documented R20 contract and the tests rely on it, and principle 3 is met
in the test harness because `test/setup.ts` fails a file when a required backend is missing. Library
users who do not check the returned list will see only a log line. Serialization also means a
backend whose adapter request never settles blocks every later `init()` call, where before only that
backend hung. Neither needs a change in this range; decide whether either belongs in the README
note.

Smaller notes, no action required:

- The commit combines two plan items (R20 and R23) in one commit. Its subject names both, so it is
  honest, but it is larger than principle 7 prefers.
- Each traced eager dispatch now owns a 4096-query set and two 32 KiB buffers, created and destroyed
  per encoder; this costs only while tracing is on.
- (Fixed after review: `pipelineSubmit` now calls `discardTracingEncoder` on failure.) Failure
  inside `pipelineSubmit` for an own encoder (shape mismatch thrown for a later shader in
  `pipelines`) does not call `discardTracingEncoder`, so the query set of an earlier shader waits
  for GC. The plan's claim covers abandoned tape encoding only, which is accurate.

## Documentation claims checked against code

- README: init serialization, logging, omission of unavailable backends: matches `init` in
  [backend.ts](../../src/backend.ts) and the tests.
- `FEATURES.md`: tape and eager tracing, one measurement per shader across passes, `withBatch`
  deferral, no truncation across query sets, resolution in the writing encoder, uncovered paths,
  `profileGpu` precedence: each matches [tracing.ts](../../src/backend/webgpu/tracing.ts) and
  [webgpu.ts](../../src/backend/webgpu.ts) and a passing test.
- Plan: the numbers in the "current continuation validation" paragraph are the measured ones above.

## Rework (`python3 scripts/history-churn.py --since c6d3b05`)

| Scope  | Commits | New lines | Redoes earlier commits (≤20) | Corrective share |
| ------ | ------- | --------- | ---------------------------- | ---------------- |
| `src`  | 1       | 142       | none                         | 0.00             |
| `code` | 1       | 346       | none                         | 0.00             |

A single commit gives little rework signal, and follow-ups of the newest commits are not yet
observable.

## Deferrals and uncovered commits

None deferred by this range. No commit is left uncovered between `c6d3b05` and `f532bc9`.

## Human sign-off

Signed off by the human on 2026-11-09 for items 5 (R20) and 6 (R23, regular kernel/routine tape
dispatches) of phase 1, covering `c6d3b05..f532bc9` plus the review-fix commit `5ebaad7`. F1 is
fixed; F2 is accepted as the R20 contract and documented in the README. The dedicated scan,
block-map, scatter-add and copy dispatch tracing stays open. This is not a phase 1 sign-off.
