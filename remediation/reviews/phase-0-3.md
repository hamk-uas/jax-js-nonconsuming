# Review: phase 0, round 3

- **Range:** `e0c0437..4e7a27e`.
- **Scope:** F2.1 only, and the implementation response in the round-2 report. Previously resolved
  findings and the homepage were not reopened.
- **Reviewer:** the separate, non-implementing review session (`c70303a7`).
- **Date:** 2026-11-08.
- **Verdict:** pass -- F2.1 is resolved.
- **Human sign-off:** given 2026-11-08; see [Human sign-off](#human-sign-off) at the end.

This passing review does not itself mark Phase 0 done. The plan still requires the report to be
committed and the human's sign-off. Only this report was added to the original repository; all
reproduction fixtures and mutations were confined to a disposable clone.

## Exit criteria and claims

| Criterion or claim                             | Result                | Evidence                                                                                                                                                                                                       |
| ---------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reviewed working tree clean                    | Verified              | `git status --short` was empty at `4e7a27e`, before writing this report.                                                                                                                                       |
| Fresh clone with dependency installation       | Verified              | `pnpm install --offline --frozen-lockfile` passed and ran the root build and website prepare generators; no linked dependencies.                                                                               |
| Build, type-check and lint                     | Verified              | `pnpm build && pnpm check && pnpm lint` passed in the fresh installed clone.                                                                                                                                   |
| Default suite green                            | Verified              | `pnpm test`: 92 files passed, 3316 tests passed, 14 skipped.                                                                                                                                                   |
| Listener attached before module initialization | Verified              | [setup.ts](../../test/setup.ts) calls `captureErrorsOfNewDevices()` before `init`; [webgpu-errors.ts](../../test/webgpu-errors.ts) attaches the listener before returning each requested device to its caller. |
| Module-load GPU error fails the next test      | Verified on both GPUs | The round-2 reproduction now fails with `(raised before this test) Query count (8192) exceeds the maximum query count (4096)`.                                                                                 |
| GPU error after the last test fails the file   | Verified on both GPUs | A temporary file creating an invalid occlusion QuerySet in its own `afterAll` failed the file through the harness's end-of-file check.                                                                         |
| Expected-error helper requires an actual error | Verified on both GPUs | The two committed harness tests pass; a temporary test declaring an error without provoking it fails with `expected a WebGPU error matching ..., but none occurred`.                                           |
| Human sign-off                                 | Pending               | Not supplied or inferred by the reviewer.                                                                                                                                                                      |

## F2.1: resolved

The original defect was a listener first installed in `beforeEach`, after the test module's
top-level code had already executed. The fix moves attachment to device creation:
`GPUAdapter.prototype.requestDevice` is wrapped in test setup, before backend initialization. The
wrapper installs the listener before exposing the device to the requesting code.

The round-2 reproduction was rerun without declaring the error as expected: an invalid timestamp
QuerySet was created during module loading, the queue was flushed, and the actual validation-error
event was awaited. The ordinary test assertion still succeeded, but the harness correctly failed the
test on both NVIDIA and Intel. It no longer merely logs the error and returns success.

The end-of-file check also makes devices process work before draining the error list. This catches
validation errors queued by a file's own trailing `afterAll` hook, which otherwise can remain
undelivered when the harness checks the list.

### Negative controls

All controls were run on both GPUs, using temporary fixtures or mutations in the clone:

| Control                                                        | Observed result                                                                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Original undeclared module-load error                          | Test fails with the correctly attributed GPU error.                                                               |
| Undeclared error in the file's `afterAll`                      | File fails with `WebGPU errors after the last test`.                                                              |
| `expectWebGpuError()` declaration with no actual error         | Test fails on the missing expectation.                                                                            |
| Remove only the early `captureErrorsOfNewDevices()` call       | The committed module-load regression test fails on its unmet `/8192/` expectation; the in-test case still passes. |
| Restore early capture, remove only the end-of-file queue flush | The trailing-error fixture incorrectly passes, establishing that the flush is necessary.                          |

These checks establish both that undeclared errors fail and that the committed tests detect removal
of the relevant fix. The new helper is not a blanket error suppression: its declared pattern must
actually occur.

No further finding was identified within the authorized F2.1 scope.

## Full-suite results

Full suites ran on unmodified `4e7a27e` after installation, before adding fixtures.

| Suite      | Passed | Skipped | Failed |
| ---------- | ------ | ------- | ------ |
| Default    | 3316   | 14      | 0      |
| NVIDIA GPU | 3369   | 14      | 0      |
| Intel GPU  | 3365   | 14      | 4      |

The Intel failures remain the four recorded R24 accuracy failures in
[numpy.test.ts](../../test/numpy.test.ts), at lines 2475 and 2505. They are unchanged Phase 1 debt,
not new failures caused by F2.1.

Each full run logged exactly one deliberate invalid-QuerySet error from the new
[harness test](../../test/webgpu-errors.test.ts), as expected. The dual-GPU command therefore logged
two deliberate errors in total; no additional uncaptured WebGPU error was reported.

## Rework and conclusion

The existing history-churn collection/measurement functions report two commits in `e0c0437..4e7a27e`
and no changes in their measured source scope. Source rework is zero; corrective share and
follow-ups/change are not informative because this range changes tests, configuration-related code
and documentation rather than `src/`. Plotting dependencies were unavailable, so no figure was
generated.

F2.1 is closed by this review. Within the agreed re-review scope, the fixes and their capability
tests pass. Phase 0 is ready for the human's sign-off after this report is committed; no
implementation change or phase-status edit was made by the reviewer.

---

## Human sign-off

Phase 0 is signed off as done by the human (@OlliNiemitalo) on 2026-11-08, answering the
implementation session's sign-off question with "Sign off Phase 0 as done" (no note). The
implementation session then pointed out that two salvage candidates of phase 0 item 3 were still
undecided, which none of the rounds had checked because the exit criteria do not name them. The
human deferred the WASM QR routine to phase 3 and the `clearCaches()` WASM-cache question to phase
6, and confirmed the sign-off stands. Recorded by the implementation session; the reviewer's text
above is unchanged except for the sign-off line in the header.

Covered: `fe2c9cc..4e7a27e`, reviewed in rounds [1](phase-0-1.md), [2](phase-0-2.md) and 3. Carried
to phase 1 as recorded in the plan: R23 (command-tape tracing) and R24 (four Intel accuracy
failures). Not covered, for the next review: `a32d106` (homepage benchmark), excluded from all three
rounds at the human's request; `b10c74c` (history-churn review mode) and `bd4540a` (this sign-off
and the plan changes with it), made after this round.
