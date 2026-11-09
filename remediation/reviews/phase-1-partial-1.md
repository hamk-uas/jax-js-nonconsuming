# Review: phase 1 partial implementation, round 1

- **Range:** `3a2aaea..0b3c72a`.
- **Date:** 2026-11-09.
- **Reviewer:** the separate, non-implementing review session (`c70303a7`).
- **Scope:** the three commits since the last review: recording the dependency re-review, CPU/WASM
  CI and the multi-GPU pre-push gate, and setup/source-type-check integrity.
- **Verdict:** pass within this incremental scope; no new finding.
- **Phase status:** Phase 1 remains incomplete. This is not a phase exit review or human sign-off.
- **Exclusion:** homepage content remains excluded at the human's request.

## Findings and limitations

No new correctness regression or test-integrity defect was identified in this range.

The real pre-push command correctly exits **1**, rather than permitting a push, because the four
previously recorded Intel R24 accuracy failures remain. Default and NVIDIA suites pass. This is the
intended enforcement of existing debt, not a new failure caused by the gate.

Hosted CI is **not verified**. The workflow exists and its local checks pass, but `gh` has no
authentication in this environment; the alternative GitHub Actions API lookup returned 404. Neither
establishes a green hosted run, and the implementation does not claim one.

AMD detection is covered by the new runner tests and the AMD configuration uses the shared factory,
but no AMD hardware was available. Actual GPU validation covered NVIDIA and Intel.

## Phase 1 item dispositions

These are incremental dispositions against the [plan](../../REMEDIATION-PLAN.md), not a claim that
the remaining items were implemented or fully reviewed.

| Item                                                                  | Disposition                                                   | Evidence                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1: push/PR CPU/WASM workflow                                          | Implemented in `c1faa1d`; hosted result open                  | [checks.yaml](../../.github/workflows/checks.yaml) selects all pushes and PRs, installs Chromium, and runs build/check/lint, plugin and gate tests, and the browser CPU/WASM suite. Those commands pass locally.                                                                                     |
| 1: full local pre-push gate                                           | Done in `c1faa1d` within the supported hardware scope         | [prepush.ts](../../scripts/prepush.ts) runs default, then every detected supported GPU profile. The real command ran NVIDIA and Intel, enforced WebGPU/WebGL, and blocked on Intel. Five runner tests also pass, including CPU fallback, earlier failures followed by later suites, and termination. |
| 2: website prepare failure                                            | Done in `0b3c72a`                                             | Actual prepare succeeds normally, fails on invalid configuration, and no longer converts that failure to success.                                                                                                                                                                                    |
| 2: benchmark source type-checking                                     | Done in `0b3c72a`                                             | [tsconfig.json](../../tsconfig.json) resolves the root package to source; the compiler includes all 16 benchmark files and zero workspace `dist/` declarations. Check passes with root `dist/` absent.                                                                                               |
| 2: diagnostic cleanup without losing assertions                       | Done in `0b3c72a`                                             | The six size/direction cases are consolidated in [lax-associative-scan.test.ts](../../test/lax-associative-scan.test.ts). Both GPU runs pass all 65 tests. The two N=65 cases were duplicates; the stale `debug-wgsl` exclusion is removed.                                                          |
| 3: missing backends fail loudly                                       | Done previously in `43b61eb`, carried forward                 | Required-backend setup remains active; all three real GPU-enabled runs require WebGPU/WebGL. Earlier phase-0 review evidence is not reopened.                                                                                                                                                        |
| 4: R19 failures                                                       | Done previously in `8d7d5b9`, `fb8ea22`, `d2534a2`, `8fbf8c8` | The plan records the constituent fixes. Current default/NVIDIA runs remain green; only the separately recorded R24 Intel failures remain.                                                                                                                                                            |
| 5: independent backend initialization and injected WebGL-failure test | Open                                                          | Not implemented by this range.                                                                                                                                                                                                                                                                       |
| 6: WebGPU error capture                                               | Done previously in `82bb49d` and `648d367`                    | Carried forward from the passing [phase-0 review](phase-0-3.md); no error suppression is added here.                                                                                                                                                                                                 |
| 6: command-tape tracing and JIT trace assertion                       | Open                                                          | Not implemented by this range.                                                                                                                                                                                                                                                                       |
| 7: skip/return/tolerance/timeout/exclusion inventory and dispositions | Open                                                          | This range cleans up the named diagnostics, not the complete inventory.                                                                                                                                                                                                                              |
| 8: steady-state leak mode                                             | Open                                                          | Not implemented by this range.                                                                                                                                                                                                                                                                       |
| 9: R24 implementation/tolerance decision                              | Open                                                          | All four existing Intel failures remain visible and block the real gate.                                                                                                                                                                                                                             |
| 10: conditioning-derived finite-difference tolerances                 | Open                                                          | Not implemented by this range.                                                                                                                                                                                                                                                                       |
| 11: remaining website pages start GPU work only on request            | Open                                                          | Not implemented by this range; homepage remains outside this review.                                                                                                                                                                                                                                 |
| 12: TTS per-step benchmark using the website model                    | Open                                                          | Not implemented by this range.                                                                                                                                                                                                                                                                       |

No new deferral or human decision is inferred.

## Phase exit criteria

| Criterion                                                                | Result                          | Evidence                                                                                                       |
| ------------------------------------------------------------------------ | ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| CI green on `wasm-multithreading`                                        | Not verified                    | Local workflow-equivalent checks pass; no hosted run could be verified.                                        |
| Default suite green, required backends enforced                          | Verified                        | Real pre-push default run: 3326 passed, 14 skipped; required WebGPU/WebGL.                                     |
| Both GPU suites green, required backends enforced                        | Not met                         | NVIDIA: 3373 passed, 14 skipped. Intel: 3369 passed, 14 skipped, four R24 failures; both require WebGPU/WebGL. |
| Every skip/tolerance/timeout has a reason and issue                      | Not verified; item remains open | This range does not perform the required repository-wide inventory.                                            |
| Steady-state leak mode enabled for linalg, scan, foriLoop and transforms | Not met                         | Item 8 remains unimplemented.                                                                                  |

## Independent validation

All commands ran in a fresh, independent clone detached at `0b3c72a`, after
`pnpm install --offline --frozen-lockfile` completed its normal root-build and website-prepare
lifecycle. Dependencies were not linked from the original worktree.

| Check                                                       | Result                                                              |
| ----------------------------------------------------------- | ------------------------------------------------------------------- |
| `pnpm build`, `pnpm check`, `pnpm lint`                     | Pass                                                                |
| `pnpm run test:eslint-plugin`                               | 106 passed                                                          |
| `pnpm run test:push-gate`                                   | 5 passed                                                            |
| `pnpm run test:cpu`                                         | 89 files passed, 3 skipped; 2265 tests passed, 1075 skipped         |
| `pnpm run test:push`: full default suite                    | 92 files passed; 3326 tests passed, 14 skipped                      |
| Same command: full NVIDIA suite                             | 101 files passed; 3373 tests passed, 14 skipped                     |
| Same command: full Intel suite                              | 100 files passed, 1 failed; 3369 tests passed, 14 skipped, 4 failed |
| Complete real pre-push command                              | Exit 1 after running all three suites                               |
| `scripts/gpu-test.sh run test/lax-associative-scan.test.ts` | 65 passed, none skipped, on each GPU                                |
| Website `prepare` and `check`                               | Pass; Svelte check reports zero errors and warnings                 |
| `tsc --listFilesOnly`                                       | Includes 16 benchmark files; zero workspace `dist/` declarations    |

The four Intel failures are the same WebGPU/WebGL pairs at
[numpy.test.ts](../../test/numpy.test.ts), lines 2475 and 2505: asin/acos difference `8.5848e-5` and
atan2 difference `6.5128e-6`, against `5e-6`. No tolerance or suppression was changed to accommodate
them.

### Negative and mutation controls

Controls affected only the disposable clone and were restored afterward.

1. **Real website sync failure:** `BASE_PATH=review-invalid pnpm run prepare` exits 1 with
   `BASE_PATH must be empty or start with "/"`. Running the old `svelte-kit sync || echo ''` recipe
   under the same configuration exits 0 despite the same error. Normal prepare passes again after
   removing the invalid environment setting.
2. **No stale build dependency:** move root `dist/` aside; `pnpm check` still passes. Revert only
   the package alias to its old `"."` target: check exits 2, including TS2307 errors from the
   benchmarks. Restore the source alias: check passes with `dist/` still absent. Restore `dist/`.
3. **CPU-only means no initialized GPU:** retain CPU-mode setup but temporarily substitute the
   normal NVIDIA launch options. The selected file fails before collection with
   `CPU-only test run initialized GPU backend(s): webgl, webgpu`. Restore the disabling flags: the
   same CPU-mode file passes, with GPU cases reported as skipped.
4. **Gate failure handling:** the committed runner tests inject failures into default, NVIDIA and
   Intel independently, verify later suites still run, and verify that failures and signals produce
   nonzero results. The actual Intel failure independently confirms the real command's nonzero exit.

### Coverage and test-integrity changes

The CPU profile intentionally disables both hardware and software GPU backends, requires CPU/WASM,
and explicitly rejects accidental GPU initialization. Its skipped GPU cases are not a substitute for
the separate, enforced hardware runs.

The scan consolidation adds N=128 and N=200 in both directions to the main regression, retains the
N=65 pair, and removes the duplicated diagnostic file. Its numerical assertion, existing
availability check and tolerance are unchanged. Consequently the default count rises by four,
whereas the full GPU count drops by two duplicate cases. CPU skips rise by four; hardware-suite
skips remain 14. No new timeout override or permissive numerical threshold was introduced.

Commit subjects match the changes; there is no version bump or WIP commit in the reviewed range. The
first commit records the already reviewed dependency fix, rather than introducing a new one.

## Rework and conclusion

`python3 -S scripts/history-churn.py --since 3a2aaea` passed without plotting dependencies. Both
scopes contain three commits. `src` has zero measured new lines; `code` has 263, with
ratio-within-20 `0.00`, ratio-within-100 `0.03`, follow-ups/change `0.00`, and corrective share
`0.00`. No recent rework pairs are listed. Recent follow-ups remain right-censored, and these scopes
exclude documentation and package manifests.

The incremental changes pass review. Phase 1 cannot be signed off: hosted CI is unverified, Intel
still fails as expected, and the remaining phase items and exit criteria are open.

Only this report was added to the original repository. No implementation or phase-status file was
edited. The report is left uncommitted for the implementation session to commit unchanged.
