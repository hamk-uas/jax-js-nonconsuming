# Review: phase 0, round 1

- **Range:** `fe2c9cc..a32d106` (homepage change excluded at the human's request)
- **Reviewer:** separate agent session (`c70303a7`), no editing role; report pasted by the human on
  2026-11-08 and committed below by the implementation session, wording unchanged (headings demoted
  one level, lines reflowed by Prettier)
- **Verdict:** fail — "not ready to close Phase 0"
- **Human sign-off:** none (the phase is not done)

---

## Reviewer report

### Verdict: good progress, but not ready to close Phase 0

Reviewed `fe2c9cc..a32d106`, excluding the homepage as requested. No code changed.

### Findings

1. **P1 — New GPU-selection regression.**  
   `chromiumEnv` selects hardware by whether a driver manifest exists. Mesa installs Intel manifests
   on AMD-only machines too. With Intel and Radeon manifests present, I reproduced selection of
   **Intel**, excluding Radeon and potentially restoring the silent GPU skips this remediation
   targets. Selection must distinguish installed drivers from available hardware.

2. **P1 — Passing tests still accept invalid GPU work (known R21).**  
   `startProfiling` requests 8,192 queries against the 4,096 limit. Both GPU runs logged invalid
   QuerySets and command buffers without failing those tests. The tracing-conflict fix is valid, but
   “green” still does not certify profiling or profiling-based dispatch tests.

3. **P2 — The clean-checkout exit criterion remains unmet (known R16).**  
   In a fresh checkout with dependencies linked, build/check/lint passed, but tests produced **two
   failed suites and five failed tests** because `website/tsconfig.json` requires generated Svelte
   configuration. Running `svelte-kit sync` made all **33 affected tests pass**. Pull that
   prerequisite forward, or explicitly revise the Phase 0 gate.

### Verification

| Check                           | Result                                 |
| ------------------------------- | -------------------------------------- |
| Build, type-check, lint         | Passed                                 |
| Prepared-worktree default suite | 3,312 passed, 14 skipped               |
| NVIDIA full suite               | 3,365 passed, 14 skipped               |
| Intel full suite                | 3,361 passed, 14 skipped, **4 failed** |

Intel’s failures match the recorded R24 transcendental-accuracy issue; they are not new regressions.

### What looks sound

- The quarantine patch applies against its documented base.
- The WASM triangular-solve fix removes duplicated parameters and exercises the actual compiled-loop
  bug.
- The small-carry foriLoop fix passes on both GPUs.
- Opt-in calibration, sequential probes, and meaningful cache assertions improve the previous
  design.

Rework metrics show **zero recent-source rework** across four source-changing commits—encouraging,
but too small a sample to establish a trend.

**Next priority:** correct GPU selection, make missing backends/errors fail tests, and resolve the
clean-checkout prerequisite before sign-off.

---

## Implementation response (claims for the re-review to verify)

Written by the implementation session. Per the review process these are claims, not evidence; the
re-review checks only these findings.

The human decided to pull two phase 1 items into phase 0, as the reviewer recommended: required
backends fail loudly (phase 1 item 3) and WebGPU errors fail tests (phase 1 item 6).

| Finding                         | Response                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Commit    |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| 1. GPU selection by manifest    | Confirmed: `mesa-vulkan-drivers` owns `intel_icd.json` and `radeon_icd.json`. `detectGpu()` now requires the PCI vendor in `/sys/class/drm/cardN/device/vendor` and the driver manifest. A detected GPU's WebGPU and WebGL must initialize or every test file fails. To check: forcing the Radeon driver on a machine without AMD hardware fails with "Required backend(s) did not initialize" (passed with 12 skips before)                                                                                     | `43b61eb` |
| 2. Invalid GPU work accepted    | Every test runs inside WebGPU error scopes; errors outside tests fail the next test or the file. QuerySet holds 4096 queries (2048 passes). To check: reverting the QuerySet change fails 5 of 7 `test/profile-gpu.test.ts` tests, including "measures nonzero pass durations" and "records at most 2048 passes, then reports truncation"                                                                                                                                                                        | `82bb49d` |
| 3. Clean-checkout tests failing | Not reproduced with `pnpm install`: the reviewer's checkout linked `node_modules`, so the `prepare` scripts that run `svelte-kit sync` and build `dist/` never ran. A fresh `git clone` at `a32d106` and again at `82bb49d`, then `pnpm install --offline --frozen-lockfile && pnpm build && pnpm check && pnpm lint && pnpm test`: all green (`82bb49d`: 3314 passed, 14 skipped). The exit criterion and review checklist item 3 now name this procedure; R16 is corrected and keeps the silent `\|\| echo ''` | plan      |

Runs at `82bb49d` on this machine: default suite 3314 passed, 14 skipped (6 WebGL-specific skips, 8
orchestrator-worker skips, none from a missing backend); NVIDIA suite 3367 passed; Intel suite 3363
passed, 4 failed (R24, unchanged). No WebGPU error logged in any run.
