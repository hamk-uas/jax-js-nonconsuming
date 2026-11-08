# Review: dependency updates and post-Phase-0 work

- **Range:** `4e7a27e..5a2c612` (12 commits).
- **Date:** 2026-11-08.
- **Reviewer:** the separate, non-implementing review session (`c70303a7`).
- **Verdict:** fail -- one new P2 workflow regression requires correction.
- **Scope:** all changes since the previous review, including dependency migrations, benchmark
  migration, JIT cache correction, count-only leak checking, the rework script and plan updates.
  Homepage content remains excluded at the human's request.
- **Phase status:** this reviews the work between phases 0 and 1; it neither reopens the recorded
  Phase 0 sign-off nor signs off Phase 1.

## Finding

### F1 -- P2: the scan-disposal website lint command no longer loads its plugin

**Location:** [package.json](../../package.json), line 84; introduced by the ESLint 10 migration in
`5ddd07d`.

ESLint 10 finds configuration starting from each target file. Consequently,
`lint:scan-dispose:website` now uses [website/eslint.config.js](../../website/eslint.config.js),
which does not register `jax-js`, instead of the root ownership-rule configuration. The CLI's
`jax-js/require-scan-result-dispose:error` override cannot resolve its plugin.

In the freshly installed clone:

```text
pnpm run lint:scan-dispose:website

ESLint: 10.14.0
A configuration object specifies rule "jax-js/require-scan-result-dispose",
but could not find plugin "jax-js".
ELIFECYCLE Command failed with exit code 2.
```

The same migration correctly added `--config eslint.config.ts` to the neighboring
`lint:ownership:website` script, but omitted this script. Adding that flag to the scan-disposal
command **at invocation time only**, without editing implementation, was an independent control: it
loaded the ownership rules and exited 0 (94 warnings, no errors). This isolates configuration
discovery as the cause.

**Impact:** the dedicated website scan-result-disposal check aborts before linting. Normal root lint
and the narrower website ownership gate do not make this command functional.

**Requested correction:** apply the same explicit root-config selection to this script and rerun
both website ownership commands. No suppression or rule removal is necessary.

## Claims and work-item disposition

| Work                                                            | Disposition                             | Independent evidence                                                                                                                                   |
| --------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 0 sign-off and salvage decisions                          | Recorded, not supplied by this reviewer | The appended human sign-off in [phase-0-3.md](phase-0-3.md) and plan agree: QR is deferred to phase 3; WASM cache clearing to phase 6.                 |
| Python-only, read-only review mode (`b10c74c`)                  | Verified                                | `python3 -S scripts/history-churn.py --since 4e7a27e` succeeds in both scopes; no worktree changes.                                                    |
| Within-range updates and tsdown migration (`a31d05b`)           | Verified                                | Fresh install/build/type-check pass; packed ESM and CJS both execute WASM JIT correctly.                                                               |
| Conv rewrite cache setting (`fd120a2`)                          | Verified                                | Committed regression passes on all four backends; removing the key suffix in the clone makes all four cases fail on both GPU runs.                     |
| Count-only leak sessions (`da649aa`)                            | Verified                                | Both focused tests pass; a temporary benchmark with a live slot-backed array fails on the slot-count assertion without creation-site tracking.         |
| Vitest 5 / Playwright and benchmark migration (`c856872`)       | Verified for exercised cases            | Full correctness suites, seven multi-task benchmark files under default config, and representative WASM/WebGPU comparisons under both GPU configs run. |
| ESLint 10 and lint stack (`5ddd07d`)                            | Partially verified; F1 open             | Root lint and 106 plugin tests pass; one dedicated lint command fails as described above.                                                              |
| TypeScript 6 (`a17d87e`)                                        | Verified                                | Root `pnpm check` and website type-check pass against the newly built declarations.                                                                    |
| SvelteKit 3 / Vite 8 / Monaco and website migration (`5a2c612`) | Verified for exercised behavior         | Website check/build/smoke pass; deployed-base REPL loads Monaco, runs the WASM example, and prints correct results.                                    |
| Newly recorded R26--R29 debt                                    | Open, not claimed fixed                 | TTS benchmark removal is explicit; remaining compiler-state cache keys, WebGPU conv performance and typedoc output ownership remain in the plan.       |
| Phase 1 completion                                              | Not claimed or verified                 | CI, full skip/tolerance inventory and remaining Phase 1 items are outside this completion claim.                                                       |

## Validation

All baseline checks ran in an independent, detached clone at `5a2c612`, after
`pnpm install --offline --frozen-lockfile` ran the normal prepare lifecycle. No linked dependencies
or implementation edits were used for baseline validation.

| Check                                                                         | Result                                                                    |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `pnpm build`, `pnpm check`, `pnpm lint`                                       | Pass                                                                      |
| `pnpm vitest run`                                                             | 92 files; 3322 passed, 14 skipped                                         |
| NVIDIA full suite                                                             | 102 files; 3375 passed, 14 skipped                                        |
| Intel full suite                                                              | 101 files passed, 1 failed; 3371 passed, 14 skipped, 4 known R24 failures |
| `pnpm run test:eslint-plugin`                                                 | 106 passed                                                                |
| Website check / lint                                                          | 0 type errors; lint passes with 30 warnings                               |
| Website build and smoke with `GITHUB_REPOSITORY=hamk-uas/jax-js-nonconsuming` | Pass                                                                      |
| Packed ESM and CJS WASM JIT                                                   | Both return `[2, 4, 6]`                                                   |
| Seven migrated benchmark files under default configuration                    | 7 files, 10 tests passed                                                  |
| Selected associative-scan, argreduce and conv A/B benchmarks on each GPU      | 3 tests passed per GPU; 25 deliberately excluded by the name selector     |

The Intel failures are unchanged `asin`/`acos` and `atan2` accuracy failures on WebGPU and WebGL in
[numpy.test.ts](../../test/numpy.test.ts), lines 2475 and 2505. They are existing R24 debt, not
dependency-update regressions.

### Capability checks

- Removing only the conv rewrite setting from the compiled-program cache key makes the new
  regression fail with `expected true to be false`; restoring it makes all four backend cases pass
  on both GPU runs.
- The migrated conv A/B reports `rewritten=true` and `rewritten=false` for its respective settings.
  The sampled 32-channel case reproduces R28: block-map averages 6.60 ms versus 2.19 ms generic-dot
  on NVIDIA, and 14.95 ms versus 2.52 ms on Intel. This is recorded existing debt, not a new
  slowdown introduced by these changes.
- Temporary benchmark tests independently fail on an intentional one-slot leak and an invalid
  occlusion QuerySet on both GPUs. Thus the migration still applies both leak and GPU-error checks
  to benchmark tests.
- The built REPL at `/jax-js-nonconsuming/repl` loads the editor and executes the WASM example:
  `5.477226` and `[0.1825742, 0.3651484, 0.5477226, 0.7302967]`. Its root link and the 404 page's
  return-home link retain the deployment prefix.

The full benchmark matrix, downloaded-model demos, live deployment and generated typedoc output were
not validated. The seven-file benchmark run and selected dual-GPU comparisons are capability checks,
not comprehensive performance gates.

## Rework

The updated script reports:

| Scope  | Commits | New lines | Ratio <=20 | Ratio <=100 | Follow-ups/change | Corrective share |
| ------ | ------- | --------- | ---------- | ----------- | ----------------- | ---------------- |
| `src`  | 12      | 149       | 0.00       | 0.09        | 0.00              | 0.00             |
| `code` | 12      | 1213      | 0.08       | 0.09        | 0.00              | 0.12             |

Recent fix-on-fix pairs in the broader scope: `b10c74c` redoes 44 lines of `deeaca6`; `c856872`
redoes 47 lines of `fb8ea22`. Follow-up counts for the newest 20 commits remain right-censored and
should not be read as evidence of no future rework.

## Conclusion

The dependency migration and directly related runtime work pass the exercised native checks, with
only the existing Intel accuracy debt in correctness suites. F1 is the sole new finding and must be
corrected before this review passes.

Only this report was added to the original repository. Temporary fixtures and mutations were
confined to the disposable clone and removed; no implementation file or phase-status document was
changed. The report is left uncommitted for the implementation session to commit unchanged.

---

## Implementation response (claims for the re-review to verify)

Written by the implementation session on 2026-11-08. The original report was committed unchanged as
`ec6caa0`. These claims do not change the review verdict or phase status.

**F1:** reproduced the missing-plugin failure (exit 2), then added `--config eslint.config.ts` to
`lint:scan-dispose:website`, matching `lint:ownership:website`. No rule, suppression, target glob or
website configuration was changed.

Validation on the corrected working tree:

- `pnpm run lint:scan-dispose:website`: exit 0; 94 warnings, no errors.
- `pnpm run lint:ownership:website`: exit 0; 5 warnings, no errors.

Ready for the separate review session to re-check F1.
