# Review: dependency updates, round 2

- **Range:** `5a2c612..3a2aaea`.
- **Date:** 2026-11-08.
- **Reviewer:** the separate, non-implementing review session (`c70303a7`).
- **Scope:** F1 from [round 1](dependencies-1.md), and its implementation response.
- **Verdict:** pass -- F1 is resolved; no further finding within this scope.
- **Phase status:** this does not sign off Phase 1 or change any phase status.

## F1: resolved

Commit `3a2aaea` adds `--config eslint.config.ts` to `lint:scan-dispose:website` in
[package.json](../../package.json), line 84, matching the existing `lint:ownership:website` command.
This is the sole implementation change since round 1: no rules, suppressions, file-selection
patterns or website configuration changed.

Both commands now load the root ownership-rule configuration under ESLint 10. The original
missing-plugin failure no longer occurs.

## Independent validation

Validation used a fresh, independent clone detached at `3a2aaea`.
`pnpm install --offline --frozen-lockfile` passed with the normal root build and website prepare
lifecycle; no linked dependencies.

| Check                                        | Result                                                |
| -------------------------------------------- | ----------------------------------------------------- |
| `pnpm run lint:scan-dispose:website`         | Exit 0; 94 warnings, no errors                        |
| `pnpm run lint:ownership:website`            | Exit 0; 5 warnings, no errors                         |
| Negative control through both commands       | Each exits 1 and reports both undisposed scan outputs |
| Commands after removing the negative fixture | Exit 0 again                                          |

The negative control was a temporary TypeScript file in the clone's `website/src/routes/repl/`
directory, inside both commands' existing target patterns. It destructured `carry` and `ys` from
`lax.scan` without disposing or returning them. Both commands reported two
`jax-js/require-scan-result-dispose` errors naming those outputs.

This verifies actual rule enforcement and file selection, not just a successful exit or the absence
of a configuration error. The temporary file was removed.

The full correctness, GPU and benchmark suites were not repeated: this re-review changes only the
lint command and carries forward round 1's results and limitations, including existing R24 and R28
debt. Homepage content remains excluded.

## Rework

`python3 -S scripts/history-churn.py --since 5a2c612` passed. Both scopes report two commits, zero
measured new lines and no recent rework pairs. These numbers are not informative for the one-line
fix: the command is stored in `package.json`, which both measured scopes exclude. Follow-up counts
for recent commits remain right-censored.

## Conclusion

The implementation response's claims are independently confirmed. F1 is closed, and the
dependency-update review passes with the scope and limitations recorded in round 1.

Only this report was added to the original repository. No implementation or phase-status file was
edited. The report is left uncommitted for the implementation session to commit unchanged.
