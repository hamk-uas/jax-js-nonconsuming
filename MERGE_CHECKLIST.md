# Merge Checklist: feat/non-consuming-ops → main

This file tracks items that must be resolved before merging this branch to main.

## Release-readiness snapshot (2026-02-15)

- ✅ `pnpm run lint` exits 0 (warnings only; no blocking errors)
- ✅ `pnpm test` exits 0
- ✅ `pnpm run check` exits 0
- ✅ `pnpm run test:website:smoke` exits 0 (opt-in temporary demo smoke, not part of regular suite)
- ✅ `pnpm run lint:ownership:website` exits 0 for lightweight REPL TS demos

### Notes

- Root lint and type-check now ignore `tmp/` scratch files to keep release gates stable.
- Demo smoke coverage is intentionally opt-in because some website demos are model/data-heavy and
  should not run in default CI/test suites.

## Go / No-Go

### Go criteria

- [x] `pnpm run lint` exits 0 (warnings allowed by current policy)
- [x] `pnpm test` exits 0 (44 files, 1260 passed, 745 skipped)
- [x] `pnpm run check` exits 0
- [x] `pnpm run test:website:smoke` exits 0
- [x] `pnpm run lint:ownership:website` exits 0
- [x] No active `KNOWN_BUG(` tests remain (grep returns empty)
- [x] Pre-commit hook strictness restored (no `|| true` bypasses)
- [x] Website builds clean (`pnpm -C website build` exits 0)
- [x] Website demo ownership audit complete (12 files, ~124 violations fixed)

### Decision

- [x] **GO** — merge `feat/non-consuming-ops` to `main`
- [ ] **NO-GO** — blockers remain (list below)

Blockers:

- None

### Sign-off

- Engineering owner: ********\_\_\_\_******** Date: ****\_\_****
- Reviewer: **************\_************** Date: ****\_\_****

## Pre-merge tasks

- [x] **Fix all KNOWN_BUG tests** — None active. All previously known bugs resolved.

- [x] **Restore strict pre-commit hook** — No `TODO(merge-to-main)` markers found in
      `.husky/pre-commit`.

- [x] **0 test failures** — `pnpm vitest run` exits 0 (44 files, 1260 passed).

- [x] **Website demo ownership audit** — Exhaustive audit of all 19 website TS/Svelte files. Fixed
      ~124 ownership violations across 12 files: - REPL demos: 01-arrays.ts,
      03-logistic-regression.ts, 04-mandelbrot.ts - Pages: mandelbrot, mnist, bench/matmul,
      bench/conv2d - Components: MatmulPerfDemo.svelte, runner.svelte.ts - TTS pipeline:
      inference.ts, pocket-tts.ts, clipInference.ts - Key fixes: removed incorrect model weight
      disposal (use-after-free), KV cache pad/cycle leak fixes, intermediate disposal in generation
      loops, `using` → explicit `.dispose()` (Svelte compiler limitation)

- [x] **Safari `using` support** — Added `SuppressedError` polyfill to `src/polyfills.ts` and
      `website/src/polyfills.ts` (imported in root `+layout.svelte`). This enables the `using`
      keyword at runtime in Safari and other browsers that lack `SuppressedError`. - `.ts` files in
      the website already use `using` where possible. - `.svelte` files still use explicit
      `.dispose()` — Svelte's parser does not support the `using` keyword yet (sveltejs/svelte#16192
      is a draft PR by Rich Harris). When it lands, the `.svelte` files can be converted; the
      polyfill infrastructure is already in place.

- [ ] **Remove this file** — `MERGE_CHECKLIST.md` is branch-specific; delete it after merge.

## Current KNOWN_BUG inventory

| Tag | File | Description     |
| --- | ---- | --------------- |
| —   | —    | All resolved ✅ |

### Resolved KNOWN_BUGs

| Tag                    | Resolution                                                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `sign-nan`             | Fixed: NaN propagation via `notEqual(x, x)` + `where` in `numpy.ts sign()`                                       |
| `bare-vmap-leak`       | Fixed: wrapper-aware primal borrow balancing in transpose + explicit input ownership in test                     |
| `bare-jacfwd-leak`     | Fixed: BatchTrace intermediate disposal + jacfwd primal-tree disposal in eager vmap contexts                     |
| `bare-jacrev-leak`     | Fixed: wrapper-aware primal borrow balancing; test now owns input explicitly                                     |
| `bare-hessian-leak`    | Fixed via jacfwd/vmap ownership cleanup + input ownership in test                                                |
| `makejaxpr-jvp`        | Fixed: avoid cascading JVPTracer Symbol.dispose when lower abstract trace owns values                            |
| `depth4-grad-leak`     | Fixed: robust unreachable Const PETracer cleanup in nested transform stacks                                      |
| `depth4-vjp-uaf`       | Fixed: robust unreachable Const PETracer cleanup in nested transform stacks                                      |
| `anonymous-const-scan` | Fixed: arrays created from raw literals during tracing are tagged as anonymous builder-owned consts in `array()` |
