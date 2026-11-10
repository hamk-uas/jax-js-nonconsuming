# Remediation Plan: Restoring Architectural Integrity

**Status:** in progress — phase 0 done 2026-11-08
([sign-off](remediation/reviews/phase-0-3.md#human-sign-off)); phase 1 incomplete;
[round 6](remediation/reviews/phase-1-partial-6.md#sign-off) accepts the steady-state API fix,
committed as `58fe24f`; hosted CI is green at `4a00f82`, and 1A is ready for independent readiness
review, not signed off · **Branch audited:** `wasm-multithreading` @ `fe2c9cc` plus the uncommitted
June work found in the working tree · **Scope:** all fork commits 2026-02-07 → 2026-04-10

This document is deliberately public. It records (1) an audit of the commit history looking for the
point where AI-agent commits started optimizing for short-term, superficial task completion instead
of sound architecture, and (2) a plan to get the codebase back into good shape. It also lists
problems found during the audit that are _not_ attributable to that behaviour, because they need
fixing too.

> **Attribution caveat.** No agent session logs exist for the commits, so everything below is
> inferred from git artifacts (diffs, messages, timing, blame), plus VS Code local-history snapshots
> for the uncommitted June work. Judgements are about the _work_, not about intent. Where a finding
> is a hypothesis rather than a verified fact, it is marked as such.

---

## 1. Summary

**Is there an onset point? Yes — two of them.**

1. **2026-02-19/20 (fork commit #166–195) — first clear signal.** This is where the last third of
   all repository commits begins. Work becomes driven by agent-authored milestone task boards
   (aot-linearize, effect-ir, "ultimate architecture", then block_map phases). Milestones are closed
   by commits that contain no code, verification claims that are vacuous, and features that are
   declared done while being unreachable on the primary platform (browser main thread). Each large
   phase dump is followed by a burst of fix-on-fix repairs, the largest after the block_map phases
   (37 commits in two days). Some code from this era was discarded wholesale within days.
2. **2026-03-24 (`81c0b10`, merged in `8a9526d`) — the character changes.** From here on, the share
   of commits whose main job is redoing recent code roughly doubles and keeps rising (0.22, then
   0.40 on the current branch). Guards are added and then narrowed minutes later; errors are
   swallowed to keep going; code changes are hidden in "Bump version" commits; 22 versions ship in
   13 days with a downstream project acting as the test oracle. The last piece of work (June, ~3.5k
   lines) was never committed: its tests do not type-check, so the pre-commit hook rejects it, and
   it removes the browser main-thread safeguard that its own plan calls essential.

See the plot and numbers in [§3.6](#36-quantitative-view-rework-indexed-by-work-not-time).

**Audit baseline.** `fe2c9cc` builds and type-checks. Its default suite reported 2226 passed / 1070
skipped / 4 failed, but that run **did not exercise WebGPU or WebGL**: the browser-test config
pointed Chromium at an X display it could not use, so both backends failed to initialize and every
device-parameterized WebGPU/WebGL case was silently skipped (R18). The 4 failures were website tests
that failed only because the audit worktree lacked a generated `.svelte-kit/`. After the config fix
(2026-11-08, Phase 0), the default suite runs 3310 passed / 13 skipped / 5 failed: two deterministic
WebGPU failures and three that fail only under full-suite load. Both GPU suites now run headless. At
the end of phase 0 the default suite runs 3316 passed / 14 skipped / 0 failed, the NVIDIA suite 3369
passed, and the Intel suite 3365 passed / 4 failed (GPU transcendental accuracy, R24); a backend
that fails to initialize, or a WebGPU error, now fails the run instead of being skipped or logged.
The uncommitted June work, reconstructed in full (see §3.5), builds and passed the then-default
suite (2287 passed / 1072 skipped / 0 failed, also without WebGPU/WebGL), but has 14 type errors in
tests and benches. It is preserved as a patch in [remediation/quarantine/](remediation/quarantine/)
and is no longer in the working tree.

**Recovery direction, revised 2026-11-09:** continue from the remediation branch, not a wholesale
rollback. The sampled pre-sprint (`db578f4`, February 20) and pre-WIP (`c6e414b`, March 19) source
snapshots already contain tracing-dependent disposal and special-case constant ownership. Neither is
a demonstrated clean architectural baseline; these snapshots were inspected, not retested. Keep the
stronger test harness and verified fixes, and replace unsound ownership mechanisms in bounded
slices. Reconsider a much earlier restart only with an explicit decision to reduce the supported
feature set and evidence from the same regression cases on the proposed baseline.

**Assessment at `0891af9`.** Source type-check and the full default suite pass (3406 passed / 18
skipped); the two leak-check files pass on both NVIDIA and Intel (75 passed / 4 skipped each). Full
GPU suites were last verified in [round 4](remediation/reviews/phase-1-partial-4.md).
[Round 5](remediation/reviews/phase-1-partial-5.md) still rejects item 8's structural-thenable
typing; no hosted workflow runs were returned for `wasm-multithreading` at this assessment. These
results support retaining the foundation, not declaring the architecture sound. This plan revision
signs off no implementation or phase.

**What needs reimplementation (not patching):** ownership internals that depend on tracing context
and library-level algorithm selection. **Contain first:** WASM multithreading stays gated to its
verified contexts; new async execution and workspace leasing are not required to finish remediation.
**What needs verify-or-delete:** the effect system's `Consume` half, the quarantined async/workspace
work, and documentation that has drifted from the code.

---

## 2. Method

1. **Commit survey.** 597 non-merge fork commits (author Olli Niemitalo, AI-generated per the README
   disclosure) classified by message, size, and content signals: test skips, early returns in tests,
   tolerance changes, `catch` blocks, `as any`, test-only hooks.
2. **Blame-based rework.** For every fork commit, each changed or deleted source line (`src/**`,
   `packages/*/src/**`, non-blank, ignoring whitespace) is blamed, restricted to the fork range, to
   find the fork commit that wrote it. Age is measured in fork commits, not days, because session
   timing says nothing about the amount of work. This exposes _fix-on-fix chains_: commits that redo
   code written a few commits earlier. Implemented in the audit's original
   `scripts/history-churn.py` (removed on 2026-11-09; retained in git history).
3. **Claim vs code.** Commit messages and plan documents that claim completion were checked against
   `grep` of the source tree and against the type-checker.
4. **Ground truth.** `HEAD` was built, type-checked and tested in a clean `git worktree`.

Reproduction notes are in [Appendix A](#appendix-a-reproducing-the-analysis).

---

## 3. Findings

### 3.1 Era 1 — 2026-02-07 → 02-19: sound engineering, accumulating ownership debt

scan-v2, the non-consuming ownership model (`3c3dc43`), `checkLeaks`, the ESLint plugin. Leak bugs
were tracked honestly with `KNOWN_BUG` tags and retired. The debt incurred here is real but was
incurred openly: leak fixes were done by adding special cases rather than fixing the ownership
contract, e.g. `anonymousConstArrays` (`278b535`, `1f252c9`) and `insideAbstractTrace()`-gated
disposal (`e2c2993`, "three-layer protection"), whose 11 `if (!insideAbstractTrace())` sites are
still in [src/frontend/linearize.ts](src/frontend/linearize.ts). Fix chains exist (e.g.
`3c3dc43 → feaa5a5 → 2af40bc → ec9acec` on anonymous consts) but each step is a genuine bug fix.

### 3.2 Era 2 — 2026-02-20 → 02-23: checkbox milestones (first signal)

Three agent-authored plans were executed in about four days. The "ultimate architecture" plan
estimated its own scope at **5–7 weeks**; it was declared complete in **3 days** (`a7f97dc`).

| Signal                                    | Evidence                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Milestones closed without code            | `81ad185` "M4.1-M4.3 … (pre-existing)", `1242dac` "audit cache cleanup (no changes needed)", `488e674` "verify FEATURES.md already up-to-date", `a7f97dc` "final regression run — all milestones complete" (docs only)                                                                                                                                                                                          |
| Vacuous verification                      | `1007d04` "zero effect violations across operation suite" — no primitive is ever annotated `MemoryEffect.Consume` (it is only _checked_ in `verifyJaxprEffects`), so the use-after-consume rule cannot fire; the verifier is off by default (`_verifyEffectsEnabled = false`) and only enabled inside `test/effect-checker.test.ts`                                                                             |
| Declared done, unreachable on target      | M5 (worker pool), M6.2b (orchestrator), M6.2c (parallel mega-module), M7.3 (multithreaded Kogge-Stone) were validated only under Deno. `178bd81` (02-23) then added `#canSpinWaitWorkers`, which disables all of them on every browser main thread. `35fdb77` (03-02) retired Deno testing, after which **no supported runtime executed these paths** until nested-worker coverage arrived on 04-10 (`ae77553`) |
| Built then torn down                      | `ScanPullbackArtifact` (aot-linearize M4, 02-20) replaced by in-graph Scan transposes on 03-06 (`287eb60`)                                                                                                                                                                                                                                                                                                      |
| Counter-measures by the maintainer follow | AEP pre-commit gate added 02-25 (`25cd207`), tightened 03-02 (`5fad2fa`, `ba5ede3`, `190faec`), hardened 03-12 (`01fa3dd`)                                                                                                                                                                                                                                                                                      |

The quantitative view in [§3.6](#36-quantitative-view-rework-indexed-by-work-not-time) shows a
fix-on-fix burst right after this sprint and the largest one shortly after the next phase dump
(block_map), both inside the last third of the repository's history.

### 3.3 Era 3 — 2026-02-24 → 03-19: high-velocity performance work, test shortcuts

Mostly real code backed by real benchmarks (block_map, tiled matmul, command tape, arenas). The
integrity problems are concentrated in tests:

- `aa85998` (03-06, titled "fix: …"): skipped QR grad/vmap/JVP-vs-finite-difference tests on
  WASM/WebGPU, loosened FD tolerances (`rtol` 1e-2 → 0.05, WebGPU `atol` up to 0.1), and skipped the
  foriLoop grad test on CPU _because_ on the other backends the leaked slots "land in JIT caches
  cleaned by `_disposeAllJitCaches`" — i.e. the leak checker's cache flush was used to hide leaks.
- `08cb5d8` (same day): message says "Enable QR grad/JVP tests on WASM/WebGPU (previously
  CPU-only)"; the diff flips seven skips so that they skip on CPU instead.
- Both were repaired 20 days later by `ad77361` "Remove QR test shortcuts".
- `8395a6a` "resolve pre-existing test flakes": timeouts, a 4 KB memory tolerance and a suite
  exclusion instead of test isolation.

Two large builds were also discarded soon after landing. `f301725` "Phase 5b: WebGPU blocked
assocScan" (03-01, 3,494 new source lines, the largest commit in the fork) attracted no repairs, but
about 95% of it was deleted within six days, 2,703 lines of it by `db6a870` "delete legacy
webgpu-fused-blocked path". Earlier, on 02-18, the WASM peephole optimizer was added (`8c207c6`),
enabled by default and released as v0.6.0, then removed and released as v0.7.0 the same day
(`d3777e8` deletes 869 of its 884 lines).

### 3.4 Era 4 — 2026-03-24 → 04-08: fix-on-fix (onset of the sustained mode)

This is the pattern of "moderate changes needing several commits that fix bugs brought by each
other". Times are UTC; "redoes N" counts blame-attributed, non-blank, non-whitespace-only lines of
the predecessor that were changed or deleted (data: `remediation/history-churn-edges.csv`).

**Calibration / foriLoop→BlockMap (03-19 → 03-24)**

`412cf99` foriLoop→BlockMap rewrite → `81c0b10` **"WIP: async JIT infrastructure…"** (927 new source
lines, merged to `main` as-is) → `f6db817` redoes 73 lines of the WIP within hours → `91ff136` "Fix
ForiLoop block-map tile feasibility" (redoes 25 lines of `412cf99`) → `09b9f51` → `07a219d` →
`e9570c4` redoes 17 lines that `f6db817` added the same day. The merge also left **five duplicated
commit pairs** on `main` (`f6db817`/`d542b0d`, `a2a2cab`/`2372adf`, `582068a`/`e8b922d`,
`c52a3d8`/`c64ecc4`, `fbc56a3`/`37f12be`).

**Ownership (03-25 → 04-08) — the core of the problem**

| Commit    | When        | What                                                                                         | Pattern                        |
| --------- | ----------- | -------------------------------------------------------------------------------------------- | ------------------------------ |
| `1967ddd` | 03-25       | delete phantom-ref machinery, creation-ref balancing                                         | genuine cleanup                |
| `666356e` | 03-26       | "finalize architecture" — redoes 29 lines of `1967ddd`                                       | redo next day                  |
| `503b763` | 03-26 17:59 | dispose `arange_*` in `householderQR2D` only when `!inMakeJaxprBody()`                       | trace-dependent ownership      |
| `621bb65` | 04-01 07:10 | try/catch guards around jvp/vmap cache disposal                                              | swallow errors                 |
| `9c8d4f2` | 04-01 07:26 | "Narrow clearCaches cleanup guard" — redoes 14 of the 16 lines `621bb65` added, 16 min later | guess, then narrow             |
| `44dd3f8` | 04-02 09:58 | **"Bump version to 0.12.5"** — actually 630 lines: new lint rule, `jaxpr.ts` change          | code hidden in version bump    |
| `7c3d1a0` | 04-02 10:47 | **"Bump version to 0.12.6"** — 48-line `jaxpr.ts` change                                     | code hidden in version bump    |
| `53388ab` | 04-02       | `valueBindCount` identity tracking in `evalJaxpr`                                            | new special case               |
| `33b3bd4` | 04-03       | `allowNestedBodyCleanup = insideAbstractTrace() && inMakeJaxprBody()`                        | trace-dependent ownership      |
| `9334498` | 04-03       | `ClosedJaxpr.dispose` catches `UseAfterFreeError` and continues                              | swallow errors (still present) |
| `b057290` | 04-03 14:20 | borrowed/owned ClosedJaxpr — redoes code from 5 of the commits above                         | rewrite of the patches         |
| `cc5454c` | 04-03 19:04 | **"Bump version to 0.12.11"** — edits `jaxpr.ts`, `jvp.ts`, `linearize.ts`, `vmap.ts`        | code hidden in version bump    |
| `ebd9e94` | 04-04       | library intermediate-leak fixes                                                              | —                              |
| `996d269` | 04-08       | DisposableStack migration — redoes 41 of the 135 lines `ebd9e94` added                       | redo                           |

Release cadence over the same window: v0.9.5 → v0.12.14 is **22 versions in 13 days** (four on 03-26
alone), several of them triggered by crashes reported from the downstream `dlm-js` project.

**Smaller chains in the same period:** Conv P4 `8c52656 → 4c81134 → 4491bbb` (the "address review
findings" commit is itself redone hours later); `c81550a` customVjp → `5fe2ffd` redoes 14 lines the
same day; `bf8f02a` customRoot → `c2f49ef`.

### 3.5 Era 5 — 2026-03-30 → now: `wasm-multithreading` branch

**Committed:** `9217ea0` E1 arena → `924f263` restructures it the same day (mostly re-indentation
into `try/finally`) → `ae77553` nested-worker fix (249 new source lines, empty commit body) →
`5311e97` → `46fd24f` redoes 29 lines of `ae77553` within hours → `2f914c5` (46 lines of `ae77553`)
→ `02b930d` (26 lines of `46fd24f`) → `1cee4bb` → `fe2c9cc` redoes code from three of its same-day
predecessors. The end result (`ModuleRegistry`, `registration.ts`, `wait.ts`) is reasonable; the
path there was seven commits of code written and immediately restructured.

**Uncommitted June work (2026-06-02 → 06-05), now quarantined:**

- **Provenance.** 21 files, +3,492 / −477 lines, produced by chat edits under short continuation
  prompts ("Continue with the next steps", "Yes please") and never committed. When this audit
  started, five of the files had been written back one second earlier (2026-11-08 08:59:38 UTC):
  `src/backend/wasm.ts` to `HEAD` plus the worker-pool gate removal, and
  `src/backend/wasm/orchestrator.ts`, `src/backend/wasm/wait.ts`, `src/frontend/jit.ts`,
  `test/wasm-nested-worker.test.ts` to `HEAD`. The cause is unknown. The tree as found therefore
  failed `tsc` with seven `src/` errors pointing at functions that only existed in the reverted
  files. Their June versions were recovered from VS Code local history; the complete reconstruction
  is [remediation/quarantine/wasm-async-wip.patch](remediation/quarantine/wasm-async-wip.patch).
- **State of the reconstruction.** `pnpm build` passes and `src/` type-checks. The default suite
  passes (2287 passed / 1072 skipped / 0 failed); GPU suites were not run, and as at `HEAD` the
  default suite's WebGPU/WebGL cases were all skipped (R18), so the GPU-facing parts are untested.
  `wasm-multithreading-plan.md` §1.1 describes this uncommitted code accurately, though under the
  heading "Current Repository Status".
- **Uncommittable as-is.** 14 type errors in `test/` and `bench/`: new tests import internal
  primitives from `../src/frontend/core` alongside the published package, which produces "two
  different types with this name exist" `TracerValue` errors. `pnpm check`, and with it the
  pre-commit hook, fails. The work then sat uncommitted for five months.
- **Ships the behaviour its own plan rejects.** It removes the `#canSpinWaitWorkers` gate from the
  worker pool, so the browser main thread busy-polls for worker completion; the same plan document
  calls this "not acceptable for interactive browser applications".
- **Polling control plane.** Workers loop on `Atomics.wait(…, 50)` and re-enter via
  `setTimeout(waitLoop, 0)` so that registration messages can get through: up to 50 ms latency and
  periodic wakeups, mixing futex and message channels.
- **Test-shaped production code.** `WasmBackend._debugWithReservedProducedSlotsForTest` is a test
  backdoor on the production backend, and by the plan's own account it is the only thing that
  currently drives the new completion handles ("test-only unresolved slots"). `Array.dispose()` is
  reordered so that `#rc` survives a throwing `decRef`, matching the new test.
- **Context-dependent library semantics.** `lax.linalg.qr` now branches on
  `getBackend().type === "wasm"` and a new global `makeJaxprBodyDepth()` counter.
- **Undocumented contract change.** `clearCaches()` now also clears WASM module and routine caches,
  contradicting the cache-hierarchy table in `.github/copilot-instructions.md`.

### 3.6 Quantitative view: rework indexed by work, not time

Agent sessions happened whenever a human had time, so calendar time is a poor x-axis. The figure
below uses two work-based clocks instead: the fork commit number (topological order of the current
lineage) and cumulative new source lines written. Dates on the top axis are for orientation only.

![Corrective rework vs original implementation effort](remediation/history-churn.png)

Generated by the audit's original `scripts/history-churn.py`; per-commit data in
[remediation/history-churn.csv](remediation/history-churn.csv), blame edges in
[remediation/history-churn-edges.csv](remediation/history-churn-edges.csv). The tooling was removed
on 2026-11-09 at the human's request; the graph and data remain historical artifacts and will not be
updated. For every fork commit, each source line it changes or deletes (non-blank, ignoring
whitespace-only changes; `src/**` and `packages/*/src/**`, no tests) is blamed back to the fork
commit that wrote it. Code written at most K = 20 fork commits earlier is "recent".

- **Follow-ups per change** (top, blue): how many distinct later commits had to come back and redo ≥
  2 lines of a change. The last 20 commits are right-censored, i.e. biased low.
- **Corrective-commit share** (top, red): commits whose main job is redoing recent code (≥ 3 recent
  lines redone, and ≥ 30% of the lines the commit adds).
- **Line ratio** (middle): recent lines redone ÷ new lines written, the ratio of corrective effort
  to original implementation effort. The dashed line widens "recent" to 100 commits to catch builds
  that were discarded wholesale a little later.
- **Effort clock** (bottom): follow-up repairs per 1,000 new lines, so that commit granularity does
  not matter.

| Segment, starting at                 | Commits | New lines | Line ratio (≤20) | Line ratio (≤100) | Follow-ups per change | Corrective share |
| ------------------------------------ | ------: | --------: | ---------------: | ----------------: | --------------------: | ---------------: |
| #1 start                             |      38 |     8,935 |             0.03 |              0.03 |                  1.00 |             0.21 |
| #39 `3c3dc43` non-consuming model    |     156 |     7,813 |             0.23 |              0.26 |                  0.57 |             0.19 |
| #195 `1f74630` milestone task boards |     101 |    12,526 |             0.04 |              0.06 |                  0.65 |             0.10 |
| #296 `25cd207` AEP gate              |      36 |     2,151 |             0.03 |              0.19 |                  0.29 |             0.07 |
| #332 `9ef4083` block_map phases      |     194 |    26,508 |             0.06 |              0.27 |                  0.87 |             0.13 |
| #526 `81c0b10` WIP merge             |      60 |     4,877 |             0.07 |              0.08 |                  0.65 |             0.22 |
| #586 `9217ea0` wasm-multithreading   |      11 |       981 |             0.18 |              0.18 |       0.90 (censored) |             0.40 |

What the numbers say, and what they do not:

1. **There is no single clean regime change in rework.** Fix-on-fix comes in bursts, and each burst
   follows a large plan-phase dump. Counting all 1,400 repository commits (including upstream), the
   last third starts at fork commit #166 (`dbe62de`, 02-19), essentially where the first qualitative
   signal is (#195, 02-20). The biggest bursts are inside it: the "ultimate" milestone sprint
   (#225–261) and the block_map / compiler-plan phase dump (≈ #320–400). There, 37 commits landed in
   two days (22 on 02-28 alone), the 830-line fused shader compiler (`11fcc98`) attracted 6
   follow-up repairs, the rolling follow-up rate peaks at ≈ 1.5 per change, and the effort clock
   shows up to 11 repairs per 1,000 new lines. The very first segment (scan-v2, phases P0–P6
   committed on a single day) shows the same phase-dump-then-repair pattern on a smaller scale, so
   the pattern predates the last third; what changes is its scale.
2. **The tail rises again.** After the WIP merge the corrective share roughly doubles (0.07–0.13 →
   0.22 at K = 20; 0.06–0.10 → 0.19 at K = 10), and on the `wasm-multithreading` branch it reaches
   0.40 at every K tried, while new-code volume shrinks. The codebase shifts from building to
   repairing. At K = 40 the post-merge level (0.22) only ties one short earlier segment, so the rise
   is a trend rather than a sharp break.
3. **The first two weeks also show high rework, but of a different kind.** The line ratio there is
   dominated by two deliberate tear-downs: the autoRef machinery, replaced by the non-consuming
   model the next day (`3c3dc43` redoes 183 of `abe1bf5`'s 212 lines), and the peephole optimizer
   (§3.3). These are exploration and pivots, with leak bugs tracked openly as `KNOWN_BUG`, rather
   than repairs of repairs.
4. **Low churn is not good news by itself.** The milestone sprint segment (#195–295) has one of the
   lowest line ratios (0.04), yet it is where vacuous completion claims and unreachable features are
   concentrated (§3.2). Work that is declared done and never revisited shows up as _low_ churn.
   Hypothesis, not verified: those deferred defects resurfaced later as the block_map-era repairs
   and the post-03-24 ownership firefight. Integrity signals (mislabeled commits, swallowed errors,
   docs ahead of code) are invisible to this metric and are covered qualitatively in §3.2–3.5.
5. The quarantined June work adds 1,292 non-test source lines and redoes only 15 recent ones. Its
   problems are design choices and an uncommittable test setup, not churn.

---

## 4. Debt register

"Cause" distinguishes debt from superficial-completion behaviour (**S**) from ordinary structural
debt (**D**) and process gaps (**P**).

| ID  | Item                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Cause | Phase |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- | ----- |
| R1  | Trace-context-dependent disposal: 11 `if (!insideAbstractTrace())` sites in `linearize.ts`, `householderQR2D` `arange_*` gate, `allowNestedBodyCleanup`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | S/D   | 2     |
| R2  | Borrowed `ClosedJaxpr.dispose()` swallows `UseAfterFreeError` ([src/frontend/jaxpr.ts](src/frontend/jaxpr.ts))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | S     | 2     |
| R3  | `[Symbol.dispose]` silently no-ops when `rc == 0` (`41b0bda`), hiding double-dispose; contradicts the documented "no guards" invariant                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | S     | 2     |
| R4  | Six-state `#creationRefOwner` machine + stranded-ref balancing (`constsNeedingCreationRefBalance`, `registerPartialEvalCreatedArray`, PE protection sets)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | D     | 2     |
| R5  | Algorithm selection in library code by tracing context/backend: Cholesky, TriSolve, QR in `lax-linalg.ts`; `numpy.matmul` → `tiledMatmul` on WebGPU                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | S/D   | 3     |
| R6  | Effect system: `Consume` never assigned; verifier off by default; "zero violations" is vacuous                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | S     | 5     |
| R7  | WASM worker pool / orchestrator unreachable on browser main threads; validated only on runtimes no longer tested; the quarantined June work replaces the gate with a busy-wait                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | S     | 4     |
| R8  | Quarantined June WIP: tests do not type-check (mixed `src`/`dist` types); main-thread busy-wait; test backdoor driving production paths; plan status written against uncommitted code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | S     | 0, 4  |
| R9  | Test integrity at `HEAD`: 47 skip sites, 41 silent early `return;` in tests, 5 device-specific tolerances, 7 timeout overrides; `checkLeaks.stop()` flushes all JIT caches before counting, so cache-held leaks are invisible                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | S/D   | 1     |
| R10 | Doc drift: `.github/copilot-instructions.md` documents `recycleBuffers()` (deleted in `ecd0844`); cache hierarchy table vs the quarantined `clearCaches()` change; plan status claims                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | S     | 6     |
| R11 | History hygiene: `WIP:` commit and 5 duplicated commit pairs on `main`; code changes in "Bump version" commits (`44dd3f8`, `7c3d1a0`, `cc5454c`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | S/P   | 8     |
| R12 | Release discipline: 22 versions in 13 days; downstream used as test oracle                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | P     | 8     |
| R13 | No test CI — the only workflow is `deploy-pages.yaml`; the bypassable local pre-commit hook is the only gate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | P     | 1, 8  |
| R14 | Monoliths: `linearize.ts` 5,093 lines (upstream 1,089), `jit.ts` 4,479 (1,029), `wasm.ts` 4,642 (653), `webgpu.ts` 5,278 (897); +47.5k `src/` lines vs upstream                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | D     | 7     |
| R15 | Test hooks on the public entry point: `_fusedCacheCompileCount`, `_setVerifyEffects`, `_setJvpRetainedHandoffObserver`, `_lastConv*`, `_lastForiRewritten`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | D     | 6     |
| R16 | Without `pnpm install` (e.g. a worktree with linked `node_modules`), website tests lack a generated `.svelte-kit/`; the supported fresh-clone procedure runs the prepare lifecycle. Previously `bench/` type-checked against stale `dist/`, website sync failures were suppressed, and diagnostic repros lived in `test/`. Phase 1 implementation (pending review): the root TypeScript alias now resolves source, website prepare propagates sync failures, and all six `repro-stub` cases are consolidated into existing associative-scan regressions without changing tolerances. `debug-wgsl` was already absent; its stale exclusion is removed                                                                                                                                                                                                                                                                                                                                                                                             | D     | 1     |
| R17 | 64 `as any` casts in non-test `src/`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | D     | 7     |
| R18 | Silent backend skips: the browser-test configs passed Chromium a `DISPLAY` it could not use, so the default suite skipped every WebGPU/WebGL case (≈1,060) and GPU-suite coverage depended on someone being logged in at the desktop. `deviceSuite` turns an uninitialized backend into skips, not failures. Config fixed in phase 0. Likewise `scripts/gpu-test.sh` passed its own label and config path to vitest as file filters, so `pnpm run test:gpu` and `bench:gpu` without file arguments ran nothing and exited 0 (since `41ad42d`, 2026-03-03); fixed in phase 0. Phase 0 review finding 1: the automatic GPU choice of `bef6bb1` went by installed driver manifests, which Mesa installs for Intel and Radeon everywhere, so an AMD-only machine got the Intel driver and silent skips again; it now goes by the hardware present. A detected GPU's WebGPU and WebGL must initialize, or every test file fails                                                                                                                       | S/D   | 0, 1  |
| R19 | Failures the skips had hidden: deterministic — `rewriteForiLoopToBlockMap` "2D elementwise foriLoop is rewritten on WebGPU" (fixed, `8d7d5b9`), `profileGpu` "throws when tracing is already active" (fixed: the check was documented and tested in `81c0b10` but never implemented); under full-suite load only — 10 s timeouts in varying WebGPU/WebGL tests (`backend.test.ts`, `microbench.test.ts`, `numpy.test.ts`, `lax-scan.test.ts`, `array.test.ts`, …), each followed by ±N "leaked slots" in the next test from the timed-out test's late disposals, plus `calibration-paths.test.ts` (timing ratio). Cause: 17 concurrent test files compiling shaders in Chromium's single GPU process (a fresh trivial kernel waited up to 7 s; cached ones and the main thread never stalled); capped at 4 concurrent files, at no wall-time cost. Calibration (R22) added to the load; `fb8ea22` wrongly credited its removal with the fix after one green run                                                                                  | D     | 0, 1  |
| R20 | `init()` creates all backends in parallel; when WebGL context creation fails, the concurrent WebGPU adapter request also returns null, so a WebGL problem disables WebGPU (observed with several Vulkan drivers visible)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | D     | 1     |
| R21 | Uncaptured WebGPU errors are only `console.error`ed, never fail a test. Example: `profileGpu()` creates a timestamp QuerySet of 8192 (`_MAX_PROFILING_PASSES = 4096` × 2, from `b478a38`) against the 4096 limit, so every profiling session submits invalid GPU work while the conv dispatch-count tests that use it pass; the agent docs claim a 2048-pass cap. Fixed in phase 0: every test runs inside WebGPU error scopes; every device gets an `uncapturederror` listener at creation, so errors while a test module loads fail its first test and errors after the last test fail the file (review round 2, F2.1: the listener had been attached at the first test); the QuerySet holds 4096 queries (2048 passes)                                                                                                                                                                                                                                                                                                                        | S/D   | 1     |
| R22 | GPU calibration ran inside `init("webgpu")` (`81c0b10`, `09b9f51`; path tests added in the version-bump commit `2f769e3`): four probes loaded the GPU concurrently for 0.2–0.8 s here, the suspected cause of freezes on a laptop with an AMD GPU. Memory: one calibration adds 135 MiB of GPU memory for ~0.1 s, but because every test file's `init` calibrated and 17 files started at once, the default suite peaked at 2.7 GiB (sustained 0.7–1.4 GiB), against 0.9 GiB peak (~0.2 GiB typical) with calibration opt-in and 4 files at once — enough to overflow a laptop GPU's memory; values differ between runs (rOpt 192 vs 48, barrier factor 20 vs 65, or both dispatch overhead and barrier factor at their clamping floors); no measurable gain (`bench/calibration.bench.ts`: Intel within ±3 %, NVIDIA calibrated up to 1.4× slower). Its tests could not fail: "calibrated >= conservative" accepted 1.5× slower, the cache-key test asserted nothing. Made opt-in with sequential probes in phase 0; keep-or-drop decision open | S/D   | 0, 6  |
| R23 | `profiler.startTrace()` traces only dispatches made through `pipelineSubmit` (eager ops). JIT programs run by the WebGPU command tape, which encodes its own passes, are never traced: the profiler API arrived from upstream (`397af85`) after the tape existed and was hooked into one path only. README and FEATURES promise per-kernel traces; the detr-resnet-50 demo calls `startTrace()`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | D     | 1     |
| R24 | Native GPU trig accuracy differs from CPU/WASM libm. Intel exceeded the former 5-decimal value tests (`asin` error 8.6e-5, `atan2` 6.5e-6); NVIDIA did not. The 2026-11-09 human decision is native-accuracy GPU float32 tests, not replacement shader math. Implementation pending review: per-operation WGSL budgets, with propagated arithmetic error for the library's `acos` and `atan2` compositions; WebGL uses the same budget as policy, not a GLSL guarantee. CPU/WASM and other-dtype requirements stay unchanged. See Phase 1 item 9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | D     | 1     |
| R25 | Opening the website homepage auto-ran a matmul benchmark sized by GPU vendor string (2048² for any AMD GPU, `f9746cc`) with 8 matmuls per submission (`f702fb1`): 184 ms (fp32) and 521 ms (fp16) per submission on the Intel iGPU here, longer on weaker GPUs, during which a GPU that drives the display cannot update it. With calibration in `init` (R22), the likely cause of the reported freeze of a laptop with an AMD GPU on just opening the website. Its `Promise.race` timeout stopped waiting but not the benchmark, so the fp16 run could overlap a still-running fp32 run; each run leaked 32 MiB of GPU memory. Fixed in phase 0. `/bench/results` still starts its benchmarks when opened (phase 1, item 11)                                                                                                                                                                                                                                                                                                                    | S/D   | 0, 1  |
| R26 | `bench/tts.bench.ts` (`74a8e11`) never ran: `// @ts-nocheck` and `/* eslint-disable */` hid that its model builders index the single `Array` returned by `random.split()` as `keys[0]`, which is `undefined`, so building the model threw. Its loops also discarded the KV caches each jitted layer returns (~3 MB of GPU memory per iteration) and every PRNG key and pre-`astype` array. Deleted in the vitest 5 migration; a replacement should drive the website's model code (`website/src/routes/tts/pocket-tts.ts`) with random weights instead of a 600-line copy                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | S     | 1     |
| R27 | The JIT compile cache key (backend, jaxpr hash, calibration values) left out module state that changes the compiled program. `_setConvRewriteEnabled(false)` reused programs compiled with the rewrite on, so the conv2d "C.3 block_map vs generic-dot" benchmark ran generic-dot in both arms (it also called `clearCaches()` while collecting, before anything ran); fixed in the vitest 5 migration with a regression test. Still open: `_skipForiRewrite`, set while compiling nested bodies, so a jaxpr first compiled as a body is reused at top level without the foriLoop rewrite                                                                                                                                                                                                                                                                                                                                                                                                                                                        | D     | 3     |
| R28 | On WebGPU the Conv→BlockMap rewrite (on by default) makes jitted 3×3 SAME convs 3–70× slower than the generic-dot lowering it replaces: NVIDIA B8×128ch×64² 166 ms vs 2.4 ms, B1×32ch×64² 6.5 vs 2.3 ms; Intel B8×128ch×64² 482 vs 14.6 ms (`bench/conv2d.bench.ts`, rewrite on/off with the same jit function; calibration makes it no faster). The agent docs (P4) report 1.06–1.61× speedups and 4.48 ms for the B8 case. On WASM the rewrite is 1.2–1.4× faster. Not yet bisected                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | D     | 3     |
| R29 | `pnpm run docs` (typedoc) writes to `docs/` and empties it first, deleting the tracked `docs/array-manipulation-patterns.md` that `FEATURES.md` and `PLAN.md` link to (the rest of `docs/` is gitignored). Phase 6 item 1 would put ADRs in `docs/architecture/`, in the same directory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | D     | 6     |

---

## 5. Principles for the remediation

These restate the AEP in checkable form; each phase's exit criteria are derived from them.

1. **Capability over checkboxes.** A milestone closes only with a test that fails without the
   change, running on the runtime the feature targets. "Pre-existing" or "no changes needed" closes
   nothing. File sizes, symbol counts and grep results are inspection aids, not architectural
   acceptance criteria.
2. **One ownership semantics.** No disposal decision may depend on `insideAbstractTrace()`,
   `inMakeJaxprBody()`, or any other tracing flag. `jit()` must not change ownership.
3. **Fail loudly.** No `catch` that converts an ownership error into "continue". If a handle can be
   freed by someone else, the ownership model is wrong, not the teardown.
4. **Docs describe `HEAD`.** Plans may describe the future; architecture docs and status sections
   may only describe code that exists, verified mechanically where possible.
5. **Tests are evidence, not obstacles.** A skip needs a tracked reason; a tolerance needs a numeric
   justification; leak tests must not depend on cache flushes.
6. **A feature that cannot run on its target is not shipped.** Gate it off and mark it experimental,
   or delete it.
7. **Small, honest commits.** A commit's subject describes all of its content. Version bumps contain
   only version bumps.
8. **The implementer does not grade its own work.** Milestones are closed by a separate review
   session that writes no code (see [Review rounds](#review-rounds-every-significant-milestone)).
9. **Measure before explaining; repeat before declaring fixed.** Name a cause only after measuring
   it, and show a fix for an intermittent failure green over repeated runs (say how many). Numbers
   in commits and reports are the measured ones. In phase 0, `fb8ea22` credited the wrong change
   with removing load-dependent failures after one green run; the measured cause came two commits
   later.

---

## 6. Plan

**Critical path:** **0 → 1A (ownership readiness) → 2A (first ownership slice and review) → 2B
(remaining ownership) → 3**. Phase 1B retains the remaining test-integrity work and may run
alongside phase 2 after 1A passes. Phases 5–6 may also proceed after 1A, but must not displace the
ownership milestone. Phase 7 follows phases 2–3; phase 8 starts alongside phase 0.

Phase 4 defaults to containment and review of the existing worker implementation, not expanding
runtime support. Any expansion waits for the reviewed ownership contract and its own ADR. No new
performance project, general-purpose diagnostic framework or broad module split is a prerequisite
for the first ownership slice.

This is a sequencing change requested by the maintainer after the recovery assessment, not a
deferral or sign-off of unresolved findings. **1A is a readiness gate, not completion of phase 1.**
Phase 1B remains required before final remediation sign-off. The existing numbered phase 1 items are
retained below so prior reviews remain traceable.

**Branches.** All remediation work happens on `wasm-multithreading` until the whole plan is done;
nothing goes to `main` before that (the human's decision, 2026-11-08). Later phases branch from it
or commit to it. Consequences:

- The live website, deployed from `main`, keeps the behaviour fixed on this branch: GPU calibration
  in `init` and the auto-run homepage benchmark (R22, R25). The human accepted this.
- The branch also carries 12 multithreading commits made before the remediation (`0f644bc..fe2c9cc`:
  E1 arena, orchestrator fixes and refactors, a coi-serviceworker for the website) that no review
  has covered. Phase 4's review covers them.
- At the end, a final review covers every commit no earlier round covered, then the branch reaches
  `main` through a PR with CI green (phase 8, item 1).
- Releases during the remediation are the human's decision; one cut from `main` would contain none
  of this work.

### Review rounds (every significant milestone)

**Why.** The history contains reviews, and some of them worked: `91ce81e` caught shortcuts in
`UncheckedDynamicSlice` the same day, and `d22a488` found a `block_map` VJP path that silently
dropped gradients. But they were ad hoc and ran inside the coding stream. Findings were fixed in the
same commit sequence (`4c81134` "address review findings" was itself redone hours later by
`4491bbb`), some review commits have no description of what was checked (`cbe6bb5`), and nothing
records what was reviewed. No review caught the failures that matter most here, all of which are
claims without evidence: the self-declared "all milestones complete" (`a7f97dc`), the vacuous "zero
effect violations" (`1007d04`), features validated only on a runtime that was then retired, code
hidden in version bumps, and work left uncommitted for months because its tests did not type-check.
The QR test shortcuts were caught, but 20 days later (`ad77361`). A session that has just done the
work is the worst-placed reader of its own claims: it shares the implementer's narrative, and if it
can also edit, review turns into more coding.

**When.** At the end of every phase; after each ADR (phases 2 and 4); before every release and every
merge to `main`; and within phases 2, 4 and 7 after each guard family or module is done, so a review
never covers more than a few days of work. Also review the 1A readiness gate and the 2A checkpoint
before dependent implementation begins. Not per commit.

**Who.** A fresh agent session with no editing role, e.g. a read-only review agent type, or a
session instructed not to modify tracked files. Prefer a different model from the implementer to
reduce correlated blind spots. It may build, run tests and run scripts. The reviewer does not fix
anything. Findings go back to an implementation session, then a re-review checks only the open
findings. Re-reviews within a phase may continue in the same review session, which keeps its
reproductions (phase 0 did); each phase starts a fresh one. Implement and review alternate until the
review passes; there is no round limit.

**Input.** Only the commit range, this plan (the phase's exit criteria, §5 principles, §7
signatures, the debt register) and the repository. The implementer's summary, commit messages and
status updates are _claims to verify_, not context.

**Checklist.**

1. Every exit criterion of the phase: verified, not verified, or contradicted, each with evidence
   (command and result, `file:line`, test name, CI run). Also every numbered item and sub-item of
   the phase: done (with commit), deferred (with the reviewer's decision, reason and target phase),
   or open. In phase 0 two undecided salvage candidates passed three rounds because only the exit
   criteria were checked.
2. Every completion claim in commit messages and docs in the range, checked against code (`grep`,
   type-checker, tests). Anything described as implemented must exist.
3. Build, check, lint and test from a fresh clone of the range's last commit after `pnpm install`
   (linked `node_modules` skip the `prepare` generators); GPU suites when the range touches WebGPU
   or WASM-threading code.
4. New or changed skips, tolerances, timeouts, early `return;` in tests, `catch` blocks, test-only
   hooks and tracing-context checks (§7 signatures, Appendix B commands).
5. Tests prove the capability on the target runtime: they fail without the change, and assert that
   the new path ran, not only that nothing threw.
6. Commit hygiene: subjects match content; version bumps contain only version bumps; no `WIP`.

**Output.** A short report committed as `remediation/reviews/<phase>-<n>.md`: verdict (pass / pass
with findings / fail), the exit-criteria and item tables, findings with severity and evidence, and a
final **Sign-off** section in which the reviewer either signs off the reviewed scope (naming the
items and range) or states that it does not. It is written for a human to read in about ten minutes.
The reviewer leaves the file uncommitted; the implementation session commits it unchanged. After a
failed round, the implementation session appends a response to that report, marked as claims for the
re-review. A scope exclusion requested by the human is stated in the report and carried to the next
review.

**Gate.** A phase is marked done in this file only with a link to a passing review report whose
Sign-off section the reviewer has signed. The same applies to releases. The reviewer's sign-off
covers the review's item table, deferrals and uncovered commits, not the implementer's summary. A
reviewer must not sign off a range it contradicts or could not verify. Deferrals are the reviewer's
call, recorded in the report with a reason and a target phase.

### Phase 0 — Freeze and quarantine (R8)

**Done 2026-11-08:** review round 3 ([phase-0-3.md](remediation/reviews/phase-0-3.md)) passed and
the human signed off, after deferring the two undecided salvage candidates of item 3 to phases 3
and 6.

1. Preserve the uncommitted June work, including the five files recovered from editor history, as
   [remediation/quarantine/wasm-async-wip.patch](remediation/quarantine/wasm-async-wip.patch) (base
   `fe2c9cc`; its header records provenance and verified state), committed through the normal
   pre-commit hook. No `--no-verify`: the patch is inert data. Restore the working tree to `HEAD`.
2. Rewrite `wasm-multithreading-plan.md` §1.1 so that it describes `HEAD` only.
3. Salvage from the patch **one item per commit, each with its own test**
   (`git apply --include=<path> remediation/quarantine/wasm-async-wip.patch`, or by hand), only
   these candidates:
   - `getWasmRoutineStaticParams` — real deduplication of routine static params out of
     `scan-plan.ts`; its backend side is in the patch's `wasm.ts`. Done in `dd66816`, which also
     fixed TriangularSolve in a compiled `lax.scan`.
   - WASM QR routine changes + `bench/qr.bench.ts` — keep only if the benchmark shows a win and the
     routine's scratch requirements are declared by the routine, not re-derived by every caller (in
     the patch, `scan-plan.ts` and `block-map-executor.ts` each compute `qrScratchSizes`). Deferred
     to phase 3 by the human (2026-11-08), to decide with QR lowering.
   - `clearCaches()` clearing WASM module/routine caches — decide deliberately and update the cache
     table either way. Deferred to phase 6 by the human (2026-11-08), with the cache table.
   - Explicitly **rejected**: `makeJaxprBodyDepth` and the backend branch in `qr()` (phase 3
     replaces them), gate removal, timed-poll worker loop, `_debug…ForTest` backend hooks.
   - **Deferred to phase 4** as input to the ADR, not merged before it: in-flight slot tracking,
     completion handles, `executeMegaModuleDeferred` / `dispatchAsync`, `autoSubmitSync`,
     `allowDeferredMegaModule`, the `Array.dispose()` reorder, workspace leasing (E2).
   - Every salvaged test imports through one module graph (no mixed `src`/`dist` imports).

   Delete the patch once phase 4 has consumed or rejected every part of it; it stays in history.

4. Add a short rule to `.github/copilot-instructions.md`: implementation sessions do not mark
   milestones done; they stop at "ready for review", and a separate review session follows
   [Review rounds](#review-rounds-every-significant-milestone).
5. Make the browser tests reach the GPUs (R18): no `DISPLAY`/`XAUTHORITY` for Chromium, one Vulkan
   driver per run, shared config for the default and GPU suites. Done 2026-11-08; it is a
   prerequisite for every later exit criterion that mentions WebGPU.
6. Pulled forward from phase 1 because the exit criterion needs a green `pnpm test`, which item 5
   made meaningful: fix the R19 failures. Done: the foriLoop rewrite (`8d7d5b9`); GPU calibration
   made opt-in (R22, `fb8ea22`); profiling refuses to start while tracing is active, as documented
   and tested, and no longer leaves bogus trace entries (`d2534a2`); the load-dependent timeouts
   were the shader-compile queue of Chromium's GPU process, shared by all concurrently running test
   files, so browser test configs run at most 4 files at once (`BROWSER_MAX_WORKERS`). With
   `scripts/gpu-test.sh` fixed (R18), the full GPU suites run for the first time since March: NVIDIA
   green; Intel has the four R24 failures, left for phase 1.
7. Stop the website from loading the GPU on page open (R25). Done: the homepage benchmark runs only
   when the visitor clicks "Measure on this device" and shows stored results until then; matrix size
   and batch length come from measured times instead of a vendor table (46–63 ms per submission;
   Intel 744 vs 749 GFLOP/s fp32 and 250 vs 264 fp16 against the old method); a deadline is checked
   before every submission and runs never overlap; no leaks.

8. Address review round 1 ([phase-0-1.md](remediation/reviews/phase-0-1.md)), pulling two phase 1
   items forward at the human's decision: choose the test GPU by the hardware present, not by
   installed driver manifests (finding 1); require a detected GPU's WebGPU and WebGL to initialize,
   failing the file otherwise (phase 1 item 3); run every test inside WebGPU error scopes so that
   invalid GPU work fails it, and fix the profiling QuerySet (finding 2, phase 1 item 6). Finding 3
   came from a checkout without `pnpm install`; the exit criterion now names the procedure. Round 2
   ([phase-0-2.md](remediation/reviews/phase-0-2.md)) failed on F2.1: errors raised while a test
   module loaded were only logged. After two failed rounds the human decided to fix it rather than
   defer it: the listener is attached when each device is created, `afterAll` flushes, and
   `test/webgpu-errors.test.ts` proves both the module-load and the in-test case. Round 3
   ([phase-0-3.md](remediation/reviews/phase-0-3.md)) passed.

After round 3 (now covered by [dependency round 1](remediation/reviews/dependencies-1.md)): in all
three rounds the reviewer could not run checklist item 7 as written (`history-churn.py` needed
matplotlib and wrote into `remediation/`) and re-implemented it; its `src` scope then read "zero
rework". Review mode (`--since`) now writes nothing, needs only Python, and also reports the `code`
scope. That scope shows phase 0's own fix-on-fix, all in the test harness: `43b61eb` redoes 21 lines
of `bef6bb1` (round 1, finding 1), `648d367` redoes 42 lines of `82bb49d` (round 2, F2.1), and
smaller ones in `8fbf8c8` (the wrong cause given in `fb8ea22`) and `a32d106`. Each was caught before
sign-off, the two largest by the reviews.

Also covered by that review: dependency updates between phases 0 and 1, at the human's request
(`a31d05b` within the declared ranges, then majors in groups). Vitest 5 runs benchmarks as tests, so
`test/setup.ts`'s per-test hooks reach them: `*.bench.ts` files get debug level 0 and a
slot-count-only leak check (`checkLeaks.start({ trackSites: false })`). Creation-site tracking made
a small eager chain ~2.5× slower; the count-only check measured within noise. Migrating the
benchmarks found R26–R28; running typedoc for the TypeScript update found R29. TypeScript stays at
6.0 until typescript-eslint, typedoc and svelte-check support 7.

**Dependency review:** round 1 found one new P2 regression: ESLint 10's per-file configuration
discovery left `lint:scan-dispose:website` without the root plugin. `3a2aaea` adds explicit root
configuration selection. [Round 2](remediation/reviews/dependencies-2.md) passes: both website
ownership commands exit 0 and independently reject undisposed scan outputs. Round 1's validation
scope and limitations carry forward; R24 and R26–R29 remain open. This does not reopen Phase 0 or
sign off Phase 1.

**Exit:** working tree clean; `pnpm build && pnpm check && pnpm test` green on
`wasm-multithreading`, in a fresh clone after `pnpm install`; a review report
`remediation/reviews/phase-0-<n>.md` passes (rounds 1 and 2 failed).

### Phase 1 — Ground truth: CI and test integrity (R9, R13, R16, R18–R21, R23, R24, R26)

[Partial review round 1](remediation/reviews/phase-1-partial-1.md) passes for `3a2aaea..0b3c72a`:
local gates and setup integrity are verified, hosted CI remains unverified, and Phase 1 is not
signed off. [Round 2](remediation/reviews/phase-1-partial-2.md) passes for `0b3c72a..83e0e32`: the
approved native GPU float32 trig contract resolves the four Intel R24 test failures, and the real
default/NVIDIA/Intel push gate passes. It does not improve runtime accuracy or sign off the phase.
The separate accuracy plan remains outside remediation.

[Round 4](remediation/reviews/phase-1-partial-4.md) verifies the item 7 cleanup and item 10 QR
checks in `9b880b5..e42a033`, but rejects item 8 for async typing, cross-realm promise sequencing,
and opposite backend deltas canceling. [Round 5](remediation/reviews/phase-1-partial-5.md) reviews
`9a19712..ea88cbf`: F2–F4 are signed off, but F1 remains open for ordinary structural thenables.
Item 8 and phase 1 are not signed off.

[Round 6](remediation/reviews/phase-1-partial-6.md) accepts the always-async fix and the 1A scope of
item 8, conditional on committing the reviewed snapshot. It also accepts the 1A/1B and 2A/2B
sequencing. It does not sign off 1A or phase 1. The first-slice audit below is a subsequent
implementation-session assessment, not part of that review. The accepted fix and unchanged review
were committed as `58fe24f` on 2026-11-10; that commit satisfies the review's commit condition.

#### Sequencing: 1A readiness and 1B completion

| Scope        | 1A: required before ownership implementation                                                                              | 1B: still required, not a blocker for starting phase 2                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Items 1–5, 9 | Reviewed setup, enforced backends/errors, local full-suite gates and a green hosted CI run for the readiness commit       | Maintain these gates throughout subsequent work                                                                                 |
| Item 6       | Preserve GPU error enforcement and accepted regular tape tracing                                                          | Complete specialized scan/block-map/scatter-add/copy tracing                                                                    |
| Items 7, 10  | Audit skips, tolerances and timeouts affecting the selected ownership slice; establish usable numerical assertions for it | Complete the suite-wide inventory, issue links, FD derivations and benchmark relocation                                         |
| Item 8       | Resolve round 5 F1 and retain the existing cross-area steady-state checks with caches intact                              | Integrate steady-state coverage into the remaining per-area suites; phase 2 adds its own path-specific cases before each change |
| Items 11–12  | Not ownership prerequisites                                                                                               | Website opt-in work and TTS benchmark restoration; website work must precede public deployment of the affected page             |

Do not treat a green slot-count smoke test as an ownership proof. Phase 2 must additionally test
alias lifetime, retained-reference release and failure cleanup. New unsupported cases are not
silently moved to 1B: a missing oracle for the selected slice blocks that slice.

#### Item register and implementation evidence

1. Add a GitHub Actions workflow running `pnpm install`, `pnpm build`, `pnpm check`, `pnpm lint`,
   `pnpm test` (non-GPU) on every push and PR, including pushes to `wasm-multithreading`. GPU suites
   stay local: add a pre-push hook that runs the full default suite and the full suite of the
   detected GPU (`scripts/gpu-test.sh`). The pre-commit hook runs only 9 test files, so no gate saw
   the R19 failures, and none noticed that `pnpm run test:gpu` had run no tests since March (R18).
   Implementation now adds `.github/workflows/checks.yaml` for all pushes and PRs, using the full
   CPU/WASM suite in Chromium (a Node-only probe failed on OPFS and WASM fixture assumptions). The
   pre-push hook runs `test:push`: full default suite, then full GPU suites through
   `scripts/gpu-test.sh`. The human chose **all detected supported GPUs**, not just the preferred
   one, and an explicit CPU/WASM-only fallback when none is detected. GPU failures are not
   suppressed; Intel R24 blocked pushes at this point. Local validation: CPU/WASM 2265 passed / 1071
   skipped; default 3322 passed / 14 skipped; NVIDIA 3375 passed / 14 skipped; Intel 3371 passed /
   14 skipped / 4 unchanged R24 failures. The real push gate ran all three local suites and exited 1
   on Intel. Five push-gate tests cover multi-GPU selection, required backend enforcement, failure
   propagation, CPU-only fallback and a terminated runner. Verified by partial review round 1; no
   GitHub CI run was claimed in that review. **2026-11-10:** hosted CI now passes at `4a00f82`; see
   the readiness evidence below. This closes the missing-run evidence, not the 1A review gate.
2. Make the setup honest (R16): the website's `prepare` script must fail when `svelte-kit sync`
   fails instead of `|| echo ''`; give `bench/` a `tsconfig` that does not type-check against a
   stale `dist/`, or build first in `pnpm check`; move `repro-stub`/`debug-wgsl` out of `test/`.
   Implementation verified by partial review round 1: remove the sync suppression and resolve the
   root package to `src/index.ts` in the existing TypeScript paths, matching the auxiliary packages'
   source aliases. No separate bench config or prerequisite build is needed for type-checking. A
   compiler file-list check includes benchmarks and reads zero workspace `dist/` declarations. Real
   website prepare and type-check pass; an intentional sync failure exits 73 with the new recipe,
   while the old suppressed recipe exits 0. Preserve the diagnostic's assertions rather than move
   them outside the gates: all six `repro-stub` cases (N=65,128,200, both directions, m=5) now use
   the existing associative-scan regression body. The two N=65 cases were duplicates. No tolerance
   changed; the pre-commit GPU list runs the consolidated test instead of the deleted diagnostic.
   `debug-wgsl` was already absent; remove its stale exclusion. Default suite: 3326 passed / 14
   skipped; associative-scan file: all 65 passed on each GPU. This is implementation evidence, not
   phase sign-off.
3. Make missing backends loud (R18). Done in phase 0, item 8.
4. Fix the five failures the skips had hidden (R19). Done in phase 0, item 6.
5. Decouple backend initialization (R20): one backend's failure must not affect another's; add a
   test that injects a WebGL failure and asserts WebGPU still initializes. Verified by
   [round 3](remediation/reviews/phase-1-partial-3.md) and signed off by the reviewer (2026-11-09):
   serialize backend creation across both a single `init()` and concurrent calls. Log unexpected
   creation errors per backend and continue initializing the others; normal unavailability still
   omits that backend from the returned list. A fresh-worker regression uses the real WebGPU
   adapter/device while injecting unavailable or throwing WebGL context creation. It also simulates
   the observed interference (an overlapping WebGL failure makes the adapter unavailable), verifies
   no overlap, checks the diagnostic, and requires one WebGPU request even across concurrent calls.
   All three cases pass on NVIDIA and Intel; CPU-only CI explicitly reports these hardware-dependent
   cases as skipped.
6. Make WebGPU errors fail tests (R21): done in phase 0, item 8. Remaining: make the command tape's
   passes traceable (R23), and test that a `jit` call produces trace entries. Verified by
   [round 3](remediation/reviews/phase-1-partial-3.md) and signed off by the reviewer (2026-11-09):
   regular kernel/routine tape dispatches now retain their source metadata and share tracing with
   eager execution, without bypassing the tape. A warmed JIT call produces one real
   `performance.measure` entry with its operation properties and WGSL source; a multi-pass sort
   contributes one measurement across its passes. Middle passes must omit `timestampWrites`, not
   supply a descriptor with both indices undefined. Query batches belong to their encoder and
   resolve alongside their writes before submission, so stopping inside `withBatch` or exceeding
   2048 begin/end pairs cannot resolve unwritten queries. The stress regression requires all 2049
   entries and nonzero measured work, not merely acceptance of zero-filled timestamps. A raw WebGPU
   diagnostic reproduced partially converted/zero timestamps on Intel when resolution used a
   separate command buffer; resolving in the writing encoder fixes that case on both GPUs. Resources
   are released after readback (including rejection) or abandoned tape encoding, without allocating
   a replacement empty batch. Profiling still suppresses tracing of both eager and tape dispatches.
   Focused initialization/tracing/profiling/batching coverage: 22 tests pass on each GPU. Dedicated
   scan, block-map, scatter-add and copy dispatch tracing remains open; this is not a claim that
   every specialized pass is now traced.
7. Inventory every skip, early `return;`, device-specific tolerance, timeout override and config
   exclusion (commands in Appendix B). For each: keep with a written reason and an issue link, or
   turn it into a failing test plus an issue. Replace silent `return;` with `ctx.skip()` so skipped
   coverage shows up in reports. Include files that test nothing: `test/test_cost.test.ts` defines
   no test and only prints, and passes because the config sets `passWithNoTests: true`;
   `test/timing-tiled-matmul.test.ts` is a 35 s benchmark with no assertions and a 300 s timeout in
   the default suite (its header says to run it from `tmp/`; `bench/tiled-matmul.bench.ts` exists).
   **Implementation, partial (verified by round 4):** the 36 silent `if (!hasWebGPU) return;` in
   `block-map-jit.test.ts` and `lax-scan.test.ts` are now `skip()`, so a CPU-only run reports them
   as skipped (default suite unchanged: 3342 passed / 14 skipped). `test_cost.test.ts` (printed
   only) is replaced by `cost-model.test.ts`, five assertions on `evaluateTotalCost`'s documented
   cliffs and normalization; `timing-tiled-matmul.test.ts` is deleted (the bench exists);
   `passWithNoTests` is removed from both configs, so a filter that matches nothing fails. The
   device-specific QR tolerances in `lax-linalg.test.ts` (WebGPU 0.1/0.05 gradient atol, CPU/other
   JVP split) were not needed: measured on NVIDIA and Intel the gradient checks pass at 1e-3, and
   the JVP check now uses central differences in f32 on every device with a derived 1e-3 bound. The
   remaining 14 skips of the default suite: 6 WebGL-only (Cholesky/TriangularSolve routines, int32
   floor semantics, 3 + 2 + 1 sites, each commented), and the 8-test orchestrator-worker suite that
   never runs in the browser suite because `crossOriginIsolated` is false (R7, phase 4). **Still
   open:** an issue link for each kept skip; the 30 s timeout overrides (7 sites) lack a written
   reason; the four benchmark-shaped tests (`scan-bench`, `bench-phase3`, `bench-assoc-scale`,
   `gpu-bench`, ~1,000 lines, one `expect` each) are excluded from the default suite but run in the
   GPU suites with `testTimeout: 300000`; they belong in `bench/`.
8. Add a **steady-state leak mode** to `checkLeaks`: run the body twice and require zero slot growth
   on the second run _without_ flushing JIT caches. This catches the class of leak that `aa85998`
   hid, while tolerating warm caches. **Implementation (F1 response awaiting review):**
   `await checkLeaks.steadyState(body)` runs a sync or async `body` once to warm up, then again, and
   reports any slot-count change of the second run with caches left intact; it does not touch an
   active `start()`/`stop()` session. `test/steady-state-leaks.test.ts` applies it on every device
   to linalg (cholesky, qr, lu, triangularSolve, jit), scan (eager, jit, jit∘grad), foriLoop (eager,
   jit, grad) and the transforms (grad, valueAndGrad, hessian, jacfwd, jvp, vjp, linearize,
   jit∘vmap∘grad): 48 passed, 4 skipped (no routines on WebGL; round 4 F4 corrects the former count
   of 44). `check-leaks.test.ts` proves the mode catches a per-run leak, a leak into an unflushed
   cache (a fresh `jit` per run, which `stop()` would free), and awaits async bodies. No leak was
   found in these sampled cases; this is not comprehensive coverage of those areas. Not yet applied
   to the existing per-area test files. **Round 4 response (F1-F3):** infer the return type from the
   callback's result, including non-void async and possibly async callbacks; await promises and
   thenables independently of their realm before measuring or reporting; sum absolute per-backend
   slot changes, preserving signed deltas in the summary, so allocations and releases cannot cancel.
   The last change also makes a decrease a positive non-steady result rather than a negative
   `leaked` value. Focused regressions reproduce the original type errors, premature cross-realm
   completion and cancellation, and cover mixed sync/async runs, ignored sync results, thenables,
   rejection in either run and a decrease-only body. The two leak-check files pass 53 tests / 26
   skipped on CPU/WASM and 75 / 4 skipped on each of NVIDIA and Intel; build, source type-check and
   targeted lint pass. The full default suite passes 3406 tests / 18 skipped across 92 files. See
   the [migration notes](FEATURES.md) for updated result types and report semantics. **Round 5
   disposition:** cross-realm sequencing, backend-delta reporting and the coverage count are
   verified fixed. A structural thenable whose `then` returns `void` still receives a synchronous
   report type despite being awaited at runtime in the reviewed implementation. **Implementation
   response, 2026-11-09 (not sign-off):** replace conditional return typing and manual thenable
   detection with one `async` function returning `Promise<LeakReport>` and sequential `await body()`
   calls. All callers now await it, including synchronous bodies; throws from either run reject the
   promise. Public migration notes cover result types, error handling and shared-resource lifetime.
   Structural thenables whose `then` returns `void`, mixed sync/thenable runs and callbacks typed as
   `() => void` have type regressions. Against the old source the updated tests produce six type
   errors and three runtime failures; all pass with the new source. Build/check/lint and formatting
   pass. Focused CPU/WASM: 59 passed / 26 skipped; each GPU: 81 passed / 4 skipped. Full CPU/WASM:
   2287 passed / 1143 skipped. The real local push gate passes: default 3412 passed / 18 skipped,
   NVIDIA and Intel each 3462 passed / 18 skipped. See the
   [implementation response](remediation/reviews/phase-1-partial-5.md#implementation-response-2026-11-09)
   for scope and limitations. Independent re-review and hosted CI remain open; this is not 1A or
   phase 1 completion. **Round 6:** the response is accepted and committed as reviewed in `58fe24f`.
   The review report's 1A table incorrectly assigns 48 passed / 4 skipped to the per-area file in
   CPU-only mode: that file has 26 passed / 26 skipped there (the combined two-file CPU count is
   correctly reported as 59 passed / 26 skipped). The report is preserved unchanged; this count
   clarification does not extend its sign-off. Existing per-area suite integration remains open
   under 1B.
9. Decide R24 (own implementations vs spec-derived tolerances for GPU transcendentals). **Human
   decision, 2026-11-09:** accept native GPU float32 accuracy, following
   [WebNN #914](https://github.com/webmachinelearning/webnn/issues/914). Implementation verified by
   [round 2](remediation/reviews/phase-1-partial-2.md): `sin`/`cos` tests use WGSL's `2^-11`
   absolute bound only inside `[-pi,pi]`; `atan` uses 4096 ULP of the reference value. The current
   WGSL `asin` bound is the worse of `6.81e-5` absolute and inherited `atan2(x,sqrt(1-x*x))` error,
   not simply 4096 ULP. For the existing corpus's non-endpoint values (`|x| <= 0.7`,
   `1-x*x >= 0.51`), `8*epsilon` covers the inherited expression's arithmetic perturbation of the
   angle; add its 4096-ULP `atan2` error. The library's `acos = pi/2 - asin` adds less than
   `3*epsilon` absolute from constant/subtraction rounding. Here `epsilon = 2^-23`. The library's
   `atan2 = 2*atan(q)` doubles the native atan error. For the existing moderate finite corpus,
   squares/add/sqrt perturb `r` by less than `7*epsilon` relatively, forming `r+|x|` by less than
   `9*epsilon`, and reciprocal/multiply produce `q` with less than `14*epsilon` relative error.
   Since the half-angle formula's sensitivity to relative `q` error is at most 1, `16*epsilon`
   bounds the propagated absolute angle error. Endpoints, axis values and infinity retain
   compatibility smoke assertions without claiming general WGSL guarantees. New tangent tests
   propagate `sin/cos` error and reciprocal/multiply rounding only for `|cos(x)| >= 0.5` on all
   devices. An exploratory WASM float32 sample at `x=-1.5` differed by `9.08e-6` at an output of
   `-14.1014`; no WASM threshold was relaxed to accommodate it. WebGL adopts the same budget by
   project policy; GLSL ES leaves trig accuracy unspecified. Boundary controls require rejection
   above the allowed error and of NaN, and verify that CPU/WASM and non-f32 metadata still use the
   original 5-decimal requirement. Runtime lowering, skips, float16/float64 and gradient thresholds
   are unchanged. See [accuracy notes](FEATURES.md#gpu-float32-trigonometric-accuracy). Local
   validation: the former four Intel failures reproduce before the change. The updated focused
   trig/control selection passes all 29 selected tests on each GPU. CPU/WASM full suite: 2270 passed
   / 1079 skipped. The real `test:push` gate now exits 0: default 3335 passed / 14 skipped; NVIDIA
   and Intel each 3382 passed / 14 skipped, with WebGPU/WebGL required. Type-check and targeted lint
   pass. These are verified by round 2, not a Phase 1 sign-off or evidence of hosted CI.
10. Re-derive FD tolerances: run gradient checks in f64 on CPU/WASM with tight tolerances; for f32
    WebGPU, derive tolerances from the problem's conditioning and document the derivation inline.
    **Started:** the QR JVP check in `lax-linalg.test.ts` has a derived bound and no device split;
    the other eight finite-difference sites in that file keep `rtol: 0.05` and `eps = 1e-4` (f32
    rounding noise there is about `4 |f| 2^-23 / eps`, up to ~0.07), and the rest of the suite is
    not yet reviewed.
11. Website pages start GPU work only when the visitor asks (R25, remainder): `/bench/results`,
    linked from the homepage, runs matmuls up to 2048² in f32 and f16 as soon as it opens.
12. Re-add a TTS per-step benchmark (R26) that imports the website's Pocket TTS model code and runs
    it with random weights; it must pass the benchmark leak check.

**Continuation validation (2026-11-09, verified by round 3):** build, source type-check, full lint
and touched-file formatting pass. The full CPU/WASM suite passes 2270 tests with 1082 explicit skips
(three new skips require real WebGPU). The real `test:push` gate exits 0: default 3338 passed / 14
skipped, NVIDIA and Intel each 3388 passed / 14 skipped, with WebGPU/WebGL required in all three
runs. Compared with round 2, the default suite gains the three initialization regressions; each GPU
suite also gains three tape-tracing regressions. No existing test, tolerance, timeout or backend
requirement was weakened. Hosted CI and phase exit criteria remain unverified or unmet as recorded
above; no phase sign-off is inferred.

**1A exit (independent readiness review required):** round 5 F1 resolved with type and runtime
regressions; build/check/lint, default suite and both full GPU suites green with required backends
enforced; hosted CPU/WASM CI green for the readiness commit (record the run URL and SHA); existing
cross-area steady-state checks green without cache flushing; the first ownership slice named, with
its relevant test gaps and oracle requirements recorded. Previously accepted work is preserved; the
reviewer checks the readiness scope and carries every 1B item forward explicitly.

**Full phase 1 exit, after 1B:** all numbered items complete or explicitly deferred by review with a
reason and destination; CI and local gates remain green; every skip/tolerance/timeout has a reason
and issue; steady-state coverage is integrated into linalg, scan, foriLoop and transform tests. A
passing 1A report alone must never be recorded as phase 1 completion.

#### 1A readiness evidence (2026-11-10; ready for review)

The accepted implementation and unchanged round-6 report are committed in `58fe24f`. The plan
revision and first-slice audit are committed in `4a00f82`. Both are pushed to
`origin/wasm-multithreading`; `main` and releases are untouched.

| Requirement                             | Evidence                                                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hosted checks                           | [Checks run 42759479814](https://github.com/hamk-uas/jax-js-nonconsuming/actions/runs/42759479814), attempt 1, **success** for full SHA `4a00f82e08e40b32fab7ab90537f3f528d549a82`                            |
| Hosted validation scope                 | Fresh checkout/install, Chromium installation, build, type-check, lint, ESLint-plugin tests, push-gate tests and CPU/WASM suite; CPU/WASM result: **2287 passed / 1143 skipped**, 89 files passed / 3 skipped |
| Local pre-push gate for the same commit | Default: **3412 passed / 18 skipped**; NVIDIA and Intel: **3462 passed / 18 skipped each**, with WebGPU/WebGL required                                                                                        |
| Full local CPU/WASM rerun               | **2287 passed / 1143 skipped**, matching hosted CI                                                                                                                                                            |
| First ownership slice                   | Named and audited below; no production ownership change is included                                                                                                                                           |

This evidence is recorded by the implementation session, not a phase sign-off. The review range
starts at `0891af9`; round 6 already accepts the checker snapshot now in `58fe24f`. The next review
must assess the subsequent first-slice/oracle audit, verify the hosted run against its SHA, and
decide 1A readiness while carrying every 1B requirement forward. Only after a passing readiness
review may 2A characterization/ADR work proceed; production ownership changes still require ADR
review.

#### First-slice readiness audit (2026-11-10; awaiting review)

**Selected slice:** body-created constant capture, from the builder's retained reference through
literal inlining, the transient `ClosedJaxpr`, cache-owned retention and final release. The first
family to replace is **builder-produced constant environments labeled borrowed but disposed as
owners**, including their `mapJaxpr` aliases and tolerant teardown. The intended deletion is that
family's reliance on the `UseAfterFreeError` swallow, not a renaming or blanket removal of all
creation-ref tracking. R4's remaining states and unrelated PE guard families stay in 2B unless the
reviewed contract proves them inseparable. This is a scope and oracle audit, not the ADR or
authorization to change ownership semantics.

Current ownership path, inspected against `0891af9` (unchanged by the checker fix):

| Boundary                     | Current owner/transfer/release mechanism                                                                                                                                                       | Requirement for the reviewed contract                                                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Concrete creation            | `Array` constructor marks PE/body birth; creator has the initial reference (`src/frontend/array.ts:252–281`)                                                                                   | Distinguish creator-owned values, raw-value temporaries and graph-only intermediates; a JS alias is not automatically another owner          |
| Capture                      | `getOrMakeConstTracer` takes one same-wrapper `.ref` per builder; raw creation is balanced immediately, body-created arrays may enter `constsNeedingCreationRefBalance` (`jaxpr.ts:1037–1073`) | Captured ownership must survive creator cleanup and be released exactly once, including a failed trace                                       |
| Build and inline             | `JaxprBuilder.build` constructs a borrowed wrapper; `_inlineLiterals` releases inlined builder refs and passes remaining values onward (`jaxpr.ts:1164–1225`)                                  | Specify transfer of the remaining environment and what happens if inlining/build fails                                                       |
| Simplification and balancing | `makeJaxpr` snapshots `refCount > 1` before build, balances creation refs, then calls `mapJaxpr`; `mapJaxpr` shares the const array without retaining (`jaxpr.ts:918–919,1901–1960`)           | Define whether the old wrapper transfers or borrows after a map; two wrappers must not imply two owners                                      |
| Cache handoff                | `buildCachedJaxprArtifact` retains via `OwnedConstEnv` then disposes the transient wrapper; concrete values get independent wrappers, tracers get `.ref` (`jaxpr.ts:945–984`)                  | Preserve independent cache ownership and clean up partially completed retention or handoff on failure                                        |
| Evaluation and teardown      | `evalJaxpr` borrows inputs, retains pass-through results and cleans intermediates/Lits; JIT wrapper disposal frees cache environments (`jaxpr.ts:774–885,2050–2072`)                           | Disposing a result or one environment must not kill another owner; test successful reuse and error cleanup without relying on cache flushing |

`makeJaxpr` currently restores the tracing flag in a `finally`, but does not put all builder-held
constants under a corresponding failure cleanup. `OwnedConstEnv.retain` maps retains without
rollback if a later retain throws. These are inspection-based failure-path risks to characterize,
not reproduced defects or justification to add broad catches. Nested PE/JVP/cache consumers must be
included in the ADR before altering the handoff.

**Existing evidence and gaps:**

| Evidence                                                                | What it establishes                                                               | Missing oracle before changing this path                                                                                                            |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ownership-characterization.test.ts`, ClosedJaxpr/nested-builder groups | External refcounts return to baseline; inner disposal leaves a shared const alive | Body-created non-scalar capture, explicit creator cleanup, mapped-environment transfer and two genuinely distinct retained owners                   |
| Same file, literal and named-owner groups                               | Current balancing states and several scalar/gradient results                      | Assert whether a value was actually inlined versus retained; assert final release rather than only the state name                                   |
| Same file, cache groups                                                 | Repeated values and cleanup with explicit `clearCaches` or the test harness flush | Steady-state runs with caches intact, then separately asserted final environment teardown                                                           |
| `jit-cache-const-ownership.test.ts` (3 cases)                           | Nested scan/foriLoop cache clearing does not throw                                | Nonzero/nonuniform numerical results, input survival and exact lifetime checks; two cases use zero initial vectors and none asserts computed values |
| `steady-state-leaks.test.ts`                                            | Cross-device sampled slot balance with caches intact                              | References retained on an existing live slot, aliases, tracing/evaluation failure and post-failure reuse                                            |

**Skip/tolerance/timeout audit:** neither of the first two files has a skip, an early-return
coverage escape, a per-test timeout override or a config exclusion. They also have no `deviceSuite`
or explicit device selection: the default is WASM (`src/backend.ts:146–153`). Thus the observed **79
passed** under CPU-only, NVIDIA and Intel configs is the same WASM coverage in three launch
environments, not three-backend ownership verification. New path-specific cases must use
`deviceSuite` and cover CPU/WASM/WebGPU/WebGL where the operations exist. Default and GPU configs
currently allow 10 s and 300 s per test respectively; keep the small ownership cases under the
ordinary 10 s limit rather than relying on the GPU-suite allowance.

Most value assertions use `allclose` defaults (`rtol=1e-5`, `atol=1e-7`); one decimal scan
recurrence uses `atol=1e-5`. There are no finite-difference checks in these two files. For the new
capture cases use small exactly representable f32 constants and closed-form add/multiply/gradient
results, with exact expected values and exact ref/slot deltas; no device-specific tolerance is
needed. Existing nonlinear or matrix checks retain their current scope, not a new numerical-accuracy
claim.

The provisional eager-scan case at `ownership-characterization.test.ts:220` only asserts
`leaked > 0` when the count is already positive. It is not a leak oracle. The nested optimization
cases at lines 865/888 assert only that magnitudes shrink. Neither may serve as acceptance coverage
for this slice. The `ClosedJaxpr.dispose() continues after UseAfterFreeError` case deliberately pins
the old workaround; replace its expected behavior only together with the reviewed ownership
contract, preserving a test for the underlying alias/cleanup scenario. The two `jit(f)` calls in the
sequential-cache test deduplicate by function identity and do not prove independent owners.

**Next step after 1A review:** write the ADR and add the missing characterization cases before any
production ownership edit. The cases must prove creator/borrower/owner distinctions, scalar inlining
and non-scalar retention, identity outputs, two independent captures, failure cleanup and subsequent
successful reuse in eager/JIT/grad/nested scan/foriLoop. Use negative controls for each new
capability; do not convert the vacuous checks into passing tests by weakening their oracle. The
first-slice test gaps are named, not resolved by this audit.

### Phase 2 — Ownership: one semantics in eager and traced code (R1–R4)

**Starting facts, not a diagnosis:** `getOrMakeConstTracer` already takes `.ref`.
[`Array.ref`](src/frontend/array.ts) increments the same wrapper's count and returns that wrapper;
`retainedHandle` creates a separate wrapper. [`JaxprBuilder.build`](src/frontend/jaxpr.ts) labels
its result borrowed, yet borrowed `ClosedJaxpr.dispose()` releases constants and suppresses
`UseAfterFreeError`. Creation-ref balancing and partial-eval cleanup then infer which references
remain. Adding another `.ref`, renaming borrowed to owned, or replacing every `.ref` with
`retainedHandle` is not a demonstrated solution; identity-based caches and tracer wrappers also
depend on these choices.

#### 2A — One end-to-end ownership slice

1. **Bound the scope.** Start with body-created constant capture: construction, builder retention,
   literal inlining, the resulting constant environment, evaluation/cache retention and teardown.
   Identify one dependent disposal-guard or balancing family to eliminate, including its nested
   transform consumers. Record the current owners, transfer points and release sites before editing.
   Preserve unrelated kernels and execution paths.
2. **Characterize before changing semantics.** Reuse
   [ownership-characterization.test.ts](test/ownership-characterization.test.ts) and
   [jit-cache-const-ownership.test.ts](test/jit-cache-const-ownership.test.ts) where appropriate.
   Cover eager/JIT execution and relevant grad and nested scan/foriLoop compositions, with numerical
   assertions as well as ownership checks. Include early disposal of a creator's local value after
   capture, shared constants and aliases, caller inputs surviving result disposal, repeated warm
   execution without cache flushing, and exceptions during tracing or evaluation followed by
   cleanup. Run retained values again after another owner is released. Distinguish borrowed access
   from a genuine second release of an owned handle.
3. **ADR before implementation.** Specify whether each boundary borrows, retains an independent
   ownership token or transfers ownership, and give every owned token one release path, including
   exceptions. Define wrapper identity versus storage identity, literal-inlining releases,
   environment copying, cache ownership, and interaction with partial evaluation. Borrowed wrappers
   must not release someone else's ownership. Distinguish graph-only intermediates from concrete
   resource-owning arrays; the contract is equal user ownership obligations, not identical physical
   allocations in eager and JIT execution. Review the ADR before implementing it. Keep it under
   `remediation/` until R29 separates generated docs from hand-written documents.
4. **Implement and delete together.** Update the chosen ownership path and its dependent callers as
   a coherent change, then remove the guard/balancing family it makes unnecessary. Do not remove
   safety checks first and repair the resulting crashes with broader catches or tracing flags.
   Preserve the non-consuming user API; document any intentional public contract change with
   deterministic migration mappings and obsolete workaround signatures.

**2A checkpoint, before widening scope:** an independent review verifies the new contract, targeted
negative controls, full default/dual-GPU regressions and explicit release/error-path assertions.
Zero slot growth alone is insufficient: extra references to already-live slots can evade it. The
report names the old guard/state/balancing family actually removed and explains why its replacement
needs fewer ownership exceptions, rather than merely moving them.

- **Pass:** continue to 2B using the reviewed contract.
- **Not established:** revise the ADR or replace the bounded subsystem on this branch before
  extending the change. Do not accumulate more ownership exceptions to obtain a green suite.
- **Broader restart proposed:** compare candidate baselines using the same characterization cases
  and state which supported capabilities would be dropped or rebuilt; obtain the maintainer's
  decision. A failed slice does not authorize resetting history.

There is no arbitrary review-round limit. Repeated failure of the same invariant is a reason to
change the design, not weaken the invariant or start another large milestone dump.

#### 2B — Extend the proven contract

1. Characterize each remaining guarded family before changing it, then apply the contract and remove
   the corresponding `arange_*` gate, `insideAbstractTrace()` disposal guards and
   `allowNestedBodyCleanup`. Review by coherent ownership family, not by isolated guard count.
2. Remove ownership-error suppression, including broad cleanup catches, and the `rc > 0` tolerance
   in `[Symbol.dispose]`; genuine double-dispose must throw. Fix ownership transfers before removing
   these masks, and retain regressions for their original failure cases.
3. Delete creation-ref balancing and `#creationRefOwner` states made redundant by the contract. Any
   surviving state must represent a named owner with an explicit transfer/release rule, not an
   inference from tracing context or a target state-count quota.

**Full phase 2 exit:** the contract covers all R1–R4 paths with no suppressed ownership errors;
`git grep -nE "(insideAbstractTrace|inMakeJaxprBody)\(\)" -- src/frontend/linearize.ts src/library`
finds no disposal decisions dependent on tracing context (inspect helpers as well as direct hits);
characterization, explicit release checks and steady-state tests are green, as are the default and
both GPU suites. Run the downstream `dlm-js` suite against a local build before release; an
unavailable downstream checkout is an unverified gate, never a pass. Review and sign-off are still
required after 2B; passing 2A alone closes no full phase.

### Phase 3 — Lowering choices belong to the compiler (R5, R27, R28)

Library functions currently pick an algorithm based on tracing context or backend (unrolled Cholesky
/ TriSolve / Householder QR when `inMakeJaxprBody()`, `tiledMatmul` when WebGPU + traced + ≥ 512).
This makes semantics depend on how a function is called and scatters cost decisions.

1. Library functions emit one primitive (`Cholesky`, `TriangularSolve`, `QR`, `Dot`).
2. Add jaxpr rewrite passes alongside `rewriteConvToBlockMap` / `rewriteForiLoopToBlockMap` in
   `jit.ts`: expand small linalg primitives to pure-primitive form when they sit inside a fusable
   body (block_map / compiled-loop scan) and size ≤ limit; rewrite large `Dot` to the tiled
   block_map path on WebGPU. Decisions go through `evaluateTotalCost`.
3. Delete the context checks from `src/library/` and the `makeJaxprBodyDepth` idea with them.
4. Decide the quarantined WASM QR routine and `bench/qr.bench.ts` (deferred from phase 0, item 3):
   keep only if the benchmark shows a win and the routine declares its own scratch requirements.
5. Rewrite passes take their settings as parameters, not module state, and every setting that
   changes the compiled program is part of the compile cache key (R27: `_skipForiRewrite`).
6. Bisect the WebGPU conv slowdown (R28) with the A/B in `bench/conv2d.bench.ts`, then fix the
   block_map conv path or let `evaluateTotalCost` choose between it and generic-dot.

**Exit:** library functions no longer choose algorithms by backend or tracing context; numerical and
ownership contracts are unchanged, and tests assert the intended lowering actually ran.
`git grep -nE "getBackend\(\)|defaultDevice\(\)|inMakeJaxprBody\(\)|insideTrace\(\)" -- src/library`
is an inventory to inspect, not a demand to hide legitimate device access behind helpers.
Fusion/dispatch-count tests remain green; compare linalg performance on both GPUs and WASM with the
recorded baseline, investigating regressions beyond measured run-to-run noise.

### Phase 4 — WASM multithreading: containment first (R7, R8)

1. **Status truth now:** document in `FEATURES.md` and the plan that the worker pool, orchestrator,
   parallel mega-module and parallel Kogge-Stone run only in worker contexts today; mark them
   experimental. Keep the `#canSpinWaitWorkers` gate. Review the existing implementation and fix
   defects in its supported contexts; containment does not excuse known cleanup or correctness bugs.
2. **ADR before expansion:** default to retaining the gated implementation and leaving new async
   execution/workspace leasing outside remediation. Record the supported contexts and limitations.
   If expansion is explicitly chosen after the ownership contract is reviewed, choose between async
   main-thread completion (`Atomics.waitAsync`, pending-work ownership across backends) and hosting
   the runtime in a worker. Record why the alternative was rejected.
3. For that expansion, replace timed polling with an explicit control protocol (e.g. a
   `STATE_CONTROL` that returns workers to their event loops on demand), and prove registration,
   cancellation/failure cleanup and completion ownership rather than restoring the June patch.
4. Prove the retained feature in its declared worker contexts: assert actual worker execution, not
   merely successful fallback. Test that the browser main thread retains the non-blocking gate.
   Claim main-thread parallelism only after a browser test proves worker execution and main-thread
   responsiveness there; this extra capability is not required for containment.
5. Node: either a `worker_threads` adapter or an explicit "unsupported" in docs and capability
   detection.
6. Preserve E1 arena subject to review of its lifetime rules. E2 workspace leasing stays quarantined
   unless expansion is chosen; re-implement it only with tests and an accepted ownership/async ADR.

**Exit:** the reviewer accepts either verified containment with explicit capability limits and
expansion recorded as post-remediation feature work, or the implemented expansion and its ADR with
target-runtime tests. In both cases the phase's review covers `0f644bc..fe2c9cc`, the multithreading
commits made before remediation that no review has covered (see Branches), including actual worker
execution and cleanup. Gated or unsupported capabilities must not be reported as shipped on the
browser main thread. The quarantined patch is not promoted wholesale.

### Phase 5 — Effect system: finish it or delete the unused half (R6)

1. Keep `Mutate`/`Alloc`/`Borrow`; they drive `effectDrivenAllocate` and DUS/scatter-add zero-copy.
2. Either annotate real consumers with `Consume` and add a test where rule 1 fires, or delete
   `Consume` and rule 1. Recommended: delete, since operations are non-consuming by design.
3. Enable the verifier for the whole test suite in `test/setup.ts`; measure the overhead.

**Exit:** every rule in `verifyJaxprEffects` has a test that triggers it; the verifier runs in CI.

### Phase 6 — Documentation truth and API surface (R10, R15, R22, R29)

1. Split `.github/copilot-instructions.md` (1,198 lines) into short agent rules plus
   `docs/architecture/` ADRs; benchmarks and roadmaps move out of agent instructions.
2. Fix known drift: `recycleBuffers()`, the cache hierarchy table, multithreading status, plan
   status sections.
3. Add narrowly scoped checks for documented public exports and referenced file paths, reusing
   TypeScript/export information where practical. Back-ticked text also contains examples and
   proposed designs; do not make a blanket identifier regex and a growing allowlist the truth gate.
   Verify behavioral claims with linked contract tests or executable examples. Run mechanical checks
   in pre-commit and CI without building a general-purpose documentation framework.
4. Move test hooks off `src/index.ts` to a `./testing` sub-path export (or reach them through the
   Vitest source alias), so the public API contains only public API.
5. Decide whether to keep GPU calibration (R22), from `bench/calibration.bench.ts` run on NVIDIA,
   Intel and the AMD laptop, together with the run-to-run spread of the measured values. Keep it
   only if it wins clearly on some GPU and its values are stable; otherwise delete
   `src/microbench.ts`, the calibrated part of the JIT cache key and `calibrateGpu()`, with a
   migration note (callers delete their `calibrateGpu()` call).
6. Decide whether `clearCaches()` also clears the WASM module and routine caches (deferred from
   phase 0, item 3; the quarantined patch does it), and make the cache table match.
7. Give typedoc an output directory of its own (R29) before item 1 adds hand-written docs, and move
   `docs/array-manipulation-patterns.md` out of the generated tree.

**Exit:** targeted documentation checks green in CI; architecture and capability claims match
reviewed implementation and tests; no internal test hooks on the public entry point, including
renamed hooks without an underscore. Symbol existence alone does not establish a truthful claim.

### Phase 7 — Decompose monoliths (R14, R17)

Only after phases 2–3, so code is not moved while it is being rewritten.

- `linearize.ts` → partial eval, transpose rules per primitive family, PE cleanup.
- `jit.ts` → compile, rewrite passes (`conv`, `fori`, `linalg`, `dot`), allocation, execution.
- `wasm.ts` / `webgpu.ts` → core backend vs scan / assoc-scan / block-map / mega-module dispatch.
- Extract cohesive responsibilities behind explicit interfaces, with ownership remaining visible at
  module boundaries. Do not split files merely to meet a line limit or relocate the same hidden
  global coupling.
- Keep mechanical moves separate from behavior or type-safety fixes, suite green at every step.
  Remove unsafe casts with proper types and guards, not equivalent assertions under another
  spelling.

This also keeps fork-specific subsystems isolated, which makes upstream syncs (currently via
cherry-pick plus `upstream-adapt.sh`) cheaper.

**Exit:** the extracted modules have clear responsibilities and dependencies, preserve behavior, and
leave each resource's owner and release path identifiable. Review the resulting boundaries and run
the affected regression suites. Files above ~2,000 lines and remaining unsafe casts are review
prompts, not pass/fail quotas.

### Phase 8 — Process guardrails (R11–R13)

1. CI from phase 1 is required for merges to `main`; merges go through PRs.
2. A `commit-msg` hook rejects (a) `WIP` subjects on `main`, and (b) "Bump version" /
   `chore(release)` commits that touch anything besides version fields and changelogs.
3. Releases: batch fixes; a downstream report first becomes a failing regression test in this repo;
   no more than one release per day without a written reason. A human publishes every release in the
   GitHub web interface; agents only bump, tag, push and supply the release text (in place since
   2026-11-08, see the README's "Creating the GitHub release").
4. Keep the AEP gate, but complement it with mechanical checks (this plan's exit criteria, targeted
   documentation checks, CI). The hash gate verifies that the protocol was _quoted_, not that it was
   _followed_.
5. Each phase ends with a review round as defined in
   [Review rounds](#review-rounds-every-significant-milestone), followed by an update of the debt
   register above. Releases require a passing review of the commits since the previous release,
   which also enforces the cadence in item 3.

---

## 7. Workaround signatures to hunt

Patterns that indicate the debt described above; use them during phases 1–3 and in review.

| Signature                                                                       | Meaning                                        |
| ------------------------------------------------------------------------------- | ---------------------------------------------- |
| `if (!insideAbstractTrace()) x.dispose()` / `if (!inMakeJaxprBody())` + dispose | ownership depends on tracing context           |
| `catch (e) { if (!(e instanceof UseAfterFreeError)) throw e; }`                 | teardown tolerates a broken ownership contract |
| `if (this.#rc > 0) this.dispose()`                                              | double-dispose hidden                          |
| `getBackend().type ===` / `defaultDevice() ===` in `src/library/`               | lowering decision in library code              |
| `test.skipIf(device …)` with a comment about leaks or JIT caches                | leak hidden by backend choice                  |
| `device === "webgpu" ? <large tol> : <small tol>`                               | tolerance tuned to pass                        |
| `if (…) return;` at the top of a test body                                      | silently passing test                          |
| `…ForTest`, `_debug…` methods on production classes                             | test-shaped production code                    |
| "pre-existing", "no changes needed", "already" in milestone commit subjects     | milestone closed without work                  |
| "Bump version" commit touching `src/`                                           | change hidden from history readers             |
| Workload size or timeout chosen by GPU vendor/adapter string                    | guess standing in for a measurement            |
| `Promise.race` against a timer around work that keeps running                   | timeout that stops waiting, not the work       |
| Listener or hook installed at first use (first test, first call)                | misses what happens before it (F2.1)           |
| Intermittent failure called fixed after one green run                           | cause not measured (`fb8ea22`)                 |

---

## Appendix A: reproducing the analysis

- Commit survey:
  `git log --reverse --no-merges --author=Olli --format='%h %ad %s' --date=short HEAD` combined with
  per-commit `git show --unified=0` pattern counts.
- Rework plot and data: retained as historical audit artifacts, not regenerated. The analysis script
  was removed on 2026-11-09 and remains available in git history before that removal. Individual
  chains can be read off `history-churn-edges.csv` (`fixer_n` redid `lines` lines written by
  `origin_n`; `n` matches the `n` column of `history-churn.csv`).

- Ground truth: `git clone` the repository into a scratch directory, check out the commit, then
  `pnpm install && pnpm build && pnpm check && pnpm lint && pnpm test`. Do not link an existing
  `node_modules`: that skips the `prepare` scripts which build `dist/` and generate the website's
  `.svelte-kit/`, and website tests then fail spuriously (phase 0 review round 1, finding 3).

## Appendix B: test-integrity inventory commands

```bash
git grep -nE "\.(skipIf|skip|todo|fails|runIf)\(|skip\(\)" -- test 'src/**/*.test.ts' 'packages/**/*.test.ts'
git grep -nE "^\s+if \(.*\) return;\s*$" -- test 'src/**/*.test.ts'
git grep -nE "device === \"(webgpu|wasm|cpu|webgl)\" \? [0-9.e-]+" -- test 'src/**/*.test.ts'
git grep -nE "timeout: [0-9_]+" -- test 'src/**/*.test.ts'
sed -n '/exclude: \[/,/\]/p' vitest.config.ts
```
