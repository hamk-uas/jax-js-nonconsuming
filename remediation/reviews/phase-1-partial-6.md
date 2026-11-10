# Re-review: phase 1 steady-state leak checks and plan restructuring, round 6

- **Range:** uncommitted working tree on top of `0891af9` (`src/frontend/check-leaks.ts`,
  `test/check-leaks.test.ts`, `test/steady-state-leaks.test.ts`, `FEATURES.md`,
  `.github/copilot-instructions.md`, `remediation/reviews/phase-1-partial-5.md`,
  `REMEDIATION-PLAN.md`).
- **Date:** 2026-11-09.
- **Scope:** open finding F1 from [round 5](phase-1-partial-5.md), its implementation response and
  regression coverage, and the maintainer-requested restructuring of `REMEDIATION-PLAN.md` (1A/1B
  and 2A/2B staging). Previously accepted work in rounds 1–5 is not reopened.
- **Verdict:** **pass** for round 5 F1 and the `checkLeaks.steadyState` API/test implementation in
  the working tree. **1A and phase 1 remain open** (uncommitted state, hosted CI not verified, and
  first ownership slice not yet named/audited for 1A exit).
- **Reviewer changes:** no implementation changes; only this report is added to the worktree and
  left uncommitted for the implementation session.
- **Exclusion carried forward:** homepage content.

## Independent validation

Validated in the working tree at `0891af9` + uncommitted changes, plus a reverse-application check
replacing `src/frontend/check-leaks.ts` with `0891af9` while keeping the new tests.

| Check                                                          | Result                                                               |
| -------------------------------------------------------------- | -------------------------------------------------------------------- |
| `pnpm run build`, `pnpm run check`, `pnpm run lint`            | Pass                                                                 |
| Generated declarations (`dist/index.d.ts`, `dist/index.d.cts`) | Both declare `steadyState(body: () => unknown): Promise<LeakReport>` |
| `git diff --check` and `npx prettier --check` on touched files | Pass                                                                 |
| Two leak-check files, CPU/WASM (`pnpm run test:cpu`)           | 2 files passed; 59 passed / 26 skipped                               |
| Two leak-check files, NVIDIA WebGPU + WebGL                    | 2 files passed; 81 passed / 4 skipped                                |
| Two leak-check files, Intel WebGPU + WebGL                     | 2 files passed; 81 passed / 4 skipped                                |
| New tests against `0891af9` `src/frontend/check-leaks.ts`      | `pnpm run check` fails with 6 type errors; runtime fails 3/33 tests  |
| Restored working-tree `src/frontend/check-leaks.ts`            | `pnpm run check` and all 33 `test/check-leaks.test.ts` tests pass    |

The observed test counts and reverse-application failure counts match the implementation response in
`remediation/reviews/phase-1-partial-5.md` (6 `expectTypeOf` failures and 3 runtime failures on the
`0891af9` source).

## Finding disposition

| Finding                                                 | Status and evidence                                                                                                                                                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Round 5 F1: ordinary structural thenables typed as sync | **Fixed.** `steadyState(body: () => unknown): Promise<LeakReport>` is unconditionally `async` and uses native `await body()` twice; structural thenables, unions and erased `() => void` callbacks return `Promise` |

### Verification of the F1 fix

`src/frontend/check-leaks.ts` (lines 740–745) deletes `isPromiseLike` and the conditional overload
in favor of:

```ts
async function steadyState(body: () => unknown): Promise<LeakReport> {
  await body();
  const before = slotCounts();
  await body();
  return steadyStateReport(before, slotCounts());
}
```

1. **Type/runtime alignment:** Native `await` unwraps any object with a callable `then` method
   (including cross-realm promises and structural thenables whose `then` returns `void`), and the
   static return type is always `Promise<LeakReport>`. There is no longer a code path where
   `steadyState` awaits a callback at runtime while TypeScript infers a synchronous `LeakReport`.
2. **Caller lifetime safety:** Because `steadyState` always yields at a microtask boundary, callers
   using lexical `using` bindings outside `body` must `await` the call before scope exit. All
   callers in `test/check-leaks.test.ts` and `test/steady-state-leaks.test.ts` (`expectSteady`)
   properly `await` the returned promise, and `FEATURES.md` / `.github/copilot-instructions.md`
   document this requirement and the error-handling migration (`try { await ... } catch`).
3. **Regression coverage:** `test/check-leaks.test.ts` covers:
   - Structural thenables whose `then` returns `void` (verifies `Promise<LeakReport>` type,
     sequential starts `[0, 1]`, completion count 2, and `cpu: +1` slot leak detection).
   - Mixed sync / structural-thenable unions across warm-up and measurement
     (`it.each([true, false])`).
   - Callbacks whose static `() => void` signature erases an underlying returned promise (using a
     single, justified test-local `@typescript-eslint/no-misused-promises` suppression).
   - Synchronous throws in run 1 and run 2 rejecting the returned promise (`rejects.toBe(error)`).

## Plan restructuring review (`REMEDIATION-PLAN.md`)

The uncommitted updates to `REMEDIATION-PLAN.md` accurately reflect the maintainer's recovery
direction and staging decisions:

- **No premature sign-off:** Explicitly states that neither 1A nor Phase 1 is complete, and that the
  plan revision signs off no implementation or phase.
- **1A vs. 1B split:** Preserves all 12 numbered Phase 1 items for traceability while defining a
  clear 1A readiness gate before Phase 2A and carrying 1B items forward as required before final
  remediation sign-off.
- **2A vs. 2B staging:** Grounds Phase 2 in the actual codebase mechanics (`Array.ref` vs.
  `retainedHandle`, borrowed `ClosedJaxpr.dispose()`, `#creationRefOwner`), requires a bounded
  end-to-end slice and reviewed ADR before implementation, and requires explicit release/alias/error
  assertions rather than slot-count smoke checks alone.
- **Commit hygiene note (non-blocking):** When committing the working tree, consider separating the
  `checkLeaks.steadyState` F1 fix (`src/`, `test/`, `FEATURES.md`,
  `.github/copilot-instructions.md`, `remediation/reviews/phase-1-partial-5.md`) from the
  `REMEDIATION-PLAN.md` restructuring so each commit's subject and scope stay self-contained.

## Phase 1 item disposition

| Item                                      | Disposition                                                                                                      |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1 CI and push gate                        | Prior local implementation reviewed; hosted CI run on `wasm-multithreading` remains open                         |
| 2 Setup integrity                         | Previously signed off; unchanged                                                                                 |
| 3 Missing backends / 4 Hidden failures    | Previously completed in phase 0; enforcement retained                                                            |
| 5 Backend-init isolation                  | Previously signed off in round 3; unchanged                                                                      |
| 6 GPU errors and tracing                  | Regular tape tracing signed off in round 3; specialized dispatch tracing remains open (1B)                       |
| 7 Test inventory                          | Partial cleanup signed off in round 4; slice-specific audit (1A) and suite-wide issue links/relocation (1B) open |
| 8 Steady-state leaks                      | `checkLeaks.steadyState` API and initial cross-area suite pass (F1–F4 resolved); per-area integration open (1B)  |
| 9 Native trig contract                    | Previously signed off in round 2; unchanged                                                                      |
| 10 FD tolerances                          | QR checks signed off in round 4; slice-specific oracle check (1A) and full suite derivations (1B) remain open    |
| 11 Website opt-in work / 12 TTS benchmark | Open under 1B                                                                                                    |

## 1A readiness and Phase 1 exit criteria

| Criterion                                                                 | Status                                                             |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Round 5 F1 resolved with type and runtime regressions                     | **Verified** in working tree (pending commit)                      |
| Build, check, lint, default suite and both GPU suites green               | Verified in working tree                                           |
| Hosted CPU/WASM CI green for the readiness commit (with run URL and SHA)  | **Not verified** (changes are uncommitted; no hosted run recorded) |
| Existing cross-area steady-state checks green without cache flushing      | **Verified** (48 passed / 4 skipped on CPU/WASM, NVIDIA and Intel) |
| First ownership slice named, with test gaps and oracle requirements       | **Open** (required before 1A exit sign-off)                        |
| Full Phase 1 exit (all 1B items, issue links, per-area suite integration) | **Unmet** (carried forward under 1B)                               |

## Sign-off

Round 5 **F1 is signed off**, and the `checkLeaks.steadyState` implementation and its initial
cross-area coverage under **item 8 (1A scope)** are signed off once committed as reviewed. Per-area
suite integration under item 8 remains open in 1B. **Neither the 1A readiness gate nor full Phase 1
is signed off** until the changes are committed, hosted CI is recorded green for the readiness
commit, and the first ownership slice with its test/oracle audit is named. This report is left
uncommitted for the implementation session.
