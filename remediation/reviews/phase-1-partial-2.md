# Review: phase 1 native trig accuracy, round 2

- **Range:** `0b3c72a..83e0e32`.
- **Date:** 2026-11-09.
- **Reviewer:** the separate, non-implementing review session (`c70303a7`).
- **Scope:** recording the preceding review and implementing the approved GPU float32 trig test
  contract; recommendation for users needing higher accuracy.
- **Human decision:** "I approved to use GPU float32 trig native accuracies in tests."
- **Verdict:** pass within this incremental scope; no new finding. R24's approved test-policy change
  is verified.
- **Phase status:** Phase 1 remains incomplete; this is not human sign-off.
- **Exclusion:** homepage content remains excluded.

## Review result

The change implements the approved accuracy policy, rather than hiding backend failures. It changes
tests and documentation only. Runtime math, autodiff rules, required backends, skips, timeouts and
float16/float64 thresholds are unchanged.

[numpy.test.ts](../../test/numpy.test.ts), lines 14-58, gates the wider budgets on the actual
result's GPU device **and** float32 dtype. CPU/WASM retain the previous five-decimal assertions. The
policy test requires acceptance at the boundary, rejection above it and rejection of NaN; non-GPU
and non-f32 metadata do not receive the GPU allowance.

I checked the constants against the official
[WGSL accuracy table](https://www.w3.org/TR/WGSL/#concrete-float-accuracy) and
[ULP definition](https://www.w3.org/TR/WGSL/#ulp):

- `sin`/`cos`: `2^-11` absolute error only inside `[-pi,pi]`. The new corpus is inside that range.
- `atan`: 4096 ULP. The helper uses WGSL's smaller adjacent spacing at binade boundaries and the
  minimum subnormal spacing at zero.
- `asin`: the worse of `6.81e-5` absolute error and the inherited `atan2(x,sqrt(1-x*x))` error, not
  a blanket 4096-ULP rule.
- `tan`, `acos` and `atan2`: budgets reflect the library's actual compositions in
  [numpy.ts](../../src/library/numpy.ts), lines 2030, 2056 and 2089. Their arithmetic allowances are
  conservative for the tested, moderate interior values. The tangent tests explicitly require
  `|cos(x)| >= 0.5`; inverse-trig references use the represented float32 input.

The documentation correctly limits the composed budgets to the corpus, distinguishes special-case
smoke tests from conformance guarantees, and does not promise accurate range reduction for arbitrary
angles. GLSL ES 3.00 section 4.5.1 indeed leaves trig precision unspecified: using the WGSL budget
for WebGL is explicitly a project policy, not a GLSL guarantee.

There is no finding requiring another implementation round. Acceptance under the approved native
contract must not be interpreted as improved runtime accuracy.

## Independent validation

Validation used a fresh, independent clone detached at `83e0e32`, with normal
`pnpm install --offline --frozen-lockfile` and its build/website generators, not linked
dependencies.

| Check                                                | Result                                                      |
| ---------------------------------------------------- | ----------------------------------------------------------- |
| `pnpm build`, `pnpm check`, `pnpm lint`              | Pass                                                        |
| Full CPU/WASM browser suite                          | 89 files passed, 3 skipped; 2270 tests passed, 1079 skipped |
| Real `pnpm run test:push`: default                   | 92 files passed; 3335 tests passed, 14 skipped              |
| Same command: NVIDIA                                 | 101 files passed; 3382 tests passed, 14 skipped             |
| Same command: Intel                                  | 101 files passed; 3382 tests passed, 14 skipped             |
| Complete real pre-push command                       | Exit 0; all three runs required WebGPU/WebGL                |
| Restored focused trig, gradient and policy selection | 29 passed on each GPU; unrelated tests filtered out         |

Compared with round 1, nine tests were added: two new value tests across four backends, plus the
policy control. CPU/WASM therefore gain five passing tests and four GPU skips. Full GPU skips remain
14; none of the four former Intel failures was skipped.

### Negative controls

All mutations were confined to the clone and restored.

1. Replace only the GPU assertion with the former five-decimal assertion. The two selected
   inverse-trig tests reproduce exactly four Intel failures: asin/acos and atan2 on WebGPU and
   WebGL. The asin difference is `8.5831e-5` against the now input-rounded reference; atan2 remains
   `6.5128e-6`, both exceeding `5e-6`.
2. Restore the native budgets, but inject a `0.01` offset into the GPU result comparison. All five
   selected value tests fail on both GPU backends, on each physical GPU: ten failures per run. The
   same ten CPU/WASM value tests pass. This verifies that the relaxed assertions still reject
   genuinely out-of-budget results without relaxing CPU/WASM.
3. Remove the injected error. The focused selection passes all 29 tests on each physical GPU,
   including the committed boundary/NaN/scope control.

## Phase 1 dispositions and exit criteria

| Item                                                                    | Disposition                                                                                                             |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 1: push/PR workflow and full pre-push gate                              | Implemented in `c1faa1d`, reviewed previously. Real local gate is now green on both GPUs; hosted CI remains unverified. |
| 2: prepare failure, source-based benchmark checking, diagnostic cleanup | Done in `0b3c72a`; carry forward [round 1](phase-1-partial-1.md).                                                       |
| 3: required backends                                                    | Done in `43b61eb`; current full gate enforces both GPU backends.                                                        |
| 4: R19 failures                                                         | Previously addressed; current full suites remain green.                                                                 |
| 5: independent backend initialization and injected failure test         | Open, unchanged.                                                                                                        |
| 6: WebGPU error capture                                                 | Previously done in `82bb49d`/`648d367`; unchanged. Command-tape tracing and its JIT trace assertion remain open.        |
| 7: complete skip/return/tolerance/timeout/exclusion inventory           | Open. This change documents the affected trig policy, not the complete inventory.                                       |
| 8: steady-state leak mode                                               | Open, unchanged.                                                                                                        |
| 9: R24 native-accuracy decision and implementation                      | Human decision confirmed in this request; implementation `83e0e32` passes this review.                                  |
| 10: conditioning-derived finite-difference tolerances                   | Open; gradient thresholds unchanged.                                                                                    |
| 11: remaining website pages start GPU work only on request              | Open; homepage remains excluded.                                                                                        |
| 12: TTS per-step benchmark                                              | Open, unchanged.                                                                                                        |

| Phase exit criterion                                          | Result                                               |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| Hosted CI green on the branch                                 | Not verified; no new hosted-run evidence is claimed. |
| Default and both GPU suites green, required backends enforced | Verified in the real pre-push run at `83e0e32`.      |
| Every skip/tolerance/timeout has a reason and issue           | Not verified; the full inventory remains open.       |
| Steady-state leak mode enabled for the named test families    | Not met; item 8 remains open.                        |

No phase deferral or sign-off is inferred. AMD hardware validation remains unavailable.

## Recommendation: opt-in higher-accuracy math

Keep native GPU float32 operations as the default. Distinguish **higher accuracy in a float32
result** from **more than float32 precision**; one switch should not imply both.

### Immediate guidance

Document explicit CPU float64 execution for numerically sensitive subcomputations, with explicit
transfers at their boundaries. Preserve the original higher-precision input where possible: casting
a previously rounded GPU input or result to float64 cannot recover lost information. CPU operations
use JavaScript `Math` via [alu.ts](../../src/alu.ts), line 1105. This is a practical higher-accuracy
option, not a promise of correctly rounded results for every composed operation. In particular,
`acos = pi/2 - asin` can still lose relative accuracy near zero.

**Do not currently recommend WASM float64 as a double-precision trig solution.**
[wasm.ts](../../src/backend/wasm.ts), lines 2177-2190, explicitly demotes inputs to float32 for
`sin`, `cos`, `asin` and `atan`, then widens the float32 result. An independent packaged-runtime
probe of float64 `sin(0.7000000000000001)` produced:

- CPU: `0.6442176872376911`, matching `Math.sin` in that run.
- WASM: `0.6442176103591919`, absolute error approximately `7.69e-8`.

This is pre-existing behavior discovered while checking the requested recommendation, not a
regression or additional acceptance condition for this test-only change. True f64 WASM
transcendentals would be a useful separate implementation task. WebGPU has no native f64.

### Recommended GPU offering

Provide explicitly named, pure opt-in functions in a small accurate-math namespace, for example
`accurate.sin(x)` and `accurate.atan2(y,x)`; these names are illustrative, not existing APIs. Prefer
composable arithmetic implementations that the existing JIT and transforms can trace over a mutable
global "precision mode". They should remain on the selected device and preserve dtype. Never
silently fall back to CPU or silently substitute native accuracy when unsupported.

Define a per-function maximum-error contract, input range and exceptional-value policy before
implementation. A target such as a few ULP for normal finite float32 inputs is useful, but must be
measured and justified before publication; it is not the same as correct rounding or cross-device
bitwise reproducibility.

Start with `atan`/`asin` and stable direct `acos`/`atan2`, addressing the observed weakness. Use
range-reduced polynomial/rational approximations and avoid cancellation and overflowing intermediate
norms. Add `sin`/`cos` with robust range reduction; naive `x % (2*pi)` is not enough for large
float32 angles. If only a bounded input range is initially supported, publish that limit.

Use an established numerical library such as [SLEEF](https://sleef.org/) as algorithmic prior art,
not as evidence that an unvalidated WGSL port inherits its guarantees. Validate against
[MPFR](https://www.mpfr.org/)-generated high-precision reference vectors, including range
boundaries, near-zero results, inverse-trig endpoints, large angles and exceptional inputs, on both
GPUs. Retain eager/JIT/transform and ownership checks. Report accuracy and slowdown separately. If
users need more than float32's significand, direct them to genuine CPU/WASM f64 support rather than
label a better f32 approximation "double precision".

## Rework and conclusion

`python3 -S scripts/history-churn.py --since 0b3c72a` passed. Both scopes contain two commits. `src`
has zero measured new lines; `code` has 113. Both report zero recent rework pairs,
ratio-within-20/100 `0.00`, follow-ups/change `0.00` and corrective share `0.00`. Recent follow-ups
remain right-censored; documentation is outside these measured scopes.

The approved R24 implementation passes review and removes the previous local gate failure under the
new contract. Remaining Phase 1 work is unchanged.

Only this report was added to the original repository, left uncommitted for the implementation
session. No implementation or phase-status file was edited.
