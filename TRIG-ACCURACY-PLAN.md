# WASM Float64 Defaults and Per-operation Trigonometric Accuracy Plan

**Status:** standalone proposal; no implementation is claimed.

This is separate feature work, not a remediation milestone or prerequisite. It does not reopen the
approved native GPU float32 test policy, change remediation exit criteria, or authorize edits to the
remediation plan or review reports. Schedule and review this work independently.

## Goal

First replace the existing WASM float64-to-float32 transcendental fallbacks with genuine float64
implementations selected by dtype alone. Float64 inputs should not require an accuracy option to
receive float64 computation.

Then let users request a more accurate implementation for an individual operation without moving
their arrays to another device. Preserve the approved native GPU float32 defaults.

Distinguish:

- **Accuracy:** how close a represented result is to the mathematical result.
- **Precision:** how many significant bits the result format can represent.

A better float32 approximation is not float64, correct rounding, or cross-device bitwise
reproducibility.

## Public API direction

Use per-operation options, following current Python JAX's `lax` approach, rather than a separate
accurate-math namespace or a mutable global precision mode.

Illustrative TypeScript API, not existing functionality:

```ts
lax.asin(x);
lax.asin(x, { accuracy: "highest" });
lax.asin(x, { accuracy: { ulps: 2 } });
```

Proposed semantics:

| Request                | Meaning                                                                                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Omitted or `"default"` | Use the ordinary implementation for the dtype/backend: preserve native GPU float32 defaults; use the new genuine f64 defaults for migrated WASM operations.                                       |
| `"highest"`            | Select the most accurate validated implementation available for the operation, dtype and backend. It can equal the default when no better implementation exists. Publish the applicable contract. |
| `{ ulps: n }`          | Require a validated implementation meeting the requested maximum-error contract. Reject the request explicitly if unavailable; do not substitute a weaker implementation.                         |

Start explicit budgets with a finite, positive ULP limit. Add absolute/relative tolerances only if a
concrete consumer requires them; define their combination semantics before adding that surface.

Keep existing `numpy` and array-method call signatures unchanged. Their WASM f64 results improve
automatically as the dtype-specific defaults are replaced; this is an intentional numerical change,
not a new opt-in API. New `lax` wrappers should share core operations and implementation helpers,
not establish another independently maintained math stack. Accuracy options are static compile-time
inputs, not array-valued or data-dependent settings. Export public option types and functions
through [src/index.ts](src/index.ts).

Do not introduce implicit CPU fallback, dtype promotion, or readback. A user requesting more than
float32 precision must choose a genuinely wider execution path explicitly.

## Accuracy contracts

Before publishing an implementation, specify its:

1. Supported operation, dtype and backend.
2. Input domain and maximum-error bound.
3. ULP definition, including binade boundaries and near-zero results.
4. Treatment of subnormal inputs/results, signed zero, NaN, infinities and out-of-domain values.
5. Arithmetic assumptions: rounding, FMA, reciprocal/square-root accuracy, reassociation and FTZ.

Use one explicit ULP definition for both advertised budgets and validation; JAX's accuracy
documentation and WGSL use different conventions at exact binade boundaries.

Aim for a few ULP for regular float32 results, but choose the published number from the actual
implementation and validation, not the source library's headline. Define and validate the f64
defaults' own bounds over their binary64 domains; genuine f64 does not imply correct rounding. Keep
mathematical-domain errors separate from unsupported accuracy requests. Special-value behavior must
be part of the contract, not silently excluded from it.

Do not publish a range-limited accurate `sin`/`cos` as a general full-domain implementation. A
bounded prototype may remain internal while its limits and eventual handling are decided; a
requested strict contract must not become an unchecked assumption about runtime inputs.

## Algorithm and coefficient sources

The intended reuse is **coefficients and their mathematical approximation recipe**, not a wholesale
import of an old C library, its error handlers, or its platform assumptions.

For every selected coefficient set, record the source/version, provenance and attribution, target
format, approximation interval, transformation, polynomial ordering and evaluation strategy. Prefer
exact representations of coefficients in their intended format. Do not rely on JavaScript constant
evaluation or historical C promotion rules to define the GPU arithmetic.

For the WASM f64 defaults, select actual double-precision algorithms and coefficient sets, not
float32 approximations with widened constants/results. Validate full binary64 range reduction,
overflow/underflow thresholds and tail behavior as applicable. Share infrastructure where useful,
but do not assume the later GPU float32 approximation is also an adequate f64 implementation.

### Cephes

[Cephes single.tgz](https://netlib.org/cephes/single.tgz) explicitly contains single-precision
implementations; [singldoc.html](https://netlib.org/cephes/singldoc.html) describes them.

- `atanf`: range reduction followed by a polynomial.
- `asinf`: small-argument polynomial and an endpoint transformation.
- `acosf`: endpoint-aware formulas avoiding subtraction cancellation.
- `sinf`/`cosf`/`tanf`: reduced-argument approximations.

Coefficients must retain their approximation interval and variable transformation. Cephes's sampled
relative-error measurements are not inherited maximum-ULP guarantees for WGSL.

### XLA

Source inspection was pinned to XLA commit `9af938ab71d667f4e5108f7c07bae60ba671de44`.

The most immediately relevant reference is
[XLA's float32 asin/acos implementation](https://github.com/openxla/xla/blob/9af938ab71d667f4e5108f7c07bae60ba671de44/xla/codegen/intrinsic/cpp/trig.cc#L228):

- Shared degree-4 correction polynomial with float32 coefficients.
- Reproducible Sollya `fpminimax` generation command.
- Reduction to a variable in `[0, 0.25]`.
- Split constants and correction terms addressing endpoint/cancellation error.
- A stated 1-ULP target and accompanying
  [range-sweep/boundary tests](https://github.com/openxla/xla/blob/9af938ab71d667f4e5108f7c07bae60ba671de44/xla/codegen/intrinsic/cpp/trig_test.cc#L284).

These generated implementations are registered for CPU targets, not existing WebGPU implementations.
Their arithmetic and error claims need independent verification after a WGSL adaptation.

Do not treat every float32-output XLA algorithm as all-float32: its
[custom CPU sin/cos](https://github.com/openxla/xla/blob/9af938ab71d667f4e5108f7c07bae60ba671de44/xla/codegen/intrinsic/cpp/custom_cpp_intrinsics.cc#L341)
evaluate internally in double precision. Those coefficients are not drop-in references for a pure
float32 WebGPU path.

XLA GPU paths also use NVIDIA libdevice and AMD OCML. Calling these device libraries is not the same
as using bare shader hardware approximations, and their accuracy does not establish a WGSL bound.

### Other prior art

Consider [SLEEF](https://sleef.org/) and independently generated Sollya minimax approximations when
Cephes/XLA choices do not meet the target. Verify source licensing and attribution before reuse. Do
not add a numerical-library runtime dependency solely to obtain coefficient data.

## Implementation sequence

### 0. Restore genuine WASM f64 defaults

The current `callFuncF32` fallback in [src/backend/wasm.ts](src/backend/wasm.ts) affects eight
operations: `sin`, `cos`, `asin`, `atan`, `exp`, `log`, `erf` and `erfc`. Reconfirm the inventory at
implementation time and distinguish implicit precision loss from explicit user-requested casts.

Start with regression tests that fail under the current fallback:

- Exact binary64 inputs distinguishable only beyond float32 precision, including inverse-trig values
  near endpoints and approximation boundaries.
- Wider-range cases such as `exp(100)` and `log(1e-100)`, whose correct results are finite but whose
  fallback results overflow or encounter an input rounded to zero.
- Large finite trig inputs requiring binary64 range reduction, and `erfc` tails lost in float32.
- Signed zero, subnormals, NaN, infinities, domain errors and true binary64 overflow/underflow.

Generate high-precision references at the exact represented f64 inputs. Use function-specific
binary64 error bounds rather than the existing five-decimal assertions; CPU results alone are not
the reference oracle.

Deliver validated f64 implementations in coherent batches covering all eight operations. Select them
through ordinary dtype dispatch, without `"highest"`, implicit CPU transfers or a legacy precision
mode. Once an operation is migrated, remove its demote/call-f32/promote fallback. Remove the
obsolete fallback helper and TODO when no operation needs them; retain explicit casts and the
existing f32 implementations.

Verify eager, JIT, mega-module and supported nested scan/block-map execution, transforms and
ownership. Use code capture or equivalent routing evidence and negative controls that restore the
old fallback, proving the tests detect its precision/range loss. Preserve existing float32 behavior.
Benchmark f64 changes separately and document the new per-operation defaults and remaining gaps.

Include focused checks for composed public functions that depend on the replaced intrinsics, without
assuming that wider arithmetic also fixes their cancellation or overflow-prone formulas.

This is the first delivery milestone, not a requirement to finish every f64 special function before
any independent GPU float32 work can proceed. The plan's f64 milestone is complete only when all
eight fallbacks are replaced and their contracts verified.

### 1. Establish GPU float32 contracts and candidate quality

- Pin reference inputs and current native outputs before implementing alternatives.
- Compare Cephes and XLA candidates using their complete mathematical recipes.
- Decide the first supported dtype/backend contracts and explicit-budget semantics.
- Generate reference data from the exact represented input values.
- Keep experimental coefficients and range-limited paths unpublished.

### 2. Deliver float32 inverse trig first

Start with float32 `asin` and stable direct `acos`, using the shared correction polynomial. Follow
with range-reduced `atan` and robust `atan2`.

Avoid:

- Accurate `acos` implemented merely as `pi/2 - nativeAsin(x)`.
- Accurate `atan2` depending internally on low-accuracy native `atan`.
- Overflowing `sqrt(x*x + y*y)` when scaled/ratio-based formulations avoid it.
- Loss of quadrant, signed-zero or endpoint semantics during approximation.

Retain non-consuming ownership in eager execution as well as tracing.

### 3. Integrate the per-operation request end to end

- Preserve the static request through tracing, transformations, simplification and lowering.
- Prevent rewrites from replacing a strict implementation with native trig.
- Ensure operation/Jaxpr/program cache identities distinguish accuracy requests and selected paths.
- Keep GPU float32 `"default"` on the existing lowering without added dispatches or approximation
  overhead; ordinary WASM f64 calls must use the new dtype-specific implementations.
- Prefer shared helpers and existing primitive parameters; add primitives only when needed for
  correct differentiation or lowering, not merely to wrap a namespace.
- Cover every supported execution path, including fused shaders and nested compiled bodies. Do not
  implement the option in only one WGSL emitter.
- Surface invalid options and unsupported strict requests through ordinary explicit errors.

Autodiff must retain mathematical derivative semantics. For example, changing the value
approximation of `asin` should not inadvertently replace its derivative rule with the derivative of
the approximation polynomial. Specify how accuracy requests affect trig operations generated by
derivative rules and test that behavior.

### 4. Add float32 forward trig only after range reduction is ready

Implement `sin`/`cos` together around shared reduction and polynomial machinery; derive `tan` from
adequately accurate constituents or a separately justified approximation.

Naive `x % (2*pi)` is not adequate for large float32 angles. Full-domain support needs robust range
reduction, including whatever multiword/integer techniques are necessary without native f64. Near
tangent poles, use conditioning-aware error analysis rather than a fixed decimal tolerance.

## Validation and release gate

Use [MPFR](https://www.mpfr.org/) or sufficiently high-precision `mpmath` references evaluated at
the exact represented input values: binary64 for the f64 defaults and binary32 for the GPU float32
alternatives. Rounded outputs from a native backend are not ground truth.

For each supported function:

- Measure regular-domain maximum ULP error and report subnormal/special-value behavior separately.
- Cover zeros, binade boundaries, approximation branch thresholds, inverse-trig endpoints,
  quadrants, poles and large-angle reduction cases.
- Combine structured sweeps, adversarial samples and exhaustive testing where feasible. Clearly
  distinguish sampled empirical bounds from exhaustive results.
- Test default, highest and explicit-budget requests, including impossible budgets and invalid
  options. Prove with negative controls that excessive error is rejected.
- Verify eager/JIT parity, supported `jvp`/`grad`/`vmap` compositions, nested scan/block-map paths
  and zero ownership leaks.
- Use code capture or equivalent evidence to prove the requested implementation actually ran;
  passing value tests alone does not establish correct routing.
- Validate f64 implementations on WASM. For GPU float32 alternatives, run actual generated shaders
  on NVIDIA and Intel. AMD coverage remains unverified until measured.
- Benchmark compile time, steady-state throughput, dispatch count and generated code size separately
  from accuracy. Build before benchmarks; use the existing test and benchmark tooling.

Publish only after the measured contract, routing, transformations, ownership and
unsupported-request behavior agree. Update public API documentation and feature notes as part of
that separate release.

## Users needing more than float32 precision

Until the f64 replacements ship, explicit CPU float64 is the current practical option for sensitive
subcomputations, with transfers chosen by the user. Keep the original wider input where possible:
widening an already rounded float32 input or result cannot recover lost information. CPU JavaScript
`Math` is not a blanket correct-rounding guarantee for the library's composed functions.

Do not presently recommend WASM float64 as double-precision trig: [wasm.ts](src/backend/wasm.ts)
demotes the eight listed transcendental operations to float32 and widens the result. After each
replacement ships, recommend WASM f64 for that operation under its validated contract and update
this guidance accordingly. Genuine f64 defaults are the initial milestone, distinct from the opt-in
GPU float32 feature. WebGPU has no native f64.

## Migration and non-goals

The additive mapping is:

```text
existing GPU float32 numpy trig calls -> unchanged native defaults
existing WASM float64 calls to the eight listed operations -> genuine f64 defaults, no option needed
selected operation needing tighter accuracy -> lax operation with an accuracy option
```

Once a validated per-operation option exists, downstream custom polynomial wrappers or ad hoc
GPU-to-CPU round trips used solely to improve that operation can be replaced. After the relevant f64
default ships, CPU detours used only to bypass WASM's f32 fallback can use ordinary WASM f64 calls
instead. Do not remove deliberate CPU execution needed for unsupported operations or stricter
contracts, or replace genuinely required float64 with a better float32 approximation.

Non-goals:

- Changing the approved default GPU test tolerances or completing remediation items.
- A global accuracy switch, separate accurate-math namespace, or vendor-name special cases.
- Claiming correct rounding, universal JAX parity, or bitwise reproducibility without evidence.
- Extending beyond the eight existing WASM f64 fallbacks and the scoped float32 trig alternatives,
  supporting all dtypes, or adding arbitrary-precision arithmetic.

## Further reading

- [JAX per-operation sine accuracy API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sin.html):
  a mode selects an available implementation; an unsatisfiable explicit tolerance is documented to
  produce a compiler error.
- [JAX numerical accuracy](https://docs.jax.dev/en/latest/accuracy.html): backend/dtype-specific
  measurements, not WGSL-native accuracy guarantees.
- [WGSL concrete-float accuracy](https://www.w3.org/TR/WGSL/#concrete-float-accuracy): the native
  shader contracts that remain the default here.
