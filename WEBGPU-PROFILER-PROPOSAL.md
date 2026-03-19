# Proposal: Parameterized Cost Modeling and Continuous Device Characterization

## 1. Executive Summary

This proposal outlines how `jax-js` estimates WebGPU device capabilities for JIT tuning decisions.
The compiler needs physical parameters (memory bandwidth $B_{mem}$, ALU throughput $T_{alu}$,
dispatch overhead $C_{dispatch}$, register cliff $R_{opt}$, shared memory ceiling $S_{opt}$) to
choose tile sizes, fusion strategies, and scan paths.

The system uses a **continuous inference model** — not discrete hardware tiers. At device init, the
runtime:

1. Classifies the vendor family from `adapter.features` bitmap + `maxBufferSize` (browser-proof,
   works even on Safari)
2. Reads `maxComputeWorkgroupStorageSize` directly as $S_{opt}$ (no inference needed)
3. Applies vendor-conditioned log-linear regression on `maxBufferSize` to infer $B_{mem}$ and
   $T_{alu}$
4. For Apple Silicon: uses `adapterInfo.architecture` (Chrome) for generation-specific lookup, or
   conservative median (Safari)
5. Optionally refines the estimate via `jax.calibrate()` micro-benchmarks using log-Kalman
   covariance updates

All parameters are continuous. No quantization into discrete tiers, no artificial discontinuities in
tuning decisions.

## 2. Motivation

- **User-Side Auto-Tuning is Unsafe**: Native A/B profiling on arbitrary programs risks Context
  Lost.
- **Replacing Arbitrary Heuristics with Regression-Based Priors**: Starting with hardware-derived
  regression coefficients is statistically sounder than hand-tuned constants.
- **Covariance is Key to Obfuscated Hardware**: Hardware capabilities are physically linked. High
  memory bandwidth implies high ALU throughput. Because these properties are strictly positive and
  scale multiplicatively (semiconductor generations multiply capability), they are **lognormally
  distributed**. Working in log-space, standard linear regression yields always-positive estimates
  via `exp()`.
- **Continuous Over Discrete**: A 4-tier system creates 3.5× discontinuities at tier boundaries
  (e.g., 3.9 vs 4.1 TFLOPS snaps between tiers with very different tuning). Continuous regression
  gives smooth, proportional estimates.
- **Risk-Aware Search for Optima**: Using an analytical cost estimator, we statically locate a path
  granting ~80% of peak efficiency while maximizing distance from the hardware's failure boundaries.

## 3. The Estimation Architecture

The JIT compiler divides the decision logic into three distinct steps:

### 3.1. Structural Legality Checks (Pass / Fail)

Hard browser constraints (like `maxComputeWorkgroupStorageSize` or `needsLeafPacking`) immediately
filter paths before cost evaluation.

### 3.2. Program-Shape Features (Variables)

For legal paths, the JIT analyzer extracts dynamic parameters:

- $N_{dispatch}$, $N_{buffers}$, $Count_{alu}$, $Count_{mem}$, $Depth_{priv}$, $Size_{shmem}$,
  $Size_{wgsl}$, $F_{subgroup}$.

### 3.3. The Unified Cost Equation

The hardware profile uses predictors
($C_{dispatch}, B_{mem}, T_{alu}, C_{compile}, R_{opt}, S_{opt}$) calculated against the program
shape. Paths approaching hardware cliffs incur an exponentially compounding danger penalty:

```
Cost_execution = (N_dispatch * C_dispatch) + (Count_mem / B_mem) + (Count_alu / (T_alu * F_subgroup))
Cost_compile = Size_wgsl * C_compile

DangerMultiplier = Penalty(Depth_priv / R_opt) * Penalty(Size_shmem / S_opt) * Penalty(N_buffers / 8)

TotalCost = (Cost_execution + Cost_compile) * DangerMultiplier
```

## 4. Origin of the Hardware Profile: Vendor-Conditioned Regression

Because strict anti-fingerprinting mandates mask `GPUAdapterInfo`, you cannot rely on getting exact
device names. But `adapter.features` and `adapter.limits` are always available — they're required
for correct rendering and can't be hidden without breaking content.

### 4.1 Two-Stage Vendor Classification (Browser-Proof)

**Stage 1: Feature Bitmap → Hardware Family**

| Features Present              | Vendor Class       | Description                                                 |
| ----------------------------- | ------------------ | ----------------------------------------------------------- |
| `etc2 + astc + shader-f16`    | `apple`            | Apple Silicon (M1–M4). Unique: mobile texture formats + f16 |
| `etc2 + astc` (no f16)        | `mobile`           | Adreno/Mali mobile SoC                                      |
| `bc + maxBuf ≤ 5GB`           | `igp`              | Intel UHD/Iris Xe, AMD APU                                  |
| `bc + shader-f16 + subgroups` | `discrete-modern`  | NVIDIA Turing+, AMD RDNA2+, Intel Arc                       |
| `bc + timestamp-query`        | `discrete-legacy`  | AMD RDNA1, older discrete                                   |
| `bc` only                     | `discrete-minimal` | Minimal desktop                                             |

The `bc` vs `etc2+astc` split is the most browser-proof signal: it separates desktop GL from mobile
GL hardware at the ISA level and cannot be faked without breaking texture loading.

**Stage 2: `maxBufferSize` → Discrete vs IGP**

Within desktop (BC-capable) hardware, `maxBufferSize` cleanly separates integrated from discrete
GPUs because:

- All discrete GPUs have ≥6 GB VRAM → `maxBufferSize ≥ 6 GB`
- All IGPs share system RAM → Dawn/Chrome reports ≤4 GB
- The 5 GB threshold has no overlap in our 49-GPU dataset

### 4.2 How Browser Masking Affects Each Stage

| Browser     | `adapter.features`       | `adapter.limits` | `adapterInfo`                              | Model Strategy                                                |
| ----------- | ------------------------ | ---------------- | ------------------------------------------ | ------------------------------------------------------------- |
| **Chrome**  | Full                     | Full             | Architecture family ("ampere", "common-3") | Best: architecture lookup → exact generation                  |
| **Firefox** | Full                     | Full             | Broad vendor only ("nvidia", "amd")        | Good: features + limits regression                            |
| **Safari**  | Clamped to spec minimums | Clamped          | "Apple GPU"                                | Limited: feature bitmap → "apple" class → conservative median |

Safari is the worst case. By clamping limits to spec minimums, it removes our continuous observable.
But the feature bitmap still identifies it as Apple, and we fall back to the Safari-specific P25
estimate from our Apple generation lookup table.

### 4.3 Continuous Inference (Not Tier Lookup)

For each vendor class, the offline pipeline has fitted:

$$\log(B_{mem}) = \alpha + \beta \cdot \log(\text{maxBufferSize})$$
$$\log(T_{alu}) = \gamma + \delta \cdot \log(B_{mem})$$

At runtime, the inference is three `Math.log`/`Math.exp` calls — no table lookup, no tier
boundaries, no discontinuities. The result is a smooth function of the device's actual
`maxBufferSize`.

For Apple specifically: Chrome provides `adapterInfo.architecture = "common-N"` (where N is the
Apple GPU generation), enabling exact per-generation lookup from our offline table. Safari gets a
conservative P25 fallback.

## 5. Offline Data Modeling: Building the Regression Coefficients

### 5.1 Joint Dataset: Physical Specs × WebGPU Limits

The model was trained on a joint dataset of 49 representative GPUs, each mapped to both physical
capabilities (bandwidth GB/s, FP32 TFLOPS) and WebGPU observables (maxBufferSize, features).
Sources: TechPowerUp, Vulkan DB, Apple developer docs, Dawn defaults, local measurements, and CTS
conformance reports.

The trained model's **Welford sufficient statistics** are embedded directly in `runtime_model.json`
under `_sufficientStats`. This makes the model self-contained — all information needed for
re-fitting lives in the JSON file itself. The offline build scripts have been retired.

### 5.2 Vendor-Conditioned Log-Linear Regression

For each vendor class (see §4.1), we fit OLS in log-space:

$$\log(B_{mem}) = \alpha_{vc} + \beta_{vc} \cdot \log(\text{maxBufferSize})$$

and the cross-covariance chain:

$$\log(T_{alu}) = \gamma_{vc} + \delta_{vc} \cdot \log(B_{mem})$$

Vendor conditioning is essential — the same `maxBufferSize` (e.g., 8 GB) means very different things
for an RTX 4060 (15 TF) vs. an AMD RX 6600 XT (9.7 TF) vs. an Apple M2 10C (3.6 TF).

### 5.3 Regression Quality by Vendor Class

| Vendor Class      | n   | $R^2$ (buf→bw) | $R^2$ (bw→tf) | Strategy                                      |
| ----------------- | --- | -------------- | ------------- | --------------------------------------------- |
| `discrete-modern` | 24  | 0.65           | 0.55          | Regression (workhorse)                        |
| `discrete-legacy` | 2   | —              | —             | Median fallback                               |
| `apple`           | 14  | 0.62           | 0.97          | Generation lookup (Chrome) or median (Safari) |
| `igp`             | 7   | 0.62           | 0.96          | Regression                                    |
| `mobile`          | 2   | —              | —             | Median fallback                               |

For classes with $n < 5$ or $R^2 < 0.5$, the runtime uses geometric-mean fallbacks instead of the
regression. The model self-reports `regressionUsable: false` for these.

### 5.4 Validation: Leave-One-In Prediction Error

| Metric               | Discrete Modern | IGP   | Apple (regression)                     | Overall |
| -------------------- | --------------- | ----- | -------------------------------------- | ------- |
| $B_{mem}$ mean error | 19.9%           | 21.8% | 38.8%                                  | 24.4%   |
| $T_{alu}$ mean error | 40.1%           | 47.7% | 37.7% (mitigated by generation lookup) | 38.6%   |
| $T_{alu}$ P90 error  | —               | —     | —                                      | 78.3%   |

The 40% mean $T_{alu}$ error is acceptable for tuning decisions because:

- The cost model makes **binary choices** (tile 128 vs 256, fuse or not) with wide margins between
  good and bad outcomes
- The Bayesian update step (`jax.calibrate()`) refines the estimate to single-digit % error
- Even a 2× error in TFLOPS estimate only shifts the cost-optimal tile size by one step, which is
  within the "3–5× of optimal" design philosophy

### 5.5 Apple Generation Lookup (Chrome-Specific)

Chrome on macOS exposes `adapterInfo.architecture = "common-N"`. The offline pipeline builds a
per-generation lookup table:

| Architecture    | BW Range (GB/s) | TF Range | Geometric Mean BW / TF |
| --------------- | --------------- | -------- | ---------------------- |
| `common-1` (M1) | 68–800          | 2.6–20.8 | 257 / 7.4              |
| `common-2` (M2) | 100–800         | 3.6–27.2 | 283 / 9.7              |
| `common-3` (M3) | 100–400         | 3.1–12.2 | 182 / 5.9              |
| `common-4` (M4) | 120–400         | 3.6–14.5 | 236 / 7.2              |

The wide ranges within each generation reflect the SKU spread (M4 10C → M4 Max 40C). The geometric
mean is a reasonable mid-point estimate. Safari fallback: P25 of all Apple GPUs (conservative:
$B_{mem}=120$ GB/s, $T_{alu}=3.62$ TF).

### 5.6 Bayesian Calibration Update (Optional)

When `jax.calibrate()` observes a real $B_{mem}$ via bandwidth probe, the log-Kalman update refines
$T_{alu}$:

$$\log(\hat{T}_{alu}) = \log(T_{prior}) + w_{TB} \cdot (\log(B_{observed}) - \log(B_{prior}))$$

where $w_{TB} = 1.21$ (Pearson $r = 0.94$, derived from the 49-GPU dataset). This is a single
multiply — no matrix math, always positive.

### 5.7 What Ships in the Repository

| File                                | Purpose                                                                                                                                                                                              | Size   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `scripts/output/runtime_model.json` | Feature classifier, regression coefficients, Apple lookup, architecture constants, median fallbacks, runtime usage pseudocode, **embedded Welford sufficient statistics** for incremental re-fitting | ~15 KB |
| `scripts/refit-model.mjs`           | Incremental GPU addition + coefficient re-derivation from sufficient stats                                                                                                                           | ~11 KB |
| `test/refit-model.test.ts`          | Welford OLS, `addGpu`, `updateCoefficients` unit tests (duplicate protection, Apple lookup, regression derivation)                                                                                   | ~15 KB |

The model is fully self-contained. To add a new GPU:

```bash
node scripts/refit-model.mjs add \
  --model "NVIDIA_RTX_5090" \
  --vendorClass "discrete-modern" \
  --maxBufGB 32 --bwGBs 1792 --tflops 105
```

This updates the Welford statistics incrementally (O(1)) and re-derives all regression coefficients.

## 6. Walk the Pareto Front: High Efficiency, Minimized risk

Instead of arbitrarily picking the absolute theoretical fastest block combination, the
`DangerMultiplier` guides the compiler to land safely on a robust plateau. We can systematically
select static sub-permutations (e.g. `[16,8]` tiling instead of `[16,16]`, or scaling back scan
bindings) that guarantee ~80% of peak performance while keeping WGSL deeply inside the empirical
"safe zone."

## 7. Model Maintenance: Sufficient Statistics + Incremental Re-fit

The offline build scripts that originally produced the model have been retired. All information
needed for re-fitting is embedded directly in `runtime_model.json` as **Welford mean-centered
sufficient statistics** (`_sufficientStats`).

### 7.1 Why Welford Form?

Storing raw data points (49 GPUs × 3 values = 147 numbers) is fragile and verbose. Instead, each
regression stores 5 numbers: `{ mean_x, mean_y, ssXX, ssXY, ssYY }`. OLS coefficients are derived in
closed form:

$$b = \frac{SS_{XY}}{SS_{XX}}, \quad a = \bar{y} - b\bar{x}, \quad R^2 = \frac{SS_{XY}^2}{SS_{XX} \cdot SS_{YY}}$$

The sufficient statistics form is **incrementally updatable** — adding a new GPU is O(1) via
Welford's online algorithm, with no access to the original data points needed.

### 7.2 Adding New GPUs

```bash
# Look up specs on TechPowerUp, maxBufferSize from Dawn defaults or local adapter query
node scripts/refit-model.mjs add \
  --model "NVIDIA_RTX_5090" \
  --vendorClass "discrete-modern" \
  --maxBufGB 32 --bwGBs 1792 --tflops 105

# Apple GPUs: include --appleGen for generation lookup table
node scripts/refit-model.mjs add \
  --model "Apple_M5_12C" \
  --vendorClass "apple" \
  --maxBufGB 16 --bwGBs 200 --tflops 8.5 \
  --appleGen "common-5"

# Re-derive coefficients without adding data (e.g., after manual stats edit)
node scripts/refit-model.mjs refit
```

### 7.3 Data Provenance (Historical)

The original 49-GPU training set was compiled from:

1. **Physical specs** — TechPowerUp, Vulkan DB, Apple developer documentation. Covers NVIDIA
   Turing→Ada, AMD RDNA1→3, Intel Arc/Iris/UHD, Apple M1→M4, AMD APU, and Qualcomm Adreno.
2. **WebGPU limits** — Dawn source defaults, local measurements on RTX 4070 Ti SUPER and Intel Arc
   iGPU, WebGPU CTS conformance reports.
3. **Market distribution** — consensus of 5 AI-synthesized estimates (~330 renderer strings, circa
   2025): desktop discrete ~55%, Apple ~15%, mobile ~12%, Intel IGP ~8%, AMD discrete ~5%.

The build scripts, CSVs, and intermediate analysis artifacts were deleted after baking the
statistics into the model JSON.

**`_seededNamesLost` (data provenance limitation):** The original 35 non-Apple GPUs
(discrete-modern: 24, discrete-legacy: 2, igp: 7, mobile: 2) were seeded from TechPowerUp spec-sheet
aggregates without recording individual GPU names. Welford sufficient statistics are destructive —
the original data points cannot be recovered. Duplicate protection only guards GPUs added _after_
the model was created. New additions are tracked in each class's `gpus` array; the original 35
entries are permanently unnamed.

## 8. Empirical WebGPU Constraints & Thresholds

Based on empirical evidence of WebGPU implementations (Chromium/Dawn, Firefox/wgpu), the following
rules anchor our cost models (specifically $C_{dispatch}$ and `DangerMultiplier` cliffs):

1. **Dispatch Latency ($C_{dispatch}$):** 10µs - 50µs (median ~25µs). Unlike native Vulkan (which
   ranges ~2-5µs), WebGPU passes through JS bindings, IPC to the GPU process, and Dawn validation.
   **Implication:** Operator fusion yields exponential returns in WebGPU. A strategy that issues
   1,000 tiny dispatches incurs ~25ms of purely CPU-side orchestration block.
2. **Register Spilling Cliffs ($R_{opt}$):** WGSL `var<private>` arrays map directly to fast
   thread-local registers up to roughly **64 ~ 255 words (32-bit)**. Deeply unrolled loops or overly
   ambitious matrix tile arrays that breach 255 words suffer massive latency regressions as
   Tint/Naga silently spills them to local VRAM. **Implication:** If $Depth_{priv} > R_{opt}$,
   `DangerMultiplier` acts as a hard cliff edge.
3. **Shared Memory Occupancy ($S_{opt}$):** Keep WGSL `var<workgroup>` memory $\le$ **16KB** per
   workgroup to ensure maximum scheduled wavefront occupancy. While WebGPU allows up to 32KB, modern
   hardware CUs generally have 64KB-100KB total shared arrays. Claiming 32KB forces the CU to only
   schedule 2 wavefronts locally, starving the ALU cores of work and exposing memory latency.
   **Implication:** The penalty $Penalty(Size_{shmem} / S_{opt})$ scales a soft linear degradation
   between 16KB and 32KB.
4. **WGSL Compilation Latency ($C_{compile}$):** Creating a pipeline involves IPC, Tint/Naga
   translation, and native driver compilation. There is a fixed IPC floor (~8ms) and a nonlinear
   driver compilation cost per KB. **Implication:** The cost heuristic is modeled as
   $C_{compile} \approx 8\text{ms} + (Size_{WGSL\_KB} \times 3\text{ms})$.
5. **Wavefront Sizes ($F_{subgroup}$):** Physical execution widths dictate shared memory alignment
   and reduction safety. Apple, Nvidia, and AMD (defaulting to Wave32 for compute) safely map to
   $F_{subgroup} = 32$. Intel IGPs heuristically drop to $16$ under register pressure.
   **Implication:** If the vendor is Intel, pad alignments assuming 16 to prevent out-of-bounds
   execution masking, otherwise use a safe constant of 32.

## 9. Implementation Path

### Phase 1: Offline Analysis (Complete ✅)

1. ✅ 49-GPU training set: physical specs × WebGPU adapter.limits, vendor-conditioned log-linear
   OLS, leave-one-in validation
2. ✅ `runtime_model.json`: feature classifier, regression coefficients, Apple generation lookup,
   architecture constants, Welford sufficient statistics
3. ✅ `refit-model.mjs`: incremental GPU addition + coefficient re-derivation (O(1) Welford update)
4. ✅ Market distribution analysis: ~330 renderer strings, 5-source AI consensus (~2025)
5. ✅ Covariance analysis: log-Kalman gains $w_{TB}=1.21$, $r=0.94$
6. ✅ Offline build scripts retired — model is self-contained
7. ✅ `test/refit-model.test.ts`: comprehensive unit tests for Welford OLS, `addGpu` duplicate
   protection, Apple generation lookup, and `updateCoefficients` regression derivation

### Phase 2: Runtime Integration (Next)

1. **Device characterizer** (`src/profiler/characterize.ts`): Import `runtime_model.json`, implement
   the two-stage classifier + regression inference at device init. Wire into `BackendCapabilities`.
2. **Cost equation module** (`src/profiler/cost.ts`): Implement the `TotalCost` equation from §3.3,
   consuming the continuous hardware estimates.
3. **JIT integration**: Replace existing ad-hoc heuristics in `chooseTileConfig()`,
   `buildNativeAssocScanPlan()`, etc. with cost equation evaluations.

### Phase 3: Optional Calibration

1. **Probe suite** (`src/profiler/probes.ts`): Safe bandwidth and dispatch probes using
   `timestamp-query` with early-termination guards.
2. **`jax.calibrate()`**: Run probes, apply log-Kalman update to refine the regression-based
   estimates. Single multiply per parameter.
3. **Pareto front walker**: Enumerate structurally legal parameter combinations , score via cost
   equation, select minimum-risk optimal path.

## 10. Real-World Distribution Analysis

The training set was cross-referenced against a WebGPU-era market dataset of ~330 renderer strings
with estimated market share among WebGPU-capable browsers (Chrome 113+, Firefox 139+, Safari 18+).
The dataset was a consensus merge of 5 independent AI-synthesized estimates (Gemini, ChatGPT,
Claude, Google AI Mode, Manus), circa 2025.

### 10.1 Key Findings

- Desktop discrete (NVIDIA RTX 30xx/40xx/50xx, AMD RX) dominates at ~55% combined share
- Apple Silicon (M1-M4) represents ~15% of the addressable market
- Mobile (Adreno, Mali, Xclipse) represents ~12%
- Intel IGP (Iris Xe, UHD 7xx) represents ~8%
- AMD discrete (RX 5000-9000) represents ~5%

### 10.2 Data Currency

The market data is WebGPU-era native — all entries are GPUs that actually ship in devices running
Chrome 113+, Firefox 139+, or Safari 18+. Unlike the legacy detect-gpu dataset (2018–2019 WebGL
traffic, 99%+ in the lowest tier), this reflects the actual hardware population our cost model
targets.

_The offline classification scripts and market CSV have been retired. The model’s vendor-class
distributions are baked into the sufficient statistics._
