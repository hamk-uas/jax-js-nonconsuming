# WebGPU AssocScan Binding Limit & einsum Compatibility

## Problem Statement

The DLM 5-tuple `associativeScan` on WebGPU requires **11 storage bindings** (1 constant + 5 input
leaves + 5 output leaves). The WebGPU device limit `maxStorageBuffersPerShaderStage` is typically
**10**. The plan-time guard in `scan-plan.ts` correctly rejects this, falling back to JS Kogge-Stone
(~25 Hz vs ~6 kHz fused).

## Key Finding: einsum Already Works

**`np.einsum("nij,njk->nik")` already works inside `associativeScan` bodies.** The `einsumFastPath`
in `numpy.ts` catches all common batch-matmul subscript patterns and lowers them to `matmul()` /
`swapaxes()` before `parseEinsumExpression` is ever called. Since `matmul` uses relative dimension
indices (`-1`, `-2`), it handles rank-reduced tracers from per-element tracing correctly.

Verified empirically: the full 5-tuple DLM compose with einsum subscripts (`"nij,njk->nik"`,
`"nji,njk->nik"`, `"nij->nji"`) runs correctly inside `associativeScan` on WASM. The fast path
covers every pattern used in the compose body.

**Action items:**

- Remove the `_compose5_einsum` "BROKEN" comment in `test/bench-assoc-scale.test.ts`
- Remove the misleading warning in `lax-associative-scan.ts` that advises against einsum
- Add an explicit test: `einsum("nij,njk->nik")` inside `associativeScan` produces correct results
- Add a test for a non-fast-path subscript that genuinely fails, confirming the general einsum path
  still needs work for exotic patterns

---

## Phase 1: Constants to Uniform Buffers

**Goal:** Move constant inputs from `var<storage, read>` to `var<uniform>`, freeing 1+ storage
binding(s). For the 5-tuple (1 constant = `np.eye(2)`), this brings bindings from 11 → 10, exactly
meeting the device limit.

### Binding arithmetic

| Scenario                    | Storage bindings | Uniform bindings | Fits 10? |
| --------------------------- | ---------------- | ---------------- | -------- |
| Current (5-tuple, 1 const)  | 11               | 0                | No       |
| Phase 1 (consts on uniform) | 10               | 1                | **Yes**  |
| Phase 2 (packed leaves)     | 2                | 1                | Yes      |

### Precedent

The scan-wrapper (`src/backend/webgpu/scan-wrapper.ts`) already uses
`@group(1) @binding(0) var<uniform> ScanOffsets` with dynamic offsets. The pattern is proven.

### Implementation steps

1. **`blockMapFusedShaderSource()` in `src/backend/webgpu/block-map.ts`**
   - Currently emits all inputs (including constants) as `@group(0) @binding(N) var<storage, read>`
   - Change: emit the first `numConsts` inputs as a packed `@group(1) @binding(0) var<uniform>`
     struct instead
   - Constants are read-only and typically small (eye matrix = 16 bytes f32). The WebGPU uniform
     buffer limit is 64 KB — more than sufficient
   - Handle mixed-dtype constants by emitting per-constant members in the struct

2. **Pipeline layout cache key (`src/backend/webgpu.ts`)**
   - Current key: `"${numInputs}:${numOutputs}:${hasUniform ? 1 : 0}"`
   - Change: subtract `numConsts` from `numInputs` in the key since they move to group(1)
   - The `hasUniform` flag is already true when the scan wrapper sets offsets — ensure the uniform
     bind group layout accommodates both offset structs and constant structs

3. **Bind group creation in the executor (`src/backend/webgpu.ts`)**
   - Currently creates a single bind group with all buffers sequential
   - Change: create `@group(0)` with only non-constant buffers, and `@group(1)` with the constant
     data packed into a uniform buffer

4. **`scan-plan.ts` binding guard update**
   - Change `neededBindings = numConsts + 2 * numLeaves` to `neededBindings = 2 * numLeaves` (since
     constants no longer consume storage bindings)
   - Add a size guard: total constant bytes must fit in 64 KB
     (`device.limits.maxUniformBufferBindingSize`)
   - If constants exceed 64 KB, keep them as storage bindings and adjust the count accordingly

### Edge cases

- **Zero constants:** No uniform buffer needed — current code path unchanged
- **Multiple constants:** Pack all into one uniform struct with typed members
- **Large constants (>64 KB):** Fall back to storage bindings for oversized constants; only move
  small ones to uniform. This is unlikely for associative scan (constants are identity matrices,
  small coefficient tensors)

### Testing

- Unit test: Generate a block-map shader with 1 constant on uniform, verify WGSL compiles and
  produces correct results
- Integration test: Run 5-tuple `associativeScan` on WebGPU with block-map path (should now succeed
  where it previously fell back)
- Regression: All existing `lax-associative-scan.test.ts` and `block-map-prototype.test.ts` pass
- Edge case test: Exactly 10 bindings (boundary condition)
- Edge case test: 4-tuple (9 bindings) and 6-tuple (13 bindings → still needs Phase 2)
- Edge case test: Oversized constants (>64 KB total) stay on storage bindings; planner correctly
  falls back when the resulting storage count exceeds the device limit
- Benchmark: 5-tuple throughput on WebGPU before/after — target ≥6 kHz (matching current WASM fused
  path)

---

## Phase 2: Leaf Packing (future, for 6+ tuples)

**Goal:** Pack multiple same-dtype pytree leaves into a single contiguous GPUBuffer, reducing
binding count from `2 * numLeaves` to as low as 2 (1 input + 1 output). Only needed when
`2 * numLeaves` exceeds the device limit minus uniform-migrated constants.

### When this is needed

| Tuple size | Bindings (Phase 1) | Fits 10? | Needs packing? |
| ---------- | ------------------ | -------- | -------------- |
| 2-tuple    | 4                  | Yes      | No             |
| 3-tuple    | 6                  | Yes      | No             |
| 5-tuple    | 10                 | Yes      | No             |
| 6-tuple    | 12                 | No       | **Yes**        |
| 8-tuple    | 16                 | No       | **Yes**        |

### Design: SoA concatenation with uniform offsets

Pack all leaves of the same dtype end-to-end in a single buffer:

```
Buffer layout: [leaf0_data (N×s0 bytes) | pad | leaf1_data (N×s1 bytes) | pad | ...]
```

Where `pad` ensures 256-byte alignment (`minStorageBufferOffsetAlignment`).

**Two implementation strategies:**

**Strategy A: Shader-side offset arithmetic (preferred)**

- Single `@group(0) @binding(0) var<storage, read> packed_in : array<f32>`
- Offsets passed via uniform struct:
  ```wgsl
  struct LeafOffsets { leaf0: u32, leaf1: u32, leaf2: u32, ... }
  @group(1) @binding(0) var<uniform> offsets: LeafOffsets;
  ```
- Shader reads: `packed_in[offsets.leaf2 + gidx * leafStride2 + local_idx]`
- Pro: Minimal buffer management. Con: More complex shader codegen.

**Strategy B: Sub-buffer binding with offsets**

- WebGPU `GPUBufferBinding` supports `{ buffer, offset, size }` on both storage and uniform
  bindings, but `offset` must be a multiple of `minStorageBufferOffsetAlignment` (256 bytes)
- For tiny leaves (2×2 = 16 bytes per element), each leaf's total size is `N × 16`; the offset for
  the next leaf can be 256-aligned since N is typically large
- Pro: No shader changes needed — each leaf still gets its own `@binding`. Con: Doesn't reduce
  binding count! Each sub-buffer binding still counts toward `maxStorageBuffersPerShaderStage`.

**Verdict: Strategy A is the only viable one** — Strategy B doesn't reduce binding count.

### Implementation steps

1. **Packing allocator in `lax-associative-scan.ts` or `scan-plan.ts`**
   - After `tree.flatten()`, group leaves by dtype
   - Allocate one contiguous GPUBuffer per dtype group
   - Compute per-leaf byte offsets with 256-byte alignment
   - Copy individual leaf buffers into the packed buffer (or allocate packed from the start)

2. **WGSL codegen changes in `blockMapFusedShaderSource()`**
   - Emit 1 packed storage binding per dtype group instead of N bindings
   - Emit offset constants or uniform members for each leaf
   - Rewrite all `inK[idx]` references to `packed_in[offset_K + idx]`

3. **Executor changes in `webgpu.ts`**
   - Create bind groups with packed buffers instead of individual leaf buffers
   - After execution, extract results from packed output buffer back into individual leaf arrays

### DLM 5-tuple specifics

All 5 leaves are f32. Sizes per element: A=16B, b=8B, C=16B, eta=8B, J=16B (total 64B). For N=12800:
packed buffer = 819,200 bytes (well within GPU limits). Single buffer for input, single buffer for
output → 2 storage bindings + 1 uniform = 3 total.

### Testing

- Unit test: Pack/unpack roundtrip with known values
- Integration test: 6-tuple or larger `associativeScan` on WebGPU succeeds
- Correctness test: Packed results match unpacked results (compare against WASM baseline)
- Alignment test: Verify 256-byte padding doesn't corrupt adjacent leaf data
- Mixed-dtype test: f32 + i32 leaves in a single scan (separate packing per dtype)
- Gradient test: `grad(associativeScan(packed_compose))` produces correct gradients

---

## Prioritization

| Phase                      | Effort                                      | Impact                         | Recommendation |
| -------------------------- | ------------------------------------------- | ------------------------------ | -------------- |
| einsum cleanup             | Small (remove comments/warnings, add tests) | Correctness of docs            | **Do first**   |
| Phase 1: Uniform constants | Medium (shader + executor + plan changes)   | **Unblocks 5-tuple on WebGPU** | **Do second**  |
| Phase 2: Leaf packing      | Large (new allocator + shader rewrite)      | Unblocks 6+ tuples             | Do when needed |

Phase 1 alone is sufficient to enable the 5-tuple DLM compose on WebGPU's fused block-map path.
Phase 2 is insurance for the future and can be deferred until a real 6+ tuple use case materializes.

---

## Downstream: dlm-js integration (from `issues/jax-js-webgpu-block-map-perf-regression.md`)

### Current state

`dlmFit(algorithm: 'assoc')` runs two `associativeScan` calls:

1. **Forward Kalman filter** — 5-tuple `{A, b, C, eta, J}` with `np.linalg.inv` and
   `np.einsum`/`np.matmul`. 11 bindings → WebGPU fallback.
2. **Backward RTS smoother** — 3-tuple `{A, b, S}`, 0 constants, 6 bindings → **already fits within
   the WebGPU limit**. Uses the 3-operand einsum `"nij,njk,nlk->nil"` (in fast path).

WebGPU `dlmFit` is currently 19–450× slower than WASM depending on N, entirely due to dispatch
overhead in the Kogge-Stone fallback.

### What Phase 1 unblocks for dlm-js

- **Forward scan:** 11 → 10 bindings → block-map fusion activates. Each Kogge-Stone round becomes 1
  fused dispatch instead of ~10 separate kernel dispatches.
- **Backward scan:** Already 6 bindings — should activate block-map fusion today. Investigate why it
  doesn't (may be a separate issue: `np.linalg.inv` or 3-operand einsum in the body preventing
  compiled-loop analysis).

### Action items for dlm-js

- After Phase 1 lands in jax-js, update dlm-js to latest jax-js dependency
- Verify forward scan activates block-map path (check with `setDebug(1)`)
- Investigate backward 3-tuple: if block-map isn't activating, file a separate issue (likely the
  3-operand einsum `"nij,njk,nlk->nil"` lowering to 2 sequential `matmul` calls, preventing
  single-kernel fusion)
- Re-run `issues/repro-webgpu-block-map-perf.ts` and update the benchmark table
- Target: WebGPU/WASM ratio ≤ 2× for warm runs (currently 19× at N=100)

### Benchmark targets (Nile order=1, m=2, warm)

| Metric             | Current   | After Phase 1 (target) |
| ------------------ | --------- | ---------------------- |
| WebGPU warm N=100  | ~126 ms   | ≤15 ms                 |
| WebGPU warm N=3200 | ~1,561 ms | ≤50 ms                 |
| GPU/WASM ratio     | 19–450×   | ≤2×                    |

These targets assume block-map fusion collapses each Kogge-Stone round to 1 dispatch. If the compose
body contains ops that prevent fusion (inv, 3-op einsum chains), the actual improvement may be less,
but dispatch count should still drop by 5–10×.

---

## Non-goals

- **General einsum rank-adaptation:** The general `parseEinsumExpression` still fails with
  rank-reduced inputs (e.g., 3-index subscript with 2D tensor). This only matters for exotic
  subscript patterns NOT in `einsumFastPath`. All common batch-matmul patterns are already handled.
  Fixing the general parser would be a separate, lower-priority effort.
- **Multiple bind groups for overflow:** WebGPU supports up to 4 bind groups, but storage bindings
  per shader stage is a global limit — splitting across groups doesn't help.
