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

**Action items:** ✅ All complete

- ~~Remove the `_compose5_einsum` "BROKEN" comment in `test/bench-assoc-scale.test.ts`~~
- ~~Remove the misleading warning in `lax-associative-scan.ts` that advises against einsum~~
- ~~Add an explicit test: `einsum("nij,njk->nik")` inside `associativeScan` produces correct
  results~~
- ~~Add a test for a non-fast-path subscript that genuinely fails, confirming the general einsum
  path still needs work for exotic patterns~~

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

| Phase                       | Effort                                 | Impact                         | Recommendation |
| --------------------------- | -------------------------------------- | ------------------------------ | -------------- |
| ~~einsum cleanup~~          | ~~Small~~                              | ~~Correctness of docs~~        | **Done** ✅    |
| ~~Phase 1: Uniform consts~~ | ~~Medium~~                             | ~~Unblocks 5-tuple on WebGPU~~ | **Done** ✅    |
| Phase 2: Leaf packing       | Large (new allocator + shader rewrite) | Unblocks 6+ tuples             | Do when needed |

Phase 1 alone is sufficient to enable the 5-tuple DLM compose on WebGPU's fused block-map path.
Phase 2 is insurance for the future and can be deferred until a real 6+ tuple use case materializes.

---

## Downstream: dlm-js integration (from `issues/jax-js-webgpu-block-map-perf-regression.md`)

### Current state (post-`2554290`)

`dlmFit(algorithm: 'assoc')` runs two `associativeScan` calls:

1. **Forward Kalman filter** — 5-tuple `{A, b, C, eta, J}` with `np.linalg.inv` and
   `np.einsum`/`np.matmul`. ~~11 bindings → WebGPU fallback.~~ Now 10 bindings via uniform constants
   → **block-map fused path activates** ✅
2. **Backward RTS smoother** — 3-tuple `{A, b, S}`, 0 constants, 6 bindings → **block-map fused path
   activates** ✅

Both scans use the fused WebGPU shader path. Correctness verified (var<private> fix in `2554290`).
The remaining performance gap is from the 373 kernel dispatches outside the scans (element
construction + diagnostics), not from the scans themselves.

### What Phase 1 unblocks for dlm-js

#### Forward scan (5-tuple)

For **m=2** (Nile model), `np.linalg.inv` uses the analytical `inv2x2` Cramer's rule path — all
elementwise ops, no Routine. The body is fully kernel-fusable. The **only** blocker is the binding
count: 1 const + 5 in + 5 out = 11 > 10. Phase 1 (constants→uniform) brings this to 10 storage
bindings, enabling the block-map fused shader.

For **m≥5**, `inv` falls through to `solve(a, eye)` which uses LU (a Routine). Routines are rejected
by `blockMapFusedShaderSource`. Phase 1 alone is insufficient; Phase 2 (leaf packing) + Routine
support would both be needed. This is out of scope for now.

- **Backward scan (3-tuple):** Already 6 bindings → block-map path should activate. If not,
  investigate whether the 3-operand einsum `"nij,njk,nlk->nil"` (lowered to 2 sequential matmuls by
  the fast path) prevents single-kernel fusion.

### Action items for dlm-js — ✅ Complete

- ~~After Phase 1 lands in jax-js, update dlm-js to latest jax-js dependency~~
- ~~Verify forward scan (m=2) activates block-map path (check with `setDebug(1)`)~~
- ~~Re-run `issues/repro-webgpu-block-map-perf.ts` and update the benchmark table~~

Both forward 5-tuple and backward 3-tuple activate the WebGPU block-map fused path.

### Benchmark results (Nile order=1, m=2, warm, RTX 4070 eGPU)

| Metric             | Before    | Phase 1 only | All fixes (2554290) |
| ------------------ | --------- | ------------ | ------------------- |
| WebGPU warm N=100  | ~126 ms   | ~123 ms      | ~124 ms             |
| WebGPU warm N=800  | —         | —            | ~416 ms             |
| WebGPU warm N=3200 | ~1,561 ms | —            | ~1,429 ms           |
| GPU/WASM ratio     | 19–450×   | 13–63×       | 19–154×             |

**Commits applied:** `10ae2aa` (uniform constants), `cd8d47a` (einsum cleanup), `a2062fa` (dispatch
batching), `2554290` (var<private> fix). All correctness issues resolved.

### Root cause analysis: 764 kernel dispatches

**Critical finding:** On this hardware (RTX 4070 via Deno),
`maxStorageBuffersPerShaderStage = 1,048,576`, so `maxArgs = 1,048,575`. The P2 (forward
binding-limit) pass in `splitGraphDataflow` **never fires**. All 764 kernel dispatches are caused by
**P1 rules** (backward black-node identification):

| P1 Rule             | Description                               | Approximate contribution |
| ------------------- | ----------------------------------------- | ------------------------ |
| Reduction endpoints | Each matmul/dot forces a kernel boundary  | 288 dispatches (38%)     |
| Diamond heuristic   | Node reaching 2+ black nodes materializes | ~200 dispatches (26%)    |
| Jaxpr outputs       | Final outputs must materialize            | ~40 dispatches (5%)      |
| Clean-shape inputs  | Pad/Concat need clean inputs              | ~30 dispatches (4%)      |
| Heterogeneous views | Gather/DynamicSlice force materialization | ~20 dispatches (3%)      |
| Constant fills      | Zero-arg identity/zero fills              | 37 dispatches (5%)       |
| Other (cascade)     | Already-black outputs trigger cascade     | ~145 dispatches (19%)    |

Full `dlmFit` dispatches (2 passes × 382 per pass = 764 total):

| Kernel property      | Count | Notes                                            |
| -------------------- | ----- | ------------------------------------------------ |
| With reduction       | 288   | Matmul/dot — inherent dispatch boundaries        |
| Without reduction    | 476   | Elementwise ops materialized by P1 rules         |
| Size ≤ 4             | 86    | Scalar/tiny — pure overhead, negligible GPU work |
| Size 5-100           | 189   | Small matrix ops                                 |
| Size 101-400         | 489   | Full-size ops (N×m² where m=2)                   |
| Multi-output kernels | 26    | Already fused (2-4 outputs each)                 |

**Overhead breakdown (warm, N=100):**

| Component                                                             | Time       | % of total |
| --------------------------------------------------------------------- | ---------- | ---------- |
| `queue.submit()`                                                      | 22ms       | 18%        |
| `createBindGroup()`                                                   | 6ms        | 5%         |
| `prepareKernelSync()`                                                 | 1ms        | 1%         |
| JIT loop overhead (scope, refcount, batch assembly, command encoding) | ~95ms      | 76%        |
| **Total**                                                             | **~124ms** | 100%       |

The bottleneck is **JS-side overhead in the JIT execution loop**, not GPU compute or even WebGPU API
calls. Each of the 764 dispatches costs ~132µs in the loop: array allocations for `ins[]` /
`outs[]`, scope lookups, `incRef`/`decRef` refcounting, batch object creation, and command encoder
recording. WASM avoids all of this via mega-module compilation (entire program → single native
function call, 6ms total).

**Scaling:** At N=3200 (M=13 blocks), each assocScan adds ~40 extra block-map dispatches. Combined
with GPU compute time scaling linearly with N, the total reaches ~1,429ms.

**Browser note:** In Chrome, `maxStorageBuffersPerShaderStage` is typically **8-10** (spec minimum).
On Chrome, the P2 pass would additionally fragment the dependency graph, producing even more
dispatches than the 764 seen here. The analysis above represents the **best case** (Deno with
driver-level limits).

### Remaining optimization opportunities (jax-js)

| ID     | Approach                                                                           | Impact                                                          | Effort                                                                      |
| ------ | ---------------------------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ~~O1~~ | ~~**Diamond heuristic relaxation: allow recomputation for cheap ops**~~            | ~~High — could reduce 476 non-reduction dispatches by ~40-60%~~ | **Done** ✅                                                                 |
| ~~O2~~ | ~~**Scalar promotion: compute size ≤ 4 kernels on CPU, pass as constants**~~       | ~~Low-Medium — eliminates 86 tiny dispatches (~14ms)~~          | **Done** ✅                                                                 |
| O3     | **Bind group caching for JIT programs with stable pipeline→slot mappings**         | Low — `createBindGroup` is only 6ms/764 calls                   | Medium — cache keyed by (pipeline, slot[])                                  |
| O4     | **Single-pass dlmFit: merge Pass 1 + Pass 2 into one jit call**                    | Medium — eliminates ~382 dispatches + 1 readback                | Medium — downstream restructuring                                           |
| O5     | **Pre-encoded command buffer: record commands once, replay with buffer rebind**    | High — eliminates per-dispatch loop overhead (~95ms)            | Large — needs WebGPU "render bundle" equivalent for compute (not available) |
| O6     | **Multi-reduction kernel: fuse chains of matmul+elementwise into single dispatch** | High — could merge pairs of dot→elemwise→dot→elemwise           | Very Large — fundamentally new codegen                                      |

~~O1 (diamond relaxation) is the highest-impact feasible optimization. The diamond heuristic
currently forces materialization whenever a node's output reaches 2+ distinct black nodes. For cheap
operations (unary, binary with small literal), the recomputation cost is negligible compared to the
dispatch overhead (~132µs). Allowing such recomputation would let these ops fuse into their
downstream reduction epilogues instead of becoming separate dispatches.~~

**O1 implemented:** The P1 diamond heuristic now exempts "cheap recompute" ops — unary elementwise
(neg, cast, sqrt, exp, etc.), binary with at least one literal input (add(x, 1), mul(x, 0.5)), and
Where with ≥2 literal inputs. These ops are duplicated into each downstream kernel instead of being
materialized as separate dispatches. The criterion matches the reduction epilogue fusability rules.
`setDebug(1)` now reports `cheapDiamonds=N` showing how many diamonds were relaxed.

**O2 implemented:** `pushLit` now computes the constant scalar value on the CPU at JIT compile time
and embeds it as `initialData` on the `malloc` step. The backend fills the buffer via
`queue.writeBuffer` (WebGPU) or `memcpy` (WASM) instead of dispatching a zero-input kernel. This
eliminates all constant-fill GPU dispatches (previously one shader dispatch per Lit scalar in the
jaxpr). The mega-module emits inline `i32.store` / `i32.store16` instructions for the data. Buffers
with `initialData` are exempt from recycling (always fresh allocation) since they're tiny (2–8
bytes).

O5 (pre-encoded commands) is the theoretical ideal — it would eliminate the 95ms JS loop overhead
entirely — but WebGPU has no equivalent of Vulkan's secondary command buffers or Metal's indirect
command buffers for compute dispatches. This is a WebGPU API limitation.

---

## O6: Multi-Reduction Kernel — Detailed Design

### Problem

Each matmul (Dot → Reduce) creates a kernel dispatch boundary in the **standalone JitProgram
execution loop**. In a DLM compose body (3-tuple, m=2), the matmul steps are:

```
newA = dot(q.A, p.A)                    # reduce(K=2), inputArgs=[q.A, p.A]
Ab   = dot(q.A, p.b)                    # reduce(K=2), inputArgs=[q.A, p.b]
newB = Ab + q.b                          # fused epilogue of Ab
AS   = dot(q.A, p.S)                    # reduce(K=2), inputArgs=[q.A, p.S]
qAT  = transpose(q.A)                   # view (zero cost)
ASAT = dot(AS, qAT)                     # reduce(K=2), inputArgs=[AS, qAT]
newS = ASAT + q.S                        # fused epilogue of ASAT
```

**4 reduction dispatches** in the standalone JIT execution loop.

**Important caveat — associativeScan WebGPU already fuses these.** On WebGPU, `associativeScan`
compiles the compose body into a **block-map fused WGSL shader**. All 4 matmul reductions execute as
sequential inline code within one shader dispatch (per-element ridx loops with `var<private>`
accumulators, barriers between steps, intermediates in `var<workgroup>`).

**Measured dispatch counts** (NVIDIA RTX 4070 Ti SUPER, Chromium headless, `dispatchCount` counter):

| N         | 2-tuple dispatches | 3-tuple dispatches |
| --------- | ------------------ | ------------------ |
| 1–256     | **1**              | **1**              |
| 257–10000 | **3**              | **3**              |

Crossover at N=257 (blockSize=256). Below blockSize, a single fused shader handles everything.
Above, 3 dispatches: Phase 1 (local scan) + Phase 3 (recursive prefix) + Phase 4 (apply). **Body
complexity has zero effect on dispatch count** — 2-tuple and 3-tuple are identical.

**O6's target audience is therefore:**

1. **Standalone JitProgram execution** — matmul chains NOT inside scan/associativeScan
2. **WASM compiled-loop / mega-module** — tighter inner loop from fewer kernel steps
3. **Fallback scan paths** — when the block-map fused shader can't be generated

### Opportunity: Two independent fusion strategies

**Strategy A: Multi-output reduction kernel (independent reductions)**

Fuse dispatches 1, 2, 3 into a **single** dispatch that produces 3 output buffers. These share the
same input arguments and the same reduction size. Each output has its own independent reduction
expression + epilogue.

```
                    ┌─ out0: reduce(q.A * p.A, Add, K=2) → newA
dispatch 1 (fused): ├─ out1: reduce(q.A * p.b, Add, K=2) → Ab + q.b → newB
                    └─ out2: reduce(q.A * p.S, Add, K=2) → AS
```

**Strategy B: Chained reduction (dependent reductions)**

Fuse dispatch 3 → dispatch 4 when the output of one reduction feeds directly into the next.
`ASAT = dot(AS, qAT)` where `AS = dot(q.A, p.S)`. The intermediate `AS` lives in registers:

```
dispatch (chained): ridx1 loop: acc1 += q.A[gidx,ridx1] * p.S[ridx1,:]
                    ridx2 loop: acc2 += acc1[:] * qAT[ridx2,gidx]  ← reads acc1 as input
                    store: acc2 + q.S
```

Strategy A is **much simpler** and covers many more cases. Strategy B is theoretically more powerful
but requires fundamentally different codegen (register-resident intermediates).

### Strategy A: Multi-output reduction kernel

#### Current barrier

`Kernel.multi()` ([alu.ts:1569](alu.ts#L1569)) explicitly forbids reductions. This was a
simplification: the codegen for multi-output kernels (`pipelineSourceMulti` in webgpu.ts,
`codegenWasmMulti` in wasm.ts) doesn't handle the ridx loop / shared-memory tree.

`flushPendingKernels()` ([jit.ts:1523](jit.ts#L1523)) routes all reduction entries to `soloEntries`,
skipping multi-output grouping.

#### Fusion criteria

Two reduction kernel entries can fuse into a multi-output reduction kernel iff:

1. **Same reduction size** — all must reduce over the same K
2. **Same reduction op** — all must be `AluOp.Add` (or all `Mul`, etc.)
3. **Same gidx size** — all output arrays have the same number of elements
4. **Same input args** — shared `inputArgs` (already required for multi-output)
5. **Independent** — no data dependency between outputs (output of one is not input to another)
6. **Combined args within limit** — `nargs + numOutputs ≤ maxArgs`

All of these are **already checked** for non-reduction multi-output kernels except (1), (2), and
(5). Criteria (1) and (2) are new. Criterion (5) is automatically satisfied because dependent
reductions are in separate `reductionEndpointEqns` and chained via epilogue.

**Critical limitation — DLM compose does NOT satisfy criterion (4).** In the 3-tuple body:

- Dot 1 (`q.A × p.A`): `inputArgs = [jitIdOf(q.A), jitIdOf(p.A)]`
- Dot 2 (`q.A × p.b`): `inputArgs = [jitIdOf(q.A), jitIdOf(p.b)]`
- Dot 3 (`q.A × p.S`): `inputArgs = [jitIdOf(q.A), jitIdOf(p.S)]`

Each dot reads a different second operand → different `inputArgs` → different grouping keys → **zero
fusions under Strategy A**. The same applies to the 5-tuple compose (every dot reads a unique input
combination).

**Relaxation: input-arg union.** To fuse dots 1-3, the kernel would need the union of their inputs:
`[q.A, p.A, p.b, p.S]` = 4 inputs + 3 outputs = 7 bindings. Each output's `exp` and `reduction`
would index into this union via remapped GlobalView gid references. This is a larger design change
than originally planned (requires gid reindexing in AluExp trees) and still wouldn't help on WebGPU
where the body is already fused by block-map.

#### Reduction dtype matching

All fused reduction outputs must share the same reduction dtype (for shared-memory workspace
sizing). If reduction dtypes differ, fall back to solo dispatch. In practice, DLM bodies use uniform
f32 throughout.

#### Kernel type changes

```typescript
// Current (alu.ts):
static multi(nargs, size, outputDescs: { exp, reduction? }[]): Kernel
// throws if any reduction is present

// Proposed:
static multi(nargs, size, outputDescs: { exp, reduction? }[]): Kernel
// ALLOW reductions IF all outputs share the same reduction op, size, and dtype
// (or all have no reduction, as before)
```

Validation: either ALL outputs have reduction with matching `(op, size, dtype)`, or NONE have
reduction. Mixed reduction/non-reduction outputs are not supported (different loop structures).

#### WASM codegen changes (`codegenWasmMulti` → `emitKernelBody`)

Current `codegenWasmMulti` generates a gidx loop that evaluates each output expression and stores.
For multi-reduction:

```
for gidx in 0..size:
  // Initialize N accumulators
  acc0 = identity; acc1 = identity; acc2 = identity;
  for ridx in 0..K:
    val0 = exp0(gidx, ridx); acc0 = combine(acc0, val0);
    val1 = exp1(gidx, ridx); acc1 = combine(acc1, val1);
    val2 = exp2(gidx, ridx); acc2 = combine(acc2, val2);
  // Apply per-output epilogues
  out0[gidx] = epilogue0(acc0)
  out1[gidx] = epilogue1(acc1)
  out2[gidx] = epilogue2(acc2)
```

**Key insight:** All N accumulators share the **same ridx loop** since they have the same reduction
size. The loop body evaluates N expressions (one per output) and accumulates into N independent
accumulators. After the loop, each output applies its own epilogue.

This is straightforward: the WASM gidx loop already exists; we add N `acc` locals, evaluate N
expressions per ridx iteration, and store N epilogue results per gidx.

Parallel dispatch (M5.3): Works unchanged — `start`/`end` locals split the gidx range across
threads, each produces partial ranges of all N outputs.

#### WebGPU codegen changes (`pipelineSourceMulti` + `pipelineSource`)

The multi-output WebGPU path currently skips reductions entirely. Two sub-strategies:

**Sub-A: Per-thread reduction (small K)**

For small reduction sizes (K ≤ ~64), each thread can accumulate independently without shared-memory
coordination:

```wgsl
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let gidx = i32(id.x);
  if (gidx >= SIZE) { return; }

  // N independent accumulators
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;

  for (var ridx: i32 = 0; ridx < K; ridx++) {
    acc0 += exp0(gidx, ridx);
    acc1 += exp1(gidx, ridx);
    acc2 += exp2(gidx, ridx);
  }

  result0[gidx] = epilogue0(acc0);
  result1[gidx] = epilogue1(acc1);
  result2[gidx] = epilogue2(acc2);
}
```

**When this applies:** `gidx_size ≥ workgroupSize` (enough threads to fill a workgroup without
needing shared-memory reduction). For matmul [M,N]×[N,K], the gidx size is M×K. For 2×2 matrices:
gidx=4, K=2 — this is the per-thread path.

**Sub-B: Shared-memory reduction (large K)**

When the tuner selects shared-memory reduction (workgroup cooperates on one gidx element), the
codegen needs N separate shmem regions (one per output) and N tree-reduction passes. This increases
shared memory by N× and adds N-1 extra workgroupBarrier() sequences.

```wgsl
var<workgroup> shmem0: array<f32, 256>;
var<workgroup> shmem1: array<f32, 256>;
var<workgroup> shmem2: array<f32, 256>;

// ridx loop with N accumulators...
// subgroup/tree reduction for each output...
// thread-0 stores all N results
```

**Feasibility:** shmem is limited to ~16 KB. For f32 with groupSize=256: 1024 bytes per output. Up
to **16 outputs** (16×1024 = 16384) fit. In practice, DLM compose produces 3-5 reduction outputs —
easily within limits.

#### Tuner changes

`tuneWebgpu()` and `tuneWasm()` must handle multi-output reduction kernels. The tuning decisions
(unroll factor, upcast, groupSize) should be uniform across all outputs since they share the ridx
loop. Use the first output's expression for tuning decisions.

#### splitGraphDataflow changes

No changes needed. P1 already marks reduction endpoints as black nodes. The fusion happens in
`flushPendingKernels`, which runs after P1.

However, we need a subtle change to the **grouping key** in `flushPendingKernels`:

```typescript
// Current key for non-reductions:
const key = `${sizeExprKey(entry.size)}:${entry.inputArgs.join(",")}`;

// Proposed key for reductions (new group):
const key = `red:${sizeExprKey(entry.size)}:${entry.inputArgs.join(",")}:${entry.reduction.op}:${sizeExprKey(entry.reduction.size)}:${entry.reduction.dtype}`;
```

Only reductions with matching `(size, inputArgs, reductionOp, reductionSize, reductionDtype)` can
merge.

#### Mega-module changes

The mega-module rejects symbolic sizes but accepts concrete reduction kernels. Multi-output
reduction kernels would need the WASM codegen changes above. Since the mega-module calls the same
`emitKernelBody` / `codegenWasm*` functions, this should propagate naturally.

### Implementation phases

#### Phase A: WASM multi-output reduction (Medium effort)

1. Remove `Kernel.multi()` restriction — allow uniform reductions
2. Add reduction grouping in `flushPendingKernels` with new key
3. Extend `codegenWasmMulti` to handle N accumulators in shared ridx loop
4. Update tuner for multi-output reduction
5. Tests: matmul chains produce fewer execute steps
6. Verify mega-module compatibility

**Expected impact on DLM 3-tuple (WASM, m=2):**

- Body dispatches: 4 → 2 (fuse independent dots 1-3 into 1, keep dependent dot 4 solo)
- Per-iteration overhead: ~528µs → ~264µs (2 dispatches × 132µs)
- But in compiled-loop scan, WASM overhead is already negligible (~6ms total for N=100)
- **Real benefit: enables mega-module to inline the fused kernel**

#### Phase B: WebGPU multi-output reduction (Medium-Large effort)

1. Extend `pipelineSourceMulti` to handle per-thread reduction (Sub-A)
2. Extend `pipelineSource` to handle shared-memory multi-output reduction (Sub-B)
3. Update ShaderInfo return for multi-output reduction metadata
4. Tests: WebGPU dispatch count reduced for matmul chains

**Expected impact:** Only affects standalone JitProgram execution (not scan/associativeScan bodies,
which are already fused by block-map). Benefits matmul chains outside of scan. Does NOT help DLM.

#### Phase C: Chained reduction (Strategy B) — Deprioritized

Dependent reduction fusion (`dot(dot(A,B), C)` in 1 dispatch). Requires register-resident
intermediates and nested ridx loops. As analyzed above, this does **not** help DLM on WebGPU
(block-map fused shader already chains through shmem). Only valuable for **standalone JitProgram
large-matrix patterns** like `softmax(Q @ K^T) @ V` where the intermediate is large.

**Architectural requirements** (unchanged):

- New AluExp nodes for "local array" (register-resident intermediate)
- Nested reduction codegen (outer gidx loop → inner ridx1 loop → store to local → inner ridx2 loop
  reading from local)
- New Kernel variant or extension to represent chained reductions
- Significant complexity; deferred until Phase A/B demonstrate value

### Risk assessment

| Risk                                                        | Likelihood | Mitigation                                              |
| ----------------------------------------------------------- | ---------- | ------------------------------------------------------- |
| Multi-output reduction increases register pressure          | Medium     | Fall back to solo for >4 outputs                        |
| Shared-memory overflow (WebGPU Sub-B, many outputs)         | Low        | Guard: `numOutputs × groupSize × byteWidth ≤ maxShmem`  |
| Tuning divergence across outputs (different optimal unroll) | Low        | Use uniform tuning from first output                    |
| Interaction with O1 cheap-recompute diamond relaxation      | None       | Diamond operates on P1; fusion in `flushPendingKernels` |
| Mega-module inline failure for multi-output reduction       | Low        | Already handles multi-output elementwise                |

### Dispatch reduction estimate (measured)

**Strategy A does NOT accelerate DLM on WebGPU** for two independent reasons:

1. **inputArgs mismatch** — The 3 independent dots in the 3-tuple compose each read a different
   second operand (`p.A`, `p.b`, `p.S`). `flushPendingKernels` groups by `size:inputArgs.join(",")`
   → different keys → no fusion. The 5-tuple is worse: ~15 dots, each with a unique input pair. Even
   with input-arg union relaxation, the optimization is moot because:

2. **Block-map fused shader** — Measured: **1 dispatch** for N≤256, **3 dispatches** for N>256. The
   entire compose body (all 4 matmul reductions + elementwise ops) compiles into a single fused WGSL
   shader. The 3 dispatches for N>256 are Phase 1/3/4 of hierarchical decomposition — each runs the
   full body, not individual steps. Body complexity is irrelevant.

**Where Strategy A helps (non-DLM, non-scan):**

| Scenario                                    | Current | After A   | Savings    |
| ------------------------------------------- | ------- | --------- | ---------- |
| Standalone `jit(() => { 3× matmul chain })` | 3       | 1 (maybe) | 2 dispatch |
| General JitProgram with N same-input dots   | N       | 1         | N-1        |

These require the "same inputArgs" criterion or the input-arg union relaxation. The value is narrow:
same-input independent reductions are uncommon in real workloads.

**Strategy A for WASM:** Fusible independent reductions sharing inputs would produce a tighter
compiled-loop / mega-module inner kernel. The benefit is marginal (native function call overhead is
~ns, not ~µs). The primary remaining value of Strategy A is **architectural cleanliness** — lifting
the Kernel.multi() restriction and letting the compiler fuse what it can.

### Strategy B: chained reduction — WebGPU DLM analysis

**Strategy B also does NOT accelerate DLM on WebGPU.**

Within the block-map fused shader, dependent reductions already chain through `var<workgroup>`:

```wgsl
// Step 2: AS = dot(q.A, p.S) — writes to shmem pingPong[outputSlot]
for (var gidx: i32 = 0; gidx < 4; gidx++) {
  var acc: f32 = 0.0;
  for (var ridx: i32 = 0; ridx < 2; ridx++) {
    acc += qA[tidx * 4 + ...] * pS[tidx * 4 + ...];
  }
  pingPong_out[tidx * 4 + gidx] = acc;
}
workgroupBarrier();

// Step 3: ASAT = dot(AS, qAT) — reads AS from shmem, no extra dispatch
for (var gidx: i32 = 0; gidx < 4; gidx++) {
  var acc: f32 = 0.0;
  for (var ridx: i32 = 0; ridx < 2; ridx++) {
    acc += pingPong_out[tidx * 4 + ridx * ...] * qAT[tidx * 4 + ...];
  }
  pingPong_out[tidx * 4 + gidx] = acc + qS[tidx * 4 + gidx]; // epilogue
}
workgroupBarrier();
```

The dependent chain (dot 2 → dot 3) is **already sequential within the fused shader**. The
intermediate `AS` lives in `var<workgroup>` shared memory — one shmem write + one shmem read of 4
floats (16 bytes) per thread. Strategy B's register-resident intermediates (`var<private>`) would
save this shmem round-trip, but for 2×2 matrices the savings are negligible (~16 bytes at shared
memory bandwidth ~100+ GB/s). The bottleneck is the Kogge-Stone ping-pong pattern and global memory
gathers, not inter-step shmem traffic.

**Where Strategy B could help (non-DLM):**

- Large matrices (e.g., 128×128) in standalone JitProgram where the chained reduction would save a
  full GPU dispatch and a global memory round-trip of the intermediate
- Patterns like `softmax(Q @ K^T) @ V` where QK^T is large and immediate consumption avoids
  materializing it to global memory

### Conclusion: O6 scope revision

O6 as designed targets a bottleneck that **does not exist on WebGPU for associativeScan workloads**.
The block-map fused shader already achieves the fusion that Strategy A and B aim for, but at the
shader level rather than the JitProgram step level.

**Remaining value:**

| Target                         | Strategy A                | Strategy B                |
| ------------------------------ | ------------------------- | ------------------------- |
| DLM scan body (WebGPU)         | No effect (already fused) | No effect (already fused) |
| DLM scan body (WASM)           | Marginal (~ns/step)       | Marginal (~ns/step)       |
| Standalone JitProgram (WebGPU) | Helps if same inputArgs   | Helps for chained dots    |
| Standalone JitProgram (WASM)   | Marginal                  | Marginal                  |
| Non-scan large-matrix chains   | Moderate                  | Significant               |

**Recommendation:** Deprioritize O6. The DLM use case — the primary motivation — is already served
by the block-map fused shader. Effort is better spent on:

- Subgroup matrix ops (WMMA) for hardware tensor cores in tiled matmul (P7)
- Conv2d tuning (P4)
- Relaxed SIMD FMA for WASM matmul (P2)

---

## P7: Subgroups & Cooperative Matrix — Forward-Looking Plan

### Overview

Subgroup operations are the single most impactful GPU feature for jax-js performance. They expose
SIMD-level (SIMT) parallelism — threads within a hardware "wave" (NVIDIA warp = 32 threads, Intel EU
= 8–32 threads) can communicate via registers without shared memory. This is strictly cheaper than
`var<workgroup>` paths.

### Chrome subgroups timeline

| Chrome  | Feature                                                                                        | Status        |
| ------- | ---------------------------------------------------------------------------------------------- | ------------- |
| 125     | Subgroups announced (feature in development)                                                   | Experimental  |
| 128     | Origin trial: `subgroupBallot`, `subgroupBroadcast`, `subgroup_invocation_id`, `subgroup_size` | Experimental  |
| 129     | Expanded: `subgroupAdd`, `subgroupAll`, `subgroupShuffle`, etc.                                | Experimental  |
| 131     | `subgroupInclusiveAdd`, `subgroupInclusiveMul`                                                 | Experimental  |
| 132     | Extended subgroups experimentation                                                             | Experimental  |
| 133     | Experimental subgroup features cleanup                                                         | Cleanup       |
| **134** | **Subgroups shipped as standard feature**                                                      | **Stable** ✅ |
| 144     | `subgroup_id` extension                                                                        | Stable        |
| 145     | `subgroup_uniformity` extension                                                                | Stable        |

Google Meet reported **2.3–2.9× speedup** using subgroups for matrix-vector multiply shaders during
the origin trial (Chrome 128–131).

### What we already use (P5 — Done)

| Feature                   | Where                              | Impact                          |
| ------------------------- | ---------------------------------- | ------------------------------- |
| `subgroupAdd/Mul/Min/Max` | JIT reduction (webgpu.ts)          | Fewer shmem steps               |
| `subgroupAdd/Mul/Min/Max` | block-map reduction (block-map.ts) | Fewer shmem steps               |
| Three-phase tree pattern  | Both paths                         | Correct for any `subgroup_size` |

**Pattern:** Subgroup reduce → SG leaders write to `var<workgroup>` → inter-subgroup tree in shmem.
Fallback: standard shared-memory tree reduction when `subgroups` feature unavailable.

### Tier 1: Remaining subgroup builtins (available now, Chrome 134+)

#### 1a. `subgroupInclusiveAdd` / `subgroupInclusiveMul` for associative scan

**What:** Replace the innermost Kogge-Stone rounds within a subgroup with a single hardware
instruction. For `subgroup_size = 32`, the first 5 rounds of Kogge-Stone (doubling 1→2→4→8→16→32)
are replaced by one `subgroupInclusiveAdd()` call.

**Where:** `associativeScan` WebGPU fused shader — the per-round Kogge-Stone loop in
`nativeScanMultiShaderSource` / block-map fused path.

**Impact:** For cumulative sum/product with N=1024, blockSize=256: each block has 8 subgroups of 32.
Currently 8 Kogge-Stone rounds per block (log₂ 256). With subgroup inclusive scan, rounds 0–4 are
free, leaving only 3 inter-subgroup rounds. **~40% fewer barrier-separated shader rounds.**

**Prerequisites:** Body function must be a simple associative op (`add` or `mul`) that maps directly
to the hardware builtin. For pytree bodies (DLM compose), the body is a general function — the
inclusive scan builtin doesn't help unless we can decompose it.

**Implementation sketch:**

1. In `runFusedPlan` Phase 1, detect if the body is pure `add` or `mul`
2. If so, emit `subgroupInclusiveAdd(val)` for the first log₂(subgroup_size) rounds
3. After the intra-subgroup prefix, the last invocation in each subgroup writes its result to shmem
4. Continue normal inter-subgroup Kogge-Stone for the remaining rounds
5. Requires `enable subgroups;` already present

#### 1b. `subgroupShuffle` / `subgroupShuffleUp` for associative scan (general bodies)

**What:** For general associative bodies (not just add/mul), replace `var<workgroup>` reads within a
Kogge-Stone round with register-to-register shuffles. Each thread gets its neighbor's value via
`subgroupShuffleUp(val, offset)` instead of writing to shmem → barrier → reading from shmem.

**Where:** Same as 1a, but applicable to ALL associative scan bodies including DLM compose.

**Impact:** Eliminates shmem traffic for the first log₂(subgroup_size) rounds. For DLM 2-tuple N=100
(1 dispatch, 1 block), this removes 5 of 8 shmem barrier pairs. The shmem barrier cost is small
relative to the actual compute, but for small bodies this could yield **10–20% improvement**.

**Implementation sketch:**

1. For rounds where `offset < subgroup_size`, emit `subgroupShuffleUp` instead of shmem write+read
2. The body function operates on register-resident values — no shmem allocation needed for these
   rounds
3. After subgroup-local rounds complete, fall back to shmem path for inter-subgroup communication
4. `subgroupShuffleUp(val, offset)` requires `offset` to be dynamically uniform or compile-time
   constant — in Kogge-Stone, the offset per round is a constant power of 2, so this works

#### 1c. `subgroupBroadcast` for scan carry / block-map constants

**What:** Broadcast a value from one invocation to all others in the subgroup without shmem.

**Where:** Block-map Phase 4 (apply-prefix) where the scanned carry is broadcast to all threads in a
workgroup. Also useful for broadcasting uniform values like block indices.

**Impact:** Minor — the broadcast step is a small fraction of total compute. Clean architectural
improvement.

### Tier 2: Cooperative Matrix / WMMA (not yet available in Chrome)

#### What is cooperative matrix?

Cooperative matrix (the WebGPU equivalent of NVIDIA's WMMA / Tensor Core, Intel's XMX / AMX)
provides hardware-accelerated small matrix multiply-accumulate operations. A subgroup collectively
owns matrix fragments; a single instruction like `cooperativeMatrixMultiplyAdd(A, B, C)` performs
`C += A × B` for hardware-native tile sizes (e.g., 16×16×16 f16 or 8×8×4 f32 on NVIDIA).

#### Spec status

| Layer  | Status (as of mid-2025)                                                  |
| ------ | ------------------------------------------------------------------------ |
| Vulkan | `VK_KHR_cooperative_matrix` — ratified KHR extension, widely supported   |
| SPIR-V | `SPV_KHR_cooperative_matrix` — stable                                    |
| Dawn   | Experimental `chromium-experimental-cooperative-matrix` behind flags     |
| WGSL   | Proposal in gpuweb/gpuweb repo, not yet in standard grammar              |
| Chrome | **Not shipped.** No origin trial announced. Requires WGSL spec stability |

**Realistic timeline:** Given the ~18-month subgroups arc (Chrome 125 proposal → Chrome 134 stable),
cooperative matrix is likely **2026** at the earliest for Chrome stable.

#### Impact on jax-js tiled matmul

Our current tiled matmul (via `block_map`) achieves **53.7% of peak FP32** at 4096×4096 on RTX 4070
Ti SUPER. The bottleneck is the software dot-product K-tile loop — each thread computes `threadTile`
(4×4 or 8×8) output elements using scalar `f32` multiply-accumulate.

With cooperative matrix:

- The K-tile inner loop would call `cooperativeMatrixMultiplyAdd(A_frag, B_frag, C_frag)` once per
  K-tile instead of a manual ridx unroll
- Hardware tensor cores deliver 2–4× the FLOP/s of scalar FP32
- Expected: **70–85% of peak FP32** (limited by shmem load bandwidth, not ALU)
- For FP16 (`shader-f16`): tensor cores are 4–8× faster than scalar f16, reaching near-peak

#### Impact on DLM

Small-matrix DLM (2×2, 4×4) would not benefit from cooperative matrix — the matrix sizes are below
the hardware tile size (typically 8×8 minimum). The cooperative matrix overhead would exceed the
scalar computation cost. **DLM improvement requires latency reduction (fewer dispatches), not
throughput increase.**

#### Implementation strategy (when WGSL spec stabilizes)

1. **Feature detection:** Add `"cooperative-matrix"` to `BackendCapabilities`, request feature from
   adapter. Query supported matrix sizes via adapter limits.
2. **block-map codegen:** In Phase 4 WGSL generation, detect tiled matmul pattern (two shmem inputs,
   one output, ridx accumulation). Replace the manual dot-product loop with cooperative matrix load
   → MMA → store.
3. **Tile size adaptation:** Hardware dictates tile shapes (e.g., 16×16×16 for f16, 16×8×8 for f32
   on NVIDIA). The `threadTile` parameter must map to these hardware native shapes. May require a
   tuner or size-specific code paths.
4. **f16 fast path:** Cooperative matrix f16 is ~4× faster than f32 on NVIDIA. For models that
   tolerate f16 precision (most ML inference), emit f16 cooperative matrix by default when
   `shader-f16` is available.
5. **Fallback:** Keep current scalar tiled matmul as fallback when cooperative matrix is
   unavailable.

### Tier 3: Timestamp queries for profiling

**What:** `timestamp-query` feature allows GPU-side timing of compute passes. Already in WebGPU spec
(Chrome 121+), not yet wired up in jax-js.

**Where:** `pipelineSubmit` / `commandEncoder.beginComputePass` — record timestamps before/after
each dispatch.

**Impact:** Enables accurate per-kernel GPU timing without CPU round-trip overhead. Critical for
validating that subgroup/WMMA optimizations actually improve wall-clock GPU time (vs just reducing
dispatch count). Would replace the indirect "dispatch count" proxy we currently use for WebGPU
performance analysis.

**Implementation:** Add `profiling: boolean` option to `JitProgram.execute()`. When enabled, insert
timestamp queries around each dispatch, readback the query buffer after execution, and return
per-step timing data.

### Priority ordering

| ID  | Feature                           | Availability | Impact        | Effort |
| --- | --------------------------------- | ------------ | ------------- | ------ |
| 1b  | `subgroupShuffleUp` in assocScan  | Now          | Medium        | Medium |
| 1a  | `subgroupInclusive*` in assocScan | Now          | Medium        | Medium |
| T3  | Timestamp queries                 | Now          | Diagnostic    | Low    |
| 1c  | `subgroupBroadcast` cleanup       | Now          | Low           | Low    |
| 2   | Cooperative matrix tiled matmul   | ~2026        | **Very High** | High   |

**Next action:** Implement Tier 1b (`subgroupShuffleUp` in associative scan) and T3 (timestamp
queries). These are available today and unblock accurate performance measurement for future work.

---

## Non-goals

- **General einsum rank-adaptation:** The general `parseEinsumExpression` still fails with
  rank-reduced inputs (e.g., 3-index subscript with 2D tensor). This only matters for exotic
  subscript patterns NOT in `einsumFastPath`. All common batch-matmul patterns are already handled.
  Fixing the general parser would be a separate, lower-priority effort.
- **Multiple bind groups for overflow:** WebGPU supports up to 4 bind groups, but storage bindings
  per shader stage is a global limit — splitting across groups doesn't help.
- **GPU/WASM ratio ≤ 2× target:** Not achievable for small-matrix DLM (m=2) at small N without
  fundamentally reducing dispatch count. The bottleneck is pure JS-side overhead in the JIT
  execution loop (95ms of 124ms total). WASM's mega-module avoids this entirely by compiling all
  steps into a single native function. WebGPU has no equivalent mechanism — each dispatch requires
  JS-side command encoding. For large-matrix workloads (matmul 4096×4096), WebGPU already achieves
  53.7% peak.
- **Binding limit optimization on high-limit hardware:** On Deno/NVIDIA with `maxArgs = 1,048,575`,
  the P2 pass has no effect. All dispatch fragmentation comes from P1 structural rules (reduction
  boundaries, diamond heuristic). Browser deployments with Chrome's `maxArgs ≈ 9` will see
  additional P2-caused fragmentation — but optimizing for Chrome's low limit would require a
  different approach (e.g., leaf packing) that doesn't address the P1 structural issue.
