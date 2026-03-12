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

The `lax.scan` wrapper (`src/backend/webgpu/scan-wrapper.ts`) already uses
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
   - The `hasUniform` flag is already true when the `lax.scan` wrapper sets offsets — ensure the
     uniform bind group layout accommodates both offset structs and constant structs

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
- Mixed-dtype test: f32 + i32 leaves in a single `associativeScan` (separate packing per dtype)
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

#### Forward `associativeScan` (5-tuple)

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

**Where Strategy A helps (standalone JitPrograms, no `lax.scan`/`associativeScan`):**

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
| Non-scan/assocScan chains      | Moderate                  | Significant               |

**Recommendation:** Deprioritize O6. The DLM use case — the primary motivation — is already served
by the block-map fused shader. Effort is better spent on:

- Subgroup matrix ops (WMMA) for hardware tensor cores in tiled matmul (P7)
- Conv2d tuning (P4)
- Relaxed SIMD FMA for WASM matmul (P2)

---

## Analytical Small-Matrix Linalg (Jaxpr-Expressible Routines)

### Background

The Routine system (`Sort`, `Cholesky`, `TriangularSolve`, `LU`, `QR`) provides backend-specific
implementations for ops that "can't be fused." On WebGPU, any Routine in a block-map body forces
fallback to per-round multi-dispatch execution — a single Routine breaks fusion for the entire body.

`np.linalg.inv` already has **analytical (non-Routine) fast paths for n ≤ 4**: Cramer's rule
(adjugate / determinant) using only elementwise ops (`mul`, `sub`, `neg`, `reciprocal`, `stack`).
These trace to fusable Kernel steps, enabling the DLM 5-tuple compose body (which calls `inv` on m×m
matrices) to fuse into the block-map shader for m ≤ 4.

The question: can the remaining linalg Routines (Cholesky, TriangularSolve, QR, LU) also be
expressed as jaxpr-traceable ops for small matrices, enabling fusion?

### Feasibility analysis

| Routine              | Algorithm for small n                  | Ops used                                | Complexity (n=4) | Feasible?                                                         |
| -------------------- | -------------------------------------- | --------------------------------------- | ---------------- | ----------------------------------------------------------------- |
| **inv** (n ≤ 4)      | Cramer's rule (adjugate/det)           | mul, sub, neg, reciprocal, stack        | ~120 mul+add     | **Done** ✅                                                       |
| **cholesky** (n ≤ 4) | Cholesky-Banachiewicz, unrolled        | mul, sub, div, sqrt                     | ~30 ops          | **Done** ✅                                                       |
| **trisolve** (n ≤ 4) | Back/forward substitution, unrolled    | mul, sub, div                           | ~20 ops          | **Done** ✅ (limit=8)                                             |
| **LU** (n ≤ 4)       | Gaussian elimination with pivoting     | mul, sub, div, argmax, swap             | ~40 ops          | **Tricky** — pivoting requires data-dependent control flow        |
| **QR** (n ≤ 4)       | Householder reflections, unrolled      | mul, sub, div, sqrt, dot, outer product | ~60 ops          | **Done** ✅ (limit=8)                                             |
| **inv** (n = 5)      | Cramer's rule (5×5 cofactor expansion) | mul, sub, neg, reciprocal, stack        | ~720 terms       | **Marginal** — large trace graph, may exceed JIT cache efficiency |

**LU is the hardest** because partial pivoting (`argmax` + row swap) introduces data-dependent
control flow that doesn't trace well. However, LU is not used directly in DLM compose bodies — it's
only used internally by `np.linalg.inv` for n ≥ 5 and by `np.linalg.det`/`slogdet`. The DLM sqrt
variant uses QR + TriangularSolve, not LU.

### Impact on DLM variants

| DLM variant            | Linalg ops in compose body | Current state                      | With analytical paths     |
| ---------------------- | -------------------------- | ---------------------------------- | ------------------------- |
| Standard (5-tuple)     | `np.linalg.inv`            | **Fuses for m ≤ 4** (already done) | No change needed          |
| Sqrt forward (3-tuple) | QR + trisolve              | **Fuses for m ≤ 8** (done)         | All paths fusable for m≤4 |
| Sqrt backward          | QR                         | **Fuses for m ≤ 8** (done)         | All paths fusable for m≤4 |

The sqrt DLM variant (`composeSqrtForward`, `composeSqrtBackward`) is numerically more stable than
the standard variant for poorly conditioned systems. Enabling it to fuse on WebGPU for small m
widens the set of models that can run efficiently on GPU.

### Implementation approach

Add threshold-based dispatch to the library functions: when the matrix dimension n ≤ threshold (4),
trace an unrolled analytical path instead of calling the Routine. This mirrors the existing pattern
in `np.linalg.inv`:

```typescript
// numpy-linalg.ts — existing pattern:
export function inv(a) {
  const n = checkSquare("inv", a);
  if (n === 1) return np.reciprocal(a);
  if (n === 2) return inv2x2(a);
  if (n === 3) return inv3x3(a);
  if (n === 4) return inv4x4(a);
  // n ≥ 5: falls through to LU Routine
  return solve(a, np.eye(n, { dtype: a.dtype }));
}
```

Apply the same pattern to:

1. **`lax.linalg.cholesky`** — n ≤ 4: unrolled Cholesky-Banachiewicz using `mul`, `sub`, `div`,
   `sqrt`. Lower-triangular output via `stack`. The algorithm is O(n³/6) — for n=4 that's ~11
   multiply-accumulate steps.
2. **`lax.linalg.triangularSolve`** — n ≤ 4: unrolled back/forward substitution. For a single RHS
   vector: n(n+1)/2 multiply-accumulate steps. For batched matrix RHS: iterate over columns.
3. **`lax.linalg.qr`** — n ≤ 4: unrolled Householder reflections. Each reflection is a rank-1
   update: compute householder vector v, then `H = I - 2vvᵀ/‖v‖²`, apply `H @ A_remaining`. For n=4:
   4 reflections, each involving a norm computation + outer product + matrix subtraction.

**Autodiff:** All analytical paths use standard ops (mul, sub, div, sqrt, stack) that already have
JVP and transpose rules. No custom JVP/transpose rules are needed — AD "just works" through the
traced ops. This is a key advantage over the Routine path, which requires hand-written JVP rules.

### Considerations

- **Trace graph size:** The 4×4 inv generates ~300 traced ops. Cholesky 4×4 would be ~50. QR 4×4
  would be ~200. These are modest — the JIT handles them easily. But for n=5+ the graph grows fast
  (inv 5×5 ≈ 720 terms).
- **Threshold selection:** n ≤ 4 is the sweet spot balancing trace size vs fusion benefit. The DLM
  use case has m=2–5; m ≤ 4 covers most practical models. For m=5, we could evaluate whether the
  inv5x5 trace graph (if implemented) is worth the compilation cost.
- **Testing:** Each analytical path must be tested against the Routine path for numerical agreement
  across all n ≤ threshold. The existing inv analytical paths have tests in `numpy-linalg.test.ts`.
- **Batched support:** The analytical paths must handle batched inputs (`[..., n, n]`) via the same
  `idx2d` / `stack` pattern used by `inv2x2`/`inv3x3`/`inv4x4`.

### Priority

**Medium-High.** Upgraded from Medium based on the Dispatch Acceleration Roadmap analysis. The
standard DLM 5-tuple variant already fuses for m ≤ 4 (inv analytical path + Phase 1 uniform
constants). Adding Cholesky + TriangularSolve analytical paths unlocks the sqrt DLM variant, which
is numerically superior for poorly conditioned systems. QR completes the backward pass path.

Implement alongside O8/O9 (independent, no dependency):

- Cholesky + TriSolve: enables sqrt DLM forward fusion for m ≤ 4
- QR: enables sqrt DLM backward fusion for m ≤ 4
- Combined: the entire sqrt DLM pipeline fuses for m ≤ 4 on WebGPU

See "Routine Jaxprification Analysis" below for the broader assessment of which Routines should and
should not be converted.

### Comparison: "eliminate Routines" vs "improve scan algorithm"

| Optimization                | Target                                   | Impact on DLM (m=2)                                                   |
| --------------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| Analytical linalg (this)    | Routine-blocked fusion in assocScan body | Already solved for standard variant (inv ≤ 4)                         |
| Decoupled Fallback (P10)    | Scan dispatch count (O(log N) → O(1))    | Saves ~0.05ms (3 dispatches → 1)                                      |
| JIT loop overhead reduction | 764 non-scan dispatches in DLM pipeline  | **95ms** of the 124ms warm total                                      |
| Mega-module for WebGPU      | Eliminate JS-side dispatch overhead      | Command tape (O8): ~4× JS overhead reduction for kernel-only programs |

The dominant DLM bottleneck is **JS-side JIT loop overhead for 764 non-scan dispatches** (element
construction + diagnostics), not the scan algorithm or scan body fusion. The scans themselves
already fuse into 1–3 dispatches for m ≤ 4.

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

### Tier 0: Decoupled Fallback — single-dispatch prefix scan (available now)

Our current associative scan uses **Kogge-Stone doubling**: O(N log N) work, O(log N) depth,
ceil(log₂ N) barrier-separated rounds. On WebGPU, this manifests as:

- **Within a workgroup** (N ≤ blockSize=256): all rounds in one dispatch via `workgroupBarrier()`
- **Across workgroups** (N > 256): 3-level hierarchical recursion = 3 dispatches

Kogge-Stone was chosen because WebGPU lacks a **cross-workgroup barrier** within a single dispatch.
However, the state-of-the-art has moved beyond this constraint.

#### The Decoupled Fallback algorithm

**Decoupled Lookback** (Merrill & Garland, 2016) achieves O(N) work in a single dispatch by having
each workgroup atomically publish its local reduction, then "look back" through prior workgroups'
published values to compute its prefix. However, standard Decoupled Lookback uses unbounded
spin-wait loops — if a prior workgroup hasn't been scheduled yet, the waiting workgroup spins
forever. This works on NVIDIA (which guarantees **Forward Progress** — all scheduled workgroups
eventually complete), but **deadlocks on other hardware** (Apple Silicon via Metal, some Intel
iGPUs) where the scheduler can't guarantee forward progress.

**Decoupled Fallback** (Smith, Levien, & Owens) solves this portably:

1. **Bounded spin:** Each lookback subgroup polls prior blocks for a fixed number of cycles
2. **Work-stealing fallback:** If the timeout fires, assume the prior block is stalled in the
   scheduler. Fetch the raw input data and compute the missing reduction locally
3. **Atomic CAS consistency:** Use `atomicCompareExchangeWeak` to publish the computed value,
   ensuring correctness even if multiple blocks fall back simultaneously

This achieves **single-dispatch, memory-bandwidth-saturating performance** on all WebGPU backends.

#### Key WebGPU-specific requirements

| Requirement                        | Why                                                                                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Single-field atomic descriptor** | WebGPU lacks global acquire-release fences. Pack status + value into one `atomic<u32>` with bit-masking to prevent compiler reordering                    |
| **No unbounded spin-wait**         | WebGPU has no Forward Progress Guarantee (FPG). Unbounded `while` loops deadlock on Metal/some Intel                                                      |
| **Subgroup operations**            | Use `requires subgroup_id; requires subgroup_uniformity;` for SIMD raking within workgroups. Native subgroups (Chrome 134+) make this safe and performant |
| **Atomic f32 or bit-packing**      | For f32 prefix sums, either use `atomicAdd` (if `shader-f32-atomic-add` available) or pack f32 into u32 bits within the descriptor                        |

#### Complexity comparison

| Algorithm                 | Work       | Dispatches      | Global sync                   | FPG required?         |
| ------------------------- | ---------- | --------------- | ----------------------------- | --------------------- |
| **Kogge-Stone** (current) | O(N log N) | O(log N) or 1–3 | None (multi-dispatch)         | No                    |
| Decoupled Lookback        | O(N)       | **1**           | Atomic lookback               | **Yes** (NVIDIA only) |
| **Decoupled Fallback**    | O(N)       | **1**           | Bounded atomic + CAS fallback | **No** ✅             |

#### Impact on jax-js

**For simple associative ops** (cumsum, cumprod, cummax — scalar `add`/`mul`/`max`):

- Replaces Kogge-Stone entirely for the WebGPU path
- Single dispatch regardless of N
- O(N) work instead of O(N log N) — significant for large N
- Memory-bandwidth saturating performance (vs current compute-bound Kogge-Stone)

**For general associative bodies** (DLM pytree compose):

- The lookback phase requires the body function to be applied during work-stealing. This is feasible
  but complex: the fallback subgroup must be able to invoke the user's body `fn` on raw input tiles
  from a prior block
- The body must also produce a reduction value that can be packed into an atomic descriptor. For
  pytree bodies with multiple leaves, this requires a packed representation or a multi-field
  descriptor scheme
- **Initial implementation should target scalar ops only.** General bodies can follow via a
  "reduction descriptor buffer" approach (one atomic per block, separate from data)

#### Implementation strategy

1. **Reference:** Start from Thomas Smith's `GPUPrefixSums` WGSL implementation (MIT license). Adapt
   the Decoupled Fallback shader structure to our codegen pipeline.
2. **Phase 1 — scalar ops: ✅ DONE.** Implemented for `add`, `mul`, `min`, `max` on f32.
   Single-dispatch kernel with workgroup-local Hillis-Steele scan → atomic publish → bounded
   lookback with work-stealing fallback → apply prefix. Detected in `buildNativeAssocScanPlan` via
   `detectDecoupledFallbackOp`. Descriptor packing: 2-bit flag + 30-bit value in single
   `atomic<u32>`. u32 excluded (30-bit packing silently truncates values > 2^30-1). i32 excluded
   (same 30-bit truncation issue as u32 — signed range ±2^29 insufficient for large cumulative
   sums). Both u32 and i32 fall back to Kogge-Stone block-map path (fixed in v0.8.4).
3. **Phase 2 — general bodies:** Extend to arbitrary associative functions using a per-block
   descriptor buffer. The lookback subgroup fetches raw input tiles and re-evaluates the body
   function when fallback triggers.
4. **Integration:** `"decoupled-fallback"` added to `AssocScanPlan`. Kogge-Stone block-map path
   remains for bodies that can't use DF (pytree, non-scalar, axis > 0) or for backends without
   atomics.

**Phase 1 remaining opportunities:**

- Subgroup-parallel lookback (first subgroup instead of thread 0 only)
- Raking pattern: multiple elements per thread for better bandwidth utilization

#### References

- Paper: Smith, Levien, & Owens — "Decoupled Fallback" (2024)
  https://github.com/b0nes164/Decoupled-Fallback-Paper
- Implementation: GPUPrefixSums — production WGSL implementation
  https://github.com/b0nes164/GPUPrefixSums

### Tier 1: Remaining subgroup builtins (available now, Chrome 134+)

#### 1a. `subgroupInclusiveAdd` / `subgroupInclusiveMul` for associative scan — **Done** ✅

**What:** Replace the innermost Kogge-Stone rounds within a subgroup with a single hardware
instruction. For `subgroup_size = 32`, the first 5 rounds of Kogge-Stone (doubling 1→2→4→8→16→32)
are replaced by one `subgroupInclusiveAdd()` call.

**Where:** `associativeScan` WebGPU fused shader — the per-round Kogge-Stone loop in the block-map
fused path.

**Impact:** For cumulative sum/product with N=1024, blockSize=256: each block has 8 subgroups of 32.
With subgroup inclusive scan, rounds 0–4 are free, leaving only 3 inter-subgroup shmem rounds.
**~40% fewer barrier-separated shader rounds.**

**Prerequisites:** Body function must be a simple associative op (`add` or `mul`) that maps directly
to the hardware builtin. For pytree bodies (DLM compose), the body is a general function — the
inclusive scan builtin doesn't help unless we can decompose the body into scalar associative ops.

**Implementation (completed):**

1. `WorkgroupAssocScanInfo.scalarOp` detects scalar `Add` or `Mul` bodies at analysis time (single
   kernel, single elem, elemCount=1, no reduction, no constants)
2. `emitInclusiveScanSection()` emits `subgroupInclusiveAdd(val)` / `subgroupInclusiveMul(val)` for
   the first 5 rounds (assuming sg_size ≥ 32)
3. 3-way runtime branch: `sg_size >= 32` → inclusive scan, `sg_size >= 8` → shuffle fallback, else
   pure shmem. All branches keep ping/pong parity deterministic.
4. After the intra-subgroup prefix, the result is written to the correct shmem buffer and normal
   inter-subgroup Kogge-Stone continues for the remaining rounds
5. Applies to all numeric dtypes (f32, f16, i32, u32) — broader than DF which is f32 only

#### 1b. `subgroupShuffle` / `subgroupShuffleUp` for associative scan (general bodies) — **Done** ✅

**What:** For general associative bodies (not just add/mul), replace `var<workgroup>` reads within a
Kogge-Stone round with register-to-register shuffles. Each thread gets its neighbor's value via
`subgroupShuffleUp(val, offset)` instead of writing to shmem → barrier → reading from shmem.

**Where:** Same as 1a, but applicable to ALL associative scan bodies including DLM compose.

**Impact:** Eliminates shmem traffic for the first 3 Kogge-Stone rounds when sg_size ≥ 8. For DLM
2-tuple N=100 (1 dispatch, 1 block, 8 rounds), this removes 3 of 8 shmem barrier pairs. The shmem
barrier cost is small relative to the actual compute, but for small bodies this could yield **10–20%
improvement**.

**Measured impact (dlm-js Nile m=2 model, RTX 4070 Ti SUPER):**

| N         | v0.8.2 baseline (ms) | 59f03a6 + subgroups (ms) | Δ     |
| --------- | -------------------- | ------------------------ | ----- |
| 100       | 1,275                | 1,286                    | +0.9% |
| 102,400   | 2,462                | 2,464                    | +0.1% |
| 819,200   | 6,087                | 5,997                    | −1.5% |
| 1,638,400 | 9,886                | 9,918                    | +0.3% |

No measurable speedup. For the Nile m=2 model the compose body is so small that shmem barriers
aren't the bottleneck — dispatch overhead and compilation time dominate. Subgroups may show more
benefit on larger state dimensions (m ≥ 5) where the compose body is heavier.

**Implementation (completed):**

1. All-or-nothing runtime guard: `if (sg_size >= 8u)` wraps the entire subgroup path. Either all 3
   register rounds execute (and the remaining rounds use shmem starting from round 3), or the pure
   shmem path handles all rounds. This keeps ping/pong buffer parity deterministic in both branches.
2. Shared emitter (`emitWasRoundBody`): both subgroup and shmem paths call the same helper with
   different resolve/write callbacks, eliminating ~200 lines of codegen duplication.
3. Register variables (`var<private>`) hold leaf values during subgroup phase
4. Compose body called via `emitWasRoundBody` with register-based callbacks — "a" reads shuffled
   registers, "b" reads current registers
5. After subgroup rounds: flush registers to shmem + single `workgroupBarrier()`
6. Remaining rounds use existing shmem Kogge-Stone path (starting from round `sgRounds`)
7. Both reduction kernel (gidx loop + ridx accumulation) and elementwise kernel paths supported
8. `@builtin(subgroup_invocation_id) sg_inv_id: u32` added to entry point when active

#### 1c. `subgroupBroadcast` for scan carry / block-map constants — Assessed, deferred

**What:** Broadcast a value from one invocation to all others in the subgroup without shmem.

**Where:** Block-map Phase 4 (apply-prefix) where the scanned carry is broadcast to all threads in a
workgroup. Also useful for broadcasting uniform values like block indices.

**Assessment:** No measurable benefit. All our broadcast patterns are workgroup-wide (workgroupSize
64–256 > subgroupSize 16–64), and `subgroupBroadcast` only works within a single subgroup. The main
candidates: (1) block-map point inputs — redundant global loads are coalesced by GPU L1 cache, (2)
routine pivot/diagonal broadcasts (`x_j`, `L_jj`, `pivot_val`) — workgroup-wide via
`var<workgroup>`, can't replace with subgroup-only broadcast, (3) decoupled fallback `shared_prefix`
— same. Adding subgroupBroadcast would increase codegen complexity for zero gain.

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

### Tier 3: Timestamp queries for profiling ✅

**What:** `timestamp-query` feature allows GPU-side timing of compute passes. Already in WebGPU spec
(Chrome 121+).

**Implementation:** Module-level profiling state (`_profilingQuerySet`, `_profilingPassIdx`) with
`_beginComputePass()` wrapper injects `timestampWrites` into all 9 compute pass creation sites.
`WebGPUBackend.startProfiling()` / `stopProfiling()` manage the query set lifecycle. Public API:
`profileGpu(fn)` returns `{ result, timing: GpuTimingResult }` with per-pass `durationMs` and
wall-clock `totalMs`. Zero overhead when not profiling (`_profilingTimestampWrites()` returns
`undefined`).

### Priority ordering

| ID  | Feature                           | Availability | Impact        | Effort               |
| --- | --------------------------------- | ------------ | ------------- | -------------------- |
| T0  | Decoupled Fallback prefix scan    | Now          | **Very High** | **Done** ✅          |
| T3  | Timestamp queries                 | Now          | Diagnostic    | **Done** ✅          |
| 1b  | `subgroupShuffleUp` in assocScan  | Now          | Medium        | **Done** ✅          |
| 1a  | `subgroupInclusive*` in assocScan | Now          | Medium        | **Done** ✅          |
| 1c  | `subgroupBroadcast` cleanup       | Now          | Low           | Assessed: no benefit |
| 2   | Cooperative matrix tiled matmul   | ~2026        | **Very High** | High                 |

**Priority rationale:** T0, 1a, 1b, T3 are complete. 1c assessed — all broadcast patterns are
workgroup-wide; `subgroupBroadcast` (subgroup-only) provides no measurable improvement over L1
cached reads. Cooperative matrix (2) is blocked on WGSL spec.

**Next action:** All P7 subgroup items are complete or assessed. Cooperative matrix (2) is blocked
on Chrome WGSL spec stability (~2026).

---

## O8: WebGPU Command Tape (Mega-Module Equivalent) ✅

**Status:** Implemented in commits `3ce29ae` (O8a core) and `caa3974` (tech debt fixes).

### Problem statement

The WASM mega-module compiles an entire `JitProgram`'s step list into a single WASM function,
eliminating all JS↔WASM boundary crossings. A DLM `jit(core)` call with 764 dispatches costs ~6ms
on WASM mega-module but **~124ms on WebGPU** because each dispatch incurs JS-side overhead.

**Measured WebGPU overhead breakdown (DLM Nile m=2, N=100, warm):**

| Component                   | Time (ms) | % of total |
| --------------------------- | --------- | ---------- |
| JIT execution loop overhead | ~95       | 76%        |
| `queue.submit()`            | ~22       | 18%        |
| `createBindGroup()`         | ~6        | 5%         |
| `prepareKernelSync()`       | ~1        | 1%         |
| **Total**                   | **~124**  | 100%       |

The 95ms loop overhead at 764 steps is **~124µs per step**, comprising:

- Scope array lookups (JitId → Slot)
- `new Array(N)` allocation for `ins[]` / `outs[]`
- `incRef` / `decRef` loops (Map lookups per input/output)
- `prepareKernelSync` (hash + Map.get)
- `dispatch()` buffer extraction (`inputs.map(slot => getBuffer(slot).buffer)`)
- Batch entry object creation (`{ source, inputs, outputs, dynamicParams }`)
- GC pressure from ~764 temporary objects per invocation

The WASM mega-module avoids all of this — JitIds are WASM locals, buffers are i32 pointers, no
refcounting, no temporary objects. WebGPU cannot compile GPU dispatches into native code (each
kernel is a separate shader), but it CAN eliminate the JS overhead by pre-compiling the dispatch
sequence into a tight command-encoding loop.

### Design: Command Tape

A **command tape** is a flattened, pre-resolved representation of a JitProgram's step list.
Pipelines are resolved once at compile time. Per-step buffer assignments are encoded as indices into
a flat `GPUBuffer[]` table, eliminating per-step scope lookup, array allocation, and refcounting.

#### Compilation phase (once per JitProgram)

```typescript
interface WebGPUCommandTape {
  // --- Pre-resolved at compile time ---

  /** One entry per dispatch (execute step). */
  dispatches: TapeDispatch[];

  /** Bulk malloc plan: [jitIdIdx, paddedSize, initialData?][] in step order. */
  mallocs: TapeMalloc[];

  /** Recycle plan: [fromIdx, toIdx][] — pointer copy in flat table. */
  recycles: [number, number][];

  /** Free plan: jitIdIdxs to free after all dispatches complete. */
  frees: number[];

  /** Number of entries in the flat buffer table (inputs + intermediates + outputs). */
  tableSize: number;

  /** Mapping: external input position → table index. */
  inputTableIdxs: number[];

  /** Mapping: external output position → table index. */
  outputTableIdxs: number[];

  /** Per-output byte size (for creating backend Slots). */
  outputSizes: number[];
}

interface TapeDispatch {
  /** Pre-compiled GPU pipeline. */
  pipeline: GPUComputePipeline;

  /** Bind group layout (from pipeline.getBindGroupLayout(0)). */
  bindGroupLayout: GPUBindGroupLayout;

  /** Indices into the flat buffer table for inputs. */
  inputIdxs: number[];

  /** Indices into the flat buffer table for outputs. */
  outputIdxs: number[];

  /** Pre-computed grid dimensions. */
  grid: [number, number];

  /** Pre-computed uniform bind group (null if no uniforms). Static uniforms only. */
  uniformBindGroup: GPUBindGroup | null;

  /** Dynamic uniform offset (for multi-pass shaders). */
  uniformOffset: number;
}

interface TapeMalloc {
  /** Index in the flat buffer table. */
  tableIdx: number;

  /** Buffer size in bytes (padded to 4-byte alignment). */
  paddedSize: number;

  /** Pre-filled constant data (for O2 scalar promotion). */
  initialData: Uint8Array | null;
}
```

**Compilation steps:**

1. Walk all steps, assign sequential table indices to each JitId
2. For each `execute` step:
   - Call `prepareKernelSync(kernel)` → resolve pipeline
   - Pre-extract `pipeline.getBindGroupLayout(0)` (avoids per-dispatch lookup)
   - Record input/output table indices (integers, not Slot refs)
   - Pre-compute grid dimensions from concrete kernel size
   - Pre-build static uniform bind groups (for non-symbolic uniforms)
3. For `malloc` steps: record size + initialData
4. For `recycle` steps: record (from, to) index pairs
5. For `free` steps: record indices (all frees deferred to post-dispatch)
6. Reject programs with step types that require complex coordination: `scan`, `dus`, `scatter_add`,
   `assoc_scan`, `block_map`, `workgroup_assoc_scan`, `reverse`, `incref`

The eligibility check mirrors `canCompileToMegaModule()` from the WASM path. `fori_loop` with
concrete bounds can be supported by unrolling into the dispatch sequence (each iteration produces N
dispatches). Routines ARE supported — they are also GPU dispatches with pre-resolved pipelines.

#### Execution phase (per invocation)

```typescript
executeTape(tape: WebGPUCommandTape, inputSlots: Slot[]): Slot[] {
  // 1. Build flat buffer table (one array, no per-step allocation)
  const table: GPUBuffer[] = new Array(tape.tableSize);

  // Map external inputs
  for (let i = 0; i < inputSlots.length; i++) {
    table[tape.inputTableIdxs[i]] = this.#getBuffer(inputSlots[i]).buffer;
  }

  // Bulk malloc all intermediates
  for (const m of tape.mallocs) {
    const buf = this.#poolPop(m.paddedSize) ?? this.#createBuffer(m.paddedSize);
    if (m.initialData) this.device.queue.writeBuffer(buf, 0, m.initialData);
    table[m.tableIdx] = buf;
  }

  // Apply recycling (just pointer copy)
  for (const [from, to] of tape.recycles) table[to] = table[from];

  // 2. Encode all dispatches in a single encoder
  const encoder = this.device.createCommandEncoder();
  for (const d of tape.dispatches) {
    const entries = new Array(d.inputIdxs.length + d.outputIdxs.length);
    for (let i = 0; i < d.inputIdxs.length; i++) {
      entries[i] = { binding: i, resource: { buffer: table[d.inputIdxs[i]] } };
    }
    for (let i = 0; i < d.outputIdxs.length; i++) {
      entries[d.inputIdxs.length + i] = {
        binding: d.inputIdxs.length + i,
        resource: { buffer: table[d.outputIdxs[i]] },
      };
    }

    const bindGroup = this.device.createBindGroup({
      layout: d.bindGroupLayout,
      entries,
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(d.pipeline);
    pass.setBindGroup(0, bindGroup);
    if (d.uniformBindGroup) pass.setBindGroup(1, d.uniformBindGroup, [d.uniformOffset]);
    pass.dispatchWorkgroups(d.grid[0], d.grid[1]);
    pass.end();
  }
  this.device.queue.submit([encoder.finish()]);

  // 3. Return to pool / free
  for (const idx of tape.frees) {
    const buf = table[idx];
    if (!this.#poolPush(buf)) {
      this.#gpuAllocatedBytes -= buf.size;
      buf.destroy();
    }
  }

  // 4. Create output slots
  const outputs: Slot[] = new Array(tape.outputTableIdxs.length);
  for (let i = 0; i < tape.outputTableIdxs.length; i++) {
    const slot = this.nextSlot++;
    this.buffers.set(slot, {
      buffer: table[tape.outputTableIdxs[i]],
      size: tape.outputSizes[i],
      ref: 1,
    });
    outputs[i] = slot;
  }
  return outputs;
}
```

### Overhead elimination analysis

| Source                         | Step-by-Step (current) | Command Tape              | Saved?      |
| ------------------------------ | ---------------------- | ------------------------- | ----------- |
| Scope lookup per step          | ~0.1µs × N             | 0 (flat table)            | ✅          |
| `ins[]`/`outs[]` array alloc   | ~0.5µs × N             | 0 (pre-computed indices)  | ✅          |
| `incRef`/`decRef` per step     | ~0.5µs × N             | 0 (tape-managed lifetime) | ✅          |
| `prepareKernelSync` per step   | ~1µs × N               | 0 (pre-resolved)          | ✅          |
| `dispatch()` buffer extraction | ~0.2µs × N             | 0 (direct table index)    | ✅          |
| Batch entry object creation    | ~0.3µs × N             | 0 (no objects)            | ✅          |
| GC pressure (temp arrays)      | ~100ms for 764 steps   | ~0 (one array, reused)    | ✅          |
| `createBindGroup` per dispatch | ~2µs × N               | ~2µs × N                  | Unavoidable |
| Compute pass encoding          | ~0.5µs × N             | ~0.5µs × N                | Unavoidable |
| `queue.submit`                 | ~22ms × 1              | ~22ms × 1                 | Unavoidable |

**Conservative estimate for DLM (764 dispatches):**

- Current: ~124ms (95ms JS loop + 22ms submit + 6ms bind groups + 1ms prepare)
- Command tape: ~30ms (22ms submit + 6ms bind groups + 2ms tape loop)
- **Speedup: ~4×**

**Optimistic estimate (bind group overhead measured lower in practice):**

- Current: ~124ms
- Command tape: ~25ms
- **Speedup: ~5×**

The 22ms `queue.submit` is the GPU-side scheduling cost for encoding 764 dispatches into a single
command buffer. This is a hardware/driver cost that cannot be reduced without reducing the number of
dispatches.

### Bind group caching (O8b, aggressive optimization)

Bind groups reference GPUBuffers by identity. If the buffer pool returns the **same GPUBuffer
objects** across invocations (LIFO pool for same-sized mallocs in same order), then bind groups from
the previous execution are still valid and can be reused.

**Approach:** Cache bind groups keyed by `(dispatch_index, buffer_identity_tuple)`:

```typescript
// Per-dispatch bind group cache
interface CachedBindGroup {
  fingerprint: number;  // hash of GPUBuffer object identities
  bindGroup: GPUBindGroup;
}

// During execution:
for (const d of tape.dispatches) {
  const fp = hashBufferIds(d.inputIdxs, d.outputIdxs, table);
  if (d.cachedBG?.fingerprint === fp) {
    // Reuse!
    pass.setBindGroup(0, d.cachedBG.bindGroup);
  } else {
    const bg = device.createBindGroup({ ... });
    d.cachedBG = { fingerprint: fp, bindGroup: bg };
    pass.setBindGroup(0, bg);
  }
}
```

**Hit rate:** For a program that runs repeatedly with a steady-state buffer pool:

- If all mallocs hit the pool and pool is LIFO → ~100% hit rate
- If some mallocs create fresh buffers → proportionally lower hit rate
- External inputs change between invocations → dispatches reading inputs miss

For DLM where the `lax.scan` loop runs N iterations with the same intermediate sizes, bind group
caching would eliminate the ~6ms `createBindGroup` cost on all iterations after the first.

**Risk:** The `GPUBuffer` identity check is not guaranteed to be stable. Pool eviction,
`configurePool`, or GC of pooled buffers would invalidate cached bind groups. The cache must be
invalidated when the pool is reconfigured.

**Priority:** O8b is secondary. O8a (basic command tape) delivers the majority of the benefit.

### Supported step types

| Step type              | Supported? | Notes                                                    |
| ---------------------- | ---------- | -------------------------------------------------------- |
| `execute` (Kernel)     | ✅         | Pre-resolved pipeline, pre-computed grid                 |
| `execute` (Routine)    | ✅         | Same as Kernel — routines are also GPU dispatches        |
| `malloc`               | ✅         | Bulk alloc into flat table                               |
| `free`                 | ✅         | Deferred to post-submit                                  |
| `recycle`              | ✅         | Table index copy (zero-cost)                             |
| `fori_loop` (concrete) | ✅         | Unroll iterations into dispatch sequence                 |
| `incref`               | ❌         | Requires refcount tracking during encode                 |
| `scan`                 | ❌         | Requires nested execution with complex coordination      |
| `dus`                  | ⚠️         | Could support via pre-encoded `copyBufferToBuffer` calls |
| `scatter_add`          | ⚠️         | Could support via pre-resolved scatter dispatch          |
| `assoc_scan`           | ❌         | Multi-phase dispatch with dynamic buffer management      |
| `block_map`            | ❌         | Complex shader generation and buffer management          |
| `workgroup_assoc_scan` | ❌         | Nested inside block_map                                  |
| `reverse`              | ⚠️         | Could support via pre-encoded copy shader                |

**Programs with unsupported steps fall back to current step-by-step execution.** This is the same
strategy as the WASM mega-module.

### When the command tape helps

| Workload                                         | Dispatches | Current (ms) | Tape (ms) | Speedup                  |
| ------------------------------------------------ | ---------- | ------------ | --------- | ------------------------ |
| DLM `jit(core)` (764 kernel dispatches)          | 764        | ~124         | ~30       | ~4×                      |
| Standalone `jit(() => chain_of_20_ops)`          | 20         | ~0.2         | ~0.05     | ~4×                      |
| TTS inference pipeline (100+ ops, no `lax.scan`) | ~100       | ~1.5         | ~0.4      | ~3.5×                    |
| `jit(f)` with `lax.scan` (scan steps block tape) | N/A        | N/A          | N/A       | Fallback                 |
| `associativeScan` (fused into 1–3 dispatches)    | 1–3        | ~3           | ~3        | 1× (overhead negligible) |

The command tape targets **kernel-only JitPrograms with many steps** — the same class the WASM
mega-module handles. Programs dominated by `lax.scan` or `block_map` dispatches are already
optimized by their respective fused shaders.

### Integration with JitProgram.execute()

Mirror the WASM mega-module integration point:

```typescript
// In JitProgram.execute():
if (this.backend.type === "webgpu") {
  if (this._commandTape === undefined) {
    this._commandTape = canCompileToCommandTape(this.steps)
      ? compileCommandTape(this.backend, this.steps, this.inputs, this.outputs)
      : null;
  }
  if (this._commandTape) {
    const outputSlots = this.backend.executeTape(this._commandTape, inputs);
    return { outputs: outputSlots, pending: [] };
  }
}
// Fall through to step-by-step execution
```

The tape is compiled on first invocation and cached on the `JitProgram` instance, matching the
`_megaModule` lazy compilation pattern.

### Implementation phases

| Phase | Scope                                           | Effort | Impact                                  |
| ----- | ----------------------------------------------- | ------ | --------------------------------------- |
| O8a   | Basic command tape (kernel+malloc+free+recycle) | Medium | ~4× for kernel-only programs            |
| O8b   | Bind group caching                              | Small  | Additional ~20% on repeated invocations |
| O8c   | DUS + scatter_add + reverse support             | Small  | Expands tape eligibility                |
| O8d   | fori_loop unrolling into tape                   | Medium | Handles loop-containing programs        |

### Comparison with alternatives

| Approach                       | Feasibility       | Impact vs current                                          |
| ------------------------------ | ----------------- | ---------------------------------------------------------- |
| **Command tape (this)**        | **Available now** | **~4× JS overhead reduction**                              |
| Arena allocator (O9)           | Available now     | Enables stable bind groups → O8b reliable. See O9 below    |
| WebGPU indirect dispatch       | Available         | Doesn't help (1 dispatch at a time, not a sequence)        |
| WebGPU compute bundles         | Not in spec       | Would be ideal; tape is the prerequisite (see below)       |
| Command buffer reuse           | Not in spec       | GPUCommandBuffer is consumed on submit                     |
| Uber-shader (all kernels in 1) | Not viable        | Different bindings, grid sizes, workgroup sizes per kernel |
| Move encoding to Worker        | Available         | Hides latency but doesn't reduce work                      |

**Compute bundles as endgame.** `GPURenderBundle` exists for pre-recorded render command sequences.
A hypothetical `GPUComputeBundle` would allow pre-recording the entire dispatch tape as a native GPU
command sequence, replayed in a single call with zero JS overhead per dispatch. The tape's data
structures (pre-resolved pipelines, pre-computed bind groups, pre-computed grids) map 1:1 to what a
compute bundle API would need. If/when the W3C introduces compute bundles, O8 becomes the natural
compilation target — no architectural changes required, only a new execution backend for the same
tape IR.

**Indirect dispatch opportunities.** `dispatchWorkgroupsIndirect(buffer, offset)` lets the GPU
determine dispatch sizes. This is useful for data-dependent grid sizes (e.g., dynamic sparsity,
variable-length sequences), but doesn't reduce the number of dispatches in a fixed-topology
JitProgram. It could become valuable for future dynamic control flow (while-loop conditions, early
exit) without round-tripping to JS. Not a priority for current DLM/ML patterns where all sizes are
statically known.

### Risks

| Risk                                             | Likelihood | Mitigation                                                  |
| ------------------------------------------------ | ---------- | ----------------------------------------------------------- |
| GC of cached pipelines invalidates tape          | Low        | Tape holds strong refs to pipelines                         |
| Buffer pool behavior changes tape correctness    | None       | Tape allocates fresh each time; pool is optimization only   |
| Programs with `incref` steps are ineligible      | Medium     | `jitCompile` avoids `incref` when possible; fallback exists |
| Initial compilation cost (pipeline resolution)   | Low        | Pipelines are already cached; tape compilation is once      |
| Large number of dispatches exceeds Chrome limits | Low        | Chrome handles 1000+ dispatches per submit in practice      |

---

## O9: WebGPU Bind Group Caching & Arena Allocation

**Status:** O9a (single-slab arena) reverted — WebGPU spec prohibits a GPUBuffer from appearing in
both `read-only-storage` and `storage` bindings within a single compute pass (buffer-identity-level
validation). O9b (bind group caching via pool LIFO identity) done. O9c (constants slab) done.

### Problem statement

The command tape (O8) eliminates most JS-side overhead, but `device.createBindGroup()` remains at
~6ms for 764 dispatches (~8µs per call). This cost is irreducible under the current discrete buffer
pool because every JIT invocation allocates different `GPUBuffer` objects, forcing fresh bind group
creation every time. O9b bind group caching is fragile because pool LIFO order isn't guaranteed
across invocations.

The root cause: **discrete per-allocation buffers** mean bind groups (which reference buffer
identities) can never be stable.

### WebGPU aliasing constraint

The WebGPU spec validates at **buffer identity** level, not byte-range:

- `read-only-storage` + `read-only-storage` (same buffer) → **OK**
- `read-only-storage` + `storage` (same buffer) → **REJECTED** (validation error)
- `storage` + `storage` (same buffer, same or different offsets) → **REJECTED**

This means a single slab buffer cannot hold both inputs and outputs of the same dispatch. The
original O9a attempted a single slab and was reverted because the same `GPUBuffer` appeared in both
read-only-storage (input) and storage (output) bindings.

### Design: role-colored multi-slab arena (O9a-v2)

The solution is to partition intermediates by **usage role** so that no buffer appears in
conflicting binding types within a single dispatch.

**Conflict graph construction** (from command tape):

1. For each dispatch, outputs conflict with all inputs and all other outputs (they share a compute
   pass, and the output needs `storage` while inputs need `read-only-storage`).
2. Inputs do NOT conflict with other inputs (multiple `read-only-storage` is OK).
3. Build an undirected conflict graph where nodes = table indices, edges = conflicting pairs.

**Graph coloring → slab assignment:**

4. Greedy-color the conflict graph. Each color → one physical GPUBuffer slab.
5. Interval-pack within each color: assign 256-byte-aligned offsets using a bump allocator over each
   buffer's live range within its slab.
6. Spill to discrete pool for edge cases with too many colors (>4).

**Expected color counts:** Most programs need 2–3 colors (inputs vs outputs, with some sharing). DLM
programs with linear chains typically need 2 colors.

### O9c: Constants slab (implemented)

All `initialData` mallocs (O2 scalar-promoted literals) are packed into a single persistent
GPUBuffer with 256-byte-aligned offsets at tape compile time. This is safe because:

- Constants are always inputs (bound as `read-only-storage`), never outputs.
- Multiple `read-only-storage` bindings to the same buffer are spec-compliant.
- The slab persists across invocations — no per-invocation `createBuffer` or `writeBuffer`.

**Bind group cache benefit:** Constants always reference the same GPUBuffer at the same offsets, so
O9b cache hits are guaranteed for the constant portion of every dispatch.

### 256-byte alignment overhead

| Buffer logical size | Padded (256-byte aligned stride) | Waste  | Typical count in DLM |
| ------------------- | -------------------------------- | ------ | -------------------- |
| 4 bytes (scalar)    | 256 bytes                        | 252B   | ~100                 |
| 16 bytes (2×2 f32)  | 256 bytes                        | 240B   | ~200                 |
| 64 bytes (4×4 f32)  | 256 bytes                        | 192B   | ~50                  |
| 256+ bytes          | size rounded up to 256           | ≤255B  | ~10                  |
| **Total DLM slab**  |                                  | ~100KB | (fits in L1 cache)   |

### Impact on bind group caching

| Scenario                     | Without arena (O9b only) | With O9c slab    | With full arena (O9a-v2) |
| ---------------------------- | ------------------------ | ---------------- | ------------------------ |
| Internal-only dispatches     | ~60% cache hit           | ~80% cache hit   | **100% cache hit**       |
| Dispatches reading inputs    | ~0% cache hit            | ~0% (ext vary)   | ~0% (ext vary)           |
| Repeated invocations (DLM)   | **Fragile** (LIFO)       | Partially stable | **Guaranteed stable**    |
| createBindGroup cost (764 d) | ~6ms → ~3ms (50%)        | ~6ms → ~2ms      | ~6ms → ~0.5ms            |

### Implementation phases

| Phase  | Scope                                          | Effort | Depends on | Status          |
| ------ | ---------------------------------------------- | ------ | ---------- | --------------- |
| O9a    | Single-slab arena (original)                   | Medium | O8a        | **Reverted** ⚠️ |
| O9a-v2 | Colored multi-slab arena                       | Medium | O9c        | Not started     |
| O9b    | Bind group cache (GPUBuffer identity key)      | Small  | Pool LIFO  | **Done** ✅     |
| O9c    | Constants slab (persistent across invocations) | Small  | O8a        | **Done** ✅     |

### Risks

| Risk                                    | Likelihood | Mitigation                                                |
| --------------------------------------- | ---------- | --------------------------------------------------------- |
| 256-byte alignment wastes memory        | Certain    | ~100KB for DLM — negligible vs GPU VRAM                   |
| Graph coloring needs too many colors    | Low        | Spill overflow to discrete pool                           |
| Slab fragmentation on varying programs  | Medium     | Per-JitProgram slab; freed as unit                        |
| Slab too small for unexpected program   | Low        | Fallback to discrete alloc for overflow                   |
| `copyBufferToBuffer` within same buffer | N/A        | WebGPU spec permits same-buffer copies (non-overlapping)  |
| External inputs can't be arena'd        | By design  | Only internal intermediates use arena; inputs stay pooled |

---

## Dispatch Acceleration Roadmap

The dominant bottleneck for DLM and ML inference workloads on WebGPU is **excessive dispatches with
high per-dispatch JS overhead**. The following optimizations form a coherent acceleration stack:

### Measured baseline (DLM Nile m=2, N=100, 764 dispatches, RTX 4070 eGPU)

| Component         | Time (ms) | % of total |
| ----------------- | --------- | ---------- |
| JS loop overhead  | 95        | 76%        |
| queue.submit      | 22        | 18%        |
| createBindGroup   | 6         | 5%         |
| prepareKernelSync | 1         | 1%         |
| **Total**         | **124**   | 100%       |

### Optimization stack

| ID         | Optimization             | Target                            | Impact estimate          | Status          |
| ---------- | ------------------------ | --------------------------------- | ------------------------ | --------------- |
| **O8a**    | Command tape             | JS loop overhead (95ms)           | 95ms → ~2ms (**47×**)    | **Done** ✅     |
| **O9a**    | Arena allocator          | createBindGroup + cache stability | 6ms → ~1ms (reliable)    | **Reverted** ⚠️ |
| **O9a-v2** | Colored multi-slab arena | createBindGroup (full)            | 6ms → ~0.5ms             | Not started     |
| **O9b**    | Bind group caching       | createBindGroup (6ms)             | 6ms → ~3ms (pool LIFO)   | **Done** ✅     |
| **O9c**    | Constants slab           | initialData buffer creation       | Eliminates const mallocs | **Done** ✅     |
| **O6**     | Multi-reduction kernels  | Dispatch count (~764)             | ~5-15% fewer dispatches  | Deprioritized   |
| **A-L**    | Analytical linalg (n≤4)  | Routine fusion barriers           | Enables sqrt DLM fusion  | **Done** ✅     |
| **T0**     | Decoupled Fallback scan  | Scan dispatch count (log N → 1)   | ~0.05ms (3→1 dispatch)   | **Done** ✅     |
| **P7-2**   | WMMA cooperative matrix  | Per-dispatch throughput           | 2-4× matmul GFLOP/s      | Blocked (~2026) |

### Projected wall-clock improvement

| Configuration            | DLM m=2 N=100 | Transformer 12-layer | Diffusion U-Net |
| ------------------------ | ------------- | -------------------- | --------------- |
| Current                  | 124 ms        | ~15 ms               | ~50 ms          |
| + O8a (command tape)     | ~30 ms        | ~5 ms                | ~15 ms          |
| + O9b (BG cache)         | ~27 ms        | ~4.5 ms              | ~13 ms          |
| + Analytical linalg (≤4) | ~25 ms\*      | ~4 ms                | ~12 ms          |
| + WMMA (future)          | ~15 ms        | ~2 ms                | ~6 ms           |

\* Analytical linalg doesn't reduce dispatch count for standard DLM (already fusing via inv
analytical paths). It unlocks the **sqrt DLM variant** and any other workload with small-matrix
Cholesky/QR/TriSolve in scan or block-map bodies.

### Sequencing rationale

1. **O8a first** — eliminates the 76% JS overhead. Largest single improvement. All other
   optimizations compound on top of the lower baseline.
2. **O9 after O8a** — requires O8a's flat buffer table; eliminates the next bottleneck
   (createBindGroup) and makes O8b reliable.
3. **Analytical linalg in parallel** — independent of O8/O9. Unblocks sqrt DLM and broadens the set
   of fusable ML patterns.
4. **WMMA when available** — hardware dependent. The tape + arena infrastructure is already the
   right execution model for WMMA dispatches.

### Who benefits beyond DLM

| ML Architecture           | Dispatch pattern                              | How this stack helps                              |
| ------------------------- | --------------------------------------------- | ------------------------------------------------- |
| **Transformers**          | ~10-15 dispatches/layer × 12+ layers          | O8: 4× overhead reduction on 120-180 dispatches   |
| **RNNs/LSTMs**            | `lax.scan` over T with matmul+activation/step | Already fused via compiled-loop; no change needed |
| **Diffusion (U-Net)**     | 500-1000 conv2d + attention + skip dispatches | O8: 4× overhead; O9: stable bind groups           |
| **MLP inference**         | matmul + bias + activation chains             | O8: moderate; chains already fuse well            |
| **Kalman filters (DLM)**  | `assocScan` + hundreds of linalg dispatches   | O8+O9: 5× total; analytical linalg: enables sqrt  |
| **Optimization (L-BFGS)** | Many small linalg ops per iteration           | O8: 4×; analytical linalg: fuses small-n solves   |

---

## Routine Jaxprification Analysis

### Question: should we generally convert Routines to jaxpr-traced ops?

The 6 current Routines are: **Sort**, **Argsort**, **Cholesky**, **TriangularSolve**, **LU**,
**QR**. Each Routine in a block-map body forces fallback to per-block multi-dispatch execution,
breaking fusion. In scan bodies, Routines are supported via preencoded-multi-step (Phase 3) but
cannot fuse into the compiled-loop shader.

### Per-Routine assessment

| Routine             | Jaxprifiable (n≤4)? | Algorithm                           | Trace graph (n=4) | Worth it?                                                     |
| ------------------- | ------------------- | ----------------------------------- | ----------------- | ------------------------------------------------------------- |
| **inv** (done)      | ✅ Done             | Cramer's rule                       | ~300 ops          | Already ships. Enables DLM 5-tuple fusion.                    |
| **Cholesky**        | ✅ Yes              | Cholesky-Banachiewicz, unrolled     | ~50 ops           | **Yes** — enables sqrt DLM. Simple, small graph.              |
| **TriangularSolve** | ✅ Yes              | Back/forward substitution, unrolled | ~20 ops           | **Yes** — enables sqrt DLM. Tiny graph.                       |
| **QR**              | ✅ Yes              | Householder reflections, unrolled   | ~200 ops          | **Maybe** — enables sqrt DLM backward. Moderate graph size.   |
| **LU**              | ⚠️ Tricky           | Gaussian elim with pivoting         | ~40 ops + control | **No** — pivoting needs data-dependent branching. Not in DLM. |
| **Sort**            | ❌ No               | Comparison-based, O(n log n)        | N/A               | **No** — fundamentally data-dependent. Not ML-critical.       |
| **Argsort**         | ❌ No               | Comparison-based, O(n log n)        | N/A               | **No** — fundamentally data-dependent. Used in topK only.     |

### Recommendation: targeted, not general

**Jaxprify Cholesky + TriangularSolve for n ≤ 4.** These are the two ops blocking sqrt DLM fusion,
both have small and simple trace graphs, and AD "just works" through the traced ops (no custom JVP
rules needed). Optionally add QR for n ≤ 4 to complete the sqrt DLM backward path.

**Do NOT attempt general jaxprification.** Sort/Argsort are fundamentally comparison-based and
cannot be expressed as fixed arithmetic traces. LU requires data-dependent pivoting. For n ≥ 5, the
trace graphs grow rapidly (inv 5×5 ≈ 720 terms, QR 5×5 ≈ 500+ terms) and the JIT compilation cost
makes them unsuitable for inner loops.

**Do NOT jaxprify large-n operations.** The backend-optimized WASM (wasmblr) and WebGPU (WGSL)
Routine implementations use parallelism, shared memory, and hardware-specific tricks that a traced
elementwise graph can never match. For n ≥ 8, a hand-written Routine will always outperform an
unrolled trace.

### Actionable scope

This analysis aligns with and reinforces the existing "Analytical Small-Matrix Linalg" section
above. The concrete next steps:

1. **Cholesky n ≤ 4:** Unrolled Cholesky-Banachiewicz in `lax-linalg.ts`. Pattern: threshold check →
   analytical path → fall through to Routine for n ≥ 5. Identical to existing `inv()` pattern.
2. **TriangularSolve n ≤ 4:** Unrolled back/forward substitution. Needs to handle both upper and
   lower triangular, single and batched RHS.
3. **QR n ≤ 4 (optional):** Unrolled Householder. Moderate effort due to reflector computation.
4. **Testing:** Each path tested against Routine output for numerical agreement at all n ≤ 4.

Priority: **Medium-High** (upgraded from Medium). The sqrt DLM variant is numerically superior for
poorly conditioned systems. Enabling it to fuse for m ≤ 4 on WebGPU is a concrete, achievable goal.

---

## Non-goals

- **General einsum rank-adaptation:** The general `parseEinsumExpression` still fails with
  rank-reduced inputs (e.g., 3-index subscript with 2D tensor). This only matters for exotic
  subscript patterns NOT in `einsumFastPath`. All common batch-matmul patterns are already handled.
  Fixing the general parser would be a separate, lower-priority effort.
- **Multiple bind groups for overflow:** WebGPU supports up to 4 bind groups, but storage bindings
  per shader stage is a global limit — splitting across groups doesn't help.
- **GPU/WASM ratio ≤ 2× target:** Not achievable for small-matrix DLM (m=2) at small N. The
  bottleneck is JS-side JIT loop overhead (~95ms of 124ms for 764 dispatches). The WebGPU command
  tape (O8) reduces this to ~30ms (~4× improvement) but cannot match WASM mega-module's ~6ms due to
  irreducible GPU API costs (`createBindGroup`, `queue.submit`). For large-matrix workloads (matmul
  4096×4096), WebGPU already achieves 53.7% peak.
- **Binding limit optimization on high-limit hardware:** On Deno/NVIDIA with `maxArgs = 1,048,575`,
  the P2 pass has no effect. All dispatch fragmentation comes from P1 structural rules (reduction
  boundaries, diamond heuristic). Browser deployments with Chrome's `maxArgs ≈ 9` will see
  additional P2-caused fragmentation — but optimizing for Chrome's low limit would require a
  different approach (e.g., leaf packing) that doesn't address the P1 structural issue.
