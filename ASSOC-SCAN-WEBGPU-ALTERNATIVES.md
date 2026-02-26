# WebGPU Associative Scan: Dispatch Architecture Alternatives

Analysis of viable strategies to reduce dispatch overhead in the WebGPU fused associative scan
(`dispatchAssocScan`).

## Device Landscape (web3dsurvey.com, Feb 2026)

WebGPU support: **78.29%** of surveyed browsers.

### Limits (cumulative — "100%" means all WebGPU devices)

| Limit                               | Value  | Coverage |
| ----------------------------------- | ------ | -------- |
| `maxComputeWorkgroupSizeX`          | 256    | 100%     |
| `maxComputeWorkgroupSizeX`          | 1024   | 85%      |
| `maxComputeInvocationsPerWorkgroup` | 256    | 100%     |
| `maxComputeInvocationsPerWorkgroup` | 1024   | 81%      |
| `maxComputeWorkgroupStorageSize`    | 16 KB  | 100%     |
| `maxComputeWorkgroupStorageSize`    | 32 KB  | 99%      |
| `maxComputeWorkgroupStorageSize`    | 49 KB  | 12%      |
| `maxComputeWorkgroupsPerDimension`  | 65535  | 100%     |
| `maxStorageBuffersPerShaderStage`   | 8      | 100%     |
| `maxStorageBuffersPerShaderStage`   | 10     | 99.6%    |
| `maxStorageBuffersPerShaderStage`   | 16     | 16%      |
| `minStorageBufferOffsetAlignment`   | 32     | 100%     |
| `minStorageBufferOffsetAlignment`   | 256    | 85%      |
| `minUniformBufferOffsetAlignment`   | 32     | 100%     |
| `minUniformBufferOffsetAlignment`   | 256    | 89%      |
| `maxBufferSize`                     | 256 MB | 100%     |
| `maxBufferSize`                     | 2 GB   | 80%      |

### Features

| Feature                   | Coverage   | Notes                                  |
| ------------------------- | ---------- | -------------------------------------- |
| `shader-f16`              | **91.7%**  | Safe to require with fallback          |
| `subgroups`               | **65.7%**  | Majority support — viable as fast path |
| `timestamp-query`         | **99.2%**  | Safe to use for profiling              |
| `indirect-first-instance` | **99.97%** | Could enable GPU-driven dispatch       |
| `dual-source-blending`    | 78.8%      | Not relevant for compute               |

### Key takeaways for associative scan design

- **Workgroup size 256 is universal.** All devices support it.
- **Shared memory: 16 KB guaranteed, 32 KB on 99%.** For f32 with ping-pong:
  `2 × 256 × 4 = 2048 bytes` for scalar scan — plenty of headroom. Multi-leaf:
  `2 × 256 × totalElemsPerPos × 4` must stay ≤ 16 KB for universal support, meaning up to
  `totalElemsPerPos = 8` (8 f32s per position). Targeting 32 KB covers 99% and allows 16 f32s per
  position.
- **Subgroups at 65.7%** — much higher than expected. Worth implementing as a fast path with
  fallback to workgroupBarrier-only scan.
- **Storage buffer limit of 8** is universal. The assoc scan shader uses 2 (ping, pong) + numConsts
  storage buffers, so numConsts ≤ 6. This is already enforced elsewhere.
- **`minStorageBufferOffsetAlignment`**: 15% of devices have 32-byte alignment (not 256). The
  current preencoded scan conservatively uses uniform buffers instead of storage buffer offsets —
  this remains the right choice.

## Current Architecture

The Kogge-Stone parallel prefix scan runs `ceil(log₂ N)` rounds. Each round:

- Thread `j` computes `fn(data[j - stride], data[j])` where `stride = 2^round`
- Threads with `j < stride` copy their value unchanged

**Current dispatch pattern:**

```
copy-in submit                          1 submit
[optional reverse submit]               0-1 submit
round 0: encoder → dispatch → submit    1 submit + 1 uniform create/destroy
round 1: encoder → dispatch → submit    1 submit + 1 uniform create/destroy
...
round K: encoder → dispatch → submit    1 submit + 1 uniform create/destroy
[optional reverse submit]               0-1 submit
copy-out submit                         1 submit
─────────────────────────────────────────────────
Total: K + 2 submits (K + 4 with reverse)
For N=65536: 18-20 submits, 16 uniform buffer create/destroy cycles
```

Each `queue.submit()` has JS↔GPU round-trip overhead. Each `createCommandEncoder()` allocates a
command buffer. Each round creates and destroys a uniform buffer for `{stride, N}`.

The scan compiled-loop shader (`nativeScanMultiShaderSource`) is the existence proof that a native
on-GPU loop works: it runs `for (var iter = 0; iter < length; iter++)` in a single dispatch. But
that works because each thread only reads/writes its own `carry[gidx]` — no cross-thread
communication. The associative scan has cross-thread dependencies (`data[j - stride]`), which is the
fundamental challenge.

---

## Alternative 0: Single Command Encoder (Batch Submits)

**Idea:** Encode all rounds into one `GPUCommandEncoder`, submit once.

**Changes:** ~30 lines in `dispatchAssocScan`. Pre-allocate all uniform buffers (or use one buffer
with dynamic offsets like `dispatchPreencodedScan` already does), encode all round dispatches into
one encoder, single `queue.submit()`.

**Dispatch count:** Still K dispatches, but **1 submit** instead of K.

**Why it works:** WebGPU guarantees sequential execution of dispatches within a command buffer. The
ping→pong→ping data dependencies are automatically respected by implicit storage barriers between
compute passes.

**Performance gain:** Eliminates K-1 JS↔GPU round-trips. For N=65536 (K=16): 16 dispatches × 1
submit vs 16 × 16 submits. The GPU can pipeline work from the command buffer without waiting for JS.

**Limitations:** Still K separate dispatches with implicit memory barriers between them. GPU must
drain the previous dispatch's workgroups before starting the next. Each dispatch has scheduling
overhead (workgroup launch, barrier).

**Risk:** Near zero — `dispatchPreencodedScan` already uses this exact pattern successfully.

**Effort:** Small (1-2 hours).

---

## Alternative 1: Workgroup-Local Native Loop (Shared Memory)

**Idea:** Each workgroup runs the full Kogge-Stone scan in shared memory using `workgroupBarrier()`
between rounds. Single dispatch handles N ≤ workgroup_size.

**Shader structure:**

```wgsl
var<workgroup> shmem_ping: array<f32, 256>;
var<workgroup> shmem_pong: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tidx = lid.x;

  // Load from global to shared memory
  shmem_ping[tidx] = input[tidx];
  workgroupBarrier();

  // Kogge-Stone rounds (all in shared memory)
  for (var stride = 1u; stride < 256u; stride *= 2u) {
    let val = shmem_ping[tidx];
    if (tidx >= stride) {
      val = fn(shmem_ping[tidx - stride], val);
    }
    shmem_pong[tidx] = val;
    workgroupBarrier();

    // Swap ping/pong (or use alternating reads)
    shmem_ping[tidx] = shmem_pong[tidx];
    workgroupBarrier();
  }

  output[tidx] = shmem_ping[tidx];
}
```

**Dispatch count:** 1 for N ≤ 256.

**Shared memory requirement** (vs device landscape):

- Scalar f32 scan: `2 × 256 × 4 = 2048 bytes` — 100% of devices (16 KB min)
- Multi-leaf scan, 8 f32s/pos: `2 × 256 × 8 × 4 = 16384` — 100% of devices
- Multi-leaf scan, 16 f32s/pos: `2 × 256 × 16 × 4 = 32768` — 99% of devices
- Larger leaf structures (>16 f32s/pos with 16 KB target, >32 with 32 KB) would need fallback to the
  current global-memory path

**Performance gain:** Eliminates ALL dispatch overhead and global memory round-trips for the rounds.
Data stays in L1/shared memory for the entire scan. For N ≤ 256: one dispatch replaces 8 dispatches.

**Limitations:**

- Only handles N ≤ workgroup_size (typically 256)
- Multi-leaf scans with large per-position data hit shared memory limits
- Must handle the `fn(a, b)` body using private variables (already done by the current `internal_N`
  arrays — those stay private, only final results go to shared memory)

**Larger N:** Requires composition with other alternatives (see Alt 2, 3).

**Codegen complexity:** Moderate — need to adapt `assocScanFusedShaderSource` to emit shared-memory
declarations, change buffer access patterns from global ping/pong to shared-memory ping/pong, and
add `workgroupBarrier()` calls. The expression codegen (`genAssocScanExpression`) needs modification
to read from shared memory instead of global `ping[]`.

**Risk:** Medium. `workgroupBarrier()` requires uniform control flow (already handled by the
`_valid` flag pattern used elsewhere). The current implementation has `if (a_pos >= 0)` branching —
must restructure to use valid-flag or ensure all threads enter the barrier.

**Effort:** Medium (2-3 days).

---

## Alternative 2: Three-Level Hierarchical Scan

**Idea:** Decompose large scans into local workgroup scans + inter-workgroup scans + propagation.
Classic GPU parallel scan algorithm.

**Phase 1 — Local scan:** Each workgroup scans its 256 elements in shared memory (Alternative 1).
Also writes its workgroup-total to an auxiliary buffer.

**Phase 2 — Scan aggregates:** Scan the per-workgroup totals (≤256 workgroups for N ≤ 65536 → fits
in one workgroup; larger → recursive).

**Phase 3 — Propagate:** Each workgroup adds the scanned prefix from phase 2 to all its local
elements.

**Dispatch count:** 3 dispatches for N ≤ 65536 (256 × 256), all in one submit.

**Performance comparison (N=65536):** | Metric | Current | Alt 0 | Alt 2 |
|---------------------|-----------|------------|-----------| | Dispatches | 16 | 16 | 3 | | Submits
| 18-20 | 1 | 1 | | Global mem accesses | 2N per round (32N total) | same | 2N × 3 phases (6N) |

**Limitations:**

- Three separate shader variants needed: local-scan, scan-aggregates, propagate
- Auxiliary buffer allocation for workgroup totals
- Multi-leaf scans multiply shared memory by leaf element count
- Recursive decomposition needed for N > 65536 (rare in practice)

**Codegen complexity:** High. Three related shaders sharing the same body expression codegen but
with different indexing and data flow patterns.

**Risk:** Medium-high. More code surface, more edge cases, but this is a well-known GPU algorithm
with extensive literature.

**Effort:** Large (4-7 days).

---

## Alternative 3: Hybrid Local + Batched Global

**Idea:** Combine Alternative 1 (workgroup-local native loop) with Alternative 0 (single-submit
batched dispatches) for the inter-workgroup rounds.

**Architecture:**

1. Phase 1: Local Kogge-Stone in shared memory (1 dispatch, handles 256 elements)
2. Phase 2+: Remaining ceil(log₂(N/256)) global rounds via batched dispatches, but now operating on
   workgroup-aggregate values instead of individual elements

This is essentially Alternative 2 but without the separate propagation pass — the global rounds
operate on the full arrays, not just aggregates. Simpler to implement but less efficient than the
full three-level decomposition.

**Dispatch count:** `1 + ceil(log₂(N/256))` in one submit. For N=65536: 1 + 8 = 9 dispatches (vs 16
current).

**Advantage over Alt 2:** Simpler codegen — only two shader variants (local and global), and the
global variant is close to the existing shader.

**Risk:** Low-medium.

**Effort:** Medium (3-4 days).

---

## Alternative 4: Uniform Buffer with Dynamic Offsets

**Idea:** Keep the current multi-dispatch architecture but eliminate per-round uniform buffer
allocation. Use a single pre-allocated uniform buffer with dynamic offsets (one stride+N pair per
round), exactly like `dispatchPreencodedScan` does.

**Changes:** Pre-allocate uniform buffer with `ceil(log₂ N)` entries at
`minUniformBufferOffsetAlignment`-aligned offsets. Use
`setBindGroup(1, uniformBG, [round * alignment])` per dispatch.

**This is a strict subset of Alternative 0** — same core idea (pre-allocate uniforms, single
encoder) but could also be done independently as a smaller refactor.

**Dispatch count:** Still K dispatches. Eliminates K uniform buffer create/destroy cycles.

**Effort:** Small (included in Alt 0).

---

## Alternative 5: Loop-in-Shader with Uniform Stride

**Idea:** Instead of one dispatch per stride, emit a shader with an inner loop over all strides. Use
global memory for ping/pong (not shared memory), with a `storageBarrier()` between rounds.

```wgsl
for (var stride = 1u; stride < N; stride *= 2u) {
  // read from ping, write to pong
  // ...
  storageBarrier();  // ← does this help?
  // swap conceptual ping/pong via conditional
}
```

**Why this doesn't work:** `storageBarrier()` only synchronizes within a workgroup. Threads in
different workgroups cannot synchronize within a single dispatch. Thread 0 in workgroup 0 might
write `data[0]` but thread 256 in workgroup 1 reads `data[0]` — no barrier can synchronize this.

**Verdict: NOT VIABLE for N > workgroup_size.** This is the fundamental "no global barrier"
constraint of WebGPU. Only `workgroupBarrier()` + shared memory works, and only within a workgroup.

---

## Alternative 6: Subgroups

**Idea:** Use `subgroupShuffle` / `subgroupInclusiveAdd` for SIMD-width parallel prefix within a
subgroup (typically 32-64 threads), reducing the number of `workgroupBarrier()` calls needed.

```wgsl
// Instead of 8 barrier rounds for 256 threads:
let partial = subgroupInclusiveAdd(val);  // 32-wide scan, 0 barriers
// Then 3 barrier rounds for inter-subgroup scan (256/32 = 8 subgroups)
```

**Performance gain:** Reduces barrier count from log₂(256) = 8 to log₂(256/subgroupSize) ≈ 3. Fewer
barriers = faster.

**Device coverage:** **65.7%** of WebGPU devices support `subgroups` (web3dsurvey.com, Feb 2026).
This is much higher than previously expected — majority coverage, not just bleeding edge.

**Limitations:**

- 34% of devices still lack support — must be a fast path with fallback.
- Only accelerates the within-workgroup scan — doesn't change the inter- workgroup architecture.
- `subgroupInclusiveAdd` only works for built-in ops (add, mul, min, max). Custom `fn(a, b)` needs
  `subgroupShuffle` + explicit combination.
- Subgroup size varies by hardware (32 on NVIDIA, 64 on AMD, variable on Intel) — shader must handle
  different sizes dynamically via `subgroup_size` builtin.

**Verdict: VIABLE as optional fast path.** Worth implementing after Alt 1 as an enhancement to the
workgroup-local scan. Generates two shader variants: one with subgroup ops (when feature available)
and one without.

**Implementation sketch:**

```wgsl
enable subgroups;

// Phase 1: subgroup-level inclusive scan (0 barriers)
var val = input[tidx];
val = subgroupInclusiveOp(val);  // for built-in ops
// OR: manual shuffle for custom fn:
// for (var s = 1u; s < subgroup_size; s *= 2u) {
//   let other = subgroupShuffleUp(val, s);
//   if (subgroup_invocation_id >= s) { val = fn(other, val); }
// }

// Phase 2: inter-subgroup scan via shared memory (3 barriers for 8 subgroups)
let sg_id = subgroup_id;  // which subgroup am I in
let sg_last = subgroup_size - 1u;
if (subgroup_invocation_id == sg_last) {
  shmem[sg_id] = val;  // write subgroup total
}
workgroupBarrier();
// ... scan shmem[0..num_subgroups] ...
// ... add prefix back to all elements ...
```

---

## Recommendation Matrix

| Alternative | Dispatches (N=64K) | Submits | Effort | Risk     | Speedup Est.       | Device Req.       |
| ----------- | ------------------ | ------- | ------ | -------- | ------------------ | ----------------- |
| Current     | 16                 | 18-20   | —      | —        | baseline           | 100%              |
| **Alt 0**   | 16                 | 1       | Small  | Low      | 2-4×               | 100%              |
| Alt 1       | 1 (N≤256 only)     | 1       | Medium | Med      | 5-10× (small N)    | 100% (16KB shmem) |
| **Alt 2**   | 3                  | 1       | Large  | Med-High | 3-8×               | 100%              |
| Alt 3       | 9                  | 1       | Medium | Low-Med  | 2-5×               | 100%              |
| Alt 4       | 16                 | 16      | Small  | Low      | 1.2-1.5×           | 100%              |
| Alt 6       | 1 (N≤256)          | 1       | Medium | Med      | +30-50% over Alt 1 | 65.7% (fast path) |

### Recommended path

1. **Do Alt 0 now** — single command encoder, pre-allocated uniforms with dynamic offsets. Minimal
   risk, immediate benefit, sets up the infrastructure for everything else. The preencoded-routine
   scan already proves this pattern.

2. **Then Alt 1** — workgroup-local native loop for N ≤ 256. This gives the "native loop" path for
   the most common small-N cases. Self-contained shader change with clear correctness criteria.

3. **Then Alt 6 as fast path on Alt 1** — subgroups at 65.7% device coverage is viable as an
   optional fast path. Generate two shader variants: one with `subgroupInclusiveOp` (fewer
   barriers), one without (workgroupBarrier only). Feature-detect at `prepareAssocScan` time.

4. **Then Alt 2 using Alt 1 as building block** — the local scan phase IS Alt 1. Adding
   aggregate-scan + propagate phases extends coverage to all N. Alt 0's single-submit infrastructure
   handles the 3-dispatch submission.

Alt 3 is an intermediate option if Alt 2's complexity is too high, but the literature on three-level
GPU scan is mature and well-understood.

### What about the scan compiled-loop?

The scan compiled-loop (`nativeScanMultiShaderSource`) runs a sequential scan as a native on-GPU
loop because the problem is embarrassingly per-thread (no cross-thread deps). It's important to note
this approach **cannot be extended** to the associative scan: the whole point of associative scan is
that the binary operation combines values from different positions, requiring cross- thread
communication.

The only way to get a "native loop" for the associative scan is via shared memory within a workgroup
(Alt 1), which is inherently limited to N ≤ workgroup_size per dispatch.

### When does associative scan matter vs sequential scan?

| Scenario               | Sequential scan                  | Associative scan  | Winner          |
| ---------------------- | -------------------------------- | ----------------- | --------------- |
| N small, simple fn     | 1 dispatch (compiled-loop)       | 1+ dispatches     | Sequential      |
| N large, grad needed   | O(N) backward                    | O(log N) backward | **Associative** |
| N large, no grad       | 1 dispatch                       | 16+ dispatches    | Sequential      |
| Body has internal deps | Compiled-loop (WASM) or fallback | Always works      | Depends         |

The associative scan's primary value is **grad performance**: O(log N) backward depth vs O(N) for
sequential scan. The dispatch overhead only matters for the forward pass — and even there, the
difference compounds in grad(assocScan) where the backward also uses O(log N) dispatches.

For `grad(assocScan)` with N=1600 (Kalman smoother), the backward does ~11 rounds × however many
dispatches per round. Reducing from 11 submits to 1 submit (Alt 0) or from 11 dispatches to 3
(Alt 2) directly impacts the backward latency that dominates DLM training time.
