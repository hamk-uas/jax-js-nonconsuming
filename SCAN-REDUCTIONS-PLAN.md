# Fused Reductions in WebGPU Scan Loops: Architecture & Implementation Plan

## 1. The Current Architecture: Why Fused Reductions Fail

The WebGPU `compiled-loop` scan path (`nativeScanMultiShaderSource`) maps each output element to a
1D GPU thread via `gidx`. All internal intermediates live as thread-private scalar `var`
declarations:

```wgsl
var result_val_2: f32 = f32(0);
if (gidx < 16) { result_val_2 = /* elementwise expression */; }
var internal_0: f32 = result_val_2;  // scalar, one per thread
```

When a subsequent **reduction step** references `internal_0`, the codegen
(`genScanExpressionWithRidx`) resolves `GlobalIndex(gid=N)` where N maps to an internal by emitting
_only_ the scalar variable name — **discarding the index expression entirely**:

```ts
// webgpu.ts L3766 — internal branch
const varName = `internal_${internalIdx}`;
return varName; // idxCode (e.g., "ridx * 4 + gidx") is THROWN AWAY
```

This means a matmul reduction loop reads the same scalar 16 times instead of indexing into a
16-element intermediate array. The result is mathematically wrong; we currently reject this pattern
entirely in `scan-plan.ts:790-800`.

Compare this to the `associativeScan` shaders which already handle the same problem correctly: they
declare internals as `var<private>` arrays and use `eidx`/`ridx` loops with proper indexing via the
`resolveGlobalIndex` callback and the shared `createWgslGen` + `emitAssocScanBodySteps`
infrastructure.

---

## 2. Root Cause & The Correct Fix: `var<private>` Internal Arrays

The original plan proposed `var<workgroup>` (shared memory) + barriers. After reviewing the
codebase, **this is the wrong approach**. Here is why:

### Why `var<workgroup>` is wrong for this problem

The scan `compiled-loop` path assigns **one gidx per output element**. Each thread's intermediate is
consumed **only by the same thread** in the following reduction step — the reduction's `ridx` loop
iterates over the elements that _this thread_ owns, reading different offsets of the same
intermediate array.

**There is no cross-thread data sharing.** Thread N's matmul reduction reads from intermediate
elements that thread N itself computed. The issue is not that threads can't see each other's data
(that would need shared memory). The issue is that intermediates are stored as _scalars_ when they
should be _arrays_.

Using `var<workgroup>`:

- Wastes shared memory (16KB limit, cannot exceed)
- Requires `workgroupBarrier()` (synchronization overhead)
- Limits workgroup size / occupancy
- Is architecturally wrong — solves a cross-thread problem that doesn't exist

### Why `var<private>` arrays are correct

The associative scan already solves this correctly:

```wgsl
// From associativeScanFlatShaderSource (webgpu.ts L4368):
var internal_0: array<f32, 16>;  // var<private> — per-thread array

// Write phase (elementwise step):
for (var eidx: i32 = 0; eidx < 16; eidx++) {
  internal_0[eidx] = /* compute element eidx */;
}

// Read phase (reduction step):
for (var eidx: i32 = 0; eidx < 4; eidx++) {
  var acc: f32 = 0.0;
  for (var ridx: i32 = 0; ridx < 4; ridx++) {
    acc += internal_0[ridx * 4 + eidx] * internal_1[ridx * 4 + eidx];
  }
  internal_2[eidx] = acc;
}
```

Each thread has its own private copy of `internal_0`. No barriers needed. No shared memory. No
workgroup size constraints. The hardware places these in registers or spills to VRAM automatically.

---

## 3. Feasibility Assessment

### What already exists

The following reusable infrastructure is **already implemented** in the associative scan codegen and
the `wgsl-gen.ts` module:

| Component                 | File                             | Purpose                                                                                                                        |
| ------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `createWgslGen`           | `src/backend/webgpu/wgsl-gen.ts` | WGSL expression generator with CSE, `resolveGlobalIndex` callback, `gidxOverride`, `ridxOverride` — handles all AluOps         |
| `ResolveGlobalIndex` type | `wgsl-gen.ts`                    | Callback: `(bufIdx, indexExpr, dtype) → WGSL string` — decouples buffer access from expression codegen                         |
| `emitAssocScanBodySteps`  | `webgpu.ts L4181`                | Emits multi-step body with `var<private>` internal arrays, proper `eidx` loops for elementwise and `ridx` loops for reductions |
| `NativeScanMultiStep`     | `webgpu.ts L62`                  | Step descriptor with `outputInternalIdx`, `outputSize` — already tracks internal array sizes                                   |

### What needs to change

**The changes are structurally modest.** The scan shader codegen (`nativeScanMultiShaderSource`)
needs to switch from its bespoke `genScanExpressionWithRidx` + scalar-`var` approach to the reusable
`createWgslGen` + `emitAssocScanBodySteps` pattern that already works in associative scan. The
scan-plan rejection logic then gets relaxed.

### Size constraints

| Block size       | Intermediate bytes | `var<private>` limit            |
| ---------------- | ------------------ | ------------------------------- |
| 4×4 (Br=Bc=Bk=4) | 2 × 64B = 128B     | No issue (register)             |
| 16×16            | 2 × 1KB = 2KB      | Fine (register spill)           |
| 32×32            | 2 × 4KB = 8KB      | Workable (VRAM spill)           |
| 64×64            | 2 × 16KB = 32KB    | Pressure — may reduce occupancy |

The WebGPU spec imposes no fixed limit on `var<private>` size (unlike `var<workgroup>` which is
capped at `maxComputeWorkgroupStorageSize`). However, excessive private memory reduces GPU
occupancy. For the target use cases (blockMap tile sizes ≤ 32×32), register pressure is manageable.

### Risk assessment

| Risk                                  | Severity | Mitigation                                                                                                                                                                                 |
| ------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Register pressure for large tiles     | Low      | Keep rejection for tiles where `totalInternalBytes > threshold` (e.g., 8KB)                                                                                                                |
| Workgroup size mismatch between steps | Medium   | All steps share `maxKernelSize` as the outer loop bound, with per-step `if (eidx < stepSize)` guards — same as associative scan                                                            |
| Expression codegen correctness        | Low      | `createWgslGen` is already battle-tested in associative scan and block-map fused shaders                                                                                                   |
| Scan-specific dataIdx/iter access     | Medium   | Need a scan-aware `resolveGlobalIndex` callback that handles `const[idx]`, `carry[idx]`, `xs[dataIdx*stride+idx]`, `internal[idx]` — similar to existing `flatResolve` in associative scan |

---

## 4. Implementation Plan

### Phase 1: Refactor `nativeScanMultiShaderSource` to use `createWgslGen`

**Goal:** Replace the bespoke `genScanExpressionWithRidx` with the reusable `createWgslGen` +
`resolveGlobalIndex` pattern, initially producing identical output for the existing elementwise-only
case.

**Changes:**

1. **Add internal array sizes to `NativeScanMultiParams`:**

   ```ts
   internalSizes: number[];   // element count per internal
   internalDtypes: DType[];   // dtype per internal
   ```

2. **Build a scan-specific `resolveGlobalIndex` callback:**

   ```ts
   const scanResolve: ResolveGlobalIndex = (gid, idxCode, dtype) => {
     if (gid < numConsts) return `const${gid}[${idxCode}]`;
     if (gid < numConsts + numCarry) {
       const ci = gid - numConsts;
       const isLocal = idxCode === "eidx" && carryElemCounts[ci] === kernelSize;
       return isLocal ? `c_${ci}` : `carry${ci}[${idxCode}]`;
     }
     if (gid < numConsts + numCarry + numX) {
       const xi = gid - numConsts - numCarry;
       return `xs${xi}[i32(dataIdx) * ${xsElemStrides[xi]} + ${idxCode}]`;
     }
     // INTERNAL — array access with proper index
     const ii = gid - numConsts - numCarry - numX;
     return `internal_${ii}[${idxCode}]`;
   };
   ```

3. **Declare internal arrays as `var<private>`:**

   ```wgsl
   var internal_0: array<f32, 16>;  // per-thread, not shared
   var internal_1: array<f32, 16>;
   ```

4. **Replace the step execution loop** with `emitAssocScanBodySteps` (or a scan-specific variant
   that shares the same structure), using `eidx` loops within each step:
   - Elementwise: `for (eidx) { internal_N[eidx] = expr; }`
   - Reduction: `for (eidx) { acc = 0; for (ridx) { acc += ...; } internal_N[eidx] = epilogue; }`
   - Carry write: copy from internal array → `carry[gidx]` and `ys[dataIdx*stride+gidx]`

5. **Delete `genScanExpressionWithRidx`** — fully replaced by `createWgslGen`.

### Phase 2: Relax scan-plan rejection

**Goal:** Allow reduction-with-internal patterns through to `compiled-loop`.

**Changes in `tryPrepareWebGPUNativeScan`:**

1. Remove the `hasInternalInput` rejection (scan-plan.ts L790-800).
2. Add per-internal size tracking:
   ```ts
   const internalSizes: number[] = [];
   const internalDtypes: DType[] = [];
   for (const [jitId, idx] of jitIdToInternalIdx) {
     const info = outputToStepInfo.get(jitId)!;
     const kernel = info.step.source as Kernel;
     internalSizes[idx] = kernel.size as number;
     internalDtypes[idx] = kernel.outputs[info.outputIdx].dtype;
   }
   ```
3. Add a total private memory budget check:
   ```ts
   const totalInternalBytes = internalSizes.reduce(
     (sum, size, i) => sum + size * byteWidth(internalDtypes[i]),
     0,
   );
   if (totalInternalBytes > MAX_PRIVATE_BYTES) return null; // fallback
   ```
4. Pass `internalSizes` and `internalDtypes` through `NativeScanMultiParams`.

### Phase 3: Adapt carry snapshot and writeback for array intermediates

**Goal:** The current `c_N` carry snapshot logic reads one element per thread. When the carry itself
becomes an eidx-indexed array, the snapshot must load all elements.

**Changes:**

- Carry snapshot: `for (eidx) { c_N[eidx] = carry_N[eidx]; }` (or keep the existing single-element
  path when `carryElemCount === maxKernelSize`, meaning each thread owns exactly one element).
- Carry writeback: after the last step that writes carry N, copy from the carry-producing internal
  or result array back to the `carry` storage buffer.
- Ys writeback: similarly index with `dataIdx * stride + eidx`.

### Phase 4: Verification

1. Re-enable `grad(tiledMatmul)` test (currently passing via fallback path).
2. Add `acceptPath: "compiled-loop"` variant that forces the fused path.
3. Verify correctness at 4×4 and 8×8 tile sizes.
4. Benchmark compiled-loop vs preencoded-multi-step for scan with matmul body.
5. Run full test suite.

### Expected outcome

The `compiled-loop` path handles the full `tiledMatmul` gradient body in a single GPU dispatch per
scan iteration, eliminating the N×S separate dispatch overhead of the `preencoded-multi-step`
fallback. For small tile sizes (4×4 to 16×16), this should be 5-50× faster depending on the number
of scan iterations.
