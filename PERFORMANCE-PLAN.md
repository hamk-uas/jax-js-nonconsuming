# Performance Improvement Plan

Follow-up performance work after the TUNER-IMPROVEMENT-PLAN (Improvements 1–8). Covers the next
layer of optimizations: compute-bound kernels, SIMD extensions, and GPU instruction-level
improvements.

---

## Priority Summary

| #   | Feature                    | Priority   | Expected Impact | Risk   |
| --- | -------------------------- | ---------- | --------------- | ------ |
| P1  | Tiled matmul (WebGPU)      | **High**   | 5–10× large mm  | Low    |
| P2  | Relaxed SIMD FMA (WASM)    | **Medium** | ~2× dot product | Medium |
| P3  | WASM i64 in wasmblr        | **Medium** | Medium-High     | Low    |
| P4  | Conv2d tuning              | **Medium** | 2–5× conv       | Low    |
| P5  | WebGPU subgroups           | **Low**    | 2–4× reductions | High   |
| P6  | Benchmark validation suite | **Medium** | N/A (tooling)   | None   |

---

## P1: Tiled Matmul (WebGPU shared-memory blocking)

**Priority:** High — single largest performance opportunity; matmul dominates transformer inference.

### Problem

Current matmul compiles as `Dot = Mul → Reduce`. Each output element `C[i,j]` is computed by one
thread doing a full dot product across the K dimension: `C[i,j] = sum(A[i,:] * B[:,j])`. Each thread
loads an entire row of A and column of B from global memory independently. For `N×N` matmul this
means `N³` global loads with no data reuse — far below peak.

**Current codegen path:**

```
numpy.matmul(x, y)                            → src/library/numpy.ts:1108
 → lax.dot(x, y, {lhsContractingDims: [-1],   → src/library/lax.ts:41
                   rhsContractingDims: [-2]})
   → Primitive.Dot                              → src/frontend/core.ts
     → jitRules[Primitive.Dot]                  → src/frontend/jit.ts:1817
       → Primitive.Mul → Primitive.Reduce        (lowered to Mul+Add reduction)
         → Kernel(nargs=2, size=N², reduction=Add over axis K)
           → pipelineSource() in webgpu.ts      → src/backend/webgpu.ts:1634
             → tuneWebgpu(kernel, caps)          → src/tuner.ts
```

The shader thread computes one output element with a serial reduction loop over K. No shared memory,
no data reuse between threads. This gets ~1/3 of peak GFLOP/s on tested hardware.

### Approach: Workgroup-level tiling

Standard GPU tiled matmul:

1. Each **workgroup** of size `TILE_M × TILE_N` threads computes a `TILE_M × TILE_N` tile of `C`.
2. The K dimension is iterated in tiles of `TILE_K`.
3. Per K-tile step:
   - Collaboratively load a `TILE_M × TILE_K` tile of A into shared memory
   - Collaboratively load a `TILE_K × TILE_N` tile of B into shared memory
   - `workgroupBarrier()`
   - Each thread accumulates its `C[i,j]` from the shared tiles
   - `workgroupBarrier()`
4. Write final accumulated values to global `C`.

This converts `O(N)` global loads per thread to `O(N/TILE_K)` shared loads — data reuse factor of
`TILE_K`.

### Implementation Plan

**Step 1: Detect tiled matmul eligibility**

In `splitGraphDataflow()` or `jitCompile()` (`src/frontend/jit.ts`), detect when a `Kernel` is a Dot
reduction (Mul+Add reduce over last axis) with two 2D matrix inputs. Set a flag on the Kernel or
emit a dedicated `"tiled_matmul"` JitStep type.

Location: `src/frontend/jit.ts:1817` — the `Primitive.Dot` jit rule is the natural detection point.
Currently it lowers to `Mul → Reduce`; instead, for eligible shapes, it can emit a specialized step.

Eligibility: `lhsContractingDims.length === 1` (single inner-product axis), both inputs 2D or
batched-2D, f32 dtype, K ≥ TILE_K (e.g., 16), M and N ≥ TILE_M (e.g., 16).

**Step 2: Tiled matmul shader generator**

Create `src/backend/webgpu/tiled-matmul.ts`:

```wgsl
// Parameters: TILE_M=16, TILE_N=16, TILE_K=16
var<workgroup> tileA: array<f32, TILE_M * TILE_K>;
var<workgroup> tileB: array<f32, TILE_K * TILE_N>;

@compute @workgroup_size(TILE_N, TILE_M)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.y * TILE_M + lid.y;
  let col = wg.x * TILE_N + lid.x;
  var acc: f32 = 0.0;

  for (var t: u32 = 0; t < K; t += TILE_K) {
    // Collaborative load
    tileA[lid.y * TILE_K + lid.x] = A[row * K + t + lid.x];
    tileB[lid.y * TILE_N + lid.x] = B[(t + lid.y) * N + col];
    workgroupBarrier();

    // Accumulate
    for (var k: u32 = 0; k < TILE_K; k++) {
      acc += tileA[lid.y * TILE_K + k] * tileB[k * TILE_N + lid.x];
    }
    workgroupBarrier();
  }

  C[row * N + col] = acc;
}
```

Tile sizes should be tunable. Start with `TILE = 16`, which uses `2 × 16 × 16 × 4 = 2048 bytes`
shared memory — well within typical limits.

**Step 3: Dispatch path**

Add a `dispatchTiledMatmul()` method in `src/backend/webgpu.ts` that:

1. Creates/caches the pipeline for the given `(M, N, K, TILE_M, TILE_N, TILE_K)` combination
2. Dispatches `ceil(N/TILE_N) × ceil(M/TILE_M)` workgroups
3. Handles edge cases (M, N, K not divisible by tile size) with bounds checks in shader

Cache key: include tile dimensions in the pipeline cache key via `ShaderPipelineCache`.

**Step 4: Batched matmul**

Extend to batched matmul: add a batch dimension loop or a 3D workgroup grid where `wg.z` indexes the
batch.

### Source References

| File                              | Line/Symbol                | Relevance                                         |
| --------------------------------- | -------------------------- | ------------------------------------------------- |
| `src/frontend/jit.ts`             | `jitRules[Primitive.Dot]`  | Where Dot is lowered to Mul+Reduce (line 1817)    |
| `src/frontend/jit.ts`             | `Primitive.Conv` rule      | Conv also lowers through Dot (line 1827)          |
| `src/backend/webgpu.ts`           | `pipelineSource()`         | Current shader codegen for reductions (line 1634) |
| `src/backend/webgpu.ts`           | `pipelineSubmit()`         | Dispatches compute pipelines (handles ShaderInfo) |
| `src/backend/webgpu/codegen.ts`   | `calculateGrid()`          | Grid splitting for large dispatches               |
| `src/backend/webgpu/codegen.ts`   | `ShaderInfo`               | Carries `sharedMemoryBytes`, `workgroupSize`      |
| `src/tuner.ts`                    | `tuneWebgpu()`             | Already has groups/local/shared-memory support    |
| `bench/matmul.bench.ts`           | 2048×2048, 4096×4096       | Existing matmul benchmarks                        |
| `.github/copilot-instructions.md` | "Tiled matmul opportunity" | Notes this is a 5–10× opportunity                 |
| `src/library/numpy.ts`            | `matmul()`                 | Public API, calls `lax.dot` (line 1108)           |
| `src/library/lax.ts`              | `dot()`                    | General contraction, transposes inputs (line 41)  |

### Tests

- `bench/matmul.bench.ts` — extend with tiled vs naive comparison
- New: `test/tiled-matmul.test.ts` — correctness (square, rectangular, batched, edge sizes)
- Verify `grad(matmul)` still works (epilogue fusion, conv backprop)

### Acceptance

- 2048×2048 f32 matmul at ≥40% of theoretical GFLOP/s (currently ~33%)
- All existing matmul tests pass (Conv, Dot, matmul, einsum)
- Conv2d implicitly benefits (Conv lowers through Dot — `jit.ts:1837`)

---

## P2: Relaxed SIMD FMA (WASM `f32x4.relaxed_madd`)

**Priority:** Medium — 2× dot-product throughput for WASM routines and potentially JIT kernels.

### Problem

WASM SIMD dot products currently use separate `f32x4.mul` + `f32x4.add` instructions. The Relaxed
SIMD proposal adds `f32x4.relaxed_madd(a, b, c)` which computes `a*b+c` in a single instruction,
potentially using hardware FMA. This doubles multiply-accumulate throughput.

### Current SIMD Usage

| Location                             | Pattern                           | Would benefit from FMA |
| ------------------------------------ | --------------------------------- | ---------------------- |
| `src/backend/wasm/wasmblr-hl.ts:594` | `simdReductionF32()`              | **Yes** — inner loop   |
| `src/backend/wasm/wasmblr-hl.ts:721` | `simdReductionF64()`              | **Yes** (f64x2 madd)   |
| `src/backend/wasm.ts:1410`           | `SIMD_OK_OPS` + JIT codegen       | **Yes** — fused a\*b+c |
| `src/backend/wasm/mega-module.ts`    | `emitExtractedSingleOutputBody()` | **Yes** — mega SIMD    |

### Implementation Plan

**Step 1: Runtime feature detection**

Add Relaxed SIMD detection in `src/backend.ts` or `src/backend/wasm.ts`:

```ts
// Relaxed SIMD validation: compile a minimal module using f32x4.relaxed_madd
function detectRelaxedSimd(): boolean {
  try {
    const bytes = new Uint8Array([
      0x00,
      0x61,
      0x73,
      0x6d, // magic
      0x01,
      0x00,
      0x00,
      0x00, // version
      // ... minimal module with relaxed_madd opcode
    ]);
    new WebAssembly.Module(bytes);
    return true;
  } catch {
    return false;
  }
}
```

Add `relaxedSimd: boolean` to `BackendCapabilities` (`src/backend.ts:20`).

**Step 2: Add `f32x4.relaxed_madd` to wasmblr**

In `src/backend/wasm/wasmblr.ts`, extend `F32x4` class (~line 990):

```ts
// Relaxed SIMD — 0x100 prefix + opcode
relaxed_madd = RELAXED_VECTOR_OP("relaxed_madd", 0xaf, ["v128", "v128", "v128"], "v128");
```

This requires adding the `0xFD` + LEB128 encoding for relaxed SIMD opcodes. The opcode is
`0xFD 0x00AF` for `f32x4.relaxed_madd`.

**Step 3: Use in `simdReductionF32`**

In `src/backend/wasm/wasmblr-hl.ts:594`, when `relaxedSimd` is available:

```ts
// Current: f32x4.mul + f32x4.add (2 instructions)
// With FMA: f32x4.relaxed_madd (1 instruction, same result)
if (caps.relaxedSimd) {
  cg.f32x4.relaxed_madd(); // a * b + acc
} else {
  cg.f32x4.mul();
  cg.f32x4.add();
}
```

**Step 4: Use in JIT SIMD path**

In `src/backend/wasm.ts`, extend `translateExpCoreSimd()` to recognize `Add(Mul(a, b), c)` patterns
and emit `f32x4.relaxed_madd` when available. This is a peephole optimization in the SIMD codegen.

### Source References

| File                              | Line/Symbol                | Relevance                                     |
| --------------------------------- | -------------------------- | --------------------------------------------- |
| `src/backend/wasm/wasmblr.ts`     | `class F32x4` (~line 990)  | Where to add `relaxed_madd` instruction       |
| `src/backend/wasm/wasmblr-hl.ts`  | `simdReductionF32` (594)   | Inner dot-product loop — primary beneficiary  |
| `src/backend/wasm/wasmblr-hl.ts`  | `simdReductionF64` (721)   | f64x2 dot products                            |
| `src/backend/wasm.ts`             | `SIMD_OK_OPS` (1410)       | JIT SIMD eligibility set (needs FMA addition) |
| `src/backend/wasm.ts`             | `translateExpCoreSimd()`   | JIT SIMD codegen (peephole target)            |
| `src/backend.ts`                  | `BackendCapabilities` (20) | Where to add `relaxedSimd` flag               |
| `.github/copilot-instructions.md` | WASM feature opportunities | Listed as "Medium priority"                   |

### Risks

- **Safari doesn't support Relaxed SIMD** (as of early 2026). Must be a runtime-detected fast path
  with non-relaxed fallback. Never hard-depend on it.
- **Numerical differences:** `relaxed_madd` may produce results that differ from separate `mul+add`
  by up to 0.5 ULP (FMA rounds once vs twice). This is acceptable for f32 ML workloads.

### Tests

- New: `test/relaxed-simd.test.ts` — feature detection, correctness, skip if unavailable
- Extend `bench/parallel-wasm.bench.ts` or new `bench/fma.bench.ts` — measure speedup
- Verify Cholesky/TriSolve/matmul numerical results are within tolerance

### Acceptance

- Relaxed SIMD path auto-detected and used when available
- Non-relaxed fallback is identical (existing tests still pass on Safari)
- Measured ≥1.5× speedup on f32 matmul inner loop (via routine bench)

---

## P3: i64 Type in wasmblr

**Priority:** Medium — unlocks proper f64 builtins and simplifies Threefry PRNG.

### Problem

The `CodeGenerator` class (`src/backend/wasm/wasmblr.ts:182`) supports `i32`, `f32`, `f64`, `v128`,
`i32x4`, `f32x4`, `f64x2` — but **no `i64`**. This means:

1. **Threefry PRNG** (`src/backend/wasm/builtins.ts:502`) uses pairs of i32 operations to simulate
   64-bit key schedule. Adding i64 would halve the instruction count for rotations and XORs.
2. **Float64 builtins** (exp/log/sin/cos for f64) require i64 for IEEE 754 bit manipulation
   (extracting exponent, mantissa). Currently these are computed via JS or f32 approximations.
3. **Memory offsets >4GB** — not relevant for browser, but i64 is the WebAssembly standard for
   addresses in memory64 proposal.

### Current i64 Workarounds

The Threefry implementation in `src/backend/wasm/builtins.ts:502` uses 20 `i32.rotl()` calls with
two i32 locals for the state. With native i64, this would be:

```wasm
;; Current (2 × i32):
local.get $x0    ;; low 32 bits
local.get $x1    ;; high 32 bits
;; ... manual rotation across two registers ...

;; With i64:
local.get $x     ;; full 64 bits
i64.rotl         ;; single instruction
```

### Implementation Plan

**Step 1: Add I64 type class**

In `src/backend/wasm/wasmblr.ts`, add `class I64` (~after `class I32`, line ~280):

```ts
class I64 implements Type {
  typeId = 0x7E; // i64
  name = "i64" as const;
  constructor(private cg: CodeGenerator) {}

  const(value: bigint): void { ... }  // 0x42 + signed LEB128(value)
  add = OP("add", 0x7c, ["i64", "i64"], "i64");
  sub = OP("sub", 0x7d, ["i64", "i64"], "i64");
  mul = OP("mul", 0x7e, ["i64", "i64"], "i64");
  and = OP("and", 0x83, ["i64", "i64"], "i64");
  or  = OP("or",  0x84, ["i64", "i64"], "i64");
  xor = OP("xor", 0x85, ["i64", "i64"], "i64");
  shl = OP("shl", 0x86, ["i64", "i64"], "i64");
  shr_s = OP("shr_s", 0x87, ["i64", "i64"], "i64");
  shr_u = OP("shr_u", 0x88, ["i64", "i64"], "i64");
  rotl = OP("rotl", 0x89, ["i64", "i64"], "i64");
  rotr = OP("rotr", 0x8a, ["i64", "i64"], "i64");
  wrap_i32 = OP("wrap_i32", 0xa7, ["i64"], "i32"); // i64 → i32
  extend_i32_s = OP("extend_i32_s", 0xac, ["i32"], "i64"); // i32 → i64 signed
  extend_i32_u = OP("extend_i32_u", 0xad, ["i32"], "i64"); // i32 → i64 unsigned
  reinterpret_f64 = OP("reinterpret_f64", 0xbd, ["f64"], "i64");
}
```

Add `i64: I64` field to `CodeGenerator` and initialize in constructor.

**Step 2: Update `i64.const` encoding**

The `encodeSigned` function in wasmblr.ts encodes LEB128 for i32. Need `encodeSigned64` for i64
which handles BigInt values.

**Step 3: Simplify Threefry**

Rewrite `wasm_threefry2x32()` in `src/backend/wasm/builtins.ts:502` to use i64:

```ts
// Old: 2 i32 locals per state word, manual rotation
// New: 1 i64 local per state word, single i64.rotl
const x = cg.local.declare(cg.i64);
cg.local.get(x);
cg.i64.const(BigInt(rot));
cg.i64.rotl();
```

Split back to two i32 outputs at the end via `i64.wrap_i32` and shift.

**Step 4: Enable f64 builtins**

With i64 available, implement `wasm_exp_f64`, `wasm_log_f64`, `wasm_sin_f64` etc. in
`src/backend/wasm/builtins.ts` using the standard Cephes/FDLIBM algorithms that extract exponent via
`i64.reinterpret_f64` + `i64.shr_u`.

### Source References

| File                                | Line/Symbol                 | Relevance                               |
| ----------------------------------- | --------------------------- | --------------------------------------- |
| `src/backend/wasm/wasmblr.ts`       | `class CodeGenerator` (182) | Where to add i64 field                  |
| `src/backend/wasm/wasmblr.ts`       | `class I32` (~280)          | Pattern to follow for I64               |
| `src/backend/wasm/wasmblr.ts`       | `encodeSigned()` (~325)     | LEB128 encoder (needs 64-bit variant)   |
| `src/backend/wasm/builtins.ts`      | `wasm_threefry2x32()` (502) | Primary beneficiary — PRNG, 20 i32.rotl |
| `src/backend/wasm/builtins.test.ts` | Threefry tests              | Correctness validation                  |
| `src/backend/wasm/wasmblr.test.ts`  | wasmblr unit tests          | Pattern for new i64 tests               |

### Risk

None — i64 is part of WebAssembly 1.0 MVP. Universally supported. No browser compatibility risk.

### Tests

- Extend `src/backend/wasm/wasmblr.test.ts` — i64 arithmetic, rotl, conversions
- Verify Threefry output matches current implementation bit-for-bit (rewrite must be exact)
- `test/random.test.ts` — exercised automatically via random.key/split/uniform

### Acceptance

- `cg.i64` available with arithmetic, bitwise, rotation, conversion ops
- Threefry rewritten and producing identical random streams
- All `test/random.test.ts` tests pass

---

## P4: Conv2d Tuning

**Priority:** Medium — Conv2d is mentioned in copilot-instructions as "not tuned yet."

### Problem

Conv2d currently compiles through the same path as matmul:

```
lax.conv_general_dilated(input, kernel, ...)
  → Primitive.Conv
    → jitRules[Primitive.Conv]    → src/frontend/jit.ts:1827
      → prepareConv() reshapes    → src/frontend/convolution.ts
      → jitRules[Primitive.Dot]   → lowered to Mul+Reduce
```

`prepareConv()` in `convolution.ts` transforms input and kernel into shapes that can be contracted
via `Dot`. The result goes through the same `pipelineSource()` codegen — no convolution-specific
optimization.

### Optimization Opportunities

1. **Tiled matmul (P1) automatically helps** — once P1 is implemented, Conv benefits because Conv
   lowers to Dot. This is the biggest win with zero Conv-specific work.

2. **Im2col avoidance** — `prepareConv()` uses ShapeTracker views to avoid materializing the im2col
   matrix. This is already efficient for memory. The compute bottleneck is the reduction.

3. **Specialized conv shader** — For small kernels (3×3, 5×5), a direct convolution shader with
   unrolled kernel loops can outperform the generic Dot path by exploiting spatial locality. This is
   lower priority since P1 covers the common case.

4. **WASM SIMD for conv** — The JIT SIMD path (`canVectorizeSimd`) currently excludes reduction
   kernels. Extending it to handle the conv reduction inner loop would help. This ties into P2 (FMA)
   — the conv inner loop is a multiply-accumulate.

### Implementation Plan

**Phase 1: Validate P1 impact on conv (no new code needed)**

After P1 (tiled matmul), benchmark Conv2d and measure improvement. If the matmul tiling provides
sufficient speedup, phase 2 may not be needed.

- **Benchmark:** Create `bench/conv2d.bench.ts` with typical sizes (batch=8, 224×224×3→64,
  56×56×64→128, etc.)
- **Measure:** Compare before/after P1 on WebGPU

**Phase 2: Direct conv shader (conditional on Phase 1 results)**

For small kernel sizes (3×3, 5×5), create a specialized WGSL shader that:

1. Reads the kernel weights into registers (unrolled — only 9 or 25 values)
2. Uses shared memory for input tile with halo
3. Each thread computes one output pixel

This would live in `src/backend/webgpu/tiled-conv.ts` and be selected in `jitCompile()` when the
kernel spatial dims are small enough.

### Source References

| File                               | Line/Symbol                 | Relevance                                       |
| ---------------------------------- | --------------------------- | ----------------------------------------------- |
| `src/frontend/jit.ts`              | `Primitive.Conv` (1827)     | Conv JIT rule, lowers through Dot               |
| `src/frontend/convolution.ts`      | `prepareConv()`             | ShapeTracker manipulation for conv→dot lowering |
| `src/frontend/convolution.ts`      | `pool()`, `poolTranspose()` | Pooling operations (same pattern)               |
| `src/library/lax.ts`               | `conv_general_dilated()`    | Public API entry point                          |
| `src/library/nn.ts`                | `conv()`                    | High-level neural network conv wrapper          |
| `test/conv.test.ts`                | Conv2d correctness tests    | Existing tests to validate                      |
| `website/src/routes/bench/conv2d/` | Conv2d benchmark page       | Existing web benchmark                          |
| `.github/copilot-instructions.md`  | "Conv2d not tuned yet"      | Confirms this is a known gap (line 124)         |

### Tests

- New: `bench/conv2d.bench.ts` — standard CNN layer sizes
- `test/conv.test.ts` — existing correctness tests
- `test/nn.test.ts` — integration with nn.conv

### Acceptance

- Conv2d 224×224×3→64 at ≥50% improvement over current (likely achieved by P1 alone)
- All `test/conv.test.ts` and `test/nn.test.ts` pass

---

## P5: WebGPU Subgroups

**Priority:** Low — spec not stable; deferred from TUNER-IMPROVEMENT-PLAN Improvement 6.

### Problem

Large reductions use workgroup shared memory (Improvement 4) with `workgroupBarrier()` tree
reduction. Subgroups would enable `subgroupAdd()` that sums across a SIMD lane (32–64 threads)
without barriers — 2–4× faster for reductions.

### When to Implement

- When `subgroups` feature lands in the WebGPU spec as stable
- When ≥2 major browsers ship it (Chrome + Firefox or Chrome + Safari)
- Current status (Feb 2026): Chrome Canary only, behind flag

### Implementation Plan

Already detailed in `TUNER-IMPROVEMENT-PLAN.md` — Improvement 6 section (line 628). Quick summary:

1. Add `subgroups: boolean` to `BackendCapabilities` (`src/backend.ts:20`)
2. Request `'subgroups'` feature at device creation (`src/backend.ts`)
3. Add subgroup reduction path in `pipelineSource()` (`src/backend/webgpu.ts`) — when
   `caps.subgroups && tune.size.groups > 1`:

   ```wgsl
   enable subgroups;
   let partial = subgroupAdd(acc);
   if (subgroup_invocation_id == 0) {
     shmem[subgroup_id] = partial;
   }
   workgroupBarrier();
   // Thread 0 reduces across subgroup results
   ```

4. Composes with Improvement 4 (shared memory): subgroups replace the inner tree reduction, reducing
   `log2(workgroupSize)` barrier steps to `log2(workgroupSize / subgroupSize)`.

### Source References

| File                        | Line/Symbol                | Relevance                                   |
| --------------------------- | -------------------------- | ------------------------------------------- |
| `TUNER-IMPROVEMENT-PLAN.md` | Improvement 6 (line 628)   | Full detailed plan with code sketches       |
| `src/backend.ts`            | `BackendCapabilities` (20) | Where to add `subgroups` flag               |
| `src/backend/webgpu.ts`     | `pipelineSource()` (1634)  | Shader codegen — where to add subgroup path |
| `src/tuner.ts`              | `tuneWebgpu()`             | Groups splitting logic                      |

### Acceptance

- Feature-gated behind `caps.subgroups`
- Falls back to shared-memory tree reduction when unavailable
- Measured ≥2× speedup for large reductions (≥4096 elements)

---

## P6: Benchmark Validation Suite

**Priority:** Medium — validates P1–P4 impact; catches regressions.

### Problem

Existing benchmarks in `bench/` are ad-hoc. No systematic tracking of:

- GFLOP/s for matmul at various sizes
- Throughput for elementwise SIMD chains
- Conv2d at standard CNN sizes
- Reduction throughput (sum, max) at various sizes

### Implementation Plan

**Step 1: Extend `bench/matmul.bench.ts`**

Add sizes: 128, 256, 512, 1024 (currently only 2048, 4096). Add batched: `[32, 64, 64]`. Report
GFLOP/s: `2 * M * N * K / time_ns`.

**Step 2: Create `bench/conv2d.bench.ts`**

Standard CNN layers:

| Layer     | Input        | Kernel | Output       |
| --------- | ------------ | ------ | ------------ |
| Conv1     | 1×224×224×3  | 7×7    | 1×112×112×64 |
| ResBlock  | 1×56×56×64   | 3×3    | 1×56×56×64   |
| MobileNet | 1×112×112×32 | 3×3 dw | 1×112×112×32 |

**Step 3: Create `bench/simd-elementwise.bench.ts`**

Measure the SIMD f32x4 path vs scalar:

- Pure add chain: `x.add(1).add(2).add(3)` over 1M, 10M elements
- Mixed: `x.mul(y).add(z).sqrt()` over 1M elements
- Broadcast: `x.add(scalar)` over 1M elements

**Step 4: Create `bench/reduction.bench.ts`**

Measure shared-memory reduction speedup:

- `sum(x)` for sizes 256, 1024, 4096, 16384, 65536, 1M
- `max(x)` for same sizes
- Both WASM and WebGPU backends

**Step 5: Results tracking**

Save baseline numbers to `docs/ULTIMATE-BENCHMARKS.md` (already exists). Add a `pnpm bench:all`
script that runs all benchmark files and generates a summary.

### Source References

| File                          | Line/Symbol       | Relevance                      |
| ----------------------------- | ----------------- | ------------------------------ |
| `bench/matmul.bench.ts`       | Existing          | Pattern to follow              |
| `bench/mega-module.bench.ts`  | Existing          | 7 benchmarks, WASM focus       |
| `bench/scatter-add.bench.ts`  | Existing          | Throughput at 1K/10K/100K      |
| `docs/ULTIMATE-BENCHMARKS.md` | Existing          | Results storage                |
| `vitest.config.ts`            | Test/bench config | Benchmark runner configuration |

### Acceptance

- Benchmark suite covers matmul, conv2d, elementwise SIMD, reductions
- GFLOP/s reported for compute-bound benchmarks
- Results baseline documented in `docs/ULTIMATE-BENCHMARKS.md`

---

## Dependency Graph

```
P6 (benchmarks) ←── independent, start anytime
   │
   ├── validates P1, P2, P4

P1 (tiled matmul) ←── highest priority, independent
   │
   └── P4 (conv2d) depends on P1 results (phase 1)

P2 (relaxed SIMD FMA) ←── independent of P1
   │
   └── benefits from P3 (i64 in wasmblr, for f64 FMA path)

P3 (i64 in wasmblr) ←── independent, enables P2 f64 path

P5 (subgroups) ←── blocked on spec stability, implement last
```

**Recommended execution order:**

1. **P6** first — establish baseline numbers
2. **P1** — highest impact (5–10× matmul)
3. **P3** — low risk, unlocks P2's f64 path
4. **P2** — after P3 (FMA for both f32 and f64)
5. **P4** — validate after P1, only do phase 2 if needed
6. **P5** — when spec stabilizes
