# The Generative Compiler Path for High-Performance Shaders

## 1. Overview

The goal is to evolve the `block_map` + `fori_loop` compiler architecture so it automatically
generates optimal WGSL shaders — no hand-written routines. This aligns with our long-term vision:
composable primitives that produce near-optimal code, eventually beating hand-written kernels for
complex fusions (multi-head attention, fused GEMMs with epilogues, etc.).

The compiler must shift from safety-first, fully-general codegen to an aggressive, statically
optimized pipeline. This document lays out the target, the problems we must solve first, the full
set of required optimizations, and the implementation order.

---

## 2. Target Shader (What the Compiler Must Generate)

Based on Nuss-and-Bolts (>1 TFLOP/s, M2 Pro), jott.live (680 GFLOP/s, M1), SitePoint, and HN
discussion on WebGPU compute ceilings.

### Example: Optimal 64×64 tiled matmul, K-tiles of 16

```wgsl
// Tile sizes as const — enables Tint IR range analysis to elide bounds checks
const BLOCK_R: u32 = 64u;
const BLOCK_C: u32 = 64u;
const BLOCK_K: u32 = 16u;
const THREAD_TILE_R: u32 = 8u;
const THREAD_TILE_C: u32 = 8u;

// Shmem with +1 padding to avoid bank conflicts (see O11)
var<workgroup> tile_A: array<array<f16, BLOCK_K + 1u>, BLOCK_R>;
var<workgroup> tile_B: array<array<f16, BLOCK_C + 1u>, BLOCK_K>;

// 64 threads, each computes 8×8 = 64 outputs
@compute @workgroup_size(8, 8)
fn main(...) {
  // Thread-local accumulators (register-resident)
  var acc: array<vec4<f16>, 16>;  // 8×8 = 16 vec4s
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f16>(0); }

  let block_row = wg_id.x * BLOCK_R;
  let block_col = wg_id.y * BLOCK_C;

  for (var k_tile = 0u; k_tile < K / BLOCK_K; k_tile++) {
    // Phase 1: Cooperative tile load (all 64 threads load 4 elements each)
    // 64 threads × 4 elements = 256 elements per tile (16×16)
    let a_base = block_row * K + k_tile * 16u;
    for (var i = 0u; i < 4u; i++) {
      let idx = tid * 4u + i;
      tile_A[idx / 16u][idx % 16u] = A[a_base + (idx / 16u) * K + idx % 16u];
    }
    // ... same for tile_B, transposed ...

    workgroupBarrier();  // ← Barrier 1: tile load complete

    // Phase 2: Per-thread compute over local 8×8 sub-tile
    @unroll
    for (var kk = 0u; kk < BLOCK_K; kk++) {     // @unroll hint for DXC/MSL backend
      let a_vec = vec4<f16>(tile_A[local_row*8+0][kk], ..., tile_A[local_row*8+3][kk]);
      let b_vec = vec4<f16>(tile_B[kk][local_col*8+0], ..., tile_B[kk][local_col*8+3]);
      acc[0] += a_vec.x * b_vec;  // outer product accumulation
      acc[1] += a_vec.y * b_vec;
      // ... 8 rows × 1 vec4 = 8 FMAs (or 8 mul+add)
    }

    workgroupBarrier();  // ← Barrier 2: ready for next tile
  }

  // Write 8×8 outputs to global memory (vec4 stores)
  for (var r = 0u; r < 8u; r++) {
    // 2 vec4 stores per row = 8 elements
    result[(block_row + local_row*8+r) * N + block_col + local_col*8] = acc[r*2];
    result[... + 4] = acc[r*2+1];
  }
}
```

**Properties the compiler must achieve:**

1. Exactly 2 `workgroupBarrier()` per K-tile
2. `var<private>` accumulators, not `var<workgroup>`
3. `vec4<f16>` loads/stores from shared memory (16-byte aligned)
4. `@unroll` on innermost loops (helps DXC/MSL register allocation)
5. `const` tile sizes (enables Tint IR range analysis to elide bounds checks)
6. No `select()`, no `min(max())`, no modular arithmetic in the inner loop
7. Cooperative tile loading: all threads participate in loading both tiles
8. Shmem arrays padded by +1 on inner dimension to avoid bank conflicts
9. `let` bindings for intermediates (helps Tint SSA optimization)

---

## 2b. Chrome Tint IR — What the Backend Compiler Does (and Doesn't Do)

As of Chrome 141+, Tint uses an IR-based compiler. Understanding its behavior is critical because
Tint is **preservative** — it faithfully translates your WGSL structure to the backend (HLSL/DXC on
Windows, MSL on Mac, SPIR-V on Linux/Android). It will NOT restructure loops, discover tiling, or
auto-vectorize. What we emit is what runs.

### What Tint IR optimizes (work WITH these):

1. **Integer Range Analysis:** Tint attempts to prove array indices are in-bounds to elide WebGPU's
   mandatory robustness checks. When it sees `i < TILE_SIZE` and `TILE_SIZE` is a `const`, it can
   often remove the bounds-check branch entirely. → **Our codegen must use `const` for tile sizes.**
   Complex `%`/`/` indexing in hot loops breaks range analysis.

2. **SSA optimization on `let` bindings:** Tint IR handles `let` declarations well for SSA form.
   Using `let` for intermediate values (vs `var` unnecessarily) helps. → **Our codegen should emit
   `let` for all non-mutated intermediates.**

3. **`@unroll` loop attribute:** Tint passes `@unroll` to the backend (DXC/MSL), which expands loops
   and enables better register allocation. This is _much_ simpler than manually unrolling at our IR
   level. → **Our codegen should emit `@unroll` on inner loops instead of expanding them.**

4. **`workgroupUniformLoad`:** (Chrome 137+) When all threads load the same shmem value, this
   built-in enables hardware broadcast optimization. → **Our codegen can use this for uniform shmem
   reads (after P0 makes most of them `let`).**

### What Tint IR does NOT do (we must generate explicitly):

1. **No auto-vectorization:** Four scalar `f16` loads → four load instructions. We must emit
   `vec4<f16>` / `vec2<f16>` explicitly.

2. **No loop restructuring:** Tint won't tile, fuse, or split loops. The loop structure we emit is
   the loop structure that runs.

3. **No bank-conflict resolution:** Tint doesn't pad shared memory arrays. We must emit
   `array<f16, TILE_SIZE + 1>` ourselves to avoid bank conflicts.

4. **No register promotion:** If a value lives in `var<workgroup>`, Tint won't promote it to
   registers. Only `var<private>` or `let` values get register-allocated.

### The Subgroup Frontier (Chrome 144+)

The `subgroups` extension exposes `subgroup_matrix` types that map directly to hardware Tensor Cores
/ Matrix Units (NVIDIA WMMA, Intel XMX, Apple AMX). This bypasses shared memory entirely —
$5{-}10\times$ faster than standard tiled matmul. This is the endgame for matmul performance but
requires the `subgroups` feature to be stable and widely available.

### Impact on Our Plan

| Review Point            | Plan Section    | Status                       |
| ----------------------- | --------------- | ---------------------------- |
| `const` tile sizes      | P2, O2, codegen | Addressed in O2, O5 ✅       |
| `@unroll` attribute     | O5              | Addressed in O5 ✅           |
| `vec4` explicit         | O3              | Addressed in O3 ✅           |
| `let` for intermediates | P0, codegen     | Addressed in P0a ✅          |
| Bank conflicts          | Shmem alloc     | Addressed in O11 ✅          |
| `workgroupUniformLoad`  | P0              | Addressed in P0a-fallback ✅ |
| Subgroup matrix         | O12             | Addressed in O12 ✅          |
| No loop restructuring   | All             | Already assumed ✅           |

---

## 3. Current Architecture — What Goes Wrong

### 3.1 The Generated Shader (256×256 f16, 16×16 tiles)

```wgsl
@compute @workgroup_size(16, 16)  // 256 threads, 1 output per thread
fn main(...) {
  shmem_3[tidx] = in0[tidx];  // init carry
  workgroupBarrier();

  for (var fl0_i: i32 = 0; fl0_i < 16; fl0_i++) {

    // ┌── PROBLEM: scalar via shmem ──────────────────────────────────────┐
    if (tidx < 1u) {
      fl0_s4[tidx] = select(0, i32(fl0_i), true) * 16;  // kIdx = i * Bk
    }
    workgroupBarrier();  // ← wasted barrier #1
    // └───────────────────────────────────────────────────────────────────┘

    // ┌── PROBLEM: select(…, true) + complex index arithmetic ────────────┐
    fl0_s5[tidx] = select(f16(0),
      in1[in_base_0 + (alu0 * 256) +
        (((alu0 * 256) + (min(max(select(0, fl0_s4[0], true), 0), 240)
          + (gidx % 16))) % 256)],
      true);
    workgroupBarrier();  // ← barrier #2 (tile A load)
    // └───────────────────────────────────────────────────────────────────┘

    fl0_s6[tidx] = select(f16(0), in2[...complex...], true);
    workgroupBarrier();  // ← barrier #3 (tile B load)

    // ┌── PROBLEM: scalar reduction, no register tiling ──────────────────┐
    {
      var fl0_s3_acc: f32 = 0;
      for (var ridx: i32 = 0; ridx < 16; ridx++) {  // NOT unrolled
        fl0_s3_acc += f32(
          select(f16(0), fl0_s5[row*16 + ridx], true)  // PROBLEM: select inside inner loop
          * select(f16(0), fl0_s6[ridx*16 + col], true));
      }
      shmem_3[tidx] = f16(select(f16(0), shmem_3[gidx % 256], true) + f16(fl0_s3_acc));
    }
    // implicit barrier #4 at loop top
    // └───────────────────────────────────────────────────────────────────┘
  }
}
```

### 3.2 Root Cause Analysis

**Problem 1: Scalar `mul(k, Bk)` → size-1 kernel → shmem `fl0_s4` + barrier**

In `tiledMatmul`, `core.mul(k, arrayFn(Bk))` creates a scalar i32 result of size 1. The JIT compiles
this as a kernel with `size=1`. The block-map codegen sees `bKernelSize (1) < blockSize (256)` and
emits `if (tidx < 1u) { fl0_s4[tidx] = ...; }`. Subsequent steps read `fl0_s4[0]`, creating a shmem
dependency → barrier.

**Root cause in the code:**

- [block-map.ts line ~1632](src/backend/webgpu/block-map.ts):
  `const needSizeGuard = bKernelSize < blockSize`
- The `fl0_s4` shmem is allocated because the JIT malloc'd a size-1 buffer for the scalar result
- It's a **uniform value** across all threads (loop iteration × constant) that should be a register

**Problem 2: `select(…, true)` on every shmem read**

The inner reduction loop reads `fl0_s5[row*16 + ridx]` wrapped in `select(f16(0), ..., true)`. These
come from `accessorGlobal()` in [alu.ts line 1778](src/alu.ts) which unconditionally wraps every
`GlobalIndex` in `AluExp.where(valid, read, 0)`. For shmem reads inside the fori_loop body, `valid`
was already checked at the block boundary level — the inner reads are always in-bounds.

But the valid flag from `ShapeTracker.toAluExp()` propagates the `dynamic_slice` mask: the mask
exists because `dynamic_slice` sets bounds `[0, dim - sliceSize]` on the start indices. Since
`fl0_s4` contains a dynamically-clamped index, the ShapeTracker can't statically prove the mask is
always satisfied → `valid = true` at runtime but can't be eliminated at compile time.

**Problem 3: `min(max(…))` clamping chains**

`DynamicSlice` in [jit.ts line 2393](src/frontend/jit.ts) emits:

```ts
start = AluExp.max(start, AluExp.i32(0));
start = AluExp.min(start, AluExp.i32(maxStart));
```

This is correct in general, but when the input is already padded to a tile-size multiple, the start
is always `i * Bk` where `i ∈ [0, numKTiles)` and `maxStart = K - Bk` = `(numKTiles-1)*Bk`. The
clamp is provably a no-op. But the compiler doesn't know this.

**Problem 4: A and B tile loads are separate barrier-synchronized steps**

The JIT compiles the fori_loop body as:

1. `fl0_s4`: scalar `kIdx` (size 1) → shmem + barrier
2. `fl0_s5`: load A tile via `dynamic_slice` (size 256) → shmem + barrier
3. `fl0_s6`: load B tile via `dynamic_slice` (size 256) → shmem + barrier
4. `fl0_s7`: reduction (reads fl0_s5 and fl0_s6) → uses barrier at loop-top

Steps 2 and 3 are **independent** — they read from different global inputs and write to different
shmem. But the barrier-placement logic in
[block-map.ts lines 662–694](src/backend/webgpu/block-map.ts) inserts a barrier between them because
the current analysis is purely sequential: "does step N read what step N-1 wrote?" It doesn't merge
independent writes into a single barrier-delimited phase.

(The actual barrier logic is at [block-map.ts lines 650–694](src/backend/webgpu/block-map.ts), using
`stepWrites`/`stepReads` maps and `needsBarrierBefore` set.)

**Problem 5: 1 output per thread — no register tiling**

`@workgroup_size(16, 16)` = 256 threads for a 16×16 block = 1 output per thread. The entire
accumulator lives in `shmem_3[tidx]` (shared memory), requiring a barrier every time the carry is
read-modify-written. Reference implementations compute 4×4 to 8×8 outputs per thread in
`var<private>` registers, reducing barrier cost by 16–64×.

**Problem 6: Modular index arithmetic**

The shmem reads use expressions like `fl0_s5[(((gidx / 16) % 16) * 16) + ridx]`. This is because the
`dot` primitive's AluExp lowers through the standard multi-dim index unraveling path. For a 16×16
shmem tile with known contiguous layout, this should simplify to `fl0_s5[tidx_0 * 16 + ridx]`. The
modular arithmetic comes from generic `unravelAlu` which doesn't specialize for the common case of
contiguous tiles where `gidx / stride % dim` = `tidx_k`.

---

## 4. Architectural Principle: Correctness by Construction

### The Insight

Instead of generating bounds checks at the WGSL level and then trying to prove they're redundant,
**structure the Jaxpr so that all accesses are unconditionally valid by construction**. If the IR
guarantees correctness, the lowest-level codegen never needs checks — no `select`, no `min/max`, no
branch, no mask. Safety is enforced at the API/primitive boundary (cheap, once), not in the inner
loop (expensive, every iteration, every thread).

This is analogous to Rust's ownership model: prove safety at compile time so the runtime has zero
overhead. Here, we prove access safety at the Jaxpr/primitive level so the WGSL has zero overhead.

### Current Architecture (Three Layers of Redundant Checks)

Currently, three independent mechanisms generate bounds checks:

**Layer 1 — Block boundary (`hasBoundary` + `valid` flag):** When array dimensions aren't divisible
by `blockShape`, partial blocks have invalid threads. Block-map codegen emits
`let valid: bool = ...` and wraps every global read/write.

- **Source:** [block-map.ts line ~108](src/backend/webgpu/block-map.ts) —
  `if (dim % blockShape[g] !== 0) hasBoundary = true`

**Layer 2 — DynamicSlice clamping (`min/max` on start indices):** `DynamicSlice` unconditionally
clamps start indices to `[0, dim - sliceSize]`.

- **Source:** [jit.ts lines 2393-2394](src/frontend/jit.ts) —
  `start = AluExp.max(start, 0); start = AluExp.min(start, maxStart);`

**Layer 3 — accessorGlobal `Where` wrapping (`select(0, read, valid)`):** `accessorGlobal()`
unconditionally wraps every `GlobalIndex` read in a validity check.

- **Source:** [alu.ts line 1778](src/alu.ts) —
  `return AluExp.where(valid, AluExp.globalIndex(...), AluExp.const(dtype, 0));`
- When the ShapeTracker has no mask (the common case inside block_map bodies for non-padded inputs),
  `valid` evaluates to constant `true` → `select(..., true)` in WGSL.
- When the input was `pad()`'d, the ShapeTracker carries a mask.

### How to Eliminate All Three

**Layer 1 (block boundary) — already solved for matmul:** `tiledMatmul` pads M, N, K to tile-size
multiples before calling `blockMap`. So `dim % blockShape[g] === 0` → `hasBoundary = false` → no
`valid` flag emitted. ✅

**Layer 2 (DynamicSlice clamping) — add unchecked mode:** When the caller guarantees in-bounds
access (because they padded the input and computed the loop bounds correctly), `DynamicSlice` should
not emit `min/max` clamping.

Concrete approach — add `Primitive.UncheckedDynamicSlice`:

```ts
// In jit.ts lowering rules:
[Primitive.UncheckedDynamicSlice](exps, avals, { sliceSizes }) {
  const operandExp = exps[0];
  const startExps = exps.slice(1);
  const operandShape = avals[0].shape as number[];
  const outIndices = unravelAlu(sliceSizes, AluVar.gidx);

  // NO clamping — caller guarantees in-bounds
  const readIndices: AluExp[] = [];
  for (let k = 0; k < operandShape.length; k++) {
    const start = AluExp.cast(DType.Int32, startExps[k]);
    readIndices.push(AluExp.add(start, outIndices[k]));
  }

  const [index] = ShapeTracker.fromShape(operandShape).toAluExp(readIndices);
  return { exp: [operandExp.substitute({ gidx: index })] };
},
```

The user-facing `lax.dynamicSlice` keeps the safe clamped version. `tiledMatmul` (and other
performance-critical block_map users) uses the unchecked variant internally, since it guarantees
correctness through padding + loop bounds.

Alternative approach — `unchecked` flag on existing `DynamicSlice`:

```ts
[Primitive.DynamicSlice](exps, avals, { sliceSizes, unchecked }) {
  // ... if (!unchecked) { add min/max clamping }
}
```

The flag approach is simpler (no new primitive), but a separate primitive is arguably cleaner
because it makes the contract explicit in the IR — a downstream pass can see that a
`UncheckedDynamicSlice` was used and knows no bounds checking is present.

**Layer 3 (accessorGlobal Where wrapping) — trivial fix, benefits BOTH lanes:** `accessorGlobal`
already computes `valid` from the ShapeTracker. When there is no mask, `valid` is the literal
constant `AluExp.bool(true)`. The fix: check for this and skip the `Where`:

```ts
export function accessorGlobal(dtype, gid, st, indices) {
  const [index, valid] = st.toAluExp(indices);
  const [, len] = st.views[0].dataRange();
  const read = AluExp.globalIndex(dtype, gid, len, index);
  // Skip Where when valid is provably true (no mask in ShapeTracker)
  // Note: AluExp.bool(true) stores Number(true) = 1, not JS `true`
  if (valid.op === AluOp.Const && valid.arg === 1) return read;
  return AluExp.where(valid, read, AluExp.const(dtype, 0));
}
```

This is a one-line fix that eliminates every `select(..., true)` in the generated shader. It
benefits ALL generated shaders (not just the fast lane) — any ShapeTracker without a mask produces
cleaner code. Do this first, regardless of any other changes.

### Correctness Contract: Two-Lane IR

The correctness-by-construction principle creates a **two-lane** architecture:

**Safe lane (default):** All user-facing primitives validate inputs and emit full bounds checks.
`lax.dynamicSlice` clamps, `lax.pad` carries masks, `accessorGlobal` wraps with `select`. This is
the default for all user code and third-party callers. No performance cost is ever traded for
correctness in this lane.

**Fast lane (internal, opt-in):** Performance-critical internal code (e.g., `tiledMatmul`) uses
unchecked variants that omit redundant checks. This lane is NEVER exposed in the public API. Entry
requires structurally proving that all accesses are in-bounds (via padding, loop bounds, tile
alignment). The proof is documented at each call site.

| Layer            | Safe Lane                            | Fast Lane                               |
| ---------------- | ------------------------------------ | --------------------------------------- |
| **User API**     | Validate shapes, dtypes              | N/A — not exposed                       |
| **Library code** | `lax.dynamicSlice` (clamped)         | `uncheckedDynamicSlice` + proof comment |
| **Pad**          | `core.pad` (mask-based, zero cost)   | `padConcrete` (real buffer, no mask)    |
| **WGSL codegen** | `select(0, read, valid)` when masked | Raw read (no mask, no select)           |

**Escape hatch:** If a transform produces indices whose in-bounds property can't be verified, the
fast lane falls back to the safe lane automatically (e.g., transpose of `UncheckedDynamicSlice` →
safe `DynamicUpdateSlice`). Safety is never silently lost.

---

## 5. Problems That Must Be Solved First (Prerequisites)

With the correctness-by-construction principle established, these prerequisites become cleaner.

### P0: The Scalar Kernel Problem (Blocks All Other Improvements)

**Why it's P0:** Even if we fix barriers, bounds checks, and indexing, a scalar kernel step always
creates a shmem intermediate + barrier. This structural issue must be resolved before anything else.

**Root cause:** `core.mul(k, arrayFn(Bk))` in the fori_loop body creates a standalone i32 kernel of
size 1. The JIT allocates a buffer, block-map allocates shmem, the codegen emits a guarded write and
a barrier.

**Two solutions (complementary):**

**P0a — Inline scalar expressions into consumers (compiler-level):** During body step analysis,
detect kernels with `size === 1` whose output is only used as an index argument to `dynamic_slice`.
Instead of emitting shmem + guarded write, resolve the expression directly in the consumer's
`resolveGlobalIndex` callback as a `let` binding:

```wgsl
let kIdx: i32 = fl0_i * 16;  // was: fl0_s4[0] via shmem
```

This eliminates the shmem allocation, the `if (tidx < 1u)` guard, and the barrier.

Using `let` (not `var`) is deliberate: Tint IR handles `let` bindings efficiently for SSA
optimization, and the value is thread-uniform so no `var<workgroup>` is needed (Tint won't promote
workgroup memory to registers — only `let`/`var<private>` values get register-allocated).

**P0a-fallback — `workgroupUniformLoad` for non-inlineable uniform reads:** If a size-1 kernel's
expression is too complex to inline (e.g., depends on multiple shmem reads), emit
`let kIdx = workgroupUniformLoad(&fl0_s4[0u]);` instead of a raw shmem read. This still requires the
shmem + barrier, but tells the hardware to use broadcast mechanisms (available since Chrome 137).
P0a is strictly better when applicable; `workgroupUniformLoad` is the fallback for when shmem is
still needed but the read is uniform across all threads.

**Capability gating:** `workgroupUniformLoad` is not universally available (missing in older Safari
and some older browser versions). The codegen must feature-check at device init and fall back to
plain `shmem[0]` reads when unavailable. Store the capability in the backend's feature flags
(alongside `shader-f16`, etc.).

**P0b — Uniformity detection (general solution for future):** Classify each body kernel by whether
all threads produce the same value:

- All inputs are either loop-uniform (loop counter, constants) or thread-invariant
- If uniform, emit as a `let` binding, never as shmem

P0a is the pragmatic fix. P0b is the principled architecture for the long term.

### P1: Barrier Phase Merging (Blocks 2-Barrier Goal)

Even after P0 (removing the scalar barrier), we still have 3 barriers: A-load, B-load, compute. The
optimal is 2.

**Root cause:** The barrier analysis walks steps sequentially. Step 2 (B-load) doesn't read anything
that step 1 (A-load) wrote, but the current code inserts a barrier before every step that reads any
preceding shmem. It doesn't recognize that two **independent writers** can be merged into one
barrier phase.

**Solution — Phase-based barrier scheduling:** Instead of "does step N read from step <N?", build a
dataflow graph of shmem read/write deps:

1. Group consecutive write-only steps (steps that only write to shmem, not read from other shmem
   written in this iteration) into a **Load Phase**
2. Insert exactly one barrier after the load phase
3. Group subsequent read-then-write steps into a **Compute Phase**
4. Insert exactly one barrier after the compute phase (for the next iteration)

For the matmul body: `kIdx(scalar) → loadA → loadB` are all independent writers → 1 barrier. Then
`dot+add` reads A+B → 1 barrier. Total: 2 per K-tile.

**Formal dependency model:** Represent each step as a node with `reads: Set<shmemId>` and
`writes: Set<shmemId>`. Two consecutive steps can share a barrier phase if
`step[N].reads ∩ phase.writes === ∅` (N doesn't read what this phase wrote). A barrier is required
only when a step reads a shmemId that a preceding step in the current phase wrote. This graph-based
model generalizes beyond matmul — when more primitives enter block_map (conv2d, attention), the
scheduler correctly handles arbitrary dependency patterns without hand-tuning.

### P2: Bounds Check Elimination (Correctness by Construction)

Instead of range analysis to prove checks redundant after the fact, **prevent the checks from being
generated in the first place**.

Three changes, applied in order:

**P2a — `accessorGlobal`: skip `Where` when `valid` is constant `true` (trivial fix):**

```ts
// AluExp.bool(true) stores Number(true) = 1, not JS boolean `true`
if (valid.op === AluOp.Const && valid.arg === 1) return read;
```

This eliminates every `select(..., true)` in any shader where the ShapeTracker has no mask. One line
of code, massive impact on generated code quality. Do this first.

**P2b — Unchecked DynamicSlice for performance-critical paths:** Add
`Primitive.UncheckedDynamicSlice` (or `unchecked` flag on `DynamicSlice`). The JIT lowering omits
`min/max` clamping. Used only when the caller structurally guarantees in-bounds access.

`tiledMatmul` switches from `lax.dynamicSlice` to the unchecked variant. The guarantee:

- Input padded to tile multiples → `operandShape[k] % sliceSizes[k] === 0`
- Loop counter `k ∈ [0, numKTiles)` → `start = k * Bk ∈ [0, Kpad - Bk]` = `[0, maxStart]`
- Therefore clamping is a no-op

Other callers continue using safe `dynamicSlice` with clamping. No behavior change for them.

**Debug-mode validation:** The unchecked variant MUST include a debug assertion that fires when
`setDebug(level >= 2)`:

```ts
if (debugLevel >= 2) {
  // Validated at trace time or JIT time, NOT in the hot shader path
  for (let k = 0; k < operandShape.length; k++) {
    assert(
      sliceSizes[k] <= operandShape[k],
      `UncheckedDynamicSlice: slice[${k}]=${sliceSizes[k]} > shape[${k}]=${operandShape[k]}`,
    );
  }
}
```

This catches incorrect usage during development without impacting production performance.

**Transform rules:** `UncheckedDynamicSlice` needs JVP, transpose, and vmap rules:

- **JVP:** Tangent is `UncheckedDynamicSlice` of the tangent operand (same indices, same unchecked
  guarantee — the indices don't change, so the safety proof carries through).
- **Transpose (adjoint):** Produces `UncheckedDynamicUpdateSlice` (or safe DUS — the adjoint writes
  to a zero buffer at the same indices, which is always in-bounds by the same proof).
- **Vmap:** Same as DynamicSlice vmap rule — batches the slice along the batched axis. The unchecked
  property is preserved because each batch element has the same tile alignment.

Since `tiledMatmul` is the only internal consumer, these transforms are exercised through
`grad(tiledMatmul)` and `vmap(tiledMatmul)`. The correctness guarantee is: if the forward pass
indices are in-bounds, the transform-derived indices are also in-bounds (they use the same index
expressions). If this invariant ever breaks (e.g., a new transform that modifies indices), the
transform should fall back to safe `DynamicSlice` — safety over performance.

**Composed transform safety:** Composition paths like `jit(grad(vmap(tiledMatmul)))` route through
multiple transform layers. Each transform must independently decide whether to preserve or downgrade
the unchecked property. The rule: if a transform can prove it preserves the index expression
structure (same indices, same bounds), keep unchecked. If it introduces new index computations
(e.g., a hypothetical transform that changes iteration order), downgrade to safe `DynamicSlice`.
Enforcement: the `UncheckedDynamicSlice` JVP/transpose/vmap rules are the ONLY places that propagate
`unchecked` — any unrecognized transform hits the default "unknown primitive" path which does NOT
propagate it.

**P2c — Pad mask elimination for tile-aligned block_map inputs:**

**Key fact: `core.pad` does NOT write zeros.** It is a pure view transform:

```ts
// jit.ts: Pad is reshapeJit — just transforms the ShapeTracker
[Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),

// shape.ts: View.pad() adds a mask to the view via #unsafeResize
pad(arg) {
  const mask = arg.map(([b, _e], i) => [b, this.shape[i] + b]);
  return this.#unsafeResize(zvarg, mask);
}
```

No new buffer is allocated. The mask tells `accessorGlobal` to return 0 for out-of-bounds positions
via `select(0, read, valid_mask)`. This is correct but slow — every access to a padded input carries
a per-element validity check in the generated shader.

When `tiledMatmul` pads a 250×256 matrix to 256×256 with `core.pad(A, aPad)`, the padded rows have
no backing data — they exist only as a mask on the original 250×256 buffer. Every read from the
padded A generates `select(f16(0), A[idx], valid_mask)` in WGSL, which correctly returns 0 for the 6
padded rows but adds a branch to every element access.

**For tile-aligned inputs** (e.g., 256×256 with tile=16): no padding is applied, no mask exists,
`valid` is constant `true` → P2a eliminates all `select`. No further work needed.

**For non-aligned inputs** (e.g., 250×256 with tile=16): `core.pad` adds a mask →
`valid_mask ≠ true` → P2a cannot help. The `select` is semantically necessary because blocks may
straddle the original/padded boundary.

**Solution — Concrete padding (`padConcrete`):** Replace `core.pad` in `tiledMatmul` with a concrete
allocation approach:

```ts
function padConcrete(x: Array, padWidths: Record<number, [number, number]>): Array {
  // Allocate zero-filled buffer at padded size, copy x into it
  using zeros = fullLike(x, 0, paddedShape);
  return dynamicUpdateSlice(zeros, x, 0, 0); // or multi-axis variant
}
```

The result is a real buffer with actual zeros at padded positions. Its ShapeTracker has **no mask**
→ `accessorGlobal` returns raw reads → P2a eliminates all `select` → zero overhead.

**Tradeoff:** Concrete padding allocates extra memory (the full padded buffer) and requires a copy.
For the matmul case this is negligible: a 250×256→256×256 pad is ~1% extra memory and a single
memcpy before the expensive matmul computation. The codegen savings far outweigh it.

**Heuristic gating:** `padConcrete` should only be used when the copy cost is negligible relative to
the computation. Gate on: `padded_elements / original_elements < 1.25` (≤25% overhead) AND
`original_elements > 1024` (not worth materializing for tiny matrices). For small matrices where
padding overhead is proportionally large, fall back to mask-based `core.pad` — the `select` overhead
is acceptable when the overall computation is cheap.

**As a materialization pass:** Rather than tying `padConcrete` directly to `tiledMatmul`, implement
it as a general "materialize masked views" pass that runs on entry to any fast-lane block_map
kernel. Any input whose ShapeTracker carries a pad mask and passes the heuristic gate gets
materialized. This keeps the optimization composable — conv2d and attention benefit without
per-primitive changes.

**JIT recycling interaction:** Inside a JIT trace, the materialized buffer is a temporary allocation
that feeds the block_map kernel. The JIT's `recycleBuffers()` pass should alias it with later
same-size frees. Verify this doesn't cause peak-memory bloat for very large matrices — if padding a
4096×4000 to 4096×4096 allocates 64MB of zeros, the peak live set grows by that amount until the
padded buffer is freed after the matmul.

**Important:** The user-facing `lax.pad` retains the mask-based approach — it's correct, general,
and composes with all transforms. `padConcrete` is an internal optimization used only in
performance-critical paths where the caller controls the subsequent access pattern.

---

## 6. The Core Optimizations (O1–O5)

After solving P0–P2, these optimizations bring the generated shader to competitive territory.

### O1: Barrier Phase Scheduling (extends P1)

**After P0+P1 are done, the fori_loop body has exactly 2 barriers per K-tile.**

Beyond phase merging, extend the scheduler to:

- Detect when a shmem write in iteration N+1 doesn't conflict with reads in iteration N
  (double-buffering opportunity — future work)
- Merge the init-carry barrier with the first loop iteration's load phase when possible

### O2: Shmem Index Simplification

**Problem:** The inner reduction reads `fl0_s5[(((gidx / 16) % 16) * 16) + ridx]`.

**Root cause:** The `dot` primitive's expression is built from generic
`unravelAlu(sliceSizes, gidx)` which decomposes the flat thread index into multi-dimensional
coordinates using division and modulo. For a [`Br`, `Bk`] shmem tile read as row-major, the flat
index IS the correct address, but the expression tree doesn't know the shmem layout matches the
access pattern.

**Why this is critical for Chrome:** Tint IR's integer range analysis works best with simple linear
expressions (`base + stride * idx` where `idx < const_bound`). Complex modulo/division chains
(`gidx / 16 % 16 * 16 + ridx`) break range analysis → Tint can't prove the access is in-bounds →
robustness bounds checks remain in the final backend code. Simplifying indices is therefore a
**double win**: fewer ALU ops AND fewer Tint-inserted bounds checks.

**Solution — ShapeTracker contiguity detection:**

When the ShapeTracker for a shmem intermediate is contiguous row-major (strides = `[Bk, 1]` for
shape `[Br, Bk]`), the flat index `tidx_0 * Bk + tidx_1` equals just `gidx` (when
`blockShape[1] === Bk`). The compiler should detect this and emit `fl0_s5[gidx]` or, when the
reduction axis differs, `fl0_s5[tidx_0 * reductionSize + ridx]` directly.

Concretely:

1. In `createGen`, when resolving a shmem read with a known-contiguous ShapeTracker, short-circuit
   the index computation to use the minimal form
2. For the `ridx` variable substitution in reductions, resolve the multi-dim index to a simple 2D
   formula using the known shmem dimensions
3. Emit tile sizes as `const` declarations in WGSL so Tint can use them for range analysis

### O3: vec4 Vectorized Memory Access

**After O2** cleans up indexing, the next multiplier is vectorized loads.

**Why this is mandatory (not optional):** Tint IR does NOT auto-vectorize. Four scalar `f16` loads
produce four separate load instructions. Emitting `vec4<f16>` maps to wide memory bus instructions
(e.g., `LDG.E.128` on NVIDIA, 128-bit load on Apple). This is a 4× reduction in load instruction
count and full utilization of the memory bus.

**Where it matters:**

- Cooperative tile loading: instead of each thread loading 1 scalar, each thread loads a `vec4` (4
  elements), reducing load instructions by 4× and using full memory bus width
- Inner reduction: accumulate with `vec4` dot products instead of scalar multiply-add
- Global output writes: `vec4` stores reduce store instruction count

**Implementation approach:**

Add a vectorization annotation to shmem entries and kernel loads:

1. During shmem allocation analysis, detect when an array's innermost dimension is ≥ 4 and
   contiguous → mark as `vec4`-eligible
2. In the codegen load phase, emit `vec4<f16>` reads with `tidx * 4` addressing
3. In the reduction loop, accumulate using `dot(a_vec, b_vec)` or component-wise FMA
4. On output write, emit `vec4` stores

**Constraint:** Vec4 requires the innermost dimension to be a multiple of 4. Our tile sizes (16,
32, 64) always satisfy this. The compiler should verify this statically.

**Alignment:** Storage buffers must be 16-byte aligned for `vec4<f32>` (or 8-byte for `vec4<f16>`).
WebGPU's `minStorageBufferOffsetAlignment` (256 bytes) already guarantees this for buffer starts.
For sub-buffer offsets within tiles, the tile size being a multiple of 4 ensures alignment.

**Critical WGSL typing constraint:** Constructing `vec4<f16>(arr[i], arr[i+1], arr[i+2], arr[i+3])`
from a scalar `array<f16>` still produces 4 discrete scalar loads in the compiled shader — Tint will
not merge them. To get true wide memory transactions, the buffer must be **declared** as
`array<vec4<f16>>`. This creates a tension with O11 (bank conflict padding): a stride of 17 scalars
cannot be cleanly stored in an array of `vec4`s.

**Resolution:** Use `array<vec4<f16>>` typing for the **global→shmem cooperative load phase** (this
is the bandwidth bottleneck — global memory is ~20× slower than shmem). For shmem storage itself,
keep scalar `array<f16, TILE+1>` with +1 bank padding and read with scalar access — shmem→register
reads are fast enough that scalar access is acceptable. Alternatively, pad the inner dimension to a
multiple of 4+pad (e.g., 20 instead of 17 for a 16-wide tile) to align vec4 access even in shmem.
The codegen should support both strategies and select based on whether the workload is
global-memory-bound or compute-bound.

### O4: Register Tiling (Thread Coarsening)

This is the single biggest performance lever: 16–64× barrier amortization.

**Current:** `blockShape=[16,16]`, `@workgroup_size(16,16)`, 1 output/thread. Accumulator in shmem.

**Target:** `blockShape=[64,64]`, `@workgroup_size(8,8)`, 64 outputs/thread. Accumulator in
`var<private>`.

**Implementation approach:**

Add `threadTile` parameter to `blockMap`:

```ts
blockMap(body, inputs, {
  blockShape: [64, 64],
  threadTile: [8, 8],  // NEW: each thread handles 8×8 outputs
  ...
});
```

**Codegen changes:**

The codegen must distinguish between:

- **Workgroup-cooperative operations** (tile loads): all `wgSize = blockShape/threadTile` threads
  participate. Each thread loads `prod(threadTile)` elements cooperatively to fill the shmem tile.
- **Thread-private operations** (accumulation): each thread iterates over its `threadTile` in
  `var<private>` without any barriers.

For the fori_loop body reduction:

```wgsl
// Instead of: shmem_carry[tidx] += dot_result;  (1 element, in shmem)
// Emit:
for (var tr = 0; tr < 8; tr++) {
  for (var tc = 0; tc < 8; tc++) {
    acc[tr][tc] += tile_A[local_row*8+tr][kk] * tile_B[kk][local_col*8+tc];
  }
}
```

**Prerequisite:** This requires the compiler to understand the distinction between "cooperative
shmem operations" and "private accumulation operations" in the fori_loop body. Currently, all body
outputs go to shmem. With register tiling, carries that are only used for accumulation should stay
in `var<private>`.

**Register pressure and occupancy:** An 8×8 thread tile requires 64 accumulators (16 vec4s = 64
registers for f16, 128 for f32). On some GPUs this exceeds the register file per-thread and spills
to local memory, erasing gains. The compiler must offer a small candidate set of thread tiles (e.g.,
2×2, 4×4, 4×8, 8×4, 8×8) and select based on dtype and target backend:

- **f16:** 8×8 is safe on most discrete GPUs (64 f16 registers are small)
- **f32:** Start with 4×4 (16 accumulators — safe everywhere), benchmark before scaling up
- **Integrated GPUs (Intel, Apple):** Often have smaller register files; 4×4 may be optimal

Occupancy tuning (threads-per-workgroup vs registers-per-thread tradeoff) belongs in Phase 3, not
Phase 5 — it fundamentally determines the code structure and must be decided before vectorization
and unrolling are layered on top.

### O5: Loop Unrolling via `@unroll`

> **STATUS: @unroll BLOCKED → Manual unrolling IMPLEMENTED** — Chrome/Tint's `@unroll` attribute
> generates incorrect code (all memory accesses produce zeros — confirmed Tint compiler bug).
> **Workaround implemented:** Manual unrolling for the register-tiled ridx reduction loop in
> `blockMapFusedShaderSource()`. When `reSize ≤ 32`, the codegen emits the loop body N times, each
> with a per-iteration `createGen()` using `ridxOverride = AluExp.i32(ri)` to substitute the ridx
> variable with a literal constant. The AluExp simplifier then constant-folds the ridx-dependent
> shmem index expressions. This applies to both the carry-fused (O12b) and general accumulator
> paths. Measured ~10% speedup at 2048×2048 matmul.

**After O4**, the inner K-tile loop and the per-thread sub-tile loops should be unrolled.

**Key insight from Tint IR:** Chrome's Tint compiler passes the `@unroll` attribute to the backend
compiler (DXC on Windows, MSL on Mac). This is _much_ simpler and more effective than manually
expanding loops at our codegen level:

- DXC/MSL can do better register allocation when they unroll (they see the full picture)
- Manual unrolling produces massive WGSL source, increasing Tint compile time
- `@unroll` lets the backend decide the optimal strategy for its register file

**Approach — Emit `@unroll` attribute on WGSL loops:**

The codegen should annotate loops with `@unroll` when:

1. The loop bound is a `const` (required for the attribute to work)
2. The loop body is the innermost hot path (reduction accumulation)
3. The iteration count is modest (≤ 64; huge unrolls cause register spilling)

```wgsl
// Generated by our codegen:
@unroll
for (var ridx: u32 = 0u; ridx < BLOCK_K; ridx++) {  // BLOCK_K is const
  acc[r] += tile_A[local_row * 8u + tr][ridx] * tile_B[ridx][local_col * 8u + tc];
}
```

**Tier 1 (immediate value):** `@unroll` on the inner reduction loop (ridx over K-tile). This is the
innermost loop — unrolling it enables the GPU compiler to schedule FMAs without loop overhead and
keep accumulators in registers.

**Tier 2 (moderate value):** `@unroll` on the per-thread sub-tile loops (tr, tc over
`THREAD_TILE_R/C`). These are small fixed-size loops that benefit from full expansion.

**Tier 3 (optional):** The K-tile loop itself. Only unroll if K-tile count is small (≤ 16). For
large K, keep the loop — `@unroll` on a 256-iteration loop causes code bloat.

**Fallback — Manual unrolling:** If `@unroll` doesn't produce hoped-for speedup on a specific
backend, the codegen can fall back to manual expansion (replacing the loop with N copies of the
body, substituting the loop variable with literal constants). This is more work but gives full
control.

---

## 7. Additional Opportunities (O6–O12)

### O6: Accumulator Precision Control

The current shader accumulates in f32 (`var fl0_s3_acc: f32`) even for f16 inputs. This is correct
for numerical precision but means every multiply requires an f16→f32 cast. For inference workloads
where f16 precision suffices, the compiler should offer a `acc_dtype` control.

### O7: Cooperative Load Scheduling

Currently, each thread loads 1 element per tile. With 256 threads and a 16×16 tile, this works out
(256 = 16×16). But with register tiling (64 threads, 64×16 tile = 1024 elements), each thread must
load 16 elements. The codegen must emit a cooperative load loop:

```wgsl
for (var i = 0u; i < 16u; i++) {
  let flat = tid * 16u + i;
  tile_A[flat / 16u][flat % 16u] = A[...];
}
```

This is closely tied to O4 and should be implemented together.

### O8: Double Buffering

Load tile N+1 while computing on tile N. Requires 2× shmem budget and careful barrier placement.
This is a future optimization — the benefit depends on whether the workload is memory-bound or
compute-bound after the other optimizations land.

### O9: Transpose-in-Register for B Tile

The current shader does `transpose` as a separate shmem operation. With register tiling, the B tile
can be loaded in transposed order directly during the cooperative load phase, avoiding an extra
shmem transpose step and its barrier.

### O10: Workgroup Size Tuning

Different GPUs prefer different workgroup sizes. The compiler should accept a tuning parameter or
auto-tune based on device capabilities:

- Apple GPUs: 32×8 or 8×32 (SIMD width 32)
- NVIDIA: 8×8 or 16×16 (warp size 32)
- Intel/AMD: varies

### O11: Shared Memory Bank Conflict Avoidance

**Problem:** When multiple threads in a warp/wavefront access the same bank of shared memory
simultaneously, the accesses are serialized. For a 16-wide tile stored as `array<f16, 16>`,
consecutive rows map to the same bank offset → column-wise reads cause N-way conflicts.

**Why Tint can't help:** Tint does NOT automatically pad shared memory arrays. Bank conflict
avoidance must be explicit in the generated WGSL.

**Solution — Pad inner shmem dimension by +1:**

```wgsl
// Instead of: var<workgroup> tile_A: array<f16, 256>;  // 16×16, col reads conflict
// Emit:       var<workgroup> tile_A: array<f16, 272>;  // 16×17, no bank conflicts
```

The +1 padding on the inner dimension shifts each row by one bank, eliminating conflicts for both
row-wise and column-wise access patterns.

**Implementation:** During shmem allocation, when the shmem is used as a 2D tile (detected from the
ShapeTracker or block dimensions), pad the innermost stride by 1. Adjust all index expressions to
use the padded stride.

**Cost:** ~6% extra shmem per tile. For 16×16 f16 tiles: 512 → 544 bytes. Well within the 16KB
workgroup memory limit.

### O12: Subgroup Matrix Operations (Endgame)

Chrome 144+ exposes `subgroup_matrix` types through the `subgroups` extension. These map directly to
hardware tensor cores (NVIDIA WMMA, Intel XMX, Apple AMX), bypassing shared memory entirely.

**Expected impact:** $5{-}10\times$ faster than standard tiled matmul because:

- No shared memory loads/stores
- No workgroup barriers
- Hardware-optimized matrix multiply-accumulate in a single instruction

**Integration path:** Add a `subgroupMatmul` primitive that the blockMap codegen emits when the
`subgroups` feature is available and the tile sizes match hardware requirements (typically 16×16 or
8×8 depending on the GPU). Fall back to the standard tiled path when unavailable.

This is the long-term endgame but depends on spec stability and broad browser support.

---

## 8. Implementation Phases

### Phase 0: Measurement Infrastructure (Before Any Optimization)

**Goal:** Establish automated quality gates and benchmark baselines so every subsequent phase has
measurable, regression-proof outcomes.

**Quality gates — automated checks on generated WGSL:**

1. No `select(..., true)` patterns (P2a regression)
2. Barrier count per K-tile ≤ target (2 after Phase 1B)
3. No `min(max(...))` clamping chains in hot loops (P2b regression)
4. Expected `@unroll` annotations present (Phase 4+)
5. No modular index arithmetic in inner loops (Phase 2+)

Implement as a `validateShader(wgsl: string, rules: Rule[])` utility that runs in debug mode and in
CI. Capture shader source via `setDebug(2)` and diff against expected patterns.

**Benchmark artifacts:** Each phase must produce before/after shader diffs and GFLOP/s numbers for
at least: 256×256 f16, 1024×1024 f32, 4096×4096 f32. Store results in `docs/` alongside the captured
WGSL for reproducibility.

**Tests:**

- Unit test: `validateShader` correctly flags `select(x, true)`, `min(max(...))`, counts barriers
- Integration test: capture WGSL from `setDebug(2)` for `jit(tiledMatmul)(A, B)` 256×256 f16, verify
  it parses and matches current baseline (snapshot test for regression detection)
- Add to `bench/matmul.bench.ts`: tiledMatmul entries at 256², 1024², 4096² for f16 and f32

**Source files:**

- New: `test/shader-quality.test.ts` (quality gate tests)
- Edit: `bench/matmul.bench.ts` (add tiledMatmul benchmarks)

### Phase 1A: Safe Low-Risk Cleanup (P2a + Quality Gates)

**Goal:** Eliminate `select(…, true)` from all generated shaders. Zero semantic changes — this is
purely recognizing a constant and skipping dead code.

**Deliverable:**

- `accessorGlobal` one-line fix in `alu.ts`
- Quality gate infrastructure wired into CI
- Baseline shader captures + benchmarks for all subsequent phases

**Risk:** Near-zero. The `valid === true` check is a tautology elimination.

**Tests:**

- Existing `test/block-map-jit.test.ts` `lax.tiledMatmul` tests must still pass (correctness)
- Existing `test/lax.test.ts` `dynamicSlice` tests must still pass
- Quality gate: captured WGSL for tiledMatmul 256×256 f16 contains zero `select(` occurrences (when
  inputs are tile-aligned, no ShapeTracker mask → no select at all)
- Regression: padded inputs (e.g., 250×256) STILL correctly produce `select` (P2a must not affect
  masked ShapeTrackers — only constant-true valid flags)

**Source files:**

- Edit: [src/alu.ts](src/alu.ts) line 1778 (`accessorGlobal`)

### Phase 1B: Semantic Foundation Changes (P0 + P1 + P2b + P2c)

**Goal:** Reduce barriers from 4→2, eliminate `min(max())` clamping, remove scalar-via-shmem. The
shader should look like a clean (if slow) tiled matmul.

**Deliverable:** Generated shader for 256×256 f16 with exactly 2 barriers per K-tile, clean shmem
indexing, no boundary-check overhead.

**Expected speedup:** 2-4× from barrier reduction and less ALU waste.

**Risk:** Medium. `UncheckedDynamicSlice` introduces a new primitive with transform rules. Mitigated
by fallback-to-safe enforcement and debug assertions.

**Tests (P0 — Scalar Promotion):**

- Quality gate: no `if (tidx < 1u)` guard in generated WGSL for tiledMatmul
- Quality gate: no shmem allocation for scalar `kIdx` — look for `let` binding instead
- Correctness: all `test/block-map-jit.test.ts` tiledMatmul tests pass

**Tests (P1 — Barrier Phase Merging):**

- Quality gate: exactly 2 `workgroupBarrier()` per K-tile loop body (count occurrences between
  `for (var fl0_i` and the closing `}`)
- Unit test: barrier scheduler with mock step deps — verify load-phase grouping for 2 independent
  writers produces 1 barrier, not 2
- Correctness: all `test/block-map-jit.test.ts` tests pass

**Tests (P2b — UncheckedDynamicSlice):**

- Correctness: `jit(tiledMatmul)` matches `np.matmul` for tile-aligned and non-aligned sizes
- Transform: `grad(tiledMatmul)` matches `grad(np.matmul)` at `Br=Bc=Bk=4`
- Transform: `vmap(tiledMatmul)` batched correctness
- Composition: `jit(grad(vmap(tiledMatmul)))` doesn't crash, produces correct results
- Quality gate: no `min(max(` in generated WGSL when using unchecked variant
- Debug assertion: `setDebug(2)` + intentionally wrong slice sizes → assertion fires
- Safety: if a transform can't prove bounds preservation, it downgrades to safe DynamicSlice (test
  by inspecting Jaxpr IR — `makeJaxpr(grad(tiledMatmul))` should contain `UncheckedDynamicSlice` in
  forward, safe `DynamicUpdateSlice` in adjoint)

**Tests (P2c — padConcrete):**

- Correctness: `tiledMatmul(250x256, 256x256)` matches `np.matmul` (non-aligned inputs)
- Quality gate: padded inputs no longer produce `select` in WGSL (pad mask eliminated)
- Heuristic: tiny matrices (e.g., 4×5) still use mask-based pad (heuristic rejects)
- Memory: verify `padConcrete` buffer is freed/recycled after matmul completes

**Source files:**

- Edit: [src/frontend/core.ts](src/frontend/core.ts) (add `UncheckedDynamicSlice` to Primitive enum)
- Edit: [src/frontend/jit.ts](src/frontend/jit.ts) line 2378 (add unchecked lowering rule)
- Edit: [src/frontend/jvp.ts](src/frontend/jvp.ts) (add JVP rule)
- Edit: [src/frontend/linearize.ts](src/frontend/linearize.ts) (add transpose rule)
- Edit: [src/frontend/vmap.ts](src/frontend/vmap.ts) (add vmap rule)
- Edit: [src/backend/webgpu/block-map.ts](src/backend/webgpu/block-map.ts) lines 650-694 (barrier
  scheduler), line 1637 (scalar promotion)
- Edit: [src/library/lax.ts](src/library/lax.ts) line 755 (`tiledMatmul` — switch to unchecked DS,
  add `padConcrete`)
- Test: [test/block-map-jit.test.ts](test/block-map-jit.test.ts) (extend existing suite)

### Phase 2: Clean Indexing (O2)

**Goal:** Replace modular index arithmetic with direct shmem addressing.

**Deliverable:** Inner loop reads are `tile_A[row * K_TILE + ridx]` not
`fl0_s5[(((gidx / 16) % 16) * 16) + ridx]`.

**Tests:**

- Quality gate: inner reduction loop body contains no `%` (modulo) operator
- Quality gate: shmem reads use `tidx_0 * TILE + ridx` or `gidx` form
- Correctness: all `test/block-map-jit.test.ts` tiledMatmul tests pass

**Source files:**

- Edit: [src/backend/webgpu/block-map.ts](src/backend/webgpu/block-map.ts) (index gen in `createGen`
  / shmem read path)
- Edit: [src/shape.ts](src/shape.ts) (contiguity detection in ShapeTracker)

### Phase 3: Register Tiling + Occupancy Tuning (O4 + O7 + O10)

**Goal:** Accumulators in `var<private>`, multiple outputs per thread.

**Deliverable:** Generated shader with `threadTile` support. Start with 4×4 (safe on all GPUs),
benchmark, then scale to 8×8 where register pressure allows. Workgroup size tuning (O10) is included
here because it's inseparable from the thread-tile decision.

**Expected speedup:** 10-30× from register reuse + barrier amortization.

**Tests:**

- Quality gate: generated WGSL contains `var<private>` accumulators, NOT `var<workgroup>`
- Quality gate: `@workgroup_size` matches `blockShape / threadTile` (e.g., 8×8 for 64/8)
- Correctness: `lax.tiledMatmul` with `blockShape=[64,64], threadTile=[4,4]` matches np.matmul
- Correctness: `lax.tiledMatmul` with `threadTile=[8,8]` matches np.matmul (f16)
- Transform: `grad(tiledMatmul)` still correct with register tiling
- Benchmark: ≥10× speedup vs Phase 2 baseline at 1024×1024

**Source files:**

- Edit: [src/backend/webgpu/block-map.ts](src/backend/webgpu/block-map.ts) (thread mapping,
  cooperative load, `var<private>` accumulator codegen)
- Edit: [src/library/lax.ts](src/library/lax.ts) (`tiledMatmul` — add threadTile option)

### Phase 4: Vectorization + Unrolling + Bank Conflicts (O3 + O5 + O11)

**Goal:** vec4 loads in cooperative tile phases, `@unroll` on inner loops, bank-conflict-free shmem
layout.

**Expected speedup:** 2-4× from vectorized memory access, 1.5-2× from unrolling, up to 2× from
eliminating bank conflicts (workload-dependent).

**Status:**

- **O11 (bank padding): ✅ DONE** — `shmemBankPad` adds +1 inner dimension padding
- **O5 (@unroll): ❌ BLOCKED → WORKAROUND ✅** — Chrome/Tint generates incorrect code for `@unroll`
  loops (all memory accesses produce zeros — Tint compiler bug). Workaround: **manual unrolling**
  implemented for the register-tiled ridx reduction loop. When `reSize ≤ 32`, the codegen emits the
  loop body N times with literal ridx indices via per-iteration `createGen()` +
  `ridxOverride = AluExp.i32(ri)`. The AluExp simplifier constant-folds ridx-dependent shmem index
  expressions, enabling the GPU compiler to optimize register allocation and FMA scheduling.
  Measured ~10% speedup at 2048×2048 matmul (launch-bound at smaller sizes).
- **O3 (vec4 loads): DEFERRED** — Requires declaring storage buffers as `array<vec4<f32>>` which is
  invasive (all reads must use vec4 indexing). Benchmarks show tiledMatmul at 1024×1024 is only
  2.68ms including WebGPU API overhead, suggesting we're launch-bound not memory-bound. vec4
  optimization would need GPU-side profiling (timestamp-query) to verify benefit.

**Tests:**

- Quality gate: shmem declaration uses `TILE + 1` inner dimension (bank padding) ✅
- ~~Quality gate: cooperative load phase uses `vec4<f16>` or `vec4<f32>` reads~~ (deferred)
- ~~Quality gate: inner reduction loop has `@unroll` annotation~~ (blocked)
- Correctness: all tiledMatmul tests pass ✅
- Benchmark: tiledMatmul 2048×2048 is 35× faster than np.matmul

### Phase 5: Polish (O6, O8, O9)

**Goal:** Double buffering, transpose optimization, precision control.

### Phase 6: Subgroup Matrix (O12)

**Goal:** Leverage hardware tensor cores when available. This is the long-term endgame.

### Dependency DAG

Not all phases are strictly linear. The true dependency structure:

```
Phase 0 (measurement) ─── required by ALL subsequent phases
    │
    ├─── Phase 1A (P2a) ─── no deps, safe, do first
    │        │
    │        └─── Phase 1B (P0 + P1 + P2b/c) ─── depends on 1A for quality gates
    │                 │
    │                 ├─── Phase 2 (O2 indexing) ─── depends on 1B for clean barriers
    │                 │        │
    │                 │        └─── Phase 3 (O4 + O7 + O10 tiling) ─── depends on O2
    │                 │                 │
    │                 │                 └─── Phase 4 (O3 + O5 + O11) ─── depends on O4
    │                 │                          │
    │                 │                          └─── Phase 5 (O6 + O8 + O9)
    │                 │
    │                 └─── Phase 6 (O12 subgroups) ─── independent of O2–O11
    │                                                   (alternative fast path)
```

Phase 6 (subgroup matrix) is an **independent branch** — it doesn't need Phases 2–5 because it
bypasses shared memory entirely. It can be prototyped in parallel once Phase 1B lands.

### Performance Progression Target (RTX 4070, 4096×4096 f32)

| Phase    | Expected GFLOP/s | % of Peak |
| -------- | ---------------- | --------- |
| Current  | ~20              | 0.07%     |
| Phase 1A | ~30              | 0.1%      |
| Phase 1B | ~80              | 0.3%      |
| Phase 2  | ~150             | 0.5%      |
| Phase 3  | ~2000-5000       | 7-17%     |
| Phase 4  | ~5000-12000      | 17-40%    |
| Phase 5  | ~8000-15000      | 28-52%    |
| Phase 6  | ~15000-29000     | 52-100%   |

---

## 9. The Evolved Compiler Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ User code: tiledMatmul(A, B) / blockMap(body, inputs, opts)                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Tracing        makeJaxpr → BlockMap + ForiLoop + DynamicSlice + Dot     │
│ 2. JIT Compile    flatten → simplify → splitGraphDataflow → jitCompile     │
│ 3. Block-Map      Eligibility check → Step classification → Shmem alloc   │
│                                                                             │
│ === NEW PASSES ============================================================│
│                                                                             │
│ 4. Scalar Promotion  [P0]  Detect size-1 kernels → inline as let bindings  │
│ 5. Check Elision     [P2]  Skip Where(true), unchecked DS, concrete pad   │
│ 6. Phase Scheduling  [P1]  Group shmem writes → minimize barriers          │
│ 7. Index Simplify    [O2]  Contiguous shmem → flat/minimal index forms     │
│ 8. Thread Mapping    [O4]  Apply threadTile → split cooperative vs private │
│ 9. Vectorization     [O3]  Promote aligned scalar ops → vec4 ops           │
│ 10. Bank Pad         [O11] Pad shmem inner dim +1 to avoid bank conflicts  │
│ 11. Tint Hints       [NEW] const tile sizes, @unroll, let intermediates    │
│                                                                             │
│ === WGSL EMIT =============================================================│
│                                                                             │
│ 12. Codegen          Emit optimized WGSL from annotated step list          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Output: Single fused WGSL compute shader                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each pass is independently testable: capture the block-map step list before and after each pass,
verify invariants. The passes compose — register tiling + vectorization + unrolling together produce
the target shader from Section 2.

This architecture doesn't just solve matmul. Conv2d, attention (Q×K^T×V), scan reductions, and
Cholesky all use `blockMap` + `foriLoop`. Every compiler improvement benefits all of them.

---

## 10. Backend Capability Matrix

Features used by the plan, gated by runtime detection at device init:

| Feature                | Chrome | Safari | Phase | Fallback              |
| ---------------------- | ------ | ------ | ----- | --------------------- |
| `shader-f16`           | 113+   | 18+    | All   | f32 accumulation      |
| `workgroupUniformLoad` | 137+   | ❌     | 1B    | Plain `shmem[0]` read |
| `@unroll` attribute    | 141+   | ❌     | 4     | Manual loop expansion |
| `subgroups`            | 144+   | ❌     | 6     | Standard tiled path   |
| `subgroup_matrix`      | 144+   | ❌     | 6     | Standard tiled path   |

The codegen must query the backend's feature set and select the best available strategy. Features
that are unavailable fall back gracefully — the generated shader is always correct, just slower.
Store capabilities in `BackendFeatures` (alongside existing `shader-f16` flag) and pass them through
to the block-map codegen passes.
