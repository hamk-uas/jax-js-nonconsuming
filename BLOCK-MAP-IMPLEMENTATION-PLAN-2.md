# BlockMap Implementation Plan v2: Pallas for WebGPU

A `lax.block_map` primitive that compiles arbitrary array sub-programs into **workgroup-local
shared-memory GPU kernels**. Tiles data into blocks, applies a body sub-jaxpr per block using
`var<workgroup>` memory, and reassembles results — the foundational primitive for tiled matmul,
flash attention, fused normalization, workgroup-local scans, and eventually replacing hand-written
WGSL routines.

**Status:** Plan. Not yet implemented. Reviewed Feb 2026; v2.2 incorporates review feedback on
matmul lowering, pattern detection, barrier placement, memory discipline, `fori_loop` primitive,
separate input/output types, and workgroup size constraints.

**Predecessor:** `BLOCK-MAP-IMPLEMENTATION-PLAN.md` (v1, deleted) viewed `block_map` primarily as an
optimization path for `associativeScan`. This revision reframes it as a **general-purpose
shared-memory compute primitive** — the WebGPU equivalent of JAX's Pallas / Triton's `tl.program_id`
model — and evaluates it against the full space of JAX ecosystem use cases.

**Related plans:** `DLM-SPEEDUP-PLAN.md` identified a deferred P4 ("WebGPU fused compose shader,
~500 LOC") that `block_map` would generalize — see use case #9 below.

---

## Strategic Motivation: Why a Shared-Memory Primitive?

### The gap in jax-js today

jax-js has a mature JIT compiler that fuses elementwise chains into single GPU dispatches. It has
hand-written WGSL routines for sort, Cholesky, LU, and triangular solve. It has cooperative group
reductions in `pipelineSource()` for large reductions. But it has **no general mechanism for tiled,
shared-memory GPU compute**.

This means:

- **Tiled matmul is impossible.** The P1 performance goal (5–10× matmul speedup) requires loading
  tiles of A and B into `var<workgroup>`, multiplying, and accumulating — a standard GPU
  optimization that jax-js cannot express. The current matmul is memory-bandwidth bound: each thread
  reads from global memory for every accumulation step.
- **Flash attention is impossible.** FlashAttention avoids materializing the $N \times N$ attention
  matrix by computing softmax iteratively over blocks in shared memory. Without a tiled compute
  model, jax-js must either materialize the full matrix (O(N²) memory) or rely on a hand-written
  WGSL routine (unscalable).
- **Fused layer normalization requires two global passes.** LayerNorm = reduction (mean, variance)
  - elementwise (normalize). Without shared memory, the mean/variance must be written to global
    memory and read back. With `block_map`, the reduction result lives in a `var<workgroup>` local
    and the normalize step reads it immediately — one dispatch instead of two.
- **Each new routine is ~200–300 lines of hand-written WGSL.** Sort (200 LOC), Cholesky (260 LOC),
  LU (180 LOC), TriangularSolve (280 LOC) — all in `routines.ts`. Each is a one-off, each needs its
  own WGSL, each is unmaintainable. A `block_map` compiler could generate correct (if not
  peak-optimal) WGSL from the same TypeScript body function that the eager mode runs.

### What JAX's Pallas provides (our reference point)

[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (JAX) and
[Triton](https://triton-lang.org) (PyTorch) both solve this problem: they let users write
block-level programs in high-level Python that compile to shared-memory GPU kernels. The key
abstraction is:

1. **Grid of blocks** — the computation is partitioned into blocks (tiles), each mapped to one
   workgroup.
2. **Block-scoped memory** — each block loads its tile from global memory into fast local
   (shared/SRAM) memory.
3. **Body function** — arbitrary operations on the block, compiled to GPU instructions.
4. **Writeback** — results written from local memory back to global memory.

`block_map` is jax-js's version of this. It is deliberately simpler than Pallas (no `BlockSpec`
input/output descriptors, no `grid` parameter — just axis + blockSize), because WebGPU is simpler
than CUDA (no explicit SMEM management, just `var<workgroup>`).

### Concrete use cases from the JAX ecosystem

| #   | Use Case                         | JAX Equivalent                                | How block_map Enables It                                                | Impact                                                                                             |
| --- | -------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1   | **Tiled matmul**                 | XLA's tiled GEMM / Pallas matmul              | 2D `block_map` loads A,B tiles into shmem, accumulates C tile           | 5–10× matmul speedup (P1 goal)                                                                     |
| 2   | **Flash attention**              | `jax.nn.dot_product_attention` / Pallas flash | Block over sequence dim, iterative softmax in shmem                     | O(N) memory vs O(N²); enables long sequences                                                       |
| 3   | **Fused LayerNorm / RMSNorm**    | XLA's fused normalize                         | Reduce + normalize in one block dispatch                                | 2× speedup over two-pass                                                                           |
| 4   | **Fused softmax**                | Triton fused softmax tutorial                 | Reduce-max + shift + exp + reduce-sum + normalize, all in shmem         | 1 dispatch vs 3–4                                                                                  |
| 5   | **Workgroup-local assocScan**    | N/A (JAX uses XLA's scan)                     | Kogge-Stone body inside block_map with barriers                         | 8× for N≤256 (replaces 1,074 LOC M7.4 fused shader)                                                |
| 6   | **Chunked state-space models**   | Mamba / S4 / Griffin                          | Sequential scan inside block, carry between blocks                      | Enables SSM inference on WebGPU                                                                    |
| 7   | **Tiled reductions**             | XLA's tree reduction                          | Block reduce in shmem → global tree of block summaries                  | Better occupancy for large reductions                                                              |
| 8   | **Workgroup-local sort**         | N/A (Pallas doesn't sort)                     | Bitonic sort passes inside block_map body                               | Could replace 200 LOC hand-written WGSL                                                            |
| 9   | **DLM-JS Kalman filter compose** | N/A (dlm-js specific)                         | 5-tuple Särkkä composition body (28 kernel ops) runs in shmem per block | Replaces deferred P4 (~500 LOC hand-written WGSL); dispatch reduction for small-matrix time series |

### Concrete example: DLM-JS Kalman filter composition (use case #9)

The [dlm-js](https://github.com/nickovchinnikov/dlm-js) library implements Bayesian dynamic linear
models using `associativeScan` over a 5-tuple Särkkä composition. The composition body contains 28
kernel operations (13 batched matmul + 15 elementwise) on **very small matrices** (m=1–13, typically
2–4) across **long time series** (n=50–1000+). With the analytical `inv` for m≤4 (P2 in
`DLM-SPEEDUP-PLAN.md`), the body is 100% kernel steps — zero Routine steps — making it fully
eligible for `block_map` shared-memory fusion.

The DLM speedup plan previously identified a **deferred P4: "WebGPU fused compose shader"** that
would have required ~500+ LOC of hand-written, algorithm-specific WGSL. `block_map` eliminates this
entirely — the same TypeScript composition body that runs in eager mode compiles automatically into
a fused shared-memory shader. For 2×2 state matrices with n=500 timesteps, the compose body needs ~8
KB of shared memory (well within the 16 KB minimum) and would reduce per-composition dispatches from
~28 (one per kernel step in the current global-dispatch model) to 1.

This use case validates `block_map`'s value for **dispatch-bound workloads** — scenarios where
individual kernels are tiny but numerous, and the dominant cost is GPU dispatch overhead rather than
compute.

---

## Design

### Primitive definition

```ts
// In src/frontend/core.ts, Primitive enum:
BlockMap = "block_map",

// In PrimitiveParamsImpl:
[Primitive.BlockMap]: {
  jaxpr: Jaxpr;              // Body sub-jaxpr
  blockShape: number[];      // Block dimensions (1D: [B], 2D: [Br, Bc])
  inAxes: any;               // PyTree describing which array dimensions map to which grid axes
  outAxes: any;              // PyTree describing which output dimensions map to which grid axes
  numConsts: number;         // Constants captured from outer scope
  numInputs: number;         // Number of input arrays (rest are outputs of body)
};
```

**Key change from v1:** `blockSize: number` → `blockShape: number[]` + `inAxes` / `outAxes`. This
supports both 1D tiling (scan, reduction, normalization) and 2D tiling (matmul, attention) from day
one, avoiding a painful API migration later.

### Public API

```ts
// In src/library/lax.ts:
function block_map<I extends ArrayOrPytree, O extends ArrayOrPytree>(
  f: (block: I) => O,
  elems: I,
  options: {
    blockShape: number[];
    inAxes?: any; // PyTree of (number | null)[] matching elems leaves
    outAxes?: any; // PyTree of (number | null)[] matching output leaves
  },
): O;
```

**Semantics:** Partitions `elems` along grid axes into blocks of shape `blockShape`, applies `f` to
each block independently, and reassembles the results. The number of blocks along grid axis `i` is
determined by the `inAxes` or `outAxes` mapping. This mirrors the `jax.vmap` API but operates on
discrete block sizes rather than single elements, enabling cross-axis tiling (e.g., matmul).

**1D example (prefix scan):**

```ts
// Each block of 256 elements is independently prefix-scanned
const result = lax.block_map((block) => koggeStoneBody(fn, block), xs, {
  blockShape: [256],
  inAxes: [0],
  outAxes: [0],
});
```

**2D example (tiled matmul):**

```ts
// Each (16, 16) tile of C accumulates A[r,:] × B[:,c]
// inAxes specifies that A maps over grid axis 0 (and broadcasts over 1),
// while B maps over grid axis 1 (and broadcasts over 0).
const C = lax.block_map(
  ({ A: aTile, B: bTile }) => tileMatmul(aTile, bTile),
  { A: A, B: B },
  {
    blockShape: [16, 16],
    inAxes: { A: [0, null], B: [null, 1] },
    outAxes: [0, 1],
  },
);
```

### Execution model

```
block_map(f, xs, { blockShape: [B], inAxes: [0], outAxes: [0] })
  │
  ├─ Eager mode: slice → f(block_pytree) → concat (JS loop over grid blocks)
  │
  └─ JIT mode:
       ├─ WebGPU (body fits shmem):
       │   → Fused WGSL shader
       │   → var<workgroup> for intermediates
       │   → workgroupBarrier() between dependent steps
       │   → 1 workgroup per block
       │   → 1 dispatch with grid=[M_x, M_y, ...] for the entire block_map
       │
       ├─ WebGPU (body exceeds shmem):
       │   → Per-block dispatch (M × body steps)
       │   → Global memory for intermediates
       │
       └─ WASM: compiled loop over blocks (single WASM module)
```

---

## Phase 0: Prototype (De-Risk the Body Tracing)

**Goal:** Verify that tracing a block body function through `makeJaxpr` produces usable IR before
building any infrastructure.

### What to validate

1. A Kogge-Stone prefix scan body traces into a clean jaxpr with `shrink`, `concatenate`, and the
   user's `fn` as equations.
2. A tiled matmul body (load tile, accumulate, store) traces into kernel + reduction equations.
3. A fused-softmax body (reduce-max, sub, exp, reduce-sum, div) traces into the expected sequence.
4. The jaxpr body size is reasonable (not O(blockSize) equations — operations should be
   blockSize-independent).

### Deliverable

A test file `test/block-map-prototype.test.ts` with ~8 tests, validating tracing only. No new
primitive, no codegen. Takes ~2 hours. If tracing produces unexpected shapes (e.g., shapes that
depend on blockSize values rather than blockSize as a dimension), stop and redesign.

---

## Phase 1: Core IR + Eager Fallback (~250 LOC)

**Goal:** Add `Primitive.BlockMap` to the IR with a correct eager-mode implementation.

### 1.1: Primitive registration

- `BlockMap = "block_map"` in `Primitive` enum
- `PrimitiveParamsImpl` entry with `{ jaxpr, blockShape, inAxes, outAxes, numConsts, numInputs }`
- Abstract eval rule: output shapes = derived from input shapes and `outAxes`.

### 1.2: Tracing API (`lax.block_map`)

Traces `f` with abstract inputs whose dimensions are replaced with `blockShape` along the mapped
`inAxes`. Captures constants. Emits `Primitive.BlockMap` equation.

### 1.3: Eager implementation

Slice/pad/concat loop over the grid of blocks:

```ts
[Primitive.BlockMap](args, { jaxpr, blockShape, inAxes, outAxes, numConsts }) {
  const consts = args.slice(0, numConsts);
  const elems = args.slice(numConsts);
  // Derive grid shape array `M` from inputs, inAxes, and blockShape
  // For each block index tuple (i0, i1, ...):
  //   slice elems according to inAxes mapping
  //   pad last block if needed
  //   evalJaxpr(jaxpr, consts, blocks)
  //   trim if padded
  //   collect results
  // Concatenate all block results according to outAxes mapping
}
```

### 1.4: Tests

- Identity body (output = input) — verifies tiling/reassembly.
- Elementwise body (double each element).
- Non-divisible N (N=10, blockShape=[4] → 3 blocks, last padded).
- 2D tiling (matmul-shaped: axes=[0,1]).
- Pytree elems.
- Length-0 edge case (N=0).

**Memory discipline from day one:** All Phase 1 tests run under the global `checkLeaks` harness
(same as every other test in the suite — see `test/setup.ts`). Additionally, add explicit leak
regression tests for:

- Padded blocks: padding creates intermediate arrays that must be disposed after trimming.
- Pytree elems: `tree.flatten` / `tree.unflatten` intermediates during slice/concat.
- Non-divisible N with `using` inside the body: verify that block intermediates don't accumulate.

The eager fallback loop slices inputs and concatenates outputs — each iteration creates intermediate
arrays. Getting `using`/`.dispose()` exactly right here prevents per-block memory spikes that would
be invisible in JIT mode (where the compiler manages lifetimes). This is the ownership correctness
principle: code must work correctly in eager mode, and `jit()` is only a performance optimization.

### Deliverables

| File                     | Change                             | LOC est  |
| ------------------------ | ---------------------------------- | -------- |
| `src/frontend/core.ts`   | Primitive, params, abstract eval   | ~20      |
| `src/library/lax.ts`     | `block_map()` public API + tracing | ~60      |
| `src/frontend/array.ts`  | Eager impl (slice/pad/concat loop) | ~70      |
| `src/frontend/jaxpr.ts`  | Abstract eval rule                 | ~10      |
| `src/index.ts`           | Re-export `block_map`              | ~2       |
| `test/block-map.test.ts` | Basic correctness tests            | ~90      |
| **Total**                |                                    | **~252** |

---

## Phase 1b: Loop Primitives — `fori_loop` and `dynamic_slice` (~100 LOC)

**Goal:** Add `fori_loop` and `dynamic_slice` primitives that enable sequential loops with
data-dependent indexing — the missing building block for tiled matmul (Phase 4), iterative
algorithms (Cholesky, Newton), and flash attention (Phase 6) inside `block_map`.

**Why `lax.scan` is insufficient for block_map bodies:**

- `scan` stacks per-iteration outputs Y into a result array — wasteful when only the final carry
  matters (matmul accumulation, iterative refinement).
- `Primitive.Scan` has no fused WGSL lowering inside `block_map` — it would trigger fallback to
  per-block global-memory dispatch, defeating the entire point of shared-memory fusion.
- Expressing sequential loops via scan requires pre-reshaping inputs to expose the iteration axis as
  a leading dimension, adding unnecessary complexity.

### 1b.1: `fori_loop` primitive

```ts
// In Primitive enum:
ForiLoop = "fori_loop",

// Params:
{ jaxpr: Jaxpr; numConsts: number; lower: number; upper: number }

// Public API (src/library/lax.ts):
function fori_loop<C extends ArrayOrPytree>(
  lower: number,
  upper: number,
  body: (i: JaxArray, carry: C) => C,
  init: C,
): C;

// Note: `lower` and `upper` are concrete `number` values, resolved at trace time.
// Phase 4 tiled matmul uses `K / Bk` as `upper`, where K is always concrete after
// pre-padding to a multiple of Bk. Supporting symbolic (SymDim) bounds is deferred
// to a future phase if needed — it would require the WGSL `for` loop to accept a
// runtime uniform for `upper`, which is straightforward but out of initial scope.
```

**Semantics:** Equivalent to `for (let i = lower; i < upper; i++) carry = body(array(i), carry)`.
The body receives `i` as a **scalar int32 `JaxArray`** (not a JS number). During tracing, `i` is an
abstract tracer; during eager execution, it is a concrete 0-D array wrapping the loop index. Returns
the final carry after `upper - lower` iterations.

> **Why `JaxArray` and not `number`?** JavaScript has no operator overloading. If `i` were a plain
> `number`, user code like `i * blockSize` would silently produce `NaN` or string concatenation
> during tracing (where `i` is actually a `Tracer` object). Making `i` a `JaxArray` forces users to
> write `i.mul(blockSize)`, which traces correctly and compiles to efficient WGSL/WASM. The
> eager-mode implementation wraps the loop index via `np.array(idx, 'int32')` each iteration.

**Compilation behavior:**

| Context                            | What happens                                                                                                                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Eager mode                         | JS `for` loop; wraps index as `np.array(i, 'int32')` before calling body per iteration                                                                                                                                          |
| JIT (standalone, WASM)             | Compiled WASM `loop` with body inlined                                                                                                                                                                                          |
| JIT (standalone, WebGPU)           | Falls back to JS loop (no workgroup context; `fori_loop` is primarily valuable inside `block_map`)                                                                                                                              |
| Inside `block_map` fused shader    | WGSL `for (var i: i32 = lower; i < upper; i++)` with carry in `var<private>` or `var<workgroup>`, body inlined, barriers per iteration if body has cross-thread deps. (Bounds checked rigorously as i32 to mirror JS semantics) |
| Inside `block_map` WASM block-loop | WASM `loop` nested inside the block iteration loop                                                                                                                                                                              |

### 1b.2: `dynamic_slice` primitive

```ts
// In Primitive enum:
DynamicSlice = "dynamic_slice",

// Public API (src/library/lax.ts):
function dynamic_slice(operand: JaxArray, startIndices: JaxArray[], sliceSizes: number[]): JaxArray;
```

**Semantics:** Extracts a slice of `operand` starting at runtime-computed `startIndices` with
compile-time-constant `sliceSizes`. This is the read counterpart to the existing
`DynamicUpdateSlice` (DUS) primitive. Unlike `shrink` (which requires compile-time offsets),
`dynamic_slice` supports data-dependent indexing — essential for `fori_loop` bodies that index into
arrays using the loop variable. **Out-of-bounds semantics**: start indices are unconditionally
clamped to `[0, dimSize - sliceSize]` per dimension, mimicking JAX's exact guarantees to preserve
valid hardware loads and correct transposition behavior.

**Compilation:** In the fused shader compiler, `dynamic_slice` on a shared-memory operand becomes
offset arithmetic: `shmem[base + startIdx * stride + tidx]`. No data copy needed.

### 1b.3: AD rules

- **`fori_loop` JVP:** Doubled body — primals + tangents carried together. Same structure as `scan`
  JVP but simpler (no Y output).
- **`fori_loop` transpose:** Reverse-order loop with transposed body.
- **`fori_loop` vmap:** Each batch element runs its own loop.
- **`dynamic_slice` JVP:** `tangent_out = dynamic_slice(tangent_in, same_indices, same_sizes)`.
- **`dynamic_slice` transpose:** `dynamic_update_slice(zeros, cotangent, same_indices)`.

### 1b.4: Why not `while_loop`?

`while_loop(cond_fn, body_fn, init)` is more general (unknown trip count) but:

- Harder to bound shared memory (loop count unknown at compile time).
- Less urgent — `fori_loop` covers matmul K-accumulation, Cholesky inner loops, Newton iterations
  with fixed step count.
- Can be added later with WGSL `loop { if (!cond) { break; } ... }`.

### Deliverables

| File                        | Change                                                        | LOC est  |
| --------------------------- | ------------------------------------------------------------- | -------- |
| `src/frontend/core.ts`      | `ForiLoop` + `DynamicSlice` primitives, params, abstract eval | ~25      |
| `src/library/lax.ts`        | `fori_loop()` + `dynamic_slice()` public API + tracing        | ~30      |
| `src/frontend/array.ts`     | Eager impl (JS for loop; dynamic slice via TypedArray offset) | ~20      |
| `src/frontend/jvp.ts`       | JVP rules for both primitives                                 | ~15      |
| `src/frontend/linearize.ts` | Transpose rules for both primitives                           | ~15      |
| `src/frontend/vmap.ts`      | Vmap rules                                                    | ~10      |
| `src/index.ts`              | Re-export                                                     | ~2       |
| `test/block-map.test.ts`    | fori_loop + dynamic_slice tests                               | ~60      |
| **Total**                   |                                                               | **~177** |

---

## Phase 2: AD Rules (~170 LOC)

**Goal:** Make `block_map` differentiable. The key insight: `block_map` is a **spatial map** — each
block is transformed independently with no cross-block dependencies. This makes AD rules trivial.

### 2.1: JVP rule

Differentiate each block's body independently. Double the inputs (primals + tangents), run the JVP'd
body through `block_map`, split outputs:

```ts
[Primitive.BlockMap]([...primals], [...tangents], params) {
  const jvpBody = jvpJaxprCache.getOrCreate(params.jaxpr, ...);
  // Tangent inputs/outputs have the same shapes → same axis mappings as primals.
  const jvpInAxes = doubleAxes(params.inAxes);   // [primalAxes..., tangentAxes...]
  const jvpOutAxes = doubleAxes(params.outAxes);
  const doubled = bind(Primitive.BlockMap,
    [...primalConsts, ...tangentConsts, ...primals, ...tangents],
    { ...params, jaxpr: jvpBody, numConsts: params.numConsts * 2,
      numInputs: params.numInputs * 2, inAxes: jvpInAxes, outAxes: jvpOutAxes },
  );
  return [doubled.slice(0, half), doubled.slice(half)];
}
// doubleAxes(axes): duplicate flat axis list — tangent leaves use same mapping as primals.
```

### 2.2: Transpose rule

The transpose of a spatial map is a spatial map of the transposed body:

```ts
// Transpose rule:
// Cotangents are output-shaped → tiled on old outAxes.
// Gradient outputs are input-shaped → reassembled on old inAxes.
const transposedBody = transposeJaxpr(params.jaxpr, ...);
return bind(Primitive.BlockMap, [...consts, ...cotangents],
  { ...params, jaxpr: transposedBody,
    inAxes: params.outAxes, outAxes: params.inAxes },
);
```

### 2.3: Vmap rule

Each batch element gets its own `block_map`. Shift `inAxes` and `outAxes` indices if batch dim ≤ any
mapped axis.

### 2.4: Tests

- `grad(sum(block_map(f, xs)))` vs finite differences.
- `jit(grad(block_map(...)))` — no leaks.
- `vmap(block_map(...))` — batching correctness.
- `grad` through a body containing reductions (`sum` inside block).

### Deliverables

| File                        | Change                   | LOC est  |
| --------------------------- | ------------------------ | -------- |
| `src/frontend/jvp.ts`       | JVP rule                 | ~30      |
| `src/frontend/linearize.ts` | PE rule + transpose rule | ~45      |
| `src/frontend/vmap.ts`      | Vmap rule                | ~25      |
| `test/block-map.test.ts`    | AD + vmap tests          | ~70      |
| **Total**                   |                          | **~170** |

---

## Phase 3: WebGPU Shared-Memory Compiler (~920 LOC)

**Goal:** Compile kernel-only `block_map` bodies into single WGSL compute shaders where each
workgroup processes one block using `var<workgroup>` shared memory.

This is the hardest phase. It is also the phase that makes `block_map` a real performance primitive
rather than just syntactic sugar.

### 3.1: What the compiler does

```
Input: Body JitProgram (steps: malloc, execute, free, recycle)
  ↓
Analysis:
  ├─ All kernel steps (no routines)? → fused shader path
  ├─ shmemBytes ≤ device.limits.maxComputeWorkgroupStorageSize? → fused shader
  └─ Otherwise → per-block dispatch fallback
  ↓
Fused shader codegen:
  1. Global storage bindings for inputs and outputs
  2. var<workgroup> arrays for every intermediate (malloc → shmem)
  3. Each kernel step → inline WGSL expression (reuse translateAluOpToWgsl / gen())
  4. workgroupBarrier() between data-dependent steps
  5. Thread i processes element i within the block
  6. Outer block index from workgroup_id
  ↓
Output: ShaderInfo { code, workgroupSize: [blockShape[0], blockShape[1] || 1, blockShape[2] || 1], passes: [{grid: gridShape}] }
```

**Hard limit: `maxComputeInvocationsPerWorkgroup`.** WebGPU requires that
`workgroupSize[0] × workgroupSize[1] × workgroupSize[2] ≤ maxComputeInvocationsPerWorkgroup`
(typically **256**). For 2D tiles this hard-caps `blockShape` to e.g. 16×16 (=256 threads). 32×32
(=1024 threads) exceeds the limit on most devices and would fail pipeline creation. The fused shader
compiler must validate `prod(blockShape) ≤ device.limits.maxComputeInvocationsPerWorkgroup` and fall
back to per-block dispatch if exceeded. This limits the P1 matmul tile size to 16×16 unless
per-thread work-tiling (each thread computing multiple output elements) is added — a Phase 4
optimization not in the initial scope.

### 3.2: Shared memory budget

WebGPU guarantees `maxComputeWorkgroupStorageSize ≥ 16,384 bytes` (16 KB). Typical GPUs provide
32–48 KB. Each `malloc` step from the body's JitProgram that would normally allocate a global buffer
becomes a `var<workgroup>` shared memory array.

```
shmemBytes = sum(step.size for step in bodySteps where step.type === "malloc")
```

If `shmemBytes > device.limits.maxComputeWorkgroupStorageSize`, fall back to per-block dispatch.

**Critical constraint: `var<workgroup>` requires compile-time constant sizes.** WGSL does not
support runtime-sized `var<workgroup>` declarations — the array size must be a compile-time literal
in the shader source. This means `blockShape` elements must be **concrete numbers**, not `SymDim`
values, at the time of WGSL codegen.

**Why this is safe:** `blockShape` is specified by the user as `number[]` in the `block_map` options
— it is always concrete. The symbolic dimension propagation (`SymDim`) applies to the _outer_ array
dimension (e.g., `N` in `xs.shape = [N, features]`), not to the block dimensions. The number of
blocks `M = ceil(N / blockShape[0])` may be symbolic (and determines the grid dispatch count), but
the per-block shared memory size is always concrete because it depends only on `blockShape` and the
body's intermediate sizes (which are functions of `blockShape`).

**Guard:** The fused shader compiler asserts `typeof blockShape[i] === "number"` for all axes before
generating `var<workgroup>` declarations. If a symbolic value leaks through (should be impossible
given the API contract), the compiler falls back to per-block dispatch rather than generating
invalid WGSL. Similarly, the body JitProgram's `malloc` step sizes are checked for
`isSymbolicSize()` — symbolic malloc sizes trigger fallback (same guard as mega-module, see Part 1).

**Practical shared memory budget (f32, blockShape=[256]):**

| Body pattern                                  | Intermediates   | shmemBytes | Fits 16 KB? |
| --------------------------------------------- | --------------- | ---------- | ----------- |
| Elementwise chain (3 steps)                   | 3 × 256 × 4     | 3,072      | ✅          |
| Kogge-Stone cumsum (2 ping-pong)              | 2 × 256 × 4     | 2,048      | ✅          |
| Fused softmax (max + shift + exp + sum + div) | 5 × 256 × 4     | 5,120      | ✅          |
| Kalman filter (2×2 matrices, 2 leaves)        | 8 × 256 × 4     | 8,192      | ✅          |
| Tiled matmul (16×16 tiles of A and B)         | 2 × 16 × 16 × 4 | 2,048      | ✅          |
| Kalman filter (4×4 matrices)                  | 32 × 256 × 4    | 32,768     | ❌ (16 KB)  |

Bodies that exceed the budget still work — they just use per-block global-memory dispatch. No
correctness regression.

### 3.3: Barrier placement

Rule: if step B reads from step A's shared-memory output (**any** data dependency), insert
`workgroupBarrier()` between them. Independent steps can share a barrier interval.

```wgsl
// Step 1: all threads write shmem_a
shmem_a[tidx] = input[blockOffset + tidx] + 1.0;
workgroupBarrier();  // ← step 2 reads shmem_a

// Step 2: all threads read shmem_a, write shmem_b
shmem_b[tidx] = shmem_a[tidx] * 2.0;
workgroupBarrier();  // ← step 3 reads shmem_b

// Step 3: writeback to global
output[blockOffset + tidx] = shmem_b[tidx];
```

**Conservative strategy (Phase 3.0):** Barrier after every step. **Optimized (Phase 3.1):** Build a
dataflow DAG of step dependencies and only insert barriers where a read-after-write hazard exists.

**Critical constraint: uniform control flow.** WebGPU requires that all active invocations in a
workgroup execute `workgroupBarrier()` — barriers must not be placed inside divergent control flow
(e.g., inside one branch of an `if/else`). This is a hard WGSL validation rule, not a performance
guideline.

**Risk:** If a body jaxpr contains `AluOp.Where` (the ternary select `cond ? a : b`), the WGSL
codegen uses `select()` which is branchless — no control flow divergence, so barriers are safe. The
risk is if future compiler optimizations convert `select` into true `if/else` branches for
cost-based reasons (as `translateExpCore()` does in WASM — see Part 1, Codegen architecture). The
mitigation:

1. **Phase 3 compiler rule:** Barriers are **always emitted at the top-level block scope**, never
   inside WGSL `if/else/switch` blocks. The compiler hoists all barriers to the outermost scope.
2. **WGSL-specific:** Unlike WASM where cost-based branching (`if/else/end`) is beneficial for
   expensive arms, WGSL `select()` is the correct strategy anyway (all SIMD lanes execute both sides
   regardless — see Part 1 "WebGPU note" on Where branching). So `block_map` bodies will always use
   branchless `select`, making this a non-issue in practice.
3. **Static check:** The fused shader compiler rejects bodies that would require barriers inside
   divergent control flow. In practice this should never trigger because the WGSL codegen path
   doesn't emit `if/else` for elementwise ops.

### 3.4: Handling reductions inside blocks

A body that contains a reduction kernel (e.g., `sum(block)`) requires a **tree reduction** within
the workgroup. This reuses the exact pattern already implemented in `pipelineSource()` for
cooperative group reductions (`groups > 1`):

1. Each thread computes a partial sum over its assigned chunk.
2. Writes partial to `var<workgroup> shmem[tidx]`.
3. `workgroupBarrier()`.
4. Tree reduction: halve active threads each step, `shmem[tidx] += shmem[tidx + stride]`.
5. Thread 0 has the final result.

This is the building block for fused LayerNorm (mean reduction → normalize) and fused softmax (max
reduction → exp → sum reduction → divide).

### 3.5: Handling the Kogge-Stone pattern (cross-thread reads)

The Kogge-Stone prefix scan is special: at each round, thread `i` reads from thread `i - stride`.
This is a **cross-thread** data dependency within the workgroup, not just a write-then-read
dependency.

**Design decision: explicit `lax.workgroupAssociativeScan` primitive, not pattern matching.**

The v1 approach of detecting `shrink` + `fn` + `concatenate` in the body jaxpr is rejected. IR
pattern matching is fragile — simplification passes, constant folding, or reordering can change the
shape of the shrink equations, silently breaking the matcher and falling back to slow loops.
Instead, we introduce an explicit low-level primitive:

```ts
// Only valid inside a block_map body
lax.workgroupAssociativeScan(fn, elems);
```

**Why this is better:**

1. **Deterministic tracing.** The body jaxpr contains an explicit
   `Primitive.WorkgroupAssociativeScan` equation with the scan operator as a sub-jaxpr. No guessing.
   No pattern fragility.
2. **Clear compiler contract.** The fused shader compiler sees `WorkgroupAssociativeScan` and emits
   the multi-round Kogge-Stone loop with `workgroupBarrier()` between rounds, ping-pong
   shared-memory arrays, and per-thread `shmem[tidx - stride]` indexed reads. This is exactly the
   code structure from M7.4's `assocScanFusedShaderSource()` — but triggered by the primitive, not
   hand-coded.
3. **Error detection.** If `lax.workgroupAssociativeScan()` appears outside a `block_map` body, it
   throws immediately — no silent fallback to broken global-dispatch code.
4. **Composability.** Users who write `lax.associativeScan` get the Phase 5 decomposition
   (block_map + inter-block carry) automatically. Users who need explicit control write
   `lax.workgroupAssociativeScan` directly inside a `block_map` body.

**Primitive registration:**

```ts
// In Primitive enum:
WorkgroupAssociativeScan = "workgroup_associative_scan",

// Params:
{ jaxpr: Jaxpr; numConsts: number }
```

**Compiler behavior:**

| Context                           | What happens                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Inside `block_map` fused shader   | Emits Kogge-Stone loop in WGSL with barriers and ping-pong shmem                                                   |
| Inside `block_map` eager fallback | Evaluates using `Primitive.WorkgroupAssociativeScan` eager rule (delegates to `associativeScanCore` for the block) |
| Outside `block_map` (eager)       | Eager impl rule runs `associativeScanCore` — correct result, useful for testing                                    |
| Outside `block_map` (JIT)         | JIT compiler asserts `WorkgroupAssociativeScan` is inside a `BlockMap` body; throws compile-time error if not      |

To prevent escaping into standard global dispatches, the JIT compiler (`splitGraphDataflow` and
`jit.ts`) simply asserts that no `Primitive.WorkgroupAssociativeScan` ever reaches the standard
dispatch queue; if it isn't claimed by a `blockMapFusedShaderSource` path, it throws a compile-time
error. This avoids fragile module-level tracing flags entirely.

**Note on eager evaluation:** `workgroupAssociativeScan` must have an eager implementation in
`src/frontend/array.ts` (delegating to `associativeScanCore`) so that `block_map`'s eager-mode
slice/concat loop can successfully evaluate the compiled body Jaxpr for each block.

**Public API guard:** The public `lax.workgroupAssociativeScan()` always traces to
`Primitive.WorkgroupAssociativeScan` — there is no module-level tracing flag. The safety net is
purely at compile time: if the JIT encounters `WorkgroupAssociativeScan` outside a `BlockMap` body,
it throws. In eager mode, the impl rule runs `associativeScanCore` directly (correct result, just
not shared-memory-accelerated). This design means calling `lax.workgroupAssociativeScan()` outside
`block_map` works in eager mode (useful for testing) but fails at JIT compile time — the desired
behavior.

**Impact on Phase 5:** The `associativeScan` decomposition in Phase 5 uses
`lax.workgroupAssociativeScan` for the per-block local scans, making the lowering explicit and
testable:

```ts
// Phase 5 decomposition (updated):
const localScans = lax.block_map(
  (block) => lax.workgroupAssociativeScan(fn, block), // explicit primitive
  xs,
  { blockShape: [256], inAxes: [0], outAxes: [0] },
);
```

**LOC impact:** ~40 additional lines for the primitive registration (core.ts, array.ts, jaxpr.ts)
and eager fallback. The fused shader codegen is the same work as before — it just has a clean entry
point now.

### 3.6: Architecture of the compiler

New file `src/backend/webgpu/block-map.ts`:

```ts
export function blockMapFusedShaderSource(
  device: GPUDevice,
  bodySteps: JitStep[],
  params: BlockMapParams,
  caps: BackendCapabilities,
): ShaderInfo | null {
  // 1. Budget check
  // 2. Classify steps: shmem allocation map
  // 3. For each execute step:
  //    - Inline kernel expression via translateAluOpToWgsl (reuse gen() for CSE)
  //    - Map buffer indices → shmem array indices
  //    - Emit workgroupBarrier() for dependencies
  // 4. Wrap in @compute @workgroup_size(blockShape[0], blockShape[1] || 1, blockShape[2] || 1)
  // 5. Return ShaderInfo with grid = gridShape
}
```

This reuses existing WGSL expression codegen (`translateAluOpToWgsl`, CSE via `gen()`) and
`ShaderInfo` dispatch infrastructure. The new work is:

- Mapping JitId buffer arguments to `var<workgroup>` declarations.
- Emitting barriers.
- Handling reductions within a block (tree reduction in shmem — same as `pipelineSource` groups).
- 2D block index calculation from `workgroup_id` for multi-axis tiling.

### 3.7: What the compiler does NOT handle (fallback to per-block dispatch)

- Bodies with routine steps (Sort, Cholesky, etc.).
- Bodies exceeding the shared memory budget.
- Bodies with symbolic sizes.
- Bodies requiring more storage bindings than `maxStorageBuffersPerShaderStage`.

### 3.8: WebGPU dispatch integration

In `src/backend/webgpu.ts`, add:

- `prepareBlockMap(bodySteps, params)` — compiles shader, creates pipeline + bind group layout.
- `dispatchBlockMap(prepared, inputs, outputs)` — encodes a single dispatch with M workgroups.

In `src/frontend/jit.ts`, when compiling a `Primitive.BlockMap` equation:

- If `backend.type === "webgpu"` and `blockMapFusedShaderSource` returns a `ShaderInfo` → emit a
  `"block_map_fused"` JitStep (single dispatch, shared memory).
- Otherwise → emit a `"block_map_dispatch"` JitStep (M × body steps, global memory).

### 3.9: WASM compiled block-loop (preserve performance)

A single WASM module iterates over M blocks, executing the body for each block with pointer
arithmetic. Structurally simpler than `codegenNativeScanGeneral()` — no carry threading, no
direct-write — just a block offset loop.

### Deliverables

| File                                    | Change                                                                                                                          | LOC est  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `src/backend/webgpu/block-map.ts` (new) | Fused shader compiler: shmem map, barriers, inline Dot/Reduce, WorkgroupAssociativeScan codegen, ForiLoop/DynamicSlice lowering | ~420     |
| `src/backend/webgpu.ts`                 | `prepareBlockMap()`, `dispatchBlockMap()`                                                                                       | ~100     |
| `src/frontend/core.ts`                  | `Primitive.WorkgroupAssociativeScan` enum, params, abstract eval                                                                | ~15      |
| `src/frontend/array.ts`                 | `WorkgroupAssociativeScan` eager impl (delegates to `associativeScanCore`)                                                      | ~15      |
| `src/frontend/jit.ts`                   | `block_map` JitStep types + compilation + execution                                                                             | ~80      |
| `src/frontend/scan-plan.ts`             | `BlockMapPlan` type                                                                                                             | ~30      |
| `src/backend/wasm.ts`                   | `codegenBlockMapLoop()`                                                                                                         | ~120     |
| `test/block-map.test.ts`                | WebGPU + WASM fused shader tests + WorkgroupAssociativeScan tests                                                               | ~140     |
| **Total**                               |                                                                                                                                 | **~920** |

---

## Phase 4: Tiled Matmul (The P1 Payoff) (~200 LOC)

**Goal:** Express matrix multiplication as a `block_map` over 2D output tiles, achieving the
long-standing P1 performance target (5–10× matmul speedup) without hand-writing WGSL.

### 4.1: Tiled matmul algorithm

For $C = A \times B$ where $A$ is $M \times K$, $B$ is $K \times N$:

```
for each output tile C[r:r+Br, c:c+Bc]:
  acc = zeros(Br, Bc)
  for k in range(0, K, Bk):
    tileA = A[r:r+Br, k:k+Bk]    // Load A tile into shmem
    tileB = B[k:k+Bk, c:c+Bc]    // Load B tile into shmem
    acc += tileA @ tileB           // Accumulate in shmem
  C[r:r+Br, c:c+Bc] = acc
```

This is a sequential scan over the K dimension (accumulating `acc`) combined with a spatial map over
the (M/Br) × (N/Bc) output tile grid. The inner `tileA @ tileB` is a small matmul that fits entirely
in shared memory.

### 4.2: Expression as block_map + fori_loop

```ts
// Pseudocode — Phase 4 lowering for matmul
const tiledMatmul = (A, B) => {
  const [M, K] = A.shape;
  const [_, N] = B.shape;
  const Br = 16,
    Bc = 16,
    Bk = 16;

  const C = lax.block_map(
    ({ A: aTile, B: bTile }) => {
      // aTile: [Br, K], bTile: [K, Bc]
      // Accumulate over K/Bk tiles using fori_loop — no reshape gymnastics needed.
      return lax.fori_loop(
        0,
        K / Bk,
        (k, acc) => {
          const a = lax.dynamic_slice(aTile, [np.array(0), k.mul(Bk)], [Br, Bk]);
          const b = lax.dynamic_slice(bTile, [k.mul(Bk), np.array(0)], [Bk, Bc]);
          return acc.add(matmul(a, b)); // Small Br×Bk × Bk×Bc in shmem
        },
        np.zeros([Br, Bc]),
      );
    },
    { A: A, B: B },
    {
      blockShape: [Br, Bc],
      inAxes: { A: [0, null], B: [null, 1] },
      outAxes: [0, 1],
    },
  );
  return C;
};
```

_(Handling non-divisible $K$):_ If the inner contraction dimension $K$ is not cleanly divisible by
$Bk$, the inputs $A$ and $B$ should be padded with zeros along their $K$ dimension _before_
traversing into `block_map`. This guarantees the WGSL `fori_loop` runs strictly within bounds
without introducing complex branch divergence or bounds-checked selective accumulations in the hot
innermost shader loop.

**How this compiles:** The `block_map` body traces to a jaxpr containing a `ForiLoop` equation. The
fused shader compiler (Phase 3) emits a WGSL `for` loop.

**Memory model clarification:** `aTile: [Br, K]` and `bTile: [K, Bc]` are **global-memory views**
(storage buffer bindings), NOT shared memory. They are too large for `var<workgroup>` when K is
sizable. Only the `Bk`-sized slices extracted by `dynamic_slice` each iteration are loaded into
shared memory. Each iteration of the WGSL `for` loop proceeds as:

1. **Global → shmem load:** `dynamic_slice(aTile, [0, k*Bk], [Br, Bk])` and
   `dynamic_slice(bTile, [k*Bk, 0], [Bk, Bc])` compile to cooperative global-memory reads into two
   `var<workgroup>` arrays (`shmem_a: array<f32, Br*Bk>`, `shmem_b: array<f32, Bk*Bc>`). Each thread
   loads one or more elements.
2. **workgroupBarrier()** — ensures all threads have finished loading before computation.
3. **Per-thread contraction:** The `matmul(a, b)` becomes an inline accumulation loop (see §4.3).
   Thread `(tr, tc)` accumulates `acc[tr, tc] += shmem_a[tr,k] * shmem_b[k,tc]` for `k in 0..Bk`.
   The accumulator `acc` lives in `var<private>` registers — it is per-thread, not shared.
4. **workgroupBarrier()** — ensures no thread starts the next iteration's global load before all
   threads finish reading shmem from this iteration.

After the final iteration, each thread writes its `acc` element to the output storage buffer.

The compiler's `inAxes: { A: [0, null], B: [null, 1] }` declaration tells it that A is tiled along
axis 0 (rows) and B along axis 1 (columns), with the K dimension (`null` axes) passed through in
full as global-memory views. The `dynamic_slice` offsets compile to base-address arithmetic into
these global storage bindings, not into shmem.

### 4.3: Inline lowering of small matmul inside block_map

**Risk identified in review:** How does the Phase 3 compiler know to lower the `matmul(a, b)` call
inside the block body into an inline WGSL thread-level contraction loop, instead of treating it as a
standard `Dot` kernel step that trips the per-block dispatch fallback?

**Answer:** The body function traces through `makeJaxpr` like any other function. Inside the
`fori_loop` body, the `matmul` call produces a `Primitive.Dot` equation in the loop body jaxpr. The
Phase 3 compiler handles this through a dispatch table of primitives with inline WGSL lowerings:

1. **Detection:** When the fused shader compiler encounters a `Primitive.Dot` equation whose inputs
   are in shared memory, it recognizes this as a block-local matmul.

2. **Inline WGSL contraction:** Instead of dispatching a separate kernel, the compiler emits an
   inline accumulation loop in WGSL. For a `Br×Bk @ Bk×Bc` multiply, each thread computes one or
   more output elements of the `Br×Bc` result:

   ```wgsl
   // Thread (tr, tc) computes C[tr, tc]
   var acc: f32 = 0.0;
   for (var k: u32 = 0; k < Bk; k++) {
     acc += shmem_a[tr * Bk + k] * shmem_b[k * Bc + tc];
   }
   shmem_c[tr * Bc + tc] = acc;
   ```

3. **Thread mapping for 2D tiles:** With `blockShape: [Br, Bc]`, the workgroup has `Br × Bc`
   threads. Thread `(tr, tc)` is responsible for output element `C[tr, tc]`. The loop over `k` is
   sequential per thread — this is the standard tiled matmul pattern.

4. **Generalization — block-local built-ins:** The compiler maintains a map of primitives that have
   inline WGSL lowerings when operating on block-local (shared-memory) operands:

   | Primitive                  | Inline WGSL                                                      | Notes                         |
   | -------------------------- | ---------------------------------------------------------------- | ----------------------------- |
   | `Dot` (matmul)             | Per-thread contraction loop                                      | As above                      |
   | `Reduce` (sum/max)         | Workgroup tree reduction                                         | Already in `pipelineSource()` |
   | `ForiLoop`                 | WGSL `for` loop with carry in `var<private>` or `var<workgroup>` | Phase 1b primitive            |
   | `DynamicSlice`             | Offset arithmetic into shmem array                               | Phase 1b primitive            |
   | `WorkgroupAssociativeScan` | Kogge-Stone loop with barriers                                   | Phase 3.5 primitive           |
   | Future: `Conv`             | Sliding window in shmem                                          | Not Phase 4                   |

   Primitives without an inline lowering that appear in a fused block_map body cause a fallback to
   per-block dispatch — the correct behavior. The set of inline-lowerable primitives grows over
   time.

**Key insight:** This is not pattern matching — it's a well-defined dispatch table. The body jaxpr
contains explicit `Primitive.Dot` equations with known input shapes. The compiler checks: "Is this a
`Dot` with both inputs in shmem and small enough dimensions?" If yes, inline. If no, fall back. This
is deterministic and easy to test.

### 4.4: Performance expectations

| Matrix size | Current (naive) | Tiled (block_map, Br=Bc=16) | Expected speedup |
| ----------- | --------------- | --------------------------- | ---------------- |
| 256×256     | ~0.5 GFLOP/s    | ~2–5 GFLOP/s                | 4–10×            |
| 1024×1024   | ~2 GFLOP/s      | ~10–20 GFLOP/s              | 5–10×            |
| 2048×2048   | ~3 GFLOP/s      | ~15–30 GFLOP/s              | 5–10×            |

**P1 acceptance criterion:** 2048×2048 f32 matmul ≥ 40% of theoretical GFLOP/s.

### 4.5: Integration path

If the tiled matmul via `block_map` meets the performance target, the existing `Dot` primitive's
WebGPU dispatch can be lowered to `block_map` for large matrices (e.g., both dims ≥ 128). Small
matrices continue using the current row×col accumulation (lower dispatch overhead).

### Deliverables

| File                                          | Change                                         | LOC est  |
| --------------------------------------------- | ---------------------------------------------- | -------- |
| `src/library/lax.ts` or `src/frontend/jit.ts` | Matmul → block_map lowering for large matrices | ~80      |
| `test/block-map.test.ts`                      | Tiled matmul correctness tests                 | ~60      |
| `bench/block-map-matmul.bench.ts`             | Performance benchmarks at various sizes        | ~60      |
| **Total**                                     |                                                | **~200** |

---

## Phase 5: AssociativeScan Lowering (~220 LOC)

**Goal:** Express `associativeScan` as `block_map` + inter-block carry propagation, using the
standard parallel prefix scan decomposition (Blelloch 1990).

### 5.1: Decomposition

```
Step 1: Local prefix scans — M blocks, each scanned via block_map + workgroupAssociativeScan
Step 2: Extract block summaries — last element of each block
Step 3: Scan summaries — dynamic dispatch on M elements
Step 4: Apply summaries — block_map to add summary[i-1] to block[i]
```

_(Note: Rather than explicitly evaluating this pattern into the Jaxpr via recursive frontend library
calls like `lax.associativeScan`, this is established directly during JIT backend lowering. See
Section 5.4 for details)._

### 5.2: Base case

When `N ≤ blockShape[0]`, a single `block_map` call runs the `workgroupAssociativeScan` body on one
block.

### 5.3: Dispatch count comparison

If `block_map` executes as a single WebGPU dispatch over `M=ceil(N/B)` workgroups:

- Step 1 (local scans): 1 dispatch (M workgroups, each running Kogge-Stone internally via barriers).
- Step 2 (gather): 1 dispatch — extract last element per block into summary buffer.
- Step 3 (scan summaries): ⌈log₂ M⌉ Kogge-Stone rounds, each a separate dispatch (reuses existing
  per-round `dispatchKoggeStoneRound` pattern). Future optimization: when M ≤ B, fuse all rounds
  into a single workgroup dispatch via `workgroupAssociativeScan` on the summary buffer.
- Step 4 (apply): 1 dispatch (M−1 workgroups, adding prefix sums to each block).

Total for $N \le B^2$ (e.g., $N \le 65536$ for $B=256$): ~5 dispatches (1 local + 1 gather + summary
rounds + 1 apply). When $N \le B$ (single block), only 1 dispatch.

| N     | B=256 | M=⌈N/B⌉ | Local scan | Gather | Summary rounds | Apply | **Total** | Current M7.4 | Winner          |
| ----- | ----- | ------- | ---------- | ------ | -------------- | ----- | --------- | ------------ | --------------- |
| 256   | 256   | 1       | 1          | 0      | 0              | 0     | **1**     | 8            | block_map 8×    |
| 512   | 256   | 2       | 1          | 1      | 1              | 1     | **4**     | 9            | block_map ~2.2× |
| 1000  | 256   | 4       | 1          | 1      | 2              | 1     | **5**     | 10           | block_map 2×    |
| 4096  | 256   | 16      | 1          | 1      | 4              | 1     | **7**     | 12           | block_map ~1.7× |
| 65536 | 256   | 256     | 1          | 1      | 8              | 1     | **11**    | 16           | block_map ~1.5× |

**Decision:** The WebGPU `block_map` decomposition theoretically wins on dispatch counts for almost
all practical values of N. Before replacing the existing M7.4 fused shader pipeline globally, this
should be benchmark-gated against real hardware to verify that memory traffic, barrier overhead, and
optimal occupancy hold up.

**General formula:** Total dispatches per recursion level = 1 (local) + 1 (gather) + ⌈log₂ M⌉
(summary rounds) + 1 (apply), where M = ⌈N/B⌉. For single-level (N ≤ B²): 3 + ⌈log₂ M⌉ dispatches.
For two levels (N > B²): the summary scan itself recurses, roughly doubling the count. Still
competitive for any practical N.

### 5.4: Addressing WASM Polymorphic Shapes & Recursion (Late Lowering)

**The Problem:** If the array length `N` is symbolic (e.g., `SymDim("T")`), the trace-time recursion
`if (N <= B)` inside `associativeScan` cannot resolve — `N` is a tracer, not a number. Static
unrolling to max depth + masking would be functionally correct, but would force small arrays (e.g.,
N=10) to execute the same number of tree levels as N=65536, wasting cycles on no-op masked rounds.

**The Solution: Late Lowering — keep `Primitive.AssociativeScan` in the Jaxpr, decompose at
execution time.**

The frontend emits a single `Primitive.AssociativeScan` equation with its body sub-jaxpr. The
recursive block decomposition happens _inside the backend executor_, where `N` is always concrete.
This is exactly what we already do today: `planAssociativeScan()` in `scan-plan.ts` compiles the
body program once, and the executor in `scan-executor.ts` receives `N` as a concrete runtime value
resolved from `dimBindings`.

The `block_map` infrastructure enters as a _compilation target_ within the existing plan/execute
split, not as a frontend-level macro expansion.

#### WASM: Single compiled module with dynamic block-loop

This builds directly on the existing `codegenNativeAssociativeScan()` pattern (currently ~300 LOC in
`wasm.ts`), which already compiles the full Kogge-Stone ladder into a single WASM function taking
`N` as a runtime `i32` parameter. The block_map version restructures this into a three-level WASM
loop:

```
WASM function assoc_scan_blocked(N: i32, B: i32, ...ptrs):
  ;; Level 1: Per-block local prefix scans
  M = ceil(N / B)
  for block_idx = 0..M-1:
    start = block_idx * B
    end = min(start + B, N)
    ;; Kogge-Stone on [start, end) — same kernel body code as today
    for stride = 1; stride < (end - start); stride *= 2:
      for i = start+stride..end-1:
        result[i] = fn(result[i-stride], result[i])

  ;; Level 2: Scan over M block summaries (extract last elem per block)
  ;;   This is a tiny scan of length M — just inline Kogge-Stone
  for stride = 1; stride < M; stride *= 2:
    for b = stride..M-1:
      summary[b] = fn(summary[b-stride], summary[b])

  ;; Level 3: Apply summaries to each block (skip block 0)
  for block_idx = 1..M-1:
    for i in block:
      result[i] = fn(summary[block_idx-1], result[i])
```

**Why this works for polymorphic N:** All three loops use `N` (and `M = ceil(N/B)`) as runtime
bounds — standard WASM `loop`/`br_if` constructs. No masking. For N=10, B=256: M=1, Levels 2–3 are
zero iterations; Level 1 runs one 10-element Kogge-Stone scan. For N=100000, B=256: M=391, all three
levels execute with exact bounds. The compiled WASM module is identical for both — only the runtime
`i32` argument changes.

**Concrete integration point:** `planAssociativeScan()` currently returns
`{ path: "compiled-loop", executable, params }`. The new path extends this:

```ts
// In planAssociativeScan(), after classifying body steps:
if (backend.type === "wasm") {
  // Reuse existing step classification and kernel reindexing (unchanged)
  const params: NativeAssocScanBlockedParams = {
    ...existingParams,
    blockSize: 256, // or tuned per leaf size
  };
  const bytes = codegenBlockedAssociativeScan(params); // new codegen
  const module = new WebAssembly.Module(bytes);
  return { path: "compiled-loop-blocked", executable: new Executable(null, { module }), params };
}
```

The `codegenBlockedAssociativeScan` function reuses `emitKernelBody()` and `translateExpCore()` —
the same kernel codegen infrastructure used by `codegenNativeAssociativeScan()` today. The only new
WASM codegen is the three-level loop structure (~80 LOC), not the kernel body compilation.

**Parallelism (M7.3 compatible):** The existing `WasmWorkerPool` dispatches work by partitioning a
`(start, end)` range across threads. Each level's inner loop naturally partitions:

- Level 1: blocks are independent → partition blocks across workers.
- Level 2: M is small (typically <1000), runs single-threaded.
- Level 3: blocks are independent → partition blocks across workers.

#### WebGPU: Reuse fused shader with JS-orchestrated block decomposition

On WebGPU, the existing `assocScanFusedShaderSource()` already generates a single WGSL compute
shader that processes one Kogge-Stone round per dispatch, with the JS executor driving
`ceil(log₂ N)` dispatches (ping-pong, uniform buffer with stride/N per round).

The block*map version restructures the \_JS executor* (`dispatchAssocScan()` in `webgpu.ts`), not
the shader:

```ts
// In executeAssociativeScan(), webgpu-fused-blocked path:
function dispatchBlockedAssocScan(prepared, params, N, B, constSlots, elemSlots, outputSlots) {
  const M = Math.ceil(N / B);
  const enc = device.createCommandEncoder();

  // Step 1: Local scans (one fused block_map dispatch)
  //   The block_map shader processes B elements per workgroup.
  //   M workgroups → 1 GPU dispatch.
  //   Internally: ceil(log₂ B) compute passes within the single shader
  //   using workgroupBarrier() between Kogge-Stone rounds.
  //   This is the Phase 3 fused shader with WorkgroupAssociativeScan body.
  dispatchBlockMapFused(enc, localScanPipeline, elemSlots, M);

  // Step 2: Extract last element per block → summary buffer (1 dispatch)
  dispatchGather(enc, localResults, summaryBuf, B, M);

  // Step 3: Scan M summaries — one dispatch per Kogge-Stone round.
  //   Reuses the existing per-round dispatch pattern.
  //   Future optimization: when M <= B, fuse into single workgroupAssociativeScan dispatch.
  const summaryRounds = Math.ceil(Math.log2(M));
  for (let r = 0; r < summaryRounds; r++) {
    dispatchKoggeStoneRound(enc, summaryPingPong, r, M);
  }

  // Step 4: Apply summaries to blocks (1 dispatch, M-1 workgroups)
  dispatchBlockMapFused(enc, applyPipeline, [localResults, summaries], M - 1);

  device.queue.submit([enc.finish()]);
}
```

**Why this works for polymorphic N:** The JS orchestrator resolves `N` from `dimBindings` before
entering `dispatchBlockedAssocScan`. The shader pipelines are compiled once for block size `B` (at
`planAssociativeScan` time). The runtime `N` only affects `M = ceil(N/B)`, which determines the grid
dispatch size — a standard dynamic parameter that `dispatchWorkgroups()` accepts. No new shader
compilation is needed per `N`.

**Integration with existing infrastructure:** The WebGPU path reuses:

- `assocScanFusedShaderSource()` for the per-block local scan shader (existing M7.4 code).
- `calculateGrid()` for 2D grid splitting when M > 65535.
- `PreparedWebGPUAssocScan` pipeline caching (same shader, different grid size per N).
- The uniform buffer pattern (stride + N per round) already used for the flat Kogge-Stone.

The new work is the JS orchestrator restructuring (~60 LOC) and the two additional trivial shaders:
gather-last-element and apply-summary (each ~20 LOC WGSL, reusable).

#### Dispatch count analysis (updated)

| N     | B=256 | M=⌈N/B⌉ | Local scan                                     | Gather | Summary rounds | Apply | **Total** |
| ----- | ----- | ------- | ---------------------------------------------- | ------ | -------------- | ----- | --------- |
| 10    | 256   | 1       | 1 (single workgroup, log₂(10)=4 rounds inside) | 0      | 0              | 0     | **1**     |
| 256   | 256   | 1       | 1                                              | 0      | 0              | 0     | **1**     |
| 1000  | 256   | 4       | 1                                              | 1      | 2 rounds       | 1     | **5**     |
| 4096  | 256   | 16      | 1                                              | 1      | 4 rounds       | 1     | **7**     |
| 65536 | 256   | 256     | 1                                              | 1      | 8 rounds       | 1     | **11**    |

For comparison, the current flat M7.4 Kogge-Stone uses `ceil(log₂ N)` dispatches directly: 16 for
N=65536, 10 for N=1000. The blocked approach wins for large N and ties for small N, with the crucial
advantage that the per-block work uses shared memory (Phase 3) for much higher throughput per
dispatch.

### Deliverables

| File                            | Change                                                                                     | LOC est  |
| ------------------------------- | ------------------------------------------------------------------------------------------ | -------- |
| `src/backend/wasm.ts`           | `codegenBlockedAssociativeScan()`: three-level WASM loop reusing existing `emitKernelBody` | ~120     |
| `src/backend/webgpu.ts`         | `dispatchBlockedAssocScan()`: JS orchestrator + gather/apply shaders                       | ~100     |
| `src/frontend/scan-plan.ts`     | New `"compiled-loop-blocked"` / `"webgpu-fused-blocked"` plan paths                        | ~40      |
| `src/frontend/scan-executor.ts` | Dispatch to new plan paths                                                                 | ~20      |
| `test/block-map.test.ts`        | Blocked assocScan tests (polymorphic N, small N, large N)                                  | ~80      |
| **Total**                       |                                                                                            | **~360** |

---

## Phase 6: Further Consumers (Future — Not Scoped)

These are not part of the initial implementation but demonstrate why `block_map` is a strategic
investment:

### 6.1: Flash attention

Standard scaled dot-product attention:
$\text{Attention}(Q, K, V) = \text{softmax}(Q K^T / \sqrt{d_k}) V$

FlashAttention tiles over the sequence dimension:

```ts
const flashAttention = (Q, K, V) =>
  lax.block_map(
    (qBlock) => {
      const init = { max: np.full([], -Infinity), sum: np.zeros([]), acc: np.zeros([Br, d]) };
      // Iterate over K,V blocks using fori_loop (compiles to WGSL for loop in shmem)
      return lax.fori_loop(
        0,
        N / Bc,
        (j, state) => {
          const kBlock = lax.dynamic_slice(K, [j.mul(Bc), np.array(0)], [Bc, d]);
          const vBlock = lax.dynamic_slice(V, [j.mul(Bc), np.array(0)], [Bc, d]);
          const scores = matmul(qBlock, kBlock.T).div(np.sqrt(dk));
          const newMax = np.maximum(state.max, scores.max(-1));
          // Online softmax correction + accumulate...
          return { max: newMax, sum: newSum, acc: correctedAcc };
        },
        init,
      ).acc;
    },
    Q,
    { blockShape: [Br], inAxes: [0], outAxes: [0] },
  );
```

**Why block_map matters here:** Without it, the intermediate `Q @ K^T` for the full sequence must be
materialized (O(N²) memory). With block_map, only the `Br × Bc` attention block lives in shared
memory at any moment.

### 6.2: Fused LayerNorm

```ts
const fusedLayerNorm = (x, gamma, beta) =>
  lax.block_map(
    (block) => {
      // All in shared memory — single dispatch
      const mean = block.mean(-1, true);
      const variance = block.sub(mean).pow(2).mean(-1, true);
      return block
        .sub(mean)
        .div(np.sqrt(variance.add(1e-5)))
        .mul(gamma)
        .add(beta);
    },
    x,
    { blockShape: [featureDim], inAxes: [0], outAxes: [0] },
  );
```

Current jax-js requires 2 dispatches (reduction pass + normalize pass). With block_map: 1 dispatch,
because the mean and variance live in `var<workgroup>` between the reduction and the normalization.

### 6.3: Chunked Mamba / S4 inference

State-space models (Mamba, S4, Griffin) chunk the sequence and run a sequential scan per chunk in
fast local memory. The per-chunk scan state is propagated between chunks:

```ts
const chunkedSSM = (xs, A, B, C) => {
  // Step 1: Per-chunk local scans in shared memory
  const localStates = lax.block_map(
    (chunk) =>
      lax.fori_loop(
        0,
        chunkSize,
        (i, state) => ssmStep(state, lax.dynamic_index(chunk, [i], [1])),
        initState,
      ),
    xs,
    { blockShape: [chunkSize], inAxes: [0], outAxes: [0] },
  );
  // Step 2: Carry propagation between chunks (small sequential scan)
  const globalCarries = lax.scan(carryProp, init, localStates)[1];
  // Step 3: Apply global carries to local results
  return lax.block_map(
    ({ local, carry }) => applyCarry(local, carry),
    { local: localResults, carry: globalCarries },
    { blockShape: [chunkSize], inAxes: { local: [0], carry: [0] }, outAxes: [0] },
  );
};
```

### 6.4: Replacing hand-written WGSL routines

Over time, routines in `routines.ts` could be **optionally** lowered through `block_map` when the
body fits in shared memory:

| Routine                         | Lines today | block_map body                       | Savings  |
| ------------------------------- | ----------- | ------------------------------------ | -------- |
| Bitonic sort (single workgroup) | ~200        | ~30 (bitonic pass as shrink+min+max) | ~170 LOC |
| Cholesky (small n)              | ~260        | ~40 (loop body via fori_loop)        | ~220 LOC |

This is NOT a near-term goal — hand-written WGSL will remain faster for some time. But it becomes an
option once the compiler matures.

---

## Phase 7: Code Deletion (Conditional on Performance Gates)

### M7.4 fused assocScan (~1,074 LOC)

Delete only if Phase 5's block_map assocScan matches within 20% of M7.4 performance at N=256, N=1000
on both Deno wgpu-rs and Chromium.

### Current naive matmul dispatch

Replace only if Phase 4's tiled matmul meets the P1 criterion (2048×2048 ≥ 40% theoretical GFLOP/s).

**Do NOT delete in the same PR that adds block_map.** Deletion is a separate, benchmark-gated PR.

---

## Consolidated Test Matrix

This section collects every test case from Phases 0–5 into a single matrix, organized by category.
Each test lists: the phase it belongs to, which backends it must pass on, and what hazard it guards
against.

**Legend:** W = WASM, G = WebGPU, C = CPU (eager), A = All three.

### T1: Phase 0 — Tracing validation (`test/block-map-prototype.test.ts`, ~8 tests)

| #    | Test                                                      | Backends | Validates                                        |
| ---- | --------------------------------------------------------- | -------- | ------------------------------------------------ |
| T1.1 | 1D elementwise body traces to blockSize-independent jaxpr | A        | Body does not unroll to O(blockSize) equations   |
| T1.2 | 2D body (axes=[0,1]) traces to consistent jaxpr           | A        | Multi-axis tracing                               |
| T1.3 | Reduction body (sum) traces correctly                     | A        | Reduction inside block doesn't break shapes      |
| T1.4 | Body with constants traces correctly                      | A        | Constants propagate through tracing              |
| T1.5 | Body with pytree input/output traces correctly            | A        | Pytree flattening during trace                   |
| T1.6 | `blockShape` does not appear in equation shapes           | A        | **Kill-signal:** if this fails, stop the project |
| T1.7 | Body with fori_loop traces correctly                      | A        | Sub-primitive tracing (Phase 1b preview)         |
| T1.8 | Body with two outputs traces correctly                    | A        | Multi-output bodies                              |

### T2: Phase 1 — Core IR + Eager correctness (`test/block-map.test.ts`, ~90 LOC)

| #     | Test                                                  | Backends | Validates                                    |
| ----- | ----------------------------------------------------- | -------- | -------------------------------------------- |
| T2.1  | Identity body — `f(block) = block`                    | A        | Tiling + reassembly roundtrip                |
| T2.2  | Elementwise body — `f(block) = block * 2`             | A        | Basic computation inside block               |
| T2.3  | Non-divisible N (N=10, blockShape=[4])                | A        | Padding + trimming of last block             |
| T2.4  | 2D tiling (axes=[0,1], blockShape=[4,4])              | A        | Multi-axis slice/concat                      |
| T2.5  | Pytree elems (object with two arrays)                 | A        | `tree.flatten`/`tree.unflatten` in loop      |
| T2.6  | Length-0 edge case (N=0)                              | A        | Empty input doesn't crash                    |
| T2.7  | Single-element input (N=1)                            | A        | Degenerate block count (1 block)             |
| T2.8  | Leak regression: padded blocks                        | A        | Padding intermediates disposed after trim    |
| T2.9  | Leak regression: pytree intermediates                 | A        | `tree.flatten` temporaries don't accumulate  |
| T2.10 | Leak regression: non-divisible N with `using` in body | A        | Per-block intermediates freed each iteration |

### T3: Phase 1b — Loop primitives (`test/block-map.test.ts`, ~60 LOC)

| #    | Test                                                         | Backends | Validates                            |
| ---- | ------------------------------------------------------------ | -------- | ------------------------------------ |
| T3.1 | `fori_loop(0, 5, f, init)` — accumulator loop                | A        | Basic carry-only loop                |
| T3.2 | `fori_loop` with `dynamic_slice` read inside body            | A        | Slice indexing correctness           |
| T3.3 | `dynamic_slice` standalone — contiguous read                 | A        | TypedArray offset (eager)            |
| T3.4 | `dynamic_slice` standalone — out-of-bounds clamping          | A        | JAX-compatible boundary semantics    |
| T3.5 | `fori_loop` inside `block_map` body                          | A        | Nested primitive tracing + execution |
| T3.6 | `fori_loop(0, 0, f, init)` — zero iterations                 | A        | Empty loop returns init unchanged    |
| T3.7 | `dynamic_slice` with non-zero start indices on multiple axes | A        | Multi-axis slicing                   |

### T4: Phase 2 — Autodiff + Vmap (`test/block-map.test.ts`, ~70 LOC)

| #    | Test                                                    | Backends | Validates                        |
| ---- | ------------------------------------------------------- | -------- | -------------------------------- |
| T4.1 | `grad(sum(block_map(f, xs)))` vs finite differences     | A        | JVP rule correctness             |
| T4.2 | `jit(grad(block_map(f, xs)))` — no leaks                | A        | AD + JIT memory discipline       |
| T4.3 | `vmap(block_map(f, xs))` — batched execution            | A        | Vmap axis shifting               |
| T4.4 | `grad` through body with reduction (`sum` inside block) | A        | Transpose rule handles reduction |
| T4.5 | `grad(block_map)` with `fori_loop` inside body vs FD    | A        | AD through nested primitives     |
| T4.6 | `jvp(block_map)` tangent shapes match primal shapes     | A        | JVP doubling correctness         |
| T4.7 | `grad(block_map)` with non-divisible N                  | A        | AD + padding interaction         |

### T5: Phase 3 — Fused shader compiler (`test/block-map.test.ts`, ~140 LOC)

#### T5a: WebGPU fused shader path

| #     | Test                                                                   | Backends | Validates                      |
| ----- | ---------------------------------------------------------------------- | -------- | ------------------------------ |
| T5a.1 | Elementwise chain: `x * 2 + 1` fused into single dispatch              | G        | Basic fusion                   |
| T5a.2 | Two-step body with data dependency → barrier inserted                  | G        | `workgroupBarrier()` placement |
| T5a.3 | Body exceeding shmem budget → per-block dispatch fallback              | G        | Graceful degradation           |
| T5a.4 | Body exceeding `maxComputeInvocationsPerWorkgroup` → error or fallback | G        | Thread limit guard             |
| T5a.5 | Reduction body (sum) → tree reduction in shared memory                 | G        | Cooperative reduction codegen  |
| T5a.6 | 2D tiling: `blockShape=[16,16]` fused shader                           | G        | Multi-dim workgroup mapping    |
| T5a.7 | Non-divisible N with fused shader (padding correctness)                | G        | Edge-block masking             |
| T5a.8 | `jit(block_map(f, xs))` matches eager result                           | G        | JIT vs eager agreement         |

#### T5b: WorkgroupAssociativeScan primitive

| #     | Test                                                            | Backends | Validates                      |
| ----- | --------------------------------------------------------------- | -------- | ------------------------------ |
| T5b.1 | `workgroupAssociativeScan(add, elems)` inside block_map         | G        | Kogge-Stone codegen + barriers |
| T5b.2 | `workgroupAssociativeScan` outside block_map → throws           | A        | Guard against misuse           |
| T5b.3 | `workgroupAssociativeScan` eager impl matches `associativeScan` | A        | Eager fallback correctness     |
| T5b.4 | `workgroupAssociativeScan` with non-power-of-2 block size       | G        | Masking in Kogge-Stone rounds  |
| T5b.5 | `workgroupAssociativeScan(mul, elems)` — non-add operator       | G        | Operator generality            |

#### T5c: WASM compiled block-loop

| #     | Test                                              | Backends | Validates                    |
| ----- | ------------------------------------------------- | -------- | ---------------------------- |
| T5c.1 | Elementwise body compiled into single WASM module | W        | `codegenBlockMapLoop()`      |
| T5c.2 | Body with reduction compiled correctly            | W        | WASM tree-reduction in block |
| T5c.3 | `jit(block_map(f, xs))` matches eager (WASM)      | W        | WASM JIT vs eager agreement  |
| T5c.4 | Block-loop with non-divisible N                   | W        | WASM padding/masking         |

### T6: Phase 4 — Tiled matmul (`test/block-map.test.ts`, ~60 LOC + bench)

| #    | Test                                                   | Backends | Validates                   |
| ---- | ------------------------------------------------------ | -------- | --------------------------- |
| T6.1 | 64×64 matmul via block_map matches `np.dot`            | G, W     | Correctness at small size   |
| T6.2 | 256×256 matmul via block_map matches `np.dot`          | G, W     | Correctness at medium size  |
| T6.3 | Non-square matmul (128×64) × (64×256)                  | G, W     | Rectangular tile handling   |
| T6.4 | `grad(tiled_matmul)` matches `grad(np.dot)`            | G, W     | AD through tiled lowering   |
| T6.5 | Small matmul (≤32×32) does NOT route through block_map | G, W     | Size gate prevents overhead |
| T6.6 | f16 matmul via block_map (if `shader-f16` available)   | G        | Half-precision support      |

#### T6-bench: Performance gates (`bench/block-map-matmul.bench.ts`)

| #     | Benchmark                                    | Target                | Kill signal       |
| ----- | -------------------------------------------- | --------------------- | ----------------- |
| T6b.1 | 256×256 f32 GFLOP/s                          | ≥40% theoretical peak | < 20% theoretical |
| T6b.2 | 512×512 f32 GFLOP/s                          | ≥40% theoretical peak | < 20% theoretical |
| T6b.3 | 1024×1024 f32 GFLOP/s                        | ≥40% theoretical peak | < 20% theoretical |
| T6b.4 | 2048×2048 f32 GFLOP/s                        | ≥40% theoretical peak | < 20% theoretical |
| T6b.5 | Tiled vs current `np.dot` speedup at 512×512 | ≥3×                   | < 1.5×            |

### T7: Phase 5 — AssociativeScan lowering (`test/block-map.test.ts`, ~80 LOC)

| #    | Test                                                       | Backends | Validates                 |
| ---- | ---------------------------------------------------------- | -------- | ------------------------- |
| T7.1 | `associativeScan(add, xs)` N=64 matches reference          | G, W     | Small N (single block)    |
| T7.2 | `associativeScan(add, xs)` N=1024 matches reference        | G, W     | Multi-block decomposition |
| T7.3 | `associativeScan(add, xs)` N=65536 matches reference       | G, W     | Two-level recursion       |
| T7.4 | `associativeScan(add, xs)` N=10 (non-power-of-2)           | G, W     | Irregular block sizes     |
| T7.5 | `associativeScan(mul, xs)` — non-add operator              | G, W     | Operator generality       |
| T7.6 | `grad(associativeScan)` via block_map matches current impl | G, W     | AD preservation           |
| T7.7 | Polymorphic N (`SymDim("T")`) — late lowering resolves     | W        | Symbolic dim support      |
| T7.8 | All 30 existing `lax-associative-scan.test.ts` tests pass  | G, W     | Regression suite          |
| T7.9 | Block_map assocScan ≥50% of M7.4 throughput for N≤256      | G, W     | Performance gate          |

### Cross-cutting test requirements

| Requirement                                           | How enforced                                                                  | Phases |
| ----------------------------------------------------- | ----------------------------------------------------------------------------- | ------ |
| **No memory leaks**                                   | All tests run under `checkLeaks` harness (`test/setup.ts`)                    | All    |
| **Eager = JIT agreement**                             | Explicit `jit(block_map(...))` vs `block_map(...)` comparison tests           | 1–5    |
| **Backend consistency**                               | Same test expectations for WASM and WebGPU (except backend-specific features) | 1–5    |
| **Transform stacking**                                | `jit(grad(block_map))`, `vmap(grad(block_map))`, `grad(vmap(block_map))`      | 2+     |
| **Benchmark gates at Decision Framework checkpoints** | Phase 4: ≥40% peak GFLOP/s. Phase 5: ≥50% of M7.4. Failure → stop.            | 4, 5   |

### Summary

| Phase     | Test file                          | Test count                   | LOC est              |
| --------- | ---------------------------------- | ---------------------------- | -------------------- |
| 0         | `test/block-map-prototype.test.ts` | ~8                           | ~60                  |
| 1         | `test/block-map.test.ts`           | ~10                          | ~90                  |
| 1b        | `test/block-map.test.ts`           | ~7                           | ~60                  |
| 2         | `test/block-map.test.ts`           | ~7                           | ~70                  |
| 3         | `test/block-map.test.ts`           | ~17                          | ~140                 |
| 4         | `test/block-map.test.ts` + bench   | ~6 + 5 bench                 | ~60 + ~60            |
| 5         | `test/block-map.test.ts`           | ~9                           | ~80                  |
| **Total** |                                    | **~64 tests + 5 benchmarks** | **~560 + ~60 bench** |

---

## Risk Assessment

### High risks

| Risk                                                    | Impact       | Mitigation                                                                                                                                                                                                                                                 |
| ------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 3 compiler more complex than estimated            | High         | Start with scalar elementwise chain; add complexity incrementally. Per-block dispatch as permanent fallback.                                                                                                                                               |
| Kogge-Stone pattern detection unreliable                | **Resolved** | Explicit `lax.workgroupAssociativeScan` primitive replaces fragile `shrink`+`concat` pattern matching. See Phase 3.5.                                                                                                                                      |
| Sequential loop inside block_map body has no fused path | **Resolved** | `fori_loop` primitive (Phase 1b) compiles to WGSL `for` loop inside fused shader. `lax.scan` inside block_map remains unsupported in fused context; use `fori_loop` for carry-only loops. See Phase 1b.                                                    |
| Tiled matmul lowering inside block_map                  | High         | `fori_loop` + `dynamic_slice` + inline `Dot` dispatch table. Deterministic: primitive-type check, not pattern matching. See Phase 4.3.                                                                                                                     |
| `maxComputeInvocationsPerWorkgroup` caps 2D tile size   | Medium       | Typically 256 threads. 16×16 fits; 32×32 does not. Per-thread work-tiling (1 thread computes multiple output elements) can increase tile size without exceeding the thread limit — but adds compiler complexity. Phase 4 starts with 16×16. See Phase 3.1. |
| Shared memory budget insufficient for large bodies      | Medium       | Runtime check; fallback to per-block dispatch. Measured: 2×2 Kalman=8 KB ✅, 4×4=32 KB ❌.                                                                                                                                                                 |
| WASM regression if assocScan routes through block_map   | High         | Phase 3.9 WASM compiled block-loop; don't delete M7.2/M7.3 until benchmarked.                                                                                                                                                                              |

### Medium risks

| Risk                                                | Impact | Mitigation                                                                                                                                                                                                         |
| --------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Barrier placement in divergent control flow         | Medium | Barriers always hoisted to top-level block scope. WGSL uses branchless `select()` for ternary ops (all SIMD lanes execute both sides). Compiler rejects bodies requiring barriers inside `if/else`. See Phase 3.3. |
| 2D tiling complicates codegen significantly         | Medium | Ship 1D tiling first (Phase 3). Extend to 2D only when Phase 4 demands it. **API is N-D from day one** (`blockShape: number[]`) — only codegen is 1D initially.                                                    |
| Body tracing produces O(blockSize) equations        | Medium | Phase 0 catches this early. Redesign body API if needed.                                                                                                                                                           |
| Eager fallback memory spikes from per-block slicing | Medium | `checkLeaks` integration from Phase 1. Explicit leak regression tests for padding, pytrees, and non-divisible N. See Phase 1.4.                                                                                    |

### Low risks

| Risk                                       | Impact       | Mitigation                                                                                                                                                                 |
| ------------------------------------------ | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `var<workgroup>` sizing with symbolic dims | **Resolved** | `blockShape` is always concrete `number[]`. Symbolic dims apply to outer array size (number of blocks), not block-local memory. Guard asserts concreteness. See Phase 3.2. |
| Non-divisible N (padding/trimming)         | Low          | Proven pattern from existing scan infrastructure.                                                                                                                          |
| API breakage                               | Low          | `block_map` is additive; all existing APIs unchanged.                                                                                                                      |
| Multi-output kernels in shared memory      | Low          | Defer to per-block dispatch initially.                                                                                                                                     |

---

## Total Effort Estimate

| Phase                  | Description                                                                | LOC est    | Dependencies |
| ---------------------- | -------------------------------------------------------------------------- | ---------- | ------------ |
| Phase 0                | Prototype (validate tracing)                                               | ~60        | None         |
| Phase 1                | Core IR + eager fallback                                                   | ~252       | Phase 0      |
| Phase 1b               | Loop primitives (fori_loop + dynamic_slice)                                | ~177       | Phase 1      |
| Phase 2                | AD rules                                                                   | ~170       | Phase 1      |
| Phase 3                | WebGPU shared-memory compiler + WASM block-loop + WorkgroupAssociativeScan | ~920       | Phase 1      |
| Phase 4                | Tiled matmul                                                               | ~200       | Phase 3, 1b  |
| Phase 5                | AssociativeScan lowering (late lowering)                                   | ~360       | Phase 3      |
| Phase 6                | Flash attention, LayerNorm, SSM, routines                                  | Future     | Phase 3      |
| Phase 7                | Code deletion (conditional)                                                | -1,074+    | Phase 4/5    |
| **Total (Phases 0–5)** |                                                                            | **~2,139** |              |
| **Net after Phase 7**  |                                                                            | **~1,065** |              |

**Critical path:** Phase 0 → Phase 1 → Phase 1b → Phase 3 → Phase 4 (tiled matmul) and Phase 5
(assocScan). Phase 2 can run in parallel with Phase 1b/3.

**Recommended execution order:**

1. **Phase 0** — 2 hours. De-risk tracing. Kill the project early if bodies don't trace cleanly.
2. **Phase 1** — Core IR. Makes the primitive real.
3. **Phase 1b** — `fori_loop` + `dynamic_slice`. Unblocks Phase 4 matmul and Phase 6 flash
   attention.
4. **Phase 2** — AD rules. Unblocks `grad(block_map(...))` for all downstream use cases.
5. **Phase 3** — The shared-memory compiler. The hard part. Ship 1D first.
6. **Phase 4** — Tiled matmul. The biggest single performance win in the project.
7. **Phase 5** — AssocScan lowering. Architectural cleanup.
8. Phase 6/7 — Future, gated on maturity.

---

## Decision Framework: When to Proceed

| Checkpoint    | Continue if...                                               | Stop if...                                                 |
| ------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| After Phase 0 | Body tracing produces clean, blockSize-independent jaxprs    | Bodies trace as O(blockSize) equations; shapes are dynamic |
| After Phase 1 | Eager fallback passes all tests; 2D tiling API feels natural | Fundamental tracing issue with multi-axis or pytrees       |
| After Phase 3 | Fused shader works for elementwise + reduction bodies        | Compiler complexity explodes; barriers incorrect           |
| After Phase 4 | Tiled matmul meets P1 target (≥40% theoretical GFLOP/s)      | Performance < 20% theoretical; overhead dominates          |
| After Phase 5 | AssocScan via block_map passes all 30 existing tests         | Performance < 50% of M7.4 for N≤256                        |

---

## Comparison with v1 Plan

| Aspect             | v1                                    | v2 (this plan)                                                                                                            |
| ------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Framing            | AssocScan optimization                | General-purpose shared-memory compute primitive                                                                           |
| `blockSize`        | `number` (1D only)                    | `blockShape: number[]` (N-D from day one)                                                                                 |
| Tiled matmul       | Not considered                        | Phase 4 (P1 performance goal)                                                                                             |
| Flash attention    | Not considered                        | Phase 6 (future consumer)                                                                                                 |
| Fused norms        | Not considered                        | Phase 6 (future consumer)                                                                                                 |
| State-space models | Not considered                        | Phase 6 (future consumer)                                                                                                 |
| LOC estimate       | ~1,430                                | ~2,139 (broader scope, 2D tiling, WorkgroupAssociativeScan + ForiLoop + DynamicSlice primitives, late-lowering assocScan) |
| Strategic value    | Low (1 consumer)                      | High (5+ consumers, solves P1)                                                                                            |
| Decision threshold | "Is it worth it for assocScan alone?" | "Does it unlock the next tier of GPU performance?"                                                                        |

---

## Files Modified (Complete List)

### New files

| File                               | Purpose                           |
| ---------------------------------- | --------------------------------- |
| `src/backend/webgpu/block-map.ts`  | Fused shader compiler (Phase 3)   |
| `test/block-map.test.ts`           | All block_map tests (Phases 1–5)  |
| `test/block-map-prototype.test.ts` | Phase 0 tracing validation        |
| `bench/block-map-matmul.bench.ts`  | Tiled matmul benchmarks (Phase 4) |

### Modified files

| File                                  | Phase    | Change                                                                                                                             |
| ------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/frontend/core.ts`                | 1, 1b, 3 | `BlockMap` + `ForiLoop` + `DynamicSlice` + `WorkgroupAssociativeScan` primitives, params, abstract eval                            |
| `src/library/lax.ts`                  | 1, 1b, 3 | `block_map()` + `fori_loop()` + `dynamic_slice()` + `workgroupAssociativeScan()` public API                                        |
| `src/frontend/array.ts`               | 1, 1b, 3 | Eager impl rules (BlockMap loop, ForiLoop JS loop, DynamicSlice TypedArray offset, WorkgroupAssociativeScan → associativeScanCore) |
| `src/frontend/jaxpr.ts`               | 1        | Abstract eval rule                                                                                                                 |
| `src/index.ts`                        | 1        | Export                                                                                                                             |
| `src/frontend/jvp.ts`                 | 1b, 2    | JVP rules (BlockMap, ForiLoop, DynamicSlice)                                                                                       |
| `src/frontend/linearize.ts`           | 1b, 2    | PE + transpose rules (BlockMap, ForiLoop, DynamicSlice)                                                                            |
| `src/frontend/vmap.ts`                | 1b, 2    | Vmap rules (BlockMap, ForiLoop, DynamicSlice)                                                                                      |
| `src/frontend/jit.ts`                 | 3        | BlockMap JitStep types + compilation + execution                                                                                   |
| `src/frontend/scan-plan.ts`           | 3        | `BlockMapPlan` type                                                                                                                |
| `src/backend/webgpu.ts`               | 3        | `prepareBlockMap()`, `dispatchBlockMap()`                                                                                          |
| `src/backend/wasm.ts`                 | 3        | `codegenBlockMapLoop()`                                                                                                            |
| `src/library/lax-associative-scan.ts` | 5        | Decomposition lowering                                                                                                             |
| `.github/copilot-instructions.md`     | 5        | Document block_map                                                                                                                 |
