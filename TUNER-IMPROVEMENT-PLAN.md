# Tuner Improvement Plan (Source-Referenced)

## Revised Assessment After SIMD Investigation

The wasmblr layer has **complete** SIMD instruction coverage — `V128`, `I32x4`, `F32x4`, `F64x2`
classes in `src/backend/wasm/wasmblr.ts` cover all standard WASM SIMD opcodes. The `WasmHl` class in
`src/backend/wasm/wasmblr-hl.ts` adds higher-level patterns: `loadF32x4`/`storeF32x4`, `f32x4Hsum`,
`f64x2Hsum`, `f32x4Splat`/`f64x2Splat`, `simdReductionF32`/`simdReductionF64`, and
`forLoopUnrolled`. These are proven in production by `src/backend/wasm/routines/cholesky-simd.ts`.

However, these helpers are **special-purpose** — they implement dot-product reduction patterns for
hand-written routines, not general elementwise vectorization. The gap is entirely in the kernel
codegen layer: `emitKernelBody()` always calls `tuneNullopt(kernel)` and `translateExpCore()` emits
purely scalar WASM. No new wasmblr instructions are needed. What's needed is a SIMD-aware code
generation mode that maps `AluExp` trees to v128 operations.

This changes priority ordering: SIMD vectorization of JIT kernels is a **codegen/tuner project**,
not an assembler project.

---

# Phase 0: Structural Prerequisites

These four restructurings are **zero-functional-change refactors** that must land before the 8
improvements below. They prevent architectural debt from accumulating as improvements are
implemented. All existing tests must pass — no behavior change whatsoever.

## R1: InstructionEmitter Strategy for `translateExpCore`

**Priority:** Phase 0 prerequisite for Improvement 5 (WASM SIMD)  
**Effort:** Medium  
**Status:** ⏭ Skipped — separate `translateExpCoreSimd()` used instead of refactoring `translateExpCore`

### Problem

Improvement 5 proposes parallel functions `translateExpSimd()`, `emitKernelBodySimd()`, and
`translateExpMegaSimd()` — tripling the expression translation surface area. Every future AluOp, CSE
change, or codegen fix would need to be replicated across all three. The mega-module's
`translateExpMega()` (`src/backend/wasm/mega-module.ts` line 978) is already a thin wrapper around
`translateExpCore` — adding a SIMD variant would make it four functions.

### Design

Define an `InstructionEmitter` interface that abstracts how arithmetic operations are lowered to
WASM instructions. `translateExpCore` takes an optional emitter parameter (default: scalar).

```ts
// In src/backend/wasm.ts (or a new wasm/emitter.ts)
export interface InstructionEmitter {
  /** Emit a binary arithmetic op. Values are already on the WASM stack. */
  emitBinaryArith(cg: CodeGenerator, op: AluOp, dtype: DType): void;
  /** Emit a unary op. Value is already on the WASM stack. */
  emitUnaryArith(cg: CodeGenerator, op: AluOp, dtype: DType): void;
  /** Emit a constant value. Leaves value on WASM stack. */
  emitConst(cg: CodeGenerator, dtype: DType, value: number): void;
  /** Emit a load from a buffer base pointer at the given gidx offset. */
  emitLoad(cg: CodeGenerator, baseLocal: number, gidxLocal: number, dtype: DType): void;
  /** Emit a store to a buffer base pointer at the given gidx offset. */
  emitStore(cg: CodeGenerator, baseLocal: number, gidxLocal: number, dtype: DType): void;
  /** Allocate a local variable for CSE caching. Returns the local index. */
  allocCSELocal(cg: CodeGenerator, dtype: DType): number;
  /** The step size for the gidx loop (1 for scalar, 4 for f32x4, 2 for f64x2). */
  readonly step: number;
}
```

The `scalarEmitter()` factory returns the current behavior extracted from `translateExpCore`:

```ts
export function scalarEmitter(): InstructionEmitter {
  return {
    emitBinaryArith(cg, op, dtype) {
      dty(cg, op, dtype).add(); /* ... dispatch */
    },
    emitConst(cg, dtype, value) {
      /* current cg.i32.const / cg.f32.const logic */
    },
    emitLoad(cg, base, gidx, dtype) {
      /* current typed load logic */
    },
    emitStore(cg, base, gidx, dtype) {
      /* current typed store logic */
    },
    allocCSELocal(cg, dtype) {
      return cg.addLocal(wasmDtype(dtype));
    },
    step: 1,
  };
}
```

A future `simdEmitter(width: 4 | 2)` would map `Add → f32x4.add`, loads → `v128.load`, etc.

### Why deferred

The interface definition is straightforward, but refactoring `translateExpCore` (260 lines with CSE,
GlobalIndex handling, 30+ AluOp cases) to call emitter methods instead of hardcoded `dty()` calls is
a significant edit. It should be done when Improvement 5 is implemented — that's when the SIMD
emitter actually exercises the abstraction and validates the interface design. Doing it earlier
risks designing the wrong interface.

### Scope when implemented

1. Define `InstructionEmitter` interface
2. Extract `scalarEmitter()` factory from current `translateExpCore` logic
3. Add `emitter?: InstructionEmitter` parameter to `translateExpCore`
4. Default to `scalarEmitter()` — zero behavior change
5. All existing callers (`translateExp`, `translateExpMega`, scan contexts) unchanged
6. Future: `simdEmitter()` + `tuneWasm()` + vectorized gidx loop in `emitKernelBody`

### Key files affected (when implemented)

| File                              | Change                                                   |
| --------------------------------- | -------------------------------------------------------- |
| `src/backend/wasm.ts`             | `InstructionEmitter` + `scalarEmitter()` + refactor core |
| `src/backend/wasm/mega-module.ts` | No change — `translateExpMega` wraps core, gets it free  |

---

## R2: Backend-Typed TuneResult Hierarchy

**Priority:** Phase 0 prerequisite for Improvements 3, 4, 5  
**Effort:** Small  
**Status:** ✅ Implemented

### Problem

The plan proposes adding `local`, `useSharedMemory`, `sharedMemoryBytes` (WebGPU-only) and
`simdWidth` (WASM-only) to `TuneResult`. These are backend-specific concerns being crammed into one
shared type. Any consumer of `TuneResult` would see all fields from all backends, creating confusion
about which fields are valid in which context.

### Design

Keep `TuneResult` as the base interface. Add `WebGPUTuneResult` and `WasmTuneResult` as subtypes:

```ts
// Base — shared by all backends. Current interface minus `groups` (WebGPU-specific).
export interface TuneResult {
  exp: AluExp;
  epilogue?: AluExp;
  outputIdxExp: AluExp;
  threadCount: SizeExpr;
  size: { reduce: SizeExpr; unroll?: number; upcast?: number };
}

// WebGPU — adds cooperative threading fields.
export interface WebGPUTuneResult extends TuneResult {
  size: TuneResult["size"] & { groups?: number; local?: number };
}

// WASM — adds vectorization fields.
export interface WasmTuneResult extends TuneResult {
  size: TuneResult["size"] & { simdWidth?: number };
}
```

Return type changes:

- `tuneNullopt()` → returns `TuneResult` (unchanged)
- `tuneWebgpu()` → returns `WebGPUTuneResult`
- Future `tuneWasm()` → returns `WasmTuneResult`

### Impact

- `pipelineSource()` and `pipelineSourceMulti()` accept the result of either `tuneWebgpu()` or
  `tuneNullopt()`. The `groups` check narrows to `WebGPUTuneResult` when needed.
- WASM codegen uses `TuneResult` from `tuneNullopt()`. Future SIMD uses `WasmTuneResult`.
- No runtime changes — this is purely type-level narrowing.

### Files changed

| File           | Change                                                        |
| -------------- | ------------------------------------------------------------- |
| `src/tuner.ts` | Add `WebGPUTuneResult`, `WasmTuneResult`; update `tuneWebgpu` |

---

## R3: Extend ShaderInfo with Workgroup Configuration

**Priority:** Phase 0 prerequisite for Improvements 3, 4  
**Effort:** Small  
**Status:** ✅ Implemented

### Problem

`ShaderInfo.workgroupSize` at `src/backend/webgpu/codegen.ts` line 7 is `number | undefined` — a
single scalar representing x-dimension-only workgroup size. WebGPU workgroups are 3-dimensional
(`@workgroup_size(x, y, z)`). When Improvements 3–4 add cooperative local threads and shared memory
reductions, the shader needs to report multi-dimensional workgroup sizes and shared memory
requirements. Without this field, that information has nowhere to go.

### Design

```ts
export interface ShaderInfo {
  code: string;
  numInputs: number;
  numOutputs: number;
  hasUniform: boolean;
  passes: { grid: [number, number]; uniform?: Uint8Array<ArrayBuffer> }[];
  isSymbolic?: boolean;
  /** Workgroup size. Number for 1-D, tuple for multi-dimensional. */
  workgroupSize?: number | [number, number?, number?];
  hasSymbolicReduction?: boolean;
  /** Bytes of workgroup shared memory required by the shader (0 = none). */
  sharedMemoryBytes?: number;
}
```

The consumer in `pipelineSubmit()` resolves the workgroup size for symbolic grid computation:

```ts
const wgSize =
  typeof shader.workgroupSize === "number"
    ? shader.workgroupSize
    : (shader.workgroupSize?.[0] ?? 256);
```

### Files changed

| File                            | Change                                               |
| ------------------------------- | ---------------------------------------------------- |
| `src/backend/webgpu/codegen.ts` | Extend `workgroupSize` type, add `sharedMemoryBytes` |
| `src/backend/webgpu.ts`         | Update `pipelineSubmit()` workgroupSize resolution   |

---

## R4: Axis Interaction Semantics (`local × groups × reduce`)

**Priority:** Phase 0 prerequisite for Improvements 3, 4  
**Effort:** Design doc only  
**Status:** ✅ Specified below

### Problem

Improvements 3 (local axis) and 4 (shared memory reductions) interact with each other and with the
existing `groups` and `reduce` axes. Without upfront design, the interaction model is discovered
ad-hoc during implementation, leading to bugs and rework.

### Design: Axis Interaction Model

The `TuneDims` axis layout is:

```
[global | local | groups | reduce | unroll | upcast]
  ↑        ↑        ↑        ↑         ↑        ↑
  0     local    groups    reduce   unroll    upcast
```

**Definitions:**

| Axis     | Maps to                     | Purpose                                    |
| -------- | --------------------------- | ------------------------------------------ |
| `global` | `workgroup_id`              | Independent output elements                |
| `local`  | `local_invocation_id`       | Cooperative threads within a workgroup     |
| `groups` | `local_invocation_id` (sub) | Cooperative threads for grouped reductions |
| `reduce` | Sequential loop             | Reduction accumulation per thread          |
| `unroll` | Unrolled loop iterations    | Explicit loop unrolling                    |
| `upcast` | Inlined expressions         | Vector-width inlined computation           |

**Interaction rules:**

1. **`local` and `groups` share `local_invocation_id`**: When both are active (local > 1 AND
   groups > 1), `local_invocation_id.x = local_idx * groups + group_idx`. Total workgroup size =
   local × groups. This must not exceed `maxComputeInvocationsPerWorkgroup` (device limit).

2. **`local` without `groups`**: Each workgroup has `local` threads processing adjacent output
   elements. No shared memory needed. Dispatch count = `ceil(totalElements / local)`.

3. **`groups` without `local`**: Each workgroup has `groups` threads cooperatively reducing. Shared
   memory needed for partial results. One output element per workgroup.

4. **`local` AND `groups`**: Each workgroup processes `local` output elements, each reduced by
   `groups` threads. Shared memory = `local × groups × sizeof(dtype)`. Total workgroup size =
   `local × groups`.

5. **`reduce` is always sequential within a thread**: After `groups` splits the reduction, each
   group thread has `reduce / groups` elements to accumulate sequentially.

6. **`unroll` and `upcast` are orthogonal**: They affect loop structure and expression shape, not
   thread mapping. Compatible with any `local`/`groups` configuration.

**Validation in `tuneWebgpu()`:**

```ts
const totalWorkgroupSize = (local ?? 1) * (groups ?? 1);
if (totalWorkgroupSize > caps.maxComputeInvocationsPerWorkgroup) {
  // Fall back to smaller local or groups
}
if (totalWorkgroupSize > caps.maxComputeWorkgroupSizeX) {
  // Need 2-D workgroup layout: @workgroup_size(local, groups)
}
```

**Grid computation with `local`:**

```ts
// Without local: grid = ceil(threadCount / workgroupSize)
// With local:    grid = ceil(outputElements / local)
// workgroupSize = local × groups (or local alone when groups = 1)
```

This design doc should be referenced when implementing Improvements 3 and 4.

---

# Phase 1: Improvements

## Improvement 1: Replace UA Sniffing with Device Limits

**Priority:** High (correctness issue — UA string is unreliable)  
**Effort:** Small  
**Status:** ✅ Implemented  
**Files:** `src/tuner.ts`, `src/backend.ts`, `src/backend/webgpu.ts`

### Problem

Line 345 of `src/tuner.ts` uses `navigator.userAgent` to disable loop unrolling on mobile:

```ts
if (!/Mobi|Android/i.test(navigator.userAgent) && ...)
```

This is fragile (desktop Chrome can send mobile UA strings, Deno/Node have no `navigator.userAgent`,
some mobile GPUs are powerful), and the check has nothing to do with what it's trying to measure —
whether the GPU has enough ALU throughput for aggressively unrolled loops.

### Plan

1. **Extend `BackendCapabilities`** in `src/backend.ts` (line 20) with a
   `maxComputeWorkgroupSizeX: number` field (and optionally `maxComputeInvocationsPerWorkgroup`).
   These are concrete device limits that correlate with GPU capability better than a UA string.

2. **Thread capabilities into `tuneWebgpu()`** — the function signature at `src/tuner.ts` line 255
   gains a `caps?: BackendCapabilities` parameter. The call site in `pipelineSource()`
   (`src/backend/webgpu.ts` line 1628) already has access to the `device` object and can pass
   `this.capabilities`.

3. **Replace the UA check** with a threshold on `maxComputeWorkgroupSizeX`. Low-end mobile GPUs
   typically report 256 (the minimum); high-end desktop GPUs report 1024. A reasonable heuristic:
   skip unrolling when `maxComputeWorkgroupSizeX <= 256`.

4. **Fallback for non-WebGPU backends:** `tuneWebgpu()` is only called from the WebGPU codegen path,
   so `caps` will always be available. Add a guard `if (!caps)` that defaults to the current
   "desktop" behavior.

### Affected call sites

| Location                                   | Change                                                  |
| ------------------------------------------ | ------------------------------------------------------- |
| `src/backend.ts` (line 20)                 | Add `maxComputeWorkgroupSizeX` to `BackendCapabilities` |
| `src/backend/webgpu.ts` (constructor ~269) | Populate from `device.limits.maxComputeWorkgroupSizeX`  |
| `src/tuner.ts` (line 255)                  | Add `caps` parameter to `tuneWebgpu()`                  |
| `src/tuner.ts` (line 345)                  | Replace `navigator.userAgent` check with caps threshold |
| `src/backend/webgpu.ts` (~line 1628)       | Pass `this.capabilities` to `tuneWebgpu()`              |

---

## Improvement 2: Wire `tuneWebgpu` into Multi-Output Kernels

**Priority:** High (leaves significant performance on the table)  
**Effort:** Medium  
**Status:** ✅ Implemented  
**Files:** `src/backend/webgpu.ts`, `src/tuner.ts`

### Problem

`pipelineSourceMulti()` at `src/backend/webgpu.ts` line 1362 explicitly bypasses tuning:

```ts
const tunes = kernel.outputs.map((o) => {
  const tmp = Kernel.single(nargs, kernel.size, o.exp, o.reduction);
  return tuneNullopt(tmp);
});
```

Multi-output kernels with reductions (e.g., `[x.sum(), x.max()]`) get no unrolling, no upcast, no
memory coalescing. Since multi-output is only emitted for non-reduction outputs currently
(`src/frontend/jit.ts` groups by size + no-reduction), this is less urgent than it appears — but it
means multi-output with future reduction support will have no tuning.

### Plan

1. **For non-reduction multi-output:** Replace `tuneNullopt` with `tuneWebgpu` for each output
   independently. The `threadCount` must be verified to be identical across all outputs (same
   `kernel.size`, same reduction=none means same thread count).

2. **For mixed reduction multi-output (future):** Each output with a reduction gets its own
   `tuneWebgpu` pass. The shader codegen in `pipelineSourceMulti` already generates per-output
   accumulator loops — the unroll/upcast factors just need to be plumbed through.

3. **Shared CSE across outputs:** The current CSE in `pipelineSourceMulti` at line ~1430 works on
   the original expressions. After tuning, the substituted expressions may share different
   subexpressions. Run CSE after substitution.

### Affected call sites

| Location                             | Change                                                       |
| ------------------------------------ | ------------------------------------------------------------ |
| `src/backend/webgpu.ts` (line 1370)  | Replace `tuneNullopt` with `tuneWebgpu` per output           |
| `src/backend/webgpu.ts` (line ~1430) | Add reduction loop codegen for multi-output (when supported) |

---

## Improvement 3: Proper `local` Axis in TuneDims

**Priority:** Medium  
**Effort:** Medium-Large  
**Status:** ✅ Implemented  
**Files:** `src/tuner.ts`, `src/backend/webgpu.ts`

### Problem

`TuneDims` at `src/tuner.ts` line 66 has a TODO:

```ts
// local: number; // TODO: Split gidx -> global and local axes during tuning.
```

And line 369 acknowledges the current `applyLocal` is a hack:

```ts
// TODO: These applyLocal() calls are a hack / bad heuristic, make this better.
```

Currently `applyLocal` (line 166) just calls `applyUpcast` — it doesn't actually create a separate
"local" axis that maps to workgroup-local indices. This means workgroup size is always 1
thread-per-element with no cooperative work within a workgroup.

### Plan

1. **Add `local` field to `TuneDims`** at line 66. Axis partitioning becomes
   `[global | local | groups | reduce | unroll | upcast]`. The `local` axis maps to
   `local_invocation_id.x` within a workgroup, and `global` maps to `workgroup_id`.

2. **Add `applyLocal()` method** that actually splits an axis into local threads rather than
   aliasing to `applyUpcast`. The method should validate that the local size doesn't exceed
   `maxComputeWorkgroupSizeX` (from Improvement 1's `BackendCapabilities`).

3. **Update `tuneWebgpu()`** to use the real `applyLocal` instead of the current hack at line 369.
   The heuristic: for upcasted axes that benefit from shared memory (when a stride-0 buffer is
   broadcast), split into local threads that cooperatively load and share data.

4. **Update `WebGPUTuneResult`** (from R2) to include a `local` size field alongside `groups`,
   `reduce`, `unroll`, `upcast`. See R4 axis interaction semantics for how `local` interacts with
   `groups`.

5. **Update `pipelineSource()`** in `src/backend/webgpu.ts` (line 1628) to emit workgroup-sized
   dispatches with `local_invocation_id` indexing. Currently workgroup size is always 1 thread for
   non-reduction kernels. With a `local` axis, it becomes `@workgroup_size(localSize)` and the
   dispatch count divides by `localSize`.

### Dependency

Improvement 1 (device limits in `BackendCapabilities`) should land first so `applyLocal` can respect
hardware constraints.

---

## Improvement 4: Workgroup Shared Memory for Reductions

**Priority:** Medium  
**Effort:** Large  
**Status:** ✅ Implemented  
**Files:** `src/tuner.ts`, `src/backend/webgpu.ts`

### Problem

The tuner's `groups` axis at `src/tuner.ts` line 64 is intended for workgroup-level cooperative
reductions, but line 1927 of `src/backend/webgpu.ts` throws:

```ts
if ((tune.size.groups ?? 1) > 1) {
  throw new Error("WebGPU backend does not support group optimization yet");
}
```

This means large reductions (e.g., summing a 1M-element array) use a single thread per output
element doing a sequential loop. A proper implementation would have multiple threads within a
workgroup cooperatively accumulate partial sums, then combine via shared memory.

### Plan

1. **Extend `WebGPUTuneResult`** (from R2) with `useSharedMemory: boolean` and
   `sharedMemoryBytes: number` fields. The tuner decides when shared memory is beneficial —
   typically when `groups > 1` (i.e., the reduction is large enough to split across workgroup
   threads). Use R3's `ShaderInfo.sharedMemoryBytes` to convey the requirement to `pipelineSubmit`.
   See R4 axis interaction semantics for `groups × local` thread layout.

2. **In `tuneWebgpu()`** at line 255: when reduction size is large enough (e.g., ≥256), set
   `groups > 1` by splitting the reduce axis. Each group thread handles `reduceSize / groups`
   elements, then they combine via shared memory.

3. **In `pipelineSource()`** at line 1927: replace the `throw` with actual shared-memory codegen:
   - Emit `var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;`
   - Each thread accumulates a partial sum into its local accumulator
   - Write partial to `shared_data[local_id]`
   - `workgroupBarrier()`
   - Tree reduction within workgroup (thread 0 accumulates all partials)
   - Thread 0 writes final result

4. **Pattern already exists in routines:** `src/backend/webgpu/routines.ts` already uses
   `var<workgroup>` and `workgroupBarrier()` for the Sort routine. The shared-memory reduction is a
   simpler version of the same pattern.

### Dependency

Improvement 3 (proper `local` axis) should land first so the `local` dimension is well-defined when
splitting the reduction.

---

## Improvement 5: WASM SIMD Vectorization of JIT Kernels

**Priority:** Medium-High (significant WASM perf opportunity)  
**Effort:** Large  
**Status:** ✅ Implemented (f32x4 contiguous + broadcast; no R1 InstructionEmitter — used separate `translateExpCoreSimd`)  
**Files:** `src/backend/wasm.ts`, `src/backend/wasm/mega-module.ts`

### Problem

`emitKernelBody()` at `src/backend/wasm.ts` line 2040 always uses `tuneNullopt` and
`translateExpCore()` at line 1640 emits purely scalar WASM. Every element is processed one at a
time. For f32 elementwise ops, SIMD could process 4 elements per instruction; for f64, 2 per
instruction.

### SIMD Capability Inventory (Already Available)

All of these exist and are production-tested:

| Layer                | What's Available                                               | Location                      |
| -------------------- | -------------------------------------------------------------- | ----------------------------- |
| **Assembler**        | `f32x4.add/sub/mul/div/min/max/abs/neg/sqrt/ceil/floor` etc.   | `wasmblr.ts` lines 985-1012   |
| **Assembler**        | `f64x2.add/sub/mul/div/min/max/abs/neg/sqrt` etc.              | `wasmblr.ts` lines 1013-1050  |
| **Assembler**        | `i32x4.add/sub/mul/shl/shr_s/shr_u`, `i32x4.splat`             | `wasmblr.ts` lines 953-984    |
| **Hl helpers**       | `loadF32x4()`, `storeF32x4()`, `f32x4Hsum()`, `f64x2Hsum()`    | `wasmblr-hl.ts` lines 362-470 |
| **Hl helpers**       | `simdReductionF32()`, `simdReductionF64()` — vectorized + tail | `wasmblr-hl.ts` lines 580-840 |
| **Proof of concept** | `cholesky-simd.ts` — hand-written, 3-4× speedup at n≥128       | `routines/cholesky-simd.ts`   |

### What's Missing

| Gap                                 | Why It Matters                                                                                                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`translateExpCore` SIMD mode**    | The core expression translator has no concept of v128 types. Every `AluOp.Add` generates scalar `f32.add`; it should optionally generate `f32x4.add`. |
| **`tuneWasm()` function**           | No WASM-specific tuner exists. Need a function that decides vectorization width and emits the vectorized gidx loop with scalar tail.                  |
| **SIMD-incompatible op detection**  | Not all ops can be vectorized. Transcendentals use `callFuncF32` (line 1736) with scalar builtins. `Threefry` has complex bit manipulation.           |
| **WasmHl SIMD elementwise helpers** | Existing helpers do dot-product reduction. General elementwise needs a `main loop + scalar tail` pattern helper.                                      |

### Plan

#### Phase 1: SIMD eligibility analysis

1. **Add `canVectorize(exp: AluExp): boolean`** — walks an `AluExp` tree and returns `false` if any
   node is SIMD-incompatible:
   - **SIMD-compatible (f32):** `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`, `Abs`, `Neg`, `Sqrt`,
     `Floor`, `Ceil`, `Cast` (f32↔i32 only), `Where` (branchless `select` only — check
     `estimateCost() < 15`)
   - **SIMD-compatible (f64):** Same minus `Floor`/`Ceil` (no `f64x2` equivalent in WASM spec)
   - **Not vectorizable:** `Sin`, `Cos`, `Asin`, `Atan`, `Exp`, `Log`, `Erf`, `Erfc`,
     `Threefry2x32`, `Bitcast`, `GlobalView` with non-contiguous stride

2. **Add `simdWidth(dtype: DType): number`** — returns 4 for f32, 2 for f64, 4 for i32, 0 for
   f16/f64+transcendentals.

#### Phase 2: Vectorized expression emission via R1 InstructionEmitter

3. **Implement R1** (see Phase 0) — refactor `translateExpCore` to use the `InstructionEmitter`
   interface. This is the prerequisite step.

4. **Create `simdEmitter(width: 4 | 2): InstructionEmitter`** — SIMD implementation that:
   - Maps `AluOp.Add` → `cg.f32x4.add()` (instead of `cg.f32.add()`)
   - Maps `AluOp.Variable` → `cg.v128.load(base, gidx)` (load 4 elements at once)
   - Maps `AluOp.Const` → `cg.f32x4.splat(value)`
   - Maps `AluOp.Where` (branchless) → `cg.v128.bitselect(t, f, mask)` (select per-lane)
   - `allocCSELocal` returns v128 locals for CSE caching
   - `step` = 4 (f32) or 2 (f64)

   Because `translateExpCore` is already polymorphic over `InstructionEmitter`, no new translation
   function is needed. The same `translateExpCore` call emits SIMD when given `simdEmitter()`.

#### Phase 3: Vectorized kernel body

5. **Extend `emitKernelBody()`** with a SIMD path controlled by `WasmTuneResult.size.simdWidth`:
   - Main loop: `gidx = start; gidx < end - (simdWidth-1); gidx += simdWidth`
   - Body: call `translateExpCore(cg, funcs, exp, ctx, simdEmitter(width))`
   - Scalar tail: `for (; gidx < end; gidx++)` with default `scalarEmitter()`
   - For reductions: v128 accumulator with `f32x4.add` in inner loop, `f32x4Hsum()` after

6. **Create `tuneWasm(kernel): WasmTuneResult`** — lightweight tuner that:
   - Returns `tuneNullopt(kernel)` if `!canVectorize(kernel.outputs[0].exp)`
   - Sets `simdWidth` on the returned `WasmTuneResult`
   - `emitKernelBody` checks `simdWidth > 1` to select the vectorized path

#### Phase 4: Mega-module integration

7. **`translateExpMega`** wraps `translateExpCore`. Since `translateExpCore` now accepts an
   `InstructionEmitter`, mega-module gets SIMD for free — just pass `simdEmitter()` when the kernel
   is vectorizable. No `translateExpMegaSimd` function needed.

### Performance Expectations

Based on the cholesky-simd measurements already in the codebase:

| Kernel Pattern                    | Expected f32 Speedup | Notes                           |
| --------------------------------- | -------------------- | ------------------------------- |
| Elementwise chain (`add→mul→sub`) | 3-4×                 | Memory-bound at large sizes     |
| Reduction (sum)                   | 2-3×                 | Accumulator becomes v128 + hsum |
| Matmul (via reduce)               | Already SIMD         | No change needed                |
| Transcendentals (`sin(x).add(y)`) | 1× (no change)       | Falls back to scalar            |

### What Does NOT Need New wasmblr Helpers

The existing `WasmHl` helpers are designed for hand-written routines with specific memory layouts.
The JIT SIMD path should use the raw wasmblr instruction methods directly (`cg.f32x4.add()`,
`cg.v128.load()`, etc.) because:

- The codegen already works at the wasmblr level (not the WasmHl level) for scalar kernels
- v128 load/store alignment is guaranteed for JIT buffers (all buffers are 4-byte aligned from
  `WasmAllocator`)
- The `simdReductionF32` pattern is too specialized (assumes two input arrays for dot product)

One useful **new WasmHl helper** to consider: `emitSimdTailGuard(gidx, end, body)` — generates the
standard `main_loop(step=simdWidth)` + `scalar_tail` pattern, since every SIMD kernel needs this.

---

## Improvement 6: Subgroups Support for WebGPU Reductions

**Priority:** Low (WebGPU `subgroups` feature not yet stable)  
**Effort:** Medium  
**Status:** ⏸ Deferred — WebGPU `subgroups` feature not stable, no broad browser support  
**Files:** `src/backend.ts`, `src/backend/webgpu.ts`

### Problem

Large reductions currently serialize across the entire reduce axis with one thread. Subgroups
(SIMD-width cooperative operations) would enable `subgroupAdd()` that sums ~32 values in one
instruction, before the shared-memory tree reduction.

### Plan

1. **Add `subgroups: boolean` to `BackendCapabilities`** at `src/backend.ts` line 20. Set from
   `device.features.has('subgroups')` at adapter creation.

2. **Request `'subgroups'` feature** at device creation in `src/backend.ts` when available on the
   adapter.

3. **Add subgroup reduction path** in `pipelineSource()` at `src/backend/webgpu.ts` line 1927 — when
   `caps.subgroups && tune.size.groups > 1`:

   ```wgsl
   enable subgroups;
   let partial = subgroupAdd(acc);
   if (subgroup_invocation_id == 0) {
     shared_data[subgroup_id] = partial;
   }
   workgroupBarrier();
   // Thread 0 reduces across subgroups
   ```

4. **Composing with Improvement 4:** Subgroups replace the inner tree reduction in shared memory.
   Instead of `log2(workgroupSize)` barrier-synchronized steps, you get
   `log2(workgroupSize / subgroupSize)` steps. For a workgroup of 256 and subgroup of 32, that's 3
   barriers instead of 8.

### Risk

The `subgroups` feature is not yet stable in the WebGPU spec. Browser support varies. This
improvement should be gated behind a runtime feature check, never assumed.

---

## Improvement 7: Bind Group Layout Caching

**Priority:** Medium (dispatch overhead reduction)  
**Effort:** Small  
**Status:** ✅ Partially implemented — layout caching done; bind group caching skipped (buffer pooling invalidates cache on most invocations)  
**Files:** `src/backend/webgpu.ts`

### Problem

`ShaderPipelineCache.#getLayout()` at `src/backend/webgpu.ts` line 2596 creates fresh
`GPUBindGroupLayout` and `GPUPipelineLayout` objects on every pipeline compilation. More critically,
`pipelineSubmit()` at line 2423 creates a new `GPUBindGroup` on every dispatch call at line 2448,
even when the same set of buffers is used repeatedly (which is common in JIT programs that are
called multiple times with the same shapes).

While `GPUBindGroupLayout` creation is amortized by the pipeline cache, `GPUBindGroup` creation
happens per-dispatch and adds GPU driver overhead.

### Plan

1. **Cache `GPUBindGroupLayout` by signature** in `ShaderPipelineCache.#getLayout()` — use
   `numInputs:numOutputs:hasUniform` as key. Most JIT programs share the same layout shape. This is
   a minor win since layout is already cached per-pipeline.

2. **Cache `GPUBindGroup` by buffer identity** in `pipelineSubmit()` — use a `WeakRef`-based cache
   keyed by the tuple of `GPUBuffer` objects. When the same dispatch is called repeatedly with the
   same buffers (e.g., repeated JIT calls), return the cached bind group. When any buffer is
   replaced (different input shape), the cache misses and creates fresh.

3. **Scope:** This mainly benefits the JIT hot path. The scan dispatch paths (line 592, line 937)
   create bind groups per-iteration which is harder to cache (changing buffers).

### Implementation sketch

```ts
// In pipelineSubmit or WebGPUBackend:
#bindGroupCache = new Map<string, WeakRef<GPUBindGroup>>();

function getOrCreateBindGroup(
  layout: GPUBindGroupLayout,
  buffers: GPUBuffer[],
): GPUBindGroup {
  const key = buffers.map((b) => b.label || String(b)).join(":");
  const cached = this.#bindGroupCache.get(key)?.deref();
  if (cached) return cached;
  const bg = device.createBindGroup({ ... });
  this.#bindGroupCache.set(key, new WeakRef(bg));
  return bg;
}
```

Note: `GPUBuffer` objects don't have stable identity strings by default. A numeric `Slot`-keyed
approach (using the existing slot map) would be more reliable.

---

## Improvement 8: Fix the Unroll-by-8 Regression

**Priority:** Low (performance tuning)  
**Effort:** Small  
**Status:** ✅ Already resolved — code uses `[4, 2]` split factors instead of `[8, 4, 2]`  
**Files:** `src/tuner.ts`

### Problem

Line ~360 of `src/tuner.ts` notes:

```ts
// Note: Unrolling by 8 previously made this faster in January 2026, but
// in later versions of Chrome on macOS, it seems to have regressed 40%.
// Seems like 4 is a more stable choice at the moment.
```

The regression is likely because Chrome's shader compiler changed its register allocation heuristic,
and 8× unrolling causes register spilling.

### Plan

1. **Add unroll factor to `BackendCapabilities`** or keep it as a tuner-internal heuristic keyed on
   `maxComputeWorkgroupSizeX` (a proxy for GPU generation).

2. **Profile-guided selection:** For the first call with a new kernel shape, try unroll factors 2,
   4, 8 and cache the fastest. This is heavier than a static heuristic but self-correcting. Could be
   gated behind a `tuneLevel: 'quick' | 'profile'` option.

3. **Simpler alternative:** Just keep unroll-by-4 and move on. The 40% regression at 8 suggests
   Chrome's shader compiler is the bottleneck, not the tuner. This can be revisited when Chrome or
   Dawn stabilizes.

---

## Suggested Implementation Order

```
Phase 0 (prerequisites — zero functional change):
  R2: Backend-typed TuneResult    [Small, done ✅]
  R3: ShaderInfo workgroup ext    [Small, done ✅]
  R1: InstructionEmitter          [Medium, skipped — separate translateExpCoreSimd used instead]
  R4: Axis interaction design     [Design doc, done ✅]

Phase 1 (improvements):
  1. UA sniffing → device limits    [done ✅]
  2. tuneWebgpu for multi-output    [done ✅]
  3. TuneDims local axis            [done ✅]
  4. Shared memory reductions       [done ✅]
  5. WASM SIMD vectorization        [done ✅ — f32x4 contiguous + broadcast]
  6. Subgroups                      [deferred ⏸ — spec not stable]
  7. Bind group layout caching      [done ✅ — bind group caching skipped]
  8. Unroll regression              [done ✅ — already resolved in existing code]
```

All actionable improvements are complete. Improvement 6 (Subgroups) is deferred until the WebGPU
`subgroups` feature stabilizes in browsers.

### Implementation notes

- **R1 (InstructionEmitter) was not needed.** A separate `translateExpCoreSimd()` function in
  `src/backend/wasm.ts` handles SIMD codegen. The interface would have been over-engineered for the
  current scope (only f32x4 elementwise vectorization). If f64x2 or i32x4 vectorization is added
  later, R1 may become worthwhile.
- **Improvement 5 (SIMD)** covers both `codegenWasm()` (eager/JIT kernels) and mega-module
  extracted kernel bodies. Handles contiguous arrays (`v128.load`) and broadcast arrays
  (`f32.load` + `f32x4.splat`). Reductions and transcendentals remain scalar.
- **Improvement 7 (Bind Group Caching)** only implemented layout caching (`GPUPipelineLayout` by
  `numInputs:numOutputs:hasUniform` signature). Full bind group caching was analyzed and skipped
  because WebGPU buffer pooling causes different `GPUBuffer` objects between invocations, making
  cache hits rare.
