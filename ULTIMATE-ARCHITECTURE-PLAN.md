# The Ultimate Architecture: Fusion, Atomics, and Mega-Modules

**Branch:** `ultimate-architecture` **Scope:** Transform `jax-js-nonconsuming` into a
fully-optimized, hardware-saturating compute engine. This plan eliminates the JS ↔ Native dispatch
boundary, minimizes VRAM bandwidth via aggressive fusion, introduces WebGPU atomics for
`scatter_add`, and parallelizes the Wasm backend via `SharedArrayBuffer`. **Execution model:**
Single autonomous, tireless coding agent.

---

## Motivation

With AOT Linearization and Effect-Typed IR complete, the library is memory-safe, deterministic, and
structurally sound. However, it currently leaves massive hardware performance on the table due to
five architectural bottlenecks:

1. **The Functional Blocker (`scatter_add`):** The Gather transpose rule
   (`src/frontend/linearize.ts` L1617–1647) only handles the permutation case (single 1-D index,
   same axis size in/out) via `argsort`-based inverse. For duplicate indices — required by embedding
   lookups, GNNs, and any `np.take` with repeated indices — it throws
   `"requires a scatter_add primitive"`. Without `scatter_add`, these models are mathematically
   impossible to train.
2. **The VRAM Bandwidth Bottleneck (single-output Kernels):** `Kernel` (`src/alu.ts` L1450) is
   single-output: `constructor(nargs, size, exp, reduction?)`. AD backward passes that produce
   `dA + dB` emit two separate kernel dispatches, forcing intermediate VRAM round-trips. Matmul
   (`Primitive.Dot`) compiles to a Mul→Reduce kernel — fusing a subsequent `relu` or `bias add`
   _into the same shader_ is possible via the existing `tune.epilogue` mechanism but not yet
   exploited by `splitGraphDataflow`.
3. **The JS ↔ Native Boundary (Wasm Overhead):** A `JitProgram` with 50 steps
   (`src/frontend/jit.ts` L155) is executed by a JS `for` loop over `JitStep`s. Each `"execute"`
   step crosses the JS↔Wasm boundary. For small/medium arrays, dispatch overhead dominates.
4. **Single-Threaded Wasm:** `WasmAllocator` (`src/backend/wasm/allocator.ts`) uses a standard
   `WebAssembly.Memory`. All kernel dispatches run on the main thread.
5. **JIT Recompilation for Variable-Length Data:** Array dimensions are baked into `Kernel.size`,
   `ShapeTracker.views[].shape`, and `effectDrivenAllocate()`'s byte-size computations. Changing a
   time series length forces a full retrace and recompile.

---

## Milestones

### M0 — Baseline Snapshot & Feature Detection (1–2 days)

#### M0.1 — Record Baseline Test Results

**What:** Run the full test suite and Deno tests, record pass/fail counts and benchmark timings.

**Commands:**

```bash
pnpm build
pnpm vitest run 2>&1 | tee tmp/m0-vitest-baseline.txt; echo $? > tmp/m0-vitest-exit.txt
pnpm run test:deno 2>&1 | tee tmp/m0-deno-baseline.txt; echo $? > tmp/m0-deno-exit.txt
deno bench --no-check --unstable-webgpu --allow-read --allow-env \
  test/deno/recycle.bench.ts 2>&1 | tee tmp/m0-bench-baseline.txt
```

**Exit criteria:** Baseline files exist in `tmp/` with full logs, exit codes, and benchmark results.

#### M0.2 — Hardware Feature Detection

**What:** Add two capabilities to the `Backend` interface (`src/backend.ts`):

```typescript
// Add to Backend interface:
readonly capabilities: BackendCapabilities;

// New type:
export interface BackendCapabilities {
  /** WebGPU: true if shader-f32-atomic-add extension is available. */
  atomicF32Add: boolean;
  /** Wasm: true if crossOriginIsolated (SharedArrayBuffer available). */
  sharedMemory: boolean;
}
```

**Files touched:**

- `src/backend.ts` — add `BackendCapabilities` type, add to `Backend` interface
- `src/backend/webgpu.ts` — detect `shader-f32-atomic-add` at adapter request time (check
  `adapter.features.has("shader-f32-atomic-add")`)
- `src/backend/wasm.ts` — detect `globalThis.crossOriginIsolated`
- `src/backend/cpu.ts` — `{ atomicF32Add: false, sharedMemory: false }`

**Exit criteria:** `backend.capabilities` exposes both flags on all backends. All tests pass.

---

### M1 — Scan Backward AOT Artifact (3–4 days)

The `vjpFlat` function (`src/frontend/linearize.ts` L2034) has two codepaths for backward pass
construction: an AOT path via `buildBackwardJaxpr` for non-scan jaxprs, and a call-time-transpose
fallback for scan-containing jaxprs. The call-time fallback exists because scan's transpose rule
(`src/frontend/linearize.ts` L1710–1900) performs **concrete forward recomputation** from
checkpoints and creates sub-jaxprs — operations that `makeJaxpr` tracing (used by
`buildBackwardJaxpr`) cannot capture.

This milestone creates a dedicated `ScanBackwardArtifact` that encapsulates the scan backward pass
as an AOT artifact, eliminating the dual-codepath maintenance burden without changing the underlying
scan transpose algorithm.

#### M1.1 — `ScanBackwardArtifact` type

**What:** Extend `src/frontend/scan-backward.ts` to make `ScanPullbackArtifact` a full AOT artifact:

```typescript
// Extend existing ScanPullbackArtifact:
export interface ScanBackwardArtifact extends Disposable {
  /** The scan transpose rule, pre-bound with residuals and checkpoint config. */
  run(cotangents: Tracer[]): Tracer[];
}
```

The `run()` method wraps the existing scan transpose rule's concrete forward recomputation +
per-segment backward transposition. It does NOT trace into a Jaxpr — it runs with concrete arrays at
call time, exactly as the current fallback does. The artifact owns its residual arrays and
checkpoint carries.

**Files touched:**

- `src/frontend/scan-backward.ts` — extend `ScanPullbackArtifact` with `run()` method
- `src/frontend/linearize.ts` — factor the scan-specific backward closure in `vjpFlat` into
  `ScanPullbackArtifact.run()`

**Exit criteria:** `ScanPullbackArtifact` encapsulates the backward pass. `vjpFlat` delegates to it.

#### M1.2 — Unify `vjpFlat` transposition strategy

**What:** Replace `jaxprNeedsCallTimeTranspose` with a unified strategy:

```typescript
function vjpFlat(f, primalsIn, auxStore?) {
  // Phase 1–2: unchanged (linearizeFlatUtil + PE cleanup)
  // Phase 3: always AOT-transpose for non-scan eqns
  const backwardJaxpr = buildBackwardJaxpr(forwardJaxpr);
  // Phase 4: check for scan artifacts in the forward jaxpr
  const scanArtifacts = collectScanArtifacts(forwardJaxpr);
  // fVjp calls backwardJaxpr.eval() + scanArtifact.run() for scan portions
}
```

The key insight: `buildBackwardJaxpr` already handles everything _except_ `Primitive.Scan`. For scan
equations, it emits a placeholder that the `ScanBackwardArtifact.run()` fills in at call time. The
scan backward pass remains **concrete call-time execution** — its runtime loops, checkpoint logic,
and sub-jaxpr construction cannot be captured into a Jaxpr by `makeJaxpr` tracing. Only the non-scan
portion of the backward pass gets AOT-compiled. In the current codebase, `grad()` calls
`fVjp(ones_like)` with concrete cotangents (not tracers), so the scan backward pass already runs
with concrete arrays — this milestone formalizes that pattern as the artifact API.

**Files touched:**

- `src/frontend/linearize.ts` — remove `jaxprNeedsCallTimeTranspose`, unify `vjpFlat`
- `src/frontend/scan-backward.ts` — `collectScanArtifacts` helper

**Test commands:**

```bash
pnpm build
pnpm vitest run test/lax-scan.test.ts        # scan grad tests
pnpm vitest run test/transform-compositions.test.ts
pnpm vitest run test/leak-diagnostic.test.ts
pnpm vitest run                               # full regression
```

**Exit criteria:** `vjpFlat` has a single codepath. `jaxprNeedsCallTimeTranspose` removed. All scan
grad/transform composition tests pass.

**Migration verification:**

```bash
grep -rn 'jaxprNeedsCallTimeTranspose' src/   # must return zero matches
pnpm check                                     # no type errors from stale references
```

---

### M2 — The Missing Primitive: `scatter_add` (4–6 days)

Implement the `scatter_add` primitive using the existing `MemoryEffect.Mutate` infrastructure
(proven by DUS).

#### M2.1 — `Primitive.ScatterAdd` IR & AD Rules

**What:** Add `scatter_add(target, indices, updates, axis)` to the IR.

**Primitive definition** (`src/frontend/core.ts`):

```typescript
// Add to Primitive enum:
ScatterAdd = "scatter_add",

// Add to PrimitiveParams:
[Primitive.ScatterAdd]: { axis: number };

// Add binding function:
export function scatterAdd(
  target: Tracer, indices: Tracer, updates: Tracer,
  axis: number,
): Tracer {
  return bind1(Primitive.ScatterAdd, [target, indices, updates], { axis });
}
```

**Memory effects** (`src/frontend/jaxpr.ts`):

```typescript
// Add to primitiveInputEffects:
[Primitive.ScatterAdd]: (n: number) => {
  // target is Mutate (in-place accumulation), indices + updates are Borrow
  const effects = globalThis.Array.from({ length: n }, () => MemoryEffect.Borrow);
  effects[0] = MemoryEffect.Mutate;
  return effects;
},
```

**AD rules** — the mathematical relationship between `gather` and `scatter_add`:

| Operation                | Forward: $y = f(x)$                   | JVP: $\dot{y} = f'(x) \cdot \dot{x}$                    | Transpose: $\bar{x} = f'^T \cdot \bar{y}$                                            |
| ------------------------ | ------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `gather(x, idx)`         | $y_j = x_{idx_j}$                     | $\dot{y}_j = \dot{x}_{idx_j}$                           | $\bar{x}_i = \sum_{j: idx_j = i} \bar{y}_j$ ← **this is `scatter_add`**              |
| `scatter_add(t, idx, u)` | $y_i = t_i + \sum_{j: idx_j = i} u_j$ | $\dot{y}_i = \dot{t}_i + \sum_{j: idx_j = i} \dot{u}_j$ | $\bar{t} = \bar{y}$ (identity), $\bar{u}_j = \bar{y}_{idx_j}$ ← **this is `gather`** |

**JVP rule** (`src/frontend/jvp.ts`):

```typescript
[Primitive.ScatterAdd]([target, indices, updates], [dTarget, _dIndices, dUpdates], { axis }) {
  const primal = scatterAdd(target, indices, updates, axis);
  const tangent = scatterAdd(dTarget, indices, dUpdates, axis);
  return [[primal], [tangent]];
},
```

**Transpose rules** (`src/frontend/linearize.ts`):

```typescript
// Replace the Gather transpose rule (L1617–1647):
[Primitive.Gather]([ct], [x, ...indices], { axis, outDim }) {
  if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
  if (indices.some((i) => i instanceof UndefPrimal))
    throw new NonlinearError(Primitive.Gather);
  // General case: scatter_add the cotangent back to source positions.
  // target = zeros_like(x), scatter_add(target, indices, ct, axis)
  const idx = indices[0] as Tracer;
  const zeros = np.zeros(x.aval.shape, { dtype: ct.dtype });
  const result = scatterAdd(zeros, idx, ct, axis[0]);
  zeros.dispose();
  return [result, null];
},

// ScatterAdd transpose:
[Primitive.ScatterAdd]([ct], [target, indices, updates], { axis }) {
  // d/d(target) = ct (identity — scatter_add is additive in target)
  const ctTarget = !(target instanceof UndefPrimal) ? null : ct;
  // d/d(updates) = gather(ct, indices) — reverse the scatter
  const ctUpdates = !(updates instanceof UndefPrimal)
    ? null
    : gather(ct, [indices], [axis], axis);
  return [
    ctTarget ?? ct,
    null,
    ctUpdates ?? np.zeros(updates.aval.shape, { dtype: ct.dtype }),
  ];
},
```

**Abstract eval** (`src/frontend/jaxpr.ts`): Output shape = target shape, output dtype = target
dtype.

**Files touched:**

- `src/frontend/core.ts` — `Primitive.ScatterAdd`, `PrimitiveParams`, `scatterAdd()`
- `src/frontend/jaxpr.ts` — abstract eval rule, `primitiveInputEffects` entry
- `src/frontend/jvp.ts` — JVP rule
- `src/frontend/linearize.ts` — Gather transpose (replace), ScatterAdd transpose (new)
- `src/frontend/array.ts` — eager impl rule
- `src/frontend/vmap.ts` — batching rule
- `src/index.ts` — export `scatterAdd` if public

**Test file:** `test/scatter-add.test.ts`

| Test name                            | What it verifies                                        |
| ------------------------------------ | ------------------------------------------------------- |
| `scatter_add basic 1-D`              | Correct accumulation with unique indices                |
| `scatter_add with duplicate indices` | Values at same index are summed                         |
| `scatter_add 2-D axis=0`             | Batched scatter along first axis                        |
| `scatter_add 2-D axis=1`             | Scatter along second axis                               |
| `scatter_add preserves target`       | Non-consuming: target is not modified                   |
| `grad(take) with duplicates`         | End-to-end: `grad(sum(take(x, [0,1,0])))` gives `[2,1]` |
| `grad(scatter_add) wrt updates`      | Cotangent gathered back correctly                       |
| `grad(scatter_add) wrt target`       | Identity cotangent                                      |
| `jit(scatter_add)`                   | JIT compilation with Mutate effect                      |
| `scatter_add passes effect checker`  | `verifyJaxprEffects` validates Mutate on target         |

**Exit criteria:** All tests pass. `grad(np.take(x, [0,1,0]))` returns correct gradients with
duplicate indices. Effect checker validates ScatterAdd's Mutate annotation.

**Migration verification:** The old Gather transpose rule (argsort-based permutation path,
L1617–1647) is **deleted entirely** — `scatter_add` handles both unique and duplicate indices. The
old code is not preserved as a fallback.

```bash
grep -rn 'requires a scatter_add primitive' src/   # must return zero matches
grep -rn 'argsort.*inverse.*permutation' src/frontend/linearize.ts  # must return zero matches
```

#### M2.2 — WebGPU Atomics (CAS Loop)

**What:** Implement `scatter_add` dispatch in the WebGPU backend.

**The problem:** Multiple GPU threads may write to the same output index simultaneously. WGSL
provides `atomicAdd` for `i32`/`u32`, but not for `f32`. The solution is a Compare-And-Swap (CAS)
loop on the bitcast representation.

**JIT compilation** (`src/frontend/jit.ts`): `ScatterAdd` is a **non-kernel black node** (like Scan,
DUS) — it gets its own `JitStep` type:

```typescript
// Add to JitStep union:
| { type: "scatter_add"; target: JitId; indices: JitId; updates: JitId;
    output: JitId; axis: number; exe: Executable }
```

`splitGraphDataflow` classifies `Primitive.ScatterAdd` as a non-kernel black (exempt from P2
maxArgs, has its own step type).

**WGSL shader** (`src/backend/webgpu.ts`):

```wgsl
// For i32/u32 (native atomicAdd):
@group(0) @binding(0) var<storage, read_write> target : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read> updates : array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&updates)) { return; }
    let idx = indices[i];
    atomicAdd(&target[idx], updates[i]);
}

// For f32 (CAS loop — always safe, even without shader-f32-atomic-add):
@group(0) @binding(0) var<storage, read_write> target : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read> updates : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&updates)) { return; }
    let idx = indices[i];
    let val = updates[i];
    // CAS loop: atomically add val to target[idx]
    var old_bits = atomicLoad(&target[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&target[idx], old_bits, new_bits);
        if (result.exchanged) { break; }
        old_bits = result.old_value;
    }
}
```

When `backend.capabilities.atomicF32Add` is true, use the native
`atomicAdd(&target[idx], bitcast<u32>(val))` path instead of CAS.

**Files touched:**

- `src/frontend/jit.ts` — `scatter_add` JitStep type, blackNode classification, execution case
- `src/backend/webgpu.ts` — `prepareScatterAdd()` shader generation, `dispatchScatterAdd()`
- `src/backend/webgpu/codegen.ts` — grid calculation for scatter dispatches

**Exit criteria:** `scatter_add` produces correct results on WebGPU with highly duplicated indices
(e.g., 10000 updates to 100 positions).

#### M2.3 — Wasm `scatter_add`

**What:** Implement sequential `scatter_add` in the Wasm backend. Since Wasm is single-threaded, no
atomics are needed — a simple loop suffices.

**wasmblr implementation** (`src/backend/wasm/routines/scatter-add.ts`):

```typescript
export function buildScatterAddModule(
  dtype: "f32" | "f64" | "i32",
  updateSize: number,
): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  cg.memory.import("env", "memory");

  // scatter_add(targetPtr, indicesPtr, updatesPtr)
  const func = cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const i = cg.local.declare(cg.i32);
    hl.forLoop(i, 0, updateSize, () => {
      // idx = indices[i]
      const idx = cg.local.declare(cg.i32);
      hl.load("i32", 1 /* indicesPtr */, hl.getExpr(i));
      cg.local.set(idx);
      // target[idx] += updates[i]
      hl.store(dtype, 0 /* targetPtr */, hl.getExpr(idx), () => {
        hl.load(dtype, 0, hl.getExpr(idx)); // target[idx]
        hl.load(dtype, 2 /* updatesPtr */, hl.getExpr(i)); // updates[i]
        hl.binOp(dtype, "add");
      });
    });
  });
  cg.export(func, "scatter_add");
  return cg.finish();
}
```

**Files touched:**

- `src/backend/wasm/routines/scatter-add.ts` — new file
- `src/backend/wasm/routines/index.ts` — export
- `src/backend/wasm/routine-provider.ts` — add to `routineBuilders` map
- `src/backend/wasm.ts` — dispatch case

**Exit criteria:** Wasm backend passes all `scatter_add` tests.

---

### M3 — Multi-Output Kernels & Epilogue Fusion (5–7 days)

#### M3.1 — Multi-Output `Kernel` Support (and merge with `Kernel`)

**What:** Extend `Kernel` to support multiple outputs, then retire the single-output class —
`Kernel` is just a degenerate `MultiKernel`. The two types should be unified rather than carried
forward in parallel.

**Current state:** `Kernel` (`src/alu.ts` L1450) is single-output:
`constructor(nargs, size, exp, reduction?)`. `splitGraphDataflow` (`src/frontend/jit.ts` L1260)
materializes every output as a separate black node.

**Migration plan — replace, don't extend:**

1. Introduce `MultiKernelOutput` and the new multi-output `Kernel` class (renaming the old class
   away is fine since it's an internal type):

```typescript
export interface KernelOutput {
  readonly exp: AluExp;
  readonly reduction?: Reduction;
  readonly dtype: DType;
  readonly bytes: number;
}

// Replaces the old single-output Kernel entirely:
export class Kernel implements FpHashable {
  constructor(
    readonly nargs: number,
    readonly size: number,
    readonly outputs: KernelOutput[],     // 1..N outputs
  ) {}

  /** Convenience: true when this is effectively single-output. */
  get isSingleOutput(): boolean { return this.outputs.length === 1; }

  /** Compatibility shim for callsites that still use kernel.exp / kernel.reduction. */
  get exp(): AluExp { return this.outputs[0].exp; }
  get reduction(): Reduction | undefined { return this.outputs[0].reduction; }
  get dtype(): DType { return this.outputs[0].dtype; }
}

/** Factory for the common single-output case, used throughout the existing codebase. */
export function singleKernel(nargs: number, size: number, exp: AluExp, reduction?: Reduction): Kernel {
  return new Kernel(nargs, size, [{ exp: exp.simplify(), reduction, dtype: ..., bytes: ... }]);
}
```

2. Replace all `new Kernel(nargs, size, exp, reduction)` call sites with `singleKernel(...)`.
3. Remove the compatibility shims (`get exp()`, `get reduction()`, `get dtype()`) once all call
   sites are migrated. This MUST happen within M3.1 — not deferred to a later milestone.

**`splitGraphDataflow` changes** (`src/frontend/jit.ts`):

Add a post-pass after P2: for each group of black-node equations that share the same set of
transitive inputs and have the same `size`, check if they can be fused into a multi-output `Kernel`.
Two kernels are fusible if:

1. Same `nargs` and same `size` (same loop bounds)
2. No data dependency between them (neither reads the other's output)
3. Combined buffer count ≤ `backend.maxArgs`

**JitStep changes** — `outputs` becomes a list (was a single JitId):

```typescript
// Update execute step to support multiple outputs:
| { type: "execute"; inputs: JitId[]; outputs: JitId[];
    source: Kernel | Routine }
```

**WASM codegen** (`src/backend/wasm.ts`): The existing `codegenWasm()` path already handles
`Kernel`. With `outputs.length === 1` the code path is identical to before; with
`outputs.length > 1`, the gidx loop emits multiple store instructions:

```typescript
// For each output: compute exp, store to outputN[gidx]
for (let oi = 0; oi < kernel.outputs.length; oi++) {
  // Push output address: outputN + gidx * byteWidth
  // Evaluate expression
  // Store
}
```

**WebGPU codegen** (`src/backend/webgpu.ts`):

```wgsl
// Multiple output bindings:
@group(0) @binding(0) var<storage, read> in0 : array<f32>;
// ...
@group(0) @binding(N) var<storage, read_write> out0 : array<f32>;
@group(0) @binding(N+1) var<storage, read_write> out1 : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let gidx = gid.x;
    if (gidx >= size) { return; }
    // Shared expression evaluation (CSE across outputs)
    let shared_val = in0[gidx] * in1[gidx];
    out0[gidx] = shared_val + bias[gidx];
    out1[gidx] = max(shared_val, 0.0);
}
```

**Files touched:**

- `src/alu.ts` — replace `Kernel` with multi-output version, add `KernelOutput`, `singleKernel()`
- `src/frontend/jit.ts` — migrate `new Kernel(...)` → `singleKernel(...)`, multi-output fusion pass
  in `splitGraphDataflow`, `execute` step outputs as list, `effectDrivenAllocate` multi-output
- `src/backend/wasm.ts` — `codegenWasm()` handles `outputs` array; single-output path unchanged
- `src/backend/webgpu.ts` — multi-output `pipelineSource()`
- All other files that construct `new Kernel(...)` — migrate to `singleKernel()`

**Test file:** `test/multi-kernel.test.ts`

| Test name                                      | What it verifies                            |
| ---------------------------------------------- | ------------------------------------------- |
| `singleKernel constructs correct Kernel`       | Factory shim, `outputs.length === 1`        |
| `two same-size outputs fuse into one dispatch` | JIT pprint shows single execute step        |
| `correctness with shared subexpressions`       | Both outputs correct                        |
| `different-size outputs remain separate`       | No incorrect fusion                         |
| `grad backward pass fuses`                     | AD backward bodies produce fewer dispatches |
| `multi-output with reduction`                  | One output has reduce, one doesn't          |
| `respects maxArgs limit`                       | Falls back to separate when too many inputs |

**Exit criteria:** `Kernel` class is unified (multi-output). `singleKernel()` factory used
everywhere the old `new Kernel(nargs, size, exp)` appeared. Compatibility shims removed.
Multi-output kernels compile and execute correctly on both backends. AD backward passes with
multiple same-size outputs show fewer dispatches.

**Migration verification:**

```bash
grep -rn 'new Kernel(' src/              # must return zero matches (all migrated to singleKernel)
grep -rn 'get exp()' src/alu.ts          # must return zero matches (shims removed)
pnpm check                                # no type errors from stale Kernel constructor usage
```

#### M3.2 — Reduction Epilogue Fusion

**What:** Extend `splitGraphDataflow`'s existing reduction epilogue logic (L1290–1340) to fuse more
aggressively. Currently, a reduction (Dot/Reduce/Conv) can only fuse with a single downstream
unary/binary op. Extend to fuse with chains of elementwise ops.

**Current behavior** (`src/frontend/jit.ts` L1290–1340): The reduction epilogue fusion checks if the
downstream equation is a simple unary/binary op and fuses it into the `Reduction.epilogue` field.

**Extended behavior:** Walk downstream from a reduction output through a chain of elementwise ops
(add, mul, relu, etc.) until hitting a black node or fork. Compose all ops into a single
`Reduction.epilogue` expression.

**Example:**

```typescript
// Before: 3 dispatches
const c = np.matmul(A, B); // Dot → Kernel(Mul+Reduce)
const d = c.add(bias); // Add → Kernel
const e = nn.relu(d); // Max(0, x) → Kernel

// After: 1 dispatch
// Dot with epilogue: relu(acc + bias[gidx])
```

**Constraints:**

- Epilogue ops must be elementwise (no reductions, no shape changes)
- Epilogue expression must not read from other reduction outputs (no cross-dependency)
- Total binding count (inputs + outputs) must stay within `backend.maxArgs`
- Pure unary epilogues (relu, neg, exp) add **zero** new buffer bindings — they reuse the reduction
  output slot. Binary epilogues with a new external input (e.g., bias add) add one binding each. The
  existing `splitGraphDataflow` P2 pass (`jit.ts` L1440–1530) already enforces `maxArgs` via
  backtracking — it counts transitive dependencies and marks excess inputs as black nodes. No new
  `maxArgs` handling is needed; the P2 pass extends naturally to longer epilogue chains.

**Files touched:**

- `src/frontend/jit.ts` — extend epilogue fusion walk in `splitGraphDataflow`
- No backend changes needed — epilogue is already an `AluExp` that both backends evaluate

**Test file:** add to `test/multi-kernel.test.ts`

| Test name                                          | What it verifies                        |
| -------------------------------------------------- | --------------------------------------- |
| `matmul + bias + relu fuses to 1 dispatch`         | JIT pprint: single kernel with epilogue |
| `matmul + bias fuses (existing behavior extended)` | Chain of 2 ops                          |
| `matmul + divergent consumers stays separate`      | Fork prevents fusion                    |

**Exit criteria:** `jit(x => relu(matmul(A, B).add(bias)))` compiles to a single dispatch on both
backends.

---

### M4 — Polymorphic Shapes (Dynamic Dimensions) (5–7 days)

Allow users to mark axes as dynamic so that JIT programs can handle variable-length inputs without
recompilation.

#### M4.1 — Symbolic Dimension Type & Shape Propagation

**What:** Introduce a `SymDim` type for symbolic dimensions, extending the shape system.

**New types** (`src/shape.ts`):

```typescript
/** A dimension is either a concrete number or a symbolic variable. */
export type Dim = number | SymDim;

export class SymDim {
  constructor(readonly name: string) {}
  toString(): string {
    return this.name;
  }
}

// ShapeTracker.shape becomes Dim[] instead of number[]:
export class ShapeTracker {
  get shape(): Dim[] { ... }
  get size(): Dim | number { ... } // product of dims, symbolic if any dim is symbolic
}
```

**Tracing with symbolic dims:** When `jit(f, { dynamic_axes: { 0: "T" } })` is called:

1. The tracer creates `ShapedArray` instances with symbolic dims: `[SymDim("T"), 64]`
2. Shape propagation rules compute output shapes symbolically:
   - `reshape([T, 64], [-1])` → `[T * 64]` (symbolic product)
   - `broadcast([T, 1], [T, 64])` → `[T, 64]`
   - `reduce([T, 64], axis=1)` → `[T]`
3. `Kernel.size` becomes `Dim | number` — symbolic when any input has a symbolic dim

**Shape guard at call time:** When the JIT function is called with concrete inputs, the symbolic
dims are resolved and checked against cached compilations.

**Operations that CANNOT support symbolic dims** (initially):

| Operation                       | Why                                        | Fallback   |
| ------------------------------- | ------------------------------------------ | ---------- |
| `reshape` with computed dim     | `-1` inference needs concrete size         | Specialize |
| `concatenate`                   | Output size depends on all input sizes     | Specialize |
| `pad`                           | Padding amounts are typically compile-time | Specialize |
| Routines (sort, cholesky, etc.) | Size-specialized wasmblr modules           | Specialize |

**Files touched:**

- `src/shape.ts` — `Dim`, `SymDim` types, update `View` and `ShapeTracker` to use `Dim[]`
- `src/frontend/core.ts` — shape propagation rules for symbolic dims
- `src/frontend/jaxpr.ts` — `ShapedArray` uses `Dim[]` for shape
- `src/alu.ts` — `Kernel.size` becomes `Dim | number`
- `src/frontend/jit.ts` — `jit()` accepts `dynamic_axes` option, shape guard logic

**Test file:** `test/polymorphic-shapes.test.ts`

| Test name                                | What it verifies                     |
| ---------------------------------------- | ------------------------------------ |
| `same JIT program for different lengths` | No recompilation                     |
| `simple elementwise with dynamic axis`   | `x.add(1)` works for varying T       |
| `reduce along static axis`               | `x.sum(axis=1)` with dynamic axis 0  |
| `matmul with dynamic batch dim`          | `[T,M] @ [M,N]` → `[T,N]`            |
| `unsupported op forces specialization`   | `reshape` with `-1` triggers retrace |

**Exit criteria:** A JIT function traced with `dynamic_axes: {0: "T"}` handles inputs of different
leading dimensions without recompilation.

#### M4.2 — Parameterized Backend Codegen ✅

**Status: DONE** — Symbolic reduction sizes are now fully supported across all backends.

**What was done:**

1. **Core type change:** `Reduction.size: number → SizeExpr` (`src/alu.ts`). Added
   `Reduction.concreteHint?: number` for expression simplification during compilation.
2. **Kernel getters:** `Kernel.hasSymbolicReduction` and `Kernel.needsDynamicParams` (`src/alu.ts`).
3. **Tuner fallback:** `tuneWebgpu()` falls back to `tuneNullopt()` when
   `kernel.hasSymbolicReduction` (`src/tuner.ts`).
4. **WASM codegen:** `codegenWasm()` and `codegenWasmMulti()` add an extra i32 parameter for
   resolved reduction size when symbolic. `emitKernelBody()` accepts `reduceSizeLocal` option for
   dynamic loop bound. `dispatch()` extracts `dynamicParams.slice(1)` as extra args.
5. **WebGPU codegen:** `pipelineSource()` builds `struct Dims` conditionally (total_size and/or
   reduce_size). Reduction for-loop uses `dims.reduce_size` when symbolic. `pipelineSubmit()` builds
   uniform buffer matching struct Dims layout from `dynamicParams`.
6. **JIT execution:** `dynamicParams` layout is `[resolvedTotalSize, resolvedReduceSize?]`. Built
   when `kernel.needsDynamicParams` is true.
7. **Pool hints:** `computePoolHints()` resolves symbolic sizes via `_currentDimBindings` instead of
   skipping them.
8. **Mega-module:** `canCompileToMegaModule()` rejects kernels with symbolic reduction sizes (cannot
   inline reduction loops with symbolic bounds as i32.const). Type signatures updated for SizeExpr.

**`dynamicParams` layout:**

```
dynamicParams[0] = resolvedTotalSize (always present when needsDynamicParams)
dynamicParams[1] = resolvedReduceSize (when kernel has symbolic reduction)
```

**WASM kernel signature:** `(start, end, ...ptrs, [reduceSize])` — extra i32 when symbolic
reduction.

**WebGPU `struct Dims`:** Conditionally includes `total_size: u32` and/or `reduce_size: u32` at
`@group(1) @binding(0)`.

**Files touched:** `src/alu.ts`, `src/tuner.ts`, `src/frontend/jit.ts`, `src/backend/wasm.ts`,
`src/backend/webgpu.ts`, `src/backend/webgpu/codegen.ts`, `src/backend/wasm/mega-module.ts`.

**Exit criteria (met):** Reducing along a dynamic axis (e.g., `np.sum(x, 0)` with
`dynamic_axes: { 0: "T" }`) produces correct results for different input sizes without
recompilation. 5 new tests in `test/polymorphic-shapes.test.ts` verify sum, max, non-uniform values,
concrete+symbolic combinations, and chained elementwise+reduce patterns.

---

### M5 — Wasm Multithreading Foundation (5–7 days)

#### M5.1 — `SharedArrayBuffer` Memory Pool

**What:** Upgrade `WasmAllocator` to optionally use `WebAssembly.Memory({ shared: true, ... })`.

**Key constraint:** `shared: true` requires `maximum` pages to be specified upfront. Strategy: start
with 256 MB maximum (4096 pages), grow as needed within that cap.

**Files touched:**

- `src/backend/wasm/allocator.ts` — constructor accepts `shared: boolean`, uses
  `new WebAssembly.Memory({ initial, maximum, shared })` when `crossOriginIsolated`
- `src/backend/wasm.ts` — pass `shared: backend.capabilities.sharedMemory` to allocator

**Exit criteria:** All existing tests pass with shared memory enabled (sequential execution). No
behavior change — just the memory type changes.

#### M5.2 — `WasmWorkerPool`

**What:** Create a pool of Web Workers that share the same `WebAssembly.Memory`.

```typescript
// src/backend/wasm/worker-pool.ts
export class WasmWorkerPool {
  readonly workers: Worker[];
  readonly numWorkers: number; // navigator.hardwareConcurrency - 1

  constructor(memory: WebAssembly.Memory) { ... }

  /** Dispatch a parallel loop: each worker runs [start, end) slice. */
  dispatchParallel(
    moduleBytes: Uint8Array,
    funcName: string,
    totalSize: number,
    args: number[], // shared memory pointers
  ): Promise<void>;

  destroy(): void;
}
```

Workers are initialized with a lightweight message handler:

```typescript
// src/backend/wasm/worker-entry.ts (inlined as Blob URL)
self.onmessage = async (e) => {
  const { moduleBytes, memory, funcName, start, end, args } = e.data;
  const module = await WebAssembly.instantiate(moduleBytes, { env: { memory } });
  module.instance.exports[funcName](start, end, ...args);
  self.postMessage({ done: true });
};
```

Synchronization uses `postMessage`/`Promise.all` for the JS→Worker dispatch in `dispatchParallel`,
which is async and works on the main thread. For intra-module synchronization (M6.2), see the
orchestrator-worker pattern below — `Atomics.wait`/`Atomics.notify` can only be used from worker
threads, not the main browser thread.

**Files touched:**

- `src/backend/wasm/worker-pool.ts` — new file
- `src/backend/wasm/worker-entry.ts` — new file (Worker script)
- `src/backend/wasm.ts` — initialize pool at backend startup when `crossOriginIsolated`

**Exit criteria:** Worker pool initializes, dispatches a basic elementwise add across workers,
produces correct results.

#### M5.3 — Parallel `wasmblr` Loops

**What:** Add `parallelForLoop` to `WasmHl` and wire it into kernel codegen.

```typescript
// src/backend/wasm/wasmblr-hl.ts
class WasmHl {
  /** Generate loop body that accepts (start, end) parameters for parallel dispatch. */
  parallelForLoop(
    i: number, // loop variable local
    body: () => void,
  ): void {
    // Function signature: (start: i32, end: i32, ...args) -> ()
    // Loop: for (i = start; i < end; i++) { body(); }
  }
}
```

**Integration with kernel dispatch:** When `WasmWorkerPool` is available and array size > threshold
(e.g., 4096 elements), `WasmBackend.dispatch()` splits the gidx range across workers:

```typescript
dispatch(exe, inputs, outputs) {
  if (this.workerPool && exe.source instanceof Kernel && exe.source.size > 4096) {
    this.workerPool.dispatchParallel(exe.module, "kernel", exe.source.size, [...ptrs]);
  } else {
    // Single-threaded: call kernel(0, size, ...ptrs)
    exe.instance.exports.kernel(0, exe.source.size, ...ptrs);
  }
}
```

**Files touched:**

- `src/backend/wasm/wasmblr-hl.ts` — `parallelForLoop`
- `src/backend/wasm.ts` — parallel dispatch in `dispatch()`, threshold logic
- `src/backend/wasm.ts` — `codegenWasm()` generates `(start, end, ...ptrs)` signature

**Test file:** `test/wasm-parallel.test.ts`

| Test name                                       | What it verifies                               |
| ----------------------------------------------- | ---------------------------------------------- |
| `parallel add produces correct results`         | Large array add matches sequential             |
| `parallel reduce`                               | Correctness with per-worker partial reductions |
| `small arrays stay single-threaded`             | Size < threshold uses main thread              |
| `graceful fallback without crossOriginIsolated` | No workers created, sequential OK              |

**Exit criteria:** Large elementwise operations show near-linear speedup on multi-core CPUs. All
existing tests pass.

**Migration verification:** All WASM kernels use the new `(start, end, ...ptrs)` signature. The old
`(ptrs...) -> ()` format is retired — single-threaded dispatch passes `(0, size, ...ptrs)`.

```bash
# Verify no codegen path still emits the old signature:
grep -rn 'kernel(.*ptrs)' src/backend/wasm.ts  # all calls must include start, end arguments
pnpm check
```

---

### M6 — Whole-Program Wasm Compilation (The "Mega-Module") (6–8 days)

Compile an entire `JitProgram` into a single wasmblr module, eliminating JS↔Wasm boundary
crossings.

#### M6.1 — `JitProgram` to Wasm Translator

**What:** Add `compileToWasmModule(program: JitProgram): WasmMegaModule` that translates the full
step list into a single Wasm function.

**Architecture:**

```typescript
// src/backend/wasm/mega-module.ts
export interface WasmMegaModule {
  readonly module: WebAssembly.Module;
  readonly instance: WebAssembly.Instance;
  /** Execute: pass input slot pointers, get output slot pointers. */
  execute(inputPtrs: number[], symDims?: Record<string, number>): number[];
}
```

**Step-by-step translation:**

| JitStep            | Wasm translation                                                                  |
| ------------------ | --------------------------------------------------------------------------------- |
| `malloc(id, size)` | `local.set(id, call $bump_alloc(size))` — calls imported `WasmAllocator.malloc`   |
| `free(id)`         | `call $free(local.get(id))` — calls imported `WasmAllocator.free`                 |
| `recycle(a → b)`   | `local.set(b, local.get(a))` — zero-cost local rename                             |
| `execute(kernel)`  | Inline the kernel's gidx loop body directly (no function call overhead)           |
| `execute(routine)` | `call $routine_N(...)` — call imported pre-compiled routine                       |
| `incref(id)`       | `call $incref(local.get(id))` — calls imported `Backend.incRef`                   |
| `scan(...)`        | Inline the scan loop (for compiled-loop) or `call $scan_dispatch(...)` (fallback) |
| `dus(...)`         | `call $memory_copy(dst, src, offset, len)` — bulk memory op                       |

**Memory management within the module:**

- Wasm locals (`local.declare(cg.i32)`) hold slot pointers for each `JitId`
- `malloc`/`free` are imported from the `WasmAllocator`
- `recycle` is a zero-cost local-to-local copy (no import call)
- Kernel bodies are inlined — the gidx loop runs directly in the mega-module
- Routines are imported as separate Wasm functions (already compiled by `routine-provider.ts`)

**Key constraint — polymorphic sizes:** When M4 is active, `malloc` sizes may be symbolic. The
mega-module function signature includes symbolic dim parameters:

```typescript
// mega_execute(input0_ptr, input1_ptr, ..., symDim_T) -> (output0_ptr, output1_ptr, ...)
// All sizes computed from symDim_T at runtime within the Wasm module
```

**Files touched:**

- `src/backend/wasm/mega-module.ts` — new file: `compileToWasmModule()`
- `src/frontend/jit.ts` — add `compiledMegaModule?: WasmMegaModule` cache on `JitProgram`
- `src/backend/wasm.ts` — integration: when `JitProgram` has Wasm backend, compile and cache

**Test file:** `test/mega-module.test.ts`

| Test name                                    | What it verifies                           |
| -------------------------------------------- | ------------------------------------------ |
| `5-step chain compiles to 1 Wasm call`       | Benchmark: 1 JS→Wasm crossing              |
| `correctness matches step-by-step execution` | Same results as current JitProgram.execute |
| `malloc/free/recycle inside mega-module`     | Internal memory management works           |
| `routine calls via imports`                  | Cholesky/sort inside mega-module           |
| `scan inside mega-module`                    | Compiled-loop scan embedded                |
| `mega-module with symbolic dims`             | Variable-length inputs (depends on M4)     |

**Exit criteria:** A JitProgram with 50 operations compiles to 1 Wasm module and executes with 1 JS
call. Correctness matches step-by-step execution on all test cases.

#### M6.2 — Mega-Module Multithreading

**What:** Wire the `WasmWorkerPool` (M5) into the mega-module. Parallelizable loops (large
elementwise kernels) dispatch to workers without returning to JS.

**Key constraint — `Atomics.wait` on the main thread:** Browsers forbid `Atomics.wait` (and the Wasm
equivalent `memory.atomic.wait32`) on the main thread to prevent UI freezing. The mega-module cannot
call `Atomics.wait` to wait for workers if it runs on the main thread. Node.js and Deno do not have
this restriction.

**Approach — orchestrator worker:** The mega-module itself runs inside a dedicated orchestrator
worker. The main thread sends input slot pointers to the orchestrator via `postMessage` (zero-copy
via `SharedArrayBuffer`). The orchestrator worker runs the mega-module synchronously — it CAN call
`Atomics.wait` since it's not the main thread. Inside the module, large kernels fan out to the
`WasmWorkerPool` via `Atomics.notify` → `Atomics.wait`:

```
Main thread                    Orchestrator Worker           Compute Workers
    │                               │                            │
    ├─ postMessage(inputPtrs) ─────→│                            │
    │                               ├─ mega_execute(...)         │
    │                               │   ├─ inline kernel A       │
    │                               │   ├─ parallel kernel B:    │
    │                               │   │   Atomics.notify ─────→│ run chunk
    │                               │   │   Atomics.wait ←───────│ done
    │                               │   ├─ inline kernel C       │
    │                               │   └─ return outputs        │
    │←── postMessage(outputPtrs) ───┤                            │
```

From the `JitProgram.execute()` API perspective, this is async (`Promise<number[]>`). The existing
sync `execute()` can remain for non-threaded backends; threaded execution returns a promise that
resolves when the orchestrator posts back.

**Files touched:**

- `src/backend/wasm/mega-module.ts` — parallel dispatch within mega-module
- `src/backend/wasm/worker-pool.ts` — shared control block protocol, orchestrator worker
- `src/backend/wasm/orchestrator-entry.ts` — new file (orchestrator worker script)

**Exit criteria:** Complex JIT programs execute entirely in Wasm, utilizing all CPU cores. Works in
both browsers (via orchestrator worker) and Node.js/Deno (direct main-thread execution).

---

### M7 — Native `associativeScan` Compilers (4–6 days)

**Design decision — Primitive with polymorphic-N-compatible transpose:**

`associativeScan` is registered as `Primitive.AssociativeScan` with a body sub-jaxpr. This gives:

1. **Clean IR representation** — one Jaxpr equation with a body sub-jaxpr vs O(body_ops × log N)
   unrolled equations. This matches how `Primitive.Scan` represents `lax.scan`.
2. **Backend specialization** — the JIT compiler can recognize the primitive and route to
   `codegenNativeAssociativeScan()` automatically.
3. **Consistency** — both `scan` and `associativeScan` are higher-order control flow operations.
4. **Polymorphic forward execution** — the `assoc_scan` JitStep runs Kogge-Stone at runtime with
   concrete N from input shapes. When M4 adds `SymDim` + `dynamic_axes`, a single JIT compilation
   reuses across different N values without re-tracing.

**AD compatibility — why "unroll-during-trace" is wrong for this primitive:**

The JVP rule doubles the body and is straightforward. For transposition (backward pass), the naive
approach — calling `associativeScanCore()` to unroll `ceil(log₂ N)` Kogge-Stone rounds into the
traced Jaxpr — is **incompatible with polymorphic N** because `ceil(log₂ N)` is not computable when
N is a `SymDim`. This also applies to `Primitive.Scan`'s transpose (which uses concrete loops).

Instead, the transpose rule implements a **reverse sequential recurrence** that walks backward
through the scan positions. The forward associative scan computes:

```
y[0] = x[0]
y[i] = fn(y[i-1], x[i])    for i = 1..N-1
```

The adjoint (backward) of this linear map is:

```
ct_carry = 0
for i = N-1 down to 1:
    ct_carry += ct_y[i]
    [ct_a, ct_b] = fn_T(ct_carry, primals y[i-1] and x[i])
    ct_x[i] = ct_b
    ct_carry = ct_a
ct_x[0] = ct_carry + ct_y[0]
```

This is a reverse sequential scan of N-1 iterations. For M7.1 it runs as a concrete loop (matching
`Primitive.Scan`'s transpose approach). For future polymorphic N, it would emit `Primitive.Scan`
instead of unrolling — the same upgrade path both primitives share.

**Scope for M7:** M7.1 introduces `Primitive.AssociativeScan` with proper PE and transpose. M7.2
adds the WASM compiled Kogge-Stone backend. M7.3 adds multithreaded inner loops.

#### M7.1 — `Primitive.AssociativeScan` & Transform Rules

**What:** Register `associativeScan` as a primitive with body sub-jaxpr.

**Primitive definition** (`src/frontend/core.ts`):

```typescript
// Add to Primitive enum:
AssociativeScan = "associative_scan",

// Add to PrimitiveParams:
[Primitive.AssociativeScan]: {
  jaxpr: Jaxpr;        // body: (consts..., a_leaves..., b_leaves...) => result_leaves
  numLeaves: number;    // number of pytree leaves
  axis: number;
  reverse: boolean;
};
```

**Transform rules:**

| Transform | Strategy                                                                                      |
| --------- | --------------------------------------------------------------------------------------------- |
| JVP       | Double the body (primal+tangent), run single associativeScan with 2× leaves                   |
| Transpose | Reverse sequential recurrence (concrete loop; emits `Primitive.Scan` when M4 adds polymorphic |
|           | length). Mirrors `Scan`'s transpose approach. Needs forward primals as residuals.             |
| Vmap      | Move batch dim, run batched associativeScan (same as current impl)                            |
| PE        | JVP-split: run primals forward (known), mark tangents unknown (mirrors `#partialEvalScan`)    |

**Key design — reverse sequential transpose:**

The transpose rule computes the backward pass as a reverse sequential recurrence:

1. Extract primal (concrete) and tangent (UndefPrimal) inputs from `args`
2. Recompute forward primals `y_P[0..N-1]` from primal inputs
3. Construct the transposed body jaxpr (transpose w.r.t. tangent inputs only, primals as residuals)
4. Run backward loop: iterate from N-1 down to 1, applying the transposed body at each position
5. Return cotangents for tangent inputs

The concrete loop runs N-1 iterations. With M4's `SymDim`, this would emit `Primitive.Scan` — the
same polymorphic-N upgrade path that `Primitive.Scan`'s own transpose will follow.

**Polymorphic N status:**

| Component   | Polymorphic? | Notes                                                            |
| ----------- | ------------ | ---------------------------------------------------------------- |
| Forward JIT | ✅ Runtime N | `assoc_scan` JitStep runs Kogge-Stone with concrete N from shape |
| JIT caching | ⏳ Needs M4  | Same shape → cache hit; different N → re-trace until M4 SymDim   |
| Backward    | ⏳ Needs M4  | Concrete loop for now; emit `Primitive.Scan` for symbolic N      |

**Files touched:**

- `src/frontend/core.ts` — `Primitive.AssociativeScan`, params type
- `src/frontend/jaxpr.ts` — abstract eval rule
- `src/frontend/array.ts` — eager impl rule (calls `associativeScanCore`)
- `src/frontend/jvp.ts` — JVP rule (double the body)
- `src/frontend/vmap.ts` — batching rule
- `src/frontend/linearize.ts` — PE dispatch (`#partialEvalAssociativeScan`), transpose rule
- `src/library/lax-associative-scan.ts` — emit primitive instead of calling core directly
- `src/frontend/jit.ts` — `associative_scan` JitStep type, blackNode classification

**Exit criteria:** All existing `test/lax-associative-scan.test.ts` tests pass. `makeJaxpr` shows a
single `associative_scan` equation. `grad(associativeScan)` produces correct results.

**Migration verification:** The public `lax.associativeScan()` entry point always emits
`Primitive.AssociativeScan` — the old direct call to `associativeScanCore()` is removed from the
public path. `associativeScanCore()` is retained as the primitive's impl rule body (for eager
execution and JIT step execution).

```bash
# lax-associative-scan.ts should emit the primitive, not call core directly:
grep -n 'associativeScanCore' src/library/lax-associative-scan.ts
# Expected: zero matches in the public function body; core is imported only by array.ts impl rule
pnpm check
```

#### M7.2 — WASM Compiled Kogge-Stone + Polymorphic Length

**What:** Add `codegenNativeAssociativeScan()` to `src/backend/wasm.ts` that compiles the full
Kogge-Stone ladder — stride-doubling loop, ping-pong buffers, per-round `fn` application — into one
Wasm module. The module takes `N` as a runtime parameter so a single compilation serves all input
lengths.

**Why polymorphic length matters:** When a user writes
`jit((xs) => lax.associativeScan(fn, xs), { dynamic_axes: { 0: "T" } })`, the JIT-compiled program
must handle different lengths without re-tracing or recompilation. The body jaxpr is N-independent
(traced with per-element shapes, scan axis removed), and the Kogge-Stone structure is the same for
any N — only loop bounds and buffer sizes change. Making length a runtime parameter is both natural
and required for the `dynamic_axes` use case.

**Algorithm inside the Wasm module:**

The compiled module runs single-threaded within one Wasm invocation. The primary win is eliminating
ceil(log₂ N) JS→Wasm crossings.

```
// N is a runtime parameter (i32 function argument)
allocate ping/pong buffers (2 × N × leafBytes)
copy input leaves to ping buffer
for stride = 1, 2, 4, ..., while stride < N:
    for i = stride..N:  // per-element fn application
        pong[i] = fn(ping[i - stride], ping[i])
    for i = 0..stride:
        pong[i] = ping[i]  // prefix: unchanged
    swap ping <-> pong
copy result from ping to output
free scratch buffers
```

**Polymorphic length design:**

The WASM module function signature is `(N: i32, ...leafPtrs: i32[]) => void`. `N` is a runtime
parameter resolved from the concrete input shape at execution time. The body kernels are compiled
with concrete per-element sizes (strides, element widths) — only the outer loop bounds
(`stride < N`, `i < N`) and buffer offsets (`i * elemSize`) use `N`. This means:

- **One compilation, any N:** The WASM module is compiled once per body-shape signature (e.g., "f32
  scalar cumsum") and cached. Different call-site Ns reuse the same module.
- **No SymDim in module codegen:** The WASM bytecode uses concrete element sizes throughout. Only
  the loop bounds reference the `N` local variable.
- **Buffer allocation at call time:** Ping/pong scratch buffers are allocated by the caller
  (JS-side) based on the concrete N, then passed as pointers to the WASM function. This avoids
  WASM-side `memory.grow` and lets the allocator/pool manage memory.

**Integration with `dynamic_axes`:**

| Layer                                     | How N flows                                             |
| ----------------------------------------- | ------------------------------------------------------- |
| `jit({ dynamic_axes: { 0: "T" } })`       | Traces with `SymDim("T")` on axis 0                     |
| `Primitive.AssociativeScan` abstract eval | Output shapes preserve `SymDim("T")`                    |
| `jitCompile()` → `assoc_scan` JitStep     | Body jaxpr compiled, `N` left as runtime param          |
| `JitProgram.execute()`                    | Resolves `N` from concrete input shape or `dimBindings` |
| WASM dispatch                             | `N` passed as first i32 arg to the compiled module      |

On WebGPU, the current unrolled path (ceil(log₂ N) dispatches) is already the hardware-imposed floor
— no compiled-loop is possible. The WebGPU path continues to derive N from concrete shapes at
execution time, which naturally supports polymorphic length through the existing
`_associativeScanCoreImpl` delegation. No WebGPU-specific changes are needed.

**Scan plan structure:**

The `assoc_scan` JitStep gains an optional `plan` field (analogous to `scan`'s `ScanPlan`):

```typescript
type AssocScanPlan =
  | { path: "compiled-loop"; executable: WasmAssocScanExecutable }
  | { path: "fallback" }; // JS Kogge-Stone (current impl)
```

`planAssociativeScan()` in the backend checks eligibility:

- Body is all elementwise Kernels (no Routines, no reductions requiring cross-element sync)
- Backend is WASM (WebGPU returns `null` → fallback, which is already optimal)
- All leaves have the same dtype and element size

**Files touched:**

- `src/backend/wasm.ts` — `codegenNativeAssociativeScan()`, `planAssociativeScan()`
- `src/backend.ts` — add optional `planAssociativeScan` to `Backend` interface
- `src/frontend/jit.ts` — `assoc_scan` JitStep gains `plan` field; execution dispatches via plan

**Test additions in** `test/lax-associative-scan.test.ts`:

| Test name                                      | What it verifies                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| `WASM compiled path matches unrolled`          | Correctness for cumsum, cumprod                                    |
| `WASM compiled path N=65536`                   | Performance: single WASM call vs ceil(log₂ 65536)                  |
| `pytree body on WASM compiled path`            | Flattened pytree through single module                             |
| `polymorphic length: different N same program` | `jit({ dynamic_axes })` reuses compiled module for N=100 and N=200 |
| `polymorphic length: N=1 and N=0 edge cases`   | Edge cases with runtime N                                          |

**Exit criteria:** `associativeScan` on WASM runs in a single JS→WASM call. A
`jit({ dynamic_axes })` wrapping `associativeScan` handles different input lengths without
recompilation. Performance matches or exceeds `lax.scan`'s compiled-loop for associative bodies.

#### M7.3 — Multithreaded Kogge-Stone (depends on M5/M6.2)

**What:** Parallelize the inner per-element loop across workers using the M6.2 orchestrator-worker
pattern.

**Feasibility:** The Kogge-Stone algorithm has natural parallelism — within each round, all elements
`i >= stride` compute `pong[i] = fn(ping[i-stride], ping[i])` independently. Workers share ping/pong
buffers via `SharedArrayBuffer`. The synchronization pattern is one barrier per round (O(log N)
total):

```
Orchestrator Worker:
    for stride = 1, 2, 4, ..., while stride < N:
        // Fan out inner loop to compute workers
        write (start=stride, end=N, pingPtr, pongPtr) to control block
        Atomics.notify(workers)
        Atomics.wait(completion_counter)  // OK — not main thread
        // Copy prefix
        for i = 0..stride: pong[i] = ping[i]
        swap ping <-> pong
```

**Speedup estimate:** With P workers: O(N log N / P + log N × barrier_cost). For N=65536 with 4
workers, expect ~3–4× speedup on the inner loop. The barrier cost (16 `Atomics.wait` calls) is
negligible versus the per-element computation.

**Files touched:**

- `src/backend/wasm.ts` — `codegenNativeAssociativeScan()` generates parallel inner loop
- `src/backend/wasm/mega-module.ts` — reuse orchestrator dispatch for assocScan

**Exit criteria:** Multithreaded `associativeScan` shows near-linear speedup on 4+ core CPUs.

---

### M8 — Cleanup, Benchmarking & Documentation (2–3 days)

#### M8.1 — Benchmark Suite

**What:** Create benchmark scripts comparing M0 baselines.

**Benchmark categories:**

| Benchmark                                  | What it measures               | Expected improvement    |
| ------------------------------------------ | ------------------------------ | ----------------------- |
| `scatter_add` 10K updates to 100 positions | WebGPU/WASM scatter throughput | ∞ (was impossible)      |
| `matmul + relu` fusion                     | JIT dispatch count             | 3→1 dispatches          |
| 5-step chain (size 4096)                   | Mega-module vs step-by-step    | 5→1 JS↔Wasm crossings  |
| Large elementwise (size 1M)                | Parallel vs single-thread WASM | ~4× on 4-core           |
| Variable-length JIT (100,150,200)          | Recompilation avoided          | 3→1 compilations        |
| `associativeScan` N=65536 WASM             | Compiled vs unrolled           | 16→1 JS↔Wasm crossings |

**Files:** `bench/scatter-add.bench.ts`, `bench/mega-module.bench.ts`,
`bench/parallel-wasm.bench.ts`

**Exit criteria:** Benchmark results saved to `docs/ULTIMATE-BENCHMARKS.md`.

#### M8.2 — Dead Code Audit & Documentation

**Dead code audit:** Systematic grep for old patterns that should no longer exist:

```bash
# M1 remnants:
grep -rn 'jaxprNeedsCallTimeTranspose' src/

# M2 remnants:
grep -rn 'requires a scatter_add primitive' src/

# M3 remnants:
grep -rn 'new Kernel(' src/
grep -rn 'get exp()\|get reduction()\|get dtype()' src/alu.ts

# M7 remnants:
grep -n 'associativeScanCore' src/library/lax-associative-scan.ts

# All must return zero matches.
```

Also check for:

- Unused imports left behind by refactored modules
- Dead branches (`if (false)`, commented-out old paths)
- TODO/FIXME comments referencing completed milestones

**Documentation:** Update `copilot-instructions.md` to document:

- `scatter_add` primitive, CAS loop, `MemoryEffect.Mutate` pattern
- Multi-output kernel fusion and epilogue fusion in `splitGraphDataflow`
- Wasm Mega-Module architecture
- Worker pool and parallel dispatch
- Polymorphic shapes API and limitations
- Native `associativeScan` on WASM

**Files touched:** `.github/copilot-instructions.md` — new sections in relevant parts.

**Exit criteria:** Dead code audit returns zero matches. Documentation reflects the new
architecture.

#### M8.3 — Final Regression Run

**What:** Full CI-equivalent check.

**Commands:**

```bash
pnpm build && pnpm check
pnpm vitest run
pnpm run test:deno
pnpm run lint && pnpm run format:check
```

**Exit criteria:** All checks pass. Zero regressions from M0 baseline.

---

## Dependency Graph

```
M0.1 (baseline) ──→ M0.2 (capabilities)
  │
  ├─→ M1.1 (ScanBackwardArtifact) ──→ M1.2 (unify vjpFlat)
  │
  ├─→ M2.1 (scatter_add IR+AD) ──→ M2.2 (WebGPU CAS) ──→ M2.3 (Wasm scatter)
  │
  ├─→ M3.1 (unified Kernel, multi-output) ──→ M3.2 (epilogue fusion)
  │
  ├─→ M4.1 (SymDim + shape propagation) ──→ M4.2 (parameterized codegen)
  │                                               │
  ├─→ M5.1 (SharedArrayBuffer) ──→ M5.2 (WorkerPool) ──→ M5.3 (parallel loops)
  │                                                              │
  │                                                              ↓
  └─→ M6.1 (Mega-Module compiler) ────────────────────────→ M6.2 (Mega + threads)
       (depends on M4.2 for symbolic dims)                       │
                                                                 │
  M7.1 (Primitive.AssociativeScan) ──→ M7.2 (WASM compiled) ────┤
                                             │                   │
                                             ├─ M7.3 (threaded) ┘
                                             │  (depends on M5+M6.2)
                                             │
                                             ↓
                                      M8.1 – M8.3 (cleanup)
```

**Independent tracks** (can be parallelized if multiple agents available):

- Track A: M1 → M2 (AD infrastructure)
- Track B: M3 (kernel fusion)
- Track C: M4 (polymorphic shapes)
- Track D: M5 → M6 (WASM performance)
- Track E: M7.1 → M7.2 (associativeScan primitive + compiled); M7.3 depends on Track D

## Risk Register

| Risk                                       | Impact                                     | Mitigation                                                                                      |
| ------------------------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `crossOriginIsolated` unavailable          | Wasm single-thread fallback                | Graceful degradation: `WasmAllocator` uses standard `ArrayBuffer`; Mega-Module runs sequential  |
| `Atomics.wait` blocked on main thread      | Mega-module can't sync with workers inline | Orchestrator-worker pattern (M6.2): run mega-module in a worker; Node.js/Deno can use direct    |
| WebGPU CAS loop contention too high        | `scatter_add` slow with many duplicates    | Sort indices first to reduce contention; segment-based scatter; rely on `shader-f32-atomic-add` |
| Mega-Module compilation latency            | JIT first-call spike                       | Cache by jaxpr hash; `wasmblr` generates fast (~1 KB modules); lazy compilation                 |
| Polymorphic shapes break View composition  | `ShapeTracker.reshape` needs concrete dims | Specialize on first concrete call; expand support incrementally                                 |
| Polymorphic shapes break buffer recycling  | `effectDrivenAllocate` needs concrete size | Key free pool by symbolic expression string; resolve at call time (M4.2)                        |
| Epilogue fusion register pressure (WebGPU) | GPU register spilling                      | Limit epilogue chain depth (e.g., ≤5 ops); fall back to separate kernel                         |
| Multi-output kernel binding limit          | `maxArgs` exceeded                         | `splitGraphDataflow` P2 pass already prevents this; extend to unified multi-output `Kernel`     |
| Parallel Wasm worker startup latency       | Cold-start overhead                        | Pre-initialize workers at backend startup; amortize over subsequent calls                       |

## Estimated Timeline

| Milestone | Effort   | Cumulative |
| --------- | -------- | ---------- |
| M0        | 1–2 days | 1–2 days   |
| M1        | 3–4 days | 4–6 days   |
| M2        | 4–6 days | 8–12 days  |
| M3        | 5–7 days | 13–19 days |
| M4        | 5–7 days | 18–26 days |
| M5        | 5–7 days | 23–33 days |
| M6        | 6–8 days | 29–41 days |
| M7        | 4–6 days | 33–47 days |
| M8        | 2–3 days | 35–50 days |

Total: **5–7 weeks** of focused implementation.

## Commit Strategy

- One commit per sub-task (M0.1, M0.2, ..., M8.3).
- Commit message format: `ultimate M{n}.{m}: {short description}`
- Every commit must pass `pnpm vitest run`.
- Branch off `main` at start. Merge back after M8.3 passes full CI.

### Migration Hygiene

Every milestone that replaces an old pattern MUST verify full migration before its final commit:

1. **Grep-verify removal:** After deleting old code, grep the codebase for old identifiers, error
   messages, and patterns. The milestone's exit criteria are not met until grep returns zero
   matches. Each milestone above includes a `Migration verification` block with the specific grep
   commands.
2. **No lingering shims:** Compatibility shims (deprecated accessors, dual codepaths, fallback
   branches) are temporary _intra-milestone_ scaffolding. They MUST be removed within the same
   milestone that introduces them — never deferred to "a follow-up" or "later."
3. **Type-check catches stragglers:** After removing an old type or changing a type signature,
   `pnpm check` (tsc) surfaces all unupdated callers. Run it before the milestone's final commit.
4. **Old code is deleted, not commented out:** Do not leave `// OLD: ...` blocks or `if (false)`
   branches. If the old path is genuinely needed as a fallback (e.g., non-threaded execution in M6),
   it is documented as an intentional dual path, not leftover code.
