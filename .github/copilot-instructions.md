# GitHub Copilot instructions for jax-js

These notes help AI coding agents be immediately productive. The document has two parts:

1. **Repository Overview** — General jax-js knowledge for any development work
2. **Scan Feature Reference** — `lax.scan` implementation details and backend-specific behavior

---

# Part 1: Repository Overview

## What is jax-js?

jax-js is a JavaScript/TypeScript port of [JAX](https://github.com/google/jax), Google's library for
high-performance numerical computing. It brings **numerical computing and machine learning** to the
web as a first-class capability — not just model inference, but the full stack: array operations,
automatic differentiation, JIT compilation, and composable transformations.

- **NumPy-like array operations** in the browser or Node.js
- **Automatic differentiation** — compute gradients of arbitrary functions for ML training
- **JIT compilation** — trace functions once, compile to optimized GPU/WASM code
- **Transformations** — `grad`, `vmap`, `jit` compose to build complex ML pipelines

The target audience is broad: scientists, artists, ML practitioners, data scientists — anyone who
needs numerical computing but shouldn't have to write GPU shaders or manage a Python environment.
jax-js makes the JAX/NumPy API available anywhere a browser (or Node.js/Deno) runs.

### Key concepts

**Tracing:** When you call a jit-wrapped function, jax-js executes it with "tracer" objects instead
of real arrays. This records what operations happen, producing an intermediate representation
(Jaxpr) that can be compiled to efficient native code. The function runs once for tracing, then the
compiled version runs for all subsequent calls.

**Kernels:** A "kernel" is a compiled computation unit. On GPU, this is a WGSL shader program; on
WASM, it's a WebAssembly module. jax-js _fuses_ multiple elementwise operations into single kernels
to minimize dispatch overhead — instead of launching one GPU dispatch per `add`, `mul`, `exp`, it
combines them into one.

**Autodiff intuition:** `grad(f)` returns a new function that computes `f`'s gradient. Internally,
jax-js traces `f` to build a computation graph, then applies the chain rule automatically. The `jvp`
(forward-mode) and `vjp` (reverse-mode) primitives enable efficient gradient computation for
different use cases.

## Project vision & design philosophy

jax-js solves two problems simultaneously: **numerical computing in the browser** (statistics,
signal processing, simulations, classical ML) and **GPU compute in the browser** (WebGPU shaders,
parallel number crunching). JAX's tracing-based design is the ideal bridge — it lets users write
high-level array code that compiles down to fast native kernels automatically.

### Generative compiler over static kernel libraries

The central architectural bet is that **generating kernels from an IR at runtime beats shipping a
library of pre-compiled kernels** (which is what TensorFlow.js and ONNX Runtime Web do). This means:

- New operations don't require new hand-written kernels — `matmul` is just reshape + multiply +
  reduce, compiled via `jit`.
- Kernel fusion happens automatically across arbitrary operation chains.
- Performance scales with compiler improvements, not manual kernel engineering.
- The library stays lightweight — XLA is 200+ KLoC, far too large for a browser bundle.

This approach draws from tinygrad's insight that a minimal set of operations + a view-tracking
system (ShapeTracker) + simple fusion heuristics can get you surprisingly far. Combined with JAX's
tracing model, it yields a composable system where `grad`, `jit`, and `vmap` all interoperate
naturally.

### Design tradeoffs to keep in mind

- **"80% of XLA" philosophy** — jax-js aims for the sweet spot of correctness and performance
  without squeezing out every last drop. We don't know what hardware we're running on (it's a
  browser library), so we target **3–5× of optimal** rather than peak.
- **Lightweight over exhaustive** — prefer a small, composable set of primitives over a large
  surface area of specialized ops. If something can be expressed via existing primitives and `jit`,
  that's preferred over adding a new backend kernel.
- **Explicit disposal over GC** — operations don't consume inputs, but GPU/WASM buffers must be
  freed explicitly via `.dispose()` when no longer needed. See
  [Memory management](#memory-management--ownership) for details.
- **Ownership-correct in both modes** — code must dispose correctly in eager mode **and** under
  `jit()` tracing. `jit()` is a pure performance optimization (kernel fusion, buffer recycling); it
  must not change ownership semantics. If code leaks or double-disposes in eager mode, that is a
  real bug to fix — not something to paper over by wrapping in `jit()`. See
  [Ownership correctness principle](#ownership-correctness-principle).
- **Compounding returns** — every improvement to the compiler makes _all_ operations faster, every
  new primitive gets autodiff for free, every `jit`-wrapped function gets kernel fusion
  automatically. Prioritize work that compounds.

### Development priorities

When deciding what to work on, prefer work in this order:

1. **Correctness first** — tests, reference-counting discipline, cross-backend consistency.
2. **API breadth** — approximate NumPy/JAX API compatibility (see `FEATURES.md` for the status
   table). The goal is that common JAX/NumPy patterns can be ported with minimal changes.
3. **Performance** — there is significant headroom, especially in:
   - WASM backend (SIMD is used for Cholesky f32; extending to matmul/elementwise kernels and adding
     multi-threading via SharedArrayBuffer are open opportunities).
   - Transformer inference (currently ~1/3 of raw matmul GFLOP/s).
   - Conv2d and other operations that haven't been tuned yet.
4. **Demos and applications** — fluid simulations, neural networks, audio processing, embedding
   search, fractals. These serve as integration tests and showcase what's possible.

### What jax-js is NOT

- Not a model-serving runtime like ONNX Runtime — it's a **framework** for writing and training
  numerical programs, not just running pre-packaged inference.
- Not trying to match XLA's peak CUDA performance — the target is the browser and web runtimes.
- Not a Python replacement — it's for when you want numerical computing **where JavaScript runs**.

## Architecture overview

- **Core library** (`@jax-js/jax`, root `src/`): array API, autodiff (`grad`, `jvp`, `vjp`), JIT
  compilation, device placement.
  - Frontend modules in `src/frontend/`: `array.ts` (Array class), `jit.ts` (kernel fusion),
    `jvp.ts`/`linearize.ts` (forward/reverse AD), `vmap.ts` (vectorization), `convolution.ts`.
  - Library namespaces in `src/library/`: `numpy.ts`, `lax.ts`, `nn.ts`, `random.ts`,
    `scipy-special.ts`, `numpy-linalg.ts`, `numpy-fft.ts`.
- **Backends** (`src/backend/`): `cpu.ts` (debug only), `wasm.ts` + `wasm/`, `webgl.ts` + `webgl/`,
  `webgpu.ts` + `webgpu/` (ML compiler & shader codegen).
- **Aux packages**: `packages/loaders` (safetensors, OPFS cache, BPE tokenizers), `packages/onnx`
  (ONNX model → native ops), `packages/optax` (optimizers).
- **Website & demos**: `website/` — live examples that double as integration tests.

## Developer workflows

```bash
pnpm install                       # requires pnpm ≥ 10
pnpm run build                     # tsdown → dist/*.js, dist/*.d.ts
pnpm run build:watch               # watch mode
pnpm exec playwright install       # one-time: install Chromium for WebGPU tests
pnpm test                          # Vitest + Playwright (browser + node)
pnpm run test:all                  # Vitest + Deno WebGPU (auto-fallback)
pnpm run test:deno                 # Deno WebGPU tests only (headless GPU)
pnpm test test/conv.test.ts        # single file
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm -C website dev                # local dev server for demos
```

### Pre-commit CI checks

Husky runs `lint-staged` on commit, which auto-fixes ESLint and Prettier issues on staged files. The
pre-commit hook also runs the full Vitest suite and Deno WebGPU tests (`pnpm vitest run` +
`pnpm run build` + `pnpm run test:deno`).

> **TEMPORARY (`feat/non-consuming-ops` branch):** The pre-commit hook uses `|| true` so known test
> failures don't block commits. Search `.husky/pre-commit` for `TODO(merge-to-main)` to find the
> relaxed lines. See `MERGE_CHECKLIST.md` for the full list of items to resolve before merge.

**Before any commit**, also run these checks manually to catch issues early:

```bash
pnpm build                         # Build the project
pnpm check                         # TypeScript type checking
pnpm exec playwright install chromium-headless-shell  # (if not already installed)
pnpm test                          # Run all Vitest tests
pnpm run test:deno                 # Run Deno WebGPU tests
```

These match the checks in `.github/workflows/ci.yaml`.

### Rebase to main

When the user asks to "rebase to main", perform these steps:

1. `git fetch origin` — update remote refs
2. `git rebase origin/main` — rebase current branch onto latest main
3. If conflicts occur, resolve them (prefer keeping both sides' intent), then
   `git add <resolved files>` and `GIT_EDITOR=true git rebase --continue` (use `GIT_EDITOR=true` to
   avoid opening an interactive editor)
4. `pnpm vitest run` — verify all tests pass after rebase
5. `git push --force-with-lease` — update remote branch (safe force-push)

**Important:** Always use `GIT_EDITOR=true` when running `git rebase --continue` to prevent the
terminal from getting stuck in vim. Always use `--force-with-lease` (not `--force`) for the push.

### Temporary files

Use `tmp/` in the project root for temporary/scratch files. This directory is gitignored and allows
file operations without manual approval in VS Code.

### Debug logging

**IMPORTANT:** Do NOT use environment variables like `DEBUG=1`. Use the runtime function:

```typescript
import { setDebug } from "@jax-js/jax";
setDebug(1); // Enable debug logging BEFORE any jit compilation
```

| Level | Output                                           |
| ----- | ------------------------------------------------ |
| 0     | No debug output (default)                        |
| 1     | JIT compile logs, scan path selection            |
| 2     | Shader code (WGSL/WebAssembly), detailed tracing |
| 3     | Expressions and metadata                         |
| 4     | JIT programs, tuning details                     |
| 5     | Most verbose operation traces                    |

### WebGPU testing on headless servers

For GPU tests on a headless server, Chrome requires specific flags:

```bash
google-chrome --headless=new --use-angle=vulkan --enable-features=Vulkan \
  --disable-vulkan-surface --enable-unsafe-webgpu --disable-software-rasterizer
```

**Alternative: Deno for headless hardware WebGPU** — Deno's WebGPU (based on wgpu-rs) works headless
without X11:

```bash
curl -fsSL https://deno.land/install.sh | sh
deno eval --unstable-webgpu 'const a = await navigator.gpu.requestAdapter(); console.log(a?.info)'
pnpm run test:deno
```

### GPU device permissions (Linux)

```bash
sudo usermod -a -G render,video $USER  # then log out/in
```

## Memory management & ownership

Operations **do not consume** their inputs. Arrays stay alive until explicitly `.dispose()`'d.
`.data()` and `.dataSync()` read the buffer without consuming.

> **Note:** The upstream jax-js repository uses **move semantics** — every operation consumes its
> inputs (refcount −1), and reusing an array requires `.ref` (refcount +1). This fork replaces that
> with a non-consuming model: operations leave inputs alive, and `.ref` is never needed in user
> code. The change eliminates a major source of `UseAfterFreeError` bugs while keeping the same
> underlying Slot-based memory system. If you encounter `.ref` patterns in upstream code or git
> history, they are not needed here.

```ts
// Arrays can be reused freely — no .ref needed
function foo(x, y) {
  return x.add(x.mul(y)); // x used twice, no problem
}

// .data() reads without consuming
const result = await arr.data(); // arr is still alive
arr.dispose(); // explicit disposal when done
```

Inside `jit()` bodies, the compiler manages intermediate lifetimes automatically (freeing at exact
last-use). In eager mode, intermediates live until collected by GC or explicit `.dispose()`. Use
`jit()` for **performance** (kernel fusion, buffer recycling) — but code must be ownership-correct
in eager mode too. See [Ownership correctness principle](#ownership-correctness-principle).

**Why `.dispose()` is required:**

GPU buffers and WASM memory are finite resources that JavaScript's GC doesn't track. Without
explicit disposal, a training loop creating arrays each step would exhaust GPU memory in seconds.
`FinalizationRegistry` is too slow and unpredictable for real-time allocation patterns. The
pool/recycler needs deterministic buffer return to maintain peak-memory guarantees. The tradeoff:
`.dispose()` is one call per array at the end of its useful life. `using` declarations
(`Symbol.dispose`) also work — `using x = np.array(...)` will auto-dispose at block end.

Canonical examples: `test/refcount.test.ts`, `test/leak-diagnostic.test.ts`,
`test/deno/webgpu.test.ts`.

### `using` declarations (TC39 Explicit Resource Management)

jax-js supports the `using` keyword
([TC39 proposal](https://github.com/tc39/proposal-explicit-resource-management)) via
`[Symbol.dispose]()` on Arrays and jit functions. A polyfill in `src/polyfills.ts` ensures
`Symbol.dispose` exists in environments that don't have it yet.

**Arrays — auto-dispose at block end:**

```ts
{
  using x = np.array([1, 2, 3]);
  using y = np.array([4, 5, 6]);
  const z = x.add(y);
  console.log(z.js()); // [5, 7, 9]
  z.dispose();
  // x and y automatically disposed here
}
```

**JIT functions — auto-dispose captured constants:**

```ts
{
  using f = jit((x: np.Array) => x.mul(x).sum());
  const result = f(np.array([1, 2, 3]));
  console.log(result.js()); // 14
  result.dispose();
  // f's cached ClosedJaxpr consts are freed here
}
```

**Linearize/VJP results — auto-dispose captured forward-pass intermediates:**

```ts
{
  const [y, fLin] = linearize((x) => np.sin(x), [np.array([1.0])]);
  using _disposeLin = fLin; // dispose when done
  const dy = fLin(np.array([1.0]));
  dy.dispose();
  y.dispose();
  // fLin's forward-pass intermediates freed here
}
```

**When to use `using` vs `.dispose()`:**

| Pattern                              | Recommendation                                 |
| ------------------------------------ | ---------------------------------------------- |
| Short-lived arrays in a function     | `using` — cleaner, exception-safe              |
| JIT functions used across many calls | `.dispose()` when truly done                   |
| Loop bodies creating intermediates   | `using` or explicit `.dispose()` per iteration |
| Test cleanup                         | `using` or `onTestFinished(() => x.dispose())` |

**When NOT to use `using`:**

- For arrays you want to return from a function — `using` would dispose before return
- For arrays stored in data structures for later use

**`using` inside `jit()` bodies — recommended:**

`using` works correctly inside `jit()` bodies and **should be used** for uniformity with eager mode.
On tracers, `dispose()` is a harmless refcount decrement. On concrete arrays during PE tracing,
`[Symbol.dispose]()` is guarded by `_peArrayCreationTracker`. The library itself uses `using`
extensively inside functions that run under tracing (`lax.ts`, `random.ts`, `numpy-fft.ts`,
`lax-linalg.ts`). This supports the ownership-correctness principle: code should look identical
regardless of whether it runs in eager or JIT mode.

**Technical detail — `[Symbol.dispose]` on tracers:**

Currently, `Tracer` base class defines `[Symbol.dispose]() {}` (no-op). `JaxprTracer` inherits this
no-op, so `using` on a JaxprTracer doesn't actually decrement `#rc`. Only `JVPTracer` and `Array`
override it. This means `using` is safe (can't break anything) but doesn't enforce ownership during
JIT tracing — it's cosmetic. A future enforcement path would be to override `[Symbol.dispose]` on
`JaxprTracer` to call `this.dispose()` and add `#rc <= 0` checks in `getVar()` /
`processPrimitive()`, matching what `PartialEvalTracer` and `Array` already do. The library code
already uses `using` everywhere, so this change should be safe.

### Memory lifecycle

A **Slot** is jax-js's internal handle to a backend memory allocation (WASM pointer or GPU buffer).

1. **Array creation** — `np.array(...)` allocates a backend `Slot` with refcount = 1.
2. **`.ref` accessor** — increments the Array object's `#rc`; same underlying Slot.
3. **Operations** — do NOT consume inputs; inputs remain alive.
4. **`.data()` / `.dataSync()`** — reads buffer; array stays alive.
5. **`.dispose()`** — decrements `#rc`; when 0, calls `backend.decRef(slot)`.
6. **`evalJaxpr` / JIT** — automatically manage intermediate lifetimes from the Jaxpr graph.
7. **Pending ops** — `PendingExecute` (batched GPU commands awaiting submission) holds refs on Slots
   until `submit()`.

### Backend memory comparison

| Aspect          | Wasm (`src/backend/wasm.ts`)                | WebGPU (`src/backend/webgpu.ts`)                      |
| --------------- | ------------------------------------------- | ----------------------------------------------------- |
| Allocation      | `WasmAllocator` over `WebAssembly.Memory`   | `device.createBuffer()` with `GPUBufferUsage.STORAGE` |
| Slot tracking   | `Map<Slot, {ptr, size, ref}>`               | `Map<Slot, {buffer, size, ref}>`                      |
| Buffer copy     | `Uint8Array.copyWithin` (aligned/unaligned) | `copyBufferToBuffer` (aligned) or WGSL copy shader    |
| Sync read       | Direct memory view                          | `SyncReader` with staging buffer + `mapAsync`         |
| Dispatch        | Instantiate Wasm module, call exported fn   | `commandEncoder.dispatchWorkgroups()`, queue submit   |
| Zero on alloc   | **Yes** — `.fill(0)` on free-list reuse     | **Fresh only** — `createBuffer` zeros; pool does not  |
| Zero on recycle | N/A (JIT recycle = slot rename)             | N/A (JIT recycle = slot rename)                       |

### Ownership internals

This section describes how the non-consuming ownership model works across different execution
layers. Understanding this is essential for implementing new primitives, transforms, or debugging
ownership-related issues. See [Part 4](#part-4-ownership-friction-points-debugging--future-work) for
known edge cases and debugging strategies.

**Layer-by-layer ownership semantics:**

| Layer                         | Consumes inputs? | How disposal works                                          |
| ----------------------------- | ---------------- | ----------------------------------------------------------- |
| `EvalTrace.processPrimitive`  | **No**           | Delegates to impl rules, which create new arrays            |
| `JaxprTrace.processPrimitive` | **No**           | Builds graph from tracer identity, never disposes tracers   |
| `JVPTrace.processPrimitive`   | **No**           | Unpacks primals/tangents, calls JVP rule, wraps outputs     |
| `BatchTrace.processPrimitive` | **No**           | Delegates to batching rules                                 |
| `evalJaxpr`                   | **Auto-managed** | Counts usages from graph, disposes intermediates after last |
| `jitCompile` / `JitProgram`   | **Auto-managed** | Emits `malloc`/`free`/`recycle` steps from graph structure  |
| `.data()` / `.dataSync()`     | **No**           | Reads buffer, array stays alive                             |
| `.dispose()`                  | **Yes** (manual) | Decrements `#rc`; frees backend resources at `#rc === 0`    |

The key insight: **the JIT compiler and `evalJaxpr` derive lifetime information from the Jaxpr
graph** (which variables are used how many times) rather than from user-provided hints. This is both
correct and ergonomic.

**`evalJaxpr` auto-disposal (the `consumeRead` pattern):**

`evalJaxpr` (in `src/frontend/jaxpr.ts`) pre-computes a `usageCount` for every `Var` — how many
times it appears as an equation input or output. As each equation is evaluated, the count is
decremented; when it hits zero **and** the variable isn't a jaxpr output, the array is disposed:

```ts
const consumeRead = (x: Atom) => {
  if (x instanceof Var) {
    const left = remainingRefs.get(x)!;
    remainingRefs.set(x, left - 1);
    if (left === 1 && !outputVars.has(x)) {
      env.get(x)?.dispose();
      env.delete(x);
    }
  }
};
```

- **Intermediates** are disposed automatically at their last use.
- **Outputs** survive — the caller owns them.
- **Arguments** from the caller survive — `evalJaxpr` never frees what it didn't create.

**JIT compilation — same idea, compile-time:**

`jitCompile` performs identical lifetime analysis at compile time, emitting explicit `malloc`,
`free`, and `recycle` steps. `insertFreeSteps()` computes each slot's last-use point. Then
`recycleBuffers()` replaces adjacent `free→malloc` pairs of the same byte size with zero-cost
`recycle` steps (see [Part 3](#part-3-buffer-recycling--webgpu-buffer-pool)). At execution time,
`JitProgram.execute()` runs these steps mechanically — no ref-counting at all.

**JVPTracer refcounting:**

`JVPTracer` (in `src/frontend/jvp.ts`) has its own `#rc` field. It starts at 1; `.ref` increments,
`.dispose()` decrements. Only when `#rc` hits 0 do the primal and tangent get disposed. This
prevents premature disposal when JVP rules create intermediate tracers that are passed to multiple
consumers.

**PETracer cascade (`linearize.ts`):**

`PartialEvalTracer.dispose()` cascades to two types of held values when `#rc` reaches 0:

1. **Known pval values** — concrete arrays from `pval.isKnown`
2. **Const recipe values** — lifted constants from `"Const"` recipe type

NOT cascaded: `JaxprEqn.tracersIn` — handled by graph-wide toposort cleanup in
`partialEvalGraphToJaxpr()`, preventing double-free when equations share inputs. This is the most
delicate part of the ownership model — see
[PETracer cascade sensitivity](#petracer-cascade-sensitivity).

**Const ownership (`getOrMakeConstTracer`):**

When tracing captures a constant (e.g., `np.array([2])` inside a `jit` body), `getOrMakeConstTracer`
calls `val.ref` to give the `ClosedJaxpr` independent ownership:

- Constant's `#rc` goes from 1 (creation) to 2 (user + `ClosedJaxpr`).
- `ClosedJaxpr.dispose()` drops it to 1 (user's ref).
- For `jit()`, `JitProgram` owns captured constants and `jit.dispose()` frees them.
- For `lax.scan`, `closedJaxpr.dispose()` is called after scan execution.

**`partialEvalGraphToJaxpr` const protection:** Before graph-wide PETracer cleanup, constants are
protected with extra `.ref` to prevent the cascade from freeing constants needed by the returned
`ClosedJaxpr`.

**`evalJaxprTransposed` arg-primal protection:** In the backward pass of `grad`, argument primals
from the caller must not be freed. An `argPrimals` set tracks which primals came from arguments vs.
computed internally — only computed primals are disposed at cleanup.

## WebGPU Backend Architecture

This section explains WebGPU constraints relevant to jax-js development. Assumes familiarity with
GPU concepts (buffers, shaders, workgroups) but not WebGPU-specific details.

### WebGPU compute model primer

WebGPU exposes GPU compute via **compute shaders** written in WGSL (WebGPU Shading Language). Key
concepts:

- **Workgroup**: A group of threads that execute together and can share memory. Threads within a
  workgroup can synchronize via `workgroupBarrier()`.
- **Dispatch**: Launches a grid of workgroups. Each thread gets a unique `global_invocation_id`.
- **Storage buffers**: GPU memory readable/writable by shaders. Used for inputs and outputs.
- **Uniform buffers**: Small, read-only memory for constants. Faster than storage buffers.

**Critical limitation:** There is **no global barrier** in WebGPU. Threads in different workgroups
cannot synchronize within a single dispatch. This fundamentally shapes how jax-js implements
operations.

### Hard limits and how jax-js handles them

| Limit                              | Typical Value | Impact on jax-js                                   |
| ---------------------------------- | ------------- | -------------------------------------------------- |
| `maxStorageBuffersPerShaderStage`  | 8-10          | Limits kernel inputs; excess args trigger fallback |
| `maxComputeWorkgroupsPerDimension` | 65535         | Large arrays need 2D grid splitting                |
| `maxComputeWorkgroupSizeX`         | 256           | Limits threads per workgroup (Sort workgroup size) |
| `minUniformBufferOffsetAlignment`  | 256 bytes     | Dynamic uniform offsets must be 256-byte aligned   |
| `minStorageBufferOffsetAlignment`  | 256 bytes     | Can't use buffer offsets for arbitrary strides     |

**Storage buffer limit handling:**

```typescript
// src/backend/webgpu.ts
const maxArgs = limits.maxStorageBuffersPerShaderStage - 1; // Reserve 1 for output
if (kernel.nargs > maxArgs) {
  throw new Error(`Kernel has ${kernel.nargs} args, max is ${maxArgs}`);
}
```

When a fused kernel would exceed the buffer limit, compilation fails. The frontend must split into
smaller kernels (handled by `splitGraphDataflow()` in JIT).

**Grid size handling:**

When array size exceeds 65535 workgroups, `calculateGrid()` in `codegen.ts` splits into a 2D grid:

```typescript
// If size > 65535, split into 2D grid: (65535, ceil(size/65535))
const gridX = Math.min(size, 65535);
const gridY = Math.ceil(size / 65535);
```

Shader code uses `global_invocation_id.x + global_invocation_id.y * 65535u` to reconstruct the
linear index.

**Storage buffer offset alignment:**

The 256-byte `minStorageBufferOffsetAlignment` means you can't bind a buffer at arbitrary offsets.
For scan with strides like 48 bytes (a 4×3 f32 matrix), buffer offsets fail. jax-js solves this with
**uniform-based offsets**: bind the entire buffer, pass the element offset as a uniform variable.
See [WebGPU preencoded-routine details](#webgpu-preencoded-routine-details-routine-body) in Part 2.

### Features exploited

| Feature                     | How jax-js uses it                                               | Location                           |
| --------------------------- | ---------------------------------------------------------------- | ---------------------------------- |
| **shader-f16**              | Float16 dtype support; requested at device creation              | `src/backend.ts` feature requests  |
| **Workgroup shared memory** | Sort uses `var<workgroup>` for bitonic sort local exchanges      | `src/backend/webgpu/routines.ts`   |
| **workgroupBarrier()**      | Synchronizes threads within Sort workgroups                      | `bitonicSortShader` in routines.ts |
| **storageBarrier()**        | Memory fence for shared variable consistency                     | Sort, Cholesky, LU in routines.ts  |
| **Pipeline caching**        | Compiled pipelines stored by shader hash                         | `pipelineCache` in webgpu.ts       |
| **Command batching**        | Multiple dispatches encoded before single queue.submit()         | `PendingExecute` in webgpu.ts      |
| **WGSL copy shader**        | Byte-level buffer copy when `copyBufferToBuffer` alignment fails | `COPY_SHADER_CODE` in webgpu.ts    |

**Scan additionally uses:**

| Feature               | How scan uses it                                        | Location                                |
| --------------------- | ------------------------------------------------------- | --------------------------------------- |
| **Ping-pong buffers** | Carry state alternates between two buffers across iters | `dispatchPreencodedScan()` in webgpu.ts |
| **Uniform buffers**   | Per-iteration offsets for preencoded routine scan       | `scan-wrapper.ts`                       |

**Pipeline caching detail:**

```typescript
// Pipelines cached by shader source hash
const cacheKey = hashShaderSource(shaderCode);
if (pipelineCache.has(cacheKey)) {
  return pipelineCache.get(cacheKey);
}
const pipeline = device.createComputePipeline({ ... });
pipelineCache.set(cacheKey, pipeline);
```

This avoids recompiling identical shaders (common with JIT-generated kernels).

**Synchronous readback trick:**

WebGPU normally requires async `buffer.mapAsync()` for reading GPU data. jax-js implements
`SyncReader` (`src/backend/webgpu/reader.ts`) using an offscreen canvas with webgpu context —
borrowed from TensorFlow.js. This enables `.dataSync()` for debugging, though `.data()` (async) is
preferred for performance.

### Features NOT exploited (opportunities)

| Feature                   | What it enables                                     | Why not used yet                         |
| ------------------------- | --------------------------------------------------- | ---------------------------------------- |
| **Subgroups**             | SIMD-width operations (shuffle, reduce within wave) | Requires `subgroups` feature; not stable |
| **Indirect dispatch**     | GPU-driven workgroup counts                         | No dynamic control flow needs it yet     |
| **Texture sampling**      | Hardware-accelerated interpolation                  | All ops use storage buffers currently    |
| **Tiled matrix multiply** | Shared memory blocking for large matmuls            | Matmul uses simple row×col accumulation  |
| **Atomic operations**     | Lock-free reductions, histograms                    | Reductions done via shader accumulation  |
| **timestamp-query**       | GPU-side profiling                                  | Requested but not wired up for profiling |
| **Render pipelines**      | Visualization without readback                      | Would need separate rendering path       |

**Subgroups opportunity:**

Subgroups enable operations like `subgroupAdd()` that sum across a SIMD lane (typically 32-64
threads) without explicit barriers. This would accelerate reductions significantly:

```wgsl
// Current (sequential accumulation):
var acc = 0.0;
for (var i = 0u; i < size; i++) { acc += data[i]; }

// With subgroups (parallel within wave):
let partial = subgroupAdd(data[local_id]);
if (subgroup_invocation_id == 0) { atomicAdd(&result, partial); }
```

**Tiled matmul opportunity:**

Current matmul computes `C[i,j] = sum(A[i,:] * B[:,j])` with each thread doing a full dot product.
Tiled matmul loads tiles of A and B into shared memory, enabling data reuse:

```wgsl
// Tiled approach (not implemented):
var<workgroup> tileA: array<f32, 16*16>;
var<workgroup> tileB: array<f32, 16*16>;
// Load tiles collaboratively, compute partial products, accumulate
```

This is a standard GPU optimization that could provide 5-10× speedup for large matrices.

### WebGPU-specific scan constraints

The "no global barrier" limitation creates scan-specific constraints documented in Part 2:

| Constraint                        | Why it exists                                                | Consequence                   |
| --------------------------------- | ------------------------------------------------------------ | ----------------------------- |
| Per-element independence required | No cross-workgroup sync between iterations                   | Complex bodies → JS fallback  |
| numCarry ≠ numY unsupported       | compiled-loop shader assumes 1:1 carry↔output mapping       | Falls back to JS loop         |
| Internal buffer deps unsupported  | Shader can't allocate scratch temporaries between statements | Mandelbrot pattern → fallback |
| Sort in scan body                 | Sort already uses uniforms (conflict with scan offsets)      | Falls back to JS loop         |

WASM backend handles all these cases because it can allocate temporaries and has true sequential
control flow. WebGPU is more restricted but faster when patterns fit.

### Key WebGPU files

| File                                 | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `src/backend.ts`                     | WebGPU init, adapter/device creation, feature reqs |
| `src/backend/webgpu.ts`              | Main backend: kernels, scan, command encoding      |
| `src/backend/webgpu/codegen.ts`      | `calculateGrid()`, WGSL helpers, `ShaderInfo`      |
| `src/backend/webgpu/routines.ts`     | Bitonic sort, Cholesky, LU, TriangularSolve WGSL   |
| `src/backend/webgpu/scan-wrapper.ts` | Transforms routine shaders for scan with offsets   |
| `src/backend/webgpu/reader.ts`       | `SyncReader` for synchronous buffer readback       |
| `src/backend/webgpu/builtins.ts`     | Shader snippets for special functions (erf, etc.)  |

### Autodiff and ownership

> ⚠️ **Critical difference from Python JAX:** Letting `vjpFn` go out of scope will **NOT** free GPU
> memory. You **MUST** call `.dispose()` explicitly.

```ts
const [y, vjpFn] = vjp(f, [x]);
const dx = vjpFn(dy);
vjpFn.dispose(); // free captured forward-pass intermediates

const jitF = jit((x) => np.multiply(x, np.array([2])));
const result = jitF(x);
jitF.dispose(); // free captured constants
```

## Codegen architecture

Expression translation and shader generation share common code between regular kernels and scan.
Understanding this structure is essential for adding scan codegen paths.

**WASM Backend:**

| Function                        | Role                                                |
| ------------------------------- | --------------------------------------------------- |
| `translateExpCore()`            | Shared core handling all `AluOp` cases              |
| `TranslateExpContext` interface | Callbacks for `getVariable` and `handleGlobalIndex` |
| `translateExp()`                | Wrapper with bounds-check GlobalIndex               |
| `emitKernelBody()`              | Shared gidx loop + reduction + store for one kernel |
| `codegenWasm()`                 | Single entry point for kernel codegen               |

Scan adds `translateExpWithGeneralScanContext()` (const/carry/xs/internal classification) and
`codegenNativeScanGeneral()` (full scan loop codegen). Both `codegenWasm` and
`codegenNativeScanGeneral` call `emitKernelBody()` for the inner per-element loop, injecting
backend-specific behavior via callbacks for output addressing, expression translation, and store
logic.

**WebGPU Backend:**

| Function                    | Role                                                    |
| --------------------------- | ------------------------------------------------------- |
| `translateAluOpToWgsl()`    | Binary/unary ops, comparisons, casts, ternary           |
| `translateErfToWgsl()`      | Erf/Erfc with f32 precision wrapper                     |
| `gen()` in `pipelineSource` | CSE (common subexpression elimination) + special cases  |
| `createShaderEmitter()`     | Returns `{emit, pushIndent, popIndent, getCode}` helper |

Scan adds `genScanExpressionWithRidx` (scan-specific GlobalIndex + inline generation) and
`nativeScanMultiShaderSource()` (full scan shader).

## Routine system

Routines are backend-specific operations (sort, cholesky, etc.) that can't be expressed as fused
elementwise kernels. They exist in three implementations:

| Backend    | Implementation          | Location                         | Algorithm Style            |
| ---------- | ----------------------- | -------------------------------- | -------------------------- |
| **CPU**    | JavaScript (TypedArray) | `src/routine.ts`                 | Sequential (for debugging) |
| **WASM**   | wasmblr (runtime gen)   | `src/backend/wasm/routines/*.ts` | Sequential (optimized)     |
| **WebGPU** | Hand-written WGSL       | `src/backend/webgpu/routines.ts` | Parallel (GPU-optimized)   |

1. **CPU backend assumes WASM unavailable** — exists for environments without WebAssembly
2. **WebGPU uses different algorithms** — GPU parallelism requires fundamentally different
   approaches:
   - Sort: Bitonic sort (parallel) vs merge sort (sequential)
   - Cholesky: Column-parallel Cholesky-Crout vs row-by-row Cholesky-Banachiewicz

### wasmblr

**Problem:** Hand-writing WASM bytecode is error-prone and unmaintainable.

**Solution:** wasmblr — a custom WASM bytecode assembler with a high-level helper layer (WasmHl).

**Benefits:**

- Runtime JIT compilation (no separate build step, no pre-compiled binaries)
- Single TypeScript syntax throughout the codebase
- Ergonomic helpers for control flow (`forLoop`, `whileLoop`, `ifElse`) and memory access
- SIMD-ready (v128, i32x4, f32x4 types available)
- Small output (~1KB per routine)
- **Size specialization**: Matrix dimensions baked at compile time enable loop unrolling and
  constant propagation
- **LRU caching**: 64-entry cache amortizes compilation cost across calls

**Key WasmHl helpers:**

- `forLoop(i, start, end, body)` — for loop with expression start/end
- `forLoopDown(i, start, end, body)` — downward for loop
- `forLoopUnrolled(n, body, threshold?)` — fully unrolls small fixed-size loops (default ≤8)
- `whileLoop(cond, body)` — while loop with condition callback
- `ifElse(resultType, then, else?)` — conditional with optional else
- `load(dtype, base, indexExpr)` — load from base + index × elemSize
- `store(dtype, base, indexExpr, valueExpr)` — store to memory
- `index2D(row, cols, col)` — compute row × cols + col
- `binOp(dtype, op)` — binary operation (add, sub, mul, div)
- `const(dtype, value)` — push constant onto stack

**SIMD helpers (f32x4/f64x2):**

- `loadF32x4(base, indexExpr)` — load 4 floats as v128
- `storeF32x4(base, indexExpr, valueExpr)` — store v128 as 4 floats
- `f32x4Hsum()` — horizontal sum v128 → f32
- `f64x2Hsum()` — horizontal sum v128 → f64
- `simdReductionF32(acc, k, end, rowABase, rowBBase, op)` — SIMD dot product with automatic tail
  handling
- `simdReductionF64(acc, k, end, rowABase, rowBBase, op)` — same for f64x2

**SIMD speedup by matrix size:**

| Matrix Size | f32x4 Speedup | f64x2 Speedup |
| ----------- | ------------- | ------------- |
| n < 32      | ~0.8x (skip)  | ~0.9x (skip)  |
| n = 32      | ~1.1x         | ~1.0x         |
| n = 64      | ~1.7x         | ~1.3x         |
| n = 128     | ~3.0x         | ~1.8x         |
| n = 256     | ~3.8x         | ~1.9x         |

SIMD is automatically selected for Cholesky when `dtype === "f32" && n >= 32`.

**Calling routines from scan loops:** Scan modules use WASM imports to call routines from separate
wasmblr modules. This avoids code duplication (each routine is 1-3KB) while keeping the entire loop
in native code. See `codegenNativeScanGeneral()` in `src/backend/wasm.ts`.

### Autodiff of routines

Routines remain **opaque primitives** — the Jaxpr just contains `cholesky a`. The internal algorithm
is NOT traced.

The JVP rule defines the derivative **in terms of other primitives**:

```typescript
[Primitive.Cholesky]([a], [da]) {
  const L = cholesky(a);
  da = da.add(mT(da)).mul(0.5);  // Symmetrize
  const W = triangularSolve(L, da, { lower: true });
  const ST = triangularSolve(L, mT(W), { lower: true });
  const dL = batchMatmulT(L, triu(ST, 1).add(triu(ST)).mul(0.5));
  return [[L], [dL]];
}
```

The gradient is computed by:

1. **JVP tracing** → produces a Jaxpr containing `cholesky`, `triangular_solve`, matmul, etc.
2. **Transpose** → walks the JVP Jaxpr and transposes each primitive

The result (`grad(sum(cholesky))`) produces a **fully expanded Jaxpr** with ~30 operations. The
derivative of `cholesky` requires `triangular_solve` — both are Routines that dispatch to native
WASM.

### Adding a new routine (checklist)

| Step | File                                   | What to add                                                          |
| ---- | -------------------------------------- | -------------------------------------------------------------------- |
| 1    | `src/backend/wasm/routines/<name>.ts`  | Size-specialized wasmblr implementation (sizes as compile-time args) |
| 2    | `src/backend/wasm/routines/index.ts`   | Export the build function                                            |
| 3    | `src/backend/wasm/routine-provider.ts` | Add builder to `routineBuilders` map with size key generation        |
| 4    | `src/routine.ts`                       | Add to `Routines` enum                                               |
| 5    | `src/frontend/core.ts`                 | Add to `routinePrimitives` map                                       |
| 6    | `src/backend/wasm.ts`                  | Add dispatch case with size params                                   |
| opt  | `src/routine.ts`                       | Add CPU fallback in `runCpuRoutine()`                                |
| opt  | `src/frontend/jvp.ts`                  | Add JVP rule if autodiff needed                                      |
| opt  | `src/frontend/linearize.ts`            | Add transpose rule if grad needed                                    |

**Size key convention:** Cache keys include dtype and all size dimensions, e.g., `cholesky_f32_4` or
`triangular_solve_f64_8_16_lower_unit`.

### WASM feature opportunities (assessed Feb 2026)

| Priority | Feature            | Browser risk       | Impact      | Notes                                                                                |
| -------- | ------------------ | ------------------ | ----------- | ------------------------------------------------------------------------------------ |
| Medium   | i64 in wasmblr     | None (MVP)         | Medium-High | Unlocks proper f64 builtins (exp/log/sin/erf) and simplifies Threefry PRNG           |
| Medium   | Relaxed SIMD (FMA) | Safari unsupported | High        | `f32x4.relaxed_madd` for 2× dot-product throughput; needs runtime feature detection  |
| Low      | Threads / atomics  | Needs COOP/COEP    | Very High   | SharedArrayBuffer + Workers for parallel matmul/routines; major architectural change |
| Low      | Sign extension ops | None               | Low         | `i32.extend8_s` etc.; marginal for float-focused workloads                           |

## Deno WebGPU test guidelines

**Critical: Avoid creating multiple `GPUDevice` instances**

- **Always reuse jax-js's WebGPU device** instead of calling `navigator.gpu.requestAdapter()` +
  `adapter.requestDevice()`.
- Creating a second `GPUDevice` can destabilize Deno's WebGPU runtime and cause flakiness, memory
  leaks, or segfaults across test files.
- Use the `getJaxJsWebGPUDevice()` helper pattern to access the backend's device.
- **Never call `device.destroy()`** on the shared backend device — let the backend manage its
  lifecycle.

**Import from `dist/` not `src/`**

- Deno tests MUST import from `../../dist/index.js` to share backend instances across test modules.
- Mixed `src/` vs `dist/` imports create separate module graphs with separate backend instances,
  causing leak detection failures.

**Buffer cleanup**

- Track all created `GPUBuffer`s in an array: `const createdBuffers: GPUBuffer[] = []`.
- Destroy them in `finally` blocks: `for (const b of createdBuffers) b.destroy()`.
- Call `await device.queue.onSubmittedWorkDone()` before destroying buffers.

**Memory leak detection:**

```ts
import { withLeakCheck, getSlotCount, assertNoLeaks } from "./harness.ts";

Deno.test({
  name: "my test",
  fn: withLeakCheck(async () => {
    const result = await someComputation();
    await result.data();
    jitF.dispose();
  }),
});
```

## Exports & public API

All public symbols must be exported from `src/index.ts`. Key exports:

- Transformations: `jit`, `grad`, `valueAndGrad`, `jvp`, `vjp`, `vmap`, `jacfwd`, `jacrev`,
  `hessian`, `linearize`, `makeJaxpr`
- Device control: `init`, `defaultDevice`, `devicePut`, `blockUntilReady`, `devices`, `getBackend`
- Namespaces: `numpy`, `lax`, `nn`, `random`, `scipySpecial`, `tree`
- Testing utilities: `ScanPath` (type)

## Extending the codebase

| Area             | Key files                                          | Notes                                                          |
| ---------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| New numpy/lax op | `src/library/{numpy,lax}.ts`                       | Follow existing function signatures; add to exports if public. |
| Backend kernel   | `src/backend/webgpu/builtins.ts`, shader templates | Mirror existing patterns; test on Chromium via Playwright.     |
| Loader/tokenizer | `packages/loaders/src/`                            | See `safetensors.ts`, `tokenizers.ts`.                         |
| ONNX op          | `packages/onnx/src/ops.ts`                         | Implement lowering; wire in `index.ts`.                        |

## JIT compiler internals

The JIT system lives in `src/frontend/jit.ts` and `src/frontend/jaxpr.ts`.

**Pipeline:**

1. **Tracing** – `makeJaxpr(f)` traces a function to produce a `Jaxpr` (intermediate representation
   in A-Normal Form, where every subexpression is named)
2. **Simplification** – `jaxpr.flatten().simplify()` canonicalizes the graph
3. **Graph splitting** – `splitGraphDataflow()` identifies "black nodes" (operations that can't be
   fused, like reductions, routines, or `DynamicUpdateSlice`) vs fusable elementwise ops
4. **Kernel fusion** – Consecutive elementwise ops merge into a single `Kernel`
5. **Compilation** – `jitCompile(backend, jaxpr)` emits a `JitProgram` (list of `JitStep`s)
6. **Execution** – `JitProgram.execute(slots)` runs steps, managing memory lifetime

**Key types:**

| Type                              | File         | Role                                       |
| --------------------------------- | ------------ | ------------------------------------------ |
| `Jaxpr`, `JaxprEqn`, `Var`, `Lit` | `jaxpr.ts`   | IR nodes and bindings                      |
| `JitProgram`, `JitStep`           | `jit.ts`     | Compiled program + step types              |
| `Kernel`                          | `alu.ts`     | Fused single-output kernel                 |
| `Routine`                         | `routine.ts` | Backend-specific op (sort, cholesky, etc.) |

**JitStep types:**

| Type      | Purpose                                                                 |
| --------- | ----------------------------------------------------------------------- |
| `execute` | Dispatch a `Kernel` or `Routine` with inputs→outputs                    |
| `malloc`  | Allocate a buffer                                                       |
| `incref`  | Increment refcount on a slot                                            |
| `free`    | Decrement refcount on a slot                                            |
| `scan`    | Scan loop (fallback, compiled-loop, or preencoded-routine via ScanPlan) |

**Kernel class:**

The `Kernel` class is single-output: `new Kernel(nargs, size, exp, reduction?)`.

- `kernel.dtype` — dtype of the output
- `kernel.size` — number of elements
- `kernel.exp` — the ALU expression tree
- `kernel.reduction` — optional reduction operation

**Adding a new primitive:**

1. Declare in `Primitive` enum (`src/frontend/core.ts`)
2. Add tracing rule in `implRules` / `jvpRules` / `transposeRules`
3. If fusable elementwise, add ALU lowering in `jit.ts`
4. If needs dedicated kernel, register in `routinePrimitives` and implement in `src/backend/*`
5. If copy-like (e.g., `DynamicUpdateSlice`), it is handled specially in `jitCompile()` (throws; DUS
   only appears inside scan bodies) and is classified as a black node in `splitGraphDataflow()`

## Common pitfalls

- Forgetting `.dispose()` → memory leak (GPU buffers not freed). See
  [Debugging Ownership Issues](#debugging-ownership-issues) for strategies.
- Exporting a symbol from library but not `src/index.ts` → missing from published types
- Changing WebGPU shaders without browser tests → silent breakage
- **CPU backend GlobalView detection**: Collect both `AluOp.GlobalIndex` AND `AluOp.GlobalView`
  (internal ALU expression types) when finding used input buffers
- **JIT pending ops before scan**: Flush pending ops before scan step execution
- **Cross-device copy of non-contiguous arrays**: `_putSync()`/`_put()` must use
  `dataSync()`/`data()` (which call `#realize()`) instead of raw `readSync()`/`read()`. Raw reads
  return bytes in memory-layout order, ignoring the ShapeTracker — transpositions and reshapes are
  silently lost. This was fixed in commit `0419dce`; the trigger required all three conditions: (1)
  non-contiguous input (reshape/transpose/flatten), (2) static argnums on jit, and (3) consts
  created inside the jit body (placed on trace-time device, becoming first arg so `#computeBackend`
  picks CPU).

## Known flaky tests

- **Deno WebGPU tests** (`test/deno/`): When running all Deno test files together in a single
  `deno test` invocation, GPU state pollution between files causes memory leak detection failures.
  The `test:deno` script runs each file as a separate `deno test` command (chained with `&&`).

## Known framework bugs (`KNOWN_BUG` tests)

The `feat/non-consuming-ops` branch has known framework bugs tracked as regular `test()` calls
tagged with `KNOWN_BUG(<id>)` in the test name and a comment above. These tests exercise the actual
broken patterns (not workarounds) and are expected to fail. When a framework fix lands, the test
starts passing automatically — no clerical changes needed.

**Policy:**

1. Write the test as it SHOULD work — the ideal behavior, no workarounds.
2. Tag the test name: `test("KNOWN_BUG(my-tag): description", () => { ... })`
3. Add a `// KNOWN_BUG(my-tag): explanation` comment above the test.
4. Keep the working workaround test nearby (e.g., jit-wrapped version, depth-3 cap).
5. Add the bug to the inventory in `MERGE_CHECKLIST.md`.

**Finding all known bugs:**

```bash
grep -rn 'KNOWN_BUG(' test/
```

**Current inventory:**

All KNOWN_BUGs have been resolved. No remaining inventory.

**Test status:** See `pnpm vitest run` output. Known failures are expected and tracked above. The LU
JVP finite-difference test was previously failing because the WASM LU routine uses native f32
arithmetic; fixed by using larger eps and looser tolerance. All previously-failing cross-device
tests (FFT, random, linalg on WASM after CPU) are fixed — see `_put`/`_putSync` in
[Common pitfalls](#common-pitfalls).

> ⚠️ **IMPORTANT: Deno WebGPU test isolation** - Due to Deno's module caching and GPU state
> persistence between test files, running all Deno tests together in a single process causes
> spurious memory leak failures. The `test:deno` script chains separate `deno test` commands for
> each file to ensure proper isolation:
>
> ```bash
> pnpm run test:deno  # Runs each file separately (RECOMMENDED)
> ```
>
> Do NOT run `deno test test/deno/` directly - use the script instead. All test files use
> `withLeakCheck` from harness.ts for memory leak detection.

## Commit checklist

**Before every commit**, AI agents MUST:

1. Run pre-commit CI checks (see above)
2. Ensure the **pre-commit hook** is installed (run `pnpm prepare` if needed). The repository will
   run linting and the _full test suite_ automatically when you commit.
3. Run the _full test suite_ locally (`pnpm vitest run`) after finishing code changes. Verify no NEW
   failures beyond the known `KNOWN_BUG` tests (see
   [Known framework bugs](#known-framework-bugs-known_bug-tests)).
4. Update documentation when adding new features or APIs
5. Add/adjust tests exercising `.dispose()` for new behavior — add focused unit tests for any
   bugfixes or edge cases
6. Export new public symbols from `src/index.ts`
7. Update `FEATURES.md` for user-visible changes
8. If you **fix** a `KNOWN_BUG` test (it starts passing), celebrate — then remove the `KNOWN_BUG`
   tag and update the inventory in both this file and `MERGE_CHECKLIST.md`

## Documentation files

| File                              | Purpose                                    | When to update                 |
| --------------------------------- | ------------------------------------------ | ------------------------------ |
| `README.md`                       | Main project intro, tutorial               | Major features, API changes    |
| `FEATURES.md`                     | JAX/NumPy API compatibility table          | New supported functions        |
| `.github/copilot-instructions.md` | AI agent onboarding, scan feature tracking | New patterns, scan development |
| `MERGE_CHECKLIST.md`              | Branch merge tasks & KNOWN_BUG inventory   | New/fixed known bugs           |
| `packages/*/README.md`            | Package-specific docs                      | Package feature changes        |

## Where to start reading

- Entry & exports: `src/index.ts`
- Memory model: `test/refcount.test.ts`
- Backends: `src/backend/webgpu/`, `src/backend/wasm/`
- Demos: `website/src/routes/repl/`, `website/src/routes/mobileclip/`
- Deno WebGPU tests: `test/deno/webgpu.test.ts` — headless hardware GPU testing
- Scan tests: `test/lax-scan.test.ts` — comprehensive scan suite (~1880 lines)

---

# Part 2: Scan Feature Reference

This section documents the `lax.scan` implementation architecture, design choices, and
backend-specific behavior.

> **API Stability:** The scan feature is under active development. Breaking API changes may occur
> without deprecation warnings. No external users depend on this API yet.

## Overview & Motivation

`lax.scan` applies a function over the leading axis of arrays, threading carry state — essential for
RNNs, Kalman filters, cumulative operations, and other sequential computations.

**Signature:**

```ts
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs, options);
// f: (carry, x) => [newCarry, y]
```

**Options:**

- `length?: number` — Number of iterations (inferred from xs if not provided)
- `reverse?: boolean` — Process xs in reverse order (default: false)
- `acceptPath?: ScanPath | ScanPath[]` — Accept only these paths; throws if actual path not in list
- `checkpoint?: boolean | number` — Control gradient checkpointing for `grad(scan)`. Default
  (undefined/true) uses √N checkpointing. A number specifies the segment size. `false` stores all
  carries (O(N) memory).
- Fallback Y stacking: the JS fallback scan path preallocates the stacked Y output buffer and writes
  each iteration's Y directly via `copyBufferToBuffer` (4-byte aligned) or the WGSL copy shader
  (unaligned). This avoids stack overflow on long scans and reduces O(length) intermediate arrays
  from `coreConcatenate`. The fallback loop and stacking are centralized in shared helpers so jit
  and non-jit scans use identical behavior.

**Scan paths (`ScanPath` type):**

- `"compiled-loop"` — Entire scan loop compiled to native code (WASM module or WebGPU shader)
- `"preencoded-routine"` — Pre-encoded GPU command dispatches with uniform offsets per iteration
  (WebGPU only)
- `"fallback"` — JS loop calling body program per iteration (one or more JS↔backend boundary
  crossings)

Use `acceptPath: ["compiled-loop", "preencoded-routine"]` in tests to ensure native compilation
doesn't regress.

**xs=null and Y=null (jax-js extensions):**

Unlike Python JAX, jax-js supports null inputs and outputs for efficiency:

- **xs=null:** When xs is null, you must provide `length` option. Body receives null as x.
- **Y=null:** Body can return `[newCarry, null]` to skip output stacking entirely.

See [API Contract](#scan-reference-contract) for code examples and ownership details.

**Use cases:**

- Cumulative sum/product
- RNN/LSTM forward pass
- Kalman filter (forward and backward passes)
- Any sequential state machine

**Key files:**

| File                                 | Role                                                     |
| ------------------------------------ | -------------------------------------------------------- |
| `src/library/lax-scan.ts`            | Public API                                               |
| `src/frontend/core.ts`               | `Primitive.Scan` enum + params type                      |
| `src/frontend/jaxpr.ts`              | Abstract eval rule                                       |
| `src/frontend/array.ts`              | Eager `Primitive.Scan` impl                              |
| `src/frontend/jit.ts`                | Scan JitStep, `Primitive.Scan` case in `jitCompile()`    |
| `src/frontend/scan-plan.ts`          | `ScanPlan` type, `planScan()`, path selection heuristics |
| `src/frontend/scan-executor.ts`      | `executeScan()` — unified scan loop, `copySliceToBuffer` |
| `src/frontend/linearize.ts`          | JVP + transpose rules for autodiff                       |
| `src/frontend/vmap.ts`               | Scan vmap rule (batches independent scans)               |
| `src/backend/wasm.ts`                | Compiled-loop codegen + WASM routine dispatch            |
| `src/backend/webgpu.ts`              | Compiled-loop + preencoded-routine for routines          |
| `src/backend/webgpu/scan-wrapper.ts` | WGSL shader transformer for uniform offsets              |

---

## Feature Status by Backend

### CPU Backend

The CPU backend uses JavaScript-interpreted evaluation. It serves as the reference implementation
for correctness testing. All scan tests are in [test/lax-scan.test.ts](test/lax-scan.test.ts).

| Feature / Test             | Status  | Notes                                |
| -------------------------- | ------- | ------------------------------------ |
| `scan basic`               | ✅ Pass |                                      |
| `scan with pytree carry`   | ✅ Pass | pytree = nested dict/array structure |
| `reverse scan`             | ✅ Pass |                                      |
| `jit + scan`               | ✅ Pass |                                      |
| `JVP (forward-mode)`       | ✅ Pass |                                      |
| `VJP (reverse-mode)`       | ✅ Pass |                                      |
| `vmap`                     | ✅ Pass |                                      |
| `vmap` > `jit(vmap(scan))` | ✅ Pass |                                      |
| `scan over views`          | ✅ Pass | sliced/transposed xs                 |

### WASM Backend

The WASM backend supports **compiled-loop**: the entire scan loop is compiled into a WebAssembly
module, eliminating JS/WASM boundary overhead per iteration.

| Feature / Test                      | Status  | Notes                               |
| ----------------------------------- | ------- | ----------------------------------- |
| `scan basic`                        | ✅ Pass |                                     |
| `scan with pytree carry`            | ✅ Pass |                                     |
| `reverse scan`                      | ✅ Pass |                                     |
| `jit + scan`                        | ✅ Pass |                                     |
| `JVP (forward-mode)`                | ✅ Pass |                                     |
| `VJP (reverse-mode)`                | ✅ Pass |                                     |
| `vmap`                              | ✅ Pass |                                     |
| `vmap` > `jit(vmap(scan))`          | ✅ Pass |                                     |
| `scan over views`                   | ✅ Pass | sliced/transposed xs                |
| `compiled-loop`                     | ✅ Pass |                                     |
| `compiled-loop` > `with constants`  | ✅ Pass |                                     |
| `compiled-loop` > `reverse=true`    | ✅ Pass | all variants support reverse        |
| `scan with routine body`            | ✅ Pass |                                     |
| `routine in scan body`              | ✅ Pass | uses compiled-loop via WASM imports |
| `grad` through `scan` with routines | ✅ Pass | works via compiled-loop             |

**Performance benchmarks:**

- Fallback (JS loop): ~500 iter/sec
- Compiled-loop: ~50-80M iter/sec (small elementwise bodies, L=1000)
- Compiled-loop with direct-write: **40-65% faster** than without for small bodies

**Small scan throughput (L=1000 iterations, WASM compiled-loop):**

| Body Pattern               | Throughput    | Notes                                    |
| -------------------------- | ------------- | ---------------------------------------- |
| Cumsum (scalar)            | ~62M iter/sec | 1 carry, 1 Y, direct-write active        |
| Carry-only (4×4, Y=null)   | ~50M iter/sec | 1 carry, no Y output                     |
| Elementwise (n=4, Y=carry) | ~78M iter/sec | 1 carry, 1 Y, direct-write active        |
| Passthrough Y (4×4)        | ~35M iter/sec | Y = old carry, direct-write not eligible |

**Scan vs jit(loop) overhead:**

Compiled-loop is consistently faster than manual `jit(loop)` at all tested sizes due to eliminating
JS↔WASM boundary crossings per iteration:

| Matrix Size | Overhead | Notes                                |
| ----------- | -------- | ------------------------------------ |
| 16×16       | **-98%** | Scan FASTER (single WASM invocation) |
| 32×32       | **-98%** | Scan FASTER                          |
| 64×64       | **-96%** | Scan FASTER                          |
| 128×128     | **-84%** | Scan FASTER                          |

Compiled-loop compiles the entire loop into one WASM module, avoiding per-iteration overhead. Memory
copies within the loop use `memory.copy` (bulk memory) for efficient carry/output transfers.

### WebGPU Backend

The WebGPU backend keeps data on GPU between iterations. Supports **compiled-loop** for elementwise
kernels, **multi-kernel scan** for bodies with multiple independent kernels, and
**preencoded-routine** for single-routine bodies meeting specific requirements (currently Cholesky).

| Feature / Test                     | Status      | Notes                                        |
| ---------------------------------- | ----------- | -------------------------------------------- |
| `scan basic`                       | ✅ Pass     | uses compiled-loop on WebGPU                 |
| `scan with pytree carry`           | ✅ Pass     |                                              |
| `reverse scan`                     | ✅ Pass     | uses compiled-loop with dataIdx              |
| `jit + scan`                       | ✅ Pass     |                                              |
| `JVP (forward-mode)`               | ✅ Pass     |                                              |
| `VJP (reverse-mode)`               | ✅ Pass     |                                              |
| `vmap`                             | ✅ Pass     |                                              |
| `vmap` > `jit(vmap(scan))`         | ✅ Pass     |                                              |
| `scan over views`                  | ✅ Pass     | sliced/transposed xs                         |
| `compiled-loop`                    | ✅ Pass     | kernel gids reindexed to scan layout         |
| `compiled-loop` > `with reduction` | ✅ Pass     | e.g., `carry += sum(x)` or matmul            |
| `compiled-loop` > `with reverse`   | ✅ Pass     | uses dataIdx like WASM                       |
| `compiled-loop` > `with constants` | ✅ Pass     | captured constants bound as storage          |
| `multi-kernel scan`                | ✅ Pass     | derives output mapping from body outputs     |
| `preencoded-routine` (Cholesky)    | ✅ Pass     | requires passthrough pattern (numCarry=numY) |
| Mixed kernel+routine bodies        | ⚠️ Fallback | e.g., Kalman filter, lstsq                   |
| Multi-routine bodies               | ⚠️ Fallback | e.g., Cholesky→TriSolve→TriSolve             |
| Sort in scan body                  | ⚠️ Fallback | Sort already uses uniforms (conflict)        |

**Note on numCarry ≠ numY:** WebGPU compiled-loop requires `numCarry === numY`. When they differ,
WebGPU falls back to JS loop. WASM compiled-loop handles this case.

**Tested on:** NVIDIA RTX 4070 Ti SUPER via Deno WebGPU (headless, no X11)

### WebGL Backend

The WebGL backend has **no compiled-loop support**. All scans use the JS fallback path, which
executes the body program per iteration. This works correctly but lacks optimization.

| Feature / Test | Status      | Notes                                         |
| -------------- | ----------- | --------------------------------------------- |
| `scan basic`   | ⚠️ Untested | Uses fallback path; requires browser with GPU |
| `jit + scan`   | ⚠️ Untested | Uses fallback path                            |

**Note:** WebGL tests exist in `test/lax-scan.test.ts` but are **untested in CI** because:

- Deno doesn't provide WebGL (only WebGPU)
- Playwright's headless Chromium doesn't expose WebGL in the test environment
- The dev system lacks a display for headed browser testing

The fallback `executeScan()` path is backend-agnostic and tested with CPU/WASM/WebGPU, so WebGL
should work identically. To verify manually, run website demos in a WebGL-capable browser.

**WASM `copyBufferToBuffer` Support:**

The WASM backend implements `copyBufferToBuffer` using `Uint8Array.copyWithin` on the main WASM
memory buffer. This allows `scan-executor.ts` to stack `xs` slices and `ys` outputs during fallback
execution without allocating temporary TypedArrays, significantly reducing GC pressure.

> **When is copyBufferToBuffer used?**  
> WASM supports `compiled-loop` for almost all scan patterns, so the fallback path is rarely hit in
> production. However, `copyBufferToBuffer` is critical for:
>
> 1.  Debugging: When fallback scan is forced (e.g., via backend capability mocking).
> 2.  Reliability: Ensuring a working fallback exists for any future unsupported pattern.
> 3.  Completeness: Fulfilling the `Backend` interface contract.

---

## Design Choices & Rationales

### Why compiled-loop vs preencoded-routine vs fallback?

| Approach               | How it works                                  | When used                                        |
| ---------------------- | --------------------------------------------- | ------------------------------------------------ |
| **compiled-loop**      | Entire scan loop in native code (WASM/shader) | Elementwise kernels (WASM+WebGPU), WASM routines |
| **preencoded-routine** | Pre-encode dispatches with uniform offsets    | WebGPU single-routine bodies (e.g., Cholesky)    |
| **fallback**           | JS loop calling body program per iteration    | Unsupported patterns, mixed bodies, Sort         |

**Rationale:** compiled-loop is preferred because:

1. Eliminates JS↔native boundary per iteration (~5000× speedup for WASM)
2. Enables compiler optimizations across iterations
3. Single WASM module instantiation vs N calls

preencoded-routine is used for WebGPU routines that can't be inlined into a shader. It transforms
routine shaders to accept per-iteration offsets via uniforms, enabling fused dispatch.

### Why wasmblr for WASM routines?

**Problem:** Hand-writing WASM bytecode is error-prone and unmaintainable.

**Solution:** wasmblr — a custom WASM bytecode assembler with a high-level helper layer (WasmHl).

**Benefits:**

- Runtime JIT compilation (no separate build step, no pre-compiled binaries)
- Single TypeScript syntax throughout the codebase
- Ergonomic helpers for control flow (`forLoop`, `whileLoop`, `ifElse`) and memory access
- SIMD-ready (v128, i32x4, f32x4 types available)
- Small output (~1KB per routine)
- **Size specialization**: Matrix dimensions baked at compile time enable loop unrolling and
  constant propagation
- **LRU caching**: 64-entry cache amortizes compilation cost across calls

See the [Routine System](#routine-system) section for implementation details and wasmblr patterns.

### Why 3 routine implementations (CPU/WASM/WebGPU)?

| Backend    | Implementation          | Location                         | Algorithm Style            |
| ---------- | ----------------------- | -------------------------------- | -------------------------- |
| **CPU**    | JavaScript (TypedArray) | `src/routine.ts`                 | Sequential (for debugging) |
| **WASM**   | wasmblr (runtime gen)   | `src/backend/wasm/routines/*.ts` | Sequential (optimized)     |
| **WebGPU** | Hand-written WGSL       | `src/backend/webgpu/routines.ts` | Parallel (GPU-optimized)   |

1. **CPU backend assumes WASM unavailable** — exists for environments without WebAssembly
2. **WebGPU uses different algorithms** — GPU parallelism requires fundamentally different
   approaches:
   - Sort: Bitonic sort (parallel) vs merge sort (sequential)
   - Cholesky: Column-parallel Cholesky-Crout vs row-by-row Cholesky-Banachiewicz

**Calling routines from scan loops:** Scan modules use WASM imports to call routines from separate
wasmblr modules. This avoids code duplication (each routine is 1-3KB) while keeping the entire loop
in native code (~1.5M iter/sec). See `codegenNativeScanGeneral()` in `src/backend/wasm.ts`.

---

## API Contract

### Scan reference contract

This contract applies to both `lax.scan()` and `jit(() => lax.scan(...))()`:

**Inputs — NOT consumed:**

```ts
const [carry, ys] = lax.scan(f, init, xs);
// init and xs are NOT consumed (non-consuming model)
// Dispose them yourself when no longer needed
```

**xs=null for carry-only scans:**

```ts
// When xs is null, you must provide length option
const [carry, ys] = lax.scan(f, init, null, { length: 100 });
// No memory allocated for xs - useful for generators, RNG sequences, etc.
// Body receives null as second argument: f(carry, null) => [newCarry, y]
```

**Y=null to skip output stacking:**

```ts
// Return null as Y to avoid allocating stacked outputs
const [carry, nullYs] = lax.scan(f, init, xs);
// f: (carry, x) => [newCarry, null]
// nullYs is null, not an empty array - no memory allocated for outputs
// Useful when you only need the final carry (e.g., Mandelbrot iteration count)
```

**Body function — no .ref needed:**

Operations do not consume inputs, so arrays can be freely reused inside the body:

```ts
const step = (carry, x) => {
  const newCarry = np.add(carry, x);
  return [newCarry, carry]; // carry used in both places — no .ref needed
};
```

**Outputs — caller owns:**

```ts
const [finalCarry, stackedYs] = lax.scan(f, init, xs);
// Caller owns these — dispose when done:
finalCarry.dispose();
stackedYs.dispose(); // or skip if Y=null
```

**Common patterns:**

| Pattern      | Code                                  | Notes                           |
| ------------ | ------------------------------------- | ------------------------------- |
| Simple body  | `return [newCarry, y]`                | Two distinct arrays             |
| Passthrough  | `return [newCarry, newCarry]`         | Same array in both              |
| Pytree carry | `return [{ a: newA, b: newB }, newA]` | Nested structure                |
| Carry-only   | `scan(f, init, null, { length: N })`  | No xs allocation (saves memory) |
| No Y output  | `return [newCarry, null]`             | No ys allocation (saves memory) |

---

## Implementation Architecture

### Execution flow

```
lax.scan(f, init, xs, { reverse })
  → Trace f → bodyJaxpr (once)
  → Primitive.Scan(jaxpr, numCarry, numConsts, length, reverse)
  → planScan(backend, bodyProgram, bodyJaxpr, ...) → ScanPlan
      ├─ { path: "compiled-loop" }       ← WASM module or WebGPU multi-kernel shader
      ├─ { path: "preencoded-routine" }  ← WebGPU uniform-offset routine scan
      └─ { path: "fallback" }            ← JS loop calling bodyProgram.execute()
  → executeScan(backend, step)
      ├─ flush pending ops on all inputs (ONE policy)
      ├─ preallocate Y stacked buffers if direct-write eligible
      ├─ dispatch based on plan.path
      ├─ manage carry lifecycle, shared-slot guards, duplicate-slot incRef
      └─ return carry + stacked ys
```

The planner produces a `ScanPlan` data structure; the executor interprets it. This gives one
execution path for all backends, one ownership policy, and one flush discipline.

**Argument layout:**

```
Primitive args:   [...consts, ...initCarry, ...xs]
Body jaxpr input: [...consts, ...carry, ...x_slice]
```

### JIT and scan interaction

Understanding how JIT interacts with scan is crucial for performance:

**Without JIT wrapper:** `lax.scan(f, init, xs)`

```
1. Trace body function f → bodyJaxpr
2. JIT-compile bodyJaxpr → bodyProgram (via jitCompile in the impl rule)
3. planScan() → ScanPlan (determines execution path)
4. executeScan(plan, bodyProgram, ...) → dispatches based on plan.path
   - compiled-loop: runs entire loop in WASM/GPU code
   - preencoded-routine: pre-encoded GPU dispatches with uniform offsets
   - fallback: JS loop calling bodyProgram.execute() per iteration
```

**With JIT wrapper:** `jit(() => lax.scan(f, init, xs))()`

```
1. Trace outer function → outerJaxpr containing Primitive.Scan
2. JIT-compile outerJaxpr → outerProgram containing scan step
3. Trace body function f → bodyJaxpr (nested inside scan step compilation)
4. JIT-compile bodyJaxpr → bodyProgram
5. planScan() → ScanPlan (embedded in the scan JitStep)
6. Execute outerProgram:
   - executeScan() dispatches based on plan.path
```

**Key insight:** The body function is **always** JIT-compiled into a `bodyProgram`. Both JIT and
non-JIT paths use the same `planScan()` + `executeScan()` flow. The difference is whether the scan
step is embedded in a larger JitProgram (with JIT) or run directly from the eager impl rule (without
JIT).

**Why use `jit()` wrapper:**

- **Caches compilation** — `jit((xs) => scan(...))` compiles once, runs many times.
- **Captures constants** — Closed-over arrays become constants in the compiled program.
- **Note:** Both JIT and non-JIT paths can use compiled-loop/preencoded-routine. The eager impl rule
  calls `planScan()` directly, so `jit()` is not required for native scan execution — but it avoids
  re-tracing and re-planning on subsequent calls.

**When to use `jit()` wrapper:**

| Pattern                              | Use case                | Notes                                      |
| ------------------------------------ | ----------------------- | ------------------------------------------ |
| `lax.scan(f, init, xs)`              | Simple scans, debugging | Body still JIT-compiled, planScan() called |
| `jit(() => lax.scan(...))()`         | Performance-critical    | Cached compilation, avoids re-tracing      |
| `jit((xs) => lax.scan(f, init, xs))` | Reusable function       | Cached compilation, constants captured     |

**Transform compositions:**

| Composition       | Works? | Notes                                       |
| ----------------- | ------ | ------------------------------------------- |
| `jit(scan(...))`  | ✅     | JIT wraps scan, body is JIT-compiled        |
| `scan(jit(...))`  | ⚠️     | JIT inside body adds overhead per iteration |
| `grad(jit(scan))` | ❌     | Not supported — jit captures forward pass   |
| `jit(grad(scan))` | ✅     | Correct pattern for gradients               |
| `vmap(jit(scan))` | ✅     | Each batch element runs JIT-compiled scan   |
| `jit(vmap(scan))` | ✅     | Outer JIT, inner vmap, body compiled once   |

**Transform sandwiches:** Compositions like `jit(grad(scan))` where transforms wrap each other. The
test suite verifies these work correctly by comparing against reference implementations.

### How tracing works (non-consuming)

The Jaxpr SSA graph records exactly which variables are used and how many times.
`JaxprTrace.processPrimitive` adds equations to the graph using `builder.getVar(tracer)` — which
maps tracer identity to Var, regardless of refcount. Since `processPrimitive` never disposes
tracers, they can be used freely in multiple operations.

At execution time, `evalJaxpr` computes `usageCount` from the graph and auto-disposes intermediates
at their last use. `jitCompile` emits precise `malloc`/`free`/`recycle` steps based on the graph
structure. The result: **identical compiled programs** whether the user wrote `.ref` or not.

**Execution time (array.ts):** The non-JIT `Primitive.Scan` impl uses `jitCompile(backend, jaxpr)`
to compile the body, then calls `planScan()` + `executeScan()` — the same unified flow used by the
JIT path. Both paths use `executeScan()` which handles all three execution paths (compiled-loop,
preencoded-routine, fallback) with identical ownership semantics.

### Debugging scan paths

**Verify expected path with acceptPath:**

```ts
// Throws if actual path is not in accepted list
const [carry, ys] = await lax.scan(f, init, xs, {
  acceptPath: ["compiled-loop", "preencoded-routine"],
});

// Works identically in eager mode (no jit):
const [carry, ys] = lax.scan(f, init, xs, { acceptPath: "compiled-loop" });
// The primitive implementation forwards options to planScan() correctly.

// Accept only a specific path
await lax.scan(f, init, xs, { acceptPath: "compiled-loop" });

// Discover which path was chosen (always throws, shows path)
await lax.scan(f, init, xs, { acceptPath: [] });
// Error: Scan path debug: chose "compiled-loop"
// For WebGPU fallback, also shows dispatch count:
// Error: Scan path debug: chose "fallback" (2 GPU dispatches per iteration)
```

**Enable debug logging:**

```ts
import { setDebug } from "@jax-js/jax";
setDebug(1); // Shows scan path selection reason
setDebug(2); // Shows shader/WASM code
```

**Common fallback reasons:**

| Reason               | Debug message                                | Fix                           |
| -------------------- | -------------------------------------------- | ----------------------------- |
| Internal buffer deps | "internal buffer dependencies not supported" | Simplify body or use WASM     |
| Carry passthrough    | "carry is passthrough, not supported"        | Ensure kernel produces carry  |
| numCarry ≠ numY      | "numCarry !== numY"                          | Match carry/output counts     |
| Unsupported routine  | "unsupported routine in scan body"           | Use supported routine or WASM |

### JIT step type

Scan uses a single unified `"scan"` JitStep type with a `ScanPlan` discriminated union:

```typescript
type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | { path: "compiled-loop"; executable: Executable; params?: NativeScanGeneralParams }
  | { path: "preencoded-routine"; preencodedParams: PreparedPreencodedScan };
```

The `executeScan()` function in `scan-executor.ts` dispatches based on `plan.path`.

### Body composition types

Scan bodies are classified by what operations they contain:

| Body Type                | Description                           | Example                          |
| ------------------------ | ------------------------------------- | -------------------------------- |
| **kernel-only**          | Only elementwise/reduction kernels    | `carry + x`, `sum(x)`            |
| **routine body**         | Single routine operation              | `cholesky(x)`, `sort(x)`         |
| **mixed kernel+routine** | Both kernels and routines in one body | `scale * x` then `cholesky(...)` |

**Execution path by body type and backend:**

| Body Type                | WASM          | WebGPU                            |
| ------------------------ | ------------- | --------------------------------- |
| kernel-only (simple)     | compiled-loop | compiled-loop                     |
| kernel-only (with deps¹) | compiled-loop | **fallback**                      |
| routine body (single)    | compiled-loop | preencoded-routine (or fallback²) |
| mixed kernel+routine     | compiled-loop | **fallback** (common in practice) |
| multiple routines        | compiled-loop | **fallback** (e.g., lstsq)        |

¹ "With deps" = internal buffer dependencies between steps, or carry passthrough pattern. ² Sort
uses fallback due to uniform buffer conflict; LU uses fallback due to multi-output (numCarry ≠
numY).

**Why `lax.linalg.triangularSolve` creates a mixed body:**

The high-level `triangularSolve` API handles `leftSide` and `lower` parameters by adding
transpose/flip operations around the primitive routine:

```ts
// What lax.linalg.triangularSolve(L, b, { leftSide: true, lower: true }) compiles to:
b_transposed = moveaxis(b, -2, -1); // Kernel step 1
L_flipped = flip(L, [-2, -1]); // Kernel step 2 (for lower=true)
x = Primitive.TriangularSolve(L_flipped, b_transposed); // Routine step
result = flip(moveaxis(x, -2, -1), [-1]); // Kernel step 3
```

This creates a mixed kernel+routine body with 3+ steps, so WebGPU falls back. WASM compiled-loop
handles it. If performance is critical, consider using WASM backend for linalg-heavy scan bodies.

**Definition: Internal buffer dependencies**

When one kernel step reads from another step's output within the same body:

```ts
// Body with internal deps (WebGPU falls back):
const body = (carry, x) => {
  const Asq = carry.A.mul(carry.A); // Step 1: produces Asq
  const newA = Asq.sub(carry.B); // Step 2: reads Asq (internal dep!)
  return [{ A: newA, B: carry.B }, newA];
};
```

WASM handles this by allocating temporary buffers. WebGPU's shader codegen doesn't support it yet.

**Definition: Carry passthrough**

When an output carry slot directly references the input carry without a kernel producing it:

```ts
// Carry passthrough (WebGPU multi-kernel falls back):
const body = (carry, x) => {
  const newA = carry.A.add(x);
  return [{ A: newA, B: carry.B }, newA]; // B is passthrough!
};
```

WebGPU multi-kernel scan requires every carry output to be produced by a kernel step.

WASM's unified `codegenNativeScanGeneral` handles all body types via compiled-loop. WebGPU has more
constraints and falls back to JS loop for complex patterns.

### Terminology glossary

The documentation uses descriptive terms that map to code constructs:

| Doc Term               | Code Step Type       | Backend      | Description                                |
| ---------------------- | -------------------- | ------------ | ------------------------------------------ |
| **compiled-loop**      | `compiled-loop`      | WASM, WebGPU | Entire scan loop compiled to native code   |
| **preencoded-routine** | `preencoded-routine` | WebGPU       | Routine body with uniform offsets per iter |
| **fallback**           | `scan`               | All          | JS loop calling body program per iteration |

Note: `preencoded-routine` transforms routine shaders to use uniform-based offsets for xs buffers,
then dispatches all iterations with pre-encoded commands. Both `compiled-loop` and
`preencoded-routine` implement the "fast" scan path.

### Compiled-loop routing

The `tryPrepareNativeScan()` dispatcher routes to backend-specific implementations:

- **WebGPU kernel-only** → `tryPrepareWebGPUNativeScan()` → uses `prepareNativeScanMulti()`
- **WebGPU routine body** → `tryPreparePreencodedScan()` → uses `preparePreencodedScan()`
- **WASM (kernels + routines)** → `tryPrepareWasmNativeScan()` → uses `prepareNativeScanGeneral()`

**Dynamic Routine Planning:**

Instead of maintaining a hardcoded list of supported routines in `scan-plan.ts`, the planner queries
the backend capabilities via `getScanRoutineInfo(routineName, routine)`.

- If the backend returns `ScanRoutineInfo`, the routine is eligible for native compilation
  (`compiled-loop` or `preencoded-routine`).
- If it returns `null`, the planner falls back to the JS loop (`fallback`).
- This allows backends to implement routine support incrementally without modifying the frontend
  planner.

### Compiled-loop eligibility

**WASM compiled-loop** (via `tryPrepareWasmNativeScan`):

- All body steps are Kernels or supported Routines
- Constants allowed, reductions allowed
- Any `numCarry`/`numY` combination
- Y outputs can be: carry passthrough, xs passthrough, or internal buffer
- Internal buffer dependencies between steps: **supported**
- Supported routines: Dynamically queried via `getScanRoutineInfo` (currently Cholesky, Sort,
  TriangularSolve, LU, Argsort)

**WebGPU compiled-loop (single kernel)** (via `prepareNativeScanMulti` with 1 step):

- Exactly 1 Kernel step (single-output)
- `numCarry === 1` and `numY === 1`
- Constants supported, reverse supported

**WebGPU compiled-loop (multi-kernel)** (via `prepareNativeScanMulti`):

- Multiple Kernel steps (kernel-only body)
- `numCarry === numY` (or `numY === 0`)
- **No internal buffer dependencies** between steps (falls back otherwise)
- **No carry passthrough** (every carry must be produced by a kernel)

**WebGPU preencoded-routine** (via `tryPreparePreencodedScan`):

- Exactly 1 Routine step (single routine body)
- `numCarry === numY` (passthrough pattern)
- Routine must not already use uniforms (excludes Sort)

### WASM compiled-loop details

All WASM scan variants use `codegenNativeScanGeneral` in `src/backend/wasm.ts`:

1. **Pre-analysis** — Build `directWriteMap` deciding which internal buffers can be redirected
2. Import routine functions and helper math functions
3. Allocate WASM locals: iteration counter, data index, element indices
4. **Step 1**: Copy initCarry to carryOut (working buffer) via `memory.copy`
5. **Step 2**: Main scan loop (iter = 0..length):
   - Compute dataIdx (reverse-aware: `length - 1 - iter` or `iter`)
   - **Step 2a**: For each step, execute kernel or call imported routine
   - **Step 2b**: Copy Y outputs to `ysStacked` at iteration offset
   - **Step 2c**: Copy carry outputs to `carryOut` for next iteration
6. Return compiled `WebAssembly.Module`

**Direct-write optimization (pre-analysis phase):**

Before generating any WASM code, `codegenNativeScanGeneral` analyzes the scan body to build a
`directWriteMap: Map<internalIdx, { carryIdx, yIdx? }>`. This maps internal buffer indices to their
redirect targets. The analysis walks expression trees via `AluExp.fold()` to collect carry read
patterns per step.

When a kernel step is eligible for direct-write:

- **Step 2a**: The kernel's store instruction targets `carryOut[carryIdx]` instead of
  `internals[internalIdx]`. If `yIdx` is also set, uses `local.tee` to store the computed value to
  both `carryOut` and `ysStacked[yIdx]` in a single expression evaluation.
- **Step 2b**: The `memory.copy` for this Y output is skipped (already written inline).
- **Step 2c**: The `memory.copy` for this carry output is skipped (already written inline).

For multi-output kernels, each output is analyzed independently — some may use direct-write while
others fall back to internal buffers.

> Note: Multi-output kernel fusion (`Kernel.multi()`) is planned but not yet implemented. Currently
> each output is a separate single-output kernel step.

Eligibility conditions (all must be met):

1. Output produced by a Kernel step (not a Routine)
2. Kernel has no reduction (prevents self-overwrite during inner loop)
3. Internal buffer not read by any other step (no data dependencies)
4. Maps to exactly one carry output
5. No Y output is a passthrough from the target carry (passthrough reads OLD carry, but direct-write
   overwrites carry during the kernel loop)
6. No later step reads the target carry as input (later steps should see the carry from the START of
   the iteration, not the partially-overwritten value)

**Why condition 5 matters:**

```ts
// This body has Y = old carry (passthrough):
const step = (carry, x) => {
  const newC = carry.add(x);
  return [newC, carry]; // Y reads OLD carry value
};
```

If we direct-wrote `newC` to `carryOut` during the kernel loop (element by element), then the
passthrough copy `Y = carry` would read a mix of old and new carry values. The passthrough copy in
Step 2b reads from `carryOut`, and at element `i`, elements `0..i-1` would already be overwritten.
This is why direct-write is disabled when any Y output is a passthrough from the target carry.

**Why condition 6 matters:**

```ts
// Multi-step body where step 2 reads the carry that step 1 writes to:
const step = (carry, x) => {
  const a = carry.A.add(x); // Step 1: writes to carry.A
  const b = carry.A.mul(x); // Step 2: reads carry.A (needs OLD value!)
  return [{ A: a, B: b }, null];
};
```

If step 1 direct-wrote to `carryOut.A`, step 2 would read partially-overwritten values instead of
the carry entering the iteration. Direct-write is disabled for `carry.A` in this case.

**Performance impact:**

For small scan bodies (L=1000), eliminating `memory.copy` provides **40-65% speedup**:

| Pattern             | Without direct-write | With direct-write | Speedup |
| ------------------- | -------------------- | ----------------- | ------- |
| Cumsum (scalar)     | ~44M iter/sec        | ~62M iter/sec     | +41%    |
| Elementwise (n=4)   | ~48M iter/sec        | ~78M iter/sec     | +63%    |
| Carry-only (4×4)    | ~40M iter/sec        | ~50M iter/sec     | +25%    |
| Passthrough Y (4×4) | ~35M iter/sec        | ~35M iter/sec     | N/A     |

The elementwise case benefits most because it eliminates both internal→carry AND internal→Y copies.
Carry-only (Y=null) only eliminates the carry copy. Passthrough Y is ineligible (condition 5).

**Y output sources (`YOutputSource` type):**

| Type             | Source                              | Use case                         |
| ---------------- | ----------------------------------- | -------------------------------- |
| `passthrough`    | Copy from carry input               | `return [newC, oldC]`            |
| `xs-passthrough` | Copy from xs slice at current iter  | `return [newC, x]`               |
| `internal`       | Copy from internal buffer (compute) | `return [newC, someComputation]` |

**Carry output sources (`CarryOutputSource` type):**

| Type          | Source                    | Use case                             |
| ------------- | ------------------------- | ------------------------------------ |
| `passthrough` | Copy from carry input     | `return [oldC, y]` (carry unchanged) |
| `internal`    | Copy from internal buffer | `return [computation, y]`            |

### WebGPU compiled-loop details

Shader codegen in `nativeScanMultiShaderSource()`:

```wgsl
for (var iter: u32 = 0; iter < length; iter++) {
  var acc: f32 = 0.0;  // reduction identity
  for (var ridx: u32 = 0; ridx < reductionSize; ridx++) {
    acc = acc + /* expression using ridx */;
  }
  carry[gidx] = /* epilogue using acc */;
  ys[iter * carrySize + gidx] = carry[gidx];
}
```

Key insight: Thread `i` only reads/writes `carry[i]` and `xs[:,i]`. No `workgroupBarrier()` needed.

### WebGPU preencoded-routine details (routine body)

For routine bodies, the approach uses pre-encoded dispatches with uniform-based offsets.

**Why uniform-based (not buffer offsets):**

- `minStorageBufferOffsetAlignment` is 256 bytes on most GPUs
- Typical strides (e.g., 120 bytes) fail alignment requirements
- Solution: Bind entire buffers, pass offset as uniform variable

**Implementation:**

The `wrapRoutineForScan` function transforms routine shaders to add offset uniforms:

1. Parse buffer bindings from WGSL source
2. Identify which bindings need offsets using `ScanBindingInfo` mapping:
   - `routineInputJitIds` maps routine input bindings → body jaxpr JitIds
   - Inputs with JitId ≥ numConsts+numCarry are xs (need offsets)
   - Outputs are always carry (ys are filled via copy-after-iteration)
3. Generate `ScanOffsets` struct with offset fields for xs bindings only
4. Transform array accesses to add offset (e.g., `x[idx]` → `x[x_offset + idx]`)

**Dispatch architecture:**

- Ping-pong buffers for carry (iteration n reads from one, writes to other). These are **transient
  backend allocations** — created and destroyed within `dispatchPreencodedScan()`, not tracked by
  `computePoolHints` or the buffer pool (see _Peak-memory guarantee_ for rationale).
- Stacked ys buffers are filled by `copyBufferToBuffer` after each iteration
- Separate uniform bind groups per iteration (dynamic offsets not supported with auto layout)

### Critical implementation patterns

**Pending ops flush:** Scan execution requires flushing pending ops before scan step:

```ts
case "scan": {
  for (const p of pending) { p.prepareSync(); p.submit(); }
  pending.length = 0;
}
```

**IncRef for duplicate slots:** When body outputs contain duplicate slots (passthrough):

```ts
const seenSlots = new Set<Slot>();
const outArrays = bodyOuts.map((slot) => {
  if (seenSlots.has(slot)) backend.incRef(slot);
  else seenSlots.add(slot);
  return new Array({ source: slot, ... });
});
```

---

## Autodiff Support

### JVP (Forward-mode AD)

JVP tracing produces a doubled scan: primals + tangents flow together:

- Body becomes `(carryP, carryT, xP, xT) → (newCarryP, newCarryT, yP, yT)`
- Single scan executes both primal and tangent computation

### VJP/Grad (Reverse-mode AD)

Uses the JVP-transpose pattern for control flow:

```
grad(f)(xs)
  → vjp(f, [xs])
  → linearizeFlatUtil(f, primals)
  → partialEvalFlat (JVP'd scan with doubled args)
  → transpose(jvpResult, cotangents)
  → Scan transpose rule: iterate backward, transpose each step
```

**Key insights:**

1. Forward pass stores √N checkpoint carries by default (or all N if `checkpoint: false`)
2. Backward pass iterates from `length-1` to `0`, recomputing from checkpoints as needed
3. `evalJaxprTransposed` propagates "known" status for residuals

**Gradient checkpointing (`checkpoint` option):**

By default, the backward pass uses √N checkpointing: only O(√N) intermediate carries are stored, and
the rest are recomputed from the nearest checkpoint during the backward pass. This trades ~2×
compute for dramatically reduced memory. Set `checkpoint: false` to store all N carries (O(N)
memory, no recomputation).

```ts
// Default: √N checkpointing is automatic
const loss = (xs) => {
  const [carry, _] = lax.scan(step, init, xs);
  return carry.sum();
};
const dxs = grad(loss)(xs); // O(√N) memory

// Opt out: store all carries
const dxs2 = grad((xs) => {
  const [carry, _] = lax.scan(step, init, xs, { checkpoint: false });
  return carry.sum();
})(xs); // O(N) memory, no recomputation
```

Implementation (in `linearize.ts` transpose rule):

1. **Checkpoint forward pass**: Run forward, save carries every `segmentSize` iterations
2. **Segment-based backward**: For each segment (reverse order):
   - Recompute forward from the segment's checkpoint to recover all segment carries
   - Run transposed body backward through the segment
3. Helper functions `runOneForwardStep` and `runOneBackwardStep` eliminate code duplication

### Vmap (Vectorized Scan)

Each batch element runs an independent scan:

1. Move batch dims: consts/carry → axis 0, xs → axis 1
2. Create vmapped body jaxpr with batch at axis 0
3. Run single scan over batched arrays
4. Move ys batch from axis 1 to axis 0

**Compositions work:** `jit(vmap(scan))` and `vmap(jit(scan))`

**Transform sandwiches tested:** The `test/lax-scan.test.ts` "transform sandwiches" suite verifies
additional compositions: `jit(grad(scan))`, `grad(vmap(scan))`, `vmap(grad(scan))`, and `vmap(scan)`
vs `scan(vmap(body))` equivalence. Note: `grad(jit(scan))` is not supported — use `jit` inside the
grad-wrapped function instead.

---

## Routine System

### Implementation status

| Routine             | Status         | Source                                          | Notes                                          |
| ------------------- | -------------- | ----------------------------------------------- | ---------------------------------------------- |
| **Cholesky**        | ✅ Implemented | `src/backend/wasm/routines/cholesky.ts`         | f32/f64, single/batched                        |
| **TriangularSolve** | ✅ Implemented | `src/backend/wasm/routines/triangular-solve.ts` | Upper/lower triangular, unit/non-unit diagonal |
| **LU**              | ✅ Implemented | `src/backend/wasm/routines/lu.ts`               | Partial pivoting                               |
| **Sort**            | ✅ Implemented | `src/backend/wasm/routines/sort.ts`             | Bottom-up merge sort, NaN-aware                |
| **Argsort**         | ✅ Implemented | `src/backend/wasm/routines/argsort.ts`          | Stable merge sort on indices                   |

Routines are compiled at runtime using wasmblr — no separate build step required.

**Key files:**

| File                                   | Purpose                                |
| -------------------------------------- | -------------------------------------- |
| `src/backend/wasm/wasmblr.ts`          | Low-level WASM bytecode assembler      |
| `src/backend/wasm/wasmblr-hl.ts`       | High-level helper layer (WasmHl class) |
| `src/backend/wasm/routines/*.ts`       | Size-specialized routine codegen       |
| `src/backend/wasm/routine-provider.ts` | 64-entry LRU module cache (by size)    |

### Adding a new routine (checklist)

| Step | File                                   | What to add                                                                        |
| ---- | -------------------------------------- | ---------------------------------------------------------------------------------- |
| 1    | `src/backend/wasm/routines/<name>.ts`  | Size-specialized wasmblr implementation (sizes as compile-time args)               |
| 2    | `src/backend/wasm/routines/index.ts`   | Export the build function                                                          |
| 3    | `src/backend/wasm/routine-provider.ts` | Add builder to `routineBuilders` map with size key generation                      |
| 4    | `src/routine.ts`                       | Add to `Routines` enum                                                             |
| 5    | `src/frontend/core.ts`                 | Add to `routinePrimitives` map                                                     |
| 6    | `src/backend/wasm.ts`                  | Add dispatch case with size params                                                 |
| 7    | `src/frontend/scan-plan.ts`            | Add to `supportedRoutines` in `tryPrepareWasmNativeScan()`                         |
| 8    | `src/backend/wasm.ts`                  | Add codegen case in `codegenNativeScanGeneral()`, add `ScanRoutineInfo.sizeParams` |
| opt  | `src/routine.ts`                       | Add CPU fallback in `runCpuRoutine()`                                              |
| opt  | `src/frontend/jvp.ts`                  | Add JVP rule if autodiff needed                                                    |
| opt  | `src/frontend/linearize.ts`            | Add transpose rule if grad needed                                                  |

**Size key convention:** Cache keys include dtype and all size dimensions, e.g., `cholesky_f32_4` or
`triangular_solve_f64_8_16_lower_unit`.

### wasmblr routine patterns

Routines are **size-specialized**: matrix dimensions are compile-time constants, enabling loop
unrolling and constant propagation. The `routine-provider.ts` caches compiled modules by a size key
(dtype + dimensions).

```typescript
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

// Size-specialized: n is a compile-time constant, not a runtime parameter
function genRoutine(cg: CodeGenerator, hl: WasmHl, dtype: "f32" | "f64", n: number): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;

  // Function signature: (inPtr: i32, outPtr: i32) - no size params at runtime
  return cg.function([cg.i32, cg.i32], [], () => {
    const inPtr = 0; // Function param indices
    const outPtr = 1;

    const i = cg.local.declare(cg.i32);
    const val = cg.local.declare(ty);

    // n is a compile-time constant - WASM compiler can unroll/optimize
    hl.forLoop(i, 0, n, () => {
      hl.load(dtype, inPtr, hl.getExpr(i));
      cg.local.set(val);

      hl.store(dtype, outPtr, hl.getExpr(i), () => {
        cg.local.get(val);
        hl.const(dtype, 2);
        hl.binOp(dtype, "mul");
      });
    });
  });
}

// Builder function called by routine-provider.ts with specific sizes
export function buildRoutineModule(dtype: "f32" | "f64", n: number): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  cg.memory.import("env", "memory");

  const func = genRoutine(cg, hl, dtype, n);
  cg.export(func, "routine"); // Single export, no dtype suffix

  return cg.finish();
}
```

**Key WasmHl helpers:**

- `forLoop(i, start, end, body)` — for loop with expression start/end
- `forLoopDown(i, start, end, body)` — downward for loop
- `forLoopUnrolled(n, body, threshold?)` — fully unrolls small fixed-size loops (default ≤8
  iterations)
- `whileLoop(cond, body)` — while loop with condition callback
- `ifElse(resultType, then, else?)` — conditional with optional else
- `load(dtype, base, indexExpr)` — load from base + index \* elemSize
- `store(dtype, base, indexExpr, valueExpr)` — store to memory
- `index2D(row, cols, col)` — compute row \* cols + col
- `binOp(dtype, op)` — binary operation (add, sub, mul, div)
- `const(dtype, value)` — push constant onto stack

**SIMD helpers (f32x4/f64x2):**

- `loadF32x4(base, indexExpr)` — load 4 floats as v128
- `storeF32x4(base, indexExpr, valueExpr)` — store v128 as 4 floats
- `f32x4Hsum()` — horizontal sum v128 → f32
- `f64x2Hsum()` — horizontal sum v128 → f64
- `simdReductionF32(acc, k, end, rowABase, rowBBase, op)` — SIMD dot product with automatic tail
  handling
- `simdReductionF64(acc, k, end, rowABase, rowBBase, op)` — same for f64x2

The `simdReductionF32`/`F64` helpers take base pointers (locals containing byte addresses) and
handle both the SIMD main loop and scalar tail automatically. The `k` local is used as the loop
counter (element index, not byte offset).

**When to use SIMD:**

| Matrix Size | f32x4 Speedup | f64x2 Speedup |
| ----------- | ------------- | ------------- |
| n < 32      | ~0.8x (skip)  | ~0.9x (skip)  |
| n = 32      | ~1.1x         | ~1.0x         |
| n = 64      | ~1.7x         | ~1.3x         |
| n = 128     | ~3.0x         | ~1.8x         |
| n = 256     | ~3.8x         | ~1.9x         |

SIMD is automatically selected for Cholesky when `dtype === "f32" && n >= 32`.

### Autodiff of routines (example: Cholesky)

Routines remain **opaque primitives** — the Jaxpr just contains `cholesky a`. The internal algorithm
is NOT traced.

The JVP rule defines the derivative **in terms of other primitives**:

```typescript
[Primitive.Cholesky]([a], [da]) {
  const L = cholesky(a);
  da = da.add(mT(da)).mul(0.5);  // Symmetrize
  const W = triangularSolve(L, da, { lower: true });
  const ST = triangularSolve(L, mT(W), { lower: true });
  const dL = batchMatmulT(L, triu(ST, 1).add(triu(ST)).mul(0.5));
  return [[L], [dL]];
}
```

The gradient is computed by:

1. **JVP tracing** → produces a Jaxpr containing `cholesky`, `triangular_solve`, matmul, etc.
2. **Transpose** → walks the JVP Jaxpr and transposes each primitive

The result (`grad(sum(cholesky))`) produces a **fully expanded Jaxpr** with ~30 operations. The
derivative of `cholesky` requires `triangular_solve` — both are Routines that dispatch to native
WASM. The gradient computation expresses the math in terms of these primitives.

---

## Codegen Architecture

Expression translation and shader generation share common code between regular kernels and scan.

**WASM Backend:**

| Function                               | Role                                                |
| -------------------------------------- | --------------------------------------------------- |
| `translateExpCore()`                   | Shared core handling all `AluOp` cases              |
| `TranslateExpContext` interface        | Callbacks for `getVariable` and `handleGlobalIndex` |
| `translateExp()`                       | Wrapper with bounds-check GlobalIndex               |
| `emitKernelBody()`                     | Shared gidx loop + reduction + store for one kernel |
| `translateExpWithGeneralScanContext()` | Wrapper with const/carry/xs/internal classification |
| `codegenWasm()`                        | Single entry point for kernel codegen               |
| `codegenNativeScanGeneral()`           | Full scan loop codegen with direct-write analysis   |

**WebGPU Backend:**

| Function                        | Role                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| `translateAluOpToWgsl()`        | Binary/unary ops, comparisons, casts, ternary                     |
| `translateErfToWgsl()`          | Erf/Erfc with f32 precision wrapper                               |
| `gen()` in `pipelineSource`     | CSE (common subexpression elimination) + special cases            |
| `genScanExpressionWithRidx`     | Scan-specific GlobalIndex + inline generation                     |
| `createShaderEmitter()`         | Returns `{emit, pushIndent, popIndent, getCode}` helper           |
| `nativeScanMultiShaderSource()` | Full scan shader implementation (handles single and multi-kernel) |

**Backend Interface:**

The `Backend` interface provides `prepareKernel()` / `prepareKernelSync()` for compiling kernels to
backend-specific executables.

### Native Scan Codegen

Native scan on both WASM and WebGPU generates single-output kernel codegen per step. Multi-kernel
scan bodies (multiple independent single-output kernel steps) are handled by
`nativeScanMultiShaderSource()` on WebGPU and `codegenNativeScanGeneral()` on WASM.

### Shared kernel body: `emitKernelBody()`

The inner per-element loop (gidx iteration, reduction accumulator, store) is shared between
`codegenWasm()` and `codegenNativeScanGeneral()` via `emitKernelBody()`. Callers inject
backend-specific behavior through three callbacks:

| Callback         | `codegenWasm` provides             | `codegenNativeScanGeneral` provides                    |
| ---------------- | ---------------------------------- | ------------------------------------------------------ |
| `emitOutputAddr` | `local.get(outputArg) + gidx * bw` | Direct-write: `carryOut[c]`; else: `internal[idx]`     |
| `emitExp`        | `translateExp(exp, {gidx, ridx})`  | `translateExpWithGeneralScanContext(exp, scanCtx)`     |
| `emitStore`      | Simple typed store                 | Dual-store: primary + ysStacked (for direct-write + Y) |

The shared function handles: gidx loop structure, bounds check, reduction identity/accumulate/
epilogue via `codegenReductionAccumulate()`, gidx increment, and loop branching.

---

## Known Limitations & Future Work

### Current limitations

| Limitation                            | Workaround                           | Backend |
| ------------------------------------- | ------------------------------------ | ------- |
| `numCarry ≠ numY` on WebGPU           | Falls back to JS loop                | WebGPU  |
| WebGPU internal buffer deps in scan   | Falls back to JS loop                | WebGPU  |
| Mixed kernel+routine bodies on WebGPU | Falls back to JS loop                | WebGPU  |
| `grad(scan)` ~2× compute overhead     | Use `{ checkpoint: false }` for O(N) | All     |
| Sort in scan body on WebGPU           | Uses JS loop (uniforms)              | WebGPU  |
| Mixed-dtype carries on WebGPU         | Use WASM backend or same-dtype carry | WebGPU  |

**WebGPU preencoded-routine requirements:** WebGPU can only use `preencoded-routine` for scan bodies
that are:

1. **Exactly one routine** (no kernels before or after)
2. **numCarry === numY** (passthrough pattern)
3. **Routine doesn't use uniforms** (excludes Sort)

In practice, this means only simple Cholesky-passthrough patterns like:

```ts
const step = (carry, x) => {
  const L = lax.linalg.cholesky(x);
  return [L, L]; // L is both carry and y
};
```

**Why most linalg patterns fall back on WebGPU:**

- **TriangularSolve**: The `lax.linalg.triangularSolve` API handles `leftSide`/`lower` via transpose
  operations (kernels), so the body has multiple steps.
- **LU**: Returns `[lu, pivots, permutation]` — three outputs, so numCarry ≠ numY.
- **lstsq/solve**: Combines Cholesky + TriangularSolve + TriangularSolve — multiple routines.
- **Kalman filters**: Mix matmul (kernel) + routines in one body.

**This is a minor limitation** because:

1. WASM `compiled-loop` handles all these cases natively via imports
2. Complex linalg patterns (Kalman, Newton, etc.) fall back regardless
3. WebGPU fallback still keeps data on GPU — the overhead is command encoding, not data transfer

**Note on Sort in scan body:** Sort already uses a uniform buffer for its configuration, which
conflicts with the scan offset uniform.

**Mixed-dtype carry on WebGPU:** The `nativeScanMultiShaderSource()` shader generator uses
`steps[0].kernel.dtype` for all buffer bindings. If a scan body has mixed dtypes (e.g., f32 carry +
i32 counter), the shader would produce incorrect results. WASM compiled-loop handles mixed dtypes
correctly since each buffer is typed independently.

**Length-0 scans:** Supported. Returns `(init, empty_ys)` matching JAX behavior.

### Code quality notes

These are not bugs but areas where the implementation uses pragmatic shortcuts that future
contributors should be aware of:

- **JVP detection heuristic:** `linearize.ts` uses `numCarry % 2 === 0 && numY % 2 === 0` to detect
  JVP-transformed scans during partial evaluation and transposition. This works because JVP always
  doubles carries/outputs, and this code only runs during autodiff. However, it could theoretically
  misclassify a user scan with even counts. Consider adding an explicit `isJvpTransformed` flag to
  `ScanParams` if this causes issues.

- **`_yTreedef` side-channel:** In `lax-scan.ts`, the Y treedef is stashed on the `flatF` function
  object via `(flatF as any)._yTreedef = yTreedef`. This is invisible to TypeScript and could be
  replaced with a closure variable.

- **Fallback Y stacking:** The `executeScanFallback()` in `scan-executor.ts` handles Y stacking via
  `copySliceToBuffer`. Both the JIT-path and non-JIT `Primitive.Scan` impl use the same
  `executeScan()` flow with preallocated buffer slots; the loop writes each iteration's Y using
  `copyBufferToBuffer` (4-byte aligned) or the WGSL copy shader `COPY_SHADER_CODE` (unaligned).
  Shared-slot protection (`protectSharedSlots: true`) incRefs shared carry/Y backend slots before
  disposal. The preencoded scan path (`dispatchPreencodedScan`) also uses the WGSL copy shader for
  ys stacking when carry sizes are not 4-byte aligned.

### Future work

| Priority | Feature                        | Notes                                                                                                                            |
| -------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| High     | Multi-output kernel            | `KernelOutput`, `Kernel.multi()`, `codegenWasmMultiPath` — bodies producing multiple outputs can't fuse into one kernel dispatch |
| Medium   | Missing test categories        | ~30 additional tests: WASM routine scan, path-documentation, advanced vmap/grad compositions                                     |
| Medium   | Mixed-dtype WebGPU scan shader | Per-binding dtype in `nativeScanMultiShaderSource`                                                                               |
| Medium   | WebGL copy for scan stacking   | Enable direct-write stacked Ys on WebGL fallback                                                                                 |

### WASM feature opportunities (assessed Feb 2026)

| Priority | Feature            | Browser risk       | Impact      | Notes                                                                                |
| -------- | ------------------ | ------------------ | ----------- | ------------------------------------------------------------------------------------ |
| Medium   | i64 in wasmblr     | None (MVP)         | Medium-High | Unlocks proper f64 builtins (exp/log/sin/erf) and simplifies Threefry PRNG           |
| Medium   | Relaxed SIMD (FMA) | Safari unsupported | High        | `f32x4.relaxed_madd` for 2× dot-product throughput; needs runtime feature detection  |
| Low      | Threads / atomics  | Needs COOP/COEP    | Very High   | SharedArrayBuffer + Workers for parallel matmul/routines; major architectural change |
| Low      | Sign extension ops | None               | Low         | `i32.extend8_s` etc.; marginal for float-focused workloads                           |

---

## Test Coverage Summary

### Test files

| File                            | Purpose                                        |
| ------------------------------- | ---------------------------------------------- |
| `test/lax-scan.test.ts`         | Main scan test suite (~1880 lines)             |
| `test/scan-backends.test.ts`    | Backend coverage & `copyBufferToBuffer` checks |
| `test/scan-bench.test.ts`       | Scan benchmark tests                           |
| `test/deno/webgpu.test.ts`      | Headless WebGPU tests via Deno                 |
| `test/deno/pool-memory.test.ts` | Pool peak memory guarantee (Deno WebGPU)       |
| `test/deno/scan.bench.ts`       | Deno WebGPU scan benchmarks                    |

### Deno WebGPU test guidelines

**Critical: Avoid creating multiple `GPUDevice` instances**

- **Always reuse jax-js's WebGPU device** instead of calling `navigator.gpu.requestAdapter()` +
  `adapter.requestDevice()`.
- Creating a second `GPUDevice` can destabilize Deno's WebGPU runtime and cause flakiness, memory
  leaks, or segfaults across test files.
- Use the `getJaxJsWebGPUDevice()` helper pattern to access the backend's device.
- **Never call `device.destroy()`** on the shared backend device — let the backend manage its
  lifecycle.

**Import from `dist/` not `src/`**

- Deno tests MUST import from `../../dist/index.js` to share backend instances across test modules.
- Mixed `src/` vs `dist/` imports create separate module graphs with separate backend instances,
  causing leak detection failures.

**Buffer cleanup**

- Track all created `GPUBuffer`s in an array: `const createdBuffers: GPUBuffer[] = []`.
- Destroy them in `finally` blocks: `for (const b of createdBuffers) b.destroy()`.
- Call `await device.queue.onSubmittedWorkDone()` before destroying buffers to ensure GPU work is
  complete.

**Module parallelism**

- Deno test runner supports parallel module execution via `--parallel` flag or `DENO_JOBS`
  environment variable.
- The `test:deno` script chains separate `deno test` commands with `&&`, running each file in its
  own process for proper GPU state isolation.

### Memory leak detection (Deno)

```ts
import { withLeakCheck, getSlotCount, assertNoLeaks } from "./harness.ts";

Deno.test({
  name: "my test",
  fn: withLeakCheck(async () => {
    const result = await someComputation();
    await result.data();
    jitF.dispose();
  }),
});
```

### Scan path verification

See [Debugging scan paths](#debugging-scan-paths) for full `acceptPath` usage and examples. Always
use `acceptPath: ["compiled-loop", "preencoded-routine"]` in tests to ensure native compilation
doesn't silently regress to JS fallback.

### Test coverage by category

| Category                    | Backend | Path               | Purpose                               |
| --------------------------- | ------- | ------------------ | ------------------------------------- |
| `scan basic`                | CPU     | fallback           | Core correctness                      |
| `native scan paths`         | WASM    | compiled-loop      | Verify fusion works                   |
| `native scan > with consts` | WASM    | compiled-loop      | Constants in body                     |
| `routine body: matmul`      | WASM    | compiled-loop      | Routine bodies via WASM imports       |
| `backend coverage`          | All     | direct call        | Verify copyBufferToBuffer & devicePut |
| `Cholesky in body`          | WebGPU  | preencoded-routine | Preencoded-routine with routines      |
| `transform sandwiches`      | varies  | varies             | `jit(grad(scan))`, `vmap(grad(scan))` |

Note: Some test categories are not yet implemented (KNOWN LIMITATIONS sentinel tests, multi-kernel
scan tests, `vmap(jit(scan))` tests).

---

# Part 3: Buffer Recycling & WebGPU Buffer Pool

This section documents the JIT-level buffer recycling optimization and the WebGPU backend buffer
pool — two complementary mechanisms that reduce memory allocation overhead.

## Overview & Motivation

GPU buffer creation (`device.createBuffer()`) and destruction (`buffer.destroy()`) are expensive
WebGPU API calls, costing ~5–10 µs each. In JIT-compiled programs that allocate and free
intermediate buffers of the same size, this overhead adds up. Two complementary optimizations
address this:

1. **JIT buffer recycling** — a compiler pass that replaces `free(a) → malloc(b)` pairs of the same
   byte size with a single `recycle(a → b)` step, reusing the backend `Slot` with zero backend
   calls.
2. **WebGPU buffer pool** — a backend-level pool of recently-freed `GPUBuffer` objects indexed by
   padded byte size, avoiding `createBuffer`/`destroy` cycles for same-size allocations that the JIT
   recycler can't catch (e.g., eager mode, cross-invocation reuse).

**Key insight:** These work at different levels and are complementary:

- **Recycling** operates within a single JIT program execution — it eliminates allocation overhead
  for intermediates whose lifetimes don't overlap.
- **Pooling** operates across JIT invocations and in eager mode — it reuses buffers returned by
  `decRef` for future `malloc` calls.

### Performance impact (WebGPU, Deno wgpu-rs)

Measured on Intel Core Ultra 5 125H:

| Benchmark                     | Without | With    | Speedup  |
| ----------------------------- | ------- | ------- | -------- |
| jit chain x5 fused (4096)     | 10.5 µs | 1.7 µs  | **6.2×** |
| jit 2-output same-size (4096) | 17.0 µs | 2.1 µs  | **8.1×** |
| jit 3-output same-size (4096) | 23.6 µs | 2.7 µs  | **8.7×** |
| jit 2× matmul 32×32           | 17.9 µs | 2.6 µs  | **6.9×** |
| scan cumsum N=100 size=64     | 4.5 ms  | 77.6 µs | **58×**  |
| scan cumsum N=500 size=256    | 4.4 ms  | 88.1 µs | **50×**  |
| eager chain x5 (4096)         | 90.1 µs | 90.0 µs | ~1×      |

The buffer pool is the dominant win — `createBuffer`/`destroy` costs dominate JIT dispatch latency.
Multi-output programs benefit most (8–9×) because they have more `malloc`/`free` pairs. Scan gets a
massive 50–58× boost because the scan executor allocates carry and stacked-ys buffers each
invocation. Eager mode is unaffected because the ~80 µs `PendingExecute` dispatch overhead
dominates.

**WASM backend:** Already has a free-list allocator (`WasmAllocator`) that coalesces freed blocks,
so the pool provides less benefit there. The JIT recycling step still helps by avoiding the
allocator's search overhead entirely.

### Peak-memory guarantee

Both features preserve peak physical GPU memory:

| Feature                | Peak GPU memory                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **JIT recycling**      | Unchanged or lower — replaces a `free`→`malloc` pair with a zero-cost slot rename; no new buffers are created                     |
| **WebGPU buffer pool** | Unchanged — `configurePool()` evicts stale entries and caps retained bytes at the program's peak live bytes before each execution |

The pool guarantee works because the JIT compiler computes `peakBytes` and `mallocSizes` at compile
time (via `computePoolHints()`). Before each `JitProgram.execute()`, the backend:

1. **Evicts** pool entries whose sizes aren't in `mallocSizes` (removes cross-program pollution).
2. **Sets the byte budget** to `peakBytes` (pool can't retain more bytes than the program needs at
   peak).

During execution, pool hits drain the pool while adding to live — total stays flat. Pool misses
create new buffers but only up to peak. After execution, freed buffers return to pool within budget.
The result: pool + live ≤ peakBytes at all times.

**Transient backend allocations (not tracked by `computePoolHints`):**

Some backend dispatch methods create short-lived GPU buffers internally — notably the preencoded-
routine scan path (`dispatchPreencodedScan`) allocates ping-pong carry buffers and copy-shader
uniform buffers. These are **not** tracked by `computePoolHints` because:

1. They have no corresponding `malloc`/`free` JitSteps — the JIT compiler never sees them.
2. They are created and destroyed within a single synchronous dispatch call, so they never persist
   across JS turns.
3. They are never pooled — they're explicitly `destroy()`'d, not returned via `decRef`/`#poolPush`.

These transient buffers cause a brief spike in `#gpuAllocatedBytes` during dispatch, but the counter
returns to its prior level before the function returns. The pool budget bounds **retained** memory
(buffers sitting idle between JIT calls), so these ephemeral internals don't affect the guarantee.

In eager mode (no JitProgram), the pool uses a static fallback budget
(`MAX_POOL_BYTES_DEFAULT = 64 MB`) since there's no compile-time peak to derive.

---

## JIT Buffer Recycling

### JitStep type

A new `"recycle"` step type was added to the `JitStep` union:

```typescript
type JitStep =
  | { type: "malloc"; output: JitId; size: number }
  | { type: "free"; input: JitId }
  | { type: "recycle"; input: JitId; output: JitId };
// ... execute, incref, scan
```

**Semantics:** `recycle(a → b)` means "reuse the backend Slot currently mapped to JitId `a` for
JitId `b`". At execution time, this is a scope remapping with zero backend calls:

```typescript
case "recycle": {
  const slot = scope.get(step.input)!;
  scope.delete(step.input);
  scope.set(step.output, slot);
  break;
}
```

### Compiler pass: `recycleBuffers()`

The `recycleBuffers()` method on `JitProgramBuilder` runs after `insertFreeSteps()` in the JIT
compilation pipeline:

```
jitCompile(backend, jaxpr)
  → ... build steps (malloc, execute, ...) ...
  → builder.insertFreeSteps(outputIds)    // emit free after last usage
  → builder.recycleBuffers()              // replace free→malloc with recycle
  → new JitProgram(...)
```

**Algorithm:**

1. Build `mallocSizes: Map<JitId, number>` — the byte size of every malloc step.
2. Walk steps looking for `free(a)`. For each free, scan forward for the next `malloc(b)` where
   `size(a) === size(b)`.
3. If found, replace the `free` step with `recycle(a → b)` and remove the `malloc` step.
4. Only skip past non-interfering steps between the free and malloc (i.e., `incref` and other `free`
   steps). Stop scanning at `execute`, `scan`, or other step types.

**Safety invariants:**

1. The freed buffer's last consumer has already been scheduled (free comes after last use).
2. The malloc'd buffer hasn't been written yet (execute step comes after malloc).
3. Sizes match exactly — no memory waste and no peak-memory increase.

**What it catches:**

| Pattern                | Example                                       | Recycle fires?                                 |
| ---------------------- | --------------------------------------------- | ---------------------------------------------- |
| Elementwise chain      | `x.add(1).mul(2).sub(3)`                      | Yes — intermediate freed before next allocated |
| Multi-output same size | `[x.add(1), x.mul(2)]`                        | Yes — input freed, output allocated            |
| Different sizes        | `x.sum()` (scalar) after `x.mul(2)` (array)   | No — sizes differ                              |
| Cross-execute          | free after execute A, malloc before execute B | Yes, if adjacent                               |
| Separated by execute   | free, execute, malloc                         | No — execute step breaks the scan              |

**Debug logging:**

```typescript
setDebug(1); // Logs: "jit: recycled 2 buffer(s)"
```

### pprint support

The `recycle` step is displayed in JIT program dumps:

```
%5 = recycle %2
```

### Key file locations

| Location                        | Purpose                                  |
| ------------------------------- | ---------------------------------------- |
| `src/frontend/jit.ts` line ~63  | `recycle` JitStep type definition        |
| `src/frontend/jit.ts` line ~120 | pprint case                              |
| `src/frontend/jit.ts` line ~188 | Execution case in `JitProgram.execute()` |
| `src/frontend/jit.ts` line ~359 | `recycleBuffers()` method                |
| `src/frontend/jit.ts` line ~717 | Call site in `jitCompile()`              |

---

## WebGPU Buffer Pool

### Design

The pool is a `Map<number, GPUBuffer[]>` keyed by **padded byte size** (already rounded to 4-byte
multiples). When `decRef` drops a buffer's refcount to 0, instead of calling `buffer.destroy()` it
tries to push the buffer into the pool. When `malloc` needs a new buffer, it checks the pool first.

```typescript
class WebGPUBackend {
  #bufferPool = new Map<number, GPUBuffer[]>();
  static readonly MAX_POOL_PER_SIZE = 4; // max buffers per size class
  static readonly MAX_POOL_TOTAL = 64; // max total pooled buffers
}
```

### Pool operations

**`#poolPop(paddedSize)`** — returns a pooled buffer of the given size or `null`:

```typescript
#poolPop(paddedSize: number): GPUBuffer | null {
  const list = this.#bufferPool.get(paddedSize);
  if (list && list.length > 0) return list.pop()!;
  return null;
}
```

**`#poolPush(buffer)`** — returns a buffer to the pool; returns `false` if pool is full:

```typescript
#poolPush(buffer: GPUBuffer): boolean {
  const paddedSize = buffer.size;
  let list = this.#bufferPool.get(paddedSize);
  if (!list) { list = []; this.#bufferPool.set(paddedSize, list); }
  if (list.length >= MAX_POOL_PER_SIZE) return false;
  let total = 0;
  for (const l of this.#bufferPool.values()) total += l.length;
  if (total >= MAX_POOL_TOTAL) return false;
  list.push(buffer);
  return true;
}
```

### Integration points

**`malloc()`** — checks pool before creating:

```typescript
// With initial data:
const pooled = this.#poolPop(paddedSize);
if (pooled) {
  buffer = pooled;
  this.device.queue.writeBuffer(buffer, 0, initialData);
}

// Without initial data:
buffer = this.#poolPop(paddedSize) ?? this.#createBuffer(paddedSize);
```

**`decRef()`** — returns to pool instead of destroying:

```typescript
if (buffer.ref === 0) {
  this.buffers.delete(slot);
  if (buffer.buffer !== this.#reusableZsb) {
    if (!this.#poolPush(buffer.buffer)) {
      buffer.buffer.destroy(); // pool full, actually destroy
    }
  }
}
```

### Capacity limits

| Limit                    | Value       | Rationale                                                   |
| ------------------------ | ----------- | ----------------------------------------------------------- |
| `MAX_POOL_PER_SIZE`      | 4           | Typical JIT programs reuse ≤4 buffers of the same size      |
| `peakBytes` (dynamic)    | per-program | Set by `configurePool()` from JIT compile-time analysis     |
| `MAX_POOL_BYTES_DEFAULT` | 64 MB       | Fallback budget for eager mode (no JIT peak to derive from) |

When `#poolPush` would exceed the byte budget, it returns `false` and the buffer is destroyed.
Before each JIT execution, `configurePool()` evicts stale entries and tightens the budget.

### Memory accounting

Pooled buffers are **not** tracked in `this.buffers` (the slot map). They're held directly as
`GPUBuffer` objects in `#bufferPool`, with total bytes tracked in `#poolCurrentBytes`. This means:

- `slotCount()` does NOT include pooled buffers (correct for leak detection).
- Pooled buffers are effectively invisible to the rest of the system until reused.
- `configurePool()` evicts stale entries before each JIT execution, so the pool self-cleans.
- If the pool is dropped (e.g., backend destroyed), pooled buffers leak. This is acceptable because
  backend destruction only happens at process exit.

### WASM backend comparison

The WASM backend uses `WasmAllocator`, which manages a contiguous `WebAssembly.Memory` with a
free-list allocator that coalesces adjacent freed blocks. This provides similar reuse semantics
without an explicit pool. The JIT recycling step still benefits WASM by skipping the allocator's
free-list search entirely.

| Aspect                      | WebGPU Pool              | WASM Allocator                  |
| --------------------------- | ------------------------ | ------------------------------- |
| Data structure              | `Map<size, GPUBuffer[]>` | Free-list with coalescing       |
| Allocation cost (pool hit)  | Array pop (~10 ns)       | Free-list search (~50 ns)       |
| Allocation cost (pool miss) | `createBuffer` (~5 µs)   | Expand memory (~1 µs)           |
| Deallocation cost           | Array push (~10 ns)      | Free-list insert (~50 ns)       |
| Cross-size reuse            | No (exact size match)    | Yes (splitting/coalescing)      |
| Zero on reuse               | **No** — stale data      | **Yes** — `.fill(0)` on realloc |

### Memory zeroing guarantees

New allocations are always zeroed:

- **WASM**: Fresh pages from `WebAssembly.Memory` are zero (spec guarantee). The `WasmAllocator`
  **also zeroes on free-list reuse** via `new Uint8Array(buffer, ptr, size).fill(0)`, so every
  `malloc()` returns zeroed memory regardless of reuse.
- **WebGPU**: `device.createBuffer()` is zero-initialized per the WebGPU spec. **Pooled buffers are
  NOT zeroed** — `#poolPop()` returns stale data.

**Current safety:** All code paths that allocate without `initialData` subsequently fully overwrite
the buffer (kernel dispatches, routine outputs, `memory.copy` in scan). No caller relies on pooled
buffers being zero. The one implicit dependency is **CPU Cholesky** (`src/routine.ts`) which only
writes the lower triangle — safe because the CPU backend allocates through the zeroing WASM
allocator or fresh JS TypedArrays (always zero). WASM Cholesky routines explicitly zero the entire
output matrix before writing.

**Rule for new code:** Never assume a buffer allocated without `initialData` contains zeros on
WebGPU. Either fully write every output element, or explicitly zero the buffer first.

---

## Test Coverage

### Test file

[test/recycle.test.ts](test/recycle.test.ts) — 7 tests covering correctness and leak detection:

| Test                                                   | What it verifies                                |
| ------------------------------------------------------ | ----------------------------------------------- |
| `chain of same-size operations is correct`             | Basic recycling correctness (add→mul→sub chain) |
| `recycling preserves correctness with different sizes` | Mixed sizes (reduction changes size)            |
| `multi-step chain does not leak slots`                 | No slot leaks with 4-step chain                 |
| `works with grad through chained ops`                  | Recycling doesn't break autodiff                |
| `works correctly with scan`                            | Recycling doesn't break scan                    |
| `chained ops produce correct results on WASM`          | WASM backend correctness                        |
| `does not leak slots on WASM`                          | WASM backend leak detection                     |

### Pool peak memory test file

[test/deno/pool-memory.test.ts](test/deno/pool-memory.test.ts) — 5 Deno WebGPU tests verifying the
peak memory guarantee using `gpuAllocatedBytes()`:

| Test                                                | What it verifies                                          |
| --------------------------------------------------- | --------------------------------------------------------- |
| `repeated JIT calls stay within peak memory`        | Pool doesn't grow across repeated same-shape calls        |
| `multi-output JIT stays within peak memory`         | Recycling + pool stable with multi-output programs        |
| `shape-varying JIT calls don't accumulate stale`    | `configurePool` evicts stale entries between programs     |
| `scan cumsum stays within peak memory`              | Scan executor's alloc/free doesn't cause pool growth      |
| `gpuAllocatedBytes tracks creates and pool returns` | Memory accounting is consistent after alloc→dispose cycle |

**GPU memory tracking:**

The WebGPU backend exposes `gpuAllocatedBytes()` (total bytes: live + pooled) and `slotCount()`
(live slots only). These are WebGPU-specific methods accessed via `getBackend() as any` in Deno
tests. The `#gpuAllocatedBytes` counter is incremented in `#createBuffer` (for storage buffers) and
decremented at all `destroy()` call sites (pool eviction, `decRef`, preencoded-scan cleanup).

### Benchmark file

[test/deno/recycle.bench.ts](test/deno/recycle.bench.ts) — Deno WebGPU benchmarks:

```bash
pnpm build && deno bench --no-check --unstable-webgpu --allow-read --allow-env test/deno/recycle.bench.ts
```

Benchmarks three categories:

- **JIT group:** fused chains (baseline), multi-output (recycle), matmul, chain+reduce
- **Scan group:** cumsum N=100/500
- **Eager group:** chain operations (pool), alloc-free cycles (pool)

To A/B test, comment out `builder.recycleBuffers()` in `jit.ts` and pool usage in `webgpu.ts`,
rebuild, and compare.

---

## Known Limitations & Future Work

| Limitation                 | Description                                             | Possible fix                                              |
| -------------------------- | ------------------------------------------------------- | --------------------------------------------------------- |
| Exact-size matching only   | Recycling requires identical byte sizes                 | Allow size-class bucketing (e.g., round up to power-of-2) |
| No cross-execute recycling | The scan stops at execute/scan steps                    | Extend analysis with liveness intervals                   |
| Eager mode uses static cap | Without a JitProgram, pool uses a 64 MB fallback budget | Derive peak from eager op sequences if needed             |
| Eager mode unaffected      | Pool helps alloc/free but dispatch overhead dominates   | Needs kernel batching / deferred dispatch improvements    |

### Future opportunities

- **Liveness-interval analysis:** Instead of scanning for adjacent `free→malloc` pairs, build a full
  liveness interval map and assign slots via graph coloring. This would catch more recycling
  opportunities across execute steps.
- **Size-class bucketing:** Round buffer sizes to the nearest power-of-2 for pooling, trading ~2×
  memory waste for much higher pool hit rates.
- **Pool memory pressure:** Register a callback to evict pool entries when GPU memory is low (not
  currently exposed by WebGPU API).
- **Staging buffer pool:** The `read()` method creates and destroys staging buffers for every
  readback. Pooling these would help workloads with frequent `.data()` calls.

---

# Part 4: Ownership Friction Points, Debugging & Future Work

This section documents known edge cases in the ownership model, debugging strategies for ownership
bugs, and design decisions about eager-mode memory management.

## Known Friction Points

### Anonymous constants in scan bodies leak 1 slot

**Problem:** If you create an anonymous `np.array(...)` inside a scan body, it leaks:

```ts
// ⚠️ LEAKS: the np.array([2, 3]) has rc=1 (creation), getOrMakeConstTracer adds rc=2,
// closedJaxpr.dispose() drops to rc=1, but nobody holds a reference to call dispose() again
const step = (carry, x) => {
  return [np.add(carry, x), np.multiply(x, np.array([2, 3]))]; // anonymous const leaks!
};
lax.scan(step, init, xs);
```

**Workaround:** Extract anonymous constants to named variables and dispose them after the scan:

```ts
const factor = np.array([2, 3]);
const step = (carry, x) => {
  return [np.add(carry, x), np.multiply(x, factor)];
};
const [carry, ys] = lax.scan(step, init, xs);
carry.dispose();
ys.dispose();
factor.dispose(); // user explicitly frees the constant
```

**Root cause:** `getOrMakeConstTracer` does `val.ref` (rc → 2). `closedJaxpr.dispose()` drops it to
rc=1. For user-held consts (like `factor`), the user's dispose drops it to 0. For anonymous consts,
nobody ever calls the final dispose.

**Potential fix (not yet implemented):** Track anonymous constants separately in the scan path —
constants whose only reference is the `ClosedJaxpr` could be freed fully by `closedJaxpr.dispose()`
without the extra `.ref`. This requires distinguishing "user-held" from "anonymous" consts at trace
time, which is not straightforward.

### PETracer cascade sensitivity

The PETracer cascade in `linearize.ts` is the most delicate part of the ownership model. It cascades
to known values and Const recipe values but NOT to `JaxprEqn.tracersIn`. Getting this wrong causes
either double-free (cascade too aggressively) or leaks (cascade too conservatively).

**Unreachable Const PETracers (fixed):** When `hasAux` captures aux values that reference input
arrays (e.g., `x.mul(x)` in aux), `instantiateConst` creates Const PETracers that `.ref` the input.
If these PETracers are unreachable from `tracersOut` (because aux values aren't in the jaxpr
outputs), `partialEvalGraphToJaxpr` never processes them, leaving dangling `.ref` calls. Fixed by
tracking all Const PETracers in `PartialEvalTrace.allConstPETracers` and disposing unreachable ones
in `partialEvalFlat` after `partialEvalGraphToJaxpr` returns.

The current design was arrived at by testing against the full suite including `jvp(grad(sin))`,
`hessian`, all scan/grad compositions, and `vmap(grad(...))` patterns. Any future change to how
`PartialEvalTracer` creates recipes or tracks values should be tested against these cases.

### JVPTracer `#rc` subtlety

JVPTracers start at `#rc = 1`. Each `.ref` increments, each `.dispose()` decrements. JVP rules
create intermediate JVPTracers that are passed around — if a rule creates a tracer, passes it to
`bind()`, and also returns it, the refcount must be correct. The current JVP rules were all audited
for this, but new JVP rules (for new operations) need to follow the same patterns.

### `ClosedJaxpr.dispose()` timing

If `closedJaxpr.dispose()` is called too early (before execution finishes reading consts), it would
free constants prematurely. If called too late (or not at all), constants leak. Currently it's
called at two points in `lax-scan.ts` — immediately after scan execution and on the length-0 path.

For `jit()`, the `JitProgram` owns captured constants and `jit.dispose()` frees them. The eager
`Primitive.Scan` impl calls `closedJaxpr.dispose()` directly.

### User-disposed constants in grad bodies (fixed)

**Problem:** When user code inside a `grad` body disposes an array that was captured as a
`ClosedJaxpr` constant, the backward pass crashes with `UseAfterFreeError`:

```ts
// Crashed before the fix:
const f = (xs) => {
  const initVal = np.array([0.0]);
  const [finalCarry, ys] = lax.scan(step, initVal, xs);
  initVal.dispose(); // ← user disposes; rc drops to 1
  ys.dispose();
  return finalCarry.sum();
};
grad(f)(xs); // UseAfterFreeError in getOrComputePrimal
```

**Root cause:** `getOrMakeConstTracer` does `val.ref` (rc → 2), but user's `.dispose()` drops rc to

1. After `partialEvalGraphToJaxpr` (net zero), `disposePeIntermediates` takes the last ref (rc → 0),
   killing the ClosedJaxpr's ownership before the backward pass reads it.

**Fix:** In both `linearizeFlat` and `vjpFlat`, before calling `disposePeIntermediates`, protect
jaxpr consts whose `c.refCount <= 1`. Normal consts (rc ≥ 2) are unaffected; only user-disposed
consts (rc = 1) are protected from over-disposal.

## Debugging Ownership Issues

If a `UseAfterFreeError` or `ReferenceError` appears:

1. **Identify the array** — the error includes the array's shape/dtype.
2. **Check disposal timing** — is some code path calling `.dispose()` prematurely? Common in
   `evalJaxprTransposed` (check `argPrimals` set) or PETracer cascade.
3. **Check `getOrMakeConstTracer`** — is a constant being disposed before the jaxpr that uses it?
4. **Add `console.log(slotCount())` checkpoints** around the failing code to narrow down where slots
   are being freed.
5. **Test with CPU backend** — CPU is simplest and has the clearest error messages.

### Leak detection

Use `slotCount()` before and after operations:

```ts
const before = (getBackend() as any).slotCount();
// ... operations ...
const after = (getBackend() as any).slotCount();
expect(after - before).toBe(0); // no leaks
```

The `test/leak-diagnostic.test.ts` file has 9 tests covering scan leak patterns.

### Transform compositions to verify

These are the compositions most sensitive to ownership bugs:

| Composition         | What it tests                                              |
| ------------------- | ---------------------------------------------------------- |
| `grad(f)(x)`        | VJP + transpose — tests evalJaxprTransposed arg protection |
| `jvp(grad(f), ...)` | Nested JVP + PETracer — tests PETracer cascade correctness |
| `hessian(f)(x)`     | Double differentiation — stress-tests all ownership layers |
| `jit(grad(scan))`   | Scan body tracing + grad + JIT — tests const ownership     |
| `vmap(grad(scan))`  | All layers combined                                        |

## Ownership Correctness Principle

> **Write code that is ownership-correct in both eager and JIT mode.**

This is the central memory management principle in jax-js. `jit()` is a **pure performance
optimization** (kernel fusion, buffer recycling, dispatch batching). It must never change program
semantics — including memory ownership. If code leaks or double-disposes in eager mode, that is a
real bug, not something to paper over by wrapping in `jit()`.

**Why this matters:** During development you frequently switch between eager and JIT modes —
debugging, profiling, adding logging, testing individual operations. If your code only works under
`jit()` tracing (where the compiler manages lifetimes), you can't freely switch. By keeping code
ownership-correct at all times, `jit()` becomes something you add or remove purely for performance.

**What "ownership-correct" means in the non-consuming model:**

1. Every array you create must eventually be `.dispose()`'d (or auto-disposed via `using`).
2. Operations do NOT consume inputs — you're responsible for disposing them when done.
3. Intermediates in expression chains (e.g., `x.mul(y).add(z)`) create arrays that are only disposed
   by GC in eager mode. Keep chains short or use `using` / explicit disposal for large tensors.
4. Transform outputs (`grad`, `vjp`, `jit`) — the caller owns the results.
5. `vjpFn.dispose()` / `jitFn.dispose()` — free captured forward-pass intermediates / constants.

**JIT's role is performance, not correctness:**

| Aspect                    | Eager mode                         | JIT mode                              |
| ------------------------- | ---------------------------------- | ------------------------------------- |
| Intermediate lifetimes    | Live until `.dispose()` / GC       | Freed at exact last-use automatically |
| Peak memory (chains)      | O(all intermediates)               | O(max concurrent live)                |
| Buffer reuse              | Pool only (on dealloc→realloc)     | Compile-time recycling + pool         |
| Kernel fusion             | None (one dispatch per op)         | Fused into single kernels             |
| **Ownership correctness** | **Must be correct** (ground truth) | **Must also be correct**              |

The eager column shows worse _performance_ characteristics — that's expected and fine. The critical
row is the last one: ownership correctness must hold in _both_ modes. JIT should only make things
faster, never fix ownership bugs.

### Static analysis: ESLint plugin (upstream jax-js)

The community [`@hamk-uas/eslint-plugin-jax-js`](https://github.com/hamk-uas/eslint-plugin-jax-js)
ESLint plugin catches array ownership violations statically for the upstream (move-semantics)
jax-js. It embodies the same ownership-correctness principle — it warns inside `jit()` callbacks on
purpose, because ownership-correct code should work in both modes.

The upstream plugin's three rules (`no-use-after-consume`, `no-unnecessary-ref`, `require-consume`)
target move semantics and are not directly applicable to this non-consuming fork. However, the
**design philosophy is identical**: code that is correct in eager mode will be correct under
`jit()`.

A future jax-js-specific linter for the non-consuming model could enforce:

- Every created array is eventually disposed (the `require-consume` analog)
- No use after `.dispose()` (the `no-use-after-consume` analog)
- No unnecessary `.ref` calls (`.ref` is never needed in user code under non-consuming semantics)

This would catch leaks and use-after-free bugs at edit time, before they surface as
`UseAfterFreeError` at runtime or silent memory leaks.

**Current in-repo status (initial implementation):**

An in-repo plugin now exists at `packages/eslint-plugin` and is wired into root `eslint.config.ts`
under plugin name `jax-js`.

Current rollout in this repo:

- `jax-js/require-using`: `warn` on `src/**`, `packages/**`
- `jax-js/no-unnecessary-ref`: `warn` on `src/**`, `packages/**`
- `jax-js/no-use-after-dispose`: `warn` on `src/**`, `packages/**`
- `jax-js/no-array-chain`: `off`
- all `jax-js/*` rules: `off` on `test/**`

Implemented rules:

- `jax-js/require-using`
- `jax-js/no-use-after-dispose`
- `jax-js/no-unnecessary-ref`
- `jax-js/no-array-chain`

Current semantics in the in-repo implementation:

- `jax-js/require-using` treats local bindings as compliant when they are:
  - declared with `using`,
  - returned later in the same block,
  - explicitly disposed later via `<name>.dispose()`, or
  - persisted to longer-lived structures (e.g., property/index assignment, `set`/`push`/`add`).
- `jax-js/no-use-after-dispose` tracks identifiers by lexical variable identity (scope-aware),
  avoiding false positives from shadowed names.

This provides a baseline Layer 1 enforcement path that can evolve with jax-js in the same repo.

**Layer 1 ESLint rule contract (implementation spec):**

1. **`jax-js/require-using`** (default: error)
   - Enforce `using`-by-default for local array-valued bindings.
   - Trigger on `VariableDeclarator` with kind `const|let` where initializer is array-producing.
   - Array-producing includes:
     - `np.array(...)`, `array(...)`, `zeros(...)`, `ones(...)`, `full(...)`, etc.
     - Member-call chains on array-like receivers (e.g., `x.mul(...)`, `x.add(...)`,
       `x.reshape(...)`)
     - Known transform calls returning arrays (`grad(...)`, `vmap(...)`, `jit(...)` call results
       excluded when binding function)
   - Pass when one of:
     - declaration kind is `using`
     - value is directly returned (`const x = ...; return x;` in same block)
     - value is persisted (assigned to object field / array slot / map value / module-scope binding)
     - declaration has explicit opt-out comment: `// jax-js-lint: allow-non-using`
   - No autofix for general case (control-flow-sensitive); optional suggestion fix:
     - `const x = expr;` → `using x = expr;` when no reassignment and no crossing `await`/`yield`.

2. **`jax-js/no-use-after-dispose`** (default: error)
   - Track local identifiers; after `.dispose()` or end of `using` scope, further reads/writes are
     errors.
   - Trigger examples:
     - `x.dispose(); x.add(1);`
     - using-declared variable escaping its block via closure/reference.
   - Must be flow-sensitive within function scope; inter-procedural analysis not required.

3. **`jax-js/no-unnecessary-ref`** (default: error)
   - Flag `.ref` usage in user/library code unless explicitly justified.
   - Allow only behind opt-out comment `// jax-js-lint: allow-ref` for rare internals.

4. **`jax-js/no-array-chain`** (default: off, strict mode)
   - Optional performance/ownership strictness rule.
   - Trigger on chained array calls depth ≥ 2 in a single expression:
     - e.g., `x.mul(w).add(b).relu()`
   - Rationale: unnamed temporaries in eager mode cannot be `using`-managed.
   - Provide suggestion (not autofix): split into `using` bindings.

**Rule scope and non-goals:**

- Scope: function-local reasoning (fast, predictable, low false positives).
- Non-goal: proving disposal across module boundaries or through alias-heavy dataflow.
- Bias toward false-negative over false-positive for persisted/global lifetimes.

### Memory management ergonomics

### Layer 1 policy (preferred)

Use a **`using`-by-default** style for all non-global array bindings:

1. If a binding is local and short-lived, write `using name = ...`.
2. If an array is returned, do not mark it `using` in the producer scope.
3. If an array is stored for later (state object/cache/module-level const), use explicit
   `.dispose()` at the true end-of-life.

This keeps eager and `jit` code structurally identical and makes linting simple.

**Lintable exceptions to `using`-by-default:**

- Returned values
- Persisted values (object fields, arrays/maps, module-scope constants)
- Explicit local `.dispose()` management (manual lifetime cuts)
- Intentional expression chains where no binding exists (discouraged in strict mode)

For **short-lived computations**, `using` declarations provide the cleanest pattern:

```ts
{
  using x = np.array([1, 2, 3]);
  using y = np.array([4, 5, 6]);
  const z = x.add(y);
  console.log(await z.data()); // [5, 7, 9]
  z.dispose();
  // x, y auto-disposed at block end
}
```

For **expression chains with large tensors**, break them up to control peak memory:

```ts
// Eager-safe: explicit intermediate disposal
const a = x.mul(weights);
const b = a.add(bias);
a.dispose(); // free intermediate before next op
const result = nn.relu(b);
b.dispose();
// result is the only live intermediate

// Preferred strict-mode style (same in eager + jit):
using a2 = x.mul(weights);
using b2 = a2.add(bias);
const result2 = nn.relu(b2);
```

For **performance-critical hot paths**, wrap in `jit()` — you get kernel fusion and automatic
intermediate recycling, on top of the already-correct ownership:

```ts
// Same ownership semantics, but faster — using works inside jit too
const forward = jit((x) => {
  using a = x.mul(weights);
  using b = a.add(bias);
  return nn.relu(b);
});
const result = forward(input);
// forward.dispose() when the function is no longer needed
```

### Alternatives evaluated (and rejected)

| Approach          | Memory benefit     | Footgun risk | Recommendation                           |
| ----------------- | ------------------ | ------------ | ---------------------------------------- |
| `using` / manual  | Explicit, correct  | None         | **Primary pattern**                      |
| `jit()`           | Auto intermediates | None         | **Performance optimization**             |
| `tidy()` (TF.js)  | Scope cleanup      | None         | Skip — no buffer reuse, sync-only        |
| `pipe()`          | Chain cleanup      | None         | Skip — too narrow                        |
| `.donate()` ¹     | Zero-alloc reuse   | High (UAF)   | Defer — pool handles it                  |
| In-place / `out=` | Zero-alloc reuse   | Breaks model | **Never** — incompatible with tracing/AD |

¹ `.donate()` is a **hypothetical future API** (inspired by JAX's `jax.device_put(x, donate=True)`)
that does not exist in the codebase. The JIT compiler's internal `recycle` pass and the WebGPU
buffer pool already provide buffer reuse without a user-facing donation API.

### `checkLeaks` diagnostic (implemented)

A zero-overhead leak detection tool that enforces ownership correctness at test time. When active,
it snapshots backend slot counts across ALL devices and tracks Array creations with lazy Error
objects for stack traces. The `leaked` count uses backend `slotCount()` deltas, so it exactly
matches existing leak detection behavior. Used in the global test setup (`test/setup.ts`) to wrap
every test with automatic leak checking — **every single test in the suite must be leak-free**.

```ts
import { checkLeaks } from "@jax-js/jax";

checkLeaks.start(); // snapshot slot count + enable stack capture
// ... user code ...
const report = checkLeaks.stop(); // diff + report
// report.leaked: number of leaked backend slots
// report.details: ["float32[512,512] created at model.ts:42", ...]
// report.summary: human-readable message with diagnostics
```

**Options:**

- `checkLeaks.start({ trackRefs: true })` — also track `.ref` call sites; the report shows
  `↳ last .ref at <location>` for arrays with rc≥2 leaks.
- `checkLeaks.snapshot()` — return a snapshot of tracked arrays mid-session without stopping.

**Key files:**

| File                           | Purpose                                                           |
| ------------------------------ | ----------------------------------------------------------------- |
| `src/frontend/check-leaks.ts`  | Core module: `checkLeaks` object, registry, `LeakReport`          |
| `src/frontend/array.ts`        | Hooks in constructor (track) and dispose (untrack)                |
| `src/frontend/jit.ts`          | Registers `_clearJitCompileCache` via `_registerJitCacheDisposer` |
| `src/frontend/jaxpr.ts`        | Registers per-function jit disposers in `_jitFunctionDisposers`   |
| `test/check-leaks.test.ts`     | 9 dedicated tests                                                 |
| `test/setup.ts`                | Global beforeEach/afterEach leak checking for all tests           |
| `test/leak-diagnostic.test.ts` | 9 scan leak tests using `checkLeaks`                              |

**Architecture — JIT cache disposal:**

Module-level `jit()` functions (like `fmod = jit(...)` in numpy.ts, `fftUpdate = jit(...)` in
numpy-fft.ts) create cached `ClosedJaxpr` objects with const arrays. These must be freed between
tests. Two mechanisms handle this:

1. **`_registerJitCacheDisposer(fn)`** — called at module load by `jit.ts` to register the global
   `_clearJitCompileCache` function. Clears the `jitCompileCache` Map.
2. **`_jitFunctionDisposers`** — a `Set<() => void>` in `check-leaks.ts`. Each `jit()` call in
   `jaxpr.ts` registers `result.dispose` (which disposes ClosedJaxpr consts + clears cache).

Both are called by `_disposeAllJitCaches()` during `checkLeaks.stop()`. The import direction is
safe: `jit.ts → check-leaks.ts` and `jaxpr.ts → check-leaks.ts` (check-leaks only imports from
`../backend`, no cycles).

**Unreachable Const PETracer disposal:**

When `hasAux` is used with `vjp`/`linearize`, aux computations may call `instantiateConst` on input
arrays, creating Const PETracers. If the aux outputs aren't in the jaxpr graph (they're captured
separately), these Const PETracers are unreachable from `tracersOut` and never processed by
`partialEvalGraphToJaxpr`. The `.ref` from `instantiateConst` is never balanced.

Fixed by tracking all Const PETracers in `PartialEvalTrace.allConstPETracers` (via
`main.globalData`) and disposing any still-alive ones after `partialEvalGraphToJaxpr` returns.

**Design:** Uses `slotCount()` across ALL backends as ground truth. The tracking Map provides
diagnostic details (array description + creation location via lazy Error objects) but is not used
for the leak count. This avoids false positives from internal AluExp-backed arrays that have no
backend Slot.

**Limitations:**

- `checkLeaks` wraps every test via `test/setup.ts`. Tests with inner `checkLeaks.start()/stop()`
  calls must be removed (they conflict with the global wrapper). See autoref.test.ts,
  recycle.test.ts for examples of tests that had inner calls removed.
- Anonymous constants in scan/jit bodies (e.g., `np.array([2, 3])` inline) get rc=2 from
  `getOrMakeConstTracer`'s `.ref`. After `_disposeAllJitCaches`, rc drops to 1. Extract constants to
  named variables and dispose them manually.

**Why this matters:** Ownership correctness is enforced at test time via `checkLeaks` (every test
must be leak-free) and at development time via IDE diagnostics. Zero overhead in production. Keeps
the API surface JAX-compatible. A future ESLint plugin for the non-consuming model would catch leaks
and use-after-free statically at edit time, complementing the runtime `checkLeaks` diagnostic.

### Fluent chain transform (v1 contract)

To add syntax headroom for anonymous intermediates without move semantics, we may introduce an
**opt-in compile-time transform** (not runtime monkey-patching) for fluent method chains.

**Core guarantees:**

1. Preserve source-level chained syntax for users.
2. Keep named bindings non-consuming unless explicitly disposed.
3. Dispose unnamed intermediates deterministically in eager mode.
4. Never dispose earlier than the transformed expression's last use.
5. Keep JS evaluation order and exception behavior unchanged.

**Allowed ergonomic forms (v1):**

- Assignment RHS chains:
  - `const y = a.op1(...).op2(...).op3(...);`
- Return chains:
  - `return a.op1(...).op2(...);`
- Expression-statement chains where result is intentionally discarded.
- Callback-local return chains (e.g., `array.map(x => x.op1(...).op2(...))`) — transformed only
  inside the callback's own function body.

**Lowering strategy (conceptual):**

- Rewrite chain to explicit temporaries in-order.
- Insert disposal of each temporary after its final transformed consumer.
- Wrap transformed region in `try/finally` when needed so exceptions do not leak intermediates.

**Bailout conditions (do NOT transform in v1):**

- `await` / `yield` in the chain expression.
- Optional chaining or computed dynamic call targets with unclear side effects.
- Patterns where a chain subexpression escapes/aliases before completion.
- Cases where ordering cannot be preserved without duplicating side effects.
- Cross-callback / higher-order-function boundaries (never rewrite a chain so that temp/dispose
  logic spans multiple function bodies).

When a bailout triggers, keep original code and optionally emit a non-blocking diagnostic: "chain
not transformed; consider `using` temporaries".

**Non-goals (v1):**

- No runtime global chain tracking.
- No refcount simulation in the transform.
- No rewriting across statement boundaries.

### Decision summary

| Approach          | Eager memory | Buffer reuse | Footgun risk | Recommendation                         |
| ----------------- | ------------ | ------------ | ------------ | -------------------------------------- |
| `using` / manual  | Correct      | Pool only    | None         | **Primary pattern**                    |
| `jit()`           | Optimal      | Full         | None         | **Performance optimization**           |
| `checkLeaks`      | Diagnostic   | N/A          | None         | **Implemented — enforces correctness** |
| `tidy()`          | Cleanup only | None         | None         | Skip — no buffer reuse, sync-only      |
| `pipe()`          | 1 extra buf  | Pool only    | None         | Skip — too narrow                      |
| `.donate()` ¹     | Zero-alloc   | True reuse   | High         | Defer — pool handles it                |
| In-place / `out=` | Zero-alloc   | True reuse   | Breaks model | **Never** — incompatible               |

¹ `.donate()` does not exist — see [Alternatives evaluated](#alternatives-evaluated-and-rejected).

## Future Work

| Priority | Feature                               | Notes                                                                                                                                                                                             |
| -------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ~~High~~ | ~~`checkLeaks` diagnostic~~           | ✅ Implemented in `src/frontend/check-leaks.ts`                                                                                                                                                   |
| ~~High~~ | ~~unreachable Const PETracer leak~~   | ✅ Fixed via `allConstPETracers` tracking in PE trace                                                                                                                                             |
| ~~High~~ | ~~user-disposed const over-disposal~~ | ✅ Fixed via `refCount <= 1` protection in `linearizeFlat`/`vjpFlat`                                                                                                                              |
| Medium   | Anonymous constant leak fix           | Distinguish user-held vs anonymous consts in scan tracing                                                                                                                                         |
| Medium   | ESLint plugin for non-consuming model | Analog of `@hamk-uas/eslint-plugin-jax-js` — enforce `using`-by-default (with return/persist exceptions), `no-use-after-dispose`, `no-unnecessary-ref`, and optional `no-array-chain` strict mode |
| Medium   | `scatter_add` primitive               | Needed for general Gather transpose (duplicate indices, multi-axis). Currently only permutation gathers (sort/argsort path) are supported. Would enable `np.take` grad with repeated indices.     |
| ~~Low~~  | ~~`using` declaration examples~~      | ✅ Documented in copilot-instructions + README                                                                                                                                                    |
