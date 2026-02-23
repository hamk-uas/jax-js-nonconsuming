These notes help AI coding agents be immediately productive. The document has two parts:

1. **Repository Overview** — General jax-js knowledge for any development work
2. **Scan Feature Reference** — `lax.scan` implementation details and backend-specific behavior
3. **Buffer Recycling & WebGPU Buffer Pool** — JIT `recycle` step and pool architecture
4. **Ownership Friction Points, Debugging & Future Work** — edge cases, debugging strategies
5. **Associative Scan** — `lax.associativeScan` Kogge-Stone parallel prefix scan

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
- **Silent leaks over noisy crashes** — move semantics crash immediately (`UseAfterFreeError`) when
  you forget `.ref`, pointing at the bug. The non-consuming model never crashes from reuse, but a
  forgotten `.dispose()` leaks GPU memory silently. `checkLeaks` and the ESLint plugin compensate,
  but they're opt-in. Method chains (`a.mul(b).add(c)`) are particularly dangerous because every
  intermediate allocates a GPU buffer nobody frees in eager mode. The bet is that silent leaks +
  tooling is more manageable than `UseAfterFreeError` + `.ref` boilerplate, especially for teams
  from Python/MATLAB — but it's a genuine tradeoff.
- **Ownership-correct in both modes** — code must dispose correctly in eager mode **and** under
  `jit()` tracing. `jit()` is a pure performance optimization (kernel fusion, buffer recycling); it
  must not change ownership semantics. If code leaks or double-disposes in eager mode, that is a
  real bug to fix — not something to paper over by wrapping in `jit()`. See
  [Ownership correctness principle](#ownership-correctness-principle).
- **Compounding returns** — every improvement to the compiler makes _all_ operations faster, every
  new primitive gets autodiff for free, every `jit`-wrapped function gets kernel fusion
  automatically. Prioritize work that compounds.

### Ownership invariants (maintainer rubric)

For transform and tracing internals, use this short review rubric before merging:

1. **Conserve handles** — every handle created/received has exactly one terminal path: transfer,
   dispose, or explicit retained ownership.
2. **Make retention boundaries explicit** — if a value outlives scope (captured consts, caches,
   pending work, closures), retain an independent handle intentionally.
3. **Release retained handles symmetrically** — retained handles must have a single release path,
   including error paths (`try/finally` parity).

This rubric is orthogonal to the non-consuming API model. It applies to internal ownership plumbing
where `.ref` still exists as a low-level mechanism.

### Development priorities

When deciding what to work on, prefer work in this order:

1. **Correctness first** — tests, reference-counting discipline, cross-backend consistency.
2. **API breadth** — approximate NumPy/JAX API compatibility (see `FEATURES.md` for the status
   table). The goal is that common JAX/NumPy patterns can be ported with minimal changes.
3. **Performance** — there is significant headroom, especially in:
   - WASM backend (SIMD is used for Cholesky f32; extending to matmul/elementwise kernels. M5 added
     `WasmWorkerPool` for parallel kernel dispatch via `SharedArrayBuffer` + Web Workers — active
     when `SharedArrayBuffer` is constructable, dispatches kernels with ≥ 4096 elements across
     cores. M6.2b added `OrchestratorWorker` for off-main-thread mega-module execution).
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
  - Scan operations in `src/library/`: `lax-scan.ts` (`lax.scan`, sequential carry threading),
    `lax-associative-scan.ts` (`lax.associativeScan`, Kogge-Stone parallel prefix scan).
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
pnpm run test:policy:strict        # strict test policy: zero failures, zero expected-failure debt
pnpm run test:arch                 # architectural mode: failures must match .ci/expected-failures.json
pnpm run test:arch:record          # record current failing tests into expected-failure manifest
pnpm run test:all                  # Vitest + Deno WebGPU (auto-fallback)
pnpm run test:deno                 # Deno WebGPU tests only (headless GPU)
pnpm run lint:ownership:internal   # maintainer-only transform ownership lint checks (now in main config)
pnpm test test/conv.test.ts        # single file
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm vitest bench bench/<file>     # run Vitest benchmarks (e.g. bench/mega-module.bench.ts)
pnpm -C website dev                # local dev server for demos
```

### Pre-commit CI checks

Husky runs `lint-staged` on commit, which auto-fixes ESLint and Prettier issues on staged files. The
pre-commit hook also runs the full Vitest suite and Deno WebGPU tests (`pnpm vitest run` +
`pnpm run build` + `pnpm run test:deno`).

**Before any commit**, also run these checks manually to catch issues early:

```bash
pnpm build                         # Build the project
pnpm check                         # TypeScript type checking
pnpm exec playwright install chromium-headless-shell  # (if not already installed)
pnpm test                          # Run all Vitest tests
pnpm run test:deno                 # Run Deno WebGPU tests
```

These match the checks in `.github/workflows/ci.yaml`.

### CI timing profile (measured Feb 2026)

Full-profile pre-commit takes ~82s total. The breakdown:

| Step                 | Wall time | Notes                                       |
| -------------------- | --------- | ------------------------------------------- |
| `test:website:smoke` | **49s**   | SvelteKit build dominates (test itself 1ms) |
| `lint`               | 7s        | ESLint across full codebase                 |
| `test:policy:strict` | 5s        | Full Vitest (1300+ tests, 46 files)         |
| `test:deno`          | 4s        | Deno WebGPU (53 tests)                      |
| `format:check`       | 4s        | Prettier                                    |
| `build`              | 3s        | tsdown (5 packages)                         |
| `check`              | 3s        | TypeScript type checking                    |
| `lint:ownership:*`   | 5s        | Website ownership checks                    |
| `core invariants`    | 2s        | refcount + transform-compositions           |
| `test:eslint-plugin` | 1s        | Plugin unit tests                           |

**Bottleneck:** `test:website:smoke` is 60% of CI time. It builds the full SvelteKit website to
catch import breakage from `src/index.ts` — `pnpm check` (tsc) doesn't type-check `.svelte` files.
This only runs in the **full profile** (main-branch commits); feature branches skip it (~33s total).
The cost is acceptable given the safety it provides — do not try to cache or skip it, because the
website imports from the library and stale `.svelte-kit/` cache can silently pass when a fresh build
would fail.

**Docs-only fast path:** When all staged files are documentation (`.md`, `.txt`, `docs/`), the
pre-commit hook skips build, tests, type-checking, and ownership lints — running only
`format:check` + `lint`. This reduces docs-only commits from ~82s to ~11s. The detection runs in
`scripts/precommit.sh` before the profile check. If you're committing docs alongside code, the full
profile runs as usual.

### Testing policy modes

The repository supports two test policies:

- **Strict mode (default):** no failing tests and no expected-failure entries.
- **Architectural mode (opt-in):** failures are allowed only if listed in
  `.ci/expected-failures.json` with owner, reason, and expiry.

Use `JAX_ARCH_MODE=1 git commit -m "..."` to run architectural mode from pre-commit. See
`docs/testing-policy.md` for workflow and review guidance.

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

### Editing Prettier-managed files

When editing `.md` files (especially `copilot-instructions.md`), **run Prettier on the file before
reading it for `replace_string_in_file`**. Prettier reformats table alignment, line wrapping, and
whitespace, so the on-disk text may differ from what you wrote. The pattern:

```bash
npx prettier --write .github/copilot-instructions.md
```

Then `read_file` to get the Prettier-canonical text, then edit. This avoids repeated
`replace_string_in_file` failures from stale match text.

For **multi-step edits** to the same Prettier-managed file, run Prettier once before the first edit,
then again between batches if a batch triggers reformatting (e.g., table column width changes).

### Temporary files

Use `tmp/` in the project root for temporary/scratch files.

**Agent default policy:** For all ad-hoc scripts, repro files, logs, and generated diagnostics
related to this repository, write into `tmp/` (repo-local) instead of system temp directories.

- ✅ Preferred: `tmp/my-repro.ts`, `tmp/debug-output.txt`
- ❌ Avoid for repo work: `/tmp/...`, `/var/tmp/...`

This directory is gitignored and allows file operations without manual approval in VS Code.

### Debug logging

**IMPORTANT:** Do NOT use environment variables like `DEBUG=1`. Use the runtime function:

```typescript
import { setDebug } from "@hamk-uas/jax-js-nonconsuming";
setDebug(1); // Enable debug logging BEFORE any jit compilation
```

| Level | Output                                                                   |
| ----- | ------------------------------------------------------------------------ |
| 0     | No debug output (default)                                                |
| 1     | JIT compile logs, scan path selection, mega-module summary               |
| 2     | Shader code (WGSL/WebAssembly), detailed tracing, mega-module per-kernel |
| 3     | Expressions and metadata                                                 |
| 4     | JIT programs, tuning details, mega-module WASM hex dump                  |
| 5     | Most verbose operation traces                                            |

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
`Symbol.dispose`, `Symbol.asyncDispose`, and `SuppressedError` exist in environments that don't have
them yet (notably Safari).

**Svelte limitation:** Svelte's parser does not support the `using` keyword in `.svelte` files
(sveltejs/svelte#16192 is a draft PR). `.svelte` and `.svelte.ts` files must use explicit
`.dispose()` instead. The website's `src/polyfills.ts` + root layout import ensures the runtime
globals exist for any plain `.ts` files that do use `using`.

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

| Aspect          | Wasm (`src/backend/wasm.ts`)                                | WebGPU (`src/backend/webgpu.ts`)                      |
| --------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| Allocation      | `WasmAllocator` over `WebAssembly.Memory`                   | `device.createBuffer()` with `GPUBufferUsage.STORAGE` |
| Slot tracking   | `Map<Slot, {ptr, size, ref}>`                               | `Map<Slot, {buffer, size, ref}>`                      |
| Buffer copy     | `Uint8Array.copyWithin` (aligned/unaligned)                 | `copyBufferToBuffer` (aligned) or WGSL copy shader    |
| Sync read       | Direct memory view                                          | `SyncReader` with staging buffer + `mapAsync`         |
| Dispatch        | Instantiate Wasm module, call `kernel(start, end, ...ptrs)` | `commandEncoder.dispatchWorkgroups()`, queue submit   |
| Parallel        | `WasmWorkerPool` (SharedArrayBuffer + Atomics, M5)          | Workgroup parallelism (hardware-managed)              |
| Zero on alloc   | **Yes** — `.fill(0)` on free-list reuse                     | **Fresh only** — `createBuffer` zeros; pool does not  |
| Zero on recycle | N/A (JIT recycle = slot rename)                             | N/A (JIT recycle = slot rename)                       |

### Float64 support & numerical precision

**Float64 runs only on WASM and CPU — never on WebGPU.** WGSL has no 64-bit floating-point type;
`dtypeToWgsl()` in `src/backend/webgpu/codegen.ts` throws on `DType.Float64`. This means any f64
computation is dispatched to the WASM or CPU backend, which have full f64 support.

**Kahan compensated summation:** Float64 Add reductions use
[Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) to reduce accumulation
error from O(n·ε) to O(ε²). This matters for workloads like dynamic linear models (Kalman filters)
where f64 precision is required and naive summation of large arrays introduces visible drift.

Implemented in three places:

| Location              | Mechanism                                                                   |
| --------------------- | --------------------------------------------------------------------------- |
| `src/alu.ts`          | `Reduction.evaluate()` — early-return Kahan loop for Float64 Add            |
| `src/backend/cpu.ts`  | `useKahan` flag + inline compensation loop in CPU kernel execution          |
| `src/backend/wasm.ts` | `codegenReductionAccumulate()` optional `kahanComp` parameter; WASM codegen |

The WASM implementation integrates with scan via the shared `emitKernelBody()` function — any scan
body containing a Float64 Add reduction automatically gets Kahan compensation.

**Why not on GPU?** Kahan compensation requires f64 arithmetic for the compensation variable. Since
WGSL has no f64 type, it's not possible. Running Kahan with f32 compensation provides no benefit
because the compensation term is rounded away by f32 precision (ε ≈ 6×10⁻⁸ for f32 vs ε ≈ 1.1×10⁻¹⁶
for f64).

**Test coverage:** `test/dtype-f64.test.ts` — includes small-sum (n=10000, tol 1e-13), large
dot-product (n=50000, tol 1e-10), and JIT dot-product (n=5000, tol 1e-13) tests.

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

> **No Float64 on WebGPU.** WGSL has no `f64` type — `dtypeToWgsl()` throws on `DType.Float64`. All
> f64 work runs on WASM/CPU. See [Float64 support](#float64-support--numerical-precision).

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

`splitGraphDataflow()`'s **P2 pass** prevents this by counting transitive fused dependencies for
every kernel-dispatched equation and backtracking (splitting the fusion boundary) when
`depCounter.size > maxArgs`. This check applies to **all kernel-dispatched equations** — including
black kernel endpoints (Jaxpr outputs, multi-use vars) — not just white (fusable) ops. Non-kernel
blacks (Scan, Routines, DUS) are exempt since they use dedicated JIT step types, not the kernel
compiler. The `throw` above is a safety net for any equation that slips through; in practice
`splitGraphDataflow` should prevent it.

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

| Function                        | Role                                                             |
| ------------------------------- | ---------------------------------------------------------------- |
| `translateExpCore()`            | Shared core handling all `AluOp` cases                           |
| `TranslateExpContext` interface | Callbacks for `getVariable` and `handleGlobalIndex`              |
| `translateExp()`                | Wrapper with bounds-check GlobalIndex; `paramOffset` for M5      |
| `emitKernelBody()`              | Shared gidx loop + reduction + store; `startLocal`/`endLocal` M5 |
| `codegenWasm()`                 | Kernel codegen: `(start, end, ...ptrs)` signature (M5)           |
| `codegenWasmMulti()`            | Multi-output kernel codegen: `(start, end, ...ptrs)` (M5)        |

**`AluOp.Where` — cost-based branching (WASM only):**

Within `translateExpCore()`, `AluOp.Where` uses a cost model to choose between two code strategies:

- **Branchless `select`** — when both arms are cheap (estimated cost < 15): evaluates all three
  operands unconditionally, ~1 cycle. Used for patterns like `relu`, `clamp`, integer ternaries.
- **True `if/else/end` branching** — when at least one arm is expensive (cost ≥ 15): only the taken
  branch executes per element. Used when arms contain transcendental function calls.

`AluExp.estimateCost()` walks the expression tree via `fold()` and sums: **20** per op in
`AluGroup.Expensive` (sin, cos, asin, atan, exp, log, erf, erfc), **15** for `Threefry2x32`, **1**
for cheap arithmetic/comparisons/casts, **0** for leaves (Const/Variable/Special). The threshold
`max(costT, costF) >= 15` triggers branching.

**CSE safety:** The CSE mechanism in `translateExpCore` saves computed values to WASM locals via
`local.tee`. If a shared subexpression is first evaluated inside one `if` branch, its local would be
uninitialized (default 0) in the other branch. The codegen pre-evaluates all arm nodes with
`references > 1` that haven't already been cached, before the branch, to guarantee the local is
always set before use.

**WebGPU note:** This optimization is WASM-only. WebGPU executes all SIMD lanes in a workgroup
simultaneously — there is no per-element branching, so an `if/else` in WGSL evaluates both sides
regardless. The `select` instruction remains the correct strategy there.

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
| 7    | `src/frontend/scan-plan.ts`            | Add to `supportedRoutines` in `tryPrepareWasmNativeScan()`           |
| 8    | `src/backend/wasm.ts`                  | Add scan codegen in `codegenNativeScanGeneral()` + `ScanRoutineInfo` |
| opt  | `src/routine.ts`                       | Add CPU fallback in `runCpuRoutine()`                                |
| opt  | `src/frontend/jvp.ts`                  | Add JVP rule if autodiff needed                                      |
| opt  | `src/frontend/linearize.ts`            | Add transpose rule if grad needed                                    |

**Size key convention:** Cache keys include dtype and all size dimensions, e.g., `cholesky_f32_4` or
`triangular_solve_f64_8_16_lower_unit`.

### WASM feature opportunities (assessed Feb 2026)

| Priority | Feature            | Browser risk       | Impact      | Notes                                                                               |
| -------- | ------------------ | ------------------ | ----------- | ----------------------------------------------------------------------------------- |
| Medium   | i64 in wasmblr     | None (MVP)         | Medium-High | Unlocks proper f64 builtins (exp/log/sin/erf) and simplifies Threefry PRNG          |
| Medium   | Relaxed SIMD (FMA) | Safari unsupported | High        | `f32x4.relaxed_madd` for 2× dot-product throughput; needs runtime feature detection |
| Low      | Sign extension ops | None               | Low         | `i32.extend8_s` etc.; marginal for float-focused workloads                          |

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
- **The main `test/*.test.ts` Vitest suite also imports from `dist/`**, not `src/`. The package
  `"main"` field resolves `@hamk-uas/jax-js-nonconsuming` to `dist/index.js`, so source edits are
  invisible to those tests until after `pnpm build`. Only files under `src/` that use relative
  imports (e.g. `src/alu.test.ts`) see in-flight source changes directly.

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
- Types: `AssociativeScanOptions`

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
3. **Graph splitting** – `splitGraphDataflow()` marks vars as "black nodes" (forced materialization
   points) and identifies fusable elementwise ops for kernel fusion. Black nodes come in two kinds:
   - **Non-kernel blacks**: `Scan`, `Routine` primitives, `DynamicUpdateSlice` — handled by their
     own JIT step type, never compiled as generic kernels; exempt from the P2 `maxArgs` check.
   - **Kernel-endpoint blacks**: Jaxpr output vars and multi-use vars — still dispatched as regular
     WebGPU/WASM kernels but forced to materialise (not fused into downstream ops); subject to the
     P2 dep-count check.
4. **Kernel fusion** – Consecutive elementwise ops merge into a single `Kernel`
5. **Compilation** – `jitCompile(backend, jaxpr)` emits a `JitProgram` (list of `JitStep`s)
6. **Buffer lifecycle** – `effectDrivenAllocate()` runs a single-pass liveness analysis with a
   free-pool, replacing adjacent `free→malloc` pairs of the same size with zero-cost `recycle`
   steps. Supersedes the earlier `insertFreeSteps()` + `recycleBuffers()` two-pass approach.
7. **Execution** – `JitProgram.execute(slots)` runs steps, managing memory lifetime

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
| `recycle` | Reuse a freed buffer for a new allocation (zero backend cost)           |
| `scan`    | Scan loop (fallback, compiled-loop, or preencoded-routine via ScanPlan) |
| `dus`     | Zero-copy DynamicUpdateSlice (copies src into dst at byte offset)       |

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
5. If copy-like (e.g., `DynamicUpdateSlice`), it emits a dedicated `dus` JitStep in `jitCompile()`
   and is classified as a non-kernel black node in `splitGraphDataflow()`. DUS carries a `Mutate`
   effect on its first input (dst), enabling `effectDrivenAllocate` to recycle the dst buffer
   directly into the output slot (zero-copy when sizes match).

## Mega-Module (WASM JIT Fusion)

The mega-module compiler (`src/backend/wasm/mega-module.ts`) compiles a JitProgram's entire step
list into a **single WASM function**, eliminating all JS↔WASM boundary crossings between kernel
dispatches. This is the M6.1 milestone.

**Key exports:**

- `WasmMegaModule` — interface: `{ module, numInputs, numOutputs, outputSizes, kernelExports }`
- `ExtractedKernelInfo` — per-kernel metadata: `{ name, size, isReduction, nInputs, nOutputs }`
- `canCompileToMegaModule(steps)` — returns `false` for `incref`, `scan`, `dus`, `scatter_add`,
  `assoc_scan`, Routine steps, or symbolic `malloc` sizes
- `compileToMegaModule(steps, inputIds, outputIds)` — returns compiled module or `null`

**Architecture:**

```
mega_execute(input0_ptr, ..., inputN_ptr, resultBufPtr) → void
```

Input pointers are function parameters. Output pointers are written to a caller-allocated result
buffer. The module imports `env.alloc` and `env.free` from the host `WasmAllocator` for internal
buffer allocations.

**Step translation:**

| JitStep   | WASM translation                                                            |
| --------- | --------------------------------------------------------------------------- |
| `malloc`  | `local.set(id, call $alloc(size))` — imported alloc                         |
| `free`    | `call $free(local.get(id))` — imported free                                 |
| `recycle` | `local.set(new, local.get(old))` — zero-cost local rename                   |
| `execute` | `call $kernel_N(0, size, ...ptrs)` — extracted function (M6.2a)             |
| `execute` | Inlined reduction loop (reductions stay in `mega_execute`, not extractable) |

**Extracted kernel functions (M6.2a):**

Non-reduction kernels are compiled as separate WASM functions with signature
`(start: i32, end: i32, ...bufPtrs: i32[]) → void` and exported from the module (e.g., `kernel_0`,
`kernel_1`). `mega_execute` calls them via direct `call $kernel_N(0, size, ...)`. V8 inlines these
at runtime, so serial performance is unchanged. Reduction kernels remain inlined in `mega_execute`
because they have accumulator dependencies that prevent parallel decomposition.

**Debug logging:**

`setDebug(1)` prints a one-line summary after compilation (step count, kernel count, extracted vs
inlined, WASM byte size). `setDebug(2)` adds per-kernel detail lines (name, size, reduction status,
input/output counts). `setDebug(4)` dumps the full WASM binary as hex for `wasm-dis` inspection.

**What it catches vs. what it rejects:**

| Pattern                          | Supported? | Why                                            |
| -------------------------------- | ---------- | ---------------------------------------------- |
| Elementwise chain (add→mul→sub)  | ✅         | All kernel execute steps, fused inline         |
| Multi-output kernel steps        | ✅         | Each output written separately                 |
| Reduction kernels (sum, max)     | ✅         | Reduction loop inlined via `emitReductionBody` |
| JitProgram with `incref`         | ❌         | Refcount tracking inside WASM not supported    |
| Routine steps (sort, cholesky)   | ❌         | Would need WASM imports for each routine       |
| Scan / DUS / scatter_add steps   | ❌         | Complex control flow not yet inlined           |
| Pass-through outputs (out=input) | ❌         | Steps may overwrite input locals               |
| Symbolic malloc sizes            | ❌         | Rejected; needs runtime alloc resolution       |
| Symbolic reduction sizes         | ❌         | Rejected; loop bound can't be i32.const        |

**Integration in JitProgram.execute():**

```typescript
if (this.backend.type === "wasm") {
  if (this._megaModule === undefined) {
    this._megaModule = canCompileToMegaModule(this.steps)
      ? compileToMegaModule(this.steps, this.inputs, this.outputs)
      : null;
  }
  if (this._megaModule) {
    const outputSlots = (this.backend as WasmBackend).executeMegaModule(this._megaModule, inputs);
    return { outputs: outputSlots, pending: [] };
  }
}
```

The mega-module is compiled lazily on first execution and cached per JitProgram (`_megaModule`
field). Programs that can't be compiled fall through to the regular step-by-step execution path.

**Key files:**

| File                              | Purpose                                                                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `src/backend/wasm/mega-module.ts` | Compiler: `compileToMegaModule()`                                                                                             |
| `src/backend/wasm.ts`             | `executeMegaModule()` method on WasmBackend                                                                                   |
| `src/frontend/jit.ts`             | `_megaModule` cache + fast path in execute                                                                                    |
| `test/mega-module.test.ts`        | Mega-module correctness + leak detection (M6.1), extracted kernels (M6.2a), orchestrator tests (M6.2b), step metadata (M6.2c) |
| `test/deno/orchestrator.test.ts`  | Deno-only orchestrator + worker pool tests (17 tests: 12 M6.2b + 5 M6.2c, requires native SAB)                                |

## Orchestrator Worker (M6.2b — Off-Main-Thread Mega-Module)

The orchestrator worker (`src/backend/wasm/orchestrator.ts`) moves mega-module execution off the
main thread via a dedicated Web Worker. This eliminates main-thread blocking during WASM execution
and is a prerequisite for M6.2c (parallel kernel dispatch within the worker).

**Architecture:**

```
Main thread                          Worker thread
─────────────────────────────────────────────────────────
jit(f)(x)
  → executeMegaModule(module, inputs)
    → orchestrator.dispatch(moduleId, ptrs, resultBuf)
      ── postMessage({type:"dispatch"}) ──→
                                        instantiate module
                                        call mega_execute()
                                        ←── postMessage({type:"done"}) ──
    ← resolve Promise with output slots
```

The worker shares the same `WebAssembly.Memory` (via `SharedArrayBuffer`) as the main thread, so
input/output data is zero-copy — only metadata (module bytes, pointer offsets, result buffer
address) crosses the postMessage boundary.

**Key design decisions:**

- **Module registration + caching:** Compiled WASM module bytes are sent to the worker once via
  `registerModule()`, which returns a numeric `moduleId`. Subsequent `dispatch()` calls reference
  the cached module by ID, avoiding repeated module transfer.
- **Lazy creation:** `WasmBackend.#orchestrator` uses three-state tracking (`undefined` = not
  attempted, `null` = unavailable, instance = created). The `#getOrchestrator()` private method
  creates it on first mega-module execution, only when `SharedArrayBuffer` is constructable.
- **Module Workers:** Both `orchestrator.ts` and `worker-pool.ts` use
  `new Worker(url, { type: "module" })` (not classic workers). Deno requires module workers for blob
  URL workers; Chromium supports both.
- **SAB detection:** Uses constructability test
  `try { new SharedArrayBuffer(1).byteLength === 1 } catch { false }` instead of
  `crossOriginIsolated`. This works in Deno (native SAB support) and browsers (with COOP/COEP
  headers).
- **PostMessage-based wake:** The worker uses pure `postMessage`/`onmessage` protocol (no
  `Atomics.wait` on the main thread). The main thread creates a `Promise` per dispatch and resolves
  it when the worker posts `{type: "done"}`.

**SharedArrayBuffer testing limitation:**

Vitest's browser mode uses iframes to isolate tests. Setting COOP+COEP headers (required for
`crossOriginIsolated=true` in browsers) breaks the iframe communication — tests hang indefinitely.
This is a fundamental Vitest limitation, not a jax-js bug. Consequently:

- **Vitest tests** (`test/mega-module.test.ts`): 8 orchestrator tests use
  `describe.skipIf(!globalThis.crossOriginIsolated)` — always skipped in Vitest.
- **Deno tests** (`test/deno/orchestrator.test.ts`): 17 tests exercise the full orchestrator +
  worker pool. Deno has native `SharedArrayBuffer` support without COOP/COEP headers.

**Worker cleanup:** `WasmBackend.destroyWorkers()` terminates both the orchestrator worker and the
worker pool. Called in `test/setup.ts` `afterAll` hook to prevent worker threads from keeping the
process alive after tests complete.

**Blob URL revocation:** Both `WasmWorkerPool` and `OrchestratorWorker` create workers from blob
URLs (`new Worker(URL.createObjectURL(...), { type: "module" })`). URL revocation is **deferred**
via `setTimeout(() => URL.revokeObjectURL(url), 0)` because Deno's module worker loading is
asynchronous — immediate revocation can cause `Module not found` errors when multiple worker sets
are created in rapid succession (as happens when both pool and orchestrator are lazily created
during the first `jit` call).

**Key files:**

| File                                       | Purpose                                                    |
| ------------------------------------------ | ---------------------------------------------------------- |
| `src/backend/wasm/orchestrator.ts`         | OrchestratorWorker: module registration + dispatch         |
| `src/backend/wasm/worker-pool.ts`          | WasmWorkerPool: parallel kernel dispatch                   |
| `src/backend/wasm/shared-memory-config.ts` | Shared config: `MAX_SHARED_PAGES`, `configureMemoryImport` |
| `src/backend/wasm.ts`                      | Lazy `#getOrchestrator()`, `#getWorkerPool()`, SAB detect  |
| `test/deno/orchestrator.test.ts`           | 17 Deno tests for orchestrator + worker pool + M6.2c       |
| `test/mega-module.test.ts`                 | 8 orchestrator tests (skipped in Vitest, covered by Deno)  |

## Parallel Kernel Dispatch (M6.2c — JS-Driven Mega-Module Step Execution)

M6.2c adds a **parallel execution path** for mega-modules: instead of running `mega_execute` as a
monolithic WASM call, the main thread walks per-step metadata (`MegaStepInfo[]`) and dispatches
large kernels across workers via `WasmWorkerPool.dispatchSync`.

**Architecture:**

```
JitProgram.execute()
  → shouldUseParallelMegaModule() → true (pure check, no side effects)
  → registerMegaModuleOnPool()    → async (first call only)
  → executeMegaModuleParallelSync(megaModule, inputs)
      ├─ for each MegaStepInfo:
      │   "malloc"  → allocator.malloc(size)
      │   "free"    → allocator.free(ptr)
      │   "recycle" → locals[toIdx] = locals[fromIdx]
      │   "kernel"  → if size >= 4096: pool.dispatchSync(kernelIdx, ...)
      │               else:            mainInstance.exports.kernel_N(0, size, ...)
      └─ return output slots
```

**Key types (`src/backend/wasm/mega-module.ts`):**

```typescript
type MegaStepInfo =
  | { type: "malloc"; outputIdx: number; size: number }
  | { type: "free"; inputIdx: number }
  | { type: "recycle"; fromIdx: number; toIdx: number }
  | {
      type: "kernel";
      kernelIdx: number;
      kernelSize: number;
      inputIdxs: number[];
      outputIdxs: number[];
    };
```

`WasmMegaModule` extended with: `stepInfos: MegaStepInfo[]`, `numLocals: number`,
`outputLocalIdxs: number[]`.

**Key design decisions:**

- **All kernels extracted (M6.2c):** Both elementwise and reduction kernels are compiled as separate
  WASM functions (`kernel_0`, `kernel_1`, ...) exported from the mega-module. Reductions are
  parallelizable because each output element's reduction loop is independent (split by gidx range).
- **Pure `shouldUseParallelMegaModule`:** This method does NOT lazily create the worker pool — it
  only checks `capabilities.sharedMemory` and kernel sizes. Pool creation is deferred to
  `registerMegaModuleOnPool()`.
- **Lazy async registration:** First call: `shouldUseParallelMegaModule` returns true → kicks off
  async `registerMegaModuleOnPool` → falls through to monolithic `executeMegaModule`. Once
  registration completes (`_megaModulePoolReady === true`), subsequent calls use the parallel path.
- **Worker stub imports:** Workers instantiate the mega-module with
  `{ alloc: throwStub, free: noop }` — workers only call extracted `kernel_N` functions, never
  `mega_execute`.
- **PARALLEL_THRESHOLD = 4096:** Kernels with fewer elements run on the main thread only (dispatch
  overhead would exceed compute savings).
- **Named kernel dispatch:** `dispatchSync` gains a `kernelIdx` parameter. Workers select functions
  via `exports["kernel_" + kernelIdx]` (index ≥ 0) or `exports.kernel` (index < 0, legacy path).

**Worker pool control buffer layout (updated):**

| Slot | Name              | Purpose                                               |
| ---- | ----------------- | ----------------------------------------------------- |
| 0    | `CTRL_STATE`      | Worker state (READY=0, WORK=1)                        |
| 1    | `CTRL_MODULE_ID`  | Module ID for kernel lookup                           |
| 2    | `CTRL_START`      | Work chunk start index                                |
| 3    | `CTRL_END`        | Work chunk end index                                  |
| 4    | `CTRL_KERNEL_IDX` | Kernel index: -1=default, 0+=named `kernel_N` (M6.2c) |
| 5    | `CTRL_NUM_ARGS`   | Number of buffer pointer arguments                    |
| 6-25 | `CTRL_ARGS`       | Buffer pointer arguments (max 20)                     |

`WORKER_STRIDE = 26` per worker.

**Key files:**

| File                              | Purpose                                                        |
| --------------------------------- | -------------------------------------------------------------- |
| `src/backend/wasm/mega-module.ts` | `MegaStepInfo`, `compileToMegaModule` (all kernels extracted)  |
| `src/backend/wasm/worker-pool.ts` | `registerMegaModule`, `CTRL_KERNEL_IDX`, named dispatch        |
| `src/backend/wasm.ts`             | `executeMegaModuleParallelSync`, `shouldUseParallelMegaModule` |
| `src/frontend/jit.ts`             | `_megaModulePoolReady`, 3-way dispatch in `execute()`          |
| `test/mega-module.test.ts`        | 6 step metadata tests (M6.2c)                                  |
| `test/deno/orchestrator.test.ts`  | 5 parallel dispatch tests (M6.2c)                              |

## Effect system (memory effects)

The Jaxpr IR carries per-equation **memory effect annotations** that model how each primitive
interacts with buffer ownership. This enables the JIT compiler's `effectDrivenAllocate` pass to make
sound buffer recycling decisions — including zero-copy DUS — without heuristic liveness analysis.

**`MemoryEffect` enum (`src/frontend/jaxpr.ts`):**

| Effect    | Meaning                                                                |
| --------- | ---------------------------------------------------------------------- |
| `Alloc`   | Creates a new buffer (output of a primitive)                           |
| `Borrow`  | Reads a buffer without taking ownership (default for all inputs)       |
| `Consume` | Takes ownership; buffer cannot be used again                           |
| `Mutate`  | In-place modification (requires exclusive ownership of the input slot) |

**`primitiveInputEffects` table:** Per-primitive overrides for input effects. Returns an array of
effects for the primitive's inputs, or `undefined` for the default (all `Borrow`). Currently only
`DynamicUpdateSlice` has an override — its first input (dst) is `Mutate`.

**`verifyJaxprEffects()` — static borrow checker:** Walks the Jaxpr and enforces:

1. **No use after consume** — once a Var is `Consume`d, any subsequent `Borrow`/`Mutate` is an
   error.
2. **Mutate exclusivity** — a Var cannot appear as both `Mutate` and `Borrow` in the same equation's
   inputs.
3. **Dead allocation detection** — an `Alloc`'d Var that is never referenced, consumed, or returned
   is flagged as dead code.

Enable verification with `_setVerifyEffects(true)` — runs automatically at the end of every
`makeJaxpr()` call. Gated behind a flag for performance (disabled in production, enabled in tests
via `test/effect-checker.test.ts`).

**How effects drive buffer recycling:**

The `effectDrivenAllocate()` method in `JitProgramBuilder` uses effect annotations to identify
recycling opportunities. For DUS specifically, the `Mutate` effect on the dst input tells the
allocator that dst's buffer is exclusively owned and can be reused as the output slot — enabling
zero-copy `copyBufferToBuffer` within the same allocation. Without the effect annotation, the
allocator would conservatively allocate a new buffer for the output.

**Key files:**

| File                          | Purpose                                                |
| ----------------------------- | ------------------------------------------------------ |
| `src/frontend/jaxpr.ts`       | `MemoryEffect` enum, `primitiveInputEffects`, verifier |
| `src/frontend/jit.ts`         | `effectDrivenAllocate()`, DUS JitStep emission         |
| `test/effect-checker.test.ts` | 23 tests: effect annotations, verifier, DUS zero-copy  |

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
- **`splitGraphDataflow` P2 black-node distinction**: Black nodes are either _non-kernel_ (Scan,
  Routine, DUS — their own JIT step, exempt from `maxArgs`) or _kernel-endpoint_ (output vars,
  multi-use vars — still compiled as kernels, must pass `depCount ≤ maxArgs`). If adding a new
  black-node class, decide which category it belongs to and set `isNonKernelBlack` accordingly.
  Previously the P2 pass skipped ALL all-black-output equations, allowing kernel endpoints to
  accumulate arbitrarily many fused inputs — causing `Too many buffers (N) for WebGPU pipeline` in
  `jit(grad(assocScan))` with 3-tuple pytrees. Fixed in commit `3d6e450`.
- **WebGPU NaN comparison (`AluOp.Cmpne`) requires bitcast, not `min()` heuristic**: WGSL NaN
  propagation is unspecified — `min(NaN, inf)` may return NaN or inf depending on driver. The
  `AluOp.Cmpne` codegen (used by `np.isnan` via `x != x`) uses `bitcast<u32>` to inspect IEEE 754
  bits directly: f32 NaN = exponent all-1s (`0x7F800000`) && mantissa non-zero; f16 NaN = `0x7C00`
  && `0x03FF` non-zero. Fixed in commit `c912f52`. Do NOT revert to `min(x, inf) != x`.
- **Mega-module silent corruption with non-concrete values**: `canCompileToMegaModule` must reject
  any step with symbolic sizes (malloc or reduction). If a non-number value (e.g., `SymbolicSize`)
  leaks through to `cg.i32.const()`, it silently encodes as 0 via NaN coercion in `encodeSigned()`,
  producing correct shapes but identity-value results (0 for sum, -Infinity for max). A runtime
  guard in `i32.const()` now catches this with an actionable error message. When adding new symbolic
  size paths, check `canCompileToMegaModule` — see the MIGRATION NOTE there.
- **`no-unnecessary-ref` autofix vs internal tracer `.ref` propagation**: The
  `jax-js/no-unnecessary-ref` eslint rule has autofix (`--fix` removes `.ref`). This is safe for
  user code but **unsafe for internal tracer plumbing** where `.ref` must propagate to inner values
  (e.g., `BatchTracer.ref` in `vmap.ts` must call `this.val.ref` so that `dispose()` remains
  balanced). Commit `b5d4274` accidentally autofix'd `BatchTracer.ref`, causing `vmap(grad(erf))` to
  crash with `UseAfterFreeError`; fixed in `af96e54`. The rule now skips `.ref` in
  `UpdateExpression` and `BinaryExpression` contexts (e.g., `buffer.ref++`, `buffer.ref === 0`) to
  avoid false positives on backend buffer tracking objects. For files with many intentional `.ref`
  calls, a file-level `// jax-js-lint: allow-ref` directive (in leading comments) suppresses all
  warnings. Per-line `// jax-js-lint: allow-ref` comments are used for isolated sites (vmap.ts,
  jaxpr.ts, tree.ts, onnx/tensor.ts, optax/transform.ts). `linearize.ts` uses the file-level
  directive (31 internal `.ref` calls in the autodiff system).

## Known flaky tests

- **Deno WebGPU tests** (`test/deno/`): When running all Deno test files together in a single
  `deno test` invocation, GPU state pollution between files causes memory leak detection failures.
  The `test:deno` script runs each file as a separate `deno test` command (chained with `&&`).
- **Deno associative-scan-perf test** (`test/deno/associative-scan-perf.test.ts`): The speedup
  threshold (≥3× vs sequential scan) is hardware-dependent and fails on some machines (measured
  ~1.3–1.4× on Intel Core Ultra 5 125H integrated GPU). This is a performance regression test, not a
  correctness issue.

## Known framework bugs (`KNOWN_BUG` tests)

All known framework bugs have been resolved. The `KNOWN_BUG` tagging convention is retained for
future use if needed:

1. Write the test as it SHOULD work — the ideal behavior, no workarounds.
2. Tag the test name: `test("KNOWN_BUG(my-tag): description", () => { ... })`
3. Add a `// KNOWN_BUG(my-tag): explanation` comment above the test.
4. Keep the working workaround test nearby (e.g., jit-wrapped version, depth-3 cap).

```bash
grep -rn 'KNOWN_BUG(' test/   # Should return nothing
```

**Test status:** See `pnpm vitest run` output. No active KNOWN_BUG failures are expected. The LU JVP
finite-difference test was previously failing because the WASM LU routine uses native f32
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
   tag and update the inventory in this file
9. For **releases** (version bump + tag + push + GitHub release), follow the Maintainer Guide in
   `README.md`

## Documentation files

| File                              | Purpose                                        | When to update                        |
| --------------------------------- | ---------------------------------------------- | ------------------------------------- |
| `README.md`                       | Main project intro, tutorial, Maintainer Guide | Major features, API changes, releases |
| `FEATURES.md`                     | JAX/NumPy API compatibility table              | New supported functions               |
| `.github/copilot-instructions.md` | AI agent onboarding, scan feature tracking     | New patterns, scan development        |
| `packages/*/README.md`            | Package-specific docs                          | Package feature changes               |

## Where to start reading

- Entry & exports: `src/index.ts`
- Memory model: `test/refcount.test.ts`
- Backends: `src/backend/webgpu/`, `src/backend/wasm/`
- Demos: `website/src/routes/repl/`, `website/src/routes/mobileclip/`
- Deno WebGPU tests: `test/deno/webgpu.test.ts` — headless hardware GPU testing
- Scan tests: `test/lax-scan.test.ts` — comprehensive scan suite (~1880 lines)
- Associative scan tests: `test/lax-associative-scan.test.ts` — 27 tests covering correctness,
  reverse, non-zero axis, pytrees, autodiff, parallel Kalman filter, WASM compiled-loop

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
import { setDebug } from "@hamk-uas/jax-js-nonconsuming";
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

**Polymorphic length:** `planScan()` accepts `length: number | Dim`. The WASM compiled-loop path
supports symbolic length (length is a runtime i32 parameter). WebGPU and preencoded-routine paths
guard against symbolic length and fall back. The `ScanParams.length` field in `core.ts` is typed as
`number | Dim` to carry the symbolic dimension through the IR.

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

**Function signature:**
`(length: i32, ...consts, ...carryIn, ...xs, ...carryOut, ...ysStacked, ...internals, aux?)`

Length is the **first i32 parameter** (arg 0), making the compiled module reusable across different
scan lengths via `dynamic_axes`. All other argument base indices are shifted by +1.

1. **Pre-analysis** — Build `directWriteMap` deciding which internal buffers can be redirected
2. Import routine functions and helper math functions
3. Allocate WASM locals: iteration counter, data index, element indices
4. **Step 1**: Copy initCarry to carryOut (working buffer) via `memory.copy`
5. **Step 2**: Main scan loop (iter = 0..`local.get(lengthArg)`):
   - Compute dataIdx (reverse-aware: `local.get(lengthArg) - 1 - iter` or `iter`)
   - **Step 2a**: For each step, execute kernel or call imported routine
   - **Step 2b**: Copy Y outputs to `ysStacked` at iteration offset
   - **Step 2c**: Copy carry outputs to `carryOut` for next iteration
6. Return compiled `WebAssembly.Module`

**Polymorphic length:** Length was previously baked as `i32.const(length)` at two locations (loop
termination and reverse dataIdx). Now both use `local.get(lengthArg)`, and the reverse dataIdx is
computed at runtime via `local.get(lengthArg) - 1 - iter`. `dispatchNativeScanGeneral()` prepends
the concrete length value to the args array at dispatch time. The `NativeScanGeneralParams`
interface no longer contains a `length` field — length is passed as a dispatch-time argument.

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

> Multi-output kernel fusion is implemented (M3.1 ✅). The `Kernel` class uses `KernelOutput[]` —
> each output is analyzed independently for direct-write eligibility.

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

## Routine System & Codegen (Scan Additions)

> The full routine system (implementation status, wasmblr patterns, WasmHl/SIMD helpers, autodiff of
> routines, adding-a-new-routine checklist) is in [Part 1 → Routine system](#routine-system). The
> codegen architecture (WASM `translateExpCore`/`emitKernelBody`/`codegenWasm` and WebGPU
> `translateAluOpToWgsl`/`gen`/`createShaderEmitter`) is in
> [Part 1 → Codegen architecture](#codegen-architecture).

**Scan-specific routine behavior:**

- Scan modules use WASM imports to call routines from separate wasmblr modules, keeping the entire
  loop in native code. See `codegenNativeScanGeneral()` in `src/backend/wasm.ts`.
- Dynamic routine planning: `getScanRoutineInfo(routineName, routine)` queries backend capabilities.
  Returns `ScanRoutineInfo` if eligible for native compilation; `null` triggers fallback.
- Supported WASM scan routines: Cholesky, Sort, TriangularSolve, LU, Argsort.

**Scan-specific codegen functions (WASM):**

| Function                               | Role                                                |
| -------------------------------------- | --------------------------------------------------- |
| `translateExpWithGeneralScanContext()` | Wrapper with const/carry/xs/internal classification |
| `codegenNativeScanGeneral()`           | Full scan loop codegen with direct-write analysis   |

**Scan-specific codegen functions (WebGPU):**

| Function                        | Role                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| `genScanExpressionWithRidx`     | Scan-specific GlobalIndex + inline generation                     |
| `nativeScanMultiShaderSource()` | Full scan shader implementation (handles single and multi-kernel) |

### Native Scan Codegen

Native scan on both WASM and WebGPU generates single-output kernel codegen per step. Multi-kernel
scan bodies are handled by `nativeScanMultiShaderSource()` on WebGPU and
`codegenNativeScanGeneral()` on WASM.

### Shared kernel body: `emitKernelBody()`

The inner per-element loop is shared between `codegenWasm()` and `codegenNativeScanGeneral()` via
`emitKernelBody()`. Accepts optional `startLocal`/`endLocal` for `WasmWorkerPool` work-splitting
(scan always runs full range). Callers inject behavior through three callbacks:

| Callback         | `codegenWasm` provides             | `codegenNativeScanGeneral` provides                    |
| ---------------- | ---------------------------------- | ------------------------------------------------------ |
| `emitOutputAddr` | `local.get(outputArg) + gidx * bw` | Direct-write: `carryOut[c]`; else: `internal[idx]`     |
| `emitExp`        | `translateExp(exp, {gidx, ridx})`  | `translateExpWithGeneralScanContext(exp, scanCtx)`     |
| `emitStore`      | Simple typed store                 | Dual-store: primary + ysStacked (for direct-write + Y) |

Handles: gidx loop, bounds check, reduction identity/accumulate/epilogue via
`codegenReductionAccumulate()`, gidx increment, and loop branching.

## Known Limitations & Future Work

### Current limitations

| Limitation                                           | Workaround                                                                                                             | Backend |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| `numCarry ≠ numY` on WebGPU                          | Falls back to JS loop                                                                                                  | WebGPU  |
| WebGPU internal buffer deps in scan                  | Falls back to JS loop (O(N) dispatches)                                                                                | WebGPU  |
| Mixed kernel+routine bodies on WebGPU                | Falls back to JS loop                                                                                                  | WebGPU  |
| **`grad(scan)` backward on WebGPU with linalg body** | **O(N) dispatches** — transposed body has intra-step deps; reformulate backward pass as `associativeScan`, or use WASM | WebGPU  |
| `grad(scan)` ~2× compute overhead                    | Use `{ checkpoint: false }` for O(N)                                                                                   | All     |
| Sort in scan body on WebGPU                          | Uses JS loop (uniforms)                                                                                                | WebGPU  |
| Mixed-dtype carries on WebGPU                        | Use WASM backend or same-dtype carry                                                                                   | WebGPU  |

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
- **Kalman filter forward pass**: Mixes matmul (kernel) + routines in one body → mixed
  kernel+routine → fallback.
- **Kalman/DLM backward pass (RTS smoother)**: The autodiff-transposed body produced by
  `jit(grad(scan))` contains sequential matmul→matmul→add chains where each step reads from the
  previous step's output. This is **internal buffer dependency** — the defining condition that
  triggers WebGPU fallback. Each of the N scan iterations requires multiple sequential GPU
  dispatches orchestrated from JS, giving **O(N) total dispatch calls**. For N=1600 this causes ~1 s
  latency per call on typical hardware. **This is the dominant performance bottleneck for any
  `grad(scan)` over a linalg-heavy body on WebGPU.**

**Why autodiff-transposed bodies predictably produce internal buffer deps:** The chain rule for a
multi-step forward body `B = f(A); C = g(B, x)` transposes to `dA = f_T(dB); dB = dB + g_T(dC)` — a
sequential dependency chain. No matter how simple the forward body is, if it contains two or more
operations that compose (output of one feeds input of next), the transposed body will have internal
deps. This is not a fixable code path in the scan executor; it is a consequence of autodiff algebra.

**Workarounds for the backward pass O(N) bottleneck:**

1. **Reformulate as `associativeScan`** — if the backward recursion can be expressed as a parallel
   prefix over associative affine maps (as in the Solin/Särkkä parallel Kalman smoother), it gets
   O(log N) dispatches on WebGPU. This requires mathematical reformulation, not a code-level fix.
   See: Särkkä, S. & García-Fernández, Á. F. (2020). "Temporal Parallelization of Bayesian
   Smoothers." _IEEE Transactions on Automatic Control_.
   [arXiv:1905.13002](https://arxiv.org/abs/1905.13002)
2. **Use WASM backend** — WASM compiled-loop handles internal buffer deps by allocating temporaries
   inside the module, running the entire N-iteration backward pass in a single WASM invocation.

**This limitation is significant** for any dynamical model using `grad(scan)` on WebGPU:

1. WASM `compiled-loop` handles all these cases natively — WebGPU is the affected backend
2. WebGPU fallback is **not** just command encoding overhead; it is a full JS↔GPU round-trip per
   iteration
3. O(N) dispatch cost dominates all other costs for N ≳ 100

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

| Priority | Feature                        | Notes                                                                                          |
| -------- | ------------------------------ | ---------------------------------------------------------------------------------------------- |
| Medium   | Missing test categories        | ~30 additional tests: WASM routine scan, path-documentation, advanced vmap/grad compositions   |
| Medium   | Mixed-dtype WebGPU scan shader | Per-binding dtype in `nativeScanMultiShaderSource`                                             |
| Medium   | WebGL copy for scan stacking   | Enable direct-write stacked Ys on WebGL fallback                                               |
| Low      | WebGPU polymorphic scan length | WebGPU compiled-loop does not support symbolic length; currently falls back to JS loop         |
| Low      | Polymorphic scan backward pass | `grad(scan)` transpose rule uses concrete loop; needs `Primitive.Scan` emission for symbolic N |

---

## Test Coverage Summary

### Test files

| File                                    | Purpose                                                                                  |
| --------------------------------------- | ---------------------------------------------------------------------------------------- |
| `test/lax-scan.test.ts`                 | Main scan test suite (~1880 lines)                                                       |
| `test/scan-backends.test.ts`            | Backend coverage & `copyBufferToBuffer` checks                                           |
| `test/scan-bench.test.ts`               | Scan benchmark tests                                                                     |
| `test/deno/webgpu.test.ts`              | Headless WebGPU tests via Deno                                                           |
| `test/deno/pool-memory.test.ts`         | Pool peak memory guarantee (Deno WebGPU)                                                 |
| `test/deno/scan.bench.ts`               | Deno WebGPU scan benchmarks                                                              |
| `test/wasm-parallel.test.ts`            | WASM parallel dispatch + kernel signature (M5)                                           |
| `test/mega-module.test.ts`              | Mega-module correctness + leak detection (M6.1), extracted kernels (M6.2a), orch (M6.2b) |
| `test/deno/orchestrator.test.ts`        | Orchestrator + worker pool tests (Deno-only, 12 tests)                                   |
| `test/deno/parallel-assoc-scan.test.ts` | M7.3 parallel Kogge-Stone tests (Deno-only, 5 tests)                                     |

### Deno WebGPU & leak detection

> See [Part 1 → Deno WebGPU test guidelines](#deno-webgpu-test-guidelines) for full Deno testing
> rules (device reuse, `dist/` imports, buffer cleanup, module parallelism, leak detection).

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

### Vitest benchmarks (M8.1)

Three Vitest benchmark files in `bench/` measure key subsystems:

```bash
pnpm build && pnpm vitest bench bench/mega-module.bench.ts
pnpm build && pnpm vitest bench bench/scatter-add.bench.ts
pnpm build && pnpm vitest bench bench/associative-scan.bench.ts
```

| File                              | Benchmarks | What it measures                                                        |
| --------------------------------- | ---------- | ----------------------------------------------------------------------- |
| `bench/mega-module.bench.ts`      | 7          | Mega-module vs step-by-step: chains, multi-output, reduce, grad, matmul |
| `bench/scatter-add.bench.ts`      | 3          | `scatterAdd` throughput at 1K/10K/100K elements                         |
| `bench/associative-scan.bench.ts` | 6          | `associativeScan` vs sequential `scan` for cumsum/cumprod               |

**Import notes:**

- `bench/scatter-add.bench.ts` imports from `../src/frontend/core` (not the public API) because
  `scatterAdd` is not exported from `src/index.ts`. Uses `type Array as JaxArray` to avoid global
  `Array` collision.
- Other bench files import from `@hamk-uas/jax-js-nonconsuming` (public API via `dist/`).
- All bench files use `DType.Float32` / `DType.Int32` enums, not string literals.

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

### Anonymous constants in scan bodies (fixed)

Inline `np.array(...)` constants created inside traced scan/jit bodies are now treated as
builder-owned anonymous consts. Arrays created from raw literals during tracing are tagged in
`array()`, so `getOrMakeConstTracer` skips the extra `.ref` and `ClosedJaxpr.dispose()` fully
balances ownership.

Regression coverage:

- `test/leak-diagnostic.test.ts`: `body with inline np.array constant does not leak`
- `test/leak-diagnostic.test.ts`:
  `inline np.array constants with xs=null length 0 and 1 do not leak`

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

1. After `partialEvalGraphToJaxpr` (net zero), residual cleanup takes the last ref (rc → 0), killing
   the ClosedJaxpr's ownership before the backward pass reads it.

**Fix:** In both `linearizeFlat` and `vjpFlat`, before residual cleanup, protect jaxpr consts whose
`c.refCount <= 1`. Normal consts (rc ≥ 2) are unaffected; only user-disposed consts (rc = 1) are
protected from over-disposal.

### AOT linearization artifacts

The autodiff system uses explicit **artifact types** to encapsulate ownership of forward-pass
residuals and backward-pass jaxprs. This replaces the earlier pattern of ad-hoc PE intermediate
disposal with structured, `Symbol.dispose`-compatible objects.

**Key types (`src/frontend/artifacts.ts`):**

| Type                | Owns                                                     | Disposed by                              |
| ------------------- | -------------------------------------------------------- | ---------------------------------------- |
| `PrimalArtifact`    | Forward jaxpr consts, `ResidualCollector` with residuals | `[Symbol.dispose]()` or `.dispose()`     |
| `PullbackArtifact`  | Backward jaxpr (locally-owned), residual refs            | `[Symbol.dispose]()` or `.dispose()`     |
| `ResidualCollector` | Residual arrays collected during forward pass            | `ResidualCollector.dispose()` at cleanup |

**Scan-specific (`src/frontend/scan-backward.ts`):**

| Type                   | Owns                                                          | NOT owned (cache-owned)                       |
| ---------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| `ScanPullbackArtifact` | `primalForwardJaxpr`, `tangentBody`, carry/xs/const residuals | `transposedBody` (from `transposeJaxprCache`) |

**`aotLinearize` flow (`src/frontend/artifacts.ts`):**

`aotLinearize(f, primals, options)` replaces the older `partialEvalFlat` + manual disposal pattern:

1. Traces forward pass with `linearizeFlat` (with `skipBackward: true` — no call-time transpose)
2. Collects residuals into a `ResidualCollector`
3. Returns `PrimalArtifact` (owns forward jaxpr + residuals) and optionally a `PullbackArtifact`
   (owns backward jaxpr + residual refs)
4. Both artifacts implement `Symbol.dispose` for `using` declarations

**Ownership contracts:**

- `transposeJaxprCache` (Map in `linearize.ts`) — **cache-owned**. Callers of `transposeJaxpr()`
  must NOT dispose the returned `ClosedJaxpr`. The cache is cleaned by `_registerJitCacheDisposer`.
- `primalForwardJaxpr` and `tangentBody` in `ScanPullbackArtifact` — **locally-owned**. Created
  fresh via `makeJaxpr`, disposed in `disposeResiduals()`.
- `markAnonymous: true` — passed to `transposeJaxpr()` and `buildBackwardJaxpr()` (inside
  `makeJaxpr` traces). Prevents UAF when cotangent zeros escape to outer jit traces by marking them
  as anonymous consts owned by the builder.

**Key files:**

| File                            | Purpose                                                                    |
| ------------------------------- | -------------------------------------------------------------------------- |
| `src/frontend/artifacts.ts`     | `aotLinearize`, `PrimalArtifact`, `PullbackArtifact`, `ResidualCollector`  |
| `src/frontend/scan-backward.ts` | `ScanBackwardSpec`, `ScanPullbackArtifact`                                 |
| `src/frontend/linearize.ts`     | Scan transpose rule (builds spec, creates artifact), `transposeJaxprCache` |

## Debugging Ownership Issues

If a `UseAfterFreeError` or `ReferenceError` appears:

1. **Identify the array** — the error includes the array's shape/dtype.
2. **Check artifact ownership** — is a `PrimalArtifact`, `PullbackArtifact`, or
   `ScanPullbackArtifact` being disposed too early? Verify that locally-owned jaxprs
   (`primalForwardJaxpr`, `tangentBody`) are not disposed before the backward pass runs.
3. **Check cache-owned jaxprs** — `transposeJaxpr()` returns cache-owned `ClosedJaxpr`s. Callers
   must NOT dispose them. The cache is cleaned by `_registerJitCacheDisposer`.
4. **Check disposal timing** — is some code path calling `.dispose()` prematurely? Common in
   `evalJaxprTransposed` (check `argPrimals` set) or PETracer cascade.
5. **Check `getOrMakeConstTracer`** — is a constant being disposed before the jaxpr that uses it?
6. **Add `console.log(slotCount())` checkpoints** around the failing code to narrow down where slots
   are being freed.
7. **Test with CPU backend** — CPU is simplest and has the clearest error messages.

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

### Static analysis: ESLint plugin

The community [`@hamk-uas/eslint-plugin-jax-js`](https://github.com/hamk-uas/eslint-plugin-jax-js)
ESLint plugin targets the **upstream move-semantics** jax-js. Its three rules
(`no-use-after-consume`, `no-unnecessary-ref`, `require-consume`) are not directly applicable to
this non-consuming fork, but the **design philosophy is identical**: code that is correct in eager
mode will be correct under `jit()`.

**In-repo plugin (publish-ready, v0.1.0):**

The in-repo plugin at `packages/eslint-plugin` (`@jax-js/eslint-plugin`) is wired into root
`eslint.config.ts` under plugin name `jax-js`. It exports `configs.recommended` and `configs.strict`
for flat-config consumers.

Current rollout in this repo:

- All `jax-js/*` rules: `warn` globally (base config)
- `configs.invariance` overlay: upgrades to `error` on `src/**`, `packages/**`, `test/**`
- `jax-js/require-using`: `off` for internal framework code (`src/frontend/**`, `src/library/**`,
  `src/backend/**`) — internal code manipulates Slots/Jaxprs, not `np.Array`
- `jax-js/no-array-chain`: `warn` everywhere (intentional benchmark chains use inline disable
  comments)
- Internal transform rules (`require-retained-release`, `require-try-finally-symmetry`,
  `require-wrapper-dispose-symmetry`): `warn` globally — no longer require a separate
  `lint:ownership:internal` script invocation

**Shared configs:**

- `configs.recommended` — `require-using: warn`, `no-use-after-dispose: error`,
  `no-dispose-then-reassign-param: warn`, `no-unnecessary-ref: warn`, `no-array-chain: off`
- `configs.strict` — all rules at `error` level, including `no-array-chain`
- `configs.internalTransforms` — maintainer-focused retention symmetry checks:
  `require-retained-release`, `require-try-finally-symmetry`, `require-wrapper-dispose-symmetry`

**User setup (external consumers):**

```ts
// eslint.config.ts
import jaxJs from "@jax-js/eslint-plugin";
export default [jaxJs.configs.recommended];
```

Implemented rules:

- `jax-js/require-using` — suggestion fix (converts `const`/`let` to `using`)
- `jax-js/no-use-after-dispose` — includes `.dispose()` line number in message
- `jax-js/no-dispose-then-reassign-param` — flags `dispose(state); state = param` callback alias
  hazards
- `jax-js/no-unnecessary-ref` — **autofix** (removes `.ref` with `--fix`)
- `jax-js/no-array-chain` — reports outermost chain only (no duplicate subchain reports)
- `jax-js/require-scan-result-dispose` — warns when `lax.scan()` destructured results are not
  disposed; currently `off` in config (planned for future enablement)
- `jax-js/no-make-disposable-alias` — warns when arrays passed to `tree.makeDisposable()` are also
  used directly afterward (the disposable wrapper would dispose them unexpectedly)
- `jax-js/require-retained-release` — retained `.ref` handles must have explicit terminal paths
- `jax-js/require-try-finally-symmetry` — `.ref` temporaries in `try` must be cleaned in `finally`
- `jax-js/require-wrapper-dispose-symmetry` — wrapper `dispose()` should release retained state
  before `this.inner.dispose()`

Current semantics in the in-repo implementation:

- `jax-js/require-using` treats local bindings as compliant when they are:
  - declared with `using`,
  - returned later in the same block,
  - explicitly disposed later via `<name>.dispose()`, or
  - persisted to longer-lived structures (e.g., property/index assignment, `set`/`push`/`add`).
- `jax-js/no-use-after-dispose` tracks identifiers by lexical variable identity (scope-aware),
  avoiding false positives from shadowed names. Error messages include the line number of the
  `.dispose()` call for easy cross-referencing.
- `jax-js/no-unnecessary-ref` has autofix: `--fix` removes `.ref` (the dot and property) from the
  chain. Safe for user code because `.ref` is never needed in the non-consuming model. **Unsafe for
  internal tracer `.ref` propagation** — see [Common pitfalls](#common-pitfalls) for the
  `BatchTracer.ref` incident. The rule automatically skips `.ref` in `UpdateExpression` and
  `BinaryExpression` contexts (e.g., `buffer.ref++`, `buffer.ref === 0`) to avoid false positives on
  backend buffer tracking objects. For files with many intentional `.ref` calls, a file-level
  `// jax-js-lint: allow-ref` directive (in leading comments) suppresses all warnings. Per-line
  `// jax-js-lint: allow-ref` comments are used for isolated sites — both on the preceding line and
  as **inline trailing comments** on the same line (e.g., `x.ref; // jax-js-lint: allow-ref`). The
  `hasAllowComment` function detects both styles via raw source-text scanning.
- `jax-js/no-array-chain` deduplicates: only the outermost qualifying chain is reported, avoiding
  noisy depth-N + depth-(N-1) + ... reports for a single expression.
- `jax-js/no-dispose-then-reassign-param` scans function bodies for adjacent
  `dispose(stateVar); stateVar = paramName;` patterns and recommends an alias guard
  (`if (stateVar !== paramName) ...`) before disposal.

See `packages/eslint-plugin/README.md` for full user-facing documentation, rule details, IDE
integration (VS Code, WebStorm, Neovim, Sublime Text), troubleshooting, and comparison with the
community HAMK plugin.

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

### Disposal patterns guide

Common disposal patterns for library users, ordered by frequency of use:

**1. `using` for intermediates (the 80% case):**

```ts
using x = np.array([1, 2, 3]);
using y = x.mul(np.array([2, 2, 2]));
const result = y.sum();
// x, y auto-disposed at block end; result is returned or disposed separately
```

**2. Block scope + `using` for early disposal:**

Free memory before continuing with non-array work:

```ts
let values;
{
  using result = tree.makeDisposable(await lax.scan(step, init, xs));
  values = await result[0].data(); // extract raw JS data
}
// GPU memory is freed here, values is a plain TypedArray
```

**3. Scan carry and output disposal — most dangerous leak site:**

After `lax.scan`, the carry is a fresh allocation. Forgetting to dispose it is silent. Use
`tree.dispose()` or `tree.makeDisposable()`:

```ts
// Option A: tree.dispose for manual cleanup
const [carry, ys] = lax.scan(step, init, xs);
// ... use carry and ys ...
tree.dispose(carry); // disposes carry.x, carry.C, etc.
tree.dispose(ys); // disposes ys.x_pred, ys.K, etc.

// Option B: tree.makeDisposable for auto-cleanup
{
  using result = tree.makeDisposable(lax.scan(step, init, xs));
  const [carry, ys] = result;
  // ... use carry and ys ...
} // all arrays in result auto-disposed
```

**4. "Extract and dispose" at the JS boundary:**

Use `consumeData()` to read data and dispose in one call:

```ts
const floats = await arr.consumeData();
// arr is disposed, floats is a plain Float32Array
```

**5. `tree.makeDisposable` for result structs:**

Attach `Symbol.dispose` to any object containing arrays:

```ts
using result = tree.makeDisposable({ x: np.array([1]), y: np.array([2]) });
// result.x and result.y auto-disposed at block end
```

**6. JIT output pytree aliasing guarantee:**

When a JIT function returns the same tracer under multiple output keys (e.g.,
`{ xf_0, yhat: xf_0 }`), the materialised result contains independent `np.Array` instances — one per
key. Each can be disposed independently.

### Migration guide from move semantics

For users migrating from the `.ref` / move-semantics model:

1. **Remove all `.ref` calls** — operations no longer consume inputs
2. **Replace `disposeAll(a, b, c)` with `using` / `.dispose()`** — or use `tree.dispose()`
3. **`using` for intermediates, `.dispose()` for object properties** — or wrap in
   `tree.makeDisposable()`
4. **`.data()` no longer auto-disposes** — add `.dispose()` after reading, or use `.consumeData()`
5. **Never use `using` on values that are returned** — `using` disposes at scope end
6. **Destructure `tree` module** to avoid `no-use-after-dispose` false positive:
   ```ts
   const { dispose: disposeTree, makeDisposable } = tree;
   ```

### `checkLeaks` diagnostic (implemented)

A zero-overhead leak detection tool that enforces ownership correctness at test time. When active,
it snapshots backend slot counts across ALL devices and tracks Array creations with lazy Error
objects for stack traces. The `leaked` count uses backend `slotCount()` deltas, so it exactly
matches existing leak detection behavior. Used in the global test setup (`test/setup.ts`) to wrap
every test with automatic leak checking — **every single test in the suite must be leak-free**.

```ts
import { checkLeaks } from "@hamk-uas/jax-js-nonconsuming";

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
the API surface JAX-compatible. The `@jax-js/eslint-plugin` catches leaks and use-after-free
statically at edit time, complementing the runtime `checkLeaks` diagnostic.

## Future Work

| Priority | Feature                             | Notes                                                                                                                                                                                           |
| -------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Medium   | Anonymous constant leak fix         | Partially addressed by `markAnonymous` flag in `transposeJaxpr`/`buildBackwardJaxpr`; full fix would distinguish user-held vs anonymous consts in scan tracing                                  |
| Low      | Chain→temporaries RFC (design-only) | Keep implementation deferred. Scope RFC to assignment/return chains only, preserve evaluation order + exceptions, and require measured eager-memory wins / lint-pressure before implementation. |

---

# Part 5: Associative Scan (`lax.associativeScan`)

## Overview

`lax.associativeScan` applies an **associative** binary operator as a parallel prefix scan —
computing the cumulative result at every position in O(log N) parallel rounds instead of O(N)
sequential steps. Unlike `lax.scan` (which threads explicit carry state step-by-step),
`associativeScan` requires only associativity of `fn` and exploits it for GPU-friendly parallelism.

**Signature:**

```ts
const result = lax.associativeScan(fn, elems, options);
// fn: (a: T, b: T) => T          — associative binary operator
// elems: T                        — pytree of Arrays with scan axis of length N
// options: { axis?: number, reverse?: boolean }
// result: T                       — same shape/structure as elems
```

**Result semantics:**

```
result[0] = elems[0]
result[i] = fn(result[i-1], elems[i])   for i ≥ 1
```

**Options:**

- `axis?: number` — Axis to scan along (default `0`).
- `reverse?: boolean` — If `true`, scan right-to-left:
  `result[i] = fn(elems[i], fn(... fn(elems[i+1], elems[N-1]))...)` (default `false`).

**Key files:**

| File                                  | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `src/library/lax-associative-scan.ts` | Public API, Kogge-Stone core, primitive-based JIT path |
| `src/frontend/core.ts`                | `Primitive.AssociativeScan` enum + params type         |
| `src/frontend/jaxpr.ts`               | Abstract eval rule                                     |
| `src/frontend/array.ts`               | Eager `Primitive.AssociativeScan` impl                 |
| `src/frontend/jit.ts`                 | JIT step for `Primitive.AssociativeScan`               |
| `src/frontend/scan-plan.ts`           | `AssocScanPlan` type, `planAssociativeScan()` planning |
| `src/backend/wasm.ts`                 | `codegenNativeAssociativeScan()` compiled-loop codegen |
| `src/frontend/jvp.ts`                 | JVP rule (forward-mode AD with doubled inputs)         |
| `src/frontend/linearize.ts`           | PE rule + transpose rule for `grad`                    |
| `src/frontend/vmap.ts`                | Vmap rule (batches independent scans along batch axis) |

**Export path:** `lax.associativeScan` (re-exported from `src/library/lax.ts`)

**Test file:** `test/lax-associative-scan.test.ts` — 27 tests

---

## Algorithm: Kogge-Stone Doubling

Each round doubles the reach of accumulated prefix results:

```
Round 1 (stride=1): result[i] = fn(result[i-1], result[i])   for i ≥ 1
Round 2 (stride=2): result[i] = fn(result[i-2], result[i])   for i ≥ 2
Round 3 (stride=4): result[i] = fn(result[i-4], result[i])   for i ≥ 4
...
```

Each round calls `fn` once with batched inputs of shape `[N-stride, ...]` — fully parallel across
all positions. The result array for round k is:

```
next = concat(current[0:stride], fn(current[0:N-stride], current[stride:N]))
```

After `ceil(log₂ N)` rounds, `result[i]` = prefix up to `i`. Complexity: O(N log N) total work,
O(log N) depth.

**Why this maps well to GPU kernels:** Each round's `fn` is a batched elementwise operation over N
elements — fully parallel across all positions. The JIT graph forces materialization before each
`Concatenate`, so each round costs exactly 1 dispatch (for elementwise `fn`). `ceil(log₂ 1024) = 10`
dispatches vs. 1024 for a sequential scan. This is also the **floor**: Kogge-Stone needs a global
barrier between rounds, which WebGPU cannot provide within a single dispatch.

---

## Implementation Details

### Backend behaviour

`associativeScan` is registered as `Primitive.AssociativeScan` with a body sub-jaxpr. When called
outside abstract tracing (eager mode or inside `jit`), it traces the body function once via
`makeJaxpr`, then dispatches the Kogge-Stone ladder using `evalJaxpr` per round with batched inputs.
If body tracing fails (e.g., einsum with batch-dimension-dependent subscripts), it falls back to
direct `associativeScanCore` which unrolls into the current trace.

The Kogge-Stone rounds use high-level array primitives internally: `core.shrink` (O(1) ShapeTracker
slice views), `core.flip`, `core.concatenate`, and `moveaxis`. On **WASM**, M7.2 added a dedicated
compiled-loop (`codegenNativeAssociativeScan()`) that compiles the entire Kogge-Stone ladder into a
single WASM module with N as a runtime i32 parameter. On **WebGPU**, the JS-driven ceil(log₂ N)
dispatch path is already the hardware-imposed floor — no further backend specialization is needed.
M7.3 added multithreaded Kogge-Stone: a parallel inner-round `kernel` export enables
`WasmWorkerPool` to split work across cores for arrays ≥ 4096 elements.

On **WASM**, M7.2 compiled the entire Kogge-Stone ladder into a single WASM module via
`codegenNativeAssociativeScan()`. N is a runtime i32 parameter enabling polymorphic length.
Ping-pong buffers are allocated by the caller at dispatch time. This eliminates all ceil(log₂ N)
JS→WASM boundary crossings.

| Backend    | What each Kogge-Stone round does                                                                                                                                                                                                                                                                                                                   | Performance vs `lax.scan`                                                                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **WebGPU** | ceil(log₂ N) JS-driven kernel dispatches total: one per round (fn output materialized before each Concatenate). Hardware-imposed floor — no cross-workgroup global barrier exists. Reductions in `fn` add extra dispatches per round. Eager mode uses cached whole-call `jit`.                                                                     | **Faster** for N≳1024: ceil(log₂ N) dispatches vs N sequential in-shader iterations (scan compiled-loop). Already optimal for WebGPU. Measured ~5–8× for N=65536 scalar prefix product.  |
| **WASM**   | **Compiled-loop (M7.2 + M7.3):** Full Kogge-Stone ladder compiled into one WASM module. Single JS→WASM call for any N (monolithic). M7.3 adds a parallel `kernel` export: for N ≥ 4096, JS drives the Kogge-Stone loop and dispatches each round's kernel across `WasmWorkerPool` workers. Per-j indexed internal buffers prevent race conditions. | **Fast** — monolithic: single WASM invocation, zero boundary overhead. Parallel (M7.3): JS-driven rounds with multi-core work-splitting per round. O(N log N) total work vs scan's O(N). |
| **CPU**    | Same as WASM (interpreted JS TypedArray ops per round)                                                                                                                                                                                                                                                                                             | **Slower** for the same reasons                                                                                                                                                          |
| **WebGL**  | 1 WebGL shader dispatch per round (scan uses JS fallback on WebGL)                                                                                                                                                                                                                                                                                 | Likely faster than `lax.scan` on WebGL since scan has no compiled-loop there; untested                                                                                                   |

### Ownership model

The function never consumes its inputs (`elems` leaves). Internal intermediates are tracked via
explicit `owned: boolean[]` parallel arrays — no identity-comparison tricks. Disposal is precise:

| Array                        | When disposed                                        |
| ---------------------------- | ---------------------------------------------------- |
| `moveaxis` views (axis≠0)    | After reverse-flip or directly after use             |
| `flip` views (reverse=true)  | After the main loop; after post-reverse flip         |
| `sliceAxis` left/right views | Immediately after `fn` returns (before concat)       |
| `fn` output leaves           | After all `next[i] = concat(prefix, output[i])` done |
| Previous `current[i]`        | After `next` array is fully built                    |
| Post-reverse flips           | After moveaxis-back                                  |

`fn` is responsible for disposing its own internal intermediates. The fn output pytree leaves are
owned by `associativeScan` and are disposed after being consumed into `next` via concat.

### Slicing strategy

All input slices use `core.shrink(a, slice)` — O(1) ShapeTracker views with no allocation. The only
allocations per round are the `fn` output leaves and the `concat` results for `next`.

### Pytree support

`tree.flatten<Array>(elems)` extracts all leaf arrays. `tree.unflatten(treedef, leaves)` rebuilds
the pytree for `fn` calls. The caller's pytree structure is preserved in the result. All leaves must
have the same size on the scan axis.

### Reverse scan

A reverse scan is implemented by:

1. Flip all input leaves along the scan axis (`core.flip(a, [0])` after `moveaxis`)
2. Run the standard forward scan
3. Flip the result back before returning

This is equivalent to JAX's approach and produces the inclusive right-to-left prefix:
`result[i] = fn(elems[i], fn(elems[i+1], ... fn(elems[N-2], elems[N-1])...))`

### Autodiff architecture (`Primitive.AssociativeScan`)

`associativeScan` is registered as `Primitive.AssociativeScan` with dedicated transform rules:

| Transform     | Rule location          | Strategy                                                                              |
| ------------- | ---------------------- | ------------------------------------------------------------------------------------- |
| **JVP**       | `src/frontend/jvp.ts`  | Doubles inputs (primal+tangent), runs Kogge-Stone with JVP'd body via `jvpJaxprCache` |
| **PE**        | `linearize.ts`         | Partial eval rule: all-unknown → opaque equation; builds transposed body for grad     |
| **Transpose** | `linearize.ts`         | Reverse sequential scan of N-1 iterations using transposed body jaxpr                 |
| **Vmap**      | `src/frontend/vmap.ts` | Maps independent scans along batch axis with axis permutation                         |

**How autodiff works:** When `grad(f)` traces through `associativeScan`, the PE rule keeps the
primitive opaque in the residual jaxpr. The transpose rule runs a reverse sequential scan of N-1
iterations, applying the transposed body to propagate cotangents backward. The JVP rule uses
`jvpJaxprCache` to cache the differentiated body jaxpr, then runs the Kogge-Stone ladder with
doubled (primal+tangent) leaves — preserving O(log N) depth.

**Body tracing fallback:** If `makeJaxpr` fails to trace the body (e.g., einsum with
batch-dimension-dependent subscripts like `"nij,njk->nik"`), the function falls back to
`associativeScanCore` which unrolls Kogge-Stone rounds directly into the current trace. AD still
works through the unrolled operations.

**Jaxpr IR size:** With the primitive, the Jaxpr contains a single `AssociativeScan` equation
referencing the body sub-jaxpr — O(1) regardless of N. The body jaxpr itself is O(body_ops).
Previously (trace-through mode), the Jaxpr grew as O(body_ops × log N) equations.

**Why the primitive helps AD (compared to trace-through):**

- **Clean IR:** One equation instead of O(log N) unrolled rounds.
- **Backend specialization:** The JIT compiler recognizes `Primitive.AssociativeScan` and emits
  specialized WASM code (M7.2 compiled-loop, M7.3 multithreaded). WebGPU uses JS-driven dispatch
  (already optimal).
- **Consistent with `Primitive.Scan`:** Same pattern of body sub-jaxpr + transform rules.
- **Transpose rule:** The backward pass is a reverse sequential scan, not an unrolled graph.

**Measured `grad` runtime (Feb 2026):**

_WASM backend:_

| N    | `grad(assocScan)` | `grad(scan)` | Speedup    |
| ---- | ----------------- | ------------ | ---------- |
| 64   | 0.021 ms          | 0.097 ms     | 4.6×       |
| 256  | 0.035 ms          | 0.448 ms     | 12.9×      |
| 1024 | 0.027 ms          | 1.037 ms     | 38.6×      |
| 4096 | 0.025 ms          | 4.757 ms     | **187.7×** |

_WebGPU backend (Deno wgpu-rs, Intel Core Ultra 5 125H):_

| N    | `grad(assocScan)` | `grad(scan)` | Speedup   |
| ---- | ----------------- | ------------ | --------- |
| 64   | 0.058 ms          | 0.360 ms     | 6.2×      |
| 256  | 0.076 ms          | 1.020 ms     | 13.5×     |
| 1024 | 0.143 ms          | 3.831 ms     | 26.9×     |
| 4096 | 0.194 ms          | 15.148 ms    | **78.2×** |

`valueAndGrad` with matmul compose (the parallel Kalman filter pattern) scales correctly: 0.2 ms →
1.3 ms across N=64→1024 on WebGPU.

**Key insight:** `grad(associativeScan)` maintains O(log N) parallel depth on both backends. The
gradient computation benefits from the same Kogge-Stone structure as the forward pass — the backward
pass transposes O(log N) rounds of standard operations, producing O(log N) transposed rounds.

---

## API Contract

**Inputs NOT consumed:**

```ts
using xs = np.array([1, 2, 3, 4]);
const ys = lax.associativeScan((a, b) => np.add(a, b), xs);
// xs is still alive — xs was NOT consumed
ys.dispose();
```

**Caller owns the result:**

```ts
// For non-pytree inputs, use `using`:
using result = lax.associativeScan((a, b) => a.mul(b), xs);
// For pytree results, dispose leaves manually or use tree.makeDisposable:
using result2 = tree.makeDisposable(lax.associativeScan(compose, { a: aArr, b: bArr }));
```

**`fn` must dispose its own intermediates:**

```ts
// CORRECT — dispose the intermediate q.a.mul(p.b) before returning:
const compose = (p: { a: Array; b: Array }, q: { a: Array; b: Array }) => {
  const newA = p.a.mul(q.a) as Array;
  using tmp = q.a.mul(p.b) as Array; // auto-disposed at block end
  const newB = tmp.add(q.b) as Array;
  return { a: newA, b: newB };
};

// LEAKS — the intermediate q.a.mul(p.b) is never disposed:
const composeLeak = (p, q) => ({
  a: p.a.mul(q.a),
  b: q.a.mul(p.b).add(q.b), // q.a.mul(p.b) leaks!
});
```

**Common patterns:**

| Pattern              | Code                                                                 |
| -------------------- | -------------------------------------------------------------------- |
| Cumulative sum       | `lax.associativeScan((a, b) => np.add(a, b), xs)`                    |
| Cumulative product   | `lax.associativeScan((a, b) => a.mul(b), xs)`                        |
| Running maximum      | `lax.associativeScan((a, b) => np.maximum(a, b), xs)`                |
| Reverse suffix sum   | `lax.associativeScan((a, b) => np.add(a, b), xs, { reverse: true })` |
| Along axis 1         | `lax.associativeScan(fn, xs, { axis: 1 })`                           |
| Pytree (affine maps) | `lax.associativeScan(compose, { a: aArr, b: bArr })`                 |

---

## Difference from `lax.scan`

| Aspect                       | `lax.scan`                               | `lax.associativeScan`                                                                                                                             |
| ---------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Algorithm                    | Sequential recurrence with carry         | Parallel prefix (Kogge-Stone)                                                                                                                     |
| `fn` signature               | `(carry, x) => [newCarry, y]`            | `(a: T, b: T) => T`                                                                                                                               |
| `fn` can be non-associative? | Yes                                      | No — must be associative                                                                                                                          |
| Output                       | `[finalCarry, stackedYs]`                | Full prefix result (same shape as input)                                                                                                          |
| Complexity                   | O(N) sequential depth                    | O(N log N) total work, O(log N) depth                                                                                                             |
| WebGPU dispatch rounds       | N (fallback) or 1 (compiled-loop)        | ceil(log₂ N) JS-driven dispatches — already the hardware-imposed floor (no cross-workgroup global barrier on WebGPU; more if `fn` has reductions) |
| WASM/CPU dispatch rounds     | 1 compiled WASM invocation (entire loop) | 1 compiled WASM invocation (entire Kogge-Stone ladder, M7.2); CPU: ceil(log₂ N) JS round-trips                                                    |
| Reverse option               | ✅                                       | ✅                                                                                                                                                |
| Pytrees                      | ✅                                       | ✅                                                                                                                                                |
| `xs=null` / `Y=null`         | ✅                                       | N/A                                                                                                                                               |
| Carry state threading        | ✅                                       | N/A (output IS the prefix)                                                                                                                        |
| Autodiff                     | ✅ (dedicated JVP/transpose rules)       | ✅ (dedicated JVP/PE/transpose/vmap rules via `Primitive.AssociativeScan`)                                                                        |

**When to use which:**

- Use `lax.scan` for sequential recurrences where `fn` is not associative, or when you need carry
  state that differs from output (e.g., RNNs, Kalman filter with complex state).
- **On WebGPU:** use `lax.associativeScan` when `fn` is associative and N is large enough that
  `log₂(N) × dispatch_cost < N × per_iter_GPU_cost`. The crossover is around N≈1024 on typical
  hardware. Ideal for cumulative reductions, compositional transforms, and parallel Kalman filters.
  **Autodiff preserves O(log N) depth** — `grad(associativeScan)` is 78× faster than `grad(scan)` at
  N=4096 on WebGPU. For the RTS backward smoother specifically, see: Särkkä, S. & García-Fernández,
  Á. F. (2020). "Temporal Parallelization of Bayesian Smoothers." _IEEE Transactions on Automatic
  Control_. [arXiv:1905.13002](https://arxiv.org/abs/1905.13002)
- **On WASM/CPU:** With M7.2's compiled-loop, `lax.associativeScan` on WASM runs the entire
  Kogge-Stone ladder in a single JS→WASM call, eliminating per-round boundary overhead. Still does
  O(N log N) total work vs scan's O(N), but the zero-boundary-overhead makes it competitive for
  moderate N. For very large N with simple scalar bodies, `lax.scan`'s compiled-loop may still be
  faster due to O(N) vs O(N log N) work. Use `associativeScan` when the function is associative and
  you need autodiff (O(log N) depth gradient).

---

## Test Coverage

`test/lax-associative-scan.test.ts` — 27 tests:

| Category                    | Tests | Notes                                                                                                                                                                                             | File                                      |
| --------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 1-D basic                   | 6     | cumsum, cumprod, running max, N=1, N=8, N=7 (non-power-of-two)                                                                                                                                    | `test/lax-associative-scan.test.ts`       |
| Reverse                     | 2     | reverse cumsum, reverse cummax                                                                                                                                                                    | `test/lax-associative-scan.test.ts`       |
| Non-zero axis               | 2     | axis=1 and axis=0 on 2-D arrays                                                                                                                                                                   | `test/lax-associative-scan.test.ts`       |
| Pytree (affine composition) | 1     | Composition of arity-2 pytree with internal intermediate disposal                                                                                                                                 | `test/lax-associative-scan.test.ts`       |
| Autodiff                    | 4     | `grad`, `grad(jit)`, `jit(grad)` with 2-tuple, `jit(grad)` with 3-tuple compose                                                                                                                   | `test/lax-associative-scan.test.ts`       |
| Parallel Kalman filter      | 2     | Sequential vs parallel correctness (8 obs), differentiable wrt obs                                                                                                                                | `test/lax-associative-scan.test.ts`       |
| Vmap                        | 1     | `vmap(associativeScan)` batches independent scans                                                                                                                                                 | `test/lax-associative-scan.test.ts`       |
| Einsum fallback             | 1     | Body tracing fallback for einsum with batch-dependent subscripts                                                                                                                                  | `test/lax-associative-scan.test.ts`       |
| WASM compiled-loop          | 8     | cumsum, cumprod, reverse, N=1, non-power-of-two, 2-D, pytree affine, grad through compiled-loop                                                                                                   | `test/lax-associative-scan.test.ts`       |
| Deno WebGPU perf            | 1     | N=65536 prefix product: assocScan (16 JS-driven rounds, ceil(log₂ 65536)) must be ≥3× faster than scan (65536 sequential in-shader iterations on 1 GPU thread); measured ~5–8× on tested hardware | `test/deno/associative-scan-perf.test.ts` |
| Deno WASM parallel (M7.3)   | 5     | cumsum N=8192, cumprod N=4096, reverse cumsum, 2-D multi-element N=8192×4, repeated calls consistency — all via `WasmWorkerPool` parallel dispatch                                                | `test/deno/parallel-assoc-scan.test.ts`   |

**Kalman filter test:** Scalar constant-coefficient Kalman filter expressed as a prefix scan of
affine maps `x_t = A_t * x_{t-1} + b_t`. The composition rule
`compose(p, q) = (p.a*q.a, q.a*p.b + q.b)` is associative — applying p first then q. Sequential
reference and parallel scan results are compared to 5 decimal places.

---

## Future Work

| Priority | Feature                       | Notes                                                                                                                                                           |
| -------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| —        | ~~Multithreaded Kogge-Stone~~ | **Done (M7.3).** Parallel inner-round `kernel` export + `WasmWorkerPool` dispatch for N ≥ 4096. Per-j indexed internal buffers for thread safety. 5 Deno tests. |
| Low      | N=0 test                      | Verify empty-sequence edge case behavior matches JAX                                                                                                            |
| Low      | WebGL performance             | WebGL has no compiled-loop for scan (JS fallback), so assocScan's O(log N) shader dispatches may already beat scan's N dispatches. Needs measurement.           |

---

# Part 6: Linear Algebra Autodiff (`solve`, `inv`)

## Overview & Motivation

`numpy.linalg.solve(A, b)` and `numpy.linalg.inv(A)` compute $A^{-1} b$ and $A^{-1}$ respectively,
using LU decomposition internally. Getting correct gradients through these operations required
finding and fixing a bug in the `Primitive.TriangularSolve` JVP rule.

**Key files:**

| File                          | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| `src/library/numpy-linalg.ts` | `solve`, `inv`, `luSolveWithP` helper       |
| `src/frontend/jvp.ts`         | JVP rules for TriangularSolve, Cholesky, LU |
| `src/frontend/linearize.ts`   | Transpose rules for TriangularSolve         |
| `src/frontend/core.ts`        | `stopGradient`, `Primitive.StopGradient`    |
| `test/numpy-linalg.test.ts`   | Gradient tests for inv, det, lstsq          |

---

## Root Cause: TriangularSolve JVP Triangle Mask Bug

### The bug

The `Primitive.TriangularSolve` JVP rule differentiates `A @ X^T = B^T` where `A` is upper
triangular. Only the upper triangle of `A` affects the solution `X`. The JVP must therefore only
propagate tangents from `triu(dA)` — perturbations below the diagonal have zero effect.

**Before fix (buggy):**

```ts
[Primitive.TriangularSolve]([a, b], [da, db], { unitDiagonal }) {
    const x = triangularSolve(a, b, { unitDiagonal });
    using dax = batchMatmulT(da, x);      // dA @ X^T — uses ALL of dA ←── BUG
    using mTdax = mT(dax);
    using rhsT = db.sub(mTdax);
    const dx = triangularSolve(a, rhsT, { unitDiagonal });
    return [[x], [dx]];
}
```

**After fix:**

```ts
[Primitive.TriangularSolve]([a, b], [da, db], { unitDiagonal }) {
    const x = triangularSolve(a, b, { unitDiagonal });
    using maskedDa = (unitDiagonal ? triu(da, 1) : triu(da)) as Tracer;  // ←── FIX
    using dax = batchMatmulT(maskedDa, x);  // triu(dA) @ X^T
    using mTdax = mT(dax);
    using rhsT = db.sub(mTdax);
    const dx = triangularSolve(a, rhsT, { unitDiagonal });
    return [[x], [dx]];
}
```

### Why the missing mask causes wrong gradients through `solve`

In `solve(A, b)`, the packed LU matrix stores L in the lower triangle and U in the upper triangle.
When `triangularSolve(luMatrix, ...)` is called:

- **Lower solve (L):** `core.triangularSolve(luMatrix, ..., {lower: true})` flips the matrix so L's
  lower triangle becomes the upper triangle of the flipped matrix. The primitive reads this upper
  triangle. With the fix, `triu(d_flipped)` correctly masks to only L's entries.
- **Upper solve (U):** `core.triangularSolve(luMatrix, ..., {lower: false})` passes the packed
  matrix directly. The primitive reads the upper triangle (U). Without the mask, `dA @ X^T` uses the
  L entries in the lower triangle — gradient leaks from L's entries into U's solve.

This gradient leak caused `grad(solve)` and `grad(inv)` to produce incorrect results (errors of
0.15–0.44 in 2×2–3×3 tests, not a precision issue).

### What was NOT the bug

- **LU JVP rule:** Verified correct both mathematically and empirically. Forward-mode tangents match
  finite differences to f32 precision (~0.0016). The LU JVP uses `triSolve(L, ...)` and
  `triSolve(U^T, ...)` where L and U are known primals — `da=0` at those calls, so the triangle mask
  bug doesn't fire.
- **TriangularSolve transpose rule** (for b input): Correct — tested independently (err < 0.001).
- **`tril`/`triu` transpose rules:** Self-adjoint, verified correct.
- **Dot transpose rule:** Correct — verified by tracing through `batchMatmulT` composition.

### Impact matrix

| Operation                              | Before fix               | After fix  |
| -------------------------------------- | ------------------------ | ---------- |
| `grad(triSolve(A, b))` w.r.t. A        | ❌ Wrong (gradient leak) | ✅ Correct |
| `grad(triSolve(A, b))` w.r.t. b        | ✅ Correct               | ✅ Correct |
| `grad(lu(A))`                          | ✅ Correct               | ✅ Correct |
| `grad(solve(A, b))` via Newton         | ✅ Correct (workaround)  | ✅ Correct |
| `grad(solve(A, b))` direct LU→triSolve | ❌ Wrong                 | ✅ Correct |
| `grad(inv(A))` analytical comparison   | ✅ via Newton            | ✅ Direct  |

### Verification results

After the fix, direct LU→triSolve gradients match the Newton refinement to machine precision:

| Test                         | Direct vs Newton | Direct vs FD | Direct vs Analytical |
| ---------------------------- | ---------------- | ------------ | -------------------- |
| 2×2 `grad(sum(solve(A, I)))` | 1e-8             | 1.8e-4       | —                    |
| 3×3 with pivoting            | 6e-8             | —            | —                    |
| 2×2 `grad(sum(inv(A)))`      | —                | —            | 0 (exact)            |

---

## Current Design: Direct LU→TriSolve Gradient Path

With the TriSolve JVP `triu(dA)` mask fix in place, `solve` differentiates directly through
`lu → triangularSolve` — no Newton refinement or `stopGradient` workarounds needed.

**Key ownership constraint:** Stop gradient only on `permRaw` (integer-valued, no gradient). Do NOT
stop gradient on `a` itself or on `luRaw` — `stopGradient(a)` would create a fully-known PETracer
that PE disposal cascades through, freeing `a` early.

```ts
// Factor A. Gradient flows through LU JVP (TriSolve triu mask ensures correctness).
// Stop gradient only on permRaw — permutation is integer-valued, no gradient.
const [lu, pivotsRaw, permRaw] = lax.linalg.lu(a);
const permutation = lax.stopGradient(permRaw);

// Solve directly — both ∂L/∂b and ∂L/∂A flow through correctly.
const x = luSolveWithP(lu, P, b, d);
```

**Why this works:**

- **∂L/∂b:** TriangularSolve transpose rule gives `ctB = A^{-T} ct` — correct.
- **∂L/∂A:** TriangularSolve JVP (now with `triu(dA)` mask) gives `dx = A^{-1} (db - triu(dA) X)` —
  the mask prevents gradient leaking from below-diagonal entries. Verified to match the analytical
  formula $-A^{-T} g x^T$ to machine precision.

**Comparison with JAX:**

JAX uses `custom_linear_solve` — a dedicated primitive that encodes the implicit function theorem.
The gradient never flows through LU. This is heavier: new primitive + 4 transform rules. By fixing
the TriSolve JVP, jax-js achieves the same correctness with zero extra infrastructure.

| Aspect               | JAX (`custom_linear_solve`)       | jax-js (direct LU→triSolve)          |
| -------------------- | --------------------------------- | ------------------------------------ |
| Forward cost         | 2 triangular solves               | 2 triangular solves (same)           |
| Backward cost        | 2 triSolves + 1 matmul            | Auto-transposed (comparable)         |
| Implementation cost  | New primitive + 4 transform rules | Zero new infrastructure              |
| Gradient correctness | Exact (hand-written)              | Exact (automatic from fixed JVP)     |
| Composability        | Manual rule for each transform    | Free (`jit`, `grad`, `vmap` compose) |
| Maintenance          | Custom primitive to maintain      | Nothing extra — standard AD pipeline |

### Why `inv` is just `solve(A, I)`

With `solve` handling gradients correctly, `inv` is simply:

```ts
export function inv(a: ArrayLike): Array {
  const n = checkSquare("inv", a);
  using eye = np.eye(n, { dtype: a.dtype });
  return solve(a, eye);
}
```

The gradient `∂L/∂A` for `inv(A)` reduces to $-A^{-T} G A^{-T}$ (where $G$ is the cotangent), which
falls out of `solve`'s gradient formula automatically.

---

## `luSolveWithP` helper

Shared helper that applies a pre-computed LU factorization with permutation matrix:

```ts
function luSolveWithP(lu, P, b, d) {
  const Pb = np.matmul(P, b); // Apply row permutation
  const LPb = triangularSolve(lu, Pb, { lower: true, unitDiagonal: true }); // L⁻¹ P b
  return triangularSolve(lu, LPb, { lower: false }); // U⁻¹ L⁻¹ P b
}
```

Used once in `solve` for the primary solve. Intermediates are pushed to the caller's `d[]` disposal
tracker.

---

## Test Coverage

| Test                                         | File                        | What it verifies                    |
| -------------------------------------------- | --------------------------- | ----------------------------------- |
| `gradient of sum(inv(A)) matches analytical` | `test/numpy-linalg.test.ts` | inv gradient vs $-A^{-T} G A^{-T}$  |
| `gradient of det is adjugate.mT`             | `test/numpy-linalg.test.ts` | det gradient (exercises LU forward) |
| `solves random batched AX = B`               | `test/numpy-linalg.test.ts` | Batched solve correctness           |
| `works with grad on b (underdetermined)`     | `test/numpy-linalg.test.ts` | lstsq gradient (Cholesky path)      |

---

## ~~TODO~~ DONE: Newton Refinement Removed (Direct LU→TriSolve Path)

Direct LU→triSolve gradient path is now the only implementation. Newton refinement and
`stopGradient(luRaw)` have been removed. See
[Current Design](#current-design-direct-lu-trisolve-gradient-path).

**What was removed and why it's no longer needed:**

### Primitive convention reference

The `Primitive.TriangularSolve` convention (important for understanding the JVP):

- `A @ X^T = B^T` where A is **upper triangular**
- Returns $X = B @ A^{-T}$ (NOT $A^{-1} @ B$)
- `core.triangularSolve(a, b, {lower: true})` converts to upper by flipping both axes of `a` and
  last axis of `b`, solves, flips result back
- `lax.linalg.triangularSolve(A, b, {leftSide: true})` transforms to right-side convention by
  transposing both arguments and flipping `lower`

---

# Part 7: Polymorphic Shapes (M4 — Dynamic Dimensions)

## Overview

Polymorphic shapes enable a single JIT-compiled `JitProgram` to be reused across different input
sizes on a specified axis (e.g., sequence length `T`). Without this, each new input shape triggers a
full re-trace + re-compile. With polymorphic shapes, the program is traced once with symbolic
dimensions and compiled to code parameterized by those dimensions.

**Key design decision: concrete compilation + symbolic caching.** The program is compiled for a
_concrete_ instantiation of the symbolic dimensions (using the first call's actual sizes), but the
compiled artifact is cached under a _symbolic_ key. Subsequent calls with different sizes on the
dynamic axis hit the cache if they share the same symbolic structure — they just pass different
concrete dimension values at execution time.

## Key Types (`src/dim.ts`)

| Type           | Purpose                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `SymDim`       | A named symbolic dimension variable (e.g., `SymDim("T")`). Propagated through shape inference during tracing.            |
| `Dim`          | `number \| SymDim` — a dimension is either concrete or symbolic.                                                         |
| `SymbolicSize` | A symbolic size expression: `factor × prod(syms)`. E.g., `SymbolicSize(4, ["T"])` = `T × 4`. Resolves with dim bindings. |
| `SizeExpr`     | `number \| SymbolicSize` — used for `Kernel.size`, `KernelOutput.bytes`, `JitStep.malloc.size`.                          |

Key utility functions: `isSymbolicDim()`, `hasSymbolicDims()`, `concreteDim()`, `concreteShape()`,
`dimEquals()`, `dimCompatible()`, `resolveShape()`, `isSymbolicSize()`, `resolveSizeExpr()`,
`sizeExprKey()`, `dimProduct()`, `sizeExprMul()`.

## How It Works

### Tracing (`makeJaxpr` with `dynamic_axes`)

```ts
const jaxpr = makeJaxpr(f, [ShapedArray([3, SymDim("T")], DType.Float32)], {
  dynamic_axes: { T: 0 }, // axis 1 is symbolic
});
```

During tracing, `SymDim("T")` propagates through all shape operations. Operations that depend on the
symbolic dimension get `SymbolicSize` kernel sizes instead of concrete numbers.

### Compilation (`_currentDimBindings` in `jit.ts`)

`_currentDimBindings` is a module-level variable set during `jitCompile()`:

1. **Set on entry**: `_currentDimBindings = dimBindings` — maps symbolic names to concrete values
   for the current call (e.g., `{ T: 128 }`).
2. **Used during compilation**: `resolveShape()` converts symbolic `Dim[]` to concrete `number[]`
   for `ShapeTracker` operations (strides, offsets, unravel indices). `setConcreteHint()` resolves
   `SizeExpr` to a concrete `concreteSizeHint` on each `Kernel` for expression simplification
   (modulo elimination).
3. **Cleared in finally**: Prevents stale state from leaking across compilations.

### Execution (`dynamicParams`)

At execution time, resolved symbolic dimensions are passed to the compiled code:

- **WASM**: As extra `i32` function parameters (planned; currently uses concrete compilation)
- **WebGPU**: Via uniform buffer containing resolved dimension values (planned; `isSymbolicSize`
  guards exist in codegen)

### Cache Key Strategy

The JIT cache uses `sizeExprKey()` for symbolic sizes — e.g., `"T*4"` instead of `"512"`. This means
a program compiled for `T=128` is reused when called with `T=256` (same symbolic structure,
different concrete value). The `Kernel.concreteSizeHint` is re-resolved from `_currentDimBindings`
on each compilation, but the cache key matches on the symbolic expression.

## Key Files

| File                              | Purpose                                                     |
| --------------------------------- | ----------------------------------------------------------- |
| `src/dim.ts`                      | `SymDim`, `Dim`, `SymbolicSize`, `SizeExpr` types + helpers |
| `src/frontend/jit.ts`             | `_currentDimBindings`, `setConcreteHint`                    |
| `src/frontend/jaxpr.ts`           | `dynamic_axes` option in `makeJaxpr()`                      |
| `test/polymorphic-shapes.test.ts` | Test coverage for symbolic shape tracing + compilation      |

## Current Limitations

- Mega-module rejects kernels with symbolic reduction sizes (cannot inline i32.const loop bounds)
- Mega-module rejects symbolic malloc sizes (cannot resolve at compile time)
- WebGPU scan compiled-loop does not support symbolic length (falls back to JS loop)
- WebGPU preencoded-routine scan does not support symbolic length

## Polymorphic Scan Length

WASM `lax.scan` compiled-loop supports **polymorphic iteration count** via `dynamic_axes`. When the
scan length derives from a symbolic dimension (e.g., `xs` has shape `[SymDim("T"), n]`), the WASM
module receives length as arg 0 and uses `local.get(lengthArg)` instead of `i32.const(length)`. This
allows a single compiled module to run scans of any length:

```ts
// Single compilation, reused for T=100 and T=200
const f = jit(
  (xs) => {
    const init = np.zeros([4]);
    return lax.scan((c, x) => [c.add(x), c.add(x)], init, xs);
  },
  { dynamic_axes: { T: 0 } },
);
const [carry4, ys4] = f(np.ones([4, 4])); // T=4
const [carry6, ys6] = f(np.ones([6, 4])); // T=6, same compiled module
```

**dimBindings resolution:** At JIT execution time, `Primitive.Jit` impl resolves `dynamic_axes` from
user input shapes. The resolution skips captured constants by using `args[numConsts]` instead of
`args[0]` — a bug fix that was required because `args` is `[...jaxpr.consts, ...userInputs]`.

**Key helper — `resolveDim(d, bindings)`:** Resolves a `Dim` to concrete `number` using dimension
bindings. Returns the number directly if already concrete; looks up `SymDim.name` in the bindings
map otherwise. Used at scan step execution time in `jit.ts` to resolve `step.length`.

---

# Part 8: ULTIMATE-ARCHITECTURE-PLAN Progress

## Plan Location

`ULTIMATE-ARCHITECTURE-PLAN.md` in the repo root. Contains detailed specifications for milestones
M0–M8 with dependency graph, code sketches, and test plans.

## Milestone Status

| Milestone | Title                         | Status   | Notes                                                                      |
| --------- | ----------------------------- | -------- | -------------------------------------------------------------------------- |
| M0.1      | Record baseline tests         | **DONE** | `tmp/m0-*` baseline files captured                                         |
| M0.2      | Hardware feature detection    | **DONE** | `BackendCapabilities` interface in `src/backend.ts`                        |
| M1.1      | `ScanBackwardArtifact` type   | **DONE** | `ScanPullbackArtifact.run()` encapsulates backward pass                    |
| M1.2      | Unify `vjpFlat` transposition | **DONE** | `jaxprNeedsCallTimeTranspose` fully removed                                |
| M2.1      | `scatter_add` IR & AD rules   | **DONE** | `Primitive.ScatterAdd` with JVP + transpose rules                          |
| M2.2      | WebGPU CAS loop shader        | **DONE** | `dispatchScatterAdd()` in `webgpu.ts`                                      |
| M2.3      | WASM sequential scatter       | **DONE** | `dispatchScatterAdd()` in `wasm.ts`                                        |
| M3.1      | Multi-output `Kernel`         | **DONE** | `KernelOutput[]`, `Kernel.single()`, multi-output codegen                  |
| M3.2      | Epilogue fusion chain walk    | **DONE** | Already implemented; verified via `stepCounts()` tests                     |
| M4.1      | `SymDim` & shape propagation  | **DONE** | `SymDim`, `Dim`, `dynamic_axes` in `makeJaxpr()`                           |
| M4.2      | Parameterized backend codegen | **DONE** | Symbolic reduction sizes, `dynamicParams` layout, mega-module rejection    |
| M5.1      | SharedArrayBuffer memory pool | **DONE** | Shared memory when SAB constructable (Deno native, browser COOP/COEP)      |
| M5.2      | WasmWorkerPool                | **DONE** | Atomics-based sync dispatch via Web Workers                                |
| M5.3      | Kernel signature + dispatch   | **DONE** | `(start, end, ...ptrs)` + parallel dispatch wiring                         |
| M6.1      | Mega-Module                   | **DONE** | `compileToMegaModule()`, single WASM call, 16 tests                        |
| M6.2a     | Extract kernel functions      | **DONE** | Extracted WASM functions per kernel, 10 tests                              |
| M6.2b     | Orchestrator worker           | **DONE** | Off-main-thread mega-module via Web Worker, 12 Deno tests                  |
| M6.2c     | Parallel kernel dispatch      | **DONE** | JS-driven step execution, workers dispatch large kernels, 5 Deno tests     |
| M7.1      | `Primitive.AssociativeScan`   | **DONE** | Body sub-jaxpr, JVP/PE/transpose/vmap rules, 19 tests                      |
| M7.2      | WASM compiled Kogge-Stone     | **DONE** | `codegenNativeAssociativeScan()`, polymorphic N, 8 tests                   |
| M7.3      | Multithreaded Kogge-Stone     | **DONE** | Parallel `kernel` export + `WasmWorkerPool`, per-j internals, 5 Deno tests |
| M8        | Cleanup & benchmarking        | **DONE** | M8.1 benchmarks ✅, M8.2 dead code audit ✅, M8.3 final regression ✅      |

## Dependency Graph (Simplified)

```
M0 ✅ ──┬──→ M1 (scan backward AOT) ✅
        ├──→ M2 (scatter_add) ✅
        ├──→ M3.1 (multi-output kernel) ✅ ──→ M3.2 (epilogue fusion) ✅
        ├──→ M4.1 ✅ ──→ M4.2 ✅ ──→ M6.1 ✅
        └──→ M5 ✅ ──→ M6.2a ✅ ──→ M6.2b ✅ ──→ M6.2c ✅
                       M7.1 ✅ ──→ M7.2 ✅ ──→ M7.3 ✅
                                                  ↓
                                            M8 (cleanup) ✅
```

## Next Available Milestones

All milestones M0–M8 are complete. The ULTIMATE-ARCHITECTURE-PLAN is fully implemented.

---

# Part 9: Session Continuity Notes for AI Agents

## Before Starting Work

1. **Build first**: Run `pnpm build` before running tests — Vitest imports from `dist/`, not `src/`.
2. **Check branch**: `git branch` to confirm you're on the right branch (currently
   `docs/ultimate-architecture-plan`).
3. **Read the plan**: `ULTIMATE-ARCHITECTURE-PLAN.md` contains detailed specifications for each
   milestone including code sketches, test plans, and acceptance criteria.
4. **Check git log**: `git log --oneline -10` to see recent commits and understand context.

## Key Implementation Patterns

### Adding new primitives

Follow the checklist in Part 1 → "Adding a new primitive" and "Adding a new routine". Key: add to
`Primitive` enum, impl rules, JVP rules, transpose rules, export from `index.ts`.

### JIT compilation flow

`makeJaxpr` → `jaxpr.flatten().simplify()` → `splitGraphDataflow()` → `jitCompile()` →
`JitProgram.execute()`. The `splitGraphDataflow` P2 pass is the most subtle — see the
`isNonKernelBlack` distinction in common pitfalls.

### Ownership debugging

If a `UseAfterFreeError` appears, check: artifact disposal timing, `transposeJaxprCache`
(cache-owned, don't dispose), `getOrMakeConstTracer` `.ref` balance, `evalJaxprTransposed`
`argPrimals` set.

### Test workflow

```bash
pnpm build                    # Required before tests
pnpm vitest run               # Full test suite (imports from dist/)
pnpm run test:eslint-plugin   # ESLint plugin tests
pnpm lint                     # Lint check
pnpm check                    # TypeScript type check
```

## What Gets Lost at Summarization Boundaries

These details are frequently lost when conversation context is summarized:

1. **The M-numbering discrepancy**: Early git commits are labeled M0–M4 but refer to foundational
   work, NOT the ULTIMATE-ARCHITECTURE-PLAN milestones. The plan's milestones are a separate
   numbering system. Always check `ULTIMATE-ARCHITECTURE-PLAN.md` for the canonical milestone
   definitions.

2. **Test imports from `dist/`**: Source edits in `src/` are invisible to Vitest tests until
   `pnpm build` runs. This causes confusion when an edit "doesn't work" in tests.

3. **eslint.config.ts structure**: All jax-js rules are `warn` globally; the invariance overlay
   upgrades to `error` for `src/**`, `packages/**`, `test/**`. Internal transform rules
   (`require-retained-release`, `require-try-finally-symmetry`, `require-wrapper-dispose-symmetry`)
   are now in the main config — the separate `lint:ownership:internal` script is no longer needed
   for pre-commit (but is kept for targeted runs).

4. **`_currentDimBindings` is module-level state**: Set/cleared in `jitCompile()` via try/finally.
   Any new compilation code that reads symbolic dimensions must go through this.

5. **Multi-output kernel access pattern**: `Kernel` has no single-output shims. All call sites
   access `kernel.outputs[0].exp`, `kernel.outputs[0].reduction`, `kernel.outputs[0].dtype`,
   `kernel.outputs[0].bytes` explicitly. Multi-output paths iterate `kernel.outputs`.

6. **The `no-array-chain` rule is NOT in the `invariance` config** — only in `strict`. The
   `invariance` config focuses on ownership correctness (eager/JIT equivalence), not performance
   patterns.

## Architecture Decisions Log

Decisions made during development that future agents should understand:

| Decision                                        | Rationale                                                                                                                                                                          |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Non-consuming ownership model                   | Eliminates `UseAfterFreeError` from `.ref` mistakes; trades for silent leaks + linting                                                                                             |
| Concrete compilation + symbolic cache           | Simpler than full symbolic IR; ShapeTracker needs concrete strides                                                                                                                 |
| `effectDrivenAllocate` over two-pass            | Single-pass liveness is cleaner; DUS zero-copy falls out naturally from `Mutate` effect                                                                                            |
| Direct LU→triSolve gradient path                | Fixing TriSolve JVP `triu(dA)` mask made Newton refinement unnecessary                                                                                                             |
| `transposeJaxprCache` is cache-owned            | Prevents repeated transposition; callers must NOT dispose returned `ClosedJaxpr`                                                                                                   |
| `invariance` ≠ `strict` ESLint config           | `invariance` = ownership correctness; `strict` adds `no-array-chain` for peak memory                                                                                               |
| WASM `(start, end, ...ptrs)` signature          | Enables work-splitting for `WasmWorkerPool`; `RANGE_PARAMS=2` prefix in all kernel codegen                                                                                         |
| WASM compiled Kogge-Stone (assocScan)           | N as runtime i32 enables polymorphic length; ping-pong by caller, not inside module; `AssocScanPlan` mirrors `ScanPlan`                                                            |
| M7.3 per-j indexed internal buffers             | Each worker sees `internal[idx] + j * internalSizes[idx]` — non-overlapping regions prevent races. Parallel path allocates `internalSize * N`, monolithic allocates `internalSize` |
| WASM compiled scan polymorphic length           | Length as runtime i32 param (arg 0); `NativeScanGeneralParams` no longer contains `length`; dispatch prepends length to args; `dimBindings` resolution uses `args[numConsts]`      |
| Mega-module rejects pass-through                | Steps (free, recycle) can overwrite input WASM locals; conservatively bail to step-by-step rather than tracking writes                                                             |
| `scatterAdd` not in public API                  | Not exported from `src/index.ts`; bench/test import from `src/frontend/core` directly. Add to public API when stable                                                               |
| M6.2 extracted-functions design                 | V8 inlines direct `call` at runtime → extracting kernels into separate WASM functions is perf-neutral serial, enables parallel. Resolves monolithic-vs-parallelizable tension.     |
| Module Workers (`type: "module"`)               | Deno doesn't support classic blob-URL workers; module workers work in both Deno and Chromium. Applied to both `worker-pool.ts` and `orchestrator.ts`.                              |
| SAB constructability over `crossOriginIsolated` | `try { new SharedArrayBuffer(1) }` works in Deno (native SAB) and browsers (with COOP/COEP); `crossOriginIsolated` is browser-only and false in Deno.                              |
| Vitest SAB tests skipped, Deno covers           | COOP+COEP headers break Vitest's iframe runner. Orchestrator/worker pool features tested exclusively via Deno. 8 Vitest tests skip, 17 Deno tests cover.                           |
| M7.3 lazy registration + monolithic fallback    | First call with N ≥ PARALLEL_THRESHOLD (4096): `pool.registerModule()` async + monolithic fallback. Subsequent: `pool.isModuleReady()` → parallel. Same pattern as mega-module.    |
