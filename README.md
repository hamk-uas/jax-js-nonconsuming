<h1 align="center">jax-js: JAX in pure JavaScript</h1>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/hamk-uas/jax-js-nonconsuming/main/website/src/lib/assets/logo-nonconsuming.svg"
    alt="jax-js-nonconsuming logo"
    width="200"
  />
</p>

<p align="center"><em>Non-consuming ownership fork</em></p>

<p align="center"><strong>
  <a href="https://hamk-uas.github.io/jax-js-nonconsuming/">Website</a> |
  <a href="https://hamk-uas.github.io/jax-js-nonconsuming/docs/">API Reference</a> |
  <a href="./FEATURES.md">Compatibility Table</a> |
  <a href="https://discord.gg/BW6YsCd4Tf">jax-js Discord</a>
</strong></p>

> **Fork notice:** This is a fork of [ekzhang/jax-js](https://github.com/ekzhang/jax-js) with a
> **non-consuming ownership model**. Operations leave inputs alive (no `.ref` needed), and `using`
> declarations provide deterministic GPU/WASM memory cleanup.
>
> **Why this fork?** The original jax-js uses move semantics, where operations consume their inputs.
> This fork was created for teams familiar with MATLAB or Python (NumPy) where move semantics are
> unexpected. We also fast-tracked a `lax.scan` implementation. The tradeoff is that leaks are
> silent instead of crashing — see [Tradeoffs](#tradeoffs-of-the-non-consuming-model) for an honest
> comparison.
>
> See [Differences from upstream](#differences-from-upstream) for a full comparison between the
> original and this fork.

**jax-js** is a machine learning framework for the browser. It aims to bring
[JAX](https://jax.dev)-style, high-performance CPU and GPU kernels to JavaScript, so you can run
numerical applications on the web.

```bash
npm install github:hamk-uas/jax-js-nonconsuming
```

To pin a specific release tag (once available):

```bash
npm install github:hamk-uas/jax-js-nonconsuming#v0.2.0
```

Under the hood, it translates array operations into a compiler representation, then synthesizes
kernels in WebAssembly and WebGPU.

The original jax-js was written from scratch with zero zero external dependencies. jax-js and this
fork maintain close API compatibility with NumPy/JAX. Since everything runs client-side, jax-js is
likely the most portable GPU ML framework, since it runs anywhere a browser can run.

## Quickstart

```js
import { numpy as np } from "@jax-js-nonconsuming/jax";

// Array operations, compatible with JAX/NumPy.
{
  using x = np.array([1, 2, 3]);
  using y = x.mul(4); // [4, 8, 12]
}
```

## Demos

- [Training neural networks on MNIST](https://hamk-uas.github.io/jax-js-nonconsuming/mnist)
- [Voice cloning: Kyutai Pocket TTS](https://hamk-uas.github.io/jax-js-nonconsuming/tts)
- [CLIP embeddings for books in-browser](https://hamk-uas.github.io/jax-js-nonconsuming/mobileclip)
- [Object detection: DETR ResNet-50 (ONNX)](https://hamk-uas.github.io/jax-js-nonconsuming/detr-resnet-50)
- [In-browser REPL](https://hamk-uas.github.io/jax-js-nonconsuming/repl)
- [Matmul benchmark](https://hamk-uas.github.io/jax-js-nonconsuming/bench/matmul)
- [Conv2d benchmark](https://hamk-uas.github.io/jax-js-nonconsuming/bench/conv2d)
- [Mandelbrot set](https://hamk-uas.github.io/jax-js-nonconsuming/mandelbrot)

## Feature comparison

Here's a quick, high-level comparison with other popular web ML runtimes:

| Feature                         | jax-js-nonconsuming | jax-js v0.1.9 | TensorFlow.js   | onnxruntime-web    |
| ------------------------------- | ------------------- | ------------- | --------------- | ------------------ |
| **Overview**                    |                     |               |                 |                    |
| API style                       | JAX/NumPy           | JAX/NumPy     | TensorFlow-like | Static ONNX graphs |
| Latest release                  | 2026                | 2026          | ⚠️ 2024         | 2026               |
| Speed                           | Fastest             | Fastest       | Fast            | Fastest            |
| Bundle size (gzip)              | 107 KB              | 80 KB         | 269 KB          | 90 KB + 24 MB Wasm |
| **Autodiff & JIT**              |                     |               |                 |                    |
| Gradients                       | ✅                  | ✅            | ✅              | ❌                 |
| Jacobian and Hessian            | ✅                  | ✅            | ❌              | ❌                 |
| `jvp()` forward differentiation | ✅                  | ✅            | ❌              | ❌                 |
| `jit()` kernel fusion           | ✅                  | ✅            | ❌              | ❌                 |
| `vmap()` auto-vectorization     | ✅                  | ✅            | ❌              | ❌                 |
| `scan()` scan over leading axis | ✅                  | ❌            | ❌              | ❌                 |
| Graph capture                   | ✅                  | ✅            | ❌              | ✅                 |
| **Backends & Data**             |                     |               |                 |                    |
| WebGPU backend                  | ✅                  | ✅            | 🟡 Preview      | ✅                 |
| WebGL backend                   | ✅                  | ✅            | ✅              | ✅                 |
| Wasm (CPU) backend              | ✅                  | ✅            | ✅              | ✅                 |
| Eager array API                 | ✅                  | ✅            | ✅              | ❌                 |
| Run ONNX models                 | 🟡 Partial          | 🟡 Partial    | ❌              | ✅                 |
| Read safetensors                | ✅                  | ✅            | ❌              | ❌                 |
| Float64                         | ✅                  | ✅            | ❌              | ❌                 |
| Float32                         | ✅                  | ✅            | ✅              | ✅                 |
| Float16                         | ✅                  | ✅            | ❌              | ✅                 |
| BFloat16                        | ❌                  | ❌            | ❌              | ❌                 |
| Packed Uint8                    | ❌                  | ❌            | ❌              | 🟡 Partial         |
| Mixed precision                 | ✅                  | ✅            | ❌              | ✅                 |
| Mixed devices                   | ✅                  | ✅            | ❌              | ❌                 |
| **Ops & Numerics**              |                     |               |                 |                    |
| Arithmetic functions            | ✅                  | ✅            | ✅              | ✅                 |
| Matrix multiplication           | ✅                  | ✅            | ✅              | ✅                 |
| General einsum                  | ✅                  | ✅            | 🟡 Partial      | 🟡 Partial         |
| Sorting                         | ✅                  | ✅            | ❌              | ❌                 |
| Activation functions            | ✅                  | ✅            | ✅              | ✅                 |
| NaN/Inf numerics                | ✅                  | ✅            | ✅              | ✅                 |
| Basic convolutions              | ✅                  | ✅            | ✅              | ✅                 |
| n-d convolutions                | ✅                  | ✅            | ❌              | ✅                 |
| Strided/dilated convolution     | ✅                  | ✅            | ✅              | ✅                 |
| Cholesky, Lstsq                 | ✅                  | ✅            | ❌              | ❌                 |
| LU, Solve, Determinant          | ✅                  | ✅            | ❌              | ❌                 |
| SVD                             | ❌                  | ❌            | ❌              | ❌                 |
| FFT                             | ✅                  | ✅            | ✅              | ✅                 |
| Basic RNG (Uniform, Normal)     | ✅                  | ✅            | ✅              | ✅                 |
| Advanced RNG                    | ✅                  | ✅            | ❌              | ❌                 |

## Tutorial

Programming in `jax-js` looks [very similar to JAX](https://docs.jax.dev/en/latest/jax-101.html),
just in JavaScript.

### Arrays

Create an array with `np.array()`:

```ts
import { numpy as np } from "@jax-js-nonconsuming/jax";

using ar = np.array([1, 2, 3]);
```

By default, this is a float32 array, but you can specify a different dtype:

```ts
using ar = np.array([1, 2, 3], { dtype: np.int32 });
```

For more efficient construction, create an array from a JS `TypedArray` buffer:

```ts
const buf = new Float32Array([10, 20, 30, 100, 200, 300]);
using ar = np.array(buf).reshape([2, 3]);
```

Once you're done with it, you can unwrap a `jax.Array` back into JavaScript. This will also apply
any pending operations or lazy updates:

```ts
// 1) Returns a possibly nested JavaScript array.
ar.js();
await ar.jsAsync(); // Faster, non-blocking

// 2) Returns a flat TypedArray data buffer.
ar.dataSync();
await ar.data(); // Fastest, non-blocking

// If you don't need the array after reading data:
const floats = await ar.consumeData();
```

Arrays can have mathematical operations applied to them. For example:

```ts
import { numpy as np, scipySpecial as special } from "@jax-js-nonconsuming/jax";

using x = np.arange(100).astype(np.float32); // array of integers [0..99]

using y1 = x.add(x); // x + x
using y2 = np.sin(x); // sin(x)
using tanhX = np.tanh(x);
using y3 = tanhX.mul(5); // 5 * tanh(x)
using y4 = special.erfc(x); // erfc(x)
```

### Memory management

Big Arrays take up a lot of memory. Python ML libraries override the `__del__()` method to free
memory, but JavaScript has no such API for running object destructors
([cf.](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry)).

In jax-js-nonconsuming, operations **do not consume** their inputs — you can freely reuse an array
in multiple expressions without any special syntax. When you're done with an array, call
`.dispose()` to free its memory, or use JavaScript's `using` keyword for automatic disposal:

```ts
{
  using x = np.array([1, 2, 3]);
  using doubled = x.add(x);
  const y = doubled.mul(x); // x used three times — no problem
  y.dispose();
  // x and doubled are automatically disposed at end of block
}
```

For best performance, wrap compute-heavy code in `jit()`. The JIT compiler automatically manages
intermediate buffers — allocating, reusing, and freeing them at the optimal points:

```ts
const f = jit((x: np.Array) => {
  using sq = x.mul(x);
  using s = sq.sum();
  return np.sqrt(s);
});
const result = f(x); // intermediates freed automatically inside jit
result.dispose(); // caller disposes the output when done
f.dispose(); // free captured constants when the function is no longer needed
```

The `@jax-js-nonconsuming/eslint-plugin-jax-js` catches the most common memory leaks (missing
`using`, use-after-dispose, unnecessary `.ref`) at edit time — see the
[plugin README](packages/eslint-plugin) for setup.

### grad(), vmap() and jit()

JAX's signature composable transformations are also supported in jax-js. Here is a simple example of
using `grad` and `vmap` to compute the derivative of a function:

```ts
import { numpy as np, grad, vmap } from "@jax-js-nonconsuming/jax";

using x = np.linspace(-10, 10, 1000);

using y1 = vmap(grad(np.sin))(x); // d/dx sin(x) = cos(x)
using y2 = np.cos(x);

np.allclose(y1, y2); // => true
```

The `jit` function is especially useful when doing long sequences of primitives on GPU, since it
fuses operations together into a single kernel dispatch. This
[improves memory bandwidth usage](https://substack.com/home/post/p-163548742) on hardware
accelerators, which is the bottleneck on GPU rather than raw FLOPs. For instance:

```ts
export const hypot = jit(function hypot(x1: np.Array, x2: np.Array) {
  using x1sq = np.square(x1);
  using x2sq = np.square(x2);
  using sum = x1sq.add(x2sq);
  return np.sqrt(sum);
});
```

Without JIT, the `hypot()` function would require four kernel dispatches: two multiplies, one add,
and one sqrt. JIT fuses these together into a single kernel that does it all at once.

All functional transformations can take typed `JsTree` of inputs and outputs. These are similar to
[JAX's pytrees](https://docs.jax.dev/en/latest/pytrees.html), and it's basically just a structure of
nested JavaScript objects and arrays. For instance:

```ts
import { grad, numpy as np, tree } from "@jax-js-nonconsuming/jax";

type Params = {
  foo: np.Array;
  bar: np.Array[];
};

function getSums(p: Params) {
  using fooSum = p.foo.sum();
  using bar0 = p.bar[0].sum();
  using bar1 = p.bar[1].sum();
  using barSum = bar0.add(bar1);
  return fooSum.add(barSum);
}

using g = tree.makeDisposable(
  grad(getSums)({
    foo: np.array([1, 2, 3]),
    bar: [np.array([10]), np.array([11, 12])],
  }),
);
// => { foo: [1, 1, 1], bar: [[1], [1, 1]] }
```

Note that you need to use `type` alias syntax rather than `interface` to define fine-grained
`JsTree` types.

### Devices

Similar to JAX, jax-js has a concept of "devices" which are a backend that stores Arrays in memory
and determines how to execute compiled operations on them.

There are currently 4 devices in jax-js:

- `cpu`: Slow, interpreted JS, only meant for debugging.
- `wasm`: [WebAssembly](https://webassembly.org/), currently single-threaded and blocking.
- `webgpu`: [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API), available on
  [supported browsers](https://caniuse.com/webgpu) (Chrome, Firefox, Safari, iOS).
- `webgl`: [WebGL2](https://developer.mozilla.org/en-US/docs/Web/API/WebGL2RenderingContext), via
  fragment shaders. This is an older graphics API that runs on almost all browsers, but it is much
  slower than WebGPU. It's offered on a best-effort basis and not as well-supported. The `webgl`
  device has not been tested during development of jax-js-nonconsuming.

**We recommend `webgpu` for best performance running neural networks and `wasm` for narrow
sequential computations.** The default device is `wasm`, but you can change this at startup time:

```ts
import { defaultDevice, init } from "@jax-js-nonconsuming/jax";

const devices = await init(); // Starts all available backends.

if (devices.includes("webgpu")) {
  defaultDevice("webgpu");
} else {
  console.warn("WebGPU is not supported, falling back to Wasm.");
}
```

You can also place individual arrays on specific devices:

```ts
import { devicePut, numpy as np } from "@jax-js-nonconsuming/jax";

using ar = np.array([1, 2, 3]); // Starts with device="wasm"
await devicePut(ar, "webgpu"); // Now device="webgpu"
```

### Helper libraries

There are other libraries in the `@jax-js-nonconsuming` namespace that can work with jax-js, or be
used in a self-contained way in other projects.

- [**`@jax-js-nonconsuming/loaders`**](packages/loaders) can load tensors from various formats like
  Safetensors, includes a fast and compliant implementation of BPE, and caches HTTP requests for
  large assets like model weights in OPFS.
- [**`@jax-js-nonconsuming/onnx`**](packages/onnx) is a model loader from the
  [ONNX](https://onnx.ai/) format into native jax-js functions.
- [**`@jax-js-nonconsuming/optax`**](packages/optax) provides implementations of optimizers like
  Adam and SGD.

### Performance

The WebGPU runtime includes an ML compiler with tile-aware optimizations, tuned for individual
browsers. Also, this library uniquely has the `jit()` feature that fuses operations together and
records an execution graph. jax-js achieves **over 7000 GFLOP/s** for matrix multiplication on an
Apple M4 Max chip ([try it](https://hamk-uas.github.io/jax-js-nonconsuming/bench/matmul)).

For that example, it's significantly faster than both
[TensorFlow.js](https://github.com/tensorflow/tfjs) and
[ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), which both use handwritten
libraries of custom kernels.

It's still early though. There's a lot of low-hanging fruit to continue optimizing the library, as
well as unique optimizations such as FlashAttention variants.

### API Reference

That's all for this short tutorial. Please see the generated
[API reference](https://hamk-uas.github.io/jax-js-nonconsuming/docs/) for detailed documentation.

## Development

_The following technical details are for contributing to jax-js-consuming and modifying its
internals._

This repository is managed by [`pnpm`](https://pnpm.io/). You can compile and build all packages in
watch mode with:

```bash
pnpm install
pnpm run build:watch
```

The `pnpm install` command automatically sets up Git hooks via
[Husky](https://typicode.github.io/husky/). Pre-commit hooks will run ESLint and Prettier on staged
files to ensure code quality.

You can also run linting and formatting manually:

```bash
pnpm lint          # Run ESLint (includes @jax-js-nonconsuming/eslint-plugin-jax-js ownership rules)
pnpm format        # Format all files with Prettier
pnpm format:check  # Check formatting without writing
pnpm check         # Run TypeScript type checking
```

Then you can run tests in a headless browser using [Vitest](https://vitest.dev/).

```bash
pnpm exec playwright install
pnpm test
```

We are currently on an older version of Playwright that supports using WebGPU in headless mode;
newer versions skip the WebGPU tests.

To start a Vite dev server running the website, demos and REPL:

```bash
pnpm -C website dev
```

## Maintainer Guide

This section is for maintainers preparing releases for the public repository.

### First public tag (recommended)

The package version is currently `0.2.0-alpha.1`. For the first public stable tag, use:

- **`v0.2.0`** (recommended)

Why: upstream `ekzhang/jax-js` is in the `0.1.x` line, and this fork already includes substantial
fork-only behavior changes (`scan`, non-consuming ownership, buffer recycling, `checkLeaks`).
Starting stable tags at `0.2.0` keeps the fork's semver clear and avoids looking older than the
current pre-release lineage.

### Release steps

1. Ensure clean state and up-to-date `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
```

2. Run release checks:

```bash
pnpm build
pnpm check
pnpm test
pnpm run test:deno
```

3. Bump `package.json` version (for the first stable release: `0.2.0`).

4. Commit release metadata:

```bash
git add package.json pnpm-lock.yaml
git commit -m "chore(release): v0.2.0"
```

5. Create and push tag:

```bash
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

6. Create GitHub Release from `v0.2.0` and include summary notes (ownership model, scan,
   compatibility notes).

### Versioning policy (fork vs upstream)

- Use **independent semver** for this fork.
- Track upstream compatibility in notes/docs, but do not mirror upstream tags exactly.
- Suggested bump rules:
  - **Patch**: bug fixes, docs-only release notes updates, internal refactors without API changes.
  - **Minor**: new public APIs/features (e.g., new transformations, backend capabilities).
  - **Major**: breaking API or ownership-model behavior changes.
- When rebasing/syncing from upstream, choose bump level by user-visible impact in this fork.

### Install snippet update after first tag

After creating the first tag, keep install examples pinned to a stable tag:

```bash
npm install github:hamk-uas/jax-js-nonconsuming#v0.2.0
```

## Contributing

Contributions are welcome! Please open issues and PRs on this repository for topics **specific to
the non-consuming fork**, such as:

- The non-consuming ownership model and `using`-based patterns
- The `@jax-js-nonconsuming/eslint-plugin-jax-js` linter
- `lax.scan`, buffer recycling, `checkLeaks`, or other fork-only features
- Documentation, demos, or examples specific to this fork

For feature requests or bugs that apply to **both branches** (e.g., new NumPy/JAX ops, backend
improvements, core tracing), please file them
[upstream at ekzhang/jax-js](https://github.com/ekzhang/jax-js/issues) instead. This avoids
duplicate work and ensures fixes land in both codebases.

**Upstream sync policy:** We are likely to periodically rebase onto upstream to pick up new features
and fixes, but there may be delays or pauses. Development of this fork may stop at any time — if
that happens, upstream jax-js continues independently.

Before submitting a PR, run the full CI checks locally:

```bash
pnpm build && pnpm check && pnpm test && pnpm run test:deno
```

## Differences from upstream

This fork replaces the upstream **move-semantics** ownership model with a **non-consuming** model.
The API is otherwise identical — all NumPy/JAX functions, `jit`, `grad`, `vmap`, `scan`, backends,
and demos work the same way. Mandelbrot has been updated with `scan` though.

| Aspect                             | Upstream (ekzhang/jax-js)                   | This fork (non-consuming)                                   |
| ---------------------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| **Ownership model**                | Move semantics                              | Non-consuming                                               |
| **Operations consume inputs?**     | Yes — every op decrements refcount          | No — inputs stay alive                                      |
| **`.ref` needed to reuse arrays?** | Yes — `x.ref` before passing to a second op | Never                                                       |
| **`UseAfterFreeError` risk**       | Common if `.ref` is forgotten               | Eliminated (but leaks become silent — see tradeoffs below)  |
| **`using` declarations**           | Not used                                    | First-class — auto-dispose at block end                     |
| **ESLint plugin**                  | `@hamk-uas/eslint-plugin-jax-js` (move)     | `@jax-js-nonconsuming/eslint-plugin-jax-js` (non-consuming) |
| **`lax.scan`**                     | Not implemented                             | Full support (JIT, autodiff, vmap, native compilation)      |
| **Buffer recycling**               | Not implemented                             | JIT-level `recycle` step + WebGPU buffer pool               |
| **`tree.makeDisposable`**          | Not available                               | Wraps any object for `using`-based cleanup                  |
| **`Array.consumeData()`**          | Not available                               | Reads data and disposes in one call                         |
| **`checkLeaks` diagnostic**        | Not available                               | Runtime leak detection with stack traces                    |

### Tradeoffs of the non-consuming model

The non-consuming model makes some things easier and other things harder. Here are the real costs:

**Silent leaks replace noisy crashes.** Move semantics crash immediately (`UseAfterFreeError`) when
you forget `.ref` — painful, but the error points straight at the bug. The non-consuming model never
crashes from reuse, but a forgotten `.dispose()` leaks GPU memory silently. You may not notice until
the GPU is out of memory hundreds of iterations later. The `checkLeaks` diagnostic and the ESLint
plugin exist specifically to compensate for this, but they are opt-in tools, not a built-in safety
net.

**Higher peak memory in eager mode.** Expression chains like `x.mul(y).add(z).sub(w)` create
intermediate arrays that linger until GC or explicit disposal. With move semantics, each
intermediate is freed as soon as the next operation consumes it. In the non-consuming model, all
intermediates stay alive simultaneously. For large tensors this can double or triple peak memory.
`jit()` solves this (it tracks last-use and frees at the optimal point), but the problem is real in
eager mode — the exact mode you use for debugging.

**JavaScript GC doesn't know about GPU memory.** The JS garbage collector tracks JS heap pressure,
not the 4 GB of VRAM on your GPU. A leaked 512×512 `f32` buffer is 1 MB of GPU memory but only ~64
bytes of JS heap. GC may never run. `FinalizationRegistry` is too slow and unpredictable to rely on.
This means leaks from forgotten `.dispose()` calls accumulate indefinitely in practice.

**Method chains become a pain point.** `a.mul(b).add(c).div(d)` is natural in NumPy. In the
non-consuming model, each `.method()` allocates a new GPU buffer that nobody frees. You need `using`
declarations (which require separate statements) or explicit `.dispose()`. This makes fluent API
style impractical for large tensors:

```ts
// ❌ Leaks two intermediate GPU buffers in eager mode:
const result = a.mul(b).add(c).div(d);

// ✅ Correct, but more verbose than the NumPy equivalent:
using t1 = a.mul(b);
using t2 = t1.add(c);
const result = t2.div(d);
```

**`using` has ecosystem gaps.** The TC39 Explicit Resource Management proposal is not yet supported
everywhere — Svelte's parser can't handle `using` in `.svelte` files, and older bundlers may need
transpilation. A polyfill is included, but it adds friction.

**More tooling required.** Move semantics enforce discipline automatically (the program crashes if
you get it wrong). The non-consuming model relies on voluntary tooling: the ESLint plugin for static
analysis, `checkLeaks` for runtime detection, and developer discipline for everything in between. If
you skip the tooling, bugs are harder to find.

**Neither model is free.** Move semantics pay with `UseAfterFreeError` bugs and `.ref` boilerplate.
The non-consuming model pays with silent leaks, higher eager-mode memory, and more reliance on
tooling. This fork bets that the second set of problems is easier to manage for teams coming from
Python/MATLAB — but it is a genuine tradeoff, not a free lunch.

### Which version should I use?

- **Use this fork** if you want a simpler ownership model where arrays can be freely reused, `using`
  declarations handle cleanup, and `lax.scan` is available.
- **Use upstream** if you prefer fail-fast ownership enforcement (crashes over silent leaks), are
  already invested in the move-semantics model and the `@hamk-uas` ESLint plugin, or if you need to
  stay on the upstream release cadence.

The two versions are **not mix-and-match** — code written for one ownership model will not work
correctly with the other. The `@jax-js-nonconsuming/eslint-plugin-jax-js` included here enforces the
non-consuming patterns and will flag `.ref` usage as unnecessary.

### Migrating from upstream

1. **Remove all `.ref` calls** — operations no longer consume inputs.
2. **Replace manual refcount juggling with `using`** — `using x = np.array(...)` auto-disposes at
   block end.
3. **Call `.dispose()` explicitly for long-lived arrays** — or wrap in `tree.makeDisposable()`.
4. **Install `@jax-js-nonconsuming/eslint-plugin-jax-js`** — it catches leaks, use-after-dispose,
   and unnecessary `.ref` at edit time. See the [plugin README](packages/eslint-plugin) for setup.

### AI-assisted development

This fork is developed primarily using AI coding agents (GitHub Copilot, Claude, GPT, Gemini) with
gentle human supervision. All changes go through the full CI pipeline (`pnpm test`, `pnpm check`,
`pnpm run test:deno`) and the pre-commit hook runs the complete test suite before every commit.
