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
  <a href="./.github/copilot-instructions.md">Copilot Instructions</a> |
  <a href="https://discord.gg/BW6YsCd4Tf">jax-js Discord</a>
</strong></p>

> **Fork notice:** This is a fork of [ekzhang/jax-js](https://github.com/ekzhang/jax-js) with a
> **non-consuming ownership model**. Operations leave inputs alive (no `.ref` needed), and `using`
> declarations provide deterministic GPU/WASM memory cleanup.
>
> **Why this fork?** The original jax-js uses move semantics, where operations consume their inputs.
> This fork was created for teams familiar with MATLAB or Python (NumPy) where move semantics are
> unexpected. We also fast-tracked `lax.scan` and `lax.associativeScan` implementations. `using`
> declarations handle the common case — block-scoped arrays are disposed automatically — but
> patterns like method chains, loop-carried state, and nested results still need manual care. A
> built-in `checkLeaks` diagnostic and ESLint plugin help catch what `using` misses. See
> [Tradeoffs](#tradeoffs-of-the-non-consuming-model) for an honest comparison.
>
> See [Differences from upstream](#differences-from-upstream) for a full comparison between the
> original and this fork.
>
> 🤖 The fork code & documentation commits are AI-generated with gentle human supervision.

**jax-js** is a machine learning framework for the browser. It aims to bring
[JAX](https://jax.dev)-style, high-performance CPU and GPU kernels to JavaScript, so you can run
numerical applications on the web.

```bash
npm install github:hamk-uas/jax-js-nonconsuming
```

To pin a specific release tag (once available):

```bash
npm install github:hamk-uas/jax-js-nonconsuming#v0.3.0
```

**pnpm users:** pnpm requires explicit permission to run build scripts from Git dependencies. Add
this to your `package.json`:

```json
{
  "pnpm": {
    "onlyBuiltDependencies": ["@hamk-uas/jax-js-nonconsuming"]
  }
}
```

Without this, `pnpm install` fails with `ERR_PNPM_GIT_DEP_PREPARE_NOT_ALLOWED`. The `prepare` script
runs `tsdown` to build the package and its sub-packages (optax, loaders, onnx, eslint-plugin).

Under the hood, it translates array operations into a compiler representation, then synthesizes
kernels in WebAssembly and WebGPU.

The original jax-js was written from scratch with zero zero external dependencies. jax-js and this
fork maintain close API compatibility with NumPy/JAX. Since everything runs client-side, jax-js is
one of the most portable GPU ML frameworks, since it runs anywhere a browser can run.

## Quickstart

```js
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

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

Here's a quick, high-level comparison with other popular web ML runtimes. Performance labels are
workload- and hardware-dependent.

| Feature                                  | jax-js-nonconsuming | jax-js v0.1.9 | TensorFlow.js   | onnxruntime-web    |
| ---------------------------------------- | ------------------- | ------------- | --------------- | ------------------ |
| **Overview**                             |                     |               |                 |                    |
| API style                                | JAX/NumPy           | JAX/NumPy     | TensorFlow-like | Static ONNX graphs |
| Speed                                    | Very fast           | Very fast     | Fast            | Very fast          |
| Bundle size (gzip)                       | 108 KB              | 80 KB         | 269 KB          | 90 KB + 24 MB Wasm |
| **Autodiff & JIT**                       |                     |               |                 |                    |
| Gradients                                | ✅                  | ✅            | ✅              | ❌                 |
| Jacobian and Hessian                     | ✅                  | ✅            | ❌              | ❌                 |
| `jvp()` forward differentiation          | ✅                  | ✅            | ❌              | ❌                 |
| `jit()` kernel fusion                    | ✅                  | ✅            | ❌              | ❌                 |
| `vmap()` auto-vectorization              | ✅                  | ✅            | ❌              | ❌                 |
| `scan()` scan over leading axis          | ✅                  | ❌            | ❌              | ❌                 |
| `associativeScan()` parallel prefix scan | ✅                  | ❌            | ❌              | ❌                 |
| Graph capture                            | ✅                  | ✅            | ❌              | ✅                 |
| **Backends & Data**                      |                     |               |                 |                    |
| WebGPU backend                           | ✅                  | ✅            | 🟡 Preview      | ✅                 |
| WebGL backend                            | ✅                  | ✅            | ✅              | ✅                 |
| Wasm (CPU) backend                       | ✅                  | ✅            | ✅              | ✅                 |
| Eager array API                          | ✅                  | ✅            | ✅              | ❌                 |
| Run ONNX models                          | 🟡 Partial          | 🟡 Partial    | ❌              | ✅                 |
| Read safetensors                         | ✅                  | ✅            | ❌              | ❌                 |
| Float64                                  | ✅                  | ✅            | ❌              | ❌                 |
| Float32                                  | ✅                  | ✅            | ✅              | ✅                 |
| Float16                                  | ✅                  | ✅            | ❌              | ✅                 |
| BFloat16                                 | ❌                  | ❌            | ❌              | ❌                 |
| Packed Uint8                             | ❌                  | ❌            | ❌              | 🟡 Partial         |
| Mixed precision                          | ✅                  | ✅            | ❌              | ✅                 |
| Mixed devices                            | ✅                  | ✅            | ❌              | ❌                 |
| **Ops & Numerics**                       |                     |               |                 |                    |
| Arithmetic functions                     | ✅                  | ✅            | ✅              | ✅                 |
| Matrix multiplication                    | ✅                  | ✅            | ✅              | ✅                 |
| General einsum                           | ✅                  | ✅            | 🟡 Partial      | 🟡 Partial         |
| Sorting                                  | ✅                  | ✅            | ❌              | ❌                 |
| Activation functions                     | ✅                  | ✅            | ✅              | ✅                 |
| NaN/Inf numerics                         | ✅                  | ✅            | ✅              | ✅                 |
| Basic convolutions                       | ✅                  | ✅            | ✅              | ✅                 |
| n-d convolutions                         | ✅                  | ✅            | ❌              | ✅                 |
| Strided/dilated convolution              | ✅                  | ✅            | ✅              | ✅                 |
| Cholesky, Lstsq                          | ✅                  | ✅            | ❌              | ❌                 |
| LU, Solve, Determinant                   | ✅                  | ✅            | ❌              | ❌                 |
| SVD                                      | ❌                  | ❌            | ❌              | ❌                 |
| FFT                                      | ✅                  | ✅            | ✅              | ✅                 |
| Basic RNG (Uniform, Normal)              | ✅                  | ✅            | ✅              | ✅                 |
| Advanced RNG                             | ✅                  | ✅            | ❌              | ❌                 |

## Tutorial

Programming in `jax-js` looks [very similar to JAX](https://docs.jax.dev/en/latest/jax-101.html),
just in JavaScript.

### Arrays

Create an array with `np.array()`:

```ts
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

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
import { numpy as np, scipySpecial as special } from "@hamk-uas/jax-js-nonconsuming";

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

The `@hamk-uas/eslint-plugin-jax-js` catches the most common memory leaks (missing `using`,
use-after-dispose, unnecessary `.ref`) at edit time — see the
[plugin README](packages/eslint-plugin) for setup.

**Ownership invariance principle:** write code that is ownership-correct in both eager mode and
`jit()` mode. `jit()` is a performance optimization (fusion, recycling), not a semantics change. If
code leaks or relies on different ownership behavior in eager mode, treat it as a real bug. For CI
enforcement in user code, use `jaxJs.configs.invariance` from `@hamk-uas/eslint-plugin-jax-js`.

### grad(), vmap() and jit()

JAX's signature composable transformations are also supported in jax-js. Here is a simple example of
using `grad` and `vmap` to compute the derivative of a function:

```ts
import { numpy as np, grad, vmap } from "@hamk-uas/jax-js-nonconsuming";

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
import { grad, numpy as np, tree } from "@hamk-uas/jax-js-nonconsuming";

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
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";

const devices = await init(); // Starts all available backends.

if (devices.includes("webgpu")) {
  defaultDevice("webgpu");
} else {
  console.warn("WebGPU is not supported, falling back to Wasm.");
}
```

You can also place individual arrays on specific devices:

```ts
import { devicePut, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

using ar = np.array([1, 2, 3]); // Starts with device="wasm"
await devicePut(ar, "webgpu"); // Now device="webgpu"
```

### Helper libraries

jax-js includes three helper libraries, available as **sub-path exports** from the main package:

- [**optax**](packages/optax) — optimizers (Adam, SGD) and gradient processing
- [**loaders**](packages/loaders) — Safetensors, BPE tokenizers, OPFS-cached downloads
- [**onnx**](packages/onnx) — load [ONNX](https://onnx.ai/) models into native jax-js functions

```ts
import { adam } from "@hamk-uas/jax-js-nonconsuming/optax";
import { cachedFetch, safetensors } from "@hamk-uas/jax-js-nonconsuming/loaders";
import { ONNXModel } from "@hamk-uas/jax-js-nonconsuming/onnx";
```

No extra install needed — they're included when you install `@hamk-uas/jax-js-nonconsuming`. The
`loaders` and `onnx` sub-packages have optional native dependencies (`@bufbuild/protobuf`,
`sentencepiece-buf`, `onnx-buf`) that are listed in the main package's `optionalDependencies` and
installed automatically if available for your platform.

### Performance

The WebGPU runtime includes an ML compiler with tile-aware optimizations, tuned for individual
browsers. Also, this library uniquely has the `jit()` feature that fuses operations together and
records an execution graph. jax-js achieves **over 7000 GFLOP/s** for matrix multiplication on an
Apple M4 Max chip ([try it](https://hamk-uas.github.io/jax-js-nonconsuming/bench/matmul)).

In that specific benchmark run, it was faster than both
[TensorFlow.js](https://github.com/tensorflow/tfjs) and
[ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), which both use handwritten
libraries of custom kernels. Results vary by model, operator mix, and hardware.

It's still early though. There's a lot of low-hanging fruit to continue optimizing the library, as
well as unique optimizations such as FlashAttention variants.

### API Reference

That's all for this short tutorial. Please see the generated
[API reference](https://hamk-uas.github.io/jax-js-nonconsuming/docs/) for detailed documentation.

## Development

_The following technical details are for contributing to jax-js-nonconsuming and modifying its
internals._

This repository is managed by [`pnpm`](https://pnpm.io/). You can compile and build all packages in
watch mode with:

```bash
pnpm install
pnpm run build:watch
```

The `pnpm install` command automatically sets up Git hooks via
[Husky](https://typicode.github.io/husky/). This repository intentionally shifts verification left:
we run the necessary quality and release-safety checks in local pre-commit rather than relying on
heavy CI gates for GitHub Pages deployment confidence.

### Local check workflow

Run checks iteratively while developing:

You can also run linting and formatting manually:

```bash
pnpm lint          # Run ESLint (includes @hamk-uas/eslint-plugin-jax-js ownership rules)
pnpm format        # Format all files with Prettier
pnpm format:check  # Check formatting without writing
pnpm check         # Run TypeScript type checking
pnpm run test:eslint-plugin   # Rule-level tests for in-repo ESLint plugin
pnpm run lint:ownership:internal  # Maintainer transform ownership checks
```

Then you can run tests in a headless browser using [Vitest](https://vitest.dev/).

```bash
pnpm exec playwright install
pnpm test
pnpm run test:policy:strict  # Strict mode: no expected-failure debt
pnpm run test:arch           # Architectural mode: failures gated by manifest
pnpm run test:deno           # Deno WebGPU tests (isolated files)
pnpm run test:website:smoke  # Website build + smoke checks
```

Architectural mode is intended for large refactors and uses `.ci/expected-failures.json` as an
explicit, expiring debt ledger. See `docs/testing-policy.md` for workflow details.

For maintainer-only transform ownership checks in framework internals, run:

```bash
pnpm run lint:ownership:internal
```

For website ownership checks used by demos/repl:

```bash
pnpm run lint:ownership:website
```

### Pre-commit policy

Pre-commit is branch-aware and runs via `scripts/precommit.sh`.

- **Feature profile** (default on non-main branches):
  - `build`, `check`, `lint`, `format:check`
  - `test:eslint-plugin`
  - `lint:ownership:internal`, `lint:ownership:website`
  - core invariants: `vitest run test/refcount.test.ts test/transform-compositions.test.ts`
- **Full profile** (default on `main`, `master`, `release/*`, `hotfix/*`):
  - everything in feature profile
  - `test:policy:strict`
  - `test:deno`
  - `test:website:smoke`

This keeps day-to-day feature iteration fast while enforcing release-grade checks when committing on
main/release branches.

For large refactors with explicit, expiring expected-failure debt, use architectural mode:

```bash
JAX_ARCH_MODE=1 git commit -m "your message"
```

Architectural mode still enforces strict core invariant suites before applying manifest-based
failure accounting. See `docs/testing-policy.md` for workflow details.

You can override profile selection explicitly:

```bash
JAX_PRECOMMIT_PROFILE=feature git commit -m "..."
JAX_PRECOMMIT_PROFILE=full git commit -m "..."
```

Before merging to `main`, run one commit (or dry run) with full profile:

```bash
JAX_PRECOMMIT_PROFILE=full git commit -m "chore: pre-merge full local checks"
```

Inspiration from `hamk-uas/eslint-plugin-jax-js`: keep hooks explicit and developer-visible, and
keep maintainer release checks documented and reproducible.

We are currently on an older version of Playwright that supports using WebGPU in headless mode;
newer versions skip the WebGPU tests.

To start a Vite dev server running the website, demos and REPL:

```bash
pnpm -C website dev
```

## Maintainer Guide

This section is for maintainers who create releases.

### Releasing

#### Steps

```bash
# 1. Make sure all checks pass
pnpm build
pnpm check
pnpm run test:policy:strict
pnpm run test:deno
pnpm run test:website:smoke
pnpm run test:eslint-plugin
pnpm run lint:ownership:internal
pnpm run lint:ownership:website

# Or equivalently (same full profile as main-branch pre-commit):
JAX_PRECOMMIT_PROFILE=full scripts/precommit.sh

# 2. Bump the version in package.json (choose patch / minor / major as
#    appropriate — see Version numbering below).
#    Then commit and tag:
git add package.json pnpm-lock.yaml
git commit -m "chore(release): v0.2.1"
git tag v0.2.1

# 3. Push the commit and tag
git push && git push --tags

# 4. Create a GitHub release
#    Go to https://github.com/hamk-uas/jax-js-nonconsuming/releases/new
#    Select the tag, write release notes summarizing changes.
```

Users install specific tags, so after releasing they can upgrade with:

```bash
npm install github:hamk-uas/jax-js-nonconsuming#v0.2.1
```

#### Version numbering

| Change type                                                 | Bump  | Example                                            |
| ----------------------------------------------------------- | ----- | -------------------------------------------------- |
| Documentation only (README, comments, copilot-instructions) | none  | Users on `main` get it automatically               |
| Bug fix, precision improvement, internal refactor           | patch | Kahan summation, ownership fix, test improvements  |
| New jax-js/NumPy ops added to API surface                   | patch | New `numpy.foo()` function                         |
| New public API or feature (transform, backend capability)   | minor | `lax.scan`, buffer recycling, new ESLint rule      |
| Breaking API or ownership-model behavior change             | major | Removing a public function, changing dispose rules |

This fork uses **independent semver** — it does not mirror upstream `ekzhang/jax-js` tags. Track
upstream compatibility in release notes, and choose bump level by user-visible impact in this fork.
When rebasing/syncing from upstream, the bump level depends on what user-facing changes come along.

### Releasing a bug fix

For simple bug-fix PRs (the common case):

1. Merge the PR to `main`.
2. Version & tag — bug fixes are always a patch bump. Follow the releasing steps above.
3. Create a GitHub release with notes describing the fix.

## Contributing

Contributions are welcome! Please open issues and PRs on this repository for topics **specific to
the non-consuming fork**, such as:

- The non-consuming ownership model and `using`-based patterns
- The `@hamk-uas/eslint-plugin-jax-js` linter
- `lax.scan`, buffer recycling, `checkLeaks`, or other fork-only features
- Documentation, demos, or examples specific to this fork

For feature requests or bugs that apply to **both branches** (e.g., new NumPy/JAX ops, backend
improvements, core tracing), please file them
[upstream at ekzhang/jax-js](https://github.com/ekzhang/jax-js/issues) instead. This avoids
duplicate work and ensures fixes land in both codebases.

**Upstream sync policy:** We may periodically rebase onto upstream to pick up new features and
fixes, but there is no guarantee of continuous updates. Maintenance debt can accumulate across
projects, and this fork may be reduced in scope or paused if priorities shift. Upstream jax-js
continues independently.

Before submitting a PR, run the full CI checks locally:

```bash
pnpm build && pnpm check && pnpm run test:policy:strict && pnpm run test:deno && pnpm run test:website:smoke && pnpm run test:eslint-plugin && pnpm run lint:ownership:internal && pnpm run lint:ownership:website
```

## Differences from upstream

This fork replaces the upstream **move-semantics** ownership model with a **non-consuming** model.
Outside ownership semantics and fork-specific features, the APIs are broadly aligned for common
NumPy/JAX usage (`jit`, `grad`, `vmap`, backends, demos), with some intentional divergence (for
example `scan`, `checkLeaks`, and ownership tooling).

| Aspect                             | Upstream (ekzhang/jax-js)                   | This fork (non-consuming)                                                                                                                                                                                                                             |
| ---------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ownership model**                | Move semantics                              | Non-consuming                                                                                                                                                                                                                                         |
| **Operations consume inputs?**     | Yes — every op decrements refcount          | No — inputs stay alive                                                                                                                                                                                                                                |
| **`.ref` needed to reuse arrays?** | Yes — `x.ref` before passing to a second op | Not in user code                                                                                                                                                                                                                                      |
| **`UseAfterFreeError` risk**       | Common if `.ref` is forgotten               | Gone for reuse; still possible after explicit `.dispose()`                                                                                                                                                                                            |
| **`using` declarations**           | Not used                                    | First-class — auto-dispose at block end                                                                                                                                                                                                               |
| **ESLint plugin**                  | `@hamk-uas/eslint-plugin-jax-js` (move)     | `@hamk-uas/eslint-plugin-jax-js` (non-consuming)                                                                                                                                                                                                      |
| **`lax.scan`**                     | Not implemented                             | Full support (JIT, autodiff, vmap, native compilation)                                                                                                                                                                                                |
| **`lax.associativeScan`**          | Not implemented                             | Kogge-Stone parallel prefix scan; pytrees, any axis, reverse, autodiff. Eager mode routes through a cached whole-call `jit` wrapper (outside abstract tracing); explicit `jit(...)` still gives maximum throughput and manual cache lifetime control. |
| **Buffer recycling**               | Not implemented                             | JIT-level `recycle` step + WebGPU buffer pool                                                                                                                                                                                                         |
| **`tree.makeDisposable`**          | Not available                               | Wraps any object for `using`-based cleanup                                                                                                                                                                                                            |
| **`Array.consumeData()`**          | Not available                               | Reads data and disposes in one call                                                                                                                                                                                                                   |
| **`checkLeaks` diagnostic**        | Not available                               | Runtime leak detection with stack traces                                                                                                                                                                                                              |

### Tradeoffs of the non-consuming model

The non-consuming model makes some things easier and other things harder. Here are the real costs:

**Important context: JIT neutralizes most intermediate-lifetime differences.** Under `jit()`, both
ownership models compile to the same Jaxpr graph, and the compiler derives buffer lifetimes from
that graph — not from the ownership model. Intermediates and buffers are freed at exact last use; in
many common workloads this makes peak-memory behavior very similar across models. The tradeoffs
below apply primarily to **eager mode** — the mode you use for debugging, prototyping, and the REPL.
Production hot paths should be JIT-wrapped anyway for performance (kernel fusion), which narrows the
practical ownership-model gap to JIT boundaries (who disposes inputs, outputs, and the `jit`
function itself) and any code that runs outside `jit()`.

**Silent leaks replace noisy crashes.** Move semantics crash immediately (`UseAfterFreeError`) when
you forget `.ref` — painful, but the error points straight at the bug. The non-consuming model never
crashes from reuse, but a missing `.dispose()` leaks GPU memory silently. For the common case —
block-scoped arrays — `using` declarations prevent this: arrays are disposed automatically at block
exit. But `using` can't help everywhere: method chains create anonymous intermediates that nobody
names (and thus nobody disposes), nested results from `scan`/`grad` need `tree.dispose()`, and
loop-carried state or arrays stored in caches require manual discipline. The `checkLeaks` diagnostic
(built into the test suite so every test is leak-checked) and the ESLint plugin catch many of these
cases, but they are developer tools, not a runtime safety net. (Move semantics can also leak — e.g.
an over-`.ref`'d array or a retained reference — but the fail-fast default for _reuse_ bugs makes
those easier to spot.)

**Higher peak memory in eager mode.** Expression chains like `x.mul(y).add(z).sub(w)` create
intermediate arrays that linger until GC or explicit disposal. With move semantics, each
intermediate is freed as soon as the next operation consumes it. In the non-consuming model, all
intermediates stay alive simultaneously — for large tensors this can significantly increase peak
memory (the exact factor depends on chain length and tensor size). Breaking chains into `using`
temporaries solves this (intermediates are disposed at block exit), but the code is more verbose
than the NumPy equivalent. Under `jit()`, both models free intermediates at the optimal point — this
is purely an eager-mode difference. But eager mode is where you debug, and debugging with higher
memory footprint is a real obstacle.

**JavaScript GC doesn't know about GPU memory.** The JS garbage collector tracks JS heap pressure,
not the 4 GB of VRAM on your GPU. A leaked 512×512 `f32` buffer is 1 MB of GPU memory but only ~64
bytes of JS heap. GC may never run. `FinalizationRegistry` is too slow and unpredictable to rely on.
This problem affects both ownership models — any jax-js program must eventually free GPU buffers
explicitly. The non-consuming model simply makes it easier to forget, because nothing crashes when
you do.

**Method chains become a pain point in eager mode.** `a.mul(b).add(c).div(d)` is natural in NumPy.
In the non-consuming model, each `.method()` allocates a new GPU buffer. The fix is `using`
declarations, but they require separate statements — one per intermediate. Under `jit()`, these
chains produce tracers (not real GPU buffers) and the compiler manages everything — so the memory
cost only appears in eager code. Still, eager code is where you prototype and learn the API:

```ts
// ❌ Leaks two intermediate GPU buffers in eager mode:
const result = a.mul(b).add(c).div(d);

// ✅ Correct, but more verbose than the NumPy equivalent:
using t1 = a.mul(b);
using t2 = t1.add(c);
const result = t2.div(d);

// Under jit(), intermediates are tracers (not real buffers), so the chain
// doesn't leak GPU memory in practice. But write the second form anyway —
// code should be ownership-correct in both eager and jit mode.
```

**`using` has ecosystem gaps.** The TC39 Explicit Resource Management proposal is not yet supported
everywhere — Svelte's parser can't handle `using` in `.svelte` files, and older bundlers may need
transpilation. A polyfill is included, but it adds friction.

**More tooling required for edge cases.** `using` handles the most common pattern (block-scoped
arrays) at the language level. But for patterns it doesn't cover — method chains, pytree results,
loop-carried state, long-lived closures — the non-consuming model leans on voluntary tooling: the
ESLint plugin for static analysis and `checkLeaks` for runtime detection. Move semantics fail fast
for _reuse_ mistakes, but have their own blind spots (over-`.ref`, retained references, forgotten
`vjpFn.dispose()`) that also need tooling and discipline.

**Neither model is free.** Move semantics pay with `UseAfterFreeError` bugs, `.ref` boilerplate, and
their own leak surfaces (over-ref, retained refs). The non-consuming model eliminates those costs
but introduces its own: silent leaks for patterns that `using` can't cover, higher eager-mode memory
for unchained intermediates, and reliance on `checkLeaks`/ESLint for the gaps. Under `jit()`, the
two models converge — the compiled programs are identical. Both models need discipline; they just
fail in different ways. This fork bets that `using`-by-default plus opt-in tooling is easier to
manage for teams coming from Python/MATLAB — but it is a genuine tradeoff, not a free lunch.

### Which version should I use?

- **Use this fork** if you want a simpler ownership model where arrays can be freely reused, `using`
  declarations handle cleanup, and `lax.scan` is available.
- **Use upstream** if you prefer fail-fast ownership enforcement (crashes over silent leaks), are
  already invested in the move-semantics model and the `@hamk-uas` ESLint plugin, or if you need to
  stay on the upstream release cadence.

The two versions are **not drop-in interchangeable** — ownership patterns from one model can behave
incorrectly or awkwardly in the other, especially around `.ref` and disposal discipline. The
`@hamk-uas/eslint-plugin-jax-js` included here enforces the non-consuming patterns and will flag
`.ref` usage as unnecessary.

### Migrating from upstream

1. **Remove all `.ref` calls** — operations no longer consume inputs.
2. **Replace manual refcount juggling with `using`** — `using x = np.array(...)` auto-disposes at
   block end.
3. **Call `.dispose()` explicitly for long-lived arrays** — or wrap in `tree.makeDisposable()`.
4. **Install `@hamk-uas/eslint-plugin-jax-js`** — it catches leaks, use-after-dispose, and
   unnecessary `.ref` at edit time. See the [plugin README](packages/eslint-plugin) for setup.

### AI-assisted development

This fork is developed primarily using AI coding agents (GitHub Copilot, Claude, GPT, Gemini) with
gentle human supervision. All changes go through the full CI pipeline (`pnpm test`, `pnpm check`,
`pnpm run test:deno`) and the pre-commit hook runs the complete test suite before every commit.
