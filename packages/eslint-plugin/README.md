# @hamk-uas/eslint-plugin-jax-js

ESLint rules for **jax-js array ownership** — enforces deterministic disposal, prevents GPU/WASM
memory leaks, and catches use-after-dispose bugs at edit time.

Built for the **non-consuming ownership model**: operations leave inputs alive, arrays must be
explicitly `.dispose()`'d (or auto-disposed via `using`), and `.ref` is never needed in user code.

> **Design philosophy:** Rules warn inside `jit()` bodies on purpose. `jit()` is a pure performance
> optimization — code must be ownership-correct in both eager and JIT modes. If a pattern leaks in
> eager mode, it's a real bug, not something to paper over by wrapping in `jit()`.

## Installation

```bash
# From the jax-js monorepo (workspace dependency):
pnpm add -D @hamk-uas/eslint-plugin-jax-js
```

**Requirements:**

- ESLint ≥ 9.0 (flat config)
- TypeScript source via [jiti](https://github.com/unjs/jiti) (ESLint loads `.ts` config files
  automatically since v9.7)

## Quick start

### Recommended config (one line)

```ts
// eslint.config.ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  jaxJs.configs.recommended,
  // ... your other configs
];
```

This enables:

| Rule                                    | Level   | What it catches                                                                          |
| --------------------------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `jax-js/require-using`                  | `warn`  | Local array bindings missing `using`                                                     |
| `jax-js/no-use-after-dispose`           | `error` | Reading/writing a variable after `.dispose()`, `.consumeData()`, or `.consumeDataSync()` |
| `jax-js/no-dispose-then-reassign-param` | `warn`  | Callback alias hazard: `dispose(state); state = param`                                   |
| `jax-js/no-make-disposable-alias`       | `warn`  | Duplicate references in `tree.makeDisposable(...)`                                       |
| `jax-js/no-unnecessary-ref`             | `warn`  | `.ref` calls (unnecessary in non-consuming)                                              |
| `jax-js/no-array-chain`                 | `off`   | Deep fluent chains (strict-mode only)                                                    |
| `jax-js/require-scan-result-dispose`    | `warn`  | Undisposed destructured `lax.scan(...)` results                                          |

### Strict config

```ts
// eslint.config.ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  jaxJs.configs.strict,
  // ... your other configs
];
```

Strict promotes everything to `error` and enables `no-array-chain` — useful for library code and CI
enforcement.

### Invariance config (eager/JIT parity)

```ts
// eslint.config.ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  jaxJs.configs.invariance,
  // ... your other configs
];
```

`invariance` is a purpose-built profile for user code that enforces ownership patterns equally in
eager mode and inside `jit()` bodies. It sets the same user-facing ownership rules as strict to
`error` so `jit()` cannot be used as a workaround for eager-mode ownership bugs.

### Internal transform config (maintainers)

For framework internals (transform wrappers, retained handles, dispose ordering), use:

```ts
// eslint.config.ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  {
    files: ["src/frontend/{jaxpr,jit,jvp,linearize,vmap}.ts"],
    ...jaxJs.configs.internalTransforms,
  },
];
```

This enables:

- `jax-js/require-retained-release`
- `jax-js/require-try-finally-symmetry`
- `jax-js/require-wrapper-dispose-symmetry`

### Individual rule control

```ts
// eslint.config.ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  {
    plugins: { "jax-js": jaxJs },
    rules: {
      "jax-js/require-using": "error",
      "jax-js/no-use-after-dispose": "error",
      "jax-js/no-dispose-then-reassign-param": "warn",
      "jax-js/no-make-disposable-alias": "warn",
      "jax-js/no-unnecessary-ref": "warn",
      "jax-js/no-array-chain": ["error", { minDepth: 3 }],
      "jax-js/require-scan-result-dispose": "warn",
      "jax-js/require-retained-release": "warn",
      "jax-js/require-try-finally-symmetry": "warn",
      "jax-js/require-wrapper-dispose-symmetry": "warn",
    },
  },
];
```

### Limit to specific directories

```ts
// eslint.config.ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  // Enable for source, disable for tests
  {
    files: ["src/**/*.ts", "packages/**/*.ts"],
    ...jaxJs.configs.recommended,
  },
  {
    files: ["test/**/*.ts"],
    plugins: { "jax-js": jaxJs },
    rules: {
      "jax-js/require-using": "off",
      "jax-js/no-use-after-dispose": "off",
      "jax-js/no-dispose-then-reassign-param": "off",
      "jax-js/no-make-disposable-alias": "off",
      "jax-js/no-unnecessary-ref": "off",
      "jax-js/no-array-chain": "off",
      "jax-js/require-scan-result-dispose": "off",
      "jax-js/require-retained-release": "off",
      "jax-js/require-try-finally-symmetry": "off",
      "jax-js/require-wrapper-dispose-symmetry": "off",
    },
  },
];
```

## Rules

Each rule reports warnings and can offer automatic code changes through ESLint:

- **Autofix** (ESLint _fix_) — applied automatically on save or via `eslint --fix`. The rule has
  verified that the change is safe for the specific code it flagged.
- **Suggestion** (ESLint _suggestion_) — shown as a 💡 lightbulb quick fix in your editor. Requires
  manual confirmation because the change may need review.

### `jax-js/require-using`

**Fixable:** 💡 suggestion (converts `const`/`let` to `using`)

Enforces `using` declarations for short-lived local array bindings. Catches the most common source
of memory leaks: forgetting to `.dispose()` temporary arrays.

```ts
// ❌ Error: Local array binding `a` should use `using`
const a = np.zeros([3, 3]);
a.dispose(); // easy to forget!

// ✅ OK: auto-disposed at block end
using a = np.zeros([3, 3]);

// ✅ OK: returned from function (caller owns it)
function makeArray() {
  const result = np.zeros([3, 3]);
  return result;
}

// ✅ OK: persisted to longer-lived structure
const cache = new Map();
const arr = np.zeros([3, 3]);
cache.set("key", arr);

// ✅ OK: explicit dispose visible in same block
const tmp = np.zeros([3, 3]);
// ... use tmp ...
tmp.dispose();
```

**Suppression:**

```ts
// jax-js-lint: allow-non-using
const specialCase = np.zeros([3, 3]);
```

**What counts as "array-producing":**

- Factory calls: `np.array(...)`, `np.zeros(...)`, `np.ones(...)`, `np.arange(...)`, etc.
- Method calls on array receivers: `x.add(...)`, `x.reshape(...)`, `x.transpose()`, etc.
- Namespace calls: `np.add(x, y)`, `np.multiply(...)`, `lax.dot(...)`, etc.
- Transform results: `grad(f)(x)`, `vmap(f)(xs)` (but not `jit(f)` which returns a function)

**What is NOT flagged:**

- Module-scope declarations (globals are long-lived by nature)
- Values that are returned, yielded, or persisted to data structures
- Values with explicit `.dispose()` calls later in the same block
- `Math.*` calls (not array-producing)

> **Inside `grad` / `jit` / `lax.scan` bodies?** `using` is safe and correct there too — see the
> [FAQ](#i-m-inside-a-grad--valueandgrad--jit-body--arent-arrays-managed-by-the-tracer).

### `jax-js/no-use-after-dispose`

**Type:** problem (error by default) · no autofix

Catches reads or writes to a variable after `.dispose()`, `.consumeData()`, or `.consumeDataSync()`
has been called on it. This prevents `UseAfterFreeError` at runtime. The error message includes the
consuming method name and line number for easy navigation.

`.dispose()`, `.consumeData()`, and `.consumeDataSync()` are all treated as consuming calls. Calling
`.dispose()` after `.consumeData()` or `.consumeDataSync()` (double-free) is flagged; calling
`.dispose()` after `.dispose()` is allowed (idempotent no-op at `rc=0`).

```ts
const x = np.array([1, 2, 3]);
x.dispose(); // line 2

// ❌ Error: `x` is used after `.dispose()` on line 2
const y = x.add(np.array([4, 5, 6]));

// ✅ OK: redundant dispose is fine (no-op at rc=0)
x.dispose();

// ❌ Error: `grad` is used after `.consumeData()` on line N
async function f() {
  const grad = computeGrad();
  const v = (await grad.consumeData())[0]; // consumes + disposes grad
  grad.dispose(); // double-free: caught!
}
```

**Scope:** Tracks variables lexically within the same function scope. Does not track across function
boundaries or through aliases.

### `jax-js/no-dispose-then-reassign-param`

**Type:** problem (`warn` in recommended, `error` in strict) · no autofix

Catches a common callback ownership footgun: disposing a state variable and immediately reassigning
it from a callback parameter. If both identifiers alias the same object, the disposal destroys live
data.

```ts
// ❌ Warn: possible alias-dispose bug
function onParamsUpdate(params) {
  tree.dispose(latestParams);
  latestParams = params;
}

// ✅ OK: guard alias case
function onParamsUpdate(params) {
  if (latestParams !== params) tree.dispose(latestParams);
  latestParams = params;
}
```

**Current scope:** Adjacent statement pattern in function bodies:
`dispose(stateVar); stateVar = paramName;`.

### `jax-js/no-make-disposable-alias`

**Type:** suggestion (`warn` in recommended, `error` in strict) · no autofix

Warns when a single identifier is reused multiple times inside a literal passed to
`tree.makeDisposable(...)`, e.g. `{ xf_0: base, yhat: base }` or `[a, a]`.

Even though runtime disposal is alias-safe, this pattern is easy to create accidentally and can hide
ownership intent.

```ts
// ❌ Warn: same array handed off under two keys
const owned = tree.makeDisposable({ xf_0: base, yhat: base });

// ✅ OK: distinct arrays
const owned2 = tree.makeDisposable({ xf_0, yhat });
```

### `jax-js/no-unnecessary-ref`

**Fixable:** ✅ autofix (removes `.ref` automatically with `--fix` or on save)

Flags `.ref` property access. In the non-consuming ownership model, `.ref` is never needed in user
code — operations do not consume inputs. The only legitimate uses are deep inside the framework
internals.

**Automatically skipped contexts** (not flagged):

- `UpdateExpression`: `buffer.ref++`, `buffer.ref--` — plain numeric property mutations
- `BinaryExpression`: `buffer.ref === 0` — property reads in comparisons

**Suppression comments:**

- **Per-line:** `// jax-js-lint: allow-ref` on the line before the flagged code
- **File-level:** `// jax-js-lint: allow-ref` in a leading comment (before any imports) suppresses
  all `.ref` warnings in the entire file

```ts
// ❌ Error: `.ref` is usually unnecessary in non-consuming jax-js
const copy = myArray.ref;

// ✅ OK: suppressed for framework internals (per-line)
// jax-js-lint: allow-ref
const copy = myArray.ref;

// ✅ OK: file-level suppression (at top of file, before imports)
// jax-js-lint: allow-ref — .ref is the core mechanism in this module
import { ... } from "...";
```

### `jax-js/no-array-chain`

**Type:** suggestion (off by default, enabled in strict config) · autofix for simple chains

Flags deep fluent method chains that create unnamed eager-mode temporaries. These intermediates
can't be `using`-managed and may accumulate in GPU memory until GC runs.

Only the outermost chain is reported — inner subchains don't produce duplicate diagnostics.

For migration, `--fix` rewrites straightforward eager chains into `using` temporaries:

- Variable assignment chains: `const y = x.add(1).mul(2)`
- Expression-statement chains: `x.add(1).mul(2)`

More complex sites (for example `return x.add(1).mul(2)`) are intentionally left as diagnostics
without rewrite.

```ts
// ❌ Error (depth 3): Array call chain depth 3 creates unnamed eager temporaries
const y = x.mul(weights).add(bias).tanh();
// (reported once, not once per subchain)

// ✅ OK: explicit intermediates with deterministic cleanup
using a = x.mul(weights);
using b = a.add(bias);
const y = b.tanh();
```

**Options:**

| Option     | Type    | Default | Description                          |
| ---------- | ------- | ------- | ------------------------------------ |
| `minDepth` | integer | `2`     | Minimum chain depth to trigger error |

**Traced-context suppression:** Chains inside tracing transform bodies are automatically suppressed,
because the JIT compiler manages intermediate lifetimes — no eager temporaries are created. This
applies to functions passed directly as the first argument to `jit`, `grad`, `valueAndGrad`, `jvp`,
`vjp`, `vmap`, `jacfwd`, `jacrev`, `hessian`, `linearize`, `makeJaxpr`, and `lax.scan`. Named
function references are also resolved: if a `const`/`let` function variable is passed to a transform
elsewhere in the same scope, chains inside that function are suppressed.

```ts
// ✅ OK: chain inside jit body — JIT manages intermediates
const f = jit((x) => x.mul(weights).add(bias).tanh());

// ✅ OK: named function passed to grad
const loss = (x) => x.mul(x).sum();
const dloss = grad(loss);
```

**Scope:** Only tracks known JAX array methods (`add`, `sub`, `mul`, `div`, `reshape`, `transpose`,
`sum`, `mean`, `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `matmul`, etc.). JavaScript collection
methods like `.map()`, `.filter()`, `.reduce()` are ignored.

### `jax-js/require-scan-result-dispose`

**Type:** problem (`warn` in recommended, `error` in strict) · no autofix

Requires destructured `lax.scan(...)` outputs to be either disposed in the same scope (for example
via `tree.dispose(carry)` / `tree.dispose(ys)`) or returned to caller-owned scope.

Underscore-prefixed bindings (for example `_ys`) are treated as intentional ignores.
`tree.makeDisposable({ carry, ys })` is also treated as explicit ownership handoff.

```ts
// ❌ Warn: scan outputs are dropped without dispose/return
function leaky(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  return np.add(ys.pred, ys.pred);
}

// ✅ OK: both scan outputs are explicitly disposed
function safe(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  tree.dispose(carry);
  try {
    return np.add(ys.pred, ys.pred);
  } finally {
    tree.dispose(ys);
  }
}

// ✅ OK: caller owns returned values
function forward(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  return { carry, ys };
}

// ✅ OK: ownership transferred to disposable wrapper
function owned(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  return tree.makeDisposable({ carry, ys });
}
```

### `jax-js/require-retained-release` (internal)

Flags retained handles created from `.ref` that never show a terminal action in scope.

Accepted terminals include `.dispose()`, transfer as function argument, return/yield, and ownership
handoff via assignment/object/array construction.

### `jax-js/require-try-finally-symmetry` (internal)

Flags `.ref` temporaries created inside `try` blocks when cleanup is not guaranteed in the matching
`finally` block.

### `jax-js/require-wrapper-dispose-symmetry` (internal)

In wrapper `dispose()` methods, enforces retained-state cleanup before `this.inner.dispose()`. If
`this.inner.dispose()` appears, it must be the last `.dispose()` call in the method body.

### Recommended ownership patterns

```ts
// Pattern 1: short-lived intermediates
using tmp = np.add(x, y);
return tmp.sum();

// Pattern 2: scan outputs disposed in same scope
const [carry, ys] = lax.scan(step, initCarry, xs);
tree.dispose(carry);
try {
  return np.add(ys.pred, ys.pred);
} finally {
  tree.dispose(ys);
}

// Pattern 3: ownership handoff to disposable wrapper
const [carry2, ys2] = lax.scan(step, initCarry, xs);
using owned = tree.makeDisposable({ carry: carry2, ys: ys2 });
return await owned.carry.data(); // read data; owned disposes all leaves at block end

// Pattern 4: avoid accidental aliasing in handoff literals
const xf_0 = np.add(x, y);
const yhat = np.add(x, y);
using out = tree.makeDisposable({ xf_0, yhat });
```

### Svelte note (`.svelte`, `.svelte.ts`)

Svelte currently does not parse `using` declarations in component scripts.

- Prefer extracting compute logic to plain `.ts` modules where `using` works.
- In component code, use explicit `.dispose()` with clear ownership handoff guards.
- Apply `require-using` primarily to non-Svelte source files.

## Suppressing warnings

Sometimes a warning is intentional (e.g., you knowingly keep an array alive in a cache). In those
cases, prefer disabling the specific rule as locally as possible, and include a short reason.

### Plugin-specific directives

Some rules support purpose-built suppression comments placed on the line immediately before the
flagged code:

| Rule                 | Suppression comment               | Scope       |
| -------------------- | --------------------------------- | ----------- |
| `require-using`      | `// jax-js-lint: allow-non-using` | Next line   |
| `no-unnecessary-ref` | `// jax-js-lint: allow-ref`       | Next line   |
| `no-unnecessary-ref` | `// jax-js-lint: allow-ref` (top) | Entire file |

### Standard ESLint directives

All rules support the standard ESLint disable comments:

**Single line:**

```ts
// eslint-disable-next-line jax-js/require-using -- intentionally leaked until process exit
const cached = np.zeros([1024]);
```

**Block:**

```ts
/* eslint-disable jax-js/require-using -- constructing a global cache */
const CACHE_A = np.zeros([128]);
const CACHE_B = np.ones([128]);
/* eslint-enable jax-js/require-using */
```

**Config-level:**

```ts
// eslint.config.ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  jaxJs.configs.recommended,
  {
    rules: {
      "jax-js/no-array-chain": "off", // turn off a rule entirely
    },
  },
];
```

## FAQ

### "I'm inside a `grad` / `valueAndGrad` / `jit` body — aren't arrays managed by the tracer?"

No. **The ownership rules are identical in eager and traced contexts.** `using` is safe and
recommended everywhere, including inside `grad`, `valueAndGrad`, `jit`, and `lax.scan` step
functions. The library itself uses `using` extensively inside traced bodies (`lax.ts`, `random.ts`,
`numpy-fft.ts`).

What each tracer type does when `[Symbol.dispose]()` is called:

| Context                                       | Tracer type   | `dispose()` behaviour                                                                                                                                         |
| --------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inside `jit(...)` body                        | `JaxprTracer` | **No-op.** Harmless.                                                                                                                                          |
| Inside `grad(...)` / `valueAndGrad(...)` body | `JVPTracer`   | Decrements its own `#rc`. When rc hits 0, primal + tangent are freed — **correct**, because the operation that used them has already captured what it needed. |
| Concrete `Array` (captured constant / input)  | `Array`       | Normal reference-counted disposal — **correct and required**.                                                                                                 |

The critical distinction: `using` on a _named_ intermediate is always correct because you own that
intermediate. The `no-array-chain` rule does suppress warnings for _unnamed_ chain intermediates
inside traced bodies (because chains create anonymous temporaries that can't be `using`-managed),
but that is a separate rule with a separate rationale.

**The rule of thumb:** write ownership the same way regardless of whether your function is wrapped
by `grad`, `jit`, or called directly. `jit()` is a pure performance optimisation — it must not
change ownership semantics, and correct code must work identically in both modes.

```ts
// ✅ Correct inside grad body — identical to eager mode
const loss = (x: np.Array) => {
  using I = np.eye(x.shape[0]);
  using Ax = np.matmul(x, I);
  return Ax.sum();
};
const dx = grad(loss)(x);

// ❌ Incorrect workaround — silences the lint warning but leaks in eager mode
/* eslint-disable jax-js/require-using */
const loss = (x: np.Array) => {
  const I = np.eye(x.shape[0]); // leaks if loss() is ever called outside grad
  const Ax = np.matmul(x, I);
  return Ax.sum();
};
```

### "`using` inside a Kalman filter / scan step function seems excessive"

It is not — the scan step function is called with real `Array` objects in eager mode (or when
compiling the JIT body). Each intermediate is a real GPU/WASM buffer. Without `using`, the buffers
live until garbage collection, which can exhaust WASM heap or GPU memory on long scans.

For a scan body with a tight inner structure, `using` keeps peak memory proportional to the step
code rather than the number of iterations.

### "I got a `UseAfterFreeError` when I added `using` to a traced body"

This means you disposed something that was still needed — not a tracing issue but a real ownership
bug in the step function itself. Check whether you wrote `using result = ...` and then returned
`result`: `using` disposes at block end, which is before the return value is consumed. Rename to
`const result = ...` (which the rule will not flag because it is returned) or restructure to return
before `using` scope ends.

```ts
// ❌ Disposes before caller can use it
const step = (carry, x) => {
  using newCarry = np.add(carry, x);
  return [newCarry, newCarry]; // newCarry already disposed here!
};

// ✅ Correct — returned value is not `using`-managed
const step = (carry, x) => {
  const newCarry = np.add(carry, x); // fine: returned immediately
  return [newCarry, newCarry];
};
```

## How it works

The plugin uses **heuristic static analysis** — no import resolution, no type information. It
recognizes array-producing patterns by:

1. **Factory names:** `array`, `zeros`, `ones`, `full`, `arange`, `linspace`, etc.
2. **Method names:** `add`, `sub`, `mul`, `div`, `reshape`, `transpose`, `sum`, etc.
3. **Namespace prefixes:** When the receiver is `np`, `numpy`, `lax`, `nn`, `random`, `jax`, etc.
4. **Call chain depth:** Counts consecutive array method calls for `no-array-chain`.

This means the rules work without a TypeScript type-checker running, keeping lint fast (~1 s for the
full jax-js codebase). The tradeoff is occasional false positives on non-jax-js code that happens to
use the same method names — suppress those with the [directives above](#suppressing-warnings).

### Design decisions & known limitations

- **Heuristic-based, no import tracking.** The rules identify jax-js arrays by recognizing factory
  calls, method names, and namespace prefixes. They do not resolve imports, so:
  - Aliased imports (`import { array as arr }`) — factory name `arr` won't be recognized.
  - Dynamic method calls (`x[methodName]()`) — not tracked.
  - This is conservative overall — it avoids false positives at the cost of occasional false
    negatives for unusual import patterns.
- **Function-local scope only.** `no-use-after-dispose` tracks variables within a single function.
  Cross-function disposal (e.g., disposing in a helper, using in the caller) is not detected.
- **Adjacent-statement pattern for `no-dispose-then-reassign-param`.** The rule detects the
  `dispose(x); x = param;` pattern only when the two statements are adjacent in a function body.
  Re-assignment separated by intervening statements is not caught.
- **`no-array-chain` only tracks known JAX methods.** JavaScript collection methods like `.map()`,
  `.filter()`, `.reduce()` are ignored — only JAX array methods increase the chain depth counter.

## IDE integration

These lint rules are designed to give you immediate feedback as you write jax-js code. Any editor or
IDE that supports ESLint will show warnings inline (red/yellow squiggles) and offer quick-fix
suggestions.

### VS Code

1. Install the
   [ESLint extension](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)
   (`dbaeumer.vscode-eslint`, v3+).
2. The extension auto-detects your `eslint.config.ts` — no extra configuration needed.
3. You will see inline warnings for leak patterns, and **Code Actions** (💡 lightbulb) to apply
   autofixes and suggestions.

Optional settings for a better experience (add to `.vscode/settings.json`):

```jsonc
{
  // Lint as you type for immediate feedback (use "onSave" if you prefer less frequent diagnostics)
  "eslint.run": "onType",

  // Validate TypeScript and JavaScript files
  "eslint.validate": ["typescript", "javascript", "typescriptreact", "javascriptreact"],

  // Auto-fix fixable rules on save (e.g., removes unnecessary .ref)
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
  },
}
```

### WebStorm / IntelliJ IDEA

1. Go to **Settings → Languages & Frameworks → JavaScript → Code Quality Tools → ESLint**.
2. Select **Automatic ESLint configuration** (or point to your config manually).
3. Check **Run eslint --fix on save** if you want autofixes applied automatically.
4. Warnings from `@hamk-uas/eslint-plugin-jax-js` will appear inline in the editor.

### Neovim

If you use [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig) with the ESLint language
server:

```lua
-- In your Neovim LSP config
require('lspconfig').eslint.setup({
  -- ESLint will automatically pick up eslint.config.ts
})
```

Or with [none-ls](https://github.com/nvimtools/none-ls.nvim) /
[efm-langserver](https://github.com/mattn/efm-langserver), configure ESLint as a diagnostics source.

### Sublime Text

Install [SublimeLinter](https://github.com/SublimeLinter/SublimeLinter) and
[SublimeLinter-eslint](https://github.com/SublimeLinter/SublimeLinter-eslint). The plugin will be
picked up automatically from your ESLint config.

### Command line

```bash
npx eslint src/
npx eslint --rule 'jax-js/require-using:error' src/my-model.ts
```

Or add it as a script in your `package.json`:

```json
{
  "scripts": {
    "lint": "eslint src/"
  }
}
```

### Troubleshooting

**Diagnostics not appearing in VS Code:**

1. Run `ESLint: Restart ESLint Server` from the command palette (Ctrl/Cmd+Shift+P).
2. Check the **ESLint** output channel (View → Output → select "ESLint") for errors.
3. Ensure your `eslint.config.ts` has `files` patterns that match your `.ts` files.
4. Reload the window (Developer: Reload Window).

**`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING` when loading the plugin:**

This error occurs on Node 22+ when a `.ts` file is loaded from inside `node_modules/`. Since v0.2.2,
the plugin's `package.json` exports point to pre-built `dist/` files, so this should not happen. If
you see this error, ensure you have the latest version installed (`prepare` must run successfully to
build `dist/`).

**Rules fire on non-jax-js code:**

The rules use heuristic pattern matching and may flag variables from non-jax-js libraries that
happen to use the same method names. Narrow the `files` glob to your jax-js source directories, or
suppress individual false positives with `eslint-disable-next-line`.

## Comparison with `@hamk-uas/eslint-plugin-jax-js`

The community [`@hamk-uas/eslint-plugin-jax-js`](https://github.com/hamk-uas/eslint-plugin-jax-js)
plugin targets the **upstream move-semantics** jax-js. This plugin targets the **non-consuming
fork** where operations leave inputs alive.

| Aspect                | `@hamk-uas/eslint-plugin-jax-js`               | `@hamk-uas/eslint-plugin-jax-js`                        |
| --------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| Ownership model       | Move semantics (consuming)                     | Non-consuming                                           |
| `.ref` guidance       | Sometimes necessary                            | Never needed in user code                               |
| Dispose terminology   | "consume" (input consumed by op)               | "dispose" (explicit cleanup)                            |
| `using` support       | Not mentioned                                  | First-class (`require-using`)                           |
| Chain detection       | No                                             | `no-array-chain` rule                                   |
| Flat config (`>=v9`)  | Yes (`configs.recommended`)                    | Yes (`configs.recommended`)                             |
| Suggested fixes       | `require-consume`, `no-use-after-consume`      | `require-using`                                         |
| Autofix               | `no-unnecessary-ref`                           | `no-unnecessary-ref`                                    |
| Dispose line in error | `no-use-after-consume` includes consuming line | `no-use-after-dispose` includes consuming method + line |
| Suppression directive | `// @jax-borrow`                               | `// jax-js-lint: allow-*`                               |

If you're using the upstream jax-js (move semantics), use the HAMK plugin. If you're using this fork
(non-consuming model with `using`), use this plugin.

## Contributing

The plugin lives in `packages/eslint-plugin/` within the jax-js monorepo.

```bash
# Run plugin rules against the codebase
pnpm exec eslint src/ packages/

# Test a specific rule in strict mode
pnpm exec eslint --rule 'jax-js/no-array-chain:error' src/library/numpy.ts

# Verify zero violations on strict sweep
pnpm exec eslint --rule 'jax-js/require-using:error' \
                  --rule 'jax-js/no-array-chain:error' \
                  --rule 'jax-js/no-use-after-dispose:error' \
                  --rule 'jax-js/no-dispose-then-reassign-param:error' \
                  --rule 'jax-js/no-unnecessary-ref:error' \
                  src/library/
```

Rule source files are in `packages/eslint-plugin/src/rules/`. Each rule is a single file exporting
an ESLint `Rule.RuleModule`.

## License

MIT
