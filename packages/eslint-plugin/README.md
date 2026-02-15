# @jax-js/eslint-plugin

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
pnpm add -D @jax-js/eslint-plugin
```

**Requirements:**

- ESLint ≥ 9.0 (flat config)
- TypeScript source via [jiti](https://github.com/unjs/jiti) (ESLint loads `.ts` config files
  automatically since v9.7)

## Quick start

### Recommended config (one line)

```ts
// eslint.config.ts
import jaxJs from "@jax-js/eslint-plugin";

export default [
  jaxJs.configs.recommended,
  // ... your other configs
];
```

This enables:

| Rule                          | Level   | What it catches                               |
| ----------------------------- | ------- | --------------------------------------------- |
| `jax-js/require-using`        | `warn`  | Local array bindings missing `using`          |
| `jax-js/no-use-after-dispose` | `error` | Reading/writing a variable after `.dispose()` |
| `jax-js/no-unnecessary-ref`   | `warn`  | `.ref` calls (unnecessary in non-consuming)   |
| `jax-js/no-array-chain`       | `off`   | Deep fluent chains (strict-mode only)         |

### Strict config

```ts
// eslint.config.ts
import jaxJs from "@jax-js/eslint-plugin";

export default [
  jaxJs.configs.strict,
  // ... your other configs
];
```

Strict promotes everything to `error` and enables `no-array-chain` — useful for library code and CI
enforcement.

### Individual rule control

```ts
// eslint.config.ts
import jaxJs from "@jax-js/eslint-plugin";

export default [
  {
    plugins: { "jax-js": jaxJs },
    rules: {
      "jax-js/require-using": "error",
      "jax-js/no-use-after-dispose": "error",
      "jax-js/no-unnecessary-ref": "warn",
      "jax-js/no-array-chain": ["error", { minDepth: 3 }],
    },
  },
];
```

### Limit to specific directories

```ts
// eslint.config.ts
import jaxJs from "@jax-js/eslint-plugin";

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
      "jax-js/no-unnecessary-ref": "off",
      "jax-js/no-array-chain": "off",
    },
  },
];
```

## Rules

### `jax-js/require-using`

**Fixable:** suggestion (auto-convertible via IDE quick-fix)

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

### `jax-js/no-use-after-dispose`

**Type:** problem (error by default)

Catches reads or writes to a variable after `.dispose()` has been called on it. This prevents
`UseAfterFreeError` at runtime. The error message includes the line number of the `.dispose()` call
for easy navigation.

```ts
const x = np.array([1, 2, 3]);
x.dispose(); // line 2

// ❌ Error: `x` is used after `.dispose()` on line 2
const y = x.add(np.array([4, 5, 6]));

// ✅ OK: redundant dispose is fine (no-op at rc=0)
x.dispose();
```

**Scope:** Tracks variables lexically within the same function scope. Does not track across function
boundaries or through aliases.

### `jax-js/no-unnecessary-ref`

**Fixable:** autofix (removes `.ref` automatically with `--fix`)

Flags `.ref` property access. In the non-consuming ownership model, `.ref` is never needed in user
code — operations do not consume inputs. The only legitimate uses are deep inside the framework
internals.

```ts
// ❌ Error: `.ref` is usually unnecessary in non-consuming jax-js
const copy = myArray.ref;

// ✅ OK: suppressed for framework internals
// jax-js-lint: allow-ref
const copy = myArray.ref;
```

### `jax-js/no-array-chain`

**Type:** suggestion (off by default, enabled in strict config)

Flags deep fluent method chains that create unnamed eager-mode temporaries. These intermediates
can't be `using`-managed and may accumulate in GPU memory until GC runs.

Only the outermost chain is reported — inner subchains don't produce duplicate diagnostics.

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

**Scope:** Only tracks known JAX array methods (`add`, `sub`, `mul`, `div`, `reshape`, `transpose`,
`sum`, `mean`, `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `matmul`, etc.). JavaScript collection
methods like `.map()`, `.filter()`, `.reduce()` are ignored.

## Suppression directives

Each rule supports its own suppression comment placed on the line immediately before the flagged
code:

| Rule                   | Suppression comment                                       |
| ---------------------- | --------------------------------------------------------- |
| `require-using`        | `// jax-js-lint: allow-non-using`                         |
| `no-unnecessary-ref`   | `// jax-js-lint: allow-ref`                               |
| `no-use-after-dispose` | `// eslint-disable-next-line jax-js/no-use-after-dispose` |
| `no-array-chain`       | `// eslint-disable-next-line jax-js/no-array-chain`       |

Standard `eslint-disable` directives work for all rules as well:

```ts
// eslint-disable-next-line jax-js/require-using
const x = np.array([1, 2, 3]);
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
use the same method names — suppress those with the directives above.

**Known heuristic boundaries:**

- Aliased imports (`import { array as arr }`) — factory name `arr` won't be recognized
- Dynamic method calls (`x[methodName]()`) — not tracked
- Cross-function disposal tracking — `no-use-after-dispose` is function-local only
- Re-assignment after dispose — not tracked (would need control-flow analysis)

## IDE integration

### VS Code

Install the
[ESLint extension](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)
(v3+). It picks up flat configs automatically. No extra settings needed — warnings and errors appear
inline with quick-fix suggestions for `require-using`.

### CLI

```bash
npx eslint src/
npx eslint --rule 'jax-js/require-using:error' src/my-model.ts
```

## Comparison with `@hamk-uas/eslint-plugin-jax-js`

The community [`@hamk-uas/eslint-plugin-jax-js`](https://github.com/hamk-uas/eslint-plugin-jax-js)
plugin targets the **upstream move-semantics** jax-js. This plugin targets the **non-consuming
fork** where operations leave inputs alive.

| Aspect                | `@hamk-uas/eslint-plugin-jax-js`               | `@jax-js/eslint-plugin`                      |
| --------------------- | ---------------------------------------------- | -------------------------------------------- |
| Ownership model       | Move semantics (consuming)                     | Non-consuming                                |
| `.ref` guidance       | Sometimes necessary                            | Never needed in user code                    |
| Dispose terminology   | "consume" (input consumed by op)               | "dispose" (explicit cleanup)                 |
| `using` support       | Not mentioned                                  | First-class (`require-using`)                |
| Chain detection       | No                                             | `no-array-chain` rule                        |
| Flat config (`>=v9`)  | Yes (`configs.recommended`)                    | Yes (`configs.recommended`)                  |
| Suggested fixes       | `require-consume`, `no-use-after-consume`      | `require-using`                              |
| Autofix               | `no-unnecessary-ref`                           | `no-unnecessary-ref`                         |
| Dispose line in error | `no-use-after-consume` includes consuming line | `no-use-after-dispose` includes dispose line |
| Suppression directive | `// @jax-borrow`                               | `// jax-js-lint: allow-*`                    |

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
                  --rule 'jax-js/no-unnecessary-ref:error' \
                  src/library/
```

Rule source files are in `packages/eslint-plugin/src/rules/`. Each rule is a single file exporting
an ESLint `Rule.RuleModule`.

## License

MIT
