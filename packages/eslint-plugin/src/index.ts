import type { ESLint, Linter } from "eslint";

import noArrayChain from "./rules/no-array-chain";
import noDisposeThenReassignParam from "./rules/no-dispose-then-reassign-param";
import noMakeDisposableAlias from "./rules/no-make-disposable-alias";
import noNestedArrayLeak from "./rules/no-nested-array-leak";
import noUnnecessaryRef from "./rules/no-unnecessary-ref";
import noUseAfterDispose from "./rules/no-use-after-dispose";
import requireRetainedRelease from "./rules/require-retained-release";
import requireScanResultDispose from "./rules/require-scan-result-dispose";
import requireTryFinallySymmetry from "./rules/require-try-finally-symmetry";
import requireUsing from "./rules/require-using";
import requireWrapperDisposeSymmetry from "./rules/require-wrapper-dispose-symmetry";

const plugin: ESLint.Plugin = {
  meta: {
    name: "@hamk-uas/eslint-plugin-jax-js",
    version: "0.1.2",
  },
  rules: {
    "require-using": requireUsing,
    "no-use-after-dispose": noUseAfterDispose,
    "no-dispose-then-reassign-param": noDisposeThenReassignParam,
    "no-make-disposable-alias": noMakeDisposableAlias,
    "no-unnecessary-ref": noUnnecessaryRef,
    "no-array-chain": noArrayChain,
    "no-nested-array-leak": noNestedArrayLeak,
    "require-scan-result-dispose": requireScanResultDispose,
    "require-retained-release": requireRetainedRelease,
    "require-try-finally-symmetry": requireTryFinallySymmetry,
    "require-wrapper-dispose-symmetry": requireWrapperDisposeSymmetry,
  },
};

// ---------------------------------------------------------------------------
// Shared configs
// ---------------------------------------------------------------------------

/**
 * Recommended config — catches ownership bugs without being overly strict.
 *
 * - `require-using`: warn
 * - `no-use-after-dispose`: error
 * - `no-unnecessary-ref`: warn
 * - `no-array-chain`: off
 */
const recommended: Linter.Config = {
  plugins: { "jax-js": plugin },
  rules: {
    "jax-js/require-using": "warn",
    "jax-js/no-use-after-dispose": "error",
    "jax-js/no-dispose-then-reassign-param": "warn",
    "jax-js/no-make-disposable-alias": "warn",
    "jax-js/no-unnecessary-ref": "warn",
    "jax-js/no-array-chain": "off",
    "jax-js/no-nested-array-leak": "error",
    "jax-js/require-scan-result-dispose": "warn",
  },
};

/**
 * Strict config — everything in recommended, plus chain-splitting enforcement.
 *
 * - `require-using`: error
 * - `no-use-after-dispose`: error
 * - `no-unnecessary-ref`: error
 * - `no-array-chain`: error
 */
const strict: Linter.Config = {
  plugins: { "jax-js": plugin },
  rules: {
    "jax-js/require-using": "error",
    "jax-js/no-use-after-dispose": "error",
    "jax-js/no-dispose-then-reassign-param": "error",
    "jax-js/no-make-disposable-alias": "error",
    "jax-js/no-unnecessary-ref": "error",
    "jax-js/no-array-chain": "error",
    "jax-js/no-nested-array-leak": "error",
    "jax-js/require-scan-result-dispose": "error",
  },
};

/**
 * Invariance config — enforces ownership patterns that should behave the same
 * in eager mode and inside `jit()` bodies.
 *
 * Use this for user code when you want explicit eager/JIT semantic invariance
 * gates in CI.  Does not include `no-array-chain` because chains affect only
 * performance (eager peak memory), not correctness.
 */
const invariance: Linter.Config = {
  plugins: { "jax-js": plugin },
  rules: {
    "jax-js/require-using": "error",
    "jax-js/no-use-after-dispose": "error",
    "jax-js/no-dispose-then-reassign-param": "error",
    "jax-js/no-make-disposable-alias": "error",
    "jax-js/no-unnecessary-ref": "error",
    "jax-js/no-nested-array-leak": "error",
    "jax-js/require-scan-result-dispose": "error",
  },
};

/**
 * Internal transform-ownership config — high-signal checks for wrapper/retention
 * symmetry in framework internals (e.g. transform plumbing).
 */
const internalTransforms: Linter.Config = {
  plugins: { "jax-js": plugin },
  rules: {
    "jax-js/require-retained-release": "warn",
    "jax-js/require-try-finally-symmetry": "warn",
    "jax-js/require-wrapper-dispose-symmetry": "warn",
  },
};

// Attach configs to plugin object for flat-config consumers:
//   import jaxJs from "@hamk-uas/eslint-plugin-jax-js";
//   export default [ jaxJs.configs.recommended, ... ];
(plugin as any).configs = {
  recommended,
  strict,
  invariance,
  internalTransforms,
};

export default plugin;
export { recommended, strict, invariance, internalTransforms };
