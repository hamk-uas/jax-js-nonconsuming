import type { ESLint, Linter } from "eslint";

import noArrayChain from "./rules/no-array-chain";
import noDisposeThenReassignParam from "./rules/no-dispose-then-reassign-param";
import noUnnecessaryRef from "./rules/no-unnecessary-ref";
import noUseAfterDispose from "./rules/no-use-after-dispose";
import requireScanResultDispose from "./rules/require-scan-result-dispose";
import requireUsing from "./rules/require-using";

const plugin: ESLint.Plugin = {
  meta: {
    name: "@jax-js/eslint-plugin",
    version: "0.1.0",
  },
  rules: {
    "require-using": requireUsing,
    "no-use-after-dispose": noUseAfterDispose,
    "no-dispose-then-reassign-param": noDisposeThenReassignParam,
    "no-unnecessary-ref": noUnnecessaryRef,
    "no-array-chain": noArrayChain,
    "require-scan-result-dispose": requireScanResultDispose,
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
    "jax-js/no-unnecessary-ref": "warn",
    "jax-js/no-array-chain": "off",
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
    "jax-js/no-unnecessary-ref": "error",
    "jax-js/no-array-chain": "error",
    "jax-js/require-scan-result-dispose": "error",
  },
};

// Attach configs to plugin object for flat-config consumers:
//   import jaxJs from "@jax-js/eslint-plugin";
//   export default [ jaxJs.configs.recommended, ... ];
(plugin as any).configs = { recommended, strict };

export default plugin;
export { recommended, strict };
