import type { ESLint } from "eslint";

import noArrayChain from "./rules/no-array-chain";
import noUnnecessaryRef from "./rules/no-unnecessary-ref";
import noUseAfterDispose from "./rules/no-use-after-dispose";
import requireUsing from "./rules/require-using";

const plugin: ESLint.Plugin = {
  meta: {
    name: "@jax-js/eslint-plugin",
    version: "0.0.0",
  },
  rules: {
    "require-using": requireUsing,
    "no-use-after-dispose": noUseAfterDispose,
    "no-unnecessary-ref": noUnnecessaryRef,
    "no-array-chain": noArrayChain,
  },
};

export default plugin;
