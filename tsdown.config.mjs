import { readdir } from "node:fs/promises";

import { defineConfig } from "tsdown";

const watchMode = process.env.TSDOWN_WATCH_MODE === "1";

// Common options for all packages.
/** @type {import("tsdown").Options} */
const opts = {
  deps: {
    // Bundle nothing from npm, except the runtime helpers that the compiler
    // and bundler toolchain inject.
    neverBundle: true,
    alwaysBundle: [/^@oxc-project\/runtime/],
  },
  format: ["cjs", "esm"],
  platform: "browser",
  // tsdown now builds CJS for Node.js unless told otherwise. Keep both
  // formats on the browser platform, as older tsdown did; the output is
  // currently identical either way.
  inputOptions: { platform: "browser" },
  dts: watchMode
    ? {
        // Without newContext, dts generation in watch mode has trouble updating
        // when files are changed.
        // https://github.com/rolldown/tsdown/issues/396
        newContext: true,
      }
    : true,
};

export default defineConfig([
  {
    name: "jax",
    ...opts,
    watch: watchMode && "src",
  },

  ...(await readdir("packages")).map((pkg) => ({
    name: pkg,
    ...opts,
    cwd: `packages/${pkg}`,
    watch: watchMode && `packages/${pkg}/src`, // Unaffected by cwd.
  })),
]);
