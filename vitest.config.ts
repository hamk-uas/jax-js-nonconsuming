import path from "node:path";

import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      // Resolve workspace sub-packages to their TypeScript sources so that
      // tests work without a prior `pnpm build` step.  The root
      // @hamk-uas/jax-js-nonconsuming package is NOT aliased here because tests in
      // src/ use relative imports — aliasing it would create duplicate module
      // instances and break leak detection.
      "@hamk-uas/jax-js-nonconsuming-optax": path.resolve(
        __dirname,
        "packages/optax/src/index.ts",
      ),
      "@hamk-uas/jax-js-nonconsuming-onnx": path.resolve(
        __dirname,
        "packages/onnx/src/index.ts",
      ),
      "@hamk-uas/jax-js-nonconsuming-loaders": path.resolve(
        __dirname,
        "packages/loaders/src/index.ts",
      ),
    },
  },
  esbuild: {
    supported: {
      using: false, // Needed to lower 'using' statements in tests.
    },
  },
  test: {
    watch: false, // Run once and exit, don't wait for 'q'
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright(),
      // https://vitest.dev/config/browser/playwright.html
      instances: [{ browser: "chromium" }],
    },
    coverage: {
      // coverage is disabled by default, run with `pnpm test:coverage`.
      enabled: false,
      provider: "v8",
    },
    passWithNoTests: true,
    exclude: ["**/node_modules/**", "**/dist/**", "test/deno/**", "tmp/**"],
    setupFiles: ["test/setup.ts"],
  },
});
