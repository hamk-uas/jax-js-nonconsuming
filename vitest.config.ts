import path from "node:path";

import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      // Resolve ALL workspace packages to their TypeScript sources so that
      // tests work without a prior `pnpm build` step.  The root package
      // alias is essential: without it, test/*.test.ts imports resolve to
      // dist/index.js while src/*.test.ts uses relative imports, creating
      // two module graphs where `instanceof Tracer` fails and leak
      // detection sees mismatched singletons.
      "@hamk-uas/jax-js-nonconsuming": path.resolve(__dirname, "src/index.ts"),
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
  // SharedArrayBuffer + Worker features (M5 WasmWorkerPool, M6.2b
  // OrchestratorWorker) require crossOriginIsolated in Chromium, which needs
  // COOP + COEP response headers.  Unfortunately those headers break vitest's
  // iframe-based test runner — the test page never connects back.  No
  // Chromium flag combination can bypass this.  Therefore SAB features are
  // tested exclusively via the Deno test suite (`pnpm run test:deno`), which
  // has native SharedArrayBuffer support.  The vitest suite exercises all
  // non-parallel WASM paths.
  test: {
    watch: false, // Run once and exit, don't wait for 'q'
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright(),
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
