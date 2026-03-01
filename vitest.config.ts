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
  // COOP + COEP response headers.  Vitest browser mode supports these via
  // server.headers since vitest-dev/vitest#4890 (Vitest ≥1.2).
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  test: {
    watch: false, // Run once and exit, don't wait for 'q'
    browser: {
      enabled: true,
      headless: false,
      screenshotFailures: false,
      provider: playwright({
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--no-sandbox",
          ],
          env: {
            DISPLAY: process.env.DISPLAY ?? ":0",
            XAUTHORITY:
              process.env.XAUTHORITY ??
              `/run/user/${process.getuid?.() ?? 1000}/gdm/Xauthority`,
          },
        },
      }),
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
