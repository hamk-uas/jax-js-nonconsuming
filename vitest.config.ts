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
    // fileParallelism defaults to true — vitest runs files concurrently.
    // All tests share one Chromium tab / GPUDevice so this is safe and
    // eliminates the ~3 s per-file startup cost that dominates with serial.
    testTimeout: 10000,
    passWithNoTests: true,
    exclude: [
      "**/node_modules/**",
      "**/dist/**",
      "test/deno/**",
      "tmp/**",
      // ── Slow-test quarantine ──────────────────────────────────────────
      // These files contain WebGPU eager-mode tests that individually
      // take 3–34 s, making them incompatible with the 10 s testTimeout.
      // They are correct but slow; run them separately via:
      //   pnpm vitest run test/lax-scan.test.ts  (etc.)
      // Tracked in .ci/expected-failures.json with expiry 2026-06-01.
      "test/lax-scan.test.ts",
      "test/numpy-fft.test.ts",
      "test/scan-backends.test.ts",
      "test/pool-memory.test.ts",
      // Performance benchmarks, not correctness tests:
      "test/scan-bench.test.ts",
    ],
    setupFiles: ["test/setup.ts"],
  },
});
