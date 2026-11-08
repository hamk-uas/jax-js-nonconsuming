import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

import {
  BROWSER_MAX_WORKERS,
  chromiumLaunchOptions,
  describeGpuSetup,
  detectGpu,
  requiredDevices,
  workspaceAliases,
} from "./test/gpu-config";

const gpu = detectGpu();
describeGpuSetup(gpu);

export default defineConfig({
  resolve: {
    // Workspace packages resolve to TypeScript sources; see test/gpu-config.ts.
    alias: workspaceAliases,
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
      headless: true,
      screenshotFailures: false,
      // No display server, one Vulkan driver; see chromiumLaunchOptions().
      provider: playwright({ launchOptions: chromiumLaunchOptions(gpu) }),
      instances: [{ browser: "chromium" }],
    },
    coverage: {
      // coverage is disabled by default, run with `pnpm test:coverage`.
      enabled: false,
      provider: "v8",
    },
    // Files run concurrently, each in its own iframe with its own module state
    // and GPUDevice; they share the Chromium GPU process, which bounds useful
    // concurrency (see BROWSER_MAX_WORKERS).
    maxWorkers: BROWSER_MAX_WORKERS,
    // A detected GPU must initialize; see requiredDevices().
    provide: { requiredDevices: requiredDevices(gpu) },
    testTimeout: 10000,
    passWithNoTests: true,
    exclude: [
      "**/node_modules/**",
      "**/dist/**",
      "tmp/**",
      // Performance benchmarks, not correctness tests:
      "test/scan-bench.test.ts",
      "test/bench-phase3.test.ts",
      "test/bench-assoc-scale.test.ts",
      // GPU-specific benchmarks (run with scripts/gpu-test.sh):
      "test/gpu-bench.test.ts",
      "test/debug-wgsl.test.ts",
      // WebGPU-only tests — must run under GPU configs, not the default suite:
      "test/o9a-arena.test.ts",
      "test/o9c-slab.test.ts",
      "test/command-tape-gpu.test.ts",
      "test/repro-stub.test.ts",
      "test/pool-memory.test.ts",
      "test/blocked-data-movement.test.ts",
      "test/vitest.nvidia.config.ts",
      "test/vitest.intel.config.ts",
    ],
    setupFiles: ["test/setup.ts"],
  },
});
