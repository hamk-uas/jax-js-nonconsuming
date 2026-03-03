/**
 * Vitest config for NVIDIA GPU benchmarks.
 *
 * Uses Vulkan backend with f16 support enabled via Dawn feature flag.
 * On multi-GPU systems, this typically selects the NVIDIA discrete GPU
 * (Chromium's Vulkan adapter priority prefers discrete over integrated).
 *
 * Usage:
 *   pnpm vitest run test/intel-matmul.test.ts -c test/vitest.nvidia.config.ts --testTimeout=300000
 */
import path from "node:path";

import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@hamk-uas/jax-js-nonconsuming": path.resolve(
        __dirname,
        "..",
        "src/index.ts",
      ),
    },
  },
  esbuild: {
    supported: { using: false },
  },
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  test: {
    watch: false,
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright({
        launchOptions: {
          args: [
            "--no-sandbox",
            "--headless=new",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--disable-vulkan-surface",
            "--enable-unsafe-webgpu",
            "--enable-dawn-features=vulkan_enable_f16_on_nvidia",
          ],
          env: {
            ...process.env,
          },
        },
      }),
      instances: [{ browser: "chromium" }],
    },
    testTimeout: 300000,
    passWithNoTests: true,
    setupFiles: [],
  },
});
