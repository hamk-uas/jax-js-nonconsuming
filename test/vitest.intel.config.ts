/**
 * Vitest config that forces Intel integrated GPU.
 *
 * Uses VK_DRIVER_FILES to bypass the NVIDIA discrete GPU and force
 * Chromium's Vulkan backend to use the Intel mesa driver. Adjust the
 * path if your system uses a different icd.d location.
 *
 * Usage:
 *   pnpm vitest run test/intel-matmul.test.ts -c test/vitest.intel.config.ts --testTimeout=300000
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
            "--use-vulkan=native",
            "--force-gpu-mem-available-mb=4096",
          ],
          env: {
            ...process.env,
            // Force Vulkan loader to only load Intel driver
            VK_DRIVER_FILES: "/usr/share/vulkan/icd.d/intel_icd.json",
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
