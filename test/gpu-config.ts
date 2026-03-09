/**
 * Shared GPU vitest config factory.
 *
 * Both NVIDIA and Intel GPU configs use this factory so that launch args,
 * server headers, aliases, and timeouts stay in sync.
 *
 * Usage from a GPU config:
 *   import { gpuConfig } from "./gpu-config";
 *   export default gpuConfig("nvidia");
 */
import path from "node:path";

import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

/** Chromium args shared by all GPU configs. */
const COMMON_ARGS = [
  "--no-sandbox",
  "--headless=new",
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--disable-vulkan-surface",
  "--enable-unsafe-webgpu",
];

/** Per-GPU overrides: extra Chromium args and env vars. */
const GPU_PROFILES = {
  nvidia: {
    args: ["--enable-dawn-features=vulkan_enable_f16_on_nvidia"],
    env: {},
  },
  intel: {
    args: ["--use-vulkan=native", "--force-gpu-mem-available-mb=4096"],
    env: {
      // Force Vulkan loader to only load the Intel mesa driver.
      VK_DRIVER_FILES: "/usr/share/vulkan/icd.d/intel_icd.json",
    },
  },
} as const;

export type GpuProfile = keyof typeof GPU_PROFILES;

/**
 * Build a complete vitest config for the given GPU.
 *
 * The config disables leak-checking (`setupFiles: []`) and uses a 300 s
 * timeout — GPU benchmarks are long-running and allocate freely.
 */
export function gpuConfig(gpu: GpuProfile) {
  const profile = GPU_PROFILES[gpu];

  return defineConfig({
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
            args: [...COMMON_ARGS, ...profile.args],
            env: {
              DISPLAY: process.env.DISPLAY ?? ":0",
              XAUTHORITY:
                process.env.XAUTHORITY ??
                `/run/user/${process.getuid?.() ?? 1000}/gdm/Xauthority`,
              ...profile.env,
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
}
