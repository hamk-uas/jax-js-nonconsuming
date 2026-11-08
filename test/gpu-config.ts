/**
 * Shared browser-test config: workspace aliases, Chromium environment, and the
 * GPU-specific vitest config factory.
 *
 * `vitest.config.ts` (default suite) and the per-GPU configs both use the
 * aliases and `chromiumEnv()` from here so they cannot drift apart.
 *
 * Usage from a GPU config:
 *   import { gpuConfig } from "./gpu-config";
 *   export default gpuConfig("nvidia");
 */
import fs from "node:fs";
import path from "node:path";

import { playwright } from "@vitest/browser-playwright";
import { configDefaults, defineConfig } from "vitest/config";

const ROOT = path.resolve(__dirname, "..");

/**
 * Resolve every workspace package to its TypeScript source, so tests need no
 * prior `pnpm build`. The root package alias is essential: without it,
 * test/*.test.ts imports resolve to dist/index.js while src/*.test.ts uses
 * relative imports, creating two module graphs where `instanceof Tracer`
 * fails and leak detection sees mismatched singletons.
 */
export const workspaceAliases = {
  "@hamk-uas/jax-js-nonconsuming": path.join(ROOT, "src/index.ts"),
  "@hamk-uas/jax-js-nonconsuming-jaxopt": path.join(
    ROOT,
    "packages/jaxopt/src/index.ts",
  ),
  "@hamk-uas/jax-js-nonconsuming-optax": path.join(
    ROOT,
    "packages/optax/src/index.ts",
  ),
  "@hamk-uas/jax-js-nonconsuming-onnx": path.join(
    ROOT,
    "packages/onnx/src/index.ts",
  ),
  "@hamk-uas/jax-js-nonconsuming-loaders": path.join(
    ROOT,
    "packages/loaders/src/index.ts",
  ),
};

/** Chromium args shared by all browser test configs. */
export const COMMON_ARGS = [
  "--no-sandbox",
  "--headless=new",
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--disable-vulkan-surface",
  "--enable-unsafe-webgpu",
];

/**
 * Test files run concurrently, each in its own iframe, but all in one
 * Chromium whose single GPU process compiles every iframe's shaders. With
 * vitest's default (CPU count − 1, i.e. 17 here) the compile queue made a
 * fresh trivial kernel take up to 7 s and pushed tests past their timeout in
 * most full runs; with 4 the worst case was 1 s and the suite took the same
 * wall time (measured 2026-11-08: 17 → 43–63 s with failures, 6 → 44 s,
 * 4 → 46 s). More files only lengthen the queue.
 */
export const BROWSER_MAX_WORKERS = 4;

/** Per-GPU overrides: extra Chromium args and the Vulkan driver manifest. */
const GPU_PROFILES = {
  nvidia: {
    args: ["--enable-dawn-features=vulkan_enable_f16_on_nvidia"],
    vulkanIcd: "/usr/share/vulkan/icd.d/nvidia_icd.json",
  },
  intel: {
    args: ["--use-vulkan=native", "--force-gpu-mem-available-mb=4096"],
    vulkanIcd: "/usr/share/vulkan/icd.d/intel_icd.json",
  },
} as const;

export type GpuProfile = keyof typeof GPU_PROFILES;

/**
 * Environment for Chromium. Playwright replaces the browser's whole
 * environment with this object.
 *
 * - No DISPLAY/XAUTHORITY: `--headless=new` initializes Vulkan WebGPU and
 *   WebGL without any display server. Pointing it at an X server it cannot
 *   use makes `requestAdapter()` return null (e.g. the login screen's
 *   Xwayland when nobody is logged in at the desktop), which silently skips
 *   every WebGPU case.
 * - `VK_DRIVER_FILES` limits the Vulkan loader to one GPU. With several GPUs
 *   visible and no display, Chromium composites canvases on a different
 *   device than the WebGPU adapter: canvas-based sync readback (`dataSync()`)
 *   returns zeros and WebGL2 is unavailable.
 *
 * Without `gpu`, picks the first installed GPU in profile order (NVIDIA
 * first, matching WebGPU's high-performance adapter choice); with no known
 * driver installed, leaves the loader unrestricted.
 *
 * Verified 2026-11-08: Ubuntu 26.04, Chromium 1208, NVIDIA 615, Mesa 26.0.
 */
export function chromiumEnv(gpu?: GpuProfile): Record<string, string> {
  const icd = gpu
    ? GPU_PROFILES[gpu].vulkanIcd
    : Object.values(GPU_PROFILES)
        .map((p) => p.vulkanIcd)
        .find((f) => fs.existsSync(f));
  return icd ? { VK_DRIVER_FILES: icd } : {};
}

/**
 * Build a complete vitest config for the given GPU.
 *
 * The config uses a 300 s timeout — GPU tests/benchmarks are long-running.
 */
export function gpuConfig(gpu: GpuProfile) {
  const profile = GPU_PROFILES[gpu];

  return defineConfig({
    resolve: { alias: workspaceAliases },
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
            env: chromiumEnv(gpu),
          },
        }),
        instances: [{ browser: "chromium" }],
      },
      maxWorkers: BROWSER_MAX_WORKERS,
      exclude: [...configDefaults.exclude, "tmp/**"],
      testTimeout: 300000,
      passWithNoTests: true,
      setupFiles: ["test/setup.ts"],
    },
  });
}
