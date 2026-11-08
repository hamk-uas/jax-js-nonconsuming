/**
 * Shared browser-test config: workspace aliases, Chromium environment, and the
 * GPU-specific vitest config factory.
 *
 * `vitest.config.ts` (default suite) and the per-GPU configs both use the
 * aliases, GPU detection and Chromium options from here so they cannot drift
 * apart.
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
const COMMON_ARGS = [
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
 *
 * It also bounds GPU memory: every file has its own GPUDevice and buffer
 * pool. Peak over the suite: 1.2 GiB with 17 files, 0.9 GiB with 4 (one
 * test's spike; ~0.2 GiB otherwise), which matters on laptop GPUs.
 */
export const BROWSER_MAX_WORKERS = 4;

/**
 * Per-GPU settings: PCI vendor ID, extra Chromium args and the Vulkan driver
 * manifest. The order is the preference when the GPU is detected (discrete
 * vendors first, matching WebGPU's high-performance adapter choice).
 */
const GPU_PROFILES = {
  nvidia: {
    pciVendor: 0x10de,
    args: ["--enable-dawn-features=vulkan_enable_f16_on_nvidia"],
    vulkanIcd: "/usr/share/vulkan/icd.d/nvidia_icd.json",
  },
  amd: {
    pciVendor: 0x1002,
    args: [],
    vulkanIcd: "/usr/share/vulkan/icd.d/radeon_icd.json",
  },
  intel: {
    pciVendor: 0x8086,
    args: ["--use-vulkan=native", "--force-gpu-mem-available-mb=4096"],
    vulkanIcd: "/usr/share/vulkan/icd.d/intel_icd.json",
  },
} as const;

export type GpuProfile = keyof typeof GPU_PROFILES;

/**
 * PCI vendor IDs of the GPUs present, from Linux sysfs; null where sysfs is
 * not available.
 */
function gpuVendorsPresent(): Set<number> | null {
  const drm = "/sys/class/drm";
  if (!fs.existsSync(drm)) return null;
  const vendors = new Set<number>();
  for (const entry of fs.readdirSync(drm)) {
    // cardN is a GPU; cardN-<connector> entries are its outputs.
    const vendorFile = path.join(drm, entry, "device/vendor");
    if (/^card\d+$/.test(entry) && fs.existsSync(vendorFile)) {
      vendors.add(Number(fs.readFileSync(vendorFile, "utf8").trim()));
    }
  }
  return vendors;
}

/**
 * The GPU that a config without an explicit GPU uses: the first profile whose
 * hardware is present and whose Vulkan driver is installed.
 *
 * Both conditions are needed. Mesa's `mesa-vulkan-drivers` installs the Intel
 * and Radeon manifests on every machine, so choosing by manifest alone picked
 * the Intel driver on an AMD-only machine, where WebGPU then found no adapter
 * and every WebGPU case was skipped (phase 0 review, finding 1).
 */
export function detectGpu(
  present: Set<number> | null = gpuVendorsPresent(),
  installed = (manifest: string) => fs.existsSync(manifest),
): GpuProfile | undefined {
  if (!present) return undefined;
  return (Object.keys(GPU_PROFILES) as GpuProfile[]).find(
    (gpu) =>
      present.has(GPU_PROFILES[gpu].pciVendor) &&
      installed(GPU_PROFILES[gpu].vulkanIcd),
  );
}

/**
 * Chromium launch options for `gpu`.
 *
 * Playwright replaces the browser's whole environment with `env`:
 * - No DISPLAY/XAUTHORITY: `--headless=new` initializes Vulkan WebGPU and
 *   WebGL without any display server. Pointing it at an X server it cannot
 *   use makes `requestAdapter()` return null (e.g. the login screen's
 *   Xwayland when nobody is logged in at the desktop).
 * - `VK_DRIVER_FILES` limits the Vulkan loader to one GPU. With several GPUs
 *   visible and no display, Chromium composites canvases on a different
 *   device than the WebGPU adapter: canvas-based sync readback (`dataSync()`)
 *   returns zeros and WebGL2 is unavailable.
 *
 * Without a GPU, the loader is left unrestricted.
 *
 * Verified 2026-11-08: Ubuntu 26.04, Chromium 1208, NVIDIA 615, Mesa 26.0.
 */
export function chromiumLaunchOptions(gpu: GpuProfile | undefined) {
  return {
    args: [...COMMON_ARGS, ...(gpu ? GPU_PROFILES[gpu].args : [])],
    env: gpu
      ? { VK_DRIVER_FILES: GPU_PROFILES[gpu].vulkanIcd }
      : ({} as Record<string, string>),
  };
}

/**
 * Browser backends that every test file requires; `test/setup.ts` fails the
 * file if one of them does not initialize. Without this, a GPU that failed to
 * initialize skipped every WebGPU and WebGL case and the run passed (R18).
 *
 * With a GPU: WebGPU and WebGL. `JAX_REQUIRE_DEVICES` overrides the list
 * (comma-separated; empty for none), e.g. for a GPU without WebGL.
 */
export function requiredDevices(gpu: GpuProfile | undefined): string[] {
  const override = process.env.JAX_REQUIRE_DEVICES;
  if (override !== undefined) {
    return override
      .split(",")
      .map((d) => d.trim())
      .filter(Boolean);
  }
  return gpu ? ["webgpu", "webgl"] : [];
}

/** Print which GPU and required backends a test run uses, once per run. */
export function describeGpuSetup(gpu: GpuProfile | undefined): void {
  // Vitest evaluates the config more than once per run.
  if (process.env.JAX_TEST_GPU_SETUP_PRINTED) return;
  process.env.JAX_TEST_GPU_SETUP_PRINTED = "1";
  const required = requiredDevices(gpu);
  console.log(
    `[test config] GPU: ${gpu ?? "none detected"}` +
      (gpu ? ` (VK_DRIVER_FILES=${GPU_PROFILES[gpu].vulkanIcd})` : "") +
      `; required backends: ${required.join(", ") || "none"}`,
  );
}

/**
 * Build a complete vitest config for the given GPU.
 *
 * The config uses a 300 s timeout — GPU tests/benchmarks are long-running.
 */
export function gpuConfig(gpu: GpuProfile) {
  describeGpuSetup(gpu);
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
        provider: playwright({ launchOptions: chromiumLaunchOptions(gpu) }),
        instances: [{ browser: "chromium" }],
      },
      maxWorkers: BROWSER_MAX_WORKERS,
      provide: { requiredDevices: requiredDevices(gpu) },
      exclude: [...configDefaults.exclude, "tmp/**"],
      testTimeout: 300000,
      passWithNoTests: true,
      setupFiles: ["test/setup.ts"],
    },
  });
}

declare module "vitest" {
  export interface ProvidedContext {
    /** Backends that test/setup.ts requires to initialize. */
    requiredDevices: string[];
  }
}
