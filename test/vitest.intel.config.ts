/**
 * Vitest config that forces the Intel integrated GPU.
 *
 * Uses VK_DRIVER_FILES to bypass the NVIDIA discrete GPU and force
 * Chromium's Vulkan backend to use the Intel mesa driver.
 *
 * Usage:
 *   pnpm vitest run test/gpu-bench.test.ts -c test/vitest.intel.config.ts
 *   pnpm vitest bench bench/matmul.bench.ts -c test/vitest.intel.config.ts
 */
import { gpuConfig } from "./gpu-config";

export default gpuConfig("intel");
