/**
 * Vitest config for NVIDIA GPU tests and benchmarks.
 *
 * Selects the NVIDIA discrete GPU (Chromium's default Vulkan adapter priority).
 * Enables f16 via Dawn feature flag.
 *
 * Usage:
 *   pnpm vitest run test/gpu-bench.test.ts -c test/vitest.nvidia.config.ts
 *   pnpm vitest bench bench/matmul.bench.ts -c test/vitest.nvidia.config.ts
 */
import { gpuConfig } from "./gpu-config";

export default gpuConfig("nvidia");
