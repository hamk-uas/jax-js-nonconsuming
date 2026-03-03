/**
 * WGSL source inspection utility.
 *
 * Dumps the generated WGSL shader source for various operations at DEBUG≥2.
 * Useful for diagnosing compiler codegen, verifying tuner decisions (upcast,
 * unroll, cooperative groups), and comparing shader output across branches.
 *
 * Usage (NVIDIA):  pnpm vitest run test/debug-wgsl.test.ts -c test/vitest.nvidia.config.ts
 * Usage (Intel):   pnpm vitest run test/debug-wgsl.test.ts -c test/vitest.intel.config.ts
 * Usage (default): pnpm vitest run test/debug-wgsl.test.ts
 *
 * Output appears in the vitest stdout (look for "=== WGSL" markers).
 * To save: pnpm vitest run test/debug-wgsl.test.ts 2>&1 | tee tmp/wgsl-dump.log
 */
import {
  blockUntilReady,
  checkLeaks,
  clearCaches,
  defaultDevice,
  DType,
  init,
  jit,
  lax,
  numpy as np,
  setDebug,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, beforeAll, describe, test } from "vitest";

describe("WGSL source inspection", () => {
  beforeAll(async () => {
    checkLeaks.stop();
    const devices = await init("webgpu");
    if (devices.includes("webgpu")) defaultDevice("webgpu");
  });

  afterAll(() => {
    setDebug(0);
    checkLeaks.start();
  });

  test("dot f32 2048×2048", async () => {
    setDebug(2);
    console.log("\n=== WGSL: dot f32 2048×2048 ===");
    using A = np.ones([2048, 2048], { dtype: DType.Float32 });
    using B = np.ones([2048, 2048], { dtype: DType.Float32 });
    const f = jit((a: np.Array, b: np.Array) => np.dot(a, b));
    using C = f(A, B);
    await blockUntilReady(C);
    f.dispose();
    clearCaches();
    setDebug(0);
  });

  test("dot f16 2048×2048", async () => {
    const gpu = navigator.gpu;
    const adapter = await gpu?.requestAdapter();
    if (!adapter?.features.has("shader-f16")) {
      console.log("Skipping f16: shader-f16 not available");
      return;
    }

    setDebug(2);
    console.log("\n=== WGSL: dot f16 2048×2048 ===");
    using A = np.ones([2048, 2048], { dtype: DType.Float16 });
    using B = np.ones([2048, 2048], { dtype: DType.Float16 });
    const f = jit((a: np.Array, b: np.Array) => np.dot(a, b));
    using C = f(A, B);
    await blockUntilReady(C);
    f.dispose();
    clearCaches();
    setDebug(0);
  });

  test("tiledMatmul f32 2048×2048", async () => {
    setDebug(2);
    console.log("\n=== WGSL: tiledMatmul f32 2048×2048 ===");
    using A = np.ones([2048, 2048], { dtype: DType.Float32 });
    using B = np.ones([2048, 2048], { dtype: DType.Float32 });
    const f = jit((a: np.Array, b: np.Array) => lax.tiledMatmul(a, b));
    using C = f(A, B);
    await blockUntilReady(C);
    f.dispose();
    clearCaches();
    setDebug(0);
  });

  test("sum reduction f32 4M elements", async () => {
    setDebug(2);
    console.log("\n=== WGSL: sum f32 [4194304] ===");
    using A = np.ones([4194304], { dtype: DType.Float32 });
    const f = jit((a: np.Array) => np.sum(a));
    using C = f(A);
    await blockUntilReady(C);
    f.dispose();
    clearCaches();
    setDebug(0);
  });

  test("elementwise chain f32 4096", async () => {
    setDebug(2);
    console.log("\n=== WGSL: elementwise chain f32 [4096] ===");
    using A = np.ones([4096], { dtype: DType.Float32 });
    using B = np.ones([4096], { dtype: DType.Float32 });
    const f = jit((a: np.Array, b: np.Array) => {
      using t1 = np.add(a, b);
      using t2 = np.multiply(t1, a);
      return np.subtract(t2, b);
    });
    using C = f(A, B);
    await blockUntilReady(C);
    f.dispose();
    clearCaches();
    setDebug(0);
  });

  test("adapter info", async () => {
    const gpu = navigator.gpu;
    const adapter = await gpu?.requestAdapter();
    console.log("\n=== Adapter Info ===");
    console.log(`  vendor: ${adapter?.info.vendor}`);
    console.log(`  architecture: ${adapter?.info.architecture}`);
    console.log(`  device: ${adapter?.info.device}`);
    console.log(`  description: ${adapter?.info.description}`);
    console.log(`  shader-f16: ${adapter?.features.has("shader-f16")}`);
    console.log(
      `  maxComputeInvocationsPerWorkgroup: ${adapter?.limits.maxComputeInvocationsPerWorkgroup}`,
    );
    console.log(
      `  maxComputeWorkgroupSizeX: ${adapter?.limits.maxComputeWorkgroupSizeX}`,
    );
    console.log(
      `  maxComputeWorkgroupStorageSize: ${adapter?.limits.maxComputeWorkgroupStorageSize}`,
    );
  });
});
