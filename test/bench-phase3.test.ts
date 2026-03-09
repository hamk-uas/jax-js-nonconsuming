/**
 * Phase 3 matmul benchmark — measures tiledMatmul with register tiling.
 *
 * Run: pnpm vitest run test/bench-phase3.test.ts --testTimeout=300000
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
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

const availableDevices = await init();
const hasWebGPU = availableDevices.includes("webgpu");

async function benchOne(
  fn: () => np.Array,
  warmup: number,
  iters: number,
): Promise<number> {
  for (let i = 0; i < warmup; i++) {
    const r = fn();
    await blockUntilReady(r);
    r.dispose();
  }
  const start = performance.now();
  for (let i = 0; i < iters; i++) {
    const r = fn();
    await blockUntilReady(r);
    r.dispose();
  }
  return (performance.now() - start) / iters;
}

describe("Phase 3 matmul benchmark", () => {
  // 512×512 f32: quick representative size
  test("512×512 benchmark", async ({ skip }) => {
    if (!hasWebGPU) skip();
    // Pause leak checker for benchmark
    checkLeaks.stop();

    await init("webgpu");
    const prev = defaultDevice("webgpu");

    const M = 512,
      N = 512,
      K = 512;
    const a = np.ones([M, K], { dtype: DType.Float32 });
    const b = np.ones([K, N], { dtype: DType.Float32 });
    const flops = 2 * M * N * K;
    const gflops = (ms: number) => (flops / (ms * 1e-3) / 1e9).toFixed(1);

    // np.matmul
    const f_np = jit((A: np.Array, B: np.Array) => np.matmul(A, B) as np.Array);
    const t1 = await benchOne(() => f_np(a, b), 3, 10);

    clearCaches();

    // tiledMatmul baseline (no threadTile)
    const f_base = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    const t2 = await benchOne(() => f_base(a, b), 3, 10);

    clearCaches();

    // tiledMatmul threadTile=[4,4]
    const f_tt44 = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 64, Bc: 64, Bk: 16, threadTile: [4, 4] }),
    );
    const t3 = await benchOne(() => f_tt44(a, b), 3, 10);

    clearCaches();

    // tiledMatmul threadTile=[8,8]
    const f_tt88 = jit((A: np.Array, B: np.Array) =>
      lax.tiledMatmul(A, B, { Br: 64, Bc: 64, Bk: 16, threadTile: [8, 8] }),
    );
    const t4 = await benchOne(() => f_tt88(a, b), 3, 10);

    console.log(`\n=== 512×512 f32 Benchmark (WebGPU) ===`);
    console.log(
      `  np.matmul:     ${t1.toFixed(3)} ms  (${gflops(t1)} GFLOP/s)`,
    );
    console.log(
      `  tiled(16):     ${t2.toFixed(3)} ms  (${gflops(t2)} GFLOP/s)`,
    );
    console.log(
      `  tt=[4,4]:      ${t3.toFixed(3)} ms  (${gflops(t3)} GFLOP/s)`,
    );
    console.log(
      `  tt=[8,8]:      ${t4.toFixed(3)} ms  (${gflops(t4)} GFLOP/s)`,
    );
    console.log(`  Speedup tt=[4,4] vs tiled(16): ${(t2 / t3).toFixed(1)}×`);
    console.log(`  Speedup tt=[8,8] vs tiled(16): ${(t2 / t4).toFixed(1)}×`);

    // Cleanup
    f_np.dispose();
    f_base.dispose();
    f_tt44.dispose();
    f_tt88.dispose();
    a.dispose();
    b.dispose();
    clearCaches();
    defaultDevice(prev);

    // Re-enable leak checker for subsequent tests
    checkLeaks.start();

    expect(true).toBe(true);
  });

  test("shader-dump: threadTile=[4,4] O5 check", async () => {
    const { setDebug } = await import("@hamk-uas/jax-js-nonconsuming");
    checkLeaks.stop();
    const prev = defaultDevice("webgpu");

    // Probe GPU identity first
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (adapter) {
        const info =
          (adapter as any).info ??
          (await (adapter as any).requestAdapterInfo?.());
        console.log("\n=== GPU Adapter Info ===");
        console.log("  vendor:", info?.vendor);
        console.log("  architecture:", info?.architecture);
        console.log("  device:", info?.device);
        console.log("  description:", info?.description);
        console.log("  isFallbackAdapter:", (adapter as any).isFallbackAdapter);
        console.log("  features:", [...adapter.features].sort().join(", "));
        console.log(
          "  maxStorageBuffers:",
          adapter.limits.maxStorageBuffersPerShaderStage,
        );
        console.log(
          "  maxComputeWorkgroupSizeX:",
          adapter.limits.maxComputeWorkgroupSizeX,
        );
      }
    }

    setDebug(2);

    {
      using A = np.ones([32, 32]);
      using B = np.ones([32, 32]);
      const f = jit((a: any, b: any) =>
        lax.tiledMatmul(a, b, { Br: 16, Bc: 16, Bk: 16, threadTile: [4, 4] }),
      );
      // eslint-disable-next-line @typescript-eslint/await-thenable
      using _result = await f(A, B);
      f.dispose();
      clearCaches();
    }
    setDebug(0);
    defaultDevice(prev);
    checkLeaks.start();
  });

  // Quick kernel-time-only benchmark (no readback between iterations)
  test("kernel timing: batch dispatch", async ({ skip }) => {
    if (!hasWebGPU) skip();
    checkLeaks.stop();
    await init("webgpu");
    const prev = defaultDevice("webgpu");

    // === Phase 1: Multi-size scaling test ===
    // If overhead is fixed per-dispatch, GFLOP/s should increase with N.
    // If overhead is proportional, GFLOP/s stays constant.
    {
      console.log(
        `\n=== Multi-Size Scaling (np.matmul, 10 dispatches + 1 readback) ===`,
      );
      for (const N of [64, 128, 256, 512, 1024]) {
        using ai = np.ones([N, N], { dtype: DType.Float32 });
        using bi = np.ones([N, N], { dtype: DType.Float32 });
        const fi = jit(
          (A: np.Array, B: np.Array) => np.matmul(A, B) as np.Array,
        );
        // warmup
        for (let j = 0; j < 3; j++) {
          const rr = fi(ai, bi);
          await blockUntilReady(rr);
          rr.dispose();
        }
        const batch = 10;
        const rs: np.Array[] = [];
        const t0 = performance.now();
        for (let j = 0; j < batch; j++) rs.push(fi(ai, bi));
        await blockUntilReady(rs[rs.length - 1]);
        const t1 = performance.now();
        for (const rr of rs) rr.dispose();
        fi.dispose();
        clearCaches();
        const ms = (t1 - t0) / batch;
        const flops = 2 * N * N * N;
        const gf = flops / (ms * 1e-3) / 1e9;
        console.log(
          `  ${N}×${N}: ${ms.toFixed(1)} ms/call  (${gf.toFixed(2)} GFLOP/s)`,
        );
      }
    }

    // === Phase 2: Tiled variants at 512×512 ===
    const M = 512,
      N = 512,
      K = 512;
    const flops = 2 * M * N * K;
    // Explicit dispose (not `using`) so disposal happens before checkLeaks.start()
    const a = np.ones([M, K], { dtype: DType.Float32 });
    const b = np.ones([K, N], { dtype: DType.Float32 });

    const configs = [
      {
        name: "np.matmul",
        fn: jit((A: np.Array, B: np.Array) => np.matmul(A, B) as np.Array),
      },
      {
        name: "tiled(16)",
        fn: jit((A: np.Array, B: np.Array) =>
          lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
        ),
      },
      {
        name: "tt=[4,4]",
        fn: jit((A: np.Array, B: np.Array) =>
          lax.tiledMatmul(A, B, { Br: 64, Bc: 64, Bk: 16, threadTile: [4, 4] }),
        ),
      },
      {
        name: "tt=[8,8]",
        fn: jit((A: np.Array, B: np.Array) =>
          lax.tiledMatmul(A, B, { Br: 64, Bc: 64, Bk: 16, threadTile: [8, 8] }),
        ),
      },
    ];

    console.log(
      `\n=== Batch Dispatch Timing (512×512, 50 dispatches → 1 readback) ===`,
    );

    for (const { name, fn: f } of configs) {
      for (let i = 0; i < 5; i++) {
        const r = f(a, b);
        await blockUntilReady(r);
        r.dispose();
      }

      const batchSize = 50;
      const results: np.Array[] = [];
      const t0 = performance.now();
      for (let i = 0; i < batchSize; i++) {
        results.push(f(a, b));
      }
      await blockUntilReady(results[results.length - 1]);
      const t1 = performance.now();
      const msPerCall = (t1 - t0) / batchSize;
      const gflopsVal = flops / (msPerCall * 1e-3) / 1e9;

      for (const r of results) r.dispose();
      f.dispose();
      clearCaches();

      console.log(
        `  ${name}: ${msPerCall.toFixed(3)} ms/call  (${gflopsVal.toFixed(1)} GFLOP/s)`,
      );
    }

    a.dispose();
    b.dispose();
    defaultDevice(prev);
    checkLeaks.start();
  });
});
