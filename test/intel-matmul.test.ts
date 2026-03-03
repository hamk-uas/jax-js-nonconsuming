/**
 * Dual-GPU matmul benchmark: sweeps N×N sizes across tile configs.
 *
 * NVIDIA (with fp16): pnpm vitest run test/intel-matmul.test.ts -c test/vitest.nvidia.config.ts
 * Intel iGPU:         pnpm vitest run test/intel-matmul.test.ts -c test/vitest.intel.config.ts
 * Default config:     pnpm vitest run test/intel-matmul.test.ts --testTimeout=300000
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
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

async function bench(
  N: number,
  dtype: DType,
  mode: "eager" | "jit-auto" | "tiled",
  opts?: any,
) {
  using k = random.key(42);
  const [k1, k2] = random.split(k, 2);
  using A = random.uniform(k1, [N, N]).astype(dtype);
  using B = random.uniform(k2, [N, N]).astype(dtype);
  k1.dispose();
  k2.dispose();
  await blockUntilReady([A, B]);

  if (mode === "eager") {
    {
      using C = np.matmul(A, B);
      await blockUntilReady(C);
    }
    const times: number[] = [];
    for (let i = 0; i < 5; i++) {
      const start = performance.now();
      using C = np.matmul(A, B);
      await blockUntilReady(C);
      times.push(performance.now() - start);
    }
    const avg = times.reduce((a, b) => a + b) / times.length;
    return { avg, gflops: (2 * N ** 3) / 1e9 / (avg / 1000) };
  }

  const f =
    mode === "tiled"
      ? jit((a: np.Array, b: np.Array) => lax.tiledMatmul(a, b, opts))
      : jit((a: np.Array, b: np.Array) => np.matmul(a, b));
  {
    using C = f(A, B);
    await blockUntilReady(C);
  }
  const times: number[] = [];
  for (let i = 0; i < 5; i++) {
    const start = performance.now();
    using C = f(A, B);
    await blockUntilReady(C);
    times.push(performance.now() - start);
  }
  f.dispose();
  const avg = times.reduce((a, b) => a + b) / times.length;
  return { avg, gflops: (2 * N ** 3) / 1e9 / (avg / 1000) };
}

function pad(s: string, w: number) {
  return s.padEnd(w);
}
function rpad(s: string, w: number) {
  return s.padStart(w);
}

describe("GPU matmul benchmark", () => {
  test("adapter info + performance", async () => {
    checkLeaks.stop();
    const gpu = navigator.gpu;
    if (!gpu) {
      console.log("No WebGPU");
      return;
    }

    const adapter = await gpu.requestAdapter();
    console.log(
      `\nAdapter: ${adapter?.info.vendor} ${adapter?.info.architecture}`,
    );
    console.log(
      `  maxComputeInvocations: ${adapter?.limits.maxComputeInvocationsPerWorkgroup}`,
    );
    console.log(
      `  maxComputeWorkgroupStorageSize: ${adapter?.limits.maxComputeWorkgroupStorageSize}`,
    );
    console.log(`  shader-f16: ${adapter?.features.has("shader-f16")}`);

    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      console.log("No WebGPU via init()");
      return;
    }
    defaultDevice("webgpu");

    const hasF16 = adapter?.features.has("shader-f16");

    type Cfg = {
      name: string;
      mode: "eager" | "jit-auto" | "tiled";
      opts?: any;
    };
    const configs: Cfg[] = [
      { name: "eager (reference)", mode: "eager" },
      { name: "jit(matmul) auto", mode: "jit-auto" },
      {
        name: "tiled 16×16",
        mode: "tiled",
        opts: { Br: 16, Bc: 16, Bk: 16 },
      },
      {
        name: "tiled 32×32 tt22",
        mode: "tiled",
        opts: { Br: 32, Bc: 32, Bk: 16, threadTile: [2, 2] },
      },
      {
        name: "tiled 64×64 tt44",
        mode: "tiled",
        opts: { Br: 64, Bc: 64, Bk: 16, threadTile: [4, 4] },
      },
      {
        name: "tiled 32×32 tt44",
        mode: "tiled",
        opts: { Br: 32, Bc: 32, Bk: 16, threadTile: [4, 4] },
      },
      {
        name: "tiled 32×32 Bk32 tt22",
        mode: "tiled",
        opts: { Br: 32, Bc: 32, Bk: 32, threadTile: [2, 2] },
      },
    ];

    for (const N of [256, 512, 1024, 2048]) {
      for (const dtype of [DType.Float32, ...(hasF16 ? [DType.Float16] : [])]) {
        const dtypeName = dtype === DType.Float16 ? "fp16" : "fp32";
        console.log(`\n=== N=${N} ${dtypeName} ===`);
        console.log(
          `  ${pad("Config", 28)} ${rpad("ms", 8)} ${rpad("GFLOP/s", 10)}`,
        );
        console.log("  " + "-".repeat(48));

        for (const cfg of configs) {
          try {
            const r = await bench(N, dtype, cfg.mode, cfg.opts);
            console.log(
              `  ${pad(cfg.name, 28)} ${rpad(r.avg.toFixed(1), 8)} ${rpad(r.gflops.toFixed(1), 10)}`,
            );
          } catch (e: any) {
            console.log(
              `  ${pad(cfg.name, 28)} FAILED: ${e.message?.slice(0, 60)}`,
            );
          }
          clearCaches();
        }
      }
    }

    expect(true).toBe(true);
    checkLeaks.start();
  });
});
