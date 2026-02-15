import type { Device, DType } from "@jax-js-nonconsuming/jax";

export async function benchFlops(
  n: number,
  device: Device,
  dtype: DType,
): Promise<number> {
  try {
    const jax = await import("@jax-js-nonconsuming/jax");
    await jax.init(device);
    jax.defaultDevice(device);

    const np = jax.numpy;

    const measurements: number[] = [];

    // 1 warmup + 2 timed runs
    for (let i = 0; i < 3; i++) {
      using key = jax.random.key(0);
      const [k1, k2] = jax.random.split(key, 2);

      using A = jax.random.uniform(k1, [n, n]).astype(dtype);
      using B = jax.random.uniform(k2, [n, n]).astype(dtype);
      k1.dispose();
      k2.dispose();
      await jax.blockUntilReady([A, B]);

      const start = performance.now();
      using C = np.matmul(A, B);
      await jax.blockUntilReady(C);
      const end = performance.now();

      if (i > 0) {
        measurements.push((end - start) / 1000);
      }
    }

    const gflops = (2 * n * n * n) / 1e9;
    const seconds =
      measurements.reduce((a, b) => a + b, 0) / measurements.length;

    return gflops / seconds;
  } catch (error: any) {
    console.error("Benchmark error:", error);
    return 0;
  }
}

export interface PerfResults {
  flops: {
    Wasm: number;
    WebGPU: number;
    "WebGPU-fp16": number | undefined;
  };
  browser: string;
  live: boolean;
}

// Fall back to these if live measurement fails
export const fallbackResults: PerfResults = {
  flops: {
    Wasm: 2.72,
    WebGPU: 2071,
    "WebGPU-fp16": 3343,
  },
  browser: "Chrome on Apple M3 Pro",
  live: false,
};

export async function measurePerf(): Promise<PerfResults> {
  // See if the current browser supports WebGPU.
  if (!navigator?.gpu?.requestAdapter) {
    return fallbackResults;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return fallbackResults;
  const hasF16 = adapter.features.has("shader-f16");

  const isApple = /iPad|iPhone|iPod|Macintosh/.test(navigator.userAgent);
  const isMobile = /Mobi/.test(navigator.userAgent);
  const isNvidia = /NVIDIA/i.test(adapter.info.vendor);
  const isGoodGpu = /NVIDIA|AMD|Qualcomm|ARM/i.test(adapter.info.vendor);

  let gpuDim: number;
  if (isApple) {
    gpuDim = isMobile ? 2048 : 4096;
  } else if (isNvidia) {
    gpuDim = 4096;
  } else if (isGoodGpu) {
    gpuDim = isMobile ? 1024 : 2048;
  } else {
    gpuDim = 1024;
  }

  return {
    flops: {
      Wasm: await benchFlops(128, "wasm", "float32" as DType),
      WebGPU: await benchFlops(gpuDim, "webgpu", "float32" as DType),
      "WebGPU-fp16": hasF16
        ? await benchFlops(gpuDim, "webgpu", "float16" as DType)
        : undefined,
    },
    browser: "Your browser (live)",
    live: true,
  };
}
