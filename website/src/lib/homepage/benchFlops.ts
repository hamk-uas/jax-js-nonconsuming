import type { Device, DType } from "@hamk-uas/jax-js-nonconsuming";

export async function benchFlops(
  n: number,
  device: Device,
  dtype: DType,
  timeoutMs = 8000,
): Promise<number | undefined> {
  // Race the actual benchmark against a timeout so weak GPUs that trigger
  // TDR (GPU timeout / device-lost) don't hang the page.
  return Promise.race([
    benchFlopsInner(n, device, dtype),
    new Promise<undefined>((resolve) =>
      setTimeout(() => resolve(undefined), timeoutMs),
    ),
  ]);
}

async function benchFlopsInner(
  n: number,
  device: Device,
  dtype: DType,
): Promise<number | undefined> {
  try {
    const jax = await import("@hamk-uas/jax-js-nonconsuming");
    await jax.init(device);
    jax.defaultDevice(device);

    const np = jax.numpy;
    const lax = jax.lax;

    // JIT-compile matmul — on WebGPU this uses tiledMatmul with
    // GPU-adaptive tile config.  The first call compiles; we exclude it.
    const matmulFn =
      device === "webgpu"
        ? jax.jit((a: any, b: any) => lax.tiledMatmul(a, b))
        : jax.jit((a: any, b: any) => np.matmul(a, b));

    // Allocate inputs once, reuse across all iterations
    using key = jax.random.key(0);
    const [k1, k2] = jax.random.split(key, 2);
    using A = jax.random.uniform(k1, [n, n]).astype(dtype);
    using B = jax.random.uniform(k2, [n, n]).astype(dtype);
    k1.dispose();
    k2.dispose();
    await jax.blockUntilReady([A, B]);

    // Warmup: triggers JIT compilation + pipeline creation (not timed)
    {
      using C = matmulFn(A, B);
      await jax.blockUntilReady(C);
      const sample = (await C.data())[0];
      if (!Number.isFinite(sample))
        throw new Error("GPU produced non-finite result");
    }

    // Timed runs
    const measurements: number[] = [];
    for (let i = 0; i < 3; i++) {
      const start = performance.now();
      using C = matmulFn(A, B);
      await jax.blockUntilReady(C);
      const end = performance.now();
      measurements.push((end - start) / 1000);
    }

    matmulFn.dispose();

    const gflops = (2 * n * n * n) / 1e9;
    const seconds =
      measurements.reduce((a, b) => a + b, 0) / measurements.length;
    const result = gflops / seconds;

    return Number.isFinite(result) && result > 0 ? result : undefined;
  } catch (error: any) {
    console.error("Benchmark error:", error);
    return undefined;
  }
}

export interface PerfResults {
  flops: {
    Wasm: number | undefined;
    WebGPU: number | undefined;
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
  const isIntel = /intel/i.test(adapter.info.vendor);
  const isGoodGpu = /NVIDIA|AMD|Qualcomm|ARM/i.test(adapter.info.vendor);
  // Intel Arc (xe-lpg, xe-hpg, …) is a discrete-class GPU; older Intel
  // UHD/Iris iGPUs have much less compute and can trigger TDR at large N.
  const isIntelArc =
    isIntel &&
    /xe-[lh]pg|alchemist|arc/i.test(
      `${adapter.info.architecture} ${adapter.info.description}`,
    );

  let gpuDim: number;
  if (isApple) {
    gpuDim = isMobile ? 2048 : 4096;
  } else if (isNvidia) {
    gpuDim = 4096;
  } else if (isGoodGpu) {
    gpuDim = isMobile ? 1024 : 2048;
  } else if (isIntelArc) {
    gpuDim = 1024;
  } else if (isIntel) {
    // Intel UHD / Iris iGPUs — keep it light to avoid device-lost.
    gpuDim = 512;
  } else {
    gpuDim = 512;
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
