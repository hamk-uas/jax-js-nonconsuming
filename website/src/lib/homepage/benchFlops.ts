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

    // Warmup: triggers JIT compilation + pipeline creation + GPU clock boost
    for (let w = 0; w < 3; w++) {
      using C = matmulFn(A, B);
      await jax.blockUntilReady(C);
      if (w === 0) {
        const sample = (await C.data())[0];
        if (!Number.isFinite(sample))
          throw new Error("GPU produced non-finite result");
      }
    }

    // Batch-submit timing: dispatch BATCH_SIZE matmuls back-to-back in a
    // single queue.submit(), fence once, divide total time by BATCH_SIZE.
    // This amortizes per-fence overhead and renderer main-thread stalls
    // (module compilation, layout, compositor BeginFrame) across many GPU
    // dispatches — far more stable than per-iteration fence timing.
    const BATCH_SIZE = 8;
    const NUM_BATCHES = 7;

    // Extra warmup: run a full batch to ramp GPU clocks before timing.
    // Single-iteration warmup above compiles shaders but doesn't generate
    // enough sustained load to bring the GPU out of low-power state.
    // Without this, the first 1–2 timed batches run at reduced clock speed
    // (observed: 183ms vs ~30ms on Intel gen-9 iGPU), occasionally
    // contaminating the median.
    {
      const wOutputs: any[] = [];
      jax.withBatch(() => {
        for (let i = 0; i < BATCH_SIZE; i++) wOutputs.push(matmulFn(A, B));
      });
      await jax.blockUntilReady(wOutputs[wOutputs.length - 1]);
      for (const o of wOutputs) o.dispose();
    }
    const measurements: number[] = [];
    for (let b = 0; b < NUM_BATCHES; b++) {
      const outputs: any[] = [];
      const start = performance.now();
      jax.withBatch(() => {
        for (let i = 0; i < BATCH_SIZE; i++) outputs.push(matmulFn(A, B));
      });
      await jax.blockUntilReady(outputs[outputs.length - 1]);
      const end = performance.now();
      for (const o of outputs) o.dispose();
      measurements.push((end - start) / 1000 / BATCH_SIZE);
    }

    matmulFn.dispose();

    const gflops = (2 * n * n * n) / 1e9;
    measurements.sort((a, b) => a - b);
    const seconds = measurements[Math.floor(measurements.length / 2)];
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
    // Intel Arc (xe-lpg, xe-hpg) handles 2048 comfortably (~25ms/iter).
    gpuDim = isMobile ? 1024 : 2048;
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
