import type {
  Device,
  DType,
  Array as JaxArray,
} from "@hamk-uas/jax-js-nonconsuming";

type Jax = typeof import("@hamk-uas/jax-js-nonconsuming");

/**
 * Target length of one submission of matmuls, in ms.
 *
 * While a submission runs, a GPU that also drives the display cannot update
 * the screen. The benchmark used to submit 8 matmuls of a size picked by GPU
 * vendor (2048² for any AMD GPU): 184 ms per submission in fp32 and 521 ms in
 * fp16 on an Intel Arc iGPU, longer on weaker GPUs, and it was suspected of
 * freezing a laptop with an AMD GPU. Matrix size and batch length now come
 * from measured matmul times. Measured 2026-11-08: 46–63 ms per submission,
 * up to 91 ms for the first batch while the GPU is still at low clocks.
 */
const SUBMIT_BUDGET_MS = 50;
const MIN_N = 128;
const MAX_N = 4096;
const MAX_BATCH = 8;
const NUM_BATCHES = 7;

class DeadlineExceeded extends Error {}

const median = (xs: number[]) =>
  [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

function randomMatrix(
  jax: Jax,
  seed: number,
  n: number,
  dtype: DType,
): JaxArray {
  using key = jax.random.key(seed);
  const u = jax.random.uniform(key, [n, n]);
  const m = u.astype(dtype);
  if (m !== u) u.dispose(); // astype returns `u` itself when the dtype is unchanged
  return m;
}

/**
 * Measure matmul GFLOP/s on `device`. Returns undefined if the measurement
 * fails or does not finish within `timeoutMs`; no new work is submitted after
 * that, and the next measurement must not start before this one returns.
 */
export async function benchFlops(
  jax: Jax,
  device: Device,
  dtype: DType,
  timeoutMs = 8000,
): Promise<number | undefined> {
  const deadline = performance.now() + timeoutMs;
  // Also give up waiting if the device stops responding (e.g. device lost),
  // in which case the measurement never reaches its next deadline check.
  let timer: ReturnType<typeof setTimeout>;
  const unresponsive = new Promise<undefined>((resolve) => {
    timer = setTimeout(() => resolve(undefined), timeoutMs + 2000);
  });
  try {
    return await Promise.race([
      measure(jax, device, dtype, deadline),
      unresponsive,
    ]);
  } finally {
    clearTimeout(timer!);
  }
}

async function measure(
  jax: Jax,
  device: Device,
  dtype: DType,
  deadline: number,
): Promise<number | undefined> {
  const previousDevice = jax.defaultDevice();
  jax.defaultDevice(device);
  const matmul =
    device === "webgpu"
      ? jax.jit((a: JaxArray, b: JaxArray) => jax.lax.tiledMatmul(a, b))
      : jax.jit((a: JaxArray, b: JaxArray) => jax.numpy.matmul(a, b));

  /** Submit `count` matmuls together and wait for them; returns ms. */
  const run = async (
    a: JaxArray,
    b: JaxArray,
    count: number,
  ): Promise<number> => {
    if (performance.now() > deadline) throw new DeadlineExceeded();
    const outputs: JaxArray[] = [];
    try {
      const start = performance.now();
      jax.withBatch(() => {
        for (let i = 0; i < count; i++) outputs.push(matmul(a, b));
      });
      await jax.blockUntilReady(outputs);
      return performance.now() - start;
    } finally {
      for (const o of outputs) o.dispose();
    }
  };

  /** Median time of one n×n matmul, after compiling for that shape. */
  const timeOne = async (n: number): Promise<number> => {
    using a = randomMatrix(jax, 1, n, dtype);
    using b = randomMatrix(jax, 2, n, dtype);
    await run(a, b, 1);
    return median([await run(a, b, 1), await run(a, b, 1), await run(a, b, 1)]);
  };

  try {
    // Grow n while a matmul of the next size is predicted to fit the budget.
    // A measured time is a fixed submit/fence latency, taken at MIN_N where
    // the matmul itself is negligible, plus the work, which is predicted to
    // grow 8× per doubling of n. GPUs are usually more efficient at larger
    // sizes, so the prediction errs on the safe side.
    let n = MIN_N;
    const fixedMs = await timeOne(n);
    let workMs = 0.05;
    while (n < MAX_N && fixedMs + 8 * workMs <= SUBMIT_BUDGET_MS) {
      n *= 2;
      workMs = Math.max((await timeOne(n)) - fixedMs, 0.05);
    }

    using a = randomMatrix(jax, 1, n, dtype);
    using b = randomMatrix(jax, 2, n, dtype);
    {
      if (performance.now() > deadline) throw new DeadlineExceeded();
      using c = matmul(a, b);
      if (!Number.isFinite((await c.data())[0])) {
        throw new Error("matmul produced a non-finite result");
      }
    }

    const batch = Math.min(
      MAX_BATCH,
      Math.max(1, Math.floor((SUBMIT_BUDGET_MS - fixedMs) / workMs)),
    );
    await run(a, b, batch); // brings the GPU out of its low-power state
    const rates: number[] = [];
    for (let i = 0; i < NUM_BATCHES; i++) {
      const ms = await run(a, b, batch);
      rates.push((batch * 2 * n ** 3) / (ms / 1000) / 1e9);
    }
    const result = median(rates);
    return Number.isFinite(result) && result > 0 ? result : undefined;
  } catch (error) {
    if (!(error instanceof DeadlineExceeded)) {
      console.error("Benchmark error:", error);
    }
    return undefined;
  } finally {
    matmul.dispose();
    jax.defaultDevice(previousDevice);
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

// Shown until the visitor measures their own device.
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
  const jax = await import("@hamk-uas/jax-js-nonconsuming");
  const devices = await jax.init("webgpu", "wasm");
  const flops: PerfResults["flops"] = {
    Wasm: undefined,
    WebGPU: undefined,
    "WebGPU-fp16": undefined,
  };
  // One measurement at a time: each returns only after it has stopped
  // submitting work.
  if (devices.includes("webgpu")) {
    flops.WebGPU = await benchFlops(jax, "webgpu", jax.DType.Float32);
    if (
      flops.WebGPU !== undefined &&
      jax.getBackend("webgpu").capabilities.shaderF16
    ) {
      flops["WebGPU-fp16"] = await benchFlops(jax, "webgpu", jax.DType.Float16);
    }
    // Release the compiled programs and pooled GPU buffers.
    jax.clearCaches();
  }
  flops.Wasm = await benchFlops(jax, "wasm", jax.DType.Float32);
  return { flops, browser: "Your browser (live)", live: true };
}
