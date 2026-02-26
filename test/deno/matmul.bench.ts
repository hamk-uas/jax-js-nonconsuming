/**
 * Deno WebGPU matmul GFLOP/s benchmark — non-consuming fork
 *
 * Run with:
 *   pnpm build && deno bench --no-check --unstable-webgpu --allow-read --allow-env test/deno/matmul.bench.ts
 *
 * Replicates the website front-page benchmark (benchFlops).
 */

import {
  blockUntilReady,
  defaultDevice,
  getBackend,
  init,
  jit,
  numpy as np,
  random,
} from "../../dist/index.js";

const devices = await init();
if (!devices.includes("webgpu")) {
  console.log("WebGPU not available, skipping benchmarks");
  Deno.exit(0);
}
defaultDevice("webgpu");

// --- Helpers ---

async function benchMatmul(n: number, dtype: string, label: string) {
  using key = random.key(42);
  const [k1, k2] = random.split(key, 2);
  using A = random.uniform(k1, [n, n]).astype(dtype);
  using B = random.uniform(k2, [n, n]).astype(dtype);
  k1.dispose();
  k2.dispose();
  await blockUntilReady([A, B]);

  // Warmup
  {
    const C = np.matmul(A, B);
    await blockUntilReady(C);
    C.dispose();
  }

  Deno.bench(`${label} eager matmul ${n}x${n}`, { group: label }, async (b) => {
    b.start();
    const C = np.matmul(A, B);
    await blockUntilReady(C);
    b.end();
    C.dispose();
  });

  // JIT version
  const matmulJit = jit((a: any, b: any) => np.matmul(a, b));

  // Warmup JIT
  {
    const C = matmulJit(A, B);
    await blockUntilReady(C);
    C.dispose();
  }

  Deno.bench(
    `${label} JIT matmul ${n}x${n}`,
    { group: label, baseline: true },
    async (b) => {
      b.start();
      const C = matmulJit(A, B);
      await blockUntilReady(C);
      b.end();
      C.dispose();
    },
  );
}

// --- Benchmarks ---

// Single matmul - eager vs JIT
await benchMatmul(1024, "float32", "f32");

const backend = getBackend("webgpu");
const hasF16 = backend.caps?.supportsFloat16 ?? false;
if (hasF16) {
  await benchMatmul(1024, "float16", "f16");
} else {
  console.log("Skipping f16 benchmarks (shader-f16 not available)");
}

// --- Manual timing for GFLOP/s reporting ---
{
  const n = 1024;
  const key = random.key(99);
  const [k1, k2] = random.split(key, 2);
  const A = random.uniform(k1, [n, n]);
  const B = random.uniform(k2, [n, n]);
  key.dispose();
  k1.dispose();
  k2.dispose();
  await blockUntilReady([A, B]);

  const matmulJit = jit((a, b) => np.matmul(a, b));

  // Warmup (3 times)
  for (let i = 0; i < 3; i++) {
    const C = matmulJit(A, B);
    await blockUntilReady(C);
    C.dispose();
  }

  const gflops = (2 * n * n * n) / 1e9;
  const times = [];
  for (let i = 0; i < 10; i++) {
    const start = performance.now();
    const C = matmulJit(A, B);
    await blockUntilReady(C);
    const end = performance.now();
    C.dispose();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const min = Math.min(...times);
  console.log(`\n=== GFLOP/s Report (${n}x${n} f32 JIT) ===`);
  console.log(`  Times: ${times.map((t) => t.toFixed(1)).join(", ")} ms`);
  console.log(
    `  Avg: ${avg.toFixed(1)} ms → ${(gflops / (avg / 1000)).toFixed(2)} GFLOP/s`,
  );
  console.log(
    `  Min: ${min.toFixed(1)} ms → ${(gflops / (min / 1000)).toFixed(2)} GFLOP/s`,
  );

  // Eager for comparison
  const eagerTimes = [];
  for (let i = 0; i < 10; i++) {
    const start = performance.now();
    const C = np.matmul(A, B);
    await blockUntilReady(C);
    const end = performance.now();
    C.dispose();
    eagerTimes.push(end - start);
  }
  const eagerAvg = eagerTimes.reduce((a, b) => a + b, 0) / eagerTimes.length;
  const eagerMin = Math.min(...eagerTimes);
  console.log(`\n=== GFLOP/s Report (${n}x${n} f32 Eager) ===`);
  console.log(`  Times: ${eagerTimes.map((t) => t.toFixed(1)).join(", ")} ms`);
  console.log(
    `  Avg: ${eagerAvg.toFixed(1)} ms → ${(gflops / (eagerAvg / 1000)).toFixed(2)} GFLOP/s`,
  );
  console.log(
    `  Min: ${eagerMin.toFixed(1)} ms → ${(gflops / (eagerMin / 1000)).toFixed(2)} GFLOP/s`,
  );

  A.dispose();
  B.dispose();
  matmulJit.dispose();
}
