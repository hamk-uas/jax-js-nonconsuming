// Conv2d benchmark: measures convolution throughput for common CNN inference shapes.
// Target cases from P4 Conv2d Tuning Plan, Phase A.
//
// Input layout: NCHW, kernel layout: OIHW (matching lax.convGeneralDilated defaults).
//
// Run: pnpm build && pnpm vitest bench bench/conv2d.bench.ts
// GPU: pnpm build && scripts/gpu-test.sh bench bench/conv2d.bench.ts

import {
  blockUntilReady,
  type CodeCaptureEntry,
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
  random,
  setCodeCapture,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

// ── Shape definitions ──────────────────────────────────────────────────────────
// Each case: [label, batchSize, inChannels, H, W, outChannels, kH, kW, stride, padding]
const CASES = [
  // Original Phase A cases (dispatch-bound at batch=1)
  ["3x3 s1 SAME 32ch  64x64", 1, 32, 64, 64, 64, 3, 3, 1, "SAME"],
  ["3x3 s1 SAME 64ch  64x64", 1, 64, 64, 64, 64, 3, 3, 1, "SAME"],
  ["3x3 s1 SAME 128ch 32x32", 1, 128, 32, 32, 128, 3, 3, 1, "SAME"],
  ["1x1 pointwise 64ch 64x64", 1, 64, 64, 64, 64, 1, 1, 1, "VALID"],
  ["1x1 pointwise 128ch 32x32", 1, 128, 32, 32, 256, 1, 1, 1, "VALID"],
  ["5x5 s1 SAME 32ch  64x64", 1, 32, 64, 64, 32, 5, 5, 1, "SAME"],
  ["3x3 s2 down 64ch 128x128", 1, 64, 128, 128, 128, 3, 3, 2, "SAME"],

  // Phase C decision-gate cases — larger tensors pushing toward compute-bound
  // 3×3 64ch 64×64 batch=8:  8 × 2×64×64×64×64×9 = 1.21 GFLOP
  ["3x3 B8 64ch  64x64", 8, 64, 64, 64, 64, 3, 3, 1, "SAME"],
  // 3×3 128ch 64×64 batch=8: 8 × 2×128×64×64×128×9 = 9.66 GFLOP
  ["3x3 B8 128ch 64x64", 8, 128, 64, 64, 128, 3, 3, 1, "SAME"],
  // 1×1 256ch 64×64:  1 × 2×256×64×64×256 = 0.54 GFLOP
  ["1x1 pw 256ch 64x64", 1, 256, 64, 64, 256, 1, 1, 1, "VALID"],
] as const;

type Case = (typeof CASES)[number];

// WASM is always available. WebGPU only runs under GPU configs
// (scripts/gpu-test.sh bench bench/conv2d.bench.ts).
const devices = await init("wasm", "webgpu");
const hasWebGPU = devices.includes("webgpu");

// ── Helper: create inputs for a case ───────────────────────────────────────────
function makeInputs(c: Case) {
  const [, N, Cin, H, W, Cout, kH, kW] = c;
  const x = random.uniform(random.key(0), [N, Cin, H, W]);
  const w = random.uniform(random.key(1), [Cout, Cin, kH, kW]);
  return { x, w };
}

// ── Helper: compute GFLOP for a conv ───────────────────────────────────────────
function convGflops(c: Case): number {
  const [, N, Cin, H, W, Cout, kH, kW, stride] = c;
  const outH = Math.ceil(H / stride);
  const outW = Math.ceil(W / stride);
  // 2 * N * Cout * outH * outW * Cin * kH * kW (multiply-accumulate = 2 FLOPs)
  return (2 * N * Cout * outH * outW * Cin * kH * kW) / 1e9;
}

// ── Capture code on first compilation ──────────────────────────────────────────
function captureOnce(): CodeCaptureEntry[] {
  const entries: CodeCaptureEntry[] = [];
  setCodeCapture((e) => entries.push(e));
  return entries;
}

// ── WASM benchmarks ────────────────────────────────────────────────────────────
suite.skipIf(!devices.includes("wasm"))("wasm conv2d", async () => {
  if (!devices.includes("wasm")) return; // vitest bench runs setup even when skipped
  defaultDevice("wasm");

  for (const c of CASES) {
    const [label, , , , , , , , stride, padding] = c;
    const { x, w } = makeInputs(c);
    await blockUntilReady([x, w]);
    afterAll(() => {
      x.dispose();
      w.dispose();
    });

    // Eager (no JIT) — dataSync() forces materialization; without it,
    // dispose() cancels the lazy PendingExecute without running the kernel.
    bench(`${label} eager`, () => {
      const out = lax.convGeneralDilated(x, w, [stride, stride], padding);
      out.dataSync();
      out.dispose();
    });

    // JIT
    const f = jit((a: np.Array, b: np.Array) =>
      lax.convGeneralDilated(a, b, [stride, stride], padding),
    );
    // Warmup
    const warmup = f(x, w);
    warmup.dispose();
    afterAll(() => f.dispose());

    bench(`${label} jit`, () => {
      const out = f(x, w);
      out.dispose();
    });
  }
});

// ── WebGPU benchmarks (run via: scripts/gpu-test.sh bench bench/conv2d.bench.ts)
suite.skipIf(!hasWebGPU)("webgpu conv2d", async () => {
  if (!hasWebGPU) return; // vitest bench runs setup even when skipped
  defaultDevice("webgpu");

  for (const c of CASES) {
    const [label, , , , , , , , stride, padding] = c;
    const gflops = convGflops(c);
    const { x, w } = makeInputs(c);
    await blockUntilReady([x, w]);
    afterAll(() => {
      x.dispose();
      w.dispose();
    });

    // Eager
    bench(`${label} eager`, async () => {
      const out = lax.convGeneralDilated(x, w, [stride, stride], padding);
      await out.blockUntilReady();
      out.dispose();
    });

    // JIT — capture compiled code on warmup
    const entries = captureOnce();
    const f = jit((a: np.Array, b: np.Array) =>
      lax.convGeneralDilated(a, b, [stride, stride], padding),
    );
    try {
      const warmup = f(x, w);
      await warmup.blockUntilReady();
      warmup.dispose();
    } finally {
      setCodeCapture(null);
    }
    afterAll(() => f.dispose());

    // Log dispatch info (code capture fires in ShaderPipelineCache,
    // only on first compilation — not repeatable after clearCaches).
    const kernelCount = entries.filter((e) => e.kind === "kernel").length;
    console.log(
      `[${label}] ${entries.length} dispatches (${kernelCount} kernels), GFLOP=${gflops.toFixed(3)}`,
    );

    bench(`${label} jit`, async () => {
      const out = f(x, w);
      await out.blockUntilReady();
      out.dispose();
    });
  }
});
