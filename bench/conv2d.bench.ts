// Conv2d benchmark: measures convolution throughput for common CNN inference shapes.
// Target cases from P4 Conv2d Tuning Plan, Phase A.
//
// Input layout: NCHW, kernel layout: OIHW (matching lax.convGeneralDilated defaults).
//
// Run: pnpm vitest bench bench/conv2d.bench.ts
// GPU: scripts/gpu-test.sh bench bench/conv2d.bench.ts

import {
  _lastConvRewritten,
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
import { afterEach, describe, test } from "vitest";

// Internal-only toggle: not part of public API, imported directly from source.
import { _setConvRewriteEnabled } from "../src/frontend/jit";

// ── Shape definitions ──────────────────────────────────────────────────────────
// Each case: [label, batchSize, inChannels, H, W, outChannels, kH, kW, stride, padding]
const PHASE_A_CASES = [
  // Original Phase A cases (dispatch-bound at batch=1)
  ["3x3 s1 SAME 32ch  64x64", 1, 32, 64, 64, 64, 3, 3, 1, "SAME"],
  ["3x3 s1 SAME 64ch  64x64", 1, 64, 64, 64, 64, 3, 3, 1, "SAME"],
  ["3x3 s1 SAME 128ch 32x32", 1, 128, 32, 32, 128, 3, 3, 1, "SAME"],
  ["1x1 pointwise 64ch 64x64", 1, 64, 64, 64, 64, 1, 1, 1, "VALID"],
  ["1x1 pointwise 128ch 32x32", 1, 128, 32, 32, 256, 1, 1, 1, "VALID"],
  ["5x5 s1 SAME 32ch  64x64", 1, 32, 64, 64, 32, 5, 5, 1, "SAME"],
  ["3x3 s2 down 64ch 128x128", 1, 64, 128, 128, 128, 3, 3, 2, "SAME"],
] as const;

// Phase C decision-gate cases — larger tensors pushing the GPU toward
// compute-bound. WebGPU only: single-threaded WASM runs conv at ~1 GFLOP/s,
// so one benchmark of the 9.66 GFLOP case would take many minutes.
const PHASE_C_CASES = [
  // 3×3 64ch 64×64 batch=8:  8 × 2×64×64×64×64×9 = 2.42 GFLOP
  ["3x3 B8 64ch  64x64", 8, 64, 64, 64, 64, 3, 3, 1, "SAME"],
  // 3×3 128ch 64×64 batch=8: 8 × 2×128×64×64×128×9 = 9.66 GFLOP
  ["3x3 B8 128ch 64x64", 8, 128, 64, 64, 128, 3, 3, 1, "SAME"],
  // 1×1 256ch 64×64:  1 × 2×256×64×64×256 = 0.54 GFLOP
  ["1x1 pw 256ch 64x64", 1, 256, 64, 64, 256, 1, 1, 1, "VALID"],
] as const;

type Case = (typeof PHASE_A_CASES)[number] | (typeof PHASE_C_CASES)[number];

// WASM is always available. WebGPU only runs under GPU configs
// (scripts/gpu-test.sh bench bench/conv2d.bench.ts).
const devices = await init("wasm", "webgpu");

// tinybench runs at least 64 measured + 16 warmup iterations by default, and
// the larger cases take up to ~1 s per eager WASM call. A synchronous
// benchmark function also blocks the browser's event loop, so the test timeout
// cannot interrupt it. The 1 s time budget still gives fast cases many samples.
const RUN_OPTIONS = { iterations: 10, warmupIterations: 2 };

// ── Helper: create inputs for a case ───────────────────────────────────────────
function makeInputs(c: Case, stack: DisposableStack) {
  const [, N, Cin, H, W, Cout, kH, kW] = c;
  const k0 = stack.use(random.key(0));
  const k1 = stack.use(random.key(1));
  const x = stack.use(random.uniform(k0, [N, Cin, H, W]));
  const w = stack.use(random.uniform(k1, [Cout, Cin, kH, kW]));
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

// ── WASM benchmarks ────────────────────────────────────────────────────────────
describe.skipIf(!devices.includes("wasm"))("wasm conv2d", () => {
  for (const c of PHASE_A_CASES) {
    const [label, , , , , , , , stride, padding] = c;
    test(label, async ({ bench }) => {
      defaultDevice("wasm");
      using stack = new DisposableStack();
      const { x, w } = makeInputs(c, stack);
      await blockUntilReady([x, w]);
      using f = jit((a: np.Array, b: np.Array) =>
        lax.convGeneralDilated(a, b, [stride, stride], padding),
      );

      await bench.compare(
        // dataSync() forces materialization; without it, dispose() cancels
        // the lazy PendingExecute without running the kernel.
        bench("eager", () => {
          using out = lax.convGeneralDilated(x, w, [stride, stride], padding);
          out.dataSync();
        }),
        bench("jit", () => {
          f(x, w).dispose();
        }),
        RUN_OPTIONS,
      );
    });
  }
});

// ── WebGPU benchmarks (run via: scripts/gpu-test.sh bench bench/conv2d.bench.ts)
describe.skipIf(!devices.includes("webgpu"))("webgpu conv2d", () => {
  for (const c of [...PHASE_A_CASES, ...PHASE_C_CASES]) {
    const [label, , , , , , , , stride, padding] = c;
    test(label, async ({ bench }) => {
      defaultDevice("webgpu");
      using stack = new DisposableStack();
      const { x, w } = makeInputs(c, stack);
      await blockUntilReady([x, w]);
      using f = jit((a: np.Array, b: np.Array) =>
        lax.convGeneralDilated(a, b, [stride, stride], padding),
      );

      // Code capture fires only when a shader is first compiled, so count
      // the dispatches on the first call, before the benchmarks run.
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));
      try {
        using first = f(x, w);
        await first.blockUntilReady();
      } finally {
        setCodeCapture(null);
      }
      const kernelCount = entries.filter((e) => e.kind === "kernel").length;
      console.log(
        `[${label}] ${entries.length} dispatches (${kernelCount} kernels), ` +
          `GFLOP=${convGflops(c).toFixed(3)}`,
      );

      await bench.compare(
        bench("eager", async () => {
          using out = lax.convGeneralDilated(x, w, [stride, stride], padding);
          await out.blockUntilReady();
        }),
        bench("jit", async () => {
          using out = f(x, w);
          await out.blockUntilReady();
        }),
        RUN_OPTIONS,
      );
    });
  }
});

// ── C.3 block_map vs generic-dot A/B (same workload) ───────────────────────────
// Runs the exact same conv workload with rewrite enabled (block_map) and disabled
// (generic-dot). This isolates the rewrite's impact from workload differences.
// [label, batch, channels (in = out), spatial size]
const AB_CASES = {
  wasm: [
    ["3x3 4ch 16x16", 1, 4, 16],
    ["3x3 4ch 32x32", 1, 4, 32],
    ["3x3 8ch 64x64", 1, 8, 64],
  ],
  webgpu: [
    ["3x3 32ch 64x64", 1, 32, 64],
    ["3x3 128ch 32x32", 1, 128, 32],
    ["3x3 B8 128ch 64x64", 8, 128, 64],
  ],
} as const;

for (const device of ["wasm", "webgpu"] as const) {
  describe.skipIf(!devices.includes(device))(
    `${device} C.3 block_map vs generic-dot (same workload)`,
    () => {
      afterEach(() => _setConvRewriteEnabled(true));

      for (const [label, N, C, S] of AB_CASES[device]) {
        test(label, async ({ bench }) => {
          defaultDevice(device);
          using k0 = random.key(0);
          using k1 = random.key(1);
          using x = random.uniform(k0, [N, C, S, S]);
          using w = random.uniform(k1, [C, C, 3, 3]);
          await blockUntilReady([x, w]);
          // The rewrite setting is part of the compile cache key, so each
          // setting compiles and caches its own program for this function.
          using f = jit((a: np.Array, b: np.Array) =>
            lax.convGeneralDilated(a, b, [1, 1], "SAME"),
          );
          for (const enabled of [true, false]) {
            _setConvRewriteEnabled(enabled);
            using out = f(x, w);
            await out.blockUntilReady();
            console.log(
              `[C.3 A/B] ${device} ${label} rewrite ${enabled ? "on" : "off"}: ` +
                `rewritten=${_lastConvRewritten()}`,
            );
          }

          const run = async () => {
            using out = f(x, w);
            await out.blockUntilReady();
          };
          // Tasks run one after another; a task's beforeAll runs before both
          // its warmup and its measured phase.
          await bench.compare(
            bench(
              "block_map",
              { beforeAll: () => _setConvRewriteEnabled(true) },
              run,
            ),
            bench(
              "generic-dot",
              { beforeAll: () => _setConvRewriteEnabled(false) },
              run,
            ),
            RUN_OPTIONS,
          );
        });
      }
    },
  );
}
