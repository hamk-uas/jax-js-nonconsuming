/**
 * M8.1 Benchmark: WASM parallel dispatch (M5) and polymorphic shapes (M4).
 *
 * Two benchmark categories:
 * 1. Large elementwise operations — measures single-thread throughput.
 *    Note: WasmWorkerPool parallel dispatch requires SharedArrayBuffer
 *    (COOP/COEP headers), which is unavailable in Vitest's iframe
 *    runner. These benchmarks establish single-thread baselines.
 *
 * 2. Variable-length JIT — verifies that dynamic_axes avoids recompilation
 *    when input sizes change on the dynamic axis.
 *
 * Run with:
 *   pnpm build && pnpm vitest bench bench/parallel-wasm.bench.ts
 */
import {
  defaultDevice,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, describe } from "vitest";

await init("wasm");
defaultDevice("wasm");

describe("large elementwise WASM (single-thread baseline)", () => {
  // --- 1M elements: typical ML intermediate size ---
  const x1M = np.ones([1_000_000]);
  afterAll(() => x1M.dispose());

  const chain3_1M = jit((x: any) => x.add(1).mul(2).sub(0.5));
  chain3_1M(x1M).dispose(); // warmup
  afterAll(() => chain3_1M.dispose());

  bench("3-step chain size=1M", () => {
    chain3_1M(x1M).dispose();
  });

  // --- 256K elements ---
  const x256K = np.ones([262144]);
  afterAll(() => x256K.dispose());

  const chain5_256K = jit((x: any) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
  chain5_256K(x256K).dispose();
  afterAll(() => chain5_256K.dispose());

  bench("5-step chain size=256K", () => {
    chain5_256K(x256K).dispose();
  });

  // --- Single exp (transcendental, compute-heavy) ---
  const exp1M = jit((x: any) => np.exp(x));
  exp1M(x1M).dispose();
  afterAll(() => exp1M.dispose());

  bench("exp size=1M", () => {
    exp1M(x1M).dispose();
  });
});

describe("polymorphic JIT (dynamic_axes, M4)", () => {
  // Variable-length JIT: single compilation reused across different sizes.
  // Without dynamic_axes, each new size triggers recompilation.

  const polyF = jit((x: any) => x.mul(2).add(1).sub(0.5), {
    dynamic_axes: { 0: "T" },
  });

  // Warmup with first size to trigger compilation
  const x100 = np.ones([100]);
  polyF(x100).dispose();
  afterAll(() => {
    x100.dispose();
    polyF.dispose();
  });

  // Create arrays of different sizes on the dynamic axis
  const x150 = np.ones([150]);
  const x200 = np.ones([200]);
  const x500 = np.ones([500]);
  afterAll(() => {
    x150.dispose();
    x200.dispose();
    x500.dispose();
  });

  bench("cached call size=100", () => {
    polyF(x100).dispose();
  });

  bench("cached call size=150 (same compiled program)", () => {
    polyF(x150).dispose();
  });

  bench("cached call size=200 (same compiled program)", () => {
    polyF(x200).dispose();
  });

  bench("cached call size=500 (same compiled program)", () => {
    polyF(x500).dispose();
  });
});
