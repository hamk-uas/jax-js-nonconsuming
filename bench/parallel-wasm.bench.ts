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
 *   pnpm vitest bench bench/parallel-wasm.bench.ts
 */
import {
  defaultDevice,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

await init("wasm");

describe("large elementwise WASM (single-thread baseline)", () => {
  test("chains and exp", async ({ bench }) => {
    defaultDevice("wasm");
    using x1M = np.ones([1_000_000]);
    using x256K = np.ones([262144]);
    using chain3 = jit((x: np.Array) => x.add(1).mul(2).sub(0.5));
    using chain5 = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
    // Transcendental, compute-heavy.
    using exp = jit((x: np.Array) => np.exp(x));

    await bench.compare(
      bench("3-step chain size=1M", () => {
        chain3(x1M).dispose();
      }),
      bench("5-step chain size=256K", () => {
        chain5(x256K).dispose();
      }),
      bench("exp size=1M", () => {
        exp(x1M).dispose();
      }),
    );
  });
});

describe("polymorphic JIT (dynamic_axes, M4)", () => {
  // Variable-length JIT: single compilation reused across different sizes.
  // Without dynamic_axes, each new size triggers recompilation.
  test("cached calls", async ({ bench }) => {
    defaultDevice("wasm");
    using polyF = jit((x: np.Array) => x.mul(2).add(1).sub(0.5), {
      dynamic_axes: { 0: "T" },
    });
    using stack = new DisposableStack();
    const registrations = [];
    for (const size of [100, 150, 200, 500]) {
      const x = stack.use(np.ones([size]));
      const suffix = size === 100 ? "" : " (same compiled program)";
      registrations.push(
        bench(`cached call size=${size}${suffix}`, () => {
          polyF(x).dispose();
        }),
      );
    }
    await bench.compare(...registrations);
  });
});
