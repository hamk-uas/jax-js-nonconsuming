/**
 * M8.1 Benchmark: Mega-Module (M6.1) vs step-by-step JIT on WASM.
 *
 * Measures the impact of compiling entire JitPrograms into a single WASM
 * function, eliminating JS↔WASM boundary crossings between kernel dispatches.
 *
 * Run with:
 *   pnpm vitest bench bench/mega-module.bench.ts
 */
import {
  defaultDevice,
  grad,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

await init("wasm");

describe("mega-module WASM", () => {
  test("jit programs", async ({ bench }) => {
    defaultDevice("wasm");
    using x4k = np.ones([4096]);
    using x64k = np.ones([65536]);
    using a32 = np.ones([32, 32]);
    using b32 = np.ones([32, 32]);

    // 5-step chain (all same-size: recycling + mega-module both activate)
    using chain5 = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
    using multiOut2 = jit((x: np.Array) => [x.add(1), x.mul(2)]);
    using multiOut3 = jit((x: np.Array) => [x.add(1), x.mul(2), x.sub(3)]);
    // Reduction changes the output size; mega-module handles it.
    using reduceSum = jit((x: np.Array) => x.mul(2).add(1).sum());
    // Autodiff-generated program.
    using gradChain = jit(grad((x: np.Array) => x.mul(x).add(x).sum()));
    // Routine steps: mega-module should fall back.
    using matmul2x = jit((a: np.Array, b: np.Array) => {
      const c = np.matmul(a, b);
      return np.matmul(c, a);
    });

    await bench.compare(
      bench("5-step chain size=4096", () => {
        chain5(x4k).dispose();
      }),
      bench("5-step chain size=65536", () => {
        chain5(x64k).dispose();
      }),
      bench("2-output same-size 4096", () => {
        for (const r of multiOut2(x4k)) r.dispose();
      }),
      bench("3-output same-size 4096", () => {
        for (const r of multiOut3(x4k)) r.dispose();
      }),
      bench("chain+reduce size=4096", () => {
        reduceSum(x4k).dispose();
      }),
      bench("grad(chain) size=4096", () => {
        gradChain(x4k).dispose();
      }),
      bench("2x matmul 32x32", () => {
        matmul2x(a32, b32).dispose();
      }),
    );
  });
});
