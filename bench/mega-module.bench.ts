/**
 * M8.1 Benchmark: Mega-Module (M6.1) vs step-by-step JIT on WASM.
 *
 * Measures the impact of compiling entire JitPrograms into a single WASM
 * function, eliminating JS↔WASM boundary crossings between kernel dispatches.
 *
 * Run with:
 *   pnpm build && pnpm vitest bench bench/mega-module.bench.ts
 */
import {
  defaultDevice,
  grad,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, describe } from "vitest";

await init("wasm");
defaultDevice("wasm");

describe("mega-module WASM", () => {
  // --- 5-step chain (all same-size: recycling + mega-module both activate) ---
  const x4k = np.ones([4096]);
  afterAll(() => x4k.dispose());

  const chain5 = jit((x: any) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
  chain5(x4k).dispose(); // warmup
  afterAll(() => chain5.dispose());

  bench("5-step chain size=4096", () => {
    chain5(x4k).dispose();
  });

  // --- Same chain at 64K (larger data, same dispatch structure) ---
  const x64k = np.ones([65536]);
  afterAll(() => x64k.dispose());

  const chain5_64k = jit((x: any) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
  chain5_64k(x64k).dispose();
  afterAll(() => chain5_64k.dispose());

  bench("5-step chain size=65536", () => {
    chain5_64k(x64k).dispose();
  });

  // --- Multi-output (2 outputs of same size) ---
  const multiOut2 = jit((x: any) => [x.add(1), x.mul(2)]);
  multiOut2(x4k).forEach((r: any) => r.dispose());
  afterAll(() => multiOut2.dispose());

  bench("2-output same-size 4096", () => {
    multiOut2(x4k).forEach((r: any) => r.dispose());
  });

  // --- Multi-output (3 outputs of same size) ---
  const multiOut3 = jit((x: any) => [x.add(1), x.mul(2), x.sub(3)]);
  multiOut3(x4k).forEach((r: any) => r.dispose());
  afterAll(() => multiOut3.dispose());

  bench("3-output same-size 4096", () => {
    multiOut3(x4k).forEach((r: any) => r.dispose());
  });

  // --- Reduction (sum — changes output size, mega-module handles it) ---
  const reduceSum = jit((x: any) => x.mul(2).add(1).sum());
  reduceSum(x4k).dispose();
  afterAll(() => reduceSum.dispose());

  bench("chain+reduce size=4096", () => {
    reduceSum(x4k).dispose();
  });

  // --- Grad through chain (tests mega-module with autodiff-generated programs) ---
  const gradChain = jit(grad((x: any) => x.mul(x).add(x).sum()));
  gradChain(x4k).dispose();
  afterAll(() => gradChain.dispose());

  bench("grad(chain) size=4096", () => {
    gradChain(x4k).dispose();
  });

  // --- Two matmuls (Routine — mega-module should fall back) ---
  const a32 = np.ones([32, 32]);
  const b32 = np.ones([32, 32]);
  afterAll(() => {
    a32.dispose();
    b32.dispose();
  });

  const matmul2x = jit((a: any, b: any) => {
    const c = np.matmul(a, b);
    return np.matmul(c, a);
  });
  matmul2x(a32, b32).dispose();
  afterAll(() => matmul2x.dispose());

  bench("2x matmul 32x32", () => {
    matmul2x(a32, b32).dispose();
  });
});
