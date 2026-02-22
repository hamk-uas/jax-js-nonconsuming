/**
 * M8.1 Benchmark: associativeScan (M7) vs sequential scan.
 *
 * Measures the Kogge-Stone parallel prefix scan compiled to WASM
 * against the sequential lax.scan compiled-loop.
 *
 * Run with:
 *   pnpm build && pnpm vitest bench bench/associative-scan.bench.ts
 */
import {
  defaultDevice,
  DType,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, describe } from "vitest";

await init("wasm");
defaultDevice("wasm");

describe("associativeScan vs scan (WASM)", () => {
  // --- Cumulative sum N=1024 ---
  const xs1k = np.ones([1024]);
  afterAll(() => xs1k.dispose());

  const assocCumsum1k = jit((xs: any) =>
    lax.associativeScan((a: any, b: any) => np.add(a, b), xs),
  );
  assocCumsum1k(xs1k).dispose();
  afterAll(() => assocCumsum1k.dispose());

  bench("associativeScan cumsum N=1024", () => {
    assocCumsum1k(xs1k).dispose();
  });

  const seqCumsum1k = jit((xs: any) => {
    const [_carry, ys] = lax.scan(
      (c: any, x: any) => {
        const nc = np.add(c, x);
        return [nc, nc];
      },
      np.array(0, { dtype: DType.Float32 }),
      xs,
    );
    _carry.dispose();
    return ys;
  });
  seqCumsum1k(xs1k).dispose();
  afterAll(() => seqCumsum1k.dispose());

  bench("scan cumsum N=1024", () => {
    seqCumsum1k(xs1k).dispose();
  });

  // --- Cumulative sum N=4096 ---
  const xs4k = np.ones([4096]);
  afterAll(() => xs4k.dispose());

  const assocCumsum4k = jit((xs: any) =>
    lax.associativeScan((a: any, b: any) => np.add(a, b), xs),
  );
  assocCumsum4k(xs4k).dispose();
  afterAll(() => assocCumsum4k.dispose());

  bench("associativeScan cumsum N=4096", () => {
    assocCumsum4k(xs4k).dispose();
  });

  const seqCumsum4k = jit((xs: any) => {
    const [_carry, ys] = lax.scan(
      (c: any, x: any) => {
        const nc = np.add(c, x);
        return [nc, nc];
      },
      np.array(0, { dtype: DType.Float32 }),
      xs,
    );
    _carry.dispose();
    return ys;
  });
  seqCumsum4k(xs4k).dispose();
  afterAll(() => seqCumsum4k.dispose());

  bench("scan cumsum N=4096", () => {
    seqCumsum4k(xs4k).dispose();
  });

  // --- Cumulative product N=1024 ---
  const xsProd = np.full([1024], 1.001);
  afterAll(() => xsProd.dispose());

  const assocCumprod = jit((xs: any) =>
    lax.associativeScan((a: any, b: any) => a.mul(b), xs),
  );
  assocCumprod(xsProd).dispose();
  afterAll(() => assocCumprod.dispose());

  bench("associativeScan cumprod N=1024", () => {
    assocCumprod(xsProd).dispose();
  });

  const seqCumprod = jit((xs: any) => {
    const [_carry, ys] = lax.scan(
      (c: any, x: any) => {
        const nc = c.mul(x);
        return [nc, nc];
      },
      np.array(1, { dtype: DType.Float32 }),
      xs,
    );
    _carry.dispose();
    return ys;
  });
  seqCumprod(xsProd).dispose();
  afterAll(() => seqCumprod.dispose());

  bench("scan cumprod N=1024", () => {
    seqCumprod(xsProd).dispose();
  });
});
