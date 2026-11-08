/**
 * M8.1 Benchmark: associativeScan (M7) vs sequential scan.
 *
 * Measures the Kogge-Stone parallel prefix scan compiled to WASM
 * against the sequential lax.scan compiled-loop.
 *
 * Run with:
 *   pnpm vitest bench bench/associative-scan.bench.ts
 */
import {
  defaultDevice,
  DType,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

await init("wasm");

/** Sequential cumulative `op` via the lax.scan compiled loop. */
function sequentialScan(
  op: (c: np.Array, x: np.Array) => np.Array,
  init: number,
) {
  return jit((xs: np.Array) => {
    const [carry, ys] = lax.scan(
      (c: np.Array, x: np.Array) => {
        const nc = op(c, x);
        return [nc, nc];
      },
      np.array(init, { dtype: DType.Float32 }),
      xs,
    );
    carry.dispose();
    return ys;
  });
}

describe("associativeScan vs scan (WASM)", () => {
  for (const N of [1024, 4096]) {
    test(`cumsum N=${N}`, async ({ bench }) => {
      defaultDevice("wasm");
      using xs = np.ones([N]);
      using assoc = jit((v: np.Array) =>
        lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), v),
      );
      using assocRev = jit((v: np.Array) =>
        lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), v, {
          reverse: true,
        }),
      );
      using seq = sequentialScan((c, x) => np.add(c, x), 0);

      await bench.compare(
        bench("associativeScan", () => {
          assoc(xs).dispose();
        }),
        bench("scan", () => {
          seq(xs).dispose();
        }),
        bench("associativeScan reverse", () => {
          assocRev(xs).dispose();
        }),
      );
    });
  }

  test("cumprod N=1024", async ({ bench }) => {
    defaultDevice("wasm");
    using xs = np.full([1024], 1.001);
    using assoc = jit((v: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => a.mul(b), v),
    );
    using seq = sequentialScan((c, x) => c.mul(x), 1);

    await bench.compare(
      bench("associativeScan", () => {
        assoc(xs).dispose();
      }),
      bench("scan", () => {
        seq(xs).dispose();
      }),
    );
  });
});
