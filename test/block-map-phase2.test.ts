import {
  DType,
  grad,
  init,
  jit,
  lax,
  numpy as np,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

await init();

describe("lax.blockMap - Phase 2 AD", () => {
  test("grad(sum(block_map(f, xs))) vs finite differences", () => {
    // f(x) = x^2
    const body = (x: np.Array) => np.multiply(x, x);
    const f = (xs: np.Array) => {
      using mapped = lax.blockMap(body, xs, { blockShape: [2] });
      return np.sum(mapped);
    };

    using x = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using g = grad(f)(x);
    // grad of x^2 is 2x -> [2, 4, 6, 8]
    expect(g).toBeAllclose([2, 4, 6, 8]);
  });

  test("jit(grad(block_map(f, xs))) computes correct gradient", () => {
    const body = (x: np.Array) => np.multiply(x, x);
    const f = (xs: np.Array) => {
      using mapped = lax.blockMap(body, xs, { blockShape: [2] });
      return np.sum(mapped);
    };

    const f_jit = jit(grad(f));
    using x = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using g = f_jit(x);
    expect(g).toBeAllclose([2, 4, 6, 8]);
    f_jit.dispose();
  });

  test("vmap(block_map)", () => {
    const body = (x: np.Array) => np.add(x, 1);
    const f = (xs: np.Array) => lax.blockMap(body, xs, { blockShape: [2] });

    // batch of 3 vectors of length 4
    using xInt = np.arange(12);
    using xReshaped = xInt.reshape([3, 4]);
    using x = xReshaped.astype(DType.Float32);
    using mapped = vmap(f)(x);

    using expected = np.add(x, 1);
    expect(mapped).toBeAllclose(expected);
  });
});
