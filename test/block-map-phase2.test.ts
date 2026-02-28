import {
  DType,
  grad,
  init,
  jit,
  jvp,
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

  test("grad through body with intermediate reduction", () => {
    // body: scale each element by block sum → d/dx_i = sum(block) + x_i
    // For block [a,b]: output = [a*(a+b), b*(a+b)]
    // d(sum)/da = (a+b) + a + b = 2(a+b), d(sum)/db = a + (a+b) + b = 2(a+b)
    // Wait, for f(x) = sum(x * sum(x, keepdims))
    // = sum_i x_i * (sum_j x_j) for each block
    // df/dx_i = sum_j x_j + x_i  ... no that's wrong
    // Simpler: body = block * mean(block, keepdims)
    // Use: body = block + sum(block, keepdims) → grad = 1 + blockSize * 1 per elem
    // Actually simplest: body(block) = block * block.sum(keepdims=True)
    // f = sum(block * block.sum()) per block
    // For [a,b]: a*(a+b) + b*(a+b) = (a+b)^2
    // df/da = 2(a+b), df/db = 2(a+b)
    const body = (block: np.Array) => {
      using s = np.sum(block, null, { keepdims: true });
      return np.multiply(block, s);
    };
    const f = (xs: np.Array) => {
      using mapped = lax.blockMap(body, xs, { blockShape: [2] });
      return np.sum(mapped);
    };
    using x = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using g = grad(f)(x);
    // Block [1,2]: sum=3, grad = 2*3 = 6 for each
    // Block [3,4]: sum=7, grad = 2*7 = 14 for each
    expect(g).toBeAllclose([6, 6, 14, 14]);
  });

  // grad(blockMap) with foriLoop: blocked on foriLoop backward pass (needs scan)
  // grad(tiledMatmul): blocked on same (tiledMatmul uses foriLoop internally)

  test("jvp(blockMap) tangent shapes match primal shapes", () => {
    const body = (block: np.Array) => np.multiply(block, block);
    const f = (xs: np.Array) => lax.blockMap(body, xs, { blockShape: [2] });

    using x = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using dx = np.ones([4], { dtype: DType.Float32 });
    const [primal, tangent] = jvp(f, [x], [dx]);
    // primal = x^2 = [1, 4, 9, 16]
    // tangent = 2x * dx = [2, 4, 6, 8]
    expect(primal).toBeAllclose([1, 4, 9, 16]);
    expect(tangent).toBeAllclose([2, 4, 6, 8]);
    primal.dispose();
    tangent.dispose();
  });

  test("grad(blockMap) with non-divisible N", () => {
    const body = (block: np.Array) => np.multiply(block, block);
    const f = (xs: np.Array) => {
      using mapped = lax.blockMap(body, xs, { blockShape: [3] });
      return np.sum(mapped);
    };
    // N=5, blockShape=3 → 2 blocks (3+2), boundary handling
    using x = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using g = grad(f)(x);
    expect(g).toBeAllclose([2, 4, 6, 8, 10]);
  });

  test("vmap(grad(blockMap)) batched gradients", () => {
    const body = (block: np.Array) => np.multiply(block, block);
    const f = (xs: np.Array) => {
      using mapped = lax.blockMap(body, xs, { blockShape: [2] });
      return np.sum(mapped);
    };
    // batch of 2 vectors of length 4
    using xInt = np.arange(8);
    using xFlat = xInt.astype(DType.Float32);
    using x = xFlat.reshape([2, 4]);
    using g = vmap(grad(f))(x);
    // grad of x^2 = 2x for each batch
    using expected = np.multiply(x, 2);
    expect(g).toBeAllclose(expected);
  });

  test("grad(vmap(blockMap)) gradient through batched forward", () => {
    const body = (block: np.Array) => np.multiply(block, block);
    const f = (xs: np.Array) => {
      using mapped = vmap((row: np.Array) =>
        lax.blockMap(body, row, { blockShape: [2] }),
      )(xs);
      return np.sum(mapped);
    };
    using xInt = np.arange(8);
    using xFlat = xInt.astype(DType.Float32);
    using x = xFlat.reshape([2, 4]);
    using g = grad(f)(x);
    // grad of sum(x^2) = 2x
    using expected = np.multiply(x, 2);
    expect(g).toBeAllclose(expected);
  });
});
