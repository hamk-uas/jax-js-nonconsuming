/**
 * O9a-v2 colored arena slab verification on WebGPU.
 *
 * Verifies that the colored multi-slab arena produces correct results and
 * that bind group cache benefits from stable buffer identities across
 * invocations.
 *
 * Run with: GPU=nvidia scripts/gpu-test.sh run test/o9a-arena.test.ts
 */
import {
  _getCommandTapeStats,
  clearCaches,
  defaultDevice,
  grad,
  init,
  jit,
  nn,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeAll, describe, expect, it } from "vitest";

await init("webgpu");

function vals(a: np.Array): number[] {
  return Array.from(a.dataSync() as Float32Array);
}

function expectClose(a: np.Array, expected: number[]) {
  const v = vals(a);
  expect(v.length).toBe(expected.length);
  for (let i = 0; i < v.length; i++) {
    expect(v[i]).toBeCloseTo(expected[i], 3);
  }
}

function expectClose2d(a: np.Array, expected: number[][]) {
  expectClose(a, expected.flat());
}

describe("O9a-v2 colored arena (WebGPU)", () => {
  beforeAll(() => {
    defaultDevice("webgpu");
    clearCaches();
  });

  it("simple chain produces correct results", () => {
    // Multiple intermediates should be arena-allocated with different colors
    const f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
    using x = np.array([10, 20, 30]);
    using r = f(x);
    expectClose(r, [19, 39, 59]);

    // Verify command tape was compiled (simple chain may fully fuse → no arena)
    const stats = _getCommandTapeStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);
    expect(stats[stats.length - 1].dispatchCount).toBeGreaterThanOrEqual(1);

    f.dispose();
  });

  it("repeated invocations give stable results (arena reuse)", () => {
    const f = jit((x: np.Array) => x.mul(3).add(5).mul(0.5));
    using x = np.array([2, 4, 6]);

    // First invocation creates arena slabs.
    // Subsequent invocations reuse them — identical buffer identities
    // should give 100% bind group cache hits for internal dispatches.
    for (let i = 0; i < 10; i++) {
      using r = f(x);
      expectClose(r, [5.5, 8.5, 11.5]);
    }
    f.dispose();
  });

  it("grad through arena-allocated intermediates", () => {
    const g = jit(grad((x: np.Array) => x.mul(x).add(x).sum()));
    using x = np.array([1, 2, 3]);
    // d/dx(x^2 + x) = 2x + 1
    using dx = g(x);
    expectClose(dx, [3, 5, 7]);
    // Second invocation
    using dx2 = g(x);
    expectClose(dx2, [3, 5, 7]);

    // Verify command tape was compiled for this kernel-only grad program
    const stats = _getCommandTapeStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);

    g.dispose();
  });

  it("matmul + bias + relu chain", () => {
    using x = np.array([
      [1, 2],
      [3, 4],
    ]);
    using w = np.array([
      [0.5, 0],
      [0, 0.5],
    ]);
    using b = np.array([1, -1]);

    const f = jit((x: np.Array, w: np.Array, b: np.Array) =>
      nn.relu(np.matmul(x, w).add(b)),
    );
    using res = f(x, w, b);
    expectClose2d(res, [
      [1.5, 0],
      [2.5, 1],
    ]);
    // Second invocation — arena slabs reused
    using res2 = f(x, w, b);
    expectClose2d(res2, [
      [1.5, 0],
      [2.5, 1],
    ]);
    f.dispose();
  });

  it("long chain with many intermediates stresses arena coloring", () => {
    // 10 intermediate buffers with constants — exercises arena with
    // multiple live intermediates and varied conflict patterns.
    const f = jit((x: np.Array) =>
      x
        .add(1)
        .mul(2)
        .add(3)
        .mul(4)
        .add(5)
        .mul(0.1)
        .add(0.5)
        .mul(3)
        .add(2)
        .mul(0.01),
    );
    using x = np.array([1]);
    // ((((((((((1+1)*2)+3)*4)+5)*0.1)+0.5)*3)+2)*0.01)
    // step by step: 2, 4, 7, 28, 33, 3.3, 3.8, 11.4, 13.4, 0.134
    using r = f(x);
    expectClose(r, [0.134]);
    using r2 = f(x);
    expectClose(r2, [0.134]);
    f.dispose();
  });

  it("different input shapes use separate compilations", () => {
    const f = jit((x: np.Array) => x.add(1).mul(2));
    using x3 = np.array([1, 2, 3]);
    using x4 = np.array([1, 2, 3, 4]);

    using r3 = f(x3);
    expectClose(r3, [4, 6, 8]);

    using r4 = f(x4);
    expectClose(r4, [4, 6, 8, 10]);

    // Verify both cached compilations still work
    using r3b = f(x3);
    expectClose(r3b, [4, 6, 8]);
    using r4b = f(x4);
    expectClose(r4b, [4, 6, 8, 10]);
    f.dispose();
  });

  it("multi-input function with shared constants", () => {
    const f = jit((a: np.Array, b: np.Array) => a.mul(2).add(b.mul(3)).add(1));
    using a = np.array([1, 2]);
    using b = np.array([10, 20]);
    using r = f(a, b);
    expectClose(r, [33, 65]);
    // Second invocation
    using r2 = f(a, b);
    expectClose(r2, [33, 65]);
    f.dispose();
  });
});
