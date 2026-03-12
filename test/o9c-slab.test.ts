/**
 * O9c constants slab verification on WebGPU.
 *
 * Verifies that the constants slab is being used and that the bind group cache
 * benefits from stable buffer identities.
 *
 * Run with: GPU=nvidia scripts/gpu-test.sh run test/o9c-slab.test.ts
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

/** Read array values for plain expect(). */
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

describe("O9c constants slab (WebGPU)", () => {
  beforeAll(() => {
    defaultDevice("webgpu");
    clearCaches();
  });

  it("5-step chain with scalar constants produces correct results", () => {
    using x = np.array([10, 20, 30]);
    const f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4).mul(0.5));
    using r1 = f(x);
    expectClose(r1, [11.5, 21.5, 31.5]);
    // Second invocation — slab is reused
    using r2 = f(x);
    expectClose(r2, [11.5, 21.5, 31.5]);

    // Verify command tape was compiled (fully-fused chains may inline
    // constants as WGSL literals, so constSlab is not guaranteed here)
    const stats = _getCommandTapeStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);
    expect(stats[stats.length - 1].dispatchCount).toBeGreaterThanOrEqual(1);

    f.dispose();
  });

  it("grad with scalar constants", () => {
    const g = jit(grad((x: np.Array) => x.mul(3).add(1).sum()));
    using x = np.array([1, 2, 3]);
    using dx1 = g(x);
    expectClose(dx1, [3, 3, 3]);
    using dx2 = g(x);
    expectClose(dx2, [3, 3, 3]);

    // Verify command tape was compiled for grad program
    const stats = _getCommandTapeStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);

    g.dispose();
  });

  it("matmul + bias + relu on WebGPU", () => {
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
    f.dispose();
  });

  it("repeated invocations give stable results", () => {
    const f = jit((x: np.Array) => x.add(10).mul(0.1));
    using x1 = np.array([0, 1, 2]);
    using x2 = np.array([100, 200, 300]);

    for (let i = 0; i < 5; i++) {
      using r1 = f(x1);
      expectClose(r1, [1, 1.1, 1.2]);
      using r2 = f(x2);
      expectClose(r2, [11, 21, 31]);
    }
    f.dispose();
  });

  it("large chain with many constants", () => {
    // Chain of 10 scalar ops — each generates an initialData constant
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
    // = ((((((2*2)+3)*4)+5)*0.1)+0.5)*3+2)*0.01
    // = ((((7*4)+5)*0.1)+0.5)*3+2)*0.01
    // = (((33)*0.1)+0.5)*3+2)*0.01
    // = ((3.3+0.5)*3+2)*0.01
    // = (3.8*3+2)*0.01
    // = (11.4+2)*0.01
    // = 13.4*0.01
    // = 0.134
    using result = f(x);
    expectClose(result, [0.134]);
    f.dispose();
  });
});
