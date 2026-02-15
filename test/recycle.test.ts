/**
 * Tests for JIT-level buffer recycling.
 *
 * Buffer recycling replaces free→malloc pairs of the same byte size with a
 * "recycle" step that reuses the backend Slot, avoiding alloc/free overhead.
 */
import {
  defaultDevice,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "@jax-js-nonconsuming/jax";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

describe("buffer recycling", () => {
  it("chain of same-size operations is correct", () => {
    // a → b → c → d, all same shape — intermediates b, c are recyclable.
    using f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
    using x = np.array([1, 2, 3, 4]);
    // (([1,2,3,4] + 1) * 2) - 3 = ([2,3,4,5]*2)-3 = [4,6,8,10]-3 = [1,3,5,7]
    using result = f(x);
    expect(result.js()).toEqual([1, 3, 5, 7]);
  });

  it("recycling preserves correctness with different sizes", () => {
    // Mix of shapes: reduction produces a different size, no recycling there.
    using f = jit((x: np.Array) => x.mul(2).sum().add(1));
    using x = np.array([1, 2, 3]);
    // (1*2 + 2*2 + 3*2) + 1 = 12 + 1 = 13
    using result = f(x);
    expect(result.js()).toEqual(13);
  });

  it("multi-step chain does not leak slots", async () => {
    using f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4));
    using x = np.array([10, 20, 30, 40]);
    using result = f(x);
    await result.data();
  });

  it("works with grad through chained ops", () => {
    // f(x) = x^2, f'(x) = 2x
    const f = (x: np.Array) => np.multiply(x, x).sum();
    const df = grad(f);
    using x = np.array([1, 2, 3]);
    using result = df(x);
    expect(result.js()).toEqual([2, 4, 6]);
  });

  it("works correctly with scan", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };
    using xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    using initCarry = np.array([0, 0]);
    using run = jit((init: np.Array, xs: np.Array) => lax.scan(step, init, xs));
    const [carry, ys] = run(initCarry, xs) as [np.Array, np.Array];
    using _carry = carry;
    using _ys = ys;
    expect(carry.js()).toEqual([9, 12]);
    expect(ys.js()).toEqual([
      [1, 2],
      [4, 6],
      [9, 12],
    ]);
  });
});

describe("buffer recycling (WASM)", () => {
  let previousDevice: string | undefined;

  beforeAll(async () => {
    const devices = await init();
    previousDevice = devices.includes("webgpu")
      ? "webgpu"
      : devices.includes("wasm")
        ? "wasm"
        : undefined;
    if (devices.includes("wasm")) {
      defaultDevice("wasm");
    }
  });

  afterAll(() => {
    if (previousDevice) defaultDevice(previousDevice as any);
  });

  it("chained ops produce correct results on WASM", () => {
    using f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
    using x = np.array([1, 2, 3, 4]);
    using result = f(x);
    expect(result.js()).toEqual([1, 3, 5, 7]);
  });

  it("does not leak slots on WASM", async () => {
    using f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4));
    using x = np.array([10, 20, 30, 40]);
    using result = f(x);
    await result.data();
  });
});
