import {
  clearCaches,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

describe("jit cache const ownership", () => {
  it("clearCaches is safe after foriLoop body calls jit with consts", async () => {
    const f = jit((x: np.Array) => {
      using eye = np.eye(2);
      return np.dot(eye, x);
    });

    using initVec = np.zeros([2]);
    using result = lax.foriLoop(
      0,
      3,
      (_i: np.Array, x: np.Array) => f(x),
      initVec,
    );
    await result.data();

    expect(() => clearCaches()).not.toThrow();
    f.dispose();
  });

  it("clearCaches is safe after scan body calls jit with consts", async () => {
    const f = jit((x: np.Array) => {
      using eye = np.eye(2);
      return np.dot(eye, x);
    });

    using initVec = np.zeros([2]);
    using xs = np.ones([3, 2]);
    const [carry, ys] = lax.scan(
      (c: np.Array, x: np.Array): [np.Array, np.Array] => {
        const y = f(x);
        return [c.add(y), y];
      },
      initVec,
      xs,
    );
    using _carry = carry;
    using _ys = ys;
    await carry.data();
    await ys.data();

    expect(() => clearCaches()).not.toThrow();
    f.dispose();
  });

  it("foriLoop body-created consts stay leak-free after cache clear", async () => {
    using initVec = np.zeros([2]);
    using result = lax.foriLoop(
      0,
      4,
      (_i: np.Array, x: np.Array) => {
        using bias = np.zeros([2]);
        return x.add(bias);
      },
      initVec,
    );

    await result.data();
    expect(() => clearCaches()).not.toThrow();
  });
});
