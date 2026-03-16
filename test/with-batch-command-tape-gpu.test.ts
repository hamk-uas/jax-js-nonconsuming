import {
  clearCaches,
  defaultDevice,
  DType,
  init,
  jit,
  lax,
  numpy as np,
  reverse,
  scatterAdd,
  withBatch,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";

const devicesAvailable = await init("webgpu");
const hasWebGPU = devicesAvailable.includes("webgpu");

function vals(a: np.Array): number[] {
  return Array.from(a.dataSync() as Float32Array);
}

describe.skipIf(!hasWebGPU)("withBatch + command tape GPU correctness", () => {
  beforeAll(() => {
    defaultDevice("webgpu");
  });

  beforeEach(() => {
    clearCaches();
  });

  it("matches unbatched execution across multiple tape programs", () => {
    const reverseStep = jit((x: np.Array) => reverse(x, 0) as np.Array);
    const dusStep = jit((dst: np.Array, src: np.Array) =>
      lax.dynamicUpdateSlice(dst, src, 2, 0),
    );
    const scatterStep = jit(
      (target: np.Array, idx: np.Array, updates: np.Array) =>
        scatterAdd(target, idx, updates, 0) as np.Array,
    );

    using expectedSeed = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
    using actualSeed = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
    using src = np.array([90, 91, 92]);
    using idx = np.array([1, 3, 5], { dtype: DType.Int32 });
    using updates = np.array([10, 20, 30]);

    let expected = expectedSeed as np.Array;
    {
      using prev = expected;
      expected = reverseStep(prev);
    }
    {
      using prev = expected;
      expected = dusStep(prev, src) as np.Array;
    }
    {
      using prev = expected;
      expected = scatterStep(prev, idx, updates);
    }

    let actual = actualSeed as np.Array;
    withBatch(() => {
      using prev = actual;
      actual = reverseStep(prev);
      using prev2 = actual;
      actual = dusStep(prev2, src) as np.Array;
      using prev3 = actual;
      actual = scatterStep(prev3, idx, updates);
    });

    expect(vals(actual)).toEqual(vals(expected));

    reverseStep.dispose();
    dusStep.dispose();
    scatterStep.dispose();
    actual.dispose();
    expected.dispose();
  });
});
