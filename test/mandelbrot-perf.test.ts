import {
  _lastForiRewritten,
  defaultDevice,
  getWebGPUDevice,
  init,
  jit,
  lax,
  numpy as np,
  setDebug,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

const width = 1000,
  height = 800;

function calculateMandelbrotForiLoop(iters: number): np.Array {
  using x = np.linspace(-2, 0.5, width);
  using y = np.linspace(-1, 1, height);
  const [X, Y] = np.meshgrid([x, y]);

  using f = jit(
    (
      A: np.Array,
      B: np.Array,
      V: np.Array,
      X: np.Array,
      Y: np.Array,
    ): np.Array => {
      type Carry = { A: np.Array; B: np.Array; V: np.Array };
      const result = lax.foriLoop(
        0,
        iters,
        (_i: np.Array, carry: Carry): Carry => {
          const { A, B, V } = carry;
          using Asq = A.mul(A);
          using Bsq = B.mul(B);
          using magSq = Asq.add(Bsq);
          using mask = magSq.less(100).astype(np.float32);
          const newV = V.add(mask);
          using diffSq = Asq.sub(Bsq);
          using realShifted = diffSq.add(X);
          const newA = np.clip(realShifted, -50, 50);
          using cross = A.mul(B);
          using crossScaled = cross.mul(2);
          using imagShifted = crossScaled.add(Y);
          const newB = np.clip(imagShifted, -50, 50);
          return { A: newA, B: newB, V: newV };
        },
        { A, B, V },
      );
      result.A.dispose();
      result.B.dispose();
      X.dispose();
      Y.dispose();
      return result.V;
    },
  );

  using A = np.zeros(X.shape);
  using B = np.zeros(Y.shape);
  using V = np.zeros(X.shape);
  const V2 = f(A, B, V, X, Y);
  X.dispose();
  Y.dispose();
  return V2;
}

await init("webgpu");
defaultDevice("webgpu");

describe("mandelbrot foriLoop perf diagnostic", () => {
  it("measures performance", async () => {
    const device = getWebGPUDevice();
    console.log(
      `maxComputeWorkgroupStorageSize: ${device.limits.maxComputeWorkgroupStorageSize}`,
    );
    console.log(
      `maxComputeWorkgroupSizeX: ${device.limits.maxComputeWorkgroupSizeX}`,
    );
    console.log(`adapterInfo: ${JSON.stringify(device.adapterInfo)}`);

    setDebug(1);
    // 1st run - compilation
    const t1 = performance.now();
    {
      using result = calculateMandelbrotForiLoop(100);
      await result.data();
      const elapsed1 = performance.now() - t1;
      console.log(
        `1st run: ${elapsed1.toFixed(1)} ms, shape=${result.shape}, foriRewritten=${_lastForiRewritten()}`,
      );
    }

    setDebug(0);

    // 2nd run - cached
    const t2 = performance.now();
    {
      using result = calculateMandelbrotForiLoop(100);
      await result.data();
      const elapsed2 = performance.now() - t2;
      console.log(`2nd run: ${elapsed2.toFixed(1)} ms`);
    }

    // 3rd run - cached
    const t3 = performance.now();
    {
      using result = calculateMandelbrotForiLoop(100);
      await result.data();
      const elapsed3 = performance.now() - t3;
      console.log(`3rd run: ${elapsed3.toFixed(1)} ms`);
    }

    expect(true).toBe(true);
  });
});
