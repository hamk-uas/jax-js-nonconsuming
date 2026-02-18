/**
 * @file Benchmark: wasmblr peephole optimizer on Kalman filter scan.
 *
 * Run with:
 *   pnpm test bench/peephole.bench.ts
 *
 * Compares WASM compiled-loop performance with peephole on vs off
 * using a diagonal Kalman filter (N=200 steps, state=[4]).
 */

import {
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
  setWasmPeephole,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init("wasm");

suite.skipIf(!devices.includes("wasm"))("peephole benchmark (WASM)", () => {
  defaultDevice("wasm");

  // --- Kalman filter benchmark -----------------------------------------------
  // Constants extracted outside jit to avoid anonymous constant leak.

  const kProcessNoise = np.array([0.01, 0.01, 0.01, 0.01]);
  const kMeasNoise = np.array([0.1]);
  const kH = np.array([1, 0, 0, 0]);
  const kOnesMask = np.ones([4]);
  const kInitState = np.zeros([4]);
  const kInitCov = np.ones([4]);
  const kObs = np.ones([200, 1]);

  type Carry = { state: np.Array; covDiag: np.Array };

  const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
    const { state, covDiag } = carry;
    const predCov = covDiag.add(kProcessNoise);
    const innovation = x.sub(np.sum(state.mul(kH)));
    const S = np.sum(predCov.mul(kH).mul(kH)).add(kMeasNoise);
    const K = predCov.mul(kH).div(S);
    const newState = state.add(K.mul(innovation));
    const newCov = predCov.mul(kOnesMask.sub(K.mul(kH)));
    return [{ state: newState, covDiag: newCov }, state];
  };

  function makeKalmanJit() {
    return jit(() =>
      lax.scan(step, { state: kInitState, covDiag: kInitCov }, kObs),
    );
  }

  // Baseline: peephole OFF
  {
    setWasmPeephole(false);
    const jitOff = makeKalmanJit();
    const [wc, wy] = jitOff() as [any, np.Array];
    tree.dispose(wc);
    wy.dispose();

    afterAll(() => jitOff.dispose());

    bench("kalman N=200 state=4 [peephole OFF]", () => {
      const [c, y] = jitOff() as [any, np.Array];
      tree.dispose(c);
      y.dispose();
    });
  }

  // Optimized: peephole ON
  {
    setWasmPeephole(true);
    const jitOn = makeKalmanJit();
    const [wc, wy] = jitOn() as [any, np.Array];
    tree.dispose(wc);
    wy.dispose();
    setWasmPeephole(false);

    afterAll(() => jitOn.dispose());

    bench("kalman N=200 state=4 [peephole ON]", () => {
      const [c, y] = jitOn() as [any, np.Array];
      tree.dispose(c);
      y.dispose();
    });
  }

  // --- Cumsum benchmark ------

  {
    const xs = np.ones([500, 64]);
    const cumsumInit = np.zeros([64]);

    // OFF
    setWasmPeephole(false);
    const cumsumOff = jit((xs: np.Array) =>
      lax.scan(
        (carry: np.Array, x: np.Array) => {
          const c = carry.add(x);
          return [c, c];
        },
        cumsumInit,
        xs,
      ),
    );
    const [wc1, wy1] = cumsumOff(xs) as [np.Array, np.Array];
    wc1.dispose();
    wy1.dispose();

    // ON
    setWasmPeephole(true);
    const cumsumOn = jit((xs: np.Array) =>
      lax.scan(
        (carry: np.Array, x: np.Array) => {
          const c = carry.add(x);
          return [c, c];
        },
        cumsumInit,
        xs,
      ),
    );
    const [wc2, wy2] = cumsumOn(xs) as [np.Array, np.Array];
    wc2.dispose();
    wy2.dispose();
    setWasmPeephole(false);

    afterAll(() => {
      cumsumOff.dispose();
      cumsumOn.dispose();
      cumsumInit.dispose();
      xs.dispose();
    });

    bench("cumsum N=500 size=64 [peephole OFF]", () => {
      const [c, y] = cumsumOff(xs) as [np.Array, np.Array];
      c.dispose();
      y.dispose();
    });

    bench("cumsum N=500 size=64 [peephole ON]", () => {
      const [c, y] = cumsumOn(xs) as [np.Array, np.Array];
      c.dispose();
      y.dispose();
    });
  }

  // --- Elementwise chain (non-scan) benchmark ---

  {
    const x = np.ones([4096]);

    // OFF
    setWasmPeephole(false);
    const chainOff = jit((x: np.Array) => x.add(x).mul(x).sub(x).add(x));
    const woff = chainOff(x) as np.Array;
    woff.dispose();

    // ON
    setWasmPeephole(true);
    const chainOn = jit((x: np.Array) => x.add(x).mul(x).sub(x).add(x));
    const won = chainOn(x) as np.Array;
    won.dispose();
    setWasmPeephole(false);

    afterAll(() => {
      chainOff.dispose();
      chainOn.dispose();
      x.dispose();
    });

    bench("chain x5 size=4096 [peephole OFF]", () => {
      const r = chainOff(x) as np.Array;
      r.dispose();
    });

    bench("chain x5 size=4096 [peephole ON]", () => {
      const r = chainOn(x) as np.Array;
      r.dispose();
    });
  }

  afterAll(() => {
    kProcessNoise.dispose();
    kMeasNoise.dispose();
    kH.dispose();
    kOnesMask.dispose();
    kInitState.dispose();
    kInitCov.dispose();
    kObs.dispose();
  });
});
