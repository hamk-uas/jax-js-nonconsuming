import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
} from "@jax-js-nonconsuming/jax";
import { afterAll, bench, suite } from "vitest";

const devices = await init("wasm", "webgpu");
const hasWebGPU = devices.includes("webgpu");

suite.skipIf(!devices.includes("wasm"))("wasm scan", async () => {
  defaultDevice("wasm");

  // cumsum: N=100, SIZE=64
  {
    const xs = np.ones([100, 64]);
    await blockUntilReady(xs);

    const cumsumJit = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const c = carry.add(x);
          return [c, c];
        },
        np.zeros([64]),
        xs,
      );
    });
    // Warmup to trigger compilation
    const [wc, wy] = cumsumJit(xs);
    wc.dispose();
    wy.dispose();

    afterAll(() => {
      cumsumJit.dispose();
      xs.dispose();
    });

    bench("cumsum N=100 size=64", () => {
      const [c, y] = cumsumJit(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
  }

  // cumsum large: N=500, SIZE=256
  {
    const xs = np.ones([500, 256]);
    await blockUntilReady(xs);

    const cumsumJit = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const c = carry.add(x);
          return [c, c];
        },
        np.zeros([256]),
        xs,
      );
    });
    const [wc, wy] = cumsumJit(xs);
    wc.dispose();
    wy.dispose();

    afterAll(() => {
      cumsumJit.dispose();
      xs.dispose();
    });

    bench("cumsum N=500 size=256", () => {
      const [c, y] = cumsumJit(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
  }

  // carry-only scan: N=200, SIZE=32
  {
    const carryOnlyJit = jit(() => {
      return lax.scan(
        (carry: any, _x: any) => {
          const c = carry.add(np.ones([32]));
          return [c, null];
        },
        np.zeros([32]),
        null,
        { length: 200 },
      );
    });
    const [wc] = carryOnlyJit();
    wc.dispose();

    afterAll(() => {
      carryOnlyJit.dispose();
    });

    bench("carry-only N=200 size=32", () => {
      const [c] = carryOnlyJit() as [any, any];
      c.dispose();
    });
  }

  // reduction body: N=100, SIZE=64
  {
    const xs = np.ones([100, 64]);
    await blockUntilReady(xs);

    const reductionJit = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const s = carry.add(np.sum(x));
          return [s, s];
        },
        np.zeros([]),
        xs,
      );
    });
    const [wc, wy] = reductionJit(xs);
    wc.dispose();
    wy.dispose();

    afterAll(() => {
      reductionJit.dispose();
      xs.dispose();
    });

    bench("reduction N=100 size=64", () => {
      const [c, y] = reductionJit(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
  }

  // reverse scan: N=200, SIZE=64
  {
    const xs = np.ones([200, 64]);
    await blockUntilReady(xs);

    const reverseJit = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const c = carry.add(x);
          return [c, c];
        },
        np.zeros([64]),
        xs,
        { reverse: true },
      );
    });
    const [wc, wy] = reverseJit(xs);
    wc.dispose();
    wy.dispose();

    afterAll(() => {
      reverseJit.dispose();
      xs.dispose();
    });

    bench("reverse N=200 size=64", () => {
      const [c, y] = reverseJit(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
  }

  // Kalman filter: N=200 steps, state=[4], cov=[4] (diagonal approx)
  // Simplified 1D Kalman with diagonal covariance (elementwise ops only)
  {
    const kalmanJit = jit(() => {
      const processNoise = np.array([0.01, 0.01, 0.01, 0.01]);
      const measNoise = np.array([0.1]);
      const H = np.array([1, 0, 0, 0]); // observe first state element

      type Carry = { state: np.Array; covDiag: np.Array };

      const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
        const { state, covDiag } = carry;
        // Predict: state propagates, cov grows
        const predState = state;
        const predCov = covDiag.add(processNoise);
        // Update: scalar innovation, diagonal Kalman gain
        const innovation = x.sub(np.sum(predState.mul(H)));
        const S = np.sum(predCov.mul(H).mul(H)).add(measNoise);
        const K = predCov.mul(H).div(S);
        const newState = predState.add(K.mul(innovation));
        const newCov = predCov.mul(np.ones([4]).sub(K.mul(H)));
        return [{ state: newState, covDiag: newCov }, predState];
      };

      const initState = np.zeros([4]);
      const initCov = np.ones([4]);
      const obs = np.ones([200, 1]); // 200 scalar observations
      return lax.scan(step, { state: initState, covDiag: initCov }, obs);
    });

    const [wCarry, wY] = kalmanJit();
    (wCarry as any).state.dispose();
    (wCarry as any).covDiag.dispose();
    wY.dispose();

    afterAll(() => {
      kalmanJit.dispose();
    });

    bench("kalman N=200 state=4", () => {
      const [c, y] = kalmanJit() as [any, any];
      c.state.dispose();
      c.covDiag.dispose();
      y.dispose();
    });
  }
});

suite.skipIf(!hasWebGPU)("webgpu scan", async () => {
  if (!hasWebGPU) return;
  defaultDevice("webgpu");

  // cumsum: N=100, SIZE=64
  {
    const xs = np.ones([100, 64]);
    await blockUntilReady(xs);

    const cumsumJit = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const c = carry.add(x);
          return [c, c];
        },
        np.zeros([64]),
        xs,
      );
    });
    const [wc, wy] = cumsumJit(xs);
    wc.dispose();
    wy.dispose();

    afterAll(() => {
      cumsumJit.dispose();
      xs.dispose();
    });

    bench("cumsum N=100 size=64", () => {
      const [c, y] = cumsumJit(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
  }

  // cumsum large: N=500, SIZE=256
  {
    const xs = np.ones([500, 256]);
    await blockUntilReady(xs);

    const cumsumJit = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const c = carry.add(x);
          return [c, c];
        },
        np.zeros([256]),
        xs,
      );
    });
    const [wc, wy] = cumsumJit(xs);
    wc.dispose();
    wy.dispose();

    afterAll(() => {
      cumsumJit.dispose();
      xs.dispose();
    });

    bench("cumsum N=500 size=256", () => {
      const [c, y] = cumsumJit(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
  }

  // reverse scan: N=200, SIZE=64
  {
    const xs = np.ones([200, 64]);
    await blockUntilReady(xs);

    const reverseJit = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const c = carry.add(x);
          return [c, c];
        },
        np.zeros([64]),
        xs,
        { reverse: true },
      );
    });
    const [wc, wy] = reverseJit(xs);
    wc.dispose();
    wy.dispose();

    afterAll(() => {
      reverseJit.dispose();
      xs.dispose();
    });

    bench("reverse N=200 size=64", () => {
      const [c, y] = reverseJit(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
  }

  // Kalman filter: N=200 steps, state=[4], cov=[4] (diagonal approx)
  {
    const kalmanJit = jit(() => {
      const processNoise = np.array([0.01, 0.01, 0.01, 0.01]);
      const measNoise = np.array([0.1]);
      const H = np.array([1, 0, 0, 0]);

      type Carry = { state: np.Array; covDiag: np.Array };

      const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
        const { state, covDiag } = carry;
        const predState = state;
        const predCov = covDiag.add(processNoise);
        const innovation = x.sub(np.sum(predState.mul(H)));
        const S = np.sum(predCov.mul(H).mul(H)).add(measNoise);
        const K = predCov.mul(H).div(S);
        const newState = predState.add(K.mul(innovation));
        const newCov = predCov.mul(np.ones([4]).sub(K.mul(H)));
        return [{ state: newState, covDiag: newCov }, predState];
      };

      const initState = np.zeros([4]);
      const initCov = np.ones([4]);
      const obs = np.ones([200, 1]);
      return lax.scan(step, { state: initState, covDiag: initCov }, obs);
    });

    const [wCarry, wY] = kalmanJit();
    (wCarry as any).state.dispose();
    (wCarry as any).covDiag.dispose();
    wY.dispose();

    afterAll(() => {
      kalmanJit.dispose();
    });

    bench("kalman N=200 state=4", () => {
      const [c, y] = kalmanJit() as [any, any];
      c.state.dispose();
      c.covDiag.dispose();
      y.dispose();
    });
  }
});
