import {
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

const devices = await init("wasm", "webgpu");

/**
 * A scan workload: `make` builds its inputs and jit function (owned by
 * `stack`) and returns one benchmark iteration.
 */
type ScanCase = {
  name: string;
  /** Also run on WebGPU (all cases run on WASM). */
  webgpu: boolean;
  make: (stack: DisposableStack) => () => void;
};

function cumsum(n: number, size: number, reverse = false) {
  return (stack: DisposableStack) => {
    const xs = stack.use(np.ones([n, size]));
    const f = stack.use(
      jit((v: np.Array) =>
        lax.scan(
          (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
            const c = carry.add(x);
            return [c, c];
          },
          np.zeros([size]),
          v,
          { reverse },
        ),
      ),
    );
    return () => {
      const [c, y] = f(xs);
      c.dispose();
      y.dispose();
    };
  };
}

const CASES: ScanCase[] = [
  { name: "cumsum N=100 size=64", webgpu: true, make: cumsum(100, 64) },
  { name: "cumsum N=500 size=256", webgpu: true, make: cumsum(500, 256) },
  {
    name: "carry-only N=200 size=32",
    webgpu: false,
    make: (stack) => {
      const f = stack.use(
        jit(() => {
          const [carry] = lax.scan(
            (c: np.Array, _x: null): [np.Array, null] => [
              c.add(np.ones([32])),
              null,
            ],
            np.zeros([32]),
            null,
            { length: 200 },
          );
          return carry;
        }),
      );
      return () => f().dispose();
    },
  },
  {
    name: "reduction N=100 size=64",
    webgpu: false,
    make: (stack) => {
      const xs = stack.use(np.ones([100, 64]));
      const f = stack.use(
        jit((v: np.Array) =>
          lax.scan(
            (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
              const s = carry.add(np.sum(x));
              return [s, s];
            },
            np.zeros([]),
            v,
          ),
        ),
      );
      return () => {
        const [c, y] = f(xs);
        c.dispose();
        y.dispose();
      };
    },
  },
  { name: "reverse N=200 size=64", webgpu: true, make: cumsum(200, 64, true) },
  {
    // Kalman filter: N=200 steps, state=[4], diagonal covariance
    // (elementwise ops only), observing the first state element.
    name: "kalman N=200 state=4",
    webgpu: true,
    make: (stack) => {
      type Carry = { state: np.Array; covDiag: np.Array };
      const f = stack.use(
        jit(() => {
          const processNoise = np.array([0.01, 0.01, 0.01, 0.01]);
          const measNoise = np.array([0.1]);
          const H = np.array([1, 0, 0, 0]);
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
          const init: Carry = { state: np.zeros([4]), covDiag: np.ones([4]) };
          const obs = np.ones([200, 1]); // 200 scalar observations
          return lax.scan(step, init, obs);
        }),
      );
      return () => {
        const [c, y] = f();
        c.state.dispose();
        c.covDiag.dispose();
        y.dispose();
      };
    },
  },
];

for (const device of ["wasm", "webgpu"] as const) {
  describe.skipIf(!devices.includes(device))(`${device} scan`, () => {
    test("scan bodies", async ({ bench }) => {
      defaultDevice(device);
      using stack = new DisposableStack();
      const registrations = CASES.filter(
        (c) => device === "wasm" || c.webgpu,
      ).map((c) => bench(c.name, c.make(stack)));
      await bench.compare(...registrations);
    });
  });
}
