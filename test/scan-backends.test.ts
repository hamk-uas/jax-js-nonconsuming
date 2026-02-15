import { beforeAll, describe, expect, it } from "vitest";

import {
  Device,
  devicePut,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "../src/index";

describe("lax.scan backend coverage", () => {
  let devices: Device[] = [];

  beforeAll(async () => {
    devices = await init();
    console.log("Available devices:", devices);
  });

  // We can't iterate dynamically in describe() easily with vitest if we want separate checks
  // but we can just run the function.
  // Actually, we can loop in the test body or use describe.each if we knew them ahead of time.
  // Since we await init(), we can't use describe.each top-level (init is async).
  // Simpler: Just one test that iterates.

  it("executes tests on all devices", async () => {
    for (const device of devices) {
      console.log(`Running sub-tests for ${device}`);

      // Manual invocation logic to reuse the code above?
      // I will inline simple logic here.

      // 1. Basic
      {
        const step = (c: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newC = np.add(c, x);
          return [newC, newC];
        };
        const initVal = await devicePut(np.zeros([1]), device);
        const xs = await devicePut(np.ones([10, 1]), device);
        const [final, _] = lax.scan(step, initVal, xs);
        const finalData = await final.data();
        expect(finalData[0]).toBe(10);
      }

      // 2. Check copyBufferToBuffer (required for fallback)
      {
        const { getBackend } = await import("@jax-js-nonconsuming/jax");
        const backend = getBackend(device) as any;

        if (typeof backend.copyBufferToBuffer === "function") {
          const slot1 = backend.malloc(16);
          const slot2 = backend.malloc(16);
          try {
            backend.copyBufferToBuffer(slot1, 0, slot2, 0, 16);
          } finally {
            backend.decRef(slot1);
            backend.decRef(slot2);
          }
        } else {
          if (device !== "webgl") {
            throw new Error(`Backend ${device} missing copyBufferToBuffer`);
          }
        }
      }

      // 3. JIT scan
      {
        const step = (c: np.Array, x: np.Array): [np.Array, np.Array] => {
          return [np.add(c, x), np.add(c, x)];
        };
        const run = jit((init, xs) => {
          return lax.scan(step, init, xs);
        });

        const initVal = await devicePut(np.zeros([1]), device);
        const xs = await devicePut(np.ones([5, 1]), device);
        // eslint-disable-next-line @typescript-eslint/await-thenable
        const [final, _] = await run(initVal, xs);

        expect((await final.data())[0]).toBe(5);
        run.dispose();
      }

      // 4. Grad scan
      {
        const loss = (xs: np.Array) => {
          const step = (c: np.Array, x: np.Array): [np.Array, np.Array] => {
            return [np.add(c, x), c];
          };
          const initVal = np.zeros([1]);
          const [final, _] = lax.scan(step, initVal, xs);
          return np.sum(final);
        };

        const xs = await devicePut(np.ones([5, 1]), device);
        const calcGrad = grad(loss);
        // eslint-disable-next-line @typescript-eslint/await-thenable
        const dxs = await calcGrad(xs);

        const dxsData = await dxs.data();
        expect(dxsData[0]).toBe(1);
      }
    }
  });
});
