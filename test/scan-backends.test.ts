import {
  defaultDevice,
  Device,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeAll, describe, expect, it } from "vitest";

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
    const originalDefault = defaultDevice();

    for (const device of devices) {
      // Set default device so internal operations (e.g., zeros in makeJaxpr)
      // create arrays on the same device as the scan inputs.
      defaultDevice(device);

      // 1. Basic
      {
        const step = (c: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newC = np.add(c, x);
          return [newC, newC];
        };
        using initVal = np.zeros([1]);
        using xs = np.ones([10, 1]);
        const [final, ys] = lax.scan(step, initVal, xs);
        const finalData = await final.data();
        expect(finalData[0]).toBe(10);
        final.dispose();
        ys.dispose();
      }

      // 2. Check copyBufferToBuffer (required for fallback)
      {
        const { getBackend } = await import("@hamk-uas/jax-js-nonconsuming");
        const backend = getBackend(device) as any;

        const slot1 = backend.malloc(16);
        const slot2 = backend.malloc(16);
        try {
          backend.copyBufferToBuffer(slot1, 0, slot2, 0, 16);
        } finally {
          backend.decRef(slot1);
          backend.decRef(slot2);
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

        using initVal = np.zeros([1]);
        using xs = np.ones([5, 1]);
        // eslint-disable-next-line @typescript-eslint/await-thenable
        const [final, ys] = await run(initVal, xs);

        expect((await final.data())[0]).toBe(5);
        final.dispose();
        ys.dispose();
        run.dispose();
      }

      // 4. Grad scan
      {
        using initVal = np.zeros([1]);
        const loss = (xs: np.Array) => {
          const step = (c: np.Array, x: np.Array): [np.Array, np.Array] => {
            return [np.add(c, x), c];
          };
          const [final, _] = lax.scan(step, initVal, xs);
          using __ = _;
          return final.sum();
        };

        using xs = np.ones([5, 1]);
        const dxs = grad(loss)(xs);

        const dxsData = await dxs.data();
        expect(dxsData[0]).toBe(1);
        dxs.dispose();
      }
    }

    // Restore original default device
    defaultDevice(originalDefault);
  });
});
