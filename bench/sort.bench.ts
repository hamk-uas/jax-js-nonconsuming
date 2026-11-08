import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

const devices = await init("wasm", "webgpu");

// GPU supports much more parallelism, so it sorts more rows.
const cases = [
  { device: "webgpu", batch: 32768 },
  { device: "wasm", batch: 128 },
] as const;

for (const { device, batch } of cases) {
  describe.skipIf(!devices.includes(device))(`${device} sort/argsort`, () => {
    test(`${batch}x1024`, async ({ bench }) => {
      defaultDevice(device);
      using key = random.key(0);
      using a = random.uniform(key, [batch, 1024]);
      await blockUntilReady(a);

      await bench.compare(
        bench("sort", async () => {
          using c = np.sort(a);
          await c.blockUntilReady();
        }),
        bench("argsort", async () => {
          using c = np.argsort(a);
          await c.blockUntilReady();
        }),
      );
    });
  });
}
