import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

const devices = await init();

describe.skipIf(!devices.includes("webgpu"))("gpu argreduce", () => {
  test("10k elements", async ({ bench }) => {
    defaultDevice("webgpu");
    using key = random.key(0);
    using x = random.uniform(key, [10000]);
    await blockUntilReady(x);
    await bench.compare(
      bench("argmax", async () => {
        using r = np.argmax(x);
        await r.blockUntilReady();
      }),
      bench("argmin", async () => {
        using r = np.argmin(x);
        await r.blockUntilReady();
      }),
    );
  });

  test("100x1000 along axis 1", async ({ bench }) => {
    defaultDevice("webgpu");
    using key = random.key(1);
    using x = random.uniform(key, [100, 1000]);
    await blockUntilReady(x);
    await bench.compare(
      bench("argmax", async () => {
        using r = np.argmax(x, 1);
        await r.blockUntilReady();
      }),
      bench("argmin", async () => {
        using r = np.argmin(x, 1);
        await r.blockUntilReady();
      }),
    );
  });
});
