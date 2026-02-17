import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init();

// Benchmark argreduce operations on WebGPU
suite.skipIf(!devices.includes("webgpu"))("gpu argreduce", async () => {
  defaultDevice("webgpu");

  // 1D array benchmarks
  const arr10k = random.uniform(random.key(0), [10000]);
  await blockUntilReady([arr10k]);
  afterAll(() => arr10k.dispose());

  bench("argmax 10k elements", async () => {
    const result = np.argmax(arr10k);
    await result.blockUntilReady();
    result.dispose();
  });

  bench("argmin 10k elements", async () => {
    const result = np.argmin(arr10k);
    await result.blockUntilReady();
    result.dispose();
  });

  // 2D array benchmarks (reduction along axis)
  const arr2d = random.uniform(random.key(1), [100, 1000]);
  await blockUntilReady([arr2d]);
  afterAll(() => arr2d.dispose());

  bench("argmax along axis (100x1000)", async () => {
    const result = np.argmax(arr2d, 1);
    await result.blockUntilReady();
    result.dispose();
  });

  bench("argmin along axis (100x1000)", async () => {
    const result = np.argmin(arr2d, 1);
    await result.blockUntilReady();
    result.dispose();
  });
});
