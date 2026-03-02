import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))("gpu matmul", async () => {
  defaultDevice("webgpu");

  const a2048 = random.uniform(random.key(0), [2048, 2048]);
  const b2048 = random.uniform(random.key(1), [2048, 2048]);
  await blockUntilReady([a2048, b2048]);
  afterAll(() => {
    a2048.dispose();
    b2048.dispose();
  });

  bench("2048x2048 eager", async () => {
    const c = np.matmul(a2048, b2048);
    await c.blockUntilReady();
    c.dispose();
  });

  // np.matmul inside jit() routes through tiledMatmul on WebGPU
  const matmul_jit_2048 = jit((a: np.Array, b: np.Array) => np.matmul(a, b));
  afterAll(() => matmul_jit_2048.dispose());

  bench("2048x2048 jit(tiledMatmul)", async () => {
    const c = matmul_jit_2048(a2048, b2048);
    await c.blockUntilReady();
    c.dispose();
  });

  const a4096 = random.uniform(random.key(0), [4096, 4096]);
  const b4096 = random.uniform(random.key(1), [4096, 4096]);
  await blockUntilReady([a4096, b4096]);
  afterAll(() => {
    a4096.dispose();
    b4096.dispose();
  });

  bench("4096x4096 eager", async () => {
    const c = np.matmul(a4096, b4096);
    await c.blockUntilReady();
    c.dispose();
  });

  const matmul_jit_4096 = jit((a: np.Array, b: np.Array) => np.matmul(a, b));
  afterAll(() => matmul_jit_4096.dispose());

  bench("4096x4096 jit(tiledMatmul)", async () => {
    const c = matmul_jit_4096(a4096, b4096);
    await c.blockUntilReady();
    c.dispose();
  });
});
