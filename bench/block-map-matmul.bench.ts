/**
 * Tiled matmul benchmarks — Phase 4 performance gates.
 *
 * Compares lax.tiledMatmul (block_map) against np.matmul (naive) at
 * 256×256, 512×512, 1024×1024, and 2048×2048.
 */

import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))("tiled matmul", async () => {
  defaultDevice("webgpu");

  for (const N of [256, 512, 1024, 2048]) {
    const a = random.uniform(random.key(0), [N, N]);
    const b = random.uniform(random.key(1), [N, N]);
    await blockUntilReady([a, b]);

    const tiledFn = jit((x: np.Array, y: np.Array) => lax.tiledMatmul(x, y));
    // Warmup JIT compilation
    const warmup = tiledFn(a, b);
    await warmup.blockUntilReady();
    warmup.dispose();

    afterAll(() => {
      a.dispose();
      b.dispose();
      tiledFn.dispose();
    });

    bench(`np.matmul ${N}x${N}`, async () => {
      const c = np.matmul(a, b);
      await c.blockUntilReady();
      c.dispose();
    });

    bench(`tiledMatmul ${N}x${N}`, async () => {
      const c = tiledFn(a, b);
      await c.blockUntilReady();
      c.dispose();
    });
  }
});
