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

suite.skipIf(!devices.includes("webgpu"))("gpu matmul", async () => {
  defaultDevice("webgpu");

  for (const N of [2048, 4096] as const) {
    const a = random.uniform(random.key(0), [N, N]);
    const b = random.uniform(random.key(1), [N, N]);
    await blockUntilReady([a, b]);
    afterAll(() => {
      a.dispose();
      b.dispose();
    });

    bench(`${N} eager`, async () => {
      const c = np.matmul(a, b);
      await c.blockUntilReady();
      c.dispose();
    });

    // np.matmul inside jit() → tiledMatmul with auto-selected tile config
    const fDef = jit((x: np.Array, y: np.Array) => np.matmul(x, y));
    afterAll(() => fDef.dispose());
    bench(`${N} jit(matmul)`, async () => {
      const c = fDef(a, b);
      await c.blockUntilReady();
      c.dispose();
    });

    // Explicit Br=Bc=16 no threadTile (pre-adaptive baseline)
    const f16 = jit((x: np.Array, y: np.Array) =>
      lax.tiledMatmul(x, y, { Br: 16, Bc: 16, Bk: 16 }),
    );
    afterAll(() => f16.dispose());
    bench(`${N} 16×16 (baseline)`, async () => {
      const c = f16(a, b);
      await c.blockUntilReady();
      c.dispose();
    });
  }
});
