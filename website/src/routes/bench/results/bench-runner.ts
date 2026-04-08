import type * as jaxMod from "@hamk-uas/jax-js-nonconsuming";

export interface CfgDef {
  label: string;
  mode: "eager" | "tiled";
  opts?: { Br: number; Bc: number; Bk: number; threadTile?: number[] };
}

type Arr = InstanceType<typeof jaxMod.numpy.Array>;

export async function bench(
  jax: typeof jaxMod,
  n: number,
  dtype: string,
  cfg: CfgDef,
): Promise<number> {
  const { numpy: np, lax, jit, random, blockUntilReady } = jax;
  const dt = dtype as any;

  using key = random.key(42);
  const [k1Raw, k2Raw] = random.split(key, 2);
  using k1 = k1Raw;
  using k2 = k2Raw;
  using A = random.uniform(k1, [n, n]).astype(dt);
  using B = random.uniform(k2, [n, n]).astype(dt);
  await blockUntilReady([A, B]);

  if (cfg.mode === "eager") {
    const warmup = np.matmul(A, B);
    await blockUntilReady(warmup);
    warmup.dispose();
    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      const start = performance.now();
      const C = np.matmul(A, B);
      await blockUntilReady(C);
      times.push(performance.now() - start);
      C.dispose();
    }
    const avg = times.reduce((a, b) => a + b) / times.length;
    return (2 * n ** 3) / 1e9 / (avg / 1000);
  }

  using f = jit((a: Arr, b: Arr) => lax.tiledMatmul(a, b, cfg.opts as any));
  const warmup = f(A, B);
  await blockUntilReady(warmup);
  warmup.dispose();
  const times: number[] = [];
  for (let i = 0; i < 3; i++) {
    const start = performance.now();
    const C = f(A, B);
    await blockUntilReady(C);
    times.push(performance.now() - start);
    C.dispose();
  }
  const avg = times.reduce((a, b) => a + b) / times.length;
  return (2 * n ** 3) / 1e9 / (avg / 1000);
}
