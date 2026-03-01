import { runBenchmark } from "$lib/benchmark";

export function createJaxJsMatmulStrategy(
  n: number,
  randomBuffer: Float32Array,
  printBufferItems: (buf: Float32Array | Float16Array) => void,
  fp16: boolean = false,
) {
  return {
    name: fp16 ? "jax-js-nonconsuming-fp16" : "jax-js-nonconsuming",
    async run(): Promise<number> {
      const jax = await import("@hamk-uas/jax-js-nonconsuming");
      await jax.init();
      jax.defaultDevice("webgpu");
      const np = jax.numpy;

      using a = np
        .array(randomBuffer as Float32Array<ArrayBuffer>, { shape: [n, n] })
        .astype(fp16 ? np.float16 : np.float32);
      using b = np
        .array(randomBuffer as Float32Array<ArrayBuffer>, { shape: [n, n] })
        .astype(fp16 ? np.float16 : np.float32);
      await jax.blockUntilReady([a, b]);

      return await runBenchmark("jax", async () => {
        using c = np.dot(a, b);
        const ar = (await c.data()) as Float16Array;
        printBufferItems(ar);
      });
    },
  };
}

export function createJaxJsTiledMatmulStrategy(
  n: number,
  randomBuffer: Float32Array,
  printBufferItems: (buf: Float32Array | Float16Array) => void,
  fp16: boolean = false,
) {
  return {
    name: fp16 ? "jax-js-tiled-fp16" : "jax-js-tiled",
    async run(): Promise<number> {
      const jax = await import("@hamk-uas/jax-js-nonconsuming");
      await jax.init();
      jax.defaultDevice("webgpu");
      const { numpy: np, lax, jit, blockUntilReady } = jax;

      const dtype = fp16 ? np.float16 : np.float32;
      using a = np
        .array(randomBuffer as Float32Array<ArrayBuffer>, { shape: [n, n] })
        .astype(dtype);
      using b = np
        .array(randomBuffer as Float32Array<ArrayBuffer>, { shape: [n, n] })
        .astype(dtype);
      await blockUntilReady([a, b]);

      using tiledFn = jit((x: any, y: any) => lax.tiledMatmul(x, y));

      return await runBenchmark("jax", async () => {
        using c = tiledFn(a, b);
        const ar = (await c.data()) as Float16Array;
        printBufferItems(ar);
      });
    },
  };
}
