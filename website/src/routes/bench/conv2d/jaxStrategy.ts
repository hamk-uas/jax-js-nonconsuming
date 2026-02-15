import { runBenchmark } from "$lib/benchmark";

export function createJaxJsConv2dStrategy(
  batchSize: number,
  channels: number,
  height: number,
  width: number,
  outChannels: number,
  filterHeight: number,
  filterWidth: number,
  randomInput: Float32Array,
  randomFilter: Float32Array,
  printBufferItems: (buf: Float32Array) => void,
  fp16: boolean = false,
) {
  return {
    name: fp16 ? "jax-js-fp16" : "jax-js",
    async run(): Promise<number> {
      const jax = await import("@hamk-uas/jax-js-nonconsuming");
      await jax.init();
      jax.defaultDevice("webgpu");
      const np = jax.numpy;

      using x = np
        .array(randomInput as Float32Array<ArrayBuffer>, {
          shape: [batchSize, channels, height, width],
        })
        .astype(fp16 ? np.float16 : np.float32);
      using filter = np
        .array(randomFilter as Float32Array<ArrayBuffer>, {
          shape: [outChannels, channels, filterHeight, filterWidth],
        })
        .astype(fp16 ? np.float16 : np.float32);
      await jax.blockUntilReady([x, filter]);

      return await runBenchmark("jax", async () => {
        using output = jax.lax.convGeneralDilated(x, filter, [1, 1], "SAME");
        const ar = (await output.data()) as Float32Array;
        printBufferItems(ar);
      });
    },
  };
}
