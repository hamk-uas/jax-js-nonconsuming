<script lang="ts">
  import { resolve } from "$app/paths";

  import { LoaderCircle } from "@lucide/svelte";
  import { onMount } from "svelte";

  type Cell = { gflops: number | null; running: boolean; err?: string };
  type TableState = {
    configs: string[];
    sizes: string[];
    grid: Cell[][];
  };

  interface CfgDef {
    label: string;
    mode: "eager" | "tiled";
    opts?: { Br: number; Bc: number; Bk: number; threadTile?: number[] };
  }

  const configs: CfgDef[] = [
    { label: "eager (ref)", mode: "eager" },
    { label: "tiled 16×16", mode: "tiled", opts: { Br: 16, Bc: 16, Bk: 16 } },
    {
      label: "tiled 32×32 tt22",
      mode: "tiled",
      opts: { Br: 32, Bc: 32, Bk: 16, threadTile: [2, 2] },
    },
    {
      label: "tiled 32×32 Bk32 tt22",
      mode: "tiled",
      opts: { Br: 32, Bc: 32, Bk: 32, threadTile: [2, 2] },
    },
    {
      label: "tiled 32×32 tt44",
      mode: "tiled",
      opts: { Br: 32, Bc: 32, Bk: 16, threadTile: [4, 4] },
    },
  ];

  const sizeDefs = [
    { n: 256, dtype: "float32" as const, label: "256 f32" },
    { n: 512, dtype: "float32" as const, label: "512 f32" },
    { n: 1024, dtype: "float32" as const, label: "1024 f32" },
    { n: 2048, dtype: "float32" as const, label: "2048 f32" },
  ];

  const fp16Sizes = [
    { n: 1024, dtype: "float16" as const, label: "1024 f16" },
    { n: 2048, dtype: "float16" as const, label: "2048 f16" },
  ];

  let adapterInfo = $state("");
  let hasF16 = $state(false);
  let running = $state(false);
  let done = $state(false);
  let error = $state<string | null>(null);
  let progress = $state("");
  let deviceProps = $state<Record<string, string>>({});

  function makeGrid(nSizes: number): Cell[][] {
    return configs.map(() =>
      Array.from({ length: nSizes }, () => ({ gflops: null, running: false })),
    );
  }

  let table = $state<TableState>({
    configs: configs.map((c) => c.label),
    sizes: sizeDefs.map((s) => s.label),
    grid: makeGrid(sizeDefs.length),
  });

  function colMaxes(grid: Cell[][]): (number | null)[] {
    const cols = grid[0]?.length ?? 0;
    return Array.from({ length: cols }, (_, c) => {
      const vals = grid
        .map((row) => row[c].gflops)
        .filter((v): v is number => v !== null);
      return vals.length > 0 ? Math.max(...vals) : null;
    });
  }

  function fmt(v: number): string {
    return v >= 100 ? v.toFixed(0) : v.toFixed(1);
  }

  let maxes = $derived(colMaxes(table.grid));

  function touchGrid() {
    table = { ...table, grid: table.grid.map((r) => [...r]) };
  }

  /** Race a promise against a timeout (ms). Returns undefined on timeout. */
  function withTimeout<T>(p: Promise<T>, ms: number): Promise<T | undefined> {
    return Promise.race([
      p,
      new Promise<undefined>((resolve) =>
        setTimeout(() => resolve(undefined), ms),
      ),
    ]);
  }

  async function runBenchmarks() {
    if (running) return;
    running = true;
    done = false;
    error = null;
    deviceProps = {};

    try {
      const jax = await import("@hamk-uas/jax-js-nonconsuming");

      const gpu = navigator.gpu;
      if (!gpu) {
        error = "WebGPU not available in this browser";
        return;
      }
      const adapter = await gpu.requestAdapter();
      if (!adapter) {
        error = "No WebGPU adapter found";
        return;
      }

      // --- Collect device diagnostics ---
      const info = adapter.info;
      adapterInfo =
        `${info.vendor} ${info.architecture} ${info.description}`.trim();
      hasF16 = adapter.features.has("shader-f16");

      const lim = adapter.limits;
      deviceProps = {
        vendor: info.vendor || "(unknown)",
        architecture: info.architecture || "(unknown)",
        description: info.description || "(unknown)",
        "shader-f16": hasF16 ? "yes" : "no",
        maxComputeInvocationsPerWorkgroup: String(
          lim.maxComputeInvocationsPerWorkgroup,
        ),
        maxComputeWorkgroupSizeX: String(lim.maxComputeWorkgroupSizeX),
        maxComputeWorkgroupSizeY: String(lim.maxComputeWorkgroupSizeY),
        maxComputeWorkgroupStorageSize: `${lim.maxComputeWorkgroupStorageSize} (${(lim.maxComputeWorkgroupStorageSize / 1024).toFixed(0)} KB)`,
        maxStorageBufferBindingSize: `${lim.maxStorageBufferBindingSize} (${(lim.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB)`,
        maxBufferSize: `${lim.maxBufferSize} (${(lim.maxBufferSize / 1024 / 1024).toFixed(0)} MB)`,
        maxStorageBuffersPerShaderStage: String(
          lim.maxStorageBuffersPerShaderStage,
        ),
      };

      const sizes = [...sizeDefs, ...(hasF16 ? fp16Sizes : [])];
      table = {
        configs: configs.map((c) => c.label),
        sizes: sizes.map((s) => s.label),
        grid: makeGrid(sizes.length),
      };

      // --- Init backend ---
      await jax.init("webgpu");
      jax.defaultDevice("webgpu");
      const { numpy: np, lax, jit, random, blockUntilReady, clearCaches } = jax;
      type Arr = InstanceType<typeof np.Array>;

      let deviceDead = false;

      async function bench(
        n: number,
        dtype: string,
        cfg: CfgDef,
      ): Promise<number> {
        const dt = dtype as any;
        const key = random.key(42);
        const [k1, k2] = random.split(key, 2);
        const A = random.uniform(k1, [n, n]).astype(dt);
        const B = random.uniform(k2, [n, n]).astype(dt);
        key.dispose();
        k1.dispose();
        k2.dispose();
        await blockUntilReady([A, B]);

        try {
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

          const f = jit((a: Arr, b: Arr) =>
            lax.tiledMatmul(a, b, cfg.opts as any),
          );

          try {
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
          } finally {
            f.dispose();
          }
        } finally {
          A.dispose();
          B.dispose();
        }
      }

      const total = sizes.length * configs.length;
      let completed = 0;

      // Timeout per cell: generous for large matrices, tight for small
      function cellTimeout(n: number): number {
        if (n >= 2048) return 30000;
        if (n >= 1024) return 15000;
        return 8000;
      }

      for (let si = 0; si < sizes.length; si++) {
        const { n, dtype } = sizes[si];

        for (let ci = 0; ci < configs.length; ci++) {
          if (deviceDead) {
            table.grid[ci][si] = {
              gflops: null,
              running: false,
              err: "device lost",
            };
            completed++;
            continue;
          }

          table.grid[ci][si] = { gflops: null, running: true };
          progress = `${completed}/${total}`;
          touchGrid();

          try {
            const result = await withTimeout(
              bench(n, dtype, configs[ci]),
              cellTimeout(n),
            );
            if (result === undefined) {
              table.grid[ci][si] = {
                gflops: null,
                running: false,
                err: "timeout (TDR)",
              };
              deviceDead = true;
            } else {
              table.grid[ci][si] = { gflops: result, running: false };
            }
          } catch (e: any) {
            const msg = e?.message ?? String(e);
            const isDeviceLost = /device.*lost|instance.*drop|mapAsync/i.test(
              msg,
            );
            table.grid[ci][si] = {
              gflops: null,
              running: false,
              err: msg.slice(0, 80),
            };
            if (isDeviceLost) {
              deviceDead = true;
            }
          }
          completed++;
          touchGrid();
          clearCaches();
        }
      }
      progress = `${total}/${total}`;
      done = true;
    } catch (e: any) {
      error = e.message ?? String(e);
    } finally {
      running = false;
    }
  }

  onMount(() => {
    runBenchmarks();
  });
</script>

<svelte:head>
  <title>Tiled Matmul Benchmark – jax-js</title>
</svelte:head>

<main class="max-w-screen-xl mx-auto px-6 py-10 font-tiktok">
  <a href={resolve("/")} class="text-sm text-neutral-500 hover:text-primary"
    >← Home</a
  >

  <h1 class="text-3xl font-bold mt-4 mb-2">Tiled Matmul Benchmark</h1>
  <p class="text-gray-500 mb-1 text-sm">
    Runs <code>jit(matmul)</code> live on your GPU across tile configs and
    matrix sizes. All numbers in <strong>GFLOP/s</strong> (higher is better).
  </p>
  <p class="text-gray-400 mb-6 text-xs">
    "tt" = threadTile (register tiling per thread). "Bk" = contraction block
    size. "auto" = <code>chooseTileConfig</code> picks the best candidate for the
    device.
  </p>

  {#if error}
    <div
      class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 text-red-700 text-sm"
    >
      {error}
    </div>
  {/if}

  {#if adapterInfo}
    <div class="text-sm text-gray-700 mb-4 flex items-center gap-2">
      <span class="text-gray-400">GPU:</span>
      {adapterInfo}
      {#if hasF16}
        <span
          class="text-xs text-gray-500 border border-gray-300 rounded px-1.5 py-0.5"
          >fp16</span
        >
      {/if}
      {#if running}
        <LoaderCircle class="w-4 h-4 animate-spin text-primary" />
        <span class="text-xs text-gray-400">{progress}</span>
      {/if}
      {#if done}
        <span class="text-green-600 text-xs">✓ done</span>
      {/if}
    </div>
  {:else if running}
    <div class="flex items-center gap-2 text-gray-400 text-sm mb-4">
      <LoaderCircle class="w-4 h-4 animate-spin" /> Detecting GPU…
    </div>
  {/if}

  <div class="overflow-x-auto -mx-6 px-6">
    <table class="w-full text-sm border-collapse min-w-[700px]">
      <thead>
        <tr class="border-b border-gray-200">
          <th class="text-left py-2 pr-4 text-gray-500 font-medium">Config</th>
          {#each table.sizes as size}
            <th
              class="text-right py-2 px-2 text-gray-500 font-medium whitespace-nowrap"
              >{size}</th
            >
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each table.configs as config, r}
          <tr class="border-b border-gray-100 hover:bg-gray-50">
            <td
              class="py-1.5 pr-4 whitespace-nowrap font-mono text-gray-700 text-xs"
              >{config}</td
            >
            {#each table.grid[r] as cell, c}
              <td class="py-1.5 px-2 text-right font-mono tabular-nums text-xs">
                {#if cell.running}
                  <LoaderCircle
                    class="inline w-3 h-3 animate-spin text-gray-400"
                  />
                {:else if cell.gflops !== null}
                  <span
                    class={cell.gflops === maxes[c]
                      ? "text-primary font-bold"
                      : "text-gray-500"}
                  >
                    {fmt(cell.gflops)}
                  </span>
                {:else if cell.err}
                  <span class="text-red-400/60" title={cell.err}>✗</span>
                {:else}
                  <span class="text-gray-300">—</span>
                {/if}
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <div class="mt-8">
    <button
      class="text-sm px-4 py-2 rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-100 disabled:opacity-40"
      disabled={running}
      onclick={runBenchmarks}
    >
      {done ? "Run again" : "Run benchmark"}
    </button>
  </div>

  <!-- Device diagnostics -->
  {#if Object.keys(deviceProps).length > 0}
    <details class="mt-10" open>
      <summary class="text-sm text-gray-400 cursor-pointer hover:text-gray-700">
        Device properties
      </summary>
      <div class="mt-2 overflow-x-auto">
        <table class="text-xs border-collapse">
          <tbody>
            {#each Object.entries(deviceProps) as [key, val]}
              <tr class="border-b border-gray-100">
                <td class="py-1 pr-4 text-gray-400 font-mono whitespace-nowrap"
                  >{key}</td
                >
                <td class="py-1 text-gray-700 font-mono">{val}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </details>
  {/if}
</main>
