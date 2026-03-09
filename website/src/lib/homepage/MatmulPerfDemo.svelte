<script lang="ts">
  import { LoaderCircle, SquareMousePointerIcon } from "@lucide/svelte";
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";

  import { fallbackResults, measurePerf, type PerfResults } from "./benchFlops";

  let results = $state<PerfResults | null>(null);

  let measuring = $state(false);

  async function measurementTask() {
    if (measuring) return;
    measuring = true;
    try {
      results = null;
      results = await measurePerf();
    } finally {
      measuring = false;
    }
  }

  onMount(() => {
    measurementTask();
  });

  // Bar chart configuration
  const barWidth = 80;
  const barGap = 16;
  const paddingX = 16;
  const paddingTop = 24;
  const paddingBottom = 28;

  const allBackends = ["Wasm", "WebGPU", "WebGPU-fp16"] as const;
  const barColors: Record<(typeof allBackends)[number], string> = {
    Wasm: "#6366f1",
    WebGPU: "#8b5cf6",
    "WebGPU-fp16": "#a855f7",
  };
  const barLabels: Record<(typeof allBackends)[number], string> = {
    Wasm: "Wasm",
    WebGPU: "WebGPU",
    "WebGPU-fp16": "fp16",
  };

  // Always render all backends for smooth transitions
  const chartWidth =
    paddingX * 2 +
    allBackends.length * barWidth +
    (allBackends.length - 1) * barGap;
  const chartHeight = 220;

  // Calculate max value for scaling
  const maxFlops = $derived(
    results
      ? Math.max(
          ...allBackends.map((b) => results!.flops[b]).filter((v) => v != null),
        )
      : fallbackResults.flops["WebGPU-fp16"]!,
  );

  // Get bar height as a percentage (0-1)
  function getBarHeight(backend: (typeof allBackends)[number]): number {
    if (!results) return 0;
    const value = results.flops[backend];
    if (value == null) return 0;
    return value / maxFlops;
  }

  // Check if a backend is available
  function isAvailable(backend: (typeof allBackends)[number]): boolean {
    if (!results) return true; // Show all as placeholders before results
    return results.flops[backend] != null;
  }

  // Format number with commas
  function formatNumber(num: number): string {
    const fractionDigits = num < 10 ? 2 : num < 100 ? 1 : 0;
    return num.toLocaleString(undefined, {
      minimumFractionDigits: fractionDigits,
      maximumFractionDigits: fractionDigits,
    });
  }
</script>

<section class="flex flex-col justify-center items-center text-center">
  <h3 class="text-lg mb-1">Matrix multiplication</h3>
  <p class="text-gray-700 text-sm mb-6 max-w-[30ch]">
    Billions of floating-point operations (GFLOPs) per second
  </p>

  <!-- Bar chart -->
  <svg
    viewBox="0 0 {chartWidth} {chartHeight}"
    class="overflow-visible max-w-full"
    style="width: {chartWidth}px; height: auto;"
  >
    <!-- X-axis (subtle) -->
    <line
      x1={0}
      y1={chartHeight - paddingBottom}
      x2={chartWidth}
      y2={chartHeight - paddingBottom}
      stroke="#e2e8f0"
      stroke-width="1"
    />

    {#each allBackends as backend, i}
      {@const available = isAvailable(backend)}
      {@const xPos = paddingX + (barWidth + barGap) * i}
      {@const heightPercent = getBarHeight(backend)}
      {@const availableHeight = chartHeight - paddingBottom - paddingTop}
      {@const minBarHeight = 4}
      {@const barHeight = Math.max(
        heightPercent * availableHeight,
        results ? minBarHeight : 0,
      )}
      {@const yPos = chartHeight - paddingBottom - barHeight}
      {@const value = results?.flops[backend] ?? 0}

      <g
        style="opacity: {available ? 1 : 0}; transition: opacity 0.3s ease-in;"
      >
        <!-- Bar -->
        <rect
          x={xPos}
          y={yPos}
          width={barWidth}
          height={barHeight}
          fill={barColors[backend]}
          rx="4"
          style="transition: height 0.5s ease-out, y 0.5s ease-out;"
        />

        <!-- Value label on top of bar -->
        {#if results}
          <text
            x={xPos + barWidth / 2}
            y={yPos - 8}
            text-anchor="middle"
            class="text-sm font-semibold"
            fill="#1e293b"
            in:fade={{ delay: 200, duration: 300 }}
          >
            {formatNumber(value)}
          </text>
        {/if}

        <!-- Backend label below bar -->
        <text
          x={xPos + barWidth / 2}
          y={chartHeight - paddingBottom + 20}
          text-anchor="middle"
          class="text-xs"
          fill="#64748b"
        >
          {barLabels[backend]}
        </text>
      </g>
    {/each}
  </svg>

  <div
    class="flex items-center gap-2 mt-4 text-sm"
    class:animate-pulse={!results}
  >
    {#if !results}
      <LoaderCircle size={16} class="animate-spin text-gray-400" />
      <p class="text-gray-500">Running benchmark…</p>
    {:else}
      <button
        class="flex items-center gap-2"
        onclick={() => {
          if (results?.live) {
            measurementTask();
          }
        }}
        disabled={!results.live || measuring}
      >
        {#if results.live}
          <SquareMousePointerIcon size={16} class="text-gray-500" />
        {/if}
        <p class="text-gray-800">
          {results.browser}
        </p>
      </button>
    {/if}
  </div>
</section>
