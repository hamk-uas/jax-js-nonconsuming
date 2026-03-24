<script lang="ts">
  import {
    ChevronDownIcon,
    ChevronRightIcon,
    CodeIcon,
    ImageIcon,
    InfoIcon,
    TriangleAlertIcon,
    XIcon,
  } from "@lucide/svelte";

  import type { ConsoleLine } from "./runner.svelte";

  let { line, showTime = false }: { line: ConsoleLine; showTime?: boolean } =
    $props();

  let expanded = $state(true);
  let text = $derived(line.data.join(" "));
  let multiline = $derived(text.includes("\n"));
</script>

{#if line.level === "report" && line.reportText}
  <details class="border-t border-indigo-200 py-0.5 text-[13px]">
    <summary
      class="px-1 cursor-pointer hover:bg-indigo-50 select-none flex items-center gap-x-2"
    >
      <CodeIcon size={14} class="shrink-0 text-indigo-500" />
      <span
        class="inline-block px-1.5 py-0.5 rounded text-[11px] font-medium bg-indigo-100 text-indigo-800"
      >
        {line.data[0] ?? "Compiled Code"}
      </span>
      {#if showTime}
        <span
          class="hidden md:block ml-auto shrink-0 font-mono text-gray-400 select-none"
        >
          {new Date(line.time).toLocaleTimeString()}
        </span>
      {/if}
    </summary>
    <pre
      class="px-2 py-1 overflow-x-auto bg-gray-50 border-t border-gray-200 text-[12px] leading-tight max-h-[32rem] overflow-y-auto"
      style:scrollbar-width="thin"><code>{line.reportText}</code></pre>
  </details>
{:else}
  <div
    class={[
      "py-0.5 border-t flex items-start gap-x-2",
      line.level === "error"
        ? "border-red-200 bg-red-50"
        : line.level === "warn"
          ? "border-yellow-200 bg-yellow-50"
          : "border-gray-200",
    ]}
  >
    {#if line.level === "log"}
      {#if multiline}
        <button
          class="shrink-0 text-gray-400 hover:text-gray-600 cursor-pointer"
          onclick={() => (expanded = !expanded)}
          aria-label={expanded ? "Collapse" : "Expand"}
        >
          {#if expanded}
            <ChevronDownIcon size={18} />
          {:else}
            <ChevronRightIcon size={18} />
          {/if}
        </button>
      {:else}
        <ChevronRightIcon size={18} class="shrink-0 text-gray-300" />
      {/if}
    {:else if line.level === "info"}
      <InfoIcon size={18} class="shrink-0 text-blue-500" />
    {:else if line.level === "warn"}
      <TriangleAlertIcon size={18} class="shrink-0 text-yellow-500" />
    {:else if line.level === "error"}
      <XIcon size={18} class="shrink-0 text-red-500" />
    {:else if line.level === "image"}
      <ImageIcon size={18} class="shrink-0 text-gray-400" />
    {/if}
    <p class="font-mono whitespace-pre-wrap min-w-0 overflow-x-auto">
      {#if line.level === "image"}
        <img
          src={line.data[0]}
          alt="Output from displayImage()"
          class="max-w-full my-0.5"
        />
      {:else if !expanded}
        {text.split("\n")[0]}
        <span class="text-gray-400">({text.split("\n").length} lines)</span>
      {:else}
        {text}
      {/if}
    </p>
    {#if showTime}
      <p
        class="hidden md:block ml-auto shrink-0 font-mono text-gray-400 select-none"
      >
        {new Date(line.time).toLocaleTimeString()}
      </p>
    {/if}
  </div>
{/if}
