<script lang="ts">
  import {
    ChevronDownIcon,
    ChevronRightIcon,
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
