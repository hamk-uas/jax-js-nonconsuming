<script lang="ts">
  import type { CodeCaptureEntry } from "@hamk-uas/jax-js-nonconsuming";
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

  function kindColor(kind: string): string {
    switch (kind) {
      case "kernel":
        return "bg-blue-100 text-blue-800";
      case "mega-module":
        return "bg-purple-100 text-purple-800";
      case "scan":
        return "bg-green-100 text-green-800";
      case "assoc-scan":
        return "bg-teal-100 text-teal-800";
      case "block-map":
        return "bg-orange-100 text-orange-800";
      case "routine":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  }

  function metaSummary(entry: CodeCaptureEntry): string {
    const parts: string[] = [];
    const m = entry.metadata;
    if (!m) return "";
    if (m.numInputs != null) parts.push(`in=${m.numInputs}`);
    if (m.numOutputs != null) parts.push(`out=${m.numOutputs}`);
    if (m.simd) parts.push("SIMD");
    if (m.reduction) parts.push("reduction");
    if (m.numSteps != null) parts.push(`steps=${m.numSteps}`);
    if (m.numKernels != null) parts.push(`kernels=${m.numKernels}`);
    if (m.byteLength != null) parts.push(`${m.byteLength} bytes`);
    return parts.join("  ");
  }
</script>

{#if line.level === "code" && line.codeEntry}
  {@const entry = line.codeEntry}
  <details class="border-t border-gray-200 py-0.5 text-[13px]">
    <summary
      class="px-1 cursor-pointer hover:bg-gray-50 select-none flex items-center gap-x-2"
    >
      <span
        class="inline-block px-1.5 py-0.5 rounded text-[11px] font-medium {kindColor(
          entry.kind,
        )}"
      >
        {entry.kind}
      </span>
      <span class="text-gray-500">{entry.backend}</span>
      {#if entry.label}
        <span class="text-gray-700">{entry.label}</span>
      {/if}
      {#if entry.metadata}
        <span class="text-gray-400">{metaSummary(entry)}</span>
      {/if}
      {#if showTime}
        <span
          class="hidden md:block ml-auto shrink-0 font-mono text-gray-400 select-none"
        >
          {new Date(line.time).toLocaleTimeString()}
        </span>
      {/if}
    </summary>
    {#if entry.code}
      <pre
        class="px-2 py-1 overflow-x-auto bg-gray-50 border-t border-gray-200 text-[12px] leading-tight max-h-80 overflow-y-auto"
        style:scrollbar-width="thin"><code>{entry.code}</code></pre>
    {/if}
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
