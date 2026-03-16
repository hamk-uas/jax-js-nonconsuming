<script lang="ts">
  import type { CodeCaptureEntry } from "@hamk-uas/jax-js-nonconsuming";

  let { entries }: { entries: CodeCaptureEntry[] } = $props();

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
</script>

{#if entries.length === 0}
  <p class="text-gray-400 text-sm px-4 py-2">
    No compiled code captured. Run a program with "Capture compiled code"
    enabled.
  </p>
{:else}
  {#each entries as entry, i (i)}
    <details class="mx-2 mb-1 border border-gray-200 rounded text-[13px]">
      <summary class="px-2 py-1 cursor-pointer hover:bg-gray-50 select-none">
        <span
          class="inline-block px-1.5 py-0.5 rounded text-[11px] font-medium {kindColor(
            entry.kind,
          )}"
        >
          {entry.kind}
        </span>
        <span class="text-gray-500 ml-1">{entry.backend}</span>
        {#if entry.label}
          <span class="text-gray-700 ml-1">{entry.label}</span>
        {/if}
        {#if entry.metadata}
          <span class="text-gray-400 ml-2">{metaSummary(entry)}</span>
        {/if}
      </summary>
      {#if entry.code}
        <pre
          class="px-2 py-1 overflow-x-auto bg-gray-50 border-t border-gray-200 text-[12px] leading-tight max-h-80 overflow-y-auto"
          style:scrollbar-width="thin"><code>{entry.code}</code></pre>
      {/if}
    </details>
  {/each}
{/if}
