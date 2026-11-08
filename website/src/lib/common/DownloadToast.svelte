<script lang="ts">
  import { CircleCheck, CircleX, LoaderCircle } from "@lucide/svelte";
  import { fly } from "svelte/transition";

  import { formatBytes } from "#lib/chart/format.ts";

  export interface Props {
    name: string;
    status: "downloading" | "success" | "error";
    loaded?: number;
    total?: number;
    errorMessage?: string;
  }

  let { name, status, loaded, total, errorMessage }: Props = $props();
</script>

<div
  class="fixed bottom-2 right-2 sm:bottom-4 sm:right-4 bg-white rounded-lg shadow-md sm:shadow-lg p-4 min-w-80 border border-gray-200 transition-all duration-300 z-20"
  transition:fly={{ y: 32, duration: 400, delay: 200 }}
>
  <div class="flex items-start gap-3">
    <div class="flex-shrink-0 mt-0.5">
      {#if status === "downloading"}
        <LoaderCircle
          size={16}
          class="stroke-[2.5] animate-spin text-blue-600"
        />
      {:else if status === "success"}
        <CircleCheck size={16} class="stroke-[2.5] text-green-600" />
      {:else}
        <CircleX size={16} class="stroke-[2.5] text-red-600" />
      {/if}
    </div>
    <div class="flex-1">
      <div class="font-medium text-sm mb-1">
        {#if status === "downloading"}
          <p class="text-gray-700">Downloading {name}…</p>
        {:else if status === "success"}
          <p class="text-green-700">Loaded {name}!</p>
        {:else if status === "error"}
          <p class="text-red-700">Failed to download</p>
        {/if}
      </div>

      {#if errorMessage}
        <div class="text-xs text-red-500">{errorMessage}</div>
      {/if}

      {#if status === "downloading" && loaded !== undefined && total !== undefined}
        {@const percentage = Math.min(Math.floor((loaded / total) * 100), 99)}
        <div class="flex items-center gap-2 mb-1.5">
          <div
            class="flex-1 bg-gradient-to-r from-gray-100 to-gray-200 rounded-full h-1.5 overflow-hidden"
          >
            <div
              class="bg-gradient-to-r from-blue-500 to-blue-600 h-1.5 transition-all duration-200 ease-out rounded-full"
              style:width="{percentage}%"
            ></div>
          </div>
          <span class="tabular-nums text-xs text-gray-600 whitespace-nowrap">
            {percentage}%
          </span>
        </div>
      {/if}

      {#if status === "downloading" || status === "success"}
        <div class="text-xs text-gray-500 tabular-nums">
          {#if loaded !== undefined}
            {formatBytes(loaded)}
            {#if total !== undefined}
              / {formatBytes(total)}
            {/if}
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>
