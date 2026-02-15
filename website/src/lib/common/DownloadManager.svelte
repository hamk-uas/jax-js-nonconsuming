<!-- @component Updates a `DownlodToast` with fetch progress. -->

<script lang="ts">
  import { cachedFetch } from "@jax-js-nonconsuming/loaders";

  import DownloadToast, {
    type Props as DownloadToastProps,
  } from "$lib/common/DownloadToast.svelte";

  let downloadState = $state<DownloadToastProps | null>(null);

  export async function fetch(name: string, url: string) {
    if (downloadState && downloadState.status === "downloading") {
      // Another download is in progress, wait for it to finish first.
      while (downloadState && downloadState.status === "downloading") {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }
    try {
      downloadState = {
        name,
        status: "downloading",
      };

      const data = await cachedFetch(url, {}, (progress) => {
        downloadState = {
          name,
          status: "downloading",
          loaded: progress.loadedBytes,
          total: progress.totalBytes,
        };
      });

      downloadState = {
        name,
        status: "success",
        loaded: downloadState.loaded,
      };
      setTimeout(() => {
        downloadState = null;
      }, 3000);

      return data;
    } catch (error) {
      downloadState = {
        name,
        status: "error",
        errorMessage:
          error instanceof Error ? error.message : "Download aborted",
      };
      setTimeout(() => {
        downloadState = null;
      }, 4000);
      throw error;
    }
  }
</script>

{#if downloadState !== null}
  <DownloadToast {...downloadState} />
{/if}
