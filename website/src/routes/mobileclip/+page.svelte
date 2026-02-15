<script lang="ts">
  import {
    defaultDevice,
    init,
    numpy as np,
  } from "@jax-js-nonconsuming/jax";
  import { opfs, safetensors, tokenizers } from "@jax-js-nonconsuming/loaders";
  import { BookMarkedIcon, FileTextIcon } from "@lucide/svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { type Book, downloadBook } from "./books";
  import { fromSafetensors, type MobileCLIP } from "./clipInference";
  import {
    computeEmbeddings,
    getTextParamCount,
    searchEmbeddings,
    tokenizeExcerpts,
  } from "./embedding";

  // Cached large objects to download.
  let _weights: safetensors.File | null = null;
  let _model: MobileCLIP | null = null;
  let _tokenizer: tokenizers.BpeEncoding | null = null;

  let downloadManager: DownloadManager;

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (_weights) return _weights;
    isDownloadingWeights = true;
    try {
      const weightsUrl =
        "https://huggingface.co/ekzhang/jax-js-models/resolve/main/mobileclip2-s0.safetensors";

      const data = await downloadManager.fetch("model weights", weightsUrl);
      const result = safetensors.parse(data);
      _weights = result;
      return result;
    } catch (error) {
      alert("Error downloading weights: " + error);
      throw error;
    } finally {
      isDownloadingWeights = false;
    }
  }

  async function getModel(): Promise<MobileCLIP> {
    if (_model) return _model;
    const weights = await downloadClipWeights();
    _model = fromSafetensors(weights);
    hasModel = true;
    return _model;
  }

  async function getTokenizer() {
    if (!_tokenizer) _tokenizer = await tokenizers.getBpe("clip");
    return _tokenizer;
  }

  let hasModel = $state(false);
  let isDownloadingWeights = $state(false);

  let hasData = $state(false);
  let isDownloadingData = $state(false);

  let book = $state<Book>(null as any);
  let embeddingProgress = $state<number[]>([]);
  let embeddingTotal = $state<number>(0);
  let embeddingGflops = $state<number>(0);
  let embeddingArray: np.Array;

  // Flat list mapping excerpt index to { chapterIdx, excerptIdx, text }
  let excerptList = $state<
    { chapterIdx: number; excerptIdx: number; text: string }[]
  >([]);

  // Search state
  let searchQuery = $state("");
  let searchResults = $state<
    { chapterIdx: number; excerptIdx: number; text: string; score: number }[]
  >([]);
  let isSearching = $state(false);

  const numExcerpts = $derived(
    book
      ? book.chapters.map((c) => c.excerpts.length).reduce((a, b) => a + b, 0)
      : 0,
  );

  async function setupBook(bookId: string) {
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      alert(
        "WebGPU is not enabled on this browser, try on Chrome or upgrade to iOS 26.",
      );
      return;
    }
    defaultDevice("webgpu");

    const model = await getModel();
    const tokenizer = await getTokenizer();

    isDownloadingData = true;
    try {
      book = await downloadBook(bookId);
    } catch (error: any) {
      alert("Error downloading book: " + error.message);
      return;
    } finally {
      isDownloadingData = false;
    }
    console.log($state.snapshot(book));
    hasData = true;

    const startTime = performance.now();
    const { tokens, excerptToChapter } = tokenizeExcerpts(
      book.chapters,
      tokenizer,
    );
    console.log(`Total excerpts: ${tokens.length}`);
    const endTime = performance.now();
    console.log(
      `Tokenized ${tokens.length} excerpts in ${endTime - startTime} ms`,
    );

    excerptList = [];
    for (let ci = 0; ci < book.chapters.length; ci++) {
      for (let ei = 0; ei < book.chapters[ci].excerpts.length; ei++) {
        excerptList.push({
          chapterIdx: ci,
          excerptIdx: ei,
          text: book.chapters[ci].excerpts[ei],
        });
      }
    }

    embeddingProgress = new Array(book.chapters.length).fill(0);
    embeddingTotal = 0;

    try {
      console.log("total params:", getTextParamCount(model));

      embeddingArray = await computeEmbeddings(model, tokens, (info) => {
        embeddingArray = info.embeddings;
        for (
          let j = info.batchStart;
          j < info.batchStart + info.batchSize;
          j++
        ) {
          embeddingProgress[excerptToChapter[j]]++;
        }
        embeddingProgress = embeddingProgress;
        embeddingTotal += info.batchSize;
        embeddingGflops = info.gflopsPerSec;
        console.log(
          `Processed rows ${info.batchStart} to ${info.batchStart + info.batchSize} (${info.gflopsPerSec.toFixed(1)} GFLOP/s)`,
        );
      });
    } catch (error) {
      console.error("Error in main:", error);
    }
  }

  let pendingQuery: string | null = null;
  let searchInProgress = false;

  async function search(query: string) {
    if (searchInProgress) {
      pendingQuery = query;
      return;
    }

    if (!query.trim() || !embeddingArray || embeddingTotal === 0) {
      searchResults = [];
      return;
    }

    searchInProgress = true;
    isSearching = true;
    try {
      const model = await getModel();
      const tokenizer = await getTokenizer();

      const results = await searchEmbeddings(
        model,
        tokenizer,
        embeddingArray,
        query,
        10,
      );

      searchResults = results.map((r) => ({
        ...excerptList[r.index],
        score: r.score,
      }));
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      searchInProgress = false;
      isSearching = false;

      if (pendingQuery !== null) {
        const nextQuery = pendingQuery;
        pendingQuery = null;
        search(nextQuery);
      }
    }
  }

  async function clearCache() {
    try {
      await opfs.clear();
      console.log("Cache cleared");
    } catch (error) {
      console.error("Error clearing cache:", error);
    }
  }
</script>

<DownloadManager bind:this={downloadManager} />

<div class="min-h-screen bg-white">
  <!-- Header with search bar -->

  {#if false}
    <header class="border-b border-gray-200">
      <div class="max-w-4xl mx-auto px-4 py-8">
        <div class="flex items-center gap-4 mb-4">
          <button
            onclick={getModel}
            disabled={isDownloadingWeights || hasModel}
            class="btn"
          >
            {isDownloadingWeights
              ? "Loading…"
              : hasModel
                ? "Model downloaded ✔️"
                : "Download model"}
          </button>
          <button
            onclick={clearCache}
            class="px-4 py-2 border-2 border-black hover:bg-black hover:text-white transition-colors"
          >
            Clear Cache
          </button>
        </div>
      </div>
    </header>
  {/if}

  <!-- Main content area -->
  <main class="max-w-4xl mx-auto px-4 py-8">
    <!-- Empty state -->
    {#if !hasData}
      <div class="text-center py-16">
        <h2 class="text-3xl font-normal mb-2">No data yet</h2>
        <p class="text-lg text-gray-600 mb-12">
          Download and embed a dataset to get started.
        </p>

        <div class="flex flex-col items-center gap-8">
          <div class="w-full max-w-md">
            <h3 class="font-medium mb-4">Load a prepared dataset</h3>
            <div class="flex flex-col gap-3">
              <button
                class="btn"
                onclick={() => setupBook("dickens-great-expectations")}
                disabled={isDownloadingWeights || isDownloadingData}
              >
                <BookMarkedIcon size={20} />
                <span><em>Great Expectations</em> by Charles Dickens</span>
              </button>
              <button
                class="btn"
                onclick={() => setupBook("wilde-dorian-gray")}
                disabled={isDownloadingWeights || isDownloadingData}
              >
                <BookMarkedIcon size={20} />
                <span><em>The Picture of Dorian Gray</em> by Oscar Wilde</span>
              </button>
              <button
                class="btn"
                onclick={() => setupBook("fitzgerald-great-gatsby")}
                disabled={isDownloadingWeights || isDownloadingData}
              >
                <BookMarkedIcon size={20} />
                <span><em>The Great Gatsby</em> by F. Scott Fitzgerald</span>
              </button>
            </div>
          </div>

          <div class="text-sm text-gray-400 uppercase tracking-wider">or</div>

          <div class="w-full max-w-md">
            <h3 class="font-medium mb-4">Upload your own data</h3>
            <div class="flex flex-col gap-3">
              <button class="btn" disabled>
                <FileTextIcon size={20} />
                Coming soon!
              </button>
            </div>
          </div>
        </div>
      </div>
    {:else}
      <section class="mb-8">
        <form
          class="mb-4"
          onsubmit={(e) => {
            e.preventDefault();
            search(searchQuery);
          }}
        >
          <input
            type="text"
            placeholder="Search excerpts…"
            class="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:border-primary focus:outline-none"
            bind:value={searchQuery}
            oninput={() => search(searchQuery)}
            disabled={embeddingTotal === 0}
          />
        </form>

        {#if isSearching}
          <p class="text-gray-500">Searching…</p>
        {:else if searchResults.length > 0}
          <div
            class="grid grid-cols-[auto_1fr_auto] gap-x-4 gap-y-2 items-baseline"
          >
            {#each searchResults as result, i (i)}
              <span class="text-sm font-medium text-primary whitespace-nowrap">
                {book.chapters[result.chapterIdx].title}
              </span>
              <p class="text-sm text-gray-700">{result.text}</p>
              <span class="text-xs text-gray-400"
                >{result.score.toFixed(3)}</span
              >
            {/each}
          </div>
        {:else if searchQuery.trim() && embeddingTotal > 0}
          <p class="text-gray-500">No results found.</p>
        {:else if embeddingTotal === 0}
          <p class="text-gray-500">Waiting for embeddings to complete…</p>
        {/if}
      </section>

      <section class="border-primary border-2 bg-primary/10 rounded-xl p-6">
        <div class="mb-4">
          <h2 class="text-lg font-semibold mb-0.5">
            Embedding: <em>{book.title}</em>
            {#if embeddingTotal === numExcerpts}
              ✅
            {/if}
          </h2>
          <p class="text-gray-600 text-sm">
            {embeddingTotal < numExcerpts
              ? "Currently embedding"
              : "Generated embeddings for"}
            {numExcerpts.toLocaleString()} excerpts with MobileCLIP2.
            {#if embeddingGflops}
              <span class="font-bold">{embeddingGflops.toFixed(2)} GFLOP/s</span
              >
            {/if}
          </p>
        </div>
        <div class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 items-center">
          {#each book.chapters as chapter, i (i)}
            <p class="max-w-[24ch] truncate text-sm">{chapter.title}</p>
            <div
              class="w-full bg-white/75 border border-primary/20 rounded-full h-4 overflow-hidden"
            >
              <div
                class="bg-primary h-3.5 transition-all duration-150 ease-linear"
                style="width: {chapter.excerpts.length > 0
                  ? (100 * (embeddingProgress[i] || 0)) /
                    chapter.excerpts.length
                  : 0}%"
              ></div>
            </div>
          {/each}
        </div>
      </section>
    {/if}
  </main>
</div>

<style lang="postcss">
  @reference "$app.css";

  .btn {
    @apply flex items-center justify-center gap-2 px-5 py-2.5 border-2 border-black;
    @apply enabled:hover:bg-black enabled:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors;
  }
</style>
