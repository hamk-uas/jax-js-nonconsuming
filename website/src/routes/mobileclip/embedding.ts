import { jit, numpy as np, tree, vmap } from "@hamk-uas/jax-js-nonconsuming";
import type { tokenizers } from "@hamk-uas/jax-js-nonconsuming-loaders";

import { type MobileCLIP, runMobileCLIPTextEncoder } from "./clipInference";

/** Rough estimate: ~38M non-embedding params × 77 context length × 2 */
const GFLOP_PER_TEXT_EMBED = (2 * 38e6 * 77) / 1e9;
export const D_EMBED = 512;

const runEncoder = jit(vmap(runMobileCLIPTextEncoder, [null, 0]));

/** Count total parameters in the text encoder. */
export function getTextParamCount(model: MobileCLIP): number {
  return tree
    .flatten(model.text)[0]
    .map((x: np.Array) => x.size)
    .reduce((a: number, b: number) => a + b, 0);
}

/** Tokenize all excerpts and return tokens + chapter index mapping. */
export function tokenizeExcerpts(
  chapters: { excerpts: string[] }[],
  tokenizer: tokenizers.BpeEncoding,
): { tokens: number[][]; excerptToChapter: number[] } {
  const tokens: number[][] = [];
  const excerptToChapter: number[] = [];
  for (let ci = 0; ci < chapters.length; ci++) {
    for (let ei = 0; ei < chapters[ci].excerpts.length; ei++) {
      tokens.push(tokenizer.encode(chapters[ci].excerpts[ei]));
      excerptToChapter.push(ci);
    }
  }
  return { tokens, excerptToChapter };
}

export interface EmbeddingBatchInfo {
  batchStart: number;
  batchSize: number;
  gflopsPerSec: number;
  /** Current cumulative embeddings. Valid until the next callback or function return. */
  embeddings: np.Array;
}

/**
 * Compute text embeddings in batches of 16.
 * @param onBatch Called after each batch with progress and current cumulative embeddings.
 * @returns The final cumulative embeddings array. Caller owns and must dispose.
 */
export async function computeEmbeddings(
  model: MobileCLIP,
  tokens: number[][],
  onBatch?: (info: EmbeddingBatchInfo) => void,
): Promise<np.Array> {
  using ar = np.array(tokens, { dtype: np.uint32 });
  let embeddings = np.zeros([0, D_EMBED], { dtype: np.float16 });

  for (let i = 0; i < ar.shape[0]; i += 16) {
    using batch = ar.slice([i, Math.min(i + 16, ar.shape[0])]);
    const batchSize = batch.shape[0];
    performance.mark("clip-start");
    const t0 = performance.now();
    using result = runEncoder(model.text, batch) as np.Array;
    await result.blockUntilReady();
    const t1 = performance.now();
    performance.mark("clip-end");
    performance.measure("clip", "clip-start", "clip-end");

    const prevEmbeddings = embeddings;
    embeddings = np.concatenate([prevEmbeddings, result], 0);
    await embeddings.blockUntilReady();

    onBatch?.({
      batchStart: i,
      batchSize,
      gflopsPerSec: (GFLOP_PER_TEXT_EMBED * batchSize) / (1e-3 * (t1 - t0)),
      embeddings,
    });

    prevEmbeddings.dispose();
  }
  return embeddings;
}

export interface SearchResult {
  index: number;
  score: number;
}

/**
 * Search embeddings for a query string. Returns top-K results with indices and scores.
 */
export async function searchEmbeddings(
  model: MobileCLIP,
  tokenizer: tokenizers.BpeEncoding,
  embeddingArray: np.Array,
  query: string,
  topK: number = 10,
): Promise<SearchResult[]> {
  using queryArray = np.array([tokenizer.encode(query)], { dtype: np.uint32 });
  using queryEmbedBatched = runEncoder(model.text, queryArray) as np.Array;
  using queryEmbed = queryEmbedBatched.slice(0);
  using embF32 = embeddingArray.astype(np.float32);
  using queryF32 = queryEmbed.astype(np.float32);
  using dotResult = np.dot(embF32, queryF32);
  const scores: number[] = await dotResult.jsAsync();

  const indices = Array.from({ length: scores.length }, (_, i) => i);
  indices.sort((a, b) => scores[b] - scores[a]);

  return indices.slice(0, topK).map((idx) => ({
    index: idx,
    score: scores[idx],
  }));
}
