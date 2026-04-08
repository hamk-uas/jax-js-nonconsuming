import { nn, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import {
  safetensors,
  WeightMapper,
} from "@hamk-uas/jax-js-nonconsuming-loaders";

// MobileCLIP2 model weights interfaces and forward pass.

export type MobileCLIP = {
  text: MobileCLIPTextEncoder;
  visual: any; // TODO
  logitScale: np.Array;
};

const weightMapper = new WeightMapper({
  exact: {
    logit_scale: "logitScale",
    "text.token_embedding.weight": "text.tokenEmbedding",
    "text.positional_embedding": "text.positionalEmbedding",
    "text.text_projection": "text.textProjection",
  },
  prefix: {
    "text.transformer.resblocks.": "text.transformer.",
  },
  substring: {
    ".ln_final.": ".lnFinal.",
    ".ln_1.": ".ln1.",
    ".ln_2.": ".ln2.",
    ".mlp.c_fc.": ".mlpUp.",
    ".mlp.c_proj.": ".mlpDown.",
    ".attn.in_proj_": ".attn.qkvProj.",
    ".attn.out_proj.": ".attn.outProj.",
  },
});

export function fromSafetensors(file: safetensors.File): MobileCLIP {
  const mappedWeights = weightMapper.mapObject(file.tensors);
  const hydrated: Record<string, np.Array> = {};
  for (const [key, value] of Object.entries(mappedWeights)) {
    // console.log(key, value);
    if (value.dtype === "F16") {
      hydrated[key] = np.array(value.data as Float16Array<ArrayBuffer>, {
        dtype: np.float16,
        shape: value.shape,
      });
    } else if (value.dtype === "I64") {
      // Ignored, these are metadata for BatchNorm.
      continue;
    } else {
      throw new Error(`Unexpected dtype ${value.dtype} for weight ${key}`);
    }
  }
  return safetensors.toNested(hydrated);
}

export type MobileCLIPTextEncoder = {
  tokenEmbedding: np.Array;
  positionalEmbedding: np.Array;
  transformer: MobileCLIPTextBlock[];
  lnFinal: LayerNorm;
  textProjection: np.Array;
};

export function runMobileCLIPTextEncoder(
  {
    tokenEmbedding,
    positionalEmbedding,
    transformer,
    lnFinal,
    textProjection,
  }: MobileCLIPTextEncoder,
  textTokens: np.Array,
): np.Array {
  using d = new DisposableStack();
  // Embed tokens and add positional embeddings
  let x = tokenEmbedding.slice(textTokens); // [L, D]
  d.use(x);
  x = x.add(positionalEmbedding);

  for (const block of transformer) {
    d.use(x);
    x = runMobileCLIPTextBlock(block, x);
  }
  d.use(x);
  x = runLayerNorm(lnFinal, x);

  using argmaxIdx = np.argmax(textTokens, -1);
  using finalFeatures = x.slice(argmaxIdx);
  using projT = textProjection.transpose();
  using output = np.dot(projT, finalFeatures); // [D_out]

  // Normalize output to be a unit vector
  using sq = np.square(output);
  using sumSq = np.sum(sq);
  using norm = np.sqrt(sumSq);
  using normEps = norm.add(1e-3);
  return output.div(normEps);
}

export type MobileCLIPTextBlock = {
  ln1: LayerNorm;
  attn: MultiHeadAttention;
  ln2: LayerNorm;
  mlpUp: Linear;
  mlpDown: Linear;
};

export function runMobileCLIPTextBlock(
  { ln1, attn, ln2, mlpUp, mlpDown }: MobileCLIPTextBlock,
  x: np.Array,
): np.Array {
  using d = new DisposableStack();
  // Pre-norm attention block
  using normed1 = runLayerNorm(ln1, x);
  using attnOut = runMultiHeadAttention(attn, normed1);
  d.use(x);
  x = x.add(attnOut);

  // Pre-norm MLP block
  using normed2 = runLayerNorm(ln2, x);
  let mlpOut = runLinear(mlpUp, normed2);
  d.use(mlpOut);
  mlpOut = nn.gelu(mlpOut, { approximate: false });
  d.use(mlpOut);
  mlpOut = runLinear(mlpDown, mlpOut);
  d.use(x);
  x = x.add(mlpOut);
  d.use(mlpOut);

  return x;
}

export type MultiHeadAttention = {
  qkvProj: Linear;
  outProj: Linear;
};

export function runMultiHeadAttention(
  { qkvProj, outProj }: MultiHeadAttention,
  x: np.Array,
): np.Array {
  const numHeads = 8;
  const [seqLen, embed] = x.shape;
  const headDim = embed / numHeads;

  // Project to Q, K, V
  using qkv = runLinear(qkvProj, x); // [seqLen, 3 * embed]
  const [qRaw, kRaw, vRaw] = np.split(qkv, 3, -1);
  using q = qRaw;
  using k = kRaw;
  using v = vRaw;

  using qR = q.reshape([seqLen, numHeads, headDim]);
  using kR = k.reshape([seqLen, numHeads, headDim]);
  using vR = v.reshape([seqLen, numHeads, headDim]);
  using attnResult = nn.dotProductAttention(qR, kR, vR);
  using output = attnResult.reshape([seqLen, embed]);

  // Final projection
  return runLinear(outProj, output);
}

export type Linear = {
  weight: np.Array; // [Out, In]
  bias: np.Array; // [Out]
};

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  using wT = weight.transpose();
  using dot = np.dot(x, wT);
  return dot.add(bias);
}

export type LayerNorm = {
  weight: np.Array;
  bias: np.Array;
};

export function runLayerNorm(
  { weight, bias }: LayerNorm,
  x: np.Array,
): np.Array {
  // Normalize with respect to the last dimension of x.
  const dimSize = x.shape[x.ndim - 1];
  using avg = x.mean(-1, { keepdims: true });
  using _preCenter = x;
  x = x.sub(avg);
  using sq = np.square(x);
  using variance = sq.mul(1 / dimSize).sum(-1, { keepdims: true });
  using stddev = np.sqrt(variance);
  using denom = stddev.add(1e-5);
  using normalized = x.div(denom);
  using scaled = normalized.mul(weight);
  return scaled.add(bias);
}
