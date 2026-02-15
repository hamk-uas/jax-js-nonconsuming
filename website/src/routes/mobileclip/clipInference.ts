import { nn, numpy as np } from "@jax-js-nonconsuming/jax";
import { safetensors, WeightMapper } from "@jax-js-nonconsuming/loaders";

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
  // Embed tokens and add positional embeddings
  let x = tokenEmbedding.slice(textTokens); // [L, D]
  {
    const old = x;
    x = old.add(positionalEmbedding);
    old.dispose();
  }

  for (const block of transformer) {
    const old = x;
    x = runMobileCLIPTextBlock(block, old);
    old.dispose();
  }
  {
    const old = x;
    x = runLayerNorm(lnFinal, old);
    old.dispose();
  }

  using finalFeatures = x.slice(np.argmax(textTokens, -1));
  using output = np.dot(textProjection.transpose(), finalFeatures); // [D_out]

  // Normalize output to be a unit vector
  return output.div(np.sqrt(np.sum(np.square(output))).add(1e-3));
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
  // Pre-norm attention block
  using normed1 = runLayerNorm(ln1, x);
  using attnOut = runMultiHeadAttention(attn, normed1);
  {
    const old = x;
    x = old.add(attnOut);
    old.dispose();
  }

  // Pre-norm MLP block
  using normed2 = runLayerNorm(ln2, x);
  let mlpOut = runLinear(mlpUp, normed2);
  {
    const old = mlpOut;
    mlpOut = nn.gelu(old, { approximate: false });
    old.dispose();
  }
  {
    const old = mlpOut;
    mlpOut = runLinear(mlpDown, old);
    old.dispose();
  }
  {
    const old = x;
    x = old.add(mlpOut);
    old.dispose();
  }
  mlpOut.dispose();

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
  const [q, k, v] = np.split(qkv, 3, -1);

  using output = nn
    .dotProductAttention(
      q.reshape([seqLen, numHeads, headDim]),
      k.reshape([seqLen, numHeads, headDim]),
      v.reshape([seqLen, numHeads, headDim]),
    )
    .reshape([seqLen, embed]);
  q.dispose();
  k.dispose();
  v.dispose();

  // Final projection
  return runLinear(outProj, output);
}

export type Linear = {
  weight: np.Array; // [Out, In]
  bias: np.Array; // [Out]
};

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  return np.dot(x, weight.transpose()).add(bias);
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
  {
    const old = x;
    x = old.sub(avg);
    old.dispose();
  }
  using denom = np
    .sqrt(
      np
        .square(x)
        .mul(1 / dimSize)
        .sum(-1, { keepdims: true }),
    )
    .add(1e-5);
  return x.div(denom).mul(weight).add(bias);
}
