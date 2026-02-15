import {
  jit,
  lax,
  nn,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";

import { fetchMnist } from "$lib/dataset/mnist";

export type Params = { [key: string]: np.Array };

export type ModelType = {
  init(key: np.Array): Params;
  predict(params: Params, x: np.Array): np.Array;
};

function maxPool2x2(x: np.Array): np.Array {
  return lax.reduceWindow(x, np.max, [2, 2], [2, 2]);
}

export const MLP: ModelType = {
  init(key: np.Array): Params {
    const [d0, d1, d2, d3] = [784, 256, 128, 10];
    const [k11, k12, k21, k22, k31, k32] = random.split(key, 6);
    const w1 = random.uniform(k11, [d0, d1], {
      minval: -1 / Math.sqrt(d0),
      maxval: 1 / Math.sqrt(d0),
    });
    const b1 = random.uniform(k12, [d1], {
      minval: -1 / Math.sqrt(d0),
      maxval: 1 / Math.sqrt(d0),
    });
    const w2 = random.uniform(k21, [d1, d2], {
      minval: -1 / Math.sqrt(d1),
      maxval: 1 / Math.sqrt(d1),
    });
    const b2 = random.uniform(k22, [d2], {
      minval: -1 / Math.sqrt(d1),
      maxval: 1 / Math.sqrt(d1),
    });
    const w3 = random.uniform(k31, [d2, d3], {
      minval: -1 / Math.sqrt(d2),
      maxval: 1 / Math.sqrt(d2),
    });
    const b3 = random.uniform(k32, [d3], {
      minval: -1 / Math.sqrt(d2),
      maxval: 1 / Math.sqrt(d2),
    });
    return { w1, b1, w2, b2, w3, b3 };
  },

  predict: jit((params: Params, x: np.Array): np.Array => {
    x = x.reshape([-1, 784]);
    const z1 = np.dot(x, params.w1).add(params.b1);
    const a1 = nn.relu(z1);
    const z2 = np.dot(a1, params.w2).add(params.b2);
    const a2 = nn.relu(z2);
    const z3 = np.dot(a2, params.w3).add(params.b3);
    return nn.logSoftmax(z3);
  }),
};

export const ConvNet: ModelType = {
  init(key: np.Array): Params {
    const [k11, k12, k21, k22, k31, k32, k41, k42] = random.split(key, 8);
    const w1 = random.uniform(k11, [24, 1, 5, 5], {
      minval: -1 / Math.sqrt(5 * 5),
      maxval: 1 / Math.sqrt(5 * 5),
    });
    const b1 = random.uniform(k12, [24, 1, 1], {
      minval: -1 / Math.sqrt(5 * 5),
      maxval: 1 / Math.sqrt(5 * 5),
    });
    const w2 = random.uniform(k21, [32, 24, 3, 3], {
      minval: -1 / Math.sqrt(24 * 3 * 3),
      maxval: 1 / Math.sqrt(24 * 3 * 3),
    });
    const b2 = random.uniform(k22, [32, 1, 1], {
      minval: -1 / Math.sqrt(24 * 3 * 3),
      maxval: 1 / Math.sqrt(24 * 3 * 3),
    });
    const w3 = random.uniform(k31, [800, 128], {
      minval: -1 / Math.sqrt(800),
      maxval: 1 / Math.sqrt(800),
    });
    const b3 = random.uniform(k32, [128], {
      minval: -1 / Math.sqrt(800),
      maxval: 1 / Math.sqrt(800),
    });
    const w4 = random.uniform(k41, [128, 10], {
      minval: -1 / Math.sqrt(128),
      maxval: 1 / Math.sqrt(128),
    });
    const b4 = random.uniform(k42, [10], {
      minval: -1 / Math.sqrt(128),
      maxval: 1 / Math.sqrt(128),
    });
    return { w1, b1, w2, b2, w3, b3, w4, b4 };
  },

  predict: jit((params: Params, x: np.Array): np.Array => {
    x = x.reshape([-1, 1, 28, 28]);
    const z1 = maxPool2x2(
      lax.convGeneralDilated(x, params.w1, [1, 1], "VALID").add(params.b1),
    );
    const a1 = nn.relu(z1); // [batch, 24, 12, 12]
    const z2 = maxPool2x2(
      lax.convGeneralDilated(a1, params.w2, [1, 1], "VALID").add(params.b2),
    );
    const a2 = nn.relu(z2); // [batch, 32, 5, 5]
    const a2flat = a2.reshape([-1, 800]); // Flatten to [batch, 800]
    const z3 = np.dot(a2flat, params.w3).add(params.b3);
    const a3 = nn.relu(z3);
    const z4 = np.dot(a3, params.w4).add(params.b4);
    return nn.logSoftmax(z4);
  }),
};

export function lossFn(predict: (params: Params, x: np.Array) => np.Array) {
  return (params: Params, x: np.Array, y: np.Array): np.Array => {
    const batchSize = y.shape[0];
    const logits = predict(params, x);
    return logits
      .mul(nn.oneHot(y, 10))
      .sum()
      .mul(-1 / batchSize);
  };
}

export async function loadData(): Promise<{
  X_train: np.Array;
  y_train: np.Array;
  X_test: np.Array;
  y_test: np.Array;
}> {
  const mnist = await fetchMnist();
  return {
    X_train: np
      .array(new Float32Array(mnist.train.images.data))
      .mul(1 / 255)
      .reshape([-1, 28, 28]),
    y_train: np.array(mnist.train.labels.data),
    X_test: np
      .array(new Float32Array(mnist.test.images.data))
      .mul(1 / 255)
      .reshape([-1, 28, 28]),
    y_test: np.array(mnist.test.labels.data),
  };
}

export function normalizeImage(X: np.Array): np.Array {
  // X.shape === [28, 28]
  using xgrid = np.meshgrid(
    [np.arange(28).astype(np.float32), np.arange(28).astype(np.float32)],
    { indexing: "ij" },
  )[0];
  using ygrid = np.meshgrid(
    [np.arange(28).astype(np.float32), np.arange(28).astype(np.float32)],
    { indexing: "ij" },
  )[1];
  const dx = Math.round(13.5 - X.mul(xgrid).sum().div(X.sum()).js());
  const dy = Math.round(13.5 - X.mul(ygrid).sum().div(X.sum()).js());
  if (dx > 0) {
    using old = X;
    X = np.pad(old, { 0: [dx, 0] }).slice([0, 28], []);
  }
  if (dx < 0) {
    using old = X;
    X = np.pad(old, { 0: [0, -dx] }).slice([-dx], []);
  }
  if (dy > 0) {
    using old = X;
    X = np.pad(old, { 1: [dy, 0] }).slice([], [0, 28]);
  }
  if (dy < 0) {
    using old = X;
    X = np.pad(old, { 1: [0, -dy] }).slice([], [-dy]);
  }
  return X;
}

/**
 * Run inference on a single 28×28 image and return class probabilities.
 */
export async function runInference(
  model: ModelType,
  params: Params,
  imageData: Float32Array,
): Promise<number[]> {
  using image = normalizeImage(
    np.array(imageData as Float32Array<ArrayBuffer>).reshape([28, 28]),
  );
  using logits = model.predict(params, image.reshape([1, 28, 28]));
  return (await np.exp(logits).slice(0).jsAsync()) as number[];
}
