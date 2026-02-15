import {
  blockUntilReady,
  numpy as np,
  random,
  tree,
  valueAndGrad,
} from "@jax-js/jax";
import { adam, applyUpdates } from "@jax-js/optax";
import { range, shuffle } from "es-toolkit";

import { loadData, lossFn, type ModelType, type Params } from "./models";

export interface TrainMetric {
  iteration: number;
  loss: number;
}

export interface TestMetric {
  epoch: number;
  loss: number;
  acc: number;
}

export interface TrainingCallbacks {
  log(msg: string): void;
  onTrainBatch(metric: TrainMetric): void;
  onTestEval(metric: TestMetric): void;
  onParamsUpdate(params: Params): void;
  onEpochEnd(): void;
  shouldStop(): boolean;
}

export interface TrainingOptions {
  model: ModelType;
  learningRate: number;
  batchSize: number;
}

export async function train(
  opts: TrainingOptions,
  callbacks: TrainingCallbacks,
): Promise<void> {
  const { model, learningRate, batchSize } = opts;

  let params = model.init(random.key(0));
  await blockUntilReady(params);

  const loss = lossFn(model.predict);
  callbacks.onParamsUpdate(params);

  callbacks.log(`=> Loading MNIST database from CDN or cache...`);
  const startTime = performance.now();
  const { X_train, y_train, X_test, y_test } = await loadData();
  const duration = performance.now() - startTime;
  callbacks.log(`=> Data loaded in ${duration.toFixed(1)} ms`);

  const solver = adam(learningRate);
  let optState = solver.init(params);

  try {
    const numBatches = Math.ceil(X_train.shape[0] / batchSize);
    let paramsOwnedByCallback = true; // initial onParamsUpdate already shared
    for (let epoch = 0; epoch < 10; epoch++) {
      callbacks.log(`=> Epoch ${epoch + 1}`);
      const randomIndices = shuffle(range(X_train.shape[0]));
      let paramsUpdatedThisEpoch = false;

      for (let i = 0; i < numBatches; i++) {
        if (callbacks.shouldStop()) break;
        using indices = np.array(
          randomIndices.slice(i * batchSize, (i + 1) * batchSize),
          { dtype: np.int32 },
        );

        const batchStart = performance.now();
        using X = X_train.slice(indices);
        using y = y_train.slice(indices);
        const [lossVal, lossGrad] = valueAndGrad(loss)(params, X, y);
        const [updates, newOptState] = solver.update(lossGrad, optState);
        tree.dispose(lossGrad);
        // Note: solver.update() already disposes old optState internally
        const newParams = applyUpdates(params, updates);
        tree.dispose(updates);
        // Only dispose params if we still own them (not handed to callback)
        if (!paramsOwnedByCallback) tree.dispose(params);
        paramsOwnedByCallback = false;
        params = newParams;
        optState = newOptState;
        paramsUpdatedThisEpoch = true;

        await blockUntilReady(params);
        const batchDuration = performance.now() - batchStart;
        const lossNumber = (await lossVal.jsAsync()) as number;
        lossVal.dispose();
        callbacks.log(
          `batch ${i}/${numBatches} completed in ${batchDuration.toFixed(1)} ms, loss: ${lossNumber.toFixed(4)}`,
        );
        callbacks.onTrainBatch({
          iteration: epoch * numBatches + i + 1,
          loss: lossNumber,
        });
      }

      if (paramsUpdatedThisEpoch) {
        callbacks.onParamsUpdate(params);
        paramsOwnedByCallback = true;
      }
      callbacks.onEpochEnd();

      if (callbacks.shouldStop()) break;

      callbacks.log(`=> Evaluating on test set...`);
      const testStartTime = performance.now();
      const testSize = X_test.shape[0];
      const testLoss: number[] = [];
      const testAcc: number[] = [];
      for (let i = 0; i + batchSize <= testSize; i += batchSize) {
        using X = X_test.slice([i, i + batchSize]).reshape([-1, 784]);
        using y = y_test.slice([i, i + batchSize]);
        using lossArr = loss(params, X, y);
        testLoss.push(await lossArr.jsAsync());
        using logits = model.predict(params, X);
        using preds = np.argmax(logits, 1);
        testAcc.push(await preds.equal(y).astype(np.uint32).sum().jsAsync());
      }
      const testDuration = performance.now() - testStartTime;
      const testLossAvg = testLoss.reduce((a, b) => a + b) / testLoss.length;
      const testAccAvg = testAcc.reduce((a, b) => a + b) / testSize;
      callbacks.log(
        `=> Test acc: ${testAccAvg.toFixed(4)}, loss: ${testLossAvg.toFixed(4)}, in ${testDuration.toFixed(1)} ms`,
      );
      callbacks.onTestEval({
        epoch: epoch + 1,
        loss: testLossAvg,
        acc: testAccAvg,
      });
    }
  } finally {
    X_train.dispose();
    y_train.dispose();
    X_test.dispose();
    y_test.dispose();
    tree.dispose(optState);
    // params ownership was transferred to caller via onParamsUpdate
  }
}
