import {
  grad,
  jit,
  nn,
  numpy as np,
  random,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { applyUpdates, sgd } from "@hamk-uas/jax-js-nonconsuming-optax";

// Logistic regression on a sample dataset.
//   > Classifier: y = sigmoid(X @ w)
//   > Loss: Binary cross-entropy,
//           loss(w) = y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)

using wTrue = np.array([2.0, -1.0, 0.5, -1.5]);

using key = random.key(0);
const X = random.uniform(key, [500, 4], { minval: -1, maxval: 1 });
const y = (() => {
  using dot = np.dot(X, wTrue);
  using gt = dot.greater(0);
  return gt.astype(np.float32);
})();

// Define loss function (binary cross-entropy).
const baseLossFn = (w: np.Array) => {
  using logits = np.dot(X, w);
  using logP = nn.logSigmoid(logits);
  using negLogits = np.negative(logits);
  using logNotP = nn.logSigmoid(negLogits);
  using oneMinusY = np.subtract(1, y);
  using yLogP = y.mul(logP);
  using yBarLogNotP = oneMinusY.mul(logNotP);
  using meanTerms = np.add(yLogP, yBarLogNotP).mean();
  return meanTerms.neg();
};

using lossFn = jit(baseLossFn);
// Comment out jit() to see the gradient step get slower.
using lossGrad = jit(grad(baseLossFn));

// Training loop.
const steps = 100;
const solver = sgd(0.2);

let w = np.zerosLike(wTrue);
let optState = solver.init(w);

for (let step = 0; step < steps; step++) {
  using d = new DisposableStack();
  const grads = d.use(lossGrad(w));
  const [updates, newOptState] = solver.update(grads, optState);
  d.adopt(updates, tree.dispose);
  d.adopt(optState, tree.dispose);
  w = applyUpdates(d.use(w), updates);
  optState = newOptState;
  if (step % 20 === 19) {
    const loss = await d.use(lossFn(w)).jsAsync();
    console.log(`Step ${step + 1}: loss = ${loss}`);
  }
}

// Output learned weights.
console.log("Learned weights:", await w.jsAsync());
w.dispose();
tree.dispose(optState);
X.dispose();
y.dispose();
