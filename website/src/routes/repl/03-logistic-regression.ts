import {
  grad,
  jit,
  nn,
  numpy as np,
  random,
  tree,
} from "@jax-js-nonconsuming/jax";
import { applyUpdates, sgd } from "@jax-js-nonconsuming/optax";

// Logistic regression on a sample dataset.
//   > Classifier: y = sigmoid(X @ w)
//   > Loss: Binary cross-entropy,
//           loss(w) = y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)

using wTrue = np.array([2.0, -1.0, 0.5, -1.5]);

using key = random.key(0);
const X = random.uniform(key, [500, 4], { minval: -1, maxval: 1 });
const y = np.dot(X, wTrue).greater(0).astype(np.float32);

// Define loss function (binary cross-entropy).
using lossFn = jit((w: np.Array) => {
  using logits = np.dot(X, w);
  using logP = nn.logSigmoid(logits);
  using logNotP = nn.logSigmoid(np.negative(logits));
  const loss = np.add(y.mul(logP), np.subtract(1, y).mul(logNotP)).mean().neg();
  return loss;
});

// Try adding jit() to lossGrad to see the code get faster.
const lossGrad = grad(lossFn);

// Training loop.
const steps = 100;
const solver = sgd(0.2);

let w = np.zerosLike(wTrue);
let optState = solver.init(w);

for (let step = 0; step < steps; step++) {
  using grads = lossGrad(w);
  const [newUpdates, newOptState] = solver.update(grads, optState);
  tree.dispose(optState);
  const newW = applyUpdates(w, newUpdates);
  w.dispose();
  tree.dispose(newUpdates);
  w = newW;
  optState = newOptState;
  if (step % 20 === 19) {
    using lossArr = lossFn(w);
    const loss = await lossArr.jsAsync();
    console.log(`Step ${step + 1}: loss = ${loss}`);
  }
}

// Output learned weights.
console.log("Learned weights:", await w.jsAsync());
w.dispose();
tree.dispose(optState);
X.dispose();
y.dispose();
