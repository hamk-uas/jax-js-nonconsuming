import { grad, numpy as np } from "@jax-js-nonconsuming/jax";

const f = (x: np.Array) => {
  using sq = x.mul(x);
  using s = sq.sum();
  return np.sqrt(s);
};
const df = grad(f);

using x = np.array([1, 2, 3, 4]);
using y = f(x);
using dy = df(x);
console.log(y.js());
console.log(dy.js());
