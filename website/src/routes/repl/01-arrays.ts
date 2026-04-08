import { grad, jit, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

const f = (x: np.Array) => {
  using sq = x.mul(x);
  using s = sq.sum();
  return np.sqrt(s);
};
using df = jit(grad(f));

using x = np.array([1, 2, 3, 4]);
using y = f(x);
using dy = df(x);
console.log(y.js());
console.log(dy.js());
