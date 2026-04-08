import { grad, jit, numpy as np, vmap } from "@hamk-uas/jax-js-nonconsuming";

const f = (x: np.Array) => {
  using sq = x.mul(x);
  using s = sq.sum();
  return np.sqrt(s);
};

using df = jit(grad(f));
using squareGrad = jit(vmap(grad(np.square)));

using x = np.array([1, 2, 3, 4]);

using y0 = f(x);
console.log(y0.js());

using y1 = df(x);
console.log(y1.js());

using y2 = squareGrad(x);
console.log(y2.js());
