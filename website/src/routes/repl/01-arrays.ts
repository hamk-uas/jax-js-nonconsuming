import { grad, numpy as np } from "@jax-js-nonconsuming/jax";

const f = (x: np.Array) => np.sqrt(x.mul(x).sum());
const df = grad(f);

using x = np.array([1, 2, 3, 4]);
console.log(f(x).js());
console.log(df(x).js());
