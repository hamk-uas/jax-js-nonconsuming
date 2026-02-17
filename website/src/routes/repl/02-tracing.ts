import { jvp, makeJaxpr, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

// f(x) = (x + 2) * x
const f = (x: np.Array) => x.add(2).mul(x);

// fdot(x) = 2 * x + 2
using one = np.array(1);
const fdot = (x: np.Array) => jvp(f, [x], [one])[1];

using x0 = np.array(2);
console.log(makeJaxpr(f)(x0).jaxpr.toString());

using x1 = np.array(2);
const { jaxpr } = makeJaxpr(fdot)(x1);
console.log(jaxpr.toString());
