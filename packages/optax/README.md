# @jax-js/optax

This is a port of [Optax](https://github.com/google-deepmind/optax) to jax-js, for gradient
processing and optimization. It includes implementations of common algorithms like Adam.

## Example

```ts
import { adam } from "@jax-js/optax";

let params = np.array([1.0, 2.0, 3.0]);

const solver = adam(1e-3);
let optState = solver.init(params);

using target = np.ones([3]);
const f = (x: np.Array) => {
  using err = squaredError(x, target);
  return err.sum();
};

for (let i = 0; i < 100; i++) {
  using paramsGrad = grad(f)(params);
  const [newUpdates, newOptState] = solver.update(paramsGrad, optState);
  const newParams = applyUpdates(params, newUpdates);
  params.dispose();
  tree.dispose(optState);
  newUpdates.dispose();
  params = newParams;
  optState = newOptState;
}
```
