import {
  grad,
  jit,
  JsTree,
  numpy as np,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import {
  adamw,
  applyUpdates,
  scaleByAdam,
  squaredError,
} from "@hamk-uas/jax-js-nonconsuming-optax";
import { expect, test } from "vitest";

test("adamw optimizer", () => {
  using ones = np.ones([3]);
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001);
  let optState = solver.init(params);

  const f = (x: np.Array) => squaredError(x, ones).sum();
  using paramsGrad = grad(f)(params);
  let updates: np.Array;
  [updates, optState] = solver.update(paramsGrad, optState, params);
  const newParams = applyUpdates(params, updates);
  params.dispose();
  updates.dispose();
  params = newParams;

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
  params.dispose();
  tree.dispose(optState);
});

test("adamw with custom weight decay", () => {
  using ones = np.ones([3]);
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001, { weightDecay: 0.01 });
  let optState = solver.init(params);

  const f = (x: np.Array) => squaredError(x, ones).sum();
  using paramsGrad = grad(f)(params);
  let updates: np.Array;
  [updates, optState] = solver.update(paramsGrad, optState, params);
  const newParams = applyUpdates(params, updates);
  params.dispose();
  updates.dispose();
  params = newParams;

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
  params.dispose();
  tree.dispose(optState);
});

test("adamw with nesterov", () => {
  using ones = np.ones([3]);
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001, { nesterov: true, weightDecay: 0.005 });
  let optState = solver.init(params);

  const f = (x: np.Array) => squaredError(x, ones).sum();
  using paramsGrad = grad(f)(params);
  let updates: np.Array;
  [updates, optState] = solver.update(paramsGrad, optState, params);
  const newParams = applyUpdates(params, updates);
  params.dispose();
  updates.dispose();
  params = newParams;

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
  params.dispose();
  tree.dispose(optState);
});

test("adamw with callable mask", () => {
  using ones = np.ones([3]);
  let params = np.array([1.0, 2.0, 3.0]);

  // Mask function that returns a mask tree - only apply decay to first element
  const maskFn = (updates: JsTree<np.Array>): JsTree<np.Array> => {
    return tree.map((_u: np.Array) => {
      return np.array([1.0, 0.0, 0.0]);
    }, updates);
  };

  const solver = adamw(0.001, { weightDecay: 0.01, mask: maskFn });
  let optState = solver.init(params);

  const f = (x: np.Array) => squaredError(x, ones).sum();
  using paramsGrad = grad(f)(params);
  let updates: np.Array;
  [updates, optState] = solver.update(paramsGrad, optState, params);
  const newParams = applyUpdates(params, updates);
  params.dispose();
  updates.dispose();
  params = newParams;

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
  params.dispose();
  tree.dispose(optState);
});

test("scaleByAdam update works inside jit", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  const solver = scaleByAdam();

  // Run scaleByAdam.update under jit — this verifies treeBiasCorrection
  // works with tracers (uses np.power instead of count.item()).
  const jitState = solver.init(params);
  const { count, mu, nu } = jitState as {
    count: np.Array;
    mu: np.Array;
    nu: np.Array;
  };
  using jitUpdate = jit(
    (g: np.Array, p: np.Array, c: np.Array, m: np.Array, n: np.Array) => {
      const state = { count: c, mu: m, nu: n };
      const [upd, _newState] = solver.update(g, state, p);
      tree.dispose(_newState);
      return upd;
    },
  );
  using jitUpdates = jitUpdate(updates, params, count, mu, nu);

  // Adam with count=0→1 on [0.1,0.2,0.3] should produce ≈ [1, 1, 1]
  // (mu_hat/sqrt(nu_hat) with both bias-corrected at step 1)
  expect(jitUpdates).toBeAllclose([1.0, 1.0, 1.0], { atol: 0.01 });

  tree.dispose(jitState);
});
