import { grad, numpy as np, tree } from "@jax-js-nonconsuming/jax";
import { applyUpdates, sgd, squaredError } from "@jax-js-nonconsuming/optax";
import { expect, test } from "vitest";

test("stochastic gradient descent", () => {
  using ones = np.ones([3]);
  const params = np.array([1.0, 2.0, 3.0]);

  const solver = sgd(0.11);
  const optState = solver.init(params);

  const f = (x: np.Array) => squaredError(x, ones).sum();
  using paramsGrad = grad(f)(params);
  const [updates, newState] = solver.update(paramsGrad, optState);
  tree.dispose(newState);
  const newParams = applyUpdates(params, updates);
  params.dispose();
  updates.dispose();

  expect(newParams).toBeAllclose([1.0, 1.78, 2.56]);
  newParams.dispose();
});

test("sgd with momentum", () => {
  using ones = np.ones([3]);
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = sgd(0.1, { momentum: 0.9 });
  let optState = solver.init(params);

  const f = (x: np.Array) => squaredError(x, ones).sum();

  // First update
  let paramsGrad = grad(f)(params);
  let updates: np.Array;
  [updates, optState] = solver.update(paramsGrad, optState);
  paramsGrad.dispose();
  let oldParams = params;
  params = applyUpdates(params, updates);
  oldParams.dispose();
  updates.dispose();

  // Second update
  paramsGrad = grad(f)(params);
  [updates, optState] = solver.update(paramsGrad, optState);
  paramsGrad.dispose();
  oldParams = params;
  params = applyUpdates(params, updates);
  oldParams.dispose();
  updates.dispose();

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
  params.dispose();
  tree.dispose(optState);
});

test("sgd with nesterov momentum", () => {
  using ones = np.ones([3]);
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = sgd(0.1, { momentum: 0.9, nesterov: true });
  let optState = solver.init(params);

  const f = (x: np.Array) => squaredError(x, ones).sum();

  // First update
  let paramsGrad = grad(f)(params);
  let updates: np.Array;
  [updates, optState] = solver.update(paramsGrad, optState);
  paramsGrad.dispose();
  let oldParams = params;
  params = applyUpdates(params, updates);
  oldParams.dispose();
  updates.dispose();

  // Second update
  paramsGrad = grad(f)(params);
  [updates, optState] = solver.update(paramsGrad, optState);
  paramsGrad.dispose();
  oldParams = params;
  params = applyUpdates(params, updates);
  oldParams.dispose();
  updates.dispose();

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
  params.dispose();
  tree.dispose(optState);
});
