import { grad, JsTree, numpy as np, tree } from "@hamk-uas/jax-js-nonconsuming";
import { adamw, applyUpdates, squaredError } from "@hamk-uas/jax-js-nonconsuming-optax";
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
