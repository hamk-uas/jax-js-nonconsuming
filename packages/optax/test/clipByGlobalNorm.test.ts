import { numpy as np, tree } from "@hamk-uas/jax-js-nonconsuming";
import { clipByGlobalNorm } from "@hamk-uas/optax";
import { expect, test } from "vitest";

test("clipByGlobalNorm: no clipping when norm is below threshold", () => {
  using p0 = np.zeros([1]);
  using p1 = np.zeros([1]);
  const params = [p0, p1];
  using u0 = np.array([3.0]);
  using u1 = np.array([4.0]);
  const updates = [u0, u1];

  const clipper = clipByGlobalNorm(10.0);
  const state = clipper.init(params);
  const [clipped, newState] = clipper.update(updates, state);
  tree.dispose(newState);

  expect(clipped[0]).toBeAllclose([3.0]);
  expect(clipped[1]).toBeAllclose([4.0]);
  tree.dispose(clipped);
});

test("clipByGlobalNorm: clips when norm exceeds threshold", () => {
  using p0 = np.zeros([1]);
  using p1 = np.zeros([1]);
  const params = [p0, p1];
  using u0 = np.array([3.0]);
  using u1 = np.array([4.0]);
  const updates = [u0, u1];

  const clipper = clipByGlobalNorm(2.5);
  const state = clipper.init(params);
  const [clipped, newState] = clipper.update(updates, state);
  tree.dispose(newState);

  expect(clipped[0]).toBeAllclose([1.5]);
  expect(clipped[1]).toBeAllclose([2.0]);
  tree.dispose(clipped);
});

test("clipByGlobalNorm: handles multi-dimensional gradients", () => {
  using p0 = np.zeros([1, 2]);
  using p1 = np.zeros([1, 2]);
  const params = [p0, p1];
  using u0 = np.array([[3.0, 4.0]]);
  using u1 = np.array([[5.0, 12.0]]);
  const updates = [u0, u1];

  const clipper = clipByGlobalNorm(5.0);
  const state = clipper.init(params);
  const [clipped, newState] = clipper.update(updates, state);
  tree.dispose(newState);

  expect(clipped[0]).toBeAllclose([[1.08, 1.44]], { atol: 0.01 });
  expect(clipped[1]).toBeAllclose([[1.8, 4.31]], { atol: 0.01 });
  tree.dispose(clipped);
});

test("clipByGlobalNorm: zero gradients", () => {
  using p0 = np.zeros([1]);
  using p1 = np.zeros([1]);
  const params = [p0, p1];
  using u0 = np.array([0.0]);
  using u1 = np.array([0.0]);
  const updates = [u0, u1];

  const clipper = clipByGlobalNorm(1.0);
  const state = clipper.init(params);
  const [clipped, newState] = clipper.update(updates, state);
  tree.dispose(newState);

  expect(clipped[0]).toBeAllclose([0.0]);
  expect(clipped[1]).toBeAllclose([0.0]);
  tree.dispose(clipped);
});
