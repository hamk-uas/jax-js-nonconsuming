import { numpy as np, tree } from "@hamk-uas/jax-js-nonconsuming";
import { chain, scale } from "@hamk-uas/jax-js-nonconsuming-optax";
import { expect, test } from "vitest";

test("chain function combines transformations", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  // Chain two simple transformations
  const combined = chain(scale(2.0), scale(0.5));
  const state = combined.init(params);

  const [newUpdates, newState] = combined.update(updates, state, params);
  tree.dispose(newState);

  // 2.0 * 0.5 = 1.0, so updates should be unchanged
  expect(newUpdates).toBeAllclose([0.1, 0.2, 0.3]);
  tree.dispose(newUpdates);
});
