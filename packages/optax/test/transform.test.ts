import { numpy as np, tree } from "@jax-js-nonconsuming/jax";
import {
  addDecayedWeights,
  scaleByLearningRate,
  scaleBySchedule,
  trace,
} from "@jax-js-nonconsuming/optax";
import { expect, test } from "vitest";

test("scaleByLearningRate function", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  const transform = scaleByLearningRate(0.1);
  const state = transform.init(params);

  const [newUpdates, newState] = transform.update(updates, state, params);
  tree.dispose(newState);

  // Should scale by -0.1 (negative learning rate)
  expect(newUpdates).toBeAllclose([-0.01, -0.02, -0.03]);
  tree.dispose(newUpdates);
});

test("addDecayedWeights function with scalar", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  const transform = addDecayedWeights({ weightDecay: 0.01 });
  const state = transform.init(params);

  const [newUpdates, newState] = transform.update(updates, state, params);
  tree.dispose(newState);

  // Should add weight decay: [0.1 + 0.01*1, 0.2 + 0.01*2, 0.3 + 0.01*3]
  expect(newUpdates).toBeAllclose([0.11, 0.22, 0.33]);
  tree.dispose(newUpdates);
});

test("addDecayedWeights function with schedule", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  // Weight decay schedule that increases: 0.01, 0.02, 0.03, ...
  const weightDecaySchedule = (step: number) => 0.01 * (step + 1);
  const transform = addDecayedWeights({ weightDecay: weightDecaySchedule });
  let state = transform.init(params);

  // First update (step 0): weight_decay = 0.01
  let [newUpdates, newState] = transform.update(updates, state, params);
  state = newState;
  expect(newUpdates).toBeAllclose([0.11, 0.22, 0.33]);
  tree.dispose(newUpdates);

  // Second update (step 1): weight_decay = 0.02
  [newUpdates, newState] = transform.update(updates, state, params);
  expect(newUpdates).toBeAllclose([0.12, 0.24, 0.36]);
  tree.dispose(newUpdates);
  tree.dispose(newState);
});

test("addDecayedWeights function with mask", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);
  // Mask: apply weight decay to first and third params only
  using mask = np.array([1.0, 0.0, 1.0]);

  const transform = addDecayedWeights({ weightDecay: 0.01, mask });
  const state = transform.init(params);

  const [newUpdates, newState] = transform.update(updates, state, params);
  tree.dispose(newState);

  // Should add weight decay only where mask = 1: [0.1 + 0.01*1, 0.2 + 0, 0.3 + 0.01*3]
  expect(newUpdates).toBeAllclose([0.11, 0.2, 0.33]);
  tree.dispose(newUpdates);
});

test("addDecayedWeights throws error when params is undefined", () => {
  using updates = np.array([0.1, 0.2, 0.3]);

  const transform = addDecayedWeights({ weightDecay: 0.01 });
  using initP = np.array([1.0, 2.0, 3.0]);
  const state = transform.init(initP);

  expect(() => {
    transform.update(updates, state, undefined);
  }).toThrow("addDecayedWeights requires params to be provided");
});

test("scaleBySchedule function with dynamic learning rate", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  // Learning rate starts at 1.0 and decreases by 10% each step
  const schedule = (step: number) => Math.pow(0.9, step);
  const transform = scaleBySchedule(schedule);
  let state = transform.init(params);

  // First update (step 0)
  let [newUpdates, newState] = transform.update(updates, state, params);
  state = newState;
  expect(newUpdates).toBeAllclose([0.1, 0.2, 0.3]); // 1.0 * updates
  tree.dispose(newUpdates);

  // Second update (step 1)
  [newUpdates, newState] = transform.update(updates, state, params);
  expect(newUpdates).toBeAllclose([0.09, 0.18, 0.27]); // 0.9 * updates
  tree.dispose(newUpdates);
  tree.dispose(newState);
});

test("trace transformation", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  const transform = trace({ decay: 0.9 });
  let state = transform.init(params);

  // First update: trace should be equal to updates
  let newUpdates: np.Array;
  [newUpdates, state] = transform.update(updates, state, params);
  expect(newUpdates).toBeAllclose([0.1, 0.2, 0.3]);
  tree.dispose(newUpdates);

  // Second update: trace = updates + 0.9 * prev_trace
  [newUpdates, state] = transform.update(updates, state, params);
  expect(newUpdates).toBeAllclose([0.19, 0.38, 0.57]); // 0.1 + 0.9*0.1, etc.
  tree.dispose(newUpdates);
  tree.dispose(state);
});

test("trace transformation with nesterov", () => {
  using params = np.array([1.0, 2.0, 3.0]);
  using updates = np.array([0.1, 0.2, 0.3]);

  const transform = trace({ decay: 0.9, nesterov: true });
  let state = transform.init(params);

  // First update with nesterov
  let newUpdates: np.Array;
  [newUpdates, state] = transform.update(updates, state, params);
  expect(newUpdates).toBeAllclose([0.19, 0.38, 0.57]); // g + decay * (g + decay * 0)
  tree.dispose(newUpdates);

  // Second update with nesterov
  [newUpdates, state] = transform.update(updates, state, params);
  expect(newUpdates.shape).toEqual([3]);
  expect(newUpdates.dtype).toEqual(np.float32);
  tree.dispose(newUpdates);
  tree.dispose(state);
});
