import { numpy as np } from "@jax-js-nonconsuming/jax";
import { expect, test } from "vitest";

import { treeMax, treeNorm, treeSum } from "../src/treeUtils";

test("treeSum: sums all elements across arrays", () => {
  using a = np.array([1, 2, 3]);
  using b = np.array([4, 5]);
  const tr = [a, b];
  using result = treeSum(tr);
  expect(result).toBeAllclose(15);
});

test("treeSum: handles empty tree", () => {
  const tr: np.Array[] = [];
  using result = treeSum(tr);
  expect(result).toBeAllclose(0);
});

test("treeSum: handles nested tree structure", () => {
  using a = np.array([1, 2]);
  using c = np.array([3, 4, 5]);
  const tr = { a, b: { c } };
  using result = treeSum(tr);
  expect(result).toBeAllclose(15);
});

test("treeMax: finds max across arrays", () => {
  using a = np.array([1, 2, 3]);
  using b = np.array([4, 5]);
  const tr = [a, b];
  using result = treeMax(tr);
  expect(result).toBeAllclose(5);
});

test("treeMax: handles negative numbers", () => {
  using a = np.array([-5, -2, -10]);
  using b = np.array([-1]);
  const tr = [a, b];
  using result = treeMax(tr);
  expect(result).toBeAllclose(-1);
});

test("treeMax: handles empty tree", () => {
  const tr: np.Array[] = [];
  using result = treeMax(tr);
  expect(result).toBeAllclose(-Infinity);
});

test("treeNorm: L2 norm (default)", () => {
  using a = np.array([3]);
  using b = np.array([4]);
  const tr = [a, b];
  using result = treeNorm(tr);
  expect(result).toBeAllclose(5);
});

test("treeNorm: L2 norm squared", () => {
  using a = np.array([3]);
  using b = np.array([4]);
  const tr = [a, b];
  using result = treeNorm(tr, 2, true);
  expect(result).toBeAllclose(25);
});

test("treeNorm: L1 norm", () => {
  using a = np.array([-3, 4]);
  using b = np.array([-5]);
  const tr = [a, b];
  using result = treeNorm(tr, 1);
  expect(result).toBeAllclose(12);
});

test("treeNorm: inf norm", () => {
  using a = np.array([-3, 4]);
  using b = np.array([-10]);
  const tr = [a, b];
  using result = treeNorm(tr, "inf");
  expect(result).toBeAllclose(10);
});

test("treeNorm: throws on unsupported ord", () => {
  using a = np.array([1, 2, 3]);
  const tr = [a];
  expect(() => treeNorm(tr, 3)).toThrow("Unsupported ord: 3");
});
