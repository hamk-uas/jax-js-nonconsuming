import "vitest";
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

interface CustomMatchers<R = unknown> {
  toBeAllclose: (
    expected: Parameters<typeof np.array>[0],
    options: { rtol?: number; atol?: number; equalNaN?: boolean } = {},
  ) => R;
  toBeWithinRange: (min: number, max: number) => R;
}

declare module "vitest" {
  interface Assertion<T = any> extends CustomMatchers<T> {}
  interface AsymmetricMatchersContaining extends CustomMatchers {}
}
