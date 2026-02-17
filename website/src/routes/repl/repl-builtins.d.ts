import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

declare global {
  function displayImage(param: np.Array): Promise<void>;
}

export {};
