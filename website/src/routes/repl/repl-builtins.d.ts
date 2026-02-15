import { numpy as np } from "@jax-js-nonconsuming/jax";

declare global {
  function displayImage(param: np.Array): Promise<void>;
}

export {};
