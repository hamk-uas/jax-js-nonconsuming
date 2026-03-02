// Tests for functions in `jax.scipySpecial`.

import {
  defaultDevice,
  devices,
  grad,
  init,
  numpy as np,
  scipySpecial as special,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.scipySpecial.erf()", () => {
    const erfValues = [
      [-10, -5, -1, -0.5, 0, 0.001, 0.1, 1, 5, 10],
      [
        // Reference was computed in float64 for comparison accuracy.
        -1, -1, -0.84270079, -0.52049988, 0, 0.00112838, 0.11246292, 0.84270079,
        1, 1,
      ],
    ];

    test("erf values", () => {
      using x = np.array(erfValues[0]);
      using y = special.erf(x);
      expect(y).toBeAllclose(erfValues[1], { atol: 1e-6 });
    });

    test("erfc values", () => {
      using x = np.array(erfValues[0]);
      using y = special.erfc(x);
      const expected = erfValues[1].map((v) => 1 - v);
      expect(y).toBeAllclose(expected, { atol: 1e-6 });
    });

    test("erf derivative", () => {
      using x = np.linspace(-3, 3, 10);
      using dy = vmap(grad(special.erf))(x);
      using xSq = np.square(x);
      using negSq = np.negative(xSq);
      using expNegSq = np.exp(negSq);
      using expected = np.multiply(2 / Math.sqrt(Math.PI), expNegSq);
      expect(dy).toBeAllclose(expected);
    });

    test("erfc derivative", () => {
      using x = np.linspace(-3, 3, 10);
      using dy = vmap(grad(special.erfc))(x);
      using xSq = np.square(x);
      using negSq = np.negative(xSq);
      using expNegSq = np.exp(negSq);
      using expected = np.multiply(-2 / Math.sqrt(Math.PI), expNegSq);
      expect(dy).toBeAllclose(expected);
    });
  });

  suite("jax.scipySpecial.logit()", () => {
    test("logit values", () => {
      using x = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]);
      using y = special.logit(x);
      const expected = [
        -4.59511985, -2.19722458, -1.09861229, 0, 1.09861229, 2.19722458,
        4.59511985,
      ];
      expect(y).toBeAllclose(expected);
    });
  });
});
