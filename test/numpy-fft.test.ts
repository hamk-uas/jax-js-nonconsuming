import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

import { deviceSuite } from "./device-suite.js";

await deviceSuite(() => {
  suite("jax.numpy.fft.fft()", () => {
    test("computes FFT of a simple real input", async () => {
      using real = np.array([0, 1, 2, 3]);
      using imag = np.array([0, 0, 0, 0]);
      const result = np.fft.fft({ real, imag });
      using rReal = result.real;
      using rImag = result.imag;
      expect(await rReal.jsAsync()).toBeAllclose([6, -2, -2, -2]);
      expect(await rImag.jsAsync()).toBeAllclose([0, 2, 0, -2]);
    });
  });

  suite("jax.numpy.fft.ifft()", () => {
    test("computes IFFT of a simple complex input", async () => {
      using real = np.array([6, -2, -2, -2]);
      using imag = np.array([0, 2, 0, -2]);
      const result = np.fft.ifft({ real, imag });
      using rReal = result.real;
      using rImag = result.imag;
      expect(await rReal.jsAsync()).toBeAllclose([0, 1, 2, 3]);
      expect(await rImag.jsAsync()).toBeAllclose([0, 0, 0, 0]);
    });

    test("FFT followed by IFFT returns original input", async () => {
      using real = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
      using imag = np.array([-5, 9, 0, 3, -1, 4, 2, 8]);
      const fftResult = np.fft.fft({ real: real, imag: imag });
      using _fftReal = fftResult.real;
      using _fftImag = fftResult.imag;
      const ifftResult = np.fft.ifft(fftResult);
      using ifftReal = ifftResult.real;
      using ifftImag = ifftResult.imag;
      expect(await ifftReal.jsAsync()).toBeAllclose(real, {
        atol: 1e-5,
      });
      expect(await ifftImag.jsAsync()).toBeAllclose(imag, {
        atol: 1e-5,
      });
    });
  });
});
