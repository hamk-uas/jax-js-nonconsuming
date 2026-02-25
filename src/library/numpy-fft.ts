// Port of the `jax.numpy.fft` module, Fast Fourier Transform.

import { arange, Array, concatenate, cos, sin } from "./numpy";
import { isFloatDtype } from "../alu";
import { jit } from "../frontend/jaxpr";
import { checkAxis, deepEqual, invertPermutation, range, rep } from "../utils";

/**
 * A pair of arrays representing real and imaginary part `a + bj`. Both arrays
 * must have the same shape.
 */
export type ComplexPair = {
  real: Array;
  imag: Array;
};

function checkPairInput(name: string, a: ComplexPair) {
  const fullName = `jax.numpy.fft.${name}`;
  if (!deepEqual(a.real.shape, a.imag.shape)) {
    throw new Error(
      `${fullName}: real and imaginary parts must have the same shape, got ${JSON.stringify(a.real.shape)} and ${JSON.stringify(a.imag.shape)}`,
    );
  }
  if (a.real.dtype !== a.imag.dtype) {
    throw new Error(
      `${fullName}: real and imaginary parts must have the same dtype, got ${a.real.dtype} and ${a.imag.dtype}`,
    );
  }
  if (!isFloatDtype(a.real.dtype)) {
    throw new Error(
      `${fullName}: input must have a float dtype, got ${a.real.dtype}`,
    );
  }
}

function checkPowerOfTwo(name: string, n: number) {
  if ((n & (n - 1)) !== 0) {
    throw new Error(
      `jax.numpy.fft.${name}: size must be a power of two, got ${n}`,
    );
  }
}

const fftUpdate = jit(
  function fftUpdate(i: number, { real, imag }: ComplexPair): ComplexPair {
    const half = 2 ** i;

    real = real.reshape([-1, 2 * half]);
    imag = imag.reshape([-1, 2 * half]);

    const k = arange(0, half, 1, { dtype: real.dtype });
    using theta = k.mul(-Math.PI / half);
    const wr = cos(theta);
    const wi = sin(theta);

    const ur = real.slice([], [0, half]);
    const ui = imag.slice([], [0, half]);
    const vr = real.slice([], [half, 2 * half]);
    const vi = imag.slice([], [half, 2 * half]);

    // t = w * v
    using vrWr = vr.mul(wr);
    using viWi = vi.mul(wi);
    const tr = vrWr.sub(viWi);
    using vrWi = vr.mul(wi);
    using viWr = vi.mul(wr);
    const ti = vrWi.add(viWr);

    // store [u + t, u - t]
    return {
      real: concatenate([ur.add(tr), ur.sub(tr)], -1),
      imag: concatenate([ui.add(ti), ui.sub(ti)], -1),
    };
  },
  { staticArgnums: [0] },
);

/**
 * Compute a one-dimensional discrete Fourier transform.
 *
 * Currently, the size of the axis must be a power of two.
 */
export function fft(a: ComplexPair, axis: number = -1): ComplexPair {
  checkPairInput("fft", a);
  let { real, imag } = a;
  axis = checkAxis(axis, real.ndim);
  const n = real.shape[axis];
  checkPowerOfTwo("fft", n);
  const logN = Math.log2(n);

  // Track all intermediates; dispose everything except the returned pair.
  const d: Array[] = [];
  try {
    // If axis is not at the end, move it to the end
    let perm: number[] | null = null;
    if (axis !== real.ndim - 1) {
      perm = range(real.ndim);
      perm.splice(axis, 1);
      perm.push(axis);
      real = real.transpose(perm);
      d.push(real);
      imag = imag.transpose(perm);
      d.push(imag);
    }

    // Cooley-Tukey FFT (radix-2) — bit-reversal permutation
    const originalShape = real.shape;
    const realR = real.reshape([-1, ...rep(logN, 2)]);
    d.push(realR);
    const realT = realR.transpose([0, ...range(1, logN + 1).reverse()]);
    d.push(realT);
    real = realT.flatten();
    d.push(real);

    const imagR = imag.reshape([-1, ...rep(logN, 2)]);
    d.push(imagR);
    const imagT = imagR.transpose([0, ...range(1, logN + 1).reverse()]);
    d.push(imagT);
    imag = imagT.flatten();
    d.push(imag);

    // Butterfly passes
    for (let i = 0; i < logN; i++) {
      ({ real, imag } = fftUpdate(i, { real, imag }));
      d.push(real, imag);
    }
    real = real.reshape(originalShape);
    d.push(real);
    imag = imag.reshape(originalShape);
    d.push(imag);

    // If axis was moved, move it back
    if (perm !== null) {
      real = real.transpose(invertPermutation(perm));
      d.push(real);
      imag = imag.transpose(invertPermutation(perm));
      d.push(imag);
    }
    return { real, imag };
  } finally {
    for (const v of d) {
      if (v !== real && v !== imag) v[Symbol.dispose]();
    }
  }
}

/**
 * Compute a one-dimensional inverse discrete Fourier transform.
 *
 * Currently, the size of the axis must be a power of two.
 */
export function ifft(a: ComplexPair, axis: number = -1): ComplexPair {
  checkPairInput("ifft", a);
  const { real, imag } = a;
  axis = checkAxis(axis, real.ndim);
  const n = real.shape[axis];
  checkPowerOfTwo("ifft", n);

  // ifft(a) = 1/n * conj(fft(conj(a)))
  using negImag = imag.mul(-1);
  const result = fft({ real, imag: negImag }, axis);
  using fftReal = result.real;
  using fftImag = result.imag;
  using negFftImag = fftImag.mul(-1);
  return {
    real: fftReal.div(n),
    imag: negFftImag.div(n),
  };
}
