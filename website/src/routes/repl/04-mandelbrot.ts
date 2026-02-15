import { jit, numpy as np } from "@jax-js-nonconsuming/jax";

// This example draws a Mandelbrot fractal using array operations and jit() for each step.
const width = 750;
const height = 600;

function mandelbrotIteration(
  A: np.Array,
  B: np.Array,
  V: np.Array,
  X: np.Array,
  Y: np.Array,
) {
  using Asq = A.mul(A);
  using Bsq = B.mul(B);
  V = V.add(Asq.add(Bsq).less(100).astype(np.float32));
  const A2 = np.clip(Asq.sub(Bsq).add(X), -50, 50);
  const B2 = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
  return [A2, B2, V];
}

function calculateMandelbrot(iters: number) {
  using x = np.linspace(-2, 0.5, width);
  using y = np.linspace(-1, 1, height);

  const [X, Y] = np.meshgrid([x, y]);

  using f = jit(mandelbrotIteration);

  let A = np.zeros(X.shape);
  let B = np.zeros(Y.shape);
  let V = np.zeros(X.shape);
  for (let i = 0; i < iters; i++) {
    const [newA, newB, newV] = f(A, B, V, X, Y);
    A.dispose();
    B.dispose();
    V.dispose();
    A = newA;
    B = newB;
    V = newV;
  }
  X.dispose();
  Y.dispose();
  A.dispose();
  B.dispose();

  return V;
}

using ar = calculateMandelbrot(100);
using image = np.subtract(1, ar.div(100));

// The REPL has a displayImage() builtin for drawing image pixels.
await displayImage(image);
