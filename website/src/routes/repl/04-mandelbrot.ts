import { jit, lax, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

declare function displayImage(image: np.Array): Promise<void>;

// Mandelbrot set using jit(foriLoop) — the compiler wraps the loop in a
// block_map on WebGPU, producing a single fused shader dispatch.
const width = 750;
const height = 600;

using x = np.linspace(-2, 0.5, width);
using y = np.linspace(-1, 1, height);
const [X, Y] = np.meshgrid([x, y]);

type Carry = { A: np.Array; B: np.Array; V: np.Array };

using f = jit(
  (
    A: np.Array,
    B: np.Array,
    V: np.Array,
    X: np.Array,
    Y: np.Array,
  ): np.Array => {
    const result = lax.foriLoop(
      0,
      100,
      (_i: np.Array, carry: Carry): Carry => {
        const { A, B, V } = carry;
        using Asq = A.mul(A);
        using Bsq = B.mul(B);
        using mask = Asq.add(Bsq).less(100).astype(np.float32);
        const newV = V.add(mask);
        const newA = np.clip(Asq.sub(Bsq).add(X), -50, 50);
        const newB = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
        return { A: newA, B: newB, V: newV };
      },
      { A, B, V },
    );
    result.A.dispose();
    result.B.dispose();
    return result.V;
  },
);

using A = np.zeros(X.shape);
using B = np.zeros(Y.shape);
using V = np.zeros(X.shape);
using d = new DisposableStack();
d.use(X);
d.use(Y);
using result = f(A, B, V, X, Y);

using scaled = result.div(100);
using image = np.subtract(1, scaled);

// The REPL has a displayImage() builtin for drawing image pixels.
await displayImage(image);
