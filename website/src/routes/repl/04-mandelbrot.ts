import { jit, lax, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

// Mandelbrot set using lax.scan — the entire iteration loop compiles to native code.
const width = 750;
const height = 600;

using x = np.linspace(-2, 0.5, width);
using y = np.linspace(-1, 1, height);
const [X, Y] = np.meshgrid([x, y]);

type Carry = { A: np.Array; B: np.Array; V: np.Array };

using f = jit(
  (A: np.Array, B: np.Array, V: np.Array, X: np.Array, Y: np.Array) => {
    const step = (carry: Carry, _x: null): [Carry, null] => {
      const { A, B, V } = carry;
      using Asq = A.mul(A);
      using Bsq = B.mul(B);
      using magSq = Asq.add(Bsq);
      using mask = magSq.less(100).astype(np.float32);
      const newV = V.add(mask);
      using diffSq = Asq.sub(Bsq);
      using realShifted = diffSq.add(X);
      const newA = np.clip(realShifted, -50, 50);
      using cross = A.mul(B);
      using crossScaled = cross.mul(2);
      using imagShifted = crossScaled.add(Y);
      const newB = np.clip(imagShifted, -50, 50);
      return [{ A: newA, B: newB, V: newV }, null];
    };

    const init: Carry = { A, B, V };
    const [final, _ys] = lax.scan(step, init, null, { length: 100 });
    final.A.dispose();
    final.B.dispose();
    return final.V;
  },
);

using A = np.zeros(X.shape);
using B = np.zeros(Y.shape);
using V = np.zeros(X.shape);
using result = f(A, B, V, X, Y);
X.dispose();
Y.dispose();

using scaled = result.div(100);
using image = np.subtract(1, scaled);

// The REPL has a displayImage() builtin for drawing image pixels.
await displayImage(image);
