import { jit, lax, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

export const width = 1000;
export const height = 800;

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

const mandelbrotMultiple = (iters: number) =>
  jit((A: np.Array, B: np.Array, V: np.Array, X: np.Array, Y: np.Array) => {
    for (let i = 0; i < iters; i++) {
      [A, B, V] = mandelbrotIteration(A, B, V, X, Y);
    }
    X.dispose();
    Y.dispose();
    return [A, B, V];
  });

export function calculateMandelbrot(iters: number): np.Array {
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

export function calculateMandelbrotJitLoop(iters: number): np.Array {
  using x = np.linspace(-2, 0.5, width);
  using y = np.linspace(-1, 1, height);

  const [X, Y] = np.meshgrid([x, y]);

  using f = mandelbrotMultiple(iters);

  using A = np.zeros(X.shape);
  using B = np.zeros(Y.shape);
  using V = np.zeros(X.shape);
  const [_A2, _B2, V2] = f(A, B, V, X, Y);
  X.dispose();
  Y.dispose();
  _A2.dispose();
  _B2.dispose();

  return V2;
}

export function calculateMandelbrotScan(iters: number): np.Array {
  using x = np.linspace(-2, 0.5, width);
  using y = np.linspace(-1, 1, height);

  const [X, Y] = np.meshgrid([x, y]);

  // Use lax.scan with Y=null (no output stacking needed)
  using f = jit(
    (
      A: np.Array,
      B: np.Array,
      V: np.Array,
      X: np.Array,
      Y: np.Array,
    ): [np.Array, np.Array, np.Array] => {
      type Carry = { A: np.Array; B: np.Array; V: np.Array };

      const step = (carry: Carry, _x: null): [Carry, null] => {
        const { A, B, V } = carry;
        using Asq = A.mul(A);
        using Bsq = B.mul(B);
        const newV = V.add(Asq.add(Bsq).less(100).astype(np.float32));
        const newA = np.clip(Asq.sub(Bsq).add(X), -50, 50);
        const newB = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
        return [{ A: newA, B: newB, V: newV }, null];
      };

      const init: Carry = { A, B, V };
      const [final, _ys] = lax.scan(step, init, null, { length: iters });
      X.dispose();
      Y.dispose();
      return [final.A, final.B, final.V];
    },
  );

  using A = np.zeros(X.shape);
  using B = np.zeros(Y.shape);
  using V = np.zeros(X.shape);
  const [_A2, _B2, V2] = f(A, B, V, X, Y);
  X.dispose();
  Y.dispose();
  _A2.dispose();
  _B2.dispose();

  return V2;
}
