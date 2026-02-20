import * as lax from "./lax";
import { triangularSolve } from "./lax-linalg";
import * as np from "./numpy";
import { Array, ArrayLike, fudgeArray } from "../frontend/array";
import { generalBroadcast } from "../utils";

/**
 * Apply a pre-computed LU factorization to solve A x = b.
 *
 * @param lu - Combined L/U matrix from `lax.linalg.lu(A)`.
 * @param P  - Permutation matrix derived from the LU permutation vector.
 * @param b  - Right-hand side to solve.
 * @param d  - Disposal tracker for intermediate arrays.
 * @returns Solution x; caller must push to `d` if not the final return.
 */
function luSolveWithP(lu: Array, P: Array, b: Array, d: Array[]): Array {
  const Pb = np.matmul(P, b);
  d.push(Pb);
  const LPb = triangularSolve(lu, Pb, {
    leftSide: true,
    lower: true,
    unitDiagonal: true,
  });
  d.push(LPb);
  return triangularSolve(lu, LPb, { leftSide: true, lower: false });
}

function checkSquare(name: string, a: Array) {
  if (a.ndim < 2 || a.shape[a.ndim - 1] !== a.shape[a.ndim - 2]) {
    throw new Error(
      `${name}: input must be at least 2D square matrix, got ${a.aval}`,
    );
  }
  return a.shape[a.ndim - 1];
}

/**
 * Compute the Cholesky decomposition of a (batched) positive-definite matrix.
 *
 * This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
 * the input matrix, which is on by default.
 */
export function cholesky(
  a: ArrayLike,
  {
    upper = false,
    symmetrizeInput = true,
  }: {
    upper?: boolean;
    symmetrizeInput?: boolean;
  } = {},
): Array {
  a = fudgeArray(a);
  checkSquare("cholesky", a);
  if (symmetrizeInput) {
    using at = np.matrixTranspose(a);
    using sum = a.add(at);
    using sym = sum.mul(0.5);
    return lax.linalg.cholesky(sym, { upper });
  }
  return lax.linalg.cholesky(a, { upper });
}

/** Compute the determinant of a square matrix (batched). */
export function det(a: ArrayLike): Array {
  a = fudgeArray(a);
  const n = checkSquare("det", a);
  const luResult = lax.linalg.lu(a);
  using lu = luResult[0];
  using pivots = luResult[1];
  using _perm = luResult[2];
  using indices = np.arange(n);
  using neq = pivots.notEqual(indices);
  using neqInt = neq.astype(np.int32);
  using sumAxis = neqInt.sum(-1);
  using parity = sumAxis.mod(2);
  using neg2 = parity.mul(-2);
  using sign = neg2.add(1); // (-1)^parity
  using diag = lu.diagonal(0, -1, -2);
  using prod = np.prod(diag, -1);
  return prod.mul(sign);
}

export { diagonal } from "./numpy";

/** Compute the inverse of a square matrix (batched). */
export function inv(a: ArrayLike): Array {
  a = fudgeArray(a);
  const n = checkSquare("inv", a);
  using eye = np.eye(n, { dtype: a.dtype });
  return solve(a, eye);
}

/**
 * Return the least-squares solution to a linear equation.
 *
 * For overdetermined systems, this finds the `x` that minimizes `norm(ax - b)`.
 * For underdetermined systems, this finds the minimum-norm solution for `x`.
 *
 * This currently uses Cholesky decomposition to solve the normal equations,
 * under the hood. The method is not as robust as QR or SVD.
 *
 * @param a coefficient matrix of shape `(M, N)`
 * @param b right-hand side of shape `(M,)` or `(M, K)`
 * @return least-squares solution of shape `(N,)` or `(N, K)`
 */
export function lstsq(a: ArrayLike, b: ArrayLike): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (a.ndim !== 2)
    throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
  const [m, n] = a.shape;
  if (b.shape[0] !== m)
    throw new Error(
      `lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`,
    );
  using at = np.matrixTranspose(a);
  if (m <= n) {
    // Underdetermined or square system: A.T @ (A @ A.T)^-1 @ B
    using aat = np.matmul(a, at); // A @ A.T, shape (M, M)
    using l = cholesky(aat, { symmetrizeInput: false }); // L @ L.T = A @ A.T
    using lb = triangularSolve(l, b, { leftSide: true, lower: true }); // L^-1 @ B
    using llb = triangularSolve(l, lb, {
      leftSide: true,
      lower: true,
      transposeA: true,
    }); // (A @ A.T)^-1 @ B
    return np.matmul(at, llb); // A.T @ (A @ A.T)^-1 @ B
  } else {
    // Overdetermined system: (A.T @ A)^-1 @ A.T @ B
    using ata = np.matmul(at, a); // A.T @ A, shape (N, N)
    using l = cholesky(ata, { symmetrizeInput: false }); // L @ L.T = A.T @ A
    using atb = np.matmul(at, b); // A.T @ B
    using lb = triangularSolve(l, atb, { leftSide: true, lower: true }); // L^-1 @ A.T @ B
    return triangularSolve(l, lb, {
      leftSide: true,
      lower: true,
      transposeA: true,
    }); // (A.T @ A)^-1 @ A.T @ B
  }
}

export { matmul } from "./numpy";

/** Raise a square matrix to an integer power, via repeated squarings. */
export function matrixPower(a: ArrayLike, n: number): Array {
  if (!Number.isInteger(n))
    throw new Error(`matrixPower: exponent must be an integer, got ${n}`);
  a = fudgeArray(a);
  const m = checkSquare("matrixPower", a);
  if (n === 0) {
    using eye = np.eye(m, { dtype: a.dtype });
    return np.broadcastTo(eye, a.shape);
  }
  const isInputOwned = n < 0;
  if (n < 0) {
    a = inv(a);
    n = -n;
  }
  let result: Array | null = null;
  let a2k = a; // a^(2^k)
  for (let k = 0; n; k++) {
    if (k > 0) {
      const prev = a2k;
      a2k = np.matmul(a2k, a2k);
      // Dispose old a2k unless it's the original input (not owned by us)
      if (prev !== a || isInputOwned) prev[Symbol.dispose]();
    }
    if (n % 2 === 1) {
      const prev = result;
      result = result === null ? a2k : np.matmul(result, a2k);
      if (prev !== null && prev !== a2k) prev[Symbol.dispose]();
    }
    n = Math.floor(n / 2);
  }
  // Dispose a2k if it's not the result and it's not the input
  if (a2k !== result && (a2k !== a || isInputOwned)) a2k[Symbol.dispose]();
  return result!;
}

export { matrixTranspose } from "./numpy";
export { outer } from "./numpy";

/** Return sign and natural logarithm of the determinant of `a`. */
export function slogdet(a: ArrayLike): [Array, Array] {
  a = fudgeArray(a);
  const n = checkSquare("slogdet", a);
  const luResult = lax.linalg.lu(a);
  using lu = luResult[0];
  using pivots = luResult[1];
  using _perm = luResult[2];
  using indices = np.arange(n);
  using neq = pivots.notEqual(indices);
  using neqInt = neq.astype(np.int32);
  using parityBase = neqInt.sum(-1);
  using diag = lu.diagonal(0, -1, -2);
  using diagNeg = diag.less(0);
  using diagNegInt = diagNeg.astype(np.int32);
  using diagNegSum = diagNegInt.sum(-1);
  using parityTotal = parityBase.add(diagNegSum);
  using parity = parityTotal.mod(2);
  using neg2 = parity.mul(-2);
  const sign = neg2.add(1); // (-1)^parity — returned, NOT using
  using absDiag = np.abs(diag);
  using logDiag = np.log(absDiag);
  const logabsdet = logDiag.sum(-1); // returned, NOT using
  return [sign, logabsdet];
}

/**
 * Solve a linear system of equations.
 *
 * This solves a (batched) linear system of equations `a @ x = b` for `x` given
 * `a` and `b`. If `a` is singular, this will return `nan` or `inf` values.
 *
 * Gradient flows through the LU decomposition via the TriangularSolve JVP rule
 * (which correctly masks dA with triu()). Both ∂L/∂A and ∂L/∂b are supported.
 *
 * @param a - Coefficient matrix of shape `(..., N, N)`.
 * @param b - Values of shape `(N,)` or `(..., N, M)`.
 * @returns Solution `x` of shape `(..., N)` or `(..., N, M)`.
 */
export function solve(a: ArrayLike, b: ArrayLike): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  const n = checkSquare("solve", a);
  if (b.ndim === 0) throw new Error(`solve: b cannot be scalar`);
  const bIs1d = b.ndim === 1;
  const d: Array[] = [];
  try {
    if (bIs1d) {
      b = b.reshape([...b.shape, 1]);
      d.push(b);
    }
    if (b.shape[b.ndim - 2] !== n) {
      throw new Error(
        `solve: leading dimension of b must match size of a, got a=${a.aval}, b=${b.aval}`,
      );
    }
    const m = b.shape[b.ndim - 1];
    const batchDims = generalBroadcast(
      a.shape.slice(0, -2),
      b.shape.slice(0, -2),
    );
    const aTargetShape = [...batchDims, n, n];
    if (
      a.shape.length !== aTargetShape.length ||
      a.shape.some((dim, i) => dim !== aTargetShape[i])
    ) {
      a = np.broadcastTo(a, aTargetShape);
      d.push(a);
    }
    const bTargetShape = [...batchDims, n, m];
    if (
      b.shape.length !== bTargetShape.length ||
      b.shape.some((dim, i) => dim !== bTargetShape[i])
    ) {
      b = np.broadcastTo(b, bTargetShape);
      d.push(b);
    }

    // Factor A. Gradient flows freely through the LU JVP (TriSolve triu mask fixed).
    // Stop gradient only on permRaw — permutation is integer-valued, no gradient.
    // Do NOT stop gradient on 'a' itself — stopGradient(a) creates a fully-known
    // PETracer that PE disposal cascades through, which would free 'a' early.
    const [lu, pivotsRaw, permRaw] = lax.linalg.lu(a);
    d.push(lu, pivotsRaw, permRaw);
    const permutation = lax.stopGradient(permRaw);

    // Build permutation matrix P (derived from stopGradient'd factorization).
    const arangeN = np.arange(n);
    d.push(arangeN);
    const permR = permutation.reshape([...permutation.shape, 1]);
    d.push(permR);
    const eq = arangeN.equal(permR);
    d.push(eq);
    const P = eq.astype(b.dtype);
    d.push(P);

    // Solve x = A^{-1} b via the LU factorization.
    // Gradient flows through both b (TriSolve transpose rule) and A (TriSolve JVP).
    let x = luSolveWithP(lu, P, b, d);
    if (bIs1d) {
      d.push(x);
      x = np.squeeze(x, -1);
    }
    return x;
  } finally {
    for (const v of d) v[Symbol.dispose]();
  }
}

export { tensordot } from "./numpy";
export { trace } from "./numpy";
export { vecdot } from "./numpy";
