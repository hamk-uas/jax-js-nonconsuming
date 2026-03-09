/**
 * Profile WebGPU assocScan with DLM-like 5-tuple compose body.
 * Run: pnpm vitest run test/bench-assoc-scale.test.ts
 */
import {
  blockUntilReady,
  defaultDevice,
  DType,
  init,
  jit,
  lax,
  numpy as np,
  setDebug,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { expect, test } from "vitest";

// Simple 2-tuple compose (baseline)
const compose2 = (
  p: { A: np.Array; b: np.Array },
  q: { A: np.Array; b: np.Array },
) => {
  using tmp = np.matmul(q.A, p.b) as np.Array;
  return {
    A: np.matmul(q.A, p.A) as np.Array,
    b: tmp.add(q.b) as np.Array,
  };
};

// DLM-like 5-tuple forward compose with inv (mirrors dlm-js composeForward)
// Uses einsum — identical to compose5 (fast path lowers einsum to matmul/swapaxes)
const _compose5_einsum = (
  a: { A: np.Array; b: np.Array; C: np.Array; eta: np.Array; J: np.Array },
  b_elem: { A: np.Array; b: np.Array; C: np.Array; eta: np.Array; J: np.Array },
) => {
  // M = inv(I + C_i * J_j)
  using CiJj = np.einsum("nij,njk->nik", a.C, b_elem.J) as np.Array;
  using I_eye = np.eye(2, { dtype: DType.Float32 });
  using I1 = np.reshape(I_eye, [1, 2, 2]);
  using X = np.add(I1, CiJj) as np.Array;
  using M = np.linalg.inv(X);

  // A_ij = A_j M A_i
  using AjM = np.einsum("nij,njk->nik", b_elem.A, M) as np.Array;
  const A_comp = np.einsum("nij,njk->nik", AjM, a.A) as np.Array;

  // b_ij = A_j M (b_i + C_i eta_j) + b_j
  using CiEtaj = np.einsum("nij,njk->nik", a.C, b_elem.eta) as np.Array;
  using bi_plus = np.add(a.b, CiEtaj) as np.Array;
  using AjM_b = np.einsum("nij,njk->nik", AjM, bi_plus) as np.Array;
  const b_comp = np.add(AjM_b, b_elem.b) as np.Array;

  // C_ij = A_j M C_i A_j' + C_j
  using AjMCi = np.einsum("nij,njk->nik", AjM, a.C) as np.Array;
  using AjT = np.einsum("nij->nji", b_elem.A) as np.Array;
  using C_tmp = np.einsum("nij,njk->nik", AjMCi, AjT) as np.Array;
  const C_comp = np.add(C_tmp, b_elem.C) as np.Array;

  // eta_ij = A_i' N (eta_j - J_j b_i) + eta_i  (N = I - J_j M C_i)
  using MCi = np.einsum("nij,njk->nik", M, a.C) as np.Array;
  using JjMCi = np.einsum("nij,njk->nik", b_elem.J, MCi) as np.Array;
  using N = np.subtract(I1, JjMCi) as np.Array;
  using Jjbi = np.einsum("nij,njk->nik", b_elem.J, a.b) as np.Array;
  using eta_diff = np.subtract(b_elem.eta, Jjbi) as np.Array;
  using N_eta = np.einsum("nij,njk->nik", N, eta_diff) as np.Array;
  using AtNeta = np.einsum("nji,njk->nik", a.A, N_eta) as np.Array;
  const eta_comp = np.add(AtNeta, a.eta) as np.Array;

  // J_ij = A_i' N J_j A_i + J_i
  using NJ = np.einsum("nij,njk->nik", N, b_elem.J) as np.Array;
  using NJAi = np.einsum("nij,njk->nik", NJ, a.A) as np.Array;
  using AtNJAi = np.einsum("nji,njk->nik", a.A, NJAi) as np.Array;
  const J_comp = np.add(AtNJAi, a.J) as np.Array;

  return { A: A_comp, b: b_comp, C: C_comp, eta: eta_comp, J: J_comp };
};

// Alternative: uses np.matmul / np.swapaxes directly (equivalent to einsum fast path)
const compose5 = (
  a: { A: np.Array; b: np.Array; C: np.Array; eta: np.Array; J: np.Array },
  b_elem: { A: np.Array; b: np.Array; C: np.Array; eta: np.Array; J: np.Array },
) => {
  // M = inv(I + C_i * J_j)
  using CiJj = np.matmul(a.C, b_elem.J) as np.Array;
  using I_eye = np.eye(2, { dtype: DType.Float32 });
  using X = np.add(I_eye, CiJj) as np.Array;
  using M = np.linalg.inv(X);

  // A_ij = A_j M A_i
  using AjM = np.matmul(b_elem.A, M) as np.Array;
  const A_comp = np.matmul(AjM, a.A) as np.Array;

  // b_ij = A_j M (b_i + C_i eta_j) + b_j
  using CiEtaj = np.matmul(a.C, b_elem.eta) as np.Array;
  using bi_plus = np.add(a.b, CiEtaj) as np.Array;
  using AjM_b = np.matmul(AjM, bi_plus) as np.Array;
  const b_comp = np.add(AjM_b, b_elem.b) as np.Array;

  // C_ij = A_j M C_i A_j' + C_j
  using AjMCi = np.matmul(AjM, a.C) as np.Array;
  using AjT = np.swapaxes(b_elem.A, -2, -1) as np.Array;
  using C_tmp = np.matmul(AjMCi, AjT) as np.Array;
  const C_comp = np.add(C_tmp, b_elem.C) as np.Array;

  // eta_ij = A_i' N (eta_j - J_j b_i) + eta_i  (N = I - J_j M C_i)
  using MCi = np.matmul(M, a.C) as np.Array;
  using JjMCi = np.matmul(b_elem.J, MCi) as np.Array;
  using N = np.subtract(I_eye, JjMCi) as np.Array;
  using Jjbi = np.matmul(b_elem.J, a.b) as np.Array;
  using eta_diff = np.subtract(b_elem.eta, Jjbi) as np.Array;
  using N_eta = np.matmul(N, eta_diff) as np.Array;
  using AiT = np.swapaxes(a.A, -2, -1) as np.Array;
  using AtNeta = np.matmul(AiT, N_eta) as np.Array;
  const eta_comp = np.add(AtNeta, a.eta) as np.Array;

  // J_ij = A_i' N J_j A_i + J_i
  using NJ = np.matmul(N, b_elem.J) as np.Array;
  using NJAi = np.matmul(NJ, a.A) as np.Array;
  using AtNJAi = np.matmul(AiT, NJAi) as np.Array;
  const J_comp = np.add(AtNJAi, a.J) as np.Array;

  return { A: A_comp, b: b_comp, C: C_comp, eta: eta_comp, J: J_comp };
};

test("DLM 5-tuple compose scaling", async () => {
  const devices = await init("wasm", "webgpu");
  const hasGPU = devices.includes("webgpu");

  const Ns = [100, 800, 3200, 12800];

  // WASM
  console.log("\n=== WASM 5-tuple ===");
  defaultDevice("wasm");
  for (const N of Ns) {
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    const C = np.full([N, 2, 2], 0.5, { dtype: DType.Float32 });
    const eta = np.full([N, 2, 1], 0.0, { dtype: DType.Float32 });
    const J = np.full([N, 2, 2], 0.1, { dtype: DType.Float32 });

    const fn = jit(
      (AA: np.Array, bb: np.Array, CC: np.Array, ee: np.Array, JJ: np.Array) =>
        lax.associativeScan(compose5, { A: AA, b: bb, C: CC, eta: ee, J: JJ }),
    );
    tree.dispose(fn(A, b, C, eta, J));
    const ITERS = 3;
    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) tree.dispose(fn(A, b, C, eta, J));
    console.log(
      `  N=${String(N).padStart(6)}: ${((performance.now() - t0) / ITERS).toFixed(2)} ms`,
    );
    A.dispose();
    b.dispose();
    C.dispose();
    eta.dispose();
    J.dispose();
    fn.dispose();
  }

  if (!hasGPU) {
    console.log("no webgpu");
    return;
  }

  // WebGPU
  console.log("\n=== WebGPU 5-tuple (GPU-synced) ===");
  defaultDevice("webgpu");
  for (const N of Ns) {
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    const C = np.full([N, 2, 2], 0.5, { dtype: DType.Float32 });
    const eta = np.full([N, 2, 1], 0.0, { dtype: DType.Float32 });
    const J = np.full([N, 2, 2], 0.1, { dtype: DType.Float32 });
    await blockUntilReady(A);

    const fn = jit(
      (AA: np.Array, bb: np.Array, CC: np.Array, ee: np.Array, JJ: np.Array) =>
        lax.associativeScan(compose5, { A: AA, b: bb, C: CC, eta: ee, J: JJ }),
    );
    if (N === Ns[0]) setDebug(1);
    await tree.consumeData(fn(A, b, C, eta, J) as any);
    if (N === Ns[0]) setDebug(0);
    const ITERS = 3;
    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) {
      await tree.consumeData(fn(A, b, C, eta, J) as any);
    }
    console.log(
      `  N=${String(N).padStart(6)}: ${((performance.now() - t0) / ITERS).toFixed(2)} ms`,
    );
    A.dispose();
    b.dispose();
    C.dispose();
    eta.dispose();
    J.dispose();
    fn.dispose();
  }

  expect(true).toBe(true);
}, 120_000);

test("assocScan AND lax.scan scaling (GPU-synced)", async () => {
  const devices = await init("wasm", "webgpu");
  const hasGPU = devices.includes("webgpu");

  const Ns = [100, 800, 3200, 12800, 25600];

  // === WASM assocScan ===
  console.log("\n=== WASM assocScan ===");
  defaultDevice("wasm");
  for (const N of Ns) {
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    const fn = jit((AA: np.Array, bb: np.Array) =>
      lax.associativeScan(compose2, { A: AA, b: bb }),
    );
    tree.dispose(fn(A, b));
    const ITERS = 3;
    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) tree.dispose(fn(A, b));
    console.log(
      `  N=${String(N).padStart(6)}: ${((performance.now() - t0) / ITERS).toFixed(2)} ms`,
    );
    A.dispose();
    b.dispose();
    fn.dispose();
  }

  // === WASM lax.scan (sequential) ===
  console.log("\n=== WASM lax.scan ===");
  for (const N of Ns) {
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    const fn = jit((AA: np.Array, bb: np.Array) => {
      const [carry, ys] = lax.scan(
        (c: { A: np.Array; b: np.Array }, x: { A: np.Array; b: np.Array }) => {
          const nc = compose2(c, x);
          return [nc, nc];
        },
        { A: np.eye(2, { dtype: DType.Float32 }), b: np.zeros([2, 1]) },
        { A: AA, b: bb },
      );
      tree.dispose(carry);
      return ys;
    });
    tree.dispose(fn(A, b));
    const ITERS = 3;
    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) tree.dispose(fn(A, b));
    console.log(
      `  N=${String(N).padStart(6)}: ${((performance.now() - t0) / ITERS).toFixed(2)} ms`,
    );
    A.dispose();
    b.dispose();
    fn.dispose();
  }

  if (!hasGPU) {
    console.log("no webgpu");
    return;
  }

  // === WebGPU assocScan (GPU-synced via consumeData) ===
  console.log("\n=== WebGPU assocScan (GPU-synced) ===");
  defaultDevice("webgpu");
  for (const N of Ns) {
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    await blockUntilReady(A);
    await blockUntilReady(b);
    const fn = jit((AA: np.Array, bb: np.Array) =>
      lax.associativeScan(compose2, { A: AA, b: bb }),
    );
    if (N === Ns[0]) setDebug(1);
    await tree.consumeData(fn(A, b) as any);
    if (N === Ns[0]) setDebug(0);
    const ITERS = 3;
    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) {
      await tree.consumeData(fn(A, b) as any);
    }
    console.log(
      `  N=${String(N).padStart(6)}: ${((performance.now() - t0) / ITERS).toFixed(2)} ms`,
    );
    A.dispose();
    b.dispose();
    fn.dispose();
  }

  // === WebGPU lax.scan (GPU-synced via consumeData) ===
  console.log("\n=== WebGPU lax.scan (GPU-synced) ===");
  for (const N of Ns) {
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    await blockUntilReady(A);
    await blockUntilReady(b);
    const fn = jit((AA: np.Array, bb: np.Array) => {
      const [carry, ys] = lax.scan(
        (c: { A: np.Array; b: np.Array }, x: { A: np.Array; b: np.Array }) => {
          const nc = compose2(c, x);
          return [nc, nc];
        },
        { A: np.eye(2, { dtype: DType.Float32 }), b: np.zeros([2, 1]) },
        { A: AA, b: bb },
      );
      tree.dispose(carry);
      return ys;
    });
    if (N === Ns[0]) setDebug(1);
    await tree.consumeData(fn(A, b) as any);
    if (N === Ns[0]) setDebug(0);
    const ITERS = 3;
    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) {
      await tree.consumeData(fn(A, b) as any);
    }
    console.log(
      `  N=${String(N).padStart(6)}: ${((performance.now() - t0) / ITERS).toFixed(2)} ms`,
    );
    A.dispose();
    b.dispose();
    fn.dispose();
  }

  expect(true).toBe(true);
}, 120_000);
