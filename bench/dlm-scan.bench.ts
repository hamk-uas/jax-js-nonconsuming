/**
 * DLM (Dynamic Linear Model) scan benchmarks.
 *
 * Benchmarks associativeScan and sequential scan with Kalman-filter-like
 * composition bodies — the primary use case from dlm-js.
 *
 * Patterns:
 *   1. 2-tuple affine compose: (A, b) → { A: q.A @ p.A, b: q.A @ p.b + q.b }
 *   2. 3-tuple Särkkä compose: adds covariance S → q.A @ p.S @ q.A^T + q.S
 *   3. Scalar cumulative product (baseline comparison)
 *
 * Run with:
 *   pnpm build && pnpm vitest bench bench/dlm-scan.bench.ts
 */
import {
  blockUntilReady,
  defaultDevice,
  DType,
  grad,
  init,
  jit,
  lax,
  numpy as np,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init("wasm", "webgpu");
const hasWebGPU = devices.includes("webgpu");

// --- DLM composition functions ---

/** 2-tuple affine: compose(p, q) = { A: q.A @ p.A, b: q.A @ p.b + q.b } */
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

/** 3-tuple Särkkä: adds S → q.A @ p.S @ q.A^T + q.S */
const compose3 = (
  p: { A: np.Array; b: np.Array; S: np.Array },
  q: { A: np.Array; b: np.Array; S: np.Array },
) => {
  using tmp = np.matmul(q.A, p.b) as np.Array;
  using AS = np.matmul(q.A, p.S) as np.Array;
  using qAT = np.transpose(q.A, [-2, -1]) as np.Array;
  using ASAT = np.matmul(AS, qAT) as np.Array;
  return {
    A: np.matmul(q.A, p.A) as np.Array,
    b: tmp.add(q.b) as np.Array,
    S: ASAT.add(q.S) as np.Array,
  };
};

// ===================== WASM =====================

suite.skipIf(!devices.includes("wasm"))("wasm DLM scan", () => {
  defaultDevice("wasm");

  // --- 2-tuple compose, 2x2, N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });

    const assocJit = jit((AA: np.Array, bb: np.Array) =>
      lax.associativeScan(compose2, { A: AA, b: bb }),
    );
    // Warmup
    const w = assocJit(A, b) as { A: np.Array; b: np.Array };
    w.A.dispose();
    w.b.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      assocJit.dispose();
    });

    bench("assocScan 2-tuple 2×2 N=200", () => {
      const r = assocJit(A, b) as { A: np.Array; b: np.Array };
      r.A.dispose();
      r.b.dispose();
    });
  }

  // --- 2-tuple compose, 2x2, N=500 ---
  {
    const N = 500;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });

    const assocJit = jit((AA: np.Array, bb: np.Array) =>
      lax.associativeScan(compose2, { A: AA, b: bb }),
    );
    const w = assocJit(A, b) as { A: np.Array; b: np.Array };
    w.A.dispose();
    w.b.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      assocJit.dispose();
    });

    bench("assocScan 2-tuple 2×2 N=500", () => {
      const r = assocJit(A, b) as { A: np.Array; b: np.Array };
      r.A.dispose();
      r.b.dispose();
    });
  }

  // --- 3-tuple Särkkä compose, 2x2, N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    const S = np.full([N, 2, 2], 0.05, { dtype: DType.Float32 });

    const assocJit = jit((AA: np.Array, bb: np.Array, SS: np.Array) =>
      lax.associativeScan(compose3, { A: AA, b: bb, S: SS }),
    );
    const w = assocJit(A, b, S) as { A: np.Array; b: np.Array; S: np.Array };
    w.A.dispose();
    w.b.dispose();
    w.S.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      S.dispose();
      assocJit.dispose();
    });

    bench("assocScan 3-tuple Särkkä 2×2 N=200", () => {
      const r = assocJit(A, b, S) as {
        A: np.Array;
        b: np.Array;
        S: np.Array;
      };
      r.A.dispose();
      r.b.dispose();
      r.S.dispose();
    });
  }

  // --- 2-tuple compose, 4x4, N=200 ---
  {
    const N = 200;
    const A = np.full([N, 4, 4], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 4, 1], 0.1, { dtype: DType.Float32 });

    const assocJit = jit((AA: np.Array, bb: np.Array) =>
      lax.associativeScan(compose2, { A: AA, b: bb }),
    );
    const w = assocJit(A, b) as { A: np.Array; b: np.Array };
    w.A.dispose();
    w.b.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      assocJit.dispose();
    });

    bench("assocScan 2-tuple 4×4 N=200", () => {
      const r = assocJit(A, b) as { A: np.Array; b: np.Array };
      r.A.dispose();
      r.b.dispose();
    });
  }

  // --- Sequential scan comparison, 2-tuple, 2x2, N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });

    const scanJit = jit((AA: np.Array, bb: np.Array) => {
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
    const w = scanJit(A, b) as { A: np.Array; b: np.Array };
    tree.dispose(w);

    afterAll(() => {
      A.dispose();
      b.dispose();
      scanJit.dispose();
    });

    bench("scan 2-tuple 2×2 N=200", () => {
      const r = scanJit(A, b) as { A: np.Array; b: np.Array };
      tree.dispose(r);
    });
  }

  // --- grad(assocScan) 2-tuple 2×2 N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });

    const gradFn = jit(
      grad((AA: np.Array, bb: np.Array) => {
        const r = lax.associativeScan(compose2, { A: AA, b: bb }) as {
          A: np.Array;
          b: np.Array;
        };
        using sA = r.A.sum();
        using sB = r.b.sum();
        r.A.dispose();
        r.b.dispose();
        return sA.add(sB) as np.Array;
      }),
    );
    const w = gradFn(A, b) as np.Array;
    w.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      gradFn.dispose();
    });

    bench("grad(assocScan) 2-tuple 2×2 N=200", () => {
      const r = gradFn(A, b) as np.Array;
      r.dispose();
    });
  }

  // --- grad(scan) 2-tuple 2×2 N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });

    const gradFn = jit(
      grad((AA: np.Array, bb: np.Array) => {
        const scanResult = lax.scan(
          (
            c: { A: np.Array; b: np.Array },
            x: { A: np.Array; b: np.Array },
          ) => {
            const nc = compose2(c, x);
            return [nc, nc];
          },
          { A: np.eye(2, { dtype: DType.Float32 }), b: np.zeros([2, 1]) },
          { A: AA, b: bb },
        );
        tree.dispose(scanResult[0]);
        const r = scanResult[1] as { A: np.Array; b: np.Array };
        using sA = r.A.sum();
        using sB = r.b.sum();
        tree.dispose(r);
        return sA.add(sB) as np.Array;
      }),
    );
    const w = gradFn(A, b) as np.Array;
    w.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      gradFn.dispose();
    });

    bench("grad(scan) 2-tuple 2×2 N=200", () => {
      const r = gradFn(A, b) as np.Array;
      r.dispose();
    });
  }
});

// ===================== WebGPU =====================

suite.skipIf(!hasWebGPU)("webgpu DLM scan", async () => {
  if (!hasWebGPU) return;
  defaultDevice("webgpu");

  // --- 2-tuple compose, 2x2, N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    await blockUntilReady(A);
    await blockUntilReady(b);

    const assocJit = jit((AA: np.Array, bb: np.Array) =>
      lax.associativeScan(compose2, { A: AA, b: bb }),
    );
    const w = assocJit(A, b) as { A: np.Array; b: np.Array };
    w.A.dispose();
    w.b.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      assocJit.dispose();
    });

    bench("assocScan 2-tuple 2×2 N=200", () => {
      const r = assocJit(A, b) as { A: np.Array; b: np.Array };
      r.A.dispose();
      r.b.dispose();
    });
  }

  // --- 2-tuple compose, 2x2, N=500 ---
  {
    const N = 500;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    await blockUntilReady(A);
    await blockUntilReady(b);

    const assocJit = jit((AA: np.Array, bb: np.Array) =>
      lax.associativeScan(compose2, { A: AA, b: bb }),
    );
    const w = assocJit(A, b) as { A: np.Array; b: np.Array };
    w.A.dispose();
    w.b.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      assocJit.dispose();
    });

    bench("assocScan 2-tuple 2×2 N=500", () => {
      const r = assocJit(A, b) as { A: np.Array; b: np.Array };
      r.A.dispose();
      r.b.dispose();
    });
  }

  // --- 3-tuple Särkkä compose, 2x2, N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    const S = np.full([N, 2, 2], 0.05, { dtype: DType.Float32 });
    await blockUntilReady(A);
    await blockUntilReady(b);
    await blockUntilReady(S);

    const assocJit = jit((AA: np.Array, bb: np.Array, SS: np.Array) =>
      lax.associativeScan(compose3, { A: AA, b: bb, S: SS }),
    );
    const w = assocJit(A, b, S) as { A: np.Array; b: np.Array; S: np.Array };
    w.A.dispose();
    w.b.dispose();
    w.S.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      S.dispose();
      assocJit.dispose();
    });

    bench("assocScan 3-tuple Särkkä 2×2 N=200", () => {
      const r = assocJit(A, b, S) as {
        A: np.Array;
        b: np.Array;
        S: np.Array;
      };
      r.A.dispose();
      r.b.dispose();
      r.S.dispose();
    });
  }

  // --- Sequential scan comparison, 2-tuple, 2x2, N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    await blockUntilReady(A);
    await blockUntilReady(b);

    const scanJit = jit((AA: np.Array, bb: np.Array) => {
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
    const w = scanJit(A, b) as { A: np.Array; b: np.Array };
    tree.dispose(w);

    afterAll(() => {
      A.dispose();
      b.dispose();
      scanJit.dispose();
    });

    bench("scan 2-tuple 2×2 N=200", () => {
      const r = scanJit(A, b) as { A: np.Array; b: np.Array };
      tree.dispose(r);
    });
  }

  // --- grad(assocScan) 2-tuple 2×2 N=200 ---
  {
    const N = 200;
    const A = np.full([N, 2, 2], 0.95, { dtype: DType.Float32 });
    const b = np.full([N, 2, 1], 0.1, { dtype: DType.Float32 });
    await blockUntilReady(A);
    await blockUntilReady(b);

    const gradFn = jit(
      grad((AA: np.Array, bb: np.Array) => {
        const r = lax.associativeScan(compose2, { A: AA, b: bb }) as {
          A: np.Array;
          b: np.Array;
        };
        using sA = r.A.sum();
        using sB = r.b.sum();
        r.A.dispose();
        r.b.dispose();
        return sA.add(sB) as np.Array;
      }),
    );
    const w = gradFn(A, b) as np.Array;
    w.dispose();

    afterAll(() => {
      A.dispose();
      b.dispose();
      gradFn.dispose();
    });

    bench("grad(assocScan) 2-tuple 2×2 N=200", () => {
      const r = gradFn(A, b) as np.Array;
      r.dispose();
    });
  }
});
