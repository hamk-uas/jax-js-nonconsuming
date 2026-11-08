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
 *   pnpm vitest bench bench/dlm-scan.bench.ts
 */
import {
  defaultDevice,
  DType,
  grad,
  init,
  jit,
  lax,
  numpy as np,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

const devices = await init("wasm", "webgpu");

// --- DLM composition functions ---

type Affine = { A: np.Array; b: np.Array };
type Sarkka = Affine & { S: np.Array };

/** 2-tuple affine: compose(p, q) = { A: q.A @ p.A, b: q.A @ p.b + q.b } */
const compose2 = (p: Affine, q: Affine): Affine => {
  using tmp = np.matmul(q.A, p.b);
  return { A: np.matmul(q.A, p.A), b: tmp.add(q.b) };
};

/** 3-tuple Särkkä: adds S → q.A @ p.S @ q.A^T + q.S */
const compose3 = (p: Sarkka, q: Sarkka): Sarkka => {
  using tmp = np.matmul(q.A, p.b);
  using AS = np.matmul(q.A, p.S);
  using qAT = np.transpose(q.A, [-2, -1]);
  using ASAT = np.matmul(AS, qAT);
  return { A: np.matmul(q.A, p.A), b: tmp.add(q.b), S: ASAT.add(q.S) };
};

/** Inputs for N compose elements of m×m state, owned by `stack`. */
function affineInputs(stack: DisposableStack, N: number, m: number): Affine {
  return {
    A: stack.use(np.full([N, m, m], 0.95, { dtype: DType.Float32 })),
    b: stack.use(np.full([N, m, 1], 0.1, { dtype: DType.Float32 })),
  };
}

/** Sequential scan of compose2 from the identity; returns the stacked ys. */
function scanCompose2(xs: Affine): Affine {
  const [carry, ys] = lax.scan(
    (c: Affine, x: Affine): [Affine, Affine] => {
      const nc = compose2(c, x);
      return [nc, nc];
    },
    { A: np.eye(2, { dtype: DType.Float32 }), b: np.zeros([2, 1]) },
    xs,
  );
  tree.dispose(carry);
  return ys;
}

/** Sum of all leaves of a compose result, disposing the result. */
function sumLeaves(r: Affine): np.Array {
  using sA = r.A.sum();
  using sB = r.b.sum();
  tree.dispose(r);
  return sA.add(sB);
}

/**
 * A DLM workload: `make` builds its inputs and jit function (owned by
 * `stack`) and returns one benchmark iteration.
 */
type DlmCase = {
  name: string;
  /** Also run on WebGPU (all cases run on WASM). */
  webgpu: boolean;
  make: (stack: DisposableStack) => () => void;
};

function assocCompose2(N: number, m: number) {
  return (stack: DisposableStack) => {
    const { A, b } = affineInputs(stack, N, m);
    const f = stack.use(
      jit((AA: np.Array, bb: np.Array) =>
        lax.associativeScan(compose2, { A: AA, b: bb }),
      ),
    );
    return () => tree.dispose(f(A, b));
  };
}

const CASES: DlmCase[] = [
  {
    name: "assocScan 2-tuple 2×2 N=200",
    webgpu: true,
    make: assocCompose2(200, 2),
  },
  {
    name: "assocScan 2-tuple 2×2 N=500",
    webgpu: true,
    make: assocCompose2(500, 2),
  },
  {
    name: "assocScan 3-tuple Särkkä 2×2 N=200",
    webgpu: true,
    make: (stack) => {
      const { A, b } = affineInputs(stack, 200, 2);
      const S = stack.use(np.full([200, 2, 2], 0.05, { dtype: DType.Float32 }));
      const f = stack.use(
        jit((AA: np.Array, bb: np.Array, SS: np.Array) =>
          lax.associativeScan(compose3, { A: AA, b: bb, S: SS }),
        ),
      );
      return () => tree.dispose(f(A, b, S));
    },
  },
  {
    name: "assocScan 2-tuple 4×4 N=200",
    webgpu: false,
    make: assocCompose2(200, 4),
  },
  {
    name: "scan 2-tuple 2×2 N=200",
    webgpu: true,
    make: (stack) => {
      const { A, b } = affineInputs(stack, 200, 2);
      const f = stack.use(
        jit((AA: np.Array, bb: np.Array) => scanCompose2({ A: AA, b: bb })),
      );
      return () => tree.dispose(f(A, b));
    },
  },
  {
    name: "grad(assocScan) 2-tuple 2×2 N=200",
    webgpu: true,
    make: (stack) => {
      const { A, b } = affineInputs(stack, 200, 2);
      const f = stack.use(
        jit(
          grad((AA: np.Array, bb: np.Array) =>
            sumLeaves(lax.associativeScan(compose2, { A: AA, b: bb })),
          ),
        ),
      );
      return () => f(A, b).dispose();
    },
  },
  {
    name: "grad(scan) 2-tuple 2×2 N=200",
    webgpu: false,
    make: (stack) => {
      const { A, b } = affineInputs(stack, 200, 2);
      const f = stack.use(
        jit(
          grad((AA: np.Array, bb: np.Array) =>
            sumLeaves(scanCompose2({ A: AA, b: bb })),
          ),
        ),
      );
      return () => f(A, b).dispose();
    },
  },
];

for (const device of ["wasm", "webgpu"] as const) {
  describe.skipIf(!devices.includes(device))(`${device} DLM scan`, () => {
    test("compose bodies", async ({ bench }) => {
      defaultDevice(device);
      using stack = new DisposableStack();
      const registrations = CASES.filter(
        (c) => device === "wasm" || c.webgpu,
      ).map((c) => bench(c.name, c.make(stack)));
      await bench.compare(...registrations);
    });
  });
}
