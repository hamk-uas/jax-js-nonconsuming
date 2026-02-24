/**
 * Deno WebGPU regression tests for M7.4 fused assocScan reduction codegen.
 *
 * The M7.4 fused Kogge-Stone shader had a bug: reduction kernels with
 * kernelSize > 1 (e.g., matmul producing 2×2 = 4 output elements) only
 * computed one element (writing to internal_out[0]) and used the wrong
 * variable (gidx = scan position instead of eidx = output element index).
 *
 * This caused matrix compositions (DLM Kalman filter pattern) to produce
 * exponentially growing garbage values (~1e12-1e34) from the very first
 * Kogge-Stone round.
 *
 * Tests exercise every codegen branch:
 *   1. Reduction kernel, kernelSize === 1 (scalar sum reduction)
 *   2. Reduction kernel, kernelSize > 1  (matmul → 4 output elements)
 *   3. Mixed body: reduction + elementwise (matmul + add = affine compose)
 *   4. Reverse variant of matmul compose
 *   5. Longer sequence (8 steps, 3 Kogge-Stone rounds)
 *
 * Run with:
 *   pnpm build && deno test --no-check --unstable-webgpu --allow-read --allow-env test/deno/assoc-scan-reduction.test.ts
 */

import {
  assertAlmostEquals,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";

import { initWebGPU, withLeakCheck } from "./harness.ts";
import { DType, jit, lax, numpy as np } from "../../dist/index.js";

// ---------------------------------------------------------------------------
// 1. Reduction kernel, kernelSize > 1: matrix multiply compose
//    THE EXACT BUG CASE — matmul produces 2×2 = 4 output elements.
// ---------------------------------------------------------------------------

Deno.test({
  name: "M7.4 fix: matmul compose (reduction kernelSize > 1)",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    // 3 time steps of 2×2 matrices
    const xs = np.array(
      [
        [
          [1, 2],
          [0, 1],
        ],
        [
          [2, 0],
          [1, 3],
        ],
        [
          [1, 1],
          [0, 2],
        ],
      ],
      { dtype: DType.Float32 },
    );

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.matmul(b, a) as any, x),
    );
    const result = f(xs);
    const data = await result.data();

    // i=0: [[1,2],[0,1]] (unchanged)
    assertAlmostEquals(data[0], 1, 1e-4);
    assertAlmostEquals(data[1], 2, 1e-4);
    assertAlmostEquals(data[2], 0, 1e-4);
    assertAlmostEquals(data[3], 1, 1e-4);

    // i=1: xs[1] @ result[0] = [[2,0],[1,3]] @ [[1,2],[0,1]] = [[2,4],[1,5]]
    assertAlmostEquals(data[4], 2, 1e-4);
    assertAlmostEquals(data[5], 4, 1e-4);
    assertAlmostEquals(data[6], 1, 1e-4);
    assertAlmostEquals(data[7], 5, 1e-4);

    // i=2: xs[2] @ result[1] = [[1,1],[0,2]] @ [[2,4],[1,5]] = [[3,9],[2,10]]
    assertAlmostEquals(data[8], 3, 1e-4);
    assertAlmostEquals(data[9], 9, 1e-4);
    assertAlmostEquals(data[10], 2, 1e-4);
    assertAlmostEquals(data[11], 10, 1e-4);

    result.dispose();
    xs.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 2. Mixed body: matmul (reduction) + add (elementwise) — DLM affine pattern
// ---------------------------------------------------------------------------

Deno.test({
  name: "M7.4 fix: affine compose matmul+add (mixed reduction+elementwise)",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const compose = (p: { A: any; b: any }, q: { A: any; b: any }) => {
      const tmp = np.matmul(q.A, p.b);
      const result = {
        A: np.matmul(q.A, p.A),
        b: tmp.add(q.b),
      };
      tmp.dispose();
      return result;
    };

    const A = np.array(
      [
        [
          [1, 0.5],
          [0, 1],
        ],
        [
          [0.9, 0],
          [0, 0.9],
        ],
        [
          [1, 0],
          [0.1, 1],
        ],
        [
          [0.8, 0.2],
          [0, 0.8],
        ],
      ],
      { dtype: DType.Float32 },
    );
    const b = np.array(
      [
        [[1], [0]],
        [[0], [1]],
        [[2], [0]],
        [[0], [3]],
      ],
      { dtype: DType.Float32 },
    );

    using f = jit((Ain: any, bin: any) =>
      lax.associativeScan(compose, { A: Ain, b: bin }),
    );

    const result = f(A, b) as any;
    const resultAData = await result.A.data();
    const resultBData = await result.b.data();

    // i=0: unchanged
    assertAlmostEquals(resultAData[0], 1, 1e-4);
    assertAlmostEquals(resultAData[1], 0.5, 1e-4);
    assertAlmostEquals(resultAData[2], 0, 1e-4);
    assertAlmostEquals(resultAData[3], 1, 1e-4);
    assertAlmostEquals(resultBData[0], 1, 1e-4);
    assertAlmostEquals(resultBData[1], 0, 1e-4);

    // i=1: A = [[0.9,0.45],[0,0.9]], b = [[0.9],[1]]
    assertAlmostEquals(resultAData[4], 0.9, 1e-4);
    assertAlmostEquals(resultAData[5], 0.45, 1e-4);
    assertAlmostEquals(resultAData[6], 0, 1e-4);
    assertAlmostEquals(resultAData[7], 0.9, 1e-4);
    assertAlmostEquals(resultBData[2], 0.9, 1e-4);
    assertAlmostEquals(resultBData[3], 1, 1e-4);

    result.A.dispose();
    result.b.dispose();
    A.dispose();
    b.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 3. Reverse matmul compose — verifies reverse + reduction codegen interaction
// ---------------------------------------------------------------------------

Deno.test({
  name: "M7.4 fix: reverse matmul compose",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const xs = np.array(
      [
        [
          [1, 2],
          [0, 1],
        ],
        [
          [2, 0],
          [1, 3],
        ],
        [
          [1, 1],
          [0, 2],
        ],
      ],
      { dtype: DType.Float32 },
    );

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.matmul(b, a) as any, x, {
        reverse: true,
      }),
    );
    const result = f(xs);
    const data = await result.data();

    // Reverse uses flip-forward-flip: flip input, scan forward, flip output.
    // For non-commutative fn(a,b)=matmul(b,a), this gives:
    // After flip: [C, B, A]. Forward scan: [C, B@C, A@(B@C)]. Flip back:
    // position 0 = A@(B@C), position 1 = B@C, position 2 = C.

    // i=2 (last): C = [[1,1],[0,2]]
    assertAlmostEquals(data[8], 1, 1e-4);
    assertAlmostEquals(data[9], 1, 1e-4);
    assertAlmostEquals(data[10], 0, 1e-4);
    assertAlmostEquals(data[11], 2, 1e-4);

    // i=1: B@C = [[2,2],[1,7]]
    assertAlmostEquals(data[4], 2, 1e-4);
    assertAlmostEquals(data[5], 2, 1e-4);
    assertAlmostEquals(data[6], 1, 1e-4);
    assertAlmostEquals(data[7], 7, 1e-4);

    // i=0: A@(B@C) = [[4,16],[1,7]]
    assertAlmostEquals(data[0], 4, 1e-4);
    assertAlmostEquals(data[1], 16, 1e-4);
    assertAlmostEquals(data[2], 1, 1e-4);
    assertAlmostEquals(data[3], 7, 1e-4);

    result.dispose();
    xs.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 4. 8-step sequence — multiple Kogge-Stone rounds (log2(8) = 3)
//    Catches issues that compound across rounds.
// ---------------------------------------------------------------------------

Deno.test({
  name: "M7.4 fix: 8-step matmul prefix product (3 Kogge-Stone rounds)",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const mats: number[][][] = [];
    for (let i = 0; i < 8; i++) {
      mats.push([
        [1 + i * 0.1, i * 0.05],
        [0, 1 + i * 0.05],
      ]);
    }
    const xs = np.array(mats, { dtype: DType.Float32 });

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.matmul(b, a) as any, x),
    );
    const result = f(xs);
    const data = await result.data();

    // Compute sequential reference
    const seqMats: number[][][] = [];
    for (let i = 0; i < 8; i++) {
      if (i === 0) {
        seqMats.push(mats[0]);
      } else {
        const prev = seqMats[i - 1];
        const curr = mats[i];
        seqMats.push([
          [
            curr[0][0] * prev[0][0] + curr[0][1] * prev[1][0],
            curr[0][0] * prev[0][1] + curr[0][1] * prev[1][1],
          ],
          [
            curr[1][0] * prev[0][0] + curr[1][1] * prev[1][0],
            curr[1][0] * prev[0][1] + curr[1][1] * prev[1][1],
          ],
        ]);
      }
    }

    for (let i = 0; i < 8; i++) {
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const idx = i * 4 + r * 2 + c;
          assertAlmostEquals(data[idx], seqMats[i][r][c], 1e-3);
        }
      }
    }

    result.dispose();
    xs.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 5. Scalar reduction (kernelSize === 1) — verify the kernelSize===1 branch
// ---------------------------------------------------------------------------

Deno.test({
  name: "M7.4 fix: scalar sum reduction (kernelSize === 1)",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    // cumsum of scalars — each element is a scalar sum (reduction over size-1)
    const xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], { dtype: DType.Float32 });

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.add(a, b), x),
    );
    const result = f(xs);
    const data = await result.data();

    assertEquals(data.length, 5);
    assertAlmostEquals(data[0], 1, 1e-5);
    assertAlmostEquals(data[1], 3, 1e-5);
    assertAlmostEquals(data[2], 6, 1e-5);
    assertAlmostEquals(data[3], 10, 1e-5);
    assertAlmostEquals(data[4], 15, 1e-5);

    result.dispose();
    xs.dispose();
  }),
});
