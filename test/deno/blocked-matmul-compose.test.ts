/**
 * Deno WebGPU tests for blocked associativeScan with matmul compose bodies.
 *
 * The blocked path (webgpu-fused-blocked) uses B=256 workgroup-level scan.
 * When N > 256, M = ceil(N/B) > 1, triggering the full 4-stage pipeline:
 *   1. Local scan within each block (B elements)
 *   2. Gather last element per block → summary array
 *   3. Summary scan (flat Kogge-Stone on M elements)
 *   4. Apply: combine scanned summary prefix with each block's local results
 *
 * The existing assoc-scan-reduction.test.ts tests N=3..8 (M=1, single block).
 * These tests use N=512..2048 (M=2..8) to exercise the multi-block pipeline
 * with matmul compose bodies (reduction kernels, kernelSize > 1).
 *
 * Run with:
 *   pnpm build && deno test --no-check --unstable-webgpu --allow-read --allow-env test/deno/blocked-matmul-compose.test.ts
 */

import {
  assertAlmostEquals,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";

import { initWebGPU, withLeakCheck } from "./harness.ts";
import { DType, jit, lax, numpy as np } from "../../dist/index.js";

// ---------------------------------------------------------------------------
// Helper: generate N random 2x2 rotation matrices (spectral radius = 1,
// products stay bounded in f32 for arbitrary N).
// ---------------------------------------------------------------------------
function rotationMats2x2(n: number, seed: number = 42): number[][][] {
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff; // [0, 1)
  };
  const mats: number[][][] = [];
  for (let i = 0; i < n; i++) {
    const theta = rand() * 2 * Math.PI;
    const c = Math.cos(theta);
    const sn = Math.sin(theta);
    mats.push([
      [c, -sn],
      [sn, c],
    ]);
  }
  return mats;
}

// Sequential reference for matmul prefix product
function seqMatmulPrefix(mats: number[][][]): number[][][] {
  const result: number[][][] = [mats[0]];
  for (let i = 1; i < mats.length; i++) {
    const prev = result[i - 1];
    const curr = mats[i];
    result.push([
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
  return result;
}

// ---------------------------------------------------------------------------
// 1. N=512 (M=2 blocks): minimal multi-block matmul compose
// ---------------------------------------------------------------------------

Deno.test({
  name: "blocked matmul compose: N=512, M=2 blocks",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const N = 512;
    const mats = rotationMats2x2(N);
    const xs = np.array(mats, { dtype: DType.Float32 });

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.matmul(b, a) as any, x),
    );
    const result = f(xs);
    const data = await result.data();

    const ref = seqMatmulPrefix(mats);

    // Check first element (unchanged)
    for (let r = 0; r < 2; r++)
      for (let c = 0; c < 2; c++)
        assertAlmostEquals(data[r * 2 + c], ref[0][r][c], 1e-3);

    // Check boundary elements around block boundary (index 255, 256, 257)
    for (const i of [255, 256, 257]) {
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const idx = i * 4 + r * 2 + c;
          const relTol = Math.max(Math.abs(ref[i][r][c]) * 1e-3, 1e-2);
          assertAlmostEquals(
            data[idx],
            ref[i][r][c],
            relTol,
            `mat[${i}][${r}][${c}]`,
          );
        }
      }
    }

    // Check last element
    const last = N - 1;
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const idx = last * 4 + r * 2 + c;
        const relTol = Math.max(Math.abs(ref[last][r][c]) * 1e-3, 1e-2);
        assertAlmostEquals(
          data[idx],
          ref[last][r][c],
          relTol,
          `mat[${last}][${r}][${c}]`,
        );
      }
    }

    result.dispose();
    xs.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 2. N=1024 (M=4 blocks): summary scan has multiple KS rounds
// ---------------------------------------------------------------------------

Deno.test({
  name: "blocked matmul compose: N=1024, M=4 blocks",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const N = 1024;
    const mats = rotationMats2x2(N);
    const xs = np.array(mats, { dtype: DType.Float32 });

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.matmul(b, a) as any, x),
    );
    const result = f(xs);
    const data = await result.data();

    const ref = seqMatmulPrefix(mats);

    // Spot-check several positions spanning block boundaries
    // Block boundaries at 256, 512, 768
    for (const i of [0, 127, 255, 256, 511, 512, 767, 768, N - 1]) {
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const idx = i * 4 + r * 2 + c;
          const relTol = Math.max(Math.abs(ref[i][r][c]) * 1e-3, 1e-2);
          assertAlmostEquals(
            data[idx],
            ref[i][r][c],
            relTol,
            `mat[${i}][${r}][${c}]`,
          );
        }
      }
    }

    result.dispose();
    xs.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 3. Reverse blocked matmul compose, N=512
// ---------------------------------------------------------------------------

Deno.test({
  name: "blocked matmul compose: reverse, N=512",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const N = 512;
    const mats = rotationMats2x2(N, 99);
    const xs = np.array(mats, { dtype: DType.Float32 });

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.matmul(b, a) as any, x, {
        reverse: true,
      }),
    );
    const result = f(xs);
    const data = await result.data();

    // Reference: reverse input, scan forward, reverse output
    const reversed = [...mats].reverse();
    const fwdRef = seqMatmulPrefix(reversed);
    const ref = [...fwdRef].reverse();

    // Check boundary elements and endpoints
    for (const i of [0, 255, 256, N - 1]) {
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const idx = i * 4 + r * 2 + c;
          const relTol = Math.max(Math.abs(ref[i][r][c]) * 1e-3, 1e-2);
          assertAlmostEquals(
            data[idx],
            ref[i][r][c],
            relTol,
            `mat[${i}][${r}][${c}]`,
          );
        }
      }
    }

    result.dispose();
    xs.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 4. Blocked affine compose: matmul+add (mixed reduction+elementwise), N=512
// ---------------------------------------------------------------------------

Deno.test({
  name: "blocked affine compose: matmul+add, N=512",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const N = 512;

    // Generate random affine transforms: x -> Ax + b
    // Composition: (A2, b2) ∘ (A1, b1) = (A2*A1, A2*b1 + b2)
    let seed = 77;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
      return (seed / 0x7fffffff) * 0.4 + 0.8; // [0.8, 1.2]
    };

    const As: number[][][] = [];
    const bs: number[][] = [];
    for (let i = 0; i < N; i++) {
      As.push([
        [rand(), rand() * 0.1],
        [rand() * 0.1, rand()],
      ]);
      bs.push([rand() * 0.5, rand() * 0.5]);
    }

    const A = np.array(As, { dtype: DType.Float32 });
    const b = np.array(
      bs.map((v) => [[v[0]], [v[1]]]),
      { dtype: DType.Float32 },
    );

    using f = jit((Ain: any, bin: any) =>
      lax.associativeScan(
        (p: { A: any; b: any }, q: { A: any; b: any }) => {
          const tmp = np.matmul(q.A, p.b);
          const result = {
            A: np.matmul(q.A, p.A),
            b: tmp.add(q.b),
          };
          tmp.dispose();
          return result;
        },
        { A: Ain, b: bin },
      ),
    );

    const result = f(A, b) as any;
    const resultAData = await result.A.data();
    const resultBData = await result.b.data();

    // Sequential reference
    const refA: number[][][] = [As[0]];
    const refB: number[][] = [bs[0]];
    for (let i = 1; i < N; i++) {
      const pA = refA[i - 1];
      const pb = refB[i - 1];
      const qA = As[i];
      const qb = bs[i];
      // A = qA * pA
      refA.push([
        [
          qA[0][0] * pA[0][0] + qA[0][1] * pA[1][0],
          qA[0][0] * pA[0][1] + qA[0][1] * pA[1][1],
        ],
        [
          qA[1][0] * pA[0][0] + qA[1][1] * pA[1][0],
          qA[1][0] * pA[0][1] + qA[1][1] * pA[1][1],
        ],
      ]);
      // b = qA * pb + qb
      refB.push([
        qA[0][0] * pb[0] + qA[0][1] * pb[1] + qb[0],
        qA[1][0] * pb[0] + qA[1][1] * pb[1] + qb[1],
      ]);
    }

    // Spot-check across block boundaries
    for (const i of [0, 255, 256, N - 1]) {
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const aIdx = i * 4 + r * 2 + c;
          const relTol = Math.max(Math.abs(refA[i][r][c]) * 1e-3, 1e-2);
          assertAlmostEquals(
            resultAData[aIdx],
            refA[i][r][c],
            relTol,
            `A[${i}][${r}][${c}]`,
          );
        }
      }
      // b is [N, 2, 1] shape → index i * 2 + row
      for (let r = 0; r < 2; r++) {
        const bIdx = i * 2 + r;
        const relTol = Math.max(Math.abs(refB[i][r]) * 1e-3, 1e-2);
        assertAlmostEquals(
          resultBData[bIdx],
          refB[i][r],
          relTol,
          `b[${i}][${r}]`,
        );
      }
    }

    result.A.dispose();
    result.b.dispose();
    A.dispose();
    b.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 5. Large N=2048 (M=8 blocks): stress test multi-block matmul compose
// ---------------------------------------------------------------------------

Deno.test({
  name: "blocked matmul compose: N=2048, M=8 blocks",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const N = 2048;
    const mats = rotationMats2x2(N, 123);
    const xs = np.array(mats, { dtype: DType.Float32 });

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.matmul(b, a) as any, x),
    );
    const result = f(xs);
    const data = await result.data();

    assertEquals(data.length, N * 4);

    const ref = seqMatmulPrefix(mats);

    // Spot-check at every block boundary and endpoints
    for (const i of [
      0,
      255,
      256,
      511,
      512,
      767,
      768,
      1023,
      1024,
      1535,
      1536,
      N - 1,
    ]) {
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const idx = i * 4 + r * 2 + c;
          const relTol = Math.max(Math.abs(ref[i][r][c]) * 1e-3, 1e-2);
          assertAlmostEquals(
            data[idx],
            ref[i][r][c],
            relTol,
            `mat[${i}][${r}][${c}]`,
          );
        }
      }
    }

    result.dispose();
    xs.dispose();
  }),
});

// ---------------------------------------------------------------------------
// 6. Scalar cumsum at large N (blocked, kernelSize=1 branch)
// ---------------------------------------------------------------------------

Deno.test({
  name: "blocked cumsum: N=1024, kernelSize=1 sanity check",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    const N = 1024;
    const data = new Float32Array(N);
    for (let i = 0; i < N; i++) data[i] = 1.0;

    const xs = np.array(data, { dtype: DType.Float32 });

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.add(a, b), x),
    );
    const result = f(xs);
    const d = await result.data();

    assertEquals(d.length, N);
    // cumsum of all-ones: result[i] = i + 1
    for (let i = 0; i < N; i++) {
      assertAlmostEquals(d[i], i + 1, 1e-2, `cumsum[${i}]`);
    }

    result.dispose();
    xs.dispose();
  }),
});
