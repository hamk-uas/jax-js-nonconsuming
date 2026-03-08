/**
 * Tests for blocked-data-movement primitives:
 * - gatherAxisPoints
 * - copyAxisRange
 * - mapOverBlocks
 *
 * These primitives are internal to block-map-executor.ts. We test them
 * indirectly through lax.associativeScan on WebGPU (webgpu-block-map path),
 * which exercises all three in its blocked Kogge-Stone decomposition:
 *
 *   copyAxisRange  — M=1 single-block fast path + block-0 copy
 *   gatherAxisPoints — block-tail extraction for summary scan
 *   mapOverBlocks  — apply scanned summaries to blocks 1..M-1
 *
 * Test cases are parameterized on N to hit block-boundary edge cases
 * relative to B=256 (the WebGPU block size).
 */

import {
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

const B = 256; // WebGPU block size

// Attempt top-level WebGPU init. If WebGPU is unavailable the entire file
// is effectively a no-op (all describes use `describe.skipIf`).
let hasWebGPU = false;
try {
  const devs = await init("webgpu");
  if (devs.includes("webgpu")) {
    defaultDevice("webgpu");
    hasWebGPU = true;
  }
} catch {
  await init();
}

// Reference cumsum for verification
function referenceCumsum(values: number[]): number[] {
  const out = new Array(values.length);
  out[0] = values[0];
  for (let i = 1; i < values.length; i++) out[i] = out[i - 1] + values[i];
  return out;
}

// Reference cumprod for verification
function referenceCumprod(values: number[]): number[] {
  const out = new Array(values.length);
  out[0] = values[0];
  for (let i = 1; i < values.length; i++) out[i] = out[i - 1] * values[i];
  return out;
}

// ============================================================================
// copyAxisRange — exercised by the M=1 (single-block) fast path
// ============================================================================

describe("copyAxisRange (via single-block assocScan)", () => {
  test.skipIf(!hasWebGPU)("N < B: single block, entire range", () => {
    const values = Array.from({ length: 100 }, (_, i) => i + 1);
    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose(referenceCumsum(values));
  });

  test.skipIf(!hasWebGPU)("N = 1: scalar edge case", () => {
    using xs = np.array([42.0]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose([42]);
  });

  test.skipIf(!hasWebGPU)("N = B: exactly one full block", () => {
    const values = Array.from({ length: B }, () => 1);
    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose(referenceCumsum(values));
  });
});

// ============================================================================
// gatherAxisPoints — exercised when M > 1 (multi-block)
// ============================================================================

describe("gatherAxisPoints (via multi-block assocScan)", () => {
  test.skipIf(!hasWebGPU)("N = B+1: two blocks, partial trailing block", () => {
    const N = B + 1;
    const values = Array.from({ length: N }, (_, i) => i + 1);
    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose(referenceCumsum(values));
  });

  test.skipIf(!hasWebGPU)("N = 2*B: exactly two full blocks", () => {
    const N = 2 * B;
    const values = Array.from({ length: N }, () => 1);
    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose(referenceCumsum(values));
  });

  test.skipIf(!hasWebGPU)(
    "N = 2*B-1: partial last block (one element short)",
    () => {
      const N = 2 * B - 1;
      const values = Array.from({ length: N }, (_, i) => (i % 7) + 1);
      using xs = np.array(values);
      using result = lax.associativeScan(
        (a: np.Array, b: np.Array) => np.add(a, b),
        xs,
      );
      expect(result).toBeAllclose(referenceCumsum(values));
    },
  );

  test.skipIf(!hasWebGPU)("N = 3*B+17: three full blocks + partial", () => {
    const N = 3 * B + 17;
    const values = Array.from({ length: N }, (_, i) => (i % 5) + 1);
    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    expect(result).toBeAllclose(referenceCumsum(values));
  });
});

// ============================================================================
// mapOverBlocks — exercised by the apply phase (blocks 1..M-1)
// ============================================================================

describe("mapOverBlocks (via multi-block apply phase)", () => {
  test.skipIf(!hasWebGPU)(
    "cumulative product — non-linear associative op",
    () => {
      // cumprod exercises that mapOverBlocks correctly applies the prefix
      // via multiplication, not just addition
      const N = B + 50;
      // Use small values to avoid float overflow
      const values = Array.from(
        { length: N },
        () => 1.0 + Math.random() * 0.01,
      );
      using xs = np.array(values);
      using result = lax.associativeScan(
        (a: np.Array, b: np.Array) => np.multiply(a, b),
        xs,
      );
      expect(result).toBeAllclose(referenceCumprod(values), { rtol: 1e-4 });
    },
  );

  test.skipIf(!hasWebGPU)("running max — non-additive associative op", () => {
    const N = B + 100;
    const values = Array.from({ length: N }, (_, i) => Math.sin(i * 0.1) * 100);
    const expected = new Array(N);
    expected[0] = values[0];
    for (let i = 1; i < N; i++)
      expected[i] = Math.max(expected[i - 1], values[i]);

    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.maximum(a, b),
      xs,
    );
    expect(result).toBeAllclose(expected);
  });

  test.skipIf(!hasWebGPU)("partial trailing block correctness", () => {
    // N = B + 3: block 1 has only 3 elements. Verifies that mapOverBlocks
    // correctly writes only valid elements and doesn't corrupt output.
    const N = B + 3;
    const values = Array.from({ length: N }, (_, i) => i + 1);
    using xs = np.array(values);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );
    const data = result.js() as number[];
    const expected = referenceCumsum(values);
    // Verify every single element, especially the 3 in the partial block
    for (let i = 0; i < N; i++) {
      expect(data[i]).toBeCloseTo(expected[i], 0);
    }
  });

  test.skipIf(!hasWebGPU)("jit(assocScan) with multi-block N", () => {
    // Ensures that the primitives work under JIT compilation too
    const N = 2 * B + 10;
    const values = Array.from({ length: N }, (_, i) => i + 1);
    using f = jit((xs: np.Array) =>
      lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), xs),
    );
    using xs = np.array(values);
    using result = f(xs) as np.Array;
    expect(result).toBeAllclose(referenceCumsum(values));
  });
});

// ============================================================================
// Multi-level recursion (N > B²) — exercises the full recursive decomposition
// ============================================================================

describe("multi-level recursion (N > B²)", () => {
  test.skipIf(!hasWebGPU)(
    "N = B² + B: three-level blocked decomposition",
    () => {
      // B=256, B²=65536, M=257 > B → triggers a second recursion level.
      // Level 0: N=65792 → M=257 blocks
      // Level 1: N=257 → M=2 blocks
      // Level 2: N=2 → M=1 block (terminates)
      const N = B * B + B;
      const values = Array.from({ length: N }, () => 1);
      using xs = np.array(values);
      using result = lax.associativeScan(
        (a: np.Array, b: np.Array) => np.add(a, b),
        xs,
      );
      // cumsum of all-ones = [1, 2, 3, ..., N]
      const data = result.js() as number[];
      expect(data[0]).toBe(1);
      expect(data[N - 1]).toBe(N);
      // Spot-check block boundaries
      expect(data[B - 1]).toBe(B);
      expect(data[B]).toBe(B + 1);
      expect(data[2 * B - 1]).toBe(2 * B);
    },
  );
});

// ============================================================================
// 2-D elements — verifies that per-element shapes > scalar work through
// the gatherAxisPoints / copyAxisRange / mapOverBlocks pipeline
// ============================================================================

describe("blocked assocScan with multi-dimensional elements", () => {
  test.skipIf(!hasWebGPU)("cumsum over [N, 4] with N > B", () => {
    const N = B + 32;
    const data = new Float32Array(N * 4);
    for (let i = 0; i < N * 4; i++) data[i] = (i % 7) + 1;

    using flat = np.array(data);
    using xs = flat.reshape([N, 4]);
    using result = lax.associativeScan(
      (a: np.Array, b: np.Array) => np.add(a, b),
      xs,
    );

    // Verify cumsum independently for each of the 4 columns
    const resultData = result.js() as number[][];
    for (let col = 0; col < 4; col++) {
      let acc = 0;
      for (let row = 0; row < N; row++) {
        acc += data[row * 4 + col];
        expect(resultData[row][col]).toBeCloseTo(acc, 0);
      }
    }
  });
});
