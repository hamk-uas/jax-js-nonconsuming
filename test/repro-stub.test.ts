/**
 * Focused repro of stub-block corruption in reverse assocScan with 3-field compose.
 * Tests N=65 (stub=1) and N=200 (stub=8), comparing WASM vs WebGPU.
 */
import {
  clearCaches,
  defaultDevice,
  DType,
  init,
  jit,
  lax,
  numpy as np,
  setDebug,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

describe("stub-block reverse assocScan repro", () => {
  for (const N of [65, 128, 200]) {
    for (const rev of [false, true]) {
      it(`N=${N} m=5 ${rev ? "reverse" : "forward"} 3-field compose`, async ({
        skip,
      }) => {
        await init("wasm");
        const devs = await init("webgpu");
        if (!devs.includes("webgpu")) {
          skip();
          return;
        }

        const m = 5;
        // Deterministic PRNG
        let seed = 42;
        const rng = () => {
          seed = (seed * 1103515245 + 12345) & 0x7fffffff;
          return seed / 0x7fffffff;
        };
        const A_data: number[][][] = [];
        const b_data: number[][][] = [];
        const S_data: number[][][] = [];
        for (let i = 0; i < N; i++) {
          const aRow: number[][] = [],
            sRow: number[][] = [];
          for (let j = 0; j < m; j++) {
            const ar: number[] = [],
              sr: number[] = [];
            for (let k = 0; k < m; k++) {
              ar.push(j === k ? 0.8 + 0.1 * rng() : 0.02 * (rng() - 0.5));
              sr.push(j === k ? 0.5 + 0.5 * rng() : 0.01 * rng());
            }
            aRow.push(ar);
            sRow.push(sr);
          }
          A_data.push(aRow);
          b_data.push(Array.from({ length: m }, () => [rng() * 10]));
          S_data.push(sRow);
        }

        const composeBackward = (
          p: { A: np.Array; b: np.Array; S: np.Array },
          q: { A: np.Array; b: np.Array; S: np.Array },
        ) => {
          const newA = np.einsum("ij,jk->ik", q.A, p.A) as np.Array;
          using Ab = np.einsum("ij,jk->ik", q.A, p.b) as np.Array;
          const newB = Ab.add(q.b) as np.Array;
          using AS = np.einsum("ij,jk->ik", q.A, p.S) as np.Array;
          using qAT = np.transpose(q.A, [-2, -1]) as np.Array;
          using ASAT = np.einsum("ij,jk->ik", AS, qAT) as np.Array;
          const newS = ASAT.add(q.S) as np.Array;
          return { A: newA, b: newB, S: newS };
        };

        // WASM reference
        defaultDevice("wasm");
        using aWasm = np.array(A_data, { dtype: DType.Float32 });
        using bWasm = np.array(b_data, { dtype: DType.Float32 });
        using sWasm = np.array(S_data, { dtype: DType.Float32 });
        using wasmFn = jit((A: np.Array, b: np.Array, S: np.Array) =>
          lax.associativeScan(composeBackward, { A, b, S }, { reverse: rev }),
        );
        const wasmResult = wasmFn(aWasm, bWasm, sWasm) as {
          A: np.Array;
          b: np.Array;
          S: np.Array;
        };
        const wasmB = await wasmResult.b.data();
        const wasmA = await wasmResult.A.data();
        tree.dispose(wasmResult);

        // WebGPU under test
        defaultDevice("webgpu");
        clearCaches();
        using aGpu = np.array(A_data, { dtype: DType.Float32 });
        using bGpu = np.array(b_data, { dtype: DType.Float32 });
        using sGpu = np.array(S_data, { dtype: DType.Float32 });
        setDebug(0);
        using gpuFn = jit((A: np.Array, b: np.Array, S: np.Array) =>
          lax.associativeScan(composeBackward, { A, b, S }, { reverse: rev }),
        );
        const gpuResult = gpuFn(aGpu, bGpu, sGpu) as {
          A: np.Array;
          b: np.Array;
          S: np.Array;
        };
        setDebug(0);
        const gpuB = await gpuResult.b.data();
        const gpuA = await gpuResult.A.data();
        tree.dispose(gpuResult);
        // setDebug(0);

        defaultDevice("wasm");

        // Compare
        let maxDiff = 0;
        let maxIdx = -1;
        let corrupted = 0;
        for (let i = 0; i < N * m; i++) {
          const diff = Math.abs(wasmB[i] - gpuB[i]);
          if (diff > maxDiff) {
            maxDiff = diff;
            maxIdx = i;
          }
          if (diff > 0.1) corrupted++;
        }

        const stubSize = N % 64;
        console.log(
          `N=${N} stub=${stubSize}: maxDiff=${maxDiff.toFixed(4)} at t=${Math.floor(maxIdx / m)}, corrupted=${corrupted}/${N * m}`,
        );

        if (corrupted > 0) {
          // Print ACTUAL values at first corrupted position to diagnose
          for (let t = 0; t < N; t++) {
            let tMax = 0;
            for (let j = 0; j < m; j++) {
              tMax = Math.max(
                tMax,
                Math.abs(wasmB[t * m + j] - gpuB[t * m + j]),
              );
            }
            if (tMax > 0.1) {
              const w = Array.from({ length: m }, (_, j) =>
                wasmB[t * m + j].toFixed(4),
              );
              const g = Array.from({ length: m }, (_, j) =>
                gpuB[t * m + j].toFixed(4),
              );
              console.log(`  t=${t} b: wasm=[${w}] gpu=[${g}]`);
              // Also show A row 0 for this timestep
              const wA = Array.from({ length: m }, (_, j) =>
                wasmA[t * m * m + j].toFixed(4),
              );
              const gA = Array.from({ length: m }, (_, j) =>
                gpuA[t * m * m + j].toFixed(4),
              );
              console.log(`  t=${t} A[0]: wasm=[${wA}] gpu=[${gA}]`);
              if (t > 5) break; // enough
            }
          }
        }

        expect(maxDiff).toBeLessThan(1.0);
      });
    }
  }
});
