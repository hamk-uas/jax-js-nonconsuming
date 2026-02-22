/**
 * Deno WASM tests for M7.3: Multithreaded Kogge-Stone (associative scan).
 *
 * Verifies that parallel inner-round dispatch via WasmWorkerPool produces
 * correct results for large N. Deno has native SharedArrayBuffer support
 * (no crossOriginIsolated needed), so WasmWorkerPool is available.
 *
 * The parallel path kicks in when:
 *   - WasmWorkerPool exists (SharedArrayBuffer constructable)
 *   - Module registered on pool (async, first call triggers registration)
 *   - N >= PARALLEL_THRESHOLD (4096)
 *
 * First call always uses the monolithic WASM path (async registration in
 * progress). After a short delay, subsequent calls use the parallel path.
 *
 * Run with:
 *   pnpm build && deno test --no-check --allow-read --allow-env test/deno/parallel-assoc-scan.test.ts
 */

import {
  assertAlmostEquals,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";

import { withLeakCheck } from "./harness.ts";
import {
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
} from "../../dist/index.js";

// ---------------------------------------------------------------------------
// Helper: ensure WASM backend is initialized
// ---------------------------------------------------------------------------

let _wasmInitDone = false;

async function initWasm(): Promise<void> {
  if (!_wasmInitDone) {
    await init("wasm");
    defaultDevice("wasm");
    _wasmInitDone = true;
  }
}

// ---------------------------------------------------------------------------
// M7.3: Parallel associative scan tests
// ---------------------------------------------------------------------------

Deno.test({
  name: "M7.3: parallel assocScan cumsum correctness (N=8192)",
  fn: withLeakCheck(async () => {
    await initWasm();

    const N = 8192; // above PARALLEL_THRESHOLD (4096)
    const data = new Float32Array(N);
    for (let i = 0; i < N; i++) data[i] = 1.0;

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.add(a, b), x),
    );
    const input = np.array(data);

    // First call: triggers async registration, uses monolithic path
    const r1 = f(input);
    const d1 = await r1.data();
    assertEquals(d1.length, N);
    assertEquals(d1[0], 1);
    assertEquals(d1[N - 1], N);
    r1.dispose();

    // Wait for async pool registration
    await new Promise((r) => setTimeout(r, 50));

    // Second call: should use parallel path (workers registered)
    const r2 = f(input);
    const d2 = await r2.data();
    // Verify cumsum: result[i] = i + 1
    for (let i = 0; i < N; i++) {
      assertEquals(d2[i], i + 1, `element ${i} should be ${i + 1}`);
    }
    r2.dispose();

    input.dispose();
  }),
});

Deno.test({
  name: "M7.3: parallel assocScan cumprod correctness",
  fn: withLeakCheck(async () => {
    await initWasm();

    const N = 4096; // exactly at threshold
    const data = new Float32Array(N);
    for (let i = 0; i < N; i++) data[i] = 1.0 + 0.0001 * i;

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => a.mul(b), x),
    );
    const input = np.array(data);

    // First call (monolithic)
    const r1 = f(input);
    await r1.data(); // force execution
    r1.dispose();

    await new Promise((r) => setTimeout(r, 50));

    // Second call (parallel if N >= threshold and registered)
    const r2 = f(input);
    const d2 = await r2.data();

    // Compute sequential reference
    const ref = new Float32Array(N);
    ref[0] = data[0];
    for (let i = 1; i < N; i++) ref[i] = ref[i - 1] * data[i];

    // Check first and last few elements (f32 precision)
    for (let i = 0; i < 10; i++) {
      assertAlmostEquals(d2[i], ref[i], 1e-4, `elem ${i}`);
    }
    assertAlmostEquals(d2[N - 1], ref[N - 1], Math.abs(ref[N - 1]) * 1e-3);
    r2.dispose();

    input.dispose();
  }),
});

Deno.test({
  name: "M7.3: parallel assocScan reverse cumsum",
  fn: withLeakCheck(async () => {
    await initWasm();

    const N = 8192;
    const data = new Float32Array(N);
    for (let i = 0; i < N; i++) data[i] = 1.0;

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.add(a, b), x, {
        reverse: true,
      }),
    );
    const input = np.array(data);

    // Warm up and register
    const r1 = f(input);
    r1.dispose();
    await new Promise((r) => setTimeout(r, 50));

    // Parallel call
    const r2 = f(input);
    const d2 = await r2.data();
    // Reverse cumsum of all-ones: result[i] = N - i
    for (let i = 0; i < N; i++) {
      assertEquals(d2[i], N - i, `element ${i} should be ${N - i}`);
    }
    r2.dispose();

    input.dispose();
  }),
});

Deno.test({
  name: "M7.3: parallel assocScan multi-element (2-D, axis=0)",
  fn: withLeakCheck(async () => {
    await initWasm();

    const N = 8192;
    const cols = 4;
    // Shape [N, cols], scan along axis 0
    const data = new Float32Array(N * cols);
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < cols; c++) {
        data[i * cols + c] = (c + 1) * 1.0; // col 0=1, col 1=2, col 2=3, col 3=4
      }
    }

    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.add(a, b), x),
    );
    using flat = np.array(data);
    const input = flat.reshape([N, cols]);

    // Warm up and register
    const r1 = f(input);
    r1.dispose();
    await new Promise((r) => setTimeout(r, 50));

    // Parallel path
    const r2 = f(input);
    const d2 = await r2.data();
    // result[i, c] = (i+1) * (c+1)
    for (let i = 0; i < 10; i++) {
      for (let c = 0; c < cols; c++) {
        assertEquals(d2[i * cols + c], (i + 1) * (c + 1), `[${i},${c}]`);
      }
    }
    r2.dispose();

    input.dispose();
  }),
});

Deno.test({
  name: "M7.3: parallel assocScan repeated calls produce consistent results",
  fn: withLeakCheck(async () => {
    await initWasm();

    const N = 8192;
    using f = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => np.add(a, b), x),
    );

    // First call triggers registration
    const x0 = np.array(new Float32Array(N).fill(1));
    const r0 = f(x0);
    r0.dispose();
    x0.dispose();
    await new Promise((r) => setTimeout(r, 50));

    // Run 3 calls with different values to verify stability
    for (let trial = 0; trial < 3; trial++) {
      const val = trial + 1;
      const x = np.array(new Float32Array(N).fill(val));
      const result = f(x);
      const data = await result.data();

      // cumsum of constant val: result[i] = (i+1) * val
      for (let i = 0; i < 10; i++) {
        assertEquals(data[i], (i + 1) * val, `trial ${trial} elem ${i}`);
      }
      assertEquals(data[N - 1], N * val, `trial ${trial} last elem`);

      result.dispose();
      x.dispose();
    }
  }),
});
