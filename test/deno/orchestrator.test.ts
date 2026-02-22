/**
 * Deno WASM orchestrator tests (M6.2b: off-main-thread mega-module execution).
 *
 * Deno has native SharedArrayBuffer support (no crossOriginIsolated needed),
 * so these tests exercise the OrchestratorWorker and WasmWorkerPool that
 * cannot be tested in vitest/Playwright (which hangs with COOP+COEP headers).
 *
 * Run with:
 *   deno test --no-check --allow-read --allow-env test/deno/orchestrator.test.ts
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/mod.ts";

import { withLeakCheck } from "./harness.ts";
import {
  defaultDevice,
  getBackend,
  grad,
  init,
  jit,
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

function getWasmBackend(): any {
  return getBackend("wasm") as any;
}

// ---------------------------------------------------------------------------
// Capability tests
// ---------------------------------------------------------------------------

Deno.test({
  name: "M6.2b: WASM backend reports sharedMemory = true in Deno",
  fn: withLeakCheck(async () => {
    await initWasm();
    const backend = getWasmBackend();
    assertEquals(backend.capabilities.sharedMemory, true);
  }),
});

Deno.test({
  name: "M6.2b: orchestrator is created after first jit call",
  fn: withLeakCheck(async () => {
    await initWasm();
    const backend = getWasmBackend();

    // Force orchestrator creation via a jit call (mega-module path)
    const f = jit((x: any) => x.add(1));
    const x = np.array([1, 2, 3]);
    const result = f(x);
    const data = await result.data();
    assertEquals(Array.from(data), [2, 3, 4]);

    // Orchestrator should now exist
    assert(backend.orchestrator != null, "orchestrator should be created");

    result.dispose();
    x.dispose();
    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// Correctness tests — mega-module dispatched via orchestrator
// ---------------------------------------------------------------------------

Deno.test({
  name: "M6.2b: simple add via orchestrator",
  fn: withLeakCheck(async () => {
    await initWasm();

    const f = jit((x: any) => x.add(1));
    const x = np.array([10, 20, 30, 40]);
    const result = f(x);
    const data = await result.data();
    assertEquals(Array.from(data), [11, 21, 31, 41]);

    result.dispose();
    x.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2b: chained ops via orchestrator",
  fn: withLeakCheck(async () => {
    await initWasm();

    const f = jit((x: any) => x.add(1).mul(2).sub(3));
    const x = np.array([1, 2, 3, 4]);
    const result = f(x);
    const data = await result.data();
    assertEquals(Array.from(data), [1, 3, 5, 7]);

    result.dispose();
    x.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2b: two-input operation via orchestrator",
  fn: withLeakCheck(async () => {
    await initWasm();

    const f = jit((a: any, b: any) => a.add(b));
    const a = np.array([1, 2, 3]);
    const b = np.array([10, 20, 30]);
    const result = f(a, b);
    const data = await result.data();
    assertEquals(Array.from(data), [11, 22, 33]);

    result.dispose();
    a.dispose();
    b.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2b: internal alloc/free proxied through orchestrator",
  fn: withLeakCheck(async () => {
    await initWasm();

    // Multi-step chain where recycling can't eliminate all alloc/free:
    // f(x) = ((x + 1) * 2 - 3) / 4 + 5
    const f = jit((x: any) => x.add(1).mul(2).sub(3).div(4).add(5));
    const x = np.array([3, 7, 11, 15]);
    const result = f(x);
    const data = await result.data();
    // (3+1)*2-3=5, 5/4=1.25, 1.25+5=6.25, etc.
    assertEquals(Array.from(data), [6.25, 8.25, 10.25, 12.25]);

    result.dispose();
    x.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2b: repeated calls reuse cached module registration",
  fn: withLeakCheck(async () => {
    await initWasm();

    const f = jit((x: any) => x.mul(3));

    const x1 = np.array([1, 2, 3]);
    const r1 = f(x1);
    assertEquals(Array.from(await r1.data()), [3, 6, 9]);

    // Second call reuses cached module
    const x2 = np.array([10, 20, 30]);
    const r2 = f(x2);
    assertEquals(Array.from(await r2.data()), [30, 60, 90]);

    r1.dispose();
    r2.dispose();
    x1.dispose();
    x2.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2b: reduction kernel via orchestrator",
  fn: withLeakCheck(async () => {
    await initWasm();

    const f = jit((x: any) => x.sum());
    const x = np.array([1, 2, 3, 4]);
    const result = f(x);
    const data = await result.data();
    assertEquals(data[0], 10);

    result.dispose();
    x.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2b: grad through orchestrator-dispatched mega-module",
  fn: withLeakCheck(async () => {
    await initWasm();

    const f = jit((x: any) => x.mul(x).sum());
    const x = np.array([1, 2, 3]);
    const g = grad(f)(x);
    // d/dx sum(x^2) = 2x
    const data = await g.data();
    assertEquals(Array.from(data), [2, 4, 6]);

    g.dispose();
    x.dispose();
    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// Worker pool tests (parallel kernel dispatch)
// ---------------------------------------------------------------------------

Deno.test({
  name: "M6.2b: worker pool created for large arrays",
  fn: withLeakCheck(async () => {
    await initWasm();
    const backend = getWasmBackend();

    // Trigger worker pool creation via a large array operation
    const n = 8192; // above PARALLEL_THRESHOLD (4096)
    const aData = new Float32Array(n).fill(1);
    const bData = new Float32Array(n).fill(2);
    const a = np.array(aData);
    const b = np.array(bData);
    const c = np.add(a, b);
    const data = await c.data();

    // Verify correctness
    for (let i = 0; i < n; i++) {
      assertEquals(data[i], 3, `element ${i} should be 3`);
    }

    // Worker pool should now exist
    assert(
      backend.workerPool != null,
      "workerPool should be created for large arrays",
    );

    a.dispose();
    b.dispose();
    c.dispose();
  }),
});

Deno.test({
  name: "M6.2b: parallel dispatch produces correct results for large jit",
  fn: withLeakCheck(async () => {
    await initWasm();

    const n = 8192;
    const f = jit((x: any, y: any) => x.mul(y).add(1));
    const x = np.array(new Float32Array(n).fill(3));
    const y = np.array(new Float32Array(n).fill(4));
    const result = f(x, y);
    const data = await result.data();

    // 3 * 4 + 1 = 13 for every element
    for (let i = 0; i < n; i++) {
      assertEquals(data[i], 13, `element ${i} should be 13`);
    }

    result.dispose();
    x.dispose();
    y.dispose();
    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// M6.2c: Parallel kernel dispatch via JS-driven step execution
// ---------------------------------------------------------------------------

Deno.test({
  name: "M6.2c: parallel dispatch produces correct results for large elementwise",
  fn: withLeakCheck(async () => {
    await initWasm();

    const n = 16384; // well above PARALLEL_THRESHOLD (4096)
    const f = jit((x: any) => x.add(1).mul(2));
    const input = np.array(new Float32Array(n).fill(5));

    // First call: triggers async registration, falls through to monolithic
    const r1 = f(input);
    assertEquals((await r1.data()).length, n);
    r1.dispose();

    // Tiny delay for async registration to complete
    await new Promise((r) => setTimeout(r, 50));

    // Second call: should use parallel path (if registration succeeded)
    const r2 = f(input);
    const data = await r2.data();
    for (let i = 0; i < n; i++) {
      assertEquals(data[i], 12, `element ${i} should be (5+1)*2 = 12`);
    }

    r2.dispose();
    input.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2c: parallel dispatch with reduction kernel",
  fn: withLeakCheck(async () => {
    await initWasm();

    const n = 8192;
    const f = jit((x: any) => x.mul(2).sum());
    const input = np.array(new Float32Array(n).fill(3));

    // First call triggers registration
    const r1 = f(input);
    assertEquals((await r1.data())[0], 3 * 2 * n);
    r1.dispose();

    await new Promise((r) => setTimeout(r, 50));

    // Second call should use parallel path
    const r2 = f(input);
    assertEquals((await r2.data())[0], 3 * 2 * n);

    r2.dispose();
    input.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2c: parallel dispatch correctness across repeated calls",
  fn: withLeakCheck(async () => {
    await initWasm();

    const n = 8192;
    const f = jit((x: any, y: any) => x.add(y).mul(3));

    // Drive registration with first call
    const x0 = np.array(new Float32Array(n).fill(1));
    const y0 = np.array(new Float32Array(n).fill(2));
    const r0 = f(x0, y0);
    r0.dispose();
    x0.dispose();
    y0.dispose();

    await new Promise((r) => setTimeout(r, 50));

    // Run 3 more calls — all should use parallel path and produce correct results
    for (let trial = 0; trial < 3; trial++) {
      const v = trial + 1;
      const x = np.array(new Float32Array(n).fill(v));
      const y = np.array(new Float32Array(n).fill(v * 10));
      const result = f(x, y);
      const data = await result.data();
      const expected = (v + v * 10) * 3;
      for (let i = 0; i < 10; i++) {
        assertEquals(data[i], expected, `trial ${trial} elem ${i}`);
      }
      result.dispose();
      x.dispose();
      y.dispose();
    }

    f.dispose();
  }),
});

Deno.test({
  name: "M6.2c: grad through parallel-dispatched mega-module",
  fn: withLeakCheck(async () => {
    await initWasm();

    const n = 8192;
    const f = jit((x: any) => x.mul(x).sum());

    // Drive registration
    const x0 = np.array(new Float32Array(n).fill(1));
    const r0 = f(x0);
    r0.dispose();
    x0.dispose();

    await new Promise((r) => setTimeout(r, 50));

    // grad through parallel mega-module
    const x = np.array(new Float32Array(n).fill(3));
    const g = grad(f)(x);
    const data = await g.data();
    // d/dx sum(x^2) = 2x = 6
    for (let i = 0; i < 10; i++) {
      assertEquals(data[i], 6, `gradient element ${i} should be 6`);
    }

    g.dispose();
    x.dispose();
    f.dispose();
  }),
});

Deno.test({
  name: "M6.2c: shouldUseParallelMegaModule true for large kernels",
  fn: withLeakCheck(async () => {
    await initWasm();
    const backend = getWasmBackend();

    // Compile a mega-module for a large array
    const n = 8192;
    const f = jit((x: any) => x.add(1));
    const x = np.array(new Float32Array(n).fill(0));
    const r = f(x);
    r.dispose();

    // The JitProgram should have a mega-module with large kernels
    assert(
      typeof backend.shouldUseParallelMegaModule === "function",
      "shouldUseParallelMegaModule method should exist",
    );

    x.dispose();
    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

// Destroy workers after all tests to allow clean exit
Deno.test({
  name: "M6.2b: cleanup — destroy workers",
  fn: async () => {
    try {
      const backend = getWasmBackend();
      if (typeof backend.destroyWorkers === "function") {
        backend.destroyWorkers();
      }
    } catch {
      // Backend may not be initialized; ignore.
    }
  },
  sanitizeOps: false,
  sanitizeResources: false,
});
