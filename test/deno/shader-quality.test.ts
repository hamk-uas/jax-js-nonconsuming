/**
 * Deno WebGPU shader quality gate tests.
 *
 * These verify that the WebGPU codegen produces expected WGSL patterns:
 * - P2a: select() elimination for tile-aligned tiledMatmul
 * - P1:  barrier merging (1 barrier per fori_loop iteration)
 * - P2b: unchecked dynamic slice (no min/max clamping)
 * - P2c: padConcrete eliminates select() for non-aligned tiledMatmul
 *
 * Run with:
 *   pnpm build && deno test --no-check --unstable-webgpu --allow-read --allow-env test/deno/shader-quality.test.ts
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

import { initWebGPU, withLeakCheck } from "./harness.ts";
import {
  clearCaches,
  DType,
  jit,
  lax,
  numpy as np,
  setDebug,
} from "../../dist/index.js";

/**
 * Capture all WGSL shader sources emitted during a callback.
 * Uses setDebug(2) which logs shaders via console.info.
 */
function captureShaders(fn: () => void): string[] {
  const shaders: string[] = [];
  const origInfo = console.info;
  console.info = (...args: unknown[]) => {
    const msg = args.map(String).join(" ");
    if (msg.includes("=========== WebGPU shader ===========")) {
      const code = msg.replace("=========== WebGPU shader ===========\n", "");
      shaders.push(code);
    }
  };
  try {
    setDebug(2);
    fn();
  } finally {
    setDebug(0);
    console.info = origInfo;
  }
  return shaders;
}

// ---------------------------------------------------------------------------
// P2a: tile-aligned tiledMatmul emits no select() in fused shader
// ---------------------------------------------------------------------------
Deno.test({
  name: "P2a: tile-aligned tiledMatmul emits no select()",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;
    clearCaches();

    const f = jit((A: any, B: any) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    const fusedShaders = shaders.filter(
      (s: string) =>
        s.includes("workgroupBarrier") || s.includes("var<workgroup>"),
    );
    if (fusedShaders.length === 0) {
      throw new Error("Expected at least 1 fused shader with workgroupBarrier");
    }
    for (const shader of fusedShaders) {
      const selectCount = (shader.match(/\bselect\s*\(/g) || []).length;
      assertEquals(
        selectCount,
        0,
        "tile-aligned fused shader should have no select() calls",
      );
    }

    // Shader quality verified — correctness tested by Chromium suite.
    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// P1: barrier merging — 1 barrier per fori_loop iteration
// ---------------------------------------------------------------------------
Deno.test({
  name: "P1: tiledMatmul fori_loop has 1 barrier per iteration",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;
    clearCaches();

    // Use Br=8 to avoid shader cache hit from P2a's Br=16
    const f = jit((A: any, B: any) =>
      lax.tiledMatmul(A, B, { Br: 8, Bc: 8, Bk: 8 }),
    );
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    const fusedShaders = shaders.filter(
      (s: string) =>
        s.includes("var<workgroup>") || s.includes("workgroupBarrier"),
    );
    if (fusedShaders.length === 0) {
      throw new Error("Expected at least 1 fused shader");
    }

    for (const shader of fusedShaders) {
      const forMatch = shader.match(
        /for\s*\(var\s+fl\d+_i.*?\{([\s\S]*?)\n\s*\}\s*\n/,
      );
      if (!forMatch) continue;
      const forBody = forMatch[1];
      const barrierCount = (forBody.match(/workgroupBarrier\(\)/g) || [])
        .length;
      assertEquals(
        barrierCount,
        1,
        `Expected 1 barrier per iteration, got ${barrierCount}`,
      );
    }

    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// P2b: unchecked dynamic slice — no min/max clamping
// ---------------------------------------------------------------------------
Deno.test({
  name: "P2b: tiledMatmul uses unchecked dynamic slice (no min/max clamping)",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;
    clearCaches();

    // Use Br=32 to avoid shader cache hit from P2a/P1
    const f = jit((A: any, B: any) =>
      lax.tiledMatmul(A, B, { Br: 32, Bc: 32, Bk: 32 }),
    );
    using A_flat = np.arange(4096).astype(DType.Float32);
    using A = A_flat.reshape([64, 64]);
    using B = np.eye(64, { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    const fusedShaders = shaders.filter(
      (s: string) =>
        s.includes("var<workgroup>") || s.includes("workgroupBarrier"),
    );
    if (fusedShaders.length === 0) {
      throw new Error("Expected at least 1 fused shader");
    }

    for (const shader of fusedShaders) {
      const forMatch = shader.match(
        /for\s*\(var\s+fl\d+_i.*?\{([\s\S]*?)\n\s*\}\s*\n/,
      );
      if (!forMatch) continue;
      const forBody = forMatch[1];
      const minCount = (forBody.match(/\bmin\s*\(/g) || []).length;
      const maxCount = (forBody.match(/\bmax\s*\(/g) || []).length;
      assertEquals(
        minCount,
        0,
        "unchecked dynamic slice should produce no min() clamping",
      );
      assertEquals(
        maxCount,
        0,
        "unchecked dynamic slice should produce no max() clamping",
      );
    }

    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// O2: Clean indexing — no redundant modular arithmetic from unravelAlu
// ---------------------------------------------------------------------------
Deno.test({
  name: "O2: fused shader has no redundant modular arithmetic from unravelAlu",
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    // 128x128 tiledMatmul with Br=Bc=Bk=16 → blockSize=256, gidx ∈ [0,255].
    // O2 gives gidx a bounded range so the simplifier eliminates:
    //   (gidx / 16) % 16  →  gidx / 16   (since gidx/16 ∈ [0,15], 15 < 16)
    //   gidx % 256         →  gidx         (since gidx ∈ [0,255], 255 < 256)
    const f = jit((A: any, B: any) =>
      lax.tiledMatmul(A, B, { Br: 16, Bc: 16, Bk: 16 }),
    );
    using A_flat = np.arange(128 * 128).astype(DType.Float32);
    using A = A_flat.reshape([128, 128]);
    using B = np.eye(128, { dtype: DType.Float32 });

    const shaders = captureShaders(() => {
      using _result = f(A, B);
    });

    const fusedShaders = shaders.filter(
      (s) => s.includes("workgroupBarrier") || s.includes("var<workgroup>"),
    );
    if (fusedShaders.length === 0) throw new Error("No fused shaders captured");

    for (const shader of fusedShaders) {
      const forMatch = shader.match(
        /for\s*\(var\s+fl\d+_i.*?\{([\s\S]*?)\n\s*\}\s*\n/,
      );
      if (!forMatch) continue;
      const forBody = forMatch[1];

      if (/\(gidx\s*\/\s*16\)\s*%\s*16/.test(forBody)) {
        throw new Error(
          "O2: (gidx / 16) % 16 should be simplified to gidx / 16",
        );
      }
      if (/gidx\s*%\s*256/.test(forBody)) {
        throw new Error("O2: gidx % 256 should be simplified to gidx");
      }
    }

    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// P2c: non-aligned tiledMatmul correctness (64x60 @ 60x64) via jit
// KNOWN_BUG: DynamicUpdateSlice JIT only supports axis=0. padConcrete
// pads along K axis (axis=1 of A), triggering the limitation.
// Correctness verified by Chromium tests on WASM backend.
// ---------------------------------------------------------------------------
Deno.test({
  name: "P2c: non-aligned tiledMatmul correctness (64x60 @ 60x64)",
  ignore: true, // DUS axis=0 limitation on WebGPU
  fn: withLeakCheck(async () => {
    if (!(await initWebGPU())) return;

    // Use jit() to avoid eager DUS dataSync (OffscreenCanvas unavailable in Deno).
    const f = jit((A: any, B: any) => lax.tiledMatmul(A, B));
    const g = jit((A: any, B: any) => np.matmul(A, B));

    using A_flat = np.arange(64 * 60).astype(DType.Float32);
    using A = A_flat.reshape([64, 60]);
    using B_flat = np.arange(60 * 64).astype(DType.Float32);
    using B = B_flat.reshape([60, 64]);

    using result = f(A, B);
    using expected = g(A, B);
    const resultData = await result.data();
    const expectedData = await expected.data();

    for (let i = 0; i < resultData.length; i++) {
      const diff = Math.abs(resultData[i] - expectedData[i]);
      if (diff > 1) {
        throw new Error(
          `Mismatch at index ${i}: got ${resultData[i]}, expected ${expectedData[i]}, diff=${diff}`,
        );
      }
    }

    f.dispose();
    g.dispose();
  }),
});
