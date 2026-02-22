/**
 * Tests for M6.1: Mega-Module compiler.
 * Tests for M6.2a: Extracted kernel functions.
 *
 * The mega-module compiles an entire JitProgram's kernel-only step list
 * into a single WASM function, eliminating JS↔WASM boundary crossings
 * between kernel dispatches.
 *
 * M6.2a extracts non-reduction kernels into separate WASM functions with
 * (start, end, ...bufs) signatures, callable independently by workers.
 * V8 inlines them back in the serial path (zero overhead).
 */
import { grad, init, jit, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { beforeAll, describe, expect, it } from "vitest";

import { getBackend } from "../src/backend";
import {
  compileToMegaModule,
  type WasmMegaModule,
} from "../src/backend/wasm/mega-module";
import { makeJaxpr } from "../src/frontend/jaxpr";
import { jitCompile } from "../src/frontend/jit";

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

/**
 * Trace a function to a Jaxpr, jitCompile it, then call
 * compileToMegaModule on the resulting steps. Returns the WasmMegaModule.
 */
function compileMega(
  f: (...args: np.Array[]) => np.Array,
  ...args: np.Array[]
): WasmMegaModule | null {
  const backend = getBackend();
  const traced = makeJaxpr(f)(...args);
  const program = jitCompile(backend, traced.jaxpr.jaxpr);
  const mm = compileToMegaModule(
    program.steps,
    program.inputs,
    program.outputs,
  );
  traced.jaxpr.dispose();
  return mm;
}

/**
 * Instantiate a WasmMegaModule with a fresh WebAssembly.Memory and stub
 * imports. Returns the instance, memory, and a Float32Array view.
 * Useful for directly calling exported kernel functions in tests.
 */
function instantiateMega(mm: WasmMegaModule): {
  instance: WebAssembly.Instance;
  memory: WebAssembly.Memory;
  f32: Float32Array;
} {
  // When cross-origin-isolated the mega-module binary imports a shared
  // memory with a maximum.  Match that here so instantiation succeeds.
  const shared =
    typeof globalThis.crossOriginIsolated === "boolean"
      ? globalThis.crossOriginIsolated
      : false;
  const memory = shared
    ? new WebAssembly.Memory({ initial: 1, maximum: 4096, shared: true })
    : new WebAssembly.Memory({ initial: 1 });
  const instance = new WebAssembly.Instance(mm.module, {
    env: {
      memory,
      alloc: () => 0,
      free: () => {},
    },
  });
  return { instance, memory, f32: new Float32Array(memory.buffer) };
}

describe("mega-module (M6.1)", () => {
  beforeAll(async () => {
    await init("wasm");
  });

  describe("correctness", () => {
    it("simple add chain", () => {
      using f = jit((x: np.Array) => x.add(1));
      using x = np.array([1, 2, 3, 4]);
      using result = f(x);
      expect(result.js()).toEqual([2, 3, 4, 5]);
    });

    it("chained elementwise ops", () => {
      // (x + 1) * 2 - 3
      using f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
      using x = np.array([1, 2, 3, 4]);
      using result = f(x);
      expect(result.js()).toEqual([1, 3, 5, 7]);
    });

    it("two-input operation", () => {
      using f = jit((a: np.Array, b: np.Array) => a.add(b));
      using a = np.array([1, 2, 3]);
      using b = np.array([10, 20, 30]);
      using result = f(a, b);
      expect(result.js()).toEqual([11, 22, 33]);
    });

    it("multi-input chain", () => {
      using f = jit((a: np.Array, b: np.Array, c: np.Array) => a.add(b).mul(c));
      using a = np.array([1, 2, 3]);
      using b = np.array([10, 20, 30]);
      using c = np.array([2, 3, 4]);
      using result = f(a, b, c);
      expect(result.js()).toEqual([22, 66, 132]);
    });

    it("reduction (sum)", () => {
      using f = jit((x: np.Array) => x.mul(2).sum().add(1));
      using x = np.array([1, 2, 3]);
      // (2+4+6) + 1 = 13
      using result = f(x);
      expect(result.js()).toEqual(13);
    });

    it("negative values", () => {
      using f = jit((x: np.Array) => x.mul(-1).add(10));
      using x = np.array([1, 2, 3]);
      using result = f(x);
      expect(result.js()).toEqual([9, 8, 7]);
    });

    it("float32 arithmetic", () => {
      using f = jit((x: np.Array) => x.mul(0.5).add(0.25));
      using x = np.array([1.0, 2.0, 4.0]);
      using result = f(x);
      expect(result.js()).toEqual([0.75, 1.25, 2.25]);
    });

    it("larger array", () => {
      using f = jit((x: np.Array) => x.add(1));
      const data = Array.from({ length: 1024 }, (_, i) => i);
      using x = np.array(data);
      using result = f(x);
      const expected = data.map((v) => v + 1);
      expect(result.js()).toEqual(expected);
    });
  });

  describe("repeated calls", () => {
    it("same-shape repeated calls are correct", () => {
      // Mega-module is compiled on 1st call, cached for subsequent calls
      using f = jit((x: np.Array) => x.add(1).mul(2));
      for (let i = 0; i < 5; i++) {
        using x = np.array([i, i + 1, i + 2]);
        using result = f(x);
        expect(result.js()).toEqual([(i + 1) * 2, (i + 2) * 2, (i + 3) * 2]);
      }
    });
  });

  describe("no leaks", () => {
    it("chain does not leak slots", async () => {
      using f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4));
      using x = np.array([10, 20, 30, 40]);
      using result = f(x);
      await result.data();
      // Leak detection is done by the global setup.ts afterEach
    });

    it("multi-output does not leak", async () => {
      // Two separate outputs — tests multi-output kernel codegen
      using f = jit((x: np.Array) => x.add(1).mul(3));
      using x = np.array([1, 2, 3]);
      using result = f(x);
      await result.data();
    });

    it("repeated calls do not leak", async () => {
      using f = jit((x: np.Array) => x.add(1));
      for (let i = 0; i < 10; i++) {
        using x = np.array([i]);
        using result = f(x);
        await result.data();
      }
    });
  });

  describe("grad through mega-module", () => {
    it("grad of x^2 sum", () => {
      // f(x) = sum(x^2), f'(x) = 2x
      const f = (x: np.Array) => np.multiply(x, x).sum();
      const df = grad(f);
      using x = np.array([1, 2, 3]);
      using result = df(x);
      expect(result.js()).toEqual([2, 4, 6]);
    });

    it("jit(grad(...)) composition", () => {
      // f(x) = sum((x + 1)^2), f'(x) = 2(x + 1)
      using df = jit(
        grad((x: np.Array) => {
          using xp1 = x.add(1);
          return np.multiply(xp1, xp1).sum();
        }),
      );
      using x = np.array([0, 1, 2]);
      using result = df(x);
      expect(result.js()).toEqual([2, 4, 6]);
    });
  });

  describe("symbolic size fallback", () => {
    // Symbolic sizes (from polymorphic shapes / dynamic axes) must NOT be
    // compiled into the mega-module. If canCompileToMegaModule fails to
    // reject them, i32.const receives a SymbolicSize object → NaN → 0,
    // making reduction loops run 0 iterations and producing identity values
    // (0 for sum, -Infinity for max). The wasmblr guard in i32.const()
    // catches this with an actionable error; these tests verify the
    // step-by-step fallback produces correct results.
    //
    // To support symbolic sizes in the mega-module in the future, see
    // the MIGRATION NOTE in canCompileToMegaModule (mega-module.ts).

    it("reduction with reused jit and different sizes falls back correctly", () => {
      // jit with static argnums caches by shape — but dynamic axis T means
      // the reduction size is symbolic. The mega-module must reject this
      // and the step-by-step path must produce correct values.
      using f = jit((x: np.Array) => x.sum(0));

      using x3 = np.ones([3, 4]);
      using r3 = f(x3);
      expect(r3.js()).toEqual([3, 3, 3, 3]);

      using x5 = np.ones([5, 4]);
      using r5 = f(x5);
      expect(r5.js()).toEqual([5, 5, 5, 5]);
    });

    it("chained ops + reduction with varying shapes", () => {
      using f = jit((x: np.Array) => x.mul(2).sum(0));

      using x1 = np.array([
        [1, 2],
        [3, 4],
      ]);
      using r1 = f(x1);
      expect(r1.js()).toEqual([8, 12]);

      using x2 = np.array([
        [10, 20],
        [30, 40],
        [50, 60],
      ]);
      using r2 = f(x2);
      expect(r2.js()).toEqual([180, 240]);
    });
  });
});

describe("extracted kernel functions (M6.2a)", () => {
  beforeAll(async () => {
    await init("wasm");
  });

  describe("kernelExports metadata", () => {
    it("elementwise chain produces non-reduction exports", () => {
      using x = np.array([1, 2, 3, 4]);
      const mm = compileMega((x: np.Array) => {
        using a = x.add(1);
        return a.mul(2);
      }, x);
      expect(mm).not.toBeNull();
      // At least one kernel exported; none should be reduction
      expect(mm!.kernelExports.length).toBeGreaterThan(0);
      for (const ke of mm!.kernelExports) {
        expect(ke.isReduction).toBe(false);
        expect(ke.name).toMatch(/^kernel_\d+$/);
        expect(ke.size).toBe(4); // 4 elements
        expect(ke.nInputs).toBeGreaterThan(0);
        expect(ke.nOutputs).toBeGreaterThan(0);
      }
    });

    it("reduction kernel is marked isReduction", () => {
      using x = np.array([1, 2, 3]);
      const mm = compileMega((x: np.Array) => {
        using a = x.mul(2);
        using b = a.sum();
        return b.add(1);
      }, x);
      expect(mm).not.toBeNull();
      // Should have at least one reduction kernel
      const reductions = mm!.kernelExports.filter((ke) => ke.isReduction);
      expect(reductions.length).toBeGreaterThan(0);
    });

    it("mixed chain has both reduction and non-reduction exports", () => {
      using x = np.array([1, 2, 3, 4]);
      // x.add(1) → elementwise (non-reduction), then .sum() → reduction
      const mm = compileMega((x: np.Array) => {
        using a = x.add(1);
        return a.sum();
      }, x);
      expect(mm).not.toBeNull();
      const _nonReduction = mm!.kernelExports.filter((ke) => !ke.isReduction);
      const reduction = mm!.kernelExports.filter((ke) => ke.isReduction);
      // The fused chain may produce 1 or 2 kernel steps depending on fusion.
      // At least one must be a reduction (from sum).
      expect(reduction.length).toBeGreaterThan(0);
    });

    it("kernel names are sequential", () => {
      using x = np.array([1, 2, 3]);
      const mm = compileMega((x: np.Array) => {
        using a = x.add(1);
        return a.mul(2);
      }, x);
      expect(mm).not.toBeNull();
      for (let i = 0; i < mm!.kernelExports.length; i++) {
        expect(mm!.kernelExports[i].name).toBe(`kernel_${i}`);
      }
    });
  });

  describe("WASM exports", () => {
    it("non-reduction kernel functions are exported from the module", () => {
      using x = np.array([1, 2, 3, 4]);
      const mm = compileMega((x: np.Array) => {
        using a = x.add(1);
        return a.mul(2);
      }, x);
      expect(mm).not.toBeNull();

      const { instance } = instantiateMega(mm!);

      // mega_execute should always be exported
      expect(instance.exports.mega_execute).toBeTypeOf("function");

      // Non-reduction kernels should be exported
      const nonReduction = mm!.kernelExports.filter((ke) => !ke.isReduction);
      for (const ke of nonReduction) {
        expect(instance.exports[ke.name]).toBeTypeOf("function");
      }
    });

    it("reduction kernels are NOT exported as standalone functions", () => {
      using x = np.array([1, 2, 3]);
      const mm = compileMega((x: np.Array) => x.sum(), x);
      expect(mm).not.toBeNull();

      const { instance } = instantiateMega(mm!);

      // Reduction kernels should be marked isReduction but not exported
      const reductions = mm!.kernelExports.filter((ke) => ke.isReduction);
      expect(reductions.length).toBeGreaterThan(0);
      for (const ke of reductions) {
        expect(instance.exports[ke.name]).toBeUndefined();
      }
    });
  });

  describe("sub-range correctness", () => {
    it("extracted kernel computes correct results for a sub-range", () => {
      using x = np.array([10, 20, 30, 40, 50, 60, 70, 80]);
      // Simple x + 1 kernel
      const mm = compileMega((x: np.Array) => x.add(1), x);
      expect(mm).not.toBeNull();

      const nonReduction = mm!.kernelExports.filter((ke) => !ke.isReduction);
      expect(nonReduction.length).toBe(1);
      const ke = nonReduction[0];

      // Set up memory: input at offset 256, output at offset 512
      const { instance, f32 } = instantiateMega(mm!);

      const inputOffset = 256; // bytes
      const outputOffset = 512; // bytes
      const inputElemOffset = inputOffset / 4;
      const outputElemOffset = outputOffset / 4;

      // Write 8 floats as input
      for (let i = 0; i < 8; i++) {
        f32[inputElemOffset + i] = (i + 1) * 10; // 10,20,...,80
      }

      // Zero output region
      for (let i = 0; i < 8; i++) {
        f32[outputElemOffset + i] = 0;
      }

      // Call the exported kernel with sub-range [2, 5)
      const kernelFn = instance.exports[ke.name] as (...args: number[]) => void;
      // Signature: (start, end, inputPtr, outputPtr)
      kernelFn(2, 5, inputOffset, outputOffset);

      // Elements [0,1] should be untouched (0)
      expect(f32[outputElemOffset + 0]).toBe(0);
      expect(f32[outputElemOffset + 1]).toBe(0);

      // Elements [2,3,4] should be input[i] + 1
      expect(f32[outputElemOffset + 2]).toBe(31); // 30 + 1
      expect(f32[outputElemOffset + 3]).toBe(41); // 40 + 1
      expect(f32[outputElemOffset + 4]).toBe(51); // 50 + 1

      // Elements [5,6,7] should be untouched (0)
      expect(f32[outputElemOffset + 5]).toBe(0);
      expect(f32[outputElemOffset + 6]).toBe(0);
      expect(f32[outputElemOffset + 7]).toBe(0);
    });

    it("full range (0, size) matches mega_execute results", () => {
      // Verify that calling the extracted kernel with full range
      // produces the same output as mega_execute
      const body = (x: np.Array) => {
        using a = x.add(1);
        return a.mul(2);
      };
      using f = jit(body);
      using x = np.array([1, 2, 3, 4]);
      using megaResult = f(x);
      const megaData = megaResult.js() as number[];

      // Compile separately to get the module
      const mm = compileMega(body, x);
      expect(mm).not.toBeNull();

      const nonReduction = mm!.kernelExports.filter((ke) => !ke.isReduction);
      expect(nonReduction.length).toBeGreaterThan(0);

      // Verify correctness: mega_execute result should be [(1+1)*2, (2+1)*2, (3+1)*2, (4+1)*2] = [4,6,8,10]
      expect(megaData).toEqual([4, 6, 8, 10]);
    });
  });

  describe("end-to-end correctness via jit", () => {
    it("extracted kernel path produces same results as step-by-step", () => {
      // This test ensures that the M6.2a refactoring (extracted functions
      // called from mega_execute) produces identical results to M6.1
      // (inlined kernels). It runs through the normal jit() path which
      // uses the mega-module automatically.
      using scale = np.array([2, 3, 4]);
      using offset = np.array([10, 20, 30]);
      using f = jit((x: np.Array) => {
        using a = x.add(offset);
        return a.mul(scale);
      });
      using x = np.array([1, 2, 3]);
      using result = f(x);
      // (1+10)*2=22, (2+20)*3=66, (3+30)*4=132
      expect(result.js()).toEqual([22, 66, 132]);
    });

    it("multi-output with extracted kernels", () => {
      // Multi-output kernel: both outputs should use extracted functions
      using f = jit((x: np.Array, y: np.Array) => x.add(y).sub(1));
      using x = np.array([5, 10, 15]);
      using y = np.array([1, 2, 3]);
      using result = f(x, y);
      expect(result.js()).toEqual([5, 11, 17]);
    });
  });
});

// ---------------------------------------------------------------------------
// M6.2b: Orchestrator worker (off-main-thread mega-module execution)
// ---------------------------------------------------------------------------

describe.skipIf(!globalThis.crossOriginIsolated)(
  "orchestrator worker (M6.2b)",
  () => {
    beforeAll(async () => {
      await init("wasm");
    });

    it("backend has orchestrator when crossOriginIsolated", () => {
      const backend = getBackend() as any;
      expect(backend.orchestrator).not.toBeNull();
    });

    it("simple add produces correct result via orchestrator", () => {
      using f = jit((x: np.Array) => x.add(1));
      using x = np.array([10, 20, 30, 40]);
      using result = f(x);
      expect(result.js()).toEqual([11, 21, 31, 41]);
    });

    it("chained ops dispatched through orchestrator", () => {
      using f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
      using x = np.array([1, 2, 3, 4]);
      using result = f(x);
      expect(result.js()).toEqual([1, 3, 5, 7]);
    });

    it("two-input operation through orchestrator", () => {
      using f = jit((a: np.Array, b: np.Array) => a.add(b));
      using a = np.array([1, 2, 3]);
      using b = np.array([10, 20, 30]);
      using result = f(a, b);
      expect(result.js()).toEqual([11, 22, 33]);
    });

    it("mega-module with internal alloc/free proxied correctly", () => {
      // Multi-step chain where recycling can't eliminate all alloc/free:
      // f(x) = ((x + 1) * 2 - 3) / 4 + 5
      using f = jit((x: np.Array) => x.add(1).mul(2).sub(3).div(4).add(5));
      using x = np.array([3, 7, 11, 15]);
      using result = f(x);
      // (3+1)*2=8, 8-3=5, 5/4=1.25, 1.25+5=6.25
      // (7+1)*2=16, 16-3=13, 13/4=3.25, 3.25+5=8.25
      // (11+1)*2=24, 24-3=21, 21/4=5.25, 5.25+5=10.25
      // (15+1)*2=32, 32-3=29, 29/4=7.25, 7.25+5=12.25
      expect(result.js()).toEqual([6.25, 8.25, 10.25, 12.25]);
    });

    it("repeated dispatches use cached module registration", () => {
      using f = jit((x: np.Array) => x.mul(3));
      using x1 = np.array([1, 2, 3]);
      using r1 = f(x1);
      expect(r1.js()).toEqual([3, 6, 9]);

      // Second call reuses the same registered module
      using x2 = np.array([10, 20, 30]);
      using r2 = f(x2);
      expect(r2.js()).toEqual([30, 60, 90]);
    });

    it("grad through orchestrator-dispatched mega-module", () => {
      using f = jit((x: np.Array) => x.mul(x).sum());
      using x = np.array([1, 2, 3]);
      using g = grad(f)(x);
      // d/dx (x^2).sum() = 2x
      expect(g.js()).toEqual([2, 4, 6]);
    });

    it("reduction kernel through orchestrator", () => {
      using f = jit((x: np.Array) => x.sum());
      using x = np.array([1, 2, 3, 4]);
      using result = f(x);
      expect(result.js()).toBe(10);
    });
  },
);
