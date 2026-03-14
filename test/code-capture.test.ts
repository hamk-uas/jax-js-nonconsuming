// Tests for setCodeCapture API and conv lowering kind signal.

import {
  _lastConvLoweringKind,
  clearCaches,
  type CodeCaptureEntry,
  type ConvLoweringKind,
  jit,
  lax,
  numpy as np,
  setCodeCapture,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterEach, expect, test } from "vitest";

import { deviceSuite } from "./device-suite.js";

afterEach(() => {
  // Always unregister the callback
  setCodeCapture(null);
});

await deviceSuite(
  (device) => {
    test("setCodeCapture receives entries on jit execution", () => {
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));

      using x = np.array([1, 2, 3, 4]);
      const f = jit((a: typeof x) => a.mul(np.array([2, 2, 2, 2])));
      using _result = f(x);
      f.dispose();

      // Should have captured at least one compilation
      expect(entries.length).toBeGreaterThanOrEqual(1);
      for (const e of entries) {
        expect(e.backend).toBe(device === "webgpu" ? "webgpu" : "wasm");
        expect(["kernel", "mega-module"]).toContain(e.kind);
      }
    });

    test("setCodeCapture emits code string", () => {
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));

      using x = np.array([1, 2, 3, 4]);
      const f = jit((a: typeof x) => a.add(np.array([1, 1, 1, 1])));
      using _result = f(x);
      f.dispose();

      expect(entries.length).toBeGreaterThanOrEqual(1);
      for (const e of entries) {
        // Code should be a non-empty string (WGSL or WAT)
        expect(typeof e.code).toBe("string");
        expect(e.code!.length).toBeGreaterThan(0);
      }
    });

    test("setCodeCapture(null) disables capture", () => {
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));

      // First call should capture
      using x = np.array([1, 2, 3, 4]);
      const f = jit((a: typeof x) => a.mul(np.array([3, 3, 3, 3])));
      using _r1 = f(x);

      const countAfterFirst = entries.length;
      expect(countAfterFirst).toBeGreaterThanOrEqual(1);

      // Disable capture and clear caches to force recompilation
      setCodeCapture(null);
      f.dispose();
      clearCaches();

      // New function with capture disabled should not capture
      const g = jit((a: typeof x) => a.add(np.array([5, 5, 5, 5])));
      using _r2 = g(x);
      g.dispose();

      expect(entries.length).toBe(countAfterFirst);
    });

    test("conv lowering kind: 1x1 kernel classified as fast-1x1-dot", () => {
      using x = np.ones([1, 1, 4]); // NCW format
      using w = np.ones([1, 1, 1]); // OIW format, 1x1 kernel
      const f = jit((a: typeof x, b: typeof w) =>
        lax.convGeneralDilated(a, b, [1], "VALID"),
      );
      using _result = f(x, w);
      f.dispose();

      const kind: ConvLoweringKind | null = _lastConvLoweringKind();
      expect(kind).toBe("fast-1x1-dot");
    });

    test("conv lowering kind: 3x3 kernel classified as generic-dot", () => {
      using x = np.ones([1, 1, 8]); // NCW format
      using w = np.ones([1, 1, 3]); // OIW format, 3-wide kernel
      const f = jit((a: typeof x, b: typeof w) =>
        lax.convGeneralDilated(a, b, [1], "VALID"),
      );
      using _result = f(x, w);
      f.dispose();

      const kind: ConvLoweringKind | null = _lastConvLoweringKind();
      expect(kind).toBe("generic-dot");
    });
  },
  ["wasm"],
);
