// Tests for setCodeCapture API and conv classification signal.

import {
  _lastConvClass,
  clearCaches,
  type CodeCaptureEntry,
  type ConvClass,
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
        expect(["kernel", "mega-module", "program"]).toContain(e.kind);
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

    test("conv classification: 1x1 kernel classified as fast-1x1-dot", () => {
      using x = np.ones([1, 1, 4]); // NCW format
      using w = np.ones([1, 1, 1]); // OIW format, 1x1 kernel
      const f = jit((a: typeof x, b: typeof w) =>
        lax.convGeneralDilated(a, b, [1], "VALID"),
      );
      using _result = f(x, w);
      f.dispose();

      const kind: ConvClass | null = _lastConvClass();
      expect(kind).toBe("fast-1x1-dot");
    });

    test("conv classification: 3x3 kernel classified as generic-dot", () => {
      using x = np.ones([1, 1, 8]); // NCW format
      using w = np.ones([1, 1, 3]); // OIW format, 3-wide kernel
      const f = jit((a: typeof x, b: typeof w) =>
        lax.convGeneralDilated(a, b, [1], "VALID"),
      );
      using _result = f(x, w);
      f.dispose();

      const kind: ConvClass | null = _lastConvClass();
      expect(kind).toBe("generic-dot");
    });

    test("scan code capture emits kind=scan with WAT code", () => {
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));

      using init = np.zeros([4]);
      using xs = np.ones([10, 4]);
      const f = jit((c: typeof init, x: typeof xs) =>
        lax.scan(
          (carry, xi) => {
            const nc = np.add(carry, xi);
            return [nc, nc];
          },
          c,
          x,
          { acceptPath: "compiled-loop" },
        ),
      );
      const [carry, ys] = f(init, xs) as [typeof init, typeof xs];
      carry.dispose();
      ys.dispose();
      f.dispose();

      const scanEntries = entries.filter((e) => e.kind === "scan");
      expect(scanEntries.length).toBeGreaterThanOrEqual(1);
      for (const e of scanEntries) {
        expect(e.backend).toBe("wasm");
        expect(typeof e.code).toBe("string");
        expect(e.code!.length).toBeGreaterThan(0);
        expect(e.metadata?.numSteps).toBeGreaterThanOrEqual(1);
      }
    });

    test("assoc-scan code capture emits kind=assoc-scan", async () => {
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));

      using xs = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
      const f = jit((x: typeof xs) =>
        lax.associativeScan((a, b) => a.add(b), x),
      );
      using _result = f(xs);
      f.dispose();

      const assocEntries = entries.filter((e) => e.kind === "assoc-scan");
      expect(assocEntries.length).toBeGreaterThanOrEqual(1);
      for (const e of assocEntries) {
        expect(e.backend).toBe("wasm");
        expect(typeof e.code).toBe("string");
        expect(e.code!.length).toBeGreaterThan(0);
        expect(e.metadata?.numLeaves).toBeGreaterThanOrEqual(1);
      }
    });

    test("block-map code capture emits kind=block-map", () => {
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));

      using x = np.ones([8]);
      const f = jit((a: typeof x) =>
        lax.blockMap((block) => block.mul(np.array([2, 2, 2, 2])), a, {
          blockShape: [4],
          inAxes: [0],
          outAxes: [0],
        }),
      );
      using _result = f(x) as typeof x;
      f.dispose();

      const bmEntries = entries.filter((e) => e.kind === "block-map");
      expect(bmEntries.length).toBeGreaterThanOrEqual(1);
      for (const e of bmEntries) {
        expect(e.backend).toBe("wasm");
        expect(typeof e.code).toBe("string");
        expect(e.code!.length).toBeGreaterThan(0);
        expect(e.metadata?.numSteps).toBeGreaterThanOrEqual(1);
      }
    });

    test("routine code capture emits kind=routine for sort", () => {
      const entries: CodeCaptureEntry[] = [];
      setCodeCapture((e) => entries.push(e));

      using x = np.array([3, 1, 4, 1, 5]);
      const f = jit((a: typeof x) => np.sort(a));
      using _result = f(x);
      f.dispose();

      const routineEntries = entries.filter((e) => e.kind === "routine");
      expect(routineEntries.length).toBeGreaterThanOrEqual(1);
      const sortEntry = routineEntries.find((e) => e.label === "sort");
      expect(sortEntry).toBeDefined();
      expect(sortEntry!.backend).toBe("wasm");
      expect(sortEntry!.metadata?.n).toBe(5);
    });
  },
  ["wasm"],
);
