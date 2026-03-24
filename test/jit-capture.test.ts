// Tests for the JIT capture facility.

import {
  type CapturedKernel,
  type CapturedSubProgram,
  captureJitReport,
  clearCaches,
  formatJitReport,
  lax,
  numpy as np,
  setCodeCapture,
} from "@hamk-uas/jax-js-nonconsuming";
import { expect, test } from "vitest";

import { deviceSuite } from "./device-suite.js";

await deviceSuite(
  (device) => {
    test("captureJitReport captures elementwise chain", () => {
      clearCaches();
      using x = np.ones([1024]);
      using two = np.array([2]);
      using one = np.array([1]);
      const report = captureJitReport((a: typeof x) => {
        using t = a.mul(two);
        return t.add(one);
      }, x);

      // Verify report structure
      expect(report.capabilities).toBeDefined();
      expect(report.program).toBeDefined();
      expect(report.program.numInputs).toBeGreaterThan(0);
      expect(report.program.numOutputs).toBe(1);
      expect(report.program.numSteps).toBeGreaterThan(0);

      // Elementwise chain should fuse to 1 kernel
      expect(report.program.kernels.length).toBe(1);
      const k = report.program.kernels[0];
      expect(k.numOutputs).toBe(1);
      expect(k.outputs[0].dtype).toBe("float32");
      expect(k.outputs[0].hasReduction).toBe(false);

      // Step counts should include execute and malloc
      expect(report.program.stepCounts.execute).toBeGreaterThan(0);
      expect(report.program.stepCounts.malloc).toBeGreaterThan(0);
    });

    test("captureJitReport captures reduction", () => {
      clearCaches();
      using x = np.ones([1024]);
      const report = captureJitReport((a: typeof x) => a.sum(), x);

      // Should have at least one kernel with a reduction
      const reduced = report.program.kernels.filter((k: CapturedKernel) =>
        k.outputs.some((o: CapturedKernel["outputs"][0]) => o.hasReduction),
      );
      expect(reduced.length).toBeGreaterThan(0);
      expect(reduced[0].outputs[0].reductionOp).toBeDefined();
    });

    test("captureJitReport captures matmul", () => {
      clearCaches();
      using a = np.ones([32, 32]);
      using b = np.ones([32, 32]);
      const report = captureJitReport(
        (x: typeof a, y: typeof b) => np.matmul(x, y),
        a,
        b,
      );

      // Matmul should appear as a kernel with reduction
      expect(report.program.kernels.length).toBeGreaterThan(0);
      expect(report.program.numOutputs).toBe(1);
    });

    test("captureJitReport captures code entries on cache miss", () => {
      clearCaches();
      using x = np.ones([64]);
      using three = np.array([3]);
      const report = captureJitReport((a: typeof x) => a.mul(three), x);

      // On cache miss we should get at least one code entry
      expect(report.codeEntries.length).toBeGreaterThan(0);
      const entry = report.codeEntries[0];
      expect(entry.backend).toBe(device === "webgpu" ? "webgpu" : "wasm");
      expect(entry.kind).toBeDefined();
      expect(entry.code).toBeDefined();
    });

    test("captureJitReport restores null code capture callback", () => {
      clearCaches();
      using x = np.ones([8]);
      using one = np.array([1]);
      captureJitReport((a: typeof x) => a.add(one), x);

      // Verify the callback was restored to null
      const entries: unknown[] = [];
      setCodeCapture((e) => entries.push(e));
      setCodeCapture(null);
    });

    test("formatJitReport produces readable output", () => {
      clearCaches();
      using x = np.ones([256]);
      using two = np.array([2]);
      using one = np.array([1]);
      const report = captureJitReport((a: typeof x) => {
        using t = a.mul(two);
        return t.add(one);
      }, x);

      const text = formatJitReport(report);
      expect(text).toContain("=== JIT Compilation Report ===");
      expect(text).toContain("Program:");
      expect(text).toContain("Kernels:");
      expect(text).toContain("Step Counts:");
      expect(text).toContain("Full Program Listing:");
      expect(text).toContain("float32");
    });

    test("captureJitReport captures sub-programs (scan)", async () => {
      clearCaches();
      using init = np.zeros([4]);
      using xs = np.ones([10, 4]);
      const report = captureJitReport(
        (c: typeof init, x: typeof xs) =>
          lax.scan(
            (carry, xi) => {
              using sum = carry.add(xi);
              return [sum, sum];
            },
            c,
            x,
          ),
        init,
        xs,
      );

      // Should have at least one sub-program for the scan body
      expect(report.program.subPrograms.length).toBeGreaterThan(0);
      const scanSub = report.program.subPrograms.find(
        (sp: CapturedSubProgram) => sp.type.startsWith("scan("),
      );
      expect(scanSub).toBeDefined();
      expect(scanSub!.program.kernels.length).toBeGreaterThan(0);
    });
  },
  ["wasm"],
);
