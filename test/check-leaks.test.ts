/**
 * Tests for the checkLeaks diagnostic API.
 *
 * checkLeaks uses backend slot count deltas (same as slotCount()) for accurate
 * leak detection, plus a tracking map for diagnostic details.
 */
import { checkLeaks, init, jit, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

await init();

describe("checkLeaks", () => {
  it("reports zero leaks when all arrays disposed", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5, 6]);
    x.dispose();
    y.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
    expect(report.summary).toBe("No leaks detected.");
  });

  it("detects a single leaked array", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    // deliberately do NOT dispose x
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(1);
    expect(report.details.length).toBeGreaterThanOrEqual(1);
    expect(report.summary).toContain("1 slot(s) leaked");
    // Clean up for next test
    x.dispose();
  });

  it("detects multiple leaked arrays", () => {
    checkLeaks.start();
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);
    const c = np.array([7, 8, 9]);
    a.dispose(); // only a is disposed
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(2);
    b.dispose();
    c.dispose();
  });

  it("jit intermediates do not leak", () => {
    checkLeaks.start();
    // Use a function without anonymous constants to avoid the known
    // anonymous-constant leak (see copilot-instructions).
    const f = jit((x: np.Array) => x.add(x).mul(x));
    const x = np.array([2, 3]);
    const result = f(x);
    expect(result.js()).toEqual([8, 18]);
    result.dispose();
    x.dispose();
    f.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
  });

  it("active property reflects tracking state", () => {
    checkLeaks.stop(); // override setup.ts beforeEach which calls start()
    expect(checkLeaks.active).toBe(false);
    checkLeaks.start();
    expect(checkLeaks.active).toBe(true);
    checkLeaks.stop();
    expect(checkLeaks.active).toBe(false);
  });

  it("arrays created before start are not counted", () => {
    const preExisting = np.array([99, 100]);
    checkLeaks.start();
    const tracked = np.array([1, 2]);
    tracked.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0); // preExisting not counted (existed before start)
    preExisting.dispose();
  });

  it("report details include array description", () => {
    checkLeaks.start();
    const x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(1);
    // Details should contain dtype and shape info from tracked arrays
    expect(report.details.length).toBeGreaterThanOrEqual(1);
    const hasArrayInfo = report.details.some(
      (d) => /float32|int32/.test(d) && /2,2|2, 2/.test(d),
    );
    expect(hasArrayInfo).toBe(true);
    x.dispose();
  });

  it("snapshot returns live tracked arrays", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5]);
    const snap = checkLeaks.snapshot();
    expect(snap.length).toBe(2);
    expect(snap.every((e) => e.rc === 1)).toBe(true);
    x.dispose();
    y.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
  });

  it("trackRefs records .ref call sites", () => {
    checkLeaks.start({ trackRefs: true });
    const x = np.array([1, 2, 3]);
    // jax-js-lint: allow-ref — intentional test of .ref tracking
    const y = x.ref; // extra ref
    const snap = checkLeaks.snapshot();
    expect(snap.length).toBe(1);
    expect(snap[0].rc).toBe(2);
    expect(snap[0].lastRef).not.toBeNull();
    x.dispose();
    y.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
  });
});
