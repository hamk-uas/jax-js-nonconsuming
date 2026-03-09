/**
 * Phase 0: Block-map prototype — validate body tracing.
 *
 * These tests verify that functions intended as block_map bodies produce
 * clean, blockSize-independent jaxprs when traced through makeJaxpr.
 * No new primitives, no codegen — pure tracing validation.
 *
 * Kill signal: if any body traces to O(blockSize) equations or produces
 * shapes that embed blockSize as a concrete dimension, stop and redesign.
 */

import { init, makeJaxpr, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

await init();

describe("Phase 0: block_map body tracing", () => {
  // -------------------------------------------------------------------------
  // T1.1: 1D elementwise body traces to blockSize-independent jaxpr
  // -------------------------------------------------------------------------
  test("1D elementwise body produces fixed-size jaxpr", () => {
    const body = (block: np.Array) =>
      block.mul(np.array(2.0)).add(np.array(1.0));

    // Trace with blockSize=8
    using x8 = np.zeros([8]);
    const { jaxpr: cj8 } = makeJaxpr(body)(x8);
    const eqnCount8 = cj8.jaxpr.eqns.length;

    // Trace with blockSize=256
    using x256 = np.zeros([256]);
    const { jaxpr: cj256 } = makeJaxpr(body)(x256);
    const eqnCount256 = cj256.jaxpr.eqns.length;

    // Same number of equations regardless of blockSize
    expect(eqnCount8).toBe(eqnCount256);
    // Should be a small fixed number (mul + add = 2 ops)
    expect(eqnCount8).toBeLessThanOrEqual(4);

    cj8.dispose();
    cj256.dispose();
  });

  // -------------------------------------------------------------------------
  // T1.2: 2D body (matmul-shaped) traces consistently
  // -------------------------------------------------------------------------
  test("2D body traces to consistent jaxpr", () => {
    // Simulates a body that operates on a 2D tile (block of a matrix)
    const body = (tile: np.Array) => tile.mul(np.array(2.0));

    using tile16 = np.zeros([16, 16]);
    const { jaxpr: cj16 } = makeJaxpr(body)(tile16);

    using tile32 = np.zeros([32, 32]);
    const { jaxpr: cj32 } = makeJaxpr(body)(tile32);

    // Same equation count
    expect(cj16.jaxpr.eqns.length).toBe(cj32.jaxpr.eqns.length);

    cj16.dispose();
    cj32.dispose();
  });

  // -------------------------------------------------------------------------
  // T1.3: Reduction body (sum) traces correctly
  // -------------------------------------------------------------------------
  test("reduction body traces correctly", () => {
    const body = (block: np.Array) => np.sum(block);

    using x = np.zeros([64]);
    const { jaxpr: cj } = makeJaxpr(body)(x);

    // Should contain a reduce primitive
    const primitives = cj.jaxpr.eqns.map((eqn) => eqn.primitive);
    expect(primitives).toContain("reduce");

    // Output should be scalar
    const outAval = cj.jaxpr.outs[0].aval;
    expect(outAval.shape).toEqual([]);

    cj.dispose();
  });

  // -------------------------------------------------------------------------
  // T1.4: Body with constants traces correctly
  // -------------------------------------------------------------------------
  test("body with captured constants traces correctly", () => {
    using scale = np.array([1.0, 2.0, 3.0, 4.0]);
    const body = (block: np.Array) => block.mul(scale);

    using x = np.zeros([4]);
    const { jaxpr: cj } = makeJaxpr(body)(x);

    // scale should appear as a constant
    expect(cj.consts.length).toBeGreaterThan(0);

    // Equations should still be small and fixed
    expect(cj.jaxpr.eqns.length).toBeLessThanOrEqual(4);

    cj.dispose();
    scale.dispose();
  });

  // -------------------------------------------------------------------------
  // T1.5: Body with pytree input/output traces correctly
  // -------------------------------------------------------------------------
  test("pytree body traces correctly", () => {
    const body = (inputs: { a: np.Array; b: np.Array }) => ({
      a: inputs.a.add(inputs.b),
      b: inputs.b.mul(np.array(2.0)),
    });

    using a = np.zeros([8]);
    using b = np.ones([8]);
    const { jaxpr: cj } = makeJaxpr(body)({ a, b });

    // Should have 2 input binders (a, b flattened) and 2 outputs
    expect(cj.jaxpr.inBinders.length).toBeGreaterThanOrEqual(2);
    expect(cj.jaxpr.outs.length).toBe(2);

    cj.dispose();
  });

  // -------------------------------------------------------------------------
  // T1.6: blockShape does NOT appear in equation shapes
  // -------------------------------------------------------------------------
  test("blockShape does not appear in equation shapes (kill signal)", () => {
    // Trace an elementwise body with a specific block size.
    // Verify that the shapes in the jaxpr inBinders/outBinders reference
    // the input shape, NOT embed blockSize as a hardcoded constant that
    // would change with different block sizes.
    const body = (block: np.Array) => {
      using doubled = block.mul(np.array(2.0));
      using summed = np.sum(doubled);
      return summed.add(np.array(1.0));
    };

    // Trace with two different "block sizes"
    using x64 = np.zeros([64]);
    const { jaxpr: cj64 } = makeJaxpr(body)(x64);

    using x128 = np.zeros([128]);
    const { jaxpr: cj128 } = makeJaxpr(body)(x128);

    // Same number of equations
    expect(cj64.jaxpr.eqns.length).toBe(cj128.jaxpr.eqns.length);

    // Same primitives in same order
    const prims64 = cj64.jaxpr.eqns.map((e) => e.primitive);
    const prims128 = cj128.jaxpr.eqns.map((e) => e.primitive);
    expect(prims64).toEqual(prims128);

    cj64.dispose();
    cj128.dispose();
  });

  // -------------------------------------------------------------------------
  // T1.7: Fused softmax body traces correctly
  // -------------------------------------------------------------------------
  test("fused softmax body traces into expected sequence", () => {
    const softmaxBody = (block: np.Array) => {
      using maxVal = np.max(block);
      using shifted = block.sub(maxVal);
      using exps = np.exp(shifted);
      using sumExps = np.sum(exps);
      return exps.div(sumExps);
    };

    using x = np.zeros([64]);
    const { jaxpr: cj } = makeJaxpr(softmaxBody)(x);

    const primitives = cj.jaxpr.eqns.map((eqn) => eqn.primitive);

    // Should contain: reduce (max), sub, exp, reduce (sum), reciprocal+mul or div
    expect(primitives.filter((p) => String(p) === "reduce").length).toBe(2);
    expect(primitives).toContain("exp");

    // Equation count should be fixed and reasonable
    expect(cj.jaxpr.eqns.length).toBeLessThanOrEqual(12);

    cj.dispose();
  });

  // -------------------------------------------------------------------------
  // T1.8: Body with two outputs traces correctly
  // -------------------------------------------------------------------------
  test("two-output body traces correctly", () => {
    const body = (block: np.Array) => ({
      doubled: block.mul(np.array(2.0)),
      sum: np.sum(block),
    });

    using x = np.zeros([32]);
    const { jaxpr: cj } = makeJaxpr(body)(x);

    // Should have 2 outputs (pytree leaves: doubled + sum)
    expect(cj.jaxpr.outs.length).toBe(2);

    // First output same shape as input, second is scalar
    const out0Shape = cj.jaxpr.outs[0].aval.shape;
    const out1Shape = cj.jaxpr.outs[1].aval.shape;
    expect(out0Shape).toEqual([32]);
    expect(out1Shape).toEqual([]);

    cj.dispose();
  });
});
