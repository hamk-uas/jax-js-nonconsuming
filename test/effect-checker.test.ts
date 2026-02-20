import { describe, expect, test } from "vitest";
import {
  ClosedJaxpr,
  makeJaxpr,
  MemoryEffect,
  numpy as np,
  type Array,
} from "@hamk-uas/jax-js-nonconsuming";

describe("effect-checker", () => {
  describe("MemoryEffect enum", () => {
    test("has all four effect variants", () => {
      expect(MemoryEffect.Alloc).toBe("Alloc");
      expect(MemoryEffect.Borrow).toBe("Borrow");
      expect(MemoryEffect.Consume).toBe("Consume");
      expect(MemoryEffect.Mutate).toBe("Mutate");
    });
  });

  describe("M1 — effect tracing", () => {
    test("elementwise add assigns Borrow to inputs and Alloc to output", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr(
        (a: Array, b: Array) => np.add(a, b),
      )(np.zeros([4]), np.zeros([4]));
      expect(closedJaxpr).toBeInstanceOf(ClosedJaxpr);
      const eqn = closedJaxpr.jaxpr.eqns[0];
      expect(eqn).toBeDefined();
      closedJaxpr.dispose();
      // Placeholder: effects are undefined until M1.2
      // Once M1.2 lands, uncomment these assertions:
      // expect(eqn.inputEffects).toEqual([MemoryEffect.Borrow, MemoryEffect.Borrow]);
      // expect(eqn.outputEffects).toEqual([MemoryEffect.Alloc]);
    });

    test("pprint includes effect annotations after M1.1", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr(
        (a: Array) => np.multiply(a, a),
      )(np.zeros([3]));
      const pprintStr = closedJaxpr.jaxpr.toString();
      closedJaxpr.dispose();
      // After M1.1, pprint will include effect annotations
      // For now, just verify pprint works without errors
      expect(pprintStr).toContain("mul");
    });

    test("pprint renders effects when set manually", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr(
        (a: Array, b: Array) => np.add(a, b),
      )(np.zeros([2]), np.zeros([2]));
      const eqn = closedJaxpr.jaxpr.eqns[0];
      // Manually set effects to verify pprint rendering
      eqn.inputEffects = [MemoryEffect.Borrow, MemoryEffect.Borrow];
      eqn.outputEffects = [MemoryEffect.Alloc];
      const pprintStr = eqn.pprint().toString();
      closedJaxpr.dispose();
      expect(pprintStr).toContain("{in[Borrow,Borrow] out[Alloc]}");
    });

    test("Var pprint shows effect when set", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr(
        (a: Array) => np.add(a, a),
      )(np.zeros([2]));
      const outVar = closedJaxpr.jaxpr.eqns[0].outBinders[0];
      outVar.effect = MemoryEffect.Alloc;
      const pprintStr = closedJaxpr.jaxpr.eqns[0].pprint().toString();
      closedJaxpr.dispose();
      expect(pprintStr).toContain("{Alloc}");
    });
  });

  describe("M2 — borrow checker", () => {
    test.skip("placeholder: verifyJaxprEffects accepts safe graph", () => {
      // Will be implemented in M2.1
    });

    test.skip("placeholder: verifyJaxprEffects rejects use-after-consume", () => {
      // Will be implemented in M2.1
    });

    test.skip("placeholder: Mutate requires exclusive ownership", () => {
      // Will be implemented in M2.1
    });
  });

  describe("M3 — JIT integration", () => {
    test.skip("placeholder: effect-driven recycling matches or beats heuristic", () => {
      // Will be implemented in M3.1
    });

    test.skip("placeholder: zero-copy DUS on WebGPU", () => {
      // Will be implemented in M3.2
    });
  });
});
