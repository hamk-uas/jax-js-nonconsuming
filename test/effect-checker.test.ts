import {
  _setVerifyEffects,
  type Array,
  ClosedJaxpr,
  lax,
  makeJaxpr,
  MemoryEffect,
  numpy as np,
  verifyJaxprEffects,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

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
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array, b: Array) =>
        np.add(a, b),
      )(np.zeros([4]), np.zeros([4]));
      expect(closedJaxpr).toBeInstanceOf(ClosedJaxpr);
      const eqn = closedJaxpr.jaxpr.eqns[0];
      expect(eqn).toBeDefined();
      // M1.2: effects are now assigned by tracing
      expect(eqn.inputEffects).toEqual([
        MemoryEffect.Borrow,
        MemoryEffect.Borrow,
      ]);
      expect(eqn.outputEffects).toEqual([MemoryEffect.Alloc]);
      // Output Vars also carry the effect
      expect(eqn.outBinders[0].effect).toBe(MemoryEffect.Alloc);
      closedJaxpr.dispose();
    });

    test("pprint includes effect annotations after M1.1", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array) => np.multiply(a, a))(
        np.zeros([3]),
      );
      const pprintStr = closedJaxpr.jaxpr.toString();
      closedJaxpr.dispose();
      // After M1.1, pprint will include effect annotations
      // For now, just verify pprint works without errors
      expect(pprintStr).toContain("mul");
    });

    test("pprint renders effects when set manually", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array, b: Array) =>
        np.add(a, b),
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
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array) => np.add(a, a))(
        np.zeros([2]),
      );
      const outVar = closedJaxpr.jaxpr.eqns[0].outBinders[0];
      outVar.effect = MemoryEffect.Alloc;
      const pprintStr = closedJaxpr.jaxpr.eqns[0].pprint().toString();
      closedJaxpr.dispose();
      expect(pprintStr).toContain("{Alloc}");
    });

    test("DynamicUpdateSlice emits Mutate on its target (first input)", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr((dst: Array, src: Array) =>
        lax.dynamicUpdateSlice(dst, src, 0),
      )(np.zeros([6]), np.zeros([3]));
      const eqns = closedJaxpr.jaxpr.eqns;
      // Find the DUS equation
      // eslint-disable-next-line @typescript-eslint/no-unsafe-enum-comparison
      const dusEqn = eqns.find((e) => e.primitive === "dynamic_update_slice");
      expect(dusEqn).toBeDefined();
      // First input (dst) should be Mutate, second input (src) should be Borrow
      expect(dusEqn!.inputEffects).toEqual([
        MemoryEffect.Mutate,
        MemoryEffect.Borrow,
      ]);
      // Output is still Alloc (produces a new buffer in the functional model)
      expect(dusEqn!.outputEffects).toEqual([MemoryEffect.Alloc]);
      closedJaxpr.dispose();
    });
  });

  describe("M2 — borrow checker", () => {
    test("verifyJaxprEffects accepts safe elementwise graph", () => {
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array, b: Array) =>
        np.add(a, b),
      )(np.zeros([4]), np.zeros([4]));
      const result = verifyJaxprEffects(closedJaxpr.jaxpr);
      expect(result.ok).toBe(true);
      expect(result.errors).toHaveLength(0);
      closedJaxpr.dispose();
    });

    test("verifyJaxprEffects rejects use-after-consume", () => {
      // Build a jaxpr, then manually set an input to Consume to simulate
      // a consume followed by a borrow
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array) => {
        const b = np.add(a, a); // uses a twice
        return np.multiply(b, a); // uses a again
      })(np.zeros([2]));
      const jaxpr = closedJaxpr.jaxpr;

      // Manually set the first equation's first input to Consume
      // (simulating a primitive that consumes its input)
      jaxpr.eqns[0].inputEffects![0] = MemoryEffect.Consume;

      const result = verifyJaxprEffects(jaxpr);
      // The second equation also uses 'a' (as Borrow), which should fail
      expect(result.ok).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors[0]).toContain("Use-after-consume");
      closedJaxpr.dispose();
    });

    test("Mutate exclusivity: rejects Mutate+Borrow on same var in one eqn", () => {
      // Build a jaxpr with DUS where dst is also borrowed
      const { jaxpr: closedJaxpr } = makeJaxpr((dst: Array, src: Array) =>
        lax.dynamicUpdateSlice(dst, src, 0),
      )(np.zeros([6]), np.zeros([3]));
      const jaxpr = closedJaxpr.jaxpr;

      // Find the DUS equation
      const dusEqn = jaxpr.eqns.find(
        // eslint-disable-next-line @typescript-eslint/no-unsafe-enum-comparison
        (e) => e.primitive === "dynamic_update_slice",
      );
      expect(dusEqn).toBeDefined();

      // Normally src is a different var. Simulate aliasing by making src
      // reference the same Var as dst, with Borrow effect
      const dstVar = dusEqn!.inputs[0];
      dusEqn!.inputs[1] = dstVar; // Same var as dst
      // dst=Mutate, src(=same var)=Borrow → exclusivity violation
      dusEqn!.inputEffects = [MemoryEffect.Mutate, MemoryEffect.Borrow];

      const result = verifyJaxprEffects(jaxpr);
      expect(result.ok).toBe(false);
      expect(result.errors[0]).toContain("Mutate exclusivity violation");
      closedJaxpr.dispose();
    });

    test("verifyJaxprEffects accepts graph with outputs (no consume needed)", () => {
      // Outputs of the jaxpr don't need to be consumed
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array, b: Array) =>
        np.add(a, b),
      )(np.zeros([2]), np.zeros([2]));
      const result = verifyJaxprEffects(closedJaxpr.jaxpr);
      expect(result.ok).toBe(true);
      closedJaxpr.dispose();
    });

    test("verifyJaxprEffects accepts intermediates (non-consuming model)", () => {
      // Intermediates that are borrowed by later equations are valid
      const { jaxpr: closedJaxpr } = makeJaxpr((a: Array, b: Array) => {
        const c = np.multiply(a, b); // intermediate
        return np.add(c, a); // uses intermediate + original
      })(np.zeros([4]), np.zeros([4]));
      const result = verifyJaxprEffects(closedJaxpr.jaxpr);
      expect(result.ok).toBe(true);
      expect(result.errors).toHaveLength(0);
      closedJaxpr.dispose();
    });
  });

  describe("M2.2 — validator integration in makeJaxpr", () => {
    test("makeJaxpr succeeds with verification enabled for elementwise ops", () => {
      _setVerifyEffects(true);
      try {
        const { jaxpr: closedJaxpr } = makeJaxpr((a: Array, b: Array) =>
          np.add(a, b),
        )(np.zeros([4]), np.zeros([4]));
        // If we get here, verification passed
        expect(closedJaxpr).toBeInstanceOf(ClosedJaxpr);
        closedJaxpr.dispose();
      } finally {
        _setVerifyEffects(false);
      }
    });

    test("makeJaxpr succeeds with verification enabled for chained ops", () => {
      _setVerifyEffects(true);
      try {
        const { jaxpr: closedJaxpr } = makeJaxpr((a: Array, b: Array) => {
          const c = np.multiply(a, b);
          const d = np.add(c, a);
          return np.subtract(d, b);
        })(np.zeros([3]), np.zeros([3]));
        expect(closedJaxpr).toBeInstanceOf(ClosedJaxpr);
        closedJaxpr.dispose();
      } finally {
        _setVerifyEffects(false);
      }
    });

    test("makeJaxpr succeeds with verification enabled for DUS", () => {
      _setVerifyEffects(true);
      try {
        const { jaxpr: closedJaxpr } = makeJaxpr((dst: Array, src: Array) =>
          lax.dynamicUpdateSlice(dst, src, 0),
        )(np.zeros([6]), np.zeros([3]));
        expect(closedJaxpr).toBeInstanceOf(ClosedJaxpr);
        closedJaxpr.dispose();
      } finally {
        _setVerifyEffects(false);
      }
    });

    test("makeJaxpr succeeds with verification enabled for reductions", () => {
      _setVerifyEffects(true);
      try {
        const { jaxpr: closedJaxpr } = makeJaxpr((a: Array) => {
          const b = np.multiply(a, a);
          return np.sum(b);
        })(np.zeros([8]));
        expect(closedJaxpr).toBeInstanceOf(ClosedJaxpr);
        closedJaxpr.dispose();
      } finally {
        _setVerifyEffects(false);
      }
    });
  });

  describe("M2.3 — zero violations across operations", () => {
    test("broad operation coverage: no effect violations", () => {
      _setVerifyEffects(true);
      try {
        // Multi-op chain (exp, sin, cos)
        const { jaxpr: j1 } = makeJaxpr((a: Array) =>
          np.exp(np.sin(np.cos(a))),
        )(np.zeros([4]));
        j1.dispose();

        // Deep chain with reduction
        const { jaxpr: j2 } = makeJaxpr((a: Array, b: Array) => {
          const c = np.multiply(a, b);
          const d = np.add(c, a);
          const e = np.subtract(d, b);
          return np.sum(e);
        })(np.zeros([4]), np.zeros([4]));
        j2.dispose();

        // Matmul
        const { jaxpr: j3 } = makeJaxpr((a: Array, b: Array) =>
          np.matmul(a, b),
        )(np.zeros([3, 4]), np.zeros([4, 5]));
        j3.dispose();

        // Transpose
        const { jaxpr: j4 } = makeJaxpr((a: Array) => np.transpose(a))(
          np.zeros([3, 4]),
        );
        j4.dispose();

        // Where / conditional
        const { jaxpr: j5 } = makeJaxpr((a: Array, b: Array) =>
          np.where(np.greater(a, b), a, b),
        )(np.zeros([4]), np.zeros([4]));
        j5.dispose();

        // Flip
        const { jaxpr: j6 } = makeJaxpr((a: Array) => np.flip(a))(
          np.zeros([4]),
        );
        j6.dispose();

        // Concatenate
        const { jaxpr: j7 } = makeJaxpr((a: Array, b: Array) =>
          np.concatenate([a, b], 0),
        )(np.zeros([3]), np.zeros([4]));
        j7.dispose();

        // Softmax-like pattern
        const { jaxpr: j8 } = makeJaxpr((a: Array) => {
          const maxA = np.max(a);
          const shifted = np.subtract(a, maxA);
          const e = np.exp(shifted);
          const s = np.sum(e);
          return np.divide(e, s);
        })(np.zeros([4]));
        j8.dispose();
      } finally {
        _setVerifyEffects(false);
      }
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
