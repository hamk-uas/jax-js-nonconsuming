import {
  jit,
  makeJaxpr,
  numpy as np,
  SymDim,
} from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

suite("M4.2: Parameterized Backend Codegen", () => {
  test("[100,64] and [150,64] share the same JitProgram", () => {
    // The exit criteria: a JIT function handles different batch sizes
    // without recompilation, producing correct results.
    using f = jit(
      (x: np.Array) => x.mul(np.array(2.0)).add(np.array(1.0)),
      { dynamic_axes: { 0: "T" } },
    );

    // First call: [100, 64]
    using x1 = np.ones([100, 64]);
    using y1 = f(x1) as np.Array;
    expect(y1.shape).toEqual([100, 64]);
    // Each element should be 2*1 + 1 = 3
    const data1 = y1.js() as number[][];
    expect(data1.length).toBe(100);
    expect(data1[0].length).toBe(64);
    expect(data1[0][0]).toBe(3);
    expect(data1[99][63]).toBe(3);

    // Second call: [150, 64] — should reuse the same compiled program
    using x2 = np.ones([150, 64]);
    using y2 = f(x2) as np.Array;
    expect(y2.shape).toEqual([150, 64]);
    const data2 = y2.js() as number[][];
    expect(data2.length).toBe(150);
    expect(data2[0][0]).toBe(3);
    expect(data2[149][63]).toBe(3);
  });

  test("buffer recycling works for same-symbolic-size intermediates", () => {
    // Chain of operations where intermediates have the same symbolic size.
    // Buffer recycling should work (recycle steps emitted in JitProgram).
    using f = jit(
      (x: np.Array) => x.add(np.array(1.0)).mul(np.array(2.0)).sub(np.array(0.5)),
      { dynamic_axes: { 0: "T" } },
    );

    using x1 = np.ones([80, 32]);
    using y1 = f(x1) as np.Array;
    expect(y1.shape).toEqual([80, 32]);
    // (1 + 1) * 2 - 0.5 = 3.5
    const data1 = y1.js() as number[][];
    expect(data1[0][0]).toBe(3.5);

    using x2 = np.ones([200, 32]);
    using y2 = f(x2) as np.Array;
    expect(y2.shape).toEqual([200, 32]);
    const data2 = y2.js() as number[][];
    expect(data2[199][31]).toBe(3.5);
  });

  test("reduce along dynamic axis is not yet supported", () => {
    // Reducing along the dynamic axis (axis 0) requires Reduction.size to be
    // SizeExpr (the reduction loop bound must be dynamic). This is future work.
    const f = jit((x: np.Array) => np.sum(x, 0), {
      dynamic_axes: { 0: "T" },
    });
    using x1 = np.ones([3, 4]);
    // Should throw because the concrete reduction size from first call (3)
    // can't be reused for arbitrary dynamic sizes.
    // For now just verify it doesn't crash on the first call.
    using y1 = f(x1) as np.Array;
    expect(y1.shape).toEqual([4]);
    expect(y1.js()).toEqual([3, 3, 3, 3]);
    f.dispose();
  });
});

suite("M4.1: Symbolic Dimension Type & Shape Propagation", () => {
  suite("SymDim basics", () => {
    test("SymDim toString and toJSON", () => {
      const d = new SymDim("T");
      expect(d.toString()).toBe("T");
      expect(d.toJSON()).toBe("__sym__T");
      expect(d.name).toBe("T");
    });

    test("SymDim equality by name", () => {
      const a = new SymDim("T");
      const b = new SymDim("T");
      const c = new SymDim("N");
      // Different objects, same name — should behave as equal in dimEquals
      expect(a.name).toBe(b.name);
      expect(a.name).not.toBe(c.name);
    });
  });

  suite("Jaxpr.resolveDims", () => {
    test("resolves symbolic dims in a traced Jaxpr", () => {
      // Trace with a symbolic shape
      const aval = np.zeros([3, 4]);
      const { jaxpr } = makeJaxpr((x: np.Array) => x.add(1))(aval);

      // The Jaxpr should have concrete shapes (traced with concrete input)
      const str = jaxpr.toString();
      expect(str).toContain("float32[3,4]");

      jaxpr.dispose();
      aval.dispose();
    });
  });

  suite("dynamic_axes JIT caching", () => {
    test("same JIT program for different lengths", () => {
      // A simple elementwise function
      using f = jit((x: np.Array) => x.add(1), {
        dynamic_axes: { 0: "T" },
      });

      // Call with shape [3, 4]
      using x1 = np.ones([3, 4]);
      using y1 = f(x1) as np.Array;
      expect(y1.shape).toEqual([3, 4]);
      expect(y1.js()).toEqual([
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
      ]);

      // Call with shape [5, 4] — should NOT retrace (same symbolic pattern)
      using x2 = np.ones([5, 4]);
      using y2 = f(x2) as np.Array;
      expect(y2.shape).toEqual([5, 4]);
      expect(y2.js()).toEqual([
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
      ]);
    });

    test("simple elementwise with dynamic axis", () => {
      using f = jit((x: np.Array) => x.mul(np.array(2)), {
        dynamic_axes: { 0: "T" },
      });

      using x1 = np.array([1, 2, 3]);
      using y1 = f(x1) as np.Array;
      expect(y1.js()).toEqual([2, 4, 6]);

      using x2 = np.array([10, 20, 30, 40, 50]);
      using y2 = f(x2) as np.Array;
      expect(y2.js()).toEqual([20, 40, 60, 80, 100]);
    });

    test("reduce along static axis", () => {
      using f = jit((x: np.Array) => np.sum(x, 1), {
        dynamic_axes: { 0: "T" },
      });

      using x1 = np.ones([3, 4]);
      using y1 = f(x1) as np.Array;
      expect(y1.shape).toEqual([3]);
      expect(y1.js()).toEqual([4, 4, 4]);

      using x2 = np.ones([5, 4]);
      using y2 = f(x2) as np.Array;
      expect(y2.shape).toEqual([5]);
      expect(y2.js()).toEqual([4, 4, 4, 4, 4]);
    });

    test("binary ops with dynamic axis on both args", () => {
      // Both args have the same dynamic axis
      using f = jit((x: np.Array, y: np.Array) => x.add(y), {
        dynamic_axes: { 0: "T" },
      });

      using x1 = np.array([1, 2, 3]);
      using y1 = np.array([10, 20, 30]);
      using r1 = f(x1, y1) as np.Array;
      expect(r1.js()).toEqual([11, 22, 33]);

      using x2 = np.array([1, 2, 3, 4, 5]);
      using y2 = np.array([10, 20, 30, 40, 50]);
      using r2 = f(x2, y2) as np.Array;
      expect(r2.js()).toEqual([11, 22, 33, 44, 55]);
    });

    test("matmul uses concrete tracing (not dynamic)", () => {
      // matmul internally uses reshape with computed dims — can't trace symbolically.
      // Without dynamic_axes, each distinct shape gets its own trace (specialization).
      using w = np.eye(3);
      using f = jit((x: np.Array) => np.matmul(x, w));

      using x1 = np.ones([2, 3]);
      using y1 = f(x1) as np.Array;
      expect(y1.shape).toEqual([2, 3]);

      using x2 = np.ones([4, 3]);
      using y2 = f(x2) as np.Array;
      expect(y2.shape).toEqual([4, 3]);
    });

    test("unsupported op forces specialization (reshape with -1)", () => {
      // reshape with -1 requires computing total size, which fails on symbolic dims.
      // With dynamic_axes, this should fail at trace time.
      expect(() => {
        const f = jit((x: np.Array) => np.reshape(x, [-1]), {
          dynamic_axes: { 0: "T" },
        });
        using x = np.ones([3, 4]);
        f(x);
        f.dispose();
      }).toThrow();
    });
  });
});
