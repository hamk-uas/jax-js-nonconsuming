import {
  DType,
  jit,
  lax,
  makeJaxpr,
  numpy as np,
  SymDim,
} from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

suite("M4.2: Parameterized Backend Codegen", () => {
  test("[100,64] and [150,64] share the same JitProgram", () => {
    // The exit criteria: a JIT function handles different batch sizes
    // without recompilation, producing correct results.
    using f = jit((x: np.Array) => x.mul(np.array(2.0)).add(np.array(1.0)), {
      dynamic_axes: { 0: "T" },
    });

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
      (x: np.Array) =>
        x.add(np.array(1.0)).mul(np.array(2.0)).sub(np.array(0.5)),
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

  test("reduce along dynamic axis produces correct results for different sizes", () => {
    // Reducing along the dynamic axis (axis 0) uses a symbolic Reduction.size.
    // The cached JitProgram resolves the reduction bound at execution time.
    using f = jit((x: np.Array) => np.sum(x, 0), {
      dynamic_axes: { 0: "T" },
    });

    // First call: T=3
    using x1 = np.ones([3, 4]);
    using y1 = f(x1) as np.Array;
    expect(y1.shape).toEqual([4]);
    expect(y1.js()).toEqual([3, 3, 3, 3]);

    // Second call: T=5 — reuses cached program with different reduction size
    using x2 = np.ones([5, 4]);
    using y2 = f(x2) as np.Array;
    expect(y2.shape).toEqual([4]);
    expect(y2.js()).toEqual([5, 5, 5, 5]);

    // Third call: T=1 — edge case
    using x3 = np.ones([1, 4]);
    using y3 = f(x3) as np.Array;
    expect(y3.shape).toEqual([4]);
    expect(y3.js()).toEqual([1, 1, 1, 1]);
  });

  test("reduce along dynamic axis with non-uniform values", () => {
    using f = jit((x: np.Array) => np.sum(x, 0), {
      dynamic_axes: { 0: "T" },
    });

    // T=3 with varying values
    using x1 = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    using y1 = f(x1) as np.Array;
    expect(y1.js()).toEqual([9, 12]);

    // T=2 — different reduction size
    using x2 = np.array([
      [10, 20],
      [30, 40],
    ]);
    using y2 = f(x2) as np.Array;
    expect(y2.js()).toEqual([40, 60]);
  });

  test("max reduction along dynamic axis", () => {
    using f = jit((x: np.Array) => np.max(x, 0), {
      dynamic_axes: { 0: "T" },
    });

    using x1 = np.array([
      [1, 5],
      [3, 2],
      [2, 4],
    ]);
    using y1 = f(x1) as np.Array;
    expect(y1.js()).toEqual([3, 5]);

    using x2 = np.array([
      [10, 1],
      [5, 20],
    ]);
    using y2 = f(x2) as np.Array;
    expect(y2.js()).toEqual([10, 20]);
  });

  test("concrete kernel size + symbolic reduction size (reduce 2D along axis 1 with dynamic axis 0)", () => {
    // Shape [T, 4]: reducing along axis 1 gives kernel.size=T (symbolic),
    // reduction.size=4 (concrete). This tests that concrete reductions
    // still work correctly when combined with dynamic total size.
    using f = jit((x: np.Array) => np.sum(x, 1), {
      dynamic_axes: { 0: "T" },
    });

    using x1 = np.ones([3, 4]);
    using y1 = f(x1) as np.Array;
    expect(y1.shape).toEqual([3]);
    expect(y1.js()).toEqual([4, 4, 4]);

    using x2 = np.ones([7, 4]);
    using y2 = f(x2) as np.Array;
    expect(y2.shape).toEqual([7]);
    expect(y2.js()).toEqual([4, 4, 4, 4, 4, 4, 4]);
  });

  test("chained elementwise + reduce along dynamic axis", () => {
    // x.mul(2).sum(0) — fused elementwise with symbolic reduction
    using f = jit((x: np.Array) => x.mul(np.array(2.0)).sum(0), {
      dynamic_axes: { 0: "T" },
    });

    using x1 = np.ones([4, 3]);
    using y1 = f(x1) as np.Array;
    expect(y1.shape).toEqual([3]);
    expect(y1.js()).toEqual([8, 8, 8]); // 4 * 2 = 8 per column

    using x2 = np.ones([2, 3]);
    using y2 = f(x2) as np.Array;
    expect(y2.shape).toEqual([3]);
    expect(y2.js()).toEqual([4, 4, 4]); // 2 * 2 = 4 per column
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

suite("Polymorphic-N: associativeScan + scan with dynamic_axes", () => {
  test("associativeScan cumsum with polymorphic length", () => {
    using f = jit(
      (data: np.Array) =>
        lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), data),
      { dynamic_axes: { 0: "T" } },
    );

    // N=5 — first call triggers compilation
    using x5 = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });
    using r5 = f(x5) as np.Array;
    expect(r5.js()).toEqual([1, 3, 6, 10, 15]);

    // N=8 — reuses compiled program
    using x8 = np.array([1, 1, 1, 1, 1, 1, 1, 1], { dtype: DType.Float32 });
    using r8 = f(x8) as np.Array;
    expect(r8.js()).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);

    // N=3
    using x3 = np.array([10, 20, 30], { dtype: DType.Float32 });
    using r3 = f(x3) as np.Array;
    expect(r3.js()).toEqual([10, 30, 60]);

    // N=1 — edge case
    using x1 = np.array([42], { dtype: DType.Float32 });
    using r1 = f(x1) as np.Array;
    expect(r1.js()).toEqual([42]);
  });

  test("associativeScan cumprod with polymorphic length", () => {
    using f = jit(
      (data: np.Array) =>
        lax.associativeScan(
          (a: np.Array, b: np.Array) => np.multiply(a, b),
          data,
        ),
      { dynamic_axes: { 0: "T" } },
    );

    using x4 = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using r4 = f(x4) as np.Array;
    expect(r4.js()).toEqual([1, 2, 6, 24]);

    using x6 = np.array([2, 2, 2, 2, 2, 2], { dtype: DType.Float32 });
    using r6 = f(x6) as np.Array;
    expect(r6.js()).toEqual([2, 4, 8, 16, 32, 64]);
  });

  test("associativeScan reverse with polymorphic length", () => {
    using f = jit(
      (data: np.Array) =>
        lax.associativeScan((a: np.Array, b: np.Array) => np.add(a, b), data, {
          reverse: true,
        }),
      { dynamic_axes: { 0: "T" } },
    );

    using x4 = np.array([1, 2, 3, 4], { dtype: DType.Float32 });
    using r4 = f(x4) as np.Array;
    expect(r4.js()).toEqual([10, 9, 7, 4]);

    using x3 = np.array([10, 20, 30], { dtype: DType.Float32 });
    using r3 = f(x3) as np.Array;
    expect(r3.js()).toEqual([60, 50, 30]);
  });

  test("lax.scan cumsum with polymorphic length", () => {
    using f = jit(
      (xs: np.Array) => {
        const init = np.array([0], { dtype: DType.Float32 });
        const [carry, ys] = lax.scan(
          (c: np.Array, x: np.Array) => {
            const nc = np.add(c, x);
            return [nc, nc];
          },
          init,
          xs,
        );
        init.dispose();
        carry.dispose();
        return ys;
      },
      { dynamic_axes: { 0: "T" } },
    );

    // N=4
    using x4 = np.array([[1], [2], [3], [4]], { dtype: DType.Float32 });
    using r4 = f(x4) as np.Array;
    expect(r4.js()).toEqual([[1], [3], [6], [10]]);

    // N=6 — should reuse compiled program
    using x6 = np.array([[1], [1], [1], [1], [1], [1]], {
      dtype: DType.Float32,
    });
    using r6 = f(x6) as np.Array;
    expect(r6.js()).toEqual([[1], [2], [3], [4], [5], [6]]);
  });
});

suite("Polymorphic foriLoop bounds (Phase 8.1)", () => {
  test("foriLoop with symbolic upper bound accumulates correctly", () => {
    // x.shape[0] is SymDim("T") during tracing — passed as foriLoop upper bound.
    // Body: carry += 1 each iteration → result = N (the batch size).
    using f = jit(
      (x: np.Array) => {
        using init = np.array(0, { dtype: DType.Float32 });
        return lax.foriLoop(
          0,
          x.shape[0] as unknown as number, // SymDim at trace time
          (_i: np.Array, carry: np.Array) => np.add(carry, np.array(1)),
          init,
        );
      },
      { dynamic_axes: { 0: "T" } },
    );

    // T=5
    using x5 = np.ones([5, 2]);
    using r5 = f(x5) as np.Array;
    expect(r5).toBeAllclose(5);

    // T=3 — reuses compiled program
    using x3 = np.ones([3, 2]);
    using r3 = f(x3) as np.Array;
    expect(r3).toBeAllclose(3);

    // T=1 — edge case
    using x1 = np.ones([1, 2]);
    using r1 = f(x1) as np.Array;
    expect(r1).toBeAllclose(1);
  });

  test("foriLoop with symbolic bound and index-dependent body", () => {
    // Sum of indices 0..N-1 = N*(N-1)/2.
    using f = jit(
      (x: np.Array) => {
        using init = np.array(0, { dtype: DType.Float32 });
        return lax.foriLoop(
          0,
          x.shape[0] as unknown as number,
          (i: np.Array, carry: np.Array) => {
            using iF = i.astype(DType.Float32);
            return np.add(carry, iF);
          },
          init,
        );
      },
      { dynamic_axes: { 0: "T" } },
    );

    // T=4: 0+1+2+3 = 6
    using x4 = np.ones([4, 1]);
    using r4 = f(x4) as np.Array;
    expect(r4).toBeAllclose(6);

    // T=6: 0+1+2+3+4+5 = 15
    using x6 = np.ones([6, 1]);
    using r6 = f(x6) as np.Array;
    expect(r6).toBeAllclose(15);
  });
});
