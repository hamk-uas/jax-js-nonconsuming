/**
 * Tests for multi-output kernel fusion in JIT compilation.
 *
 * When multiple independent kernels share the same inputs and output size,
 * jitCompile batches them into a single multi-output Kernel dispatch, reducing
 * dispatch overhead. These tests verify correctness of the fused outputs.
 */
import {
  defaultDevice,
  grad,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

// ---------------------------------------------------------------------------
// WASM backend tests
// ---------------------------------------------------------------------------
describe("multi-output kernel (WASM)", () => {
  let prev: ReturnType<typeof defaultDevice>;
  beforeAll(() => {
    prev = defaultDevice("wasm");
  });
  afterAll(() => defaultDevice(prev));

  it("two independent elementwise outputs are correct", () => {
    // Two outputs computed from the same input via scalar ops.
    // JIT should batch same-size same-input non-reduction kernels.
    using f = jit((x: np.Array) => {
      using a = x.add(1) as np.Array;
      using b = x.mul(2) as np.Array;
      return [a, b] as [np.Array, np.Array];
    });

    using x = np.array([1, 2, 3, 4]);
    const [r1, r2] = f(x);
    expect(r1.js()).toEqual([2, 3, 4, 5]); // x + 1
    expect(r2.js()).toEqual([2, 4, 6, 8]); // x * 2
    r1.dispose();
    r2.dispose();
  });

  it("three independent outputs are correct", () => {
    using f = jit((x: np.Array) => {
      using a = x.add(10) as np.Array;
      using b = x.mul(3) as np.Array;
      using c = x.sub(1) as np.Array;
      return [a, b, c] as [np.Array, np.Array, np.Array];
    });

    using x = np.array([2, 5]);
    const [a, b, c] = f(x);
    expect(a.js()).toEqual([12, 15]); // 2+10, 5+10
    expect(b.js()).toEqual([6, 15]); // 2*3, 5*3
    expect(c.js()).toEqual([1, 4]); // 2-1, 5-1
    a.dispose();
    b.dispose();
    c.dispose();
  });

  it("grad through function with multi-output opportunity", () => {
    // grad(f) produces backward pass with multiple same-size outputs
    const f = (x: np.Array) => {
      using a = x.mul(x) as np.Array; // x^2
      using b = x.add(x) as np.Array; // 2x
      using s = a.add(b) as np.Array; // x^2 + 2x
      return s.sum();
    };

    using x = np.array([1, 2, 3]);
    using dx = grad(f)(x);
    // d/dx(x^2 + 2x) = 2x + 2
    expect(dx).toBeAllclose([4, 6, 8]);
  });

  it("multi-output preserves correctness with different expressions", () => {
    // Each output uses a different binary expression on the same two inputs
    using f = jit((a: np.Array, b: np.Array) => {
      using r1 = a.add(b) as np.Array;
      using r2 = a.sub(b) as np.Array;
      using r3 = a.mul(b) as np.Array;
      return [r1, r2, r3] as [np.Array, np.Array, np.Array];
    });

    using a = np.array([10, 20, 30]);
    using b = np.array([1, 2, 3]);
    const [r1, r2, r3] = f(a, b);
    expect(r1.js()).toEqual([11, 22, 33]);
    expect(r2.js()).toEqual([9, 18, 27]);
    expect(r3.js()).toEqual([10, 40, 90]);
    r1.dispose();
    r2.dispose();
    r3.dispose();
  });

  it("reduction outputs remain correct (not merged with elementwise)", () => {
    // A function with both elementwise and reduction outputs.
    // The reduction has a different output size, so it stays solo.
    using f = jit((x: np.Array) => {
      using elem = x.mul(2) as np.Array;
      const reduced = x.sum(); // reduction → scalar
      return [elem, reduced] as [np.Array, np.Array];
    });

    using x = np.array([1, 2, 3]);
    const [elem, reduced] = f(x);
    expect(elem.js()).toEqual([2, 4, 6]);
    expect(reduced.js()).toBe(6);
    elem.dispose();
    reduced.dispose();
  });

  it("repeated calls produce consistent results", () => {
    using f = jit((x: np.Array) => {
      using a = x.add(1) as np.Array;
      using b = x.mul(2) as np.Array;
      return [a, b] as [np.Array, np.Array];
    });

    // First call — triggers tracing + compilation
    using x1 = np.array([10, 20]);
    const [a1, b1] = f(x1);
    expect(a1.js()).toEqual([11, 21]);
    expect(b1.js()).toEqual([20, 40]);
    a1.dispose();
    b1.dispose();

    // Second call — uses cached program
    using x2 = np.array([100, 200]);
    const [a2, b2] = f(x2);
    expect(a2.js()).toEqual([101, 201]);
    expect(b2.js()).toEqual([200, 400]);
    a2.dispose();
    b2.dispose();
  });
});

// ---------------------------------------------------------------------------
// WebGPU backend tests (only run if available)
// ---------------------------------------------------------------------------
describe("multi-output kernel (WebGPU)", () => {
  let prev: ReturnType<typeof defaultDevice>;
  let available = false;

  beforeAll(async () => {
    try {
      await init("webgpu");
      prev = defaultDevice("webgpu");
      available = true;
    } catch {
      // WebGPU not available, skip
    }
  });
  afterAll(() => {
    if (available) defaultDevice(prev);
  });

  it.skipIf(!available)(
    "two independent elementwise outputs are correct",
    () => {
      using f = jit((x: np.Array) => {
        using a = x.add(1) as np.Array;
        using b = x.mul(2) as np.Array;
        return [a, b] as [np.Array, np.Array];
      });

      using x = np.array([1, 2, 3, 4]);
      const [r1, r2] = f(x);
      expect(r1.js()).toEqual([2, 3, 4, 5]);
      expect(r2.js()).toEqual([2, 4, 6, 8]);
      r1.dispose();
      r2.dispose();
    },
  );

  it.skipIf(!available)("three independent outputs are correct", () => {
    using f = jit((x: np.Array) => {
      using a = x.add(10) as np.Array;
      using b = x.mul(3) as np.Array;
      using c = x.sub(1) as np.Array;
      return [a, b, c] as [np.Array, np.Array, np.Array];
    });

    using x = np.array([2, 5]);
    const [a, b, c] = f(x);
    expect(a.js()).toEqual([12, 15]);
    expect(b.js()).toEqual([6, 15]);
    expect(c.js()).toEqual([1, 4]);
    a.dispose();
    b.dispose();
    c.dispose();
  });

  it.skipIf(!available)("grad through multi-output opportunity", () => {
    const f = (x: np.Array) => {
      using a = x.mul(x) as np.Array;
      using b = x.add(x) as np.Array;
      using s = a.add(b) as np.Array;
      return s.sum();
    };

    using x = np.array([1, 2, 3]);
    using dx = grad(f)(x);
    expect(dx).toBeAllclose([4, 6, 8]);
  });
});
