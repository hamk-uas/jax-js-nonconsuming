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
  nn,
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
// Reduction epilogue fusion tests
// ---------------------------------------------------------------------------
describe("reduction epilogue fusion (WASM)", () => {
  let prev: ReturnType<typeof defaultDevice>;
  beforeAll(() => {
    prev = defaultDevice("wasm");
  });
  afterAll(() => defaultDevice(prev));

  it("matmul + bias + relu fuses correctly", () => {
    // Dot (reduction) → Add (binary with external bias) → Max(0) (relu)
    // The epilogue chain should be: relu(acc + bias[gidx])
    using f = jit((A: np.Array, B: np.Array, bias: np.Array) => {
      using c = np.matmul(A, B) as np.Array;
      using d = c.add(bias) as np.Array;
      return nn.relu(d);
    });

    // A=[2,3], B=[3,2], bias=[2]
    using A = np.array([
      [1, 0, 2],
      [0, 3, 1],
    ]);
    using B = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    using bias = np.array([10, -100]);
    // matmul: [[1*1+0*3+2*5, 1*2+0*4+2*6], [0*1+3*3+1*5, 0*2+3*4+1*6]]
    //       = [[11, 14], [14, 18]]
    // + bias: [[21, -86], [24, -82]]
    // relu:   [[21, 0], [24, 0]]
    using result = f(A, B, bias);
    expect(result.js()).toEqual([
      [21, 0],
      [24, 0],
    ]);
  });

  it("sum().mul(2).add(1) chains two epilogue ops", () => {
    // Reduce(sum) → Mul(2) → Add(1)
    using f = jit((x: np.Array) => {
      using s = x.sum() as np.Array;
      using m = s.mul(2) as np.Array;
      return m.add(1);
    });

    using x = np.array([1, 2, 3, 4]); // sum=10, *2=20, +1=21
    using result = f(x);
    expect(result.js()).toBe(21);
  });

  it("dot + neg (unary epilogue)", () => {
    // Dot → Neg
    using f = jit((a: np.Array, b: np.Array) => {
      using d = np.dot(a, b) as np.Array;
      return np.negative(d);
    });

    using a = np.array([1, 2, 3]);
    using b = np.array([4, 5, 6]);
    // dot = 1*4+2*5+3*6 = 32, neg = -32
    using result = f(a, b);
    expect(result.js()).toBe(-32);
  });

  it("sum + exp (transcendental epilogue)", () => {
    // Reduce(sum) → Exp
    using f = jit((x: np.Array) => {
      using s = x.sum() as np.Array;
      return np.exp(s);
    });

    using x = np.array([1, 1]); // sum=2, exp(2)≈7.389
    using result = f(x);
    expect(result).toBeAllclose(Math.exp(2));
  });

  it("sum + where (ternary epilogue)", () => {
    // Reduce(sum) → Where(mask, sum_result, 0)
    using f = jit((x: np.Array, mask: np.Array) => {
      using s = x.sum() as np.Array;
      return np.where(mask, s, 0);
    });

    using x = np.array([1, 2, 3]); // sum=6
    using mask = np.array(true); // truthy → pick sum
    using result = f(x, mask);
    expect(result.js()).toBe(6);

    using mask2 = np.array(false); // falsy → pick 0
    using result2 = f(x, mask2);
    expect(result2.js()).toBe(0);
  });

  it("sum + compare (comparison epilogue)", () => {
    // Reduce(sum) → Compare(sum_result, threshold, gt)
    using f = jit((x: np.Array, threshold: np.Array) => {
      using s = x.sum() as np.Array;
      return np.greater(s, threshold);
    });

    using x = np.array([1, 2, 3]); // sum=6
    using t1 = np.array(5);
    using result1 = f(x, t1);
    expect(result1.js()).toBe(true); // 6 > 5 = true

    using t2 = np.array(10);
    using result2 = f(x, t2);
    expect(result2.js()).toBe(false); // 6 > 10 = false
  });

  it("matmul + scale preserves correctness", () => {
    // Dot → Mul(scalar)
    using f = jit((A: np.Array, B: np.Array) => {
      using c = np.matmul(A, B) as np.Array;
      return c.mul(0.5);
    });

    using A = np.array([[2, 4]]);
    using B = np.array([[1], [3]]); // matmul=[[14]], *0.5=[[7]]
    using result = f(A, B);
    expect(result).toBeAllclose([[7]]);
  });

  it("reduction epilogue does not fuse divergent consumers", () => {
    // sum() used twice → should NOT fuse, multi-use stops the chain
    using f = jit((x: np.Array) => {
      using s = x.sum() as np.Array;
      using a = s.mul(2) as np.Array;
      using b = s.add(1) as np.Array;
      return [a, b] as [np.Array, np.Array];
    });

    using x = np.array([1, 2, 3]); // sum=6
    const [a, b] = f(x);
    expect(a.js()).toBe(12); // 6*2
    expect(b.js()).toBe(7); // 6+1
    a.dispose();
    b.dispose();
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
