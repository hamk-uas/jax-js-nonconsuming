import {
  defaultDevice,
  DType,
  grad,
  jit,
  jvp,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, expect, suite, test } from "vitest";

beforeEach(() => {
  defaultDevice("cpu"); // float16 is not available on Wasm
});

suite("dtype-op edge cases", () => {
  test("uint32 subtraction underflow", () => {
    using a = np.array(3, { dtype: np.uint32 });
    using b = np.array(5, { dtype: np.uint32 });
    using c = a.sub(b);
    expect(c.dtype).toBe(np.uint32);
    expect(c.js()).toEqual(4294967294); // 2^32 - 2
  });
});

suite("dtype promotion rules", () => {
  test("promote uint32 and int32 to int32", () => {
    using a = np.array(3, { dtype: np.uint32 });
    using b = np.array(-2, { dtype: np.int32 });
    using c = a.add(b);
    expect(c.dtype).toBe(np.int32);
    expect(c.js()).toEqual(1);
  });

  test("promote int32 and float16 to float16", () => {
    using a = np.array(3, { dtype: np.int32 });
    using b = np.array(2.5, { dtype: np.float16 });
    using c = a.mul(b);
    expect(c.dtype).toBe(np.float16);
    expect(c.js()).toEqual(7.5);
  });

  test("promote uint32 and float32 to float32", () => {
    using a = np.array(4, { dtype: np.uint32 });
    using b = np.array(1.5, { dtype: np.float32 });
    using c = a.sub(b);
    expect(c.dtype).toBe(np.float32);
    expect(c.js()).toEqual(2.5);
  });

  test("promote bool and int32 to int32", () => {
    using a = np.array(true, { dtype: np.bool });
    using b = np.array(10, { dtype: np.int32 });
    using c = a.add(b);
    expect(c.dtype).toBe(np.int32);
    expect(c.js()).toEqual(11);
  });

  test("promote float16 and float32 to float32", () => {
    using a = np.array(2.5, { dtype: np.float16 });
    using b = np.array(1.5, { dtype: np.float32 });
    using c = a.div(b);
    expect(c.dtype).toBe(np.float32);
    expect(c).toBeAllclose(2.5 / 1.5);
  });
});

suite("weak types", () => {
  test("number constants are weak", () => {
    using a = np.array(5);
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(true);
    using b = np.multiply(3, 5);
    expect(b.dtype).toBe(np.float32);
    expect(b.weakType).toBe(true);
  });

  test("bool constants are not weak type", () => {
    using a = np.array(true);
    expect(a.dtype).toBe(np.bool);
    expect(a.weakType).toBe(false);
    using b = np.array([true, false]);
    expect(b.dtype).toBe(np.bool);
    expect(b.weakType).toBe(false);
  });

  test("arrays of numbers are not weak", () => {
    using a = np.array([1, 2, 3]);
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(false);
  });

  test("constant as operand is cast to int32", () => {
    using a = np.array(5, { dtype: np.int32 });
    using b = a.add(3); // 3 is a JS number constant
    expect(b.dtype).toBe(np.int32);
    expect(b.weakType).toBe(false);
  });

  test("constant as operand is cast to uint32", () => {
    using a = np.array(5, { dtype: np.uint32 });
    using b = a.add(2.8); // Should truncate to 2, which fits in uint32
    expect(b.dtype).toBe(np.uint32);
    expect(b.weakType).toBe(false);
    expect(b.js()).toEqual(7);
  });

  test("ops preserve weak float", () => {
    using a = np.array(5, { dtype: np.int32 });
    using mul = np.multiply(3, 3);
    using b = a.add(mul);
    expect(b.dtype).toBe(np.int32);
    expect(b.weakType).toBe(false);
    expect(b.js()).toEqual(14);
  });

  test("weak type in jit constants", () => {
    using f = jit(() => {
      return np.sin(3);
    });
    using a = f();
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(true);
    using two = np.array(2, { dtype: np.float16 });
    using b = a.add(two);
    expect(b.dtype).toBe(np.float16);
    expect(b.weakType).toBe(false);
    expect(b.js()).toBeCloseTo(Math.sin(3) + 2, 2);
  });

  test("weak type added in jit op", () => {
    using f = jit((x: np.Array) => x.add(3));
    for (const dtype of [np.int32, np.float32]) {
      using a = np.array(4, { dtype });
      expect(a.weakType).toBe(false);
      using b = f(a);
      expect(b.dtype).toBe(dtype);
      expect(b.weakType).toBe(false);
      expect(b.js()).toEqual(7);
    }
  });

  test("weak type preserved by jit op", () => {
    using f = jit((x: np.Array) => x.add(3));
    using a = f(5); // should be weak
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(true);

    using twoI32 = np.array(2, { dtype: np.int32 });
    using b = a.add(twoI32);
    expect(b.dtype).toBe(np.int32);
    expect(b.weakType).toBe(false);
    expect(b.js()).toEqual(10);
  });
});

suite("cast autodiff", () => {
  test("grad through f32 -> f64 cast", () => {
    // cast is linear: grad should cast cotangent back to input dtype
    const f = (x: np.Array) => {
      using y = x.astype(DType.Float64);
      return y.sum();
    };
    using x = np.array([1.0, 2.0, 3.0]);
    using gx = grad(f)(x);
    expect(gx.dtype).toBe(DType.Float32);
    expect(gx.js()).toEqual([1, 1, 1]);
  });

  test("grad through f64 -> f32 cast", () => {
    const f = (x: np.Array) => {
      using y = x.astype(DType.Float32);
      return y.sum();
    };
    using x = np.array([1.0, 2.0, 3.0], { dtype: DType.Float64 });
    using gx = grad(f)(x);
    expect(gx.dtype).toBe(DType.Float64);
    expect(gx).toBeAllclose([1, 1, 1]);
  });

  test("jit(grad) through cast", () => {
    const f = (x: np.Array) => {
      using y = x.astype(DType.Float64);
      return y.sum();
    };
    using x = np.array([1.0, 2.0, 3.0]);
    using gx = jit(grad(f))(x);
    expect(gx.dtype).toBe(DType.Float32);
    expect(gx.js()).toEqual([1, 1, 1]);
  });

  test("grad through cast in expression chain", () => {
    // f(x) = sum(cast(2*x, f64)) → df/dx = 2
    const f = (x: np.Array) => {
      using scaled = x.mul(np.array(2.0));
      using casted = scaled.astype(DType.Float64);
      return casted.sum();
    };
    using x = np.array([1.0, 2.0, 3.0]);
    using gx = grad(f)(x);
    expect(gx.dtype).toBe(DType.Float32);
    expect(gx.js()).toEqual([2, 2, 2]);
  });

  test("jvp through cast preserves tangent dtype", () => {
    const f = (x: np.Array) => x.astype(DType.Float64);
    using x = np.array([1.0, 2.0]);
    using dx = np.array([1.0, 0.0]);
    const [y, dy] = jvp(f, [x], [dx]);
    expect(y.dtype).toBe(DType.Float64);
    expect(dy.dtype).toBe(DType.Float64);
    y.dispose();
    dy.dispose();
  });
});
