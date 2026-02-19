import {
  defaultDevice,
  devices,
  DType,
  grad,
  init,
  jit,
  jvp,
  lax,
  numpy as np,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.numpy.sum()", () => {
    test("can take multiple axes", () => {
      using _x = np.arange(24);
      using x = _x.reshape([2, 3, 4]);
      using y = x.sum([0, 2]);
      expect(y.js()).toEqual([60, 92, 124]);
    });

    test("keepdims preserves dim of size 1", () => {
      using _x = np.arange(24);
      using x = _x.reshape([2, 3, 4]);
      using y = x.sum([0, 2], { keepdims: true });
      expect(y.shape).toEqual([1, 3, 1]);
      expect(y.js()).toEqual([[[60], [92], [124]]]);
    });

    test("is identity on empty axes", () => {
      using _x = np.arange(24);
      using x = _x.reshape([2, 3, 4]);
      // eslint-disable-next-line jax-js/require-using -- sum([]) may return same backing object
      const y = x.sum([]); // may return same object — no using
      expect(x.js()).toEqual(y.js());
    });
  });

  suite("jax.numpy.cumsum()", () => {
    test("computes cumsum along axis", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      using y = np.cumsum(x, 0);
      expect(y.js()).toEqual([
        [1, 2, 3],
        [5, 7, 9],
      ]);
      using z = np.cumsum(x, 1);
      expect(z.js()).toEqual([
        [1, 3, 6],
        [4, 9, 15],
      ]);
    });
  });

  suite("jax.numpy.eye()", () => {
    test("computes a square matrix", () => {
      using x = np.eye(3);
      expect(x).toBeAllclose([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });

    test("computes a rectangular matrix", () => {
      using x = np.eye(2, 3);
      expect(x).toBeAllclose([
        [1, 0, 0],
        [0, 1, 0],
      ]);
    });

    test("can be multiplied", () => {
      using e = np.eye(3, 5);
      using x = e.mul(-42);
      using s = x.sum();
      expect(s).toBeAllclose(-126);
      expect(x).toBeAllclose([
        [-42, 0, 0, 0, 0],
        [0, -42, 0, 0, 0],
        [0, 0, -42, 0, 0],
      ]);
    });
  });

  suite("jax.numpy.diag()", () => {
    test("constructs diagonal from 1D array", () => {
      using x = np.array([1, 2, 3]);
      using y = np.diag(x);
      expect(y.js()).toEqual([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
      ]);
    });

    test("fetches diagonal of 2D array", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      using y = np.diag(x);
      expect(y.js()).toEqual([1, 5, 9]);
      using z = np.diag(x, 1);
      expect(z.js()).toEqual([2, 6]);
    });

    test("can construct off-diagonal", () => {
      {
        using a = np.array([1, 2]);
        using d1 = np.diag(a, 1);
        expect(d1.js()).toEqual([
          [0, 1, 0],
          [0, 0, 2],
          [0, 0, 0],
        ]);
      }
      {
        using a = np.array([1, 2]);
        using d2 = np.diag(a, -2);
        expect(d2.js()).toEqual([
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [1, 0, 0, 0],
          [0, 2, 0, 0],
        ]);
      }
    });
  });

  suite("jax.numpy.diagonal()", () => {
    test("diagonal defaults to first two axes", () => {
      using _a = np.arange(4);
      using a = _a.reshape([2, 2]);
      {
        using d = a.diagonal();
        expect(d.js()).toEqual([0, 3]);
      }
      {
        using d = a.diagonal(1);
        expect(d.js()).toEqual([1]);
      }
      {
        using d = a.diagonal(-1);
        expect(d.js()).toEqual([2]);
      }

      using _b = np.arange(8);
      using b = _b.reshape([2, 2, 2]);
      {
        using d = b.diagonal();
        expect(d.js()).toEqual([
          [0, 6],
          [1, 7],
        ]);
      }
    });

    test("can take diagonal over other axes", () => {
      using _a = np.arange(12);
      using a = _a.reshape([3, 2, 2]);
      {
        using d = a.diagonal(0, 1, 2);
        expect(d.js()).toEqual([
          [0, 3],
          [4, 7],
          [8, 11],
        ]);
      }

      // a[:, :, 0] = [[0, 2], [4, 6], [8, 10]]
      {
        using d = np.diagonal(a, 0, 0, 1);
        expect(d.js()).toEqual([
          [0, 6],
          [1, 7],
        ]);
      }
      {
        using d = np.diagonal(a, 1, 0, 1);
        expect(d.js()).toEqual([[2], [3]]);
      }
      {
        using d = np.diagonal(a, 1, 1, 0);
        expect(d.js()).toEqual([
          [4, 10],
          [5, 11],
        ]);
      }
    });

    test("gradient over diagonal sum-of-squares", () => {
      using __a = np.arange(6);
      using _a = __a.astype(np.float32);
      using a = _a.reshape([2, 3]);
      const f = (a: np.Array) => a.mul(a).diagonal(1).sum();
      using g = grad(f)(a);
      expect(g.js()).toEqual([
        [0, 2, 0],
        [0, 0, 10],
      ]);
    });

    test("computes trace", () => {
      using _x = np.arange(9);
      using x = _x.reshape([3, 3]);
      {
        using t = np.trace(x);
        expect(t.js()).toEqual(12);
      }
      {
        using t = np.trace(x, 1);
        expect(t.js()).toEqual(6);
      }
    });
  });

  suite("jax.numpy.tri()", () => {
    test("computes lower-triangular matrix", () => {
      using x = np.tri(3);
      expect(x.js()).toEqual([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
      ]);
    });

    test("computes rectangular lower-triangular matrix", () => {
      using x = np.tri(2, 4, 1);
      expect(x.js()).toEqual([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
      ]);
    });

    test("triu works", () => {
      using _x = np.arange(24);
      using x = _x.reshape([2, 3, 4]);
      using y = np.triu(x);
      expect(y.js()).toEqual([
        [
          [0, 1, 2, 3],
          [0, 5, 6, 7],
          [0, 0, 10, 11],
        ],
        [
          [12, 13, 14, 15],
          [0, 17, 18, 19],
          [0, 0, 22, 23],
        ],
      ]);
    });

    test("tril works", () => {
      using _x = np.arange(24);
      using x = _x.reshape([2, 3, 4]);
      using y = np.tril(x);
      expect(y.js()).toEqual([
        [
          [0, 0, 0, 0],
          [4, 5, 0, 0],
          [8, 9, 10, 0],
        ],
        [
          [12, 0, 0, 0],
          [16, 17, 0, 0],
          [20, 21, 22, 0],
        ],
      ]);
    });
  });

  suite("jax.numpy.arange()", () => {
    test("can be called with 1 argument", () => {
      let x = np.arange(5);
      expect(x.js()).toEqual([0, 1, 2, 3, 4]);

      x.dispose();
      x = np.arange(0);
      expect(x.js()).toEqual([]);

      x.dispose();
      x = np.arange(-10);
      expect(x.js()).toEqual([]);
      x.dispose();
    });

    test("can be called with 2 arguments", () => {
      let x = np.arange(50, 60);
      expect(x.js()).toEqual([50, 51, 52, 53, 54, 55, 56, 57, 58, 59]);

      x.dispose();
      x = np.arange(-10, -5);
      expect(x.js()).toEqual([-10, -9, -8, -7, -6]);
      x.dispose();
    });

    test("can be called with 3 arguments", () => {
      let x = np.arange(0, 10, 2);
      expect(x.js()).toEqual([0, 2, 4, 6, 8]);

      x.dispose();
      x = np.arange(10, 0, -2);
      expect(x.js()).toEqual([10, 8, 6, 4, 2]);

      x.dispose();
      x = np.arange(0, -10, -2);
      expect(x.js()).toEqual([0, -2, -4, -6, -8]);
      x.dispose();
    });

    test("works with non-integer step", () => {
      // By default, it uses Int32 dtype, so this rounds down.
      let x = np.arange(0, 1, 0.2);
      expect(x.js()).toEqual([0, 0, 0, 0, 0]);

      // Explicitly set dtype to Float32.
      x.dispose();
      x = np.arange(0, 1, 0.2, { dtype: np.float32 });
      expect(x).toBeAllclose([0, 0.2, 0.4, 0.6, 0.8]);
      x.dispose();
    });
  });

  suite("jax.numpy.linspace()", () => {
    test("creates a linear space with 5 elements", () => {
      using x = np.linspace(0, 1, 5);
      expect(x.js()).toEqual([0, 0.25, 0.5, 0.75, 1]);
    });

    test("creates a linear space with 1-3 elements", () => {
      let x = np.linspace(0, 1, 3);
      expect(x.js()).toEqual([0, 0.5, 1]);

      x.dispose();
      x = np.linspace(0, 1, 2);
      expect(x.js()).toEqual([0, 1]);

      x.dispose();
      x = np.linspace(0, 1, 1);
      expect(x.js()).toEqual([0]);
      x.dispose();
    });

    test("defaults to 50 elements", () => {
      using x = np.linspace(0, 1);
      expect(x.shape).toEqual([50]);
      const ar = x.js() as number[];
      expect(ar[0]).toEqual(0);
      expect(ar[49]).toEqual(1);
      expect(ar[25]).toBeCloseTo(25 / 49);
    });
  });

  suite("jax.numpy.logspace()", () => {
    test("creates log-spaced values with base 10", () => {
      // logspace(0, 2, 3) should give 10^0, 10^1, 10^2 = [1, 10, 100]
      using x = np.logspace(0, 2, 3);
      expect(x.js()).toBeAllclose([1, 10, 100]);
    });

    test("creates log-spaced values with base 2", () => {
      // logspace(0, 3, 4, base=2) should give 2^0, 2^1, 2^2, 2^3 = [1, 2, 4, 8]
      using x = np.logspace(0, 3, 4, true, 2);
      expect(x.js()).toBeAllclose([1, 2, 4, 8]);
    });

    test("handles endpoint=false", () => {
      // logspace(0, 2, 4, endpoint=false) should give [1, ~3.16, 10, ~31.6]
      using x = np.logspace(0, 2, 4, false);
      const result = x.js() as number[];
      expect(result[0]).toBeCloseTo(1, 5);
      expect(result[1]).toBeCloseTo(Math.pow(10, 0.5), 5);
      expect(result[2]).toBeCloseTo(10, 5);
      expect(result[3]).toBeCloseTo(Math.pow(10, 1.5), 5);
    });

    test("defaults to 50 elements with base 10", () => {
      using x = np.logspace(0, 1);
      expect(x.shape).toEqual([50]);
      const ar = x.js() as number[];
      expect(ar[0]).toBeCloseTo(1, 5); // 10^0
      expect(ar[49]).toBeCloseTo(10, 5); // 10^1
    });
  });

  suite("jax.numpy.where()", () => {
    test("computes where", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 5, 6]);
      using z = np.array([true, false, true]);
      using result = np.where(z, x, y);
      expect(result.js()).toEqual([1, 5, 3]);
    });

    test("works with jvp", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 5, 6]);
      using z = np.array([true, false, true]);
      using t1 = np.array([1, 1, 1]);
      using t2 = np.zeros([3]);
      const [primal, tangent] = jvp(
        (x: np.Array, y: np.Array) => np.where(z, x, y),
        [x, y],
        [t1, t2],
      );
      using _p = primal;
      using _t = tangent;
      expect(primal.js()).toEqual([1, 5, 3]);
      expect(tangent.js()).toEqual([1, 0, 1]);
    });

    test("works with grad reverse-mode", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 5, 6]);
      using z = np.array([true, false, true]);
      const f = ({ x, y }: { x: np.Array; y: np.Array }) =>
        np.where(z, x, y).sum();
      const grads = grad(f)({ x, y });
      expect(grads.x.js()).toEqual([1, 0, 1]);
      expect(grads.y.js()).toEqual([0, 1, 0]);
      grads.x.dispose();
      grads.y.dispose();
    });

    test("where broadcasting", () => {
      using z = np.array([true, false, true, true]);
      using r1 = np.where(z, 1, 3);
      expect(r1.js()).toEqual([1, 3, 1, 1]);
      using r2 = np.where(false, 1, 3);
      expect(r2.js()).toEqual(3);
      {
        using a = np.array([10, 11]);
        using r3 = np.where(false, 1, a);
        expect(r3.js()).toEqual([10, 11]);
      }
      {
        using a = np.array([10, 11, 12]);
        using r4 = np.where(true, 7, a);
        expect(r4.js()).toEqual([7, 7, 7]);
      }
    });
  });

  suite("jax.numpy.equal()", () => {
    test("computes equal", () => {
      using x = np.array([1, 2, 3, 4]);
      using y = np.array([4, 5, 3, 4]);
      using eq = np.equal(x, y);
      expect(eq.js()).toEqual([false, false, true, true]);
      using ne = np.notEqual(x, y);
      expect(ne.js()).toEqual([true, true, false, false]);
    });

    test("does not propagate gradients", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([0, 5, 6]);
      const f = ({ x, y }: { x: np.Array; y: np.Array }) =>
        np.where(np.equal(x, y), 1, 0).sum();
      const grads = grad(f)({ x, y });
      expect(grads.x.js()).toEqual([0, 0, 0]);
      expect(grads.y.js()).toEqual([0, 0, 0]);
      grads.x.dispose();
      grads.y.dispose();
    });
  });

  suite("jax.numpy.transpose()", () => {
    test("transposes a 1D array (no-op)", () => {
      using x = np.array([1, 2, 3]);
      using y = np.transpose(x);
      expect(y.js()).toEqual([1, 2, 3]);
    });

    test("transposes a 2D array", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      using y = np.transpose(x);
      expect(y.js()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    test("composes with jvp", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      using t = np.ones([2, 3]);
      const [y, dy] = jvp(
        (x: np.Array) => x.transpose().mul(x.transpose()),
        [x],
        [t],
      );
      using _y = y;
      using _dy = dy;
      {
        using xsq = x.mul(x);
        using expected = xsq.transpose();
        expect(y).toBeAllclose(expected);
      }
      {
        using x2 = x.mul(2);
        using expected = x2.transpose();
        expect(dy).toBeAllclose(expected);
      }
    });

    test("composes with grad", () => {
      using x = np.ones([3, 4]);
      using dx = grad((x: np.Array) => x.transpose().sum())(x);
      expect(dx).toBeAllclose(x);
    });
  });

  suite("jax.numpy.swapaxes()", () => {
    test("swaps axis of an array", () => {
      using _x = np.arange(12);
      using x = _x.reshape([2, 2, 3]);
      using s = np.swapaxes(x, 1, 2);
      expect(s.js()).toEqual([
        [
          [0, 3],
          [1, 4],
          [2, 5],
        ],
        [
          [6, 9],
          [7, 10],
          [8, 11],
        ],
      ]);
    });
  });

  suite("jax.numpy.reshape()", () => {
    test("reshapes a 1D array", () => {
      using x = np.array([1, 2, 3, 4]);
      using y = np.reshape(x, [2, -1]);
      expect(y.js()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    test("raises Error on incompatible shapes", () => {
      using x = np.array([1, 2, 3, 4]);
      expect(() => np.reshape(x, [3, 2])).toThrow(Error);
      expect(() => np.reshape(x, [2, 3])).toThrow(Error);
      expect(() => np.reshape(x, [2, 2, 2])).toThrow(Error);
      expect(() => np.reshape(x, [3, -1])).toThrow(Error);
      expect(() => np.reshape(x, [-1, -1])).toThrow(Error);
    });

    test("composes with jvp", () => {
      using x = np.array([1, 2, 3, 4]);
      using t = np.ones([4]);
      const [y, dy] = jvp(
        (x: np.Array) => np.reshape(x, [2, 2]).sum(),
        [x],
        [t],
      );
      using _y = y;
      using _dy = dy;
      expect(y).toBeAllclose(10);
      expect(dy).toBeAllclose(4);
    });
  });

  suite("jax.numpy.flip()", () => {
    test("flips a 1D array", () => {
      using x = np.array([1, 2, 3]);
      using f = np.flip(x);
      expect(f.js()).toEqual([3, 2, 1]);
    });

    test("flips a 2D array", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      {
        using f = np.flip(x);
        expect(f.js()).toEqual([
          [6, 5, 4],
          [3, 2, 1],
        ]);
      }
      {
        using f = np.flip(x, 0);
        expect(f.js()).toEqual([
          [4, 5, 6],
          [1, 2, 3],
        ]);
      }
      {
        using f = np.flip(x, 1);
        expect(f.js()).toEqual([
          [3, 2, 1],
          [6, 5, 4],
        ]);
      }
    });
  });

  suite("jax.numpy.matmul()", () => {
    test("acts as vector dot product", () => {
      using x = np.array([1, 2, 3, 4]);
      using y = np.array([10, 100, 1000, 1]);
      using z = np.matmul(x, y);
      expect(z.js()).toEqual(3214);
    });

    test("computes 2x2 matmul", () => {
      using x = np.array([
        [1, 2],
        [3, 4],
      ]);
      using y = np.array([
        [5, 6],
        [7, 8],
      ]);
      using z = np.matmul(x, y);
      expect(z.js()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    test("computes 2x3 matmul", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      using y = np.array([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);
      using z = np.matmul(x, y);
      expect(z.js()).toEqual([
        [58, 64],
        [139, 154],
      ]);
    });

    test("computes stacked 3x3 matmul", () => {
      using a = np.array([
        [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
        [
          [10, 11, 12],
          [13, 14, 15],
          [16, 17, 18],
        ],
      ]);
      using b = np.array([
        [20, 21, 22],
        [23, 24, 25],
        [26, 27, 28],
      ]);
      using c = np.matmul(a, b);
      expect(c.shape).toEqual([2, 3, 3]);
      expect(c.js()).toEqual([
        [
          [144, 150, 156],
          [351, 366, 381],
          [558, 582, 606],
        ],
        [
          [765, 798, 831],
          [972, 1014, 1056],
          [1179, 1230, 1281],
        ],
      ]);
    });

    test("jit with fused bias and relu", () => {
      using matmulWithBiasAndRelu = jit(
        (x: np.Array, w: np.Array, b: np.Array) => {
          using y = np.matmul(x, w).add(b);
          return np.maximum(y, 0);
        },
      );

      using x = np.array([
        [1, -1],
        [-1, 1],
      ]);
      using w = np.array([
        [2, 3],
        [4, 6],
      ]);
      using b = np.array([10, -10]);

      using y = matmulWithBiasAndRelu(x, w, b);
      expect(y.js()).toEqual([
        [8, 0],
        [12, 0],
      ]);
    });
  });

  suite("jax.numpy.dot()", () => {
    test("acts as scalar multiplication", () => {
      using z = np.dot(3, 4);
      expect(z.js()).toEqual(12);
    });

    test("computes 1D dot product", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 5, 6]);
      using z = np.dot(x, y);
      expect(z.js()).toEqual(32);
    });

    test("computes 2D dot product", () => {
      using x = np.array([
        [1, 2],
        [3, 4],
      ]);
      using y = np.array([
        [5, 6],
        [7, 8],
      ]);
      using z = np.dot(x, y);
      expect(z.js()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    test("produces correct shape", () => {
      using x = np.zeros([2, 3, 4, 5]);
      using y = np.zeros([1, 4, 5, 6]);
      using z = np.dot(x, y);
      expect(z.shape).toEqual([2, 3, 4, 1, 4, 6]);
    });

    if (device !== "cpu") {
      test("200-256-200 matrix product", async () => {
        using _a = np.arange(200);
        using _b = _a.astype(np.float32);
        using _c = _b.reshape([200, 1]);
        using _d = np.ones([200, 256]);
        using x = _c.mul(_d);
        using y = np.ones([256, 200]);
        await Promise.all([x.data(), y.data()]);
        using dotResult = np.dot(x, y);
        const buf = await dotResult.data();
        expect(buf.length).toEqual(200 * 200);
        expect(buf[0]).toEqual(0);
        expect(buf[200]).toEqual(256);
        expect(buf[200 * 200 - 1]).toEqual(199 * 256);
      });
    }

    // This test observes a past tuning / shape tracking issue where indices
    // would be improperly calculated applying the Unroll optimization.
    test("1-784-10 matrix product", async () => {
      using __x = np.arange(784);
      using _x = __x.astype(np.float32);
      using x = _x.reshape([1, 784]);
      using y = np.ones([784, 10]);
      await Promise.all([x.data(), y.data()]);
      using dotResult = np.dot(x, y);
      const buf = await dotResult.data();
      expect(buf.length).toEqual(10);
      expect(buf).toEqual(
        new Float32Array(Array.from({ length: 10 }, () => (784 * 783) / 2)),
      );
    });
  });

  suite("jax.numpy.tensordot()", () => {
    test("2-3-4 with 3-4-5", async () => {
      using _x1 = np.arange(24);
      using x1 = _x1.reshape([2, 3, 4]);
      using x2 = np.ones([3, 4, 5]);
      let z = np.tensordot(x1, x2);
      expect(await z.jsAsync()).toEqual([
        [66, 66, 66, 66, 66],
        [210, 210, 210, 210, 210],
      ]);
      // Equivalent to the above as explicit sequences.
      z.dispose();
      z = np.tensordot(x1, x2, [
        [1, 2],
        [0, 1],
      ]);
      expect(await z.jsAsync()).toEqual([
        [66, 66, 66, 66, 66],
        [210, 210, 210, 210, 210],
      ]);
      z.dispose();
    });
  });

  suite("jax.numpy.einsum()", () => {
    test("basic einsum matmul", () => {
      using _a = np.arange(6);
      using a = _a.reshape([2, 3]);
      using b = np.ones([3, 4]);
      using c = np.einsum("ik,kj->ij", a, b);
      expect(c.js()).toEqual([
        [3, 3, 3, 3],
        [12, 12, 12, 12],
      ]);
    });

    test("einsum one-array sums", () => {
      using _a = np.arange(6);
      using a = _a.reshape([2, 3]);
      let c = np.einsum("ij->", a);
      expect(c.js()).toEqual(15);

      c.dispose();
      c = np.einsum(a, [0, 1], []);
      expect(c.js()).toEqual(15);

      c.dispose();
      c = np.einsum(a, [0, 1], []);
      expect(c.js()).toEqual(15);

      c.dispose();
      c = np.einsum("ij->j", a);
      expect(c.js()).toEqual([3, 5, 7]);

      c.dispose();
      c = np.einsum("ji->j", a);
      expect(c.js()).toEqual([3, 12]);

      c.dispose();
      using _sliced = a.slice([0, 2], [1, 3]);
      c = np.einsum("ii->", _sliced);
      expect(c.js()).toEqual(6);
      c.dispose();
    });

    test("einsum transposition", () => {
      using _a = np.arange(6);
      using a = _a.reshape([2, 3]);
      using b = np.einsum("ji", a);
      expect(b.js()).toEqual([
        [0, 3],
        [1, 4],
        [2, 5],
      ]);
    });

    test("examples from jax docs", () => {
      // https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html
      using _M = np.arange(16);
      using M = _M.reshape([4, 4]);
      using x = np.arange(4);
      using y = np.array([5, 4, 3, 2]);

      const results: np.Array[] = [];
      const e = (...args: any[]) => {
        const r = np.einsum(...args);
        results.push(r);
        return r;
      };

      // Vector product
      expect(e("i,i", x, y).js()).toEqual(16);
      expect(e("i,i->", x, y).js()).toEqual(16);
      expect(e(x, [0], y, [0]).js()).toEqual(16);
      expect(e(x, [0], y, [0], []).js()).toEqual(16);

      // Matrix product
      expect(e("ij,j->i", M, x).js()).toEqual([14, 38, 62, 86]);
      expect(e("ij,j", M, x).js()).toEqual([14, 38, 62, 86]);
      expect(e(M, [0, 1], x, [1], [0]).js()).toEqual([14, 38, 62, 86]);
      expect(e(M, [0, 1], x, [1]).js()).toEqual([14, 38, 62, 86]);

      // Outer product
      const outerExpected = [
        [0, 0, 0, 0],
        [5, 4, 3, 2],
        [10, 8, 6, 4],
        [15, 12, 9, 6],
      ];
      expect(e("i,j->ij", x, y).js()).toEqual(outerExpected);
      expect(e("i,j", x, y).js()).toEqual(outerExpected);
      expect(e(x, [0], y, [1], [0, 1]).js()).toEqual(outerExpected);
      expect(e(x, [0], y, [1]).js()).toEqual(outerExpected);

      // 1D array sum
      expect(e("i->", x).js()).toEqual(6);
      expect(e(x, [0], []).js()).toEqual(6);

      // Sum along an axis
      expect(e("...j->...", M).js()).toEqual([6, 22, 38, 54]);

      // Matrix transpose
      using y2 = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const transposeExpected = [
        [1, 4],
        [2, 5],
        [3, 6],
      ];
      expect(e("ij->ji", y2).js()).toEqual(transposeExpected);
      expect(e("ji", y2).js()).toEqual(transposeExpected);
      expect(e(y2, [1, 0]).js()).toEqual(transposeExpected);
      expect(e(y2, [0, 1], [1, 0]).js()).toEqual(transposeExpected);

      // Matrix diagonal
      expect(e("ii->i", M).js()).toEqual([0, 5, 10, 15]);

      // Matrix trace
      expect(e("ii", M).js()).toEqual(30);

      // Tensor products
      using _tx = np.arange(30);
      using tx = _tx.reshape([2, 3, 5]);
      using _ty = np.arange(60);
      using ty = _ty.reshape([3, 4, 5]);
      const tensorExpected = [
        [3340, 3865, 4390, 4915],
        [8290, 9940, 11590, 13240],
      ];
      expect(e("ijk,jlk->il", tx, ty).js()).toEqual(tensorExpected);
      expect(e("ijk,jlk", tx, ty).js()).toEqual(tensorExpected);
      expect(e(tx, [0, 1, 2], ty, [1, 3, 2], [0, 3]).js()).toEqual(
        tensorExpected,
      );
      expect(e(tx, [0, 1, 2], ty, [1, 3, 2]).js()).toEqual(tensorExpected);

      // Chained dot products
      using _w = np.arange(5, 9);
      using w = _w.reshape([2, 2]);
      using _cx = np.arange(6);
      using cx = _cx.reshape([2, 3]);
      using _cy = np.arange(-2, 4);
      using cy = _cy.reshape([3, 2]);
      using z = np.array([
        [2, 4, 6],
        [3, 5, 7],
      ]);
      const chainedExpected = [
        [481, 831, 1181],
        [651, 1125, 1599],
      ];
      expect(e("ij,jk,kl,lm->im", w, cx, cy, z).js()).toEqual(chainedExpected);
      expect(e(w, [0, 1], cx, [1, 2], cy, [2, 3], z, [3, 4]).js()).toEqual(
        chainedExpected,
      );

      for (const r of results) r.dispose();
    });

    test("shape tests", () => {
      const checkEinsumShapes = (expr: string, ...shapes: number[][]) => {
        const inputs = shapes.slice(0, -1).map((shape) => np.zeros(shape));
        using result = np.einsum(expr, ...inputs);
        expect(result.shape).toEqual(shapes[shapes.length - 1]);
        for (const inp of inputs) inp.dispose();
      };

      // Tests without ellipsis
      checkEinsumShapes("", [], []);
      checkEinsumShapes("i,i->", [3], [3], []);
      checkEinsumShapes("ijj->i", [2, 3, 3], [2]);
      checkEinsumShapes("i,i->i", [3], [3], [3]);
      checkEinsumShapes("ij,j->i", [2, 3], [3], [2]);
      checkEinsumShapes("ij,ji", [3, 4], [4, 3], []);
      checkEinsumShapes("ij,jk", [2, 3], [3, 4], [2, 4]);
      checkEinsumShapes("ij,jk->ki", [2, 3], [3, 4], [4, 2]);
      checkEinsumShapes("abc,cde->abde", [2, 3, 4], [4, 5, 6], [2, 3, 5, 6]);
      checkEinsumShapes(
        "abcd,cdef->abef",
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [2, 3, 6, 7],
      );
      checkEinsumShapes(
        "abcd,efcd->abef",
        [2, 3, 4, 5],
        [6, 7, 4, 5],
        [2, 3, 6, 7],
      );
      checkEinsumShapes(
        "abc,bcd,efa,fab",
        [2, 3, 4],
        [3, 4, 5],
        [10, 6, 2],
        [6, 2, 3],
        [5, 10],
      );

      // Tests with ellipsis (can be in middle of indices)
      checkEinsumShapes("...", [5, 1], [5, 1]);
      checkEinsumShapes("i...", [5, 1], [1, 5]);
      checkEinsumShapes("...,...->...", [2, 3, 4], [3, 4], [2, 3, 4]);
      checkEinsumShapes("...i,i->...", [2, 3, 4], [4], [2, 3]);
      checkEinsumShapes("i,...i->...", [4], [2, 3, 4], [2, 3]);
      checkEinsumShapes("...ij,jk->...ik", [5, 2, 3], [3, 4], [5, 2, 4]);
      checkEinsumShapes(
        "...ij,...jk->...ik",
        [6, 5, 2, 3],
        [5, 3, 4],
        [6, 5, 2, 4],
      );
      checkEinsumShapes(
        "ab...cd,cd...ef->ab...ef",
        [2, 3, 4, 5, 6, 7],
        [6, 7, 8, 9],
        [2, 3, 4, 5, 8, 9],
      );

      // Tests with broadcasting dims
      checkEinsumShapes("ii->i", [3, 3], [3]);
      checkEinsumShapes("ii->i", [3, 1], [1]);
      checkEinsumShapes("i,i->i", [3], [1], [3]);
      checkEinsumShapes("i,i->i", [1], [3], [3]);
      checkEinsumShapes("ii,i->i", [3, 3], [1], [3]);
      checkEinsumShapes("ii,i->i", [1, 1], [3], [3]);
      checkEinsumShapes("ij,ij->ij", [1, 10], [5, 1], [5, 10]);
      checkEinsumShapes("...,...->...", [1, 10], [5, 1], [5, 10]);
    });

    test("jit(scan) with einsum in body", async () => {
      // Regression: jit(scan(body_with_einsum)) crashed on WebGPU because the
      // Scan equation has many inputs (consts + carry + xs) that can exceed
      // backend.maxArgs. splitGraphDataflow must skip the maxArgs check for
      // equations whose outputs are already black nodes (like Scan).
      using G = np.array(
        [
          [1, 0],
          [1, 1],
        ],
        { dtype: DType.Float32 },
      );
      using W = np.array(
        [
          [1, 0],
          [0, 1],
        ],
        { dtype: DType.Float32 },
      );
      using one = np.array([[1.0]], { dtype: DType.Float32 });

      const step = (
        carry: { x: np.Array; C: np.Array },
        inp: { y: np.Array; V2: np.Array; FF: np.Array },
      ): [{ x: np.Array; C: np.Array }, { xPred: np.Array; Cp: np.Array }] => {
        const { x, C } = carry;
        const yi = inp.y;
        const FFi = inp.FF;
        const V2i = inp.V2;

        using isNan = np.isnan(yi);
        using zero = np.zerosLike(yi);
        using mask = np.where(isNan, zero, one);
        using ySafe = np.where(isNan, zero, yi);

        using FFx = np.matmul(FFi, x);
        using diff = np.subtract(ySafe, FFx);
        using v = np.multiply(mask, diff);
        // Three-operand einsum: batched matmul-transpose pattern
        using FCFt = np.einsum("ij,jk,lk->il", FFi, C, FFi);
        using Cp = np.add(FCFt, V2i);
        using GCFt = np.einsum("ij,jk,lk->il", G, C, FFi);
        using CpInv = np.reciprocal(Cp);
        using GCFtS = np.multiply(GCFt, CpInv);
        using K = np.multiply(mask, GCFtS);
        using KFF = np.matmul(K, FFi);
        using L = np.subtract(G, KFF);
        using Gx = np.matmul(G, x);
        using Kv = np.matmul(K, v);
        const xNext = np.add(Gx, Kv);
        using GCLt = np.einsum("ij,jk,lk->il", G, C, L);
        const CNext = np.add(GCLt, W);

        return [
          { x: xNext, C: CNext },
          { xPred: x, Cp },
        ];
      };

      const n = 5;
      using yArr = np.array(
        globalThis.Array.from({ length: n }, () => [[1000]]),
        { dtype: DType.Float32 },
      );
      using V2Arr = np.array(
        globalThis.Array.from({ length: n }, () => [[14400]]),
        { dtype: DType.Float32 },
      );
      using FFBase = np.array([[[1, 0]]], { dtype: DType.Float32 });
      using FFArr = np.tile(FFBase, [n, 1, 1]);
      using x0 = np.array([[0], [0]], { dtype: DType.Float32 });
      using C0 = np.array(
        [
          [1e7, 0],
          [0, 1e7],
        ],
        { dtype: DType.Float32 },
      );

      const core = (
        x0_: np.Array,
        C0_: np.Array,
        y_: np.Array,
        V2_: np.Array,
        FF_: np.Array,
      ) => {
        const [carry, out] = lax.scan(
          step,
          { x: x0_, C: C0_ },
          { y: y_, V2: V2_, FF: FF_ },
        );
        tree.dispose(carry);
        return out!;
      };

      using f = jit(core);
      const out = f(x0, C0, yArr, V2Arr, FFArr) as {
        xPred: np.Array;
        Cp: np.Array;
      };
      const xPred = await out.xPred.data();
      // First prediction should be [0, 0] (initial state)
      expect(xPred[0]).toBeCloseTo(0, 1);
      expect(xPred[1]).toBeCloseTo(0, 1);
      tree.dispose(out);
    });
  });

  const _JsArray = globalThis.Array;

  suite("jax.numpy.meshgrid()", () => {
    test("creates xy meshgrid", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 5]);
      const [X, Y] = np.meshgrid([x, y]);
      using _X = X;
      using _Y = Y;
      expect(X.js()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
      ]);
      expect(Y.js()).toEqual([
        [4, 4, 4],
        [5, 5, 5],
      ]);
    });

    test("works with ij indexing", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 5]);
      const [X, Y] = np.meshgrid([x, y], { indexing: "ij" });
      using _X = X;
      using _Y = Y;
      expect(X.js()).toEqual([
        [1, 1],
        [2, 2],
        [3, 3],
      ]);
      expect(Y.js()).toEqual([
        [4, 5],
        [4, 5],
        [4, 5],
      ]);
    });

    test("works with 3D arrays", () => {
      // Note: XYZ -> [Y, X, Z]
      using x = np.array([1, 2]);
      using y = np.array([3, 4, 5]);
      using z = np.array([6, 7, 8, 9]);
      const [X, Y, Z] = np.meshgrid([x, y, z]); // "xy" indexing
      using _X = X;
      using _Y = Y;
      using _Z = Z;
      expect(X.shape).toEqual([3, 2, 4]);
      expect(Y.shape).toEqual([3, 2, 4]);
      expect(Z.shape).toEqual([3, 2, 4]);
    });
  });

  suite("jax.numpy.minimum()", () => {
    test("computes element-wise minimum", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 2, 0]);
      using z = np.minimum(x, y);
      expect(z.js()).toEqual([1, 2, 0]);
    });

    test("works with jvp", () => {
      using x = np.array([1, 3, 3]);
      using y = np.array([4, 2, 0]);
      using t1 = np.ones([3]);
      using t2 = np.zeros([3]);
      const [z, dz] = jvp(
        (x: np.Array, y: np.Array) => np.minimum(x, y),
        [x, y],
        [t1, t2],
      );
      using _z = z;
      using _dz = dz;
      expect(z.js()).toEqual([1, 2, 0]);
      expect(dz.js()).toEqual([1, 0, 0]);
    });

    test("minimum of bools", () => {
      using x = np.array([true, false, true]);
      using y = np.array([false, false, true]);
      using z = np.minimum(x, y);
      expect(z.js()).toEqual([false, false, true]);
    });
  });

  suite("jax.numpy.maximum()", () => {
    test("computes element-wise maximum", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 2, 0]);
      using z = np.maximum(x, y);
      expect(z.js()).toEqual([4, 2, 3]);
    });

    test("works with jvp", () => {
      using x = np.array([1, 1, 3]);
      using y = np.array([4, 2, 0]);
      using t1 = np.ones([3]);
      using t2 = np.zeros([3]);
      const [z, dz] = jvp(
        (x: np.Array, y: np.Array) => np.maximum(x, y),
        [x, y],
        [t1, t2],
      );
      using _z = z;
      using _dz = dz;
      expect(z.js()).toEqual([4, 2, 3]);
      expect(dz.js()).toEqual([0, 0, 1]);
    });
  });

  suite("jax.numpy.absolute()", () => {
    test("computes absolute value", () => {
      using x = np.array([-1, 2, -3]);
      using y = np.absolute(x);
      expect(y.js()).toEqual([1, 2, 3]);

      using z = np.abs(x); // Alias for absolute
      expect(z.js()).toEqual([1, 2, 3]);
    });
  });

  suite("jax.numpy.sign()", () => {
    test("computes sign function", () => {
      using x = np.array([-10, 0, 5]);
      using y = np.sign(x);
      expect(y.js()).toEqual([-1, 0, 1]);
    });

    test("works with NaN", () => {
      using x = np.array(NaN);
      using s = np.sign(x);
      expect(s.js()).toBeNaN();
    });
  });

  suite("jax.numpy.reciprocal()", () => {
    test("computes element-wise reciprocal", () => {
      using x = np.array([1, 2, 3]);
      using y = np.reciprocal(x);
      expect(y.js()).toBeAllclose([1, 0.5, 1 / 3]);
    });

    test("works with jvp", () => {
      using x = np.array([1, 2, 3]);
      using t = np.ones([3]);
      const [y, dy] = jvp((x: np.Array) => np.reciprocal(x), [x], [t]);
      using _y = y;
      using _dy = dy;
      expect(y).toBeAllclose([1, 0.5, 1 / 3]);
      expect(dy).toBeAllclose([-1, -0.25, -1 / 9]);
    });

    test("can be used in grad", () => {
      using x = np.array([1, 2, 3]);
      using dx = grad((x: np.Array) => np.reciprocal(x).sum())(x);
      expect(dx).toBeAllclose([-1, -0.25, -1 / 9]);
    });

    test("called via Array.div() and jax.numpy.divide()", () => {
      using x = np.array([1, 2, 3]);
      using y = np.array([4, 5, 6]);
      using z = x.div(y);
      expect(z).toBeAllclose([0.25, 0.4, 0.5]);

      using w = np.divide(x, y);
      expect(w.js()).toBeAllclose([0.25, 0.4, 0.5]);
    });

    test("recip of 0 is infinity", () => {
      using x = np.reciprocal(0);
      expect(x.js()).toEqual(Infinity);

      using y = np.array(9.0).div(0);
      expect(y.js()).toEqual(Infinity);
    });
  });

  suite("jax.numpy.floorDivide()", () => {
    test("computes element-wise floor division", () => {
      using x = np.array([7, 7, -7, -7]);
      using y = np.array([3, -3, 3, -3]);
      using z = np.floorDivide(x, y);
      // floor(7/3)=2, floor(7/-3)=-3, floor(-7/3)=-3, floor(-7/-3)=2
      expect(z.js()).toEqual([2, -3, -3, 2]);
    });

    test("handles integer division that rounds toward negative infinity", () => {
      using x = np.array([5, -5, 10, -10]);
      using y = np.array([2, 2, 3, 3]);
      using z = np.floorDivide(x, y);
      // floor(5/2)=2, floor(-5/2)=-3, floor(10/3)=3, floor(-10/3)=-4
      expect(z.js()).toEqual([2, -3, 3, -4]);
    });

    test("works with scalars", () => {
      {
        using r = np.floorDivide(7, 3);
        expect(r.js()).toBeCloseTo(2, 5);
      }
      {
        using r = np.floorDivide(-7, 3);
        expect(r.js()).toBeCloseTo(-3, 5);
      }
    });

    test("works with int32 dtype", () => {
      using x = np.array([7, 7, -7, -7], { dtype: np.int32 });
      using y = np.array([3, -3, 3, -3], { dtype: np.int32 });
      using z = np.floorDivide(x, y);
      // Should round toward -infinity, not toward zero
      // floor(7/3)=2, floor(7/-3)=-3, floor(-7/3)=-3, floor(-7/-3)=2
      expect(z.js()).toEqual([2, -3, -3, 2]);
      expect(z.dtype).toBe(np.int32);
    });
  });

  suite("jax.numpy.fmod()", () => {
    test("computes element-wise fmod", () => {
      using x = np.array([5, 7, -9, -11]);
      using y = np.array([3, -4, 2, -3]);
      using z = np.fmod(x, y);
      expect(z.js()).toEqual([2, 3, -1, -2]);
    });

    test("gradient is correct", () => {
      using x = np.array([5, 7, -9, -11]);
      using y = np.array([3, -4, 2, -3]);
      const { x: dx, y: dy } = grad(({ x, y }: { x: np.Array; y: np.Array }) =>
        np.fmod(x, y).sum(),
      )({ x, y }) as unknown as { x: np.Array; y: np.Array };
      expect(dx.js()).toEqual([1, 1, 1, 1]);
      expect(dy.js()).toEqual([
        -Math.trunc(5 / 3),
        -Math.trunc(7 / -4),
        -Math.trunc(-9 / 2),
        -Math.trunc(-11 / -3),
      ]);
      dx.dispose();
      dy.dispose();
    });
  });

  suite("jax.numpy.remainder()", () => {
    test("computes element-wise remainder", () => {
      using x = np.array([5, 5, -5, -5]);
      using y = np.array([3, -3, 3, -3]);
      using z = np.remainder(x, y);
      // Should follow the sign of the divisor, like Python (but unlike JS).
      expect(z.js()).toEqual([2, -1, 1, -2]);
    });

    test("remainder gradient is correct", () => {
      using x = np.array([5, 5, -5, -5]);
      using y = np.array([3, -3, 3, -3]);
      const { x: dx, y: dy } = grad(({ x, y }: { x: np.Array; y: np.Array }) =>
        np.remainder(x, y).sum(),
      )({ x, y }) as unknown as { x: np.Array; y: np.Array };
      expect(dx.js()).toEqual([1, 1, 1, 1]);
      expect(dy.js()).toEqual([
        -Math.floor(5 / 3),
        -Math.floor(5 / -3),
        -Math.floor(-5 / 3),
        -Math.floor(-5 / -3),
      ]);
      dx.dispose();
      dy.dispose();
    });
  });

  suite("jax.numpy.divmod()", () => {
    test("returns floor division and remainder", () => {
      using x = np.array([7, 7, -7, -7]);
      using y = np.array([3, -3, 3, -3]);
      const [q, r] = np.divmod(x, y);
      using _q = q;
      using _r = r;
      // floor(7/3)=2, floor(7/-3)=-3, floor(-7/3)=-3, floor(-7/-3)=2
      expect(q.js()).toEqual([2, -3, -3, 2]);
      // remainder follows sign of divisor y
      expect(r.js()).toEqual([1, -2, 2, -1]);
    });

    test("satisfies invariant x == q*y + r", () => {
      using x = np.array([5, -5, 10, -10]);
      using y = np.array([3, 3, 4, 4]);
      const [q, r] = np.divmod(x, y);
      using _q = q;
      using _r = r;
      // Verify: x == q * y + r
      using _qy = np.multiply(q, y);
      using reconstructed = np.add(_qy, r);
      expect(reconstructed.js()).toEqual([5, -5, 10, -10]);
    });

    test("works with scalars", () => {
      const [q, r] = np.divmod(7, 3);
      using _q = q;
      using _r = r;
      expect(q.js()).toBeCloseTo(2, 5);
      expect(r.js()).toBeCloseTo(1, 5);
    });

    test("works with int32 dtype", () => {
      using x = np.array([7, -7], { dtype: np.int32 });
      using y = np.array([3, 3], { dtype: np.int32 });
      const [q, r] = np.divmod(x, y);
      using _q = q;
      using _r = r;
      expect(q.js()).toEqual([2, -3]);
      expect(r.js()).toEqual([1, 2]);
      expect(q.dtype).toBe(np.int32);
      expect(r.dtype).toBe(np.int32);
    });
  });

  suite("jax.numpy.exp()", () => {
    test("computes element-wise exponential", () => {
      using x = np.array([-Infinity, 0, 1, 2, 3]);
      using y = np.exp(x);
      expect(y.js()).toBeAllclose([0, 1, Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("exp(-Infinity) = 0", () => {
      using x = np.exp(-Infinity);
      expect(x.js()).toEqual(0);
    });

    test("works with small and large numbers", () => {
      using x = np.array([-1000, -100, -50, -10, 0, 10, 50, 100, 1000]);
      using y = np.exp(x);
      expect(y.js()).toBeAllclose([
        0,
        3.720075976020836e-44,
        1.9287498479639178e-22,
        4.5399929762484854e-5,
        1,
        22026.465794806718,
        5.184705528587072e21,
        2.6881171418161356e43,
        Infinity,
      ]);
    });

    test("works with jvp", () => {
      using x = np.array([1, 2, 3]);
      using t = np.ones([3]);
      const [y, dy] = jvp((x: np.Array) => np.exp(x), [x], [t]);
      using _y = y;
      using _dy = dy;
      expect(y.js()).toBeAllclose([Math.E, Math.E ** 2, Math.E ** 3]);
      expect(dy.js()).toBeAllclose([Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("can be used in grad", () => {
      using x = np.array([1, 2, 3]);
      using dx = grad((x: np.Array) => np.exp(x).sum())(x);
      expect(dx.js()).toBeAllclose([Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("exp2(10) = 1024", () => {
      using x = np.exp2(10);
      expect(x.js()).toBeCloseTo(1024);
    });

    test("exp2(0) = 1", () => {
      using x = np.exp2(0);
      expect(x.js()).toBeCloseTo(1);
    });
  });

  suite("jax.numpy.log()", () => {
    test("computes element-wise natural logarithm", () => {
      using x = np.array([1, Math.E, Math.E ** 2]);
      using y = np.log(x);
      expect(y.js()).toBeAllclose([0, 1, 2]);
    });

    test("log(0) is -Infinity", () => {
      using x = np.log(0);
      expect(x.js()).toEqual(-Infinity);
    });

    test("works with jvp", () => {
      using x = np.array([1, Math.E, Math.E ** 2]);
      using t = np.ones([3]);
      const [y, dy] = jvp((x: np.Array) => np.log(x), [x], [t]);
      using _y = y;
      using _dy = dy;
      expect(y.js()).toBeAllclose([0, 1, 2]);
      expect(dy.js()).toBeAllclose([1, 1 / Math.E, 1 / Math.E ** 2]);
    });

    test("can be used in grad", () => {
      using x = np.array([1, Math.E, Math.E ** 2]);
      using dx = grad((x: np.Array) => np.log(x).sum())(x);
      expect(dx.js()).toBeAllclose([1, 1 / Math.E, 1 / Math.E ** 2]);
    });

    test("log2 and log10", () => {
      using x = np.array([1, 2, 4, 8]);
      using y2 = np.log2(x);
      using y10 = np.log10(x);
      expect(y2.js()).toBeAllclose([0, 1, 2, 3]);
      expect(y10.js()).toBeAllclose([
        0,
        Math.log10(2),
        Math.log10(4),
        Math.log10(8),
      ]);
    });
  });

  suite("jax.numpy.sqrt()", () => {
    test("computes element-wise square root", () => {
      using x = np.array([1, 4, 9]);
      using y = np.sqrt(x);
      expect(y.js()).toBeAllclose([1, 2, 3]);
    });

    test("returns NaN for negative inputs", () => {
      using x = np.array([-1, -4, 9]);
      using y = np.sqrt(x);
      expect(y.js()).toEqual([NaN, NaN, 3.0]);
    });
  });

  suite("jax.numpy.cbrt()", () => {
    test("computes element-wise cube root", () => {
      using x = np.array([-8, -1, 0, 1, 8]);
      using y = np.cbrt(x);
      expect(y).toBeAllclose([-2, -1, 0, 1, 2]);
    });

    test("works with jvp", () => {
      using x = np.array([-8, -1, 0, 1, 8]);
      using t = np.ones([5]);
      const [y, dy] = jvp(np.cbrt, [x], [t]);
      using _y = y;
      using _dy = dy;
      expect(y).toBeAllclose([-2, -1, 0, 1, 2]);
      expect(dy).toBeAllclose([1 / 12, 1 / 3, NaN, 1 / 3, 1 / 12]);
    });
  });

  suite("jax.numpy.power()", () => {
    test("computes element-wise power", () => {
      using x = np.array([-1, 2, 3, 4]);
      using y = np.power(x, 3);
      expect(y).toBeAllclose([-1, 8, 27, 64]);
    });

    test("multiple different exponents", () => {
      using _exp = np.array([-2, 0, 0.5, 1, 2]);
      using y = np.power(3, _exp);
      expect(y).toBeAllclose([1 / 9, 1, Math.sqrt(3), 3, 9]);
    });

    test("works with negative numbers", () => {
      // const y = np.power(-3, np.array([-2, -1, 0, 1, 2, 3, 4, 5]));
      // expect(y).toBeAllclose([1 / 9, -1 / 3, 1, -3, 9, -27, 81, -243]);
      using _neg_exp = np.array([0.5, 1.5, 2.5]);
      using z = np.power(-3, _neg_exp);
      expect(z.js()).toEqual([NaN, NaN, NaN]);
    });

    test("power of zero", () => {
      using _exp = np.array([-2, -1, 0, 0.5, 1, 2]);
      using y = np.power(0, _exp);
      expect(y.js()).toEqual([Infinity, Infinity, NaN, 0, 0, 0]);
    });
  });

  suite("jax.numpy.min()", () => {
    test("computes minimum of 1D array", () => {
      using x = np.array([3, 1, 4, 2]);
      using y = np.min(x);
      expect(y.js()).toEqual(1);
    });

    test("computes minimum of 2D array along axis", () => {
      using x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      using y = np.min(x, 0);
      expect(y.js()).toEqual([2, 1, 0]);
    });

    test("computes minimum of 2D array without axis", () => {
      using x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      using y = np.min(x);
      expect(y.js()).toEqual(0);
    });

    test("can have grad of min", () => {
      using x = np.array([3, 1, 4, 1]);
      using dx = grad((x: np.Array) => np.min(x))(x);
      expect(dx.js()).toEqual([0, 0.5, 0, 0.5]); // Gradient is 1 at the minimum
    });
  });

  suite("jax.numpy.max()", () => {
    test("computes maximum of 1D array", () => {
      using x = np.array([3, 1, 4, 2]);
      using y = np.max(x);
      expect(y.js()).toEqual(4);
    });

    test("computes maximum of 2D array along axis", () => {
      using x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      using y = np.max(x, 0);
      expect(y.js()).toEqual([3, 5, 4]);
    });

    test("computes maximum of 2D array without axis", () => {
      using x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      using y = np.max(x);
      expect(y.js()).toEqual(5);
    });

    test("can have grad of max", () => {
      using x = np.array([10, 3, 4, 10]);
      using dx = grad((x: np.Array) => np.max(x))(x);
      expect(dx.js()).toEqual([0.5, 0, 0, 0.5]); // Gradient is 1 at the maximum
    });
  });

  suite("jax.numpy.pad()", () => {
    test("pads an array equally", () => {
      using a = np.array([1, 2, 3]);
      using b = np.pad(a, 1);
      expect(b.js()).toEqual([0, 1, 2, 3, 0]);

      using c = np.array([
        [1, 2],
        [3, 4],
      ]);
      using d = np.pad(c, 1);
      expect(d.js()).toEqual([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
      ]);
    });

    test("pads an array with uneven widths", () => {
      using a = np.array([[1]]);
      using b = np.pad(a, [
        [1, 2],
        [3, 0],
      ]);
      expect(b.js()).toEqual([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);
    });

    test("raises TypeError on axis mismatch", () => {
      using a = np.zeros([1, 2, 3]);
      expect(() => np.pad(a, [])).toThrow(Error);
      expect(() => np.pad(a, [[0, 1]])).not.toThrow(Error);
      expect(() =>
        np.pad(a, [
          [0, 1],
          [1, 2],
        ]),
      ).toThrow(Error);
    });

    test("pad handles backprop", () => {
      using a = np.array([1, 2, 3]);
      using da = grad((x: np.Array) => np.pad(x, 1).sum())(a);
      expect(da.js()).toEqual([1, 1, 1]);
    });

    test("works with jit and a prior operation", () => {
      // See comment about `needsCleanShapePrimitives` in JIT.
      using f = jit((x: np.Array) => {
        using y = x.add(2);
        return np.pad(y, 1);
      });
      using a = np.array([1, 2, 3]);
      using b = f(a);
      expect(b.js()).toEqual([0, 3, 4, 5, 0]);
    });

    test("pad with explicit indices", () => {
      using x = np.zeros([2, 3, 4, 5]);
      using y = np.pad(x, { 1: [0, 2], [-1]: [3, 0] });
      expect(y.shape).toEqual([2, 5, 4, 8]);
    });
  });

  suite("jax.numpy.split()", () => {
    test("splits into equal parts with integer", () => {
      using x = np.arange(6);
      const [a, b, c] = np.split(x, 3);
      using _a = a;
      using _b = b;
      using _c = c;
      expect(a.js()).toEqual([0, 1]);
      expect(b.js()).toEqual([2, 3]);
      expect(c.js()).toEqual([4, 5]);
    });

    test("splits 2D array along axis 0", () => {
      using _x = np.arange(12);
      using x = _x.reshape([4, 3]);
      const [a, b] = np.split(x, 2, 0);
      using _a = a;
      using _b = b;
      expect(a.js()).toEqual([
        [0, 1, 2],
        [3, 4, 5],
      ]);
      expect(b.js()).toEqual([
        [6, 7, 8],
        [9, 10, 11],
      ]);
    });

    test("splits 2D array along axis 1", () => {
      using _x = np.arange(12);
      using x = _x.reshape([3, 4]);
      const [a, b] = np.split(x, 2, 1);
      using _a = a;
      using _b = b;
      expect(a.js()).toEqual([
        [0, 1],
        [4, 5],
        [8, 9],
      ]);
      expect(b.js()).toEqual([
        [2, 3],
        [6, 7],
        [10, 11],
      ]);
    });

    test("splits at indices", () => {
      using x = np.arange(10);
      const [a, b, c] = np.split(x, [3, 7]);
      using _a = a;
      using _b = b;
      using _c = c;
      expect(a.js()).toEqual([0, 1, 2]);
      expect(b.js()).toEqual([3, 4, 5, 6]);
      expect(c.js()).toEqual([7, 8, 9]);
    });

    test("splits at indices with empty sections", () => {
      using x = np.arange(5);
      const [a, b, c, d] = np.split(x, [0, 0, 3]);
      using _a = a;
      using _b = b;
      using _c = c;
      using _d = d;
      expect(a.js()).toEqual([]);
      expect(b.js()).toEqual([]);
      expect(c.js()).toEqual([0, 1, 2]);
      expect(d.js()).toEqual([3, 4]);
    });

    test("throws on uneven split", () => {
      using x = np.arange(5);
      expect(() => np.split(x, 2)).toThrow(Error);
      expect(() => np.split(x, 3)).toThrow(Error);
    });

    test("works with negative axis", () => {
      using _x = np.arange(12);
      using x = _x.reshape([3, 4]);
      const [a, b] = np.split(x, 2, -1);
      using _a = a;
      using _b = b;
      expect(a.js()).toEqual([
        [0, 1],
        [4, 5],
        [8, 9],
      ]);
      expect(b.js()).toEqual([
        [2, 3],
        [6, 7],
        [10, 11],
      ]);
    });

    test("works with grad", () => {
      using _x = np.arange(6);
      using x = _x.astype(np.float32);
      const f = (x: np.Array) => {
        const [a, b] = np.split(x, 2);
        return a.sum().add(b.mul(2).sum());
      };
      using dx = grad(f)(x);
      expect(dx.js()).toEqual([1, 1, 1, 2, 2, 2]);
    });

    test("works inside jit", () => {
      using f = jit((x: np.Array) => {
        const [a, b] = np.split(x, 2);
        return a.add(b);
      });
      using x = np.arange(6);
      using y = f(x);
      expect(y.js()).toEqual([3, 5, 7]);
    });

    test("splits an array into 20 parts", () => {
      using x = np.arange(20);
      for (const [i, a] of np.split(x, 20).entries()) {
        expect(a.js()).toEqual([i]);
        a.dispose();
      }
    });
  });

  suite("jax.numpy.concatenate()", () => {
    // This suite also handles stack, hstack, vstack, dstack, etc.

    test("can concatenate 1D arrays", () => {
      using a = np.array([1, 2, 3]);
      using b = np.array([4, 5, 6]);
      using c = np.concatenate([a, b]);
      expect(c.js()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test("concatenation size mismatch", () => {
      using a = np.zeros([2, 3]);
      using b0 = np.zeros([3, 2]);
      expect(() => np.concatenate([a, b0])).toThrow(Error);
      expect(() => np.concatenate([a, b0], 1)).toThrow(Error);
      using b = b0.transpose();
      expect(() => np.concatenate([a, b]).dispose()).not.toThrow(Error);
    });

    test("stack() and variants work", () => {
      {
        using r = np.stack([2, 3]);
        expect(r.js()).toEqual([2, 3]);
      }
      {
        using r = np.stack([2, 3], -1);
        expect(r.js()).toEqual([2, 3]);
      }
      expect(() => np.stack([2, 3], 1)).toThrow(Error); // invalid axis
      expect(() => np.stack([2, 3], 2)).toThrow(Error); // invalid axis

      {
        using r = np.vstack([1, 2, 3]);
        expect(r.js()).toEqual([[1], [2], [3]]);
      }
      {
        using a = np.array([1, 2, 3]);
        using b = np.ones([3]);
        using r = np.vstack([a, b]);
        expect(r.js()).toEqual([
          [1, 2, 3],
          [1, 1, 1],
        ]);
      }

      {
        using r = np.hstack([1, 2, 3]);
        expect(r.js()).toEqual([1, 2, 3]);
      }
      {
        using a = np.array([1, 2, 3]);
        using b = np.ones([3]);
        using r = np.hstack([a, b]);
        expect(r.js()).toEqual([1, 2, 3, 1, 1, 1]);
      }

      {
        using r = np.dstack([1, 2, 3]);
        expect(r.js()).toEqual([[[1, 2, 3]]]);
      }
      {
        using a = np.array([1, 2, 3]);
        using b = np.ones([3]);
        using r = np.dstack([a, b]);
        expect(r.js()).toEqual([
          [
            [1, 1],
            [2, 1],
            [3, 1],
          ],
        ]);
      }
    });

    test("concatenate works in jit", () => {
      using f = jit(np.concatenate);
      using _a1 = np.array([1, 2]);
      using _a1f = np.flip(_a1);
      using _a2 = np.array([3, 4]);
      using _a3 = np.array([5]);
      using c = f([_a1f, _a2, _a3]);
      expect(c.js()).toEqual([2, 1, 3, 4, 5]);
    });
  });

  suite("jax.numpy.argmax()", () => {
    test("finds maximum of logits", () => {
      using _a = np.array([0.1, 0.2, 0.3, 0.2]);
      using x = np.argmax(_a);
      expect(x.js()).toEqual(2);
    });

    test("retrieves first index of maximum", () => {
      using _a = np.array([
        [0.1, -0.2, -0.3, 0.1],
        [0, 0.1, 0.3, 0.3],
      ]);
      using x = np.argmax(_a, 1);
      expect(x.js()).toEqual([0, 2]);
    });

    test("runs on flattened array by default", () => {
      using _a = np.array([
        [0.1, -0.2],
        [0.3, 0.1],
      ]);
      using x = np.argmax(_a);
      expect(x.js()).toEqual(2); // Index of maximum in flattened array
    });
  });

  suite("jax.numpy.tanh()", () => {
    const vals = [-1, -0.7, 0, 0.5, 1.7, 10, 50, 100, 1000];

    test("sinh values", () => {
      for (const x of vals) {
        using r = np.sinh(x);
        expect(r).toBeAllclose(Math.sinh(x));
      }
    });

    test("cosh values", () => {
      for (const x of vals) {
        using r = np.cosh(x);
        expect(r).toBeAllclose(Math.cosh(x));
      }
    });

    test("tanh values", () => {
      for (const x of vals) {
        using r = np.tanh(x);
        expect(r).toBeAllclose(Math.tanh(x));
      }
      using r = np.tanh(Infinity);
      expect(r.js()).toEqual(1);
    });
  });

  suite("jax.numpy.sinc()", () => {
    test("sinc(0) = 1", () => {
      using r = np.sinc(0);
      expect(r.js()).toBeCloseTo(1, 5);
    });

    test("sinc at integer values is 0", () => {
      // sinc(n) = sin(πn) / (πn) = 0 for non-zero integers
      using x = np.array([1, 2, 3, -1, -2, -3]);
      using _r = np.sinc(x);
      const result: number[] = _r.js();
      for (const val of result) {
        expect(val).toBeCloseTo(0, 5);
      }
    });

    test("sinc at 0.5", () => {
      // sinc(0.5) = sin(π/2) / (π/2) = 1 / (π/2) = 2/π
      using r = np.sinc(0.5);
      expect(r.js()).toBeCloseTo(2 / Math.PI, 5);
    });

    test("sinc is symmetric", () => {
      using x = np.array([0.1, 0.5, 1.5, 2.5]);
      using negX = np.array([-0.1, -0.5, -1.5, -2.5]);
      using r1 = np.sinc(x);
      using r2 = np.sinc(negX);
      expect(r1.js()).toBeAllclose(r2.js());
    });

    test("sinc on array", () => {
      using x = np.array([0, 0.5, 1]);
      const expected = [1, 2 / Math.PI, 0];
      using r = np.sinc(x);
      expect(r.js()).toBeAllclose(expected);
    });
  });

  suite("jax.numpy.atan()", () => {
    test("arctan values", () => {
      const vals = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000, Infinity];
      using _a = np.array(vals);
      using _r = np.atan(_a);
      const atanvals: number[] = _r.js();
      for (let i = 0; i < vals.length; i++) {
        expect(atanvals[i]).toBeCloseTo(Math.atan(vals[i]), 5);
      }
    });

    test("arcsin and arccos values", () => {
      const vals = [-1, -0.7, 0, 0.5, 1];
      using _a = np.array(vals);
      using _asin = np.asin(_a);
      using _acos = np.acos(_a);
      const asinvals: number[] = _asin.js();
      const acosvals: number[] = _acos.js();
      for (let i = 0; i < vals.length; i++) {
        expect(asinvals[i]).toBeCloseTo(Math.asin(vals[i]), 5);
        expect(acosvals[i]).toBeCloseTo(Math.acos(vals[i]), 5);
      }
    });

    test("grad of arctan", () => {
      using x = np.array([1, Math.sqrt(3), 0]);
      using dx = grad((x: np.Array) => np.atan(x).sum())(x);
      const expected = [0.5, 0.25, 1];
      expect(dx.js()).toBeAllclose(expected);
    });

    test("grad of arcsin", () => {
      using x = np.array([-0.5, 0, 0.5]);
      using dx = grad((x: np.Array) => np.asin(x).sum())(x);
      const expected = [2 / Math.sqrt(3), 1, 2 / Math.sqrt(3)];
      expect(dx.js()).toBeAllclose(expected);
    });
  });

  suite("jax.numpy.atan2()", () => {
    test("arctan2 values", () => {
      // Test all four quadrants and special cases with various values
      const y = [3, 5, -7, -2, 4, -6, 0, 0, 1.5, -2.5];
      const x = [4, -2, -3, 8, 0, 0, 5, -9, 1.5, -2.5];
      using _ya = np.array(y);
      using _xa = np.array(x);
      using _r = np.atan2(_ya, _xa);
      const result: number[] = _r.js();
      for (let i = 0; i < y.length; i++) {
        expect(result[i]).toBeCloseTo(Math.atan2(y[i], x[i]), 5);
      }
    });
  });

  suite("jax.numpy.repeat()", () => {
    test("repeats elements of 1D array", () => {
      using x = np.array([1, 2, 3]);
      using y = np.repeat(x, 2);
      expect(y.js()).toEqual([1, 1, 2, 2, 3, 3]);
    });

    test("repeats elements of 2D array along axis", () => {
      using x = np.array([
        [1, 2],
        [3, 4],
      ]);
      using y = np.repeat(x, 2, 0);
      expect(y.js()).toEqual([
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
      ]);

      using z = np.repeat(x, 3, 1);
      expect(z.js()).toEqual([
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
      ]);
    });

    test("flattens input when axis is null", () => {
      using x = np.array([
        [1, 2],
        [3, 4],
      ]);
      using y = np.repeat(x, 2);
      expect(y.js()).toEqual([1, 1, 2, 2, 3, 3, 4, 4]);
    });
  });

  suite("jax.numpy.tile()", () => {
    test("tiles 1D array", () => {
      using x = np.array([1, 2, 3]);
      using y = np.tile(x, 2);
      expect(y.js()).toEqual([1, 2, 3, 1, 2, 3]);
    });

    test("tiles 2D array along multiple axes", () => {
      using x = np.array([
        [1, 2],
        [3, 4],
      ]);
      using y = np.tile(x, [2, 1]);
      expect(y.js()).toEqual([
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
      ]);

      using z = np.tile(x, 3);
      expect(z.js()).toEqual([
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
      ]);
    });

    test("tiles with reps having more dimensions than array", () => {
      using x = np.array([1, 2]);
      using y = np.tile(x, [2, 2]);
      expect(y.js()).toEqual([
        [1, 2, 1, 2],
        [1, 2, 1, 2],
      ]);
    });
  });

  suite("jax.numpy.var_()", () => {
    test("computes variance", () => {
      using x = np.array([1, 2, 3, 4]);
      using y = np.var_(x);
      expect(y).toBeAllclose(1.25);
    });

    test("computes standard deviation", () => {
      using x = np.array([1, 2, 3, 4]);
      using y = np.std(x);
      expect(y).toBeAllclose(Math.sqrt(1.25));
    });
  });

  suite("jax.numpy.cov()", () => {
    test("computes covariance matrix", () => {
      using x = np.array([
        [0, 1, 2],
        [0, 1, 2],
      ]);
      using cov1 = np.cov(x);
      expect(cov1.js()).toBeAllclose([
        [1, 1],
        [1, 1],
      ]);
    });

    test("computes covariance matrix for anti-correlated data", () => {
      using x = np.array([
        [-1, 0, 1],
        [1, 0, -1],
      ]);
      using cov2 = np.cov(x);
      expect(cov2.js()).toBeAllclose([
        [1, -1],
        [-1, 1],
      ]);
    });

    test("computes covariance matrix from separate arrays", () => {
      using x = np.array([-1, 0, 1]);
      using y = np.array([1, 0, -1]);
      using cov3 = np.cov(x, y);
      expect(cov3.js()).toBeAllclose([
        [1, -1],
        [-1, 1],
      ]);
    });
  });

  suite("jax.numpy.isnan()", () => {
    test("identify special values", () => {
      // Test isnan and related functions (isinf, isfinite, etc.)
      using x = np.array([NaN, Infinity, -Infinity, 1]);
      {
        using r = np.isnan(x);
        expect(r.js()).toEqual([true, false, false, false]);
      }
      {
        using r = np.isinf(x);
        expect(r.js()).toEqual([false, true, true, false]);
      }
      {
        using r = np.isfinite(x);
        expect(r.js()).toEqual([false, false, false, true]);
      }
      {
        using r = np.isneginf(x);
        expect(r.js()).toEqual([false, false, true, false]);
      }
      {
        using r = np.isposinf(x);
        expect(r.js()).toEqual([false, true, false, false]);
      }
    });
  });

  suite("jax.numpy.nanToNum()", () => {
    test("replaces NaN with 0 by default", () => {
      using x = np.array([1, NaN, 3]);
      using y = np.nanToNum(x);
      expect(y.js()).toEqual([1, 0, 3]);
    });

    test("replaces NaN with custom value", () => {
      using x = np.array([NaN, 2, NaN]);
      using y = np.nanToNum(x, { nan: 99 });
      expect(y.js()).toEqual([99, 2, 99]);
    });

    test("replaces positive infinity when specified", () => {
      using x = np.array([1, Infinity, 3]);
      using y = np.nanToNum(x, { posinf: 999 });
      expect(y.js()).toEqual([1, 999, 3]);
    });

    test("replaces negative infinity when specified", () => {
      using x = np.array([1, -Infinity, 3]);
      using y = np.nanToNum(x, { neginf: -999 });
      expect(y.js()).toEqual([1, -999, 3]);
    });

    test("sets infinity to limit values when not specified", () => {
      using x = np.array([Infinity, -Infinity]);
      using y = np.nanToNum(x);
      expect(y).toBeAllclose([3.40282347e38, -3.40282347e38]);
    });

    test("handles all special values together", () => {
      using x = np.array([NaN, Infinity, -Infinity, 42]);
      using y = np.nanToNum(x, { nan: 0, posinf: 100, neginf: -100 });
      expect(y.js()).toEqual([0, 100, -100, 42]);
    });
  });

  suite("jax.numpy.convolve()", () => {
    test("computes 1D convolution", () => {
      using x = np.array([1, 2, 3, 2, 1]);
      using y = np.array([4, 1, 2]);

      using full = np.convolve(x, y);
      expect(full.js()).toEqual([4, 9, 16, 15, 12, 5, 2]);

      using same = np.convolve(x, y, "same");
      expect(same.js()).toEqual([9, 16, 15, 12, 5]);

      using valid = np.convolve(x, y, "valid");
      expect(valid.js()).toEqual([16, 15, 12]);
    });

    test("computes 1D correlation", () => {
      using x = np.array([1, 2, 3, 2, 1]);
      using y = np.array([4, 5, 6]);

      using valid = np.correlate(x, y);
      expect(valid.js()).toEqual([32, 35, 28]);

      using full = np.correlate(x, y, "full");
      expect(full.js()).toEqual([6, 17, 32, 35, 28, 13, 4]);

      using same = np.correlate(x, y, "same");
      expect(same.js()).toEqual([17, 32, 35, 28, 13]);

      using x1 = np.array([1, 2, 3, 2, 1]);
      using y1 = np.array([4, 5, 4]);
      using corr = np.correlate(x1, y1, "full");
      using conv = np.convolve(x1, y1, "full");
      expect(corr.js()).toEqual([4, 13, 26, 31, 26, 13, 4]);
      expect(conv.js()).toEqual([4, 13, 26, 31, 26, 13, 4]);
    });
  });

  suite("jax.numpy.all()", () => {
    test("returns true when all elements are true", () => {
      using x = np.array([true, true, true]);
      using r = np.all(x);
      expect(r.js()).toEqual(true);
    });

    test("returns false when any element is false", () => {
      using x = np.array([true, false, true]);
      using r = np.all(x);
      expect(r.js()).toEqual(false);
    });

    test("works along axis", () => {
      using x = np.array([
        [true, false],
        [true, true],
      ]);
      {
        using r = np.all(x, 0);
        expect(r.js()).toEqual([true, false]);
      }
      {
        using r = np.all(x, 1);
        expect(r.js()).toEqual([false, true]);
      }
    });

    test("works with numeric arrays (truthy values)", () => {
      using x = np.array([1, 2, 3]);
      {
        using r = np.all(x);
        expect(r.js()).toEqual(true);
      }

      using y = np.array([1, 0, 3]);
      {
        using r = np.all(y);
        expect(r.js()).toEqual(false);
      }
    });

    test("supports keepdims", () => {
      using x = np.array([
        [true, true],
        [true, false],
      ]);
      using result = np.all(x, 1, { keepdims: true });
      expect(result.shape).toEqual([2, 1]);
      expect(result.js()).toEqual([[true], [false]]);
    });
  });

  suite("jax.numpy.any()", () => {
    test("returns true when any element is true", () => {
      using x = np.array([false, true, false]);
      using r = np.any(x);
      expect(r.js()).toEqual(true);
    });

    test("returns false when all elements are false", () => {
      using x = np.array([false, false, false]);
      using r = np.any(x);
      expect(r.js()).toEqual(false);
    });

    test("works along axis", () => {
      using x = np.array([
        [false, false],
        [true, false],
      ]);
      {
        using r = np.any(x, 0);
        expect(r.js()).toEqual([true, false]);
      }
      {
        using r = np.any(x, 1);
        expect(r.js()).toEqual([false, true]);
      }
    });

    test("works with numeric arrays (truthy values)", () => {
      using x = np.array([0, 0, 0]);
      {
        using r = np.any(x);
        expect(r.js()).toEqual(false);
      }

      using y = np.array([0, 1, 0]);
      {
        using r = np.any(y);
        expect(r.js()).toEqual(true);
      }
    });

    test("supports keepdims", () => {
      using x = np.array([
        [false, false],
        [true, false],
      ]);
      using result = np.any(x, 1, { keepdims: true });
      expect(result.shape).toEqual([2, 1]);
      expect(result.js()).toEqual([[false], [true]]);
    });
  });

  suite("jax.numpy.expandDims()", () => {
    test("expands dims at position 0", () => {
      using x = np.array([1, 2, 3]);
      using y = np.expandDims(x, 0);
      expect(y.shape).toEqual([1, 3]);
      expect(y.js()).toEqual([[1, 2, 3]]);
    });

    test("expands dims at position 1", () => {
      using x = np.array([1, 2, 3]);
      using y = np.expandDims(x, 1);
      expect(y.shape).toEqual([3, 1]);
      expect(y.js()).toEqual([[1], [2], [3]]);
    });

    test("expands dims with negative axis", () => {
      using x = np.array([1, 2, 3]);
      using y = np.expandDims(x, -1);
      expect(y.shape).toEqual([3, 1]);
      expect(y.js()).toEqual([[1], [2], [3]]);
    });

    test("expands multiple dims at once", () => {
      using x = np.array([1, 2]);
      using y = np.expandDims(x, [0, 2]);
      expect(y.shape).toEqual([1, 2, 1]);
      expect(y.js()).toEqual([[[1], [2]]]);
    });

    test("expands dims on 2D array", () => {
      using x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      using y = np.expandDims(x, 0);
      expect(y.shape).toEqual([1, 2, 3]);

      using z = np.expandDims(x, 2);
      expect(z.shape).toEqual([2, 3, 1]);
    });

    test("throws on out of bounds axis", () => {
      using x = np.array([1, 2, 3]);
      expect(() => np.expandDims(x, 3)).toThrow(Error);
      expect(() => np.expandDims(x, -4)).toThrow(Error);
    });

    test("throws on repeated axis", () => {
      using x = np.array([1, 2, 3]);
      expect(() => np.expandDims(x, [0, 0])).toThrow(Error);
    });

    test("works with jvp", () => {
      using x = np.array([1, 2, 3]);
      using t = np.ones([3]);
      const [y, dy] = jvp((x: np.Array) => np.expandDims(x, 0), [x], [t]);
      using _y = y;
      using _dy = dy;
      expect(y.shape).toEqual([1, 3]);
      expect(dy.shape).toEqual([1, 3]);
    });

    test("works with grad", () => {
      using x = np.array([1, 2, 3]);
      using dx = grad((x: np.Array) => np.expandDims(x, 0).sum())(x);
      expect(dx.js()).toEqual([1, 1, 1]);
    });
  });

  if (device !== "webgl") {
    suite("jax.numpy.sort()", () => {
      test("sorts 1D array", () => {
        using x = np.array([3, 1, 4, 1, 5, 9, 2, 6]);
        using y = np.sort(x);
        expect(y.js()).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
      });

      test("sorts 2D array along axis", () => {
        using x = np.array([
          [3, 1, 2],
          [6, 4, 5],
        ]);
        using y0 = np.sort(x, 0);
        expect(y0.js()).toEqual([
          [3, 1, 2],
          [6, 4, 5],
        ]);
        using y1 = np.sort(x, 1);
        expect(y1.js()).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });

      test("sorts NaN to the end", () => {
        using x = np.array([3, NaN, 1, NaN, 2]);
        using y = np.sort(x);
        expect(y.js()).toEqual([1, 2, 3, NaN, NaN]);
      });

      test("works with jvp", () => {
        using x = np.array([3, 1, 2]);
        using t = np.array([10, 20, 30]);
        const [y, dy] = jvp(np.sort, [x], [t]);
        using _y = y;
        using _dy = dy;
        expect(y.js()).toEqual([1, 2, 3]);
        expect(dy.js()).toEqual([20, 30, 10]);
      });

      test("works with grad", () => {
        using x = np.array([3, 1, 4, 2]);
        const f = (x: np.Array) => np.sort(x).slice([0, 2]).sum();
        using dx = grad(f)(x);
        expect(dx.js()).toEqual([0, 1, 0, 1]);
      });

      test("works inside a jit function", () => {
        using x = np.array([5, 2, 8, 1]);
        using f = jit((x: np.Array) => np.sort(x));
        using y = f(x);
        expect(y.js()).toEqual([1, 2, 5, 8]);
      });

      test("works for int and bool dtypes", () => {
        for (const dtype of [np.int32, np.uint32]) {
          using x = np.array([3, 1, 4, 1, 5], { dtype });
          using y = np.sort(x);
          expect(y.js()).toEqual([1, 1, 3, 4, 5]);
          expect(y.dtype).toBe(dtype);
        }
        using x = np.array([true, false, true, false, true]);
        using y = np.sort(x);
        expect(y.js()).toEqual([false, false, true, true, true]);
        expect(y.dtype).toBe(np.bool);
      });

      test("handles zero-sized arrays", () => {
        using x = np.array([[], [], []], { dtype: np.float32 });
        using y = np.sort(x);
        expect(y.shape).toEqual([3, 0]);
        expect(y.dtype).toBe(np.float32);
      });

      test("can sort 8192 elements", async () => {
        // If the maximum workgroup size is 1024, then only 2048 elements can fit
        // into a single-workgroup sort. This test exercises multi-pass sorting in
        // global memory for GPUs.
        using x = np.linspace(0, 1, 8192);
        using _fx = np.flip(x);
        using y = np.sort(_fx);
        expect(y).toBeAllclose(x);
      });
    });

    suite("jax.numpy.argsort()", () => {
      test("argsorts 1D array", () => {
        using x = np.array([3, 1, 4, 2, 5]);
        using idx = np.argsort(x);
        expect(idx.js()).toEqual([1, 3, 0, 2, 4]);
        expect(idx.dtype).toBe("int32");
      });

      test("argsorts 2D array", () => {
        using x = np.array([
          [3, 1, 2],
          [6, 4, 5],
        ]);
        using idx = np.argsort(x, 1);
        expect(idx.js()).toEqual([
          [1, 2, 0],
          [1, 2, 0],
        ]);
      });

      test("is a stable sorting algorithm", () => {
        using x = np.array([
          3,
          1,
          1,
          NaN,
          Infinity,
          2,
          NaN,
          1,
          0,
          -0,
          Infinity,
        ]);
        using idx = np.argsort(x);
        expect(idx.js()).toEqual([8, 9, 1, 2, 7, 5, 0, 4, 10, 3, 6]);
      });

      test("produces zero gradient", () => {
        using x = np.array([3, 1, 2]);
        const f = (x: np.Array) => np.argsort(x).astype(np.float32).sum();
        using dx = grad(f)(x);
        expect(dx.js()).toEqual([0, 0, 0]);
      });

      test("can argsort 8191 elements", async () => {
        // Testing 8191 as it's not exactly a power-of-two size.
        using x = np.linspace(0, 1, 8191);
        using _fx = np.flip(x);
        using y = np.argsort(_fx);
        const ar = y.js() as number[];
        expect(ar).toEqual(Array.from({ length: 8191 }, (_, i) => 8190 - i));
      });
    });
  }

  suite("jax.numpy.take()", () => {
    test("takes elements from 1D array", () => {
      using x = np.array([10, 20, 30, 40, 50]);
      using indices = np.array([3, 0, 4, 1]);
      using y = np.take(x, indices);
      expect(y.js()).toEqual([40, 10, 50, 20]);
    });

    test("takes elements from 2D array along axis", () => {
      using x = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ]);
      using indices = np.array([2, 0]);
      using y0 = np.take(x, indices, 0);
      expect(y0.js()).toEqual([
        [70, 80, 90],
        [10, 20, 30],
      ]);
      using y1 = np.take(x, indices, 1);
      expect(y1.js()).toEqual([
        [30, 10],
        [60, 40],
        [90, 70],
      ]);
    });
  });
});
