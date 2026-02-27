/**
 * Comprehensive tests for lax.scan implementation.
 *
 * Ported from feat/scan (v1) for scan-v2 P1.
 *
 * Structure:
 * - Basic scan tests (eager mode, CPU/WASM default)
 * - Multi-backend device tests (all available devices)
 * - xs=null carry-only tests
 * - Y=null (no output stacking) tests
 * - Scan over views (sliced/transposed/reshaped xs)
 * - Scan with routines in body (Cholesky, matmul, etc.)
 * - acceptPath option tests
 * - Ownership edge cases (duplicate-slot, passthrough, xs-passthrough)
 * - Autodiff tests (JVP, VJP, checkpointing, vmap)
 * - Native scan path tests (compiled-loop, preencoded-routine)
 */

import {
  defaultDevice,
  devices,
  grad,
  init,
  jit,
  jvp,
  lax,
  numpy as np,
  tree,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import {
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  suite,
  test,
} from "vitest";

const devicesAvailable = await init();

// ============================================================================
// Basic scan tests (eager mode)
// ============================================================================

describe("lax.scan", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("tree.map for stacking pytrees (JAX tree_stack pattern)", () => {
    it("stacks list of pytrees", () => {
      type Tree = { a: np.Array; b: np.Array };
      const trees: [Tree, Tree, Tree] = [
        { a: np.array([1.0]), b: np.array([2.0]) },
        { a: np.array([3.0]), b: np.array([4.0]) },
        { a: np.array([5.0]), b: np.array([6.0]) },
      ];

      const stacked = tree.map(
        (...v: np.Array[]) =>
          np.stack(
            v.map((a) => a),
            0,
          ),
        ...trees,
      );

      const a = (stacked as { a: np.Array; b: np.Array }).a;
      const b = (stacked as { a: np.Array; b: np.Array }).b;

      expect(a).toBeAllclose([[1], [3], [5]]);
      expect(b).toBeAllclose([[2], [4], [6]]);
      trees[0].a.dispose();
      trees[0].b.dispose();
      trees[1].a.dispose();
      trees[1].b.dispose();
      trees[2].a.dispose();
      trees[2].b.dispose();
      a.dispose();
      b.dispose();
    });
  });

  describe("scan basic", () => {
    it("computes cumulative sum", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([15.0]);
      expect(outputs).toBeAllclose([[1], [3], [6], [10], [15]]);
      finalCarry.dispose();
      outputs.dispose();
    });

    it("computes factorial-like recurrence", () => {
      const step = (carry: np.Array, t: np.Array): [np.Array, np.Array] => {
        const newCarry = np.multiply(carry, t);
        return [newCarry, newCarry];
      };

      using initVal = np.array([1.0]);
      using ts = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [final, outputs] = lax.scan(step, initVal, ts);

      expect(final).toBeAllclose([120.0]);
      expect(outputs).toBeAllclose([[1], [2], [6], [24], [120]]);
      final.dispose();
      outputs.dispose();
    });

    it("handles length-0 scans with xs array (returns init and empty ys)", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.array([42.0]);
      using xs = np.zeros([0, 1]);

      const [final, outputs] = lax.scan(step, initVal, xs);

      expect(final).toBeAllclose([42.0]);
      expect(outputs.shape).toEqual([0, 1]);
      final.dispose();
      outputs.dispose();
    });

    it("handles xs=null length-0 scans with explicit length (returns init and empty ys)", () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        return [carry, carry];
      };

      using initVal = np.array([7.0]);

      const [final, outputs] = lax.scan(step, initVal, null, { length: 0 });

      expect(final).toBeAllclose([7.0]);
      expect(outputs.shape).toEqual([0, 1]);
      final.dispose();
      outputs.dispose();
    });

    it("length-0 scan with pytree Y returns empty per-leaf arrays", () => {
      type Y = { a: np.Array; b: np.Array };

      const step = (carry: np.Array, x: np.Array): [np.Array, Y] => {
        const newCarry = np.add(carry, x);
        return [
          newCarry,
          { a: newCarry, b: np.multiply(newCarry, np.array([2.0])) },
        ];
      };

      using initVal = np.array([3.0]);
      using xs = np.zeros([0, 1]);

      const [final, ys] = lax.scan(step, initVal, xs);

      expect(final).toBeAllclose([3.0]);
      expect(ys).not.toBeNull();
      expect(ys.a.shape[0]).toBe(0);
      expect(ys.b.shape[0]).toBe(0);
      final.dispose();
      tree.dispose(ys);
    });

    it("length-0 scan with Y=null returns null", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, null] => {
        const newCarry = np.add(carry, x);
        return [newCarry, null];
      };

      using initVal = np.array([5.0]);
      using xs = np.zeros([0, 1]);

      const [final, ys] = lax.scan(step, initVal, xs);

      expect(final).toBeAllclose([5.0]);
      expect(ys).toBeNull();
      final.dispose();
      tree.dispose(ys);
    });
  });

  describe("scan with pytree carry", () => {
    it("tracks two values simultaneously", () => {
      type Carry = { sum: np.Array; count: np.Array };

      using _one = np.array([1.0]);
      const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
        const newSum = np.add(carry.sum, x);
        const newCount = np.add(carry.count, _one);
        return [{ sum: newSum, count: newCount }, np.divide(newSum, newCount)];
      };

      const initVal = { sum: np.array([0.0]), count: np.array([0.0]) };
      using xs = np.array([[2.0], [4.0], [6.0], [8.0]]);

      const [final, runningMeans] = lax.scan(step, initVal, xs);

      expect(final.sum).toBeAllclose([20.0]);
      expect(final.count).toBeAllclose([4.0]);
      // Running means: 2/1, 6/2, 12/3, 20/4 = 2, 3, 4, 5
      expect(runningMeans).toBeAllclose([[2], [3], [4], [5]]);
      initVal.sum.dispose();
      initVal.count.dispose();
      tree.dispose(final);
      runningMeans.dispose();
    });
  });

  describe("scan with pytree inputs", () => {
    it("handles pytree xs", () => {
      type X = { a: np.Array; b: np.Array };
      type Carry = np.Array;
      type Y = np.Array;

      const step = (carry: Carry, x: X): [Carry, Y] => {
        const sum = np.add(x.a, x.b);
        const newCarry = np.add(carry, sum);
        return [newCarry, newCarry];
      };

      using initVal = np.array([0.0]);
      const xs = {
        a: np.array([[1.0], [2.0], [3.0]]),
        b: np.array([[10.0], [20.0], [30.0]]),
      };

      const [final, _outputs] = lax.scan(step, initVal, xs);

      // (1+10) + (2+20) + (3+30) = 11 + 22 + 33 = 66
      expect(final).toBeAllclose([66.0]);
      xs.a.dispose();
      xs.b.dispose();
      final.dispose();
      _outputs.dispose();
    });
  });
});

// ============================================================================
// Multi-backend scan tests — runs on all available devices
// ============================================================================

suite.each(devices)("lax.scan device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("cumulative sum", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

    expect(finalCarry).toBeAllclose([15.0]);
    expect(outputs).toBeAllclose([[1], [3], [6], [10], [15]]);
    finalCarry.dispose();
    outputs.dispose();
  });

  test("jit + scan", () => {
    const step = jit((carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    });

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

    expect(finalCarry).toBeAllclose([6.0]);
    expect(outputs).toBeAllclose([[1], [3], [6]]);
    finalCarry.dispose();
    outputs.dispose();
  });

  test("pytree carry with multiple arrays", () => {
    type Carry = { sum: np.Array; product: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const newSum = np.add(carry.sum, x);
      const newProduct = np.multiply(carry.product, x);
      return [{ sum: newSum, product: newProduct }, newSum];
    };

    const initCarry = { sum: np.array([0.0]), product: np.array([1.0]) };
    using xs = np.array([[2.0], [3.0], [4.0]]);

    const [final, outputs] = lax.scan(step, initCarry, xs);

    expect(final.sum).toBeAllclose([9.0]); // 2+3+4
    expect(final.product).toBeAllclose([24.0]); // 2*3*4
    expect(outputs).toBeAllclose([[2], [5], [9]]);
    initCarry.sum.dispose();
    initCarry.product.dispose();
    tree.dispose(final);
    outputs.dispose();
  });

  test("larger iteration count", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const n = 100;
    using initCarry = np.array([0.0]);
    using xs = np.ones([n, 1]);

    const [finalCarry, _outputs] = lax.scan(step, initCarry, xs);

    expect(finalCarry).toBeAllclose([n]);
    finalCarry.dispose();
    _outputs.dispose();
  });

  test("elementwise ops in scan body", async () => {
    // More complex body: carry = tanh(carry + x * 0.1)
    using _scale = np.array([0.1]);
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const scaled = np.multiply(x, _scale);
      const added = np.add(carry, scaled);
      const newCarry = np.tanh(added);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]);

    const [finalCarry, _outputs] = lax.scan(step, initCarry, xs);
    using __outputs = _outputs;

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeGreaterThan(0);
    expect(finalData[0]).toBeLessThan(1);
    finalCarry.dispose();
  });

  describe("scan with routine body", () => {
    test("matmul in body (routine)", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.matmul(carry, x);
        return [newCarry, newCarry];
      };

      using initCarry = np.eye(2);
      using xs = np.array([
        [
          [2, 0],
          [0, 2],
        ], // scale by 2
        [
          [1, 1],
          [0, 1],
        ], // shear
        [
          [0, -1],
          [1, 0],
        ], // rotate 90 degrees
      ]);

      const [finalCarry, _outputs] = lax.scan(step, initCarry, xs);
      using __outputs = _outputs;

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(2.0);
      expect(finalData[1]).toBeCloseTo(-2.0);
      expect(finalData[2]).toBeCloseTo(2.0);
      expect(finalData[3]).toBeCloseTo(0.0);
      finalCarry.dispose();
    });

    test("matmul in body with reverse", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.matmul(carry, x);
        return [newCarry, newCarry];
      };

      using initCarry = np.eye(2);
      using xs = np.array([
        [
          [2, 0],
          [0, 2],
        ],
        [
          [1, 1],
          [0, 1],
        ],
        [
          [0, -1],
          [1, 0],
        ],
      ]);

      const [finalCarry, _outputs] = lax.scan(step, initCarry, xs, {
        reverse: true,
      });
      using __outputs = _outputs;

      // Processing order: xs[2], xs[1], xs[0]
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(0.0);
      expect(finalData[1]).toBeCloseTo(-2.0);
      expect(finalData[2]).toBeCloseTo(2.0);
      expect(finalData[3]).toBeCloseTo(2.0);
      finalCarry.dispose();
    });

    test("complex composite body (Kalman-like)", async () => {
      using F = np.array([
        [1, 0.1],
        [0, 1],
      ]);
      using H = np.array([[1, 0]]);
      using Q = np.array([
        [0.01, 0],
        [0, 0.01],
      ]);
      using _R = np.array([[0.1]]);

      type Carry = { state: np.Array; cov: np.Array };
      type X = { obs: np.Array };
      type Y = { pred: np.Array; innovation: np.Array };

      using _half = np.array([[0.5]]);
      using _decay = np.array([[0.9]]);
      const step = (carry: Carry, x: X): [Carry, Y] => {
        const { state, cov } = carry;
        const { obs } = x;

        const statePred = np.matmul(F, state);
        const FCov = np.matmul(F, cov);
        const covPred = np.add(np.matmul(FCov, F.transpose()), Q);
        const innovation = np.subtract(obs, np.matmul(H, statePred));
        const covH = np.matmul(covPred, H.transpose());
        const K = np.multiply(covH, _half);
        const stateNew = np.add(statePred, np.matmul(K, innovation));
        const covNew = np.multiply(covPred, _decay);

        return [
          { state: stateNew, cov: covNew },
          { pred: statePred, innovation },
        ];
      };

      using initState = np.array([[0], [0]]);
      using initCov = np.array([
        [1, 0],
        [0, 1],
      ]);
      using observations = np.array([[[1]], [[2]], [[2.5]], [[3]], [[3.5]]]);

      const [finalCarry, outputs] = lax.scan(
        step,
        { state: initState, cov: initCov },
        { obs: observations },
      );

      expect(finalCarry.state.shape).toEqual([2, 1]);
      expect(finalCarry.cov.shape).toEqual([2, 2]);
      expect(outputs.pred.shape).toEqual([5, 2, 1]);
      expect(outputs.innovation.shape).toEqual([5, 1, 1]);

      const finalStateData = await finalCarry.state.data();
      expect(Math.abs(finalStateData[0])).toBeGreaterThan(0.1);

      tree.dispose(finalCarry);
      tree.dispose(outputs);
    });

    test("with reduction (dot product accumulation)", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const sumX = np.sum(x);
        const newCarry = np.add(carry, sumX);
        return [newCarry, newCarry];
      };

      using initCarry = np.array([0.0]);
      using xs = np.array([
        [1.0, 2.0, 3.0, 4.0], // sum = 10
        [5.0, 5.0, 0.0, 0.0], // sum = 10
        [1.0, 1.0, 1.0, 1.0], // sum = 4
      ]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

      expect(finalCarry).toBeAllclose([24.0]);
      expect(outputs).toBeAllclose([[10], [20], [24]]);
      finalCarry.dispose();
      outputs.dispose();
    });
  });

  describe("reverse scan", () => {
    test("basic", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initCarry = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
        reverse: true,
      });

      expect(finalCarry).toBeAllclose([6.0]);
      // Reverse: processes xs[2]=3, xs[1]=2, xs[0]=1
      // outputs[2]=3, outputs[1]=5, outputs[0]=6
      expect(outputs).toBeAllclose([[6], [5], [3]]);
      finalCarry.dispose();
      outputs.dispose();
    });
  });

  describe("scan with xs=null (carry-only)", () => {
    test("generates sequence with no input arrays", () => {
      using _one = np.array([1.0]);
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const newCarry = np.add(carry, _one);
        return [newCarry, carry];
      };

      using initCarry = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 5,
      });

      expect(finalCarry).toBeAllclose([5.0]);
      expect(outputs).toBeAllclose([[0], [1], [2], [3], [4]]);
      finalCarry.dispose();
      outputs.dispose();
    });

    test("generates fibonacci sequence", async () => {
      using _idx0 = np.array(0, { dtype: np.int32 });
      using _idx1 = np.array(1, { dtype: np.int32 });
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const a = np.take(carry, _idx0);
        const b = np.take(carry, _idx1);
        const newCarry = np.stack([b, np.add(a, b)]);
        return [newCarry, a];
      };

      using initCarry = np.array([0.0, 1.0]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 8,
      });

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([0, 1, 1, 2, 3, 5, 8, 13]);

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(21.0); // fib(8)
      expect(finalData[1]).toBeCloseTo(34.0); // fib(9)
      finalCarry.dispose();
      outputs.dispose();
    });

    test("with jit", () => {
      using _one = np.array([1.0]);
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const newCarry = np.add(carry, _one);
        return [newCarry, carry];
      };

      const jitScan = jit((initVal: np.Array) =>
        lax.scan(step, initVal, null, { length: 5 }),
      );

      using initCarry = np.array([0.0]);
      const [finalCarry, outputs] = jitScan(initCarry);

      expect(finalCarry).toBeAllclose([5.0]);
      expect(outputs).toBeAllclose([[0], [1], [2], [3], [4]]);

      finalCarry.dispose();
      outputs.dispose();
      jitScan.dispose();
    });

    test("throws when length not provided", () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        return [carry, carry];
      };

      using initCarry = np.array([0.0]);

      expect(() => lax.scan(step, initCarry, null)).toThrow();
    });

    test("with reverse", () => {
      using _one = np.array([1.0]);
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const newCarry = np.add(carry, _one);
        return [newCarry, carry];
      };

      using initCarry = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 5,
        reverse: true,
      });

      expect(finalCarry).toBeAllclose([5.0]);
      // Reverse: outputs = [4, 3, 2, 1, 0] (carry values in reverse stacking order)
      expect(outputs).toBeAllclose([[4], [3], [2], [1], [0]]);
      finalCarry.dispose();
      outputs.dispose();
    });

    test("with pytree carry", async () => {
      type Carry = { a: np.Array; b: np.Array };

      using _one = np.array([1.0]);
      const step = (carry: Carry, _x: null): [Carry, np.Array] => {
        const newA = np.add(carry.a, carry.b);
        const newB = np.add(carry.b, _one);
        return [{ a: newA, b: newB }, carry.a];
      };

      const initCarry = { a: np.array([0.0]), b: np.array([1.0]) };

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 5,
      });

      const aData = await finalCarry.a.data();
      const bData = await finalCarry.b.data();
      expect(aData[0]).toBeCloseTo(15.0);
      expect(bData[0]).toBeCloseTo(6.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([0, 1, 3, 6, 10]);
      initCarry.a.dispose();
      initCarry.b.dispose();
      tree.dispose(finalCarry);
      outputs.dispose();
    });
  });

  describe("scan with Y=null (no output stacking)", () => {
    test("basic carry-only with Y=null", async () => {
      using _one = np.array([1.0]);
      const step = (carry: np.Array, _x: null): [np.Array, null] => {
        const newCarry = np.add(carry, _one);
        return [newCarry, null];
      };

      using initVal = np.array([0.0]);
      const [finalCarry, ys] = lax.scan(step, initVal, null, { length: 5 });

      const data = await finalCarry.data();
      expect(data[0]).toBeCloseTo(5.0);
      expect(ys).toBeNull();
      finalCarry.dispose();
      tree.dispose(ys);
    });

    test("Y=null with jit", () => {
      using _one = np.array([1.0]);
      const scanFn = (initVal: np.Array) => {
        const step = (carry: np.Array, _x: null): [np.Array, null] => {
          const newCarry = np.add(carry, _one);
          return [newCarry, null];
        };
        return lax.scan(step, initVal, null, { length: 5 });
      };

      const f = jit(scanFn as any);
      using arg = np.array([0.0]);
      const [finalCarry, ys] = f(arg) as [np.Array, null];

      expect(finalCarry).toBeAllclose([5.0]);
      expect(ys).toBeNull();
      finalCarry.dispose();
      f.dispose();
    });

    test("Y=null with xs array (not null)", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, null] => {
        const newCarry = np.add(carry, x);
        return [newCarry, null];
      };

      using initVal = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const [finalCarry, ys] = lax.scan(step, initVal, xs);

      const data = await finalCarry.data();
      expect(data[0]).toBeCloseTo(15.0);
      expect(ys).toBeNull();
      finalCarry.dispose();
      tree.dispose(ys);
    });

    test("Y=null with pytree carry", async () => {
      type Carry = { sum: np.Array; count: np.Array };

      using _ten = np.array([10.0]);
      using _one = np.array([1.0]);
      const step = (carry: Carry, _x: null): [Carry, null] => {
        const newSum = np.add(carry.sum, _ten);
        const newCount = np.add(carry.count, _one);
        return [{ sum: newSum, count: newCount }, null];
      };

      const initVal = { sum: np.array([0.0]), count: np.array([0.0]) };
      const [finalCarry, ys] = lax.scan(step, initVal, null, { length: 5 });

      const sumData = await finalCarry.sum.data();
      const countData = await finalCarry.count.data();
      expect(sumData[0]).toBeCloseTo(50.0);
      expect(countData[0]).toBeCloseTo(5.0);
      expect(ys).toBeNull();
      initVal.sum.dispose();
      initVal.count.dispose();
      tree.dispose(finalCarry);
      tree.dispose(ys);
    });
  });

  describe("scan over views (sliced/transposed xs)", () => {
    it("scan over sliced xs", () => {
      using full = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
      using xs = full.slice([2, 5]);

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      using initVal = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([9.0]); // 2 + 3 + 4
      expect(outputs).toBeAllclose([[2], [5], [9]]); // cumsum of [2,3,4]
      finalCarry.dispose();
      outputs.dispose();
    });

    it("scan over transposed xs", () => {
      using original = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      using xs = np.transpose(original); // shape [2, 3]

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      using initVal = np.zeros([3]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([3, 7, 11]);
      expect(outputs).toBeAllclose([
        [1, 3, 5],
        [3, 7, 11],
      ]);
      finalCarry.dispose();
      outputs.dispose();
    });

    it("jit(scan) over sliced xs", () => {
      using full = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]);
      using xs = full.slice([1, 4]); // [1.0], [2.0], [3.0]

      const f = jit(() => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        using initVal = np.array([0.0]);
        return lax.scan(step, initVal, xs);
      });

      const [finalCarry, outputs] = f();

      expect(finalCarry).toBeAllclose([6.0]); // 1 + 2 + 3
      expect(outputs).toBeAllclose([[1], [3], [6]]);

      finalCarry.dispose();
      outputs.dispose();
      f.dispose();
    });

    it("scan over reshaped xs", () => {
      using original = np.array([
        [
          [1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0],
        ],
        [
          [7.0, 8.0],
          [9.0, 10.0],
          [11.0, 12.0],
        ],
      ]);

      using xs = np.reshape(original, [2, 6]); // shape [2, 6]

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const xSum = np.sum(x);
        const newCarry = np.add(carry, xSum);
        return [newCarry, newCarry];
      };
      using initVal = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([78.0]);
      expect(outputs).toBeAllclose([[21.0], [78.0]]);
      finalCarry.dispose();
      outputs.dispose();
    });
  });

  describe("scan with routines", () => {
    it("scan with Cholesky in body", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const scaled = np.multiply(carry, x);
        const L = lax.linalg.cholesky(scaled);
        const reconstructed = np.matmul(L, L.transpose());
        return [reconstructed, L];
      };

      using initVal = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      using xs = np.array([[1.0], [1.0], [1.0]]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      const outputData = await outputs.data();
      expect(outputData.length).toBe(3 * 4); // 3 iterations, 2x2 matrices

      const finalData = await finalCarry.data();
      expect(finalData.length).toBe(4);
      expect(finalData[0]).toBeGreaterThan(0);
      expect(finalData[3]).toBeGreaterThan(0);
      finalCarry.dispose();
      outputs.dispose();
    });

    it("jit + scan with Cholesky", async () => {
      const f = jit(() => {
        const step = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
          const L = lax.linalg.cholesky(carry);
          const reconstructed = np.matmul(L, L.transpose());
          return [reconstructed, L];
        };

        using initVal = np.array([
          [4.0, 2.0],
          [2.0, 5.0],
        ]);
        using xs = np.array([[1.0], [1.0]]);

        return lax.scan(step, initVal, xs);
      });

      const [finalCarry, outputs] = f();

      const finalData = await finalCarry.data();
      expect(finalData.length).toBe(4);
      expect(finalData[0]).toBeGreaterThan(0);

      const outputData = await outputs.data();
      expect(outputData.length).toBe(2 * 4); // 2 iterations, 2x2 matrices

      finalCarry.dispose();
      outputs.dispose();
      f.dispose();
    });
  });

  describe("acceptPath option", () => {
    it("succeeds when acceptPath includes fallback", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0]]);

      // On fallback-only (P1), fallback should always be accepted
      const f = jit(() =>
        lax.scan(step, initVal, xs, {
          acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
        }),
      );

      const [carry, ys] = f();

      expect(carry).toBeAllclose([6.0]);
      expect(ys).toBeAllclose([[1], [3], [6]]);

      carry.dispose();
      ys.dispose();
      f.dispose();
    });

    it("allows array of paths", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.array([0.0]);
      using xs = np.array([[1.0], [2.0]]);

      const f = jit(() =>
        lax.scan(step, initVal, xs, {
          acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
        }),
      );

      const [carry, ys] = f();
      expect(carry).toBeAllclose([3.0]);

      carry.dispose();
      ys.dispose();
      f.dispose();
    });
  });

  describe("ownership edge cases", () => {
    it("duplicate-slot output: return [x, x]", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const result = np.add(carry, x);
        return [result, result];
      };

      using initCarry = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = lax.scan(step, initCarry, xs) as [
        np.Array,
        np.Array,
      ];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 3.0, 6.0]));
      finalCarry.dispose();
      ys.dispose();
    });

    it("carry passthrough: return [carry, y]", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const y = np.add(carry, x);
        return [carry, y];
      };

      using initCarry = np.array([10.0]);
      using xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = lax.scan(step, initCarry, xs) as [
        np.Array,
        np.Array,
      ];

      expect(await finalCarry.data()).toEqual(new Float32Array([10.0]));
      expect(await ys.data()).toEqual(new Float32Array([11.0, 12.0, 13.0]));
      finalCarry.dispose();
      ys.dispose();
    });

    it("xs passthrough: return [newCarry, x]", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, x];
      };

      using initCarry = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = lax.scan(step, initCarry, xs) as [
        np.Array,
        np.Array,
      ];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 2.0, 3.0]));
      finalCarry.dispose();
      ys.dispose();
    });

    it("jit(scan) with duplicate-slot output", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const result = np.add(carry, x);
        return [result, result];
      };

      using initCarry = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 3.0, 6.0]));
      finalCarry.dispose();
      ys.dispose();
      f.dispose();
    });

    it("jit(scan) with xs passthrough, reverse=true", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, x];
      };

      using initCarry = np.array([0.0]);
      using xs = np.array([[1.0], [2.0], [3.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs, { reverse: true }));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      // xs passthrough with reverse: ys maintains xs order
      expect(await ys.data()).toEqual(new Float32Array([1.0, 2.0, 3.0]));
      finalCarry.dispose();
      ys.dispose();
      f.dispose();
    });

    it("jit(scan) with carry passthrough and multiple iterations", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const y = np.multiply(carry, x);
        return [carry, y];
      };

      using initCarry = np.array([2.0]);
      using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([2.0]));
      expect(await ys.data()).toEqual(
        new Float32Array([2.0, 4.0, 6.0, 8.0, 10.0]),
      );
      finalCarry.dispose();
      ys.dispose();
      f.dispose();
    });
  });
});

// ============================================================================
// Scan preallocate tests (Y stacking correctness)
// ============================================================================

describe("scan preallocate", () => {
  test("basic stacked outputs", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    expect(ys).toBeAllclose([[1.0], [3.0], [6.0]]);
    finalCarry.dispose();
    ys.dispose();
  });

  test("duplicate-slot ys: carry and y are same slot", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const result = np.add(carry, x);
      return [result, result]; // carry = y (same underlying slot)
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    expect(ys).toBeAllclose([[1.0], [3.0], [6.0]]);
    finalCarry.dispose();
    ys.dispose();
  });

  test("passthrough ys: y is the carry value (no new computation for y)", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, carry]; // y = old carry (passthrough)
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    // ys = carry BEFORE each update: [0, 1, 3]
    expect(ys).toBeAllclose([[0.0], [1.0], [3.0]]);
    finalCarry.dispose();
    ys.dispose();
  });

  test("reverse scan preallocate", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs, { reverse: true });
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    // Reverse processes [3,2,1]: cumsum = [3,5,6]
    // Output stored in original indices: ys[2]=3, ys[1]=5, ys[0]=6
    expect(ys).toBeAllclose([[6.0], [5.0], [3.0]]);
    finalCarry.dispose();
    ys.dispose();
  });

  test("length-0 scan preallocate", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([42.0]);
    using xs = np.zeros([0, 1]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([42.0]);
    expect(ys.shape).toEqual([0, 1]);
    finalCarry.dispose();
    ys.dispose();
  });
});

// ============================================================================
// DLM / Kalman-like patterns (jit-scan-dlm tests)
// ============================================================================

describe("jit(scan) DLM patterns", () => {
  test("jit(scan) with passthrough Y + flip", async () => {
    type Carry = { state: np.Array; P: np.Array };

    using _decay = np.array([0.9]);
    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const { state, P } = carry;
      const newState = np.add(state, np.multiply(P, x));
      const newP = np.multiply(P, _decay);
      return [{ state: newState, P: newP }, state]; // y = old state (passthrough)
    };

    const initCarry = {
      state: np.array([0.0]),
      P: np.array([1.0]),
    };
    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const f = jit(() => lax.scan(step, initCarry, xs));
    const [finalCarry, outputs] = f();

    // Carry tracks accumulated state
    const stateData = await finalCarry.state.data();
    const PData = await finalCarry.P.data();
    expect(stateData[0]).toBeGreaterThan(0);
    expect(PData[0]).toBeGreaterThan(0);
    expect(PData[0]).toBeLessThan(1);

    // Outputs should be the previous state values (passthrough pattern)
    const outputsData = await outputs.data();
    expect(outputsData.length).toBe(5);
    expect(outputsData[0]).toBeCloseTo(0.0); // initial state

    initCarry.state.dispose();
    initCarry.P.dispose();
    finalCarry.state.dispose();
    finalCarry.P.dispose();
    outputs.dispose();
    f.dispose();
  });

  test("non-jit scan baseline (DLM-like)", async () => {
    type Carry = { state: np.Array; P: np.Array };

    using _decay = np.array([0.9]);
    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const { state, P } = carry;
      const newState = np.add(state, np.multiply(P, x));
      const newP = np.multiply(P, _decay);
      return [{ state: newState, P: newP }, state];
    };

    const initCarry = {
      state: np.array([0.0]),
      P: np.array([1.0]),
    };
    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

    const stateData = await finalCarry.state.data();
    expect(stateData[0]).toBeGreaterThan(0);
    expect(outputs.shape).toEqual([5, 1]);
    initCarry.state.dispose();
    initCarry.P.dispose();
    tree.dispose(finalCarry);
    outputs.dispose();
  });

  test("two-pass forward + reverse (smoother-like pattern)", async () => {
    // Forward pass: accumulate
    const forwardStep = (
      carry: np.Array,
      x: np.Array,
    ): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    using initFwd = np.array([0.0]);
    const [_fwdFinal, fwdOutputs] = lax.scan(forwardStep, initFwd, xs);

    // Reverse pass: use forward outputs as input, accumulate backwards
    const reverseStep = (
      carry: np.Array,
      x: np.Array,
    ): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initRev = np.array([0.0]);
    const [revFinal, _revOutputs] = lax.scan(reverseStep, initRev, fwdOutputs, {
      reverse: true,
    });

    const revData = await revFinal.data();
    // Sum of cumulative sums = 1 + 3 + 6 + 10 + 15 = 35
    expect(revData[0]).toBeCloseTo(35.0);
    revFinal.dispose();
    _fwdFinal.dispose();
    fwdOutputs.dispose();
    _revOutputs.dispose();
  });

  test("scan with pytree carry + output + slicing", async () => {
    type Carry = { sum: np.Array; count: np.Array };
    type Y = { running_mean: np.Array };

    using _one = np.array([1.0]);
    const step = (carry: Carry, x: np.Array): [Carry, Y] => {
      const newSum = np.add(carry.sum, x);
      const newCount = np.add(carry.count, _one);
      const mean = np.divide(newSum, newCount);
      return [{ sum: newSum, count: newCount }, { running_mean: mean }];
    };

    using xs = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]]);
    const initCarry = { sum: np.array([0.0]), count: np.array([0.0]) };

    const f = jit(() => lax.scan(step, initCarry, xs));
    const [finalCarry, outputs] = f();

    expect(finalCarry.sum).toBeAllclose([30.0]);
    expect(finalCarry.count).toBeAllclose([5.0]);

    // Running means: 2/1=2, 6/2=3, 12/3=4, 20/4=5, 30/5=6
    expect(outputs.running_mean).toBeAllclose([[2], [3], [4], [5], [6]]);

    initCarry.sum.dispose();
    initCarry.count.dispose();
    finalCarry.sum.dispose();
    finalCarry.count.dispose();
    outputs.running_mean.dispose();
    f.dispose();
  });
});

// ============================================================================
// Scan autodiff
// ============================================================================

describe("scan autodiff", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("JVP (forward-mode)", () => {
    it("computes jvp of cumulative sum", async () => {
      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, _outputs] = lax.scan(step, initVal, xs);
        using __outputs = _outputs;
        return finalCarry;
      };

      using xs = np.array([[1.0], [2.0], [3.0]]);
      using xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(cumsumScan, [xs], [xs_dot]);

      expect(await primal.data()).toEqual(new Float32Array([6]));
      expect(await tangent.data()).toEqual(new Float32Array([3]));
      primal.dispose();
      tangent.dispose();
    });

    it("computes jvp of cumulative product", async () => {
      using initVal = np.ones([1]);
      const cumprodScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, _outputs] = lax.scan(step, initVal, xs);
        using __outputs = _outputs;
        return finalCarry;
      };

      using xs = np.array([[2.0], [3.0], [4.0]]);
      using xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(cumprodScan, [xs], [xs_dot]);

      expect(await primal.data()).toEqual(new Float32Array([24]));
      expect(await tangent.data()).toEqual(new Float32Array([26]));
      primal.dispose();
      tangent.dispose();
    });

    it("jvp works with reverse scan", async () => {
      using initVal = np.zeros([1]);
      const revCumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
        using __ = _;
        return finalCarry;
      };

      using xs = np.array([[1.0], [2.0], [3.0]]);
      using xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(revCumsumScan, [xs], [xs_dot]);

      expect(await primal.data()).toEqual(new Float32Array([6]));
      expect(await tangent.data()).toEqual(new Float32Array([3]));
      primal.dispose();
      tangent.dispose();
    });
  });

  describe("VJP (reverse-mode)", () => {
    it("computes gradient through scan (sum of final carry)", async () => {
      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        using __ = _;
        return finalCarry.sum();
      };

      using xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(cumsumScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
      dxs.dispose();
    });

    it("computes gradient through scan (sum of all cumsum values)", async () => {
      using initVal = np.zeros([1]);
      const sumOfCumsum = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [_, ys] = lax.scan(step, initVal, xs);
        using __ = _;
        return ys.sum();
      };

      using xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(sumOfCumsum)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([3, 2, 1]));
      dxs.dispose();
    });

    it("computes gradient through reverse scan", async () => {
      using initVal = np.zeros([1]);
      const reverseCumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
        using __ = _;
        return finalCarry.sum();
      };

      using xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(reverseCumsumScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
      dxs.dispose();
    });

    it("grad works when function disposes initVal inside body", async () => {
      // Regression: user-disposed constants inside a grad body caused
      // UseAfterFreeError in getOrComputePrimal because disposePeIntermediates
      // would free the ClosedJaxpr's last reference to the constant.
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const f = (xs: np.Array) => {
        const initVal = np.array([0.0]);
        const [finalCarry, ys] = lax.scan(step, initVal, xs);
        initVal.dispose();
        ys.dispose();
        const result = finalCarry.sum();
        finalCarry.dispose();
        return result;
      };

      using xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(f)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
      dxs.dispose();
    });
  });

  describe("gradient checkpointing", () => {
    it("default (sqrt-N) produces same gradient as checkpoint: false", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.zeros([1]);
      const cumsumScanNoCheckpoint = (xs: np.Array) => {
        const [finalCarry, _] = lax.scan(step, initVal, xs, {
          checkpoint: false,
        });
        using __ = _;
        return finalCarry.sum();
      };

      const cumsumScanDefault = (xs: np.Array) => {
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        using __ = _;
        return finalCarry.sum();
      };

      using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const dxsRef = grad(cumsumScanNoCheckpoint)(xs);
      const dxsDefault = grad(cumsumScanDefault)(xs);

      expect(await dxsDefault.data()).toEqual(await dxsRef.data());
      dxsRef.dispose();
      dxsDefault.dispose();
    });

    it("default checkpointing produces correct gradient (sum of all outputs)", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.zeros([1]);
      const sumOfCumsum = (xs: np.Array) => {
        const [_, ys] = lax.scan(step, initVal, xs);
        using __ = _;
        return ys.sum();
      };

      using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const dxs = grad(sumOfCumsum)(xs);
      expect(dxs.shape).toEqual([5, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([5, 4, 3, 2, 1]));
      dxs.dispose();
    });

    it("checkpoint: number uses custom segment size", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const [finalCarry, _] = lax.scan(step, initVal, xs, { checkpoint: 2 });
        using __ = _;
        return finalCarry.sum();
      };

      using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
      const dxs = grad(cumsumScan)(xs);
      expect(dxs.shape).toEqual([6, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
      dxs.dispose();
    });

    it("checkpoint works with reverse scan", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.zeros([1]);
      const reverseScan = (xs: np.Array) => {
        const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
        using __ = _;
        return finalCarry.sum();
      };

      using xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(reverseScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
      dxs.dispose();
    });

    it("checkpoint works with larger iteration count", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        using __ = _;
        return finalCarry.sum();
      };

      const data = new Float32Array(100).fill(1.0);
      using xs = np.array(data).reshape([100, 1]);

      const dxs = grad(cumsumScan)(xs);
      expect(dxs.shape).toEqual([100, 1]);

      const gradData = await dxs.data();
      for (let i = 0; i < 100; i++) {
        expect(gradData[i]).toBeCloseTo(1.0);
      }
      dxs.dispose();
    });

    it("checkpoint works with nonlinear body (multiplicative)", async () => {
      using initVal = np.array([1.0]);
      const mulScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry, x);
          return [newCarry, carry];
        };
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        using __ = _;
        return finalCarry.sum();
      };

      using xs = np.array([[2.0], [3.0], [4.0]]);
      const dxsRef = grad((xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry, x);
          return [newCarry, carry];
        };
        const [finalCarry, _] = lax.scan(step, initVal, xs, {
          checkpoint: false,
        });
        using __ = _;
        return finalCarry.sum();
      })(xs);
      const dxsDefault = grad(mulScan)(xs);

      expect(await dxsDefault.data()).toEqual(await dxsRef.data());
      dxsRef.dispose();
      dxsDefault.dispose();
    });

    it("jit(grad(scan)) works with default checkpointing", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        using __ = _;
        return finalCarry.sum();
      };

      const jitGrad = jit(grad(cumsumScan));
      using xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = jitGrad(xs);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
      dxs.dispose();
      jitGrad.dispose();
    });
  });

  describe("vmap", () => {
    it("vmaps cumulative sum over batch dimension", async () => {
      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, outputs] = lax.scan(step, initVal, xs);
        using _finalCarry = finalCarry;
        outputs.dispose();
        return finalCarry;
      };

      using xs = np.array([
        [[1.0], [2.0], [3.0], [4.0]],
        [[2.0], [4.0], [6.0], [8.0]],
        [[1.0], [1.0], [1.0], [1.0]],
      ]);

      const batchedCumsum = vmap(cumsumScan);
      const result = batchedCumsum(xs);

      expect(result.shape).toEqual([3, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(10.0);
      expect(data[1]).toBeCloseTo(20.0);
      expect(data[2]).toBeCloseTo(4.0);
      result.dispose();
    });

    it("jit(vmap(scan)) works correctly", async () => {
      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        using _finalCarry = finalCarry;
        using __ = _;
        return finalCarry;
      };

      using xs = np.array([
        [[1.0], [2.0], [3.0], [4.0]],
        [[2.0], [4.0], [6.0], [8.0]],
        [[1.0], [1.0], [1.0], [1.0]],
      ]);

      const jittedBatchedCumsum = jit(vmap(cumsumScan));
      const result = jittedBatchedCumsum(xs);

      expect(result.shape).toEqual([3, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(10.0);
      expect(data[1]).toBeCloseTo(20.0);
      expect(data[2]).toBeCloseTo(4.0);

      result.dispose();
      jittedBatchedCumsum.dispose();
    });
  });

  describe("transform sandwiches", () => {
    it("jit(grad(scan)) computes gradient through JIT-compiled grad", async () => {
      using initVal = np.zeros([1]);
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        using __ = _;
        return finalCarry.sum();
      };

      const jitGrad = jit(grad(cumsumScan));

      using xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs1 = jitGrad(xs);
      expect(dxs1.shape).toEqual([3, 1]);
      expect(await dxs1.data()).toEqual(new Float32Array([1, 1, 1]));

      using xs2 = np.array([[4.0], [5.0], [6.0]]);
      const dxs2 = jitGrad(xs2);
      expect(await dxs2.data()).toEqual(new Float32Array([1, 1, 1]));

      dxs1.dispose();
      dxs2.dispose();
      jitGrad.dispose();
    });
  });
});

// ============================================================================
// Native scan path tests — P2+ (skipped until compiled-loop/preencoded available)
// ============================================================================

describe("native scan paths (P2+)", () => {
  beforeAll(async () => {
    const devs = await init();
    if (!devs.includes("wasm")) return; // skip if no WASM
    defaultDevice("wasm");
  });
  test("small array with acceptPath: compiled-loop", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const size = 64;
    using initCarry = np.zeros([size]);
    using xs = np.ones([10, size]);

    const [finalCarry, _outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    using _expected = np.full([size], 10.0);
    expect(finalCarry).toBeAllclose(_expected);
    finalCarry.dispose();
    _outputs.dispose();
  });

  test("native scan with constants", () => {
    using scale = np.array([2.0]);
    using offset = np.array([1.0]);

    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const scaled = np.multiply(x, scale);
      const shifted = np.add(scaled, offset);
      const newCarry = np.add(carry, shifted);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    expect(finalCarry).toBeAllclose([15.0]);
    expect(outputs).toBeAllclose([[3], [8], [15]]);
    finalCarry.dispose();
    outputs.dispose();
  });

  test("native scan with reduction in body", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const sumX = np.sum(x);
      const newCarry = np.add(carry, sumX);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 5.0, 0.0, 0.0],
      [1.0, 1.0, 1.0, 1.0],
    ]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    expect(finalCarry).toBeAllclose([24.0]);
    expect(outputs).toBeAllclose([[10], [20], [24]]);
    finalCarry.dispose();
    outputs.dispose();
  });

  test("large native scan (many iterations)", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const n = 500;
    using initCarry = np.zeros([64]);
    using xs = np.ones([n, 64]);

    const [finalCarry, _] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    using _expectedLarge = np.full([64], n);
    expect(finalCarry).toBeAllclose(_expectedLarge);
    finalCarry.dispose();
    _.dispose();
  });

  test("routine body: matmul in native scan", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.matmul(carry, x);
      return [newCarry, newCarry];
    };

    using initCarry = np.eye(4);
    using _e1 = np.eye(4);
    using _s1 = _e1.mul(2);
    using _e2 = np.eye(4);
    using _s2 = _e2.mul(3);
    using xs = np.stack([_s1, _s2]);

    const [finalCarry, _] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    using _e3 = np.eye(4);
    using _expectedMatmul = _e3.mul(6);
    expect(finalCarry).toBeAllclose(_expectedMatmul);
    finalCarry.dispose();
    _.dispose();
  });
});

// ============================================================================
// Preencoded multi-step scan tests (WebGPU Phase 2)
// ============================================================================

describe("preencoded multi-step scan (WebGPU)", () => {
  const hasWebGPU = devicesAvailable.includes("webgpu");

  beforeEach(({ skip }) => {
    if (!hasWebGPU) skip();
    defaultDevice("webgpu");
  });

  test("numCarry ≠ numY: two carries, one Y output", () => {
    // Tier 1 rejects because numCarry=2 ≠ numY=1.
    // Two independent kernels (different input sets: [a,x] vs [b,x]) → 2 execute steps.
    const step = (
      carry: [np.Array, np.Array],
      x: np.Array,
    ): [[np.Array, np.Array], np.Array] => {
      const [a, b] = carry;
      const newA = np.add(a, x);
      const newB = np.multiply(b, x);
      return [[newA, newB], newA];
    };

    using initA = np.array([0.0, 0.0]);
    using initB = np.array([1.0, 1.0]);
    using xs = np.array([
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ]);

    const [[finalA, finalB], ys] = lax.scan(step, [initA, initB], xs, {
      acceptPath: "preencoded-multi-step",
    });

    // a accumulates: [0,0]+[1,2]=[1,2], +[3,4]=[4,6], +[5,6]=[9,12]
    expect(finalA).toBeAllclose([9, 12]);
    // b multiplies: [1,1]*[1,2]=[1,2], *[3,4]=[3,8], *[5,6]=[15,48]
    expect(finalB).toBeAllclose([15, 48]);
    // ys is the running sum (= newA at each step)
    expect(ys).toBeAllclose([
      [1, 2],
      [4, 6],
      [9, 12],
    ]);
    finalA.dispose();
    finalB.dispose();
    ys.dispose();
  });

  test("Y ≠ carry: same numCarry=numY but different computations", () => {
    // Tier 1 rejects because yOutIds[0] ≠ carryOutIds[0].
    // Two independent kernels: carry+x vs x*x (different input sets) → 2 execute steps.
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      const y = np.multiply(x, x);
      return [newCarry, y];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0], [4.0]]);

    const [finalCarry, ys] = lax.scan(step, initCarry, xs, {
      acceptPath: "preencoded-multi-step",
    });

    // carry: 0+1=1, 1+2=3, 3+3=6, 6+4=10
    expect(finalCarry).toBeAllclose([10]);
    // y: 1²=1, 2²=4, 3²=9, 4²=16
    expect(ys).toBeAllclose([[1], [4], [9], [16]]);
    finalCarry.dispose();
    ys.dispose();
  });

  test("reverse scan with numCarry ≠ numY", () => {
    const step = (
      carry: [np.Array, np.Array],
      x: np.Array,
    ): [[np.Array, np.Array], np.Array] => {
      const [a, b] = carry;
      const newA = np.add(a, x);
      const newB = np.multiply(b, x);
      return [[newA, newB], newA];
    };

    using initA = np.array([0.0]);
    using initB = np.array([1.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [[finalA, finalB], ys] = lax.scan(step, [initA, initB], xs, {
      reverse: true,
      acceptPath: "preencoded-multi-step",
    });

    // reverse order: x=[3],[2],[1]
    // a: 0+3=3, 3+2=5, 5+1=6
    expect(finalA).toBeAllclose([6]);
    // b: 1*3=3, 3*2=6, 6*1=6
    expect(finalB).toBeAllclose([6]);
    // ys (in reverse iteration order, stacked as original axis order):
    // iter 0 (x=[3]): newA=[3] → ys[2]
    // iter 1 (x=[2]): newA=[5] → ys[1]
    // iter 2 (x=[1]): newA=[6] → ys[0]
    expect(ys).toBeAllclose([[6], [5], [3]]);
    finalA.dispose();
    finalB.dispose();
    ys.dispose();
  });

  test("Y ≠ carry with larger arrays", () => {
    const size = 64;
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      const y = np.multiply(x, x);
      return [newCarry, y];
    };

    using initCarry = np.zeros([size]);
    using xs = np.ones([100, size]);

    const [finalCarry, ys] = lax.scan(step, initCarry, xs, {
      acceptPath: "preencoded-multi-step",
    });

    using expected = np.full([size], 100);
    expect(finalCarry).toBeAllclose(expected);
    // ys[i] = 1*1 = 1 for all i (xs are all ones)
    using _sumYs = np.sum(ys);
    expect(_sumYs.js()).toBeCloseTo(100 * size, 0);
    finalCarry.dispose();
    ys.dispose();
  });

  test("jit-wrapped preencoded-multi-step scan", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      const y = np.multiply(x, x);
      return [newCarry, y];
    };

    using initCarry = np.array([0.0, 0.0]);
    using xs = np.array([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);

    using f = jit((init: np.Array, data: np.Array) =>
      lax.scan(step, init, data, {
        acceptPath: "preencoded-multi-step",
      }),
    );

    const [finalCarry, ys] = f(initCarry, xs);

    expect(finalCarry).toBeAllclose([4, 6]);
    expect(ys).toBeAllclose([
      [1, 4],
      [9, 16],
    ]);
    finalCarry.dispose();
    ys.dispose();
  });

  test("numCarry ≠ numY with pytree carry", () => {
    // Pytree carry with 2 leaves, 1 Y output → numCarry=2, numY=1
    const step = (
      carry: { sum: np.Array; count: np.Array },
      x: np.Array,
    ): [{ sum: np.Array; count: np.Array }, np.Array] => {
      const newSum = np.add(carry.sum, x);
      const newCount = np.add(carry.count, np.ones([]));
      return [{ sum: newSum, count: newCount }, newSum];
    };

    using initSum = np.array([0.0]);
    using initCount = np.array([0.0]);
    using xs = np.array([[10.0], [20.0], [30.0]]);

    const [finalCarry, ys] = lax.scan(
      step,
      { sum: initSum, count: initCount },
      xs,
      { acceptPath: "preencoded-multi-step" },
    );

    expect(finalCarry.sum).toBeAllclose([60]);
    expect(finalCarry.count).toBeAllclose([3]);
    expect(ys).toBeAllclose([[10], [30], [60]]);
    tree.dispose(finalCarry);
    ys.dispose();
  });

  test("grad through preencoded-multi-step scan (Y ≠ carry)", () => {
    // grad(scan) with Y ≠ carry: backward pass should correctly propagate
    const loss = (xs: np.Array): np.Array => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        const y = np.multiply(x, x);
        return [newCarry, y];
      };
      using initCarry = np.zeros([1]);
      const [finalCarry, ys] = lax.scan(step, initCarry, xs);
      // loss = sum(carry) + sum(ys) = sum(xs) + sum(xs²)
      using sumCarry = np.sum(finalCarry);
      using sumYs = np.sum(ys);
      finalCarry.dispose();
      ys.dispose();
      return np.add(sumCarry, sumYs);
    };

    using xs = np.array([[1.0], [2.0], [3.0]]);
    using dxs = grad(loss)(xs);

    // d/dx_i [sum(xs) + sum(xs²)] = 1 + 2*x_i
    expect(dxs).toBeAllclose([[3], [5], [7]]);
  });
});
