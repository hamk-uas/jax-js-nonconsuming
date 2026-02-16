/**
 * Deno-based WebGPU scan tests for jax-js
 *
 * Mirrors test/lax-scan.test.ts but uses Deno's test runner and `using`
 * declarations for automatic resource cleanup.
 *
 * NOTE: Uses assertAllcloseAsync (async .data()) because Deno has no
 * OffscreenCanvas, so dataSync()-based assertAllclose would fail.
 *
 * NOTE: Scan body functions must NOT use `using` on returned arrays —
 * `using` would dispose them at function scope exit before scan can use
 * them. Only use `using` at the test level for cleanup.
 *
 * Run with: deno test --no-check --unstable-webgpu --allow-read --allow-env test/deno/scan.test.ts
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { assertThrows } from "https://deno.land/std@0.224.0/assert/mod.ts";

import {
  assertAllcloseAsync,
  hasWebGPU,
  initWebGPU,
  withLeakCheck,
} from "./harness.ts";
import { grad, jit, jvp, lax, numpy as np, vmap } from "../../dist/index.js";

// ============================================================================
// Basic scan
// ============================================================================

Deno.test({
  name: "scan: cumulative sum",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initVal = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, initVal, xs);
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [6.0]);
    await assertAllcloseAsync(outputs, [[1], [3], [6]]);
  }),
});

Deno.test({
  name: "scan: factorial scan",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.multiply(carry, x);
      return [newCarry, newCarry];
    };

    using initVal = np.array([1.0]);
    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = lax.scan(step, initVal, xs);
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [120.0]);
    await assertAllcloseAsync(outputs, [[1], [2], [6], [24], [120]]);
  }),
});

Deno.test({
  name: "scan: length-0 returns init",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([42.0]);
    using xs = np.zeros([0, 1]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs);
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [42.0]);
    assertEquals(outputs.shape, [0, 1]);
  }),
});

// ============================================================================
// Pytree carry
// ============================================================================

Deno.test({
  name: "scan: pytree carry",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newA = np.add(carry.a, x);
      const newB = np.add(carry.b, carry.a);
      return [{ a: newA, b: newB }, newA];
    };

    using initA = np.array([0.0]);
    using initB = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, { a: initA, b: initB }, xs);
    using _fcA = finalCarry.a;
    using _fcB = finalCarry.b;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry.a, [6.0]);
    await assertAllcloseAsync(finalCarry.b, [4.0]);
  }),
});

// ============================================================================
// Reverse scan
// ============================================================================

Deno.test({
  name: "scan: reverse cumulative sum",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initVal = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, initVal, xs, {
      reverse: true,
    });
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [6.0]);
    // Reverse processes [3,2,1]: cumsum = [3,5,6], stored at indices [2,1,0]
    await assertAllcloseAsync(outputs, [[6], [5], [3]]);
  }),
});

// ============================================================================
// jit + scan
// ============================================================================

Deno.test({
  name: "scan: jit(scan) cumulative sum",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    using f = jit(() => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      return lax.scan(step, initVal, xs);
    });

    const [finalCarry, outputs] = f();
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [6.0]);
    await assertAllcloseAsync(outputs, [[1], [3], [6]]);
  }),
});

Deno.test({
  name: "scan: jit(scan) with pytree carry and multiple arrays",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);
    const step = (carry: any, x: any) => {
      const sumVal = np.add(carry.sum, x);
      const countVal = np.add(carry.count, _one);
      return [{ sum: sumVal, count: countVal }, sumVal];
    };

    using initSum = np.array([0.0]);
    using initCount = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    using f = jit(() => lax.scan(step, { sum: initSum, count: initCount }, xs));

    const [finalCarry, outputs] = f();
    using _fcSum = finalCarry.sum;
    using _fcCount = finalCarry.count;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry.sum, [6.0]);
    await assertAllcloseAsync(finalCarry.count, [3.0]);
    await assertAllcloseAsync(outputs, [[1], [3], [6]]);
  }),
});

// ============================================================================
// xs=null (carry-only scan)
// ============================================================================

Deno.test({
  name: "scan: xs=null sequence generator",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);
    const step = (carry: any, _x: null) => {
      const newCarry = np.add(carry, _one);
      return [newCarry, carry];
    };

    using initCarry = np.array([0.0]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
      length: 5,
    });
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [5.0]);
    await assertAllcloseAsync(outputs, [[0], [1], [2], [3], [4]]);
  }),
});

Deno.test({
  name: "scan: xs=null fibonacci",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, _x: null) => {
      const { a, b } = carry;
      const newB = np.add(a, b);
      return [{ a: b, b: newB }, a];
    };

    using initA = np.array([0.0]);
    using initB = np.array([1.0]);
    const [finalCarry, outputs] = lax.scan(step, { a: initA, b: initB }, null, {
      length: 8,
    });
    using _fcA = finalCarry.a;
    using _fcB = finalCarry.b;
    using _out = outputs;

    const data = await outputs.data();
    assertEquals(Array.from(data), [0, 1, 1, 2, 3, 5, 8, 13]);
  }),
});

Deno.test({
  name: "scan: xs=null with jit",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);
    using jitScan = jit((initCarry: any) => {
      const step = (carry: any, _x: null) => {
        const newCarry = np.add(carry, _one);
        return [newCarry, carry];
      };
      return lax.scan(step, initCarry, null, { length: 5 });
    });

    using initCarry = np.array([0.0]);
    const [finalCarry, outputs] = jitScan(initCarry);
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [5.0]);
    await assertAllcloseAsync(outputs, [[0], [1], [2], [3], [4]]);
  }),
});

Deno.test({
  name: "scan: xs=null throws when length not provided",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, _x: null): [any, any] => {
      return [carry, carry];
    };

    using initCarry = np.array([0.0]);
    assertThrows(() => lax.scan(step, initCarry, null));
  }),
});

Deno.test({
  name: "scan: xs=null with reverse",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);
    const step = (carry: any, _x: null) => {
      const newCarry = np.add(carry, _one);
      return [newCarry, carry];
    };

    using initCarry = np.array([0.0]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
      length: 5,
      reverse: true,
    });
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [5.0]);
    await assertAllcloseAsync(outputs, [[4], [3], [2], [1], [0]]);
  }),
});

Deno.test({
  name: "scan: xs=null with pytree carry",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);
    const step = (carry: any, _x: null) => {
      const newA = np.add(carry.a, carry.b);
      const newB = np.add(carry.b, _one);
      return [{ a: newA, b: newB }, carry.a];
    };

    using initA = np.array([0.0]);
    using initB = np.array([1.0]);

    const [finalCarry, outputs] = lax.scan(step, { a: initA, b: initB }, null, {
      length: 5,
    });
    using _fcA = finalCarry.a;
    using _fcB = finalCarry.b;
    using _out = outputs;

    const aData = await finalCarry.a.data();
    const bData = await finalCarry.b.data();
    assertEquals(Math.round(aData[0] * 100) / 100, 15);
    assertEquals(Math.round(bData[0] * 100) / 100, 6);

    const outputData = await outputs.data();
    assertEquals(Array.from(outputData), [0, 1, 3, 6, 10]);
  }),
});

// ============================================================================
// Y=null (no output stacking)
// ============================================================================

Deno.test({
  name: "scan: Y=null basic carry-only",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);
    const step = (carry: any, _x: null): [any, null] => {
      const newCarry = np.add(carry, _one);
      return [newCarry, null];
    };

    using initVal = np.array([0.0]);
    const [finalCarry, _ys] = lax.scan(step, initVal, null, { length: 5 });
    using _fc = finalCarry;

    const data = await finalCarry.data();
    assertEquals(Math.round(data[0] * 100) / 100, 5);
    assertEquals(_ys, null);
  }),
});

Deno.test({
  name: "scan: Y=null with jit",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);

    using f = jit((initVal: any) => {
      const step = (carry: any, _x: null): [any, null] => {
        const newCarry = np.add(carry, _one);
        return [newCarry, null];
      };
      return lax.scan(step, initVal, null, { length: 5 });
    });

    using initVal = np.array([0.0]);
    const [finalCarry, ys] = f(initVal);
    using _fc = finalCarry;

    await assertAllcloseAsync(finalCarry, [5.0]);
    assertEquals(ys, null);
  }),
});

Deno.test({
  name: "scan: Y=null with xs array",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any): [any, null] => {
      const newCarry = np.add(carry, x);
      return [newCarry, null];
    };

    using initVal = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);
    const [finalCarry, _ys] = lax.scan(step, initVal, xs);
    using _fc = finalCarry;

    await assertAllcloseAsync(finalCarry, [6.0]);
    assertEquals(_ys, null);
  }),
});

// ============================================================================
// Scan over views (sliced/transposed xs)
// ============================================================================

Deno.test({
  name: "scan: sliced xs",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using full = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
    using xs = full.slice([2, 5]);

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };
    using initVal = np.array([0.0]);

    const [finalCarry, outputs] = lax.scan(step, initVal, xs);
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [9.0]); // 2 + 3 + 4
    await assertAllcloseAsync(outputs, [[2], [5], [9]]);
  }),
});

Deno.test({
  name: "scan: transposed xs",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using original = np.array([
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ]);
    using xs = np.transpose(original); // shape [2, 3]

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };
    using initVal = np.zeros([3]);

    const [finalCarry, outputs] = lax.scan(step, initVal, xs);
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [3, 7, 11]);
    await assertAllcloseAsync(outputs, [
      [1, 3, 5],
      [3, 7, 11],
    ]);
  }),
});

Deno.test({
  name: "scan: reshaped xs",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

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

    using xs = np.reshape(original, [2, 6]);

    const step = (carry: any, x: any) => {
      const xSum = np.sum(x);
      const newCarry = np.add(carry, xSum);
      return [newCarry, newCarry];
    };
    using initVal = np.array([0.0]);

    const [finalCarry, outputs] = lax.scan(step, initVal, xs);
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [78.0]);
    await assertAllcloseAsync(outputs, [[21.0], [78.0]]);
  }),
});

// ============================================================================
// Scan with routines
// ============================================================================

Deno.test({
  name: "scan: Cholesky in body",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
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
    using _fc = finalCarry;
    using _out = outputs;

    const outputData = await outputs.data();
    assertEquals(outputData.length, 3 * 4); // 3 iterations, 2x2 matrices

    const finalData = await finalCarry.data();
    assertEquals(finalData.length, 4);
  }),
});

Deno.test({
  name: "scan: jit + Cholesky in body",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using f = jit(() => {
      const step = (carry: any, _x: any) => {
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
    using _fc = finalCarry;
    using _out = outputs;

    const finalData = await finalCarry.data();
    assertEquals(finalData.length, 4);

    const outputData = await outputs.data();
    assertEquals(outputData.length, 2 * 4); // 2 iterations, 2x2 matrices
  }),
});

// ============================================================================
// Ownership edge cases
// ============================================================================

Deno.test({
  name: "scan: duplicate-slot output [result, result]",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const result = np.add(carry, x);
      return [result, result];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, ys] = lax.scan(step, initCarry, xs);
    using _fc = finalCarry;
    using _ys = ys;

    const fcData = await finalCarry.data();
    assertEquals(Array.from(fcData), [6]);

    const ysData = await ys.data();
    assertEquals(Array.from(ysData), [1, 3, 6]);
  }),
});

Deno.test({
  name: "scan: carry passthrough [carry, y]",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const y = np.add(carry, x);
      return [carry, y];
    };

    using initCarry = np.array([10.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, ys] = lax.scan(step, initCarry, xs);
    using _fc = finalCarry;
    using _ys = ys;

    const fcData = await finalCarry.data();
    assertEquals(Array.from(fcData), [10]);

    const ysData = await ys.data();
    assertEquals(Array.from(ysData), [11, 12, 13]);
  }),
});

Deno.test({
  name: "scan: xs passthrough [newCarry, x]",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, x];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, ys] = lax.scan(step, initCarry, xs);
    using _fc = finalCarry;
    using _ys = ys;

    const fcData = await finalCarry.data();
    assertEquals(Array.from(fcData), [6]);

    const ysData = await ys.data();
    assertEquals(Array.from(ysData), [1, 2, 3]);
  }),
});

Deno.test({
  name: "scan: jit(scan) with duplicate-slot output",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const result = np.add(carry, x);
      return [result, result];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    using f = jit(() => lax.scan(step, initCarry, xs));
    const [finalCarry, ys] = f();
    using _fc = finalCarry;
    using _ys = ys;

    const fcData = await finalCarry.data();
    assertEquals(Array.from(fcData), [6]);

    const ysData = await ys.data();
    assertEquals(Array.from(ysData), [1, 3, 6]);
  }),
});

Deno.test({
  name: "scan: jit(scan) with carry passthrough and multiple iters",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const y = np.multiply(carry, x);
      return [carry, y];
    };

    using initCarry = np.array([2.0]);
    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    using f = jit(() => lax.scan(step, initCarry, xs));
    const [finalCarry, ys] = f();
    using _fc = finalCarry;
    using _ys = ys;

    const fcData = await finalCarry.data();
    assertEquals(Array.from(fcData), [2]);

    const ysData = await ys.data();
    assertEquals(Array.from(ysData), [2, 4, 6, 8, 10]);
  }),
});

// ============================================================================
// Preallocate tests (Y stacking correctness)
// ============================================================================

Deno.test({
  name: "scan preallocate: passthrough ys (y = old carry)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, carry]; // y = old carry
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, ys] = lax.scan(step, initCarry, xs);
    using _fc = finalCarry;
    using _ys = ys;

    await assertAllcloseAsync(finalCarry, [6.0]);
    await assertAllcloseAsync(ys, [[0.0], [1.0], [3.0]]);
  }),
});

Deno.test({
  name: "scan preallocate: reverse scan",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, ys] = lax.scan(step, initCarry, xs, { reverse: true });
    using _fc = finalCarry;
    using _ys = ys;

    await assertAllcloseAsync(finalCarry, [6.0]);
    await assertAllcloseAsync(ys, [[6.0], [5.0], [3.0]]);
  }),
});

// ============================================================================
// DLM / Kalman-like patterns
// ============================================================================

Deno.test({
  name: "scan: DLM passthrough Y + decay",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _decay = np.array([0.9]);
    const step = (carry: any, x: any) => {
      const { state, P } = carry;
      const px = np.multiply(P, x);
      const newState = np.add(state, px);
      const newP = np.multiply(P, _decay);
      return [{ state: newState, P: newP }, state]; // y = old state
    };

    using initState = np.array([0.0]);
    using initP = np.array([1.0]);
    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    using f = jit(() => lax.scan(step, { state: initState, P: initP }, xs));
    const [finalCarry, outputs] = f();
    using _fcState = finalCarry.state;
    using _fcP = finalCarry.P;
    using _out = outputs;

    const stateData = await finalCarry.state.data();
    const PData = await finalCarry.P.data();
    // State should be positive (accumulated)
    assertEquals(stateData[0] > 0, true);
    // P should decay below 1
    assertEquals(PData[0] > 0, true);
    assertEquals(PData[0] < 1, true);

    const outputsData = await outputs.data();
    assertEquals(outputsData.length, 5);
    // First output is initial state = 0
    assertEquals(Math.round(outputsData[0] * 100) / 100, 0);
  }),
});

Deno.test({
  name: "scan: two-pass forward + reverse (smoother-like)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const forwardStep = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    using initFwd = np.array([0.0]);
    const [_fwdFinal, fwdOutputs] = lax.scan(forwardStep, initFwd, xs);
    using __fwdFinal = _fwdFinal;
    using _fwdOutputs = fwdOutputs;

    const reverseStep = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initRev = np.array([0.0]);
    const [revFinal, _revOutputs] = lax.scan(reverseStep, initRev, fwdOutputs, {
      reverse: true,
    });
    using _fc = revFinal;
    using __revOutputs = _revOutputs;

    const revData = await revFinal.data();
    // Sum of cumulative sums = 1 + 3 + 6 + 10 + 15 = 35
    assertEquals(Math.round(revData[0] * 100) / 100, 35);
  }),
});

Deno.test({
  name: "scan: pytree carry + output + running mean",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using _one = np.array([1.0]);
    const step = (carry: any, x: any) => {
      const newSum = np.add(carry.sum, x);
      const newCount = np.add(carry.count, _one);
      const mean = np.divide(newSum, newCount);
      return [{ sum: newSum, count: newCount }, { running_mean: mean }];
    };

    using xs = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]]);
    using initSum = np.array([0.0]);
    using initCount = np.array([0.0]);

    using f = jit(() => lax.scan(step, { sum: initSum, count: initCount }, xs));
    const [finalCarry, outputs] = f();
    using _fcSum = finalCarry.sum;
    using _fcCount = finalCarry.count;
    using _outMean = outputs.running_mean;

    await assertAllcloseAsync(finalCarry.sum, [30.0]);
    await assertAllcloseAsync(finalCarry.count, [5.0]);
    await assertAllcloseAsync(outputs.running_mean, [[2], [3], [4], [5], [6]]);
  }),
});

// ============================================================================
// acceptPath option
// ============================================================================

Deno.test({
  name: "scan: acceptPath includes fallback",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initVal = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    using f = jit(() =>
      lax.scan(step, initVal, xs, {
        acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
      }),
    );

    const [carry, ys] = f();
    using _carry = carry;
    using _ys = ys;

    await assertAllcloseAsync(carry, [6.0]);
    await assertAllcloseAsync(ys, [[1], [3], [6]]);
  }),
});

// ============================================================================
// Scan autodiff: JVP
// ============================================================================

Deno.test({
  name: "scan JVP: cumulative sum",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
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
    using _p = primal;
    using _t = tangent;

    const pData = await primal.data();
    assertEquals(Array.from(pData), [6]);

    const tData = await tangent.data();
    assertEquals(Array.from(tData), [3]);
  }),
});

Deno.test({
  name: "scan JVP: cumulative product",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.ones([1]);
    const cumprodScan = (xs: any) => {
      const step = (carry: any, x: any) => {
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
    using _p = primal;
    using _t = tangent;

    const pData = await primal.data();
    assertEquals(Array.from(pData), [24]);

    const tData = await tangent.data();
    assertEquals(Array.from(tData), [26]);
  }),
});

Deno.test({
  name: "scan JVP: reverse scan",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const revCumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
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
    using _p = primal;
    using _t = tangent;

    const pData = await primal.data();
    assertEquals(Array.from(pData), [6]);

    const tData = await tangent.data();
    assertEquals(Array.from(tData), [3]);
  }),
});

// ============================================================================
// Scan autodiff: VJP / grad
// ============================================================================

Deno.test({
  name: "scan VJP: gradient of sum(final carry)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs);
      using __ = _;
      return finalCarry.sum();
    };

    using xs = np.array([[1.0], [2.0], [3.0]]);
    using dxs = grad(cumsumScan)(xs);
    assertEquals(dxs.shape, [3, 1]);

    const data = await dxs.data();
    assertEquals(Array.from(data), [1, 1, 1]);
  }),
});

Deno.test({
  name: "scan VJP: gradient of sum(all cumsum values)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const sumOfCumsum = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [_, ys] = lax.scan(step, initVal, xs);
      using __ = _;
      return ys.sum();
    };

    using xs = np.array([[1.0], [2.0], [3.0]]);
    using dxs = grad(sumOfCumsum)(xs);
    assertEquals(dxs.shape, [3, 1]);

    const data = await dxs.data();
    assertEquals(Array.from(data), [3, 2, 1]);
  }),
});

Deno.test({
  name: "scan VJP: gradient through reverse scan",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const reverseCumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
      using __ = _;
      return finalCarry.sum();
    };

    using xs = np.array([[1.0], [2.0], [3.0]]);
    using dxs = grad(reverseCumsumScan)(xs);
    assertEquals(dxs.shape, [3, 1]);

    const data = await dxs.data();
    assertEquals(Array.from(data), [1, 1, 1]);
  }),
});

// ============================================================================
// Gradient checkpointing
// ============================================================================

Deno.test({
  name: "scan checkpoint: default matches checkpoint:false",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initVal = np.zeros([1]);

    const cumsumScanNoCheckpoint = (xs: any) => {
      const [finalCarry, _] = lax.scan(step, initVal, xs, {
        checkpoint: false,
      });
      using __ = _;
      return finalCarry.sum();
    };

    const cumsumScanDefault = (xs: any) => {
      const [finalCarry, _] = lax.scan(step, initVal, xs);
      using __ = _;
      return finalCarry.sum();
    };

    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    using dxsRef = grad(cumsumScanNoCheckpoint)(xs);
    using dxsDefault = grad(cumsumScanDefault)(xs);

    const refData = await dxsRef.data();
    const defaultData = await dxsDefault.data();
    assertEquals(Array.from(defaultData), Array.from(refData));
  }),
});

Deno.test({
  name: "scan checkpoint: correct gradient (sum of all outputs)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initVal = np.zeros([1]);
    const sumOfCumsum = (xs: any) => {
      const [_, ys] = lax.scan(step, initVal, xs);
      using __ = _;
      return ys.sum();
    };

    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    using dxs = grad(sumOfCumsum)(xs);
    assertEquals(dxs.shape, [5, 1]);

    const data = await dxs.data();
    assertEquals(Array.from(data), [5, 4, 3, 2, 1]);
  }),
});

Deno.test({
  name: "scan checkpoint: custom segment size",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs, { checkpoint: 2 });
      using __ = _;
      return finalCarry.sum();
    };

    using xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
    using dxs = grad(cumsumScan)(xs);
    assertEquals(dxs.shape, [6, 1]);

    const data = await dxs.data();
    assertEquals(Array.from(data), [1, 1, 1, 1, 1, 1]);
  }),
});

Deno.test({
  name: "scan checkpoint: reverse scan",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const reverseScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
      using __ = _;
      return finalCarry.sum();
    };

    using xs = np.array([[1.0], [2.0], [3.0]]);
    using dxs = grad(reverseScan)(xs);
    assertEquals(dxs.shape, [3, 1]);

    const data = await dxs.data();
    assertEquals(Array.from(data), [1, 1, 1]);
  }),
});

Deno.test({
  name: "scan checkpoint: larger iteration count (N=100)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs);
      using __ = _;
      return finalCarry.sum();
    };

    const arr = new Float32Array(100).fill(1.0);
    using xs = np.array(arr).reshape([100, 1]);

    using dxs = grad(cumsumScan)(xs);
    assertEquals(dxs.shape, [100, 1]);

    const gradData = await dxs.data();
    for (let i = 0; i < 100; i++) {
      assertEquals(Math.abs(gradData[i] - 1.0) < 1e-4, true);
    }
  }),
});

Deno.test({
  name: "scan checkpoint: nonlinear body (multiplicative)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.array([1.0]);
    const mulScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.multiply(carry, x);
        return [newCarry, carry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs);
      using __ = _;
      return finalCarry.sum();
    };

    const mulScanNoCheckpoint = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.multiply(carry, x);
        return [newCarry, carry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs, {
        checkpoint: false,
      });
      using __ = _;
      return finalCarry.sum();
    };

    using xs = np.array([[2.0], [3.0], [4.0]]);
    using dxsRef = grad(mulScanNoCheckpoint)(xs);
    using dxsDefault = grad(mulScan)(xs);

    const refData = await dxsRef.data();
    const defaultData = await dxsDefault.data();
    assertEquals(Array.from(defaultData), Array.from(refData));
  }),
});

Deno.test({
  name: "scan checkpoint: jit(grad(scan))",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const [finalCarry, _] = lax.scan(step, initVal, xs);
      using __ = _;
      return finalCarry.sum();
    };

    using jitGrad = jit(grad(cumsumScan));
    using xs = np.array([[1.0], [2.0], [3.0]]);
    using dxs = jitGrad(xs);

    const data = await dxs.data();
    assertEquals(Array.from(data), [1, 1, 1]);
  }),
});

// ============================================================================
// Vmap
// ============================================================================

Deno.test({
  name: "scan vmap: cumulative sum over batch",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, outputs] = lax.scan(step, initVal, xs);
      using _fc = finalCarry;
      outputs.dispose();
      return finalCarry;
    };

    using xs = np.array([
      [[1.0], [2.0], [3.0], [4.0]],
      [[2.0], [4.0], [6.0], [8.0]],
      [[1.0], [1.0], [1.0], [1.0]],
    ]);

    using result = vmap(cumsumScan)(xs);
    assertEquals(result.shape, [3, 1]);

    const data = await result.data();
    assertEquals(Math.round(data[0] * 100) / 100, 10);
    assertEquals(Math.round(data[1] * 100) / 100, 20);
    assertEquals(Math.round(data[2] * 100) / 100, 4);
  }),
});

Deno.test({
  name: "scan vmap: jit(vmap(scan))",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs);
      using _fc = finalCarry;
      using __ = _;
      return finalCarry;
    };

    using xs = np.array([
      [[1.0], [2.0], [3.0], [4.0]],
      [[2.0], [4.0], [6.0], [8.0]],
      [[1.0], [1.0], [1.0], [1.0]],
    ]);

    using jittedBatchedCumsum = jit(vmap(cumsumScan));
    using result = jittedBatchedCumsum(xs);

    assertEquals(result.shape, [3, 1]);
    const data = await result.data();
    assertEquals(Math.round(data[0] * 100) / 100, 10);
    assertEquals(Math.round(data[1] * 100) / 100, 20);
    assertEquals(Math.round(data[2] * 100) / 100, 4);
  }),
});

// ============================================================================
// Transform sandwiches
// ============================================================================

Deno.test({
  name: "scan transform: jit(grad(scan))",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using initVal = np.zeros([1]);
    const cumsumScan = (xs: any) => {
      const step = (carry: any, x: any) => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const [finalCarry, _] = lax.scan(step, initVal, xs);
      using __ = _;
      return finalCarry.sum();
    };

    using jitGrad = jit(grad(cumsumScan));

    using xs = np.array([[1.0], [2.0], [3.0]]);
    using dxs1 = jitGrad(xs);
    assertEquals(dxs1.shape, [3, 1]);
    const data1 = await dxs1.data();
    assertEquals(Array.from(data1), [1, 1, 1]);

    // Second call should work (cached)
    using xs2 = np.array([[4.0], [5.0], [6.0]]);
    using dxs2 = jitGrad(xs2);
    const data2 = await dxs2.data();
    assertEquals(Array.from(data2), [1, 1, 1]);
  }),
});

// ============================================================================
// Native scan paths (WebGPU compiled-loop / preencoded-routine)
// ============================================================================

Deno.test({
  name: "scan native: small array with acceptPath",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const size = 64;
    using initCarry = np.zeros([size]);
    using xs = np.ones([10, size]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
    });
    using _fc = finalCarry;
    using _out = outputs;

    using expected = np.full([size], 10.0);
    await assertAllcloseAsync(finalCarry, expected);
  }),
});

Deno.test({
  name: "scan native: with constants",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    using scale = np.array([2.0]);
    using offset = np.array([1.0]);

    const step = (carry: any, x: any) => {
      const scaled = np.multiply(x, scale);
      const shifted = np.add(scaled, offset);
      const newCarry = np.add(carry, shifted);
      return [newCarry, newCarry];
    };

    using initCarry = np.array([0.0]);
    using xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
    });
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [15.0]);
    await assertAllcloseAsync(outputs, [[3], [8], [15]]);
  }),
});

Deno.test({
  name: "scan native: with reduction in body",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
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
      acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
    });
    using _fc = finalCarry;
    using _out = outputs;

    await assertAllcloseAsync(finalCarry, [24.0]);
    await assertAllcloseAsync(outputs, [[10], [20], [24]]);
  }),
});

Deno.test({
  name: "scan native: large iteration count (N=500)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const n = 500;
    using initCarry = np.zeros([64]);
    using xs = np.ones([n, 64]);

    const [finalCarry, _] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
    });
    using _fc = finalCarry;
    using __ = _;

    using expected = np.full([64], n);
    await assertAllcloseAsync(finalCarry, expected);
  }),
});

Deno.test({
  name: "scan native: matmul routine in body",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    await initWebGPU();

    const step = (carry: any, x: any) => {
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
      acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
    });
    using _fc = finalCarry;
    using __ = _;

    using _e3 = np.eye(4);
    using expected = _e3.mul(6);
    await assertAllcloseAsync(finalCarry, expected);
  }),
});
