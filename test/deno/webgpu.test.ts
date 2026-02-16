/**
 * Deno-based WebGPU tests for jax-js
 *
 * Run with: deno test --unstable-webgpu --allow-read --allow-env test/deno/
 *
 * This uses Deno's native wgpu-rs WebGPU implementation which works
 * headless without X11, enabling hardware GPU testing on servers.
 *
 * jax-js includes a workaround for Deno's createComputePipelineAsync bug
 * by using the synchronous createComputePipeline when running in Deno.
 *
 * jax-js Ownership Model:
 * - Operations do NOT consume their inputs — arrays stay alive until .dispose()
 * - .data() reads the buffer without consuming — array stays alive
 * - .dispose() is required to free GPU buffers when done
 * - Inside jit() bodies, the compiler manages intermediate lifetimes automatically
 *
 * Memory Leak Detection:
 * - Tests use withLeakCheck() to verify no GPU buffers are leaked
 * - getSlotCount() queries the backend's allocated buffer count
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

// Import jax-js from local build
import { withLeakCheck } from "./harness.ts";
import {
  defaultDevice,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "../../dist/index.js";

// Import leak detection harness

// Check if WebGPU is available
const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

Deno.test({
  name: "WebGPU adapter available (hardware GPU)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    // Use init() to get adapter info via jax-js instead of creating a separate adapter.
    // This avoids potential GPU resource conflicts when running tests together.
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("WebGPU not available in jax-js");
      return;
    }
    // Access adapter info through the initialized backend
    const adapter = await navigator.gpu.requestAdapter();
    console.log("GPU Adapter:", adapter?.info);
    assertEquals(adapter !== null, true, "Should have a GPU adapter");
    assertEquals(
      adapter?.info?.isFallbackAdapter,
      false,
      "Should be hardware GPU, not fallback",
    );
  }),
});

Deno.test({
  name: "jax-js init detects webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    console.log("Available devices:", devices);
    assertEquals(
      devices.includes("webgpu"),
      true,
      "WebGPU should be available",
    );
  }),
});

// Test with WebGPU backend (uses sync createComputePipeline workaround)
Deno.test({
  name: "basic array operations on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    const a = np.array([1, 2, 3, 4]);
    const b = np.array([5, 6, 7, 8]);
    const c = np.add(a, b);

    const result = await c.data();
    assertEquals(Array.from(result), [6, 8, 10, 12]);

    a.dispose();
    b.dispose();
    c.dispose();
  }),
});

Deno.test({
  name: "matmul on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // Wrap in jit to manage matmul intermediates
    const matmulFn = jit((a: np.Array, b: np.Array) => np.matmul(a, b));
    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    const c = matmulFn(a, b);

    const result = await c.data();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assertEquals(Array.from(result), [19, 22, 43, 50]);

    a.dispose();
    b.dispose();
    c.dispose();
    matmulFn.dispose();
  }),
});

Deno.test({
  name: "JIT compilation on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // Extract constant outside jit body to avoid anonymous constant leak
    const two = np.array([2]);
    const f = (x: np.Array) => {
      const x2 = np.multiply(x, x);
      return np.add(x2, np.multiply(x, two));
    };
    const jitF = jit(f);

    const x = np.array([1, 2, 3, 4]);
    const result = jitF(x);

    const data = await result.data();
    // x^2 + 2x for x = [1,2,3,4] => [3, 8, 15, 24]
    assertEquals(Array.from(data), [3, 8, 15, 24]);

    x.dispose();
    result.dispose();
    // JIT functions are OwnedFunction - must dispose to release captured constants
    jitF.dispose();
    two.dispose();
  }),
});

Deno.test({
  name: "autodiff (grad) on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // f(x) = sum(x^2), so grad(f)(x) = 2x
    // Wrap in jit to manage autodiff intermediates
    const f = (x: np.Array) => np.sum(np.multiply(x, x));
    const gradFn = jit(grad(f));

    const x = np.array([1, 2, 3]);
    const dx = gradFn(x);

    const data = await dx.data();
    // d/dx(sum(x^2)) = 2x => [2, 4, 6]
    assertEquals(Array.from(data), [2, 4, 6]);

    x.dispose();
    dx.dispose();
    gradFn.dispose();
  }),
});

Deno.test({
  name: "reduction operations on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // Wrap in jit to manage reduction intermediates
    const reductions = jit((x: np.Array) => [
      np.sum(x),
      np.sum(x, 0),
      np.sum(x, 1),
    ]);

    const x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    const [sumAll, sumAxis0, sumAxis1] = reductions(x) as np.Array[];

    assertEquals(Array.from(await sumAll.data()), [21]);
    assertEquals(Array.from(await sumAxis0.data()), [5, 7, 9]);
    assertEquals(Array.from(await sumAxis1.data()), [6, 15]);

    x.dispose();
    sumAll.dispose();
    sumAxis0.dispose();
    sumAxis1.dispose();
    reductions.dispose();
  }),
});

// ---------------------------------------------------------------------------
// lax.scan tests on WebGPU (P3: compiled-loop path)
// ---------------------------------------------------------------------------

Deno.test({
  name: "lax.scan cumsum on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) return;
    defaultDevice("webgpu");

    // Extract init outside jit body to avoid anonymous constant leak
    const initCarry = np.array([0]);
    const f = jit((xs: np.Array) =>
      lax.scan(
        (carry: np.Array, x: np.Array) => {
          const out = np.add(carry, x);
          return [out, out];
        },
        initCarry,
        xs,
        { acceptPath: ["compiled-loop", "preencoded-routine"] },
      ),
    );
    const xs = np.array([[1], [2], [3], [4]]);
    const [carry, ys] = f(xs);
    const carryData = await carry.data();
    const ysData = await ys.data();
    xs.dispose();
    carry.dispose();
    ys.dispose();
    f.dispose();
    initCarry.dispose();
    assertEquals(Array.from(carryData), [10]);
    assertEquals(Array.from(ysData), [1, 3, 6, 10]);
  }),
});

Deno.test({
  name: "lax.scan cumsum jit on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) return;
    defaultDevice("webgpu");

    // Extract init outside jit body to avoid anonymous constant leak
    const initCarry = np.array([0]);
    const f = jit((xs: np.Array) => {
      const [carry, ys] = lax.scan(
        (c: np.Array, x: np.Array) => {
          const out = np.add(c, x);
          return [out, out];
        },
        initCarry,
        xs,
        { acceptPath: ["compiled-loop", "preencoded-routine"] },
      );
      return [carry, ys];
    });
    const xs = np.array([[1], [2], [3], [4], [5]]);
    const [carry, ys] = f(xs);
    const carryData = await carry.data();
    const ysData = await ys.data();
    xs.dispose();
    carry.dispose();
    ys.dispose();
    f.dispose();
    initCarry.dispose();
    assertEquals(Array.from(carryData), [15]);
    assertEquals(Array.from(ysData), [1, 3, 6, 10, 15]);
  }),
});
// ---------------------------------------------------------------------------
// lax.scan with matmul body — preencoded-routine path (P4)
// ---------------------------------------------------------------------------

Deno.test({
  name: "lax.scan matmul (preencoded-routine) on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) return;
    defaultDevice("webgpu");

    // Accumulate matrix products: carry = carry @ x for each x in xs
    const n = 2;
    const identity = np.eye(n); // [[1,0],[0,1]]

    // Separate reshape to avoid losing the intermediate flat array
    const xsFlat = np.array([2, 0, 0, 1, 1, 0, 0, 3]);
    const xs = xsFlat.reshape([2, n, n]);

    const f = jit((initC: np.Array, xsIn: np.Array) =>
      lax.scan(
        (carry: np.Array, x: np.Array) => {
          const result = np.matmul(carry, x);
          return [result, result];
        },
        initC,
        xsIn,
        { acceptPath: ["preencoded-routine", "compiled-loop", "fallback"] },
      ),
    );

    const [carry, ys] = f(identity, xs);
    const carryData = await carry.data();
    const ysData = await ys.data();
    identity.dispose();
    xs.dispose();
    xsFlat.dispose();
    carry.dispose();
    ys.dispose();
    f.dispose();

    // After iter 0: I @ [[2,0],[0,1]] = [[2,0],[0,1]]
    // After iter 1: [[2,0],[0,1]] @ [[1,0],[0,3]] = [[2,0],[0,3]]
    assertEquals(Array.from(carryData), [2, 0, 0, 3]);
    // ys[0] = [[2,0],[0,1]], ys[1] = [[2,0],[0,3]]
    assertEquals(Array.from(ysData), [2, 0, 0, 1, 2, 0, 0, 3]);
  }),
});

Deno.test({
  name: "lax.scan matmul 3 iters (preencoded-routine) on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) return;
    defaultDevice("webgpu");

    // 3x3 matrix accumulation over 3 iterations
    // This tests the ping-pong carry pattern more thoroughly
    const n = 3;
    const identity = np.eye(n);

    // xs[0]: scale first row by 2
    // xs[1]: add row0 to row1
    // xs[2]: identity (no-op)
    const xsData = new Float32Array([
      2,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1, // iter 0: diag(2,1,1)
      1,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1, // iter 1: shear
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1, // iter 2: identity
    ]);
    // Separate reshape to avoid losing the intermediate flat array
    const xsFlat = np.array(xsData);
    const xs = xsFlat.reshape([3, n, n]);

    const f = jit((initC: np.Array, xsIn: np.Array) =>
      lax.scan(
        (carry: np.Array, x: np.Array) => {
          const out = np.matmul(carry, x);
          return [out, out];
        },
        initC,
        xsIn,
        { acceptPath: ["preencoded-routine", "compiled-loop", "fallback"] },
      ),
    );

    const [carry, ys] = f(identity, xs);
    const carryData = await carry.data();
    const ysData = await ys.data();
    identity.dispose();
    xs.dispose();
    xsFlat.dispose();
    carry.dispose();
    ys.dispose();
    f.dispose();

    // iter 0: I @ diag(2,1,1) = [[2,0,0],[0,1,0],[0,0,1]]
    // iter 1: that @ shear    = [[2,0,0],[1,1,0],[0,0,1]]
    // iter 2: that @ I        = [[2,0,0],[1,1,0],[0,0,1]]
    assertEquals(Array.from(carryData), [2, 0, 0, 1, 1, 0, 0, 0, 1]);

    // ys stacked: 3 x 3 x 3 = 27 elements
    const expected = [
      2,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1, // ys[0]
      2,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1, // ys[1]
      2,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1, // ys[2]
    ];
    assertEquals(Array.from(ysData), expected);
  }),
});
