/**
 * Tests for WASM parallel dispatch (M5: WASM Multithreading Foundation).
 *
 * Tests the kernel signature change (start, end, ...ptrs), worker pool
 * creation, and graceful fallback when shared memory is not available.
 *
 * Note: In browser test environments (Playwright), `crossOriginIsolated`
 * is typically `false`, so the worker pool is not created. These tests
 * verify the fallback works correctly and the new kernel signature
 * produces correct results in single-threaded mode.
 */
import {
  getBackend,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

describe("WASM parallel dispatch", () => {
  it("kernel (start, end) signature works for basic add", async () => {
    await init("wasm");

    const a = np.array([1, 2, 3, 4]);
    const b = np.array([5, 6, 7, 8]);
    const c = np.add(a, b);
    expect(await c.data()).toEqual(new Float32Array([6, 8, 10, 12]));
    a.dispose();
    b.dispose();
    c.dispose();
  });

  it("kernel (start, end) works for large arrays", async () => {
    await init("wasm");

    const n = 8192; // above PARALLEL_THRESHOLD
    const aData = new Float32Array(n);
    const bData = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      aData[i] = i;
      bData[i] = n - i;
    }
    const a = np.array(aData);
    const b = np.array(bData);
    const c = np.add(a, b);
    const result = await c.data();
    // Every element should be n
    for (let i = 0; i < n; i++) {
      expect(result[i]).toBe(n);
    }
    a.dispose();
    b.dispose();
    c.dispose();
  });

  it("kernel (start, end) works with jit fusion", async () => {
    await init("wasm");

    using a = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
    using b = np.array([2, 2, 2, 2, 2, 2, 2, 2]);
    using one = np.array([1]);

    using f = jit((x: np.Array, y: np.Array) => x.mul(y).add(one));
    using result = f(a, b);
    expect(await result.data()).toEqual(
      new Float32Array([3, 5, 7, 9, 11, 13, 15, 17]),
    );
  });

  it("kernel (start, end) works with reductions", async () => {
    await init("wasm");

    using a = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
    using result = a.sum();
    expect(await result.data()).toEqual(new Float32Array([36]));
  });

  it("kernel (start, end) works with multi-output kernels", async () => {
    await init("wasm");

    using a = np.array([1, 2, 3, 4]);
    using b = np.array([5, 6, 7, 8]);

    using f = jit((x: np.Array, y: np.Array) => [x.add(y), x.mul(y)]);
    const [sum, prod] = f(a, b) as [np.Array, np.Array];
    expect(await sum.data()).toEqual(new Float32Array([6, 8, 10, 12]));
    expect(await prod.data()).toEqual(new Float32Array([5, 12, 21, 32]));
    sum.dispose();
    prod.dispose();
  });

  it("kernel (start, end) works with scan", async () => {
    await init("wasm");

    using init_ = np.array([0]);
    using xs = np.array([1, 2, 3, 4, 5]);
    const [carry, ys] = lax.scan(
      (c: np.Array, x: np.Array) => {
        using next = np.add(c, x);
        return [next, next];
      },
      init_,
      xs,
    );
    expect(await carry.data()).toEqual(new Float32Array([15]));
    expect(await ys.data()).toEqual(new Float32Array([1, 3, 6, 10, 15]));
    carry.dispose();
    ys.dispose();
  });

  it("kernel (start, end) works with grad", async () => {
    await init("wasm");

    const f = (x: np.Array) => {
      using sq = x.mul(x) as np.Array;
      return sq.sum();
    };
    using x = np.array([1, 2, 3]);
    using dx = grad(f)(x) as np.Array;
    // d/dx sum(x^2) = 2x
    expect(await dx.data()).toEqual(new Float32Array([2, 4, 6]));
  });

  it("graceful fallback without crossOriginIsolated", async () => {
    await init("wasm");

    const backend = getBackend("wasm") as any;
    // In browser test environment, crossOriginIsolated is typically false
    // so workerPool should be null. In crossOriginIsolated environments,
    // workerPool exists but small arrays should still use single-threaded path.
    const _pool = backend.workerPool;

    // Regardless of pool status, small arrays should work correctly
    using a = np.array([10, 20, 30]);
    using b = np.array([1, 2, 3]);
    using c = np.add(a, b);
    expect(await c.data()).toEqual(new Float32Array([11, 22, 33]));
  });

  it("shared memory capability is reported correctly", async () => {
    await init("wasm");

    const backend = getBackend("wasm") as any;
    // The backend should report its shared memory support
    expect(typeof backend.capabilities.sharedMemory).toBe("boolean");
  });
});
