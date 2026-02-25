/**
 * Tests for getCacheSizes() — direct observability of all JIT-related caches.
 *
 * Verifies that cache sizes grow correctly when operations are performed,
 * that clearCaches() resets them, and that different backends produce
 * separate jitCompile entries.
 */
import {
  clearCaches,
  getBackend,
  getCacheSizes,
  grad,
  init,
  jit,
  numpy as np,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

await init();

function addOne(x: np.Array) {
  return x.add(1);
}

function squareSum(x: np.Array) {
  using sq = x.mul(x) as np.Array;
  return sq.sum();
}

describe("getCacheSizes", () => {
  it("returns zero for all caches initially", () => {
    // checkLeaks.stop() from the previous test calls _disposeAllJitCaches(),
    // which clears all caches. So at the start of each test, counts are 0.
    const sizes = getCacheSizes();
    expect(sizes.jitCompile).toBe(0);
    expect(sizes.jvpJaxpr).toBe(0);
    expect(sizes.transposeJaxpr).toBe(0);
    expect(sizes.vmapJaxpr).toBe(0);
    // jitFunctions may be > 0 from module-level jit() registrations in the
    // library (numpy.ts, etc.), so we don't assert it's exactly 0.
  });

  it("jitCompile grows when a function is JIT-compiled", () => {
    const before = getCacheSizes();
    {
      using x = np.array([1, 2, 3]);
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    const after = getCacheSizes();
    expect(after.jitCompile).toBe(before.jitCompile + 1);
    jit(addOne).dispose();
  });

  it("jitCompile does not grow on cache hit", () => {
    {
      using x = np.array([1, 2, 3]);
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    const baseline = getCacheSizes();

    // Same function, same shape → cache hit.
    {
      using x = np.array([4, 5, 6]);
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    const after = getCacheSizes();
    expect(after.jitCompile).toBe(baseline.jitCompile);
    jit(addOne).dispose();
  });

  it("jitCompile grows for different shapes", () => {
    {
      using x = np.array([1, 2, 3]);
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    const baseline = getCacheSizes();

    // Different shape → new jaxpr → new compile cache entry.
    {
      using x = np.array([1, 2, 3, 4, 5]);
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    const after = getCacheSizes();
    expect(after.jitCompile).toBe(baseline.jitCompile + 1);
    jit(addOne).dispose();
  });

  it("jitFunctions grows for distinct function references", () => {
    const f1 = jit(addOne);
    const after1 = getCacheSizes();
    // addOne was already registered in earlier tests; dedup means no new entry.
    // Use a fresh arrow function to guarantee a new registration.
    const fresh = (x: np.Array) => x.add(2);
    const f2 = jit(fresh);
    const after2 = getCacheSizes();
    expect(after2.jitFunctions).toBe(after1.jitFunctions + 1);

    // Dedup: calling jit(fresh) again does NOT add another entry.
    const f3 = jit(fresh);
    const after3 = getCacheSizes();
    expect(after3.jitFunctions).toBe(after2.jitFunctions);
    expect(f2).toBe(f3);

    f1.dispose();
    f2.dispose();
  });

  it("grad(jit(f)) populates jvpJaxpr and transposeJaxpr caches", () => {
    // Transform caches are only populated when the transform encounters an
    // inner Primitive.Jit (or Scan/AssociativeScan). Plain grad(f) traces
    // inline without caching. grad(jit(f)) forces the JVP/transpose rules
    // for Primitive.Jit, which populate jvpJaxprCache and transposeJaxprCache.
    const before = getCacheSizes();
    {
      using x = np.array([1.0, 2.0, 3.0]);
      using dx = grad(jit(squareSum))(x) as np.Array;
      void dx;
    }
    const after = getCacheSizes();
    expect(after.jvpJaxpr).toBeGreaterThan(before.jvpJaxpr);
    expect(after.transposeJaxpr).toBeGreaterThan(before.transposeJaxpr);
    jit(squareSum).dispose();
  });

  it("vmap(jit(f)) populates vmapJaxpr cache", () => {
    // vmapJaxprCache is populated when vmap encounters Primitive.Jit.
    const rowOp = (row: np.Array) => row.mul(row).sum();
    const before = getCacheSizes();
    {
      using x = np.array([
        [1, 2],
        [3, 4],
      ]);
      using r = vmap(jit(rowOp))(x) as np.Array;
      void r;
    }
    const after = getCacheSizes();
    expect(after.vmapJaxpr).toBeGreaterThan(before.vmapJaxpr);
    jit(rowOp).dispose();
  });

  it("clearCaches resets all module-level caches to zero", () => {
    // Populate some caches.
    {
      using x = np.array([1, 2, 3]);
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    {
      // grad(jit(f)) populates jvp + transpose caches.
      using x = np.array([1.0, 2.0]);
      using dx = grad(jit(squareSum))(x) as np.Array;
      void dx;
    }

    const before = getCacheSizes();
    expect(before.jitCompile).toBeGreaterThan(0);

    clearCaches();

    const after = getCacheSizes();
    expect(after.jitCompile).toBe(0);
    expect(after.jvpJaxpr).toBe(0);
    expect(after.transposeJaxpr).toBe(0);
    expect(after.vmapJaxpr).toBe(0);
  });
});

describe("getCacheSizes across devices", () => {
  it("jitCompile creates separate entries per backend", async () => {
    const before = getCacheSizes();

    // Run on default device (typically webgpu or cpu in browser tests).
    {
      using x = np.array([1, 2, 3]);
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    const afterFirst = getCacheSizes();
    expect(afterFirst.jitCompile).toBe(before.jitCompile + 1);

    // Run on wasm device — different backend type in cache key.
    await init("wasm");
    {
      using x = np.array([1, 2, 3], { device: "wasm" });
      using r = jit(addOne)(x) as np.Array;
      void r;
    }
    const afterWasm = getCacheSizes();
    // If the default device is already wasm, no new entry (cache hit).
    // If the default is different (cpu/webgpu), we get a separate entry.
    const defaultBackend = getBackend();
    if (defaultBackend.type !== "wasm") {
      expect(afterWasm.jitCompile).toBe(afterFirst.jitCompile + 1);
    } else {
      // Same backend type — cache hit.
      expect(afterWasm.jitCompile).toBe(afterFirst.jitCompile);
    }

    jit(addOne).dispose();
  });

  it("transform caches are shared across devices", async () => {
    // jvpJaxpr is keyed by Jaxpr object identity (from the per-function
    // tracing cache). grad(jit(f)) populates it via jvpRules[Primitive.Jit].
    const before = getCacheSizes();
    {
      using x = np.array([1.0, 2.0, 3.0]);
      using dx = grad(jit(squareSum))(x) as np.Array;
      void dx;
    }
    const afterDefault = getCacheSizes();
    const jvpGrowth = afterDefault.jvpJaxpr - before.jvpJaxpr;
    expect(jvpGrowth).toBeGreaterThan(0);

    // Same grad(jit(f)) call on wasm — jvpJaxpr should NOT grow because
    // the inner Jaxpr identity is the same (jit dedup + same shape).
    await init("wasm");
    {
      using x = np.array([1.0, 2.0, 3.0], { device: "wasm" });
      using dx = grad(jit(squareSum))(x) as np.Array;
      void dx;
    }
    const afterWasm = getCacheSizes();
    expect(afterWasm.jvpJaxpr).toBe(afterDefault.jvpJaxpr);

    // jitCompile should grow (different backend type in key).
    const defaultBackend = getBackend();
    if (defaultBackend.type !== "wasm") {
      expect(afterWasm.jitCompile).toBeGreaterThan(afterDefault.jitCompile);
    }

    jit(squareSum).dispose();
  });
});

describe("getCacheSizes polymorphic (dynamic_axes)", () => {
  // jitCompileCache is a single frontend cache keyed by
  // `backend.type + "," + FpHash.hash(jaxpr)`. With dynamic_axes the jaxpr
  // contains SymDim nodes, so the hash is identical regardless of concrete
  // size. This dedup is backend-agnostic — no need to iterate devices.
  // (WASM has its own module-level caches but those aren't exposed here.)
  const polyAdd = (x: np.Array) => x.add(1);

  it("dynamic_axes avoids jitCompile duplication for different sizes", () => {
    using f = jit(polyAdd, { dynamic_axes: { 0: "T" } });

    // First call — compiles once.
    {
      using x = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      using r = f(x) as np.Array;
      void r;
    }
    const afterFirst = getCacheSizes();

    // Second call with different size on axis 0 — cache hit (same symbolic hash).
    {
      using x = np.array([
        [10, 20],
        [30, 40],
        [50, 60],
        [70, 80],
        [90, 100],
      ]);
      using r = f(x) as np.Array;
      expect(r.shape).toEqual([5, 2]);
    }
    const afterSecond = getCacheSizes();
    expect(afterSecond.jitCompile).toBe(afterFirst.jitCompile);
  });

  it("without dynamic_axes, different sizes create separate entries", () => {
    using f = jit(polyAdd);

    {
      using x = np.array([1, 2, 3]);
      using r = f(x) as np.Array;
      void r;
    }
    const afterFirst = getCacheSizes();

    {
      using x = np.array([1, 2, 3, 4, 5]);
      using r = f(x) as np.Array;
      void r;
    }
    const afterSecond = getCacheSizes();

    // Different shapes → different jaxpr hashes → new compile entry.
    expect(afterSecond.jitCompile).toBe(afterFirst.jitCompile + 1);
  });
});
