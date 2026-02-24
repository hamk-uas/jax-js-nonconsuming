import {
  getBackend,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

describe("WASM allocator memory management", () => {
  // These tests verify the allocator's ability to reclaim memory across
  // different input shapes — preventing OOM when JIT-compiled functions
  // are called with many different sizes (cross-shape accumulation).
  // See: dlm-js issue jax-js-wasm-allocator-cross-shape-oom.md

  test("repeated jit calls with different shapes do not grow memory monotonically", async () => {
    await init("wasm");
    const backend = getBackend("wasm") as any;

    // Run a jit function with many different input sizes.
    // Without the allocator fix, each new size class would bump-allocate
    // without reusing freed memory from previous sizes, leading to OOM.
    const f = jit((x: np.Array) => x.mul(x).sum());
    const sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800];
    const allocations: number[] = [];

    for (const n of sizes) {
      using x = np.array(new Float32Array(n));
      using result = f(x) as np.Array;
      result.js();
      allocations.push(backend.allocatorStats().totalAllocated);
    }

    f.dispose();

    // Run another pass — memory should NOT grow beyond previous peak.
    const g = jit((x: np.Array) => x.add(x).sum());
    for (const n of sizes) {
      using x = np.array(new Float32Array(n));
      using result = g(x) as np.Array;
      result.js();
    }

    const statsAfterSecondPass = backend.allocatorStats();
    g.dispose();

    // The second pass's peak should not significantly exceed the first.
    const firstPeak = Math.max(...allocations);
    expect(statsAfterSecondPass.totalAllocated).toBeLessThan(firstPeak * 3);
  });

  test("allocator reset-on-empty reclaims all memory", async () => {
    await init("wasm");
    const backend = getBackend("wasm") as any;
    const statsBefore = backend.allocatorStats().totalAllocated;

    // Allocate some buffers of different sizes.
    // Use non-zero data to avoid the all-equal short-circuit in arrayFromData.
    const a = np.array(Float32Array.from({ length: 1000 }, (_, i) => i));
    const b = np.array(Float32Array.from({ length: 2000 }, (_, i) => i));
    const c = np.array(Float32Array.from({ length: 4000 }, (_, i) => i));

    const statsWithBuffers = backend.allocatorStats().totalAllocated;
    expect(statsWithBuffers).toBeGreaterThan(statsBefore);

    // Free all buffers — allocator should reset.
    a.dispose();
    b.dispose();
    c.dispose();

    const statsAfterFree = backend.allocatorStats().totalAllocated;
    // After freeing all, the bump pointer should have been reset.
    expect(statsAfterFree).toBeLessThanOrEqual(statsBefore);
  });

  test("top-of-heap compaction reclaims trailing freed blocks", async () => {
    await init("wasm");
    const backend = getBackend("wasm") as any;

    // Use non-zero data to avoid the all-equal short-circuit in arrayFromData
    // (arrays with all-equal elements use fullInternal, not backend.malloc).
    const data = Float32Array.from({ length: 200 }, (_, i) => i);

    // Allocate 3 blocks: a, b, c (in order on the heap).
    const a = np.array(data);
    const statsA = backend.allocatorStats().totalAllocated;

    const b = np.array(data);
    const statsB = backend.allocatorStats().totalAllocated;
    expect(statsB).toBeGreaterThan(statsA);

    const c = np.array(data);
    const statsC = backend.allocatorStats().totalAllocated;
    expect(statsC).toBeGreaterThan(statsB);

    // Free c (top of heap) — should compact.
    c.dispose();
    const afterFreeC = backend.allocatorStats().totalAllocated;
    expect(afterFreeC).toBeLessThanOrEqual(statsB);

    // Free b (now top of heap) — should compact further.
    b.dispose();
    const afterFreeB = backend.allocatorStats().totalAllocated;
    expect(afterFreeB).toBeLessThanOrEqual(statsA);

    // 'a' is still alive, so full reset doesn't trigger.
    a.dispose();
  });

  test("ascending N sequence does not OOM", async () => {
    await init("wasm");
    // Adapted from dlm-js repro-wasm-oom-large-n.ts test 3:
    // Ascending sequence of sizes should not accumulate memory.
    const f = jit((x: np.Array) => x.mul(x).sum());

    // Use 20 ascending sizes (sqrt(2) increments from 100 to ~51k elements).
    const sizes = [];
    for (let i = 0; i < 20; i++) {
      sizes.push(Math.round(100 * Math.pow(2, i * 0.5)));
    }

    for (const n of sizes) {
      // Cap at 1M elements for test speed.
      const actualN = Math.min(n, 1_000_000);
      using x = np.array(new Float32Array(actualN));
      using result = f(x) as np.Array;
      result.js();
    }

    f.dispose();

    // If we got here without OOM, the test passes.
    const backend = getBackend("wasm") as any;
    const stats = backend.allocatorStats();
    expect(stats.totalAllocated).toBeLessThan(100_000_000); // < 100MB
  });
});
