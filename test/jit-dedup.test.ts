/**
 * Tests for jit() function-identity deduplication.
 *
 * When `jit(fn)(args)` is called inline (no persistent reference to the
 * wrapper), each call used to create a new OwnedFunction with a new cache,
 * accumulating GPU-backed ClosedJaxpr consts indefinitely. The dedup registry
 * ensures same `(fn, opts)` → same OwnedFunction + shared cache.
 */
import {
  getBackend,
  init,
  jit,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

await init();

// Named functions for identity-based dedup testing.
// Use raw number literals (not np.array([1])) so the jit body const goes
// through pureArray → anonymousConstArrays → properly builder-owned.
function addOne(x: np.Array) {
  return x.add(1);
}

function mulTwo(x: np.Array) {
  return x.mul(2);
}

describe("jit dedup", () => {
  it("returns the same OwnedFunction for the same function reference", () => {
    const f1 = jit(addOne);
    const f2 = jit(addOne);
    expect(f1).toBe(f2);
    // Single dispose is sufficient — both references point to the same object.
    f1.dispose();
  });

  it("returns different OwnedFunctions for different function references", () => {
    const f1 = jit(addOne);
    const f2 = jit(mulTwo);
    expect(f1).not.toBe(f2);
    f1.dispose();
    f2.dispose();
  });

  it("returns different OwnedFunctions for different opts", () => {
    const f1 = jit(addOne);
    const f2 = jit(addOne, { staticArgnums: [0] });
    expect(f1).not.toBe(f2);
    f1.dispose();
    f2.dispose();
  });

  it("returns same OwnedFunction for equivalent opts", () => {
    // Same staticArgnums in different order — should serialize identically.
    const f1 = jit(addOne, { staticArgnums: [1, 0] });
    const f2 = jit(addOne, { staticArgnums: [0, 1] });
    expect(f1).toBe(f2);
    f1.dispose();
  });

  it("inline jit(fn)(args) does not accumulate slots", () => {
    const backend = getBackend();

    // Warm up: first call traces and populates the cache.
    {
      using input = np.array([1, 2, 3]);
      using r0 = jit(addOne)(input) as np.Array;
      expect(r0.js()).toEqual([2, 3, 4]);
    }

    // Snapshot after first call — only cache const remains (input + result disposed).
    const slotsAfterFirst = backend.slotCount();

    // Second call should hit the shared cache — no new consts allocated.
    {
      using input = np.array([4, 5, 6]);
      using r1 = jit(addOne)(input) as np.Array;
      expect(r1.js()).toEqual([5, 6, 7]);
    }

    const slotsAfterSecond = backend.slotCount();

    // With dedup: cache hit, no new const. Slot count unchanged.
    // Without dedup: new cache, new const → slotsAfterSecond > slotsAfterFirst.
    expect(slotsAfterSecond).toBe(slotsAfterFirst);

    // Clean up the deduped function's cache.
    jit(addOne).dispose();
  });

  it("repeated inline calls stay flat in slot count", () => {
    const backend = getBackend();

    // First call warms the cache.
    {
      using input = np.array([10]);
      using r0 = jit(addOne)(input) as np.Array;
      expect(r0.js()).toEqual([11]);
    }
    const baseline = backend.slotCount();

    // 10 more calls — slot count should not grow (cache hit each time,
    // inputs and results disposed immediately).
    for (let i = 0; i < 10; i++) {
      using input = np.array([i]);
      using result = jit(addOne)(input) as np.Array;
      void result;
    }

    const afterLoop = backend.slotCount();
    // With dedup: all cache hits, no new consts. Slot count unchanged.
    expect(afterLoop).toBe(baseline);

    jit(addOne).dispose();
  });

  it("dispose clears cache, next call re-traces correctly", () => {
    // First call populates cache.
    {
      using input = np.array([1, 2]);
      using r1 = jit(addOne)(input) as np.Array;
      expect(r1.js()).toEqual([2, 3]);
    }

    // Dispose clears the cache (consts freed).
    jit(addOne).dispose();

    // Next call should re-trace (cache miss) and still produce correct results.
    {
      using input = np.array([3, 4]);
      using r2 = jit(addOne)(input) as np.Array;
      expect(r2.js()).toEqual([4, 5]);
    }

    // Still the same OwnedFunction from the registry.
    const f1 = jit(addOne);
    const f2 = jit(addOne);
    expect(f1).toBe(f2);
    f1.dispose();
  });

  it("arrow functions in a loop are correctly NOT deduped", () => {
    // Each iteration creates a new arrow function object — different identity.
    // This is correct: each arrow could capture different closure state.
    const fns: ReturnType<typeof jit>[] = [];
    for (let i = 0; i < 3; i++) {
      // Capture i by value. Use raw literal so const is properly anonymous.
      const offset = i;
      fns.push(jit((x: np.Array) => x.add(offset)));
    }

    // All different OwnedFunctions.
    expect(fns[0]).not.toBe(fns[1]);
    expect(fns[1]).not.toBe(fns[2]);

    // And they produce different results (different closure state).
    {
      using input = np.array([10]);
      using r0 = fns[0](input) as np.Array;
      expect(r0.js()).toEqual([10]); // i=0
    }
    {
      using input = np.array([10]);
      using r1 = fns[1](input) as np.Array;
      expect(r1.js()).toEqual([11]); // i=1
    }
    {
      using input = np.array([10]);
      using r2 = fns[2](input) as np.Array;
      expect(r2.js()).toEqual([12]); // i=2
    }

    for (const f of fns) f.dispose();
  });
});
