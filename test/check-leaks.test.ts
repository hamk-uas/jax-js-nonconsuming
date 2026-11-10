/**
 * Tests for the checkLeaks diagnostic API.
 *
 * checkLeaks uses backend slot count deltas (same as slotCount()) for accurate
 * leak detection, plus a tracking map for diagnostic details.
 */
import {
  checkLeaks,
  init,
  jit,
  type LeakReport,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, expectTypeOf, it } from "vitest";

await init();

describe("checkLeaks", () => {
  it("reports zero leaks when all arrays disposed", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5, 6]);
    x.dispose();
    y.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
    expect(report.userLeaked).toBe(0);
    expect(report.internalLeaked).toBe(0);
    expect(report.summary).toBe("No leaks detected.");
  });

  it("detects a single leaked array", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    // deliberately do NOT dispose x
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(1);
    expect(report.userLeaked).toBe(1);
    expect(report.internalLeaked).toBe(0);
    expect(report.details.length).toBeGreaterThanOrEqual(1);
    expect(report.summary).toContain("1 slot(s) leaked");
    // Clean up for next test
    x.dispose();
  });

  it("detects multiple leaked arrays", () => {
    checkLeaks.start();
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);
    const c = np.array([7, 8, 9]);
    a.dispose(); // only a is disposed
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(2);
    b.dispose();
    c.dispose();
  });

  it("jit intermediates do not leak", () => {
    checkLeaks.start();
    // Use a function without anonymous constants to avoid the known
    // anonymous-constant leak (see copilot-instructions).
    const f = jit((x: np.Array) => x.add(x).mul(x));
    const x = np.array([2, 3]);
    const result = f(x);
    expect(result.js()).toEqual([8, 18]);
    result.dispose();
    x.dispose();
    f.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
  });

  it("active property reflects tracking state", () => {
    checkLeaks.stop(); // override setup.ts beforeEach which calls start()
    expect(checkLeaks.active).toBe(false);
    checkLeaks.start();
    expect(checkLeaks.active).toBe(true);
    checkLeaks.stop();
    expect(checkLeaks.active).toBe(false);
  });

  it("arrays created before start are not counted", () => {
    const preExisting = np.array([99, 100]);
    checkLeaks.start();
    const tracked = np.array([1, 2]);
    tracked.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0); // preExisting not counted (existed before start)
    preExisting.dispose();
  });

  it("report details include array description", () => {
    checkLeaks.start();
    const x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(1);
    // Details should contain dtype and shape info from tracked arrays
    expect(report.details.length).toBeGreaterThanOrEqual(1);
    const hasArrayInfo = report.details.some(
      (d) => /float32|int32/.test(d) && /2,2|2, 2/.test(d),
    );
    expect(hasArrayInfo).toBe(true);
    x.dispose();
  });

  it("snapshot returns live tracked arrays", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5]);
    const snap = checkLeaks.snapshot();
    expect(snap.length).toBe(2);
    expect(snap.every((e) => e.rc === 1)).toBe(true);
    x.dispose();
    y.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
  });

  it("trackRefs records .ref call sites", () => {
    checkLeaks.start({ trackRefs: true });
    const x = np.array([1, 2, 3]);
    // jax-js-lint: allow-ref — intentional test of .ref tracking
    const y = x.ref; // extra ref
    const snap = checkLeaks.snapshot();
    expect(snap.length).toBe(1);
    expect(snap[0].rc).toBe(2);
    expect(snap[0].lastRef).not.toBeNull();
    x.dispose();
    y.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
  });

  it("trackSites: false counts leaks without recording sites", () => {
    checkLeaks.start({ trackSites: false });
    const x = np.array([1, 2, 3]);
    expect(checkLeaks.snapshot()).toEqual([]);
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(1);
    expect(report.userLeaked).toBe(0);
    expect(report.details).toEqual([]);
    expect(report.summary).toContain("Creation sites were not tracked");
    x.dispose();
  });

  it("trackRefs requires trackSites", () => {
    expect(() =>
      checkLeaks.start({ trackRefs: true, trackSites: false }),
    ).toThrow("trackRefs requires trackSites");
  });
});

describe("checkLeaks.steadyState", () => {
  it("accepts a balanced body whose jit cache stays warm", async () => {
    const f = jit((a: np.Array) => a.mul(2));
    try {
      const report = await checkLeaks.steadyState(() => {
        using x = np.array([1, 2, 3]);
        using y = f(x);
        y.js();
      });
      expect(report.leaked).toBe(0);
      expect(report.summary).toBe("No leaks detected.");
    } finally {
      f.dispose();
    }
  });

  it("reports a body that leaks on every run", async () => {
    const kept: np.Array[] = [];
    try {
      const report = await checkLeaks.steadyState(() => {
        kept.push(np.array([1, 2, 3]));
      });
      expect(report.leaked).toBe(1);
      expect(report.summary).toContain("Not steady");
    } finally {
      for (const x of kept) x.dispose();
    }
  });

  it("reports a leak into a cache that stop() would flush", async () => {
    // A fresh jit function per run keeps its compiled constants until it is
    // disposed or caches are cleared.
    const fns: ReturnType<typeof jit>[] = [];
    try {
      const report = await checkLeaks.steadyState(() => {
        const f = jit((a: np.Array) => a.add(np.array([1, 1, 1])));
        fns.push(f);
        using x = np.array([1, 2, 3]);
        using y = f(x);
        y.js();
      });
      expect(report.leaked).toBeGreaterThan(0);
    } finally {
      for (const f of fns) f.dispose();
    }
  });

  it("awaits an async body", async () => {
    const kept: np.Array[] = [];
    try {
      const report = await checkLeaks.steadyState(async () => {
        using x = np.array([1, 2, 3]);
        await x.data();
        kept.push(np.array([4, 5]));
      });
      expect(report.leaked).toBe(1);
    } finally {
      for (const x of kept) x.dispose();
    }
  });

  it("types an async body with an ignored result as a promise", async () => {
    let completed = 0;
    const result = checkLeaks.steadyState(async () => ++completed);
    expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
    const report = await result;
    expect(completed).toBe(2);
    expect(report.leaked).toBe(0);
    expect(report.summary).toBe("No leaks detected.");
  });

  it("returns a promise for an ignored synchronous result", async () => {
    let completed = 0;
    const result = checkLeaks.steadyState(() => ++completed);
    expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
    const report = await result;
    expect(result).toBeInstanceOf(Promise);
    expect(completed).toBe(2);
    expect(report.summary).toBe("No leaks detected.");
  });

  it.each([true, false])(
    "handles a possibly async body (async warm-up: %s)",
    async (asyncWarmUp) => {
      let calls = 0;
      const body = (): void | Promise<number> => {
        calls++;
        if ((calls === 1) === asyncWarmUp) return Promise.resolve(calls);
      };
      const result = checkLeaks.steadyState(body);
      expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
      const report = await result;
      expect(calls).toBe(2);
      expect(report.summary).toBe("No leaks detected.");
    },
  );

  it("allows an unknown callback result to be asynchronous", async () => {
    const body = (): unknown => Promise.resolve(123);
    const result = checkLeaks.steadyState(body);
    expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
    expect((await result).summary).toBe("No leaks detected.");
  });

  it("awaits a promise even when the callback type discards it", async () => {
    let completed = 0;
    // eslint-disable-next-line @typescript-eslint/no-misused-promises -- intentionally erase the returned promise to test the completion contract
    const body: () => void = async () => ++completed;
    const result = checkLeaks.steadyState(body);
    expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
    const report = await result;
    expect(completed).toBe(2);
    expect(report.summary).toBe("No leaks detected.");
  });

  it("awaits thenables with ignored results sequentially", async () => {
    let completed = 0;
    const starts: number[] = [];
    const body = (): PromiseLike<number> => {
      starts.push(completed);
      return {
        then: (resolve, reject) =>
          Promise.resolve()
            .then(() => ++completed)
            .then(resolve, reject),
      };
    };
    const result = checkLeaks.steadyState(body);
    expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
    const report = await result;
    expect(completed).toBe(2);
    expect(starts).toEqual([0, 1]);
    expect(report.summary).toBe("No leaks detected.");
  });

  it("types and awaits structural thenables whose then returns void", async () => {
    let completed = 0;
    const starts: number[] = [];
    const kept: np.Array[] = [];
    const body = () => {
      starts.push(completed);
      return {
        then(resolve: (value: number) => void) {
          queueMicrotask(() => {
            kept.push(np.array([1, 2], { device: "cpu" }));
            resolve(++completed);
          });
        },
      };
    };
    try {
      const result = checkLeaks.steadyState(body);
      expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
      const report = await result;
      expect(completed).toBe(2);
      expect(starts).toEqual([0, 1]);
      expect(report.leaked).toBe(1);
      expect(report.summary).toContain("cpu: +1");
    } finally {
      for (const x of kept) x.dispose();
    }
  });

  it.each([true, false])(
    "handles structural thenable unions (async warm-up: %s)",
    async (asyncWarmUp) => {
      let completed = 0;
      const starts: number[] = [];
      const body = () => {
        starts.push(completed);
        if ((starts.length === 1) === asyncWarmUp) {
          return {
            then(resolve: (value: number) => void) {
              queueMicrotask(() => resolve(++completed));
            },
          };
        }
        return ++completed;
      };
      const result = checkLeaks.steadyState(body);
      expectTypeOf(result).toEqualTypeOf<Promise<LeakReport>>();
      const report = await result;
      expect(completed).toBe(2);
      expect(starts).toEqual([0, 1]);
      expect(report.summary).toBe("No leaks detected.");
    },
  );

  it.each([1, 2])(
    "rejects a synchronous throw from run %s",
    async (failedRun) => {
      let calls = 0;
      const error = new Error("body failed");
      const body = () => {
        if (++calls === failedRun) throw error;
      };
      await expect(checkLeaks.steadyState(body)).rejects.toBe(error);
      expect(calls).toBe(failedRun);
    },
  );

  it.each([1, 2])("propagates rejection from run %s", async (failedRun) => {
    let calls = 0;
    const error = new Error("body failed");
    const body = async () => {
      if (++calls === failedRun) throw error;
    };
    await expect(checkLeaks.steadyState(body)).rejects.toBe(error);
    expect(calls).toBe(failedRun);
  });

  it("awaits cross-realm promises sequentially", async () => {
    const frame = document.createElement("iframe");
    document.body.append(frame);
    const pending: Promise<void>[] = [];
    const kept: np.Array[] = [];
    let completed = 0;
    const starts: number[] = [];
    try {
      if (!frame.contentWindow) throw new Error("Missing iframe window");
      const OtherPromise = Reflect.get(
        frame.contentWindow,
        "Promise",
      ) as PromiseConstructor;
      const report = await checkLeaks.steadyState(() => {
        starts.push(completed);
        const promise = new OtherPromise<void>((resolve) => {
          setTimeout(() => {
            kept.push(np.array([1, 2], { device: "cpu" }));
            completed++;
            resolve();
          }, 0);
        });
        pending.push(promise);
        return promise;
      });
      expect(completed).toBe(2);
      expect(starts).toEqual([0, 1]);
      expect(report.leaked).toBe(1);
      expect(report.summary).toContain("cpu: +1");
    } finally {
      await Promise.all(pending);
      for (const x of kept) x.dispose();
      frame.remove();
    }
  });

  it("does not cancel opposite backend slot changes", async () => {
    // Nonuniform data forces backend storage instead of inline constants.
    const inputs = [
      np.array([1, 2], { device: "cpu" }),
      np.array([2, 3], { device: "cpu" }),
    ];
    const kept: np.Array[] = [];
    try {
      const report = await checkLeaks.steadyState(() => {
        inputs.pop()!.dispose();
        kept.push(np.array([3, 4], { device: "wasm" }));
      });
      expect(report.leaked).toBe(2);
      expect(report.summary).toContain("Not steady");
      expect(report.summary).toContain("cpu: -1");
      expect(report.summary).toContain("wasm: +1");
    } finally {
      for (const x of inputs) x.dispose();
      for (const x of kept) x.dispose();
    }
  });

  it("reports a decrease rather than claiming steady state", async () => {
    const inputs = [
      np.array([1, 2], { device: "cpu" }),
      np.array([2, 3], { device: "cpu" }),
    ];
    try {
      const report = await checkLeaks.steadyState(() =>
        inputs.pop()!.dispose(),
      );
      expect(report.leaked).toBe(1);
      expect(report.summary).toContain("Not steady");
      expect(report.summary).toContain("cpu: -1");
    } finally {
      for (const x of inputs) x.dispose();
    }
  });

  it("leaves an active session untouched", async () => {
    checkLeaks.start();
    const report = await checkLeaks.steadyState(() => {
      using x = np.array([1, 2, 3]);
      x.js();
    });
    expect(report.leaked).toBe(0);
    expect(checkLeaks.active).toBe(true);
    expect(checkLeaks.stop().leaked).toBe(0);
  });
});
