/**
 * WebGPU buffer pool peak-memory guarantee tests.
 *
 * Tests buffer-pool invariants on real Chromium WebGPU.
 *
 * Key mechanisms tested:
 * - `configurePool()` called before each JIT execution caps retained bytes
 * - pool + live ≤ peakBytes after warmup
 * - Stale pool entries evicted on shape change
 * - gpuAllocatedBytes tracking
 */
import {
  blockUntilReady,
  clearCaches,
  defaultDevice,
  getBackend,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";

describe("WebGPU buffer pool memory", () => {
  let hasWebGPU = false;
  let prevDevice: ReturnType<typeof defaultDevice>;

  function getGpuBytes(): number {
    return (getBackend() as any).gpuAllocatedBytes();
  }

  function getSlotCount(): number {
    return (getBackend() as any).slotCount();
  }

  async function measurePeakSlotsDuring(
    run: () => Promise<void>,
  ): Promise<{ baseline: number; peak: number; delta: number; end: number }> {
    const backend = getBackend() as any;
    const origMalloc = backend.malloc.bind(backend);
    const origDecRef = backend.decRef.bind(backend);

    const baseline = getSlotCount();
    let peak = baseline;
    const sample = () => {
      const current = getSlotCount();
      if (current > peak) peak = current;
    };

    backend.malloc = (...args: any[]) => {
      const out = origMalloc(...args);
      sample();
      return out;
    };
    backend.decRef = (...args: any[]) => {
      const out = origDecRef(...args);
      sample();
      return out;
    };

    try {
      sample();
      await run();
      sample();
    } finally {
      backend.malloc = origMalloc;
      backend.decRef = origDecRef;
    }

    return { baseline, peak, delta: peak - baseline, end: getSlotCount() };
  }

  // Under fileParallelism, concurrent GPU tests can delay readbacks.
  // Allow enough time for contended GPU queue to drain.
  beforeEach(() => clearCaches());

  beforeAll(async () => {
    const devices = await init();
    hasWebGPU = devices.includes("webgpu");
    if (hasWebGPU) {
      prevDevice = defaultDevice("webgpu");
    }
  });

  afterAll(() => {
    if (hasWebGPU) defaultDevice(prevDevice as any);
  });

  it(
    "repeated JIT calls stay within peak memory",
    { timeout: 30_000 },
    async ({ skip }) => {
      if (!hasWebGPU) skip();

      using f = jit((x: any) => x.add(1).mul(2).sub(3));
      using x = np.ones([1024]);

      // Warmup
      const warmup = f(x);
      await warmup.data();
      warmup.dispose();

      const baselineBytes = getGpuBytes();

      for (let i = 0; i < 20; i++) {
        const result = f(x);
        await result.data();
        result.dispose();
      }

      expect(getGpuBytes()).toBeLessThanOrEqual(baselineBytes);
    },
  );

  it(
    "multi-output JIT stays within peak memory",
    { timeout: 30_000 },
    async ({ skip }) => {
      if (!hasWebGPU) skip();

      using f = jit((x: any) => [x.add(1), x.mul(2)]);
      using x = np.ones([2048]);

      // Warmup
      const [a, b] = f(x) as any[];
      await a.data();
      await b.data();
      a.dispose();
      b.dispose();

      const baselineBytes = getGpuBytes();

      for (let i = 0; i < 15; i++) {
        const [r1, r2] = f(x) as any[];
        await r1.data();
        await r2.data();
        r1.dispose();
        r2.dispose();
      }

      expect(getGpuBytes()).toBeLessThanOrEqual(baselineBytes);
    },
  );

  it("shape-varying JIT calls don't accumulate stale buffers", async ({
    skip,
  }) => {
    if (!hasWebGPU) skip();

    using fSmall = jit((x: any) => x.add(1).mul(2));
    using fLarge = jit((x: any) => x.add(1).mul(2));
    using xSmall = np.ones([256]);
    using xLarge = np.ones([4096]);

    // Warmup both
    (await fSmall(xSmall)).dispose();
    (await fLarge(xLarge)).dispose();

    const baselineBytes = getGpuBytes();

    for (let i = 0; i < 10; i++) {
      (await fSmall(xSmall)).dispose();
      (await fLarge(xLarge)).dispose();
    }

    expect(getGpuBytes()).toBeLessThanOrEqual(baselineBytes);
  });

  it("eager vs jit peak-memory parity", async ({ skip }) => {
    if (!hasWebGPU) skip();

    using x = np.ones([4096]);
    using addV = np.ones([4096]);
    using mulV = np.full([4096], 2);
    using fJit = jit((a: any, b: any, c: any) => a.sort().add(b).mul(c));

    const eagerRun = async () => {
      using t1 = x.sort();
      using t2 = t1.add(addV);
      using out = t2.mul(mulV);
      await out.data();
    };
    const jitRun = async () => {
      using out = fJit(x, addV, mulV);
      await out.data();
    };

    // Warmup JIT compile outside measurement
    await jitRun();

    const eagerPeak = await measurePeakSlotsDuring(eagerRun);
    const jitPeak = await measurePeakSlotsDuring(jitRun);

    // JIT should not use significantly MORE peak slots than eager.
    // The command tape (O8) manages intermediates as raw GPUBuffers outside
    // the slot system, so jitPeak.delta can be lower than eagerPeak.delta —
    // that's strictly better, not a regression.
    expect(jitPeak.delta).toBeLessThanOrEqual(eagerPeak.delta + 1);
  });

  it(
    "scan cumsum stays within peak memory",
    { timeout: 30_000 },
    async ({ skip }) => {
      if (!hasWebGPU) skip();

      const initCarry = np.zeros([64]);
      using scanF = jit((xs: any) =>
        lax.scan(
          (carry: any, x: any) => {
            const s = carry.add(x);
            return [s, s];
          },
          initCarry,
          xs,
        ),
      );
      using xs = np.ones([100, 64]);

      // Warmup
      const [c, ys] = scanF(xs) as any[];
      await c.data();
      await ys.data();
      c.dispose();
      ys.dispose();

      const baselineBytes = getGpuBytes();

      for (let i = 0; i < 5; i++) {
        const [c2, ys2] = scanF(xs) as any[];
        await c2.data();
        await ys2.data();
        c2.dispose();
        ys2.dispose();
      }

      // Allow small tolerance: concurrent tests may add pool entries between
      // baseline measurement and final check (fileParallelism shares one GPUDevice).
      const tolerance = 4096;
      expect(getGpuBytes()).toBeLessThanOrEqual(baselineBytes + tolerance);
      initCarry.dispose();
    },
  );

  it("gpuAllocatedBytes tracks creates and pool returns", async ({ skip }) => {
    if (!hasWebGPU) skip();

    // Use a unique size unlikely to be in the pool from previous tests
    const arr = np.ones([7919]);
    await blockUntilReady(arr);

    const withArrayBytes = getGpuBytes();

    arr.dispose();

    const afterBytes = getGpuBytes();
    // After dispose: pool may hold buffer (bytes same) or destroy it (bytes decrease)
    expect(afterBytes).toBeLessThanOrEqual(withArrayBytes);
    expect(afterBytes).toBeGreaterThanOrEqual(0);
  });
});
