/**
 * Scan performance benchmark — measures scan paths on the current backend.
 * Run with: pnpm vitest run test/scan-bench.test.ts
 */

import {
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
} from "@jax-js-nonconsuming/jax";
import { describe, expect, it } from "vitest";

async function timeMs(fn: () => void, warmup = 3, runs = 10): Promise<number> {
  for (let i = 0; i < warmup; i++) fn();
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

describe("scan performance benchmarks", () => {
  it("cumsum: N=100, SIZE=64", async () => {
    await init();
    const N = 100;
    const SIZE = 64;
    using xs = np.ones([N, SIZE]);
    using initCarry = np.zeros([SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry];
        },
        initCarry,
        xs,
      );
    });

    // Verify correctness
    const [c, _y] = scanAuto(xs);
    const data = await c.data();
    expect(data[0]).toBeCloseTo(N, 1);
    _y.dispose();
    c.dispose();

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `cumsum          (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
  });

  it("cumsum large: N=500, SIZE=256", async () => {
    await init();
    const N = 500;
    const SIZE = 256;
    using xs = np.ones([N, SIZE]);
    using initCarry = np.zeros([SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry];
        },
        initCarry,
        xs,
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `cumsum-large    (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
  });

  it("carry-only scan: N=200, SIZE=32", async () => {
    await init();
    const N = 200;
    const SIZE = 32;
    using onesConst = np.ones([SIZE]);
    using initCarry = np.zeros([SIZE]);

    const scanAuto = jit(() => {
      return lax.scan(
        (carry: any, _x: any) => {
          const newCarry = carry.add(onesConst);
          return [newCarry, null];
        },
        initCarry,
        null,
        { length: N },
      );
    });

    const autoMs = await timeMs(() => {
      const [c, _] = scanAuto() as [any, any];
      c.dispose();
    });
    console.log(
      `carry-only      (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
  });

  it("multi-carry scan: N=100, SIZE=32", async () => {
    await init();
    const N = 100;
    const SIZE = 32;
    using xs = np.ones([N, SIZE]);
    using decay = np.array([0.99]);
    using scale = np.array([0.01]);
    using initA = np.zeros([SIZE]);
    using initB = np.zeros([SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const [a, b] = carry;
          const newA = a.add(x);
          const newB = b.mul(decay).add(x.mul(scale));
          return [
            [newA, newB],
            [newA, newB],
          ] as any;
        },
        [initA, initB] as any,
        xs,
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      (c as any)[0].dispose();
      (c as any)[1].dispose();
      (y as any)[0].dispose();
      (y as any)[1].dispose();
    });
    console.log(
      `multi-carry     (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
  });

  it("scan with reduction: N=100, SIZE=64", async () => {
    await init();
    const N = 100;
    const SIZE = 64;
    using xs = np.ones([N, SIZE]);
    using initCarry = np.zeros([]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const s = carry.add(np.sum(x));
          return [s, s];
        },
        initCarry,
        xs,
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `reduction       (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
  });

  it("reverse scan: N=200, SIZE=64", async () => {
    await init();
    const N = 200;
    const SIZE = 64;
    using xs = np.ones([N, SIZE]);
    using initCarry = np.zeros([SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry];
        },
        initCarry,
        xs,
        { reverse: true },
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `reverse         (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
  });
});
