/**
 * Leak detection tests for scan fallback path.
 *
 * CPU backend always uses fallback (no compiled-loop support), making it
 * ideal for verifying that the fallback loop doesn't leak slots.
 *
 * NOTE on lazy constants: np.array([scalar]) and np.array([v,...v]) with
 * <128 elements create AluExp-backed arrays — no backend.malloc. Use
 * distinct-valued multi-element arrays to force eager allocation if you
 * need precise slot counting.
 *
 * Each test uses checkLeaks.start()/stop() to verify zero leaked arrays.
 */
import {
  checkLeaks,
  defaultDevice,
  init,
  lax,
  numpy as np,
} from "@jax-js-nonconsuming/jax";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

let previousDevice: string | undefined;

describe("scan fallback leak detection (CPU)", () => {
  beforeAll(async () => {
    const devices = await init();
    // Remember the best non-CPU device so we can restore it in afterAll
    previousDevice = devices.includes("webgpu")
      ? "webgpu"
      : devices.includes("wasm")
        ? "wasm"
        : undefined;
    defaultDevice("cpu");
  });

  afterAll(() => {
    if (previousDevice) defaultDevice(previousDevice as any);
  });

  it("cumsum body does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, null] => {
      return [np.add(carry, x), null];
    };

    checkLeaks.start();
    const [carry] = lax.scan(step, initC, xs);
    await carry.data();
    carry.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("body with captured constant does not leak", async () => {
    const bias = np.array([100, 200]); // distinct → eager alloc
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, null] => {
      return [np.add(np.add(carry, x), bias), null];
    };

    checkLeaks.start();
    const [carry] = lax.scan(step, initC, xs);
    await carry.data();
    carry.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    xs.dispose();
    initC.dispose();
    bias.dispose();
  });

  it("passthrough body (carry = y) does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, any] => {
      const newC = np.add(carry, x);
      return [newC, newC];
    };

    checkLeaks.start();
    const [carry, ys] = lax.scan(step, initC, xs);
    await carry.data();
    await ys.data();
    carry.dispose();
    ys.dispose();
    const report = checkLeaks.stop();
    expect(
      report.leaked,
      `${report.summary}\n${report.details.join("\n")}`,
    ).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("body with independent carry and y does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initC = np.array([10, 20]);
    const factor = np.array([2, 3]);

    const step = (carry: any, x: any): [any, any] => {
      const newC = np.add(carry, x);
      return [newC, np.multiply(x, factor)];
    };

    checkLeaks.start();
    const [carry, ys] = lax.scan(step, initC, xs);
    await carry.data();
    await ys.data();
    carry.dispose();
    ys.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    xs.dispose();
    initC.dispose();
    factor.dispose();
  });

  it("body with inline np.array constant does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, any] => {
      const newC = np.add(carry, x);
      return [newC, np.multiply(x, np.array([2, 3]))];
    };

    checkLeaks.start();
    const [carry, ys] = lax.scan(step, initC, xs);
    await carry.data();
    await ys.data();
    carry.dispose();
    ys.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("xs=null carry-only scan does not leak", async () => {
    const initC = np.array([10, 20]);
    const increment = np.array([1, 2]);

    const step = (carry: any, _x: any): [any, null] => {
      const newC = np.add(carry, increment);
      return [newC, null];
    };

    checkLeaks.start();
    const [carry, ys] = lax.scan(step, initC, null, { length: 5 });
    expect(ys).toBeNull();
    await carry.data();
    carry.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    initC.dispose();
    increment.dispose();
  });

  it("inline np.array constants with xs=null length 0 and 1 do not leak", async () => {
    const step = (carry: any, _x: any): [any, any] => {
      const newC = np.add(carry, np.array([1, 2]));
      return [newC, np.multiply(newC, np.array([2, 3]))];
    };

    for (const length of [0, 1]) {
      checkLeaks.start();
      const initC = np.array([10, 20]);
      const [carry, ys] = lax.scan(step, initC, null, { length });
      await carry.data();
      await ys.data();

      // length=0 returns init carry directly; avoid double-dispose.
      if (carry !== initC) carry.dispose();
      ys.dispose();
      initC.dispose();

      const report = checkLeaks.stop();
      expect(report.leaked, `length=${length} ${report.summary}`).toBe(0);
    }
  });

  it("pytree carry does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initA = np.array([10, 20]);
    const initB = np.array([30, 40]);
    const bIncrement = np.array([1, 2]);
    const yFactor = np.array([2, 3]);

    const step = (carry: { a: any; b: any }, x: any): [any, any] => {
      const newA = np.add(carry.a, x);
      const newB = np.add(carry.b, bIncrement);
      return [{ a: newA, b: newB }, np.multiply(x, yFactor)];
    };

    checkLeaks.start();
    const [carry, ys] = lax.scan(step, { a: initA, b: initB }, xs);
    await (carry as any).a.data();
    await (carry as any).b.data();
    await ys.data();
    (carry as any).a.dispose();
    (carry as any).b.dispose();
    ys.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    xs.dispose();
    initA.dispose();
    initB.dispose();
    bIncrement.dispose();
    yFactor.dispose();
  });

  it("reverse scan does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const initC = np.array([10, 20]);
    const yFactor = np.array([3, 4]);

    const step = (carry: any, x: any): [any, any] => {
      return [np.add(carry, x), np.multiply(x, yFactor)];
    };

    checkLeaks.start();
    const [carry, ys] = lax.scan(step, initC, xs, {
      reverse: true,
    });
    await carry.data();
    await ys.data();
    carry.dispose();
    ys.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    xs.dispose();
    initC.dispose();
    yFactor.dispose();
  });

  it("longer scan (L=50) does not leak", async () => {
    const data = Array.from({ length: 50 }, (_, i) => [i * 2, i * 2 + 1]);
    const xs = np.array(data);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, any] => {
      return [np.add(carry, x), null];
    };

    checkLeaks.start();
    const [carry] = lax.scan(step, initC, xs);
    await carry.data();
    carry.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("acceptPath is enforced in eager mode", () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, null] => {
      return [np.add(carry, x), null];
    };

    // CPU only supports fallback — requesting compiled-loop should throw
    expect(() =>
      lax.scan(step, initC, xs, { acceptPath: "compiled-loop" }),
    ).toThrow();

    xs.dispose();
    initC.dispose();
  });
});
