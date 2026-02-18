/**
 * @file Tests for the wasmblr peephole optimizer.
 *
 * Verifies that enabling the peephole pass:
 * 1. Does not change numerical results (correctness).
 * 2. Applies meaningful rewrites (unit tests on raw bytecodes).
 *
 * Constants are extracted outside `jit()` bodies to avoid the anonymous
 * constant leak (see copilot-instructions §4 "Anonymous constants").
 */

import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  lax,
  numpy as np,
  setWasmPeephole,
  tree,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterEach, describe, expect, test } from "vitest";

// Direct import for unit-testing the peephole on raw bytes
import {
  newPeepholeStats,
  optimizeFunctionBody,
} from "../src/backend/wasm/wasmblr-peephole";

const devices = await init("wasm");
const hasWasm = devices.includes("wasm");

afterEach(() => {
  setWasmPeephole(false);
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe.skipIf(!hasWasm)("wasmblr peephole optimizer", () => {
  test("cumsum correctness: peephole on matches peephole off", async () => {
    defaultDevice("wasm");
    using xs = np.ones([50, 8]);
    await blockUntilReady(xs);

    // Extract init outside jit to avoid anonymous constant leak
    using init = np.zeros([8]);

    // --- Run without peephole ---
    setWasmPeephole(false);
    let cOffData: Float32Array, yOffData: Float32Array;
    {
      using f = jit((xs: np.Array) =>
        lax.scan(
          (carry: np.Array, x: np.Array) => {
            const c = carry.add(x);
            return [c, c];
          },
          init,
          xs,
        ),
      );
      const [cOff, yOff] = f(xs) as [np.Array, np.Array];
      cOffData = (await cOff.data()) as Float32Array;
      yOffData = (await yOff.data()) as Float32Array;
      cOff.dispose();
      yOff.dispose();
    }

    // --- Run with peephole ---
    setWasmPeephole(true);
    {
      using f = jit((xs: np.Array) =>
        lax.scan(
          (carry: np.Array, x: np.Array) => {
            const c = carry.add(x);
            return [c, c];
          },
          init,
          xs,
        ),
      );
      const [cOn, yOn] = f(xs) as [np.Array, np.Array];
      const cOnData = await cOn.data();
      const yOnData = await yOn.data();
      cOn.dispose();
      yOn.dispose();

      expect(Array.from(cOnData)).toEqual(Array.from(cOffData));
      expect(Array.from(yOnData)).toEqual(Array.from(yOffData));
    }
  });

  test("Kalman filter correctness: peephole on matches peephole off", async () => {
    defaultDevice("wasm");

    // Extract all constants outside jit to avoid anonymous constant leak
    using processNoise = np.array([0.01, 0.01, 0.01, 0.01]);
    using measNoise = np.array([0.1]);
    using H = np.array([1, 0, 0, 0]);
    using onesMask = np.ones([4]);
    using initState = np.zeros([4]);
    using initCov = np.ones([4]);
    using obs = np.ones([200, 1]);

    type Carry = { state: np.Array; covDiag: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const { state, covDiag } = carry;
      const predCov = covDiag.add(processNoise);
      const innovation = x.sub(np.sum(state.mul(H)));
      const S = np.sum(predCov.mul(H).mul(H)).add(measNoise);
      const K = predCov.mul(H).div(S);
      const newState = state.add(K.mul(innovation));
      const newCov = predCov.mul(onesMask.sub(K.mul(H)));
      return [{ state: newState, covDiag: newCov }, state];
    };

    // --- Run without peephole ---
    setWasmPeephole(false);
    let stateOff: Float32Array, covOff: Float32Array, yDataOff: Float32Array;
    {
      using f = jit(() =>
        lax.scan(step, { state: initState, covDiag: initCov }, obs),
      );
      const [carry, y] = f() as [any, np.Array];
      stateOff = (await carry.state.data()) as Float32Array;
      covOff = (await carry.covDiag.data()) as Float32Array;
      yDataOff = (await y.data()) as Float32Array;
      tree.dispose(carry);
      y.dispose();
    }

    // --- Run with peephole ---
    setWasmPeephole(true);
    {
      using f = jit(() =>
        lax.scan(step, { state: initState, covDiag: initCov }, obs),
      );
      const [carry, y] = f() as [any, np.Array];
      const stateOn = (await carry.state.data()) as Float32Array;
      const covOn = (await carry.covDiag.data()) as Float32Array;
      const yDataOn = (await y.data()) as Float32Array;
      tree.dispose(carry);
      y.dispose();

      expect(Array.from(stateOn)).toEqual(Array.from(stateOff));
      expect(Array.from(covOn)).toEqual(Array.from(covOff));
      expect(Array.from(yDataOn)).toEqual(Array.from(yDataOff));
    }
  });

  test("peephole unit: local.set+get → tee", () => {
    // local.set 3 ; local.get 3 → local.tee 3
    const bytes = [0x21, 0x03, 0x20, 0x03]; // set 3, get 3
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.setGetToTee).toBe(1);
    // Should produce: local.tee 3 (0x22, 0x03)
    expect(out).toEqual([0x22, 0x03]);
  });

  test("peephole unit: i32.const 0 + i32.add → removed", () => {
    // i32.const 0 ; i32.add → both deleted
    const bytes = [0x41, 0x00, 0x6a]; // i32.const 0, i32.add
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.identity).toBe(1);
    expect(out).toEqual([]);
  });

  test("peephole unit: i32.const 1 + i32.mul → removed", () => {
    // i32.const 1 ; i32.mul → both deleted
    const bytes = [0x41, 0x01, 0x6c]; // i32.const 1, i32.mul
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.identity).toBe(1);
    expect(out).toEqual([]);
  });

  test("peephole unit: strength reduction mul→shl", () => {
    // i32.const 4 ; i32.mul → i32.const 2 ; i32.shl
    const bytes = [0x41, 0x04, 0x6c]; // i32.const 4, i32.mul
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.strengthReduction).toBe(1);
    // Should produce: i32.const 2, i32.shl
    expect(out).toEqual([0x41, 0x02, 0x74]);
  });

  test("peephole unit: tee+drop → set", () => {
    // local.tee 5 ; drop → local.set 5
    const bytes = [0x22, 0x05, 0x1a]; // tee 5, drop
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.teeDropToSet).toBe(1);
    expect(out).toEqual([0x21, 0x05]);
  });

  test("peephole unit: realistic load pattern", () => {
    // Simulates: local.get(base); local.get(gidx); i32.const(4); i32.mul; i32.add; f32.load 2 0
    const bytes = [
      0x20,
      0x00, // local.get 0
      0x20,
      0x01, // local.get 1
      0x41,
      0x04, // i32.const 4
      0x6c, // i32.mul
      0x6a, // i32.add
      0x2a,
      0x02,
      0x00, // f32.load align=2 offset=0
    ];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.strengthReduction).toBe(1);
    // i32.const 4 ; i32.mul → i32.const 2 ; i32.shl
    expect(out).toEqual([
      0x20,
      0x00, // local.get 0
      0x20,
      0x01, // local.get 1
      0x41,
      0x02, // i32.const 2 (log2(4))
      0x74, // i32.shl
      0x6a, // i32.add
      0x2a,
      0x02,
      0x00, // f32.load align=2 offset=0
    ]);
  });

  test("peephole unit: chained rewrites in one pass", () => {
    // local.set 2 ; local.get 2 ; i32.const 0 ; i32.add
    // → local.tee 2 ; (identity add removed)
    const bytes = [
      0x21,
      0x02, // local.set 2
      0x20,
      0x02, // local.get 2
      0x41,
      0x00, // i32.const 0
      0x6a, // i32.add
    ];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.setGetToTee).toBe(1);
    expect(stats.identity).toBe(1);
    expect(out).toEqual([0x22, 0x02]); // local.tee 2
  });

  test("peephole unit: zero-init removal at function start", () => {
    // i32.const 0 (0x41 0x00) ; local.set 3 (0x21 0x03) ; i32.const 0 ; local.set 4
    // followed by actual code: local.get 0 (0x20 0x00)
    const bytes = [0x41, 0x00, 0x21, 0x03, 0x41, 0x00, 0x21, 0x04, 0x20, 0x00];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.zeroInit).toBe(2);
    // Only the local.get 0 should remain
    expect(out).toEqual([0x20, 0x00]);
  });

  test("peephole unit: zero-init stops at non-zero-init instruction", () => {
    // i32.const 0 ; local.set 3 ; local.get 0 (non-zero-init) ; i32.const 0 ; local.set 4
    // Only the first pair should be removed
    const bytes = [0x41, 0x00, 0x21, 0x03, 0x20, 0x00, 0x41, 0x00, 0x21, 0x04];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.zeroInit).toBe(1);
    // local.get 0 ; i32.const 0 ; local.set 4 remain
    expect(out).toEqual([0x20, 0x00, 0x41, 0x00, 0x21, 0x04]);
  });

  test("peephole unit: zero-init skips non-zero const", () => {
    // i32.const 5 ; local.set 3 — NOT a zero-init, should not be removed
    const bytes = [0x41, 0x05, 0x21, 0x03, 0x20, 0x00];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.zeroInit).toBe(0);
    expect(out).toEqual(bytes);
  });

  test("peephole unit: unknown opcode returns bytes unchanged", () => {
    // 0xfe is not a recognized opcode
    const bytes = [0x20, 0x00, 0xfe, 0x6a];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    // Should return original bytes unchanged (parse failed → fail-safe)
    expect(out).toEqual(bytes);
  });

  test("elementwise chain correctness with peephole", async () => {
    defaultDevice("wasm");

    using a = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
    using b = np.array([2, 3, 4, 5, 6, 7, 8, 9]);

    // Without peephole
    setWasmPeephole(false);
    let dOff: Float32Array;
    {
      using f = jit((x: np.Array, y: np.Array) => x.mul(y).add(x).sub(y));
      using r = f(a, b);
      dOff = (await r.data()) as Float32Array;
    }

    // With peephole
    setWasmPeephole(true);
    {
      using f = jit((x: np.Array, y: np.Array) => x.mul(y).add(x).sub(y));
      using r = f(a, b);
      const dOn = (await r.data()) as Float32Array;
      expect(Array.from(dOn)).toEqual(Array.from(dOff));
    }
  });

  test("routine body (sort in scan) correctness with peephole", async () => {
    defaultDevice("wasm");

    const sortStep = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const sorted = np.sort(x);
      return [sorted, sorted];
    };

    using xs = np.array([
      [3, 1, 2],
      [6, 4, 5],
      [9, 7, 8],
    ]);
    using init = np.zeros([3]);

    // Without peephole
    setWasmPeephole(false);
    let cDataOff: Float32Array, yDataOff: Float32Array;
    {
      using f = jit((xs: np.Array) => lax.scan(sortStep, init, xs));
      const [c, y] = f(xs) as [np.Array, np.Array];
      cDataOff = (await c.data()) as Float32Array;
      yDataOff = (await y.data()) as Float32Array;
      c.dispose();
      y.dispose();
    }

    // With peephole
    setWasmPeephole(true);
    {
      using f = jit((xs: np.Array) => lax.scan(sortStep, init, xs));
      const [c, y] = f(xs) as [np.Array, np.Array];
      const cDataOn = (await c.data()) as Float32Array;
      const yDataOn = (await y.data()) as Float32Array;
      c.dispose();
      y.dispose();

      expect(Array.from(cDataOn)).toEqual(Array.from(cDataOff));
      expect(Array.from(yDataOn)).toEqual(Array.from(yDataOff));
    }
  });

  test("reduction body correctness with peephole", async () => {
    defaultDevice("wasm");
    using xs = np.ones([20, 16]);
    await blockUntilReady(xs);
    using init = np.zeros([]);

    // Without peephole
    setWasmPeephole(false);
    let cDataOff: Float32Array, yDataOff: Float32Array;
    {
      using f = jit((xs: np.Array) =>
        lax.scan(
          (carry: np.Array, x: np.Array) => {
            const s = carry.add(np.sum(x));
            return [s, s];
          },
          init,
          xs,
        ),
      );
      const [c, y] = f(xs) as [np.Array, np.Array];
      cDataOff = (await c.data()) as Float32Array;
      yDataOff = (await y.data()) as Float32Array;
      c.dispose();
      y.dispose();
    }

    // With peephole
    setWasmPeephole(true);
    {
      using f = jit((xs: np.Array) =>
        lax.scan(
          (carry: np.Array, x: np.Array) => {
            const s = carry.add(np.sum(x));
            return [s, s];
          },
          init,
          xs,
        ),
      );
      const [c, y] = f(xs) as [np.Array, np.Array];
      const cDataOn = (await c.data()) as Float32Array;
      const yDataOn = (await y.data()) as Float32Array;
      c.dispose();
      y.dispose();

      expect(Array.from(cDataOn)).toEqual(Array.from(cDataOff));
      expect(Array.from(yDataOn)).toEqual(Array.from(yDataOff));
    }
  });

  // --------------------------------------------------------------------------
  // New Binaryen-inspired rules (rules 5-7)
  // --------------------------------------------------------------------------

  test("peephole unit: i32.const 0 + i32.sub → removed", () => {
    const bytes = [0x41, 0x00, 0x6b]; // i32.const 0, i32.sub
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.identity).toBe(1);
    expect(out).toEqual([]);
  });

  test("peephole unit: i32.const -1 + i32.and → removed", () => {
    // -1 in signed LEB128: [0x7f]
    const bytes = [0x41, 0x7f, 0x71]; // i32.const -1, i32.and
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.identity).toBe(1);
    expect(out).toEqual([]);
  });

  test("peephole unit: i32.const 0 + i32.shl → removed", () => {
    const bytes = [0x41, 0x00, 0x74]; // i32.const 0, i32.shl
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.identity).toBe(1);
    expect(out).toEqual([]);
  });

  test("peephole unit: dead set elimination", () => {
    // local.set 2 ; local.set 2 → drop ; local.set 2
    const bytes = [0x21, 0x02, 0x21, 0x02]; // set 2, set 2
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.deadSet).toBe(1);
    expect(out).toEqual([0x1a, 0x21, 0x02]); // drop, set 2
  });

  test("peephole unit: i32 constant folding", () => {
    // i32.const 3 ; i32.const 5 ; i32.add → i32.const 8
    const bytes = [0x41, 0x03, 0x41, 0x05, 0x6a];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.constFold).toBe(1);
    expect(out).toEqual([0x41, 0x08]);
  });

  test("peephole unit: offset absorption into f32.load", () => {
    // local.get 0 ; i32.const 8 ; i32.add ; f32.load align=2 offset=0
    // → local.get 0 ; f32.load align=2 offset=8
    const bytes = [
      0x20,
      0x00, // local.get 0
      0x41,
      0x08, // i32.const 8
      0x6a, // i32.add
      0x2a,
      0x02,
      0x00, // f32.load align=2 offset=0
    ];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.offsetAbsorb).toBe(1);
    expect(out).toEqual([
      0x20,
      0x00, // local.get 0
      0x2a,
      0x02,
      0x08, // f32.load align=2 offset=8
    ]);
  });

  test("peephole unit: cascading constant fold + offset absorption", () => {
    // local.get 0 ; const 3 ; const 4 ; mul ; add ; f32.load align=2 offset=0
    // Pass 1: const 3; const 4; mul → const 12 (constant fold subsumes strength reduction)
    // Pass 2: const 12; add; f32.load offset=0 → f32.load offset=12 (offset absorption)
    const bytes = [
      0x20,
      0x00, // local.get 0
      0x41,
      0x03, // i32.const 3
      0x41,
      0x04, // i32.const 4
      0x6c, // i32.mul
      0x6a, // i32.add
      0x2a,
      0x02,
      0x00, // f32.load align=2 offset=0
    ];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.constFold).toBe(1);
    expect(stats.offsetAbsorb).toBe(1);
    expect(out).toEqual([
      0x20,
      0x00, // local.get 0
      0x2a,
      0x02,
      0x0c, // f32.load align=2 offset=12
    ]);
  });

  // --- Rule 9: comparison simplification ---

  test("peephole unit: i32.const 1 ; i32.lt_u → i32.eqz", () => {
    // x ; i32.const 1 ; i32.lt_u → x ; i32.eqz
    const bytes = [
      0x20,
      0x00, // local.get 0
      0x41,
      0x01, // i32.const 1
      0x49, // i32.lt_u
    ];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.compSimplify).toBe(1);
    expect(out).toEqual([
      0x20,
      0x00, // local.get 0
      0x45, // i32.eqz
    ]);
  });

  test("peephole unit: i32.const 0 ; i32.eq → i32.eqz", () => {
    const bytes = [
      0x20,
      0x00, // local.get 0
      0x41,
      0x00, // i32.const 0
      0x46, // i32.eq
    ];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.compSimplify).toBe(1);
    expect(out).toEqual([
      0x20,
      0x00, // local.get 0
      0x45, // i32.eqz
    ]);
  });

  test("peephole unit: offset absorption does NOT apply to stores", () => {
    // For i32.store, the i32.add modifies the VALUE (top of stack), not the address.
    // i32.store offset=N adds N to the ADDRESS, which is semantically different.
    // addr ; value ; i32.const 10 ; i32.add ; i32.store
    // must NOT become: addr ; value ; i32.store offset=10
    const bytes = [
      0x20,
      0x00, // local.get 0 (address)
      0x20,
      0x01, // local.get 1 (value)
      0x41,
      0x0a, // i32.const 10
      0x6a, // i32.add (value + 10)
      0x36,
      0x02,
      0x00, // i32.store align=2 offset=0
    ];
    const stats = newPeepholeStats();
    const out = optimizeFunctionBody(bytes, stats);
    expect(stats.offsetAbsorb).toBe(0); // Must NOT absorb into store
    // Bytes should be unchanged (no offset absorption on stores)
    expect(out).toEqual(bytes);
  });
});
