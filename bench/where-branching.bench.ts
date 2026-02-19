/**
 * Benchmark: WASM Where branching optimization.
 *
 * Compares performance of `np.where` with expensive arms (exp, log, sin)
 * on the WASM backend, which now uses `if/else` true branching instead of
 * the branchless `select` instruction when one arm is costly.
 *
 * Run:
 *   pnpm exec vitest bench bench/where-branching.bench.ts
 */
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  nn,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init("wasm");

suite.skipIf(!devices.includes("wasm"))("wasm where branching", () => {
  defaultDevice("wasm");

  // ── ELU: where(x > 0, x, alpha * (exp(x) - 1)) ──────────────────────
  // Expensive arm contains exp(). Branching should help when most values > 0.
  {
    const N = 4096;
    const x = np.array(
      Array.from({ length: N }, (_, i) => (i / N) * 8 - 2), // range [-2, 6)
    );
    const eluJit = jit((x: np.Array) => nn.elu(x));
    // Warmup
    const warmup = eluJit(x);
    warmup.dispose();

    afterAll(() => {
      eluJit.dispose();
      x.dispose();
    });

    bench(`elu ${N} elements (75% positive)`, () => {
      const r = eluJit(x);
      r.dispose();
    });
  }

  // ── SELU ──────────────────────────────────────────────────────────────
  {
    const N = 4096;
    const x = np.array(
      Array.from({ length: N }, (_, i) => (i / N) * 8 - 2),
    );
    const seluJit = jit((x: np.Array) => nn.selu(x));
    const warmup = seluJit(x);
    warmup.dispose();

    afterAll(() => {
      seluJit.dispose();
      x.dispose();
    });

    bench(`selu ${N} elements (75% positive)`, () => {
      const r = seluJit(x);
      r.dispose();
    });
  }

  // ── Custom: where(cond, x, exp(x)) ── direct pattern ─────────────────
  {
    const N = 4096;
    // 50/50 mix
    const x = np.array(
      Array.from({ length: N }, (_, i) => (i % 2 === 0 ? 1.0 : -1.0)),
    );
    const f = jit((x: np.Array) => {
      using cond = x.greater(0);
      return np.where(cond, x, np.exp(x));
    });
    const warmup = f(x);
    warmup.dispose();

    afterAll(() => {
      f.dispose();
      x.dispose();
    });

    bench(`where(cond, x, exp(x)) ${N} elements (50% true)`, () => {
      const r = f(x);
      r.dispose();
    });
  }

  // ── Same pattern but mostly-true (90%) ────────────────────────────────
  {
    const N = 4096;
    const x = np.array(
      Array.from({ length: N }, (_, i) => (i < N * 0.9 ? 1.0 : -1.0)),
    );
    const f = jit((x: np.Array) => {
      using cond = x.greater(0);
      return np.where(cond, x, np.exp(x));
    });
    const warmup = f(x);
    warmup.dispose();

    afterAll(() => {
      f.dispose();
      x.dispose();
    });

    bench(`where(cond, x, exp(x)) ${N} elements (90% true)`, () => {
      const r = f(x);
      r.dispose();
    });
  }

  // ── Both arms expensive: where(cond, exp(x), sin(x)) ─────────────────
  {
    const N = 4096;
    const x = np.array(
      Array.from({ length: N }, (_, i) => (i % 2 === 0 ? 1.0 : -1.0)),
    );
    const f = jit((x: np.Array) => {
      using cond = x.greater(0);
      return np.where(cond, np.exp(x), np.sin(x));
    });
    const warmup = f(x);
    warmup.dispose();

    afterAll(() => {
      f.dispose();
      x.dispose();
    });

    bench(`where(cond, exp(x), sin(x)) ${N} elements (50%)`, () => {
      const r = f(x);
      r.dispose();
    });
  }

  // ── Control: cheap arms (relu) — should NOT benefit from branching ────
  {
    const N = 4096;
    const x = np.array(
      Array.from({ length: N }, (_, i) => (i / N) * 8 - 2),
    );
    const reluJit = jit((x: np.Array) => nn.relu(x));
    const warmup = reluJit(x);
    warmup.dispose();

    afterAll(() => {
      reluJit.dispose();
      x.dispose();
    });

    bench(`relu ${N} elements (control, cheap arms)`, () => {
      const r = reluJit(x);
      r.dispose();
    });
  }
});
