/**
 * Minimal reproducer for the jit + scan + valueAndGrad leak.
 *
 * Run: pnpm vitest run test/leak-repro.test.ts
 *
 * Downstream issue: diffSVMC/issues/jax-js-nonconsuming-leak-jit-scan-optax.md
 */
import {
  clearCaches,
  init,
  jit,
  lax,
  numpy as np,
  tree,
  valueAndGrad,
} from "@hamk-uas/jax-js-nonconsuming";
import {
  adam,
  applyUpdates,
  chain,
  clipByGlobalNorm,
  type OptState,
} from "@hamk-uas/jax-js-nonconsuming-optax";
import { describe, expect, test } from "vitest";

await init();

// grad(sum(x*x)) = 2*x, so x - lr*2*x = x*(1-2*lr). With lr=0.01: x*0.98.
// After 10 steps: x * 0.98^10.

describe("jit + scan + valueAndGrad leak", () => {
  // Test 1: The full downstream pattern — jit(scan(valueAndGrad + optax + bestTracking))
  test("jit wrapping scan with valueAndGrad + optax + best tracking", () => {
    const objective = (params: np.Array) => np.sum(params.mul(params));
    const OPTIMIZER = chain(clipByGlobalNorm(10.0), adam(0.05));

    type Carry = {
      params: np.Array;
      optState: OptState;
      bestParams: np.Array;
      bestLoss: np.Array;
    };

    const impl = (x: np.Array) => {
      const initCarry: Carry = {
        params: x,
        optState: OPTIMIZER.init(x),
        bestParams: np.array([4.0, 1.0]),
        bestLoss: np.array(Infinity),
      };

      const step = (carry: Carry, _: null): [Carry, null] => {
        const [loss, grads] = valueAndGrad(objective)(carry.params);
        const better = loss.less(carry.bestLoss);
        const bestParams = np.where(better, carry.params, carry.bestParams);
        const bestLoss = np.where(better, loss, carry.bestLoss);
        loss.dispose();
        better.dispose();
        const [updates, optState] = OPTIMIZER.update(
          grads,
          carry.optState,
          carry.params,
        );
        grads.dispose();
        const params = applyUpdates(carry.params, updates);
        tree.dispose(updates);
        return [{ params, optState, bestParams, bestLoss }, null];
      };

      const [finalCarry] = lax.scan(step, initCarry, null, { length: 20 });
      using _fc = tree.makeDisposable(finalCarry as Carry);
      const result = _fc.bestParams.add(np.array(0)); // clone
      return result;
    };

    const f = jit(impl);
    const x = np.array([3.0, 4.0]);
    const result = f(x);
    result.dispose();
    x.dispose();
    f.dispose();
    clearCaches();
  });

  // Test 1b: Same but without jit — pure eager
  test("scan with valueAndGrad + optax + best tracking (no jit)", () => {
    const objective = (params: np.Array) => np.sum(params.mul(params));
    const OPTIMIZER = chain(clipByGlobalNorm(10.0), adam(0.05));

    type Carry = {
      params: np.Array;
      optState: OptState;
      bestParams: np.Array;
      bestLoss: np.Array;
    };

    const x = np.array([3.0, 4.0]);
    const initBestParams = np.array([4.0, 1.0]);
    const initBestLoss = np.array(Infinity);
    const initOptState = OPTIMIZER.init(x);
    const initCarry: Carry = {
      params: x,
      optState: initOptState,
      bestParams: initBestParams,
      bestLoss: initBestLoss,
    };

    const step = (carry: Carry, _: null): [Carry, null] => {
      const [loss, grads] = valueAndGrad(objective)(carry.params);
      const better = loss.less(carry.bestLoss);
      const bestParams = np.where(better, carry.params, carry.bestParams);
      const bestLoss = np.where(better, loss, carry.bestLoss);
      loss.dispose();
      better.dispose();
      const [updates, optState] = OPTIMIZER.update(
        grads,
        carry.optState,
        carry.params,
      );
      grads.dispose();
      const params = applyUpdates(carry.params, updates);
      tree.dispose(updates);
      return [{ params, optState, bestParams, bestLoss }, null];
    };

    const [finalCarry] = lax.scan(step, initCarry, null, { length: 20 });
    using _fc = tree.makeDisposable(finalCarry as Carry);
    // Dispose initial carry arrays (non-consuming: scan doesn't free them)
    tree.dispose(initOptState);
    initBestParams.dispose();
    initBestLoss.dispose();
    x.dispose();
  });

  // Test 1c: Closer to downstream — multiple closed-over scalar args + argnums
  test("jit wrapping scan + valueAndGrad with closed-over args", () => {
    const OPTIMIZER = chain(clipByGlobalNorm(10.0), adam(0.05));

    type Carry = {
      params: np.Array;
      optState: OptState;
      bestParams: np.Array;
      bestLoss: np.Array;
    };

    // Simulate downstream: objective differentiates only w.r.t. first arg,
    // other args are closed-over scalars
    const impl = (
      x: np.Array,
      a: np.Array,
      b: np.Array,
      c: np.Array,
      d: np.Array,
    ) => {
      const objective = (
        params: np.Array,
        a2: np.Array,
        b2: np.Array,
        c2: np.Array,
        d2: np.Array,
      ) => {
        const t1 = params.mul(params);
        const t2 = t1.mul(a2);
        t1.dispose();
        const t3 = t2.add(b2.mul(c2));
        t2.dispose();
        const t4 = t3.sub(d2);
        t3.dispose();
        const result = np.sum(t4);
        t4.dispose();
        return result;
      };

      const initCarry: Carry = {
        params: x,
        optState: OPTIMIZER.init(x),
        bestParams: np.array([4.0, 1.0]),
        bestLoss: np.array(Infinity),
      };

      const step = (carry: Carry, _: null): [Carry, null] => {
        const [loss, grads] = valueAndGrad(objective, { argnums: 0 })(
          carry.params,
          a,
          b,
          c,
          d,
        );
        const better = loss.less(carry.bestLoss);
        const bestParams = np.where(better, carry.params, carry.bestParams);
        const bestLoss = np.where(better, loss, carry.bestLoss);
        loss.dispose();
        better.dispose();
        const [updates, optState] = OPTIMIZER.update(
          grads,
          carry.optState,
          carry.params,
        );
        grads.dispose();
        const params = applyUpdates(carry.params, updates);
        tree.dispose(updates);
        return [{ params, optState, bestParams, bestLoss }, null];
      };

      const [finalCarry] = lax.scan(step, initCarry, null, { length: 20 });
      using _fc = tree.makeDisposable(finalCarry as Carry);
      const result = _fc.bestParams.add(np.array(0));
      return result;
    };

    const f = jit(impl);
    const x = np.array([3.0, 4.0]);
    const a = np.array(1.0);
    const b = np.array(2.0);
    const c = np.array(0.5);
    const d = np.array(1.0);
    const result = f(x, a, b, c, d);
    result.dispose();
    x.dispose();
    a.dispose();
    b.dispose();
    c.dispose();
    d.dispose();
    f.dispose();
    clearCaches();
  });

  // Test 2: The basic pattern — jit(scan(valueAndGrad)) without optax
  test("jit wrapping scan with valueAndGrad body", () => {
    const objective = (params: np.Array) => np.sum(params.mul(params));

    const step = (carry: np.Array, _: null): [np.Array, null] => {
      const [_loss, grads] = valueAndGrad(objective)(carry);
      _loss.dispose();
      const newCarry = carry.sub(grads.mul(0.01));
      grads.dispose();
      return [newCarry, null];
    };

    const impl = (x: np.Array) => {
      const [finalCarry] = lax.scan(step, x, null, { length: 10 });
      return finalCarry;
    };

    const f = jit(impl);
    const x = np.array([3.0, 4.0]);
    const result = f(x);
    expect(result.js()[0]).toBeCloseTo(3 * 0.98 ** 10, 2);
    result.dispose();
    x.dispose();
    f.dispose();
    clearCaches();
  });

  // Test 3: scan(valueAndGrad) without jit — isolate scan body tracing
  test("scan with valueAndGrad body (no jit)", () => {
    const objective = (params: np.Array) => np.sum(params.mul(params));

    const step = (carry: np.Array, _: null): [np.Array, null] => {
      const [_loss, grads] = valueAndGrad(objective)(carry);
      _loss.dispose();
      const newCarry = carry.sub(grads.mul(0.01));
      grads.dispose();
      return [newCarry, null];
    };

    const x = np.array([3.0, 4.0]);
    const [result] = lax.scan(step, x, null, { length: 10 });
    using _result = result;
    expect((result as np.Array).js()[0]).toBeCloseTo(3 * 0.98 ** 10, 2);
    x.dispose();
  });

  // Test 4: jit(valueAndGrad) without scan — isolate JVP inside JIT
  test("jit wrapping valueAndGrad (no scan)", () => {
    const objective = (params: np.Array) => np.sum(params.mul(params));

    const impl = (x: np.Array) => {
      const [loss, grads] = valueAndGrad(objective)(x);
      return [loss, grads];
    };

    const f = jit(impl);
    const x = np.array([3.0, 4.0]);
    const [loss, grads] = f(x) as [np.Array, np.Array];
    expect(loss.js()).toBeCloseTo(25, 1);
    loss.dispose();
    grads.dispose();
    x.dispose();
    f.dispose();
    clearCaches();
  });
});
