/**
 * evalJaxpr identity-return disposal tests.
 *
 * Primitives like stopGradient or convert_element_type may return an input
 * Array by identity.  evalJaxpr's usage-based early disposal must account
 * for this: if two Vars share the same backing Array (via identity-returning
 * primitive), consumeRead on one must not free the Array while the other Var
 * still references it.
 *
 * Before the fix, these patterns caused UseAfterFreeError because consumeRead
 * on the input Var disposed the shared Array, then consumeRead on the output
 * Var tried to dispose it again.
 */
import {
  grad,
  hessian,
  type JsTree,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, it } from "vitest";

describe("evalJaxpr identity-return disposal", () => {
  it("stopGradient inside foriLoop body", () => {
    const body = (_i: np.Array, x: np.Array): np.Array => {
      const y = lax.stopGradient(x);
      return y.add(1);
    };
    using init = np.array(1.0);
    using result = lax.foriLoop(0, 3, body, init) as np.Array;
    expect(result.js()).toBe(4);
  });

  it("stopGradient inside scan body", () => {
    const scanBody = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
      const y = lax.stopGradient(carry);
      const out = y.add(1);
      return [out, out];
    };
    using init = np.array(0.0);
    using xs = np.array([0, 0, 0]);
    const [result, ys] = lax.scan(
      scanBody as unknown as (
        c: JsTree<np.Array>,
        x: JsTree<np.Array>,
      ) => [JsTree<np.Array>, JsTree<np.Array>],
      init as unknown as JsTree<np.Array>,
      xs as unknown as JsTree<np.Array>,
      { length: 3 },
    ) as unknown as [np.Array, np.Array];
    expect(result.js()).toBe(3);
    result.dispose();
    ys.dispose();
  });

  it("grad inside foriLoop body", () => {
    using alpha = np.array(2.5);
    const body = (_i: np.Array, x: np.Array): np.Array => {
      const lossFn = (v: np.Array): np.Array => v.mul(alpha).mul(v);
      const g = grad(lossFn)(x);
      using step = g.mul(0.01);
      return x.sub(step);
    };
    using init = np.array(1.0);
    using result = lax.foriLoop(0, 2, body, init) as np.Array;
    expect(Number.isFinite(result.js() as number)).toBe(true);
  });

  it("grad inside scan body", () => {
    using alpha = np.array(2.5);
    const scanBody = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
      const lossFn = (v: np.Array): np.Array => v.mul(alpha).mul(v);
      const g = grad(lossFn)(carry);
      using step = g.mul(0.01);
      const out = carry.sub(step);
      return [out, out];
    };
    using init = np.array(1.0);
    using xs = np.array([0.0]);
    const [result, ys] = lax.scan(
      scanBody as unknown as (
        c: JsTree<np.Array>,
        x: JsTree<np.Array>,
      ) => [JsTree<np.Array>, JsTree<np.Array>],
      init as unknown as JsTree<np.Array>,
      xs as unknown as JsTree<np.Array>,
      { length: 1 },
    ) as unknown as [np.Array, np.Array];
    expect(Number.isFinite(result.js() as number)).toBe(true);
    result.dispose();
    ys.dispose();
  });

  it("scan → foriLoop → grad composition", () => {
    using alpha = np.array(2.5);
    const scanBody = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
      const hourlyBody = (_i: np.Array, x: np.Array): np.Array => {
        const lossFn = (v: np.Array): np.Array => v.mul(alpha).mul(v);
        const g = grad(lossFn)(x);
        using step = g.mul(0.01);
        return x.sub(step);
      };
      const inner = lax.foriLoop(0, 2, hourlyBody, carry) as np.Array;
      return [inner, inner];
    };
    using init = np.array(1.0);
    using xs = np.array([0.0]);
    const [result, ys] = lax.scan(
      scanBody as unknown as (
        c: JsTree<np.Array>,
        x: JsTree<np.Array>,
      ) => [JsTree<np.Array>, JsTree<np.Array>],
      init as unknown as JsTree<np.Array>,
      xs as unknown as JsTree<np.Array>,
      { length: 1 },
    ) as unknown as [np.Array, np.Array];
    expect(Number.isFinite(result.js() as number)).toBe(true);
    result.dispose();
    ys.dispose();
  });

  it("np.linalg.solve inside foriLoop body", () => {
    const body = (_i: np.Array, x: np.Array): np.Array => {
      using a = np.array([
        [2, 0],
        [0, 2],
      ]);
      using step = np.linalg.solve(a, x);
      using scaled = step.mul(0.1);
      return x.sub(scaled);
    };
    using init = np.array([4.0, 3.0]);
    using result = lax.foriLoop(0, 2, body, init) as np.Array;
    const js = result.js() as number[];
    expect(Number.isFinite(js[0])).toBe(true);
    expect(Number.isFinite(js[1])).toBe(true);
  });

  it("hessian + eye + solve inside foriLoop (Newton solver pattern)", () => {
    const objFn = (params: np.Array): np.Array => {
      const a = lax.dynamicIndexInDim(params, 0, 0, false);
      const b = lax.dynamicIndexInDim(params, 1, 0, false);
      return a.mul(a).add(b.mul(b));
    };
    const body = (_i: np.Array, x: np.Array): np.Array => {
      const g = grad(objFn)(x);
      using h = hessian(objFn)(x);
      using eye = np.eye(2, { dtype: x.dtype });
      using reg = eye.mul(0.01);
      using hReg = h.add(reg);
      using step = np.linalg.solve(hReg, g);
      using scaled = step.mul(0.1);
      return x.sub(scaled);
    };
    using init = np.array([4.0, 3.0]);
    using result = lax.foriLoop(0, 3, body, init) as np.Array;
    const js = result.js() as number[];
    // Should converge towards [0, 0]
    expect(Math.abs(js[0])).toBeLessThan(4.0);
    expect(Math.abs(js[1])).toBeLessThan(3.0);
  });
});
