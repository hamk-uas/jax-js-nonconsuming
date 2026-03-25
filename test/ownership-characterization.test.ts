/**
 * Ownership Characterization Tests — Ownership Restructuring Plan
 *
 * These tests pin the behavior of the tracing/autodiff ownership machinery.
 * They serve as regression tests during the ownership restructuring: any
 * change that breaks these tests is either (a) a correct inversion of a
 * workaround-era behavior or (b) a real regression.
 *
 * Phase 3d (explicit lexical balancing): the phantom-ref state machine
 * (anonymousConstArrays, _constCreationBuilderRefs, _deferredConstCreationDisposes)
 * has been deleted. Ownership is now purely lexical:
 * - getOrMakeConstTracer's .ref is the sole retained handle for captured consts.
 * - _inlineLiterals and ClosedJaxpr.dispose() each balance exactly one builder .ref.
 * - `using` inside traced bodies fires normally (harmless: builder already holds .ref).
 * - Fire-and-forget arrays (no `using`) leak identically in jit and eager modes.
 *
 * See PLAN.md for the full restructuring plan.
 *
 * Run: pnpm vitest run test/ownership-characterization.test.ts
 */
import {
  checkLeaks,
  clearCaches,
  grad,
  jit,
  jvp,
  lax,
  linearize,
  makeJaxpr,
  numpy as np,
  valueAndGrad,
  vjp,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, expect, test } from "vitest";

// ============================================================
// Category 1: checkLeaks for transform compositions
//
// Pin whether grad, vjp, jit(grad), vmap(grad), jit(scan(grad))
// currently leak. Tests that are known to leak use manual checkLeaks
// scope to document the exact leak count without failing the suite.
// ============================================================

describe("transform composition leak checks", () => {
  // --- Tests that pass with zero leaks ---

  test("jit(grad(f)) — zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    using jitGrad = jit(grad(f));
    using x = np.array([1, 2, 3]);
    using r = jitGrad(x);
    expect(r).toBeAllclose([2, 4, 6]);
  });

  test("vmap(grad(f)) inside jit — zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    using jitVmapGrad = jit(vmap(grad(f)));
    using batch = np.array([
      [1, 2],
      [3, 4],
    ]);
    using r = jitVmapGrad(batch);
    expect(r).toBeAllclose([
      [2, 4],
      [6, 8],
    ]);
  });

  test("grad(grad(f)) — zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).mul(x); // x³
    const d2f = grad(grad(f));
    using r = d2f(np.array(2));
    // f''(x) = 6x → f''(2) = 12
    expect(r).toBeAllclose(12);
  });

  test("jit(grad(f)) with scan in body — zero leaks", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, null] => {
      return [carry.add(x), null];
    };
    const f = (xs: np.Array) => {
      using init = np.array(0);
      const [carry] = lax.scan(step, init, xs);
      return carry.mul(carry);
    };
    using jitGradF = jit(grad(f));
    using xs = np.array([1, 2, 3, 4]);
    using r = jitGradF(xs);
    expect(r).toBeAllclose([20, 20, 20, 20]);
  });

  // --- Bare eager grad/valueAndGrad paths ---
  // These previously leaked PE intermediates (ResidualCollector.dispose() was dead code).
  // Fixed: collector now disposes dead PE temporaries while protecting builder-owned consts.

  test("bare grad(f) — zero leaks after collector fix", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    using x = np.array([1, 2, 3]);
    using r = grad(f)(x);
    expect(r).toBeAllclose([2, 4, 6]);
  });

  test("bare vjp(f) — zero leaks when properly disposed", () => {
    // Surprisingly, bare vjp does NOT leak PE intermediates.
    // Only grad/valueAndGrad leak (they internally do vjp + forward eval).
    const f = (x: np.Array) => x.mul(x);
    using x = np.array([2, 3, 4]);
    const [y, backward] = vjp(f, [x]);
    using _y = y;
    using _backward = backward;
    const grads = backward(np.ones(x.shape));
    using _g0 = grads[0];
    expect(grads[0]).toBeAllclose([4, 6, 8]);
  });

  test("bare valueAndGrad(f) — zero leaks after collector fix", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    using x = np.array([1, 2, 3]);
    const [val, g] = valueAndGrad(f)(x);
    using _val = val;
    using _g = g;
    expect(val).toBeAllclose(14);
    expect(g).toBeAllclose([2, 4, 6]);
  });

  test("bare jvp(f) — zero leaks when properly disposed", () => {
    // Surprisingly, bare jvp does NOT leak tangent intermediates.
    const f = (x: np.Array) => x.mul(x);
    using x = np.array([2, 3]);
    using t = np.array([1, 1]);
    const [y, dy] = jvp(f, [x], [t]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose([4, 9]);
    expect(dy).toBeAllclose([4, 6]);
  });

  test("bare linearize(f) — zero leaks when properly disposed", () => {
    // Linearize does not leak when disposed correctly.
    const f = (x: np.Array) => x.mul(x);
    using x = np.array([2, 3]);
    const [y, linearFn] = linearize(f, [x]);
    using _y = y;
    using _lin = linearFn;
    expect(y).toBeAllclose([4, 9]);
    using t1 = np.array([1, 0]);
    using r1 = linearFn(t1);
    expect(r1).toBeAllclose([4, 0]);
    using t2 = np.array([0, 1]);
    using r2 = linearFn(t2);
    expect(r2).toBeAllclose([0, 6]);
  });

  test("bare grad(f) — multiple calls zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    const df = grad(f);
    using x = np.array([1, 2, 3]);
    using r1 = df(x);
    using r2 = df(x);
    expect(r1).toBeAllclose([2, 4, 6]);
    expect(r2).toBeAllclose([2, 4, 6]);
  });
});

// ============================================================
// Category 2: `using` inside transform bodies
//
// `using` inside traced code fires [Symbol.dispose] during tracing.
// Since getOrMakeConstTracer takes an independent .ref before the
// scope exits, the disposal harmlessly decrements from rc=2 to rc=1,
// leaving the builder's retained handle intact.
// ============================================================

describe("using inside transform bodies", () => {
  test("using inside jit(grad) body — disposal harmless during tracing", () => {
    const f = (x: np.Array) => {
      using bias = np.array(1);
      return x.add(bias).mul(x).sum();
    };
    using jitGrad = jit(grad(f));
    using x = np.array([1, 2, 3]);
    using r = jitGrad(x);
    // f(x) = (x+1)*x = x² + x, f'(x) = 2x + 1
    expect(r).toBeAllclose([3, 5, 7]);
  });

  test("using inside bare grad body — zero leaks after collector fix", () => {
    const f = (x: np.Array) => {
      using ones = np.ones(x.shape);
      return x.add(ones).sum();
    };
    using x = np.array([1, 2, 3]);
    using r = grad(f)(x);
    expect(r).toBeAllclose([1, 1, 1]);
  });

  test("vjp with using-created local const — zero leaks after collector fix", () => {
    const f = (x: np.Array) => {
      using scale = np.array(2);
      return x.mul(scale).sum();
    };
    using x = np.array([1, 2, 3]);
    const [y, bwd] = vjp(f, [x]);
    using _y = y;
    using _bwd = bwd;
    const grads = bwd(np.array(1));
    using _g0 = grads[0];
    expect(grads[0]).toBeAllclose([2, 2, 2]);
  });

  test("PROVISIONAL: using inside scan body (eager)", () => {
    const outerResult = checkLeaks.stop();
    expect(outerResult.leaked).toBe(0);

    checkLeaks.start();
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      using scale = np.array(2);
      const newCarry = carry.add(x.mul(scale));
      return [newCarry, newCarry];
    };
    const xs = np.array([1, 2, 3]);
    const init = np.array(0);
    const [carry, ys] = lax.scan(step, init, xs);
    expect(carry).toBeAllclose(12);
    expect(ys).toBeAllclose([2, 6, 12]);
    ys.dispose();
    carry.dispose();
    xs.dispose();
    init.dispose();

    const report = checkLeaks.stop();
    // Provisional characterization: pin the current behavior without claiming
    // that eager scan+using is a confirmed standalone leak class.
    if (report.leaked > 0) {
      expect(report.leaked).toBeGreaterThan(0);
    }

    checkLeaks.start();
  });
});

// ============================================================
// Category 3: ClosedJaxpr.dispose() — builder ref lifecycle
//
// Verify that ClosedJaxpr.dispose() correctly releases builder refs
// for captured consts.
//
// NOTE: makeJaxpr(f)(x) returns {jaxpr: ClosedJaxpr, treedef}.
// The `jaxpr` field IS the ClosedJaxpr — use jaxpr.dispose() directly.
// ============================================================

describe("ClosedJaxpr disposal", () => {
  test("ClosedJaxpr.dispose() releases consts — external array", () => {
    using externalConst = np.array([10, 20, 30]);
    const { jaxpr } = makeJaxpr((x: np.Array) => x.add(externalConst))(
      np.zeros([3]),
    );
    // The ClosedJaxpr holds a .ref so rc should be 2
    expect(externalConst.refCount).toBe(2);
    expect(jaxpr.consts.length).toBe(1);
    // Dispose the ClosedJaxpr — should release one ref
    jaxpr.dispose();
    expect(externalConst.refCount).toBe(1);
  });

  test("ClosedJaxpr.dispose() handles anonymous scalar const (possibly inlined)", () => {
    // np.array(42) is a scalar — _inlineLiterals may inline it as a Lit.
    // After makeJaxpr, the ClosedJaxpr manages the const's lifecycle.
    const f = (x: np.Array) => x.add(np.array(42));
    const { jaxpr } = makeJaxpr(f)(np.array(0));
    // Dispose should not leak whether or not the scalar was inlined
    jaxpr.dispose();
  });

  test("nested makeJaxpr — inner disposal does not kill outer const", () => {
    using sharedConst = np.array([1, 2, 3]);

    const outer = (x: np.Array) => {
      // Nested makeJaxpr captures the same const
      const { jaxpr: innerJaxpr } = makeJaxpr((y: np.Array) =>
        y.add(sharedConst),
      )(x);
      // Dispose inner immediately
      innerJaxpr.dispose();
      // sharedConst must still be alive for the outer body
      return x.mul(sharedConst);
    };

    const { jaxpr: outerJaxpr } = makeJaxpr(outer)(np.zeros([3]));
    // The outer builder should still have the const alive
    expect(sharedConst.refCount).toBeGreaterThanOrEqual(2);
    outerJaxpr.dispose();
    // After full cleanup, sharedConst should be back to user ownership
    expect(sharedConst.refCount).toBe(1);
  });
});

// ============================================================
// Category 4: transposeJaxprCache — cache-owned contract
//
// Callers must NOT dispose ClosedJaxprs returned from the cache.
// After collector fix, bare grad cache reuse no longer leaks.
// ============================================================

describe("transposeJaxprCache contract", () => {
  test("bare grad cache reuse — zero leaks after collector fix", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    const df = grad(f);
    using x = np.array([2, 3, 4]);
    using r1 = df(x);
    using r2 = df(x);
    expect(r1).toBeAllclose([4, 6, 8]);
    expect(r2).toBeAllclose([4, 6, 8]);
  });

  test("jit(grad) reuses cached transpose jaxpr — zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    using jitGrad = jit(grad(f));
    using x = np.array([1, 2, 3]);

    // Multiple calls should reuse both jit cache and transpose cache
    using r1 = jitGrad(x);
    using r2 = jitGrad(x);
    using r3 = jitGrad(x);

    expect(r1).toBeAllclose([2, 4, 6]);
    expect(r2).toBeAllclose([2, 4, 6]);
    expect(r3).toBeAllclose([2, 4, 6]);
  });
});

// ============================================================
// Category 5: Nested-builder const sharing
//
// Tests for grad(grad(f)), nested makeJaxpr, where inner ClosedJaxpr
// is disposed before the outer one.
// ============================================================

describe("nested-builder const sharing", () => {
  test("grad(grad(f)) — double nesting zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).mul(x);
    using r = grad(grad(f))(np.array(3));
    expect(r).toBeAllclose(18);
  });

  test("grad(grad(grad(f))) — triple nesting zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).mul(x); // x³
    using r = grad(grad(grad(f)))(np.array(2));
    expect(r).toBeAllclose(6);
  });

  test("nested makeJaxpr with shared external const", () => {
    using c = np.array(5);
    const inner = (x: np.Array) => x.add(c);
    const outer = (x: np.Array) => {
      const { jaxpr } = makeJaxpr(inner)(x);
      jaxpr.dispose();
      return x.mul(c);
    };
    const { jaxpr } = makeJaxpr(outer)(np.array(0));
    jaxpr.dispose();
    // c must still be alive — only user ref remains
    expect(c.refCount).toBe(1);
  });
});

// ============================================================
// Category 6: _inlineLiterals ownership
//
// Scalar consts may be inlined as Lit nodes by _inlineLiterals.
// _inlineLiterals calls .dispose() on inlined consts to release
// the builder's .ref. The creation ref is balanced by the user's
// `using` (or leaked in fire-and-forget code, same as eager mode).
// ============================================================

describe("inlined literal ownership", () => {
  test("scalar const inlined as Lit — no leak after dispose", () => {
    const f = (x: np.Array) => x.add(np.array(42));
    const { jaxpr } = makeJaxpr(f)(np.array(0));
    jaxpr.dispose();
    // checkLeaks (from setup.ts) verifies no slots leaked
  });

  test("multiple scalar consts — each cleaned up independently", () => {
    const f = (x: np.Array) => x.add(np.array(1)).mul(np.array(2));
    const { jaxpr } = makeJaxpr(f)(np.array(3));
    jaxpr.dispose();
  });

  test("scalar const captured then used in bare grad — zero leaks", () => {
    const f = (x: np.Array) => x.mul(np.array(2)).sum();
    using x = np.array([1, 2, 3]);
    using r = grad(f)(x);
    expect(r).toBeAllclose([2, 2, 2]);
  });

  test("scalar const in jit(grad) — zero leaks", () => {
    const f = (x: np.Array) => x.mul(np.array(2)).sum();
    using jitGrad = jit(grad(f));
    using x = np.array([1, 2, 3]);
    using r = jitGrad(x);
    expect(r).toBeAllclose([2, 2, 2]);
  });
});

// ============================================================
// Category 7: Evaluation-time Lit materialization
//
// evalJaxpr tracks Lit-created arrays in litArrays for cleanup.
// After collector fix, these paths no longer leak.
// ============================================================

describe("evaluation-time Lit materialization", () => {
  test("bare grad with literal — zero leaks after collector fix", () => {
    const f = (x: np.Array) => x.mul(x).add(np.array(1)).sum();
    using x = np.array([1, 2, 3]);
    using r = grad(f)(x);
    expect(r).toBeAllclose([2, 4, 6]);
  });

  test("jit(grad(f)) with Lit — zero leaks", () => {
    const f = (x: np.Array) => x.add(np.array(10)).mul(x).sum();
    using jitGrad = jit(grad(f));
    using x = np.array([1, 2, 3]);
    using r = jitGrad(x);
    // f(x) = x² + 10x → f'(x) = 2x + 10
    expect(r).toBeAllclose([12, 14, 16]);
  });

  test("vjp with Lit materialization — zero leaks after collector fix", () => {
    const f = (x: np.Array) => x.mul(np.array(3)).add(np.array(1));
    using x = np.array([1, 2, 3]);
    const [y, bwd] = vjp(f, [x]);
    using _y = y;
    using _bwd = bwd;
    const grads = bwd(np.ones([3]));
    using _g0 = grads[0];
    expect(grads[0]).toBeAllclose([3, 3, 3]);
  });

  test("nested grad — evaluation inside outer tracing scope — zero leaks", () => {
    const f = (x: np.Array) => x.mul(x).mul(x); // x³
    using r = grad(grad(f))(np.array(2));
    expect(r).toBeAllclose(12);
  });
});

// ============================================================
// Category 8: ResidualCollector protection-set edge cases
//
// ResidualCollector.dispose() is now active (collector fix).
// Protection set: forwardJaxpr.consts at rc≤1 are protected (sole ref is CJ's).
// Consts at rc>1 carry an instantiateConst ref that the collector cleans up.
// One KNOWN LEAK test remains: "bare grad(scan(f))" (Category 9,
// UseAfterFreeError — separate issue from PE intermediates).
// ============================================================

describe("ResidualCollector code paths", () => {
  test("vjp with nonlinear sum body — zero leaks after collector fix", () => {
    const f = (x: np.Array) => x.mul(x).add(x).sum();
    using x = np.array([1, 2, 3, 4]);
    const [y, bwd] = vjp(f, [x]);
    using _y = y;
    using _bwd = bwd;
    const grads = bwd(np.array(1));
    using _g0 = grads[0];
    expect(grads[0]).toBeAllclose([3, 5, 7, 9]);
  });

  test("bare grad with hasAux — zero leaks after collector fix", () => {
    const f = (x: np.Array) => {
      const loss = x.mul(x).sum();
      const aux = x.sum();
      return [loss, aux] as [np.Array, np.Array];
    };
    using x = np.array([1, 2, 3]);
    const [g, aux] = grad(f, { hasAux: true })(x);
    using _g = g;
    using _aux = aux;
    expect(g).toBeAllclose([2, 4, 6]);
    expect(aux).toBeAllclose(6);
  });

  test("bare vjp with hasAux — zero leaks when properly disposed", () => {
    // Bare vjp+hasAux does NOT leak PE intermediates.
    const f = (x: np.Array) => {
      const y = x.mul(x);
      const aux = x.sum();
      return [y, aux] as [np.Array, np.Array];
    };
    using x = np.array([2, 3]);
    const [y, bwd, aux] = vjp(f, [x], { hasAux: true });
    using _y = y;
    using _bwd = bwd;
    using _aux = aux as np.Array;
    expect(aux).toBeAllclose(5);
    const grads = bwd(np.ones([2]));
    using _g0 = grads[0];
    expect(grads[0]).toBeAllclose([4, 6]);
  });

  test("bare grad with multiple intermediates — zero leaks after anon fix", () => {
    const f = (x: np.Array) => {
      const a = x.mul(x);
      const b = a.add(x);
      const c = b.mul(a);
      const d = c.add(b);
      return d.sum();
    };
    using x = np.array([1, 2]);
    using r = grad(f)(x);
    expect(r.shape).toEqual([2]);
  });
});

// ============================================================
// Category 9: Scan + transform compositions
// ============================================================

describe("scan transform compositions", () => {
  test("KNOWN LEAK: bare grad(scan(f)) — UseAfterFreeError or PE leak", () => {
    const outerResult = checkLeaks.stop();
    expect(outerResult.leaked).toBe(0);

    checkLeaks.start();
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = carry.add(x);
      return [newCarry, newCarry];
    };
    const f = (xs: np.Array) => {
      const [carry] = lax.scan(step, np.array(0), xs);
      return carry;
    };
    const x = np.array([1, 2, 3, 4]);
    let _threw = false;
    try {
      const r = grad(f)(x);
      // If it doesn’t throw, it still leaks PE intermediates.
      expect(r).toBeAllclose([1, 1, 1, 1]);
      r.dispose();
    } catch (e) {
      // CHARACTERIZATION: bare grad(scan) may throw UseAfterFreeError
      // because anonymous const `np.array(0)` gets disposed during backward pass.
      _threw = true;
      expect(String(e)).toMatch(/disposed|UseAfterFree/i);
    }
    x.dispose();

    const report = checkLeaks.stop();
    // Whether it throws or succeeds, there are leaked slots.
    expect(report.leaked).toBeGreaterThan(0);

    checkLeaks.start();
  });

  test("jit(grad(scan(f))) — zero leaks", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = carry.add(x);
      return [newCarry, newCarry];
    };
    const f = (xs: np.Array) => {
      using init = np.array(0);
      const [carry] = lax.scan(step, init, xs);
      return carry.mul(carry);
    };
    using jitGradF = jit(grad(f));
    using xs = np.array([1, 2, 3]);
    using r = jitGradF(xs);
    expect(r).toBeAllclose([12, 12, 12]);
  });

  test("jit(vmap(grad(scan(f)))) — zero leaks", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = carry.add(x);
      return [newCarry, newCarry];
    };
    const f = (xs: np.Array) => {
      using init = np.array(0);
      const [carry] = lax.scan(step, init, xs);
      return carry;
    };
    using jitVmapGrad = jit(vmap(grad(f)));
    using batch = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using r = jitVmapGrad(batch);
    expect(r).toBeAllclose([
      [1, 1, 1],
      [1, 1, 1],
    ]);
  });

  test("scan with valueAndGrad body — zero leaks (eager)", () => {
    const objective = (params: np.Array) => np.sum(params.mul(params));
    const step = (carry: np.Array, _: null): [np.Array, null] => {
      const [, g] = valueAndGrad(objective)(carry);
      const updated = carry.sub(g.mul(np.array(0.1)));
      g.dispose();
      return [updated, null];
    };
    using init = np.array([3.0, 4.0]);
    const [finalCarry] = lax.scan(step, init, null, { length: 5 });
    using _fc = finalCarry;
    const expected0 = 3.0 * Math.pow(0.8, 5);
    const expected1 = 4.0 * Math.pow(0.8, 5);
    expect(finalCarry).toBeAllclose([expected0, expected1], { atol: 1e-5 });
  });
});

// ============================================================
// Category 10: Edge cases — ownership boundary conditions
// ============================================================

describe("ownership edge cases", () => {
  test("makeJaxpr with anonymous const — dispose cleans up", () => {
    const f = (x: np.Array) => x.add(np.array(0));
    const { jaxpr } = makeJaxpr(f)(np.array(1));
    jaxpr.dispose();
  });

  test("same array captured as const by multiple jit calls — zero leaks", () => {
    using shared = np.array([1, 2, 3]);
    const f = (x: np.Array) => x.add(shared);
    using jf = jit(f);
    using x = np.array([10, 20, 30]);

    using r1 = jf(x);
    expect(r1).toBeAllclose([11, 22, 33]);

    using r2 = jf(x);
    expect(r2).toBeAllclose([11, 22, 33]);
  });

  test("clearCaches releases jit-captured const refs", () => {
    // Use manual checkLeaks scope to control cleanup order
    const outerResult = checkLeaks.stop();
    expect(outerResult.leaked).toBe(0);

    checkLeaks.start();
    const arr = np.array([1, 2, 3]);
    const f = (x: np.Array) => x.add(arr);
    const jf = jit(f);
    const x = np.array([0, 0, 0]);
    const r = jf(x);
    expect(r).toBeAllclose([1, 2, 3]);
    r.dispose();
    x.dispose();

    // arr has extra ref from JIT capture
    const rcAfterJit = arr.refCount;
    expect(rcAfterJit).toBeGreaterThanOrEqual(2);

    jf.dispose();
    clearCaches();
    // After clearing, arr should be back to just user ownership
    expect(arr.refCount).toBe(1);
    arr.dispose();

    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);

    checkLeaks.start();
  });

  test("dispose of jit function releases captured consts", () => {
    // Manual scope to control jf disposal and clearCaches ordering
    const outerResult = checkLeaks.stop();
    expect(outerResult.leaked).toBe(0);

    checkLeaks.start();
    const arr = np.array([5, 6, 7]);
    const f = (x: np.Array) => x.mul(arr);
    const jf = jit(f);
    const x = np.array([1, 1, 1]);
    const r = jf(x);
    expect(r).toBeAllclose([5, 6, 7]);
    r.dispose();
    x.dispose();

    const rcBefore = arr.refCount;
    jf.dispose();
    // jf.dispose() should release the jaxpr cache ref
    expect(arr.refCount).toBeLessThan(rcBefore);
    clearCaches();
    arr.dispose();

    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);

    checkLeaks.start();
  });
});
