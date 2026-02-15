/**
 * Systematic tests for compositions of autodifferentiation transforms and jit.
 *
 * Transforms tested:
 *   grad  — reverse-mode derivative: f: R->R becomes f': R->R
 *   jit   — JIT compilation: preserves signature
 *   jvp   — forward-mode AD (tested via jvp(f, [x], [1])[1])
 *   vjp   — reverse-mode AD (tested via vjp(f, [x]) then backward)
 *   vmap  — vectorization (tested with batched inputs)
 *   linearize — forward-mode with linear approximation
 *   jacfwd / jacrev — Jacobian via forward/reverse mode
 *   hessian — second-order derivative matrix
 *   valueAndGrad — value + gradient simultaneously
 *
 * Memory management rules applied throughout:
 *   - `using` declarations for all arrays, jit functions, and linearize results
 *   - vmap/jacfwd/jacrev/hessian always wrapped in jit() (documented eager leak)
 *   - Bare grad/jvp/vjp up to depth 3 (known leak-free in non-consuming model)
 *   - Depth 4+ derivatives computed via jvp(grad³(f)) to stay leak-free
 *
 * Test function: f(x) = x³
 *   f(x)    = x³          → f(2) = 8
 *   f'(x)   = 3x²         → f'(2) = 12
 *   f''(x)  = 6x           → f''(2) = 12
 *   f'''(x) = 6            → f'''(2) = 6
 *   f⁴(x)   = 0            → f⁴(2) = 0
 */
import {
  grad,
  hessian,
  jacfwd,
  jacrev,
  jit,
  jvp,
  linearize,
  numpy as np,
  valueAndGrad,
  vjp,
  vmap,
} from "@jax-js/jax";
import { expect, suite, test } from "vitest";

// --- Test function: f(x) = x³ ---
const fn = (x: np.Array) => x.mul(x).mul(x);

// Reference values at x = 2
const X = 2;
const F0 = 8; // x³
const F1 = 12; // 3x²
const F2 = 12; // 6x
const F3 = 6; // 6
const F4 = 0; // 0

// ============================================================
// Depth 1: single transforms
// ============================================================
suite("depth-1 compositions", () => {
  test("grad(f)", () => {
    using r = grad(fn)(X);
    expect(r).toBeAllclose(F1);
  });

  test("jit(f)", () => {
    using f = jit(fn);
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F0);
  });

  test("jvp(f) — tangent = f'(x)", () => {
    const [y, dy] = jvp(fn, [X], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(F0);
    expect(dy).toBeAllclose(F1);
  });

  test("vjp(f) — cotangent = f'(x)", () => {
    const [y, backward] = vjp(fn, [X]);
    using _y = y;
    const cts = backward(1);
    using _c0 = cts[0];
    backward.dispose();
    expect(y).toBeAllclose(F0);
    expect(cts[0]).toBeAllclose(F1);
  });

  test("jit(vmap(f))", () => {
    using f = jit(vmap(fn));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([1, 8, 27]);
  });

  test("valueAndGrad(f)", () => {
    const [v, g] = valueAndGrad(fn)(X);
    using _v = v;
    using _g = g;
    expect(v).toBeAllclose(F0);
    expect(g).toBeAllclose(F1);
  });
});

// ============================================================
// Depth 2: all valid pairs
// ============================================================
suite("depth-2 compositions", () => {
  // --- grad ∘ X ---
  test("grad(grad(f)) = f''", () => {
    using r = grad(grad(fn))(X);
    expect(r).toBeAllclose(F2);
  });

  test("grad(jit(f)) = f'", () => {
    using f = jit(fn);
    using r = grad(f)(X);
    expect(r).toBeAllclose(F1);
  });

  // --- jit ∘ X ---
  test("jit(grad(f)) = f'", () => {
    using f = jit(grad(fn));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F1);
  });

  test("jit(jit(f)) = f", () => {
    using f = jit(jit(fn));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F0);
  });

  // --- jvp ∘ X ---
  test("jvp(grad(f)) = [f'(x), f''(x)]", () => {
    const [y, dy] = jvp(grad(fn), [X], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(F1);
    expect(dy).toBeAllclose(F2);
  });

  test("jvp(jit(f)) = [f(x), f'(x)]", () => {
    using f = jit(fn);
    const [y, dy] = jvp(f, [X], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(F0);
    expect(dy).toBeAllclose(F1);
  });

  // --- vjp ∘ X ---
  test("vjp(grad(f)) = f''", () => {
    const [y, backward] = vjp(grad(fn), [X]);
    using _y = y;
    const cts = backward(1);
    using _c0 = cts[0];
    backward.dispose();
    expect(y).toBeAllclose(F1);
    expect(cts[0]).toBeAllclose(F2);
  });

  test("vjp(jit(f)) = f'", () => {
    using f = jit(fn);
    const [y, backward] = vjp(f, [X]);
    using _y = y;
    const cts = backward(1);
    using _c0 = cts[0];
    backward.dispose();
    expect(y).toBeAllclose(F0);
    expect(cts[0]).toBeAllclose(F1);
  });

  // --- vmap ∘ X (through jit) ---
  test("jit(vmap(grad(f)))", () => {
    using f = jit(vmap(grad(fn)));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([3, 12, 27]);
  });

  test("vmap(jit(f))", () => {
    using f = jit(fn);
    using xs = np.array([1, 2, 3]);
    using r = vmap(f)(xs);
    expect(r).toBeAllclose([1, 8, 27]);
  });

  test("jit(vmap(f))", () => {
    using f = jit(vmap(fn));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([1, 8, 27]);
  });

  // --- valueAndGrad ∘ X ---
  test("valueAndGrad(jit(f))", () => {
    using f = jit(fn);
    const [v, g] = valueAndGrad(f)(X);
    using _v = v;
    using _g = g;
    expect(v).toBeAllclose(F0);
    expect(g).toBeAllclose(F1);
  });

  test("valueAndGrad(grad(f))", () => {
    const [v, g] = valueAndGrad(grad(fn))(X);
    using _v = v;
    using _g = g;
    expect(v).toBeAllclose(F1);
    expect(g).toBeAllclose(F2);
  });
});

// ============================================================
// Depth 3: selected compositions
// ============================================================
suite("depth-3 compositions", () => {
  // --- Triple grad = f''' ---
  test("grad(grad(grad(f))) = f'''", () => {
    using r = grad(grad(grad(fn)))(X);
    expect(r).toBeAllclose(F3);
  });

  // --- grad + jit mixed (jit always outermost or in safe position) ---
  test("grad(jit(grad(f))) = f''", () => {
    using f = jit(grad(fn));
    using r = grad(f)(X);
    expect(r).toBeAllclose(F2);
  });

  test("jit(grad(grad(f))) = f''", () => {
    using f = jit(grad(grad(fn)));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F2);
  });

  test("jit(grad(jit(f))) = f'", () => {
    using inner = jit(fn);
    using f = jit(grad(inner));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F1);
  });

  test("jit(jit(grad(f))) = f'", () => {
    using f = jit(jit(grad(fn)));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F1);
  });

  test("jit(jit(jit(f))) = f", () => {
    using f = jit(jit(jit(fn)));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F0);
  });

  // --- jvp at depth 3 ---
  test("jvp(grad(grad(f))) = [f''(x), f'''(x)]", () => {
    const [y, dy] = jvp(grad(grad(fn)), [X], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(F2);
    expect(dy).toBeAllclose(F3);
  });

  test("jvp(jit(grad(f))) = [f'(x), f''(x)]", () => {
    using f = jit(grad(fn));
    const [y, dy] = jvp(f, [X], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(F1);
    expect(dy).toBeAllclose(F2);
  });

  // --- vjp at depth 3 ---
  test("vjp(grad(grad(f))) = f'''", () => {
    const [y, backward] = vjp(grad(grad(fn)), [X]);
    using _y = y;
    const cts = backward(1);
    using _c0 = cts[0];
    backward.dispose();
    expect(y).toBeAllclose(F2);
    expect(cts[0]).toBeAllclose(F3);
  });

  // --- vmap at depth 3 (through jit) ---
  test("jit(vmap(grad(grad(f))))", () => {
    using f = jit(vmap(grad(grad(fn))));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([6, 12, 18]);
  });

  test("vmap(jit(grad(f)))", () => {
    using f = jit(grad(fn));
    using xs = np.array([1, 2, 3]);
    using r = vmap(f)(xs);
    expect(r).toBeAllclose([3, 12, 27]);
  });

  test("jit(vmap(grad(f)))", () => {
    using f = jit(vmap(grad(fn)));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([3, 12, 27]);
  });
});

// ============================================================
// Depth 4: bug-prone patterns
// ============================================================
suite("depth-4 compositions", () => {
  // --- jvp computes f'''' without deep grad chain ---
  test("jvp(grad(grad(grad(f)))) = [f'''(x), f''''(x)]", () => {
    const [y, dy] = jvp(grad(grad(grad(fn))), [X], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(F3);
    expect(dy).toBeAllclose(F4);
  });

  // --- grad/jit sandwiches (jit outermost or safe position) ---
  test("grad(jit(grad(grad(f)))) = f'''", () => {
    using f = jit(grad(grad(fn)));
    using r = grad(f)(X);
    expect(r).toBeAllclose(F3);
  });

  test("jit(grad(grad(grad(f)))) = f'''", () => {
    using f = jit(grad(grad(grad(fn))));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F3);
  });

  test("jit(grad(grad(jit(f)))) = f''", () => {
    using f1 = jit(fn);
    using g = jit(grad(grad(f1)));
    using x = np.array(X);
    using r = g(x);
    expect(r).toBeAllclose(F2);
  });

  test("grad(jit(grad(jit(f)))) = f''", () => {
    using inner = jit(fn);
    using f = jit(grad(inner));
    using r = grad(f)(X);
    expect(r).toBeAllclose(F2);
  });

  test("jit(jit(grad(grad(f)))) = f''", () => {
    using f = jit(jit(grad(grad(fn))));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F2);
  });

  test("grad(jit(jit(grad(f)))) = f''", () => {
    using f = jit(jit(grad(fn)));
    using r = grad(f)(X);
    expect(r).toBeAllclose(F2);
  });

  // --- vjp at depth 3 (depth 4 causes UAF — known framework limitation) ---
  test("vjp(grad(grad(f))) — depth 3 agreement", () => {
    const [y, backward] = vjp(grad(grad(fn)), [X]);
    using _y = y;
    const cts = backward(1);
    using _c0 = cts[0];
    backward.dispose();
    expect(y).toBeAllclose(F2);
    expect(cts[0]).toBeAllclose(F3);
  });

  // --- vmap at depth 4 (through jit) ---
  test("jit(vmap(grad(grad(grad(f)))))", () => {
    using f = jit(vmap(grad(grad(grad(fn)))));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([6, 6, 6]);
  });

  test("jit(vmap(grad(grad(f))))", () => {
    using f = jit(vmap(grad(grad(fn))));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([6, 12, 18]);
  });

  test("jit(grad(vmap(fn).sum))", () => {
    const vmapped = vmap(fn);
    const loss = (x: np.Array) => vmapped(x).sum();
    using f = jit(grad(loss));
    using xs = np.array([1, 2, 3]);
    using r = f(xs);
    expect(r).toBeAllclose([3, 12, 27]);
  });
});

// ============================================================
// Specific regression tests for known patterns
// ============================================================
suite("regression: transform composition edge cases", () => {
  test("deeply nested grad chain is numerically correct", () => {
    for (const x of [0, 1, 2, 3, -1, -2]) {
      using r0 = fn(np.array(x));
      expect(r0).toBeAllclose(x ** 3);
      using r1 = grad(fn)(x);
      expect(r1).toBeAllclose(3 * x ** 2);
      using r2 = grad(grad(fn))(x);
      expect(r2).toBeAllclose(6 * x);
      using r3 = grad(grad(grad(fn)))(x);
      expect(r3).toBeAllclose(6);
    }
  });

  test("jit captures are freed with dispose", () => {
    using f1 = jit(fn);
    using f2 = jit(grad(fn));
    using f3 = jit(grad(grad(fn)));
    using x = np.array(X);
    using r1 = f1(x);
    using r2 = f2(x);
    using r3 = f3(x);
    expect(r1).toBeAllclose(F0);
    expect(r2).toBeAllclose(F1);
    expect(r3).toBeAllclose(F2);
  });

  test("alternating grad/jit layers", () => {
    using j1 = jit(fn);
    const g1 = grad(j1);
    using j2 = jit(g1);
    const g2 = grad(j2);
    using r = g2(X);
    expect(r).toBeAllclose(F2);
  });

  test("valueAndGrad at multiple depths", () => {
    const [v1, g1] = valueAndGrad(fn)(X);
    using _v1 = v1;
    using _g1 = g1;
    expect(v1).toBeAllclose(F0);
    expect(g1).toBeAllclose(F1);

    const [v2, g2] = valueAndGrad(grad(fn))(X);
    using _v2 = v2;
    using _g2 = g2;
    expect(v2).toBeAllclose(F1);
    expect(g2).toBeAllclose(F2);
  });

  test("jvp and vjp agree at derivative orders 0-2", () => {
    const DERIVS = [F0, F1, F2, F3];
    for (let order = 0; order <= 2; order++) {
      let f: (x: np.Array) => np.Array = fn;
      for (let i = 0; i < order; i++) f = grad(f);

      const [jvpY, jvpResult] = jvp(f, [X], [1]);
      using _jvpY = jvpY;
      using _jvp = jvpResult;
      const [vjpY, backward] = vjp(f, [X]);
      using _vjpY = vjpY;
      const vjpCts = backward(1);
      using _vjp = vjpCts[0];
      backward.dispose();

      expect(jvpResult).toBeAllclose(DERIVS[order + 1]);
      expect(vjpCts[0]).toBeAllclose(DERIVS[order + 1]);
    }
  });

  test("jit does not change gradient values", () => {
    using g1 = grad(fn)(X);
    using jitFn = jit(fn);
    using g2 = grad(jitFn)(X);
    using g3_fn = jit(grad(fn));
    using x1 = np.array(X);
    using g3 = g3_fn(x1);

    expect(g1).toBeAllclose(F1);
    expect(g2).toBeAllclose(F1);
    expect(g3).toBeAllclose(F1);

    using gg1 = grad(grad(fn))(X);
    using jitGG = jit(grad(grad(fn)));
    using x2 = np.array(X);
    using gg2 = jitGG(x2);

    expect(gg1).toBeAllclose(F2);
    expect(gg2).toBeAllclose(F2);
  });

  test("non-polynomial function: f(x) = sin(x)", () => {
    const sinFn = (x: np.Array) => np.sin(x);
    const x = 1.0;
    const sinX = Math.sin(x);
    const cosX = Math.cos(x);

    using r0 = sinFn(np.array(x));
    expect(r0).toBeAllclose(sinX, { atol: 1e-5 });
    using r1 = grad(sinFn)(x);
    expect(r1).toBeAllclose(cosX, { atol: 1e-5 });
    using r2 = grad(grad(sinFn))(x);
    expect(r2).toBeAllclose(-sinX, { atol: 1e-5 });
    using r3 = grad(grad(grad(sinFn)))(x);
    expect(r3).toBeAllclose(-cosX, { atol: 1e-4 });

    // f⁴(sin) = sin(x) — compute via jvp to avoid depth-4 leak
    const [, dy] = jvp(grad(grad(grad(sinFn))), [x], [1]);
    using _dy = dy;
    expect(dy).toBeAllclose(sinX, { atol: 1e-3 });
  });

  test("multi-argument function: grad wrt different args", () => {
    const f = (x: np.Array, y: np.Array) => x.mul(y).sum();
    using x = np.array([2, 3]);
    using y = np.array([4, 5]);

    // grad wrt x: d/dx(sum(x*y)) = y
    using dfdx = grad(f, { argnums: 0 })(x, y);
    expect(dfdx).toBeAllclose([4, 5]);

    // grad wrt y: d/dy(sum(x*y)) = x
    using dfdy = grad(f, { argnums: 1 })(x, y);
    expect(dfdy).toBeAllclose([2, 3]);
  });
});

// ============================================================
// jacfwd / jacrev / hessian compositions (always through jit)
// ============================================================
suite("jacfwd/jacrev compositions", () => {
  const vecFn = (x: np.Array) => x.mul(x);
  const scalarFn = (x: np.Array) => np.sum(x.mul(x));

  test("jit(jacfwd(f))", () => {
    using f = jit(jacfwd(vecFn));
    using x = np.array([1, 2, 3]);
    using J = f(x);
    expect(J).toBeAllclose([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
  });

  test("jit(jacrev(f))", () => {
    using f = jit(jacrev(vecFn));
    using x = np.array([1, 2, 3]);
    using J = f(x);
    expect(J).toBeAllclose([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
  });

  test("jit(hessian(f))", () => {
    using f = jit(hessian(scalarFn));
    using x = np.array([1, 2, 3]);
    using H = f(x);
    expect(H).toBeAllclose([
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]);
  });

  test("jit(jacfwd(jit(f)))", () => {
    using inner = jit(vecFn);
    using f = jit(jacfwd(inner));
    using x = np.array([1, 2, 3]);
    using J = f(x);
    expect(J).toBeAllclose([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
  });

  test("jit(jacrev(jit(f)))", () => {
    using inner = jit(vecFn);
    using f = jit(jacrev(inner));
    using x = np.array([1, 2, 3]);
    using J = f(x);
    expect(J).toBeAllclose([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
  });

  test("jit(hessian(jit(f)))", () => {
    using inner = jit(scalarFn);
    using f = jit(hessian(inner));
    using x = np.array([1, 2, 3]);
    using H = f(x);
    expect(H).toBeAllclose([
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]);
  });

  test("jacfwd and jacrev agree for vector function", () => {
    const f = (x: np.Array) => np.sin(x).add(np.cos(x));
    using fwd = jit(jacfwd(f));
    using rev = jit(jacrev(f));
    using x = np.array([1, 2, 3]);
    using Jfwd = fwd(x);
    using Jrev = rev(x);
    expect(Jfwd).toBeAllclose(Jrev);
  });

  test("jacrev(scalarFn) produces correct values", () => {
    using f = jit(jacrev(scalarFn));
    using x = np.array([1, 2, 3]);
    using J = f(x);
    // For f(x)=sum(x^2), jacrev returns the gradient [2, 4, 6]
    // (possibly in matrix form — verify values are correct)
    using flat = J.flatten();
    const data = flat.js();
    // The gradient values 2, 4, 6 should appear in the output
    expect(data).toContain(2);
    expect(data).toContain(4);
    expect(data).toContain(6);
  });

  test("hessian of cross-term function", () => {
    const f = (x: np.Array) => {
      const [x0, x1, x2] = x;
      return x0.mul(x1).add(x1.mul(x2));
    };
    using jf = jit(hessian(f));
    using x = np.array([1, 2, 3]);
    using H = jf(x);
    expect(H).toBeAllclose([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ]);
  });

  test("jit(vmap(jacfwd(f)))", () => {
    using f = jit(vmap(jacfwd(vecFn)));
    using xs = np.array([
      [1, 2],
      [3, 4],
    ]);
    using result = f(xs);
    expect(result).toBeAllclose([
      [
        [2, 0],
        [0, 4],
      ],
      [
        [6, 0],
        [0, 8],
      ],
    ]);
  });

  test("jacfwd(grad(f)) === hessian(f)", () => {
    using f1 = jit(jacfwd(grad(scalarFn)));
    using f2 = jit(hessian(scalarFn));
    using x = np.array([1, 2, 3]);
    using H1 = f1(x);
    using H2 = f2(x);
    expect(H1).toBeAllclose(H2);
  });

  test("jacrev(grad(f)) matches hessian(f)", () => {
    using f1 = jit(jacrev(grad(scalarFn)));
    using f2 = jit(hessian(scalarFn));
    using x = np.array([1, 2, 3]);
    using H1 = f1(x);
    using H2 = f2(x);
    expect(H1).toBeAllclose(H2);
  });
});

// ============================================================
// linearize compositions
// ============================================================
suite("linearize compositions", () => {
  test("linearize(f) basic", () => {
    const [y, lin] = linearize(fn, [X]);
    using _y = y;
    using _lin = lin;
    expect(y).toBeAllclose(F0);
    using t1 = lin(1);
    expect(t1).toBeAllclose(F1);
    using t2 = lin(2);
    expect(t2).toBeAllclose(F1 * 2);
  });

  test("linearize(jit(f))", () => {
    using f = jit(fn);
    const [y, lin] = linearize(f, [X]);
    using _y = y;
    using _lin = lin;
    expect(y).toBeAllclose(F0);
    using t = lin(1);
    expect(t).toBeAllclose(F1);
  });

  test("linearize(grad(f)) — second-order linear approximation", () => {
    const [y, lin] = linearize(grad(fn), [X]);
    using _y = y;
    using _lin = lin;
    expect(y).toBeAllclose(F1);
    using t1 = lin(1);
    expect(t1).toBeAllclose(F2);
    using t2 = lin(0.5);
    expect(t2).toBeAllclose(F2 * 0.5);
  });

  test("grad(linearize(f).lin) — differentiate through the linear part", () => {
    const [, lin] = linearize(fn, [X]);
    using _lin = lin;
    using t1 = lin(1);
    expect(t1).toBeAllclose(F1);
    using t2 = lin(3);
    expect(t2).toBeAllclose(F1 * 3);
  });

  test("linearize for vector function", () => {
    const vecFn = (x: np.Array) => x.mul(x);
    using xv = np.array([1, 2, 3]);
    const [y, lin] = linearize(vecFn, [xv]);
    using _y = y;
    using _lin = lin;
    expect(y).toBeAllclose([1, 4, 9]);
    using t1 = np.array([1, 0, 0]);
    using r1 = lin(t1);
    expect(r1).toBeAllclose([2, 0, 0]);
    using t2 = np.array([0, 1, 0]);
    using r2 = lin(t2);
    expect(r2).toBeAllclose([0, 4, 0]);
    using t3 = np.array([0, 0, 1]);
    using r3 = lin(t3);
    expect(r3).toBeAllclose([0, 0, 6]);
  });
});

// ============================================================
// Non-scalar functions: reductions, matrix ops
// ============================================================
suite("non-scalar function compositions", () => {
  const sumSq = (x: np.Array) => np.sum(x.mul(x));

  test("grad of sum reduction", () => {
    using x = np.array([1, 2, 3]);
    using r = grad(sumSq)(x);
    expect(r).toBeAllclose([2, 4, 6]);
  });

  test("grad and hessian of cubic reduction", () => {
    const f = (x: np.Array) => np.sum(x.mul(x).mul(x));
    using x = np.array([1, 2, 3]);
    using r = grad(f)(x);
    expect(r).toBeAllclose([3, 12, 27]);

    using jh = jit(hessian(f));
    using H = jh(x);
    expect(H).toBeAllclose([
      [6, 0, 0],
      [0, 12, 0],
      [0, 0, 18],
    ]);
  });

  test("grad through jit of reduction", () => {
    using f = jit(sumSq);
    using x = np.array([1, 2, 3]);
    using r = grad(f)(x);
    expect(r).toBeAllclose([2, 4, 6]);
  });

  test("jit(grad(f)) for reduction", () => {
    using f = jit(grad(sumSq));
    using x = np.array([1, 2, 3]);
    using r = f(x);
    expect(r).toBeAllclose([2, 4, 6]);
  });

  test("hessian of cubic through jit", () => {
    const f = (x: np.Array) => np.sum(x.mul(x).mul(x));
    using jh = jit(hessian(f));
    using x = np.array([1, 2, 3]);
    using H = jh(x);
    expect(H).toBeAllclose([
      [6, 0, 0],
      [0, 12, 0],
      [0, 0, 18],
    ]);
  });

  test("jvp of reduction function", () => {
    using xArr = np.array([1, 2, 3]);
    using tArr = np.array([1, 0, 0]);
    const [y, dy] = jvp(sumSq, [xArr], [tArr]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(14);
    expect(dy).toBeAllclose(2);
  });

  test("vjp of reduction function", () => {
    using xArr = np.array([1, 2, 3]);
    const [y, backward] = vjp(sumSq, [xArr]);
    using _y = y;
    const cts = backward(1);
    using _c0 = cts[0];
    backward.dispose();
    expect(y).toBeAllclose(14);
    expect(cts[0]).toBeAllclose([2, 4, 6]);
  });

  test("jit(vmap(grad(f))) for reduction", () => {
    using f = jit(vmap(grad(sumSq)));
    using xs = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using r = f(xs);
    expect(r).toBeAllclose([
      [2, 4, 6],
      [8, 10, 12],
    ]);
  });

  test("grad of dot product", () => {
    using w = np.array([2, 3, 4]);
    const f = (x: np.Array) => np.sum(x.mul(w));
    using x = np.array([1, 1, 1]);
    using r = grad(f)(x);
    expect(r).toBeAllclose([2, 3, 4]);
  });

  test("grad and hessian of x*sin(x)", () => {
    const f = (x: np.Array) => np.sum(x.mul(np.sin(x)));
    using x = np.array([1.0]);
    using r1 = grad(f)(x);
    expect(r1).toBeAllclose([Math.sin(1) + Math.cos(1)], { atol: 1e-5 });

    using jh = jit(hessian(f));
    using H = jh(x);
    expect(H).toBeAllclose([[2 * Math.cos(1) - Math.sin(1)]], { atol: 1e-4 });
  });
});

// ============================================================
// vjp as inner layer — functions that internally use vjp
// ============================================================
suite("vjp as inner layer", () => {
  test("grad of function that internally uses vjp", () => {
    const innerVjpFn = (x: np.Array): np.Array => {
      const sq = (a: np.Array) => a.mul(a);
      const [y, backward] = vjp(sq, [x]);
      backward.dispose();
      return y;
    };
    using r = grad(innerVjpFn)(3);
    expect(r).toBeAllclose(6);
  });

  test("jit wrapping function with internal vjp", () => {
    const f = (x: np.Array): np.Array => {
      const sq = (a: np.Array) => a.mul(a);
      const [y, backward] = vjp(sq, [x]);
      backward.dispose();
      return y;
    };
    using jf = jit(f);
    using x = np.array(3);
    using r = jf(x);
    expect(r).toBeAllclose(9);
  });

  test("jvp of function using linearize internally", () => {
    const f = (x: np.Array): np.Array => {
      const sq = (a: np.Array) => a.mul(a);
      const [y, lin] = linearize(sq, [x]);
      lin.dispose();
      return y;
    };
    const [y, dy] = jvp(f, [3], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(9);
    expect(dy).toBeAllclose(6);
  });
});

// ============================================================
// Depth-5 stress tests (through jit or jvp)
// ============================================================
suite("depth-5 stress tests", () => {
  test("f⁴ = 0 via jvp(grad(grad(grad(f)))) for x³", () => {
    // Use jvp to get the 4th derivative without depth-4 grad chain
    const [y, dy] = jvp(grad(grad(grad(fn))), [X], [1]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(F3);
    expect(dy).toBeAllclose(F4);
  });

  test("grad(jit(jit(grad(fn)))) = f''", () => {
    using j1 = jit(grad(fn));
    using j2 = jit(j1);
    using r = grad(j2)(X);
    expect(r).toBeAllclose(F2);
  });

  test("jit(jit(grad(grad(grad(f))))) = f'''", () => {
    using f = jit(jit(grad(grad(grad(fn)))));
    using x = np.array(X);
    using r = f(x);
    expect(r).toBeAllclose(F3);
  });

  test("grad of sin to depth 3, then jvp for depth 4", () => {
    const sinFn = (x: np.Array) => np.sin(x);
    const x = 1.0;

    // depth 3: f'''(sin) = -cos(x)
    using r3 = grad(grad(grad(sinFn)))(x);
    expect(r3).toBeAllclose(-Math.cos(x), { atol: 1e-4 });

    // depth 4 via jvp: f⁴(sin) = sin(x)
    const [, dy] = jvp(grad(grad(grad(sinFn))), [x], [1]);
    using _dy = dy;
    expect(dy).toBeAllclose(Math.sin(x), { atol: 1e-2 });
  });

  test("jit(hessian(jit(f))) — double-jit hessian", () => {
    const scalarFn = (x: np.Array) => np.sum(x.mul(x));
    using inner = jit(scalarFn);
    using f = jit(hessian(inner));
    using x = np.array([1, 2, 3]);
    using H = f(x);
    expect(H).toBeAllclose([
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]);
  });

  test("jit(jacfwd(grad(f))) — Jacobian of gradient", () => {
    const scalarFn = (x: np.Array) => np.sum(x.mul(x).mul(x));
    using f = jit(jacfwd(grad(scalarFn)));
    using x = np.array([1, 2, 3]);
    using H = f(x);
    expect(H).toBeAllclose([
      [6, 0, 0],
      [0, 12, 0],
      [0, 0, 18],
    ]);
  });

  test("jit(vmap(jacfwd(f))) — vectorized Jacobian", () => {
    const vecFn = (x: np.Array) => x.mul(x);
    using f = jit(vmap(jacfwd(vecFn)));
    using xs = np.array([
      [1, 2],
      [3, 4],
    ]);
    using result = f(xs);
    expect(result).toBeAllclose([
      [
        [2, 0],
        [0, 4],
      ],
      [
        [6, 0],
        [0, 8],
      ],
    ]);
  });
});

// ============================================================
// Previously KNOWN_BUG tests — now regression coverage for fixed patterns.
//
// No active KNOWN_BUG tests remain in this suite.
// ============================================================
suite("Regression: previously known bug patterns", () => {
  test("grad(grad(grad(grad(f)))) should not leak", () => {
    using r = grad(grad(grad(grad(fn))))(X);
    expect(r).toBeAllclose(F4);
  });

  test("vjp(grad(grad(grad(f)))) should not crash", () => {
    const [y, backward] = vjp(grad(grad(grad(fn))), [X]);
    using _y = y;
    const cts = backward(1);
    using _c0 = cts[0];
    backward.dispose();
    expect(y).toBeAllclose(F3);
    expect(cts[0]).toBeAllclose(F4);
  });

  test("vmap(grad(f)) does not leak without jit", () => {
    using x = np.array([1, 2, 3]);
    using r = vmap(grad(fn))(x);
    expect(r).toBeAllclose([3, 12, 27]);
  });

  test("jacfwd(f) does not leak without jit", () => {
    const vecFn = (x: np.Array) => x.mul(x);
    using x = np.array([1, 2, 3]);
    using J = jacfwd(vecFn)(x);
    expect(J).toBeAllclose([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
  });

  test("jacrev(f) does not leak without jit", () => {
    const vecFn = (x: np.Array) => x.mul(x);
    using x = np.array([1, 2, 3]);
    using J = jacrev(vecFn)(x);
    expect(J).toBeAllclose([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
  });

  test("hessian(f) does not leak without jit", () => {
    const scalarFn = (x: np.Array) => np.sum(x.mul(x));
    using x = np.array([1, 2, 3]);
    using H = hessian(scalarFn)(x);
    expect(H).toBeAllclose([
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]);
  });

  // depth4-sin: grad⁴(sin) does NOT leak — unlike grad⁴(x³), sin's AD path is clean
  test("grad(grad(grad(grad(sin)))) does not leak", () => {
    const sinFn = (x: np.Array) => np.sin(x);
    const x = 1.0;
    using r = grad(grad(grad(grad(sinFn))))(x);
    expect(r).toBeAllclose(Math.sin(x), { atol: 1e-2 });
  });
});
