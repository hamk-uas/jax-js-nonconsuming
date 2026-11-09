/**
 * Steady-state leak checks (`checkLeaks.steadyState`) for linalg, scan,
 * foriLoop and the transforms.
 *
 * The per-test `checkLeaks.stop()` in test/setup.ts flushes all JIT caches
 * before counting, so a leak that lands in a cache is freed first and goes
 * unseen. Here every body runs twice and the second run must not change the
 * slot count, with the caches left intact.
 */
import {
  checkLeaks,
  grad,
  hessian,
  jacfwd,
  jit,
  jvp,
  lax,
  linearize,
  numpy as np,
  tree,
  valueAndGrad,
  vjp,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

import { deviceSuite } from "./device-suite.js";

function expectSteady(body: () => void) {
  const report = checkLeaks.steadyState(body);
  expect(report.leaked, report.summary).toBe(0);
}

await deviceSuite((device) => {
  suite("linalg", () => {
    const hasRoutines = device !== "webgl";

    test("cholesky, eager and under grad", ({ skip }) => {
      if (!hasRoutines) skip();
      expectSteady(() => {
        using A = np.array([
          [4, 1],
          [1, 3],
        ]);
        using L = lax.linalg.cholesky(A);
        const f = (a: np.Array) => {
          using l = lax.linalg.cholesky(a);
          return l.sum();
        };
        using g = grad(f)(A);
        g.js();
        L.js();
      });
    });

    test("qr, eager and under jvp", ({ skip }) => {
      if (!hasRoutines) skip();
      expectSteady(() => {
        using A = np.array([
          [3, 1],
          [1, 4],
          [2, 2],
        ]);
        using dA = np.ones([3, 2]);
        const f = (a: np.Array) => {
          const [q, r] = lax.linalg.qr(a);
          q.dispose();
          return r;
        };
        const [r, dr] = jvp(f, [A], [dA]);
        using _r = r;
        using _dr = dr;
        dr.js();
      });
    });

    test("lu and triangularSolve", ({ skip }) => {
      if (!hasRoutines) skip();
      expectSteady(() => {
        using A = np.array([
          [4, 3],
          [6, 3],
        ]);
        const [lu, pivots, permutation] = lax.linalg.lu(A);
        using _lu = lu;
        using _pivots = pivots;
        using _permutation = permutation;
        using L = np.array([
          [2, 0],
          [1, 3],
        ]);
        using b = np.array([[4], [7]]);
        using x = lax.linalg.triangularSolve(L, b, {
          leftSide: true,
          lower: true,
        });
        x.js();
      });
    });

    test("jit(cholesky solve)", ({ skip }) => {
      if (!hasRoutines) skip();
      using f = jit((a: np.Array) => {
        using l = lax.linalg.cholesky(a);
        return l.sum();
      });
      expectSteady(() => {
        using A = np.array([
          [4, 1],
          [1, 3],
        ]);
        using r = f(A);
        r.js();
      });
    });
  });

  suite("scan", () => {
    const step = (c: np.Array, x: np.Array): [np.Array, np.Array] => {
      const next = np.add(c, x);
      return [next, next];
    };

    test("eager scan", () => {
      expectSteady(() => {
        using init = np.array([0.0]);
        using xs = np.array([[1.0], [2.0], [3.0]]);
        const [c, ys] = lax.scan(step, init, xs);
        using _c = c;
        using _ys = ys;
        c.js();
      });
    });

    test("jit of scan", () => {
      using f = jit((init: np.Array, xs: np.Array) => {
        const [c, ys] = lax.scan(step, init, xs);
        ys.dispose();
        return c;
      });
      expectSteady(() => {
        using init = np.array([0.0]);
        using xs = np.array([[1.0], [2.0], [3.0]]);
        using c = f(init, xs);
        c.js();
      });
    });

    test("jit of grad of scan", () => {
      using f = jit(
        grad((xs: np.Array) => {
          using init = np.array([0.0]);
          const [c, ys] = lax.scan(step, init, xs);
          ys.dispose();
          return c.sum();
        }),
      );
      expectSteady(() => {
        using xs = np.array([[1.0], [2.0], [3.0]]);
        using g = f(xs);
        g.js();
      });
    });
  });

  suite("foriLoop", () => {
    test("eager foriLoop", () => {
      expectSteady(() => {
        using init = np.array([0.0, 1.0]);
        using r = lax.foriLoop(
          0,
          4,
          (_i: np.Array, acc: np.Array) => np.add(acc, 1),
          init,
        );
        r.js();
      });
    });

    test("jit of foriLoop", () => {
      using f = jit((init: np.Array) =>
        lax.foriLoop(
          0,
          4,
          (_i: np.Array, acc: np.Array) => np.add(acc, 1),
          init,
        ),
      );
      expectSteady(() => {
        using init = np.array([0.0, 1.0]);
        using r = f(init);
        r.js();
      });
    });

    test("grad of foriLoop", () => {
      const g = grad((x: np.Array) =>
        lax
          .foriLoop(
            0,
            3,
            (_i: np.Array, acc: np.Array) => np.multiply(acc, x),
            x,
          )
          .sum(),
      );
      expectSteady(() => {
        using x = np.array([1.5, 2.0]);
        using r = g(x);
        r.js();
      });
    });
  });

  suite("transforms", () => {
    const fn = (x: np.Array) => np.sin(x).mul(x).sum();

    test("grad, valueAndGrad, hessian, jacfwd", () => {
      expectSteady(() => {
        using x = np.array([0.5, 1.5]);
        using g = grad(fn)(x);
        const [v, g2] = valueAndGrad(fn)(x);
        using _v = v;
        using _g2 = g2;
        using h = hessian(fn)(x);
        using j = jacfwd((a: np.Array) => np.sin(a))(x);
        g.js();
        h.js();
        j.js();
      });
    });

    test("jvp, vjp, linearize", () => {
      expectSteady(() => {
        using x = np.array([0.5, 1.5]);
        using t = np.ones([2]);
        const [y, dy] = jvp(fn, [x], [t]);
        using _y = y;
        using _dy = dy;
        const [y2, backward] = vjp(fn, [x]);
        using _y2 = y2;
        const cts = backward(1);
        tree.dispose(cts);
        backward.dispose();
        const [y3, lin] = linearize(fn, [x]);
        using _lin = lin;
        using _y3 = y3;
        using dy3 = lin(t);
        dy3.js();
      });
    });

    test("jit(vmap(grad)) and jit(grad)", () => {
      using f = jit(vmap(grad((x: np.Array) => np.sin(x).sum())));
      using h = jit(grad(fn));
      expectSteady(() => {
        using xs = np.array([
          [0.5, 1.5],
          [1.0, 2.0],
        ]);
        using r = f(xs);
        using x = np.array([0.5, 1.5]);
        using r2 = h(x);
        r.js();
        r2.js();
      });
    });
  });
});
