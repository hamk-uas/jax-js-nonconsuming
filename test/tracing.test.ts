import {
  grad,
  jit,
  jvp,
  linearize,
  makeJaxpr,
  nn,
  numpy as np,
  tree,
  valueAndGrad,
  vjp,
  vmap,
} from "@jax-js-nonconsuming/jax";
import { expect, suite, test } from "vitest";

suite("jax.makeJaxpr()", () => {
  test("tracks a nullary function", () => {
    const { jaxpr } = makeJaxpr(() => np.multiply(2, 2))();
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda  .
        ( 4 ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("tracks a unary function", () => {
    using x = np.array([
      [2, 4, 10],
      [1, 1, 1],
    ]);
    const { jaxpr } = makeJaxpr((x: np.Array) => np.multiply(x.add(2), x))(x);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[2,3] .
        let b:float32[2,3] = add a 2
            c:float32[2,3] = mul b a
        in ( c ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("composes with jvp", () => {
    const f = (x: np.Array) => np.multiply(x.add(2), x);
    const fdot = (x: np.Array) => {
      const [, dy] = jvp(f, [x], [1]);
      return dy;
    };

    using x = np.array(2);
    const { jaxpr } = makeJaxpr(fdot)(x);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add a 2
            c:float32[] = add b a
        in ( c ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("composes with grad", () => {
    const f = (x: np.Array) => {
      using y = x.add(2);
      return x.mul(x).add(y);
    };
    const { jaxpr } = makeJaxpr(grad(f))(3);
    expect(jaxpr.consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add 1 a
            c:float32[] = add b a
        in ( c ) }"
    `);
  });

  test("can flatten() nested Jaxprs", () => {
    const f = (x: np.Array) => {
      using y = x.add(2);
      return x.mul(x).add(y);
    };
    using jf = jit(f);

    const { jaxpr } = makeJaxpr((x) => f(jf(x)))(3);
    expect(jaxpr.consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = jit [ name=f
                                jaxpr={ lambda a:float32[] .
                                  let b:float32[] = add a 2
                                      c:float32[] = mul a a
                                      d:float32[] = add c b
                                  in ( d ) }
                                numConsts=0 ] a
            c:float32[] = add b 2
            d:float32[] = mul b b
            e:float32[] = add d c
        in ( e ) }"
    `);
    expect(jaxpr.jaxpr.flatten().toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add a 2
            c:float32[] = mul a a
            d:float32[] = add c b
            e:float32[] = add d 2
            f:float32[] = mul d d
            g:float32[] = add f e
        in ( g ) }"
    `);
  });
});

suite("jax.linearize()", () => {
  test("works for scalars", () => {
    const [y, lin] = linearize(np.sin, [3]);
    using _y = y;
    using _lin = lin;
    using sin3 = np.sin(3);
    expect(y).toBeAllclose(sin3);
    using r1 = lin(1);
    using cos3 = np.cos(3);
    expect(r1).toBeAllclose(cos3);
    using r2 = lin(-42);
    using expected = cos3.mul(-42);
    expect(r2).toBeAllclose(expected);
  });

  test("works for simple arrays", () => {
    using input = np.array([2, 3]);
    const [y, lin] = linearize((x: np.Array) => x.mul(x), [input]);
    using _y = y;
    using _lin = lin;
    expect(y).toBeAllclose([4, 9]);
    using t1 = np.array([1, 0]);
    using r1 = lin(t1);
    expect(r1).toBeAllclose([4, 0]);
    using t2 = np.array([0, 1]);
    using r2 = lin(t2);
    expect(r2).toBeAllclose([0, 6]);
  });

  test("can take and return jstrees", () => {
    const [y, lin] = linearize(
      (x: { a: np.Array; b: np.Array }) => ({
        r1: x.a.mul(x.a).add(x.b),
        r2: x.b,
      }),
      [{ a: 1, b: 2 }],
    );
    using _lin = lin;
    using _yr1 = y.r1;
    using _yr2 = y.r2;
    expect(y.r1).toBeAllclose(3);
    expect(y.r2).toBeAllclose(2);

    const { r1: r1Dot, r2: r2Dot } = lin({ a: 1, b: 0 });
    using _r1Dot = r1Dot;
    using _r2Dot = r2Dot;
    expect(r1Dot).toBeAllclose(2);
    expect(r2Dot).toBeAllclose(0);
  });
});

suite("jax.vjp()", () => {
  test("works for scalars", () => {
    const [y, backward] = vjp(np.sin, [3]);
    using _y = y;
    using _backward = backward;
    using sin3 = np.sin(3);
    expect(y).toBeAllclose(sin3);
    const grads = backward(1);
    using _g0 = grads[0];
    using cos3 = np.cos(3);
    expect(grads[0]).toBeAllclose(cos3);
  });

  test("hasAux returns aux and computes correct gradients", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      using loss = x.sum();
      using aux = x.mul(2);
      return [loss, aux];
    };

    using x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjp(f, [x], { hasAux: true });
    using _loss = loss;
    using _vjpFn = vjpFn;
    using _aux = aux;

    expect(loss).toBeAllclose(6);
    expect(aux).toBeAllclose([2, 4, 6]);

    const [g] = vjpFn(1);
    using _g = g;
    expect(g).toBeAllclose([1, 1, 1]);
  });

  test("hasAux handles pytree aux", () => {
    type Aux = { predictions: np.Array; squared: np.Array };
    const f = (x: np.Array): [np.Array, Aux] => {
      using loss = x.sum();
      const aux = {
        predictions: x.mul(2),
        squared: x.mul(x),
      };
      return [loss, aux];
    };

    using x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjp(f, [x], { hasAux: true });
    using _loss = loss;
    using _vjpFn = vjpFn;
    using _pred = aux.predictions;
    using _sq = aux.squared;

    expect(loss).toBeAllclose(6);
    expect(aux.predictions).toBeAllclose([2, 4, 6]);
    expect(aux.squared).toBeAllclose([1, 4, 9]);
  });

  test("hasAux handles pytree main output", () => {
    type Main = { a: np.Array; b: np.Array };
    const f = (x: np.Array): [Main, np.Array] => {
      const main = { a: x.sum(), b: x.prod() };
      using aux = x.mul(2);
      return [main, aux];
    };

    using x = np.array([1, 2, 3]);
    const [main, vjpFn, _aux] = vjp(f, [x], { hasAux: true });
    using _mainA = main.a;
    using _mainB = main.b;
    using _vjpFn = vjpFn;
    using _auxArr = _aux;

    expect(main.a).toBeAllclose(6);
    expect(main.b).toBeAllclose(6);

    using ct_a = np.ones([]);
    using ct_b = np.ones([]);
    const [g] = vjpFn({ a: ct_a, b: ct_b });
    using _g = g;
    expect(g).toBeAllclose([7, 4, 3]);
  });

  test("hasAux throws if function does not return tuple", () => {
    const f = (x: np.Array) => x.sum();
    using x = np.array([1, 2, 3]);
    expect(() => vjp(f as any, [x], { hasAux: true })).toThrow(/tuple/);
  });

  test("hasAux gradients match vjp without aux", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [x.sum(), x.mul(2)];
    const fWithoutAux = (x: np.Array): np.Array => x.sum();

    using x = np.array([1, 2, 3]);

    const [loss1, vjpFn1, aux1] = vjp(fWithAux, [x], { hasAux: true });
    using _loss1 = loss1;
    using _aux1 = aux1;
    const [y2, vjpFn2] = vjp(fWithoutAux, [x]);
    using _y2 = y2;
    using _vjpFn1 = vjpFn1;
    using _vjpFn2 = vjpFn2;

    using ones = np.ones([]);
    const [grad1] = vjpFn1(ones);
    const [grad2] = vjpFn2(ones);
    using _grad1 = grad1;
    using _grad2 = grad2;

    expect(grad1).toBeAllclose(grad2);
  });

  test("hasAux works with jit wrapper", () => {
    using f = jit((x: np.Array): [np.Array, np.Array] => [x.sum(), x.mul(2)]);

    using x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjp(f, [x], { hasAux: true });
    using _loss = loss;
    using _vjpFn = vjpFn;
    using _aux = aux;

    expect(loss).toBeAllclose(6);
    expect(aux).toBeAllclose([2, 4, 6]);
    using ones = np.ones([]);
    const [g] = vjpFn(ones);
    using _g = g;
    expect(g).toBeAllclose([1, 1, 1]);
  });

  test("hasAux works inside jit", () => {
    const inner = (x: np.Array): [np.Array, np.Array] => [x.sum(), x.mul(2)];

    using outer = jit((x: np.Array): [np.Array, np.Array] => {
      const [y, vjpFn, aux] = vjp(inner, [x], { hasAux: true });
      tree.dispose(y);
      const [g] = vjpFn(np.ones([]));
      vjpFn.dispose();
      return [g, aux];
    });

    using x = np.array([1, 2, 3]);
    const [g, aux] = outer(x);
    using _g = g;
    using _aux = aux;

    expect(g).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });
});

suite("jax.grad()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => x.mul(x).mul(x); // d/dx (x^3) = 3x^2
    const df = grad(f);
    using r1 = df(4);
    expect(r1).toBeAllclose(48);
    using r2 = df(5);
    expect(r2).toBeAllclose(75);
    using r3 = df(0);
    expect(r3).toBeAllclose(0);
    using r4 = df(-4);
    expect(r4).toBeAllclose(48);
  });

  test("can compute higher derivatives", () => {
    const f = (x: np.Array) => np.sin(np.cos(x));
    const df = grad(f); // d/dx sin(cos(x)) = -sin(x)cos(cos(x))
    const ddf = grad(df); // d^2/dx^2 sin(cos(x)) = -sin^2(x)sin(cos(x)) - cos(x)cos(cos(x))
    using r1 = df(3);
    expect(r1).toBeAllclose(-0.077432003);
    using r2 = ddf(3);
    expect(r2).toBeAllclose(0.559854311);
  });

  test("can compute grad of products", () => {
    using x = np.array([1, 2, 3, 4]);
    using gradProd = grad((x: np.Array) => np.prod(x))(x);
    expect(gradProd.js()).toEqual([24, 12, 8, 6]);
  });

  test("passing const argnums", () => {
    const f = (x: np.Array, y: [np.Array, np.Array]) =>
      x.mul(y[0]).add(y[1]).sum();
    const df_dx = grad(f, { argnums: 0 });
    const df_dy = grad(f, { argnums: 1 });
    using x = np.array([2, 3]);
    using y0 = np.array([4, 5]);
    using y1 = np.array([10, 20]);
    const y: [np.Array, np.Array] = [y0, y1];

    using r1 = df_dx(x, y);
    expect(r1.js()).toEqual([4, 5]); // dy/dx = y0
    const w = df_dy(x, y);
    using _w0 = w[0];
    using _w1 = w[1];
    expect(w[0].js()).toEqual([2, 3]); // dy/dy0 = x
    expect(w[1].js()).toEqual([1, 1]); // dy/dy1 = 1

    // Now try with a tuple of argnums
    const df_both = grad(f, { argnums: [1, 0] });
    const [[dy0, dy1], dx] = df_both(x, y);
    using _dx = dx;
    using _dy0 = dy0;
    using _dy1 = dy1;
    expect(dx.js()).toEqual([4, 5]);
    expect(dy0.js()).toEqual([2, 3]);
    expect(dy1.js()).toEqual([1, 1]);
  });

  test("backprops through auto-broadcast", () => {
    using x1 = np.array([[2], [4]]);
    using y1 = np.array([4, 5, 6]);
    const [dx, dy] = grad(([x, y]: [np.Array, np.Array]) => x.mul(y).sum())([
      x1,
      y1,
    ]);
    using _dx = dx;
    using _dy = dy;
    expect(dx.js()).toEqual([[15], [15]]);
    expect(dy.js()).toEqual([6, 6, 6]);

    using x2 = np.array([[2], [4]]);
    using y2 = np.array([4, 5, 6]);
    const [dx2, dy2] = grad(([x, y]: [np.Array, np.Array]) => x.add(y).sum())([
      x2,
      y2,
    ]);
    using _dx2 = dx2;
    using _dy2 = dy2;
    expect(dx2.js()).toEqual([[3], [3]]);
    expect(dy2.js()).toEqual([2, 2, 2]);
  });

  test("backprop handles dense layer", () => {
    type Params = {
      w: np.Array;
      b: np.Array;
    };

    // x is of shape [batch, dim_in]
    // params.w is of shape [dim_in, dim_out]
    // params.b is of shape [dim_out]

    // const dense = (params: Params, x: np.Array) =>
    //   np
    //     .multiply(x.reshape([...x.shape, 1]), params.w)
    //     .sum(-2)
    //     .add(params.b);

    const dense = (params: Params, x: np.Array) => {
      using dot = np.dot(x, params.w);
      return dot.add(params.b);
    };

    const loss = (params: Params, x: np.Array) =>
      nn.logSoftmax(dense(params, x)).slice([], 0).sum().mul(0.5);

    using w = np.array([
      [0.1, 0.2, -0.3, 0.0],
      [0.5, -0.1, 0.3, 0.4],
    ]);
    using b = np.array([0, 0, 0, 0]);
    const params: Params = { w, b };
    using x = np.array([
      [0.1, 0.2],
      [0.2, 0.3],
    ]);

    // These numbers are checked for consistency with JAX.
    using jitLoss = jit(loss);
    using lossVal = jitLoss(params, x);
    expect(lossVal).toBeAllclose(-1.3165712);

    // console.log(makeJaxpr(grad(loss))(params, x).jaxpr.toString());
    const grads = grad(loss)(params, x);
    using _gw = grads.w;
    using _gb = grads.b;
    expect(grads.w).toBeAllclose([
      [0.1095776, -0.03502218, -0.03585088, -0.03870453],
      [0.18276905, -0.05844341, -0.05986644, -0.06445917],
    ]);
    expect(grads.b).toBeAllclose([
      0.7319144, -0.2342123, -0.24015564, -0.25754642,
    ]);
  });

  test("hasAux returns aux and computes correct gradient", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      using loss = x.sum();
      using aux = x.mul(2);
      return [loss, aux];
    };

    using x = np.array([1, 2, 3]);
    const [gradient, aux] = grad(f, { hasAux: true })(x);
    using _gradient = gradient;
    using _aux = aux;

    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("hasAux handles pytree aux", () => {
    type Aux = { predictions: np.Array; squared: np.Array };
    const f = (x: np.Array): [np.Array, Aux] => {
      using loss = x.sum();
      const aux = {
        predictions: x.mul(2),
        squared: x.mul(x),
      };
      return [loss, aux];
    };

    using x = np.array([1, 2, 3]);
    const [gradient, aux] = grad(f, { hasAux: true })(x);
    using _gradient = gradient;
    using _pred = aux.predictions;
    using _sq = aux.squared;

    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux.predictions).toBeAllclose([2, 4, 6]);
    expect(aux.squared).toBeAllclose([1, 4, 9]);
  });

  test("hasAux gradients match grad without aux", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [x.sum(), x.mul(2)];
    const fWithoutAux = (x: np.Array) => x.sum();

    using x = np.array([1, 2, 3]);

    const [grad1, aux1] = grad(fWithAux, { hasAux: true })(x);
    using _grad1 = grad1;
    using _aux1 = aux1;
    using grad2 = grad(fWithoutAux)(x);

    expect(grad1).toBeAllclose(grad2);
  });

  test("hasAux works with jit wrapper", () => {
    using f = jit((x: np.Array): [np.Array, np.Array] => [x.sum(), x.mul(2)]);

    using x = np.array([1, 2, 3]);
    const [gradient, aux] = grad(f, { hasAux: true })(x);
    using _gradient = gradient;
    using _aux = aux;

    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("hasAux throws on non-scalar output", () => {
    const f = (x: np.Array): [np.Array, np.Array] => [x, x.mul(2)];
    using x = np.array([1, 2, 3]);
    expect(() => grad(f, { hasAux: true })(x)).toThrow("scalar");
  });

  test("hasAux throws on non-float dtype", () => {
    const f = (x: np.Array): [np.Array, np.Array] => [x.sum(), x.mul(2)];
    using x = np.array([1, 2, 3], { dtype: np.int32 });
    expect(() => grad(f, { hasAux: true })(x)).toThrow("floating-point");
  });

  test("grad is not stopped in aux values", () => {
    const f = grad((x: np.Array) => {
      const [one, xsquare] = grad(
        (y: np.Array): [np.Array, np.Array] => [y, y.mul(y)],
        { hasAux: true },
      )(x);
      one.dispose();
      return xsquare;
    });
    using r = f(10);
    expect(r).toBeAllclose(20);
  });
});

suite("jax.valueAndGrad()", () => {
  test("returns value and gradient", () => {
    const f = (x: np.Array) => x.mul(x).sum();
    using x = np.array([2, 3]);
    const [value, gradient] = valueAndGrad(f)(x);
    using _value = value;
    using _gradient = gradient;

    expect(value).toBeAllclose(13); // 4 + 9 = 13
    expect(gradient).toBeAllclose([4, 6]); // 2x = [4, 6]
  });

  test("hasAux returns value, gradient, and aux", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      using loss = x.sum();
      using aux = x.mul(2);
      return [loss, aux];
    };

    using x = np.array([1, 2, 3]);
    const [[value, aux], gradient] = valueAndGrad(f, { hasAux: true })(x);
    using _value = value;
    using _aux = aux;
    using _gradient = gradient;

    expect(value).toBeAllclose(6);
    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("hasAux matches valueAndGrad for value and gradient", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [x.sum(), x.mul(2)];
    const fWithoutAux = (x: np.Array) => x.sum();

    using x = np.array([1, 2, 3]);

    const [[value1, aux1], grad1] = valueAndGrad(fWithAux, { hasAux: true })(x);
    using _value1 = value1;
    using _aux1 = aux1;
    using _grad1 = grad1;
    const [value2, grad2] = valueAndGrad(fWithoutAux)(x);
    using _value2 = value2;
    using _grad2 = grad2;

    expect(value1).toBeAllclose(value2);
    expect(grad1).toBeAllclose(grad2);
  });
});

suite("jax.jit()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => {
      using tmp = x.mul(x);
      return tmp.mul(x);
    };
    using f2 = jit(f);
    using x1 = np.array(2);
    using r1 = f(x1);
    expect(r1).toBeAllclose(8);
    using x2 = np.array(2);
    using r2 = f2(x2);
    expect(r2).toBeAllclose(8);
  });

  test("works with identity function", () => {
    using f = jit((x: np.Array) => x);
    using _in = np.array(3);
    using a = f(_in);
    expect(a.js()).toEqual(3);
  });

  test("works with duplicate output", () => {
    using f = jit((x: np.Array) => [x, x]);
    using _in = np.array(3);
    const [a, b] = f(_in);
    using _a = a;
    using _b = b;
    expect(a.js()).toEqual(3);
    expect(b.js()).toEqual(3);
  });

  test("processes gather ops", () => {
    using indices = np.array([1, 3, 2, 0], { dtype: np.int32 });
    using f = jit((x: np.Array) => x.slice(indices));
    using _in = np.array([10, 20, 30, 40]);
    using a = f(_in);
    expect(a.js()).toEqual([20, 40, 30, 10]);
  });

  test("supports staticArgnums", () => {
    using f = jit((x: np.Array, idx: number) => x.slice(idx), {
      staticArgnums: [1],
    });
    using _in1 = np.arange(20);
    using r1 = f(_in1, 0);
    expect(r1.js()).toEqual(0);
    using _in2 = np.arange(20);
    using r2 = f(_in2, 3);
    expect(r2.js()).toEqual(3);
    using _in3 = np.array([30, 1, 20, 11]);
    using r3 = f(_in3, 3);
    expect(r3.js()).toEqual(11);
  });

  test("jit-of-jit", () => {
    using f = jit((x: np.Array) => x.mul(x));
    using g = jit((x: np.Array) => f(f(x)));
    using r1 = g(3);
    expect(r1).toBeAllclose(81);
    using h = jit(jit(g));
    using r2 = h(3);
    expect(r2).toBeAllclose(81);
  });

  test("jvp-of-jit", () => {
    using f = jit((x: np.Array) => x.mul(x));
    using p = np.array(3);
    using t = np.array(1);
    const [y, dy] = jvp(f, [p], [t]);
    using _y = y;
    using _dy = dy;
    expect(y).toBeAllclose(9);
    expect(dy).toBeAllclose(6);
  });

  test("grad-of-jit", () => {
    using f = jit((x: np.Array) => x.mul(x));

    using r1 = grad(f)(3);
    expect(r1).toBeAllclose(6);
    using r2 = grad(f)(10);
    expect(r2).toBeAllclose(20);
    // Wrap jvp(grad(jit)) and grad(grad(jit)) in outer jit to manage
    // internal intermediates (evalJaxprTransposed creates concrete arrays
    // inside the JVP trace that can't be disposed due to insideTrace guard).
    {
      using jvpGradF = jit((x: np.Array) => {
        const [yy, dyy] = jvp(grad(f), [x], [np.array(1)]);
        return np.stack([yy, dyy]);
      });
      using arg10 = np.array(10);
      using jvpResult = jvpGradF(arg10);
      {
        using s = jvpResult.slice(0);
        expect(s.js()).toBeCloseTo(20);
      }
      {
        using s = jvpResult.slice(1);
        expect(s.js()).toBeCloseTo(2);
      }
    }
    {
      using ggf = jit(grad(grad(f)));
      using r3 = ggf(10);
      expect(r3).toBeAllclose(2);
    }
    using gf = jit(grad(f));
    using r4 = grad(gf)(10);
    expect(r4).toBeAllclose(2);
  });

  test("vmap-of-jit", () => {
    using s = jit((x: np.Array) => x.sum());
    using ar = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using r1 = s(ar);
    expect(r1.js()).toEqual(21);
    using r2 = vmap(s)(ar);
    expect(r2.js()).toEqual([6, 15]);
    using r3 = vmap(s, 1)(ar);
    expect(r3.js()).toEqual([5, 7, 9]);
  });
});
