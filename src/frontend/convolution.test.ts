import { expect, test } from "vitest";

import { AluExp, DType } from "../alu";
import { ShapeTracker } from "../shape";
import { range } from "../utils";
import { pool, poolTranspose } from "./convolution";

function evalShape(
  st: ShapeTracker,
): (...indices: number[]) => [number, boolean] {
  const vars = range(st.shape.length).map((i) =>
    AluExp.variable(DType.Int32, `i${i}`),
  );
  const [iexpr, vexpr] = st.toAluExp(vars);
  return (...indices) => {
    const sub = Object.fromEntries(vars.map((v, i) => [v.arg, indices[i]]));
    return [iexpr.evaluate(sub), Boolean(vexpr.evaluate(sub))];
  };
}

test("transpose of pool in 2d", () => {
  const st = ShapeTracker.fromShape([7, 12]);
  const pooled = pool(st, [2, 2], [1, 2]);
  expect(pooled.shape).toEqual([6, 6, 2, 2]);

  const pooledE = evalShape(pooled);
  expect(pooledE(0, 0, 0, 0)).toEqual([0, true]);
  expect(pooledE(0, 0, 0, 1)).toEqual([1, true]);
  expect(pooledE(0, 0, 1, 0)).toEqual([12, true]);
  expect(pooledE(0, 4, 1, 0)).toEqual([20, true]);

  const unpooled = poolTranspose(pooled, [7, 12], [2, 2], [1, 2]);
  expect(unpooled.shape.slice(0, 2)).toEqual(st.shape);
  expect(unpooled.shape).toMatchInlineSnapshot(`
    [
      7,
      12,
      3,
      5,
    ]
  `);

  const unpooledE = evalShape(unpooled);
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 12; j++) {
      const idx = i * 12 + j;
      for (let a = 0; a < unpooled.shape[2]; a++) {
        for (let b = 0; b < unpooled.shape[3]; b++) {
          const [x, v] = unpooledE(i, j, a, b);
          expect(!v || x === idx).toBe(true);
        }
      }
    }
  }
});
