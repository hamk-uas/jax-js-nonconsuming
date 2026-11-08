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
  defaultDevice,
  init,
  jit,
  nn,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

const devices = await init("wasm");

const N = 4096;
/** Range [-2, 6): 75% positive. */
const ramp = (i: number) => (i / N) * 8 - 2;
const alternating = (i: number) => (i % 2 === 0 ? 1.0 : -1.0);
const mostlyTrue = (i: number) => (i < N * 0.9 ? 1.0 : -1.0);

const cases: [string, (i: number) => number, (x: np.Array) => np.Array][] = [
  // ELU: where(x > 0, x, alpha * (exp(x) - 1)). The expensive arm contains
  // exp(); branching should help when most values > 0.
  [`elu ${N} elements (75% positive)`, ramp, (x) => nn.elu(x)],
  [`selu ${N} elements (75% positive)`, ramp, (x) => nn.selu(x)],
  // Direct pattern: where(cond, x, exp(x)).
  [
    `where(cond, x, exp(x)) ${N} elements (50% true)`,
    alternating,
    (x) => {
      using cond = x.greater(0);
      return np.where(cond, x, np.exp(x));
    },
  ],
  [
    `where(cond, x, exp(x)) ${N} elements (90% true)`,
    mostlyTrue,
    (x) => {
      using cond = x.greater(0);
      return np.where(cond, x, np.exp(x));
    },
  ],
  // Both arms expensive.
  [
    `where(cond, exp(x), sin(x)) ${N} elements (50%)`,
    alternating,
    (x) => {
      using cond = x.greater(0);
      return np.where(cond, np.exp(x), np.sin(x));
    },
  ],
  // Control: cheap arms; should NOT benefit from branching.
  [`relu ${N} elements (control, cheap arms)`, ramp, (x) => nn.relu(x)],
];

describe.skipIf(!devices.includes("wasm"))("wasm where branching", () => {
  test("expensive vs cheap arms", async ({ bench }) => {
    defaultDevice("wasm");
    using stack = new DisposableStack();
    const registrations = cases.map(([name, values, body]) => {
      const x = stack.use(
        np.array(Array.from({ length: N }, (_, i) => values(i))),
      );
      const f = stack.use(jit(body));
      return bench(name, () => {
        f(x).dispose();
      });
    });
    await bench.compare(...registrations);
  });
});
