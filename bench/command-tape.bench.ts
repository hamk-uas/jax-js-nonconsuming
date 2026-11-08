/**
 * O8+O9 Command tape + arena benchmark on WebGPU.
 *
 * Measures per-invocation latency for JIT programs that compile to command
 * tapes (kernel-only, concrete sizes). Compares:
 *   - Short chains (5 dispatches)
 *   - Medium chains (20 dispatches)
 *   - Multi-output programs
 *   - grad() backward passes (~10-15 dispatches for a 5-step chain)
 *   - Matmul + elementwise epilogues
 *
 * Run with:
 *   pnpm vitest bench bench/command-tape.bench.ts
 */
import {
  blockUntilReady,
  defaultDevice,
  grad,
  init,
  jit,
  nn,
  numpy as np,
  random,
} from "@hamk-uas/jax-js-nonconsuming";
import { describe, test } from "vitest";

const devices = await init("webgpu");

describe.skipIf(!devices.includes("webgpu"))(
  "webgpu command tape + arena",
  () => {
    test("jit programs", async ({ bench }) => {
      defaultDevice("webgpu");
      using stack = new DisposableStack();
      const normal = (seed: number, shape: number[]) =>
        stack.use(random.normal(stack.use(random.key(seed)), shape));

      using x = np.ones([4096]);
      // matmul + bias + relu (inference-like)
      const [mx, mw, mb] = [
        normal(0, [32, 128]),
        normal(1, [128, 64]),
        normal(2, [64]),
      ];
      // 2-layer MLP chain
      const [px, pw1, pb1, pw2, pb2] = [
        normal(10, [16, 256]),
        normal(11, [256, 128]),
        normal(12, [128]),
        normal(13, [128, 64]),
        normal(14, [64]),
      ];
      await blockUntilReady([x, mx, mw, mb, px, pw1, pb1, pw2, pb2]);

      using chain5 = jit((v: np.Array) =>
        v.add(1).mul(2).sub(3).add(4).mul(0.5),
      );
      using chain20 = jit((v: np.Array) => {
        let y = v;
        for (let i = 0; i < 20; i++) y = y.add(1).mul(0.99);
        return y;
      });
      using multi3 = jit((v: np.Array) => [v.add(1), v.mul(2), v.sub(0.5)]);
      using g = jit(grad((v: np.Array) => v.add(1).mul(2).sub(3).sum()));
      using mlp = jit((a: np.Array, w: np.Array, b: np.Array) =>
        nn.relu(np.matmul(a, w).add(b)),
      );
      using mlp2 = jit(
        (
          a: np.Array,
          w1: np.Array,
          b1: np.Array,
          w2: np.Array,
          b2: np.Array,
        ) => {
          const h = nn.relu(np.matmul(a, w1).add(b1));
          return np.matmul(h, w2).add(b2);
        },
      );

      await bench.compare(
        bench("5-step chain size=4096", () => {
          chain5(x).dispose();
        }),
        bench("20-step chain size=4096", () => {
          chain20(x).dispose();
        }),
        bench("3-output same-size size=4096", () => {
          for (const r of multi3(x)) r.dispose();
        }),
        bench("grad(5-step chain) size=4096", () => {
          g(x).dispose();
        }),
        bench("matmul+bias+relu 32×128×64", () => {
          mlp(mx, mw, mb).dispose();
        }),
        bench("2-layer MLP 16×256→128→64", () => {
          mlp2(px, pw1, pb1, pw2, pb2).dispose();
        }),
      );
    });
  },
);
