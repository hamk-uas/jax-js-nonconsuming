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
 *   pnpm build && pnpm vitest bench bench/command-tape.bench.ts
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
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))(
  "webgpu command tape + arena",
  async () => {
    defaultDevice("webgpu");

    // --- 5-step elementwise chain ---
    {
      const x = np.ones([4096]);
      await blockUntilReady(x);

      const chain5 = jit((x: np.Array) =>
        x.add(1).mul(2).sub(3).add(4).mul(0.5),
      );
      chain5(x).dispose(); // warmup (compiles tape)
      afterAll(() => {
        x.dispose();
        chain5.dispose();
      });

      bench("5-step chain size=4096", () => {
        chain5(x).dispose();
      });
    }

    // --- 20-step elementwise chain ---
    {
      const x = np.ones([4096]);
      await blockUntilReady(x);

      const chain20 = jit((x: np.Array) => {
        let y = x;
        for (let i = 0; i < 20; i++) y = y.add(1).mul(0.99);
        return y;
      });
      chain20(x).dispose(); // warmup
      afterAll(() => {
        x.dispose();
        chain20.dispose();
      });

      bench("20-step chain size=4096", () => {
        chain20(x).dispose();
      });
    }

    // --- Multi-output (3 outputs, same size) ---
    {
      const x = np.ones([4096]);
      await blockUntilReady(x);

      const multi3 = jit((x: np.Array) => [x.add(1), x.mul(2), x.sub(0.5)]);
      (multi3(x) as np.Array[]).forEach((r) => r.dispose()); // warmup
      afterAll(() => {
        x.dispose();
        multi3.dispose();
      });

      bench("3-output same-size size=4096", () => {
        (multi3(x) as np.Array[]).forEach((r: np.Array) => r.dispose());
      });
    }

    // --- grad() of 5-step chain ---
    {
      const x = np.ones([4096]);
      await blockUntilReady(x);

      const g = jit(grad((x: np.Array) => x.add(1).mul(2).sub(3).sum()));
      g(x).dispose(); // warmup
      afterAll(() => {
        x.dispose();
        g.dispose();
      });

      bench("grad(5-step chain) size=4096", () => {
        g(x).dispose();
      });
    }

    // --- matmul + bias + relu (inference-like) ---
    {
      const x = random.normal(random.key(0), [32, 128]);
      const w = random.normal(random.key(1), [128, 64]);
      const b = random.normal(random.key(2), [64]);
      await blockUntilReady([x, w, b]);

      const mlp = jit((x: np.Array, w: np.Array, b: np.Array) =>
        nn.relu(np.matmul(x, w).add(b)),
      );
      mlp(x, w, b).dispose(); // warmup
      afterAll(() => {
        x.dispose();
        w.dispose();
        b.dispose();
        mlp.dispose();
      });

      bench("matmul+bias+relu 32×128×64", () => {
        mlp(x, w, b).dispose();
      });
    }

    // --- 2-layer MLP chain ---
    {
      const x = random.normal(random.key(10), [16, 256]);
      const w1 = random.normal(random.key(11), [256, 128]);
      const b1 = random.normal(random.key(12), [128]);
      const w2 = random.normal(random.key(13), [128, 64]);
      const b2 = random.normal(random.key(14), [64]);
      await blockUntilReady([x, w1, b1, w2, b2]);

      const mlp2 = jit(
        (
          x: np.Array,
          w1: np.Array,
          b1: np.Array,
          w2: np.Array,
          b2: np.Array,
        ) => {
          const h = nn.relu(np.matmul(x, w1).add(b1));
          return np.matmul(h, w2).add(b2);
        },
      );
      mlp2(x, w1, b1, w2, b2).dispose(); // warmup
      afterAll(() => {
        x.dispose();
        w1.dispose();
        b1.dispose();
        w2.dispose();
        b2.dispose();
        mlp2.dispose();
      });

      bench("2-layer MLP 16×256→128→64", () => {
        mlp2(x, w1, b1, w2, b2).dispose();
      });
    }
  },
);
