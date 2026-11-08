/**
 * Calibrated vs conservative cost-model decisions on WebGPU.
 *
 * `calibrateGpu()` is opt-in (REMEDIATION-PLAN.md, R22). This benchmark
 * measures what it buys: each case runs once with programs compiled under the
 * conservative defaults and once under the values measured by
 * `calibrateGpu()`. The cases exercise the cost-model consumers: reduction
 * tuning (tuneWebgpu), foriLoop→BlockMap block shape, and associativeScan
 * block size.
 *
 * Run on both GPUs:
 *   pnpm build && scripts/gpu-test.sh bench bench/calibration.bench.ts
 */
import {
  _applyCalibration,
  _resetCalibration,
  blockUntilReady,
  calibrateGpu,
  defaultDevice,
  getBackend,
  init,
  jit,
  lax,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))(
  "calibrated vs conservative",
  async () => {
    defaultDevice("webgpu");

    await calibrateGpu();
    const caps = getBackend("webgpu").capabilities;
    const measured = {
      dispatchOverheadUs: caps.dispatchOverheadUs!,
      bandwidthGBs: caps.bandwidthGBs!,
      tflops: caps.tflops!,
      barrierCostFactor: caps.barrierCostFactor!,
      rOptWords: caps.rOptWords!,
    };
    console.log("calibrated values:", measured);
    _resetCalibration();

    // JIT cache keys include the calibrated values, so one jit function
    // compiles separately under each setting; warmup absorbs the compile.
    const conservative = { setup: () => _resetCalibration() };
    const calibrated = { setup: () => _applyCalibration(measured) };

    const x = np.ones([1_000_000]);
    const a = np.ones([128, 128]);
    const b = np.ones([128, 128]);
    const xs = np.ones([10_000]);
    afterAll(() => {
      x.dispose();
      a.dispose();
      b.dispose();
      xs.dispose();
      _resetCalibration();
    });

    const sum = jit((v: np.Array) => np.sum(v));
    const fori = jit((p: np.Array, q: np.Array): [np.Array, np.Array] =>
      lax.foriLoop(
        0,
        20,
        (_i: np.Array, [u, w]: [np.Array, np.Array]): [np.Array, np.Array] => {
          using du = w.mul(0.01);
          using dw = u.mul(0.01);
          return [u.add(du), w.sub(dw)];
        },
        [p, q] as [np.Array, np.Array],
      ),
    );
    const cumsum = jit((v: np.Array) =>
      lax.associativeScan((s: np.Array, t: np.Array) => s.add(t), v),
    );
    afterAll(() => {
      sum.dispose();
      fori.dispose();
      cumsum.dispose();
    });

    const runSum = async () => {
      using r = sum(x);
      await r.blockUntilReady();
    };
    const runFori = async () => {
      const [ra, rb] = fori(a, b);
      await blockUntilReady([ra, rb]);
      ra.dispose();
      rb.dispose();
    };
    const runCumsum = async () => {
      using r = cumsum(xs);
      await r.blockUntilReady();
    };

    const cases: [string, () => Promise<void>][] = [
      ["sum N=1e6", runSum],
      ["foriLoop 128×128 ×20", runFori],
      ["associativeScan cumsum N=1e4", runCumsum],
    ];
    for (const [name, run] of cases) {
      suite(name, () => {
        bench("conservative", run, conservative);
        bench("calibrated", run, calibrated);
      });
    }
  },
);
