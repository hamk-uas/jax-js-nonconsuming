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
 *   scripts/gpu-test.sh bench bench/calibration.bench.ts
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
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  test,
  type TestContext,
} from "vitest";

const devices = await init("webgpu");

describe.skipIf(!devices.includes("webgpu"))(
  "calibrated vs conservative",
  () => {
    let measured: Parameters<typeof _applyCalibration>[0];

    beforeAll(async () => {
      defaultDevice("webgpu");
      await calibrateGpu();
      const caps = getBackend("webgpu").capabilities;
      measured = {
        dispatchOverheadUs: caps.dispatchOverheadUs!,
        bandwidthGBs: caps.bandwidthGBs!,
        tflops: caps.tflops!,
        barrierCostFactor: caps.barrierCostFactor!,
        rOptWords: caps.rOptWords!,
      };
      console.log("calibrated values:", measured);
      _resetCalibration();
    });
    afterEach(() => _resetCalibration());
    afterAll(() => _resetCalibration());

    // JIT cache keys include the calibrated values, so one jit function
    // compiles separately under each setting; warmup absorbs the compile.
    // The tasks of one compare() run one after another, and a task's
    // beforeAll runs before both its warmup and its measured phase, so every
    // call sees its own setting.
    function compareSettings(
      bench: TestContext["bench"],
      run: () => Promise<void>,
    ) {
      return bench.compare(
        bench("conservative", { beforeAll: () => _resetCalibration() }, run),
        bench(
          "calibrated",
          { beforeAll: () => _applyCalibration(measured) },
          run,
        ),
      );
    }

    test("sum N=1e6", async ({ bench }) => {
      defaultDevice("webgpu");
      using x = np.ones([1_000_000]);
      using sum = jit((v: np.Array) => np.sum(v));
      await compareSettings(bench, async () => {
        using r = sum(x);
        await r.blockUntilReady();
      });
    });

    test("foriLoop 128×128 ×20", async ({ bench }) => {
      defaultDevice("webgpu");
      using a = np.ones([128, 128]);
      using b = np.ones([128, 128]);
      using fori = jit((p: np.Array, q: np.Array): [np.Array, np.Array] =>
        lax.foriLoop(
          0,
          20,
          (
            _i: np.Array,
            [u, w]: [np.Array, np.Array],
          ): [np.Array, np.Array] => {
            using du = w.mul(0.01);
            using dw = u.mul(0.01);
            return [u.add(du), w.sub(dw)];
          },
          [p, q] as [np.Array, np.Array],
        ),
      );
      await compareSettings(bench, async () => {
        const [ra, rb] = fori(a, b);
        await blockUntilReady([ra, rb]);
        ra.dispose();
        rb.dispose();
      });
    });

    test("associativeScan cumsum N=1e4", async ({ bench }) => {
      defaultDevice("webgpu");
      using xs = np.ones([10_000]);
      using cumsum = jit((v: np.Array) =>
        lax.associativeScan((s: np.Array, t: np.Array) => s.add(t), v),
      );
      await compareSettings(bench, async () => {
        using r = cumsum(xs);
        await r.blockUntilReady();
      });
    });
  },
);
