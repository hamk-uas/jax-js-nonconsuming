/**
 * Deno WebGPU performance test: lax.associativeScan vs lax.scan
 *
 * Verifies that associativeScan is faster than lax.scan for a large-N
 * prefix-product task where GPU parallelism can be exploited.
 *
 * Why this task shows a clear speedup:
 *   - lax.scan compiled-loop: 1 GPU dispatch, but the WGSL shader places the
 *     N-step loop INSIDE the shader. With a scalar carry, exactly 1 GPU thread
 *     runs, doing N sequential multiply iterations with no CPU roundtrips.
 *     At N=32768 this single thread saturates the GPU pipeline for ~8 ms.
 *   - lax.associativeScan: only ceil(log2(32768)) = 15 parallel rounds.
 *     Each round dispatches a kernel over ALL ~N elements simultaneously,
 *     using many GPU threads in parallel. 15 dispatch roundtrips vs.
 *     32768 in-shader sequential iterations on a single thread mean the
 *     associative scan is ~8× faster for this N.
 *
 * The crossover point is where N × (GPU iter cost) > log2(N) × dispatch cost.
 * On this hardware dispatch cost ≈ 70µs, GPU iter ≈ 0.26µs/iter:
 *   scan:  32768 × 0.26µs + 50µs ≈ 8.5 ms per call
 *   assoc: 15 × 70µs           ≈ 1.1 ms per call  → ~8× speedup
 *
 * Run with:
 *   pnpm build && deno test --no-check --unstable-webgpu --allow-read --allow-env test/deno/associative-scan-perf.test.ts
 */

import { assert } from "https://deno.land/std@0.224.0/assert/mod.ts";

import { hasWebGPU, initWebGPU, withLeakCheck } from "./harness.ts";
import { blockUntilReady, jit, lax, numpy as np } from "../../dist/index.js";

// Number of prefix-product elements.
// ceil(log2(65536)) = 16 rounds for associativeScan vs 65536 sequential
// in-shader iterations for scan. Large enough that the single-thread GPU loop
// in scan's compiled-loop clearly exceeds log2(N) dispatch roundtrips.
const N = 65536;

// Number of timed iterations (after warmup).
// Fewer iterations needed since each scan call takes ~10ms.
const TIMED_ITERS = 10;

// Minimum speedup ratio we require: associativeScan must be this much faster.
// Predicted ~6× on this hardware; require 3× for headroom on slower GPUs.
const MIN_SPEEDUP = 3;

/**
 * Count compute dispatch calls while running `fn`.
 *
 * We instrument GPUComputePassEncoder dispatch methods, run `fn`, then restore
 * prototypes in a finally block to avoid cross-test side effects.
 */
async function countGpuDispatches(fn: () => Promise<void>): Promise<number> {
  let count = 0;
  const proto = GPUComputePassEncoder.prototype as any;
  const origDispatch = proto.dispatchWorkgroups;
  const origDispatchIndirect = proto.dispatchWorkgroupsIndirect;

  proto.dispatchWorkgroups = function (...args: any[]) {
    count += 1;
    return origDispatch.apply(this, args);
  };
  proto.dispatchWorkgroupsIndirect = function (...args: any[]) {
    count += 1;
    return origDispatchIndirect.apply(this, args);
  };

  try {
    await fn();
    return count;
  } finally {
    proto.dispatchWorkgroups = origDispatch;
    proto.dispatchWorkgroupsIndirect = origDispatchIndirect;
  }
}

/**
 * Measure wall-clock time (ms) for `iters` calls of `fn`.
 * `fn` must return the primary output array so we can `blockUntilReady` on it
 * as a GPU fence (Deno wgpu-rs has no OffscreenCanvas, so we need an async
 * readback-free fence — blockUntilReady on a real output suffices).
 * Runs `warmup` ignored calls first.
 */
async function timeMs(
  fn: () => any,
  warmup: number,
  iters: number,
): Promise<number> {
  // Warmup: run and await each call to fully flush the GPU pipeline.
  for (let i = 0; i < warmup; i++) {
    const out = fn();
    await blockUntilReady(out);
    out.dispose();
  }
  // Timed runs: enqueue all iterations then wait once at the end.
  const start = performance.now();
  let last: any;
  for (let i = 0; i < iters; i++) {
    if (last) last.dispose();
    last = fn();
  }
  await blockUntilReady(last);
  const elapsed = performance.now() - start;
  last.dispose();
  return elapsed;
}

Deno.test({
  name: "associativeScan is faster than scan for large-N prefix product on WebGPU",
  ignore: !hasWebGPU,
  fn: async () => {
    const ok = await initWebGPU();
    if (!ok) return;

    // Input: N scalars, all slightly above 1 so the product stays finite.
    // Shape [N, 1] for scan (leading axis = steps), shape [N] for associativeScan.
    // Using 1.0001^65536 ≈ 669, well within float32 range.
    const xsScan = np.full([N, 1], 1.0001);
    const xsAssoc = np.full([N], 1.0001);
    await blockUntilReady(xsScan);
    await blockUntilReady(xsAssoc);

    // -----------------------------------------------------------------------
    // lax.scan — compiled-loop on WebGPU.
    // Step: scalar carry *= scalar x.
    // With scalar carry (1 element), the compiled-loop WGSL shader dispatches
    // exactly 1 thread that loops N=65536 times sequentially inside the shader.
    // This single thread is the sequential bottleneck we are measuring.
    // -----------------------------------------------------------------------
    const initCarry = np.array([1.0]);

    const scanJit = jit((xs: any) =>
      lax.scan(
        (carry: any, x: any) => {
          const c = carry.mul(x);
          return [c, c];
        },
        initCarry,
        xs,
        { acceptPath: ["compiled-loop"] },
      ),
    );

    // timeMs expects fn() to return the primary output; we return `c` (scalar
    // carry) and dispose `y` internally since scan always produces both.
    const scanFn = () => {
      const [c, y] = scanJit(xsScan) as [any, any];
      y.dispose();
      return c; // caller (timeMs) disposes this
    };

    const scanMs = await timeMs(scanFn, /* warmup */ 3, TIMED_ITERS);

    // -----------------------------------------------------------------------
    // lax.associativeScan — Kogge-Stone, ceil(log2(N)) parallel rounds.
    // fn: (a, b) => a * b   (associative)
    //
    // Wrapped in jit() to avoid re-tracing on every call. Without jit,
    // each call would re-trace all ceil(log2(N)) rounds of the Kogge-Stone
    // ladder, making benchmark times dominated by compilation overhead.
    // -----------------------------------------------------------------------
    const assocJit = jit((xs: any) =>
      lax.associativeScan((a: any, b: any) => a.mul(b), xs),
    );

    const assocFn = () => assocJit(xsAssoc) as any;

    const assocMs = await timeMs(assocFn, /* warmup */ 3, TIMED_ITERS);

    const speedup = scanMs / assocMs;

    const rounds = Math.ceil(Math.log2(N));
    console.log(
      `\n  scan  (N=${N}, 1 dispatch, ${N} sequential GPU iters): ` +
        `${scanMs.toFixed(1)} ms / ${TIMED_ITERS} iters  (${(scanMs / TIMED_ITERS).toFixed(2)} ms/call)`,
    );
    console.log(
      `  assoc (N=${N}, ${rounds} parallel dispatches, Kogge-Stone): ` +
        `${assocMs.toFixed(1)} ms / ${TIMED_ITERS} iters  (${(assocMs / TIMED_ITERS).toFixed(2)} ms/call)`,
    );
    console.log(
      `  speedup: ${speedup.toFixed(1)}× (required ≥ ${MIN_SPEEDUP}×)`,
    );

    xsScan.dispose();
    xsAssoc.dispose();
    initCarry.dispose();
    scanJit.dispose();
    assocJit.dispose();

    assert(
      speedup >= MIN_SPEEDUP,
      `associativeScan speedup ${speedup.toFixed(1)}× is below minimum ${MIN_SPEEDUP}×.\n` +
        `  scan:  ${scanMs.toFixed(1)} ms\n` +
        `  assoc: ${assocMs.toFixed(1)} ms`,
    );
  },
});

Deno.test({
  name: "associativeScan dispatch count is near log2(N) rounds on WebGPU",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const ok = await initWebGPU();
    if (!ok) return;

    const n = 1024;
    const expectedRounds = Math.ceil(Math.log2(n));

    using xs = np.full([n], 1.0001);
    using assocJit = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => a.mul(b), x),
    );

    // Warm up compilation and pipeline caches before counting dispatches.
    {
      using warm = assocJit(xs);
      await blockUntilReady(warm);
    }

    const dispatches = await countGpuDispatches(async () => {
      using y = assocJit(xs);
      await blockUntilReady(y);
    });

    // Lower bound: at least one dispatch per Kogge-Stone round.
    assert(
      dispatches >= expectedRounds,
      `associativeScan dispatches ${dispatches} is below expected rounds ${expectedRounds}`,
    );

    // Upper bound: allow some overhead from bookkeeping/copy kernels, but
    // ensure growth remains O(log N) rather than O(N).
    assert(
      dispatches <= expectedRounds * 3,
      `associativeScan dispatches ${dispatches} too high for N=${n}; expected near log2(N)=${expectedRounds}`,
    );
  }),
});

Deno.test({
  name: "associativeScan dispatch growth is logarithmic between nearby powers of two",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const ok = await initWebGPU();
    if (!ok) return;

    const n1 = 512;
    const n2 = 1024;

    using xs1 = np.full([n1], 1.0001);
    using xs2 = np.full([n2], 1.0001);
    using assocJit = jit((x: any) =>
      lax.associativeScan((a: any, b: any) => a.mul(b), x),
    );

    // Warm per-shape caches so counting excludes compile/setup work.
    {
      using w1 = assocJit(xs1);
      await blockUntilReady(w1);
      using w2 = assocJit(xs2);
      await blockUntilReady(w2);
    }

    const d1 = await countGpuDispatches(async () => {
      using y1 = assocJit(xs1);
      await blockUntilReady(y1);
    });
    const d2 = await countGpuDispatches(async () => {
      using y2 = assocJit(xs2);
      await blockUntilReady(y2);
    });

    const rounds1 = Math.ceil(Math.log2(n1));
    const rounds2 = Math.ceil(Math.log2(n2));
    const roundDelta = rounds2 - rounds1; // = 1

    assert(
      d2 >= d1,
      `dispatches should be non-decreasing with N: N=${n1} -> ${d1}, N=${n2} -> ${d2}`,
    );

    // Doubling N adds exactly one Kogge-Stone round. Allow a small constant
    // slack for backend bookkeeping kernels while still ruling out linear growth.
    assert(
      d2 <= d1 + 6,
      `dispatch growth too high for doubling N: N=${n1}(${d1}) -> N=${n2}(${d2}); rounds +${roundDelta}`,
    );
  }),
});
