/**
 * Scan backward pass artifact for `grad(scan)`.
 *
 * Encapsulates the checkpoint-based backward pass through a scan operation,
 * including forward recomputation (√N checkpointing) and backward transposition.
 *
 * @module
 */
// jax-js-lint: allow-ref — .ref is essential for evalJaxpr input ownership

import { add, broadcast, concatenate, reshape, shrink, Tracer } from "./core";
import { ClosedJaxpr, evalJaxpr } from "./jaxpr";

// ---------------------------------------------------------------------------
// ScanBackwardSpec — compile-time info for the scan backward pass
// ---------------------------------------------------------------------------

/**
 * All compile-time (structural/shape) information needed to execute a scan
 * backward pass.  Built once per unique scan shape, then reused across calls.
 */
export interface ScanBackwardSpec {
  /** Forward-only body jaxpr (locally-owned — disposed by the artifact). */
  primalForwardJaxpr: ClosedJaxpr;

  /**
   * Tangent-only body jaxpr (locally-owned — disposed by the artifact).
   * Created via `makeJaxpr`, needed only as an intermediate to produce
   * `transposedBody` but must be disposed to free its const zeros.
   */
  tangentBody: ClosedJaxpr;

  /**
   * Transposed tangent body jaxpr (**cache-owned** — NOT disposed by the
   * artifact). Produced by `transposeJaxpr()` and cached in
   * `transposeJaxprCache`; the cache handles disposal via
   * `_disposeAllJitCaches` in `checkLeaks.stop()`.
   */
  transposedBody: ClosedJaxpr;

  // Dimension info (all counts refer to the scan body's input/output layout)
  numConsts: number;
  numCarry: number;
  numY: number;
  numPrimalCarry: number;
  numPrimalY: number;
  numPrimalX: number;
  numTangentConsts: number;
  numTangentCarry: number;
  numTangentX: number;

  // Scan parameters
  length: number;
  reverse: boolean;
  checkpoint: boolean | number | undefined;

  /** Which original scan args are tangent (true) vs primal (false). */
  undefMask: boolean[];

  /** Which original scan args are `UndefPrimal` instances. */
  actualUndefMask: boolean[];
}

// ---------------------------------------------------------------------------
// ScanPullbackArtifact
// ---------------------------------------------------------------------------

/**
 * Owns the residual arrays for a scan backward pass and executes the
 * checkpoint-based backward loop.
 *
 * **Lifecycle:**
 *   1. Created by the `Primitive.Scan` transpose rule with residuals + spec.
 *   2. `.run(cts)` is called once to produce input cotangents.
 *   3. `[Symbol.dispose]()` frees remaining owned resources.
 *
 * Residuals are consumed (disposed) during `.run()`.  Calling `[Symbol.dispose]()`
 * after `.run()` is safe (idempotent).
 */
export class ScanPullbackArtifact {
  #disposed = false;

  constructor(
    readonly spec: ScanBackwardSpec,
    private constResiduals: Tracer[],
    private carryResiduals: Tracer[],
    private xsResiduals: Tracer[],
  ) {}

  // -------------------------------------------------------------------------
  // Forward step: advance one iteration of the primal scan
  // -------------------------------------------------------------------------
  private forwardStep(iter: number, carry: Tracer[]): Tracer[] {
    const { reverse, length, numPrimalCarry } = this.spec;
    const { primalForwardJaxpr } = this.spec;
    const dataIdx = reverse ? length - 1 - iter : iter;

    // Slice xs residuals for this iteration
    const xSlices: Tracer[] = [];
    for (const xs of this.xsResiduals) {
      const slice = shrink(xs, [
        [dataIdx, dataIdx + 1],
        ...xs.shape
          .slice(1)
          .map((_, i) => [0, xs.shape[i + 1]] as [number, number]),
      ]);
      const reshaped = reshape(slice, xs.shape.slice(1));
      slice.dispose();
      xSlices.push(reshaped);
    }

    const forwardInputs = [
      ...this.constResiduals.map((c) => c.ref),
      ...carry.map((c) => c.ref),
      ...xSlices,
    ];
    const forwardOuts = evalJaxpr(primalForwardJaxpr.jaxpr, [
      ...primalForwardJaxpr.consts.map((c) => c.ref),
      ...forwardInputs,
    ]);
    const newCarry = forwardOuts.slice(0, numPrimalCarry);
    for (let i = numPrimalCarry; i < forwardOuts.length; i++) {
      forwardOuts[i].dispose();
    }
    for (const x of xSlices) {
      if (x.refCount > 0) x.dispose();
    }
    return newCarry;
  }

  // -------------------------------------------------------------------------
  // Backward step: transpose one iteration
  // -------------------------------------------------------------------------
  private backwardStep(
    iter: number,
    primalCarry: Tracer[],
    ctCarryRunning: Tracer[],
    ctConstsAccum: Tracer[] | null,
    ctXsAccum: Tracer[][],
    ctYsAll: Tracer[],
  ): { ctCarryNew: Tracer[]; ctConstsAccum: Tracer[] | null } {
    const {
      reverse,
      length,
      numY,
      numCarry,
      numPrimalCarry,
      numTangentConsts,
      numTangentX,
      transposedBody,
    } = this.spec;
    const dataIdx = reverse ? length - 1 - iter : iter;

    // Slice primal xs for this iteration
    const xSlices: Tracer[] = [];
    for (const xs of this.xsResiduals) {
      const slice = shrink(xs, [
        [dataIdx, dataIdx + 1],
        ...xs.shape
          .slice(1)
          .map((_, i) => [0, xs.shape[i + 1]] as [number, number]),
      ]);
      const reshaped = reshape(slice, xs.shape.slice(1));
      slice.dispose();
      xSlices.push(reshaped);
    }

    // Slice cotangent of y for this iteration
    // numY is doubled by JVP (primal + tangent outputs); the tangent y's
    // are in the second half. Assert the invariant that JVP doubling
    // produces an even numY.
    if (numY % 2 !== 0) {
      throw new Error(
        `scan backward: expected even numY from JVP doubling, got ${numY}`,
      );
    }
    const ctYSlices: Tracer[] = [];
    for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++) {
      const ctY = ctYsAll[i];
      const slice = shrink(ctY, [
        [dataIdx, dataIdx + 1],
        ...ctY.shape
          .slice(1)
          .map((_, j) => [0, ctY.shape[j + 1]] as [number, number]),
      ]);
      const reshaped = reshape(slice, ctY.shape.slice(1));
      slice.dispose();
      ctYSlices.push(reshaped);
    }

    // Build cotangents for tangentBody outputs
    const bodyOutCotangents: Tracer[] = [];
    bodyOutCotangents.push(...ctCarryRunning.map((c) => c.ref));
    bodyOutCotangents.push(...ctYSlices);

    // Run transposed body
    const transposedInputs = [
      ...transposedBody.consts.map((c) => c.ref),
      ...this.constResiduals.map((c) => c.ref),
      ...primalCarry.map((c) => c.ref),
      ...xSlices,
      ...bodyOutCotangents,
    ];

    const transposedOuts = evalJaxpr(transposedBody.jaxpr, transposedInputs);

    // Extract cotangents
    let outIdx = 0;
    const ctConstsIter: Tracer[] = [];
    for (let i = 0; i < numTangentConsts; i++) {
      ctConstsIter.push(transposedOuts[outIdx++]);
    }

    const ctCarryNew: Tracer[] = [];
    const numTangentCarryLocal = numCarry - numPrimalCarry;
    for (let i = 0; i < numTangentCarryLocal; i++) {
      ctCarryNew.push(transposedOuts[outIdx++]);
    }

    const ctXIter: Tracer[] = [];
    for (let i = 0; i < numTangentX; i++) {
      ctXIter.push(transposedOuts[outIdx++]);
    }

    // Accumulate const cotangents
    let newCtConstsAccum: Tracer[] | null;
    if (ctConstsAccum === null) {
      newCtConstsAccum = ctConstsIter;
    } else {
      const next: Tracer[] = [];
      for (let i = 0; i < ctConstsAccum.length; i++) {
        const summed = add(ctConstsAccum[i], ctConstsIter[i]);
        ctConstsAccum[i].dispose();
        ctConstsIter[i].dispose();
        next.push(summed);
      }
      newCtConstsAccum = next;
    }

    // Store x cotangents (will stack later)
    for (let i = 0; i < numTangentX; i++) {
      ctXsAccum[i].push(ctXIter[i]);
    }

    // Dispose old carry cotangent
    for (const c of ctCarryRunning) c.dispose();

    return { ctCarryNew, ctConstsAccum: newCtConstsAccum };
  }

  // -------------------------------------------------------------------------
  // run() — execute the full checkpoint-based backward pass
  // -------------------------------------------------------------------------

  /**
   * Execute the scan backward pass and return input cotangents.
   *
   * @param cts Cotangents for scan outputs: `[ct_carry..., ct_ys...]`
   * @returns Array matching the scan's input args, with `null` for known primals.
   */
  run(cts: Tracer[]): (Tracer | null)[] {
    const {
      numCarry,
      numY,
      numPrimalCarry,
      numTangentX,
      length,
      reverse,
      checkpoint,
      undefMask,
      actualUndefMask,
      numConsts,
    } = this.spec;

    // ---- Step 1: Forward pass to collect checkpoint carries ----

    const useCheckpointing = checkpoint !== false;
    const segmentSize = useCheckpointing
      ? typeof checkpoint === "number"
        ? checkpoint
        : Math.max(1, Math.ceil(Math.sqrt(length)))
      : length;

    const allCarries: Tracer[][] | null = useCheckpointing ? null : [];
    const checkpointCarries: Map<number, Tracer[]> | null = useCheckpointing
      ? new Map()
      : null;

    {
      let currentCarry = this.carryResiduals.map((c) => c.ref);
      if (allCarries) {
        allCarries.push(currentCarry.map((c) => c.ref));
      } else {
        checkpointCarries!.set(
          0,
          currentCarry.map((c) => c.ref),
        );
      }

      for (let iter = 0; iter < length; iter++) {
        const newCarry = this.forwardStep(iter, currentCarry);
        for (const c of currentCarry) c.dispose();
        currentCarry = newCarry;

        if (allCarries) {
          allCarries.push(currentCarry.map((c) => c.ref));
        } else if ((iter + 1) % segmentSize === 0) {
          checkpointCarries!.set(
            iter + 1,
            currentCarry.map((c) => c.ref),
          );
        }
      }
      for (const c of currentCarry) c.dispose();
    }

    // ---- Step 2: Backward pass ----

    const ctCarryAll = cts.slice(0, numCarry);
    const ctYsAll = cts.slice(numCarry);

    // Initialize running cotangent for carry (tangent carry cotangents only)
    let ctCarryRunning = ctCarryAll.slice(numPrimalCarry).map((c) => c.ref);
    // Dispose primal carry cotangents
    for (let i = 0; i < numPrimalCarry; i++) ctCarryAll[i].dispose();

    // Accumulate cotangents for xs and consts
    const ctXsAccum: Tracer[][] = [];
    for (let i = 0; i < numTangentX; i++) {
      ctXsAccum.push([]);
    }
    let ctConstsAccum: Tracer[] | null = null;

    if (useCheckpointing) {
      const numSegments = Math.ceil(length / segmentSize);

      for (let seg = numSegments - 1; seg >= 0; seg--) {
        const segStart = seg * segmentSize;
        const segEnd = Math.min(segStart + segmentSize, length);

        // Recompute carries for this segment from checkpoint
        const segCarries: Tracer[][] = [];
        let carry = checkpointCarries!.get(segStart)!.map((c) => c.ref);
        segCarries.push(carry.map((c) => c.ref));

        for (let iter = segStart; iter < segEnd - 1; iter++) {
          const newCarry = this.forwardStep(iter, carry);
          for (const c of carry) c.dispose();
          carry = newCarry;
          segCarries.push(carry.map((c) => c.ref));
        }
        for (const c of carry) c.dispose();

        // Process segment backward
        for (let iter = segEnd - 1; iter >= segStart; iter--) {
          const localIdx = iter - segStart;
          const result = this.backwardStep(
            iter,
            segCarries[localIdx],
            ctCarryRunning,
            ctConstsAccum,
            ctXsAccum,
            ctYsAll,
          );
          ctCarryRunning = result.ctCarryNew;
          ctConstsAccum = result.ctConstsAccum;
          for (const c of segCarries[localIdx]) c.dispose();
        }

        // Dispose checkpoint
        for (const c of checkpointCarries!.get(segStart)!) c.dispose();
        checkpointCarries!.delete(segStart);
      }

      // Dispose any remaining checkpoints
      for (const [, carries] of checkpointCarries!) {
        for (const c of carries) c.dispose();
      }
    } else {
      for (let iter = length - 1; iter >= 0; iter--) {
        const result = this.backwardStep(
          iter,
          allCarries![iter],
          ctCarryRunning,
          ctConstsAccum,
          ctXsAccum,
          ctYsAll,
        );
        ctCarryRunning = result.ctCarryNew;
        ctConstsAccum = result.ctConstsAccum;
        for (const c of allCarries![iter]) c.dispose();
      }
      // Dispose the last allCarries entry
      for (const c of allCarries![length]) c.dispose();
    }

    // Dispose remaining cotangent Y arrays
    for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++)
      ctYsAll[i].dispose();
    for (let i = 0; i < Math.floor(numY / 2); i++) ctYsAll[i].dispose();

    // ---- Step 3: Stack x cotangents ----

    const ctXsStacked: Tracer[] = [];
    for (let i = 0; i < numTangentX; i++) {
      const reversed = ctXsAccum[i].reverse();
      if (reverse) reversed.reverse();
      const expanded = reversed.map((ct) =>
        broadcast(ct, [1, ...ct.shape], [0]),
      );
      const stacked = concatenate(expanded, 0);
      const disposed = new Set<Tracer>();
      for (const ct of expanded) {
        if (!disposed.has(ct)) {
          disposed.add(ct);
          ct.dispose();
        }
      }
      for (const ct of reversed) {
        if (!disposed.has(ct)) {
          disposed.add(ct);
          ct.dispose();
        }
      }
      ctXsStacked.push(stacked);
    }

    // ---- Step 4: Assemble output cotangents ----

    const result: (Tracer | null)[] = [];
    let ctConstIdx = 0;
    let ctCarryIdx = 0;
    let ctXIdx = 0;

    for (let i = 0; i < actualUndefMask.length; i++) {
      const isJvpTangent = undefMask[i];

      if (!actualUndefMask[i]) {
        // This arg is a known primal (Tracer), return null
        if (isJvpTangent) {
          if (i < numConsts) {
            ctConstsAccum![ctConstIdx++].dispose();
          } else if (i < numConsts + numCarry) {
            ctCarryRunning[ctCarryIdx++].dispose();
          } else {
            ctXsStacked[ctXIdx++].dispose();
          }
        }
        result.push(null);
      } else if (i < numConsts) {
        result.push(ctConstsAccum![ctConstIdx++]);
      } else if (i < numConsts + numCarry) {
        result.push(ctCarryRunning[ctCarryIdx++]);
      } else {
        result.push(ctXsStacked[ctXIdx++]);
      }
    }

    // Dispose any remaining accumulated cotangents
    const remainingCtConsts: Tracer[] = ctConstsAccum ?? [];
    for (let i = ctConstIdx; i < remainingCtConsts.length; i++) {
      remainingCtConsts[i].dispose();
    }
    for (let i = ctCarryIdx; i < ctCarryRunning.length; i++) {
      ctCarryRunning[i].dispose();
    }
    for (let i = ctXIdx; i < ctXsStacked.length; i++) {
      ctXsStacked[i].dispose();
    }

    // ---- Step 5: Cleanup residuals (consumed) ----
    this.disposeResiduals();

    return result;
  }

  // -------------------------------------------------------------------------
  // Disposal
  // -------------------------------------------------------------------------

  private disposeResiduals(): void {
    if (this.#disposed) return;
    this.#disposed = true;

    // Dispose locally-owned jaxprs
    this.spec.primalForwardJaxpr.dispose();
    this.spec.tangentBody.dispose();
    // NOTE: transposedBody is cache-owned — NOT disposed here.

    for (const c of this.constResiduals) c.dispose();
    for (const c of this.carryResiduals) c.dispose();
    for (const c of this.xsResiduals) c.dispose();
    this.constResiduals = [];
    this.carryResiduals = [];
    this.xsResiduals = [];
  }

  [Symbol.dispose](): void {
    this.disposeResiduals();
  }
}
