/**
 * @file AOT Linearization Artifacts — explicit lifecycle management for autodiff.
 *
 * These artifact types replace ad-hoc PE intermediate cleanup with scoped
 * ownership: each artifact owns exactly the resources it needs, implements
 * `[Symbol.dispose]`, and can be cleaned up deterministically.
 *
 * See AOT-LINEARIZATION-PLAN.md for design rationale and milestone tracking.
 */
// jax-js-lint: allow-ref

import type { Tracer } from "./core";
import type { ClosedJaxpr } from "./jaxpr";
import { evalJaxpr } from "./jaxpr";
import { buildBackwardJaxpr, linearizeFlatUtil } from "./linearize";

// ---------------------------------------------------------------------------
// ResidualPack — owns concrete residual arrays from the forward pass
// ---------------------------------------------------------------------------

/** Concrete residual arrays produced by PrimalArtifact.run(). */
export interface ResidualPack extends Disposable {
  /** The residual arrays. Throws UseAfterFreeError if already disposed. */
  readonly arrays: Tracer[];
  /** Whether this pack has been disposed. */
  readonly consumed: boolean;
}

/** Concrete implementation of ResidualPack. */
class ResidualPackImpl implements ResidualPack {
  #arrays: Tracer[];
  #disposed = false;

  constructor(arrays: Tracer[]) {
    this.#arrays = arrays;
  }

  get arrays(): Tracer[] {
    if (this.#disposed)
      throw new ReferenceError("ResidualPack already disposed");
    return this.#arrays;
  }

  get consumed(): boolean {
    return this.#disposed;
  }

  [Symbol.dispose](): void {
    if (!this.#disposed) {
      for (const a of this.#arrays) a.dispose();
      this.#disposed = true;
    }
  }
}

// ---------------------------------------------------------------------------
// PrimalArtifact — owns the forward jaxpr + stored primal outputs
// ---------------------------------------------------------------------------

/** Forward-pass artifact produced by aotLinearize(). */
export interface PrimalArtifact extends Disposable {
  /** The forward-pass jaxpr (tangent computation from partial evaluation). */
  readonly forwardJaxpr: ClosedJaxpr;
  /**
   * Return the primal outputs and a ResidualPack for the backward pass.
   *
   * Currently returns stored results from the initial aotLinearize trace.
   * The `primals` argument is reserved for future reusable forward-pass
   * support.
   */
  run(primals: Tracer[]): { primalsOut: Tracer[]; residuals: ResidualPack };
}

/**
 * PrimalArtifact implementation.
 *
 * Owns:
 * - The forward jaxpr (ClosedJaxpr) — whose consts are the residual arrays
 * - Stored primal outputs from the initial trace
 *
 * Each `run()` call creates a new ResidualPack with .ref'd copies of the
 * residual arrays. The pack holds independent ownership — callers must
 * dispose it when done.
 */
class PrimalArtifactImpl implements PrimalArtifact {
  #forwardJaxpr: ClosedJaxpr;
  #storedPrimalsOut: Tracer[];
  #disposed = false;

  constructor(forwardJaxpr: ClosedJaxpr, primalsOut: Tracer[]) {
    this.#forwardJaxpr = forwardJaxpr;
    this.#storedPrimalsOut = primalsOut;
  }

  get forwardJaxpr(): ClosedJaxpr {
    if (this.#disposed)
      throw new ReferenceError("PrimalArtifact already disposed");
    return this.#forwardJaxpr;
  }

  run(_primals: Tracer[]): {
    primalsOut: Tracer[];
    residuals: ResidualPack;
  } {
    if (this.#disposed)
      throw new ReferenceError("PrimalArtifact already disposed");
    // Create a ResidualPack with .ref'd copies of the forward jaxpr's consts.
    // Each run() call produces an independent pack with its own ownership.
    const residualArrays = this.#forwardJaxpr.consts.map((c) => c.ref);
    return {
      primalsOut: this.#storedPrimalsOut,
      residuals: new ResidualPackImpl(residualArrays),
    };
  }

  [Symbol.dispose](): void {
    if (!this.#disposed) {
      this.#forwardJaxpr.dispose();
      this.#disposed = true;
    }
  }
}

// ---------------------------------------------------------------------------
// PullbackArtifact — owns the transposed backward jaxpr
// ---------------------------------------------------------------------------

/**
 * Backward-pass artifact produced by aotLinearize().
 *
 * May be called multiple times with different cotangents — the residuals
 * must not be consumed/destroyed if reuse is expected.
 */
export interface PullbackArtifact extends Disposable {
  /** The backward (transposed) jaxpr. */
  readonly backwardJaxpr: ClosedJaxpr;
  /** Run the backward pass: residuals + cotangents → input gradients. */
  run(residuals: ResidualPack, cotangents: Tracer[]): Tracer[];
}

/**
 * PullbackArtifact implementation.
 *
 * Owns the backward jaxpr (built by `buildBackwardJaxpr`). The backward
 * jaxpr's inBinders are `[...backwardConsts, ...residualVars, ...cotangentVars]`.
 *
 * `run(residuals, cotangents)` evaluates the backward jaxpr with the
 * residuals and cotangents as inputs, producing input gradients.
 */
class PullbackArtifactImpl implements PullbackArtifact {
  #backwardJaxpr: ClosedJaxpr;
  #disposed = false;

  constructor(backwardJaxpr: ClosedJaxpr) {
    this.#backwardJaxpr = backwardJaxpr;
  }

  get backwardJaxpr(): ClosedJaxpr {
    if (this.#disposed)
      throw new ReferenceError("PullbackArtifact already disposed");
    return this.#backwardJaxpr;
  }

  run(residuals: ResidualPack, cotangents: Tracer[]): Tracer[] {
    if (this.#disposed)
      throw new ReferenceError("PullbackArtifact already disposed");
    // The backward jaxpr's inputs are:
    //   [...backwardConsts, ...residualValues, ...cotangentValues]
    // backwardConsts come from the ClosedJaxpr (captured during transpose).
    // residualValues are the forward-pass concrete intermediate arrays.
    // cotangentValues are the gradient seeds from the caller.
    //
    // All inputs are .ref'd because evalJaxpr disposes every input at its
    // last use within the jaxpr.  Without .ref the caller's arrays would
    // be freed, breaking both residual reuse and caller expectations.
    return evalJaxpr(this.#backwardJaxpr.jaxpr, [
      ...this.#backwardJaxpr.consts.map((c) => c.ref),
      ...residuals.arrays.map((a) => a.ref),
      ...cotangents.map((c) => c.ref),
    ]);
  }

  [Symbol.dispose](): void {
    if (!this.#disposed) {
      this.#backwardJaxpr.dispose();
      this.#disposed = true;
    }
  }
}

// ---------------------------------------------------------------------------
// aotLinearize — top-level factory
// ---------------------------------------------------------------------------

/** Result of aotLinearize(). */
export interface AotLinearizeResult {
  readonly primal: PrimalArtifact;
  readonly pullback: PullbackArtifact;
}

/**
 * Ahead-of-time linearization: trace `f` once with `exampleArgs`, split into
 * a reusable forward artifact (PrimalArtifact) and backward artifact
 * (PullbackArtifact).
 *
 * Pipeline:
 * 1. JVP the function and partial-evaluate → forward jaxpr + primal outputs
 * 2. Transpose the forward jaxpr → backward jaxpr
 * 3. Dispose PE intermediates via ResidualCollector
 * 4. Wrap results in PrimalArtifact + PullbackArtifact
 *
 * Ownership after return:
 * - PrimalArtifact owns the forward jaxpr (whose consts are residual arrays)
 * - PullbackArtifact owns the backward jaxpr
 * - Each `primal.run()` creates an independent ResidualPack
 * - Caller must dispose all artifacts when done
 */
export function aotLinearize(
  f: (...args: Tracer[]) => Tracer[],
  exampleArgs: Tracer[],
): AotLinearizeResult {
  // Phase 1: JVP + partial evaluation → forward jaxpr + primal outputs
  const {
    primalsOut,
    jaxpr: forwardJaxpr,
    collector,
  } = linearizeFlatUtil(f, exampleArgs);

  // Phase 2: Transpose → backward jaxpr
  const backwardJaxpr = buildBackwardJaxpr(forwardJaxpr);

  // Phase 3: Dispose PE intermediates.
  // Protect primal outputs and forward jaxpr consts (residuals) from disposal.
  const protectedVals = new Set<Tracer>(primalsOut);
  for (const c of forwardJaxpr.consts) {
    protectedVals.add(c);
  }
  collector.dispose(protectedVals);

  // Phase 4: Create artifacts
  const primal = new PrimalArtifactImpl(forwardJaxpr, primalsOut);
  const pullback = new PullbackArtifactImpl(backwardJaxpr);

  return { primal, pullback };
}
