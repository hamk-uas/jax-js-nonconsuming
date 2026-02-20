/**
 * @file AOT Linearization Artifacts — explicit lifecycle management for autodiff.
 *
 * These artifact types replace ad-hoc PE intermediate cleanup with scoped
 * ownership: each artifact owns exactly the resources it needs, implements
 * `[Symbol.dispose]`, and can be cleaned up deterministically.
 *
 * See AOT-LINEARIZATION-PLAN.md for design rationale and milestone tracking.
 */

import type { Tracer } from "./core";
import type { ClosedJaxpr } from "./jaxpr";

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

// ---------------------------------------------------------------------------
// PrimalArtifact — owns the forward jaxpr + compiled program
// ---------------------------------------------------------------------------

/** Forward-pass artifact produced by aotLinearize(). */
export interface PrimalArtifact extends Disposable {
  /** The forward-pass jaxpr (includes residual outputs). */
  readonly forwardJaxpr: ClosedJaxpr;
  /** Execute the forward pass, producing user outputs and residuals. */
  run(primals: Tracer[]): { primalsOut: Tracer[]; residuals: ResidualPack };
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
 * Stub — implementation will be added in M2.4.
 */
export function aotLinearize(
  _f: (...args: Tracer[]) => Tracer[],
  _exampleArgs: Tracer[],
): AotLinearizeResult {
  throw new Error("aotLinearize: not yet implemented (see M2.4)");
}
