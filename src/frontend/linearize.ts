/** @file Implementations of vjp() and partial evaluation. */
// jax-js-lint: allow-ref — .ref is the core ownership mechanism in autodiff internals

import { AluOp, isFloatDtype } from "../alu";
import {
  dispose as treeDispose,
  flatten as treeFlatten,
  map as treeMap,
  unflatten as treeUnflatten,
} from "../tree";
import {
  checkInts,
  DEBUG,
  deepEqual,
  generalBroadcast,
  invertPermutation,
  partitionList,
  range,
  toposort,
  unzip2,
} from "../utils";
import {
  anonymousConstArrays,
  array,
  eye,
  Array as JaxArray,
  onesLike,
  pureArray,
  zeros,
} from "./array";
import { aotLinearize } from "./artifacts";
import { _registerJitCacheDisposer } from "./check-leaks";
import {
  _peArrayCreationTracker,
  _setPACT,
  AbstractValue,
  add,
  argsort,
  bind,
  broadcast,
  concatenate,
  conv,
  currentTraceLevel,
  flattenFun,
  flattenFunWithAux,
  flip,
  fullRaise,
  gather,
  insideAbstractTrace,
  mul,
  ndim,
  neg,
  newMain,
  pad,
  Primitive,
  PrimitiveParams,
  reduce,
  reshape,
  ShapedArray,
  shrink,
  split,
  stopGradient,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  triangularSolve,
  UseAfterFreeError,
  where,
} from "./core";
import {
  abstractEvalRules,
  ClosedJaxpr,
  evalJaxpr,
  Jaxpr,
  JaxprEqn,
  Lit,
  makeJaxpr,
  OwnedFunction,
  typecheckJaxpr,
  Var,
} from "./jaxpr";
import { jvp, lowerAux } from "./jvp";
import { jacfwd, moveaxis, vmap } from "./vmap";

/** Array value that can either be known or unknown. */
class PartialVal {
  constructor(
    readonly val: Tracer | null,
    readonly aval: ShapedArray,
  ) {}

  static known(val: Tracer): PartialVal {
    return new PartialVal(val, ShapedArray.fromAval(val.aval));
  }

  static unknown(aval: AbstractValue): PartialVal {
    return new PartialVal(null, ShapedArray.fromAval(aval));
  }

  get isKnown(): boolean {
    return this.val !== null;
  }

  toString(): string {
    return this.val ? this.val.toString() : this.aval.toString();
  }
}

/**
 * Phase 1 of partial evaluation: trace `f` under a PartialEvalTrace and
 * produce a forward jaxpr together with PE intermediates for ownership disposal.
 *
 * This is the core of partial evaluation. The returned intermediates must be
 * disposed by the caller via `ResidualCollector.dispose()` (or a future
 * `ResidualCollector`) to balance rc=1 from creation.
 *
 * Extracted from `partialEvalFlat` in M1.1 to enable composition with
 * `buildBackwardJaxpr` and future artifact types.
 */
function buildForwardJaxpr(
  f: (...args: any[]) => any,
  pvalsIn: PartialVal[],
): {
  jaxpr: ClosedJaxpr;
  pvalsOut: PartialVal[];
  peIntermediates: Tracer[];
  literalIntermediates: Tracer[];
} {
  let jaxpr: ClosedJaxpr;
  let pvalsOut: PartialVal[];
  let knownIntermediates: Tracer[];
  let literalIntermediates: Tracer[];
  {
    using main = newMain(PartialEvalTrace);
    main.isAbstract = true;
    const trace = new PartialEvalTrace(main);
    const tracersIn = pvalsIn.map((pval) => trace.newArg(pval));
    const unknownTracersIn = tracersIn
      .filter((t) => !t.pval.isKnown)
      .map((t) => t.ref);

    // Track all Array constructions during PE scope for intermediate disposal.
    // This catches: (1) EvalTrace-level bind results from JVP rules,
    // (2) anonymous constants (np.array([...]) inside grad body) that bypass
    // bind() and PE tracking. Save/restore handles nested PE scopes (e.g.
    // grad(grad(f))).
    const previousTracker = _peArrayCreationTracker;
    const peCreatedArrays: Tracer[] = [];
    _setPACT(peCreatedArrays);
    let outs: any;
    try {
      outs = f(...tracersIn);
    } catch (e) {
      _setPACT(previousTracker);
      // Dispose any arrays created during the partial trace before the throw.
      for (const t of peCreatedArrays) {
        try {
          t.dispose();
        } catch {
          /* already disposed */
        }
      }
      throw e;
    }
    _setPACT(previousTracker);

    const tracersOut: PartialEvalTracer[] = outs.map((out: TracerValue) =>
      fullRaise(trace, out),
    );

    pvalsOut = tracersOut.map((t) => t.pval);
    const unknownTracersOut = tracersOut.filter((t) => !t.pval.isKnown);
    jaxpr = partialEvalGraphToJaxpr(unknownTracersIn, unknownTracersOut);

    // Dispose unreachable Const PETracers: instantiateConst calls .ref on the
    // underlying value for each Const PETracer it creates. The toposort in
    // partialEvalGraphToJaxpr disposes reachable Const PETracers (balancing
    // the .ref). But Const PETracers from unreachable computations (e.g. aux
    // branches in hasAux) are never processed, leaving dangling .ref calls.
    // Dispose any Const PETracer not already disposed by the toposort.
    //
    // In deeply nested transform stacks, a Const PETracer can occasionally
    // resolve to an already-disposed underlying value (owned and released by
    // an outer cleanup path). Keep draining the PETracer wrappers while
    // tolerating those stale-value cases.
    for (const ct of trace.allConstPETracers) {
      if (!ct.isAlive) continue;
      try {
        ct.dispose();
      } catch {
        // Already-disposed underlying value in nested transform cleanup.
      }
    }

    knownIntermediates = trace.knownIntermediates;
    literalIntermediates = trace.literalIntermediates;
    // Merge PE-scope Array creations into knownIntermediates for disposal.
    knownIntermediates.push(...peCreatedArrays);
  }

  return {
    jaxpr,
    pvalsOut,
    peIntermediates: knownIntermediates,
    literalIntermediates,
  };
}

/** Thin wrapper for backward compatibility. Delegates to `buildForwardJaxpr`. */
function _partialEvalFlat(
  f: (...args: any[]) => any,
  pvalsIn: PartialVal[],
): {
  jaxpr: ClosedJaxpr;
  pvalsOut: PartialVal[];
  peIntermediates: Tracer[];
  literalIntermediates: Tracer[];
} {
  return buildForwardJaxpr(f, pvalsIn);
}
/**
 * Unwrap tracer wrappers (BatchTracer, etc.) to find the underlying concrete
 * Array. Chases the `.val` chain until it reaches a non-wrapper value.
 * Returns the input unchanged if it's already a concrete Array.
 */
function unwrapToConcreteArray(t: Tracer): Tracer {
  let current: any = t;
  while (current && typeof current === "object" && "val" in current) {
    const inner = current.val;
    if (inner === current || !(inner instanceof Tracer)) break;
    current = inner;
  }
  return current;
}

/**
 * Collects PE intermediates from a forward partial-evaluation pass and
 * provides a `dispose(protectedVals)` method for deterministic cleanup.
 *
 * This class encapsulates the `disposePeIntermediates` logic: it owns the
 * list of PE intermediates and literal intermediates, and disposes those
 * that are NOT in the protected set (outputs, aux captures, low-rc consts).
 *
 * Precursor to `ResidualPack` in the AOT linearization artifact model.
 * Extracted in M1.3.
 */
class ResidualCollector {
  readonly peIntermediates: Tracer[];
  readonly literalIntermediates: Tracer[];

  constructor(peIntermediates: Tracer[], literalIntermediates: Tracer[]) {
    this.peIntermediates = peIntermediates;
    this.literalIntermediates = literalIntermediates;
  }

  /**
   * Dispose PE intermediates that aren't protected (outputs, aux, low-rc consts).
   *
   * During PE, all-known evaluations create concrete arrays. Output values
   * (in protectedVals) are returned to the caller. ClosedJaxpr consts have
   * independent ownership via .ref in partialEvalGraphToJaxpr. Everything
   * else (pure intermediates AND computed consts) needs one dispose here
   * to balance the rc=1 from creation.
   */
  dispose(protectedVals: Set<Tracer>): void {
    if (insideAbstractTrace()) return;
    // Build a protection set that includes both the wrapper tracers and their
    // underlying concrete arrays. PE-created arrays (peIntermediates) are raw
    // Arrays, but protectedVals may contain BatchTracers wrapping those arrays.
    // Without unwrapping, identity checks fail through the wrapper layer.
    //
    // CRITICAL: Multiple wrappers can share the same underlying concrete array
    // (e.g., Reduce with axis=[] returns the input as-is, so both the
    // intermediate and the output wrap the same raw Array). We must protect
    // the concrete array from disposal via ANY wrapper, not just the one
    // in protectedVals.
    const allProtected = new Set<Tracer>(protectedVals);
    for (const v of protectedVals) {
      const concrete = unwrapToConcreteArray(v);
      if (concrete !== v) allProtected.add(concrete);
    }
    const targets = this.peIntermediates;
    const disposed = new Set<Tracer>();
    for (const t of targets) {
      if (allProtected.has(t)) continue;
      if (disposed.has(t)) continue;
      // Before disposing a wrapper, check if its underlying concrete array
      // is protected. If so, skip this disposal entirely — cascading would
      // free the protected array.
      const concrete = unwrapToConcreteArray(t);
      if (concrete !== t && allProtected.has(concrete)) continue;
      disposed.add(t);
      // Mark the unwrapped concrete array as disposed too, so the raw Array
      // entry in peIntermediates (from _peArrayCreationTracker) won't
      // double-free it.
      if (concrete !== t) disposed.add(concrete);
      try {
        t.dispose();
      } catch {
        // Already disposed.
      }
    }
  }
}

/**
 * Extract concrete Arrays from a value tree that may contain JVPTracers,
 * PartialEvalTracers, or other wrappers. Uses lowerAux to walk through
 * the tracer chain (JVPTracer→PETracer→concrete Array).
 */
export function collectConcreteArrays(value: any): Tracer[] {
  const result: Tracer[] = [];
  const lowered = lowerAux(value);
  treeMap((x: any) => {
    if (x instanceof Tracer) result.push(x);
    return x;
  }, lowered);
  return result;
}

/** Result of linearizeFlatUtil: forward jaxpr + primals + disposal collector. */
export interface ForwardResult {
  primalsOut: Tracer[];
  jaxpr: ClosedJaxpr;
  collector: ResidualCollector;
}

/**
 * Helper function with shared Jaxpr logic between linearize and vjp.
 *
 * Internally, vjp() looks very similar to linearize() but returns a function
 * evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
 * of pushing forward tangents.
 *
 * Uses `buildForwardJaxpr` (M1.1) and returns a `ResidualCollector` (M1.3)
 * instead of raw intermediate arrays.
 */
export function linearizeFlatUtil(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): ForwardResult {
  const pvalsIn = [
    ...primalsIn.map(PartialVal.known),
    ...primalsIn.map((t) => PartialVal.unknown(t.aval)),
  ];
  const fJvp = (...x: Tracer[]) => {
    // Args contain both primals and tangents, concatenated.
    const k = x.length / 2;
    const [primalsOut, tangentsOut] = jvp(f, x.slice(0, k), x.slice(k, 2 * k));
    return [...primalsOut, ...tangentsOut];
  };
  const { jaxpr, pvalsOut, peIntermediates, literalIntermediates } =
    buildForwardJaxpr(fJvp, pvalsIn);
  const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
  if (!primalPvals.every((pval) => pval.isKnown)) {
    throw new Error("Not all primal values are known after partial evaluation");
  }
  const primalsOut = primalPvals.map((pval) => pval.val!);
  const collector = new ResidualCollector(
    peIntermediates,
    literalIntermediates,
  );
  return { primalsOut, jaxpr, collector };
}

function linearizeFlat(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
  auxStore?: { value: any },
): [Tracer[], (...args: Tracer[]) => Tracer[], () => void] {
  // AOT linearize: trace f, build forward jaxpr, dispose PE intermediates.
  // skipBackward: linearize only needs the forward jaxpr, not the backward.
  // Pass auxStore ref (not pre-computed arrays) — auxStore.value is only
  // populated during linearizeFlatUtil, which runs inside aotLinearize.
  const { primal, pullback } = aotLinearize(f, primalsIn, {
    auxStore,
    skipBackward: true,
  });

  // Pullback is unused for linearize — dispose immediately (no-op stub).
  pullback[Symbol.dispose]();

  // Get primal outputs and residuals; residuals are only needed to keep the
  // forward jaxpr consts alive until fLin is done — dispose them in the
  // outer dispose function.
  const { primalsOut, residuals } = primal.run(primalsIn);

  // fLin evaluates the forward jaxpr with tangent inputs (same as before).
  const forwardJaxpr = primal.forwardJaxpr;
  const fLin = (...tangents: Tracer[]) =>
    evalJaxpr(forwardJaxpr.jaxpr, [
      ...forwardJaxpr.consts.map((c) => c.ref),
      ...tangents,
    ]);

  // Dispose residuals + primal artifact (which owns the forward jaxpr).
  const dispose = () => {
    residuals[Symbol.dispose]();
    primal[Symbol.dispose]();
  };

  return [primalsOut, fLin, dispose];
}

export function linearize(
  f: (...primals: any[]) => any,
  primalsIn: any[],
  { hasAux = false } = {},
): [any, OwnedFunction<(...tangents: any[]) => any>, any?] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  let fFlat, outTree, aux;
  if (hasAux) {
    [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [fFlat, outTree] = flattenFun(f, inTree);
  }
  // Wrap scalar primals to Arrays; track which are newly created for disposal.
  const wrappedPrimals = primalsInFlat.map(pureArray);
  const [primalsOutFlat, fLinFlat, dispose] = linearizeFlat(
    fFlat,
    wrappedPrimals,
    hasAux ? aux : undefined,
  );
  // Dispose newly-created pureArray wrappers. After linearizeFlat returns, the
  // wrappers are only used for .aval (shape/dtype metadata), which is safe
  // to read after disposal. Skip wrappers that appear in primalsOutFlat
  // (identity function case: output IS the input primal).
  if (!insideAbstractTrace()) {
    const primalsOutSet = new Set(primalsOutFlat);
    for (let i = 0; i < wrappedPrimals.length; i++) {
      if (
        wrappedPrimals[i] !== primalsInFlat[i] &&
        !primalsOutSet.has(wrappedPrimals[i])
      ) {
        wrappedPrimals[i].dispose();
      }
    }
  }
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in linearize");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const fLin = ((...tangentsIn: any[]) => {
    const [tangentsInFlat, inTree2] = treeFlatten(tangentsIn);
    if (!inTree.equals(inTree2)) {
      throw new TreeMismatchError("linearize", inTree, inTree2);
    }
    // pureArray wrappers for tangents are consumed by evalJaxpr inside
    // fLinFlat (evalJaxpr auto-disposes input args at last use), so no
    // explicit disposal is needed here.
    const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
    return treeUnflatten(outTree.value!, tangentsOutFlat);
  }) as OwnedFunction<(...tangents: any[]) => any>;
  fLin.dispose = dispose;
  fLin[Symbol.dispose] = dispose;
  if (hasAux) {
    return [primalsOut, fLin, lowerAux(aux!.value)];
  }
  return [primalsOut, fLin];
}

// Used in PartialEvalTracer to track recipes for "unknown" partial vals.
type JaxprRecipe =
  | {
      type: "LambdaBinding";
    }
  | {
      // Note: Not really a constant, actually just a "known" value translated
      // into unknown for abstract evaluation rules.
      type: "Const";
      val: Tracer; // holds reference
    }
  | {
      type: "JaxprEqn";
      prim: Primitive;
      tracersIn: PartialEvalTracer[]; // holds reference
      params: Record<string, any>;
      avalsOut: ShapedArray[];
      tracerRefsOut: WeakRef<PartialEvalTracer>[];
    };

class PartialEvalTracer extends Tracer {
  #rc: number; // PartialEvalTracer reference count, used to free references.

  // Note: Either pval is known and recipe is null, or pval is unknown and
  // recipe describes how to compute the value.
  constructor(
    trace: Trace,
    readonly pval: PartialVal,
    readonly recipe: JaxprRecipe | null,
  ) {
    super(trace);
    this.#rc = 1;
  }

  get aval(): AbstractValue {
    return this.pval.aval;
  }

  toString(): string {
    if (!this.recipe) {
      return `PartialEvalTracer(${this.pval.toString()})`;
    } else {
      return `PartialEvalTracer<${this.recipe.type}>(${this.pval.toString()})`;
    }
  }

  get ref() {
    if (this.#rc <= 0) {
      throw new UseAfterFreeError(this);
    }
    this.#rc++;
    return this;
  }
  /** Whether this PETracer hasn't been fully disposed yet. */
  get isAlive(): boolean {
    return this.#rc > 0;
  }
  dispose() {
    if (this.#rc <= 0) {
      throw new UseAfterFreeError(this);
    }
    if (--this.#rc === 0) {
      // Cascade dispose to owned values. Known pval values and Const recipe
      // values are ref'd by partialEvalGraphToJaxpr before cleanup, so the
      // cascade here just releases the PETracer's share of ownership.
      // JaxprEqn tracersIn are NOT cascaded — they are handled by the
      // graph-wide toposort cleanup in partialEvalGraphToJaxpr.
      if (this.pval.isKnown) {
        this.pval.val!.dispose();
      } else if (this.recipe?.type === "Const") {
        this.recipe.val.dispose();
      }
    }
  }

  fullLower(): Tracer {
    if (this.pval.isKnown) return this.pval.val!;
    return this;
  }
}

class PartialEvalTrace extends Trace {
  newArg(pval: PartialVal) {
    if (pval.isKnown) return new PartialEvalTracer(this, pval, null);
    return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
  }

  pure(val: TracerValue): Tracer {
    const arr = pureArray(val);
    // Track literal-created Arrays so ResidualCollector can clean them up.
    // pureArray() returns existing Tracers as-is, so non-Tracer inputs (numbers,
    // TypedArrays) produce genuinely new allocations that need disposal.
    // Without this, literals used as args in JVP rules (0.0, 1.0, etc.) leak
    // a backend slot each.
    if (!(val instanceof Tracer)) {
      this.knownIntermediates.push(arr);
      this.literalIntermediates.push(arr);
    }
    return new PartialEvalTracer(this, PartialVal.known(arr), null);
  }
  lift = this.pure;

  // Track concrete Arrays created during all-known evaluation.
  // Stored on main.globalData so all PETrace instances created by findTopTrace
  // (which creates new trace instances for the same MainTrace) share the list.
  get knownIntermediates(): Tracer[] {
    let arr = this.main.globalData?._knownIntermediates;
    if (!arr) {
      arr = [];
      if (!this.main.globalData) (this.main as any).globalData = {};
      this.main.globalData._knownIntermediates = arr;
    }
    return arr;
  }

  // Track Arrays created from non-Tracer literals in pure(). These are safe
  // to dispose even while tracing, unlike general knownIntermediates which can
  // include tracer-backed values from nested transforms.
  get literalIntermediates(): Tracer[] {
    let arr = this.main.globalData?._literalIntermediates;
    if (!arr) {
      arr = [];
      if (!this.main.globalData) (this.main as any).globalData = {};
      this.main.globalData._literalIntermediates = arr;
    }
    return arr;
  }

  // All Const PETracers created by instantiateConst. Stored on main.globalData
  // so all PETrace instances from the same MainTrace share the list. Used to
  // dispose Const PETracers unreachable from tracersOut (e.g. hasAux aux
  // computations that reference input arrays but aren't in the jaxpr outputs).
  get allConstPETracers(): PartialEvalTracer[] {
    let arr = this.main.globalData?._allConstPETracers;
    if (!arr) {
      arr = [];
      if (!this.main.globalData) (this.main as any).globalData = {};
      this.main.globalData._allConstPETracers = arr;
    }
    return arr;
  }

  instantiateConst(tracer: PartialEvalTracer) {
    if (!tracer.pval.isKnown) {
      return tracer;
    } else {
      // Translate known value into unknown "Const" recipe for abstract eval.
      // .ref gives ClosedJaxpr independent ownership of the const value.
      // The matching dispose happens in partialEvalGraphToJaxpr cleanup
      // (for reachable consts) or partialEvalFlat (for unreachable ones).
      const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
      const val = tracer.pval.val!.ref;
      const constTracer = new PartialEvalTracer(this, pval, {
        type: "Const",
        val,
      });
      this.allConstPETracers.push(constTracer);
      return constTracer;
    }
  }

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: PartialEvalTracer[],
    params: PrimitiveParams<P>,
  ): Tracer[] {
    if (tracers.every((t) => t.pval.isKnown)) {
      const results = bind(
        primitive,
        tracers.map((t) => t.fullLower()),
        params,
      );
      // Track concrete results for disposal of intermediates after PE.
      for (const r of results) this.knownIntermediates.push(r);
      return results;
    }
    if (primitive === Primitive.Jit) {
      // Special case, needs its own PartialEvalTrace handling because unlike
      // other primtiives, Jit can have subexpressions that are known while
      // other outputs are unknown.
      const { name, jaxpr, numConsts } =
        params as PrimitiveParams<Primitive.Jit>;
      return this.#partialEvalJaxpr(name, jaxpr, numConsts, tracers);
    }
    if (primitive === Primitive.Scan) {
      // Special case for JVP'd scan: primal outputs depend only on primal inputs
      return this.#partialEvalScan(
        params as PrimitiveParams<Primitive.Scan>,
        tracers,
      );
    }
    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsIn = tracersIn.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: primitive,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [], // Populated later on
    };
    const tracersOut = avalsOut.map((aval, i) => {
      if (i > 0) {
        // Make sure we increment reference count for each tracer in the recipe,
        // since they belong to multiple PartialEvalTracers.
        tracersIn.forEach((t) => t.ref);
      }
      return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
    });
    recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
    return tracersOut;
  }

  /**
   * Evaluate a Jaxpr on a set of PartialEvalTracers, computing as many known
   * values as possible (with JIT) and forwarding the unknown ones.
   *
   * Used when encountering a Jit rule during the trace.
   */
  #partialEvalJaxpr(
    name: string,
    jaxpr: Jaxpr,
    numConsts: number,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    void numConsts; // Unused
    jaxpr = jaxpr.flatten(); // Otherwise, we don't partially evaluate nested Jaxprs well.

    const inUnknowns = tracers.map((t) => !t.pval.isKnown);
    const { jaxpr1, jaxpr2, outUnknowns, numRes } = partialEvalJaxpr(
      jaxpr,
      inUnknowns,
    );

    const [knownTracers, unknownTracers] = partitionList(inUnknowns, tracers);

    const outs1Res = bind(
      Primitive.Jit,
      knownTracers.map((t) => t.ref.fullLower()),
      { name: `${name}_peval`, jaxpr: jaxpr1, numConsts: 0 },
    );
    const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
    const res = outs1Res.slice(jaxpr1.outs.length - numRes);

    const resTracers = res.map((x) =>
      this.instantiateConst(fullRaise(this, x) as PartialEvalTracer),
    );
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.Jit,
      tracersIn: resTracers.concat(unknownTracers),
      params: { name: `${name}_resid`, jaxpr: jaxpr2, numConsts: 0 },
      avalsOut: jaxpr2.outs.map((x) => x.aval),
      tracerRefsOut: [], // populated later
    };
    const outs2 = jaxpr2.outs.map((x, i) => {
      if (i > 0) {
        // Make sure we increment reference count for each tracer in the recipe,
        // since they belong to multiple PartialEvalTracers.
        recipe.tracersIn.forEach((t) => t.ref);
      }
      return new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe);
    });
    recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));

    // Stitch the known and unknown output tracers together, both with Jit.
    let i = 0;
    let j = 0;
    return outUnknowns.map((unk) => (unk ? outs2[j++] : outs1[i++]));
  }

  /**
   * Partial eval for Scan primitive.
   *
   * When scan is encountered during partial evaluation (e.g., inside JVP for VJP):
   * - If all inputs are known, just run the scan
   * - If this is a JVP'd scan (doubled carry/xs), we can split primal (known)
   *   from tangent (unknown) outputs
   * - Otherwise, mark all outputs as unknown
   */
  #partialEvalScan(
    params: PrimitiveParams<Primitive.Scan>,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    const { numConsts: _numConsts, numCarry } = params;

    // Determine which tracers are known/unknown
    const isKnown = tracers.map((t) => t.pval.isKnown);

    // Check if any inputs are unknown
    const hasUnknown = isKnown.some((k) => !k);

    if (!hasUnknown) {
      // All inputs known, just run the scan
      const inputs = tracers.map((t) => t.fullLower());
      return bind(Primitive.Scan, inputs, params);
    }

    // Get abstract values for all outputs
    const avalsIn = tracers.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[Primitive.Scan](avalsIn, params);
    const numY = avalsOut.length - numCarry;

    // Check if this looks like a JVP'd scan (even numCarry and numY).
    const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0;

    if (!isJvpScan) {
      // Not a JVP scan, mark all outputs as unknown
      const tracersIn = tracers.map((t) => this.instantiateConst(t));
      const recipe: JaxprRecipe = {
        type: "JaxprEqn",
        prim: Primitive.Scan,
        tracersIn,
        params,
        avalsOut,
        tracerRefsOut: [],
      };
      const tracersOut = avalsOut.map((aval, i) => {
        if (i > 0) tracersIn.forEach((t) => t.ref);
        return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
      });
      recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
      return tracersOut;
    }

    // This is a JVP'd scan. We need to:
    // 1. Run primal-only computation to get known outputs
    // 2. Create a residual jaxpr for tangent computation

    const numPrimalCarry = numCarry / 2;
    const numPrimalY = numY / 2;

    // Run primal-only computation using known inputs + zeros for tangent
    const synthesizedZeroInputs: Tracer[] = [];
    const fullInputs = tracers.map((t) => {
      if (t.pval.isKnown) {
        return (t.pval.val as Tracer).ref;
      } else {
        const z = zeros(t.pval.aval.shape, { dtype: t.pval.aval.dtype });
        synthesizedZeroInputs.push(z);
        return z;
      }
    });

    const fullOuts = bind(Primitive.Scan, fullInputs, params);

    // Create tracersIn for the residual jaxpr
    const tracersIn = tracers.map((t) => this.instantiateConst(t));

    // Build recipe for the full scan
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.Scan,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [],
    };

    // Build output tracers
    const tracersOut: PartialEvalTracer[] = [];

    // Primal carry outputs (first numPrimalCarry) are known
    for (let i = 0; i < numPrimalCarry; i++) {
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), null),
      );
    }

    // Tangent carry outputs are unknown
    let isFirstUnknown = true;
    for (let i = numPrimalCarry; i < numCarry; i++) {
      fullOuts[i].dispose();
      if (!isFirstUnknown) tracersIn.forEach((t) => t.ref);
      isFirstUnknown = false;
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe),
      );
    }

    // Primal Y outputs are known
    for (let i = 0; i < numPrimalY; i++) {
      tracersOut.push(
        new PartialEvalTracer(
          this,
          PartialVal.known(fullOuts[numCarry + i]),
          null,
        ),
      );
    }

    // Tangent Y outputs are unknown
    for (let i = numPrimalY; i < numY; i++) {
      fullOuts[numCarry + i].dispose();
      tracersIn.forEach((t) => t.ref);
      tracersOut.push(
        new PartialEvalTracer(
          this,
          PartialVal.unknown(avalsOut[numCarry + i]),
          recipe,
        ),
      );
    }

    const retainedKnownOutputs = new Set<Tracer>();
    for (let i = 0; i < numPrimalCarry; i++) {
      retainedKnownOutputs.add(fullOuts[i]);
    }
    for (let i = 0; i < numPrimalY; i++) {
      retainedKnownOutputs.add(fullOuts[numCarry + i]);
    }

    for (const inp of fullInputs) {
      if (!retainedKnownOutputs.has(inp) && inp.refCount > 0) {
        inp.dispose();
      }
    }

    // tracerRefsOut: known positions get null ref
    recipe.tracerRefsOut = tracersOut.map((t) =>
      t.pval.isKnown ? (null as any) : new WeakRef(t),
    );

    return tracersOut;
  }
}

/** Partially evaluate a Jaxpr, returning an immediate and residual Jaxpr. */
function partialEvalJaxpr(
  jaxpr: Jaxpr,
  inUnknowns: boolean[],
  instantiate?: boolean[],
): { jaxpr1: Jaxpr; jaxpr2: Jaxpr; outUnknowns: boolean[]; numRes: number } {
  jaxpr = jaxpr.flatten(); // Otherwise, we don't partially evaluate nested Jaxprs well.

  const knownIns = jaxpr.inBinders.filter((_, i) => !inUnknowns[i]);
  const knownVars = new Set(knownIns); // Var that we can evaluate immediately.
  const residuals = new Set<Var>(); // Vars to evaluate in eqns1, and pass to eqns2 (subset of knownVars).

  const eqns1: JaxprEqn[] = [];
  const eqns2: JaxprEqn[] = [];
  for (const eqn of jaxpr.eqns) {
    if (eqn.primitive === Primitive.Jit) {
      throw new TypeError("partialEvalJaxpr requires flattened Jaxpr");
    }
    const hasUnknowns = eqn.inputs.some(
      (x) => x instanceof Var && !knownVars.has(x),
    );
    if (hasUnknowns) {
      for (const x of eqn.inputs) {
        if (x instanceof Var && knownVars.has(x)) {
          residuals.add(x);
        }
      }
      eqns2.push(eqn);
    } else {
      eqns1.push(eqn);
      for (const v of eqn.outBinders) {
        knownVars.add(v);
      }
    }
  }
  const outUnknowns = jaxpr.outs.map(
    (x) => x instanceof Var && !knownVars.has(x),
  );
  // If instantiate is provided, move selected outputs into residuals.
  if (instantiate !== undefined) {
    for (let i = 0; i < jaxpr.outs.length; i++) {
      const x = jaxpr.outs[i];
      if (instantiate[i] && !outUnknowns[i] && x instanceof Var) {
        residuals.add(x);
        outUnknowns[i] = true; // Mark as unknown.
      }
    }
  }

  const residualsL = Array.from(residuals);
  const [ins1, ins2] = partitionList(inUnknowns, jaxpr.inBinders);
  const [outs1, outs2] = partitionList(outUnknowns, jaxpr.outs);
  const jaxpr1 = new Jaxpr(ins1, eqns1, outs1.concat(residualsL));
  const jaxpr2 = new Jaxpr(residualsL.concat(ins2), eqns2, outs2);
  return { jaxpr1, jaxpr2, outUnknowns, numRes: residualsL.length };
}

/**
 * Convert the graph representation of a partial eval to a standard Jaxpr.
 * Also called `tracers_to_jaxpr()` in JAX.
 */
function partialEvalGraphToJaxpr(
  tracersIn: PartialEvalTracer[],
  tracersOut: PartialEvalTracer[],
): ClosedJaxpr {
  const tracerToVar = new Map<PartialEvalTracer, Var>();
  const constToVar = new Map<Tracer, Var>();
  const processedEqns = new Set<JaxprRecipe>(); // Avoid translating the same equation multiple times.
  const constPETracers: PartialEvalTracer[] = []; // Intermediate Const PETracers to dispose.
  const eqns: JaxprEqn[] = [];

  for (const t of tracersIn) {
    tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
  }

  for (const t of toposort(tracersOut, (t) =>
    t.recipe?.type === "JaxprEqn" ? t.recipe.tracersIn : [],
  )) {
    if (!t.recipe) {
      throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
    }
    if (t.recipe.type === "LambdaBinding") {
      // Check that the binding is in the input list.
      if (!tracersIn.includes(t)) {
        throw new TypeError("LambdaBinding tracer not in input list");
      }
    } else if (t.recipe.type === "Const") {
      const val = t.recipe.val;
      let binder = constToVar.get(val);
      if (!binder) {
        binder = new Var(ShapedArray.fromAval(val.aval));
        constToVar.set(val, binder);
      }
      tracerToVar.set(t, binder);
      constPETracers.push(t); // Track for disposal
    } else if (t.recipe.type === "JaxprEqn") {
      if (!processedEqns.has(t.recipe)) {
        processedEqns.add(t.recipe);
        const tracersIn = t.recipe.tracersIn.map((t) => tracerToVar.get(t)!);
        const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
        for (let i = 0; i < outBinders.length; i++) {
          const ref = t.recipe.tracerRefsOut[i];
          // ref can be null for known outputs in partial-eval of JVP'd scan
          const tracerOut = ref?.deref?.();
          if (tracerOut) {
            tracerToVar.set(tracerOut, outBinders[i]);
          }
        }
        eqns.push(
          new JaxprEqn(t.recipe.prim, tracersIn, t.recipe.params, outBinders),
        );
      }
    }
  }

  const [consts, constvars] = unzip2(constToVar.entries());
  const inBinders = [
    ...constvars,
    ...tracersIn.map((t) => tracerToVar.get(t)!),
  ];
  const outVars = tracersOut.map((t) => tracerToVar.get(t)!);
  let jaxpr = new Jaxpr(inBinders, eqns, outVars);
  typecheckJaxpr(jaxpr); // sanity check

  // Give ClosedJaxpr independent ownership of its consts before PETracer
  // cleanup. Without this .ref, the constPETracer disposal cascade (below)
  // consumes the only ref from instantiateConst, leaving ClosedJaxpr with
  // a borrowed reference. When ClosedJaxpr.dispose() is called (e.g., from
  // fVjp.dispose() in grad), it would free user-owned arrays.
  for (const c of consts) (c as Tracer).ref;

  // Cleanup PETracer wrappers:
  // 1) Const PETracers: their recipe.val was .ref'd by instantiateConst,
  //    so cascade to recipe.val.dispose() just balances the .ref (safe).
  //    The extra .ref above ensures ClosedJaxpr retains its own ownership.
  // 2) Unknown (non-Const) PETracers in tracersIn/tracersOut: no val to
  //    cascade to, so dispose() is effectively a no-op.
  // SKIP known PETracers — they hold borrowed references to caller-owned
  // Arrays (e.g., user inputs, forward pass results). Disposing would free
  // the caller's arrays prematurely.
  for (const t of constPETracers) t.dispose();
  for (const t of tracersIn) {
    if (!t.pval.isKnown) t.dispose();
  }
  for (const t of tracersOut) {
    if (!t.pval.isKnown) t.dispose();
  }

  jaxpr = jaxpr.simplify();
  if (DEBUG >= 5) {
    console.info("jaxpr from partial evaluation:\n" + jaxpr.toString());
  }

  return new ClosedJaxpr(jaxpr, consts);
}

// implementation of vjp and grad

/** Marker type for pullback, used by transpose rules. */
class UndefPrimal {
  readonly aval: ShapedArray;

  constructor(aval: AbstractValue) {
    this.aval = ShapedArray.fromAval(aval);
  }
}

/**
 * Helper to get or compute a primal (known) variable's value during transpose.
 * For intermediate variables that are known (computed from only known inputs),
 * we need to evaluate the equations that produce them.
 */
function getOrComputePrimal(
  jaxpr: Jaxpr,
  knownVars: Set<Var>,
  knownPrimals: Map<Var, Tracer>,
  v: Var,
  internalArrays?: Set<Tracer>,
): Tracer {
  // Return .ref so the caller (transpose rule) gets an independent copy.
  // Transpose rules like Add and Where explicitly dispose known primals;
  // without .ref the shared value in knownPrimals would be freed too.
  if (knownPrimals.has(v)) {
    const r = knownPrimals.get(v)!.ref;
    if (internalArrays) internalArrays.add(r);
    return r;
  }

  // Find the equation that produces this variable
  const eqn = jaxpr.eqns.find((eq) => eq.outBinders.some((out) => out === v));
  if (!eqn) {
    throw new Error(
      `Internal error: could not find equation producing variable`,
    );
  }

  // Recursively get values for inputs
  const inputVals = eqn.inputs.map((inp) =>
    inp instanceof Lit
      ? array(inp.value, { dtype: inp.dtype })
      : getOrComputePrimal(jaxpr, knownVars, knownPrimals, inp, internalArrays),
  );

  // Evaluate this equation
  const results = bind(eqn.primitive, inputVals, eqn.params as any);

  // Store all output values
  for (let i = 0; i < eqn.outBinders.length; i++) {
    knownPrimals.set(eqn.outBinders[i], results[i]);
  }

  // Return .ref so the caller gets an independent copy.
  const result = knownPrimals.get(v);
  if (!result) {
    throw new Error(`Internal error: variable not produced by equation`);
  }
  const r = result.ref;
  if (internalArrays) internalArrays.add(r);
  return r;
}

/**
 * Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
 *
 * Will raise a TypeError if the provided Jaxpr is not a linear function of its,
 * inputs, as general expressions cannot be transposed.
 */
function evalJaxprTransposed(
  jaxpr: Jaxpr,
  args: (Tracer | UndefPrimal)[],
  cotangents: Tracer[],
  { markAnonymous = false }: { markAnonymous?: boolean } = {},
): Tracer[] {
  // Track which variables are known (primal) vs unknown (tangent).
  // A variable is known if ALL its inputs are known (primal values propagate).
  // A variable is unknown if ANY of its inputs are unknown (tangent dependency).
  const knownVars = new Set<Var>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      knownVars.add(jaxpr.inBinders[i]);
    }
  }

  // Forward pass: propagate "known" status through equations
  for (const eqn of jaxpr.eqns) {
    const allInputsKnown = eqn.inputs.every(
      (v) => v instanceof Lit || knownVars.has(v),
    );
    if (allInputsKnown) {
      // All inputs are known → all outputs are known (primal computation)
      for (const outVar of eqn.outBinders) {
        knownVars.add(outVar);
      }
    }
  }

  // Now collect actual Tracer values for known input variables
  const knownPrimals = new Map<Var, Tracer>();
  const argPrimals = new Set<Var>(); // Track which primals are from args (owned by caller)
  const argPrimalInitRc = new Map<Var, number>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      const arg = args[i] as Tracer;
      knownPrimals.set(jaxpr.inBinders[i], arg);
      argPrimals.add(jaxpr.inBinders[i]);
      const concrete = unwrapToConcreteArray(arg);
      argPrimalInitRc.set(
        jaxpr.inBinders[i],
        concrete instanceof JaxArray ? concrete.refCount : arg.refCount,
      );
    }
  }

  const ctStore = new Map<Var, Tracer>();

  // Track arrays created internally for batch disposal at the end.
  // Includes: zeros from readCotangent, accumulated sums from writeCotangent,
  // literal arrays from primalsIn, and non-external values entering ctStore.
  const internalArrays = new Set<Tracer>();
  // Track externally-owned cotangents (seeds from caller) — never dispose these.
  const externalCts = new Set<Tracer>();
  for (const ct of cotangents) {
    if (ct instanceof Tracer) externalCts.add(ct);
  }

  const readCotangent = (v: Var) => {
    const ct = ctStore.get(v);
    if (ct) {
      // We should read a cotangent at most once, as an out binder.
      ctStore.delete(v);
      return ct;
    } else {
      const z = zeros(v.aval.shape, { dtype: v.aval.dtype });
      // Mark as anonymous so getOrMakeConstTracer (when inside a makeJaxpr
      // trace like transposeJaxpr) skips .ref — the ClosedJaxpr becomes
      // the sole owner and dispose() fully frees the backing Slot.
      // Only safe when the enclosing makeJaxpr builder is the sole capturer;
      // when evalJaxprTransposed runs directly inside an outer jit trace,
      // arrays may escape to the outer trace and need normal .ref there.
      if (markAnonymous && z instanceof JaxArray) anonymousConstArrays.add(z);
      internalArrays.add(z);
      return z;
    }
  };

  const writeCotangent = (v: Var, ct: Tracer | null) => {
    if (ct !== null) {
      // Track non-external cotangent values for batch disposal.
      // Don't dispose eagerly — ct may be aliased (e.g. Add transpose returns [ct, ct]).
      if (!externalCts.has(ct)) internalArrays.add(ct);
      const oldCt = ctStore.get(v);
      // May need to accumulate cotangents if used in multiple JaxprEqns.
      if (oldCt) {
        const sum = add(oldCt, ct);
        internalArrays.add(sum);
        ctStore.set(v, sum);
      } else {
        ctStore.set(v, ct);
      }
    }
  };

  for (let i = 0; i < jaxpr.outs.length; i++) {
    const v = jaxpr.outs[i];
    if (v instanceof Var) writeCotangent(v, cotangents[i]);
  }

  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    // Inputs are primalsIn and cotangentsOut, outputs are cotangentsIn. We're
    // using the known primal values to _pull back_ cotangents for unknown
    // values. Tricky!

    // Check if all inputs are known (using our forward-propagated knownVars)
    const allInputsKnown = eqn.inputs.every(
      (v) => v instanceof Lit || knownVars.has(v),
    );

    if (allInputsKnown) {
      // Skip equations where all inputs are known (residual equations).
      // These don't depend on unknowns and don't contribute to the linear function.
      continue;
    }

    // For equations with mixed inputs, we need to get residual values for known inputs
    // and mark unknown inputs as UndefPrimal
    const primalsIn = eqn.inputs.map((v) => {
      if (v instanceof Lit) {
        const lit = array(v.value, { dtype: v.dtype });
        if (markAnonymous && lit instanceof JaxArray) {
          anonymousConstArrays.add(lit);
        }
        internalArrays.add(lit);
        return lit;
      }
      return knownVars.has(v)
        ? getOrComputePrimal(jaxpr, knownVars, knownPrimals, v, internalArrays)
        : new UndefPrimal(v.aval);
    });

    const cotangentsOut = eqn.outBinders.map(readCotangent);
    const rule = transposeRules[eqn.primitive];
    if (!rule) {
      throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
    }
    const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params as any);
    for (let j = 0; j < eqn.inputs.length; j++) {
      const v = eqn.inputs[j];
      if (v instanceof Var && !knownVars.has(v)) {
        writeCotangent(v, cotangentsIn[j]);
      } else if (cotangentsIn[j] !== null) {
        throw new Error("internal: cotangent should be null");
      }
    }
  }

  const results: Tracer[] = [];
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (args[i] instanceof UndefPrimal) {
      results.push(readCotangent(jaxpr.inBinders[i]));
    }
  }

  // Flush result arrays' pending backend dispatches before disposing
  // intermediates. Pending operations hold incRef on their I/O slots; if
  // results carry unsubmitted PEs while intermediates are disposed, shared
  // PE refcounts never reach zero and the slots they reference leak.
  // Submitting first releases those cross-references cleanly.
  if (!insideAbstractTrace()) {
    for (const r of results) {
      if (r instanceof JaxArray) {
        r._flushPendingSync();
      }
    }
  }

  // Always restore input-known primals to their initial refcount.
  // This balances temporary .ref borrows from getOrComputePrimal even when
  // running inside a trace (e.g. grad(scan) checkpoint path).
  for (const v of argPrimals) {
    const val = knownPrimals.get(v);
    const initialRc = argPrimalInitRc.get(v);
    if (val && initialRc !== undefined) {
      const concrete = unwrapToConcreteArray(val);
      const currentRc =
        concrete instanceof JaxArray ? concrete.refCount : val.refCount;
      const excess = currentRc - initialRc;
      for (let i = 0; i < excess; i++) {
        try {
          val.dispose();
        } catch {
          break;
        }
      }
    }
  }

  // Dispose internally-created arrays and computed primals.
  // When inside an abstract trace (e.g., inner grad running during outer grad's
  // tracing), computed primals and internal arrays may be tracers from outer
  // traces. Disposing them would cascade (JVPTracer.dispose → primal.dispose)
  // and free values still needed by outer evaluations. Skip disposal when
  // abstract traces are on the stack. BatchTrace is safe — values are concrete.
  if (!insideAbstractTrace()) {
    const returnedSet = new Set(results);

    // 1. For computed primals (created by getOrComputePrimal forward recomputation):
    //    Fully dispose — nobody else owns them.
    for (const [v, t] of knownPrimals.entries()) {
      if (!argPrimals.has(v) && !returnedSet.has(t)) {
        try {
          while (t.refCount > 0) t.dispose();
        } catch {
          // Already disposed.
        }
      }
    }

    // 2. Dispose internal arrays (zeros, accumulated sums, literal arrays).
    //    Skip arg primals (same objects due to .ref returning `this`).
    //    Skip returned results and externally-owned cotangents.
    for (const arr of internalArrays) {
      if (!returnedSet.has(arr) && !externalCts.has(arr)) {
        // Check if this is an arg primal (same object via .ref)
        let isArgPrimal = false;
        for (const v of argPrimals) {
          if (knownPrimals.get(v) === arr) {
            isArgPrimal = true;
            break;
          }
        }
        if (!isArgPrimal) {
          try {
            arr.dispose();
          } catch {
            // Already disposed by a transpose rule or nested operation.
          }
        }
      }
    }
  }

  return results;
}

/**
 * Inverse operation of `generalBroadcast()` for backpropagation.
 *
 * `x` has the shape of the result of an operation that was broadcasted with
 * `target` (it's a cotangent during backprop). Returns a tracer with rank and
 * shape equal to `target`.
 */
function unbroadcast(x: Tracer, target: UndefPrimal): Tracer {
  const shape = target.aval.shape;

  // 1. Remove extra dimensions from x, if any.
  //    x can either have rank == target.ndim (fine!), or rank > target.ndim.
  //    In the latter case, we need to trim off extra dimensions on the left.
  const extraDims = x.ndim > shape.length ? range(x.ndim - shape.length) : [];
  if (x.ndim < shape.length) {
    throw new Error(
      `unbroadcast: x.ndim (${x.shape}) < target.ndim (${shape})`,
    );
  }

  // 2. Reduce (but keep) dimensions of x that are 1 in target.
  const unsqueeze: number[] = [];
  const keptReduceDims: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    // i is indexed according to target.
    const indexFromEnd = shape.length - i; // >= 1
    const indexInX = x.ndim - indexFromEnd;
    const xLen = x.shape[indexInX];
    if (xLen > 1 && shape[i] === 1) {
      unsqueeze.push(i);
      keptReduceDims.push(indexInX);
    } else if (shape[i] !== xLen) {
      throw new Error("internal: unbroadcast shape mismatch");
    }
  }

  const reductionDims = [...extraDims, ...keptReduceDims];
  if (reductionDims.length === 0) return x;
  let result = x.sum(reductionDims);
  if (!deepEqual(result.shape, shape)) {
    const sumResult = result;
    result = broadcast(sumResult, shape, unsqueeze); // keep dims selectively
    sumResult.dispose();
  }
  return result;
}

class NonlinearError extends TypeError {
  constructor(primitive: Primitive) {
    super(`Nonlinear operation in backward pass for ${primitive}`);
  }
}

type TransposeRule<P extends Primitive> = (
  cotangents: Tracer[],
  primals: (Tracer | UndefPrimal)[],
  params: PrimitiveParams<P>,
) => (Tracer | null)[];

// You need a transpose rule for a primitive p if:
//  - p is used in jvpRules, while computing a tangent (not primal)
//  - in this use, at least one argument to p is a tangent
//
// This computes a backward pass, so it pulls back cotangents to the inputs of p
// that are UndefPrimal (i.e., tangents that weren't sent forward).
const transposeRules: Partial<{ [P in Primitive]: TransposeRule<P> }> = {
  [Primitive.Mul]([ct], [x, y]) {
    // TODO: For transpose rules on operations that have type promotion rules,
    // make sure the gradient is cast back to the correct dtype.
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Mul);
    if (x instanceof UndefPrimal) {
      const prod = mul(ct, y as Tracer);
      const result = unbroadcast(prod, x);
      if (result !== prod) prod.dispose();
      return [result, null];
    } else {
      const prod = mul(x as Tracer, ct);
      const result = unbroadcast(prod, y as UndefPrimal);
      if (result !== prod) prod.dispose();
      return [null, result];
    }
  },
  [Primitive.Neg]([ct], [x]) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
    return [neg(ct)];
  },
  [Primitive.Add]([ct], [x, y]) {
    if (!(x instanceof UndefPrimal || y instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Add);
    if (x instanceof UndefPrimal && y instanceof UndefPrimal)
      // Non-consuming: ct survives both unbroadcast calls, no .ref needed.
      return [unbroadcast(ct, x), unbroadcast(ct, y)];
    return x instanceof UndefPrimal
      ? ((y as Tracer).dispose(), [unbroadcast(ct, x), null])
      : ((x as Tracer).dispose(), [null, unbroadcast(ct, y as UndefPrimal)]);
  },
  [Primitive.Reduce]([ct], [x], { op, axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Reduce);
    if (op === AluOp.Add) {
      return [broadcast(ct, x.aval.shape, axis)];
    } else {
      // Forward-mode jvp of product does not involve any products.
      // The same applies to min/max as non-additive reductions.
      throw new NonlinearError(Primitive.Reduce);
    }
  },
  [Primitive.Pool]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pool);
    return bind(Primitive.PoolTranspose, [ct], {
      inShape: x.aval.shape,
      window,
      strides,
    });
  },
  [Primitive.PoolTranspose]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.PoolTranspose);
    return bind(Primitive.Pool, [ct], { window, strides });
  },
  [Primitive.Dot]([ct], [x, y]) {
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Dot);
    const axisSize = generalBroadcast(x.aval.shape, y.aval.shape).slice(-1)[0];
    const ctBroad = broadcast(ct, ct.shape.concat(axisSize), [-1]); // Undo the contraction.
    if (x instanceof UndefPrimal) {
      const prod = mul(ctBroad, y as Tracer);
      const result = unbroadcast(prod, x);
      if (result !== prod) prod.dispose();
      ctBroad.dispose();
      return [result, null];
    } else {
      const prod = mul(x as Tracer, ctBroad);
      const result = unbroadcast(prod, y as UndefPrimal);
      if (result !== prod) prod.dispose();
      ctBroad.dispose();
      return [null, result];
    }
  },
  [Primitive.Conv]([ct], [lhs, rhs], params) {
    if (lhs instanceof UndefPrimal === rhs instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Conv);
    // See rules for transposing a convolution in `convolution.ts`.
    const v = params.vmapDims;
    // Permutation to swap batch/channel dims (axes v and v+1), keeping vmapDims first.
    const rev01 = [...range(v), v + 1, v, ...range(v + 2, ct.ndim)];
    if (lhs instanceof UndefPrimal) {
      // Transpose to LHS (activations).
      using kernelTransposed = transpose(rhs as Tracer, rev01) as Tracer; // Reverse in <-> out channels.
      using kernel = flip(
        kernelTransposed,
        range(v + 2, kernelTransposed.ndim),
      ) as Tracer; // Flip spatial dimensions.
      const result = conv(ct, kernel, {
        vmapDims: v,
        strides: params.lhsDilation,
        // Reference: _conv_general_vjp_lhs_padding()
        padding: params.padding.map<[number, number]>(([pl, _pr], i) => {
          // dilated kernel_size in this dimension
          const dilatedKernel =
            (kernel.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
          const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
          const padBefore = dilatedKernel - 1 - pl;
          // Cannot calculate `padAfter = dilatedKernel - 1 - pr` because strides
          // may not produce an equal dilated kernel, for instance, 6 / stride 2
          // produces [X_X_X_], but dilating the cotangents recovers [Y_Y_Y].
          //
          // Instead, we set it to make the output shape (before strides) match
          // with dilatedLhs, currently it's less than dilatedLhs.
          const dilatedLhs =
            (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
          const padAfter =
            dilatedLhs + dilatedKernel - 1 - dilatedCt - padBefore;
          return [padBefore, padAfter];
        }),
        lhsDilation: params.strides,
        rhsDilation: params.rhsDilation,
      });
      return [result, null];
    } else {
      // Transpose to RHS (filter).
      using newLhs = transpose(lhs as Tracer, rev01) as Tracer; // Reverse batch <-> in channels.
      using newRhs = transpose(ct, rev01) as Tracer; // Reverse batch <-> out channels.
      using convResult = conv(newLhs, newRhs, {
        vmapDims: v,
        strides: params.rhsDilation,
        // Reference: _conv_general_vjp_rhs_padding()
        padding: params.padding.map<[number, number]>(([pl, _pr], i) => {
          const dilatedLhs =
            (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
          const dilatedKernel =
            (rhs.aval.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
          const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
          const padFromLhs = dilatedCt - dilatedLhs;
          const padFromRhs = dilatedKernel - pl - 1;
          return [pl, padFromLhs + padFromRhs];
        }),
        lhsDilation: params.lhsDilation,
        rhsDilation: params.strides,
      }) as Tracer;
      const result = transpose(convResult, rev01); // Reverse in <-> out channels.
      return [null, result];
    }
  },
  [Primitive.Where]([ct], [cond, x, y]) {
    // Cotangent should be zero where cond doesn't apply.
    const cts: (Tracer | null)[] = [null, null, null];
    if (cond instanceof UndefPrimal) throw new NonlinearError(Primitive.Where);
    if (x instanceof UndefPrimal) {
      const masked = where(cond, ct, 0);
      cts[1] = unbroadcast(masked, x);
      if (cts[1] !== masked) masked.dispose();
    } else {
      x.dispose();
    }
    if (y instanceof UndefPrimal) {
      const masked = where(cond, 0, ct);
      cts[2] = unbroadcast(masked, y);
      if (cts[2] !== masked) masked.dispose();
    } else {
      y.dispose();
    }
    // ct and cond are in internalArrays — batch cleanup handles them.
    return cts;
  },
  [Primitive.Concatenate]([ct], inputs, { axis }) {
    // The backprop of concatenate is split along `axis`.
    //
    // Inputs that are `UndefPrimal` are tangent variables (unknown in the
    // residual sense). Inputs that are concrete arrays arise from
    // `linearTangentsJvp` substituting `zeros_like(primal)` for the zero
    // tangent of a constant-inside-the-function (e.g. `np.ones(...)` called
    // inside a `grad`-traced body). These are semantically zero tangents, so
    // no gradient flows back to them — they are NOT primal values leaking into
    // the tangent computation.
    //
    // If ALL inputs are concrete (no tangent variables at all), the
    // concatenate is truly nonlinear (a primal leaked in) — throw.
    if (inputs.every((x) => !(x instanceof UndefPrimal)))
      throw new NonlinearError(Primitive.Concatenate);

    const sizes = inputs.map((x) => x.aval.shape[axis]);
    const splits = split(ct, axis, sizes);

    // Fast path: all inputs are tangent variables.
    if (inputs.every((x) => x instanceof UndefPrimal)) return splits;

    // Mixed case: some inputs are concrete zero-tangent constants.
    // Return the corresponding split for tangent-variable inputs, and
    // dispose the unused split + return null for known-constant inputs
    // (null signals "no cotangent needed for this known input").
    return inputs.map((inp, i) => {
      if (inp instanceof UndefPrimal) return splits[i];
      splits[i].dispose();
      return null;
    });
  },
  [Primitive.Split](cts, [x], { axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Split);
    return [concatenate(cts, axis)];
  },
  [Primitive.Gather]([ct], [x, ...indices], { axis, outDim }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
    if (indices.some((i) => i instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Gather);
    // Permutation-case transpose: when a single 1-D index gathers along one
    // axis and the output has the same size as the input, the index is a
    // permutation.  Its transpose is gather(ct, inverse_perm). This covers
    // the sort/argsort JVP path.  The general case (duplicate indices,
    // multi-axis) requires a scatter_add primitive — see future work.
    if (
      indices.length === 1 &&
      axis.length === 1 &&
      ct.shape[axis[0]] === x.aval.shape[axis[0]]
    ) {
      const idx = indices[0] as Tracer;
      // argsort of the permutation gives the inverse permutation.
      const [sortedVals, invIdx] = argsort(idx);
      const result = gather(ct, [invIdx], axis, outDim);
      sortedVals.dispose();
      invIdx.dispose();
      return [result, null];
    }
    throw new Error(
      "Gather transpose rule is only implemented for permutation gathers. " +
        "General case (duplicate indices) requires a scatter_add primitive.",
    );
  },
  [Primitive.Transpose]([ct], [x], { perm }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Transpose);
    return [transpose(ct, invertPermutation(perm))];
  },
  [Primitive.Broadcast]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Broadcast);
    return [reduce(ct, AluOp.Add, axis)];
  },
  [Primitive.Reshape]([ct], [x], _) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Reshape);
    return [reshape(ct, x.aval.shape)];
  },
  [Primitive.Flip]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Flip);
    return [flip(ct, axis)];
  },
  [Primitive.Shrink]([ct], [x], { slice }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Shrink);
    const width = slice.map(
      ([s, e], i) => [s, x.aval.shape[i] - e] as [number, number],
    );
    return [pad(ct, width)];
  },
  [Primitive.Pad]([ct], [x], { width }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pad);
    const slice = width.map(
      ([s, _e], i) => [s, s + x.aval.shape[i]] as [number, number],
    );
    return [shrink(ct, slice)];
  },
  [Primitive.TriangularSolve]([ct], [a, b], { unitDiagonal }) {
    if (a instanceof UndefPrimal || !(b instanceof UndefPrimal))
      throw new NonlinearError(Primitive.TriangularSolve);
    // The adjoint of solving a @ x.T = b.T for x, when differentiating w.r.t. b:
    //   If forward is: x.T = a^{-1} @ b.T
    //   Then adjoint is: ct_b.T = a^{-T} @ ct_x.T, so we just transpose A
    // Note: The primitive always operates on upper triangular a, so a^T is lower.
    using aT = moveaxis(a, -2, -1) as Tracer;
    const ctB = triangularSolve(aT, ct, {
      lower: true,
      unitDiagonal,
    });
    return [null, ctB];
  },
  [Primitive.Jit](cts, args, { name, jaxpr }) {
    // We need this one because the jvp() rule for Jit generates a Jit
    // with the transformed Jaxpr. So grad-of-jit will result in a transposed
    // Jit, which we need to handle.
    const undefPrimals = args.map((x) => x instanceof UndefPrimal);
    const newJaxpr = transposeJaxpr(jaxpr, undefPrimals);
    const residuals = args.filter((x, i) => !undefPrimals[i]) as Tracer[];
    const outs = bind(
      Primitive.Jit,
      [...newJaxpr.consts.map((c) => c.ref), ...residuals, ...cts],
      {
        name: `${name}_t`,
        jaxpr: newJaxpr.jaxpr,
        numConsts: newJaxpr.consts.length,
      },
    );
    // Now pull cotangents back to the corresponding UndefPrimal inputs.
    let i = 0;
    return undefPrimals.map((isUndef) => (isUndef ? outs[i++] : null));
  },
  [Primitive.DynamicUpdateSlice]() {
    throw new Error("DynamicUpdateSlice transpose: not yet implemented");
  },
  [Primitive.Scan](
    cts,
    args,
    { jaxpr, numCarry, numConsts, length, reverse, checkpoint },
  ) {
    // Scan transpose rule for backward pass through scan.
    //
    // Supports two strategies controlled by the `checkpoint` option:
    // - Default: Store only √N checkpoints, recompute intermediate carries
    //   from the nearest checkpoint during the backward pass — O(√N) memory, ~2× compute.
    // - checkpoint=false: Store all N intermediate carries — O(N) memory.
    //   See: Griewank & Walther, "Algorithm 799: Revolve"
    //
    // Forward scan: (consts, init_carry, xs) -> (final_carry, ys)
    // body: (consts, carry, x) -> (new_carry, y)
    //
    // For a JVP-transformed scan, the layout is:
    // args: [jvpConsts..., constsP..., constsT..., carryP..., carryT..., xsP..., xsT...]
    // cts:  [ct_carryP..., ct_carryT..., ct_ysP..., ct_ysT...]

    const numX = args.length - numConsts - numCarry;
    const numY = cts.length - numCarry;

    // Detect JVP structure: even numCarry, numX, and numY.
    const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0 && numX % 2 === 0;

    const numPrimalCarry = isJvpScan ? numCarry / 2 : 0;
    const numPrimalX = isJvpScan ? numX / 2 : 0;
    const numPrimalY = isJvpScan ? numY / 2 : 0;

    // Identify which args are effectively UndefPrimal (need cotangents)
    const undefMask = args.map((x, i) => {
      if (x instanceof UndefPrimal) return true;
      if (!isJvpScan) return false;

      // JVP structure: second half of each group is tangent
      if (i < numConsts) {
        return false;
      } else if (i < numConsts + numCarry) {
        const carryIdx = i - numConsts;
        return carryIdx >= numPrimalCarry;
      } else {
        const xIdx = i - numConsts - numCarry;
        return xIdx >= numPrimalX;
      }
    });

    const bodyNumConsts = numConsts;
    const bodyNumCarry = numCarry;

    // Build transposed body jaxpr
    // Body input layout: [consts..., carry..., x...]
    const bodyUndefPrimals: boolean[] = [];
    for (let i = 0; i < jaxpr.inBinders.length; i++) {
      if (i < bodyNumConsts) {
        bodyUndefPrimals.push(undefMask[i]);
      } else if (i < bodyNumConsts + bodyNumCarry) {
        bodyUndefPrimals.push(undefMask[numConsts + (i - bodyNumConsts)]);
      } else {
        bodyUndefPrimals.push(
          undefMask[numConsts + numCarry + (i - bodyNumConsts - bodyNumCarry)],
        );
      }
    }

    // Get residual (primal) values from scan args
    const constArgs = args.slice(0, numConsts);
    const carryArgs = args.slice(numConsts, numConsts + numCarry);
    const xsArgs = args.slice(numConsts + numCarry);

    // Split into primal and tangent parts
    const constResiduals = constArgs.filter(
      (_, i) => !undefMask[i],
    ) as Tracer[];
    const carryResiduals = carryArgs.filter(
      (_, i) => !undefMask[numConsts + i],
    ) as Tracer[];
    const xsResiduals = xsArgs.filter(
      (_, i) => !undefMask[numConsts + numCarry + i],
    ) as Tracer[];

    const actualNumPrimalCarry = isJvpScan
      ? numPrimalCarry
      : carryArgs.map((_, i) => !undefMask[numConsts + i]).filter((x) => x)
          .length;

    if (actualNumPrimalCarry === 0 || carryResiduals.length === 0) {
      throw new Error(
        "Scan transpose: no carry residuals available. grad() through scan " +
          "requires primal carry values to be available as residuals.",
      );
    }

    // Step 1: Re-run forward pass to get all intermediate primal carries

    // Create forward body that only computes primals
    const forwardInTypes = jaxpr.inBinders
      .filter((_, i) => !bodyUndefPrimals[i])
      .map((v) => v.aval);

    const { jaxpr: primalForwardJaxpr } = makeJaxpr(
      (...primalInputs: Tracer[]): Tracer[] => {
        // Build full inputs with zeros for tangent slots
        const fullInputs: Tracer[] = [];
        let primalIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            const aval = jaxpr.inBinders[i].aval;
            fullInputs.push(zeros(aval.shape, { dtype: aval.dtype }));
          } else {
            fullInputs.push(primalInputs[primalIdx++].ref);
          }
        }
        const outs = evalJaxpr(jaxpr, fullInputs);
        const primalCarryOuts = outs.slice(0, numPrimalCarry);
        const primalYOuts = outs.slice(
          numCarry,
          numCarry + Math.floor(numY / 2),
        );
        // Dispose tangent outputs
        for (let i = numPrimalCarry; i < numCarry; i++) outs[i].dispose();
        for (let i = numCarry + Math.floor(numY / 2); i < outs.length; i++)
          outs[i].dispose();
        return [...primalCarryOuts, ...primalYOuts];
      },
      { validateRefs: false },
    )(...forwardInTypes);

    // Helper: run one forward step
    const runOneForwardStep = (iter: number, carry: Tracer[]): Tracer[] => {
      const dataIdx = reverse ? length - 1 - iter : iter;
      const xSlices: Tracer[] = [];
      for (const xs of xsResiduals) {
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
        ...constResiduals.map((c) => c.ref),
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
    };

    // Forward pass: collect carries for backward pass
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
      let currentCarry = carryResiduals.map((c) => c.ref);
      if (allCarries) {
        allCarries.push(currentCarry.map((c) => c.ref));
      } else {
        checkpointCarries!.set(
          0,
          currentCarry.map((c) => c.ref),
        );
      }

      for (let iter = 0; iter < length; iter++) {
        const newCarry = runOneForwardStep(iter, currentCarry);
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

    // Step 2: Create a tangent-only body for transposition
    const numTangentConsts = numConsts - constResiduals.length;
    const numTangentCarry = numCarry - numPrimalCarry;
    const numTangentX = numX - numPrimalX;

    const tangentBodyInAvals = [
      ...jaxpr.inBinders
        .filter((_, i) => !bodyUndefPrimals[i])
        .map((v) => v.aval), // primal inputs (residuals)
      ...jaxpr.inBinders
        .filter((_, i) => bodyUndefPrimals[i])
        .map((v) => v.aval), // tangent inputs
    ];

    const { jaxpr: tangentBody } = makeJaxpr(
      (...tangentBodyArgs: Tracer[]): Tracer[] => {
        const numPrimalInputs = jaxpr.inBinders.filter(
          (_, i) => !bodyUndefPrimals[i],
        ).length;
        const primalResiduals = tangentBodyArgs.slice(0, numPrimalInputs);
        const tangentInputs = tangentBodyArgs.slice(numPrimalInputs);

        // Build full body inputs in original order
        const fullInputs: Tracer[] = [];
        let primalIdx = 0;
        let tangentIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            fullInputs.push(tangentInputs[tangentIdx++].ref);
          } else {
            fullInputs.push(primalResiduals[primalIdx++].ref);
          }
        }

        // Evaluate full body
        const fullOuts = evalJaxpr(jaxpr, fullInputs);

        // Return only tangent outputs
        const tangentOuts: Tracer[] = [];
        for (let i = numPrimalCarry; i < numCarry; i++) {
          tangentOuts.push(fullOuts[i]);
        }
        for (let i = numCarry + numPrimalY; i < fullOuts.length; i++) {
          tangentOuts.push(fullOuts[i]);
        }

        // Dispose primal outputs
        for (let i = 0; i < numPrimalCarry; i++) fullOuts[i].dispose();
        for (let i = numCarry; i < numCarry + numPrimalY; i++)
          fullOuts[i].dispose();

        return tangentOuts;
      },
      { validateRefs: false },
    )(...tangentBodyInAvals);

    // Transpose the tangent-only body
    const tangentBodyUndefPrimals = [
      ...Array(
        tangentBody.jaxpr.inBinders.length -
          (numTangentConsts + numTangentCarry + numTangentX),
      ).fill(false), // primal residuals
      ...Array(numTangentConsts + numTangentCarry + numTangentX).fill(true), // tangent inputs
    ];

    const transposedBody = transposeJaxpr(
      tangentBody.jaxpr,
      tangentBodyUndefPrimals,
    );

    // Step 3: Run backward pass in reverse
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

    // Helper: run one backward step
    const runOneBackwardStep = (iter: number, primalCarry: Tracer[]) => {
      const dataIdx = reverse ? length - 1 - iter : iter;

      // Slice primal xs for this iteration
      const xSlices: Tracer[] = [];
      for (const xs of xsResiduals) {
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
        ...constResiduals.map((c) => c.ref),
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
      if (ctConstsAccum === null) {
        ctConstsAccum = ctConstsIter;
      } else {
        const next: Tracer[] = [];
        for (let i = 0; i < ctConstsAccum.length; i++) {
          const summed = add(ctConstsAccum[i], ctConstsIter[i]);
          ctConstsAccum[i].dispose();
          ctConstsIter[i].dispose();
          next.push(summed);
        }
        ctConstsAccum = next;
      }

      // Store x cotangents (will stack later)
      for (let i = 0; i < numTangentX; i++) {
        ctXsAccum[i].push(ctXIter[i]);
      }

      // Update running carry cotangent
      for (const c of ctCarryRunning) c.dispose();
      ctCarryRunning = ctCarryNew;
    };

    // Execute backward pass
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
          const newCarry = runOneForwardStep(iter, carry);
          for (const c of carry) c.dispose();
          carry = newCarry;
          segCarries.push(carry.map((c) => c.ref));
        }
        for (const c of carry) c.dispose();

        // Process segment backward
        for (let iter = segEnd - 1; iter >= segStart; iter--) {
          const localIdx = iter - segStart;
          runOneBackwardStep(iter, segCarries[localIdx]);
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
        runOneBackwardStep(iter, allCarries![iter]);
        for (const c of allCarries![iter]) c.dispose();
      }
      // Dispose the last allCarries entry
      for (const c of allCarries![length]) c.dispose();
    }

    // Dispose remaining cotangents
    for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++)
      ctYsAll[i].dispose();
    for (let i = 0; i < Math.floor(numY / 2); i++) ctYsAll[i].dispose();

    // Stack x cotangents
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

    // Build output cotangents
    const actualUndefMask = args.map((x) => x instanceof UndefPrimal);

    const result: (Tracer | null)[] = [];
    let ctConstIdx = 0;
    let ctCarryIdx = 0;
    let ctXIdx = 0;

    for (let i = 0; i < args.length; i++) {
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

    // Cleanup
    primalForwardJaxpr.dispose();
    // NOTE: transposedBody is NOT disposed here — it's owned by
    // transposeJaxprCache (returned from transposeJaxpr()). The cache
    // handles disposal via _disposeAllJitCaches in checkLeaks.stop().
    for (const c of constResiduals) c.dispose();
    for (const c of carryResiduals) c.dispose();
    for (const c of xsResiduals) c.dispose();

    return result;
  },
};

const transposeJaxprCache = new Map<Jaxpr, Map<string, ClosedJaxpr>>();

// Register for cleanup during checkLeaks.stop() to avoid leaking
// ClosedJaxpr consts across test boundaries.
_registerJitCacheDisposer(() => {
  for (const inner of transposeJaxprCache.values()) {
    for (const cj of inner.values()) {
      cj.dispose();
    }
  }
  transposeJaxprCache.clear();
});

function transposeJaxpr(jaxpr: Jaxpr, undefPrimals: boolean[]): ClosedJaxpr {
  const cacheKey = JSON.stringify(undefPrimals); // deterministic
  const prevResult = transposeJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult) return prevResult;

  // This handles grad-of-jit or transpose-of-jit. To do this, it needs to
  // evaluate the Jaxpr transposed and then retrace it. See the comment in
  // jvpJaxpr() to explain more about what's going on here.
  const { inTypes, outTypes } = typecheckJaxpr(jaxpr);

  // Need to remove the UndefPrimals from the input types, as they are not
  // inputs to the Jaxpr while tracing.
  const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);
  const { jaxpr: newJaxpr } = makeJaxpr(
    (forwardIn: Tracer[], cotangents: Tracer[]) => {
      const args: (Tracer | UndefPrimal)[] = [];
      let forwardInIdx = 0; // index in forwardIn
      for (let i = 0; i < undefPrimals.length; i++) {
        if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
        else args.push(forwardIn[forwardInIdx++]);
      }
      return evalJaxprTransposed(jaxpr, args, cotangents, {
        markAnonymous: true,
      });
    },
    { validateRefs: false },
  )(forwardInTypes, outTypes);
  typecheckJaxpr(newJaxpr.jaxpr); // sanity check

  if (!transposeJaxprCache.has(jaxpr))
    transposeJaxprCache.set(jaxpr, new Map());
  transposeJaxprCache.get(jaxpr)!.set(cacheKey, newJaxpr);
  return newJaxpr;
}

/**
 * Phase 2 of partial evaluation: transpose a forward jaxpr to produce a
 * backward jaxpr suitable for VJP/grad.
 *
 * This is a thin wrapper around `transposeJaxpr` that constructs the
 * `undefPrimals` mask from the forward jaxpr's input count and the number
 * of primals. In the forward jaxpr, the inputs are `[...consts, ...primals, ...tangents]`.
 * The `undefPrimals` mask marks which inputs should be treated as
 * unknowns in the transpose (i.e., the tangent slots → true, primal slots → false).
 *
 * Extracted from `vjpFlat` in M1.2 to enable composition with
 * `buildForwardJaxpr` and future artifact types.
 */
export function buildBackwardJaxpr(forwardJaxpr: ClosedJaxpr): ClosedJaxpr {
  // The forward jaxpr's inBinders are [...constVars, ...tangentVars].
  // constVars correspond to the residuals from partial evaluation (known
  // in the backward pass). tangentVars correspond to the JVP tangent
  // inputs (unknown — we want gradients w.r.t. them).
  const numConsts = forwardJaxpr.consts.length;
  const undefPrimals = forwardJaxpr.jaxpr.inBinders.map(
    (_, i) => i >= numConsts,
  );

  // Build a fresh backward jaxpr. We bypass transposeJaxprCache because
  // the returned ClosedJaxpr will be owned (and disposed) by the caller
  // (PullbackArtifact). Caching would create stale-entry hazards.
  const jaxpr = forwardJaxpr.jaxpr;
  const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
  const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);

  const { jaxpr: newJaxpr } = makeJaxpr(
    (forwardIn: Tracer[], cotangents: Tracer[]) => {
      const args: (Tracer | UndefPrimal)[] = [];
      let forwardInIdx = 0;
      for (let i = 0; i < undefPrimals.length; i++) {
        if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
        else args.push(forwardIn[forwardInIdx++]);
      }
      return evalJaxprTransposed(jaxpr, args, cotangents, {
        markAnonymous: true,
      });
    },
    { validateRefs: false },
  )(forwardInTypes, outTypes);
  typecheckJaxpr(newJaxpr.jaxpr);
  return newJaxpr;
}

/**
 * Check whether a Jaxpr contains any equation that `buildBackwardJaxpr`
 * cannot trace (currently: `Primitive.Scan`, whose transpose rule creates
 * sub-jaxprs and does concrete forward recomputation incompatible with
 * makeJaxpr tracing — see M4 in AOT-LINEARIZATION-PLAN.md).
 */
function jaxprNeedsCallTimeTranspose(jaxpr: Jaxpr): boolean {
  return jaxpr.eqns.some((eqn) => eqn.primitive === Primitive.Scan);
}

function vjpFlat(
  f: (...x: Tracer[]) => Tracer[],
  primalsIn: Tracer[],
  auxStore?: { value: any },
): [Tracer[], (...cotangents: Tracer[]) => Tracer[], () => void] {
  // Phase 1: JVP + partial evaluation → forward jaxpr + primal outputs.
  // We always need this, regardless of the transposition strategy.
  const {
    primalsOut,
    jaxpr: forwardJaxpr,
    collector,
  } = linearizeFlatUtil(f, primalsIn);

  // Phase 2: Dispose PE intermediates.
  const protectedVals = new Set<Tracer>(primalsOut);
  for (const c of forwardJaxpr.consts) {
    if (c.refCount <= 1) protectedVals.add(c);
  }
  if (auxStore?.value != null) {
    for (const arr of collectConcreteArrays(auxStore.value)) {
      protectedVals.add(arr);
    }
  }
  collector.dispose(protectedVals);

  // CRITICAL: Flush primals' pending backend dispatches. PE tracing created
  // concrete forward-pass arrays with lazy PendingExecute chains. Submitting
  // them here prevents orphaned Slot references.
  if (!insideAbstractTrace()) {
    for (const p of primalsOut) {
      if (p instanceof JaxArray) {
        p._flushPendingSync();
      }
    }
  }

  // Phase 3: Choose transposition strategy.
  //
  // Scan's transpose rule creates sub-jaxprs and does concrete forward
  // recomputation — incompatible with `buildBackwardJaxpr`'s makeJaxpr
  // tracing (M4 will add a dedicated scan backward artifact). For now,
  // fall back to call-time transposition for scan-containing jaxprs.
  if (jaxprNeedsCallTimeTranspose(forwardJaxpr.jaxpr)) {
    // Call-time transposition: evalJaxprTransposed runs with concrete arrays
    // when fVjp is invoked (the old, proven pattern).
    const fVjp = (...cotangents: Tracer[]) => {
      const transposeInputs = [
        ...forwardJaxpr.consts,
        ...primalsIn.map((t) => new UndefPrimal(t.aval)),
      ];
      return evalJaxprTransposed(
        forwardJaxpr.jaxpr,
        transposeInputs,
        cotangents,
      );
    };
    const dispose = () => forwardJaxpr.dispose();
    return [primalsOut, fVjp, dispose];
  }

  // AOT transposition: build backward jaxpr eagerly.
  const backwardJaxpr = buildBackwardJaxpr(forwardJaxpr);
  const primal = {
    forwardJaxpr,
    run(_primalsIn: Tracer[]) {
      const residualArrays = forwardJaxpr.consts.map((c) => c.ref);
      return { primalsOut, residuals: residualArrays };
    },
  };

  // Pullback closure: evaluate backward jaxpr with residuals + cotangents.
  const residuals = primal.run(primalsIn);
  const fVjp = (...cotangents: Tracer[]) => {
    return evalJaxpr(backwardJaxpr.jaxpr, [
      ...backwardJaxpr.consts.map((c) => c.ref),
      ...residuals.residuals.map((a) => a.ref),
      ...cotangents.map((c) => c.ref),
    ]);
  };

  // Dispose all owned resources.
  const dispose = () => {
    for (const r of residuals.residuals) r.dispose();
    forwardJaxpr.dispose();
    backwardJaxpr.dispose();
  };

  return [primalsOut, fVjp, dispose];
}

export function vjp(
  f: (...primals: any) => any,
  primalsIn: any[],
  { hasAux = false } = {},
): [any, OwnedFunction<(...cotangents: any) => any>, any?] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  let fFlat, outTree, aux;
  if (hasAux) {
    [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [fFlat, outTree] = flattenFun(f, inTree);
  }
  // Wrap scalar primals to Arrays; track which are newly created for disposal.
  const wrappedPrimals = primalsInFlat.map(pureArray);
  const [primalsOutFlat, fVjpFlat, innerDispose] = vjpFlat(
    fFlat,
    wrappedPrimals,
    hasAux ? aux : undefined,
  );
  // Dispose newly-created pureArray wrappers. After vjpFlat returns, the
  // wrappers are only used for .aval (shape/dtype metadata), which is safe
  // to read after disposal. Skip wrappers that appear in primalsOutFlat
  // (identity function case: output IS the input primal).
  if (!insideAbstractTrace()) {
    const primalsOutSet = new Set(primalsOutFlat);
    for (let i = 0; i < wrappedPrimals.length; i++) {
      if (
        wrappedPrimals[i] !== primalsInFlat[i] &&
        !primalsOutSet.has(wrappedPrimals[i])
      ) {
        wrappedPrimals[i].dispose();
      }
    }
  }
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in vjp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);

  // "cotangentsOut" because pullback
  const fVjp = ((cotangentsOut: any) => {
    const [cotangentsOutFlat, outTree2] = treeFlatten(cotangentsOut);
    if (!outTree.value!.equals(outTree2)) {
      throw new TreeMismatchError("vjp", outTree.value!, outTree2);
    }
    // Wrap scalar cotangents to Arrays; dispose wrappers after transpose.
    const wrappedCots = cotangentsOutFlat.map(pureArray);
    const cotangentsInFlat = fVjpFlat(...wrappedCots);
    if (!insideAbstractTrace()) {
      for (let i = 0; i < wrappedCots.length; i++) {
        if (wrappedCots[i] !== cotangentsOutFlat[i]) {
          wrappedCots[i].dispose();
        }
      }
    }
    return treeUnflatten(inTree, cotangentsInFlat);
  }) as OwnedFunction<(...cotangents: any) => any>;
  fVjp.dispose = innerDispose;
  fVjp[Symbol.dispose] = innerDispose;

  if (hasAux) {
    return [primalsOut, fVjp, lowerAux(aux!.value)];
  }
  return [primalsOut, fVjp];
}

/** @inline */
export type GradOpts = {
  /**
   * Integer or sequence of integers. Specifies which positional argument(s) to
   * differentiate with respect to.
   *
   * Defaults to `0` (the first argument).
   */
  argnums?: number | number[];

  /**
   * The input function returns a pair of `[out, aux]` including an auxiliary
   * value. This `aux` is not differentiated, but is returned alongside the
   * gradient when evaluating the function.
   */
  hasAux?: boolean;
};

export function grad(f: (...primals: any) => Tracer, opts?: GradOpts) {
  const valueAndGradFn = valueAndGrad(f, opts);
  return (...x: any) => {
    if (opts?.hasAux) {
      const [[y, aux], dx] = valueAndGradFn(...x);
      if (!insideAbstractTrace()) y.dispose();
      return [dx, aux];
    } else {
      const [y, dx] = valueAndGradFn(...x);
      if (!insideAbstractTrace()) y.dispose();
      return dx;
    }
  };
}

export function valueAndGrad(f: (...primals: any) => Tracer, opts?: GradOpts) {
  const argnums = opts?.argnums ?? 0; // By default, differentiate w.r.t. first arg.
  const hasAux = opts?.hasAux ?? false;
  checkInts(argnums);
  const argnumsSet = new Set(typeof argnums === "number" ? [argnums] : argnums);
  return (...x: any) => {
    if (x.length === 0) {
      throw new Error("grad requires at least one argument to differentiate");
    }
    // Differentiate only with respect to the argnums.
    // Track stopGradient results for disposal after vjp completes.
    // We track (sgResult, original) pairs so we only dispose sg results that
    // are distinct from their inputs (stopGradient returns the same object in
    // eager mode, so disposing it would dispose the user's input).
    const sgArrays: Tracer[] = [];
    const sgOriginals = new Set<Tracer>();
    for (let i = 0; i < x.length; i++) {
      if (!argnumsSet.has(i)) {
        x[i] = treeMap((leaf: any) => {
          const sg = stopGradient(leaf);
          if (sg instanceof Tracer) {
            if (leaf instanceof Tracer) sgOriginals.add(leaf);
            sgArrays.push(sg);
          }
          return sg;
        }, x[i]);
      }
    }
    const [y, fVjp, aux] = vjp(f, x, { hasAux });
    if (!(y instanceof Tracer) || ndim(y) !== 0) {
      if (!insideAbstractTrace()) {
        fVjp.dispose();
        treeDispose(y);
        if (hasAux) treeDispose(aux);
        for (const a of sgArrays) {
          if (!sgOriginals.has(a)) a.dispose();
        }
      }
      throw new TypeError("grad requires a scalar output");
    }
    if (!isFloatDtype(y.dtype)) {
      if (!insideAbstractTrace()) {
        fVjp.dispose();
        treeDispose(y);
        if (hasAux) treeDispose(aux);
        for (const a of sgArrays) {
          if (!sgOriginals.has(a)) a.dispose();
        }
      }
      throw new TypeError("grad only supports floating-point dtypes");
    }
    const seed = onesLike(y);
    const cts = fVjp(seed); // backprop from scalar 1
    let seedEscapes = false;
    for (const ct of cts) {
      if (ct === seed) {
        seedEscapes = true;
        break;
      }
      for (const arr of collectConcreteArrays(ct)) {
        if (arr === seed) {
          seedEscapes = true;
          break;
        }
      }
      if (seedEscapes) break;
    }
    const shouldDisposeSeed = !seedEscapes;
    if (!insideAbstractTrace()) {
      for (const a of sgArrays) {
        if (!sgOriginals.has(a)) a.dispose();
      }
    }
    fVjp.dispose();
    if (shouldDisposeSeed) {
      if (!insideAbstractTrace()) {
        seed.dispose();
      } else if (
        currentTraceLevel() === 1 &&
        seed instanceof JaxArray &&
        seed.refCount > 0
      ) {
        seed.dispose();
      }
    }
    for (let i = 0; i < cts.length; i++) {
      if (!argnumsSet.has(i)) treeDispose(cts[i]);
    }
    const grads =
      typeof argnums === "number" ? cts[argnums] : argnums.map((i) => cts[i]);
    return hasAux ? [[y, aux], grads] : [y, grads];
  };
}

// See also: jacfwd()
export function jacrev(f: any) {
  return function jacobianReverse(x: Tracer) {
    if (x.shape.length !== 1) {
      throw new TypeError("jacrev only supports 1D inputs");
    }
    const [size] = x.shape;
    const pullback = (ct: Tracer) => {
      const [y, fVjp] = vjp(f, [x]);
      y.dispose();
      const [ret] = fVjp(ct);
      fVjp.dispose();
      return ret;
    };
    const eyeMatrix = eye(size, undefined, { dtype: x.dtype });
    const result = vmap(pullback, [1])(eyeMatrix);
    eyeMatrix.dispose();
    return result;
  };
}

// See also: jacfwd()
export function hessian(f: any) {
  return jacfwd(grad(f));
}
