# jax-js Customization API Roadmap

## Goal

Define a durable public customization surface for jax-js that:

1. Preserves compiler optimization freedom.
2. Improves performance for advanced use cases like implicit differentiation.
3. Gives expert users more leverage than JAX where that leverage is safe.
4. Avoids freezing tracer, ownership, partial-eval, and backend invariants into stable public API.

This is an architectural roadmap, not a single-PR implementation plan.

## Executive Summary

The main customization boundary should remain at the transform layer.

That means:

1. Stable public APIs should primarily express semantic intent such as custom differentiation,
   rematerialization, implicit differentiation for solves, and batching behavior.
2. Expert-facing inspection and artifact tooling can be more open than JAX, but should live behind
   an explicit `unstable` or `experimental` namespace.
3. Raw tracer construction, primitive registration, rule-table mutation, partial evaluation, and
   backend lowering hooks should not become stable public API.

This is intentionally similar to JAX in layering, but jax-js can be more permissive than JAX in a
useful way by exposing more semantic expert APIs and more read-only inspection/artifact surfaces.

## Why This Boundary Is Correct

### Optimization preservation

If a customization is expressed at the transform layer, then under `jit` both the forward and
backward programs still trace into ordinary jax-js primitives. The compiler still sees normal math
and can continue to apply:

1. elementwise fusion,
2. command-tape compilation,
3. WebGPU batching,
4. WASM mega-module lowering,
5. scan and block-map lowering,
6. backend-specific tuning and cache reuse.

If the public extension story instead centers on custom primitives or backend hooks, those user
boundaries are much more likely to become opaque optimization barriers.

### Maintenance containment

The transform layer is the last place where internal compiler structure can still change freely.
Once primitive authoring, transpose tables, or lowering contracts are public and stable, every
internal refactor becomes an API migration problem.

### Ownership safety

jax-js has explicit disposal and ownership-sensitive internals. Public low-level extension points
would couple users to details such as:

1. eager vs traced lifetime differences,
2. `stopGradient` identity behavior,
3. refcount and retain semantics,
4. cache ownership,
5. backend resource teardown behavior.

That is the wrong contract to freeze.

## Design Principle

Expose intent, not mechanism.

Good public APIs express mathematical or transform semantics:

1. “this function has a custom reverse rule”
2. “this function has a custom forward rule”
3. “differentiate this linear solve implicitly”
4. “differentiate this root solve via the implicit function theorem”
5. “recompute instead of storing this region”

Bad public APIs expose internal machinery directly:

1. tracer construction
2. stable primitive registration
3. JVP/VJP rule table mutation
4. partial-evaluation scopes
5. backend lowering hooks
6. ownership / reference-count internals

The good list increases leverage. The bad list freezes implementation debt.

## Performance Review

### What performance means in jax-js

Performance here is not only FLOPs. It also includes:

1. JS-side dispatch overhead,
2. GPU command submission count,
3. memory traffic,
4. retained intermediates,
5. readback behavior,
6. ownership overhead.

Because of that, semantic customization APIs are especially valuable when they avoid tracing through
long iterative implementations or large backward graphs.

### Why transform-layer customization is good for performance

Transform-layer customizations preserve compiler visibility. If `fwd`, `bwd`, or JVP rules are
written in terms of ordinary jax-js operations, the resulting differentiated programs can still be
fused and lowered efficiently.

This is the best boundary for:

1. custom derivative formulas,
2. numerically stable gradients,
3. gradient conventions,
4. implicit differentiation wrappers,
5. structured solver APIs.

### Why `customVjp` has high leverage

`customVjp` can replace a large reverse-mode graph with a smaller mathematically equivalent backward
rule. That helps for:

1. implicit differentiation of optimization loops,
2. fixed-point iteration,
3. root-finding,
4. clipped or convention-based gradients,
5. numerically stable reverse rules.

The critical point is that the replacement backward rule still traces to ordinary ops, so backend
optimization remains available.

### Why `customJvp` is also important

Many numerically stable or mathematically natural derivative overrides are better expressed as JVPs
than VJPs.

Benefits:

1. one rule can often serve both forward- and reverse-mode paths,
2. higher-order differentiation often behaves more naturally,
3. the API matches the local linearization directly.

### Why solver-specific semantic APIs have outsized value

`customLinearSolve` and `customRoot` are likely more valuable than exposing raw internal extension
points.

They allow users to replace differentiation through many solver iterations with implicit
differentiation at the solution. In a dispatch-sensitive runtime, that can be a larger win than in
XLA-based JAX.

## Public API Tiers

### Tier 1: Stable Core Customization APIs

These should be the main user-facing customization surfaces.

#### 1. `customJvp`

Purpose:

1. numerical stability,
2. custom local linearization,
3. differentiation conventions at boundaries,
4. support for both forward- and reverse-mode differentiation.

Why it should be early:

1. many “custom gradient” use cases are actually forward-rule problems,
2. it avoids the reverse-only limitation of `customVjp`,
3. it provides a cleaner story for higher-order differentiation.

Suggested shape:

```ts
export function customJvp<F extends (...args: any[]) => any>(
  fn: F,
  jvpRule: (primals: Parameters<F>, tangents: Parameters<F>) => [ReturnType<F>, ReturnType<F>],
  opts?: { nondiffArgnums?: number[] },
): F;
```

#### 2. `customVjp`

Purpose:

1. reverse-only custom rules,
2. implicit differentiation wrappers,
3. custom backward logic for iterative or opaque implementations,
4. debugging or gradient-surgery use cases.

Why it should be early:

1. it is the critical building block for solver and optimizer libraries,
2. it directly addresses cases where tracing the primal backward pass is a bad plan.

Suggested shape:

```ts
export function customVjp<F extends (...args: any[]) => any>(
  fwd: (...args: Parameters<F>) => [ReturnType<F>, any],
  bwd: (residuals: any, cotangents: ReturnType<F>) => any,
  opts?: { nondiffArgnums?: number[] },
): (...args: Parameters<F>) => ReturnType<F>;
```

#### 3. `customGradient`

Purpose:

1. ergonomic convenience API for common scalar-output cases.

Recommendation:

Do not implement it as a separate mechanism. Implement it as thin sugar over `customVjp` or
`customJvp`.

#### 4. `checkpoint` / rematerialization

Purpose:

1. explicit memory-vs-recompute control,
2. better scaling for large autodiff graphs,
3. tighter control over saved residuals.

Suggested shape:

```ts
export function checkpoint<F extends (...args: any[]) => any>(
  fn: F,
  opts?: { policy?: "always" | "compiler" | "minimal" },
): F;
```

### Tier 2: Stable Expert Semantic APIs

These expose more power than the basic decorators but still preserve optimizer freedom.

#### 1. `customLinearSolve`

Purpose:

1. implicit gradients for matrix-free or iterative linear solves,
2. avoiding differentiation through solve implementations,
3. support for optimizer internals and implicit layers.

Why high priority:

1. directly useful for optimization libraries,
2. often reduces memory and dispatch count dramatically,
3. maps cleanly to mathematical structure.

Suggested shape:

```ts
export function customLinearSolve<T>(
  matvec: (x: T) => T,
  b: T,
  solve: (matvec: (x: T) => T, b: T) => T,
  opts?: {
    transposeSolve?: (vecmat: (x: T) => T, b: T) => T;
    symmetric?: boolean;
    hasAux?: boolean;
  },
): T;
```

#### 2. `customRoot` ✅

**Status: Implemented.** See `src/frontend/linearize.ts` and `test/custom-root.test.ts`.

Uses the Implicit Function Theorem via
`stopGradient(x*) + customLinearSolve(∂f/∂x, f(stopGradient(x*)))`. Gradients flow through `f`'s
closure parameters; the solver is opaque.

Final shape:

```ts
export function customRoot(
  f: (x: any) => any,
  initialGuess: any,
  solve: (f: (x: any) => any, initialGuess: any) => any,
  tangentSolve: (g: (x: any) => any, y: any) => any,
): any;
```

#### 3. `linearTranspose`

Purpose:

1. explicit access to the transpose of linearized computations,
2. building block for advanced solver and operator APIs.

Why later:

1. expert-facing,
2. easy to misuse if introduced too early,
3. best added after core custom derivative APIs settle.

#### 4. `closureConvert` or equivalent helper

Purpose:

1. make advanced APIs practical with closed-over values,
2. reduce awkwardness around explicit parameter threading.

Why later:

1. enabling infrastructure rather than the main user need,
2. best designed after `customRoot` and `customLinearSolve` prove out.

#### 5. `customBatching`

Purpose:

1. allow `vmap` override behavior for special abstractions.

Why late:

1. niche,
2. easy to increase complexity disproportionately,
3. batching semantics interact with many corners of the system.

### Tier 3: Explicitly Unstable Expert APIs

This is where jax-js can reasonably be more open than JAX.

These APIs should live under an explicit `unstable` or `experimental` namespace and carry no
stability guarantee.

Good candidates:

1. richer read-only Jaxpr inspection,
2. AOT forward/backward artifacts,
3. compiled code capture and lowering inspection,
4. detailed profiling hooks,
5. reusable linearization artifacts.

Possible shapes:

```ts
unstable.makeJaxprDetails(fn);
unstable.linearizeArtifact(fn);
unstable.captureLowering(fn);
unstable.profileCompiled(fn);
unstable.inspectResiduals(fn);
```

These are useful because they expose information and reusable artifacts without freezing the inner
compiler representation as a stable construction API.

## What Must Stay Private

Even if jax-js chooses to be more open than JAX, these should remain private or at most unstable:

1. tracer construction and lifecycle,
2. primitive registration as stable public API,
3. raw JVP and transpose rule tables,
4. partial-evaluation internals,
5. ownership and reference-count machinery,
6. backend lowering contracts,
7. command-tape internals,
8. buffer-pool and arena allocation internals.

These are backend-coupled, ownership-sensitive, and exactly where optimization freedom is
concentrated.

## Why Not Expose General Primitive Authoring Publicly

It is tempting to let power users register new primitives directly.

That is not a good stable public story because a usable primitive requires coordinated support for:

1. abstract evaluation,
2. eager execution,
3. JVP,
4. transpose / VJP,
5. batching,
6. shape rules,
7. ownership discipline,
8. lowering or fallback execution across backends.

In practice, a public primitive API either becomes too weak to be useful or exposes so much of the
compiler that refactoring becomes prohibitively expensive.

If primitive authoring is ever exposed, it should be explicitly unstable and framed as research
infrastructure.

## Recommended Order of Delivery

### Ordering for general autodiff completeness

1. `customJvp`
2. `customVjp`
3. `customGradient` as sugar
4. `checkpoint`
5. `customLinearSolve`
6. `customRoot`
7. `linearTranspose`
8. `closureConvert`
9. `customBatching`
10. unstable expert inspection surfaces

### Ordering for optimization-library value

If the near-term target is `jaxopt`-style functionality, use this order instead:

1. `customVjp`
2. `customLinearSolve`
3. `customRoot`
4. `customJvp`
5. `checkpoint`
6. `linearTranspose`
7. `closureConvert`
8. `customBatching`
9. unstable expert inspection surfaces

This second ordering is the better fit for current project goals.

## Mapping to Near-Term Use Cases

### Optimization libraries

Needed first:

1. `customVjp`
2. `customLinearSolve`
3. `customRoot`

Reason:

These cover implicit differentiation through fixed points, line searches, and iterative solvers
without differentiating through every internal step.

### Numerical stability improvements

Needed first:

1. `customJvp`
2. `customVjp`

Reason:

Many stable derivative formulas are naturally local linearization problems.

### Memory-sensitive workloads

Needed first:

1. `checkpoint`
2. `customVjp`

Reason:

The first controls rematerialization globally, the second allows manual restructuring of expensive
backward paths.

### Power users and researchers

Needed first:

1. unstable inspection APIs,
2. AOT linearization artifacts,
3. profiling and compiled-code capture.

Reason:

They usually want observability more than they want to own compiler invariants.

## Concrete Recommendations for jax-js

1. Keep `customVjp` at the transform layer. Do not rework it into a stable primitive-level feature.
2. Prioritize `customJvp`, `customLinearSolve`, and `customRoot` over any public primitive-authoring
   API.
3. Create an `unstable` namespace for expert inspection surfaces rather than opening raw internals
   in the stable namespace.
4. Treat `customGradient` as a user-experience wrapper, not a new implementation path.
5. If a future plugin story is desired, constrain it to inspection and artifact reuse first.

## Risks and Failure Modes

### Risk: Overexposing internals too early

Likely consequences:

1. blocked compiler refactors,
2. backend inconsistencies,
3. ownership bugs leaking into user code,
4. permanent optimization barriers.

### Risk: Only shipping `customVjp`

Likely consequences:

1. users force forward-mode-friendly problems into reverse-only APIs,
2. numerical-stability use cases become more awkward than necessary,
3. higher-order differentiation stories remain incomplete.

### Risk: Delaying structured solve APIs

Likely consequences:

1. downstream libraries build ad hoc implicit-diff wrappers,
2. duplicated logic for fixed-point and linear-solve transposes,
3. more brittle solver code in external packages.

## Proposed Stability Contract

### Stable

1. `customJvp`
2. `customVjp`
3. `customGradient`
4. `checkpoint`
5. `customLinearSolve`
6. `customRoot`
7. `linearTranspose`

### Unstable

1. read-only Jaxpr detail inspection,
2. AOT linearization artifacts,
3. compiled code capture helpers,
4. detailed profiling helpers,
5. research-grade primitive authoring if it is ever added.

### Private

1. tracer lifecycle,
2. primitive internals,
3. rule tables,
4. partial-eval internals,
5. ownership internals,
6. lowering contracts.

## Implementation Discipline Implied by This Roadmap

1. Semantic customization APIs should lower back into ordinary traceable math wherever possible.
2. Backward rules must respect ownership symmetry in eager and traced modes.
3. Unstable APIs should emphasize read-only or artifact-oriented access before mutation-oriented
   hooks.
4. Public semantic APIs should be backend-agnostic even if implementations are backend-sensitive.

## Final Recommendation

Use jax-js's additional freedom selectively.

Be more open than JAX about:

1. semantic expert APIs,
2. inspection and artifact tooling,
3. solver-oriented abstractions.

Do not be more open than JAX about:

1. tracer construction,
2. primitive authoring as stable public API,
3. backend lowering hooks,
4. ownership and partial-eval machinery.

That is the path that maximizes user leverage, preserves performance headroom, and keeps the
compiler architecture evolvable.
