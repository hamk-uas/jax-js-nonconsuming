# Effect-Typed Jaxpr: Architectural Redesign Plan

**Branch:** `effect-typed-ir-plan`
**Scope:** Evolve the Jaxpr Intermediate Representation (IR) from purely functional SSA to an Effect-Typed IR with explicit memory semantics (`Alloc`, `Borrow`, `Consume`, `Mutate`). Enable static borrow checking and zero-copy WebGPU mutations.
**Execution model:** Single autonomous agent. No human review gates.

---

## Motivation

The current Jaxpr IR is purely functional. Memory lifetimes are inferred heuristically at execution time (`evalJaxpr` usage counts) or compile time (`jitCompile` free/recycle passes). While safe, this functional abstraction hides hardware realities and limits performance:

1. **Missed WebGPU Optimizations:** Operations like `dynamic_update_slice` currently require allocating a full copy because the IR cannot express in-place mutation safely.
2. **Suboptimal Buffer Recycling:** The JIT compiler's `recycleBuffers` pass relies on finding adjacent `free -> malloc` pairs. Explicit `Consume -> Alloc` effects would allow global graph-coloring-based register allocation for GPU buffers.
3. **Opaque Lifetimes in AD:** The autodiff system cannot statically verify if a residual is consumed or borrowed, leading to defensive copies or complex artifact management.
4. **Missing Primitives:** Operations like `scatter_add` require atomic mutations, which don't map well to a purely functional IR without explicit effect tracking.

The redesign introduces **Effect-Typed Jaxpr**:
- Variables and Equations carry explicit memory effects.
- A static **Borrow Checker** validates the IR during tracing (preventing use-after-consume at compile time).
- The WebGPU backend leverages `Mutate` for zero-copy `var<storage, read_write>` operations.

---

## Milestones

### M0 — Baseline snapshot & type stubs (1–2 days)

Establish regression baselines and define the core effect types.

#### M0.1 — Record baseline test results
**What:** Run the full test suite and Deno tests, record pass/fail counts.
**Commands:**
```bash
pnpm build
pnpm vitest run > tmp/m0-vitest-baseline.txt 2>&1; echo $? > tmp/m0-vitest-exit.txt
pnpm run test:deno > tmp/m0-deno-baseline.txt 2>&1; echo $? > tmp/m0-deno-exit.txt
```
**Exit criteria:** Baseline files exist in `tmp/` with full logs and exit codes.

#### M0.2 — Create Effect type stubs
**What:** Define the effect taxonomy in `src/frontend/jaxpr.ts`.
**Types to define:**
```typescript
export enum MemoryEffect {
  Alloc = "Alloc",     // Creates a new buffer
  Borrow = "Borrow",   // Reads a buffer without taking ownership
  Consume = "Consume", // Takes ownership, buffer cannot be used again
  Mutate = "Mutate"    // In-place modification (requires exclusive ownership)
}

// Update Var and JaxprEqn to track effects
```
**Exit criteria:** Types compile. All existing tests pass.

#### M0.3 — Effect-focused test file
**What:** Create `test/effect-checker.test.ts` with placeholder tests.
**Exit criteria:** Test file runs. Placeholder tests pass.

---

### M1 — IR Effect Types & Tracing (3–5 days)

Propagate effect types through the tracing system.

#### M1.1 — Update `Var` and `JaxprEqn`
**What:** Add `effect` fields to `Var` and `JaxprEqn`. Update the `pprint` system to display effects (e.g., `%1:f32[4] = add %0 {Borrow}`).
**Exit criteria:** `pprint` tests updated and passing.

#### M1.2 — Default Effect Assignment in Tracing
**What:** Update `JaxprTrace.processPrimitive` to assign default effects:
- Inputs to standard elementwise ops are `Borrow`.
- Outputs are `Alloc`.
- Jaxpr outputs are `Borrow` (or `Consume` if returned directly).
**Exit criteria:** Tracing produces Jaxprs with correct default effects.

#### M1.3 — Explicit Mutation Tracing
**What:** Implement explicit `Mutate` and `Consume` tracing for specific primitives (e.g., `DynamicUpdateSlice`).
**Exit criteria:** `DynamicUpdateSlice` emits a `Mutate` effect on its target array.

---

### M2 — Static Borrow Checker (3–5 days)

Implement a static analysis pass to verify memory safety at trace time.

#### M2.1 — Implement `verifyJaxprEffects`
**What:** Create a validator that walks a Jaxpr and enforces ownership rules:
1. No `Borrow` or `Mutate` after `Consume`.
2. `Mutate` requires exclusive ownership (no active `Borrow` aliases).
3. All `Alloc` variables must eventually be `Consume`d or returned.
**Exit criteria:** Validator correctly accepts safe graphs and rejects unsafe ones.

#### M2.2 — Integrate Validator into `makeJaxpr`
**What:** Run `verifyJaxprEffects` automatically at the end of `makeJaxpr` (can be gated behind a debug flag initially).
**Exit criteria:** All existing tests pass with the validator enabled.

#### M2.3 — Fix Effect Violations
**What:** Fix any existing primitives or transform rules that violate the new strict effect rules (e.g., implicit aliasing in scan bodies).
**Exit criteria:** Zero validation errors across the entire test suite.

---

### M3 — JIT Compiler & WebGPU Backend Integration (4–6 days)

Translate IR effects into concrete WebGPU optimizations.

#### M3.1 — Effect-Driven Buffer Recycling
**What:** Replace the heuristic `insertFreeSteps` and `recycleBuffers` passes in `jit.ts` with an effect-driven allocator.
- `Consume` effects explicitly mark the end of a buffer's lifetime.
- Implement graph-coloring register allocation for optimal buffer reuse.
**Exit criteria:** JIT memory peak is equal to or lower than the baseline.

#### M3.2 — Zero-Copy `DynamicUpdateSlice`
**What:** Update the WebGPU backend to compile `Mutate` effects into `var<storage, read_write>` bindings.
- `DynamicUpdateSlice` no longer allocates a new buffer; it writes directly to the input buffer.
**Exit criteria:** `DynamicUpdateSlice` benchmark shows significant speedup and zero allocation overhead.

#### M3.3 — Implement `scatter_add`
**What:** Add a new `scatter_add` primitive using `Mutate` and WebGPU atomics (`atomic<i32>`, `atomic<u32>`).
**Exit criteria:** `scatter_add` works correctly on WebGPU and WASM.

---

### M4 — Autodiff Integration (3–5 days)

Leverage explicit effects to optimize the autodiff pipeline.

#### M4.1 — Residual Borrowing
**What:** Update `linearize` and `vjp` to emit explicit `Borrow` effects for residuals, ensuring the backward pass does not consume them (allowing multiple pullback calls).
**Exit criteria:** VJP reusability invariant is statically verified by the Borrow Checker.

#### M4.2 — Scan Backward Pass Optimization
**What:** Optimize the scan backward pass memory usage using explicit `Consume` effects for checkpoint carries, allowing the JIT to recycle checkpoint memory immediately after the local backward step.
**Exit criteria:** `grad(scan)` memory footprint is reduced.

---

### M5 — Cleanup & Documentation (2–3 days)

#### M5.1 — Remove Legacy Passes
**What:** Remove the old heuristic `insertFreeSteps` and `recycleBuffers` passes from `jit.ts`.
**Exit criteria:** Codebase is clean, tests pass.

#### M5.2 — Update Documentation
**What:** Update `copilot-instructions.md` to document the Effect-Typed IR, the Borrow Checker, and how to write custom primitives with effects.
**Exit criteria:** Documentation reflects the new architecture.

#### M5.3 — Final Regression Run
**What:** Full CI-equivalent check.
**Commands:**
```bash
pnpm build
pnpm run check
pnpm vitest run
pnpm run test:deno
pnpm run lint
pnpm run format:check
```
**Exit criteria:** All checks pass. Zero regressions from M0 baseline.

---

## Dependency Graph

```
M0.1 (baseline)
  │
  ├─→ M0.2 (effect stubs) ──→ M0.3 (test file)
  │
  ├─→ M1.1 (Var/Eqn effects)
  │     │
  │     ├─→ M1.2 (Default tracing)
  │     │
  │     └─→ M1.3 (Mutate tracing)
  │           │
  │           └─→ M2.1 (Borrow Checker)
  │                 │
  │                 ├─→ M2.2 (Integrate validator) ──→ M2.3 (Fix violations)
  │                 │
  │                 └─→ M3.1 (Effect-driven JIT recycling)
  │                       │
  │                       ├─→ M3.2 (Zero-copy DUS)
  │                       │
  │                       ├─→ M3.3 (scatter_add)
  │                       │
  │                       └─→ M4.1 (AD Residual Borrowing)
  │                             │
  │                             └─→ M4.2 (Scan AD optimization)
  │                                   │
  │                                   └─→ M5.1–M5.3 (cleanup)
```

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Borrow Checker is too strict for existing AD patterns | AD tracing fails | Introduce `unsafe_borrow` escape hatches initially; refine AD rules in M4 |
| WebGPU `read_write` storage limits | Kernel compilation fails | Fall back to copy-on-write if `maxStorageBuffersPerShaderStage` is exceeded |
| Graph-coloring allocator increases JIT compile time | Slower first execution | Use a fast linear-scan allocator for large graphs |
| `Mutate` breaks JAX functional purity | User confusion | Restrict `Mutate` to internal JIT/AD passes; expose only functional APIs to users |

## Invariants to Maintain

1. **Functional Public API:** The user-facing API remains purely functional. `Mutate` and `Consume` are internal compiler optimizations.
2. **Slot-count parity:** `slotCount()` before and after any operation matches.
3. **Transform composition correctness:** All depth-1 through depth-4 compositions pass.
4. **WebGPU Peak Memory:** Peak memory usage must not exceed the baseline.

## Estimated Timeline

| Milestone | Effort | Cumulative |
|-----------|--------|------------|
| M0 | 1–2 days | 1–2 days |
| M1 | 3–5 days | 4–7 days |
| M2 | 3–5 days | 7–12 days |
| M3 | 4–6 days | 11–18 days |
| M4 | 3–5 days | 14–23 days |
| M5 | 2–3 days | 16–26 days |

Total: **3–4 weeks** of focused implementation.

## Commit Strategy

- One commit per task (M0.1, M0.2, ..., M5.3).
- Commit message format: `effect-ir M{n}.{m}: {short description}`
- Every commit must pass `pnpm vitest run`.
- Branch off `main` at start. Merge back after M5.3 passes full CI.
