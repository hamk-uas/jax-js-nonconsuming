# WASM Multithreading & Performance Plan

## 1. Problem Statement

The upstream `jax-js` repository achieves **2.90 GFLOP/s** on the WASM matmul benchmark, while our
`jax-js-nonconsuming` fork currently hits **1.11 GFLOP/s**.

Analysis of Chrome traces reveals that upstream successfully dispatches work across 8 Web Workers
simultaneously, while our WASM backend executes entirely on the single main thread.

## 2. The Root Cause: Event-Loop Deadlock & `#canSpinWaitWorkers`

Our codebase contains a `WasmWorkerPool` and an `OrchestratorWorker`, but both are being permanently
disabled in the browser.

In `src/backend/wasm.ts`, we check if the environment supports `Atomics.wait` on the current thread
using `#canSpinWaitWorkers`. Because the Web Spec strictly forbids `Atomics.wait` on the main window
thread, this check returns `false`, gracefully falling back to single-threaded execution.

**Why did we disable it? The `postMessage` Deadlock** Our current parallel architecture attempts a
synchronous dispatch:

1. Main thread writes params to a `SharedArrayBuffer`.
2. Main thread calls `worker.postMessage({ type: "wake" })`.
3. Main thread enters a `while(true)` spin-loop (using `Atomics.load`) waiting for completion.

_The flaw:_ In modern browsers (Chromium especially), `postMessage` goes through the browser's Event
Loop. By immediately entering a `while(true)` spin-loop on the main thread, the Event Loop is
blocked. The `postMessage` is never delivered to the worker, and the main thread spins forever
(deadlock). To prevent this, the system disables the worker pool entirely on the main thread.

## 3. Does the Orchestrator Worker Help?

Our `OrchestratorWorker` (M6.2b) was originally designed to solve this by moving the mega-module
execution off the main thread. If the execution is on a worker, it has its own event loop and can
freely use `Atomics.wait`.

**Does it work right now? No.** Our current Orchestrator relies on the main thread spin-looping to
service `alloc` and `free` proxies (`while(true) { check alloc_req... }`). This triggers the exact
same event-loop starvation issue, so the Orchestrator is also gated behind `#canSpinWaitWorkers` and
is never activated in the browser.

**Can it help if redesigned? Yes.** If the Orchestrator was redesigned to be fully
decentralized—managing its own WASM memory allocations without proxying to the main thread—it could
serve as the master thread. The main thread would just `await` the final result without
spin-looping, and the Orchestrator could freely coordinate the generic `WasmWorkerPool`.

## 4. Alternative Solutions

### Option A: Async Promise-based Dispatch (Browser-First Recommendation)

This is the same broad architectural direction upstream took in PR #102, but the idea should be
framed in our own terms: make in-flight execution explicit instead of pretending results are ready
synchronously.

- **How it works:** Change `WasmWorkerPool.dispatch()` to be asynchronous. Instead of the main
  thread spin-looping, completion is represented explicitly as a completion handle or `Promise`. In
  SAB-backed browser runtimes, the preferred wakeup primitive is `Atomics.waitAsync()` on a shared
  control word: the main thread can await worker completion without blocking the event loop, while
  workers still signal completion with the existing shared-memory protocol plus `Atomics.notify()`.
  Where SAB is unavailable, completion falls back to normal worker messaging.
- **Dependency Tracking:** Add backend-owned pending-work tracking keyed by produced slots, but do
  **not** assume this means a distinct Promise per slot. In the common case, multiple slots produced
  by one dispatch should share one completion handle or producing-program record. If a user calls
  `.data()` or reads a value, the backend resolves that slot through its producing completion state.
- **JSPI where available:** When JavaScript Promise Integration (JSPI) is available, we should use
  it selectively at the JS↔WASM boundary. Wrapped WASM exports can return Promises, and wrapped
  async imports can suspend and resume the WASM computation with straight-line code. This is a good
  fit for coarse exported executors such as mega-module entry points or other WASM compiled-loop
  exports whose host interactions are asynchronous.
- **Interaction with disposal / recycle / mega-module:** Pending-work tracking must gate
  `.dispose()` and `free()` — a slot cannot be freed or recycled while its completion handle is
  still in-flight. The `recycleBuffers()` JIT pass must treat pending slots as live. For mega-module
  execution, the Orchestrator's `alloc`/`free` proxy calls become async messages rather than
  synchronous spin-loops; the Orchestrator itself can run synchronously on its own thread, but the
  main thread must await the final result via the same async completion path instead of spin-looping
  on `STATE_DONE`.
- **Migration constraint:** Option A should change the **control plane**, not the compute
  granularity. We should keep `executeMegaModule()`, native compiled loops, and other coarse
  WASM-side executors as the unit of work. The async boundary belongs around those existing
  program-level calls, not around individual internal steps. This is a strong fit for mega-modules
  and native scan/assoc-scan paths; it is **not** a complete description of the current fallback JIT
  step loop, which still executes as a JS-side sequential interpreter and therefore needs either to
  remain synchronous for now or to gain its own async-capable execution model.
- **Pros:** Never blocks the main UI thread. `Atomics.waitAsync()` gives the browser main thread a
  clean non-blocking way to await a shared-memory completion signal while preserving the existing
  worker-control architecture. Preserves the current strength of this codebase, which is that
  substantial execution already happens inside WASM rather than as many JS dispatches.
- **Cons:** Requires migrating synchronous `dispatch()` / `executeMegaModule()` paths in the backend
  to an asynchronous tracker, and auditing every `.dispose()` / recycle / free call site for
  pending-work safety. The `Backend` interface is the choke point for this work, so an additive
  async/extended interface is safer than mutating the existing synchronous contract in-place.
  `waitAsync()` improves the wakeup primitive, and JSPI improves the WASM boundary ergonomics, but
  neither removes the need for pending-work tracking or async read semantics.

### Option B: Synchronous Wake via `Atomics.notify` (Stopgap / Benchmark-Only)

If we _must_ keep the synchronous dispatch pattern on the main thread, we can bypass the event loop
completely by switching workers from `onmessage`-based wake to `Atomics.wait`-based sleep.

- **The Trick:** Instead of the workers waiting in the JS Event Loop for a
  `postMessage({type: 'wake'})`, the workers run an infinite loop calling
  `Atomics.wait(control, STATE, READY)`.
- **Waking up:** The main thread writes work parameters and calls `Atomics.store(..., WORK)` and
  `Atomics.notify(...)`. This immediately wakes the worker via futex, bypassing the browser's JS
  Event mechanism entirely. The main thread can then safely spin-loop on `Atomics.load` without
  deadlocking!
- **What about `register` / `register-mega` / `destroy` events?** A worker blocked in `Atomics.wait`
  can't process _any_ `onmessage` events. The current worker pool uses `onmessage` for five distinct
  message types: `init`, `control`, `wake`, `register`, `register-mega`, and `destroy` (see
  `worker-pool.ts` lines 65–131). Replacing only the wake path means either (a) adding a new
  `PROCESS_MESSAGES` state to the shared control buffer plus a secondary state machine so workers
  can yield back to the event loop for registration traffic, or (b) moving module registration
  itself into the shared-memory protocol (writing `WebAssembly.Module` handles via `postMessage`
  while the worker is temporarily yielded). Both require a new lifecycle state machine — not a
  narrow swap.
- **Pros:** Preserves synchronous `dispatchSync()` API. No Promises on the hot path. Keeps the
  existing backend architecture.
- **Cons — main-thread stall (not "minor jank"):** The main browser thread enters a busy-spin
  `while(Atomics.load(...) !== READY)` loop for the full duration of the dispatch. During that time,
  **all** rendering, input events, `setTimeout`/`setInterval`, Promise continuations, and
  `requestAnimationFrame` callbacks are frozen. This is the exact behavior that
  `src/backend/wasm.ts` lines 208–221 already document as "fundamentally unsuitable" for browser
  main threads. Acceptable for Node.js benchmarks or one-shot CLI tools, but not for interactive
  browser applications.
- **Status:** Stopgap / proof-of-concept only. Suitable for validating that the futex wake path
  works in Chromium before committing to Option A.

## 5. Updated Recommendation

After the review rounds, the recommendation should no longer be stated as a single global winner.
The right recommendation depends on **runtime** and **phase**.

### 5.1 Browser recommendation

We should still pursue **Option A (Async Promise Dispatch)** as the primary browser-facing
direction. That conclusion survived every review round: it is the only option that directly
addresses the fundamental browser problem, which is main-thread/event-loop coordination rather than
lack of parallel work. It also aligns with the current backend contract drift we would eventually
have to handle anyway (`pending` work, safe disposal, sync reads).

Where the runtime supports it, we should also use **JSPI at selected JS↔WASM boundaries** as a
companion mechanism inside Option A. The important constraint is scope: JSPI is about letting a WASM
export suspend on async JS imports and resume later, not about replacing worker coordination,
pending-work tracking, or the overall async dispatch model.

### 5.2 Arena recommendation

We should promote **Option E (WASM Arena / Workspace Planning)** from a speculative side option to
an **early implementation track**, not just a later companion to Option A.

Recommended phasing:

1. **E1: fixed-size arena for mega-modules.** Use `stepInfos` / liveness to replace internal
   mega-module malloc/free traffic with planned offsets.
2. **E2: parametric workspace arena for predictable polymorphic compiled loops.** Blocked
   associative scan is now the clearest example: its runtime-sized scratch is formulaic in `N`, not
   arbitrary.

Why this changed: the code review established that Option E is stronger and lower-risk than it first
looked. `MegaStepInfo[]` already gives us concrete malloc/free/recycle structure for mega-modules,
and blocked associative scan already computes a closed-form workspace from `N`, block size, and leaf
widths. That makes E1/E2 independently valuable performance work that also simplifies the later
async/orchestrator story, rather than work that has to wait behind it.

### 5.3 Node recommendation

For **Node**, the immediate recommendation is **not** to start with Option A. The current blocker in
this environment is that the implementation expects a web-style global `Worker`, while plain Node
here exposes shared memory and legal spin-waits but no `Worker` global. So the first Node-specific
action should be a worker-runtime abstraction or `node:worker_threads` adapter.

Once that exists:

- **Option C** becomes materially more attractive in Node than in browsers.
- **Option E** also becomes more valuable, because allocator/orchestrator simplification compounds
  with legal synchronous coordination.
- **Option A** remains useful if we want one cross-runtime async model, but it is no longer the
  obvious first move for Node specifically.

### 5.4 Options to deprioritize

- **Option B** should remain a stopgap / benchmark-only path. The later findings do not improve its
  browser suitability.
- **Standalone Option F** should remain deprioritized as an architecture. However,
  `Atomics.waitAsync()` should now be treated as the preferred wakeup primitive inside the
  SAB-backed Option A path rather than as a speculative side note.
- **JSPI** should not be treated as a replacement architecture either. It is a boundary-layer
  enhancement for runtimes that support it.

### 5.5 Bottom line

The recommendation should now read as:

- **Browser primary:** Option A, preferably using `Atomics.waitAsync()` as the completion wait
  primitive when SAB-backed workers are available.
- **Browser boundary enhancement where available:** JSPI on selected exported executors / async
  imports, to keep coarse WASM-side logic straight-line across async host calls.
- **Browser fallback without SAB / COOP-COEP:** Option D.
- **Early cross-cutting implementation track:** Option E, in two phases (fixed-size then parametric
  workspace).
- **Node first step:** worker abstraction / `node:worker_threads` integration, then revisit C vs A
  on top of that.

If we eventually want the Orchestrator pattern in browsers, it should still be built only _after_
the async pipeline is stable. But Option E no longer has to wait behind A at all; E1/E2 are credible
first implementations because the arena/workspace logic is useful in its own right and reduces later
coordination pressure.

### 5.6 Minimal-change migration path

The migration should be organized around one principle: **preserve coarse WASM-side execution,
change only how the host waits for it.** Our current mega-module path is an asset and should remain
the center of the design.

Recommended sequence:

1. **Keep the existing execution units.** `executeMegaModule()`, native `scan` compiled-loop,
   blocked `associativeScan`, and other already-coarse WASM entry points stay intact. Do **not**
   decompose them into many JS-managed sub-dispatches just to make async easier.
2. **Implement E1 first: fixed-size arena for mega-modules.** `stepInfos` already exposes enough
   local liveness information to replace internal mega-module malloc/free traffic with planned
   offsets. This is low-risk, gives standalone performance value, and directly simplifies any future
   Orchestrator path.
3. **Implement E2 next: parametric workspace for blocked associative scan and similar compiled
   loops.** These paths already have predictable runtime-sized scratch formulas, so a single
   workspace allocation plus offset carving is more realistic than generic allocator traffic.
4. **Add pending-work tracking at slot ownership boundaries.** The first async semantic change is
   not compiler structure; it is slot lifecycle. `dispose`, `free`, recycle, `data`, and `dataSync`
   need clear rules for values whose producing program is still running.
5. **Introduce async completion at coarse program boundaries.** Make worker-pool and mega-module
   completion explicit via a completion handle / Promise, but keep the internal step orchestration
   where it already belongs: inside WASM or inside the existing worker-side executor.
6. **Use Option A when SAB-backed workers are available.** In cross-origin-isolated browser
   environments, the main thread becomes an async submitter of coarse WASM jobs rather than a
   synchronous spinner.
7. **Prefer `Atomics.waitAsync()` over message-only completion waits in the SAB path.** This keeps
   the control protocol close to the existing shared-memory design: workers still publish state
   through shared control words and signal completion with `Atomics.notify()`, while the main thread
   awaits that state transition without blocking the event loop.
8. **Use JSPI only after there are real async host imports/exports to suspend on.** This is
   especially attractive for coarse WASM executors that need async host calls but should otherwise
   keep straight-line WASM control flow. We should wrap only the exports and imports that actually
   suspend; most boundaries should stay unwrapped.
9. **Use Option D as the browser fallback shell, not as a different compute architecture.** Without
   SAB/COOP-COEP, move the same style of coarse execution into a runtime worker. The worker should
   still favor mega-modules and other WASM-side executors; the UI thread should only handle API-edge
   upload/download and result materialization.

What this migration explicitly avoids:

- turning mega-modules into many JS callbacks or per-step Promise chains
- moving orchestration out of WASM when it already composes well there
- treating Option D as a reason to abandon the current mega-module-heavy backend shape

One explicit caveat belongs in the migration text: the current non-mega fallback
`JitProgram.execute()` path is still a JS-side sequential step interpreter. Boundary-only async
changes map cleanly to mega-modules and native compiled loops, but not to that fallback path. We
should therefore either keep the fallback interpreter synchronous initially or give it a separate
async-capable execution model rather than pretending the same boundary rule covers every program
shape.

### 5.7 Likely answers to the remaining implementation questions

The feasibility review answered several open questions well enough that the plan can now state
likely implementation directions instead of treating them as fully open.

1. **Completion should become explicit at coarse program boundaries.** For async WASM paths,
   `dispatch()` / `executeMegaModule()` should conceptually return both result slots and a
   completion handle. In SAB-backed browser runtimes, `Atomics.waitAsync()` is now the most
   practical way for the main thread to await that completion signal without blocking. The important
   semantic change is that input-release, `free`, recycle, and disposal decisions must be tied to
   actual completion rather than to the moment work is submitted.
2. **Pending-work tracking belongs at slot ownership boundaries.** A backend-owned pending-work
   table keyed by slot or produced value is the most plausible way to enforce this. The goal is not
   per-step Promise chaining inside mega-modules, and not necessarily a distinct Promise per slot;
   the natural unit is usually the producing program or completion handle shared by all of that
   dispatch's outputs. The existing `PendingExecute` pattern in `array.ts` is a useful local
   precedent, but it currently tracks lazy prepare/submit semantics rather than backend completion
   after submission.
3. **`waitAsync()` reduces plumbing, but not semantics.** Because `Atomics.waitAsync()` returns an
   object whose `value` is either an immediate status (`"not-equal"`, `"timed-out"`) or a Promise
   that fulfills with `"ok"` / `"timed-out"`, it can sit neatly under a completion-handle
   abstraction. That makes it a good fit for waiting on program completion without requiring
   message-based completion bookkeeping in every SAB-backed case or a full Promise-per-slot tracker
   in the common multi-output case. It still does not eliminate pending-work tracking for ownership
   or reads.
4. **The `Backend` interface is the architectural choke point.** Because `dispatch()`, `readSync()`,
   `read()`, and slot lifetime operations all sit on the shared backend contract, an additive
   async-capable extension interface is lower-risk than mutating the existing synchronous interface
   in-place. CPU and single-threaded WASM paths can satisfy the extended contract with
   already-resolved completions.
5. **JSPI improves boundary ergonomics, not ownership semantics.** When available, JSPI lets
   selected WASM exports return Promises and lets selected async JS imports suspend/resume WASM with
   very little internal restructuring. That is attractive for keeping mega-module-style or
   compiled-loop WASM bodies straight-line across async host calls. But JSPI cannot suspend
   JavaScript itself, and it does not remove the need for async-aware backend state once control has
   crossed back into JS. It also has little immediate value until those exports/imports actually
   become async.
6. **Browser `readSync()` / `dataSync()` should not pretend pending worker-backed results are
   already available.** On the browser main thread, the likely contract is to throw if a synchronous
   read targets still-pending async work, and to steer callers toward `.data()` /
   `blockUntilReady()`. Worker-only and Node-like runtimes can keep a blocking implementation where
   that is actually legal.
7. **Internal synchronous read sites will need an explicit audit.** The main issue is not user
   ergonomics but internal assumptions that output slots are immediately ready after submission.
   Wherever possible, dependent work should stay inside the existing coarse WASM executors;
   otherwise those boundaries need to become async-aware. The good news is that the current WASM
   backend's single-threaded `readSync()` is still just a memory copy, so existing sync-heavy tests
   do not have to become async immediately if pending-work checks short-circuit in already-complete
   cases.
8. **Option E is an early track, not just a companion to Option A.** We do not need to wait for
   async dispatch to start it. Fixed-size mega-module arenas and parametric scan workspaces deliver
   standalone value and reduce the amount of later coordination work.
9. **The fallback JIT interpreter needs explicit scope.** `executeMegaModule()` and native compiled
   loops are natural async boundaries; the non-mega fallback `JitProgram.execute()` loop is not. The
   plan should treat that as a separate execution surface rather than quietly assuming the same
   migration step covers both.
10. **Option D should preserve worker-local residency rather than become a clone-heavy RPC system.**
    The UI side should treat arrays as proxies or handles to worker-owned slots. Data should cross
    the boundary only at explicit API edges, and those transfers must use transferables; structured
    clone and per-primitive round-trips are outside the intended design.

### Option C: The Autonomous Orchestrator (No Main-Thread Proxies)

If we want to keep JAX dispatch synchronous _internally_ without blocking the UI, we could deploy an
Orchestrator Worker that totally owns the WebAssembly memory.

- **How it works:** Instead of the Orchestrator bouncing `#malloc` and `#free` requests back to the
  main thread via shared memory locks, the Orchestrator instantiates its own `WasmAllocator`. The
  main thread acts merely as a fire-and-forget RPC client:
  `worker.postMessage({cmd: 'runProgram', ...})`.
- **Pros:** The Orchestrator thread can legally use `Atomics.wait()` because it is not a window
  thread. It can manage the 8-thread worker pool flawlessly using synchronous spin-waits.
- **Cons:** Requires migrating memory ownership wholly into the thread.

### Option D: The "UI is a Dumb Terminal" Approach (Full-Runtime Worker)

Move the entire JAX-JS system (Tracing, JIT compiling, Array allocation, WebGPU/WASM execution)
inside a Web Worker.

- **Pros:** Because the entire user script runs in a worker, `Atomics.wait` just works and no
  backend compute code changes are needed. **Uniquely, this option also provides value without
  COOP/COEP headers** — even without `SharedArrayBuffer`, the runtime worker keeps all intermediates
  local and only crosses the boundary at API edges (input upload / output download), preserving UI
  responsiveness. As a fallback, it can preserve the current backend's strengths if the worker
  continues to use mega-modules and other coarse WASM-side executors rather than bouncing execution
  back into many UI-thread RPCs.
- **Cons:** Major API boundary work — the proxy layer between the UI thread and the runtime worker
  _becomes_ the architecture. Users can no longer arbitrarily touch JS objects from the window
  during execution inside JAX bodies. The RPC serialization overhead on each API call may dominate
  for fine-grained interactive use cases, so the boundary must stay coarse.

### Option G: JSPI At Selected WASM Boundaries

Use JavaScript Promise Integration (JSPI) where the runtime supports it, but keep it scoped to the
JS↔WASM boundary rather than treating it as a replacement for the worker or ownership design.

- **What it is:** JSPI lets selected WebAssembly exports return Promises and lets selected
  JavaScript imports that themselves return Promises suspend and later resume the WASM computation.
  The key benefit is that WASM code can remain straight-line across async host calls with minimal
  internal restructuring.
- **How it would fit here:** JSPI is most attractive for coarse exported WASM executors such as
  mega-module entry points, compiled-loop exports, or other WASM bodies that occasionally need an
  async host import. Instead of manually splitting those executors around host async boundaries, a
  wrapped export can surface a Promise while the suspended WASM resumes when the host Promise
  resolves. In the current codebase, that value is mostly future-facing because the relevant
  mega-module imports (`alloc` / `free`) are still synchronous today.
- **Pros:** Minimal boundary churn where it applies. Good fit for preserving the existing coarse
  WASM-side execution model. Particularly attractive if some current JS imports (allocation proxies,
  async service calls, future loader hooks) become naturally Promise-returning.
- **Cons:** JSPI only helps at the JS↔WASM boundary. It does **not** solve worker-pool
  coordination, pending-work tracking, disposal ordering, or `dataSync()` semantics by itself. It
  also cannot suspend JavaScript itself; once control is back in JS, normal async rules still apply.
  Not every export/import should be wrapped.
- **Availability / rollout:** Treat as feature-detected and opportunistic. The V8 write-up presents
  it as standardized and available in modern Chrome and Firefox, but we should still gate it at
  runtime and keep the existing `waitAsync()`-based / message-based async paths as fallback.
  Sequence it after there are actual async host imports/exports worth wrapping.
- **Bottom line:** JSPI should be used where available, but as a boundary enhancement inside Option
  A or hybrid async paths, not as a new primary architecture.

### Option E: WASM Colored Arena Pre-allocation (JIT-Time Memory Planning)

Bring the WebGPU JIT "Colored Arena" memory management model and apply it to WASM.

- **The Flaw We're Fixing:** The Orchestrator worker currently blocks on `Atomics.wait` to ask the
  main thread to dynamically allocate and free memory (`malloc_req` / `free`).
- **How it works:** Instead of runtime dynamic allocations during mega-module execution, the JIT
  tape pre-computes array lifetimes via structural analysis of the conflict graph at compile time.
  It assigns arrays to specific "colors" (memory slabs/offsets), turning dynamic mallocs into static
  WASM heap offsets known _before_ execution starts.
- **Pros:** Radically simplifies the Orchestrator. The orchestrator never has to proxy an allocation
  call to the main thread. It's a pure compute task. The main thread can just wait for completion.
  Completely removes the `STATE_ALLOC_REQ` proxy wait deadlock.
- **Cons:** Requires rewriting the WASM JIT path to use the WebGPU arena logic. Intermediate memory
  is strictly pre-allocated, meaning memory isn't reclaimed mid-run via a generic heap. Can't
  support dynamically-sized outputs inside JIT.
- **Feasibility in this repo:** **Moderate-to-high for two scoped targets: (1) the existing
  mega-module subset, and (2) parametric-workspace compiled loops such as blocked WASM associative
  scan.** The current mega-module compiler already rejects symbolic malloc sizes, pass-through
  outputs, and many dynamic step types, and it emits concrete `stepInfos` entries containing exact
  malloc/free/recycle boundaries. Separately, blocked WASM associative scan does have runtime-sized
  scratch space, but that scratch is not arbitrary: the dispatcher computes it from a small
  closed-form layout based on `N`, `B`, and leaf byte widths (`ping/pong = totalLeafElemSize * N`,
  summaries = `totalLeafElemSize * ceil(N / B)`, plus fixed internal buffers). That means Option E
  does **not** have to be all-or-nothing between "fully static" and "unsupported dynamic sizes".
- **Practical shape of an MVP:** Start with mega-module-only arena planning, preferably for the
  existing `stepInfos` path before touching the monolithic `mega_execute` import ABI. Required work
  is finite and local: (1) compute local liveness from `stepInfos`, (2) assign offsets or colors,
  (3) grow memory once up front if needed, (4) replace internal `malloc`/`free` with local pointer
  assignment, and (5) preserve normal allocation for result buffers and non-mega-module execution.
  This is a credible optimization/refactor path, not just theory.
- **What the associative-scan connection adds:** It suggests a natural Phase 2 for Option E: a
  **parametric arena** or workspace descriptor, where compile time fixes the layout formula and
  dispatch time plugs in `N`. For blocked associative scan, a single invocation could allocate one
  contiguous workspace block sized from `N`, then carve out `pingPtr`, `pongPtr`, `summaryPingPtr`,
  `summaryPongPtr`, and fixed internals by precomputed offsets instead of doing multiple allocator
  round-trips. That is still an arena, just not a constant-size one.
- **Polymorphic-length caveat:** WASM already has polymorphic-length compiled loops today, not just
  for native `scan` compiled-loop but also for blocked native `associativeScan`. Both paths pass the
  trip count `N` as a runtime `i32` and reuse one compilation across lengths. This weakens the
  earlier simplistic objection to arenas for dynamic shapes: the right distinction is not "static vs
  dynamic" but **"arbitrary runtime allocation vs predictable runtime-sized workspace."**
  Mega-modules stay in the fully static bucket; blocked associative scan fits the
  predictable-workspace bucket.
- **Associative-scan parallelism caveat:** `associativeScan` is indeed structurally more parallel
  than `scan` (parallel rounds, blocked decomposition, low span). That improves the **payoff** of a
  future worker-backed WASM strategy and makes a parametric arena more valuable there. But in the
  current implementation, the WASM blocked associative-scan path still dispatches a single
  `blocked_assoc_scan` export on one WASM instance. So this changes the attractiveness of Option E
  more than it changes the conclusion about Option A: it gives us a better second target for arena
  planning, but it still does not solve browser-side worker coordination by itself.

### Option F: `Atomics.waitAsync` as the SAB-Path Wakeup Primitive

Rather than treating `Atomics.waitAsync()` as a competing architecture, we should now treat it as
the preferred non-blocking wait primitive inside the SAB-backed async design.

- **What MDN changes practically:** `Atomics.waitAsync()` is now broadly available across modern
  runtimes and browsers, including main-thread browser use. It works on an `Int32Array` or
  `BigInt64Array` view over a `SharedArrayBuffer` and returns an object with `{ async, value }`:
  either an immediate status such as `"not-equal"` / `"timed-out"`, or a Promise that fulfills with
  `"ok"` / `"timed-out"`. The Promise never rejects.
- **How it fits this plan:** In the SAB-backed Option A path, workers can continue publishing
  completion through shared control words and `Atomics.notify()`. The main thread can then use
  `waitAsync()` to await that state change without blocking the event loop. This is a better fit
  than main-thread spin loops and often cleaner than message-only completion plumbing.
- **Pros:** Gives the browser main thread a futex-like non-blocking wait that maps well onto the
  current shared-memory worker design. Allows a completion-handle abstraction to be built from the
  existing control buffer rather than requiring message completion in every case. It is a practical
  improvement, not just a theoretical one.
- **Cons:** It still requires SAB and therefore COOP/COEP. It only changes how completion is
  awaited; it does not change ownership, recycling, disposal, or read semantics. If a location is
  already not equal to the expected value, the fast path returns immediately, so callers still need
  correct state management around the control word.
- **Feasibility in this repo:** **High as an implementation detail inside Option A or a hybrid SAB
  path; low as a standalone architecture.** This is the cleanest way to modernize the main-thread
  wait story without discarding the current shared-memory control scheme.
- **Bottom line:** `waitAsync()` should no longer be framed as a marginal curiosity. It is now the
  preferred wakeup primitive for the SAB-backed async path. What it does **not** do is remove the
  need for pending-work tracking, safe disposal, or async dispatch / read semantics.
- **Polymorphic-length interaction:** `waitAsync` does not materially improve or worsen WASM
  polymorphic length support. Symbolic-length kernels already pass resolved runtime sizes at
  dispatch time; the hard part remains ownership and synchronous-read semantics after dispatch, not
  dimension binding itself.

---

## 6. Cross-Origin Realities & COOP/COEP Implications

Multithreading WebAssembly relies fundamentally on `SharedArrayBuffer` (SAB). Because of Spectre
mitigations, modern browsers require strict security headers to enable SAB:

- `Cross-Origin-Opener-Policy: same-origin` (COOP)
- `Cross-Origin-Embedder-Policy: require-corp` (COEP)

How this limitation intersects with our worker architecture options:

### 1. If COOP/COEP Headers ARE Present (Cross-Origin Isolated = `true`)

- `SharedArrayBuffer` is enabled.
- **Implication:** _All options (A, B, C, D, E, F, G) are available._ We can freely spawn
  `WasmWorkerPool` threads, build an `OrchestratorWorker`, and share memory between them and the
  main thread. This is also the prerequisite for using `Atomics.waitAsync()` as the preferred
  main-thread completion wait in the Option A path.

### 2. If COOP/COEP Headers ARE NOT Present (Cross-Origin Isolated = `false`)

- `SharedArrayBuffer` is forcefully disabled by the browser. `Atomics` is largely useless.
- **Effect on shared-memory parallelism (Options A, B, C, E, F):** Worker Pool and Orchestrator
  approaches can **no longer share memory**. The only way to move data is via `Transferable`
  messages or structured cloning. The serialization overhead for a 4096×4096 matrix (~64 MB)
  completely destroys any multithreading gains. These options are actively detrimental without SAB.
  In particular, `Atomics.waitAsync()` is unavailable here because it only works on an `Int32Array`
  or `BigInt64Array` backed by `SharedArrayBuffer`.
- **What this does _not_ rule out:** JSPI itself is not fundamentally tied to `SharedArrayBuffer`,
  because it works at the Promise boundary between JS imports/exports and WASM. However, in this
  plan its main value is as a companion to the async worker-backed paths, not as a replacement for
  the lack of shared memory.
- **Exception — Option D (Full-Runtime Worker):** A single runtime worker that owns all WASM state
  can still be useful _without_ SAB. It won't get shared-memory multithreading, but it keeps all
  intermediates worker-local and only crosses the boundary at API edges (input upload / output
  download). This preserves UI responsiveness during long-running computations. The worker degrades
  to single-threaded WASM internally, but the main thread stays free.
- **Verdict:** Shared-memory parallelism (Options A/B/C/E/F) requires SAB and therefore COOP/COEP.
  Option D can provide a useful UI-responsiveness fallback even without these headers, at the cost
  of API-boundary serialization. JSPI may still be usable as a boundary mechanism in some no-SAB
  environments, but it does not by itself restore shared-memory parallelism.

### Conclusion on Headers

**Shared-memory multithreaded WASM requires `SharedArrayBuffer`, which mandates COOP/COEP headers.**
Options A, B, C, E, and F are all dead without them.

However, "no SAB" does not mean "no useful worker deployment." Option D (Full-Runtime Worker) can
still offload compute to a single worker for UI responsiveness, even if it runs single-threaded WASM
internally.

If a downstream user is trying to embed `jax-js-nonconsuming` on a 3rd-party domain or a site where
they cannot manage HTTP headers (e.g. general CDNs, iframe widgets), the library must support three
tiers:

1. **With COOP/COEP:** Full shared-memory multithreading (Option A preferred).
2. **Without COOP/COEP, with Workers:** Single-threaded WASM in a runtime worker (Option D fallback)
   — UI stays responsive.
3. **Without Workers:** Synchronous single-threaded WASM on the main thread (current fallback via
   `#canSpinWaitWorkers` + `typeof SharedArrayBuffer` checks).

Upstream's async dispatch (Option A) remains the recommended primary path because it supports
environments _with_ COOP/COEP cleanly without freezing the UI, while naturally degrading through
tiers 2–3 when SAB is unavailable.

---

## 7. Non-Browser Runtime Status (Node / Deno)

The browser deadlock analysis should not be projected wholesale onto non-browser runtimes.

### Node.js today

- **What is already true:** Node is treated as a supported runtime for CPU and WASM in the test
  suite. The WASM backend explicitly documents that `Atomics.wait` works on the Node main thread and
  that message delivery is independent of the browser event loop, so the browser main-thread
  deadlock argument does not apply there.
- **What we observed in this environment:** Plain Node v24 here exposes `SharedArrayBuffer` and the
  backend reports `canSpinWaitWorkers = true`, but `typeof Worker === "undefined"`. Because the
  current worker pool and orchestrator are gated on `typeof Worker !== "undefined"`, they still do
  **not** activate in this runtime even though spin-waiting itself is legal.
- **Implication for the options:**
  - Option B is still poor for browsers, but in Node it is an engineering tradeoff rather than a
    correctness trap because freezing a GUI event loop is not the issue.
  - Options C and E are more attractive in Node than in browsers because synchronous coordination is
    legal there.
  - Option F is less compelling in Node because the main reason to use it—avoiding browser UI
    starvation—is absent.
  - Option G (JSPI) is potentially interesting in V8-based runtimes as a way to simplify selected
    async JS↔WASM boundaries, but it still does not remove the need for a worker runtime
    abstraction in this repo.
- **Actual blocker in Node:** The current implementation assumes a web-style global `Worker` API and
  creates Blob-URL module workers. To make parallel WASM paths work reliably in Node, we likely need
  an adapter over `node:worker_threads` (or an explicit runtime abstraction for worker creation)
  rather than more work on futex semantics.

### Deno / worker-like runtimes

- The code already carries small Deno-specific lifecycle comments around delayed Blob URL
  revocation, which suggests the Worker construction model is expected to work better there than in
  plain Node.
- That makes Deno-like runtimes closer to the browser-worker model or dedicated-worker model than to
  plain Node's current global environment.

### Practical conclusion for runtime prioritization

- **Browser:** Option A remains the primary direction because the event-loop problem is fundamental.
- **Node:** The immediate gap is worker API integration, not `Atomics.wait` legality.
- **WASM polymorphic compiled loops:** Existing polymorphic-length support in native scan and
  blocked associative scan is not just a constraint; it also points to a more general arena design.
  After a fixed-size mega-module MVP, the next plausible extension is a parametric arena for
  predictable runtime-sized workspaces.
- **Associative-scan upside:** Because associative scan is genuinely parallelizable, it raises the
  ceiling for future Node or worker-backed WASM execution more than ordinary sequential scan does.
  More importantly for Option E, it provides a concrete non-mega-module use case where runtime-sized
  scratch space is regular enough to arena-pack.

---

## 8. Chrome Timing Model For Option Comparison

To compare the plans quantitatively, we need the cost of the actual building blocks they are
composed from. The table below is measured in **HeadlessChrome 145** on this Linux machine
(`hardwareConcurrency = 18`, `crossOriginIsolated = true`) using the reproducible harness in
`tmp/measure-chrome-plan.mjs`.

### 8.1 Measured primitive timings (Chrome)

All numbers below are **median wall time** unless otherwise stated.

| Primitive                       | Measurement                                   |          Median | Notes                                                                     |
| ------------------------------- | --------------------------------------------- | --------------: | ------------------------------------------------------------------------- |
| Promise microtask               | `await Promise.resolve()`                     | ~0.000-0.005 ms | Below timer resolution; effectively free compared to worker/message costs |
| Worker cold create + first ping | Blob URL module worker                        |    **3.885 ms** | First sample 4.57 ms                                                      |
| Small worker round-trip         | `postMessage('ping') -> 'pong'`               |    **0.015 ms** | Current Option A-style completion signal scale                            |
| Futex wake + ack                | `Atomics.notify` worker wake, ack via message |    **0.010 ms** | Lower bound for Option B/C-style shared-control wake                      |
| Module register + instantiate   | Empty `WebAssembly.Module` to existing worker |    **0.015 ms** | Lower bound; real kernels can be higher                                   |
| `ArrayBuffer` alloc             | 64 MB                                         |    **0.030 ms** | First sample can spike ~0.52 ms                                           |
| `SharedArrayBuffer` alloc       | 64 MB                                         |    **0.010 ms** | Very cheap in this Chromium build                                         |
| WASM memory create              | 64 MB unshared `WebAssembly.Memory`           |    **0.025 ms** | p95 ~0.395 ms                                                             |
| WASM memory create              | 64 MB shared `WebAssembly.Memory`             |    **0.005 ms** |                                                                           |
| WASM memory grow                | +64 MB unshared                               |    **0.130 ms** |                                                                           |
| WASM memory grow                | +64 MB shared                                 |    **0.005 ms** | Chromium appears to amortize this heavily                                 |
| WasmAllocator malloc            | 4 KB cold                                     |    **0.005 ms** | Small alloc path                                                          |
| WasmAllocator malloc            | 4 KB reuse                                    |    **0.005 ms** | Same order as one message RTT                                             |
| WasmAllocator free              | 4 KB                                          |    **0.005 ms** |                                                                           |
| WasmAllocator malloc            | 64 MB cold                                    |    **0.145 ms** | Includes grow/size-class bookkeeping                                      |
| WasmAllocator malloc            | 64 MB reuse                                   |    **0.115 ms** | Reuse still pays for zero-fill                                            |
| WasmAllocator free              | 64 MB                                         |    **0.145 ms** |                                                                           |
| Worker transfer round-trip      | 64 MB transferable `ArrayBuffer`              |    **0.690 ms** | About 0.345 ms one-way if symmetric                                       |
| Worker clone round-trip         | 16 MB structured clone                        |    **24.63 ms** | Catastrophic if transferables are not used                                |

### 8.2 Immediate quantitative conclusions

1. **Async Promise dispatch is not meaningfully slower than sync futex wake in Chrome.**
   - Option A control overhead per kernel is on the order of one small dispatch message plus one
     completion message: roughly **2 × 0.015 ms + <=0.005 ms** Promise work.
   - Option B/C-style futex wake saves only a few microseconds relative to that (`0.010 ms` vs
     `0.015 ms` scale). The difference is too small to justify browser main-thread spin semantics.

2. **Worker startup is a one-time tens-of-milliseconds event, not a per-dispatch problem.**
   - One worker creation is ~3.9 ms median.
   - An 8-worker pool costs roughly **31 ms** median to create serially.
   - An 8-worker pool plus 1 orchestrator is roughly **35 ms** median serial startup, with cold
     samples plausibly in the **40+ ms** range.
   - This is acceptable if workers are persistent, unacceptable if recreated per invocation.

3. **Transferable messaging is viable; structured clone is not.**
   - 64 MB round-trip transfer is only ~0.69 ms.
   - 16 MB round-trip clone is already ~24.6 ms.
   - Therefore any Option D / no-SAB worker design must enforce transferables or persistent
     worker-local residency. Accidentally falling back to clone destroys the plan.

4. **Allocator cost is real but bounded; for small buffers, coordination can dominate.**
   - A 4 KB malloc/free is only ~0.005 ms, which is the same order as a Promise hop or futex wake.
   - A 64 MB malloc/free is ~0.115-0.145 ms per operation.
   - Therefore Option E matters most in two cases:
     - many small alloc/free calls, where per-allocation coordination overhead rivals the allocator
       itself;
     - repeated large scratch alloc/free, where the allocator cost itself becomes visible.

### 8.3 Plan-by-plan expected timing model

These formulas use the measured primitives above. They are not exact wall-clock predictions for a
whole program; they are the cost terms that distinguish the plans.

#### Option A — Async Promise Dispatch

**Cold startup:**

- Worker pool startup: ~`3.9 ms × W` where `W` is number of workers.

**Steady-state per parallel kernel dispatch:**

- dispatch signal + completion wait settlement
- in a message-based implementation, approximately **0.03-0.04 ms** control overhead per dispatched
  kernel
- in a `waitAsync()`-backed SAB implementation, the control cost should stay in the same
  microsecond-class regime as futex wake plus Promise settlement rather than introducing a new large
  overhead term

**Interpretation:**

- This is already small enough that any kernel taking >=0.1 ms will be dominated by compute, not by
  async orchestration.
- `waitAsync()` strengthens the practical case for Option A because it improves the completion
  primitive without changing that basic cost model.
- The main engineering cost of Option A is ownership/pending-work semantics, not raw Chrome
  overhead.

#### Option B — Sync `Atomics.notify` Wake On Main Thread

**Steady-state per kernel:**

- futex wake path is marginally cheaper than Option A in raw control cost
- best-case savings are only on the order of **~0.005-0.02 ms** per kernel

**Interpretation:**

- The browser UI freeze cost is effectively the entire kernel duration, so the microsecond control
  savings do not matter.
- The timing data makes B look even worse, not better: the measurable control-speed win is tiny.

#### Option C — Autonomous Orchestrator (Without Arena)

**Cold startup:**

- orchestrator + worker pool startup: roughly **35 ms** median for 9 workers total (8 pool + 1
  orchestrator), serial creation

**Steady-state per mega-program:**

- main thread -> orchestrator dispatch signal: ~message scale
- orchestrator -> worker wake signals: ~futex/message scale
- **plus per alloc/free proxy traffic** if internal buffers are still dynamically allocated

For `K_small` 4 KB allocations and `K_large` 64 MB allocations, the allocator/proxy term is roughly:

`K_small × (0.005 ms allocator + 0.01-0.015 ms coordination)`

`K_large × (0.115-0.145 ms allocator + 0.01-0.015 ms coordination)`

**Interpretation:**

- C is attractive only if the orchestrator does enough work per dispatch to amortize startup and if
  alloc/free chatter is controlled.
- Without Option E, allocator/proxy traffic can easily add tenths of a millisecond to
  low-single-millisecond programs.

#### Option D — Full Runtime Worker

**With SAB:**

- behaves more like a worker-owned runtime; messaging cost is mostly API-edge orchestration

**Without SAB, but with transferables:**

- one 64 MB upload + one 64 MB download costs roughly **0.69 ms round-trip**

**Without transferables / accidental clone:**

- even 16 MB round-trip clone is ~24.6 ms

**Interpretation:**

- Option D remains viable only if large payloads cross the boundary rarely and always via transfer.
- Fine-grained interactive RPC or clone-heavy designs are not competitive.

#### Option E1 — Fixed-Size Arena For Mega-Modules

**What it removes:**

- repeated internal allocator calls during mega-module execution
- any future alloc/free proxy traffic for those same internal buffers

**Expected savings:**

- roughly linear in internal malloc/free count
- if a mega-module would otherwise perform `K` internal large-buffer alloc/free pairs, savings are
  on the order of **`K × 0.26-0.29 ms`** just from allocator time, before counting proxy
  coordination

**Interpretation:**

- E1 is most valuable on orchestration-heavy mega-programs, not because JS malloc is slow in
  absolute terms, but because it is avoidable repeated work.

#### Option E2 — Parametric Workspace Arena (Blocked Associative Scan Class)

Blocked associative scan is the clearest example where runtime-sized scratch is still predictable.
Today it allocates:

- `ping`
- `pong`
- `summaryPing`
- `summaryPong`
- fixed internal buffers

The first four are formulaic in `N`, so E2 can replace four allocator round-trips with one workspace
sizing calculation plus offset carving.

**Expected savings for large scans:**

- at minimum, remove **4 large allocs + 4 frees** worth of allocator traffic for the ping/pong +
  summary buffers
- if those buffers are large, that is roughly **1.0-1.2 ms** of allocator work removed at the 64 MB
  scale used in the microbenchmarks, before counting any cross-thread proxy overhead

**Interpretation:**

- This is the strongest quantitative reason Option E moved up in priority after the review rounds.
- Associative scan does not just raise the parallelism ceiling; it provides a concrete workload
  where parametric-workspace planning should pay off materially.

#### Option F — `Atomics.waitAsync`

**Raw timing effect:**

- Promise overhead is <=0.005 ms; wait completion is still driven by the same shared-memory signal
  path and `Atomics.notify()` cadence

**Interpretation:**

- The timing data confirms `waitAsync()` is not a separate performance architecture.
- Its importance is practical: it removes the main-thread blocking problem from the SAB wait path
  without forcing the design into message-only completion.
- This is why it should now be treated as the preferred implementation detail inside Option A rather
  than as an alternative to it.

#### Option G — JSPI

**Raw timing expectation:**

- JSPI is primarily a boundary-ergonomics feature, not a throughput story.
- The V8 write-up describes suspend/resume overhead as essentially constant-time and shows roughly
  microsecond-scale overhead for a deliberately Promise-heavy demo, which is far smaller than real
  browser I/O latencies but still not free.

**Interpretation:**

- JSPI is attractive where it prevents invasive restructuring of coarse WASM exports around async
  host imports.
- It should be applied selectively at boundaries that actually suspend; wrapping everything would
  add complexity for little gain.
- In this plan, JSPI complements Option A and the coarse mega-module design, but it does not change
  the need for ownership-safe async state in the JS host.

### 8.4 Recommendation impact from the timing data

The Chrome timings support the updated recommendations directly:

- **Keep Option A as the browser-primary path.** Its measured control overhead is already tiny.
- **Do not revive Option B as a serious browser plan.** The measurable speed delta versus A is too
  small to matter.
- **Use `Atomics.waitAsync()` in the SAB-backed Option A path.** It improves the main-thread
  completion story without changing the larger ownership work the async migration still requires.
- **Use JSPI where available on selected WASM boundaries, but only after real async host imports
  exist.** It is the cleanest way to keep coarse WASM-side logic straight-line across async host
  calls without re-architecting the whole executor around JS callbacks.
- **Promote Option E to the front of the queue.** The measured allocator costs are large enough, and
  the parametric associative-scan workspace is regular enough, that E is no longer speculative and
  does not need to wait for the async migration.
- **Treat Option D carefully.** It is viable only under transferable-at-API-edge discipline.
- **For Node, prioritize worker integration.** The browser timing model is no longer the blocker
  once `Atomics.wait` is legal; the missing piece is runtime worker plumbing.
