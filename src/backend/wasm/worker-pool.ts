/**
 * Worker pool for parallel WASM kernel dispatch.
 *
 * Workers share a `WebAssembly.Memory` (backed by `SharedArrayBuffer`).
 * Coordination uses a shared Int32Array control buffer for work params
 * and completion signaling, combined with `postMessage` for wake-up.
 *
 * Protocol (per worker):
 *   1. Main thread writes work params to control buffer via Atomics.store
 *   2. Main thread sends postMessage({ type: "wake" }) to worker
 *   3. Worker's onmessage fires, reads params, executes kernel
 *   4. Worker writes STATE_READY to control buffer, calls Atomics.notify
 *   5. Main thread waits for STATE_READY (Atomics.wait when available,
 *      otherwise a busy-poll fallback)
 *
 * Workers still sit in the natural event loop so they can receive module
 * registration messages; the main side uses `Atomics.wait` when available to
 * avoid starving nested-worker execution.
 * Browser main threads never construct this path because WasmBackend gates it
 * on Atomics.wait support.
 */

import { registerAsyncModule } from "./registration";

function canWaitOnThisThread(): boolean {
  try {
    const probe = new Int32Array(new SharedArrayBuffer(4));
    Atomics.wait(probe, 0, 1, 0);
    return true;
  } catch {
    return false;
  }
}

function waitWhileState(
  control: Int32Array,
  index: number,
  expected: number,
  canWait: boolean,
): void {
  if (canWait) {
    Atomics.wait(control, index, expected);
    return;
  }
  while (Atomics.load(control, index) === expected) {
    // busy wait fallback
  }
}

// ---------------------------------------------------------------------------
// Control buffer layout constants
// ---------------------------------------------------------------------------

/** Worker is idle, waiting for postMessage wake-up. */
const STATE_READY = 0;
/** Work params written, worker should process. */
const STATE_WORK = 1;

// Per-worker control layout (Int32 slots):
const CTRL_STATE = 0;
const CTRL_MODULE_ID = 1;
const CTRL_START = 2;
const CTRL_END = 3;
/** Kernel index: -1 = default "kernel" export, 0+ = "kernel_N" export. */
const CTRL_KERNEL_IDX = 4;
const CTRL_NUM_ARGS = 5;
const CTRL_ARGS = 6;

/** Maximum number of buffer-pointer arguments per kernel dispatch. */
const MAX_ARGS = 20;
/** Total Int32 slots per worker in the control buffer. */
const WORKER_STRIDE = CTRL_ARGS + MAX_ARGS;

// ---------------------------------------------------------------------------
// Worker entry code (inlined as Blob URL)
// ---------------------------------------------------------------------------

/**
 * Build the worker script source. Constants are baked in so the worker
 * code is fully self-contained (no imports needed).
 */
function buildWorkerCode(): string {
  return `
"use strict";
var memory = null;
var control = null;
var workerIdx = -1;
var workBase = -1;
var instances = Object.create(null); // moduleId -> WebAssembly.Instance

self.onmessage = function (e) {
  var msg = e.data;
  if (msg.type === "init") {
    memory = msg.memory;
    workerIdx = msg.workerIdx;
  } else if (msg.type === "control") {
    control = new Int32Array(msg.buffer);
    workBase = workerIdx * ${WORKER_STRIDE};
  } else if (msg.type === "wake") {
    // Main thread wrote work params — execute kernel
    if (workBase < 0) return;
    var state = Atomics.load(control, workBase + ${CTRL_STATE});
    if (state !== ${STATE_WORK}) return; // spurious or already processed

    var moduleId = Atomics.load(control, workBase + ${CTRL_MODULE_ID});
    var start = Atomics.load(control, workBase + ${CTRL_START});
    var end = Atomics.load(control, workBase + ${CTRL_END});
    var kernelIdx = Atomics.load(control, workBase + ${CTRL_KERNEL_IDX});
    var numArgs = Atomics.load(control, workBase + ${CTRL_NUM_ARGS});

    var args = new Array(numArgs);
    for (var a = 0; a < numArgs; a++) {
      args[a] = Atomics.load(control, workBase + ${CTRL_ARGS} + a);
    }

    var instance = instances[moduleId];
    if (instance) {
      // M6.2c: select kernel by index (-1 = default "kernel", 0+ = "kernel_N")
      var fn = kernelIdx < 0
        ? instance.exports.kernel
        : instance.exports["kernel_" + kernelIdx];
      fn(start, end, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], args[7], args[8], args[9],
        args[10], args[11], args[12], args[13], args[14], args[15],
        args[16], args[17], args[18], args[19]);
    }

    // Signal completion: set READY and notify main thread.
    Atomics.store(control, workBase + ${CTRL_STATE}, ${STATE_READY});
    Atomics.notify(control, workBase + ${CTRL_STATE});
  } else if (msg.type === "register") {
    try {
      var inst = new WebAssembly.Instance(msg.module, { env: { memory: memory } });
      instances[msg.moduleId] = inst;
      self.postMessage({ type: "registered", moduleId: msg.moduleId });
    } catch (err) {
      self.postMessage({ type: "error", moduleId: msg.moduleId, error: String(err) });
    }
  } else if (msg.type === "register-mega") {
    // Register a mega-module with stub alloc/free imports.
    // Workers only call extracted kernel_N functions, never mega_execute.
    try {
      var inst = new WebAssembly.Instance(msg.module, {
        env: {
          memory: memory,
          alloc: function() { throw new Error("worker: unexpected alloc"); },
          free: function() {},
        },
      });
      instances[msg.moduleId] = inst;
      self.postMessage({ type: "registered", moduleId: msg.moduleId });
    } catch (err) {
      self.postMessage({ type: "error", moduleId: msg.moduleId, error: String(err) });
    }
  } else if (msg.type === "unregister") {
    delete instances[msg.moduleId];
  } else if (msg.type === "destroy") {
    self.close();
  }
};
`;
}

// ---------------------------------------------------------------------------
// WasmWorkerPool
// ---------------------------------------------------------------------------

export class WasmWorkerPool {
  readonly numWorkers: number;

  #workers: Worker[];
  #controlBuf: SharedArrayBuffer;
  #control: Int32Array;
  #canWait: boolean;
  #destroyed = false;

  // Module registration tracking
  #nextModuleId = 0;
  #moduleIds = new WeakMap<WebAssembly.Module, number>();
  #registeredOnWorkers = new Set<number>();
  #registeringModules = new Map<number, Promise<void>>();

  constructor(memory: WebAssembly.Memory, numWorkers?: number) {
    const n =
      numWorkers ??
      Math.max(
        1,
        ((globalThis as any).navigator?.hardwareConcurrency ?? 4) - 1,
      );
    this.numWorkers = n;

    // Shared control buffer
    this.#controlBuf = new SharedArrayBuffer(n * WORKER_STRIDE * 4);
    this.#control = new Int32Array(this.#controlBuf);
    this.#canWait = canWaitOnThisThread();

    // Create workers from inlined Blob URL
    const code = buildWorkerCode();
    const blob = new Blob([code], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);

    this.#workers = [];
    for (let i = 0; i < n; i++) {
      const w = new Worker(url, { type: "module" });
      // Init: send shared memory
      w.postMessage({ type: "init", memory, workerIdx: i });
      // Send control buffer so worker can start its work loop
      w.postMessage({ type: "control", buffer: this.#controlBuf });
      this.#workers.push(w);
    }
    // Defer revocation — Deno module workers may not have loaded the blob
    // URL synchronously by the time revokeObjectURL is called.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  /**
   * Register a WASM module on all workers (async — workers instantiate
   * the module with the shared memory). Returns a module ID.
   *
   * Must be called (and awaited) before `dispatchSync` can use the module.
   * Repeated calls with the same module return the cached ID instantly.
   */
  async registerModule(module: WebAssembly.Module): Promise<number> {
    return this.#registerOnAllWorkers(module, "register");
  }

  async #registerOnAllWorkers(
    module: WebAssembly.Module,
    messageType: "register" | "register-mega",
  ): Promise<number> {
    const { id } = await registerAsyncModule(
      {
        ownerLabel: "Worker pool",
        moduleIds: this.#moduleIds,
        registeringModules: this.#registeringModules,
        allocateModuleId: () => this.#nextModuleId++,
        isRegistered: (existingId) => this.#registeredOnWorkers.has(existingId),
        markRegistered: (registeredId) => {
          this.#registeredOnWorkers.add(registeredId);
        },
        clearRegistered: (registeredId) => {
          this.#registeredOnWorkers.delete(registeredId);
        },
      },
      module,
      (moduleId) =>
        Promise.all(
          this.#workers.map(
            (worker) =>
              new Promise<void>((resolve, reject) => {
                const handler = (event: MessageEvent) => {
                  if (event.data.moduleId !== moduleId) return;
                  worker.removeEventListener("message", handler);
                  if (event.data.type === "registered") {
                    resolve();
                  } else {
                    reject(
                      new Error(
                        event.data.error ??
                          `worker ${messageType === "register" ? "registration" : "mega registration"} failed`,
                      ),
                    );
                  }
                };
                worker.addEventListener("message", handler);
                worker.postMessage({
                  type: messageType,
                  moduleId,
                  module,
                });
              }),
          ),
        ).then(() => undefined),
      (moduleId) => {
        for (const worker of this.#workers) {
          worker.postMessage({ type: "unregister", moduleId });
        }
      },
    );

    return id;
  }

  /**
   * Register a mega-module on all workers (M6.2c). Mega-modules import
   * `env.alloc` and `env.free` — workers get stub implementations since
   * they only call extracted `kernel_N` functions, never `mega_execute`.
   *
   * Must be called (and awaited) before `dispatchSync` can use the module.
   * Repeated calls with the same module return the cached ID instantly.
   */
  async registerMegaModule(module: WebAssembly.Module): Promise<number> {
    return this.#registerOnAllWorkers(module, "register-mega");
  }

  /**
   * Check if a module has been registered on all workers (i.e.,
   * `registerModule` has been awaited for this module).
   */
  isModuleReady(module: WebAssembly.Module): boolean {
    const id = this.#moduleIds.get(module);
    return id !== undefined && this.#registeredOnWorkers.has(id);
  }

  /**
   * Get the module ID for a previously registered module.
   * Returns undefined if not registered.
   */
  getModuleId(module: WebAssembly.Module): number | undefined {
    return this.#moduleIds.get(module);
  }

  /**
   * Synchronous parallel dispatch.
   *
   * Splits `[0, totalSize)` into `numWorkers + 1` chunks. Workers process
   * their chunks in parallel while the main thread processes chunk 0.
   * Main thread spin-waits (via `Atomics.load`) for all workers to finish.
   *
   * @param moduleId     Module ID from `registerModule()`
   * @param mainInstance Instance on the main thread (same module, shared memory)
   * @param totalSize    Total number of elements to process
   * @param args         Buffer pointers (shared memory addresses)
   * @param kernelIdx    Kernel index: -1 = default "kernel" export, 0+ = "kernel_N" (M6.2c)
   */
  dispatchSync(
    moduleId: number,
    mainInstance: WebAssembly.Instance,
    totalSize: number,
    args: number[],
    kernelIdx: number = -1,
  ): void {
    if (this.#destroyed) throw new Error("WasmWorkerPool destroyed");
    if (!this.#registeredOnWorkers.has(moduleId)) {
      throw new Error(`Module ${moduleId} not registered on workers`);
    }
    if (args.length > MAX_ARGS) {
      throw new Error(`Too many args (${args.length} > ${MAX_ARGS})`);
    }

    const n = this.numWorkers;
    const nChunks = n + 1; // workers + main thread
    const chunkSize = Math.ceil(totalSize / nChunks);

    // Write work params for each worker, then wake via postMessage
    for (let i = 0; i < n; i++) {
      const start = (i + 1) * chunkSize;
      const end = Math.min(start + chunkSize, totalSize);
      if (start >= end) continue;

      const base = i * WORKER_STRIDE;
      Atomics.store(this.#control, base + CTRL_MODULE_ID, moduleId);
      Atomics.store(this.#control, base + CTRL_START, start);
      Atomics.store(this.#control, base + CTRL_END, end);
      Atomics.store(this.#control, base + CTRL_KERNEL_IDX, kernelIdx);
      Atomics.store(this.#control, base + CTRL_NUM_ARGS, args.length);
      for (let a = 0; a < args.length; a++) {
        Atomics.store(this.#control, base + CTRL_ARGS + a, args[a]);
      }
      // Signal work (must be last — worker reads STATE_WORK on wake)
      Atomics.store(this.#control, base + CTRL_STATE, STATE_WORK);
      // Wake worker via postMessage (no Atomics.waitAsync/notify needed)
      this.#workers[i].postMessage({ type: "wake" });
    }

    // Main thread processes chunk [0, chunkSize)
    const mainEnd = Math.min(chunkSize, totalSize);
    if (mainEnd > 0) {
      // M6.2c: select kernel by index
      const fn =
        kernelIdx < 0
          ? (mainInstance.exports.kernel as (...a: number[]) => void)
          : (mainInstance.exports[`kernel_${kernelIdx}`] as (
              ...a: number[]
            ) => void);
      fn(0, mainEnd, ...args);
    }

    // Wait for all active workers to finish.
    for (let i = 0; i < n; i++) {
      const start = (i + 1) * chunkSize;
      if (start >= totalSize) continue;
      const base = i * WORKER_STRIDE;
      while (true) {
        const state = Atomics.load(this.#control, base + CTRL_STATE);
        if (state === STATE_READY) break;
        waitWhileState(this.#control, base + CTRL_STATE, state, this.#canWait);
      }
    }
  }

  /** Terminate all workers and release resources. */
  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    for (const w of this.#workers) {
      w.postMessage({ type: "destroy" });
      w.terminate();
    }
    this.#workers.length = 0;
  }
}
