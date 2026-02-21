/**
 * Worker pool for parallel WASM kernel dispatch.
 *
 * Workers share a `WebAssembly.Memory` (backed by `SharedArrayBuffer`).
 * Coordination uses a shared Int32Array control buffer with per-worker state
 * managed via `Atomics.wait` (workers) and `Atomics.load` spin-wait (main thread).
 *
 * Protocol (per worker):
 *   READY (0) — worker is blocked on `Atomics.wait`, idle.
 *   WORK  (1) — main thread wrote work params; worker wakes, executes, resets to READY.
 *
 * Main thread spin-waits via `Atomics.load` (no `Atomics.wait` — blocked on browser main thread).
 */

// ---------------------------------------------------------------------------
// Control buffer layout constants
// ---------------------------------------------------------------------------

/** Worker is idle, waiting for work. */
const STATE_READY = 0;
/** Work params written, worker should process. */
const STATE_WORK = 1;

// Per-worker control layout (Int32 slots):
const CTRL_STATE = 0;
const CTRL_MODULE_ID = 1;
const CTRL_START = 2;
const CTRL_END = 3;
const CTRL_NUM_ARGS = 4;
const CTRL_ARGS = 5;

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
var instances = Object.create(null); // moduleId -> WebAssembly.Instance

self.onmessage = function (e) {
  var msg = e.data;
  if (msg.type === "init") {
    memory = msg.memory;
    workerIdx = msg.workerIdx;
  } else if (msg.type === "control") {
    control = new Int32Array(msg.buffer);
    workLoop();
  } else if (msg.type === "register") {
    try {
      var inst = new WebAssembly.Instance(msg.module, { env: { memory: memory } });
      instances[msg.moduleId] = inst;
      self.postMessage({ type: "registered", moduleId: msg.moduleId });
    } catch (err) {
      self.postMessage({ type: "error", moduleId: msg.moduleId, error: String(err) });
    }
  } else if (msg.type === "destroy") {
    self.close();
  }
};

function workLoop() {
  var STRIDE = ${WORKER_STRIDE};
  var base = workerIdx * STRIDE;

  while (true) {
    // Block until state changes from READY.
    Atomics.wait(control, base + ${CTRL_STATE}, ${STATE_READY});

    var state = Atomics.load(control, base + ${CTRL_STATE});
    if (state !== ${STATE_WORK}) continue; // spurious wakeup

    var moduleId = Atomics.load(control, base + ${CTRL_MODULE_ID});
    var start = Atomics.load(control, base + ${CTRL_START});
    var end = Atomics.load(control, base + ${CTRL_END});
    var numArgs = Atomics.load(control, base + ${CTRL_NUM_ARGS});

    var args = new Array(numArgs);
    for (var a = 0; a < numArgs; a++) {
      args[a] = Atomics.load(control, base + ${CTRL_ARGS} + a);
    }

    var instance = instances[moduleId];
    if (instance) {
      instance.exports.kernel(start, end, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], args[7], args[8], args[9],
        args[10], args[11], args[12], args[13], args[14], args[15],
        args[16], args[17], args[18], args[19]);
    }

    // Signal completion: set READY and notify main thread.
    Atomics.store(control, base + ${CTRL_STATE}, ${STATE_READY});
    Atomics.notify(control, base + ${CTRL_STATE});
  }
}
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
  #destroyed = false;

  // Module registration tracking
  #nextModuleId = 0;
  #moduleIds = new WeakMap<WebAssembly.Module, number>();
  #registeredOnWorkers = new Set<number>();

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

    // Create workers from inlined Blob URL
    const code = buildWorkerCode();
    const blob = new Blob([code], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);

    this.#workers = [];
    for (let i = 0; i < n; i++) {
      const w = new Worker(url);
      // Init: send shared memory
      w.postMessage({ type: "init", memory, workerIdx: i });
      // Send control buffer so worker can start its work loop
      w.postMessage({ type: "control", buffer: this.#controlBuf });
      this.#workers.push(w);
    }
    URL.revokeObjectURL(url);
  }

  /**
   * Register a WASM module on all workers (async — workers instantiate
   * the module with the shared memory). Returns a module ID.
   *
   * Must be called (and awaited) before `dispatchSync` can use the module.
   * Repeated calls with the same module return the cached ID instantly.
   */
  async registerModule(module: WebAssembly.Module): Promise<number> {
    const existing = this.#moduleIds.get(module);
    if (existing !== undefined) return existing;

    const id = this.#nextModuleId++;
    this.#moduleIds.set(module, id);

    // Send module to all workers and wait for acknowledgement
    const promises = this.#workers.map(
      (w) =>
        new Promise<void>((resolve, reject) => {
          const handler = (e: MessageEvent) => {
            if (e.data.moduleId !== id) return;
            w.removeEventListener("message", handler);
            if (e.data.type === "registered") resolve();
            else
              reject(new Error(e.data.error ?? "worker registration failed"));
          };
          w.addEventListener("message", handler);
          w.postMessage({ type: "register", moduleId: id, module });
        }),
    );
    await Promise.all(promises);
    this.#registeredOnWorkers.add(id);
    return id;
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
   */
  dispatchSync(
    moduleId: number,
    mainInstance: WebAssembly.Instance,
    totalSize: number,
    args: number[],
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

    // Write work params for each worker
    for (let i = 0; i < n; i++) {
      const start = (i + 1) * chunkSize;
      const end = Math.min(start + chunkSize, totalSize);
      if (start >= end) continue;

      const base = i * WORKER_STRIDE;
      Atomics.store(this.#control, base + CTRL_MODULE_ID, moduleId);
      Atomics.store(this.#control, base + CTRL_START, start);
      Atomics.store(this.#control, base + CTRL_END, end);
      Atomics.store(this.#control, base + CTRL_NUM_ARGS, args.length);
      for (let a = 0; a < args.length; a++) {
        Atomics.store(this.#control, base + CTRL_ARGS + a, args[a]);
      }
      // Signal work (must be last — triggers Atomics.wait return in worker)
      Atomics.store(this.#control, base + CTRL_STATE, STATE_WORK);
      Atomics.notify(this.#control, base + CTRL_STATE);
    }

    // Main thread processes chunk [0, chunkSize)
    const mainEnd = Math.min(chunkSize, totalSize);
    if (mainEnd > 0) {
      (mainInstance.exports.kernel as (...a: number[]) => void)(
        0,
        mainEnd,
        ...args,
      );
    }

    // Spin-wait for all active workers to finish
    for (let i = 0; i < n; i++) {
      const start = (i + 1) * chunkSize;
      if (start >= totalSize) continue;
      const base = i * WORKER_STRIDE;
      // Spin: safe on all threads (does not block event loop, just busy-polls)
      while (Atomics.load(this.#control, base + CTRL_STATE) !== STATE_READY) {
        // busy wait
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
