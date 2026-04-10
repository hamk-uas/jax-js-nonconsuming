import { registerAsyncModule } from "./registration";

/**
 * Shared module registration state for WASM worker owners (OrchestratorWorker,
 * WasmWorkerPool). Encapsulates ID allocation, dedup, and in-flight tracking.
 */
export class ModuleRegistry {
  #label: string;
  #nextId = 0;
  #ids = new WeakMap<WebAssembly.Module, number>();
  #registered = new Set<number>();
  #registering = new Map<number, Promise<void>>();

  constructor(label: string) {
    this.#label = label;
  }

  isReady(module: WebAssembly.Module): boolean {
    const id = this.#ids.get(module);
    return id !== undefined && this.#registered.has(id);
  }

  hasId(id: number): boolean {
    return this.#registered.has(id);
  }

  getId(module: WebAssembly.Module): number | undefined {
    return this.#ids.get(module);
  }

  /**
   * Synchronous fast path — throws if the module has not been registered yet.
   */
  getIdSync(module: WebAssembly.Module): number {
    const existing = this.#ids.get(module);
    if (existing !== undefined && this.#registered.has(existing)) {
      return existing;
    }
    throw new Error(
      `${this.#label}: module not pre-registered; call registerModule() before synchronous dispatch`,
    );
  }

  async register(
    module: WebAssembly.Module,
    startRegistration: (id: number) => Promise<void>,
    onFailure: (id: number) => void | Promise<void>,
  ): Promise<{ id: number; created: boolean }> {
    return registerAsyncModule(
      {
        ownerLabel: this.#label,
        moduleIds: this.#ids,
        registeringModules: this.#registering,
        allocateModuleId: () => this.#nextId++,
        isRegistered: (id) => this.#registered.has(id),
        markRegistered: (id) => {
          this.#registered.add(id);
        },
        clearRegistered: (id) => {
          this.#registered.delete(id);
        },
      },
      module,
      startRegistration,
      onFailure,
    );
  }
}
