import { describe, expect, it } from "vitest";

import { OrchestratorWorker } from "../src/backend/wasm/orchestrator";
import { WasmWorkerPool } from "../src/backend/wasm/worker-pool";

const canRunWorkerRegistrationUnitTests =
  typeof Worker === "function" &&
  typeof SharedArrayBuffer === "function" &&
  globalThis.crossOriginIsolated === true;

const EMPTY_MODULE_BYTES = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
]);

type WorkerMessageHandler = (event: MessageEvent) => void;
type RegistrationOutcome = "error" | "registered";

class FakeWorker {
  static instances: FakeWorker[] = [];
  static registrationOutcomes: RegistrationOutcome[] = [];

  readonly unregisterIds: number[] = [];

  #messageHandlers = new Set<WorkerMessageHandler>();

  constructor(_url: string | URL, _options?: WorkerOptions) {
    FakeWorker.instances.push(this);
  }

  addEventListener(type: string, handler: EventListenerOrEventListenerObject) {
    if (type !== "message") return;
    if (typeof handler === "function") {
      this.#messageHandlers.add(handler as WorkerMessageHandler);
      return;
    }
    this.#messageHandlers.add(((event: MessageEvent) =>
      handler.handleEvent(event)) as WorkerMessageHandler);
  }

  removeEventListener(
    type: string,
    handler: EventListenerOrEventListenerObject,
  ) {
    if (type !== "message") return;
    if (typeof handler === "function") {
      this.#messageHandlers.delete(handler as WorkerMessageHandler);
    }
  }

  postMessage(message: { type: string; moduleId?: number }) {
    if (message.type === "register" || message.type === "register-mega") {
      const outcome = FakeWorker.registrationOutcomes.shift() ?? "registered";
      queueMicrotask(() => {
        this.#emitMessage({
          type: outcome,
          moduleId: message.moduleId,
          error:
            outcome === "error" ? "synthetic registration failure" : undefined,
        });
      });
      return;
    }

    if (message.type === "unregister" && message.moduleId !== undefined) {
      this.unregisterIds.push(message.moduleId);
    }
  }

  terminate() {}

  #emitMessage(data: unknown) {
    for (const handler of this.#messageHandlers) {
      handler({ data } as MessageEvent);
    }
  }
}

function makeEmptyModule(): WebAssembly.Module {
  return new WebAssembly.Module(EMPTY_MODULE_BYTES);
}

function withFakeWorker<T>(
  registrationOutcomes: RegistrationOutcome[],
  run: () => Promise<T>,
): Promise<T> {
  const originalWorker = globalThis.Worker;
  FakeWorker.instances = [];
  FakeWorker.registrationOutcomes = [...registrationOutcomes];
  (globalThis as typeof globalThis & { Worker: typeof Worker }).Worker =
    FakeWorker as unknown as typeof Worker;

  return run().finally(() => {
    (globalThis as typeof globalThis & { Worker: typeof Worker }).Worker =
      originalWorker;
    FakeWorker.instances = [];
    FakeWorker.registrationOutcomes = [];
  });
}

describe.skipIf(!canRunWorkerRegistrationUnitTests)(
  "WASM worker registration cleanup",
  () => {
    it("orchestrator unregisters failed registrations before retrying", async () => {
      await withFakeWorker(["error", "registered"], async () => {
        const orchestrator = new OrchestratorWorker(
          new WebAssembly.Memory({ initial: 1 }),
        );
        const module = makeEmptyModule();

        await expect(orchestrator.registerModule(module)).rejects.toThrow(
          "synthetic registration failure",
        );

        expect(FakeWorker.instances).toHaveLength(1);
        expect(FakeWorker.instances[0].unregisterIds).toEqual([0]);

        await expect(orchestrator.registerModule(module)).resolves.toBe(1);
        expect(orchestrator.isModuleReady(module)).toBe(true);
        expect(orchestrator.getModuleId(module)).toBe(1);
        expect(orchestrator.registerModuleSync(module)).toBe(1);

        orchestrator.destroy();
      });
    });

    it("worker pool unregisters failed registrations on every worker before retrying", async () => {
      await withFakeWorker(
        ["registered", "error", "registered", "registered"],
        async () => {
          const pool = new WasmWorkerPool(
            new WebAssembly.Memory({ initial: 1 }),
            2,
          );
          const module = makeEmptyModule();

          await expect(pool.registerMegaModule(module)).rejects.toThrow(
            "synthetic registration failure",
          );

          expect(FakeWorker.instances).toHaveLength(2);
          for (const worker of FakeWorker.instances) {
            expect(worker.unregisterIds).toEqual([0]);
          }

          await expect(pool.registerMegaModule(module)).resolves.toBe(1);
          expect(pool.isModuleReady(module)).toBe(true);
          expect(pool.getModuleId(module)).toBe(1);

          pool.destroy();
        },
      );
    });
  },
);
