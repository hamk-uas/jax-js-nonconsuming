import { describe, expect, it } from "vitest";

const canRunNestedWorkerWasm =
  typeof Worker === "function" &&
  typeof SharedArrayBuffer === "function" &&
  globalThis.crossOriginIsolated === true;

async function runNestedWorkerScenario<TResult>(
  workerCode: string,
  timeoutMs: number = 10000,
): Promise<{ result: TResult; stages: string[] }> {
  const stages: string[] = [];
  const result = await new Promise<TResult>((resolve, reject) => {
    const blob = new Blob([workerCode], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url, { type: "module" });
    const cleanup = () => {
      worker.terminate();
      URL.revokeObjectURL(url);
    };
    const timeout = setTimeout(() => {
      cleanup();
      reject(
        new Error(
          `Timed out waiting for nested worker WASM result. Stages: ${stages.join(", ") || "none"}`,
        ),
      );
    }, timeoutMs);

    worker.addEventListener("message", (event) => {
      if (event.data?.type === "stage") {
        stages.push(String(event.data.stage));
        return;
      }
      clearTimeout(timeout);
      cleanup();
      resolve(event.data as TResult);
    });
    worker.addEventListener("error", (event) => {
      clearTimeout(timeout);
      cleanup();
      reject(
        event.error instanceof Error
          ? event.error
          : new Error(event.message || String(event.error)),
      );
    });
    worker.postMessage({ type: "run" });
  });

  return { result, stages };
}

describe.skipIf(!canRunNestedWorkerWasm)(
  "WASM nested worker shared-memory path",
  () => {
    it("runs eager and jit WASM paths inside a browser worker", async () => {
      const jaxUrl = new URL("../src/index.ts", import.meta.url).href;
      const backendUrl = new URL("../src/backend.ts", import.meta.url).href;
      const orchestratorUrl = new URL(
        "../src/backend/wasm/orchestrator.ts",
        import.meta.url,
      ).href;
      const workerCode = `
import { init, jit, numpy as np } from ${JSON.stringify(jaxUrl)};
import { getBackend } from ${JSON.stringify(backendUrl)};
import { OrchestratorWorker } from ${JSON.stringify(orchestratorUrl)};

let orchestratorReady = false;
let orchestratorDispatched = false;

const originalRegisterModule = OrchestratorWorker.prototype.registerModule;
OrchestratorWorker.prototype.registerModule = async function (...args) {
  self.postMessage({ type: "stage", stage: "orch-register-start" });
  try {
    const result = await originalRegisterModule.apply(this, args);
    orchestratorReady = true;
    self.postMessage({ type: "stage", stage: "orch-register-end" });
    return result;
  } catch (error) {
    self.postMessage({
      type: "stage",
      stage: "orch-register-error",
      error: String(error),
    });
    throw error;
  }
};

const originalDispatch = OrchestratorWorker.prototype.dispatch;
OrchestratorWorker.prototype.dispatch = function (...args) {
  self.postMessage({ type: "stage", stage: "orch-dispatch-start" });
  const result = originalDispatch.apply(this, args);
  orchestratorDispatched = true;
  self.postMessage({ type: "stage", stage: "orch-dispatch-end" });
  return result;
};

self.onmessage = async (event) => {
  if (!event.data || event.data.type !== "run") return;
  try {
    self.postMessage({ type: "stage", stage: "starting" });
    await init("wasm");
    const backend = getBackend("wasm");
    self.postMessage({
      type: "stage",
      stage: "after-init",
      sharedMemory: backend.capabilities.sharedMemory,
    });

    const a = np.ones([8192]);
    const eager = a.add(a);
    const eagerData = Array.from(await eager.data()).slice(0, 4);
    a.dispose();
    eager.dispose();
    self.postMessage({ type: "stage", stage: "after-eager" });

    const f = jit((x) => x.add(1).mul(2));
  const n = 1024;
  const x = np.array(Array.from({ length: n }, (_, i) => i + 1));
    self.postMessage({ type: "stage", stage: "before-jit-exec-1" });
    const y = f(x);
    self.postMessage({ type: "stage", stage: "after-jit-exec-1" });
  const jitData = Array.from(await y.data()).slice(0, 4);
    self.postMessage({ type: "stage", stage: "after-jit-read-1" });
    x.dispose();
    y.dispose();

    let jitData2 = null;
    for (let attempt = 0; attempt < 20; attempt++) {
      const x2 = np.array(Array.from({ length: n }, (_, i) => i + 5));
      self.postMessage({
        type: "stage",
        stage:
          attempt === 0
            ? "before-jit-exec-2"
            : "before-jit-exec-2-" + String(attempt + 1),
      });
      const y2 = f(x2);
      self.postMessage({
        type: "stage",
        stage:
          attempt === 0
            ? "after-jit-exec-2"
            : "after-jit-exec-2-" + String(attempt + 1),
      });
      jitData2 = Array.from(await y2.data()).slice(0, 4);
      self.postMessage({
        type: "stage",
        stage:
          attempt === 0
            ? "after-jit-read-2"
            : "after-jit-read-2-" + String(attempt + 1),
      });
      x2.dispose();
      y2.dispose();
      if (orchestratorReady) {
        self.postMessage({ type: "stage", stage: "after-orch-ready" });
      }
      if (orchestratorDispatched) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    f.dispose();

    if (!orchestratorDispatched) {
      throw new Error("orchestrator dispatch did not occur");
    }

    self.postMessage({
      type: "result",
      ok: true,
      sharedMemory: backend.capabilities.sharedMemory,
      orchestrator: backend.orchestrator !== null,
      eagerData,
      jitData,
      jitData2,
    });
  } catch (error) {
    self.postMessage({
      type: "result",
      ok: false,
      error: String(error),
      stack: error && typeof error === "object" && "stack" in error ? String(error.stack) : undefined,
    });
  }
};
`;

      const { result, stages } = await runNestedWorkerScenario<{
        type?: string;
        ok: boolean;
        error?: string;
        stack?: string;
        sharedMemory?: boolean;
        orchestrator?: boolean;
        eagerData?: number[];
        jitData?: number[];
        jitData2?: number[];
      }>(workerCode);

      if (!result.ok) {
        throw new Error(
          [result.error ?? "nested worker failed", result.stack]
            .filter(Boolean)
            .join("\n"),
        );
      }

      expect(result.sharedMemory).toBe(true);
      expect(result.orchestrator).toBe(true);
      expect(result.eagerData).toEqual([2, 2, 2, 2]);
      expect(result.jitData).toEqual([4, 6, 8, 10]);
      expect(result.jitData2).toEqual([12, 14, 16, 18]);
      expect(stages).toContain("orch-register-start");
      expect(stages).toContain("orch-register-end");
      expect(stages).toContain("orch-dispatch-start");
      expect(stages).toContain("orch-dispatch-end");
      expect(stages).not.toContain("orch-register-error");
    }, 15000);

    it("runs the M6.2c parallel worker-pool path inside a browser worker", async () => {
      const jaxUrl = new URL("../src/index.ts", import.meta.url).href;
      const backendUrl = new URL("../src/backend.ts", import.meta.url).href;
      const orchestratorUrl = new URL(
        "../src/backend/wasm/orchestrator.ts",
        import.meta.url,
      ).href;
      const workerPoolUrl = new URL(
        "../src/backend/wasm/worker-pool.ts",
        import.meta.url,
      ).href;
      const workerCode = `
import { init, jit, numpy as np } from ${JSON.stringify(jaxUrl)};
import { getBackend } from ${JSON.stringify(backendUrl)};
import { OrchestratorWorker } from ${JSON.stringify(orchestratorUrl)};
import { WasmWorkerPool } from ${JSON.stringify(workerPoolUrl)};

let poolReady = false;

const originalOrchestratorRegister = OrchestratorWorker.prototype.registerModule;
OrchestratorWorker.prototype.registerModule = async function (...args) {
  self.postMessage({ type: "stage", stage: "orch-register-start" });
  const result = await originalOrchestratorRegister.apply(this, args);
  self.postMessage({ type: "stage", stage: "orch-register-end" });
  return result;
};

const originalRegisterMegaModule = WasmWorkerPool.prototype.registerMegaModule;
WasmWorkerPool.prototype.registerMegaModule = async function (...args) {
  self.postMessage({ type: "stage", stage: "pool-register-start" });
  const result = await originalRegisterMegaModule.apply(this, args);
  poolReady = true;
  self.postMessage({ type: "stage", stage: "pool-register-end" });
  return result;
};

const originalDispatchSync = WasmWorkerPool.prototype.dispatchSync;
WasmWorkerPool.prototype.dispatchSync = function (...args) {
  self.postMessage({ type: "stage", stage: "pool-dispatch-start" });
  const result = originalDispatchSync.apply(this, args);
  self.postMessage({ type: "stage", stage: "pool-dispatch-end" });
  return result;
};

async function waitForPoolReady(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (!poolReady && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  if (!poolReady) {
    throw new Error("worker pool registration did not complete");
  }
}

self.onmessage = async (event) => {
  if (!event.data || event.data.type !== "run") return;
  try {
    self.postMessage({ type: "stage", stage: "starting" });
    await init("wasm");
    const backend = getBackend("wasm");
    self.postMessage({
      type: "stage",
      stage: "after-init",
      sharedMemory: backend.capabilities.sharedMemory,
    });

    const n = 8192;
    const x = np.array(Array.from({ length: n }, (_, i) => i));
    const f = jit((input) => input.add(1).mul(2));

    self.postMessage({ type: "stage", stage: "before-jit-exec-1" });
    const y = f(x);
    self.postMessage({ type: "stage", stage: "after-jit-exec-1" });
    const jitData = Array.from(await y.data()).slice(0, 4);
    self.postMessage({ type: "stage", stage: "after-jit-read-1" });
    x.dispose();
    y.dispose();

    await waitForPoolReady(2000);
    self.postMessage({ type: "stage", stage: "after-pool-ready" });

    const x2 = np.array(Array.from({ length: n }, (_, i) => i + 10));
    self.postMessage({ type: "stage", stage: "before-jit-exec-2" });
    const y2 = f(x2);
    self.postMessage({ type: "stage", stage: "after-jit-exec-2" });
    const jitData2 = Array.from(await y2.data()).slice(0, 4);
    self.postMessage({ type: "stage", stage: "after-jit-read-2" });
    x2.dispose();
    y2.dispose();
    f.dispose();

    self.postMessage({
      type: "result",
      ok: true,
      sharedMemory: backend.capabilities.sharedMemory,
      workerPool: backend.workerPool !== null,
      jitData,
      jitData2,
    });
  } catch (error) {
    self.postMessage({
      type: "result",
      ok: false,
      error: String(error),
      stack: error && typeof error === "object" && "stack" in error ? String(error.stack) : undefined,
    });
  }
};
`;

      const { result, stages } = await runNestedWorkerScenario<{
        type?: string;
        ok: boolean;
        error?: string;
        stack?: string;
        sharedMemory?: boolean;
        workerPool?: boolean;
        jitData?: number[];
        jitData2?: number[];
      }>(workerCode, 15000);

      if (!result.ok) {
        throw new Error(
          [result.error ?? "nested worker failed", result.stack]
            .filter(Boolean)
            .join("\n"),
        );
      }

      expect(result.sharedMemory).toBe(true);
      expect(result.workerPool).toBe(true);
      expect(result.jitData).toEqual([2, 4, 6, 8]);
      expect(result.jitData2).toEqual([22, 24, 26, 28]);
      expect(stages).toContain("pool-register-start");
      expect(stages).toContain("pool-register-end");
      expect(stages).toContain("after-pool-ready");
      expect(stages).toContain("pool-dispatch-start");
      expect(stages).toContain("pool-dispatch-end");
      expect(stages).not.toContain("orch-register-start");
      expect(stages).not.toContain("orch-register-end");
    }, 20000);
  },
);
