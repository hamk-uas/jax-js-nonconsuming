import { describe, expect, it } from "vitest";

const canRunNestedWorkerWasm =
  typeof Worker === "function" &&
  typeof SharedArrayBuffer === "function" &&
  globalThis.crossOriginIsolated === true;

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

const originalRegisterModule = OrchestratorWorker.prototype.registerModule;
OrchestratorWorker.prototype.registerModule = async function (...args) {
  self.postMessage({ type: "stage", stage: "orch-register-start" });
  const result = await originalRegisterModule.apply(this, args);
  self.postMessage({ type: "stage", stage: "orch-register-end" });
  return result;
};

const originalDispatch = OrchestratorWorker.prototype.dispatch;
OrchestratorWorker.prototype.dispatch = function (...args) {
  self.postMessage({ type: "stage", stage: "orch-dispatch-start" });
  const result = originalDispatch.apply(this, args);
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
    const x = np.array([1, 2, 3, 4]);
    self.postMessage({ type: "stage", stage: "before-jit-exec-1" });
    const y = f(x);
    self.postMessage({ type: "stage", stage: "after-jit-exec-1" });
    const jitData = Array.from(await y.data());
    self.postMessage({ type: "stage", stage: "after-jit-read-1" });
    x.dispose();
    y.dispose();

    await new Promise((resolve) => setTimeout(resolve, 50));

    const x2 = np.array([5, 6, 7, 8]);
    self.postMessage({ type: "stage", stage: "before-jit-exec-2" });
    const y2 = f(x2);
    self.postMessage({ type: "stage", stage: "after-jit-exec-2" });
    const jitData2 = Array.from(await y2.data());
    self.postMessage({ type: "stage", stage: "after-jit-read-2" });
    x2.dispose();
    y2.dispose();
    f.dispose();

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

      const stages: string[] = [];
      const result = await new Promise<{
        type?: string;
        ok: boolean;
        error?: string;
        stack?: string;
        sharedMemory?: boolean;
        orchestrator?: boolean;
        eagerData?: number[];
        jitData?: number[];
        jitData2?: number[];
      }>((resolve, reject) => {
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
        }, 10000);

        worker.addEventListener("message", (event) => {
          if (event.data?.type === "stage") {
            stages.push(String(event.data.stage));
            return;
          }
          clearTimeout(timeout);
          cleanup();
          resolve(event.data);
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
    }, 15000);
  },
);
