import {
  defaultDevice,
  devicePut,
  devices,
  init,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import { expect, suite, test } from "vitest";

const availableDevices = await init();

defaultDevice("cpu");

test("setup has wasm and cpu devices", () => {
  // We'll need these two devices to test behavior with arrays on different
  // devices when they interact.
  expect(devices).toContain("cpu");
  expect(devices).toContain("wasm");
});

test("binop moves to committed device", () => {
  using x = np.array([1, 2, 3], { device: "wasm" });
  using y = np.array(4);

  expect(x.device).not.toBe(y.device);
  using z = x.add(y);
  expect(z.device).toBe("wasm"); // committed
  expect(z.js()).toEqual([5, 6, 7]);
});

test("devicePut moves device", async () => {
  using ar0 = np.array([1, 2, 3]);
  expect(ar0.device).toBe("cpu");
  using ar1 = await devicePut(ar0, "wasm");
  expect(ar1.device).toBe("wasm");
  const ar2 = await devicePut(ar1, "wasm");
  // ar2 may be the same object as ar1 if already on wasm
  if (ar2 !== ar1) ar2.dispose();
  expect(ar2.device).toBe("wasm");
});

test("devicePut can be called with no device", async () => {
  using ar0 = np.array([1, 2, 3]);
  expect(ar0.device).toBe("cpu");
  const ar1 = await devicePut(ar0);
  // ar1 may be the same object as ar0 if already on default device
  if (ar1 !== ar0) ar1.dispose();
  expect(ar1.device).toBe("cpu");

  // ar should still be uncommitted, as devicePut is a no-op in this case.
  using two = np.array(2, { device: "wasm" });
  using ar2 = ar0.add(two);
  expect(ar2.device).toBe("wasm");
  expect(ar2.js()).toEqual([3, 4, 5]);

  // also works for scalars
  const { a, b } = await devicePut({ a: 3, b: true });
  using _a = a;
  using _b = b;
  expect(a.dtype).toBe(np.float32);
  expect(b.dtype).toBe(np.bool);
});

test("devicePut works with scalars", async () => {
  const x = await devicePut(5, "wasm");
  const y = await devicePut(10);
  expect(x.device).toBe("wasm");
  expect(x.dtype).toBe(np.float32);
  expect(x.weakType).toBe(true);
  expect(y.device).toBe("cpu");
  expect(y.dtype).toBe(np.float32);
  expect(y.weakType).toBe(true);

  using z = x.add(y);
  expect(z.device).toBe("wasm");
  expect(z.js()).toBe(15);
});

// R20 requires real WebGPU initialization; CPU-only CI reports these as skips.
suite.skipIf(!availableDevices.includes("webgpu"))(
  "independent backend initialization (R20)",
  () => {
    test.each([
      { failure: "unavailable", concurrent: false },
      { failure: "throw", concurrent: false },
      { failure: "throw", concurrent: true },
    ])(
      "WebGPU survives WebGL $failure (concurrent calls: $concurrent)",
      async ({ failure, concurrent }) => {
        // Blob workers lack Vitest's dynamic-import runtime; use Vite's worker pipeline.
        const worker = new Worker(
          new URL("./backend-init.worker.ts", import.meta.url),
          { type: "module" },
        );
        try {
          const result = await new Promise<{
            available?: string[];
            overlap: boolean;
            adapterRequests: number;
            diagnostics: string[];
            error?: string;
          }>((resolve, reject) => {
            worker.onmessage = (event) => resolve(event.data);
            worker.onerror = (event) => reject(new Error(event.message));
            worker.postMessage({ failure, concurrent });
          });
          expect(result.error).toBeUndefined();
          expect(result.available, JSON.stringify(result)).toContain("webgpu");
          expect(result.available).not.toContain("webgl");
          expect(result.overlap).toBe(false);
          expect(result.adapterRequests).toBe(1);
          expect(result.diagnostics).toEqual(
            failure === "throw"
              ? [
                  "Unexpected error initializing webgl backend: Error: injected WebGL context failure",
                ]
              : [],
          );
        } finally {
          worker.terminate();
        }
      },
    );
  },
);
