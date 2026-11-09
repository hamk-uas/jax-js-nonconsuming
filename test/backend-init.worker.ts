import { getWebGPUDevice, init } from "../src/backend";

self.onmessage = async (
  event: MessageEvent<{ failure: string; concurrent: boolean }>,
) => {
  const { failure, concurrent } = event.data;
  const diagnostics: string[] = [];
  console.error = (...args) => diagnostics.push(args.map(String).join(" "));
  let requestingAdapter = false;
  let overlap = false;
  let adapterRequests = 0;
  const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
  navigator.gpu.requestAdapter = async (...args) => {
    adapterRequests++;
    requestingAdapter = true;
    try {
      const adapter = await requestAdapter(...args);
      return overlap ? null : adapter;
    } finally {
      requestingAdapter = false;
    }
  };
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: class {},
  });
  Object.defineProperty(OffscreenCanvas.prototype, "getContext", {
    value() {
      overlap ||= requestingAdapter;
      if (failure === "throw") {
        throw new Error("injected WebGL context failure");
      }
      return null;
    },
  });

  try {
    const available = concurrent
      ? (await Promise.all([init("webgpu"), init("webgl"), init("webgpu")])).at(
          -1,
        )!
      : await init("webgpu", "webgl");
    self.postMessage({ available, overlap, adapterRequests, diagnostics });
    if (available.includes("webgpu")) getWebGPUDevice().destroy();
  } catch (error) {
    self.postMessage({ error: String(error) });
  }
};
