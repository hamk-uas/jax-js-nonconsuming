import { describe, expect, it } from "vitest";

import { _canUseWasmSharedMemoryRuntime } from "../src/backend/wasm";

class FakeSharedArrayBuffer {
  byteLength: number;

  constructor(byteLength: number) {
    this.byteLength = byteLength;
  }
}

describe("WASM shared-memory runtime gating", () => {
  it("allows shared memory when SharedArrayBuffer is constructible", () => {
    expect(
      _canUseWasmSharedMemoryRuntime({
        sharedArrayBufferCtor: FakeSharedArrayBuffer,
      }),
    ).toBe(true);
  });

  it("disables shared memory when SharedArrayBuffer is unavailable", () => {
    expect(
      _canUseWasmSharedMemoryRuntime({
        sharedArrayBufferCtor: null,
      }),
    ).toBe(false);
  });
});