import { describe, expect, it } from "vitest";

import { installExplicitResourceManagementPolyfills } from "../src/polyfills";

function createPolyfilledGlobal() {
  const fakeGlobal = {} as typeof globalThis;
  installExplicitResourceManagementPolyfills(fakeGlobal);
  return fakeGlobal;
}

describe("DisposableStack polyfill", () => {
  it("rejects non-disposable values passed to use", () => {
    const DisposableStackPolyfill = createPolyfilledGlobal().DisposableStack;
    const stack = new DisposableStackPolyfill();

    expect(() => stack.use({} as Disposable)).toThrowError(TypeError);
  });

  it("rejects non-callable adopt and defer callbacks", () => {
    const DisposableStackPolyfill = createPolyfilledGlobal().DisposableStack;
    const stack = new DisposableStackPolyfill();

    expect(() =>
      stack.adopt("x", null as unknown as (value: string) => void),
    ).toThrowError(TypeError);
    expect(() => stack.defer(null as unknown as () => void)).toThrowError(
      TypeError,
    );
  });

  it("exposes the expected toStringTag and move semantics", () => {
    const DisposableStackPolyfill = createPolyfilledGlobal().DisposableStack;
    const calls: string[] = [];
    const stack = new DisposableStackPolyfill();

    stack.defer(() => calls.push("disposed"));

    expect(Object.prototype.toString.call(stack)).toBe(
      "[object DisposableStack]",
    );

    const moved = stack.move();
    expect(stack.disposed).toBe(true);

    moved.dispose();
    expect(calls).toEqual(["disposed"]);
  });
});
