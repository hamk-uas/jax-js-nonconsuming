/** @file Polyfills for using this library. */

// Polyfill for `Symbol.dispose` and `Symbol.asyncDispose`
// https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-2.html
(Symbol as any).dispose ??= Symbol.for("Symbol.dispose");
(Symbol as any).asyncDispose ??= Symbol.for("Symbol.asyncDispose");

// Polyfill for `SuppressedError` (required by the `using` keyword).
// When a `using` block throws AND the `[Symbol.dispose]()` also throws,
// the runtime wraps both errors in a `SuppressedError`.  Safari does not
// have this yet, so we provide a minimal shim.
if (typeof globalThis.SuppressedError === "undefined") {
  (globalThis as any).SuppressedError = class SuppressedError extends Error {
    error: unknown;
    suppressed: unknown;
    constructor(error: unknown, suppressed: unknown, message?: string) {
      super(message);
      this.name = "SuppressedError";
      this.error = error;
      this.suppressed = suppressed;
    }
  };
}
