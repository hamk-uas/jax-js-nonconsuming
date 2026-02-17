/**
 * @file Browser polyfills for the `using` keyword (TC39 Explicit Resource Management).
 *
 * This runs before any @hamk-uas/jax-js-nonconsuming import so that `using` declarations
 * work in Safari and other browsers that lack native support.
 *
 * The core library (`@hamk-uas/jax-js-nonconsuming`) also ships these polyfills, but the
 * website layout imports this file first to guarantee the globals exist
 * before any component code runs.
 */

// 1. Symbol.dispose / Symbol.asyncDispose
(Symbol as any).dispose ??= Symbol.for("Symbol.dispose");
(Symbol as any).asyncDispose ??= Symbol.for("Symbol.asyncDispose");

// 2. SuppressedError — required when a `using` block throws AND the
//    [Symbol.dispose]() also throws (both errors must be reported).
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
