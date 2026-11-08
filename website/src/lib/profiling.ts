// Utilities for profiling, especially for GPU.

import { browser } from "$app/env";

export function isMobile(): boolean {
  return browser && /Mobi|Android/i.test(navigator.userAgent);
}

export function countMethodCalls(cls: any, functionName: string): () => number {
  const method = cls.prototype[functionName];
  let callCount = 0;
  cls.prototype[functionName] = function (...args: any[]) {
    callCount++;
    return method.apply(this, args);
  };
  return () => {
    // Restore original method
    cls.prototype[functionName] = method;
    return callCount;
  };
}
