/**
 * Tests for the debug-mode diagnostic that reports JIT-rescued intermediates.
 *
 * When `setDebug(1)`, makeJaxpr and disposePeIntermediates emit console.warn
 * messages listing intermediates that would leak in eager mode but were
 * silently cleaned up by the tracing infrastructure.
 *
 * Run: pnpm vitest run test/ownership-diagnostic.test.ts
 */
import {
  clearCaches,
  grad,
  jit,
  nn,
  numpy as np,
  scipySpecial,
  setDebug,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterEach, describe, expect, test, vi } from "vitest";

import { disposePeIntermediates } from "../src/frontend/linearize";

afterEach(() => {
  setDebug(0);
  clearCaches();
});

describe("ownership divergence diagnostic", () => {
  test("makeJaxpr warns about rescued concrete consts at debug level 1", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      setDebug(1);

      // np.array([2]) inside the body creates a concrete array that becomes
      // a const. Without `using`, its creation ref is balanced by makeJaxpr.
      const f = jit((x: np.Array) => {
        return x.mul(np.array([2])).sum();
      });
      using x = np.array([1, 2, 3]);
      using _result = f(x);

      const rescueWarnings = warnSpy.mock.calls.filter(
        (args) =>
          typeof args[0] === "string" && args[0].includes("makeJaxpr rescued"),
      );
      expect(rescueWarnings.length).toBeGreaterThan(0);
      expect(rescueWarnings[0][0]).toContain("would leak in eager mode");
      expect(rescueWarnings[0][0]).toContain("using");
    } finally {
      warnSpy.mockRestore();
    }
  });

  test("grad warns about rescued PE intermediates at debug level 1", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      setDebug(1);

      // Inside grad, intermediate arrays from the forward pass are cleaned
      // up by disposePeIntermediates. The diagnostic should fire.
      const df = grad((x: np.Array) => {
        return x.mul(x).sum();
      });
      using x = np.array([1, 2, 3]);
      using _result = df(x);

      const rescueWarnings = warnSpy.mock.calls.filter(
        (args) => typeof args[0] === "string" && args[0].includes("rescued"),
      );
      expect(rescueWarnings.length).toBeGreaterThan(0);
      expect(rescueWarnings[0][0]).toContain("would leak in eager mode");
    } finally {
      warnSpy.mockRestore();
    }
  });

  test("no diagnostic when user disposes intermediates properly", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      setDebug(1);

      // With proper `using`, no rescue is needed.
      const f = jit((x: np.Array) => {
        using two = np.array([2]);
        using scaled = x.mul(two);
        return scaled.sum();
      });
      using x = np.array([1, 2, 3]);
      using _result = f(x);

      const rescueWarnings = warnSpy.mock.calls.filter(
        (args) =>
          typeof args[0] === "string" && args[0].includes("makeJaxpr rescued"),
      );
      expect(rescueWarnings.length).toBe(0);
    } finally {
      warnSpy.mockRestore();
    }
  });

  test("no diagnostic at debug level 0", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      setDebug(0);

      const f = jit((x: np.Array) => {
        return x.mul(x).sum();
      });
      using x = np.array([1, 2, 3]);
      using _result = f(x);

      const rescueWarnings = warnSpy.mock.calls.filter(
        (args) => typeof args[0] === "string" && args[0].includes("rescued"),
      );
      expect(rescueWarnings.length).toBe(0);
    } finally {
      warnSpy.mockRestore();
    }
  });

  test("diagnostic includes array descriptions", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      setDebug(1);

      const f = jit((x: np.Array) => {
        return x.mul(np.array([2])).sum();
      });
      using x = np.array([1, 2, 3]);
      using _result = f(x);

      const rescueWarnings = warnSpy.mock.calls.filter(
        (args) =>
          typeof args[0] === "string" && args[0].includes("makeJaxpr rescued"),
      );
      expect(rescueWarnings.length).toBeGreaterThan(0);
      // Should contain the Array toString format like "Array:float32[1]"
      expect(rescueWarnings[0][0]).toMatch(/Array:\w+\[/);
    } finally {
      warnSpy.mockRestore();
    }
  });

  test("PE diagnostic deduplicates multiple wrappers for the same concrete array", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      setDebug(1);

      using concrete = np.array([1, 2, 3]);
      // jax-js-lint: allow-ref -- retain extra ownership so both wrappers can release the same concrete array
      concrete.ref;
      // jax-js-lint: allow-ref -- retain extra ownership so both wrappers can release the same concrete array
      concrete.ref;

      const wrap = () => {
        let alive = true;
        return {
          val: concrete,
          isAliveForCleanup: () => alive,
          dispose: () => {
            if (!alive) return;
            alive = false;
            concrete.dispose();
          },
        };
      };

      disposePeIntermediates([wrap(), wrap()] as any, new Set());

      const rescueWarnings = warnSpy.mock.calls.filter(
        (args) =>
          typeof args[0] === "string" && args[0].includes("grad/vjp rescued"),
      );
      expect(rescueWarnings.length).toBe(1);
      expect(rescueWarnings[0][0]).toContain("rescued 1 PE intermediate");
      const desc = concrete.toString();
      expect(rescueWarnings[0][0].split(desc).length - 1).toBe(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  test("public library helpers stay warning-free under debug level 1", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      setDebug(1);

      const f = jit((x: np.Array) => {
        using div = np.divide(1, x);
        using sub = np.subtract(1, x);
        using sinc = np.sinc(x);
        using silu = nn.silu(x);
        using probs = np.clip(x.mul(0.1).add(0.2), 1e-3, 1 - 1e-3);
        using logit = scipySpecial.logit(probs);
        using total = div.add(sub).add(sinc).add(silu).add(logit);
        return total.sum();
      });

      using x = np.array([2, 4, 8]);
      using _result = f(x);

      const rescueWarnings = warnSpy.mock.calls.filter(
        (args) =>
          typeof args[0] === "string" && args[0].includes("makeJaxpr rescued"),
      );
      expect(rescueWarnings).toHaveLength(0);
    } finally {
      warnSpy.mockRestore();
    }
  });
});
