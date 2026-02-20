/**
 * @file Tests for AOT linearization artifacts.
 *
 * M0.3: Placeholder tests that import types and verify they exist.
 * Later milestones (M2.x) will add real correctness and leak tests.
 */
import { describe, expect, it } from "vitest";

import { aotLinearize } from "../src/frontend/artifacts";
import type {
  AotLinearizeResult,
  PrimalArtifact,
  PullbackArtifact,
  ResidualPack,
} from "../src/frontend/artifacts";

describe("artifact type stubs", () => {
  it("aotLinearize is a function", () => {
    expect(typeof aotLinearize).toBe("function");
  });

  it("aotLinearize throws not-yet-implemented", () => {
    expect(() => aotLinearize(() => [], [])).toThrow(
      "aotLinearize: not yet implemented",
    );
  });

  it("artifact interfaces are importable", () => {
    // Type-only check — if this file compiles, the interfaces exist.
    const _check: AotLinearizeResult | null = null;
    const _p: PrimalArtifact | null = null;
    const _pb: PullbackArtifact | null = null;
    const _r: ResidualPack | null = null;
    expect(_check).toBeNull();
    expect(_p).toBeNull();
    expect(_pb).toBeNull();
    expect(_r).toBeNull();
  });
});
