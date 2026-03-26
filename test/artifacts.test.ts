/**
 * @file Tests for AOT linearization artifacts.
 *
 * M0.3: Placeholder tests that import types and verify they exist.
 * M2: Real correctness, reuse, and leak tests.
 */
import { describe, expect, it } from "vitest";

import { checkLeaks, init, numpy as np } from "../src";
import { aotLinearize } from "../src/frontend/artifacts";
import type {
  AotLinearizeResult,
  PrimalArtifact,
  PullbackArtifact,
  ResidualPack,
} from "../src/frontend/artifacts";

await init("wasm");

describe("artifact type stubs", () => {
  it("aotLinearize is a function", () => {
    expect(typeof aotLinearize).toBe("function");
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

describe("ResidualPack", () => {
  it("arrays returns arrays before disposal", () => {
    const f = (x: any) => [np.sin(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    expect(residuals.consumed).toBe(false);
    expect(residuals.arrays.length).toBeGreaterThan(0);

    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });

  it("arrays after dispose throws ReferenceError", () => {
    const f = (x: any) => [np.sin(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    residuals[Symbol.dispose]();
    expect(residuals.consumed).toBe(true);
    expect(() => residuals.arrays).toThrow(ReferenceError);

    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });
});

describe("PullbackArtifact", () => {
  it("run produces correct gradients for sin", async () => {
    const f = (x: any) => [np.sin(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    // sin(1.0) ≈ 0.8415
    const sinVal = await (primalsOut[0] as any).data();
    expect(sinVal[0]).toBeCloseTo(Math.sin(1.0), 4);

    // grad(sin)(1.0) = cos(1.0) ≈ 0.5403
    using ct = np.array([1.0]);
    const grads = pullback.run(residuals, [ct]);
    expect(grads.length).toBe(1);
    const gradVal = await (grads[0] as any).data();
    expect(gradVal[0]).toBeCloseTo(Math.cos(1.0), 4);

    grads.forEach((g: any) => g.dispose());
    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });

  it("run can be called multiple times (reusability)", async () => {
    const f = (x: any) => [np.sin(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    // First call with ct=1
    using ct1 = np.array([1.0]);
    const grads1 = pullback.run(residuals, [ct1]);
    const g1 = await (grads1[0] as any).data();
    grads1.forEach((g: any) => g.dispose());

    // Second call with ct=2
    using ct2 = np.array([2.0]);
    const grads2 = pullback.run(residuals, [ct2]);
    const g2 = await (grads2[0] as any).data();
    grads2.forEach((g: any) => g.dispose());

    expect(g1[0]).toBeCloseTo(Math.cos(1.0), 4);
    expect(g2[0]).toBeCloseTo(2 * Math.cos(1.0), 4);

    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });

  it("residual packs from separate runs are independently owned", async () => {
    const f = (x: any) => [np.sin(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);

    const first = primal.run([x]);
    const second = primal.run([x]);

    first.residuals[Symbol.dispose]();
    first.primalsOut.forEach((p: any) => p.dispose());

    using ct = np.array([2.0]);
    const grads = pullback.run(second.residuals, [ct]);
    const gradVal = await (grads[0] as any).data();
    expect(gradVal[0]).toBeCloseTo(2 * Math.cos(1.0), 4);

    grads.forEach((g: any) => g.dispose());
    second.residuals[Symbol.dispose]();
    second.primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });

  it("after dispose throws ReferenceError", () => {
    const f = (x: any) => [np.sin(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    pullback[Symbol.dispose]();
    using ct = np.array([1.0]);
    expect(() => pullback.run(residuals, [ct])).toThrow(ReferenceError);

    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
  });
});

describe("PrimalArtifact", () => {
  it("run produces correct primal outputs", async () => {
    const f = (x: any) => [x.mul(x)];
    using x = np.array([3.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    const val = await (primalsOut[0] as any).data();
    expect(val[0]).toBeCloseTo(9.0, 4);

    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });

  it("run returns independently owned primal outputs across calls", async () => {
    const f = (x: any) => [x.mul(x)];
    using x = np.array([3.0]);
    const { primal, pullback } = aotLinearize(f, [x]);

    const first = primal.run([x]);
    const firstVal = await (first.primalsOut[0] as any).data();
    expect(firstVal[0]).toBeCloseTo(9.0, 4);
    first.residuals[Symbol.dispose]();
    first.primalsOut.forEach((p: any) => p.dispose());

    const second = primal.run([x]);
    const secondVal = await (second.primalsOut[0] as any).data();
    expect(secondVal[0]).toBeCloseTo(9.0, 4);

    second.residuals[Symbol.dispose]();
    second.primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });

  it("after dispose throws ReferenceError", () => {
    const f = (x: any) => [np.sin(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);

    primal[Symbol.dispose]();
    expect(() => primal.run([x])).toThrow(ReferenceError);
    expect(() => primal.forwardJaxpr).toThrow(ReferenceError);

    pullback[Symbol.dispose]();
  });
});

describe("aotLinearize integration", () => {
  it("artifact lifecycle is leak-free when disposed correctly", () => {
    const outerResult = checkLeaks.stop();
    expect(outerResult.leaked).toBe(0);

    checkLeaks.start();
    const x = np.array([1.0, 2.0, 3.0]);
    const f = (arg: any) => {
      using sinArg = np.sin(arg);
      using weighted = sinArg.mul(arg);
      return [weighted.sum()];
    };
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    const ct = np.array(1.0);
    const grads = pullback.run(residuals, [ct]);

    grads.forEach((g: any) => g.dispose());
    ct.dispose();
    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
    x.dispose();

    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
    checkLeaks.start();
  });

  it("produces correct grad for polynomial x^3", async () => {
    const f = (x: any) => {
      using x2 = x.mul(x) as any;
      return [x2.mul(x)];
    };
    using x = np.array([2.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    // f(2) = 8
    const val = await (primalsOut[0] as any).data();
    expect(val[0]).toBeCloseTo(8.0, 4);

    // f'(x) = 3x^2, f'(2) = 12
    using ct = np.array([1.0]);
    const grads = pullback.run(residuals, [ct]);
    const gradVal = await (grads[0] as any).data();
    expect(gradVal[0]).toBeCloseTo(12.0, 4);

    grads.forEach((g: any) => g.dispose());
    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });

  it("residual from primal feeds into pullback correctly", async () => {
    // exp(x) — simple function where residuals are the forward-pass values
    const f = (x: any) => [np.exp(x)];
    using x = np.array([1.0]);
    const { primal, pullback } = aotLinearize(f, [x]);
    const { primalsOut, residuals } = primal.run([x]);

    // exp(1) ≈ 2.7183
    const val = await (primalsOut[0] as any).data();
    expect(val[0]).toBeCloseTo(Math.exp(1.0), 3);

    // grad(exp)(1) = exp(1) ≈ 2.7183
    using ct = np.array([1.0]);
    const grads = pullback.run(residuals, [ct]);
    const gradVal = await (grads[0] as any).data();
    expect(gradVal[0]).toBeCloseTo(Math.exp(1.0), 3);

    grads.forEach((g: any) => g.dispose());
    residuals[Symbol.dispose]();
    primalsOut.forEach((p: any) => p.dispose());
    primal[Symbol.dispose]();
    pullback[Symbol.dispose]();
  });
});
