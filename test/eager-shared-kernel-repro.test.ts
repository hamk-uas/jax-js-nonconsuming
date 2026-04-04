import { expect, test } from "vitest";

import {
  checkLeaks,
  clearCaches,
  grad,
  hessian,
  lax,
  numpy as np,
  tree,
} from "../src";

type Variant = {
  name: string;
  useMetRolling: boolean;
  useRunningTotal: boolean;
  useConditionalTotal: boolean;
  disposeConditionalTemps: boolean;
  useAux: boolean;
  carryPrevObs: boolean;
};

type KernelCarry = {
  params: np.Array;
  metRolling: np.Array;
  isFirstMet: np.Array;
  total: np.Array;
  prevObs: np.Array;
  aux: np.Array;
};

type KernelOutput = {
  params: np.Array;
  total: np.Array;
};

type ScalarCarry = {
  params: number[];
  metRolling: number[];
  isFirstMet: boolean;
  total: number;
  prevObs: number;
  aux: number;
};

function materializeCarry(carry: ScalarCarry): KernelCarry {
  return {
    params: np.array(carry.params),
    metRolling: np.array(carry.metRolling),
    isFirstMet: np.array(carry.isFirstMet),
    total: np.array(carry.total),
    prevObs: np.array(carry.prevObs),
    aux: np.array(carry.aux),
  };
}

function scalarizeCarry(carry: KernelCarry): ScalarCarry {
  return {
    params: carry.params.js() as number[],
    metRolling: carry.metRolling.js() as number[],
    isFirstMet: Boolean(carry.isFirstMet.item()),
    total: carry.total.item() as number,
    prevObs: carry.prevObs.item() as number,
    aux: carry.aux.item() as number,
  };
}

function disposeCarryIfNotAliased(
  initial: KernelCarry,
  escaped: KernelCarry,
): void {
  const disposeIfNotAliased = (i: np.Array, e: np.Array) => {
    if (i === e) return;
    try {
      i.dispose();
    } catch (error) {
      if (
        !(error instanceof ReferenceError) ||
        !String(error.message).includes("Referenced tracer Array")
      ) {
        throw error;
      }
    }
  };

  disposeIfNotAliased(initial.params, escaped.params);
  disposeIfNotAliased(initial.metRolling, escaped.metRolling);
  disposeIfNotAliased(initial.isFirstMet, escaped.isFirstMet);
  disposeIfNotAliased(initial.total, escaped.total);
  disposeIfNotAliased(initial.prevObs, escaped.prevObs);
  disposeIfNotAliased(initial.aux, escaped.aux);
}

function disposeOutputIfNotAliased(
  output: KernelOutput,
  carry: KernelCarry,
): void {
  if (output.params !== carry.params) output.params.dispose();
  if (output.total !== carry.total) output.total.dispose();
}

function makeSharedStep(alpha: np.Array, variant: Variant) {
  const objective = (params: np.Array): np.Array => {
    using a = lax.dynamicIndexInDim(params, 0, 0, false);
    using b = lax.dynamicIndexInDim(params, 1, 0, false);
    return a.mul(a).add(b.mul(b)).mul(alpha);
  };

  return (
    carry: KernelCarry,
    forcing: { obs: np.Array },
  ): [KernelCarry, KernelOutput] => {
    const hourlyBody = (_i: np.Array, params: np.Array): np.Array => {
      const g = grad(objective)(params);
      using h = hessian(objective)(params);
      using eye = np.eye(2, { dtype: params.dtype });
      using reg = eye.mul(0.01);
      using hReg = h.add(reg);
      using step = np.linalg.solve(hReg, g);
      using scaled = step.mul(0.1);
      const next = params.sub(scaled);
      g.dispose();
      return next;
    };

    const nextParams = lax.foriLoop(0, 2, hourlyBody, carry.params) as np.Array;
    const nextTotal = variant.useRunningTotal
      ? (() => {
          if (variant.useConditionalTotal) {
            if (variant.disposeConditionalTemps) {
              using nextParamsSum = nextParams.sum();
              using better = nextParamsSum.greater(carry.total);
              using fallback = carry.total.add(forcing.obs);
              return np.where(better, nextParamsSum, fallback);
            }
            using better = nextParams.sum().greater(carry.total);
            return np.where(
              better,
              nextParams.sum(),
              carry.total.add(forcing.obs),
            );
          }
          return carry.total.add(forcing.obs);
        })()
      : carry.total;
    const nextAux = variant.useAux
      ? (() => {
          using positive = forcing.obs.greater(0.0);
          return np.where(positive, carry.aux.add(forcing.obs), carry.aux);
        })()
      : carry.aux;

    let nextMet = carry.metRolling;
    let nextFirst = carry.isFirstMet;
    if (variant.useMetRolling) {
      using obsPair = np.stack([forcing.obs, forcing.obs.add(1.0)]);
      using smoothOld = carry.metRolling.mul(0.5);
      using smoothNew = obsPair.mul(0.5);
      using smoothed = smoothOld.add(smoothNew);
      nextMet = np.where(carry.isFirstMet, obsPair, smoothed);
      nextFirst = np.where(carry.isFirstMet, false, carry.isFirstMet);
    }

    const nextCarry: KernelCarry = {
      params: nextParams,
      metRolling: nextMet,
      isFirstMet: nextFirst,
      total: nextTotal,
      prevObs: variant.carryPrevObs ? forcing.obs : carry.prevObs,
      aux: nextAux,
    };
    return [nextCarry, { params: nextParams, total: nextTotal }];
  };
}

function measureLeak(variant: Variant): number {
  const outer = checkLeaks.stop();
  expect(outer.leaked).toBe(0);
  checkLeaks.start();

  using alpha = np.array(2.5);
  const step = makeSharedStep(alpha, variant);
  let carry: ScalarCarry = {
    params: [2.0, 1.0],
    metRolling: [0.0, 0.0],
    isFirstMet: true,
    total: 0.0,
    prevObs: 0.0,
    aux: 0.0,
  };

  try {
    for (const obs of [0.25, 0.5]) {
      const materialized = materializeCarry(carry);
      const forcing = { obs: np.array(obs) };
      let nextCarry: KernelCarry | null = null;
      let output: KernelOutput | null = null;
      try {
        [nextCarry, output] = step(materialized, forcing);
        carry = scalarizeCarry(nextCarry);
      } finally {
        if (nextCarry == null || output == null) {
          tree.dispose(materialized);
          forcing.obs.dispose();
        } else {
          disposeCarryIfNotAliased(materialized, nextCarry);
          if (forcing.obs !== nextCarry.prevObs) forcing.obs.dispose();
          disposeOutputIfNotAliased(output, nextCarry);
          tree.dispose(nextCarry);
        }
      }
    }
  } finally {
    clearCaches();
  }

  const report = checkLeaks.stop();
  checkLeaks.start();
  return report.leaked;
}

// Characterization: the reused traced kernel is not inherently leaky here.
// The leak comes from inline branch temporaries passed into `where(...)`
// without explicit ownership in the eager wrapper pattern.
test("characterization: eager wrapper leaks when conditional branch temps are inline", () => {
  const leaked = measureLeak({
    name: "full",
    useMetRolling: true,
    useRunningTotal: true,
    useConditionalTotal: true,
    disposeConditionalTemps: false,
    useAux: true,
    carryPrevObs: true,
  });
  expect(leaked).toBeGreaterThan(0);
});

test("bisect: leak does not require prevObs or aux carry fields", () => {
  const full = measureLeak({
    name: "full",
    useMetRolling: true,
    useRunningTotal: true,
    useConditionalTotal: true,
    disposeConditionalTemps: false,
    useAux: true,
    carryPrevObs: true,
  });
  const withoutPrevObsAndAux = measureLeak({
    name: "no-prevobs-no-aux",
    useMetRolling: true,
    useRunningTotal: true,
    useConditionalTotal: true,
    disposeConditionalTemps: false,
    useAux: false,
    carryPrevObs: false,
  });

  expect(full).toBeGreaterThan(0);
  expect(withoutPrevObsAndAux).toBeGreaterThan(0);
});

test("bisect: leak still reproduces with params + running total only", () => {
  const leaked = measureLeak({
    name: "params-plus-total",
    useMetRolling: false,
    useRunningTotal: true,
    useConditionalTotal: true,
    disposeConditionalTemps: false,
    useAux: false,
    carryPrevObs: false,
  });

  expect(leaked).toBeGreaterThan(0);
});

test("bisect: unconditional running total is leak-free", () => {
  const leaked = measureLeak({
    name: "params-plus-unconditional-total",
    useMetRolling: false,
    useRunningTotal: true,
    useConditionalTotal: false,
    disposeConditionalTemps: false,
    useAux: false,
    carryPrevObs: false,
  });

  expect(leaked).toBe(0);
});

test("bisect: conditional running total is leak-free when branch temps are explicitly disposed", () => {
  const leaked = measureLeak({
    name: "params-plus-conditional-total-explicit-temps",
    useMetRolling: false,
    useRunningTotal: true,
    useConditionalTotal: true,
    disposeConditionalTemps: true,
    useAux: false,
    carryPrevObs: false,
  });

  expect(leaked).toBe(0);
});

test("bisect: params-only eager wrapper is leak-free", () => {
  const leaked = measureLeak({
    name: "params-only",
    useMetRolling: false,
    useRunningTotal: false,
    useConditionalTotal: false,
    disposeConditionalTemps: false,
    useAux: false,
    carryPrevObs: false,
  });

  expect(leaked).toBe(0);
});
