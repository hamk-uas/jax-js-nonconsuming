import { expect, test } from "vitest";

import { erf, erfc } from "../../alu";
import {
  wasm_atan,
  wasm_cos,
  wasm_erf,
  wasm_erfc,
  wasm_exp,
  wasm_log,
  wasm_sin,
  wasm_threefry2x32,
} from "./builtins";
import { CodeGenerator } from "./wasmblr";

function relativeError(wasmResult: number, jsResult: number): number {
  return Math.abs(wasmResult - jsResult) / Math.max(Math.abs(jsResult), 1e-3);
}

test("wasm_exp has relative error < 2e-7", async () => {
  const cg = new CodeGenerator();

  const expFunc = wasm_exp(cg);
  cg.export(expFunc, "exp");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { exp } = instance.exports as { exp(x: number): number };

  const testValues = [
    -5, -2, -1, -0.5, 0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.99,
    1, 2, 3, 5, 10,
  ];
  for (const x of testValues) {
    expect(relativeError(exp(x), Math.exp(x))).toBeLessThan(2e-7);
  }

  // Test edge cases
  expect(exp(Infinity)).toBe(Infinity);
  expect(exp(-Infinity)).toBe(0);
  expect(exp(NaN)).toBeNaN();
});

test("wasm_log has relative error < 5e-7", async () => {
  const cg = new CodeGenerator();

  const logFunc = wasm_log(cg);
  cg.export(logFunc, "log");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { log } = instance.exports as { log(x: number): number };

  const testValues = [0.01, 0.1, 0.5, 1, 1.5, 2, Math.E, 5, 10, 100];
  for (const x of testValues) {
    expect(relativeError(log(x), Math.log(x))).toBeLessThan(5e-7);
  }

  // Test edge cases: log(x < 0) should return NaN
  expect(log(-1)).toBeNaN();
  expect(log(0)).toBe(-Infinity);
  expect(log(Infinity)).toBe(Infinity);
  expect(log(NaN)).toBeNaN();
});

test("wasm_sin has absolute error < 5e-7", async () => {
  const cg = new CodeGenerator();

  const sinFunc = wasm_sin(cg);
  cg.export(sinFunc, "sin");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { sin } = instance.exports as { sin(x: number): number };

  // Test a range of values including critical points
  const testValues = [
    -2 * Math.PI,
    -Math.PI,
    -Math.PI / 2,
    -Math.PI / 4,
    0,
    Math.PI / 6,
    Math.PI / 4,
    Math.PI / 3,
    Math.PI / 2,
    Math.PI,
    (3 * Math.PI) / 2,
    2 * Math.PI,
    5,
    10,
    -5,
    -10,
  ];

  for (const x of testValues) {
    expect(Math.abs(sin(x) - Math.sin(x))).toBeLessThan(5e-7);
  }

  expect(sin(Infinity)).toBeNaN();
  expect(sin(-Infinity)).toBeNaN();
  expect(sin(NaN)).toBeNaN();
});

test("wasm_cos has absolute error < 5e-7", async () => {
  const cg = new CodeGenerator();

  const cosFunc = wasm_cos(cg);
  cg.export(cosFunc, "cos");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { cos } = instance.exports as { cos(x: number): number };

  // Test a range of values including critical points
  const testValues = [
    -2 * Math.PI,
    -Math.PI,
    -Math.PI / 2,
    -Math.PI / 4,
    0,
    Math.PI / 6,
    Math.PI / 4,
    Math.PI / 3,
    Math.PI / 2,
    Math.PI,
    (3 * Math.PI) / 2,
    2 * Math.PI,
    5,
    10,
    -5,
    -10,
  ];

  for (const x of testValues) {
    expect(Math.abs(cos(x) - Math.cos(x))).toBeLessThan(5e-7);
  }

  expect(cos(Infinity)).toBeNaN();
  expect(cos(-Infinity)).toBeNaN();
  expect(cos(NaN)).toBeNaN();
});

test("wasm_atan has relative error < 2e-6", async () => {
  const cg = new CodeGenerator();

  const atanFunc = wasm_atan(cg);
  cg.export(atanFunc, "atan");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { atan } = instance.exports as { atan(x: number): number };

  // Test a range of values including critical points
  const testValues = [
    -1000, -100, -50, -10, -5, -2, -1, -0.5, -0.1, -0.01, 0, 0.01, 0.1, 0.5, 1,
    2, 5, 10, 50, 100, 1000,
  ];

  for (const x of testValues) {
    expect(relativeError(atan(x), Math.atan(x))).toBeLessThan(2e-6);
  }

  expect(atan(Infinity)).toBeCloseTo(Math.PI / 2);
  expect(atan(-Infinity)).toBeCloseTo(-Math.PI / 2);
  expect(atan(NaN)).toBeNaN();
});

test("wasm_erf has relative error < 2e-6", async () => {
  const cg = new CodeGenerator();

  const erfFunc = wasm_erf(cg, wasm_exp(cg));
  cg.export(erfFunc, "erf");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { erf: wasmErf } = instance.exports as { erf(x: number): number };

  const testValues = [-3, -2, -1, -0.5, -0.1, 0, 0.01, 0.1, 0.5, 1, 2, 3];

  for (const x of testValues) {
    expect(relativeError(wasmErf(x), erf(x))).toBeLessThan(2e-6);
  }

  expect(wasmErf(Infinity)).toBe(1);
  expect(wasmErf(-Infinity)).toBe(-1);
  expect(wasmErf(NaN)).toBeNaN();
});

test("wasm_erfc has relative error < 2e-7", async () => {
  const cg = new CodeGenerator();

  const erfcFunc = wasm_erfc(cg, wasm_exp(cg));
  cg.export(erfcFunc, "erfc");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { erfc: wasmErfc } = instance.exports as { erfc(x: number): number };

  const testValues = [-3, -2, -1, -0.5, -0.1, 0, 0.01, 0.1, 0.5, 1, 2, 3];

  for (const x of testValues) {
    expect(relativeError(wasmErfc(x), erfc(x))).toBeLessThan(2e-7);
  }

  expect(wasmErfc(Infinity)).toBe(0);
  expect(wasmErfc(-Infinity)).toBe(2);
  expect(wasmErfc(NaN)).toBeNaN();
});

test("wasm_threefry2x32 produces expected results", async () => {
  const cg = new CodeGenerator();

  const threefryFunc = wasm_threefry2x32(cg);
  cg.export(threefryFunc, "threefry2x32");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);

  const threefry2x32 = instance.exports.threefry2x32 as CallableFunction;

  // Test known vector: all zeros input
  const result0 = threefry2x32(0, 0, 0, 0) as [number, number];
  expect(Array.isArray(result0)).toBe(true);
  expect(result0).toHaveLength(2);

  // Convert to unsigned 32-bit for comparison
  const x0 = result0[0] >>> 0; // Convert to unsigned 32-bit
  const x1 = result0[1] >>> 0;

  expect(x0).toBe(1797259609);
  expect(x1).toBe(2579123966);

  // Test that different inputs produce different outputs
  const result1 = threefry2x32(0, 0, 0, 0) as [number, number];
  const result2 = threefry2x32(0, 0, 0, 1) as [number, number];
  const result3 = threefry2x32(1, 0, 0, 0) as [number, number];

  expect(result1).not.toEqual(result2);
  expect(result1).not.toEqual(result3);
  expect(result2).not.toEqual(result3);

  // Test with non-zero keys
  const result4 = threefry2x32(
    0xdeadbeef,
    0xcafebabe,
    0x12345678,
    0x87654321,
  ) as [number, number];
  expect(Array.isArray(result4)).toBe(true);
  expect(result4).toHaveLength(2);
});
