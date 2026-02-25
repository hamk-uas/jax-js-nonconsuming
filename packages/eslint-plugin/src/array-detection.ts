/**
 * Shared heuristic for detecting array-producing calls in ESLint rules.
 *
 * Used by `require-using` and `no-nested-array-leak`.
 */

import { getMemberName } from "./types";

export const ARRAY_FACTORIES = new Set([
  "array",
  "zeros",
  "zerosLike",
  "ones",
  "onesLike",
  "full",
  "empty",
  "arange",
  "linspace",
  "eye",
  "identity",
  "concatenate",
  "stack",
  "hstack",
  "vstack",
  "dstack",
  "columnStack",
  "diag",
  "diagonal",
  "flip",
  "flipud",
  "fliplr",
  "tile",
  "repeat",
  "sort",
  "argsort",
  "clip",
  "absolute",
  "abs",
  "sign",
  "square",
  "einsum",
  "tensordot",
  "inner",
  "outer",
  "vdot",
  "vecdot",
  "convolve",
  "correlate",
  "cumsum",
  "where",
  "take",
  "meshgrid",
  "broadcastTo",
  "expandDims",
  "squeeze",
  "ravel",
  "swapaxes",
  "matrixTranspose",
  "hamming",
  "hann",
  // numpy long-form function names (aliases in numpy.ts)
  "subtract",
  "multiply",
  "divide",
  "trueDivide",
  "floorDivide",
  "negative",
  "reciprocal",
  "remainder",
  "fmod",
  "power",
  "positive",
  "heaviside",
  "hypot",
  "atan2",
  "moveaxis",
  "pad",
  // numpy functions returning arrays
  "allclose",
  "argmax",
  "argmin",
  "corrcoef",
  "cov",
  "deg2rad",
  "divmod",
  "exp2",
  "expm1",
  "frexp",
  "isinf",
  "isnan",
  "isneginf",
  "isposinf",
  "ldexp",
  "log10",
  "log1p",
  "log2",
  "nanToNum",
  "ptp",
  "rad2deg",
  "std",
  "trace",
  "trunc",
  "var_",
  // numpy trig aliases
  "arccosh",
  "arcsinh",
  "arctanh",
  "cbrt",
  "cosh",
  "degrees",
  "radians",
  "sinc",
  "sinh",
]);

/**
 * Functions on the `lax` namespace that return arrays.
 * `lax.scan` is excluded — handled by `require-scan-result-dispose`.
 */
export const LAX_FACTORIES = new Set([
  "associativeScan",
  "dot",
  "conv",
  "convGeneralDilated",
  "convWithGeneralPadding",
  "convTranspose",
  "reduceWindow",
  "erf",
  "erfc",
  "stopGradient",
  "dynamicUpdateSlice",
  "topK",
]);

export const ARRAY_METHODS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "pow",
  "mod",
  "reshape",
  "transpose",
  "flatten",
  "sum",
  "prod",
  "mean",
  "min",
  "max",
  "astype",
  "neg",
  "abs",
  "exp",
  "log",
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "tanh",
  "sqrt",
  "square",
  "maximum",
  "minimum",
  "dot",
  "matmul",
  "clip",
  "squeeze",
  "expandDims",
  "at",
  "slice",
]);

export function isArrayProducingCall(node: any): boolean {
  if (!node || node.type !== "CallExpression") return false;

  const callee = node.callee;
  if (callee.type === "Identifier") {
    return ARRAY_FACTORIES.has(callee.name);
  }

  if (callee.type === "MemberExpression") {
    const property = getMemberName(callee.property);
    if (!property) return false;

    if (callee.object.type === "Identifier" && callee.object.name === "Math") {
      return false;
    }

    if (
      callee.object.type === "Identifier" &&
      callee.object.name === "np" &&
      ARRAY_FACTORIES.has(property)
    ) {
      return true;
    }

    if (
      callee.object.type === "Identifier" &&
      callee.object.name === "lax" &&
      LAX_FACTORIES.has(property)
    ) {
      return true;
    }

    return ARRAY_METHODS.has(property);
  }

  return false;
}
