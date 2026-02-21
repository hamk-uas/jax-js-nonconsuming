import type { Rule } from "eslint";

import { getMemberName, hasAllowComment } from "../types";

const ARRAY_FACTORIES = new Set([
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
const LAX_FACTORIES = new Set([
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

const ARRAY_METHODS = new Set([
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

function isArrayProducingCall(node: any): boolean {
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

function usesAnyIdentifier(node: any, names: ReadonlySet<string>): boolean {
  for (const name of names) {
    if (usesIdentifier(node, name)) return true;
  }
  return false;
}

function hasReturnOrYieldOfAnyIdentifier(
  node: any,
  names: ReadonlySet<string>,
): boolean {
  for (const name of names) {
    if (hasReturnOrYieldOfIdentifier(node, name)) return true;
  }
  return false;
}

function hasPersistedSinkForAnyIdentifier(
  stmt: any,
  names: ReadonlySet<string>,
): boolean {
  if (stmt.type !== "ExpressionStatement") return false;
  const expr = stmt.expression;

  if (
    expr?.type === "AssignmentExpression" &&
    usesAnyIdentifier(expr.right, names) &&
    expr.left?.type !== "Identifier"
  ) {
    return true;
  }

  if (
    expr?.type === "CallExpression" &&
    expr.callee?.type === "MemberExpression"
  ) {
    if (
      expr.callee.object?.type === "Identifier" &&
      expr.callee.object.name === "tree" &&
      getMemberName(expr.callee.property) === "makeDisposable" &&
      expr.arguments.some((arg: any) => usesAnyIdentifier(arg, names))
    ) {
      return true;
    }

    const name = getMemberName(expr.callee.property);
    if (
      name &&
      ["set", "push", "unshift", "add"].includes(name) &&
      expr.arguments.some((arg: any) => usesAnyIdentifier(arg, names))
    ) {
      return true;
    }
  }

  return false;
}

/**
 * Lightweight escape tracking for pytrees and object-property flows.
 *
 * Tracks transitive aliases within the same block:
 *   a -> obj -> node -> result
 * If any tracked alias is returned/yielded or persisted to non-local storage,
 * treat the original binding as escaping and do not require `using`.
 */
function hasEscapedUseAfterDeclaration(declStmt: any, idName: string): boolean {
  const block = declStmt.parent;
  if (!block || block.type !== "BlockStatement") return false;
  const statements = block.body as any[];
  const idx = statements.indexOf(declStmt);
  if (idx < 0) return false;

  const escapedNames = new Set<string>([idName]);

  for (let i = idx + 1; i < statements.length; i++) {
    const stmt = statements[i];

    if (hasReturnOrYieldOfAnyIdentifier(stmt, escapedNames)) {
      return true;
    }

    if (hasPersistedSinkForAnyIdentifier(stmt, escapedNames)) {
      return true;
    }

    if (stmt.type === "VariableDeclaration") {
      for (const decl of stmt.declarations as any[]) {
        if (!decl.init || decl.id?.type !== "Identifier") continue;
        if (usesAnyIdentifier(decl.init, escapedNames)) {
          escapedNames.add(decl.id.name);
        }
      }
    }
  }

  return false;
}

/**
 * Deeply search a node subtree for any return/yield statement that
 * references the given identifier. This catches identifiers returned
 * inside if/else branches, try/finally, etc.
 */
function hasReturnOrYieldOfIdentifier(
  node: any,
  idName: string,
  seen: WeakSet<object> = new WeakSet(),
): boolean {
  if (!node || typeof node !== "object") return false;
  if (seen.has(node)) return false;
  seen.add(node);

  // Don't recurse into nested function/class scopes — the binding
  // is only "returned" if the enclosing function returns it.
  if (
    node.type === "FunctionDeclaration" ||
    node.type === "FunctionExpression" ||
    node.type === "ArrowFunctionExpression" ||
    node.type === "ClassDeclaration" ||
    node.type === "ClassExpression"
  ) {
    return false;
  }

  if (
    node.type === "ReturnStatement" &&
    usesIdentifier(node.argument, idName)
  ) {
    return true;
  }

  if (
    node.type === "YieldExpression" &&
    usesIdentifier(node.argument, idName)
  ) {
    return true;
  }

  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = (node as any)[key];
    if (globalThis.Array.isArray(value)) {
      for (const item of value) {
        if (hasReturnOrYieldOfIdentifier(item, idName, seen)) return true;
      }
    } else if (value && typeof value === "object") {
      if (hasReturnOrYieldOfIdentifier(value, idName, seen)) return true;
    }
  }

  return false;
}

function hasDisposeCallForIdentifier(
  node: any,
  idName: string,
  seen: WeakSet<object> = new WeakSet(),
): boolean {
  if (!node || typeof node !== "object") return false;
  if (seen.has(node)) return false;
  seen.add(node);

  if (
    node.type === "CallExpression" &&
    node.callee?.type === "MemberExpression"
  ) {
    if (getMemberName(node.callee.property) === "dispose") {
      const obj = node.callee.object;
      if (obj?.type === "Identifier" && obj.name === idName) return true;
    }
  }

  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = (node as any)[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (hasDisposeCallForIdentifier(item, idName, seen)) return true;
      }
    } else if (value && typeof value === "object") {
      if (hasDisposeCallForIdentifier(value, idName, seen)) return true;
    }
  }

  return false;
}

function hasExplicitDisposeAfterDeclaration(
  declStmt: any,
  idName: string,
): boolean {
  const block = declStmt.parent;
  if (!block || block.type !== "BlockStatement") return false;
  const statements = block.body as any[];
  const idx = statements.indexOf(declStmt);
  if (idx < 0) return false;

  for (let i = idx + 1; i < statements.length; i++) {
    if (hasDisposeCallForIdentifier(statements[i], idName)) return true;
  }

  return false;
}

function usesIdentifier(
  node: any,
  idName: string,
  seen: WeakSet<object> = new WeakSet(),
): boolean {
  if (!node || typeof node !== "object") return false;
  if (seen.has(node)) return false;
  seen.add(node);
  if (node.type === "Identifier" && node.name === idName) return true;

  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = (node as any)[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (usesIdentifier(item, idName, seen)) return true;
      }
    } else if (value && typeof value === "object") {
      if (usesIdentifier(value, idName, seen)) return true;
    }
  }

  return false;
}

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require `using` for short-lived local array bindings in jax-js code",
    },
    schema: [],
    hasSuggestions: true,
    messages: {
      requireUsing:
        "Local array binding `{{name}}` should use `using` (or be explicitly exempted). " +
        "`using` is safe and correct inside grad/jit/scan bodies — " +
        "tracers have no-op or reference-counted dispose, so ownership rules are identical in eager and traced contexts.",
      suggestUsing: "Convert `const`/`let` to `using`.",
    },
  },
  create(context) {
    return {
      VariableDeclaration(node: any) {
        if (!(node.kind === "const" || node.kind === "let")) return;
        if (hasAllowComment(context, node, "jax-js-lint: allow-non-using"))
          return;

        const isModuleScope = node.parent?.type === "Program";
        if (isModuleScope) return;

        for (const decl of node.declarations as any[]) {
          if (!decl.init || decl.id?.type !== "Identifier") continue;
          if (!isArrayProducingCall(decl.init)) continue;
          if (hasEscapedUseAfterDeclaration(node, decl.id.name)) continue;
          if (hasExplicitDisposeAfterDeclaration(node, decl.id.name)) continue;

          context.report({
            node: decl.id,
            messageId: "requireUsing",
            data: { name: decl.id.name },
            suggest: [
              {
                messageId: "suggestUsing",
                fix(fixer) {
                  return fixer.replaceTextRange(
                    [node.range[0], node.range[0] + node.kind.length],
                    "using",
                  );
                },
              },
            ],
          });
        }
      },
    };
  },
};

export default rule;
