/**
 * Shared utilities for detecting whether code is inside a tracing transform
 * body (jit, grad, vmap, scan, etc.) where intermediates are automatically
 * managed by the JIT compiler.
 *
 * Used by `no-array-chain` and `no-nested-array-leak` to suppress warnings
 * inside traced contexts.
 */

import type { Rule } from "eslint";

import { getMemberName } from "./types";

/**
 * Tracing transforms whose first (function) argument is traced — not eagerly
 * executed — so array operations inside those bodies do NOT create real GPU
 * temporaries.  The JIT compiler manages intermediates automatically.
 */
export const TRACING_TRANSFORMS = new Set([
  "jit",
  "grad",
  "valueAndGrad",
  "jvp",
  "vjp",
  "vmap",
  "jacfwd",
  "jacrev",
  "hessian",
  "linearize",
  "makeJaxpr",
]);

/**
 * Check whether a callee AST node is a known tracing transform.
 */
export function isTransformCallee(callee: any): boolean {
  // Direct call: jit(fn), grad(fn), etc.
  if (callee?.type === "Identifier" && TRACING_TRANSFORMS.has(callee.name))
    return true;
  // Method call: lax.scan(fn, ...), lax.associativeScan(fn, ...)
  if (callee?.type === "MemberExpression") {
    const method = getMemberName(callee.property);
    if (method === "scan" || method === "associativeScan") return true;
  }
  return false;
}

/**
 * Walk up the scope chain looking for a Variable with the given name.
 */
export function findVariable(scope: any, name: string): any {
  let s = scope;
  while (s) {
    const v = s.variables?.find((v: any) => v.name === name);
    if (v) return v;
    s = s.upper;
  }
  return null;
}

/**
 * Walk up the AST from `node`.  If we find a function expression / arrow
 * function that is used as the **body argument** of a tracing transform,
 * the node is inside a traced context and should be suppressed.
 *
 * Handles two cases:
 * 1. Inline: `jit((x) => np.add(np.square(x), x))` — function is the first arg.
 * 2. Named:  `const f = (x) => np.add(np.square(x), x); grad(f)(x)` — function
 *    is assigned to a variable that is later passed to a transform.
 *
 * Also suppresses inside `lax.scan` and `lax.associativeScan` body functions.
 */
export function isInsideTracedBody(
  node: any,
  context: Rule.RuleContext,
): boolean {
  let current = node.parent;
  while (current) {
    if (
      current.type === "ArrowFunctionExpression" ||
      current.type === "FunctionExpression"
    ) {
      const parent = current.parent;

      // Case 1: Inline function as first arg to transform.
      if (
        parent?.type === "CallExpression" &&
        parent.arguments[0] === current
      ) {
        if (isTransformCallee(parent.callee)) return true;
      }

      // Case 2: Function assigned to a const/let, and at least one
      // reference site passes it as the first arg to a tracing transform.
      if (
        parent?.type === "VariableDeclarator" &&
        parent.id?.type === "Identifier"
      ) {
        const scope = context.sourceCode.getScope(parent);
        const variable = findVariable(scope, parent.id.name);
        if (variable) {
          for (const ref of variable.references) {
            const id = ref.identifier;
            // Used as: grad(f), jit(f), vmap(f), etc.
            if (
              id.parent?.type === "CallExpression" &&
              id.parent.arguments[0] === id &&
              isTransformCallee(id.parent.callee)
            ) {
              return true;
            }
          }
        }
      }
    }
    current = current.parent;
  }
  return false;
}
