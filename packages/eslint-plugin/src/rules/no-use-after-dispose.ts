import type { Rule } from "eslint";

import { getMemberName } from "../types";

function resolveVariable(context: Rule.RuleContext, node: any): any {
  if (!node || node.type !== "Identifier") return null;
  let scope: any = context.sourceCode.getScope(node);
  while (scope) {
    const found = scope.set?.get?.(node.name);
    if (found) return found;
    scope = scope.upper;
  }
  return null;
}

function isIdentifierReference(node: any, parent: any): boolean {
  if (!parent) return false;
  if (parent.type === "VariableDeclarator" && parent.id === node) return false;
  if (
    parent.type === "MemberExpression" &&
    parent.property === node &&
    !parent.computed
  )
    return false;
  if (parent.type === "Property" && parent.key === node && !parent.computed)
    return false;
  if (
    parent.type === "FunctionDeclaration" ||
    parent.type === "FunctionExpression" ||
    parent.type === "ArrowFunctionExpression"
  ) {
    return !parent.params.includes(node);
  }
  return true;
}

/**
 * Check if a `.dispose()` call is inside a callback/closure (arrow function or
 * function expression) rather than at the current statement level.  Deferred
 * callbacks like `onTestFinished(() => x.dispose())` should not mark the
 * variable as disposed in the enclosing scope.
 */
function isInsideCallback(node: any): boolean {
  let current = node.parent;
  while (current) {
    if (
      current.type === "ArrowFunctionExpression" ||
      current.type === "FunctionExpression"
    ) {
      return true;
    }
    // Stop walking at function/program boundaries — these are the scope
    // boundaries we care about.
    if (current.type === "FunctionDeclaration" || current.type === "Program") {
      return false;
    }
    current = current.parent;
  }
  return false;
}

/** Methods that consume the array — any use after calling these is a bug. */
const CONSUMING_METHODS = new Set(["dispose", "consumeData"]);

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow using local variables after calling .dispose() or .consumeData()",
    },
    schema: [],
    messages: {
      useAfterDispose:
        "`{{name}}` is used after `.{{method}}()` on line {{consumeLine}}.",
    },
  },
  create(context) {
    const disposedAt = new Map<any, { end: number; line: number; method: string }>();

    return {
      CallExpression(node: any) {
        if (node.callee?.type !== "MemberExpression") return;
        const prop = getMemberName(node.callee.property);
        if (!prop || !CONSUMING_METHODS.has(prop)) return;
        // tree.dispose(arg) disposes arg, not the receiver `tree`.
        // Only mark the receiver as consumed for zero-argument calls.
        if (node.arguments.length > 0) return;
        const obj = node.callee.object;
        if (obj?.type !== "Identifier") return;

        // Ignore .dispose() calls inside callbacks / closures — they execute
        // deferred (e.g. onTestFinished(() => x.dispose())), so the variable
        // is NOT disposed at the call-site in the enclosing scope.
        if (isInsideCallback(node)) return;

        const variable = resolveVariable(context, obj);
        if (!variable) return;
        // Don't let a .dispose() call overwrite a prior .consumeData() marker.
        // The .dispose() itself will be caught by the Identifier visitor as a
        // use-after-consumeData error.
        const existing = disposedAt.get(variable);
        if (existing && existing.method !== "dispose") return;
        disposedAt.set(variable, {
          end: node.range[1],
          line: node.loc?.start?.line ?? 0,
          method: prop,
        });
      },

      // Clear disposed state when a variable is reassigned.
      // `params.dispose(); params = newParams;` — subsequent uses are fine.
      AssignmentExpression(node: any) {
        if (node.left?.type !== "Identifier") return;
        const variable = resolveVariable(context, node.left);
        if (variable && disposedAt.has(variable)) {
          disposedAt.delete(variable);
        }
      },

      Identifier(node: any) {
        const variable = resolveVariable(context, node);
        if (!variable) return;
        const info = disposedAt.get(variable);
        if (info === undefined) return;
        if (node.range[0] <= info.end) return;
        if (!isIdentifierReference(node, node.parent)) return;

        const parent = node.parent;
        // Allow a second .dispose() call on an already-disposed variable
        // (idempotent no-op). But .dispose() after .consumeData() IS a bug.
        if (
          info.method === "dispose" &&
          parent?.type === "MemberExpression" &&
          parent.object === node &&
          getMemberName(parent.property) === "dispose"
        ) {
          return;
        }

        context.report({
          node,
          messageId: "useAfterDispose",
          data: {
            name: node.name,
            method: info.method,
            consumeLine: String(info.line),
          },
        });
      },
    };
  },
};

export default rule;
