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

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description: "Disallow using local variables after calling .dispose()",
    },
    schema: [],
    messages: {
      useAfterDispose:
        "`{{name}}` is used after `.dispose()` on line {{disposeLine}}.",
    },
  },
  create(context) {
    const disposedAt = new Map<any, { end: number; line: number }>();

    return {
      CallExpression(node: any) {
        if (node.callee?.type !== "MemberExpression") return;
        const prop = getMemberName(node.callee.property);
        if (prop !== "dispose") return;
        // tree.dispose(arg) disposes arg, not the receiver `tree`.
        // Only mark the receiver as disposed for zero-argument .dispose() calls.
        if (node.arguments.length > 0) return;
        const obj = node.callee.object;
        if (obj?.type !== "Identifier") return;

        // Ignore .dispose() calls inside callbacks / closures — they execute
        // deferred (e.g. onTestFinished(() => x.dispose())), so the variable
        // is NOT disposed at the call-site in the enclosing scope.
        if (isInsideCallback(node)) return;

        const variable = resolveVariable(context, obj);
        if (!variable) return;
        disposedAt.set(variable, {
          end: node.range[1],
          line: node.loc?.start?.line ?? 0,
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
        if (
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
            disposeLine: String(info.line),
          },
        });
      },
    };
  },
};

export default rule;
