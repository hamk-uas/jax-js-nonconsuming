import type { Rule } from "eslint";

import { getMemberName } from "../types";

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

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description: "Disallow using local variables after calling .dispose()",
    },
    schema: [],
    messages: {
      useAfterDispose: "`{{name}}` is used after `.dispose()`.",
    },
  },
  create(context) {
    const disposedAt = new Map<string, number>();

    return {
      CallExpression(node: any) {
        if (node.callee?.type !== "MemberExpression") return;
        const prop = getMemberName(node.callee.property);
        if (prop !== "dispose") return;
        const obj = node.callee.object;
        if (obj?.type !== "Identifier") return;
        disposedAt.set(obj.name, node.range[1]);
      },
      Identifier(node: any) {
        const end = disposedAt.get(node.name);
        if (end === undefined) return;
        if (node.range[0] <= end) return;
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
          data: { name: node.name },
        });
      },
    };
  },
};

export default rule;
