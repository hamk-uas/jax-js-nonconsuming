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
]);

const ARRAY_METHODS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "pow",
  "reshape",
  "transpose",
  "sum",
  "mean",
  "astype",
  "slice",
  "neg",
  "exp",
  "log",
  "sin",
  "cos",
  "tanh",
  "sqrt",
  "maximum",
  "minimum",
  "dot",
  "matmul",
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

    if (
      callee.object.type === "Identifier" &&
      callee.object.name === "np" &&
      ARRAY_FACTORIES.has(property)
    ) {
      return true;
    }

    return ARRAY_METHODS.has(property);
  }

  return false;
}

function isReturnedAfterDeclaration(declStmt: any, idName: string): boolean {
  const block = declStmt.parent;
  if (!block || block.type !== "BlockStatement") return false;
  const statements = block.body as any[];
  const idx = statements.indexOf(declStmt);
  if (idx < 0) return false;

  for (let i = idx + 1; i < statements.length; i++) {
    const stmt = statements[i];
    if (
      stmt.type === "ReturnStatement" &&
      stmt.argument?.type === "Identifier" &&
      stmt.argument.name === idName
    ) {
      return true;
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
    const stmt = statements[i];
    if (stmt.type !== "ExpressionStatement") continue;
    const expr = stmt.expression;
    if (expr?.type !== "CallExpression") continue;
    if (expr.callee?.type !== "MemberExpression") continue;
    if (getMemberName(expr.callee.property) !== "dispose") continue;
    const obj = expr.callee.object;
    if (obj?.type === "Identifier" && obj.name === idName) return true;
  }

  return false;
}

function usesIdentifier(node: any, idName: string): boolean {
  if (!node || typeof node !== "object") return false;
  if (node.type === "Identifier" && node.name === idName) return true;

  for (const key of Object.keys(node)) {
    const value = (node as any)[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (usesIdentifier(item, idName)) return true;
      }
    } else if (value && typeof value === "object") {
      if (usesIdentifier(value, idName)) return true;
    }
  }

  return false;
}

function hasPersistedUseAfterDeclaration(
  declStmt: any,
  idName: string,
): boolean {
  const block = declStmt.parent;
  if (!block || block.type !== "BlockStatement") return false;
  const statements = block.body as any[];
  const idx = statements.indexOf(declStmt);
  if (idx < 0) return false;

  for (let i = idx + 1; i < statements.length; i++) {
    const stmt = statements[i];

    if (stmt.type === "ExpressionStatement") {
      const expr = stmt.expression;

      if (
        expr?.type === "AssignmentExpression" &&
        usesIdentifier(expr.right, idName) &&
        expr.left?.type !== "Identifier"
      ) {
        return true;
      }

      if (
        expr?.type === "CallExpression" &&
        expr.callee?.type === "MemberExpression"
      ) {
        const name = getMemberName(expr.callee.property);
        if (
          name &&
          ["set", "push", "unshift", "add"].includes(name) &&
          expr.arguments.some((arg: any) => usesIdentifier(arg, idName))
        ) {
          return true;
        }
      }
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
        "Local array binding `{{name}}` should use `using` (or be explicitly exempted).",
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
          if (isReturnedAfterDeclaration(node, decl.id.name)) continue;
          if (hasExplicitDisposeAfterDeclaration(node, decl.id.name)) continue;
          if (hasPersistedUseAfterDeclaration(node, decl.id.name)) continue;

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
