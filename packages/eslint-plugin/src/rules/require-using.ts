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

function isImmediatelyReturned(decl: any, idName: string): boolean {
  const block = decl.parent?.parent;
  if (!block || block.type !== "BlockStatement") return false;
  const statements = block.body as any[];
  const idx = statements.indexOf(decl);
  if (idx < 0 || idx + 1 >= statements.length) return false;
  const next = statements[idx + 1];
  return (
    next.type === "ReturnStatement" &&
    next.argument?.type === "Identifier" &&
    next.argument.name === idName
  );
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
          if (isImmediatelyReturned(node, decl.id.name)) continue;

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
