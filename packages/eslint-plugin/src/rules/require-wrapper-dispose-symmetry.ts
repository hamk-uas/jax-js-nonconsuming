import type { Rule } from "eslint";

function isIdentifier(node: any, name: string) {
  return node && node.type === "Identifier" && node.name === name;
}

function getDisposeTarget(node: any) {
  if (!node || node.type !== "CallExpression") return null;

  const callee = node.callee;
  if (!callee || callee.type !== "MemberExpression" || callee.computed) {
    return null;
  }
  if (!isIdentifier(callee.property, "dispose")) return null;

  const target = callee.object;
  if (
    target &&
    target.type === "MemberExpression" &&
    !target.computed &&
    target.object.type === "ThisExpression" &&
    target.property.type === "Identifier"
  ) {
    return target.property.name;
  }

  return null;
}

function collectTopLevelDisposeCalls(block: any) {
  const calls: Array<{ target: string; node: any }> = [];
  if (!block || block.type !== "BlockStatement") return calls;

  for (const stmt of block.body) {
    if (stmt.type !== "ExpressionStatement") continue;
    const target = getDisposeTarget(stmt.expression);
    if (!target) continue;
    calls.push({ target, node: stmt.expression });
  }

  return calls;
}

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require wrapper dispose symmetry: release retained state before this.inner.dispose()",
    },
    schema: [],
    messages: {
      innerNotLast:
        "In dispose(), `this.inner.dispose()` should be last; clean retained state before disposing inner.",
    },
  },

  create(context) {
    return {
      MethodDefinition(node: any) {
        if (!isIdentifier(node.key, "dispose")) return;
        if (!node.value || node.value.type !== "FunctionExpression") return;

        const body = node.value.body;
        const calls = collectTopLevelDisposeCalls(body);
        if (calls.length === 0) return;

        const innerIndices = calls
          .map((call, index) => ({ ...call, index }))
          .filter((call) => call.target === "inner");

        if (innerIndices.length === 0) return;

        const lastIndex = calls.length - 1;
        for (const inner of innerIndices) {
          if (inner.index !== lastIndex) {
            context.report({
              node: inner.node,
              messageId: "innerNotLast",
            });
          }
        }
      },
    };
  },
};

export default rule;
