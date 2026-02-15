import type { Rule } from "eslint";

import { getMemberName } from "../types";

const CHAINABLE_ARRAY_METHODS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "pow",
  "mod",
  "neg",
  "reshape",
  "transpose",
  "sum",
  "mean",
  "astype",
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
  "less",
  "lessEqual",
  "greater",
  "greaterEqual",
  "equal",
  "notEqual",
]);

const rule: Rule.RuleModule = {
  meta: {
    type: "suggestion",
    docs: {
      description:
        "Disallow deep array method chains that create unnamed eager intermediates",
    },
    schema: [
      {
        type: "object",
        properties: {
          minDepth: { type: "integer", minimum: 2 },
        },
        additionalProperties: false,
      },
    ],
    messages: {
      noArrayChain:
        "Array call chain depth {{depth}} creates unnamed eager temporaries. Split into `using` bindings.",
    },
  },
  create(context) {
    const minDepth = (context.options[0] as any)?.minDepth ?? 2;

    function depthOfCallChain(node: any): number {
      let depth = 0;
      let current: any = node;
      while (
        current?.type === "CallExpression" &&
        current.callee?.type === "MemberExpression"
      ) {
        const method = getMemberName(current.callee.property);
        if (!method || !CHAINABLE_ARRAY_METHODS.has(method)) break;
        depth++;
        current = current.callee.object;
      }
      return depth;
    }

    return {
      CallExpression(node: any) {
        if (node.callee?.type !== "MemberExpression") return;
        const depth = depthOfCallChain(node);
        if (depth < minDepth) return;

        // Only report the outermost chain — skip if a parent call is also
        // a qualifying chain (avoids duplicate depth-3 + depth-2 reports).
        const parentCall = node.parent;
        if (
          parentCall?.type === "MemberExpression" &&
          parentCall.parent?.type === "CallExpression"
        ) {
          const parentDepth = depthOfCallChain(parentCall.parent);
          if (parentDepth >= minDepth) return;
        }

        context.report({
          node,
          messageId: "noArrayChain",
          data: { depth: String(depth) },
        });
      },
    };
  },
};

export default rule;
