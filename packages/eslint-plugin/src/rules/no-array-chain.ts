import type { Rule } from "eslint";

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
