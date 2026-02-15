import type { Rule } from "eslint";

import { getMemberName, hasAllowComment } from "../types";

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow .ref in non-consuming jax-js code unless explicitly opted out",
    },
    schema: [],
    messages: {
      unnecessaryRef:
        "`.ref` is usually unnecessary in non-consuming jax-js. Use `// jax-js-lint: allow-ref` if this usage is intentional.",
    },
  },
  create(context) {
    return {
      MemberExpression(node: any) {
        const name = getMemberName(node.property);
        if (name !== "ref") return;
        if (hasAllowComment(context, node, "jax-js-lint: allow-ref")) return;
        context.report({ node: node.property, messageId: "unnecessaryRef" });
      },
    };
  },
};

export default rule;
