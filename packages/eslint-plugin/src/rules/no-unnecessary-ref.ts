import type { Rule } from "eslint";

import { getMemberName, hasAllowComment } from "../types";

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow .ref in non-consuming jax-js code unless explicitly opted out",
    },
    fixable: "code",
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
        // Skip `.ref` in update expressions (e.g. `buffer.ref++`, `buffer.ref--`)
        // and comparisons (e.g. `buffer.ref === 0`).
        // These are plain numeric property accesses, not the Array `.ref` accessor.
        const parentType = node.parent?.type;
        if (
          parentType === "UpdateExpression" ||
          parentType === "BinaryExpression"
        )
          return;
        if (hasAllowComment(context, node, "jax-js-lint: allow-ref")) return;
        context.report({
          node: node.property,
          messageId: "unnecessaryRef",
          fix(fixer) {
            // Remove `.ref` from the chain: `x.ref` → `x`, `x.ref.add(1)` → `x.add(1)`
            // Range: from the dot before `ref` to the end of `ref`
            const sourceCode = context.sourceCode ?? context.getSourceCode();
            const dotToken = sourceCode.getTokenBefore(node.property);
            if (!dotToken || dotToken.value !== ".") return null;
            return fixer.removeRange([
              dotToken.range[0],
              node.property.range[1],
            ]);
          },
        });
      },
    };
  },
};

export default rule;
