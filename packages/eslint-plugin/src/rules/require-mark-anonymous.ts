import type { Rule } from "eslint";

import { hasAllowComment } from "../types";

/**
 * Flags `new Array({source: ...})` expressions in exported functions that
 * are not wrapped in `markAnonymousIfTracing()`.
 *
 * When a user-facing factory function (e.g. `arange`, `eye`, `linspace`)
 * creates an AluExp-backed array via `new Array({source: ...})`, it MUST
 * wrap the result in `markAnonymousIfTracing()`.  Otherwise, calling the
 * function inside a `jit`/`grad`/`scan` body leaks the captured constant
 * because `ClosedJaxpr.dispose()` can't perform the extra dispose.
 *
 * The rule only fires inside `export function` declarations — class
 * methods and non-exported helpers are exempt.
 *
 * Suppress with: `// jax-js-lint: allow-unmarked`
 *
 * This rule should be enabled only on `src/frontend/array.ts` via config.
 */
const requireMarkAnonymous: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require `markAnonymousIfTracing()` wrapper around `new Array({source: ...})` in exported functions",
    },
    schema: [],
    messages: {
      missingMark:
        "`new Array({source: ...})` in exported function `{{fnName}}` must be " +
        "wrapped in `markAnonymousIfTracing()` to prevent leaks when called " +
        "inside jit/grad/scan bodies. If intentionally unmarked, add a " +
        "`// jax-js-lint: allow-unmarked` comment.",
    },
  },
  create(context) {
    // Track the current exported function name (null if not in one)
    let exportedFnName: string | null = null;

    return {
      // Entering an exported function declaration
      "ExportNamedDeclaration > FunctionDeclaration"(node: any) {
        exportedFnName = node.id?.name ?? null;
      },
      "ExportNamedDeclaration > FunctionDeclaration:exit"() {
        exportedFnName = null;
      },

      NewExpression(node: any) {
        // Only check inside exported functions
        if (!exportedFnName) return;

        // Must be `new Array(...)`
        if (node.callee.type !== "Identifier" || node.callee.name !== "Array")
          return;

        // Must have an object literal argument with a `source` property
        const objArg = node.arguments[0];
        if (!objArg || objArg.type !== "ObjectExpression") return;
        const hasSource = objArg.properties.some(
          (p: any) =>
            p.type === "Property" &&
            p.key.type === "Identifier" &&
            p.key.name === "source",
        );
        if (!hasSource) return;

        // Check if already wrapped in markAnonymousIfTracing()
        const parent = node.parent;
        if (
          parent &&
          parent.type === "CallExpression" &&
          parent.callee.type === "Identifier" &&
          parent.callee.name === "markAnonymousIfTracing"
        ) {
          return;
        }

        // Check for suppression comment
        if (hasAllowComment(context, node, "jax-js-lint: allow-unmarked"))
          return;

        context.report({
          node,
          messageId: "missingMark",
          data: { fnName: exportedFnName },
        });
      },
    };
  },
};

export default requireMarkAnonymous;
