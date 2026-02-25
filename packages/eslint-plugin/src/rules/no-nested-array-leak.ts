import type { Rule } from "eslint";

import { isArrayProducingCall } from "../array-detection";
import { getMemberName, hasAllowComment } from "../types";

/**
 * Flags array-producing calls nested as arguments to other array-producing
 * calls.  The intermediate array is never bound to a variable, so it can
 * never be disposed — a silent GPU memory leak.
 *
 * Example:
 *   np.tile(np.reshape(G, [1, m, m]), [n, 1, 1]);
 *          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ← leaked intermediate
 *
 * Fix: extract to a `using` binding:
 *   using G_3d = np.reshape(G, [1, m, m]);
 *   np.tile(G_3d, [n, 1, 1]);
 *
 * NOTE: This rule uses heuristics (method name matching) and may produce
 * false positives on non-array objects whose methods share names with
 * np.Array methods (e.g. `AluExp.mul`, `console.log`).  Best suited for
 * application code; set to `off` for framework internals.
 */
const noNestedArrayLeak: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow array-producing calls nested as arguments — the intermediate is never disposed",
    },
    schema: [],
    messages: {
      nestedArrayLeak:
        "Array-producing call `{{callee}}` is passed as an argument without binding — " +
        "the intermediate array is never disposed. Extract to a `using` variable.",
    },
  },
  create(context) {
    return {
      CallExpression(node: any) {
        if (!isArrayProducingCall(node)) return;
        if (hasAllowComment(context, node, "jax-js-lint: allow-non-using"))
          return;

        for (const arg of node.arguments as any[]) {
          if (!isArrayProducingCall(arg)) continue;
          if (hasAllowComment(context, arg, "jax-js-lint: allow-non-using"))
            continue;

          const callee = arg.callee;
          let calleeName = "unknown";
          if (callee.type === "Identifier") {
            calleeName = callee.name;
          } else if (callee.type === "MemberExpression") {
            const prop = getMemberName(callee.property);
            const obj =
              callee.object?.type === "Identifier" ? callee.object.name : null;
            calleeName = obj && prop ? `${obj}.${prop}` : (prop ?? "unknown");
          }

          context.report({
            node: arg,
            messageId: "nestedArrayLeak",
            data: { callee: calleeName },
          });
        }
      },
    };
  },
};

export default noNestedArrayLeak;
