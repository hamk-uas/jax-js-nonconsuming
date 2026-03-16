import type { Rule } from "eslint";

import { isArrayProducingCall } from "../array-detection";
import { isInsideTracedBody } from "../tracing-detection";
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
 * Suppressed inside tracing transform bodies (jit, grad, vmap, scan, etc.)
 * where intermediates are automatically managed by the JIT compiler.
 *
 * NOTE: This rule uses heuristics (method name matching) and may produce
 * false positives on non-array objects whose methods share names with
 * np.Array methods (e.g. `AluExp.mul`, `console.log`).  Best suited for
 * application code; set to `off` for framework internals.
 */

function getIndent(sourceText: string, start: number): string {
  const lineStart = sourceText.lastIndexOf("\n", start - 1) + 1;
  const prefix = sourceText.slice(lineStart, start);
  const match = prefix.match(/^\s*/);
  return match?.[0] ?? "";
}

function collectUsedNames(scope: any): Set<string> {
  const names = new Set<string>();
  let current = scope;
  while (current) {
    for (const variable of current.variables ?? []) {
      if (variable?.name) names.add(variable.name);
    }
    current = current.upper;
  }
  return names;
}

function nextTempName(used: Set<string>): string {
  let i = 1;
  while (used.has(`_jaxTmp${i}`)) i++;
  const name = `_jaxTmp${i}`;
  used.add(name);
  return name;
}

/**
 * Find the enclosing statement (VariableDeclaration or ExpressionStatement)
 * that we can prepend `using` bindings before.
 */
function findEnclosingStatement(node: any, context: Rule.RuleContext): any {
  const sourceCode = context.sourceCode ?? context.getSourceCode();
  const ancestors = sourceCode.getAncestors(node);
  for (let i = ancestors.length - 1; i >= 0; i--) {
    const a = ancestors[i];
    if (
      a.type === "VariableDeclaration" ||
      a.type === "ExpressionStatement" ||
      a.type === "ReturnStatement"
    ) {
      return a;
    }
  }
  return null;
}

const noNestedArrayLeak: Rule.RuleModule = {
  meta: {
    type: "problem",
    fixable: "code",
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
    // Track outer calls that have nested violations so we can batch-fix them.
    // Map from outer CallExpression node → list of violating argument indices.
    const outerViolations = new Map<any, { argIdx: number; arg: any }[]>();

    return {
      CallExpression(node: any) {
        if (!isArrayProducingCall(node)) return;
        if (hasAllowComment(context, node, "jax-js-lint: allow-non-using"))
          return;
        if (isInsideTracedBody(node, context)) return;

        const violations: { argIdx: number; arg: any }[] = [];

        for (let i = 0; i < node.arguments.length; i++) {
          const arg = node.arguments[i];
          if (!isArrayProducingCall(arg)) continue;
          if (hasAllowComment(context, arg, "jax-js-lint: allow-non-using"))
            continue;
          violations.push({ argIdx: i, arg });
        }

        if (violations.length === 0) return;

        // Store for batch fix
        outerViolations.set(node, violations);

        // Report each violation individually
        for (const { arg } of violations) {
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
            fix: buildBatchFix(node, violations, context),
          });
        }
      },
    };

    /**
     * Build a single fix that extracts ALL nested array-producing arguments
     * from the outer call into `using` bindings. Returned from every report
     * on the same outer node — ESLint deduplicates identical range edits.
     */
    function buildBatchFix(
      outerNode: any,
      violations: { argIdx: number; arg: any }[],
      ctx: Rule.RuleContext,
    ): Rule.ReportFixer {
      return (fixer) => {
        const sourceCode = ctx.sourceCode ?? ctx.getSourceCode();
        const sourceText = sourceCode.getText();
        const statement = findEnclosingStatement(outerNode, ctx);
        if (!statement) return null;

        const scope = sourceCode.getScope(outerNode);
        const usedNames = collectUsedNames(scope);

        const indent = getIndent(sourceText, statement.range[0]);
        const bindings: string[] = [];
        const replacements: { arg: any; name: string }[] = [];

        for (const { arg } of violations) {
          const name = nextTempName(usedNames);
          const argText = sourceText.slice(arg.range[0], arg.range[1]);
          bindings.push(`${indent}using ${name} = ${argText};`);
          replacements.push({ arg, name });
        }

        const fixes: Rule.Fix[] = [];

        // Insert bindings on the line before the statement.
        // Use insertTextBeforeRange at the start of the statement's line
        // to preserve the statement's own indentation.
        const stmtStart = statement.range[0];
        const lineStart = sourceText.lastIndexOf("\n", stmtStart - 1) + 1;
        const prefix = bindings.join("\n") + "\n";
        fixes.push(fixer.insertTextBeforeRange([lineStart, lineStart], prefix));

        // Replace each nested argument with its temp variable
        for (const { arg, name } of replacements) {
          fixes.push(fixer.replaceTextRange(arg.range, name));
        }

        return fixes;
      };
    }
  },
};

export default noNestedArrayLeak;
