import type { Rule } from "eslint";

import { getMemberName } from "../types";

function collectMakeDisposableIdentifiers(
  node: any,
  out: string[],
  seen: WeakSet<object> = new WeakSet(),
) {
  if (!node || typeof node !== "object") return;
  if (seen.has(node)) return;
  seen.add(node);

  if (node.type === "ObjectExpression") {
    for (const prop of node.properties ?? []) {
      if (prop.type === "Property") {
        // Shorthand object property: { a }
        if (prop.shorthand && prop.key?.type === "Identifier") {
          out.push(prop.key.name);
          continue;
        }

        // Direct identifier value: { x: a }
        if (prop.value?.type === "Identifier") {
          out.push(prop.value.name);
          continue;
        }

        // Recurse nested object/array literal values only.
        if (
          prop.value?.type === "ObjectExpression" ||
          prop.value?.type === "ArrayExpression"
        ) {
          collectMakeDisposableIdentifiers(prop.value, out, seen);
        }
      } else if (prop.type === "SpreadElement") {
        // Spreads can hide aliases and are dynamic; skip by design.
        continue;
      }
    }
    return;
  }

  if (node.type === "ArrayExpression") {
    for (const element of node.elements ?? []) {
      if (!element) continue;
      if (element.type === "Identifier") {
        out.push(element.name);
        continue;
      }
      if (
        element.type === "ObjectExpression" ||
        element.type === "ArrayExpression"
      ) {
        collectMakeDisposableIdentifiers(element, out, seen);
      }
    }
  }
}

function duplicateNames(names: string[]): string[] {
  const counts = new Map<string, number>();
  for (const name of names) {
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return [...counts.entries()]
    .filter(([, count]) => count > 1)
    .map(([name]) => name)
    .sort();
}

const rule: Rule.RuleModule = {
  meta: {
    type: "suggestion",
    docs: {
      description:
        "Warn on duplicate identifier references in tree.makeDisposable(...) literals",
    },
    schema: [],
    messages: {
      duplicateAlias:
        "`tree.makeDisposable(...)` reuses `{{name}}` multiple times in the same literal handoff. This creates aliased outputs; use distinct arrays or keep this aliasing intentional and explicit.",
    },
  },
  create(context) {
    return {
      CallExpression(node: any) {
        if (node.callee?.type !== "MemberExpression") return;
        if (node.callee.object?.type !== "Identifier") return;
        if (node.callee.object.name !== "tree") return;
        if (getMemberName(node.callee.property) !== "makeDisposable") return;

        const arg0 = node.arguments?.[0];
        if (
          !arg0 ||
          (arg0.type !== "ObjectExpression" && arg0.type !== "ArrayExpression")
        ) {
          return;
        }

        const names: string[] = [];
        collectMakeDisposableIdentifiers(arg0, names);
        const duplicates = duplicateNames(names);
        for (const dup of duplicates) {
          context.report({
            node: arg0,
            messageId: "duplicateAlias",
            data: { name: dup },
          });
        }
      },
    };
  },
};

export default rule;
