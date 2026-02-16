import type { Rule } from "eslint";

const TS_WRAPPERS = new Set([
  "TSAsExpression",
  "TSNonNullExpression",
  "TSTypeAssertion",
  "TSInstantiationExpression",
  "ParenthesizedExpression",
]);

function isIdentifier(node: any, name: string) {
  return node && node.type === "Identifier" && node.name === name;
}

function unwrap(node: any) {
  let current = node;
  while (current && TS_WRAPPERS.has(current.type)) {
    current = current.expression;
  }
  return current;
}

function walk(node: any, fn: (node: any) => void) {
  if (!node || typeof node !== "object") return;
  fn(node);
  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = node[key];
    if (Array.isArray(value)) {
      for (const child of value) {
        if (child && typeof child.type === "string") walk(child, fn);
      }
    } else if (value && typeof value.type === "string") {
      walk(value, fn);
    }
  }
}

function retainedFromRef(declarator: any) {
  if (!declarator || declarator.type !== "VariableDeclarator") return null;
  if (!declarator.init || declarator.id.type !== "Identifier") return null;

  const init = unwrap(declarator.init);
  if (!init || init.type !== "MemberExpression") return null;
  if (init.computed) return null;
  if (!isIdentifier(init.property, "ref")) return null;

  return { name: declarator.id.name, idNode: declarator.id };
}

function retainedFromRefAssignment(node: any) {
  if (!node || node.type !== "AssignmentExpression") return null;
  if (!node.right || node.left.type !== "Identifier") return null;

  const right = unwrap(node.right);
  if (!right || right.type !== "MemberExpression" || right.computed) {
    return null;
  }
  if (!isIdentifier(right.property, "ref")) return null;

  return { name: node.left.name, idNode: node.left };
}

function collectRetainedInTryBlock(blockNode: any) {
  const retained = new Map<string, { name: string; idNode: any }>();
  walk(blockNode, (node) => {
    const found = retainedFromRef(node);
    if (found) retained.set(found.name, found);

    const assigned = retainedFromRefAssignment(node);
    if (assigned) retained.set(assigned.name, assigned);
  });
  return [...retained.values()];
}

function hasDisposeCall(node: any, varName: string) {
  let found = false;
  walk(node, (candidate) => {
    if (found || candidate.type !== "CallExpression") return;
    const callee = unwrap(candidate.callee);
    if (!callee || callee.type !== "MemberExpression" || callee.computed)
      return;
    if (!isIdentifier(callee.object, varName)) return;
    if (!isIdentifier(callee.property, "dispose")) return;
    found = true;
  });
  return found;
}

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require `.ref` temporaries created in try blocks to be released in finally",
    },
    schema: [],
    messages: {
      missingFinally:
        "Retained handle `{{name}}` is created inside try without a finally block. Add finally cleanup for symmetric release.",
      missingFinallyDispose:
        "Retained handle `{{name}}` is created inside try but not disposed in finally. Move cleanup to finally for error-path parity.",
    },
  },

  create(context) {
    return {
      TryStatement(node: any) {
        const retained = collectRetainedInTryBlock(node.block);
        if (retained.length === 0) return;

        if (!node.finalizer) {
          for (const ret of retained) {
            context.report({
              node: ret.idNode,
              messageId: "missingFinally",
              data: { name: ret.name },
            });
          }
          return;
        }

        for (const ret of retained) {
          if (!hasDisposeCall(node.finalizer, ret.name)) {
            context.report({
              node: ret.idNode,
              messageId: "missingFinallyDispose",
              data: { name: ret.name },
            });
          }
        }
      },
    };
  },
};

export default rule;
