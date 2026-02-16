import type { Rule } from "eslint";

function isIdentifier(node: any, name: string) {
  return node && node.type === "Identifier" && node.name === name;
}

function isDisposeCallOnIdentifier(refIdentifier: any, varName: string) {
  const member = refIdentifier.parent;
  if (!member || member.type !== "MemberExpression") return false;
  if (member.computed || !isIdentifier(member.object, varName)) return false;
  if (!isIdentifier(member.property, "dispose")) return false;

  const call = member.parent;
  return !!(call && call.type === "CallExpression" && call.callee === member);
}

function isTransferredUsage(refIdentifier: any) {
  const parent = refIdentifier.parent;
  if (!parent) return false;

  if (parent.type === "ReturnStatement" && parent.argument === refIdentifier) {
    return true;
  }
  if (parent.type === "YieldExpression" && parent.argument === refIdentifier) {
    return true;
  }
  if (
    parent.type === "AssignmentExpression" &&
    parent.right === refIdentifier
  ) {
    return true;
  }
  if (parent.type === "Property" && parent.value === refIdentifier) {
    return true;
  }
  if (
    parent.type === "CallExpression" &&
    parent.arguments.includes(refIdentifier)
  ) {
    return true;
  }
  if (parent.type === "ArrayExpression" || parent.type === "ObjectExpression") {
    return true;
  }

  return false;
}

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require retained `.ref` handles to have an explicit release or transfer path",
    },
    hasSuggestions: true,
    schema: [],
    messages: {
      missingRelease:
        "Retained handle `{{name}}` from `.ref` has no explicit release path in this scope.",
      suggestDispose:
        "Add `{{name}}.dispose()` at end of scope (if ownership stays local)",
    },
  },

  create(context) {
    const sourceCode = context.sourceCode;

    return {
      VariableDeclarator(node: any) {
        if (!node.init || node.id.type !== "Identifier") return;
        const varName = node.id.name;

        if (
          node.init.type !== "MemberExpression" ||
          node.init.computed ||
          !isIdentifier(node.init.property, "ref")
        ) {
          return;
        }

        const scope = sourceCode.getScope(node);
        const variable = scope.set.get(varName);
        if (!variable) return;

        const refs = variable.references.filter(
          (ref: any) => ref.identifier !== node.id,
        );

        let hasTerminal = false;
        for (const ref of refs) {
          const id = ref.identifier;
          if (
            isDisposeCallOnIdentifier(id, varName) ||
            isTransferredUsage(id)
          ) {
            hasTerminal = true;
            break;
          }
        }

        if (!hasTerminal) {
          context.report({
            node: node.id,
            messageId: "missingRelease",
            data: { name: varName },
            suggest: [
              {
                messageId: "suggestDispose",
                data: { name: varName },
                fix: (fixer) => {
                  const statement =
                    node.parent && node.parent.type === "VariableDeclaration"
                      ? node.parent
                      : node;
                  return fixer.insertTextAfter(
                    statement,
                    `\n${varName}.dispose();`,
                  );
                },
              },
            ],
          });
        }
      },
    };
  },
};

export default rule;
