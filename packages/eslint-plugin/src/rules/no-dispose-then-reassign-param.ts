import type { Rule } from "eslint";

import { getMemberName } from "../types";

function collectParamNames(param: any, out: Set<string>) {
  if (!param) return;
  if (param.type === "Identifier") {
    out.add(param.name);
    return;
  }
  if (param.type === "AssignmentPattern") {
    collectParamNames(param.left, out);
    return;
  }
  if (param.type === "RestElement") {
    collectParamNames(param.argument, out);
    return;
  }
  if (param.type === "ObjectPattern") {
    for (const prop of param.properties ?? []) {
      if (prop.type === "Property") collectParamNames(prop.value, out);
      if (prop.type === "RestElement") collectParamNames(prop.argument, out);
    }
    return;
  }
  if (param.type === "ArrayPattern") {
    for (const el of param.elements ?? []) {
      collectParamNames(el, out);
    }
  }
}

function getDisposeTargetName(stmt: any): string | null {
  if (stmt?.type !== "ExpressionStatement") return null;
  const expr = stmt.expression;
  if (expr?.type !== "CallExpression") return null;

  const callee = expr.callee;
  if (callee?.type !== "MemberExpression") return null;
  const prop = getMemberName(callee.property);
  if (prop !== "dispose") return null;

  if (callee.object?.type === "Identifier" && callee.object.name === "tree") {
    const arg0 = expr.arguments?.[0];
    if (arg0?.type === "Identifier") return arg0.name;
  }

  if (callee.object?.type === "Identifier") {
    return callee.object.name;
  }

  return null;
}

function getSimpleAssignment(
  stmt: any,
): { left: string; right: string } | null {
  if (stmt?.type !== "ExpressionStatement") return null;
  const expr = stmt.expression;
  if (expr?.type !== "AssignmentExpression") return null;
  if (expr.operator !== "=") return null;
  if (expr.left?.type !== "Identifier") return null;
  if (expr.right?.type !== "Identifier") return null;
  return { left: expr.left.name, right: expr.right.name };
}

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow disposing a state variable immediately before assigning it from a callback parameter",
    },
    schema: [],
    messages: {
      disposeThenReassignParam:
        "`{{stateName}}` is disposed then immediately assigned from parameter `{{paramName}}`. If they alias, this can dispose live data. Guard with `if ({{stateName}} !== {{paramName}})`.",
    },
  },
  create(context) {
    function checkFunctionLike(node: any) {
      if (node.body?.type !== "BlockStatement") return;
      const paramNames = new Set<string>();
      for (const p of node.params ?? []) collectParamNames(p, paramNames);
      if (paramNames.size === 0) return;

      const statements = node.body.body ?? [];
      for (let i = 0; i + 1 < statements.length; i++) {
        const disposeTarget = getDisposeTargetName(statements[i]);
        if (!disposeTarget) continue;
        const assignment = getSimpleAssignment(statements[i + 1]);
        if (!assignment) continue;
        if (assignment.left !== disposeTarget) continue;
        if (!paramNames.has(assignment.right)) continue;

        context.report({
          node: statements[i + 1],
          messageId: "disposeThenReassignParam",
          data: {
            stateName: assignment.left,
            paramName: assignment.right,
          },
        });
      }
    }

    return {
      FunctionDeclaration: checkFunctionLike,
      FunctionExpression: checkFunctionLike,
      ArrowFunctionExpression: checkFunctionLike,
    };
  },
};

export default rule;
