import type { Rule } from "eslint";

import { isArrayProducingCall } from "../array-detection";
import {
  findEnclosingFunction,
  getBorrowedParamNames,
  unwrapBorrowedReturnExpr,
} from "../borrowed-param-return";
import { getMemberName } from "../types";

function isZeroLiteral(node: any): boolean {
  if (!node) return false;
  if (node.type === "Literal") {
    return typeof node.value === "number" && Object.is(node.value, 0);
  }
  if (node.type === "UnaryExpression" && node.operator === "-") {
    return isZeroLiteral(node.argument);
  }
  return false;
}

function checkReturnedExpr(
  node: any,
  paramNames: ReadonlySet<string>,
  report: (
    node: any,
    messageId: "borrowedParamReturn" | "borrowedParamWorkaround",
    name: string,
  ) => void,
): void {
  node = unwrapBorrowedReturnExpr(node);
  if (!node) return;

  if (node.type === "Identifier" && paramNames.has(node.name)) {
    report(node, "borrowedParamReturn", node.name);
    return;
  }

  if (
    node.type === "MemberExpression" &&
    getMemberName(node.property) === "ref" &&
    node.object?.type === "Identifier" &&
    paramNames.has(node.object.name)
  ) {
    return;
  }

  if (
    node.type === "CallExpression" &&
    node.callee?.type === "MemberExpression" &&
    getMemberName(node.callee.property) === "add" &&
    node.callee.object?.type === "Identifier" &&
    paramNames.has(node.callee.object.name) &&
    node.arguments.length === 1 &&
    isZeroLiteral(node.arguments[0])
  ) {
    report(node, "borrowedParamWorkaround", node.callee.object.name);
    return;
  }

  if (node.type === "ObjectExpression") {
    for (const prop of node.properties) {
      if (prop.type === "Property") {
        checkReturnedExpr(prop.value, paramNames, report);
      } else if (prop.type === "SpreadElement") {
        checkReturnedExpr(prop.argument, paramNames, report);
      }
    }
    return;
  }

  if (node.type === "ArrayExpression") {
    for (const elt of node.elements) {
      if (elt) checkReturnedExpr(elt, paramNames, report);
    }
    return;
  }

  if (node.type === "ConditionalExpression") {
    checkReturnedExpr(node.consequent, paramNames, report);
    checkReturnedExpr(node.alternate, paramNames, report);
    return;
  }

  if (node.type === "CallExpression" || node.type === "NewExpression") {
    if (node.type === "CallExpression" && isArrayProducingCall(node)) {
      return;
    }
    for (const arg of node.arguments ?? []) {
      checkReturnedExpr(arg, paramNames, report);
    }
    return;
  }

  if (node.type === "AwaitExpression") {
    checkReturnedExpr(node.argument, paramNames, report);
    return;
  }

  if (node.type === "LogicalExpression") {
    checkReturnedExpr(node.left, paramNames, report);
    checkReturnedExpr(node.right, paramNames, report);
    return;
  }

  if (node.type === "SequenceExpression") {
    for (const expr of node.expressions)
      checkReturnedExpr(expr, paramNames, report);
  }
}

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow returning borrowed np.Array or Tracer parameters without retaining via .ref",
    },
    schema: [],
    messages: {
      borrowedParamReturn:
        "Borrowed parameter `{{name}}` escapes via return without `.ref`. Return `{{name}}.ref` so the callee retains independent ownership.",
      borrowedParamWorkaround:
        "`{{name}}.add(0)` in a return path copies only to retain ownership. Return `{{name}}.ref` instead.",
    },
  },
  create(context) {
    function checkFunctionExpressionBody(node: any) {
      if (node.body?.type === "BlockStatement") return;
      const paramNames = getBorrowedParamNames(node);
      if (paramNames.size === 0) return;
      checkReturnedExpr(node.body, paramNames, (expr, messageId, name) => {
        context.report({ node: expr, messageId, data: { name } });
      });
    }

    return {
      ReturnStatement(node: any) {
        const fnNode = findEnclosingFunction(node);
        if (!fnNode || !node.argument) return;
        const paramNames = getBorrowedParamNames(fnNode);
        if (paramNames.size === 0) return;
        checkReturnedExpr(
          node.argument,
          paramNames,
          (expr, messageId, name) => {
            context.report({ node: expr, messageId, data: { name } });
          },
        );
      },
      ArrowFunctionExpression: checkFunctionExpressionBody,
    };
  },
};

export default rule;
