import type { Rule } from "eslint";

import { getMemberName } from "../types";

function usesIdentifier(
  node: any,
  idName: string,
  seen: WeakSet<object> = new WeakSet(),
): boolean {
  if (!node || typeof node !== "object") return false;
  if (seen.has(node)) return false;
  seen.add(node);
  if (node.type === "Identifier" && node.name === idName) return true;

  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = (node as any)[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (usesIdentifier(item, idName, seen)) return true;
      }
    } else if (value && typeof value === "object") {
      if (usesIdentifier(value, idName, seen)) return true;
    }
  }

  return false;
}

function hasReturnOrYieldOfIdentifier(
  node: any,
  idName: string,
  seen: WeakSet<object> = new WeakSet(),
): boolean {
  if (!node || typeof node !== "object") return false;
  if (seen.has(node)) return false;
  seen.add(node);

  if (
    node.type === "FunctionDeclaration" ||
    node.type === "FunctionExpression" ||
    node.type === "ArrowFunctionExpression" ||
    node.type === "ClassDeclaration" ||
    node.type === "ClassExpression"
  ) {
    return false;
  }

  if (
    node.type === "ReturnStatement" &&
    usesIdentifier(node.argument, idName)
  ) {
    return true;
  }

  if (
    node.type === "YieldExpression" &&
    usesIdentifier(node.argument, idName)
  ) {
    return true;
  }

  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = (node as any)[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (hasReturnOrYieldOfIdentifier(item, idName, seen)) return true;
      }
    } else if (value && typeof value === "object") {
      if (hasReturnOrYieldOfIdentifier(value, idName, seen)) return true;
    }
  }

  return false;
}

function hasDisposeCallForIdentifier(
  node: any,
  idName: string,
  seen: WeakSet<object> = new WeakSet(),
): boolean {
  if (!node || typeof node !== "object") return false;
  if (seen.has(node)) return false;
  seen.add(node);

  if (
    node.type === "CallExpression" &&
    node.callee?.type === "MemberExpression"
  ) {
    const prop = getMemberName(node.callee.property);
    if (prop === "dispose") {
      const calleeObj = node.callee.object;

      if (calleeObj?.type === "Identifier" && calleeObj.name === idName) {
        return true;
      }

      if (calleeObj?.type === "Identifier" && calleeObj.name === "tree") {
        const arg0 = node.arguments?.[0];
        if (usesIdentifier(arg0, idName)) return true;
      }
    }
  }

  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = (node as any)[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (hasDisposeCallForIdentifier(item, idName, seen)) return true;
      }
    } else if (value && typeof value === "object") {
      if (hasDisposeCallForIdentifier(value, idName, seen)) return true;
    }
  }

  return false;
}

function hasMakeDisposableHandoffForIdentifier(
  node: any,
  idName: string,
  seen: WeakSet<object> = new WeakSet(),
): boolean {
  if (!node || typeof node !== "object") return false;
  if (seen.has(node)) return false;
  seen.add(node);

  if (
    node.type === "CallExpression" &&
    node.callee?.type === "MemberExpression" &&
    node.callee.object?.type === "Identifier" &&
    node.callee.object.name === "tree" &&
    getMemberName(node.callee.property) === "makeDisposable"
  ) {
    return node.arguments.some((arg: any) => usesIdentifier(arg, idName));
  }

  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = (node as any)[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (hasMakeDisposableHandoffForIdentifier(item, idName, seen)) {
          return true;
        }
      }
    } else if (value && typeof value === "object") {
      if (hasMakeDisposableHandoffForIdentifier(value, idName, seen)) {
        return true;
      }
    }
  }

  return false;
}

function isLaxScanCall(node: any): boolean {
  if (!node || node.type !== "CallExpression") return false;
  if (node.callee?.type !== "MemberExpression") return false;
  if (getMemberName(node.callee.property) !== "scan") return false;
  return (
    node.callee.object?.type === "Identifier" &&
    node.callee.object.name === "lax"
  );
}

function collectTopLevelArrayPatternIds(pattern: any): string[] {
  if (!pattern || pattern.type !== "ArrayPattern") return [];
  const ids: string[] = [];
  for (const element of pattern.elements ?? []) {
    if (!element) continue;
    if (element.type === "Identifier") ids.push(element.name);
    if (
      element.type === "AssignmentPattern" &&
      element.left?.type === "Identifier"
    ) {
      ids.push(element.left.name);
    }
  }
  return ids;
}

function isIgnoredBindingName(name: string): boolean {
  return name.startsWith("_");
}

/**
 * Detect `using _alias = scanOutput;` or `using _alias = scanOutput.member;`
 * pattern — a valid ownership handoff via `using` declaration.
 */
function hasUsingAliasOfIdentifier(node: any, idName: string): boolean {
  if (node?.type !== "VariableDeclaration" || node.kind !== "using") {
    return false;
  }
  for (const decl of node.declarations as any[]) {
    if (!decl.init) continue;
    if (usesIdentifier(decl.init, idName)) return true;
  }
  return false;
}

function hasReturnOrDisposeAfterDeclaration(
  declStmt: any,
  idName: string,
): boolean {
  const block = declStmt.parent;
  if (!block || block.type !== "BlockStatement") return false;
  const statements = block.body as any[];
  const idx = statements.indexOf(declStmt);
  if (idx < 0) return false;

  for (let i = idx + 1; i < statements.length; i++) {
    if (hasReturnOrYieldOfIdentifier(statements[i], idName)) return true;
    if (hasDisposeCallForIdentifier(statements[i], idName)) return true;
    if (hasMakeDisposableHandoffForIdentifier(statements[i], idName)) {
      return true;
    }
    if (hasUsingAliasOfIdentifier(statements[i], idName)) return true;
  }

  return false;
}

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require disposing or returning destructured lax.scan results to avoid leaked pytree array leaves",
    },
    schema: [],
    messages: {
      scanResultNotDisposed:
        "`{{name}}` comes from `lax.scan(...)` and should be disposed (for example `tree.dispose({{name}})`) or returned to caller-owned scope.",
    },
  },
  create(context) {
    return {
      VariableDeclaration(node: any) {
        for (const decl of node.declarations as any[]) {
          if (!isLaxScanCall(decl.init)) continue;
          const ids = collectTopLevelArrayPatternIds(decl.id);
          if (ids.length === 0) continue;

          for (const idName of ids) {
            if (isIgnoredBindingName(idName)) continue;
            if (hasReturnOrDisposeAfterDeclaration(node, idName)) continue;

            context.report({
              node: decl.id,
              messageId: "scanResultNotDisposed",
              data: { name: idName },
            });
          }
        }
      },
    };
  },
};

export default rule;
