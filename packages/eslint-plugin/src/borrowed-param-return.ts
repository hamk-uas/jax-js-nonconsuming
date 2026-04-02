import { getMemberName } from "./types";

function getQualifiedTypeName(node: any): string | null {
  if (!node) return null;
  if (node.type === "Identifier") return node.name;
  if (node.type === "TSQualifiedName") {
    const left = getQualifiedTypeName(node.left);
    const right = getQualifiedTypeName(node.right);
    if (!left || !right) return null;
    return `${left}.${right}`;
  }
  return null;
}

function isBorrowedParamType(node: any): boolean {
  if (!node) return false;
  switch (node.type) {
    case "TSTypeReference": {
      const typeName = getQualifiedTypeName(node.typeName);
      return typeName === "np.Array" || typeName === "Tracer";
    }
    case "TSUnionType":
      return node.types.some((t: any) => isBorrowedParamType(t));
    case "TSParenthesizedType":
      return isBorrowedParamType(node.typeAnnotation);
    default:
      return false;
  }
}

function getParamIdentifier(param: any): any | null {
  if (!param) return null;
  if (param.type === "Identifier") return param;
  if (param.type === "AssignmentPattern" && param.left?.type === "Identifier") {
    return param.left;
  }
  return null;
}

export function findEnclosingFunction(node: any): any | null {
  let current = node?.parent;
  while (current) {
    if (
      current.type === "FunctionDeclaration" ||
      current.type === "FunctionExpression" ||
      current.type === "ArrowFunctionExpression"
    ) {
      return current;
    }
    current = current.parent;
  }
  return null;
}

export function getBorrowedParamNames(fnNode: any): Set<string> {
  const names = new Set<string>();
  for (const param of fnNode?.params ?? []) {
    const ident = getParamIdentifier(param);
    if (!ident) continue;
    if (isBorrowedParamType(ident.typeAnnotation?.typeAnnotation)) {
      names.add(ident.name);
    }
  }
  return names;
}

export function unwrapBorrowedReturnExpr(node: any): any {
  let current = node;
  while (current) {
    if (
      current.type === "ParenthesizedExpression" ||
      current.type === "TSAsExpression" ||
      current.type === "TSTypeAssertion" ||
      current.type === "TSNonNullExpression" ||
      current.type === "ChainExpression"
    ) {
      current = current.expression;
      continue;
    }
    return current;
  }
  return current;
}

export function isReturnedValueContext(node: any): boolean {
  let current = node;
  while (current?.parent) {
    const parent = current.parent;
    if (parent.type === "ReturnStatement" && parent.argument === current) {
      return true;
    }
    if (
      parent.type === "ArrowFunctionExpression" &&
      parent.body === current &&
      parent.body?.type !== "BlockStatement"
    ) {
      return true;
    }
    if (parent.type === "Property" && parent.value === current) {
      current = parent;
      continue;
    }
    if (
      parent.type === "ObjectExpression" &&
      parent.properties.includes(current)
    ) {
      current = parent;
      continue;
    }
    if (
      parent.type === "ArrayExpression" &&
      parent.elements.includes(current)
    ) {
      current = parent;
      continue;
    }
    if (parent.type === "SpreadElement" && parent.argument === current) {
      current = parent;
      continue;
    }
    if (
      parent.type === "CallExpression" &&
      parent.arguments.includes(current)
    ) {
      current = parent;
      continue;
    }
    if (
      parent.type === "NewExpression" &&
      parent.arguments?.includes(current)
    ) {
      current = parent;
      continue;
    }
    if (
      parent.type === "ConditionalExpression" &&
      (parent.consequent === current || parent.alternate === current)
    ) {
      current = parent;
      continue;
    }
    if (
      parent.type === "LogicalExpression" &&
      (parent.left === current || parent.right === current)
    ) {
      current = parent;
      continue;
    }
    if (
      parent.type === "SequenceExpression" &&
      parent.expressions.includes(current)
    ) {
      current = parent;
      continue;
    }
    if (parent.type === "AwaitExpression" && parent.argument === current) {
      current = parent;
      continue;
    }
    return false;
  }
  return false;
}

export function isRequiredBorrowedParamReturnRef(node: any): boolean {
  if (!node || node.type !== "MemberExpression") return false;
  const name = getMemberName(node.property);
  if (name !== "ref") return false;
  if (node.object?.type !== "Identifier") return false;
  const fnNode = findEnclosingFunction(node);
  if (!fnNode) return false;
  const paramNames = getBorrowedParamNames(fnNode);
  return paramNames.has(node.object.name) && isReturnedValueContext(node);
}
