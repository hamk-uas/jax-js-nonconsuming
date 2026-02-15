import type { Rule } from "eslint";

export type RuleModule = Rule.RuleModule;

export function hasAllowComment(
  context: Rule.RuleContext,
  node: Rule.Node,
  marker: string,
): boolean {
  const source = context.sourceCode;
  const comments = source.getCommentsBefore(node);
  if (comments.length === 0) return false;
  const last = comments[comments.length - 1];
  const lastLine = last.loc?.end.line;
  const nodeLine = node.loc?.start.line;
  if (lastLine === undefined || nodeLine === undefined) return false;
  return lastLine >= nodeLine - 1 && last.value.includes(marker);
}

export function getMemberName(node: any): string | null {
  if (!node) return null;
  if (node.type === "Identifier") return node.name;
  if (node.type === "Literal" && typeof node.value === "string") {
    return node.value;
  }
  return null;
}
