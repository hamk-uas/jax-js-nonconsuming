import type { Rule } from "eslint";

export type RuleModule = Rule.RuleModule;

/**
 * Check whether a suppression comment (e.g. `// jax-js-lint: allow-ref`)
 * appears on the line immediately before `node`.
 *
 * ESLint's `getCommentsBefore(node)` only returns comments between the
 * previous sibling *token* and `node`.  For a deeply-nested expression
 * (e.g. a MemberExpression inside a VariableDeclarator), a comment placed
 * before the enclosing statement is invisible.  We therefore walk up the
 * parent chain, trying `getCommentsBefore` at each level until we reach a
 * statement boundary.
 */
export function hasAllowComment(
  context: Rule.RuleContext,
  node: Rule.Node,
  marker: string,
): boolean {
  const source = context.sourceCode;
  const nodeLine = node.loc?.start.line;
  if (nodeLine === undefined) return false;

  let current: any = node;
  while (current) {
    const comments = source.getCommentsBefore(current);
    if (comments.length > 0) {
      const last = comments[comments.length - 1];
      const lastLine = last.loc?.end.line;
      if (
        lastLine !== undefined &&
        lastLine >= nodeLine - 1 &&
        last.value.includes(marker)
      ) {
        return true;
      }
    }
    // Stop once we've checked a statement-level node — the comment
    // must be on the line immediately before the flagged code.
    const t = current.type;
    if (
      t === "VariableDeclaration" ||
      t === "ExpressionStatement" ||
      t === "ReturnStatement" ||
      t === "ThrowStatement" ||
      t === "Program"
    ) {
      break;
    }
    current = current.parent;
  }
  return false;
}

export function getMemberName(node: any): string | null {
  if (!node) return null;
  if (node.type === "Identifier") return node.name;
  if (node.type === "Literal" && typeof node.value === "string") {
    return node.value;
  }
  return null;
}
