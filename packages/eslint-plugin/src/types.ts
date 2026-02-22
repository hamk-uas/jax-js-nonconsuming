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

  // Check file-level leading comments (top-of-file directive).
  const program = source.ast;
  const progComments = source.getCommentsBefore(program.body[0] ?? program);
  for (const c of progComments) {
    if (c.value.includes(marker)) return true;
  }

  // Check inline/trailing comments on the same line as the node.
  // getCommentsBefore() only returns comments between the previous token
  // and the node, so it misses trailing `// jax-js-lint: allow-ref` on
  // the same line.  Fall back to raw source-text scanning.
  const sourceText = source.getText();
  const lines = sourceText.split("\n");
  if (nodeLine <= lines.length) {
    const lineText = lines[nodeLine - 1];
    const slashIdx = lineText.indexOf("//");
    if (slashIdx >= 0 && lineText.slice(slashIdx).includes(marker)) {
      return true;
    }
    // Also check block comments /* ... */ on the same line
    const blockMatch = lineText.match(/\/\*[\s\S]*?\*\//g);
    if (blockMatch?.some((c) => c.includes(marker))) {
      return true;
    }
  }

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
