import type { Rule } from "eslint";

import { getMemberName } from "../types";

const CHAINABLE_ARRAY_METHODS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "pow",
  "mod",
  "neg",
  "reshape",
  "transpose",
  "sum",
  "mean",
  "astype",
  "exp",
  "log",
  "sin",
  "cos",
  "tanh",
  "sqrt",
  "maximum",
  "minimum",
  "dot",
  "matmul",
  "less",
  "lessEqual",
  "greater",
  "greaterEqual",
  "equal",
  "notEqual",
]);

/**
 * Tracing transforms whose first (function) argument is traced — not eagerly
 * executed — so array chains inside those bodies do NOT create real GPU
 * temporaries.  The JIT compiler manages intermediates automatically.
 */
const TRACING_TRANSFORMS = new Set([
  "jit",
  "grad",
  "valueAndGrad",
  "jvp",
  "vjp",
  "vmap",
  "jacfwd",
  "jacrev",
  "hessian",
  "linearize",
  "makeJaxpr",
]);

/**
 * Check whether a callee AST node is a known tracing transform.
 */
function isTransformCallee(callee: any): boolean {
  // Direct call: jit(fn), grad(fn), etc.
  if (callee?.type === "Identifier" && TRACING_TRANSFORMS.has(callee.name))
    return true;
  // Method call: lax.scan(fn, ...)
  if (callee?.type === "MemberExpression") {
    const method = getMemberName(callee.property);
    if (method === "scan") return true;
  }
  return false;
}

/**
 * Walk up the scope chain looking for a Variable with the given name.
 */
function findVariable(scope: any, name: string): any {
  let s = scope;
  while (s) {
    const v = s.variables?.find((v: any) => v.name === name);
    if (v) return v;
    s = s.upper;
  }
  return null;
}

function getChainCalls(node: any): any[] {
  const calls: any[] = [];
  let current = node;
  while (
    current?.type === "CallExpression" &&
    current.callee?.type === "MemberExpression"
  ) {
    const method = getMemberName(current.callee.property);
    if (!method || !CHAINABLE_ARRAY_METHODS.has(method)) break;
    calls.unshift(current);
    current = current.callee.object;
  }
  return calls;
}

function getIndent(sourceText: string, start: number): string {
  const lineStart = sourceText.lastIndexOf("\n", start - 1) + 1;
  const prefix = sourceText.slice(lineStart, start);
  const match = prefix.match(/^\s*/);
  return match?.[0] ?? "";
}

function collectUsedNames(scope: any): Set<string> {
  const names = new Set<string>();
  let current = scope;
  while (current) {
    for (const variable of current.variables ?? []) {
      if (variable?.name) names.add(variable.name);
    }
    current = current.upper;
  }
  return names;
}

function nextTempName(used: Set<string>): string {
  let i = 1;
  while (used.has(`_jaxChain${i}`)) i++;
  const name = `_jaxChain${i}`;
  used.add(name);
  return name;
}

function replaceCallObjectText(
  sourceText: string,
  callNode: any,
  objectText: string,
): string {
  const callText = sourceText.slice(callNode.range[0], callNode.range[1]);
  const objectNode = callNode.callee.object;
  const relStart = objectNode.range[0] - callNode.range[0];
  const relEnd = objectNode.range[1] - callNode.range[0];
  return callText.slice(0, relStart) + objectText + callText.slice(relEnd);
}

function findRewriteStatement(node: any, context: Rule.RuleContext): any {
  const sourceCode = context.sourceCode ?? context.getSourceCode();
  const ancestors = sourceCode.getAncestors(node);
  for (let i = ancestors.length - 1; i >= 0; i--) {
    const ancestor = ancestors[i];
    if (
      ancestor.type === "VariableDeclaration" ||
      ancestor.type === "ExpressionStatement"
    ) {
      return ancestor;
    }
  }
  return null;
}

function buildChainRewrite(
  node: any,
  context: Rule.RuleContext,
): string | null {
  const sourceCode = context.sourceCode ?? context.getSourceCode();
  const sourceText = sourceCode.getText();
  const chainCalls = getChainCalls(node);
  if (chainCalls.length < 2) return null;

  const scope = sourceCode.getScope(node);
  const usedNames = collectUsedNames(scope);

  const statement = findRewriteStatement(node, context);
  if (!statement) return null;

  const indent = getIndent(sourceText, statement.range[0]);

  if (statement.type === "VariableDeclaration") {
    if (statement.declarations.length !== 1) return null;
    if (statement.kind !== "const" && statement.kind !== "let") return null;

    const [decl] = statement.declarations;
    if (decl.init !== node || !decl.id) return null;

    const lines: string[] = [];
    // jax-js-lint: allow-non-using
    let currentObjectText = sourceText.slice(
      chainCalls[0].callee.object.range[0],
      chainCalls[0].callee.object.range[1],
    );

    for (let i = 0; i < chainCalls.length - 1; i++) {
      const callText = replaceCallObjectText(
        sourceText,
        chainCalls[i],
        currentObjectText,
      );
      const temp = nextTempName(usedNames);
      lines.push(`${indent}using ${temp} = ${callText};`);
      currentObjectText = temp;
    }

    const lastCallText = replaceCallObjectText(
      sourceText,
      chainCalls[chainCalls.length - 1],
      currentObjectText,
    );
    const idText = sourceText.slice(decl.id.range[0], decl.id.range[1]);
    lines.push(`${indent}${statement.kind} ${idText} = ${lastCallText};`);
    return lines.join("\n");
  }

  if (statement.type === "ExpressionStatement") {
    if (statement.expression !== node) return null;
    const lines: string[] = [];
    // jax-js-lint: allow-non-using
    let currentObjectText = sourceText.slice(
      chainCalls[0].callee.object.range[0],
      chainCalls[0].callee.object.range[1],
    );

    for (let i = 0; i < chainCalls.length; i++) {
      const callText = replaceCallObjectText(
        sourceText,
        chainCalls[i],
        currentObjectText,
      );
      const temp = nextTempName(usedNames);
      lines.push(`${indent}using ${temp} = ${callText};`);
      currentObjectText = temp;
    }

    return lines.join("\n");
  }

  return null;
}

/**
 * Walk up the AST from `node`.  If we find a function expression / arrow
 * function that is used as the **body argument** of a tracing transform,
 * the chain is inside a traced context and should be suppressed.
 *
 * Handles two cases:
 * 1. Inline: `jit((x) => x.mul(x).add(x))` — function is the first arg.
 * 2. Named:  `const f = (x) => x.mul(x).add(x); grad(f)(x)` — function
 *    is assigned to a variable that is later passed to a transform.
 *
 * Also suppresses chains inside `lax.scan` body functions (first arg).
 */
function isInsideTracedBody(node: any, context: Rule.RuleContext): boolean {
  let current = node.parent;
  while (current) {
    if (
      current.type === "ArrowFunctionExpression" ||
      current.type === "FunctionExpression"
    ) {
      const parent = current.parent;

      // Case 1: Inline function as first arg to transform.
      if (
        parent?.type === "CallExpression" &&
        parent.arguments[0] === current
      ) {
        if (isTransformCallee(parent.callee)) return true;
      }

      // Case 2: Function assigned to a const/let, and at least one
      // reference site passes it as the first arg to a tracing transform.
      if (
        parent?.type === "VariableDeclarator" &&
        parent.id?.type === "Identifier"
      ) {
        const scope = context.sourceCode.getScope(parent);
        const variable = findVariable(scope, parent.id.name);
        if (variable) {
          for (const ref of variable.references) {
            const id = ref.identifier;
            // Used as: grad(f), jit(f), vmap(f), etc.
            if (
              id.parent?.type === "CallExpression" &&
              id.parent.arguments[0] === id &&
              isTransformCallee(id.parent.callee)
            ) {
              return true;
            }
          }
        }
      }
    }
    current = current.parent;
  }
  return false;
}

const rule: Rule.RuleModule = {
  meta: {
    type: "suggestion",
    fixable: "code",
    docs: {
      description:
        "Disallow deep array method chains that create unnamed eager intermediates",
    },
    schema: [
      {
        type: "object",
        properties: {
          minDepth: { type: "integer", minimum: 2 },
        },
        additionalProperties: false,
      },
    ],
    messages: {
      noArrayChain:
        "Array call chain depth {{depth}} creates unnamed eager temporaries. Split into `using` bindings.",
    },
  },
  create(context) {
    const minDepth = (context.options[0] as any)?.minDepth ?? 2;

    function depthOfCallChain(node: any): number {
      let depth = 0;
      let current: any = node;
      while (
        current?.type === "CallExpression" &&
        current.callee?.type === "MemberExpression"
      ) {
        const method = getMemberName(current.callee.property);
        if (!method || !CHAINABLE_ARRAY_METHODS.has(method)) break;
        depth++;
        current = current.callee.object;
      }
      return depth;
    }

    return {
      CallExpression(node: any) {
        if (node.callee?.type !== "MemberExpression") return;
        const depth = depthOfCallChain(node);
        if (depth < minDepth) return;

        // Only report the outermost chain — skip if a parent call is also
        // a qualifying chain (avoids duplicate depth-3 + depth-2 reports).
        const parentCall = node.parent;
        if (
          parentCall?.type === "MemberExpression" &&
          parentCall.parent?.type === "CallExpression"
        ) {
          const parentDepth = depthOfCallChain(parentCall.parent);
          if (parentDepth >= minDepth) return;
        }

        // Suppress inside traced contexts — jit/grad/vmap/etc. body
        // functions are traced (not eagerly executed), so chains inside
        // them do not create real GPU temporaries.
        if (isInsideTracedBody(node, context)) return;

        context.report({
          node,
          messageId: "noArrayChain",
          data: { depth: String(depth) },
          fix(fixer) {
            const statement = findRewriteStatement(node, context);
            if (!statement) return null;
            const replacement = buildChainRewrite(node, context);
            if (!replacement) return null;
            return fixer.replaceTextRange(statement.range, replacement);
          },
        });
      },
    };
  },
};

export default rule;
