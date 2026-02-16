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
        });
      },
    };
  },
};

export default rule;
