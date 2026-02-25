import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";

import plugin from "../src/index";

async function lint(code: string) {
  const eslint = new ESLint({
    overrideConfigFile: true,
    overrideConfig: [
      {
        languageOptions: {
          ecmaVersion: "latest",
          sourceType: "module",
        },
        plugins: {
          "jax-js": plugin as any,
        },
        rules: {
          "jax-js/require-mark-anonymous": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "array-factory.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/require-mark-anonymous",
  );
}

// ── Should report: new Array({source: ...}) in exported function ─────────

test("require-mark-anonymous: bare new Array({source: ...}) in export function", async () => {
  const code = `
export function arange(n) {
  return new Array({source: expr, st, dtype: DType.Float32});
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.ok(messages[0].message.includes("arange"));
  assert.ok(messages[0].message.includes("markAnonymousIfTracing"));
});

test("require-mark-anonymous: two bare constructors in one export function", async () => {
  const code = `
export function factory(n) {
  if (n > 0) {
    return new Array({source: a, st, dtype: DType.Float32});
  }
  return new Array({source: b, st, dtype: DType.Float32});
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
});

// ── Should NOT report: wrapped in markAnonymousIfTracing ─────────────────

test("require-mark-anonymous: wrapped in markAnonymousIfTracing passes", async () => {
  const code = `
export function arange(n) {
  return markAnonymousIfTracing(new Array({source: expr, st, dtype: DType.Float32}));
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

// ── Should NOT report: non-exported function ─────────────────────────────

test("require-mark-anonymous: non-exported function is exempt", async () => {
  const code = `
function fullInternal(n) {
  return new Array({source: expr, st, dtype: DType.Float32});
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

// ── Should NOT report: class method ──────────────────────────────────────

test("require-mark-anonymous: class method is exempt", async () => {
  const code = `
class EvalTrace {
  processPrimitive() {
    return new Array({source: expr, st, dtype: DType.Float32});
  }
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

// ── Should NOT report: no source property ────────────────────────────────

test("require-mark-anonymous: new Array without source property is exempt", async () => {
  const code = `
export function factory(n) {
  return new Array({st, dtype: DType.Float32, backend});
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

// ── Should NOT report: suppression comment ───────────────────────────────

test("require-mark-anonymous: allow-unmarked comment suppresses", async () => {
  const code = `
export function fullInternal(n) {
  // jax-js-lint: allow-unmarked
  return new Array({source: expr, st, dtype: DType.Float32});
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

// ── Should NOT report: new Array() with non-Identifier callee ────────────

test("require-mark-anonymous: new SomeOtherClass({source: ...}) is exempt", async () => {
  const code = `
export function factory(n) {
  return new SomeOtherClass({source: expr, st});
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

// ── Edge: default-exported function ──────────────────────────────────────

test("require-mark-anonymous: default export function is exempt (only named exports)", async () => {
  const code = `
export default function factory(n) {
  return new Array({source: expr, st, dtype: DType.Float32});
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});
