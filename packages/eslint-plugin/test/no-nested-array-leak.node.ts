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
          "jax-js/no-nested-array-leak": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "nested-leak.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/no-nested-array-leak",
  );
}

// ── Should report: nested array-producing calls ──────────────────────────

test("no-nested-array-leak: np.tile(np.reshape(...)) flags inner reshape", async () => {
  const code = `
function f(G) {
  const result = np.tile(np.reshape(G, [1, 2, 2]), [10, 1, 1]);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.ok(messages[0].message.includes("np.reshape"));
  assert.equal(messages[0].messageId, "nestedArrayLeak");
});

test("no-nested-array-leak: np.add(np.array(...), np.eye(...)) flags both args", async () => {
  const code = `
function f() {
  const result = np.add(np.array([1, 2]), np.eye(2));
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
  assert.ok(messages[0].message.includes("np.array"));
  assert.ok(messages[1].message.includes("np.eye"));
});

test("no-nested-array-leak: method chain as argument flags inner call", async () => {
  const code = `
function f(x, y) {
  const result = np.add(x.reshape([2, 3]), y);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.ok(messages[0].message.includes("reshape"));
});

test("no-nested-array-leak: np.multiply(np.array(...), np.eye(...)) flags both", async () => {
  const code = `
function f() {
  const result = np.multiply(np.array([2]), np.eye(3));
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
});

// ── Should NOT report: no nesting ────────────────────────────────────────

test("no-nested-array-leak: flat variable binding does not trigger", async () => {
  const code = `
function f(x) {
  using a = np.reshape(x, [2, 3]);
  const result = np.tile(a, [10, 1, 1]);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("no-nested-array-leak: non-array-producing outer call does not trigger", async () => {
  const code = `
function f(x) {
  doSomething(np.reshape(x, [2, 3]));
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("no-nested-array-leak: allow-non-using comment suppresses", async () => {
  const code = `
function f(G) {
  // jax-js-lint: allow-non-using
  const result = np.tile(np.reshape(G, [1, 2, 2]), [10, 1, 1]);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("no-nested-array-leak: lax factory nested in np call flags inner", async () => {
  const code = `
function f(x) {
  const result = np.add(lax.erf(x), x);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.ok(messages[0].message.includes("lax.erf"));
});
