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
          "jax-js/require-retained-release": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "require-retained-release.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/require-retained-release",
  );
}

test("require-retained-release: accepts explicit dispose", async () => {
  const code = `
function f(x) {
  const y = x.ref;
  y.dispose();
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("require-retained-release: accepts transfer", async () => {
  const code = `
function f(x) {
  const y = x.ref;
  return y;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("require-retained-release: accepts object-property handoff", async () => {
  const code = `
function f(x) {
  const y = x.ref;
  const box = { value: y };
  return box;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("require-retained-release: warns on missing terminal action", async () => {
  const code = `
function f(x) {
  const y = x.ref;
  y.shape;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /no explicit release path/);
});
