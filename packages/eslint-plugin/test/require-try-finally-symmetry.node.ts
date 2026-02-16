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
          "jax-js/require-try-finally-symmetry": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "require-try-finally-symmetry.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/require-try-finally-symmetry",
  );
}

test("require-try-finally-symmetry: accepts try/finally cleanup", async () => {
  const code = `
function f(x) {
  let t;
  try {
    t = x.ref;
    work(t);
  } finally {
    t.dispose();
  }
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("require-try-finally-symmetry: warns when finally is missing", async () => {
  const code = `
function f(x) {
  let t;
  try {
    t = x.ref;
    work(t);
  } catch (e) {
    handle(e);
  }
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /without a finally block/);
});

test("require-try-finally-symmetry: warns when finalizer misses dispose", async () => {
  const code = `
function f(x) {
  let t;
  try {
    t = x.ref;
    work(t);
  } finally {
    cleanup();
  }
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /not disposed in finally/);
});
