import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";

import plugin from "../src/index";

async function lintMakeDisposableAlias(code: string) {
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
          "jax-js/no-make-disposable-alias": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "make-disposable-alias.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/no-make-disposable-alias",
  );
}

test("no-make-disposable-alias: warns for duplicated object identifier", async () => {
  const code = `
function f(base) {
  return tree.makeDisposable({ xf_0: base, yhat: base });
}
`;

  const messages = await lintMakeDisposableAlias(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /reuses `base`/);
});

test("no-make-disposable-alias: warns for duplicated array identifier", async () => {
  const code = `
function f(a) {
  return tree.makeDisposable([a, a]);
}
`;

  const messages = await lintMakeDisposableAlias(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /reuses `a`/);
});

test("no-make-disposable-alias: no warn for distinct identifiers", async () => {
  const code = `
function f(a, b) {
  return tree.makeDisposable({ xf_0: a, yhat: b });
}
`;

  const messages = await lintMakeDisposableAlias(code);
  assert.equal(messages.length, 0);
});

test("no-make-disposable-alias: no warn for non-literal handoff", async () => {
  const code = `
function f(base) {
  const out = { xf_0: base, yhat: base };
  return tree.makeDisposable(out);
}
`;

  const messages = await lintMakeDisposableAlias(code);
  assert.equal(messages.length, 0);
});
