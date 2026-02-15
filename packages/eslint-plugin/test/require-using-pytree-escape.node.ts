import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";

import plugin from "../src/index";

async function lintRequireUsing(code: string) {
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
          "jax-js/require-using": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "pytree-escape.js",
  });
  return result.messages.filter((m) => m.ruleId === "jax-js/require-using");
}

test("require-using: nested pytree alias returned does not warn", async () => {
  const code = `
function step(x, y) {
  const a = np.add(x, y);
  const leaf = { a };
  const node = { left: leaf };
  return { tree: node };
}
`;

  const messages = await lintRequireUsing(code);
  assert.equal(messages.length, 0);
});

test("require-using: alias persisted through object property does not warn", async () => {
  const code = `
function store(x, y, holder) {
  const a = np.add(x, y);
  const leaf = { a };
  holder.tree = leaf;
  return holder;
}
`;

  const messages = await lintRequireUsing(code);
  assert.equal(messages.length, 0);
});

test("require-using: tree.makeDisposable ownership handoff does not warn", async () => {
  const code = `
function capture(x, y) {
  const a = np.add(x, y);
  const owner = tree.makeDisposable({ a });
  return owner;
}
`;

  const messages = await lintRequireUsing(code);
  assert.equal(messages.length, 0);
});

test("require-using: non-escaping local still warns", async () => {
  const code = `
function shortLived(x, y) {
  const a = np.add(x, y);
  return x;
}
`;

  const messages = await lintRequireUsing(code);
  assert.equal(messages.length, 1);
  assert.equal(messages[0].line, 3);
});
