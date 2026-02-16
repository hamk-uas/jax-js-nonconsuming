import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";

import plugin from "../src/index";

async function lintScanDispose(code: string) {
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
          "jax-js/require-scan-result-dispose": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "scan-dispose.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/require-scan-result-dispose",
  );
}

test("require-scan-result-dispose: warns when scan outputs are not disposed", async () => {
  const code = `
function f(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  const z = np.add(ys.pred, ys.pred);
  return z;
}
`;

  const messages = await lintScanDispose(code);
  assert.equal(messages.length, 2);
});

test("require-scan-result-dispose: no warn when carry and ys disposed", async () => {
  const code = `
function f(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  tree.dispose(carry);
  try {
    return np.add(ys.pred, ys.pred);
  } finally {
    tree.dispose(ys);
  }
}
`;

  const messages = await lintScanDispose(code);
  assert.equal(messages.length, 0);
});

test("require-scan-result-dispose: no warn when scan outputs are returned", async () => {
  const code = `
function f(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  return { carry, ys };
}
`;

  const messages = await lintScanDispose(code);
  assert.equal(messages.length, 0);
});

test("require-scan-result-dispose: no warn when makeDisposable takes ownership", async () => {
  const code = `
function f(step, initCarry, xs) {
  const [carry, ys] = lax.scan(step, initCarry, xs);
  const owned = tree.makeDisposable({ carry, ys });
  return owned;
}
`;

  const messages = await lintScanDispose(code);
  assert.equal(messages.length, 0);
});

test("require-scan-result-dispose: underscore-prefixed outputs are ignored", async () => {
  const code = `
function f(step, initCarry, xs) {
  const [final, _ys] = lax.scan(step, initCarry, xs);
  return final;
}
`;

  const messages = await lintScanDispose(code);
  assert.equal(messages.length, 0);
});

test("require-scan-result-dispose: using alias handoff is recognized", async () => {
  const code = `
async function f() {
  const [finalCarry, outputs] = lax.scan(step, initVal, xs);
  using _fc = finalCarry;
  using _out = outputs;
  await assertAllcloseAsync(finalCarry, [6.0]);
}
`;

  const messages = await lintScanDispose(code);
  assert.equal(messages.length, 0);
});

test("require-scan-result-dispose: using alias of member also works", async () => {
  const code = `
async function f() {
  const [finalCarry, outputs] = lax.scan(step, init, xs);
  using _a = finalCarry.a;
  using _b = finalCarry.b;
  using _out = outputs;
}
`;

  const messages = await lintScanDispose(code);
  assert.equal(messages.length, 0);
});
