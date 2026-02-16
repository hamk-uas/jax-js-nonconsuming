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
          "jax-js/require-wrapper-dispose-symmetry": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "require-wrapper-dispose-symmetry.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/require-wrapper-dispose-symmetry",
  );
}

test("require-wrapper-dispose-symmetry: accepts retained cleanup before inner", async () => {
  const code = `
class Wrapper {
  dispose() {
    this.cache.dispose();
    this.inner.dispose();
  }
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("require-wrapper-dispose-symmetry: accepts only inner dispose", async () => {
  const code = `
class Wrapper {
  dispose() {
    this.inner.dispose();
  }
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("require-wrapper-dispose-symmetry: warns when inner dispose is not last", async () => {
  const code = `
class Wrapper {
  dispose() {
    this.inner.dispose();
    this.cache.dispose();
  }
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /should be last/);
});
