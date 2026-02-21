import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";

import plugin from "../src/index";

async function lintDisposeThenReassign(code: string) {
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
          "jax-js/no-dispose-then-reassign-param": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "dispose-reassign.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/no-dispose-then-reassign-param",
  );
}

// --- True positives: should warn ---

test("warns on dispose then assign from param", async () => {
  const code = `
function update(newState) {
  state.dispose();
  state = newState;
}
`;
  const messages = await lintDisposeThenReassign(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /state.*newState/);
});

test("warns on tree.dispose then assign from param", async () => {
  const code = `
function update(newCarry) {
  tree.dispose(carry);
  carry = newCarry;
}
`;
  const messages = await lintDisposeThenReassign(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /carry.*newCarry/);
});

test("warns in arrow function", async () => {
  const code = `
const update = (value) => {
  current.dispose();
  current = value;
};
`;
  const messages = await lintDisposeThenReassign(code);
  assert.equal(messages.length, 1);
});

test("warns with destructured param", async () => {
  const code = `
function update({ next }) {
  state.dispose();
  state = next;
}
`;
  const messages = await lintDisposeThenReassign(code);
  assert.equal(messages.length, 1);
});

// --- True negatives: should NOT warn ---

test("allows dispose then assign from non-param", async () => {
  const code = `
function update() {
  const newVal = compute();
  state.dispose();
  state = newVal;
}
`;
  const messages = await lintDisposeThenReassign(code);
  assert.equal(messages.length, 0);
});

test("allows dispose without immediate reassignment", async () => {
  const code = `
function cleanup(x) {
  x.dispose();
  console.log("done");
}
`;
  const messages = await lintDisposeThenReassign(code);
  assert.equal(messages.length, 0);
});

test("allows dispose then assign to different variable", async () => {
  const code = `
function update(newState) {
  old.dispose();
  current = newState;
}
`;
  const messages = await lintDisposeThenReassign(code);
  assert.equal(messages.length, 0);
});
