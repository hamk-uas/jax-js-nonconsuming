import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";

import plugin from "../src/index";

async function lintUseAfterDispose(code: string) {
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
          "jax-js/no-use-after-dispose": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "use-after-dispose.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/no-use-after-dispose",
  );
}

// --- True positives: should warn ---

test("no-use-after-dispose: warns for use after dispose", async () => {
  const code = `
function f() {
  const x = createArray();
  x.dispose();
  console.log(x);
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /`x` is used after `.dispose\(\)`/);
});

test("no-use-after-dispose: warns for method call after dispose", async () => {
  const code = `
function f() {
  const arr = createArray();
  arr.dispose();
  const d = arr.data();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /`arr` is used after `.dispose\(\)`/);
});

// --- Reassignment: should NOT warn ---

test("no-use-after-dispose: no warn when variable is reassigned after dispose", async () => {
  const code = `
function f() {
  let params = createArray();
  const newParams = transform(params);
  params.dispose();
  params = newParams;
  console.log(params);
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 0);
});

test("no-use-after-dispose: no warn for optax-style dispose-then-reassign pattern", async () => {
  const code = `
function f() {
  let params = create();
  let updates;
  [updates, optState] = solver.update(grad, optState, params);
  const newParams = applyUpdates(params, updates);
  params.dispose();
  updates.dispose();
  params = newParams;

  console.log(params.shape);
  params.dispose();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 0);
});

// --- Deferred callbacks: should NOT warn ---

test("no-use-after-dispose: no warn for dispose inside onTestFinished callback", async () => {
  const code = `
function f() {
  const model = new Model(data);
  onTestFinished(() => model.dispose());
  const result = model.run({ A: a, B: b });
  console.log(result);
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 0);
});

test("no-use-after-dispose: no warn for dispose inside arrow function", async () => {
  const code = `
function f() {
  const x = createArray();
  afterEach(() => { x.dispose(); });
  const y = x.add(1);
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 0);
});

test("no-use-after-dispose: no warn for dispose inside function expression", async () => {
  const code = `
function f() {
  const x = createArray();
  cleanup(function() { x.dispose(); });
  const y = x.mul(2);
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 0);
});

// --- Edge cases ---

test("no-use-after-dispose: still warns for direct dispose (not in callback)", async () => {
  const code = `
function f() {
  const x = createArray();
  x.dispose();
  const y = x.add(1);
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
});

test("no-use-after-dispose: no warn for second dispose call", async () => {
  const code = `
function f() {
  const x = createArray();
  x.dispose();
  x.dispose();
}
`;
  const messages = await lintUseAfterDispose(code);
  // Second .dispose() should not be flagged (it's a dispose call itself)
  assert.equal(messages.length, 0);
});

test("no-use-after-dispose: reassignment resumes tracking on next dispose", async () => {
  const code = `
function f() {
  let x = createArray();
  x.dispose();
  x = createArray();
  x.dispose();
  console.log(x);
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /`x` is used after `.dispose\(\)`/);
});

test("no-use-after-dispose: ignores tree.dispose(arg) on the receiver", async () => {
  const code = `
function f() {
  const x = createArray();
  tree.dispose(x);
  console.log(tree);
}
`;
  const messages = await lintUseAfterDispose(code);
  // tree.dispose(x) has arguments, so `tree` should not be marked as disposed
  assert.equal(messages.length, 0);
});

// --- consumeData() as a consuming method ---

test("no-use-after-dispose: warns for dispose() after consumeData()", async () => {
  const code = `
async function f() {
  const grad = computeGrad();
  const gradV = (await grad.consumeData())[0];
  grad.dispose();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /`grad` is used after `.consumeData\(\)`/);
});

test("no-use-after-dispose: warns for method call after consumeData()", async () => {
  const code = `
async function f() {
  const arr = createArray();
  const raw = await arr.consumeData();
  const d = arr.data();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /`arr` is used after `.consumeData\(\)`/);
});

test("no-use-after-dispose: no warn for use before consumeData()", async () => {
  const code = `
async function f() {
  const arr = createArray();
  const v = arr.add(1);
  const raw = await arr.consumeData();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 0);
});

test("no-use-after-dispose: message includes method name for consumeData", async () => {
  const code = `
async function f() {
  const x = createArray();
  await x.consumeData();
  x.dispose();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  // Message should say consumeData, not dispose
  assert.match(messages[0].message, /`.consumeData\(\)`/);
  assert.doesNotMatch(messages[0].message, /`.dispose\(\)`/);
});

// --- consumeDataSync() as a consuming method ---

test("no-use-after-dispose: warns for dispose() after consumeDataSync()", async () => {
  const code = `
function f() {
  const arr = createArray();
  const raw = arr.consumeDataSync();
  arr.dispose();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  assert.match(
    messages[0].message,
    /`arr` is used after `.consumeDataSync\(\)`/,
  );
});

test("no-use-after-dispose: warns for method call after consumeDataSync()", async () => {
  const code = `
function f() {
  const arr = createArray();
  const raw = arr.consumeDataSync();
  const d = arr.data();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 1);
  assert.match(
    messages[0].message,
    /`arr` is used after `.consumeDataSync\(\)`/,
  );
});

test("no-use-after-dispose: no warn for use before consumeDataSync()", async () => {
  const code = `
function f() {
  const arr = createArray();
  const v = arr.add(1);
  const raw = arr.consumeDataSync();
}
`;
  const messages = await lintUseAfterDispose(code);
  assert.equal(messages.length, 0);
});
