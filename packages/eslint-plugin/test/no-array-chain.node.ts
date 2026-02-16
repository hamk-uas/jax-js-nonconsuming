import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";

import plugin from "../src/index";

async function lintArrayChain(code: string) {
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
          "jax-js/no-array-chain": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "array-chain.js",
  });
  return result.messages.filter((m) => m.ruleId === "jax-js/no-array-chain");
}

// --- True positives: should warn ---

test("no-array-chain: warns for depth-2 chain in eager code", async () => {
  const code = `
const x = createArray();
const y = x.add(1).mul(2);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /depth 2/);
});

test("no-array-chain: warns for depth-3 chain in eager code", async () => {
  const code = `
const x = createArray();
const y = x.add(1).mul(2).sub(3);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /depth 3/);
});

// --- Traced-context suppression: should NOT warn ---

test("no-array-chain: suppressed inside inline jit body", async () => {
  const code = `
const f = jit((x) => x.add(1).mul(2));
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 0);
});

test("no-array-chain: suppressed inside inline grad body", async () => {
  const code = `
const r = grad((x) => x.mul(x).sum())(input);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 0);
});

test("no-array-chain: suppressed inside inline vmap body", async () => {
  const code = `
const f = vmap((x) => x.mul(2).add(3));
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 0);
});

test("no-array-chain: suppressed inside inline lax.scan body", async () => {
  const code = `
lax.scan((carry, x) => {
  const s = carry.add(x).mul(2);
  return [s, s];
}, init, xs);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 0);
});

test("no-array-chain: suppressed for named function passed to grad", async () => {
  const code = `
const f = (x) => x.mul(x).add(x);
const df = grad(f);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 0);
});

test("no-array-chain: suppressed for named function passed to jit", async () => {
  const code = `
const f = (x) => x.add(1).mul(2).sub(3);
const jf = jit(f);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 0);
});

test("no-array-chain: suppressed for named function passed to hessian", async () => {
  const code = `
const f = (x) => x.mul(x).sum();
const H = jit(hessian(f));
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 0);
});

test("no-array-chain: NOT suppressed for named function never passed to transform", async () => {
  const code = `
const f = (x) => x.mul(x).add(x);
const y = f(input);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
});
