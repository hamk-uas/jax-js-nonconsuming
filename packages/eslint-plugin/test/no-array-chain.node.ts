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

async function lintArrayChainWithFix(code: string) {
  const eslint = new ESLint({
    fix: true,
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
  return {
    output: result.output ?? code,
    messages: result.messages.filter(
      (m) => m.ruleId === "jax-js/no-array-chain",
    ),
  };
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

test("no-array-chain: autofix rewrites variable assignment chains", async () => {
  const code = `
const x = createArray();
const y = x.add(1).mul(2).sub(3);
`;
  const { output, messages } = await lintArrayChainWithFix(code);
  assert.equal(messages.length, 0);
  assert.match(output, /using _jaxChain\d+ = x\.add\(1\);/);
  assert.match(output, /using _jaxChain\d+ = _jaxChain\d+\.mul\(2\);/);
  assert.match(output, /const y = _jaxChain\d+\.sub\(3\);/);
});

test("no-array-chain: autofix rewrites expression statement chains", async () => {
  const code = `
const x = createArray();
x.add(1).mul(2);
`;
  const { output, messages } = await lintArrayChainWithFix(code);
  assert.equal(messages.length, 0);
  assert.match(output, /using _jaxChain\d+ = x\.add\(1\);/);
  assert.match(output, /using _jaxChain\d+ = _jaxChain\d+\.mul\(2\);/);
});

test("no-array-chain: does not autofix return-expression chains", async () => {
  const code = `
function f(x) {
  return x.add(1).mul(2);
}
`;
  const { output, messages } = await lintArrayChainWithFix(code);
  assert.equal(messages.length, 1);
  assert.equal(output, code);
});

// --- Traced-context: warns with safe-to-fix message ---

test("no-array-chain: warns inside inline jit body with traced message", async () => {
  const code = `
const f = jit((x) => x.add(1).mul(2));
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-array-chain: warns inside inline grad body with traced message", async () => {
  const code = `
const r = grad((x) => x.mul(x).sum())(input);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-array-chain: warns inside inline vmap body with traced message", async () => {
  const code = `
const f = vmap((x) => x.mul(2).add(3));
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-array-chain: warns inside inline lax.scan body with traced message", async () => {
  const code = `
lax.scan((carry, x) => {
  const s = carry.add(x).mul(2);
  return [s, s];
}, init, xs);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-array-chain: warns for named function passed to grad with traced message", async () => {
  const code = `
const f = (x) => x.mul(x).add(x);
const df = grad(f);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-array-chain: warns for named function passed to jit with traced message", async () => {
  const code = `
const f = (x) => x.add(1).mul(2).sub(3);
const jf = jit(f);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-array-chain: warns for named function passed to hessian with traced message", async () => {
  const code = `
const f = (x) => x.mul(x).sum();
const H = jit(hessian(f));
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-array-chain: eager message for function never passed to transform", async () => {
  const code = `
const f = (x) => x.mul(x).add(x);
const y = f(input);
`;
  const messages = await lintArrayChain(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /unnamed eager temporaries/);
});
