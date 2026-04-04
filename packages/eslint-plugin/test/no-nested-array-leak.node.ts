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
          "jax-js/no-nested-array-leak": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "nested-leak.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/no-nested-array-leak",
  );
}

async function lintWithFix(code: string) {
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
          "jax-js/no-nested-array-leak": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "nested-leak.js",
  });
  return {
    output: result.output ?? code,
    messages: result.messages.filter(
      (m) => m.ruleId === "jax-js/no-nested-array-leak",
    ),
  };
}

// ── Should report: nested array-producing calls ──────────────────────────

test("no-nested-array-leak: np.tile(np.reshape(...)) flags inner reshape", async () => {
  const code = `
function f(G) {
  const result = np.tile(np.reshape(G, [1, 2, 2]), [10, 1, 1]);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.ok(messages[0].message.includes("np.reshape"));
  assert.equal(messages[0].messageId, "nestedArrayLeak");
});

test("no-nested-array-leak: np.add(np.array(...), np.eye(...)) flags both args", async () => {
  const code = `
function f() {
  const result = np.add(np.array([1, 2]), np.eye(2));
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
  assert.ok(messages[0].message.includes("np.array"));
  assert.ok(messages[1].message.includes("np.eye"));
});

test("no-nested-array-leak: method chain as argument flags inner call", async () => {
  const code = `
function f(x, y) {
  const result = np.add(x.reshape([2, 3]), y);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.ok(messages[0].message.includes("reshape"));
});

test("no-nested-array-leak: np.multiply(np.array(...), np.eye(...)) flags both", async () => {
  const code = `
function f() {
  const result = np.multiply(np.array([2]), np.eye(3));
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
});

// ── Should NOT report: no nesting ────────────────────────────────────────

test("no-nested-array-leak: flat variable binding does not trigger", async () => {
  const code = `
function f(x) {
  using a = np.reshape(x, [2, 3]);
  const result = np.tile(a, [10, 1, 1]);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("no-nested-array-leak: non-array-producing outer call does not trigger", async () => {
  const code = `
function f(x) {
  doSomething(np.reshape(x, [2, 3]));
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

test("no-nested-array-leak: allow-non-using comment suppresses", async () => {
  const code = `
function f(G) {
  // jax-js-lint: allow-non-using
  const result = np.tile(np.reshape(G, [1, 2, 2]), [10, 1, 1]);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 0);
});

// ── Traced-context: warns with safe-to-fix message ──────────────────────

test("no-nested-array-leak: warns inside inline jit body with traced message", async () => {
  const code = `
const f = jit((x) => np.tile(np.reshape(x, [1, 2, 2]), [10, 1, 1]));
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-nested-array-leak: warns inside inline grad body with traced message", async () => {
  const code = `
const g = grad((x) => np.add(np.square(x), np.eye(2)).sum());
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
  assert.ok(
    messages.every((m) => m.message.includes("safe inside jit/grad/scan")),
  );
});

test("no-nested-array-leak: warns inside inline vmap body with traced message", async () => {
  const code = `
const h = vmap((x) => np.tile(np.reshape(x, [1, 3]), [2, 1]));
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-nested-array-leak: warns inside lax.scan step with traced message", async () => {
  const code = `
const [carry, ys] = lax.scan(
  (c, x) => [np.add(c, np.reshape(x, [1, 2])), c],
  init,
  xs,
);
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-nested-array-leak: warns for named function passed to grad with traced message", async () => {
  const code = `
const f = (x) => np.tile(np.reshape(x, [1, 2]), [3, 1]).sum();
const df = grad(f);
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /safe inside jit\/grad\/scan/);
});

test("no-nested-array-leak: warns for named function passed to jit with traced message", async () => {
  const code = `
const f = (x) => np.add(np.square(x), np.eye(2));
const jf = jit(f);
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
  assert.ok(
    messages.every((m) => m.message.includes("safe inside jit/grad/scan")),
  );
});

test("no-nested-array-leak: warns inside hessian with traced message", async () => {
  const code = `
const h = hessian((x) => np.add(np.square(x), np.eye(2)).sum());
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
  assert.ok(
    messages.every((m) => m.message.includes("safe inside jit/grad/scan")),
  );
});

test("no-nested-array-leak: where args are flagged inside grad body with traced message", async () => {
  const code = `
const g = grad((x) => np.where(x.greater(0), x.sum(), x.add(1)).sum());
`;
  const messages = await lint(code);
  assert.equal(messages.length, 3);
  assert.ok(
    messages.every((m) => m.message.includes("safe inside jit/grad/scan")),
  );
});

test("no-nested-array-leak: where condition and branch flagged inside traced body", async () => {
  const code = `
const g = grad((x) => np.where(np.greater(x, 0), x, x.add(1)).sum());
`;
  const messages = await lint(code);
  assert.equal(messages.length, 2);
  assert.ok(
    messages.every((m) => m.message.includes("safe inside jit/grad/scan")),
  );
});

test("no-nested-array-leak: where args flagged inside lax.scan step with traced message", async () => {
  const code = `
const [carry, ys] = lax.scan(
  (c, x) => [np.where(c.greater(0), c.sum(), c.add(x)), c],
  init,
  xs,
);
`;
  const messages = await lint(code);
  assert.equal(messages.length, 3);
  assert.ok(
    messages.every((m) => m.message.includes("safe inside jit/grad/scan")),
  );
});

test("no-nested-array-leak: eager message outside tracing context", async () => {
  const code = `
function f(x) {
  return np.tile(np.reshape(x, [1, 2, 2]), [10, 1, 1]);
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /never disposed/);
});

test("no-nested-array-leak: lax factory nested in np call flags inner", async () => {
  const code = `
function f(x) {
  const result = np.add(lax.erf(x), x);
  return result;
}
`;
  const messages = await lint(code);
  assert.equal(messages.length, 1);
  assert.ok(messages[0].message.includes("lax.erf"));
});

// ── Autofix tests ────────────────────────────────────────────────────────

test("no-nested-array-leak: autofix extracts single nested arg", async () => {
  const code = `
function f(G) {
  const result = np.tile(np.reshape(G, [1, 2, 2]), [10, 1, 1]);
  return result;
}
`;
  const { output, messages } = await lintWithFix(code);
  assert.equal(messages.length, 0);
  assert.match(output, /using _jaxTmp1 = np\.reshape\(G, \[1, 2, 2\]\);/);
  assert.match(output, /np\.tile\(_jaxTmp1, \[10, 1, 1\]\)/);
});

test("no-nested-array-leak: autofix extracts both nested args", async () => {
  const code = `
function f() {
  const result = np.add(np.array([1, 2]), np.eye(2));
  return result;
}
`;
  const { output, messages } = await lintWithFix(code);
  assert.equal(messages.length, 0);
  assert.match(output, /using _jaxTmp1 = np\.array\(\[1, 2\]\);/);
  assert.match(output, /using _jaxTmp2 = np\.eye\(2\);/);
  assert.match(output, /np\.add\(_jaxTmp1, _jaxTmp2\)/);
});

test("no-nested-array-leak: autofix extracts method-call arg", async () => {
  const code = `
function f(x, y) {
  const result = np.add(x.reshape([2, 3]), y);
  return result;
}
`;
  const { output, messages } = await lintWithFix(code);
  assert.equal(messages.length, 0);
  assert.match(output, /using _jaxTmp1 = x\.reshape\(\[2, 3\]\);/);
  assert.match(output, /np\.add\(_jaxTmp1, y\)/);
});

test("no-nested-array-leak: autofix preserves indentation", async () => {
  const code = `
function f(x) {
    const result = np.add(np.square(x), x);
    return result;
}
`;
  const { output, messages } = await lintWithFix(code);
  assert.equal(messages.length, 0);
  // Should use 4-space indent matching the const statement
  assert.match(output, /\n {4}using _jaxTmp1 = np\.square\(x\);/);
});

test("no-nested-array-leak: no autofix inside return statement", async () => {
  // Return statements with nested calls get fix too — extracted before return
  const code = `
function f(x) {
  return np.tile(np.reshape(x, [1, 2, 2]), [10, 1, 1]);
}
`;
  const { output, messages } = await lintWithFix(code);
  assert.equal(messages.length, 0);
  assert.match(output, /using _jaxTmp1 = np\.reshape\(x, \[1, 2, 2\]\);/);
  assert.match(output, /return np\.tile\(_jaxTmp1, \[10, 1, 1\]\);/);
});
