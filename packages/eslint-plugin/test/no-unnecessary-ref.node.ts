import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";
import ts from "typescript-eslint";

import plugin from "../src/index";

async function lintRef(code: string) {
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
          "jax-js/no-unnecessary-ref": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "ref-test.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/no-unnecessary-ref",
  );
}

// --- True positives: should warn ---

test("no-unnecessary-ref: warns on simple .ref access", async () => {
  const code = `
function f() {
  const x = createArray();
  const y = x.ref;
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /\.ref/);
});

test("no-unnecessary-ref: warns on chained .ref.add()", async () => {
  const code = `
function f(x) {
  const y = x.ref.add(1);
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 1);
});

// --- True negatives: should NOT warn ---

test("no-unnecessary-ref: allows .ref in UpdateExpression (ref++)", async () => {
  const code = `
function f(buffer) {
  buffer.ref++;
  buffer.ref--;
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 0);
});

test("no-unnecessary-ref: allows .ref in BinaryExpression (ref === 0)", async () => {
  const code = `
function f(buffer) {
  if (buffer.ref === 0) { console.log("freed"); }
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 0);
});

test("no-unnecessary-ref: allows .ref with allow-ref comment", async () => {
  const code = `
function f(x) {
  // jax-js-lint: allow-ref
  const y = x.ref;
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 0);
});

test("no-unnecessary-ref: allows .ref with inline trailing allow-ref comment", async () => {
  const code = `
function f(x) {
  const y = x.ref; // jax-js-lint: allow-ref
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 0);
});

test("no-unnecessary-ref: allows .ref with inline allow-ref in map callback", async () => {
  const code = `
function f(arr) {
  const y = arr.map((c) => c.ref); // jax-js-lint: allow-ref
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 0);
});

test("no-unnecessary-ref: allows file-level allow-ref directive", async () => {
  const code = `// jax-js-lint: allow-ref
function f(x) {
  const y = x.ref;
}
`;
  const messages = await lintRef(code);
  assert.equal(messages.length, 0);
});

test("no-unnecessary-ref: allows .ref when returning a borrowed typed param", async () => {
  const eslint = new ESLint({
    overrideConfigFile: true,
    overrideConfig: [
      {
        languageOptions: {
          ecmaVersion: "latest",
          sourceType: "module",
          parser: ts.parser as any,
        },
        plugins: {
          "jax-js": plugin as any,
        },
        rules: {
          "jax-js/no-unnecessary-ref": "error",
        },
      },
    ],
  });

  const code = `
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function f(x: np.Array) {
  return { x: x.ref };
}
`;

  const [result] = await eslint.lintText(code, {
    filePath: "ref-return-test.js",
  });
  const messages = result.messages.filter(
    (m) => m.ruleId === "jax-js/no-unnecessary-ref",
  );
  assert.equal(messages.length, 0);
});

test("no-unnecessary-ref: allows .ref when a helper wraps the returned borrowed typed param", async () => {
  const eslint = new ESLint({
    overrideConfigFile: true,
    overrideConfig: [
      {
        languageOptions: {
          ecmaVersion: "latest",
          sourceType: "module",
          parser: ts.parser as any,
        },
        plugins: {
          "jax-js": plugin as any,
        },
        rules: {
          "jax-js/no-unnecessary-ref": "error",
        },
      },
    ],
  });

  const code = `
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function wrap(v: unknown) {
  return v;
}

function f(x: np.Array) {
  return wrap({ x: x.ref });
}
`;

  const [result] = await eslint.lintText(code, {
    filePath: "ref-return-helper-test.js",
  });
  const messages = result.messages.filter(
    (m) => m.ruleId === "jax-js/no-unnecessary-ref",
  );
  assert.equal(messages.length, 0);
});

// --- Autofix ---

test("no-unnecessary-ref: autofix removes .ref from chain", async () => {
  const code = `function f(x) { const y = x.ref; }`;

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
          "jax-js/no-unnecessary-ref": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "ref-autofix.js",
  });
  assert.equal(result.output, `function f(x) { const y = x; }`);
});
