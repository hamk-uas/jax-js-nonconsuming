import assert from "node:assert/strict";
import test from "node:test";

import { ESLint } from "eslint";
import ts from "typescript-eslint";

import plugin from "../src/index";

async function lintBorrowedReturn(code: string) {
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
          "jax-js/no-borrowed-param-return": "error",
        },
      },
    ],
  });

  const [result] = await eslint.lintText(code, {
    filePath: "borrowed-param-return-test.js",
  });
  return result.messages.filter(
    (m) => m.ruleId === "jax-js/no-borrowed-param-return",
  );
}

test("no-borrowed-param-return: warns on returned object property alias", async () => {
  const code = `
import type { Tracer } from "@hamk-uas/jax-js-nonconsuming";
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function foo(x: np.Array): { x: np.Array } {
  return { x };
}
`;
  const messages = await lintBorrowedReturn(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /x\.ref/);
});

test("no-borrowed-param-return: warns on direct Tracer return", async () => {
  const code = `
import type { Tracer } from "@hamk-uas/jax-js-nonconsuming";

function foo(x: Tracer): Tracer {
  return x;
}
`;
  const messages = await lintBorrowedReturn(code);
  assert.equal(messages.length, 1);
});

test("no-borrowed-param-return: warns on add(0) workaround", async () => {
  const code = `
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function foo(x: np.Array): { x: np.Array } {
  return { x: x.add(0.0) };
}
`;
  const messages = await lintBorrowedReturn(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /add\(0\)/);
  assert.match(messages[0].message, /x\.ref/);
});

test("no-borrowed-param-return: warns through helper-wrapped return", async () => {
  const code = `
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function wrap(v: unknown) {
  return v;
}

function foo(x: np.Array): unknown {
  return wrap({ x });
}
`;
  const messages = await lintBorrowedReturn(code);
  assert.equal(messages.length, 1);
  assert.match(messages[0].message, /x\.ref/);
});

test("no-borrowed-param-return: allows explicit .ref", async () => {
  const code = `
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function foo(x: np.Array): { x: np.Array } {
  return { x: x.ref };
}
`;
  const messages = await lintBorrowedReturn(code);
  assert.equal(messages.length, 0);
});

test("no-borrowed-param-return: allows helper-wrapped explicit .ref", async () => {
  const code = `
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function wrap(v: unknown) {
  return v;
}

function foo(x: np.Array): unknown {
  return wrap({ x: x.ref });
}
`;
  const messages = await lintBorrowedReturn(code);
  assert.equal(messages.length, 0);
});

test("no-borrowed-param-return: allows returned transformed value", async () => {
  const code = `
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

function foo(x: np.Array, y: np.Array): { x: np.Array } {
  return { x: x.add(y) };
}
`;
  const messages = await lintBorrowedReturn(code);
  assert.equal(messages.length, 0);
});
