import assert from "node:assert/strict";
import test from "node:test";

import plugin from "../src/index";

test("plugin exports invariance config", () => {
  const configs = (plugin as any).configs;
  assert.ok(configs);
  assert.ok(configs.invariance);

  const rules = configs.invariance.rules as Record<string, unknown>;
  assert.equal(rules["jax-js/require-using"], "error");
  assert.equal(rules["jax-js/no-use-after-dispose"], "error");
  assert.equal(rules["jax-js/no-dispose-then-reassign-param"], "error");
  assert.equal(rules["jax-js/no-make-disposable-alias"], "error");
  assert.equal(rules["jax-js/no-unnecessary-ref"], "error");
  assert.equal(rules["jax-js/no-array-chain"], "error");
  assert.equal(rules["jax-js/require-scan-result-dispose"], "error");
});
