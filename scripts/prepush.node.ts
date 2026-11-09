import assert from "node:assert/strict";
import {
  chmodSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { detectGpu, detectGpus } from "../test/gpu-config";
import { runPrepush } from "./prepush";

type Call = {
  args: string[];
  gpu: string | null;
  required: string;
  cpuOnly: string;
};

function withPnpm(run: (calls: () => Call[]) => void): void {
  const dir = mkdtempSync(join(tmpdir(), "jax-prepush-"));
  const log = join(dir, "calls.jsonl");
  const keys = [
    "PATH",
    "PUSH_GATE_LOG",
    "PUSH_GATE_FAIL",
    "GPU",
    "JAX_REQUIRE_DEVICES",
    "JAX_TEST_CPU_ONLY",
  ];
  const saved = keys.map((key) => [key, process.env[key]] as const);
  try {
    writeFileSync(
      join(dir, "pnpm"),
      `#!/usr/bin/env node
const fs = require("node:fs");
const args = process.argv.slice(2);
const gpu = process.env.GPU ?? null;
fs.appendFileSync(process.env.PUSH_GATE_LOG, JSON.stringify({
  args, gpu, required: process.env.JAX_REQUIRE_DEVICES,
  cpuOnly: process.env.JAX_TEST_CPU_ONLY
}) + "\\n");
const label = gpu ?? "default";
if (process.env.PUSH_GATE_FAIL === "signal") process.kill(process.pid, "SIGTERM");
if (process.env.PUSH_GATE_FAIL === label) process.exit(3);
`,
    );
    chmodSync(join(dir, "pnpm"), 0o755);
    process.env.PATH = `${dir}:${process.env.PATH}`;
    process.env.PUSH_GATE_LOG = log;
    delete process.env.PUSH_GATE_FAIL;
    delete process.env.GPU;
    process.env.JAX_REQUIRE_DEVICES = "";
    process.env.JAX_TEST_CPU_ONLY = "1";
    run(() =>
      readFileSync(log, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as Call),
    );
  } finally {
    for (const [key, value] of saved) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    rmSync(dir, { recursive: true });
  }
}

test("GPU detection selects every present supported GPU, not unrelated manifests", () => {
  const present = new Set([0x10de, 0x8086]);
  assert.deepEqual(
    detectGpus(present, () => true),
    ["nvidia", "intel"],
  );
  assert.equal(
    detectGpu(present, () => true),
    "nvidia",
  );
  assert.deepEqual(
    detectGpus(new Set([0x1002]), () => true),
    ["amd"],
  );
  assert.deepEqual(
    detectGpus(present, (path) => path.endsWith("intel_icd.json")),
    ["intel"],
  );
  assert.deepEqual(
    detectGpus(new Set(), () => true),
    [],
  );
  assert.deepEqual(
    detectGpus(null, () => true),
    [],
  );
});

test("pre-push runs the full default suite and every detected GPU with requirements enforced", () => {
  withPnpm((calls) => {
    assert.equal(runPrepush(["nvidia", "intel"]), 0);
    const runs = calls();
    assert.deepEqual(
      runs.map((run) => run.gpu),
      [null, "nvidia", "intel"],
    );
    assert.deepEqual(
      runs.map((run) => run.args),
      [
        ["vitest", "run"],
        ["vitest", "run", "-c", "test/vitest.nvidia.config.ts"],
        ["vitest", "run", "-c", "test/vitest.intel.config.ts"],
      ],
    );
    assert.ok(
      runs.every(
        (run) => run.required === "webgpu,webgl" && run.cpuOnly === "0",
      ),
    );
  });
});

test("pre-push preserves failures while still running the remaining GPU suites", () => {
  for (const failure of ["default", "nvidia", "intel"]) {
    withPnpm((calls) => {
      process.env.PUSH_GATE_FAIL = failure;
      assert.equal(runPrepush(["nvidia", "intel"]), 3);
      assert.equal(calls().length, 3);
    });
  }
});

test("without a GPU, pre-push uses CPU/WASM and does not hide its failures", () => {
  for (const failed of [false, true]) {
    withPnpm((calls) => {
      if (failed) process.env.PUSH_GATE_FAIL = "default";
      assert.equal(runPrepush([]), failed ? 3 : 0);
      assert.deepEqual(
        calls().map((run) => run.args),
        [["run", "test:cpu"]],
      );
      assert.equal(calls()[0].required, "cpu,wasm");
    });
  }
});

test("a terminated test runner blocks the push", () => {
  withPnpm(() => {
    process.env.PUSH_GATE_FAIL = "signal";
    assert.equal(runPrepush([]), 1);
  });
});
