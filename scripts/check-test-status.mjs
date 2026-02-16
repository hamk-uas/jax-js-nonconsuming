#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const USAGE = `Usage:
  node scripts/check-test-status.mjs --mode <strict|arch> [--report <path>] [--manifest <path>]

Options:
  --mode       Required. One of: strict, arch
  --report     Vitest JSON report path (default: tmp/vitest-report.json)
  --manifest   Expected failures manifest path (default: .ci/expected-failures.json)
`;

function parseArgs(argv) {
  const args = {
    report: "tmp/vitest-report.json",
    manifest: ".ci/expected-failures.json",
    mode: null,
  };

  for (let i = 2; i < argv.length; i++) {
    const token = argv[i];
    if (token === "--report") args.report = argv[++i];
    else if (token === "--manifest") args.manifest = argv[++i];
    else if (token === "--mode") args.mode = argv[++i];
    else throw new Error(`Unknown argument: ${token}`);
  }

  if (!args.mode) {
    throw new Error("Missing required --mode option.\n\n" + USAGE);
  }

  if (args.mode !== "strict" && args.mode !== "arch") {
    throw new Error(`Invalid --mode '${args.mode}'. Use 'strict' or 'arch'.`);
  }
  return args;
}

function readJson(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing file: ${filePath}`);
  }
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function normalizeRelFile(filePath) {
  const rel = path.relative(process.cwd(), filePath);
  return rel.split(path.sep).join("/");
}

function collectFailures(report) {
  const failures = [];
  const suites = report.testResults ?? [];

  for (const suite of suites) {
    const file = normalizeRelFile(suite.name ?? "unknown");
    const assertions = suite.assertionResults ?? [];
    for (const test of assertions) {
      if (test.status === "failed") {
        failures.push({
          file,
          fullName: test.fullName,
          title: test.title,
        });
      }
    }
  }

  return failures;
}

function keyOf(entry) {
  return `${entry.file}::${entry.fullName}`;
}

function validateManifest(manifest) {
  const entries = manifest.entries ?? [];
  if (!Array.isArray(entries)) {
    throw new Error("Manifest must have an 'entries' array.");
  }

  const duplicateCheck = new Set();
  for (const entry of entries) {
    if (
      !entry.file ||
      !entry.fullName ||
      !entry.reason ||
      !entry.owner ||
      !entry.expires
    ) {
      throw new Error(
        "Each manifest entry must include: file, fullName, reason, owner, expires",
      );
    }
    const k = keyOf(entry);
    if (duplicateCheck.has(k)) {
      throw new Error(`Duplicate manifest entry: ${k}`);
    }
    duplicateCheck.add(k);
  }
  return entries;
}

function isExpired(isoDate) {
  const d = new Date(isoDate);
  if (Number.isNaN(d.getTime())) return true;
  const now = new Date();
  const today = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()),
  );
  const expiryDay = new Date(
    Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()),
  );
  return expiryDay < today;
}

function printFailureList(title, entries) {
  if (entries.length === 0) return;
  console.error(`\n${title}`);
  for (const entry of entries) {
    console.error(`- ${entry.file} :: ${entry.fullName}`);
  }
}

function main() {
  const args = parseArgs(process.argv);
  const report = readJson(args.report);
  const manifest = readJson(args.manifest);

  const failures = collectFailures(report);
  const manifestEntries = validateManifest(manifest);

  const failedSet = new Set(failures.map(keyOf));
  const manifestSet = new Set(manifestEntries.map(keyOf));

  const summary = {
    total: report.numTotalTests ?? 0,
    passed: report.numPassedTests ?? 0,
    failed: report.numFailedTests ?? failures.length,
    skipped: report.numPendingTests ?? 0,
    todo: report.numTodoTests ?? 0,
  };

  console.log(
    `[test-status] mode=${args.mode} total=${summary.total} passed=${summary.passed} failed=${summary.failed} skipped=${summary.skipped} todo=${summary.todo}`,
  );

  if (args.mode === "strict") {
    if (manifestEntries.length > 0) {
      console.error(
        "\nStrict mode forbids expected-failure debt. Clear .ci/expected-failures.json.",
      );
      process.exit(1);
    }
    if (failures.length > 0) {
      printFailureList("Unexpected failing tests (strict mode):", failures);
      process.exit(1);
    }
    console.log("[test-status] strict mode passed.");
    return;
  }

  const expiredEntries = manifestEntries.filter((entry) =>
    isExpired(entry.expires),
  );
  if (expiredEntries.length > 0) {
    console.error("\nExpired expected-failure entries:");
    for (const entry of expiredEntries) {
      console.error(
        `- ${entry.file} :: ${entry.fullName} (owner=${entry.owner}, expires=${entry.expires})`,
      );
    }
    process.exit(1);
  }

  const unexpectedFailures = failures.filter((f) => !manifestSet.has(keyOf(f)));
  const resolvedManifest = manifestEntries.filter(
    (m) => !failedSet.has(keyOf(m)),
  );

  if (unexpectedFailures.length > 0) {
    printFailureList(
      "Unexpected failures (not listed in .ci/expected-failures.json):",
      unexpectedFailures,
    );
    process.exit(1);
  }

  if (resolvedManifest.length > 0) {
    console.error("\nExpected-failure entries now pass and must be removed:");
    for (const entry of resolvedManifest) {
      console.error(`- ${entry.file} :: ${entry.fullName}`);
    }
    process.exit(1);
  }

  console.log(
    "[test-status] architectural mode passed (all failures are approved + unexpired).\n",
  );
}

main();
