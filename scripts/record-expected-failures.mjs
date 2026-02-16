#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

function parseArgs(argv) {
  const args = {
    report: "tmp/vitest-report.json",
    manifest: ".ci/expected-failures.json",
    owner: "@unknown-owner",
    reason: "Architectural refactor (temporary)",
    expires: null,
    expiresDays: 14,
  };

  for (let i = 2; i < argv.length; i++) {
    const token = argv[i];
    if (token === "--report") args.report = argv[++i];
    else if (token === "--manifest") args.manifest = argv[++i];
    else if (token === "--owner") args.owner = argv[++i];
    else if (token === "--reason") args.reason = argv[++i];
    else if (token === "--expires") args.expires = argv[++i];
    else if (token === "--expires-days") args.expiresDays = Number(argv[++i]);
    else throw new Error(`Unknown argument: ${token}`);
  }

  if (!args.expires) {
    const now = new Date();
    const expiry = new Date(
      Date.UTC(
        now.getUTCFullYear(),
        now.getUTCMonth(),
        now.getUTCDate() + args.expiresDays,
      ),
    );
    args.expires = expiry.toISOString().slice(0, 10);
  }

  return args;
}

function readJson(filePath, fallback = null) {
  if (!fs.existsSync(filePath)) {
    if (fallback !== null) return fallback;
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
    for (const test of suite.assertionResults ?? []) {
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

function key(entry) {
  return `${entry.file}::${entry.fullName}`;
}

function main() {
  const args = parseArgs(process.argv);
  const report = readJson(args.report);
  const existing = readJson(args.manifest, { entries: [] });

  const failures = collectFailures(report);
  if (failures.length === 0) {
    console.log(
      "[record-expected-failures] No failing tests in report; manifest unchanged.",
    );
    return;
  }

  const existingEntries = Array.isArray(existing.entries)
    ? existing.entries
    : [];
  const existingMap = new Map(
    existingEntries.map((entry) => [key(entry), entry]),
  );

  const nextEntries = failures.map((failure) => {
    const k = key(failure);
    const prev = existingMap.get(k);
    if (prev) {
      return {
        ...prev,
        file: failure.file,
        fullName: failure.fullName,
      };
    }
    return {
      file: failure.file,
      fullName: failure.fullName,
      reason: args.reason,
      owner: args.owner,
      expires: args.expires,
    };
  });

  nextEntries.sort((a, b) => {
    const fileCmp = a.file.localeCompare(b.file);
    if (fileCmp !== 0) return fileCmp;
    return a.fullName.localeCompare(b.fullName);
  });

  const output = { entries: nextEntries };
  fs.writeFileSync(
    args.manifest,
    JSON.stringify(output, null, 2) + "\n",
    "utf8",
  );

  const retained = nextEntries.filter((entry) =>
    existingMap.has(key(entry)),
  ).length;
  const added = nextEntries.length - retained;

  console.log(
    `[record-expected-failures] Wrote ${args.manifest}: total=${nextEntries.length}, retained=${retained}, added=${added}`,
  );
  console.log(
    "[record-expected-failures] Review owner/reason/expires before commit (especially new entries).",
  );
}

main();
