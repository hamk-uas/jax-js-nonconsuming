/**
 * Shared types and utilities for the mistake telemetry system.
 *
 * Used by:  log-mistake.ts, mistakes-report.ts
 * Ledger:   tmp/copilot-mistakes.json (gitignored)
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

export type MistakeCategory =
  | "state-drift"
  | "wrong-assumption"
  | "missed-check"
  | "tooling"
  | "docs-stale"
  | "perf";

export type MistakeEntry = {
  count: number;
  lastSeen: string;
  category: MistakeCategory;
  symptom: string;
  prevention: string;
};

export type MistakeLedger = {
  _note?: string;
  _updated?: string;
  [key: string]: unknown;
};

export const VALID_CATEGORIES: ReadonlySet<string> = new Set<MistakeCategory>([
  "state-drift",
  "wrong-assumption",
  "missed-check",
  "tooling",
  "docs-stale",
  "perf",
]);

const root = resolve(dirname(new URL(import.meta.url).pathname), "../..");
export const LEDGER_PATH = resolve(root, "tmp/copilot-mistakes.json");

export function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      out[key] = "true";
    } else {
      out[key] = value;
      index++;
    }
  }
  return out;
}

export function readLedger(): MistakeLedger {
  if (!existsSync(LEDGER_PATH)) {
    return {
      _note: "Local, gitignored telemetry for repeated agent mistakes.",
      _updated: new Date().toISOString(),
    };
  }
  return JSON.parse(readFileSync(LEDGER_PATH, "utf8")) as MistakeLedger;
}

export function writeLedger(ledger: MistakeLedger): void {
  mkdirSync(dirname(LEDGER_PATH), { recursive: true });
  ledger._updated = new Date().toISOString();
  writeFileSync(LEDGER_PATH, `${JSON.stringify(ledger, null, 2)}\n`, "utf8");
}

export function isMistakeEntry(value: unknown): value is MistakeEntry {
  if (typeof value !== "object" || value == null) return false;
  const row = value as Record<string, unknown>;
  return (
    typeof row.count === "number" &&
    typeof row.lastSeen === "string" &&
    typeof row.category === "string" &&
    typeof row.symptom === "string" &&
    typeof row.prevention === "string"
  );
}
