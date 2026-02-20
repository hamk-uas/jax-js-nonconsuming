import {
  isMistakeEntry,
  type MistakeCategory,
  parseArgs,
  readLedger,
} from "./lib/mistakes";

type ReportRow = {
  key: string;
  count: number;
  category: MistakeCategory;
  ageDays: number;
  score: number;
  promote: boolean;
  symptom: string;
  prevention: string;
};

/** Category weights for priority scoring. Higher = more urgent. */
const CATEGORY_WEIGHT: Record<MistakeCategory, number> = {
  "state-drift": 1.3,
  "wrong-assumption": 1.15,
  "missed-check": 1.2,
  tooling: 0.95,
  "docs-stale": 1.0,
  perf: 1.05,
};

function help(): void {
  console.log(`Usage:
  pnpm run mistakes:report
  pnpm run mistakes:report -- --min-count 2
  pnpm run mistakes:report -- --json

Scoring model:
  score = count × categoryWeight × recencyFactor
  recencyFactor = 1 / (1 + ageDays / 90)

Promotion default:
  promote when count >= 2 (override with --min-count N)
`);
}

function daysSince(iso: string, nowMs: number): number {
  const ts = Date.parse(iso);
  if (!Number.isFinite(ts)) return 9999;
  return Math.max(0, (nowMs - ts) / (1000 * 60 * 60 * 24));
}

const args = parseArgs(process.argv.slice(2));
if (args.help === "true" || args.h === "true") {
  help();
  process.exit(0);
}

const minCount = Number(args["min-count"] ?? "2");
if (!Number.isFinite(minCount) || minCount < 1 || !Number.isInteger(minCount)) {
  console.error("[mistakes:report] --min-count must be a positive integer");
  process.exit(1);
}

const ledger = readLedger();
const nowMs = Date.now();

const rows: ReportRow[] = [];
for (const [key, value] of Object.entries(ledger)) {
  if (key.startsWith("_")) continue;
  if (!isMistakeEntry(value)) continue;
  const ageDays = daysSince(value.lastSeen, nowMs);
  const weight = CATEGORY_WEIGHT[value.category] ?? 1.0;
  const recencyFactor = 1 / (1 + ageDays / 90);
  const score = value.count * weight * recencyFactor;
  rows.push({
    key,
    count: value.count,
    category: value.category,
    ageDays,
    score,
    promote: value.count >= minCount,
    symptom: value.symptom,
    prevention: value.prevention,
  });
}

rows.sort((left, right) => right.score - left.score);

if (args.json === "true") {
  console.log(JSON.stringify({ minCount, rows }, null, 2));
  process.exit(0);
}

if (rows.length === 0) {
  console.log(
    "[mistakes:report] Ledger is empty or not found. Nothing to report.",
  );
  process.exit(0);
}

console.log(
  `Mistake priority report (promote threshold: count >= ${minCount})`,
);
console.log();
for (const row of rows) {
  const age = Math.round(row.ageDays);
  const promoteTag = row.promote ? "⬆ PROMOTE" : "  track";
  console.log(
    `${promoteTag}  ${row.key}  score=${row.score.toFixed(2)}  count=${row.count}  cat=${row.category}  age=${age}d`,
  );
  console.log(`          symptom:    ${row.symptom}`);
  console.log(`          prevention: ${row.prevention}`);
}

const promoteRows = rows.filter((row) => row.promote);
console.log();
if (promoteRows.length === 0) {
  console.log("No promotion candidates yet (all counts below threshold).");
} else {
  console.log(`${promoteRows.length} promotion candidate(s):`);
  for (const row of promoteRows) {
    console.log(`  → ${row.key} (count=${row.count}): ${row.prevention}`);
  }
}
