import {
  type MistakeCategory,
  type MistakeEntry,
  parseArgs,
  readLedger,
  VALID_CATEGORIES,
  writeLedger,
} from "./lib/mistakes";

function help(): void {
  console.log(`Usage:
  pnpm run mistakes:log -- --key <id> [--count <n>] [--category <cat>] [--symptom "..."] [--prevention "..."]

Examples:
  pnpm run mistakes:log -- --key assumed-linear-session-state
  pnpm run mistakes:log -- --key forgot-check-timings --category missed-check \\
    --symptom "Skipped timing validation" \\
    --prevention "Run pnpm run check:timings before handoff"

Notes:
  - If the key exists, count is incremented (default +1, override with --count).
  - If the key is new, --category, --symptom, and --prevention are required.
  - Ledger path: tmp/copilot-mistakes.json
`);
}

const args = parseArgs(process.argv.slice(2));
if (args.help === "true" || args.h === "true") {
  help();
  process.exit(0);
}

const key = args.key;
if (!key) {
  console.error("[mistakes:log] Missing --key");
  help();
  process.exit(1);
}

const step = Number(args.count ?? "1");
if (!Number.isFinite(step) || step < 1 || !Number.isInteger(step)) {
  console.error("[mistakes:log] --count must be a positive integer");
  process.exit(1);
}

const ledger = readLedger();
const now = new Date().toISOString();
const existing = ledger[key] as MistakeEntry | undefined;

if (existing && typeof existing === "object" && "count" in existing) {
  existing.count += step;
  existing.lastSeen = now;
  ledger[key] = existing;
} else {
  const category = args.category;
  const symptom = args.symptom;
  const prevention = args.prevention;

  if (!category || !VALID_CATEGORIES.has(category)) {
    console.error("[mistakes:log] New key requires valid --category");
    console.error(
      `[mistakes:log] Allowed: ${Array.from(VALID_CATEGORIES).join(", ")}`,
    );
    process.exit(1);
  }
  if (!symptom || !prevention) {
    console.error("[mistakes:log] New key requires --symptom and --prevention");
    process.exit(1);
  }

  ledger[key] = {
    count: step,
    lastSeen: now,
    category: category as MistakeCategory,
    symptom,
    prevention,
  } satisfies MistakeEntry;
}

writeLedger(ledger);

const updated = ledger[key] as MistakeEntry;
console.log(
  `[mistakes:log] ${key} -> count=${updated.count}, lastSeen=${updated.lastSeen}`,
);
