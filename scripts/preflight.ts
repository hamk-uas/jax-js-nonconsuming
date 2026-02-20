import { execSync, spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";

type Context = "src" | "tests" | "docs" | "bench" | "release" | "general";

type Check = {
  id: string;
  command: string;
  reason: string;
};

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

function parseArgs(argv: string[]): Record<string, string> {
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

function help(): void {
  console.log(`Usage:
  pnpm run preflight
  pnpm run preflight -- --context src,timings
  pnpm run preflight -- --strict
  pnpm run preflight -- --dry

Behavior:
  - Auto-detects context from git changes when --context is omitted.
  - Runs minimal high-value checks for the detected contexts.
  - Use --strict to include heavier checks.
`);
}

function getChangedFiles(): string[] {
  try {
    const out = execSync(
      "git diff --name-only --cached && git diff --name-only",
      {
        cwd: root,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      },
    );
    const files = out
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    return Array.from(new Set(files));
  } catch {
    return [];
  }
}

function inferContexts(files: string[]): Set<Context> {
  const contexts = new Set<Context>();
  for (const file of files) {
    if (file.startsWith("src/")) contexts.add("src");
    if (file.startsWith("test/")) contexts.add("tests");
    if (file.startsWith("bench/")) contexts.add("bench");
    if (file.startsWith("scripts/")) contexts.add("general");
    if (file.startsWith("packages/")) contexts.add("src");
    if (file.endsWith(".md")) contexts.add("docs");
    if (file.startsWith(".github/")) contexts.add("docs");
    if (file.startsWith("website/")) contexts.add("general");
    if (
      file === "package.json" ||
      file === "vitest.config.ts" ||
      file === "tsconfig.json"
    )
      contexts.add("general");
  }
  if (contexts.size === 0) contexts.add("general");
  return contexts;
}

function parseContexts(raw: string | undefined): Set<Context> {
  if (!raw) return new Set();
  const out = new Set<Context>();
  for (const token of raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)) {
    if (
      token === "src" ||
      token === "tests" ||
      token === "docs" ||
      token === "bench" ||
      token === "release" ||
      token === "general"
    ) {
      out.add(token);
    } else {
      throw new Error(`Unknown context: ${token}`);
    }
  }
  return out;
}

function buildChecks(contexts: Set<Context>, strict: boolean): Check[] {
  const checks: Check[] = [];
  const add = (id: string, command: string, reason: string): void => {
    if (checks.some((item) => item.id === id)) return;
    checks.push({ id, command, reason });
  };

  if (
    contexts.has("src") ||
    contexts.has("general") ||
    contexts.has("release")
  ) {
    add(
      "lint",
      "pnpm run lint",
      "Catch disposal/memory issues and TypeScript lint regressions in src/",
    );
    add("check", "pnpm run check", "TypeScript type checking");
  }
  if (contexts.has("docs")) {
    // Docs-only changes still benefit from lint if src/ was also touched
  }
  if (
    strict &&
    (contexts.has("src") ||
      contexts.has("tests") ||
      contexts.has("general") ||
      contexts.has("release"))
  ) {
    add("test", "pnpm run test", "Run unit tests for code-level confidence");
  }
  if (strict && contexts.has("release")) {
    add("build", "pnpm run build", "Validate distribution build");
  }

  if (checks.length === 0) {
    add("lint", "pnpm run lint", "Default safety check");
  }

  return checks;
}

const args = parseArgs(process.argv.slice(2));
if (args.help === "true" || args.h === "true") {
  help();
  process.exit(0);
}

const strict = args.strict === "true";
const dry = args.dry === "true";

let contexts = new Set<Context>();
try {
  contexts = parseContexts(args.context);
} catch (error) {
  console.error(`[preflight] ${String(error)}`);
  process.exit(1);
}

const changedFiles = getChangedFiles();
if (contexts.size === 0) {
  contexts = inferContexts(changedFiles);
}

const checks = buildChecks(contexts, strict);

console.log(`[preflight] contexts: ${Array.from(contexts).join(", ")}`);
if (changedFiles.length > 0) {
  console.log(`[preflight] changed files: ${changedFiles.length}`);
}
console.log("[preflight] checks:");
for (const check of checks) {
  console.log(`- ${check.command}`);
  console.log(`  reason: ${check.reason}`);
}

if (dry) {
  console.log("[preflight] --dry enabled, not executing checks.");
  process.exit(0);
}

for (const check of checks) {
  console.log(`\n[preflight] running: ${check.command}`);
  const result = spawnSync(check.command, {
    cwd: root,
    shell: true,
    stdio: "inherit",
  });
  if (result.status !== 0) {
    console.error(`[preflight] failed: ${check.command}`);
    process.exit(result.status ?? 1);
  }
}

console.log("\n[preflight] all checks passed.");
