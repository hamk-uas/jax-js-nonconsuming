#!/bin/bash

set -euo pipefail

# --- AEP (Agentic Evolution Protocol) gate ---
# On every commit the committer must first re-read the AEP in order to commit.
# DO NOT MISUSE! Read AEP first -- don't circumvent this check!
REQUIRED_HASH="a28bd144cdc6672c607309e4810aec66be3539ed"
INPUT_HASH=$(echo -n "${AEP:-}" | shasum | awk '{print $1}')
if [[ "$INPUT_HASH" != "$REQUIRED_HASH" ]]; then
  echo "AEP GATE: commit rejected"
  echo "Expected Hash: $REQUIRED_HASH"
  echo "Received Hash: $INPUT_HASH (from AEP=\"${AEP:-}\")"
  echo "Read the Agentic Evolution Protocol (AEP) in"
  echo ".github/copilot-instructions.md, then follow it."
  echo "Do NOT circumvent this check. Read again, don't memorize!"
  exit 1
fi

branch_name="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

if [[ -n "${JAX_PRECOMMIT_PROFILE:-}" ]]; then
  profile="${JAX_PRECOMMIT_PROFILE}"
else
  case "$branch_name" in
    main|master|release/*|hotfix/*)
      profile="full"
      ;;
    *)
      profile="feature"
      ;;
  esac
fi

echo "[pre-commit] branch=$branch_name profile=$profile"

# --- Docs-only fast path ---
# When only documentation files are staged (no code changes), skip build/test/lint
# and only run format + lint checks. Saves ~50s on main-branch doc-only commits.
staged_files="$(git diff --cached --name-only --diff-filter=ACMR)"
docs_only=true
while IFS= read -r f; do
  case "$f" in
    *.md|*.txt|.github/copilot-instructions.md|docs/*) ;;
    *) docs_only=false; break ;;
  esac
done <<< "$staged_files"

if [[ "$docs_only" == "true" && -n "$staged_files" ]]; then
  echo "[pre-commit] docs-only commit detected — running format + lint only"
  pnpm format:check
  pnpm lint --max-warnings 0
  exit 0
fi
# --- End docs-only fast path ---

if [[ "$profile" != "feature" && "$profile" != "full" ]]; then
  echo "[pre-commit] Invalid JAX_PRECOMMIT_PROFILE='$profile' (expected: feature|full)"
  exit 1
fi

arch_mode="${JAX_ARCH_MODE:-0}"
if [[ "$arch_mode" != "0" && "$arch_mode" != "1" ]]; then
  echo "[pre-commit] Invalid JAX_ARCH_MODE='$arch_mode' (expected: 0|1)"
  exit 1
fi

pnpm build
pnpm check
pnpm lint --max-warnings 0
pnpm format:check
pnpm run test:eslint-plugin
pnpm run lint:ownership:website

if [[ "$arch_mode" == "1" ]]; then
  pnpm run test:arch
else
  if [[ "$profile" == "feature" ]]; then
    pnpm vitest run test/refcount.test.ts test/transform-compositions.test.ts
  else
    pnpm run test:policy:strict
  fi
fi

if [[ "$profile" == "full" ]]; then
  pnpm run test:website:smoke
fi
