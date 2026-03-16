#!/bin/bash
# upstream-adapt.sh — Mechanical adaptations after cherry-picking from ekzhang/jax-js.
#
# Usage:
#   1. git cherry-pick <commit>
#   2. Resolve conceptual conflicts manually (if any)
#   3. ./scripts/upstream-adapt.sh [--features-keep-ours] [--fix-refs] [--fix-using] [--eslint-fix] [--all]
#   4. git add -A && git cherry-pick --continue
#
# Flags:
#   --features-keep-ours  Resolve FEATURES.md conflict by keeping HEAD (ours)
#   --fix-refs            Remove .ref() calls from staged library files
#   --fix-using           Convert const→using for array bindings in staged test files
#   --eslint-fix          Run ESLint --fix (no-nested-array-leak autofix) on staged files
#   --all                 All of the above
#   (no flags)            Same as --all
#
# This script is idempotent — safe to run multiple times.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# --- Parse flags ---
do_features=false
do_refs=false
do_using=false
do_eslint=false

if [[ $# -eq 0 ]]; then
  do_features=true; do_refs=true; do_using=true; do_eslint=true
fi

for arg in "$@"; do
  case "$arg" in
    --features-keep-ours) do_features=true ;;
    --fix-refs)           do_refs=true ;;
    --fix-using)          do_using=true ;;
    --eslint-fix)         do_eslint=true ;;
    --all)                do_features=true; do_refs=true; do_using=true; do_eslint=true ;;
    -h|--help)
      head -17 "$0" | tail -16
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      echo "Run with --help for usage." >&2
      exit 1
      ;;
  esac
done

# --- Collect changed files ---
# Works during both cherry-pick (--diff-filter on MERGE_HEAD) and after staging.
changed_files() {
  if [[ -f .git/CHERRY_PICK_HEAD ]]; then
    # During cherry-pick: files touched by the incoming commit
    git diff --name-only --diff-filter=ACMR HEAD 2>/dev/null || true
  else
    # After staging: changed from HEAD
    git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true
  fi
}

changed_test_files() {
  changed_files | grep -E '^test/.*\.ts$' || true
}

changed_src_files() {
  changed_files | grep -E '^src/.*\.ts$' || true
}

# ============================================================
# 1. FEATURES.md — keep ours on conflict
# ============================================================
if [[ "$do_features" == "true" ]]; then
  if git ls-files --unmerged FEATURES.md 2>/dev/null | grep -q .; then
    echo "[upstream-adapt] FEATURES.md: keeping HEAD version (resolve conflict)"
    git checkout --ours FEATURES.md
    git add FEATURES.md
    echo "  → Remember to manually add the new feature row from the upstream commit."
  elif [[ -f FEATURES.md ]] && grep -q '^<<<<<<' FEATURES.md 2>/dev/null; then
    echo "[upstream-adapt] FEATURES.md: conflict markers detected, keeping HEAD"
    git checkout --ours FEATURES.md
    git add FEATURES.md
    echo "  → Remember to manually add the new feature row from the upstream commit."
  else
    echo "[upstream-adapt] FEATURES.md: no conflict"
  fi
fi

# ============================================================
# 2. Remove .ref() — upstream uses ref-counting, we don't
# ============================================================
if [[ "$do_refs" == "true" ]]; then
  ref_count=0
  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    [[ -f "$file" ]] || continue
    # Remove .ref() calls (standalone method calls, not .ref used as property)
    # Patterns: foo.ref()  foo.ref();  .ref()  .ref();
    if grep -qE '\.ref\(\)' "$file"; then
      # Remove entire lines that are just .ref() calls (e.g., "  result.ref();\n")
      sed -i '/^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_.]*\.ref();[[:space:]]*$/d' "$file"
      # In chained expressions, remove .ref() (e.g., "return foo.ref();" → "return foo;")
      sed -i 's/\.ref()//g' "$file"
      ref_count=$((ref_count + 1))
    fi
  done < <(changed_src_files)
  if [[ $ref_count -gt 0 ]]; then
    echo "[upstream-adapt] Removed .ref() from $ref_count source file(s)"
  else
    echo "[upstream-adapt] No .ref() calls found in changed source files"
  fi
fi

# ============================================================
# 3. const → using for array bindings in test files
# ============================================================
if [[ "$do_using" == "true" ]]; then
  using_count=0
  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    [[ -f "$file" ]] || continue
    # Convert "const x = np.<factory>(" → "using x = np.<factory>("
    # and     "const x = <expr>.method(" → "using x = <expr>.method("
    # This is intentionally conservative — only converts lines matching
    # common array-producing patterns. Manual review still needed.
    if grep -qE 'const [a-zA-Z_]+ = (np\.|lax\.|nn\.|random\.|[a-zA-Z_]+\.(add|sub|mul|div|dot|matmul|reshape|transpose|slice|neg|abs|exp|log|sum|mean|max|min)\()' "$file"; then
      sed -i -E 's/^([[:space:]]*)const ([a-zA-Z_]+) = (np\.|lax\.|nn\.|random\.)/\1using \2 = \3/g' "$file"
      sed -i -E 's/^([[:space:]]*)const ([a-zA-Z_]+) = ([a-zA-Z_]+\.(add|sub|mul|div|dot|matmul|reshape|transpose|slice|neg|abs|exp|log|sum|mean|max|min)\()/\1using \2 = \3/g' "$file"
      using_count=$((using_count + 1))
    fi
  done < <(changed_test_files)
  if [[ $using_count -gt 0 ]]; then
    echo "[upstream-adapt] Converted const→using in $using_count test file(s)"
    echo "  → Review changes: some const bindings may be intentionally non-disposing"
  else
    echo "[upstream-adapt] No const→using conversions needed"
  fi
fi

# ============================================================
# 4. ESLint autofix (no-nested-array-leak)
# ============================================================
if [[ "$do_eslint" == "true" ]]; then
  eslint_files=()
  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    [[ -f "$file" ]] || continue
    eslint_files+=("$file")
  done < <(changed_files | grep -E '\.ts$' || true)

  if [[ ${#eslint_files[@]} -gt 0 ]]; then
    echo "[upstream-adapt] Running ESLint --fix on ${#eslint_files[@]} file(s)..."
    # Uses the project's eslint.config.ts which includes no-nested-array-leak
    npx eslint --fix "${eslint_files[@]}" 2>/dev/null || true
    echo "[upstream-adapt] ESLint autofix complete"
  else
    echo "[upstream-adapt] No TypeScript files to lint"
  fi
fi

echo ""
echo "[upstream-adapt] Done. Review changes, then:"
echo "  git add -A && git cherry-pick --continue"
