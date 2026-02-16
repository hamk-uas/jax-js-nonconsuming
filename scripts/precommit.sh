#!/bin/bash

set -euo pipefail

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
pnpm run lint:ownership:internal
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
  pnpm run test:deno
  pnpm run test:website:smoke
fi
