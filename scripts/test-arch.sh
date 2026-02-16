#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Architectural mode: strict core invariants ==="
pnpm vitest run test/refcount.test.ts test/transform-compositions.test.ts

echo ""
echo "=== Architectural mode: full suite with expected-failure manifest ==="
set +e
pnpm vitest run --reporter=json --outputFile tmp/vitest-report.json
VITEST_EXIT=$?
set -e

if [ "$VITEST_EXIT" -ne 0 ]; then
  echo "Vitest reported failures; validating against .ci/expected-failures.json..."
fi

node scripts/check-test-status.mjs \
  --mode arch \
  --report tmp/vitest-report.json \
  --manifest .ci/expected-failures.json

echo ""
echo "=== Architectural mode checks passed ==="
