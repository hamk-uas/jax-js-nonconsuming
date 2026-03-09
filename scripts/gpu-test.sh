#!/bin/bash
#
# Run vitest (bench or run) on both NVIDIA and Intel GPUs sequentially.
#
# Examples:
#   scripts/gpu-test.sh run test/gpu-bench.test.ts
#   scripts/gpu-test.sh bench bench/matmul.bench.ts
#   scripts/gpu-test.sh run                           # all tests
#   scripts/gpu-test.sh bench                          # all benches
#
# GPU selection:
#   GPU=nvidia scripts/gpu-test.sh bench bench/matmul.bench.ts   # NVIDIA only
#   GPU=intel  scripts/gpu-test.sh bench bench/matmul.bench.ts   # Intel only
#   GPU=both   scripts/gpu-test.sh bench bench/matmul.bench.ts   # both (default)
#
set -euo pipefail
cd "$(dirname "$0")/.."

gpu="${GPU:-both}"
cmd="${1:-run}"
shift || true

nvidia_cfg="test/vitest.nvidia.config.ts"
intel_cfg="test/vitest.intel.config.ts"

run_gpu() {
  local label="$1" cfg="$2"
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  GPU: $label"
  echo "  Config: $cfg"
  echo "  Command: pnpm vitest $cmd $* -c $cfg"
  echo "════════════════════════════════════════════════════════════"
  echo ""
  pnpm vitest "$cmd" "$@" -c "$cfg"
}

exit_code=0

if [[ "$gpu" == "nvidia" || "$gpu" == "both" ]]; then
  run_gpu "NVIDIA" "$nvidia_cfg" "$@" || exit_code=$?
fi

if [[ "$gpu" == "intel" || "$gpu" == "both" ]]; then
  run_gpu "Intel" "$intel_cfg" "$@" || exit_code=$?
fi

if [[ "$gpu" != "nvidia" && "$gpu" != "intel" && "$gpu" != "both" ]]; then
  echo "Error: GPU must be 'nvidia', 'intel', or 'both' (got '$gpu')"
  exit 1
fi

exit $exit_code
