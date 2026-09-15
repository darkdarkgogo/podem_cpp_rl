#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_top30_hard_2rounds_batch8_bt200"

exec "${PYTHON:-python}" -u scripts/run_smartatpg_benchmark_linux.py \
  "$TRAINING_DIR/benchmark_bundle" \
  --output-dir benchmark_results_top30_hard_2rounds_batch8_bt200 \
  --repeats 5 \
  --backtrack-limit 200 \
  --seed 14 \
  "$@"
