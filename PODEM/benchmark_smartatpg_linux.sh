#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_11d_co_nobuf_all16_8rounds_bt2000"

exec "${PYTHON:-python}" -u scripts/run_smartatpg_benchmark_linux.py \
  "$TRAINING_DIR/benchmark_bundle" \
  --output-dir benchmark_results_11d_co_nobuf_all16_8rounds_bt2000 \
  --repeats 5 \
  --backtrack-limit 2000 \
  --seed 14 \
  "$@"
