#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec "${PYTHON:-python}" -u scripts/run_smartatpg_benchmark_linux.py \
  artifacts/smartatpg_12d_co_30rounds/benchmark_bundle \
  --output-dir benchmark_results_12d_co_30rounds \
  --repeats 5 \
  --backtrack-limit 500 \
  --seed 14 \
  "$@"
