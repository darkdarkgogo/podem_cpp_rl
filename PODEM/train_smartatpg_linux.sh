#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_data_split_5rounds_bt200"

exec "${PYTHON:-python}" -u scripts/run_smartatpg_training_linux.py \
  --output-dir "$TRAINING_DIR" \
  --dataset-root data \
  --rounds 5 \
  --backtrack-limit 200 \
  --seed 2026 \
  --gpu 0 \
  "$@"
