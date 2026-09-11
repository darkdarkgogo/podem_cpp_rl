#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_12d_co_all16_8rounds_bt2000"

exec "${PYTHON:-python}" -u scripts/run_smartatpg_training_linux.py \
  --output-dir "$TRAINING_DIR" \
  --rounds 8 \
  --backtrack-limit 2000 \
  --seed 2026 \
  --gpu 0 \
  "$@"
