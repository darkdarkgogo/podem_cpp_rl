#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_12d_co_20rounds_bt2000"
LEGACY_TRAINING_DIR="artifacts/smartatpg_12d_co_30rounds_bt2000"
if [[ -d "$LEGACY_TRAINING_DIR" && ! -d "$TRAINING_DIR" ]]; then
  TRAINING_DIR="$LEGACY_TRAINING_DIR"
fi

exec "${PYTHON:-python}" -u scripts/run_smartatpg_training_linux.py \
  --output-dir "$TRAINING_DIR" \
  --rounds 20 \
  --backtrack-limit 2000 \
  --seed 2026 \
  --mean-gpu 0 \
  --gat-gpu 1 \
  "$@"
