#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec "${PYTHON:-python}" -u scripts/run_dual_smartatpg_training_linux.py \
  --output-dir artifacts/smartatpg_dual_top30_hard_2rounds_batch8_bt200 \
  --dataset-root data \
  --rounds 2 \
  --backtrack-limit 200 \
  --seed 2026 \
  --gat-gpu 0 \
  --mean-gpu 1 \
  "$@"
