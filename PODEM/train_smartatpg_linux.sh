#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec "${PYTHON:-python}" -u scripts/run_smartatpg_training_linux.py \
  --output-dir artifacts/smartatpg_12d_co_30rounds_bt2000 \
  --rounds 30 \
  --backtrack-limit 2000 \
  --seed 2026 \
  "$@"
