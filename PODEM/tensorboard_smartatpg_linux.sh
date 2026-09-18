#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_top30_hard_2rounds_batch8_bt100"

exec "${PYTHON:-python}" -m tensorboard.main \
  --logdir "$TRAINING_DIR" \
  --host 0.0.0.0 \
  --port 6006 \
  "$@"
