#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_12d_co_20rounds_bt2000"
LEGACY_TRAINING_DIR="artifacts/smartatpg_12d_co_30rounds_bt2000"
if [[ -d "$LEGACY_TRAINING_DIR" && ! -d "$TRAINING_DIR" ]]; then
  TRAINING_DIR="$LEGACY_TRAINING_DIR"
fi

exec "${PYTHON:-python}" -m tensorboard.main \
  --logdir "$TRAINING_DIR" \
  --host 0.0.0.0 \
  --port 6006 \
  "$@"
