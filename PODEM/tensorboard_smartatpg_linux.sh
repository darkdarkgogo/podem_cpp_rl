#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAINING_DIR="artifacts/smartatpg_11d_co_nobuf_all16_8rounds_bt2000"

exec "${PYTHON:-python}" -m tensorboard.main \
  --logdir "$TRAINING_DIR" \
  --host 0.0.0.0 \
  --port 6006 \
  "$@"
