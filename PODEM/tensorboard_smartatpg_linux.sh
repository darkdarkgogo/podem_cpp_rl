#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec "${PYTHON:-python}" -m tensorboard.main \
  --logdir artifacts/smartatpg_12d_co_30rounds_bt2000 \
  --host 0.0.0.0 \
  --port 6006 \
  "$@"
