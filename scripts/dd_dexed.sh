#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(find_project_root "$SCRIPT_PATH")"
PYTHON_BIN="$(resolve_python_bin)"

cd "$PROJECT_ROOT"

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name dd_train_dexed \
  ckpt_path=outputs/dd_dexed/2026-03-31_22-41-11/checkpoints/checkpoint_last.pt

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name dd_train_dexed \
  seed=2026 \
  experiment.name=dd_dexed_seed2026
