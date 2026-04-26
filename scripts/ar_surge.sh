#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(find_project_root "$SCRIPT_PATH")"
PYTHON_BIN="$(resolve_python_bin)"

cd "$PROJECT_ROOT"

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name ar_train_surge \
  trainer.limit_val_batches=2 \
  model.validation.render_batches=2 \
  model.validation.nsynth_eval.batches=2 \
  ckpt_path=outputs/ar_surge/2026-04-15_11-49-06/checkpoints/checkpoint_last.pt

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name ar_train_surge \
  seed=2026 \
  experiment.name=ar_surge_seed2026
