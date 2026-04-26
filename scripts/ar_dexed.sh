#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(find_project_root "$SCRIPT_PATH")"
PYTHON_BIN="$(resolve_python_bin)"

cd "$PROJECT_ROOT"

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name ar_train_dexed \
  ckpt_path=outputs/ar_dexed/2026-03-31_17-08-33/checkpoints/checkpoint_last.pt

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name ar_train_dexed \
  seed=6062 \
  experiment.name=ar_dexed_seed6062
# seed=1234(by default) & 4321 & 2026 & 6062

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name ar_train_dexed \
  model.order_path=configs/order/dexed_autoregressive_order_random.yaml \
  seed=6062 \
  experiment.name=ar_dexed_random_seed6062

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name ar_train_dexed \
  model.order_path=configs/order/dexed_autoregressive_order_random.yaml \
  experiment.name=ar_dexed_random

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name ar_train_dexed \
  model.order_path=configs/order/dexed_autoregressive_order_midi_last.yaml \
  experiment.name=ar_dexed_midi_last
