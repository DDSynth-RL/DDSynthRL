#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(find_project_root "$SCRIPT_PATH")"
PYTHON_BIN="$(resolve_python_bin)"

cd "$PROJECT_ROOT"

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name dd_train_surge \
  trainer.limit_val_batches=2 \
  model.validation.render_batches=2 \
  model.validation.nsynth_eval.batches=2 \
  ckpt_path=outputs/dd_surge/2026-04-15_17-17-38/checkpoints/checkpoint_last.pt

# 2026-06-17: 在云南，发现huggingface连接出问题，暂时用镜像解决。
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/hf_cache
export HF_HUB_CACHE=~/hf_cache/hub
python -m src.train --config-name dd_train_surge \
  trainer.limit_val_batches=2 \
  model.validation.render_batches=2 \
  model.validation.nsynth_eval.batches=2 \
  ckpt_path=outputs/dd_surge/2026-04-15_17-17-38/checkpoints/checkpoint_last.pt

env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.train --config-name dd_train_surge \
  trainer.limit_val_batches=2 \
  model.validation.render_batches=2 \
  model.validation.nsynth_eval.batches=2 \
  seed=2026 \
  experiment.name=dd_surge_seed2026
