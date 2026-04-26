#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(find_project_root "$SCRIPT_PATH")"
PYTHON_BIN="$(resolve_python_bin)"

if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "Set CKPT_PATH to the DD checkpoint you want to finetune." >&2
  echo "Example:" >&2
  echo "  CKPT_PATH=outputs/dd_dexed/.../checkpoints/checkpoint_last.pt scripts/dd_grpo.sh" >&2
  exit 1
fi

ARGS=("ckpt_path=${CKPT_PATH}")
if [[ -n "${REF_CKPT_PATH:-}" ]]; then
  ARGS+=("ref_ckpt_path=${REF_CKPT_PATH}")
fi

cd "$PROJECT_ROOT"
exec env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.finetune_grpo --config-name finetune/dd_grpo "${ARGS[@]}" "$@"
