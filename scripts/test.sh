#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: scripts/test.sh <checkpoint> [evaluation arguments...]" >&2
  exit 1
fi

CKPT_PATH="$1"
shift

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(find_project_root "$SCRIPT_PATH")"
PYTHON_BIN="$(resolve_python_bin)"

cd "$PROJECT_ROOT"
exec env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt "$CKPT_PATH" "$@"
