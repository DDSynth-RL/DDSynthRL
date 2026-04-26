#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: bash scripts/generate_surge_shard.sh <train|val|test> <shard_index>" >&2
  exit 1
fi

SPLIT="$1"
SHARD_INDEX="$2"
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

PROJECT_ROOT="$(find_project_root "${SCRIPT_PATH}")"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/data/surge_dataset_recipe.yaml}"
PYTHON_BIN="$(resolve_python_bin)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON_BIN}" -m src.utils.generate_surge_h5_dataset \
  --config "${CONFIG_PATH}" \
  --split "${SPLIT}" \
  --shard-index "${SHARD_INDEX}"
