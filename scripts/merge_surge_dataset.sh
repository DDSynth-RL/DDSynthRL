#!/bin/bash
set -euo pipefail

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

"${PYTHON_BIN}" -m src.utils.merge_surge_h5_shards --config "${CONFIG_PATH}"
