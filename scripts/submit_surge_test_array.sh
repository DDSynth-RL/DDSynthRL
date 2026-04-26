#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

PROJECT_ROOT="$(find_project_root "${SCRIPT_PATH}")"
LOG_DIR="${PROJECT_ROOT}/logs"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/data/surge_dataset_recipe.yaml}"

mkdir -p "${LOG_DIR}"

sbatch \
  -p i64m512u \
  -J surge_test \
  -n 1 \
  --cpus-per-task=1 \
  --time=7-00:00:00 \
  --array=0-0 \
  -o "${LOG_DIR}/surge_test_%A_%a.out" \
  -e "${LOG_DIR}/surge_test_%A_%a.err" \
  -D "${PROJECT_ROOT}" \
  --export=ALL,PYTHON_BIN="${PYTHON_BIN}",CONFIG_PATH="${CONFIG_PATH}" \
  --wrap "bash '${PROJECT_ROOT}/scripts/generate_surge_shard.sh' test \${SLURM_ARRAY_TASK_ID}"
