#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(find_project_root "$SCRIPT_PATH")"
PYTHON_BIN="$(resolve_python_bin)"

cd "$PROJECT_ROOT"

# test grpo_full
# OOD
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/dd_grpo_resume_with_crepe_23_8_workers/2026-03-29_15-48-12/checkpoints/checkpoint_best_reward_1_step_00130800.pt \
  --max-in-domain 1 \
  --max-ood 200
# In-domain
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/dd_grpo_resume_with_crepe_23_8_workers/2026-03-29_15-48-12/checkpoints/checkpoint_best_reward_1_step_00130800.pt \
  --max-in-domain 200 \
  --max-ood 1

# test grpo_clap_crepe
# OOD
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/dd_grpo_resume_with_crepe_clap_only_2/2026-04-13_18-34-46/checkpoints/checkpoint_best_reward_1_step_00134200.pt \
  --max-in-domain 1 \
  --max-ood 200
# In-domain
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/dd_grpo_resume_with_crepe_clap_only_2/2026-04-13_18-34-46/checkpoints/checkpoint_best_reward_1_step_00134200.pt \
  --max-in-domain 200 \
  --max-ood 1

# test fm_dexed
# OOD
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/fm_dexed/2026-04-16_14-41-57/checkpoints/checkpoint_best_val_nsynth_wmfcc_step_02221472_metric_15.654646.pt \
  --max-in-domain 1 \
  --max-ood 200
# In-domain
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/fm_dexed/2026-04-16_14-41-57/checkpoints/checkpoint_best_val_nsynth_wmfcc_step_02995616_metric_15.635098.pt \
  --max-in-domain 200 \
  --max-ood 1

# test dd_dexed
# OOD
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/dd_dexed/2026-03-26_14-20-11/checkpoints/checkpoint_best_val_nsynth_wmfcc_step_00077576_metric_11.424175.pt \
  --max-in-domain 1 \
  --max-ood 200
# In-domain
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/dd_dexed/2026-04-06_14-50-56/checkpoints/checkpoint_best_val_wmfcc_step_01238792_metric_4.577944.pt \
  --max-in-domain 200 \
  --max-ood 1

# test ar_dexed
# OOD
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/ar_dexed/2026-03-28_09-27-58/checkpoints/checkpoint_best_val_nsynth_wmfcc_step_00077576_metric_12.170912.pt \
  --max-in-domain 1 \
  --max-ood 200
# In-domain
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/ar_dexed/2026-04-06_14-46-40/checkpoints/checkpoint_best_val_wmfcc_step_01383944_metric_4.140687.pt \
  --max-in-domain 200 \
  --max-ood 1

# test dd_surge
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/dd_surge/2026-04-13_18-37-51/checkpoints/checkpoint_best_val_nsynth_wmfcc_step_00817500_metric_15.412776.pt \
  --max-in-domain 1 \
  --max-ood 200

# test ar_surge
env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m src.test \
  --ckpt outputs/ar_surge/2026-04-15_17-18-21/checkpoints/checkpoint_best_val_nsynth_wmfcc_step_00847500_metric_18.769318.pt \
  --max-in-domain 1 \
  --max-ood 200
