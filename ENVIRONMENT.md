# DDSynth-RL Environment (Synth Backends + H5 Dataset Generation / Loading)

This document tracks runtime dependencies required by:

- `src/data/synth_backends/dexed/dexed_bridge.py`
- `src/data/synth_backends/dexed/dexed_renderer.py`
- `src/data/synth_backends/surge/surge_bridge.py`
- `src/data/synth_backends/surge/surge_renderer.py`
- `src/data/h5_synth_dataset.py`
- `src/data/h5_synth_datamodule.py`
- `src/utils/generate_surge_h5_dataset.py`
- `src/utils/merge_surge_h5_shards.py`
- `src/utils/pack_dexed_h5_dataset.py`
- `src/train.py`
- `src/finetune_grpo.py`
- `src/grpo/renderer_pool.py`
- `src/models/autoregressive_module.py`
- `src/models/discrete_diffusion_module.py`
- `src/models/components/transformer_autoregressive.py`
- `src/models/components/transformer_discrete_diffusion.py`
- `src/utils/audio_metrics.py`

## Required Dependencies

- Python `3.10.19`
- `numpy==1.26.4`
- `PyYAML==6.0.3`
- `hydra-core==1.3.2`
- `omegaconf==2.3.0`
- `dawdreamer==0.8.3`
- `pedalboard==0.9.9`
- `torch==2.9.0`
- `torchaudio==2.9.0`
- `torchvision==0.24.0`
- `lightning==2.6.0`
- `h5py==3.15.1`
- `librosa==0.10.2.post1`
- `wandb==0.23.0`
- `fadtk==1.1.0`
- `encodec==0.1.1`
- `resampy==0.4.3`
- `einops==0.8.2`
- `transformers==4.57.1`
- `pandas==2.3.3`
- `nnaudio==0.3.3`
- `laion-clap==1.1.7`
- `hypy-utils==1.0.29`
- `msclap==1.3.4`
- `torchcrepe==0.0.24`

`numpy` and `PyYAML` are aligned with the existing `synth-matching` environment to reduce migration risk.
`dawdreamer` version is aligned with the existing `AR-Matching` environment.
`pedalboard` is pinned to `0.9.9` to stay compatible with GLIBC 2.31 hosts (manylinux2014 wheel) while correctly loading Surge XT as an instrument.
`hydra-core/omegaconf` are required by the stage-1 training entrypoint and model configs.
`torch/torchaudio/torchvision/lightning/h5py/librosa` are required to run the shared H5 dataset + Lightning datamodule stack, the H5 dataset packing/generation scripts, and the discrete-diffusion training path.
`wandb` is required by the default DD logger config.
`fadtk` is required by the default DD `clap` metric.
`encodec/resampy/einops/transformers/pandas/nnaudio/laion-clap/hypy-utils/msclap` are pinned explicitly because they are part of the `fadtk` runtime chain used by the default DD validation stack.
`torchcrepe` is required by the default `GRPO` reward stack because `reward.weights.crepe=12.0` is enabled in `configs/finetune/dd_grpo.yaml`.

## Reproducible Setup

From the project root (`DDSynth-RL`):

```bash
cd DDSynth-RL
conda env create -f environment.yaml
conda activate DDSynth-RL
```

`environment.yaml` intentionally keeps conda deps minimal (`python` + `pip`) and installs runtime packages through pip pins. This keeps solver pressure low and speeds up environment creation.

If you prefer pip-only installation in an existing environment:

```bash
cd DDSynth-RL
python -m pip install -r requirements.txt
```

## Notes

- `requirements.txt` is the pip source of truth for package pins.
- `environment.yaml` is the reproducible entry point for creating the `DDSynth-RL` conda env.
- Recipe-driven dataset generation currently uses:
  - `configs/data/surge_dataset_recipe.yaml`
  - `configs/data/dexed_dataset_recipe.yaml`
- These recipe YAMLs are the explicit build-time config entry points for dataset generation.
- Generated H5 datasets must be self-describing through frozen `dataset_metadata.json`; training/reading should not depend on re-supplying external YAML.
- As new modules (renderer, vst host, packing/generation, training stack) are added, update both files in the same commit.
- Installing `dawdreamer` does not guarantee a given plugin path is loadable; plugin binary type/path must match DawDreamer plugin loading expectations.
- Installing `pedalboard` from an unpinned/newer wheel can fail on older clusters with `GLIBC_2.35 not found`.

## Current Dataset Entry Points

Surge XT shard generation:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.utils.generate_surge_h5_dataset \
  --config configs/data/surge_dataset_recipe.yaml \
  --split train \
  --shard-index 0
```

Surge XT shard merge:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.utils.merge_surge_h5_shards \
  --config configs/data/surge_dataset_recipe.yaml
```

Dexed JSON/WAV packing:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.utils.pack_dexed_h5_dataset \
  --config configs/data/dexed_dataset_recipe.yaml
```

Before running the Dexed packer, place the raw source files under the project root:

- `dataset/dexed/raw/json`: input preset JSON files
- `dataset/dexed/raw/audio`: matching WAV files with the same stem names

The default Dexed recipe now assumes those repo-relative locations. If those directories are empty, packing will fail immediately with a clear error instead of guessing another source directory.

## Current Training Entry Points

Default stage-1 DD configs:

- `dd_train_dexed` is the Dexed stage-1 DD default.
- It assumes:
  - `data=h5_dexed`
  - `weight=finetune_3`
  - `experiment.name=dd_dexed`
  - `model.validation.log_per_param_loss=true`
  - in-domain validation limited to the first `8` validation batches (`trainer.limit_val_batches=8`)
  - render-backed validation enabled on those same `8` validation batches (`model.validation.render_batches=8`)
  - OOD validation enabled with `model.validation.nsynth_eval.enable=true`
  - OOD validation limited to `8` NSynth batches (`model.validation.nsynth_eval.batches=8`)
  - repo-local NSynth validation audio under `dataset/nsynth/valid/audio`

- `dd_train_surge` is the Surge XT stage-1 DD default.
- It assumes:
  - `data=h5_surge`
  - `weight=default`
  - `experiment.name=dd_surge`
  - the same `8/8` in-domain and OOD validation limits as Dexed
  - repo-local NSynth validation audio under `dataset/nsynth/valid/audio`

The direct equivalent of the old Dexed AR-Matching command is now:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name dd_train_dexed
```

If `dataset/nsynth/valid/audio` is missing, `dd_train_dexed` will now fail immediately instead of silently skipping OOD validation. Disable it explicitly only when you intentionally want a no-NSynth run:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name dd_train_dexed \
  model.validation.nsynth_eval.enable=false
```

Stage-1 discrete-diffusion training on Surge uses the same shared training stack, but it must not inherit the Dexed-specific token weights. Use:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name dd_train_surge
```

Default stage-1 autoregressive config:

- `ar_train_dexed` is the Dexed stage-1 autoregressive default.
- It assumes:
  - `data=h5_dexed`
  - current frozen Dexed `25-bin` dataset semantics
  - `weight=finetune_3`
  - `experiment.name=ar_dexed`
  - the same lighter validation schedule as the DD defaults:
    - `trainer.limit_val_batches=8`
    - `model.validation.render_batches=8`
    - `model.validation.nsynth_eval.batches=8`
  - repo-local NSynth validation audio under `dataset/nsynth/valid/audio`

Run it with:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name ar_train_dexed
```

- `ar_train_surge` is the Surge XT stage-1 autoregressive default.
- It assumes:
  - `data=h5_surge`
  - `weight=default`
  - `experiment.name=ar_surge`
  - the same lighter validation schedule as the Dexed AR default:
    - `trainer.limit_val_batches=8`
    - `model.validation.render_batches=8`
    - `model.validation.nsynth_eval.batches=8`
  - repo-local NSynth validation audio under `dataset/nsynth/valid/audio`

Run it with:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name ar_train_surge
```

Default stage-1 flow-matching config:

- `fm_train_dexed` is the Dexed stage-1 flow-matching default.
- It assumes:
  - `data=h5_dexed`
  - `training.batch_size=32` to keep the per-step sample budget aligned with the current DD/AR defaults
  - current frozen Dexed dataset semantics for full parameters, with FM targets rebuilt on the fly from `full_parameters_backend + midi_norm`
  - `experiment.name=fm_dexed`
  - a larger `12/12` FM backbone (`encoder.n_layers=12`, `vector_field.num_layers=12`) to move it closer to the current DD/AR model scale
  - the same lighter validation schedule as the current DD/AR defaults:
    - `trainer.limit_val_batches=8`
    - `model.validation.render_batches=8`
    - `model.validation.nsynth_eval.batches=8`
  - repo-local NSynth validation audio under `dataset/nsynth/valid/audio`
  - top-k checkpointing on `val/wmfcc` and `val_nsynth/wmfcc` via the shared callback config

Run it with:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name fm_train_dexed
```

- `fm_train_surge` is the Surge XT stage-1 flow-matching default.
- It assumes:
  - `data=h5_surge`
  - `training.batch_size=32` to keep the per-step sample budget aligned with the current DD/AR defaults
  - current frozen Surge schema semantics lifted into a `300`-dimensional FM target space (`130` scalar synth params + `32` one-hot categorical groups + `3` MIDI scalars after expansion)
  - `experiment.name=fm_surge`
  - the same `12/12` FM backbone depth used for Dexed
  - the same lighter validation schedule as the current DD/AR/FM defaults:
    - `trainer.limit_val_batches=8`
    - `model.validation.render_batches=8`
    - `model.validation.nsynth_eval.batches=8`
  - repo-local NSynth validation audio under `dataset/nsynth/valid/audio`
  - top-k checkpointing on `val/wmfcc` and `val_nsynth/wmfcc`

Run it with:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name fm_train_surge
```

Default DD GRPO finetuning config:

- `finetune/dd_grpo` is the shared DD `GRPO` default.
- It assumes:
  - you pass `ckpt_path=...` explicitly at launch
  - `ref_ckpt_path` defaults to `ckpt_path`
  - the target checkpoint is a `DiscreteDiffusionModule` checkpoint
  - synth type is inferred from the checkpoint's frozen dataset metadata, so the same entrypoint supports both Dexed and Surge
  - default sampling policy:
    - `sampling.token.top_k=16`
    - `sampling.token.epsilon=0.7`
    - `sampling.pos.top_k=16`
    - `sampling.pos.epsilon=0.7`
  - default reward weights:
    - `wmfcc=1.2`
    - `clap=8.0`
    - `crepe=12.0`
    - `mss=1.8`
    - `sot=60.0`
    - `rms=4.0`
  - `grpo.renderer_workers=8`
  - NSynth prompt audio under:
    - `dataset/nsynth/train/audio`
    - `dataset/nsynth/valid/audio`

Run it directly with:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.finetune_grpo --config-name finetune/dd_grpo \
  ckpt_path=outputs/<run>/checkpoints/checkpoint_last.pt
```

If you want a separate frozen reference policy:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.finetune_grpo --config-name finetune/dd_grpo \
  ckpt_path=outputs/<policy-run>/checkpoints/checkpoint_last.pt \
  ref_ckpt_path=outputs/<reference-run>/checkpoints/checkpoint_last.pt
```

There is also a small helper script:

```bash
cd DDSynth-RL
CKPT_PATH=outputs/<run>/checkpoints/checkpoint_last.pt scripts/dd_grpo.sh
```

Optional:

- set `REF_CKPT_PATH=...` before running `scripts/dd_grpo.sh` if the reference checkpoint should differ from the policy checkpoint
- keep `logger.wandb.mode=offline` for quick smoke tests

Current validation status for `GRPO`:

- Dexed: single-update GPU smoke ran through policy load, reference load, renderer init, render, reward computation, metric logging, and `checkpoint_last.pt` write
- Surge: same single-update GPU smoke also ran through the full path
- The current `DDSynth-RL` environment now includes `torchcrepe`, so the default `crepe` reward path is active

For a CPU smoke test that only verifies the training path end-to-end, use a smaller temporary model and one fast-dev batch:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name dd_train_dexed \
  model.validation.nsynth_eval.enable=false \
  +trainer.fast_dev_run=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  data.training.num_workers=0 \
  data.training.persistent_workers=false \
  model.d_model=128 \
  model.nhead=4 \
  model.num_layers=2 \
  model.dim_feedforward=256
```

The default DD model config is much larger than this smoke-test override. On CPU-only hosts, the default width/depth can be killed by the scheduler or OS before the first validation batch finishes.

The DD default now includes render-backed validation and OOD NSynth validation. Validation/test dataloaders already read raw `audio`, so enabling render validation does not require a second data config. Renderer defaults are repo-relative and synth-specific:

- Dexed uses `synth/Dexed.vst3`
- Surge uses `synth/Surge XT.vst3` and `presets/surge-base.vstpreset`

Minimal Dexed render-validation smoke test:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name dd_train_dexed \
  data=h5_dexed \
  +trainer.fast_dev_run=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  data.training.batch_size=1 \
  data.training.num_workers=0 \
  data.training.persistent_workers=false \
  model.validation.nsynth_eval.enable=false \
  model.d_model=128 \
  model.nhead=4 \
  model.num_layers=2 \
  model.dim_feedforward=256 \
  model.validation.render_batches=1 \
  model.validation.metrics.wmfcc=false \
  model.validation.metrics.mfcc13=true
```

Minimal Surge render-validation smoke test:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name dd_train_surge \
  +trainer.fast_dev_run=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  data.training.batch_size=1 \
  data.training.num_workers=0 \
  data.training.persistent_workers=false \
  model.validation.nsynth_eval.enable=false \
  model.d_model=128 \
  model.nhead=4 \
  model.num_layers=2 \
  model.dim_feedforward=256 \
  model.validation.render_batches=1 \
  model.validation.metrics.wmfcc=false \
  model.validation.metrics.mfcc13=true \
  model.renderer.surge.preset_load_flush_seconds=0.5 \
  model.renderer.surge.post_param_flush_seconds=0.5 \
  model.renderer.surge.post_render_flush_seconds=0.5
```

Equivalent Surge smoke test using the dedicated default config:

```bash
cd DDSynth-RL
PYTHONPATH=$(pwd) python -m src.train --config-name dd_train_surge \
  +trainer.fast_dev_run=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  data.training.batch_size=1 \
  data.training.num_workers=0 \
  data.training.persistent_workers=false \
  model.validation.nsynth_eval.enable=false \
  model.d_model=128 \
  model.nhead=4 \
  model.num_layers=2 \
  model.dim_feedforward=256 \
  model.validation.render_batches=1 \
  model.validation.metrics.wmfcc=false \
  model.validation.metrics.mfcc13=true \
  model.renderer.surge.preset_load_flush_seconds=0.5 \
  model.renderer.surge.post_param_flush_seconds=0.5 \
  model.renderer.surge.post_render_flush_seconds=0.5
```

Optional metric note:

- `logger=wandb` is part of the default DD stacks. For quick local verification without network logging, override `logger.wandb.mode=disabled` or disable the logger group entirely with `~logger`.
- `model.validation.metrics.clap=true` requires the pinned `fadtk` runtime chain from `requirements.txt`.
- DawDreamer may print an `invalid URI` warning when loading the Dexed VST3 bundle path; in the current environment that warning is noisy but non-fatal as long as training proceeds into the validation batch.
- The default trainer disables Lightning sanity validation (`num_sanity_val_steps=0`) so a randomly initialized model does not hit render-backed Dexed validation before the first training step.
