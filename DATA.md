# Data provenance

DDSynth-RL does not redistribute the processed Dexed dataset. The source preset
database and upstream augmentation implementation are available from the
official [SPINVAE-2 repository](https://github.com/gwendal-lv/spinvae2). That
repository is licensed separately under AGPL-3.0.

## Dexed data used in the paper

The local paper artifact was constructed in two stages:

1. Start from 53,759 Dexed presets and apply SPINVAE-2's preset augmentation to
   obtain four parameter variations per preset (`var000` through `var003`).
2. Render every parameter variation under four sampled MIDI
   conditions. MIDI note spans 24--96, velocity spans 1--127, and duration uses
   128 quantized values from 0.20--3.00 seconds.

This gives 53,759 x 4 x 4 = 860,144 parameter-audio pairs. Splits are grouped by
source preset so that no base preset occurs in more than one split:

| Split | Samples |
|---|---:|
| Train | 774,128 |
| Validation | 42,992 |
| Test | 43,024 |

The HDF5 files contain `audio`, `mel_spec`, `parameters`, `midi`, and
`preset_id`. MIDI values are stored in normalized form; their ranges and
cardinalities are frozen in `dataset_metadata.json`.

## Reproduction boundary

The upstream SPINVAE-2 implementation contains the four-way preset variation
logic. DDSynth-RL adds randomized MIDI note, velocity, and duration rendering.
The exact second-stage generation program used to create the paper artifact is
not present in this repository, so the repository does not claim bit-exact data
regeneration from the upstream database alone.

For already-rendered matching JSON/WAV pairs, create the training HDF5 files
with:

```bash
python -m src.utils.pack_dexed_h5_dataset \
  --config configs/data/dexed_dataset_recipe.yaml
```

Place the inputs under `dataset/dexed/raw/json` and
`dataset/dexed/raw/audio`. The packer creates the train/validation/test splits,
mel spectrograms, statistics, and frozen dataset metadata; it does not perform
preset augmentation or audio rendering.
