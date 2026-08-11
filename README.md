# DDSynth-RL

Official implementation of **DDSynth-RL: Audio Synthesizer Inversion via
Discrete Diffusion with Reinforcement Learning**, accepted at **ISMIR 2026**.

[[Paper]](https://arxiv.org/abs/2608.03032) ·
[[Demo]](https://ddsynth-rl.github.io/DDSynthRL-Demo/) ·
[[Checkpoints]](https://huggingface.co/MINNE-WU/DDSynth-RL)

DDSynth-RL treats synthesizer inversion as conditional generation over
discrete synthesizer and MIDI parameters. It first trains a masked discrete
diffusion model with supervised objectives, then fine-tunes it with GRPO using
audio-domain rewards from a non-differentiable Dexed renderer. The repository
also includes the autoregressive and flow-matching baselines from the paper.

## Installation

The reference environment uses Python 3.10 and PyTorch 2.9:

```bash
conda env create -f environment.yaml
conda activate DDSynth-RL
```

Download [Dexed](https://asb2m10.github.io/dexed/) separately and place the
VST3 plugin at:

```text
synth/Dexed.vst3
```

## Checkpoints

Install the Hugging Face CLI and download all eight checkpoints used by the
demo:

```bash
python -m pip install -U huggingface_hub
hf download MINNE-WU/DDSynth-RL --local-dir checkpoints/ddsynth-rl
```

The Hugging Face repository contains supervised AR, flow-matching, and discrete
diffusion checkpoints selected for in-domain and OOD evaluation, plus the two
GRPO checkpoints.

## Data

DDSynth-RL expects the following project-relative layout:

```text
dataset/
├── dexed/
│   ├── train.h5
│   ├── val.h5
│   ├── test.h5
│   ├── dataset_metadata.json
│   └── stats.npz
└── nsynth/
    ├── train/audio/*.wav
    ├── valid/audio/*.wav
    └── test/audio/*.wav
```

The processed Dexed dataset is not redistributed by this repository. Download
the source preset resources from the official
[SPINVAE-2 repository](https://github.com/gwendal-lv/spinvae2). The paper data
uses SPINVAE-2's four-way preset variation strategy and renders each variation
under four randomized MIDI note, velocity, and duration conditions. See
[DATA.md](DATA.md) for the exact artifact statistics, provenance, and current
reproduction boundary.

`src/utils/pack_dexed_h5_dataset.py` converts matching rendered JSON/WAV pairs
to the HDF5 layout above; it does not generate the augmented renders.

For OOD training and evaluation, download the official NSynth `json/wav`
train, valid, and test splits from the
[NSynth dataset page](https://magenta.tensorflow.org/datasets/nsynth). The code
uses the original split names and only requires each split's `audio/` folder.

## Evaluation

Evaluate a downloaded checkpoint on both Dexed and NSynth:

```bash
bash scripts/test.sh checkpoints/ddsynth-rl/ddsynth_rl_multi_reward.pt \
  --max-in-domain 200 \
  --max-ood 200
```

Results are written under `render/`. Run `python -m src.test --help` for audio
paths, sampling controls, reward weights, and output options.

## Training

Train the three supervised models:

```bash
bash scripts/ar_dexed.sh
bash scripts/dd_dexed.sh
bash scripts/fm_dexed.sh
```

Each script accepts Hydra overrides. For example:

```bash
bash scripts/dd_dexed.sh seed=2026 experiment.name=dd_dexed_seed2026
```

Fine-tune a supervised discrete diffusion checkpoint with GRPO:

```bash
CKPT_PATH=checkpoints/ddsynth-rl/dd_dexed_best_ood.pt \
  bash scripts/dd_grpo.sh
```

Training runs are saved to `outputs/<experiment>/<timestamp>/`.

## Citation

```bibtex
@inproceedings{wu2026ddsynthrl,
  title     = {DDSynth-RL: Audio Synthesizer Inversion via Discrete Diffusion with Reinforcement Learning},
  author    = {Wu, Tristan and Chin, Daniel and Zhang, Junan and Jiang, Junyan and Jing, Yansen and Xia, Gus},
  booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference},
  year      = {2026},
  address   = {Abu Dhabi, UAE}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for components under separate
attribution terms. The released checkpoints and paper are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
