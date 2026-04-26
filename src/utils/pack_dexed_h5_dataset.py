"""Pack Dexed JSON/WAV pairs into frozen-schema HDF5 datasets for DDSynth-RL.

This script ports AR-Matching's Dexed dataset packing into DDSynth-RL's current
dataset format:
- parameter semantics freeze into dataset_metadata.json via parameter_schema
- MIDI is stored as normalized values and declared in metadata
- train/val/test splits are written as separate H5 files plus stats.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import librosa
import numpy as np
import yaml

from src.project_paths import (
    find_project_root,
    project_relative_string,
    require_project_relative_path,
    resolve_project_path,
)
from src.data.synth_backends.dexed.dexed_bridge import DexedParameterHelper


_PROJECT_ROOT = find_project_root(Path(__file__).resolve())

logger = logging.getLogger(__name__)


def _resolve_path(path_like: str | Path) -> Path:
    return resolve_project_path(path_like)


def _resolve_project_relative_path(path_like: str | Path, *, label: str, source: Path) -> Path:
    return require_project_relative_path(path_like, label=label, source=source)


def _project_relative_string(path: Path) -> str:
    return project_relative_string(path)


@dataclass
class MelStatsTracker:
    """Track mean/std of mel specs across samples."""

    sum: Optional[np.ndarray] = None
    sumsq: Optional[np.ndarray] = None
    count: int = 0

    def update(self, mel: np.ndarray) -> None:
        if self.sum is None:
            self.sum = np.zeros_like(mel, dtype=np.float64)
            self.sumsq = np.zeros_like(mel, dtype=np.float64)
        if mel.shape != self.sum.shape:
            raise ValueError(f"Inconsistent mel shape {mel.shape}, expected {self.sum.shape}")
        self.sum += mel
        self.sumsq += mel**2
        self.count += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sum is None or self.sumsq is None or self.count == 0:
            raise RuntimeError("No mel samples were collected for stats.")
        mean = self.sum / float(self.count)
        var = np.maximum(self.sumsq / float(self.count) - mean**2, 0.0)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class MelConfig:
    n_mels: int
    window_seconds: float
    frames_per_second: float
    window: str
    power_to_db_ref: str


@dataclass(frozen=True)
class DexedDatasetRecipe:
    output_root: Path
    json_dir: Path
    audio_dir: Path
    summary_path: Path
    num_samples: int
    sample_rate: int
    target_duration: float
    min_loudness: float
    seed: int
    train_ratio: float
    val_ratio: float
    split_strategy: str
    mel: MelConfig
    midi_keys: tuple[str, str, str]
    sample_batch_size: int


@dataclass
class DexedSample:
    preset_id: str
    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray
    midi_norm: np.ndarray


def _require_mapping(payload: Mapping[str, Any], key: str, source: Path) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Recipe {source} must define mapping `{key}`.")
    return value


def _require_str(payload: Mapping[str, Any], key: str, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Recipe {source} must define non-empty string `{key}`.")
    return value


def _require_int(payload: Mapping[str, Any], key: str, source: Path, *, minimum: Optional[int] = None) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Recipe {source} must define integer `{key}`.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Recipe {source} must define `{key} >= {minimum}`, got {value}.")
    return int(value)


def _require_float(
    payload: Mapping[str, Any],
    key: str,
    source: Path,
    *,
    minimum: Optional[float] = None,
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Recipe {source} must define numeric `{key}`.")
    value_f = float(value)
    if minimum is not None and value_f < minimum:
        raise ValueError(f"Recipe {source} must define `{key} >= {minimum}`, got {value_f}.")
    return value_f


def _load_dexed_recipe(config_path: Path) -> DexedDatasetRecipe:
    if not config_path.exists():
        raise FileNotFoundError(f"Dexed dataset recipe YAML not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Dexed dataset recipe must be a mapping: {config_path}")

    paths_cfg = _require_mapping(raw, "paths", config_path)
    dataset_cfg = _require_mapping(raw, "dataset", config_path)
    mel_cfg = _require_mapping(raw, "mel", config_path)
    midi_cfg = _require_mapping(raw, "midi", config_path)
    runtime_cfg = _require_mapping(raw, "runtime", config_path)

    midi_keys = midi_cfg.get("keys")
    if not isinstance(midi_keys, list) or len(midi_keys) != 3 or any(not isinstance(x, str) or not x for x in midi_keys):
        raise ValueError(f"Recipe {config_path} must define `midi.keys` as a 3-item string list.")

    recipe = DexedDatasetRecipe(
        output_root=_resolve_project_relative_path(
            _require_str(paths_cfg, "output_root", config_path),
            label="paths.output_root",
            source=config_path,
        ),
        json_dir=_resolve_project_relative_path(
            _require_str(paths_cfg, "json_dir", config_path),
            label="paths.json_dir",
            source=config_path,
        ),
        audio_dir=_resolve_project_relative_path(
            _require_str(paths_cfg, "audio_dir", config_path),
            label="paths.audio_dir",
            source=config_path,
        ),
        summary_path=_resolve_project_relative_path(
            _require_str(paths_cfg, "summary_path", config_path),
            label="paths.summary_path",
            source=config_path,
        ),
        num_samples=_require_int(dataset_cfg, "num_samples", config_path, minimum=0),
        sample_rate=_require_int(dataset_cfg, "sample_rate", config_path, minimum=1),
        target_duration=_require_float(dataset_cfg, "target_duration", config_path, minimum=1e-8),
        min_loudness=_require_float(dataset_cfg, "min_loudness", config_path),
        seed=_require_int(dataset_cfg, "seed", config_path, minimum=0),
        train_ratio=_require_float(dataset_cfg, "train_ratio", config_path, minimum=1e-8),
        val_ratio=_require_float(dataset_cfg, "val_ratio", config_path, minimum=1e-8),
        split_strategy=_require_str(dataset_cfg, "split_strategy", config_path),
        mel=MelConfig(
            n_mels=_require_int(mel_cfg, "n_mels", config_path, minimum=1),
            window_seconds=_require_float(mel_cfg, "window_seconds", config_path, minimum=1e-8),
            frames_per_second=_require_float(mel_cfg, "frames_per_second", config_path, minimum=1e-8),
            window=_require_str(mel_cfg, "window", config_path),
            power_to_db_ref=_require_str(mel_cfg, "power_to_db_ref", config_path),
        ),
        midi_keys=(str(midi_keys[0]), str(midi_keys[1]), str(midi_keys[2])),
        sample_batch_size=_require_int(runtime_cfg, "sample_batch_size", config_path, minimum=1),
    )

    if recipe.train_ratio + recipe.val_ratio >= 1.0:
        raise ValueError(
            f"Recipe {config_path} must satisfy train_ratio + val_ratio < 1.0, "
            f"got {recipe.train_ratio + recipe.val_ratio}."
        )
    if recipe.split_strategy != "preset_group":
        raise ValueError(
            f"Recipe {config_path} only supports split_strategy='preset_group', got {recipe.split_strategy!r}."
        )
    if recipe.mel.power_to_db_ref != "max":
        raise ValueError(
            f"Recipe {config_path} only supports mel.power_to_db_ref='max', got {recipe.mel.power_to_db_ref!r}."
        )
    if int(recipe.sample_rate / recipe.mel.frames_per_second) <= 0:
        raise ValueError(
            f"Recipe {config_path} produces invalid mel hop_length: "
            f"sample_rate={recipe.sample_rate}, frames_per_second={recipe.mel.frames_per_second}."
        )
    if int(recipe.sample_rate * recipe.mel.window_seconds) <= 0:
        raise ValueError(
            f"Recipe {config_path} produces invalid mel n_fft: "
            f"sample_rate={recipe.sample_rate}, window_seconds={recipe.mel.window_seconds}."
        )
    return recipe


def preset_group_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def split_by_preset_group(
    stems: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(int(seed))
    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(preset_group_from_stem(stem), []).append(stem)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    total_groups = len(group_ids)
    if total_groups < 3:
        raise RuntimeError(f"Need at least 3 preset groups to split, got {total_groups}.")

    train_groups = int(total_groups * float(train_ratio))
    val_groups = int(total_groups * float(val_ratio))
    if train_groups < 1:
        train_groups = 1
    remaining = total_groups - train_groups
    if remaining < 2:
        train_groups = total_groups - 2
        remaining = 2
    if val_groups < 1:
        val_groups = 1
    if val_groups > remaining - 1:
        val_groups = remaining - 1
    test_groups = total_groups - train_groups - val_groups
    if test_groups < 1:
        test_groups = 1
        if val_groups > 1:
            val_groups -= 1

    train_ids = set(group_ids[:train_groups])
    val_ids = set(group_ids[train_groups : train_groups + val_groups])

    split_map: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for gid, stems_in_group in groups.items():
        if gid in train_ids:
            split_map["train"].extend(stems_in_group)
        elif gid in val_ids:
            split_map["val"].extend(stems_in_group)
        else:
            split_map["test"].extend(stems_in_group)

    for split_name, subset in split_map.items():
        if not subset:
            raise RuntimeError(f"{split_name} split has no samples; adjust ratios or seed.")
    return split_map


def _discover_stems(json_dir: Path, audio_dir: Path, limit: int) -> list[str]:
    if not json_dir.is_dir():
        raise FileNotFoundError(
            f"Dexed json_dir not found: {json_dir}. Place raw Dexed JSON files under "
            f"{_project_relative_string(json_dir)}."
        )
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Dexed audio_dir not found: {audio_dir}. Place raw Dexed WAV files under "
            f"{_project_relative_string(audio_dir)}."
        )

    stems = sorted(
        p.stem for p in json_dir.glob("*.json") if (audio_dir / f"{p.stem}.wav").exists()
    )
    if not stems:
        raise RuntimeError(f"No matching JSON/WAV stems found under {json_dir} and {audio_dir}.")
    if limit > 0:
        stems = stems[: min(len(stems), int(limit))]
    return stems


def _load_audio(audio_path: Path, sample_rate: int, target_duration: float) -> np.ndarray:
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    target_len = int(sample_rate * target_duration)
    if audio.shape[-1] >= target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - audio.shape[-1]))
    return np.asarray(audio, dtype=np.float32, order="C")


def _make_spectrogram(audio: np.ndarray, sample_rate: int, mel_cfg: MelConfig) -> np.ndarray:
    n_fft = int(float(mel_cfg.window_seconds) * sample_rate)
    hop_length = int(sample_rate / float(mel_cfg.frames_per_second))
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=int(mel_cfg.n_mels),
        n_fft=n_fft,
        hop_length=hop_length,
        window=str(mel_cfg.window),
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db.astype(np.float32, copy=False)


def _compute_mel_frames(sample_rate: int, target_duration: float, mel_cfg: MelConfig) -> int:
    hop = int(sample_rate / float(mel_cfg.frames_per_second))
    audio_len = int(sample_rate * float(target_duration))
    return int(np.ceil(audio_len / hop)) + 1


def _fit_mel_frames(mel: np.ndarray, target_frames: int) -> np.ndarray:
    if mel.shape[1] > target_frames:
        return mel[:, :target_frames]
    if mel.shape[1] < target_frames:
        pad = np.repeat(mel[:, -1:], repeats=target_frames - mel.shape[1], axis=1)
        return np.concatenate([mel, pad], axis=1)
    return mel


def _load_json_params(
    json_path: Path,
    midi_keys: Sequence[str],
    param_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Dexed JSON must decode to a mapping: {json_path}")

    missing_midi = [key for key in midi_keys if key not in data]
    if missing_midi:
        raise KeyError(f"{json_path} missing required MIDI keys: {missing_midi}")

    midi = np.asarray([float(data[k]) for k in midi_keys], dtype=np.float32)
    if not np.isfinite(midi).all():
        raise ValueError(f"{json_path} MIDI contains NaN/Inf: {midi!r}")
    if np.any(midi < -1e-6) or np.any(midi > 1.0 + 1e-6):
        raise ValueError(
            f"{json_path} MIDI must already be normalized to [0,1], found {midi!r}"
        )

    missing_params = [name for name in param_names if name not in data]
    if missing_params:
        preview = ", ".join(missing_params[:8])
        raise KeyError(
            f"{json_path} missing {len(missing_params)} Dexed parameters required by summary "
            f"(examples: {preview})"
        )

    extra_params = [
        key for key in data.keys() if key not in midi_keys and key not in set(param_names)
    ]
    if extra_params:
        preview = ", ".join(extra_params[:8])
        raise KeyError(
            f"{json_path} contains {len(extra_params)} unexpected non-MIDI keys "
            f"(examples: {preview})"
        )

    params = np.asarray([float(data[name]) for name in param_names], dtype=np.float32)
    if not np.isfinite(params).all():
        raise ValueError(f"{json_path} parameters contain NaN/Inf values.")
    return midi, params


def _generate_sample(
    *,
    stem: str,
    json_dir: Path,
    audio_dir: Path,
    midi_keys: Sequence[str],
    param_names: Sequence[str],
    sample_rate: int,
    target_duration: float,
    mel_cfg: MelConfig,
    mel_frames: int,
) -> DexedSample:
    json_path = json_dir / f"{stem}.json"
    audio_path = audio_dir / f"{stem}.wav"

    midi_norm, param_vec = _load_json_params(
        json_path=json_path,
        midi_keys=midi_keys,
        param_names=param_names,
    )
    audio = _load_audio(audio_path=audio_path, sample_rate=sample_rate, target_duration=target_duration)

    if not np.isfinite(audio).all():
        raise ValueError(f"{audio_path} contains NaN/Inf values.")
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    if rms < 1e-6:
        raise ValueError(f"{audio_path} RMS too low ({rms:.2e}).")

    mel_spec = _make_spectrogram(audio=audio, sample_rate=sample_rate, mel_cfg=mel_cfg)
    mel_spec = _fit_mel_frames(mel_spec, mel_frames)

    return DexedSample(
        preset_id=stem,
        audio=audio.astype(np.float32, copy=False),
        mel_spec=mel_spec.astype(np.float32, copy=False),
        param_array=param_vec.astype(np.float32, copy=False),
        midi_norm=midi_norm.astype(np.float32, copy=False),
    )


def _create_h5_datasets(
    h5: h5py.File,
    num_samples: int,
    audio_len: int,
    mel_bins: int,
    mel_frames: int,
    num_params: int,
) -> tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    audio_ds = h5.create_dataset("audio", shape=(num_samples, audio_len), dtype=np.float16, compression=None)
    mel_ds = h5.create_dataset("mel_spec", shape=(num_samples, mel_bins, mel_frames), dtype=np.float32, compression=None)
    param_ds = h5.create_dataset("parameters", shape=(num_samples, num_params), dtype=np.float32, compression=None)
    midi_ds = h5.create_dataset("midi", shape=(num_samples, 3), dtype=np.float32, compression=None)
    preset_ds = h5.create_dataset(
        "preset_id",
        shape=(num_samples,),
        dtype=h5py.string_dtype(encoding="utf-8"),
        compression=None,
    )
    return audio_ds, mel_ds, param_ds, midi_ds, preset_ds


def _save_batch(
    samples: Sequence[DexedSample],
    audio_ds: h5py.Dataset,
    mel_ds: h5py.Dataset,
    param_ds: h5py.Dataset,
    midi_ds: h5py.Dataset,
    preset_ds: h5py.Dataset,
    start_idx: int,
) -> int:
    end_idx = start_idx + len(samples)
    audio_ds[start_idx:end_idx, :] = np.stack([s.audio for s in samples], axis=0).astype(np.float16, copy=False)
    mel_ds[start_idx:end_idx, :, :] = np.stack([s.mel_spec for s in samples], axis=0)
    param_ds[start_idx:end_idx, :] = np.stack([s.param_array for s in samples], axis=0)
    midi_ds[start_idx:end_idx, :] = np.stack([s.midi_norm for s in samples], axis=0)
    preset_ds[start_idx:end_idx] = [s.preset_id for s in samples]
    return end_idx


def _write_split(
    *,
    out_path: Path,
    split_name: str,
    stems: Sequence[str],
    json_dir: Path,
    audio_dir: Path,
    midi_keys: Sequence[str],
    param_names: Sequence[str],
    sample_rate: int,
    target_duration: float,
    mel_cfg: MelConfig,
    mel_frames: int,
    sample_batch_size: int,
    stats_tracker: Optional[MelStatsTracker],
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = len(stems)
    audio_len = int(sample_rate * float(target_duration))

    with h5py.File(out_path, "w") as h5:
        audio_ds, mel_ds, param_ds, midi_ds, preset_ds = _create_h5_datasets(
            h5=h5,
            num_samples=num_samples,
            audio_len=audio_len,
            mel_bins=int(mel_cfg.n_mels),
            mel_frames=mel_frames,
            num_params=len(param_names),
        )
        audio_ds.attrs["sample_rate"] = int(sample_rate)
        audio_ds.attrs["target_duration"] = float(target_duration)

        batch: list[DexedSample] = []
        next_write_idx = 0
        for i, stem in enumerate(stems):
            sample = _generate_sample(
                stem=stem,
                json_dir=json_dir,
                audio_dir=audio_dir,
                midi_keys=midi_keys,
                param_names=param_names,
                sample_rate=sample_rate,
                target_duration=target_duration,
                mel_cfg=mel_cfg,
                mel_frames=mel_frames,
            )
            if stats_tracker is not None:
                stats_tracker.update(sample.mel_spec)
            batch.append(sample)

            if len(batch) >= sample_batch_size:
                next_write_idx = _save_batch(
                    samples=batch,
                    audio_ds=audio_ds,
                    mel_ds=mel_ds,
                    param_ds=param_ds,
                    midi_ds=midi_ds,
                    preset_ds=preset_ds,
                    start_idx=next_write_idx,
                )
                batch = []

            if (i + 1) % max(1, sample_batch_size) == 0 or i + 1 == num_samples:
                logger.info("%s: packed %d/%d", split_name, i + 1, num_samples)

        if batch:
            next_write_idx = _save_batch(
                samples=batch,
                audio_ds=audio_ds,
                mel_ds=mel_ds,
                param_ds=param_ds,
                midi_ds=midi_ds,
                preset_ds=preset_ds,
                start_idx=next_write_idx,
            )

    return int(next_write_idx)


def _build_metadata(
    *,
    helper: DexedParameterHelper,
    recipe: DexedDatasetRecipe,
    config_path: Path,
    mel_frames: int,
    splits: Mapping[str, int],
) -> dict[str, object]:
    return {
        "synth": "dexed",
        "config_path": _project_relative_string(config_path),
        "source_json_dir": _project_relative_string(recipe.json_dir),
        "source_audio_dir": _project_relative_string(recipe.audio_dir),
        "summary_path": _project_relative_string(recipe.summary_path),
        "parameter_schema": helper.export_schema(),
        "num_samples": int(sum(int(v) for v in splits.values())),
        "sample_rate": int(recipe.sample_rate),
        "target_duration": float(recipe.target_duration),
        "mel": {
            "n_mels": int(recipe.mel.n_mels),
            "window_seconds": float(recipe.mel.window_seconds),
            "frames_per_second": float(recipe.mel.frames_per_second),
            "window": str(recipe.mel.window),
            "power_to_db_ref": str(recipe.mel.power_to_db_ref),
        },
        "mel_frames": int(mel_frames),
        "use_saved_mean_and_variance": True,
        "min_rms": float(1e-6),
        "min_loudness": float(recipe.min_loudness),
        "num_params": int(helper.preset_helper.param_count),
        "midi_keys": list(recipe.midi_keys),
        "midi_representation": "normalized",
        "split_ratios": {
            "train_ratio": float(recipe.train_ratio),
            "val_ratio": float(recipe.val_ratio),
            "test_ratio": float(1.0 - recipe.train_ratio - recipe.val_ratio),
        },
        "split_strategy": str(recipe.split_strategy),
        "splits": dict(splits),
        "seed": int(recipe.seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack Dexed JSON/WAV pairs into DDSynth-RL frozen-schema HDF5 splits.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dexed dataset recipe YAML describing packing semantics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dexed dataset recipe YAML describing packing semantics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config_path = _resolve_path(args.config)
    recipe = _load_dexed_recipe(config_path)
    output_root = recipe.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    helper = DexedParameterHelper(summary_path=recipe.summary_path)
    param_names = list(helper.preset_helper.vst_param_names)

    stems = _discover_stems(recipe.json_dir, recipe.audio_dir, recipe.num_samples)
    split_map = split_by_preset_group(
        stems=stems,
        train_ratio=recipe.train_ratio,
        val_ratio=recipe.val_ratio,
        seed=recipe.seed,
    )
    mel_frames = _compute_mel_frames(recipe.sample_rate, recipe.target_duration, recipe.mel)

    logger.info("Output root: %s", output_root)
    logger.info("Recipe config: %s", config_path)
    logger.info("JSON dir: %s", recipe.json_dir)
    logger.info("Audio dir: %s", recipe.audio_dir)
    logger.info("Summary path: %s", recipe.summary_path)
    logger.info("Split counts: %s", {k: len(v) for k, v in split_map.items()})
    logger.info("Mel frames: %d", mel_frames)

    train_stats = MelStatsTracker()
    splits_written: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_stems = sorted(split_map[split_name])
        out_path = output_root / f"{split_name}.h5"
        logger.info("Writing %s split (%d samples) -> %s", split_name, len(split_stems), out_path)
        written = _write_split(
            out_path=out_path,
            split_name=split_name,
            stems=split_stems,
            json_dir=recipe.json_dir,
            audio_dir=recipe.audio_dir,
            midi_keys=recipe.midi_keys,
            param_names=param_names,
            sample_rate=recipe.sample_rate,
            target_duration=recipe.target_duration,
            mel_cfg=recipe.mel,
            mel_frames=mel_frames,
            sample_batch_size=recipe.sample_batch_size,
            stats_tracker=train_stats if split_name == "train" else None,
        )
        logger.info("%s: wrote %d samples", split_name, written)
        splits_written[split_name] = int(written)

    mean, std = train_stats.finalize()
    np.savez(output_root / "stats.npz", mean=mean, std=std)
    logger.info("Saved mel stats -> %s", output_root / "stats.npz")

    metadata = _build_metadata(
        helper=helper,
        recipe=recipe,
        config_path=config_path,
        mel_frames=mel_frames,
        splits=splits_written,
    )
    with (output_root / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata -> %s", output_root / "dataset_metadata.json")


if __name__ == "__main__":
    main()
