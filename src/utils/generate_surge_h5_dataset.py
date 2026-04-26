"""Generate Surge XT random data and pack it into AR-Matching-style HDF5 files.

This script is self-contained inside DDSynth-RL:
- generation semantics come from an explicit dataset-recipe YAML
- sampling uses DDSynth-RL's local Surge summary/rules via SurgeParameterHelper
- rendering uses DDSynth-RL's SurgePedalboardRenderer
- output format follows train/val/test.h5 + dataset_metadata.json + stats.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import librosa
import numpy as np
import yaml

from src.project_paths import find_project_root, project_relative_string, resolve_project_path
from src.data.synth_backends.surge.surge_bridge import SurgeParameterHelper
from src.data.synth_backends.surge.surge_renderer import SurgePedalboardRenderer

_PROJECT_ROOT = find_project_root(Path(__file__).resolve())
_MIDI_KEYS: tuple[str, str, str] = ("MIDI_NOTE", "MIDI_VELOCITY", "MIDI_DURATION")

logger = logging.getLogger(__name__)


def _resolve_path(path_like: str | Path) -> Path:
    return resolve_project_path(path_like)


def _normalize_surge_plugin_path(path_like: str | Path) -> Path:
    """Accept either a `.vst3` bundle path or a file inside the bundle."""
    resolved = _resolve_path(path_like)
    if resolved.suffix.lower() == ".vst3":
        return resolved

    for parent in resolved.parents:
        if parent.suffix.lower() == ".vst3":
            return parent
    return resolved


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
        if self.sum is None or self.count == 0 or self.sumsq is None:
            raise RuntimeError("No mel samples were collected for stats.")
        mean = self.sum / float(self.count)
        var = np.maximum(self.sumsq / float(self.count) - mean**2, 0.0)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class GeneratedSample:
    preset_id: str
    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray
    midi_norm: np.ndarray


@dataclass(frozen=True)
class SamplingPriors:
    """Optional non-uniform sampling priors for selected parameters."""

    continuous: Dict[int, tuple[float, float]]  # full_idx -> (constant_val_p, constant_val)
    categorical: Dict[int, np.ndarray]  # full_idx -> normalized class weights


@dataclass(frozen=True)
class MelConfig:
    n_mels: int
    window_seconds: float
    frames_per_second: float
    window: str


@dataclass(frozen=True)
class ShardingConfig:
    shard_root: Path
    train_shards: int
    val_shards: int
    test_shards: int


@dataclass(frozen=True)
class DatasetRecipe:
    output_root: Path
    plugin_path: Path
    preset_path: Path
    summary_path: Path
    sampling_priors_path: Path
    num_samples: int
    sample_rate: int
    block_size: int
    target_duration: float
    min_rms: float
    max_tries: int
    sample_batch_size: int
    seed: int
    train_ratio: float
    val_ratio: float
    log_interval: int
    mel: MelConfig
    preset_load_flush_seconds: float
    post_param_flush_seconds: float
    post_render_flush_seconds: float
    allow_zero_velocity: bool
    fixed_midi_note: Optional[int]
    fixed_midi_velocity: Optional[int]
    fixed_midi_duration: Optional[float]
    reset_between_renders: bool
    sharding: ShardingConfig


def _require_mapping(payload: Mapping[str, Any], key: str, source: Path) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Recipe {source} must define mapping `{key}`.")
    return value


def _require_bool(payload: Mapping[str, Any], key: str, source: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Recipe {source} must define boolean `{key}`.")
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


def _optional_int(payload: Mapping[str, Any], key: str, source: Path) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Recipe {source} must define integer-or-null `{key}`.")
    return int(value)


def _optional_float(payload: Mapping[str, Any], key: str, source: Path) -> Optional[float]:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Recipe {source} must define numeric-or-null `{key}`.")
    return float(value)


def _load_dataset_recipe(config_path: Path) -> DatasetRecipe:
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset recipe YAML not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Dataset recipe must be a mapping: {config_path}")

    paths_cfg = _require_mapping(raw, "paths", config_path)
    dataset_cfg = _require_mapping(raw, "dataset", config_path)
    mel_cfg = _require_mapping(raw, "mel", config_path)
    render_cfg = _require_mapping(raw, "render", config_path)
    midi_cfg = _require_mapping(raw, "midi", config_path)
    runtime_cfg = _require_mapping(raw, "runtime", config_path)
    sharding_cfg = _require_mapping(raw, "sharding", config_path)

    recipe = DatasetRecipe(
        output_root=_resolve_path(_require_str(paths_cfg, "output_root", config_path)),
        plugin_path=_normalize_surge_plugin_path(_require_str(paths_cfg, "plugin_path", config_path)),
        preset_path=_resolve_path(_require_str(paths_cfg, "preset_path", config_path)),
        summary_path=_resolve_path(_require_str(paths_cfg, "summary_path", config_path)),
        sampling_priors_path=_resolve_path(_require_str(paths_cfg, "sampling_priors_path", config_path)),
        num_samples=_require_int(dataset_cfg, "num_samples", config_path, minimum=1),
        sample_rate=_require_int(dataset_cfg, "sample_rate", config_path, minimum=1),
        block_size=_require_int(dataset_cfg, "block_size", config_path, minimum=1),
        target_duration=_require_float(dataset_cfg, "target_duration", config_path, minimum=1e-8),
        min_rms=_require_float(dataset_cfg, "min_rms", config_path, minimum=0.0),
        max_tries=_require_int(dataset_cfg, "max_tries", config_path, minimum=1),
        sample_batch_size=_require_int(runtime_cfg, "sample_batch_size", config_path, minimum=1),
        seed=_require_int(dataset_cfg, "seed", config_path, minimum=0),
        train_ratio=_require_float(dataset_cfg, "train_ratio", config_path, minimum=1e-8),
        val_ratio=_require_float(dataset_cfg, "val_ratio", config_path, minimum=1e-8),
        log_interval=_require_int(runtime_cfg, "log_interval", config_path, minimum=1),
        mel=MelConfig(
            n_mels=_require_int(mel_cfg, "n_mels", config_path, minimum=1),
            window_seconds=_require_float(mel_cfg, "window_seconds", config_path, minimum=1e-8),
            frames_per_second=_require_float(mel_cfg, "frames_per_second", config_path, minimum=1e-8),
            window=_require_str(mel_cfg, "window", config_path),
        ),
        preset_load_flush_seconds=_require_float(render_cfg, "preset_load_flush_seconds", config_path, minimum=0.0),
        post_param_flush_seconds=_require_float(render_cfg, "post_param_flush_seconds", config_path, minimum=0.0),
        post_render_flush_seconds=_require_float(render_cfg, "post_render_flush_seconds", config_path, minimum=0.0),
        allow_zero_velocity=_require_bool(midi_cfg, "allow_zero_velocity", config_path),
        fixed_midi_note=_optional_int(midi_cfg, "fixed_note", config_path),
        fixed_midi_velocity=_optional_int(midi_cfg, "fixed_velocity", config_path),
        fixed_midi_duration=_optional_float(midi_cfg, "fixed_duration", config_path),
        reset_between_renders=_require_bool(render_cfg, "reset_between_renders", config_path),
        sharding=ShardingConfig(
            shard_root=_resolve_path(_require_str(sharding_cfg, "shard_root", config_path)),
            train_shards=_require_int(sharding_cfg, "train_shards", config_path, minimum=1),
            val_shards=_require_int(sharding_cfg, "val_shards", config_path, minimum=1),
            test_shards=_require_int(sharding_cfg, "test_shards", config_path, minimum=1),
        ),
    )

    if not recipe.mel.window:
        raise ValueError(f"Recipe {config_path} must define non-empty `mel.window`.")
    if recipe.train_ratio + recipe.val_ratio >= 1.0:
        raise ValueError(
            f"Recipe {config_path} must satisfy train_ratio + val_ratio < 1.0, "
            f"got {recipe.train_ratio + recipe.val_ratio}."
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


def _load_sampling_priors(
    priors_path: Path,
    helper: SurgeParameterHelper,
) -> SamplingPriors:
    if not priors_path.exists():
        raise FileNotFoundError(f"Sampling priors YAML not found: {priors_path}")

    payload = yaml.safe_load(priors_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Sampling priors must be a mapping: {priors_path}")

    name_to_idx = {name: i for i, name in enumerate(helper.preset_helper.vst_param_names)}
    modes = helper.preset_helper.vst_param_modes
    cardinals = helper.preset_helper.vst_param_cardinals
    mins = helper.preset_helper.vst_param_min
    maxs = helper.preset_helper.vst_param_max

    continuous: Dict[int, tuple[float, float]] = {}
    categorical: Dict[int, np.ndarray] = {}

    raw_cont = payload.get("continuous", {})
    if raw_cont is None:
        raw_cont = {}
    if not isinstance(raw_cont, Mapping):
        raise ValueError("`continuous` section in sampling priors must be a mapping.")

    for name, spec in raw_cont.items():
        if not isinstance(name, str):
            raise ValueError(f"Invalid continuous prior key (expected str): {name!r}")
        if name not in name_to_idx:
            raise KeyError(f"Continuous prior references unknown parameter: {name}")
        idx = int(name_to_idx[name])
        if modes[idx] != "num":
            raise ValueError(
                f"Continuous prior for `{name}` requires mode `num`, found `{modes[idx]}`"
            )
        if not isinstance(spec, Mapping):
            raise ValueError(f"Continuous prior for `{name}` must be a mapping")

        p = float(spec.get("constant_val_p", 0.0))
        v = float(spec.get("constant_val", 0.0))
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"`constant_val_p` for `{name}` must be in [0,1], got {p}")
        if not (float(mins[idx]) <= v <= float(maxs[idx])):
            raise ValueError(
                f"`constant_val` for `{name}`={v} outside configured range "
                f"[{mins[idx]}, {maxs[idx]}]"
            )
        continuous[idx] = (p, v)

    raw_cat = payload.get("categorical", {})
    if raw_cat is None:
        raw_cat = {}
    if not isinstance(raw_cat, Mapping):
        raise ValueError("`categorical` section in sampling priors must be a mapping.")

    for name, spec in raw_cat.items():
        if not isinstance(name, str):
            raise ValueError(f"Invalid categorical prior key (expected str): {name!r}")
        if name not in name_to_idx:
            raise KeyError(f"Categorical prior references unknown parameter: {name}")
        idx = int(name_to_idx[name])
        if modes[idx] != "cat":
            raise ValueError(
                f"Categorical prior for `{name}` requires mode `cat`, found `{modes[idx]}`"
            )
        if not isinstance(spec, Mapping):
            raise ValueError(f"Categorical prior for `{name}` must be a mapping")
        weights = spec.get("weights")
        if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
            raise ValueError(f"Categorical prior `{name}` must provide `weights` list")
        cardinal = int(cardinals[idx])
        if len(weights) != cardinal:
            raise ValueError(
                f"Categorical prior `{name}` has {len(weights)} weights, expected {cardinal}"
            )
        weight_arr = np.asarray([float(w) for w in weights], dtype=np.float64)
        if np.any(weight_arr < 0.0):
            raise ValueError(f"Categorical prior `{name}` has negative weights")
        wsum = float(np.sum(weight_arr))
        if not np.isfinite(wsum) or wsum <= 0.0:
            raise ValueError(f"Categorical prior `{name}` has invalid weight sum: {wsum}")
        categorical[idx] = (weight_arr / wsum).astype(np.float64, copy=False)

    return SamplingPriors(continuous=continuous, categorical=categorical)


class SurgeSummarySampler:
    """Random sampler over learnable Surge parameters defined in the summary YAML."""

    def __init__(self, helper: SurgeParameterHelper, priors: SamplingPriors) -> None:
        self._helper = helper
        self._space = helper.preset_helper
        self._num_priors = dict(priors.continuous)
        self._cat_priors = dict(priors.categorical)

        self._defaults = np.zeros((self._space.param_count,), dtype=np.float32)
        for idx, value in self._space.default_values.items():
            if 0 <= int(idx) < self._space.param_count:
                self._defaults[int(idx)] = float(value)

        self._num_indices: list[int] = []
        self._cat_indices: list[int] = []
        for idx, mode in enumerate(self._space.vst_param_modes):
            if mode == "num":
                self._num_indices.append(idx)
            elif mode == "cat":
                self._cat_indices.append(idx)
        self._learnable_indices = sorted(self._num_indices + self._cat_indices)

    def sample_full_vector(self, rng: np.random.Generator) -> np.ndarray:
        vec = self._defaults.copy()

        for idx in self._num_indices:
            min_v = float(self._space.vst_param_min[idx])
            max_v = float(self._space.vst_param_max[idx])
            if not math.isfinite(min_v) or not math.isfinite(max_v):
                raise ValueError(f"Non-finite numeric range at parameter index {idx}: min={min_v}, max={max_v}")
            prior = self._num_priors.get(idx)
            if prior is not None and float(prior[0]) > 0.0 and float(rng.random()) < float(prior[0]):
                vec[idx] = float(prior[1])
            elif max_v <= min_v:
                vec[idx] = min_v
            else:
                vec[idx] = float(rng.uniform(min_v, max_v))

        for idx in self._cat_indices:
            cardinal = int(self._space.vst_param_cardinals[idx])
            raw_values = self._space.vst_param_raw_values[idx]
            if cardinal <= 0:
                continue

            weights = self._cat_priors.get(idx)
            if weights is not None:
                cls = int(rng.choice(cardinal, p=weights))
            else:
                cls = int(rng.integers(0, cardinal))
            if raw_values is not None and len(raw_values) == cardinal:
                vec[idx] = float(raw_values[cls])
            elif cardinal == 1:
                vec[idx] = 0.0
            else:
                vec[idx] = float(cls) / float(cardinal - 1)

        return np.clip(vec, 0.0, 1.0).astype(np.float32, copy=False)

    @property
    def learnable_indices(self) -> list[int]:
        return list(self._learnable_indices)

    def render_param_dict_from_full(self, full_vec: np.ndarray) -> dict[str, float]:
        """Build renderer input dict using only sampled learnable parameters.

        This matches synth-permutations behavior: sampled subset is written,
        all other plugin parameters stay at preset/base state.
        """

        vec = np.asarray(full_vec, dtype=np.float32)
        names = self._space.vst_param_names
        return {names[i]: float(vec[i]) for i in self._learnable_indices}


class MidiSampler:
    """Sample MIDI absolute values and corresponding normalized labels."""

    def __init__(
        self,
        cfg: Mapping[str, float | int],
        allow_zero_velocity: bool = False,
        fixed_note: Optional[int] = None,
        fixed_velocity: Optional[int] = None,
        fixed_duration: Optional[float] = None,
    ) -> None:
        self.note_min = int(cfg["note_min"])
        self.note_classes = int(cfg["note_classes"])
        self.velocity_classes = int(cfg["velocity_classes"])
        self.duration_min = float(cfg["duration_min"])
        self.duration_max = float(cfg["duration_max"])
        self.duration_classes = int(cfg["duration_classes"])
        self.allow_zero_velocity = bool(allow_zero_velocity)
        self.fixed_note = int(fixed_note) if fixed_note is not None else None
        self.fixed_velocity = int(fixed_velocity) if fixed_velocity is not None else None
        self.fixed_duration = float(fixed_duration) if fixed_duration is not None else None

        if self.note_classes < 1:
            raise ValueError(f"Invalid note_classes={self.note_classes}; must be >= 1")
        if self.velocity_classes < 2:
            raise ValueError(f"Invalid velocity_classes={self.velocity_classes}; must be >= 2")
        if self.duration_classes < 1:
            raise ValueError(f"Invalid duration_classes={self.duration_classes}; must be >= 1")
        if self.duration_max <= self.duration_min:
            raise ValueError(
                f"Invalid duration range: duration_min={self.duration_min}, duration_max={self.duration_max}"
            )
        if self.note_min + self.note_classes - 1 > 127:
            raise ValueError(
                "MIDI note range exceeds 127. "
                f"note_min={self.note_min}, note_classes={self.note_classes}"
            )
        if self.velocity_classes - 1 > 127:
            raise ValueError(
                "velocity_classes implies values above MIDI 127. "
                f"velocity_classes={self.velocity_classes}"
            )

        if self.fixed_note is not None:
            note_hi = self.note_min + self.note_classes - 1
            if not (self.note_min <= self.fixed_note <= note_hi):
                raise ValueError(
                    f"fixed_note={self.fixed_note} is out of range [{self.note_min}, {note_hi}]"
                )
        if self.fixed_velocity is not None:
            vel_lo = 0 if self.allow_zero_velocity else 1
            vel_hi = self.velocity_classes - 1
            if not (vel_lo <= self.fixed_velocity <= vel_hi):
                raise ValueError(
                    f"fixed_velocity={self.fixed_velocity} is out of range [{vel_lo}, {vel_hi}]"
                )
        if self.fixed_duration is not None:
            if not (self.duration_min <= self.fixed_duration <= self.duration_max):
                raise ValueError(
                    f"fixed_duration={self.fixed_duration} is out of range "
                    f"[{self.duration_min}, {self.duration_max}]"
                )

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        if self.fixed_note is None:
            note_cls = int(rng.integers(0, self.note_classes))
            note_abs = float(self.note_min + note_cls)
        else:
            note_abs = float(self.fixed_note)
            note_cls = int(np.clip(round(note_abs - self.note_min), 0, self.note_classes - 1))

        vel_low = 0 if self.allow_zero_velocity else 1
        if self.fixed_velocity is None:
            vel_cls = int(rng.integers(vel_low, self.velocity_classes))
            velocity_abs = float(vel_cls)
        else:
            velocity_abs = float(self.fixed_velocity)
            vel_cls = int(np.clip(round(velocity_abs), vel_low, self.velocity_classes - 1))

        if self.fixed_duration is None:
            dur_cls = int(rng.integers(0, self.duration_classes))
            if self.duration_classes <= 1:
                duration_abs = float(self.duration_min)
                dur_norm = 0.0
            else:
                dur_norm = float(dur_cls) / float(self.duration_classes - 1)
                duration_abs = float(self.duration_min + dur_norm * (self.duration_max - self.duration_min))
        else:
            duration_abs = float(self.fixed_duration)
            if self.duration_classes <= 1:
                dur_cls = 0
                dur_norm = 0.0
            else:
                denom = max(self.duration_max - self.duration_min, 1e-8)
                ratio = np.clip((duration_abs - self.duration_min) / denom, 0.0, 1.0)
                dur_cls = int(np.clip(round(ratio * (self.duration_classes - 1)), 0, self.duration_classes - 1))
                dur_norm = float(dur_cls) / float(self.duration_classes - 1)

        note_den = float(max(self.note_classes - 1, 1))
        vel_den = float(max(self.velocity_classes - 1, 1))
        midi_norm = np.asarray(
            [
                float(note_cls) / note_den,
                float(vel_cls) / vel_den,
                float(dur_norm),
            ],
            dtype=np.float32,
        )
        midi_abs = np.asarray([note_abs, velocity_abs, duration_abs], dtype=np.float32)

        midi_payload = {
            "note": note_abs,
            "velocity": velocity_abs,
            "duration": duration_abs,
            "release": duration_abs * 0.5,
        }
        return midi_abs, midi_norm, midi_payload


def make_spectrogram(audio: np.ndarray, sample_rate: int, mel_cfg: MelConfig) -> np.ndarray:
    """Build mel spectrogram with explicit frozen recipe settings."""

    n_fft = int(float(mel_cfg.window_seconds) * sample_rate)
    hop_length = int(sample_rate / float(mel_cfg.frames_per_second))
    window = mel_cfg.window

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=int(mel_cfg.n_mels),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db.astype(np.float32, copy=False)


def _fit_audio_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if audio.shape[-1] > target_len:
        return audio[:target_len]
    if audio.shape[-1] < target_len:
        return np.pad(audio, (0, target_len - audio.shape[-1]))
    return audio


def _fit_mel_frames(mel: np.ndarray, target_frames: int) -> np.ndarray:
    if mel.shape[1] > target_frames:
        return mel[:, :target_frames]
    if mel.shape[1] < target_frames:
        pad = np.repeat(mel[:, -1:], repeats=target_frames - mel.shape[1], axis=1)
        return np.concatenate([mel, pad], axis=1)
    return mel


def _split_seed(base_seed: int, split_name: str) -> int:
    split_order = {"train": 1, "val": 2, "test": 3}
    if split_name not in split_order:
        raise KeyError(f"Unsupported split name: {split_name}")
    return int(base_seed) + 1009 * int(split_order[split_name])


def _rng_for_sample(split_seed: int, global_sample_index: int, attempt: int, stream_id: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence(
        [int(split_seed), int(global_sample_index), int(attempt), int(stream_id)]
    )
    return np.random.default_rng(seed_seq)


def _partition_count(total: int, num_parts: int) -> list[int]:
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if num_parts < 1:
        raise ValueError(f"num_parts must be >= 1, got {num_parts}")
    base = total // num_parts
    remainder = total % num_parts
    return [base + (1 if i < remainder else 0) for i in range(num_parts)]


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    if total < 3:
        raise ValueError("Need at least 3 samples for train/val/test splits.")
    if train_ratio <= 0.0 or val_ratio <= 0.0:
        raise ValueError(f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1.0, got {train_ratio + val_ratio}")

    test_ratio = 1.0 - train_ratio - val_ratio
    weights = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    weights = weights / float(np.sum(weights))

    # Keep every split non-empty, then distribute the remainder proportionally.
    base = np.asarray([1, 1, 1], dtype=np.int64)
    remaining = int(total - int(np.sum(base)))
    if remaining == 0:
        return {"train": int(base[0]), "val": int(base[1]), "test": int(base[2])}

    raw = weights * float(remaining)
    extra = np.floor(raw).astype(np.int64)
    leftover = int(remaining - int(np.sum(extra)))

    if leftover > 0:
        fractional = raw - extra.astype(np.float64)
        order = np.argsort(-fractional)
        for i in range(leftover):
            extra[int(order[i % len(order)])] += 1

    counts = base + extra
    return {
        "train": int(counts[0]),
        "val": int(counts[1]),
        "test": int(counts[2]),
    }


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
    samples: Sequence[GeneratedSample],
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


def _generate_one(
    *,
    split_name: str,
    split_seed: int,
    global_sample_index: int,
    renderer: SurgePedalboardRenderer,
    helper: SurgeParameterHelper,
    sampler: SurgeSummarySampler,
    midi_sampler: MidiSampler,
    target_duration: float,
    mel_cfg: MelConfig,
    mel_frames: int,
    min_rms: float,
    max_tries: int,
) -> GeneratedSample:
    target_len = int(renderer.sample_rate * float(target_duration))

    for attempt in range(int(max_tries)):
        param_rng = _rng_for_sample(split_seed, global_sample_index, attempt, 0)
        midi_rng = _rng_for_sample(split_seed, global_sample_index, attempt, 1)
        full_vec = sampler.sample_full_vector(param_rng)
        param_dict = sampler.render_param_dict_from_full(full_vec)
        _midi_abs, midi_norm, midi_payload = midi_sampler.sample(midi_rng)

        audio = renderer.render_single(params=param_dict, midi=midi_payload)
        audio_arr = np.asarray(audio, dtype=np.float32)
        if audio_arr.ndim == 2:
            if audio_arr.shape[0] == 1:
                audio_arr = audio_arr[0]
            else:
                audio_arr = np.mean(audio_arr, axis=0, dtype=np.float32)
        audio_arr = np.asarray(audio_arr, dtype=np.float32).reshape(-1)

        if not np.isfinite(audio_arr).all():
            continue
        audio_arr = _fit_audio_length(audio_arr, target_len)

        rms = float(np.sqrt(np.mean(audio_arr**2))) if audio_arr.size else 0.0
        if rms < float(min_rms):
            continue

        mel_spec = make_spectrogram(audio_arr, renderer.sample_rate, mel_cfg)
        mel_spec = _fit_mel_frames(mel_spec, mel_frames)

        raw_id = f"{split_name}:{split_seed}:{global_sample_index}:{attempt}".encode("utf-8")
        preset_id = hashlib.sha1(raw_id).hexdigest()[:16]
        return GeneratedSample(
            preset_id=preset_id,
            audio=audio_arr.astype(np.float32, copy=False),
            mel_spec=mel_spec.astype(np.float32, copy=False),
            param_array=full_vec.astype(np.float32, copy=False),
            midi_norm=midi_norm.astype(np.float32, copy=False),
        )

    raise RuntimeError(
        f"Failed to generate a valid sample after {max_tries} tries "
        f"(split={split_name}, global_index={global_sample_index}, min_rms={min_rms})."
    )


def _write_split(
    *,
    out_path: Path,
    split_name: str,
    num_samples: int,
    sample_start_index: int,
    renderer: SurgePedalboardRenderer,
    helper: SurgeParameterHelper,
    sampler: SurgeSummarySampler,
    midi_sampler: MidiSampler,
    target_duration: float,
    mel_cfg: MelConfig,
    mel_frames: int,
    min_rms: float,
    max_tries: int,
    batch_size: int,
    split_seed: int,
    stats_tracker: Optional[MelStatsTracker],
    log_interval: int,
    file_attrs: Optional[Mapping[str, object]] = None,
) -> int:
    audio_len = int(renderer.sample_rate * float(target_duration))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as h5:
        if file_attrs is not None:
            for key, value in file_attrs.items():
                h5.attrs[str(key)] = value
        h5.attrs["written_samples"] = 0
        h5.attrs["write_complete"] = False
        audio_ds, mel_ds, param_ds, midi_ds, preset_ds = _create_h5_datasets(
            h5=h5,
            num_samples=num_samples,
            audio_len=audio_len,
            mel_bins=int(mel_cfg.n_mels),
            mel_frames=mel_frames,
            num_params=helper.preset_helper.param_count,
        )
        audio_ds.attrs["sample_rate"] = int(renderer.sample_rate)
        audio_ds.attrs["target_duration"] = float(target_duration)

        batch: list[GeneratedSample] = []
        next_write_idx = 0
        for i in range(num_samples):
            global_sample_index = int(sample_start_index) + int(i)
            sample = _generate_one(
                split_name=split_name,
                split_seed=split_seed,
                global_sample_index=global_sample_index,
                renderer=renderer,
                helper=helper,
                sampler=sampler,
                midi_sampler=midi_sampler,
                target_duration=target_duration,
                mel_cfg=mel_cfg,
                mel_frames=mel_frames,
                min_rms=min_rms,
                max_tries=max_tries,
            )
            if stats_tracker is not None:
                stats_tracker.update(sample.mel_spec)
            batch.append(sample)

            if len(batch) >= batch_size:
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

            if (i + 1) % max(1, int(log_interval)) == 0 or i + 1 == num_samples:
                logger.info("%s: generated %d/%d", split_name, i + 1, num_samples)

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

        if next_write_idx != int(num_samples):
            raise RuntimeError(
                f"Split `{split_name}` wrote {next_write_idx} samples but expected {num_samples}."
            )

        h5.attrs["written_samples"] = int(next_write_idx)
        h5.attrs["write_complete"] = True
        h5.flush()

    return int(next_write_idx)


def _compute_mel_frames(sample_rate: int, target_duration: float, mel_cfg: MelConfig) -> int:
    audio_len = int(sample_rate * float(target_duration))
    hop = int(sample_rate / float(mel_cfg.frames_per_second))
    # librosa with center=True produces 1 + floor(N/hop) frames for this setup.
    return int(audio_len // hop) + 1


def _build_metadata(
    *,
    helper: SurgeParameterHelper,
    plugin_path: Path,
    preset_path: Optional[Path],
    summary_path: Path,
    sampling_priors_path: Path,
    sampling_priors: SamplingPriors,
    sample_rate: int,
    block_size: int,
    target_duration: float,
    mel_cfg: MelConfig,
    mel_frames: int,
    splits: Mapping[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_rms: float,
    max_tries: int,
    fixed_midi_note: Optional[int],
    fixed_midi_velocity: Optional[int],
    fixed_midi_duration: Optional[float],
    allow_zero_velocity: bool,
    reset_between_renders: bool,
    preset_load_flush_seconds: float,
    post_param_flush_seconds: float,
    post_render_flush_seconds: float,
    sharding: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    metadata = {
        "synth": "surge_xt",
        "plugin_path": project_relative_string(plugin_path),
        "preset_path": project_relative_string(preset_path) if preset_path is not None else None,
        "summary_path": project_relative_string(summary_path),
        "parameter_schema": helper.export_schema(),
        "sampling_priors_path": project_relative_string(sampling_priors_path),
        "sampling_priors": {
            "num_constant_overrides": int(len(sampling_priors.continuous)),
            "cat_weight_overrides": int(len(sampling_priors.categorical)),
        },
        "num_samples": int(sum(int(v) for v in splits.values())),
        "sample_rate": int(sample_rate),
        "block_size": int(block_size),
        "target_duration": float(target_duration),
        "mel": {
            "n_mels": int(mel_cfg.n_mels),
            "window_seconds": float(mel_cfg.window_seconds),
            "frames_per_second": float(mel_cfg.frames_per_second),
            "window": str(mel_cfg.window),
        },
        "mel_frames": int(mel_frames),
        "use_saved_mean_and_variance": True,
        "min_rms": float(min_rms),
        "max_tries": int(max_tries),
        "num_params": int(helper.preset_helper.param_count),
        "midi_keys": list(_MIDI_KEYS),
        "midi_representation": "normalized",
        "split_ratios": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(1.0 - train_ratio - val_ratio),
        },
        "splits": dict(splits),
        "seed": int(seed),
        "render_param_mode": "learnable_only",
        "allow_zero_velocity": bool(allow_zero_velocity),
        "render_flush_seconds": {
            "preset_load": float(preset_load_flush_seconds),
            "post_param": float(post_param_flush_seconds),
            "post_render": float(post_render_flush_seconds),
        },
        "reset_between_renders": bool(reset_between_renders),
        "fixed_midi": {
            "note": fixed_midi_note,
            "velocity": fixed_midi_velocity,
            "duration": fixed_midi_duration,
        },
    }
    if sharding is not None:
        metadata["sharding"] = dict(sharding)
    return metadata


def _build_common_generation_payload(
    *,
    helper: SurgeParameterHelper,
    plugin_path: Path,
    preset_path: Optional[Path],
    summary_path: Path,
    sampling_priors_path: Path,
    sampling_priors: SamplingPriors,
    sample_rate: int,
    block_size: int,
    target_duration: float,
    mel_cfg: MelConfig,
    mel_frames: int,
    seed: int,
    min_rms: float,
    max_tries: int,
    fixed_midi_note: Optional[int],
    fixed_midi_velocity: Optional[int],
    fixed_midi_duration: Optional[float],
    allow_zero_velocity: bool,
    reset_between_renders: bool,
    preset_load_flush_seconds: float,
    post_param_flush_seconds: float,
    post_render_flush_seconds: float,
) -> dict[str, object]:
    return {
        "plugin_path": project_relative_string(plugin_path),
        "preset_path": project_relative_string(preset_path) if preset_path is not None else None,
        "summary_path": project_relative_string(summary_path),
        "sampling_priors_path": project_relative_string(sampling_priors_path),
        "sampling_priors": {
            "num_constant_overrides": int(len(sampling_priors.continuous)),
            "cat_weight_overrides": int(len(sampling_priors.categorical)),
        },
        "parameter_schema": helper.export_schema(),
        "sample_rate": int(sample_rate),
        "block_size": int(block_size),
        "target_duration": float(target_duration),
        "mel": {
            "n_mels": int(mel_cfg.n_mels),
            "window_seconds": float(mel_cfg.window_seconds),
            "frames_per_second": float(mel_cfg.frames_per_second),
            "window": str(mel_cfg.window),
        },
        "mel_frames": int(mel_frames),
        "seed": int(seed),
        "min_rms": float(min_rms),
        "max_tries": int(max_tries),
        "midi_keys": list(_MIDI_KEYS),
        "midi_representation": "normalized",
        "allow_zero_velocity": bool(allow_zero_velocity),
        "reset_between_renders": bool(reset_between_renders),
        "fixed_midi": {
            "note": fixed_midi_note,
            "velocity": fixed_midi_velocity,
            "duration": fixed_midi_duration,
        },
        "render_flush_seconds": {
            "preset_load": float(preset_load_flush_seconds),
            "post_param": float(post_param_flush_seconds),
            "post_render": float(post_render_flush_seconds),
        },
    }


def _fingerprint_payload(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _build_shard_file_attrs(
    *,
    split_name: str,
    shard_index: int,
    shard_count: int,
    global_start_index: int,
    num_samples: int,
    split_seed: int,
    common_fingerprint: str,
) -> dict[str, object]:
    return {
        "soundmgm_format": "surge_h5_shard_v1",
        "split_name": str(split_name),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "global_start_index": int(global_start_index),
        "num_samples": int(num_samples),
        "split_seed": int(split_seed),
        "common_fingerprint": str(common_fingerprint),
    }


def _shard_count_for_split(recipe: DatasetRecipe, split_name: str) -> int:
    mapping = {
        "train": int(recipe.sharding.train_shards),
        "val": int(recipe.sharding.val_shards),
        "test": int(recipe.sharding.test_shards),
    }
    if split_name not in mapping:
        raise KeyError(f"Unsupported split name: {split_name}")
    return int(mapping[split_name])


def _shard_layout_for_split(recipe: DatasetRecipe, split_name: str, split_total: int) -> list[tuple[int, int]]:
    shard_count = _shard_count_for_split(recipe, split_name)
    shard_sizes = _partition_count(int(split_total), int(shard_count))
    layout: list[tuple[int, int]] = []
    start = 0
    for shard_size in shard_sizes:
        layout.append((int(start), int(shard_size)))
        start += int(shard_size)
    return layout


def _build_generation_context(
    recipe: DatasetRecipe,
    recipe_path: Path,
) -> tuple[
    SurgeParameterHelper,
    SamplingPriors,
    dict[str, float | int],
    dict[str, int],
    int,
    dict[str, object],
    str,
]:
    helper = SurgeParameterHelper(summary_path=recipe.summary_path)
    sampling_priors = _load_sampling_priors(recipe.sampling_priors_path, helper)
    midi_cfg: dict[str, float | int] = {
        "note_min": int(helper.midi_cfg.note_min),
        "note_classes": int(helper.midi_cfg.note_classes),
        "velocity_classes": int(helper.midi_cfg.velocity_classes),
        "duration_min": float(helper.midi_cfg.duration_min),
        "duration_max": float(helper.midi_cfg.duration_max),
        "duration_classes": int(helper.midi_cfg.duration_classes),
    }
    if float(recipe.target_duration) < float(midi_cfg["duration_max"]):
        raise ValueError(
            f"target_duration={float(recipe.target_duration):.3f} is shorter than "
            f"midi.duration_max={float(midi_cfg['duration_max']):.3f}."
        )
    splits = _split_counts(int(recipe.num_samples), float(recipe.train_ratio), float(recipe.val_ratio))
    mel_frames = _compute_mel_frames(int(recipe.sample_rate), float(recipe.target_duration), recipe.mel)
    common_payload = _build_common_generation_payload(
        helper=helper,
        plugin_path=recipe.plugin_path,
        preset_path=recipe.preset_path,
        summary_path=recipe.summary_path,
        sampling_priors_path=recipe.sampling_priors_path,
        sampling_priors=sampling_priors,
        sample_rate=int(recipe.sample_rate),
        block_size=int(recipe.block_size),
        target_duration=float(recipe.target_duration),
        mel_cfg=recipe.mel,
        mel_frames=mel_frames,
        seed=int(recipe.seed),
        min_rms=float(recipe.min_rms),
        max_tries=int(recipe.max_tries),
        fixed_midi_note=recipe.fixed_midi_note,
        fixed_midi_velocity=recipe.fixed_midi_velocity,
        fixed_midi_duration=recipe.fixed_midi_duration,
        allow_zero_velocity=bool(recipe.allow_zero_velocity),
        reset_between_renders=bool(recipe.reset_between_renders),
        preset_load_flush_seconds=float(recipe.preset_load_flush_seconds),
        post_param_flush_seconds=float(recipe.post_param_flush_seconds),
        post_render_flush_seconds=float(recipe.post_render_flush_seconds),
    )
    common_fingerprint = _fingerprint_payload(common_payload)
    logger.info("Recipe config: %s", recipe_path)
    logger.info("Sampling summary: %s", recipe.summary_path)
    logger.info("Sampling priors: %s", recipe.sampling_priors_path)
    logger.info("Split counts: %s", splits)
    logger.info("Mel frames: %d", mel_frames)
    logger.info(
        "Sampling priors overrides: %d numeric constants, %d categorical weight vectors",
        len(sampling_priors.continuous),
        len(sampling_priors.categorical),
    )
    learnable_param_count = int(sum(1 for m in helper.preset_helper.vst_param_modes if m in {"num", "cat"}))
    logger.info(
        "Renderer parameter write mode: learnable_only (%d/%d params per sample)",
        learnable_param_count,
        int(helper.preset_helper.param_count),
    )
    logger.info(
        "Render flush seconds: preset_load=%.3f post_param=%.3f post_render=%.3f",
        float(recipe.preset_load_flush_seconds),
        float(recipe.post_param_flush_seconds),
        float(recipe.post_render_flush_seconds),
    )
    logger.info("Generation fingerprint: %s", common_fingerprint)
    return helper, sampling_priors, midi_cfg, splits, mel_frames, common_payload, common_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Surge XT shard H5 described by the dataset recipe.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dataset recipe YAML describing generation semantics.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        required=True,
        help="Split name for the shard to generate.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        required=True,
        help="0-based shard index within the selected split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    recipe_path = _resolve_path(args.config)
    recipe = _load_dataset_recipe(recipe_path)

    helper, sampling_priors, midi_cfg, splits, mel_frames, common_payload, common_fingerprint = (
        _build_generation_context(recipe, recipe_path)
    )

    shard_output_root = recipe.sharding.shard_root
    shard_output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Shard root: %s", shard_output_root)
    shard_split = str(args.split)
    split_total = int(splits[shard_split])
    shard_layout = _shard_layout_for_split(recipe, shard_split, split_total)
    shard_index = int(args.shard_index)
    if not (0 <= shard_index < len(shard_layout)):
        raise ValueError(
            f"Invalid shard_index={shard_index} for split={shard_split}; "
            f"expected range [0, {len(shard_layout) - 1}]"
        )
    sample_start_index, shard_num_samples = shard_layout[shard_index]
    split_seed = _split_seed(int(recipe.seed), shard_split)
    split_dir = shard_output_root / shard_split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{shard_split}_{shard_index:03d}.h5"
    logger.info(
        "Writing shard: split=%s shard=%03d/%03d start=%d count=%d -> %s",
        shard_split,
        shard_index,
        len(shard_layout),
        sample_start_index,
        shard_num_samples,
        out_path,
    )

    renderer = SurgePedalboardRenderer(
        plugin_path=recipe.plugin_path,
        preset_path=recipe.preset_path,
        sample_rate=int(recipe.sample_rate),
        block_size=int(recipe.block_size),
        channels=2,
        fadeout_seconds=0.1,
        convert_to_mono=True,
        normalize_audio=False,
        note_on_delay=0.01,
        strict_parameter_check=True,
        reset_between_renders=bool(recipe.reset_between_renders),
        runtime_flush_seconds=1.0,
        preset_load_flush_seconds=float(recipe.preset_load_flush_seconds),
        post_param_flush_seconds=float(recipe.post_param_flush_seconds),
        post_render_flush_seconds=float(recipe.post_render_flush_seconds),
    )

    try:
        summary_names = list(helper.preset_helper.vst_param_names)
        missing = [name for name in summary_names if name not in renderer.param_names]
        if missing:
            preview = ", ".join(missing[:10])
            raise KeyError(
                f"Summary contains {len(missing)} parameter names not exposed by renderer "
                f"(examples: {preview})"
            )

        sampler = SurgeSummarySampler(helper=helper, priors=sampling_priors)
        midi_sampler = MidiSampler(
            cfg=midi_cfg,
            allow_zero_velocity=bool(recipe.allow_zero_velocity),
            fixed_note=recipe.fixed_midi_note,
            fixed_velocity=recipe.fixed_midi_velocity,
            fixed_duration=recipe.fixed_midi_duration,
        )
        shard_attrs = _build_shard_file_attrs(
            split_name=shard_split,
            shard_index=shard_index,
            shard_count=len(shard_layout),
            global_start_index=sample_start_index,
            num_samples=shard_num_samples,
            split_seed=split_seed,
            common_fingerprint=common_fingerprint,
        )
        written = _write_split(
            out_path=out_path,
            split_name=shard_split,
            num_samples=shard_num_samples,
            sample_start_index=sample_start_index,
            renderer=renderer,
            helper=helper,
            sampler=sampler,
            midi_sampler=midi_sampler,
            target_duration=float(recipe.target_duration),
            mel_cfg=recipe.mel,
            mel_frames=mel_frames,
            min_rms=float(recipe.min_rms),
            max_tries=int(recipe.max_tries),
            batch_size=int(recipe.sample_batch_size),
            split_seed=split_seed,
            stats_tracker=None,
            log_interval=int(recipe.log_interval),
            file_attrs=shard_attrs,
        )
        logger.info("%s shard %03d: wrote %d samples", shard_split, shard_index, written)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
