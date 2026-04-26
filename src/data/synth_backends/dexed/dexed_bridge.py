"""Dexed parameter bridge.

This module maps between:
- full Dexed preset vectors (all VST parameters)
- learnable model vectors (num -> scalar, cat -> one-hot block)
- optional MIDI learnable vectors appended as one-hot blocks
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from src.project_paths import resolve_project_path


def _resolve_summary_path(
    summary_path: Union[str, Path],
) -> Path:
    return resolve_project_path(summary_path)


@dataclass(frozen=True)
class DexedParamMeta:
    full_index: int
    learnable_indices: Sequence[int]
    name: str
    cardinality: int
    mode: Optional[str]


@dataclass(frozen=True)
class MidiConfig:
    note_min: int = 24
    note_classes: int = 73
    velocity_classes: int = 128
    duration_min: float = 0.2
    duration_max: float = 3.0
    duration_classes: int = 128


def _parse_midi_cfg(raw_midi: Any, summary_source: Union[str, Path]) -> MidiConfig:
    if not isinstance(raw_midi, Mapping):
        raise ValueError(
            f"Dexed summary must define top-level `midi` mapping: {summary_source}"
        )

    required = (
        "note_min",
        "note_classes",
        "velocity_classes",
        "duration_min",
        "duration_max",
        "duration_classes",
    )
    missing = [key for key in required if key not in raw_midi]
    if missing:
        raise ValueError(
            f"Dexed summary missing required midi keys {missing}: {summary_source}"
        )

    return MidiConfig(
        note_min=int(raw_midi["note_min"]),
        note_classes=int(raw_midi["note_classes"]),
        velocity_classes=int(raw_midi["velocity_classes"]),
        duration_min=float(raw_midi["duration_min"]),
        duration_max=float(raw_midi["duration_max"]),
        duration_classes=int(raw_midi["duration_classes"]),
    )


def _normalize_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"num", "cat"}:
            return lowered
    raise ValueError(f"Unsupported Dexed parameter mode: {value!r}")


class SummaryParameterSpace:
    """Parameter-space view built from `dexed_params_summary.yaml` entries."""

    def __init__(self, entries: Sequence[Dict[str, Any]]) -> None:
        numeric_entries = [entry for entry in entries if isinstance(entry.get("index"), int)]
        numeric_entries.sort(key=lambda item: int(item["index"]))

        self.param_count = len(numeric_entries)
        self.vst_param_names: List[str] = [str(entry["name"]) for entry in numeric_entries]
        self.vst_param_modes: List[Optional[str]] = [
            _normalize_mode(entry.get("mode")) for entry in numeric_entries
        ]

        cardinals: List[int] = []
        for entry, mode in zip(numeric_entries, self.vst_param_modes):
            if mode == "cat":
                cardinal = int(entry.get("cardinality", 1) or 1)
                cardinals.append(max(cardinal, 1))
            else:
                # `num` is represented by one learnable scalar; frozen params have none.
                cardinals.append(1)
        self.vst_param_cardinals = cardinals

        self.default_values: Dict[int, float] = {}
        for entry in numeric_entries:
            default = entry.get("default")
            if default is not None:
                self.default_values[int(entry["index"])] = float(default)

        self.full_to_learnable: List[Optional[Union[int, List[int]]]] = []
        self.learnable_to_full: List[int] = []
        cursor = 0
        for full_idx, mode in enumerate(self.vst_param_modes):
            if mode is None:
                self.full_to_learnable.append(None)
                continue
            if mode == "num":
                self.full_to_learnable.append(cursor)
                self.learnable_to_full.append(full_idx)
                cursor += 1
                continue

            cardinal = self.vst_param_cardinals[full_idx]
            onehot_indices = list(range(cursor, cursor + cardinal))
            self.full_to_learnable.append(onehot_indices)
            self.learnable_to_full.extend([full_idx] * cardinal)
            cursor += cardinal

        self.learnable_preset_size = cursor

    def full_from_learnable(
        self, learnable: np.ndarray, apply_defaults: bool = True
    ) -> np.ndarray:
        learnable = np.asarray(learnable, dtype=np.float32)
        squeeze = False
        if learnable.ndim == 1:
            learnable = learnable[None, :]
            squeeze = True

        batch = learnable.shape[0]
        full = np.zeros((batch, self.param_count), dtype=np.float32)

        if apply_defaults:
            for idx, value in self.default_values.items():
                full[:, idx] = value

        for full_idx, mapping in enumerate(self.full_to_learnable):
            mode = self.vst_param_modes[full_idx]
            if mode is None or mapping is None:
                continue

            if mode == "num":
                full[:, full_idx] = learnable[:, int(mapping)]
                continue

            indices = mapping if isinstance(mapping, list) else [mapping]
            logits = learnable[:, indices]
            classes = np.argmax(logits, axis=1)
            cardinal = self.vst_param_cardinals[full_idx]
            if cardinal <= 1:
                full[:, full_idx] = 0.0
            else:
                full[:, full_idx] = classes / float(cardinal - 1)

        return full[0] if squeeze else full

    def learnable_from_full(self, full: np.ndarray) -> np.ndarray:
        full = np.asarray(full, dtype=np.float32)
        squeeze = False
        if full.ndim == 1:
            full = full[None, :]
            squeeze = True

        batch = full.shape[0]
        learnable = np.zeros((batch, self.learnable_preset_size), dtype=np.float32)

        for full_idx, mapping in enumerate(self.full_to_learnable):
            mode = self.vst_param_modes[full_idx]
            if mode is None or mapping is None:
                continue

            values = full[:, full_idx]
            if mode == "num":
                learnable[:, int(mapping)] = values
                continue

            indices = mapping if isinstance(mapping, list) else [mapping]
            cardinal = self.vst_param_cardinals[full_idx]
            if cardinal <= 1:
                continue
            classes = np.clip(
                np.round(values * (cardinal - 1)), 0, cardinal - 1
            ).astype(np.int64)
            for bi, cls in enumerate(classes):
                learnable[bi, indices[int(cls)]] = 1.0

        return learnable[0] if squeeze else learnable


class DexedParameterHelper:
    """Minimal Dexed helper used by renderer/model glue code."""

    def __init__(self, summary_path: Union[str, Path]) -> None:
        path = _resolve_summary_path(summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Dexed parameter summary not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self._init_from_schema(raw=raw, schema_source=str(path))

    @classmethod
    def from_schema(cls, schema: Mapping[str, Any]) -> "DexedParameterHelper":
        helper = cls.__new__(cls)
        helper._init_from_schema(raw=schema, schema_source="dataset metadata")
        return helper

    def export_schema(self) -> Dict[str, Any]:
        return copy.deepcopy(self._schema)

    def _init_from_schema(self, raw: Any, schema_source: str) -> None:
        if not isinstance(raw, dict) or "parameters" not in raw:
            raise ValueError(
                f"Dexed summary must be a mapping containing 'parameters': {schema_source}"
            )

        self._schema = copy.deepcopy(dict(raw))
        self._space = SummaryParameterSpace(raw["parameters"])
        self._meta = self._build_meta()

        self.midi_cfg = _parse_midi_cfg(raw.get("midi"), schema_source)

        self.midi_learnable_size = (
            self.midi_cfg.note_classes
            + self.midi_cfg.velocity_classes
            + self.midi_cfg.duration_classes
        )
        self.learnable_preset_size = self._space.learnable_preset_size + self.midi_learnable_size

    @property
    def preset_helper(self) -> SummaryParameterSpace:
        return self._space

    def _build_meta(self) -> List[DexedParamMeta]:
        meta: List[DexedParamMeta] = []
        for full_idx, mapping in enumerate(self._space.full_to_learnable):
            if mapping is None:
                learnable_indices: Sequence[int] = []
            elif isinstance(mapping, list):
                learnable_indices = list(mapping)
            else:
                learnable_indices = [int(mapping)]

            meta.append(
                DexedParamMeta(
                    full_index=full_idx,
                    learnable_indices=learnable_indices,
                    name=self._space.vst_param_names[full_idx],
                    cardinality=self._space.vst_param_cardinals[full_idx],
                    mode=self._space.vst_param_modes[full_idx],
                )
            )
        return meta

    def get_param_meta(self) -> List[DexedParamMeta]:
        return list(self._meta)

    def learnable_to_full(
        self,
        learnable_tensor: np.ndarray,
        apply_defaults: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        learnable = np.asarray(learnable_tensor, dtype=np.float32)
        squeeze = False
        if learnable.ndim == 1:
            learnable = learnable[None, :]
            squeeze = True

        synth_part = learnable[:, : self._space.learnable_preset_size]
        midi_part = learnable[:, self._space.learnable_preset_size :]
        synth_full = self._space.full_from_learnable(
            synth_part, apply_defaults=apply_defaults
        )
        midi_values = self._midi_from_learnable(midi_part)

        if squeeze:
            return synth_full[0], midi_values[0]
        return synth_full, midi_values

    def full_to_learnable(
        self,
        full_tensor: np.ndarray,
        midi: Union[np.ndarray, Mapping[str, float]],
    ) -> np.ndarray:
        synth_learnable = self._space.learnable_from_full(full_tensor)
        midi_learnable = self._midi_to_learnable(midi)
        synth_learnable = np.asarray(synth_learnable, dtype=np.float32)
        midi_learnable = np.asarray(midi_learnable, dtype=np.float32)

        if synth_learnable.ndim == 1:
            return np.concatenate([synth_learnable, midi_learnable], axis=0)
        return np.concatenate([synth_learnable, midi_learnable], axis=1)

    def _normalize_midi_input(
        self,
        midi: Union[np.ndarray, Mapping[str, float]],
        batch: int,
    ) -> np.ndarray:
        if midi is None:
            raise ValueError("MIDI must be provided explicitly; default MIDI fallback is not allowed.")

        if isinstance(midi, Mapping):
            required = ("note", "velocity", "duration")
            missing = [key for key in required if key not in midi]
            if missing:
                raise ValueError(
                    f"MIDI mapping must define keys {required}; missing {missing}"
                )
            note = float(midi["note"])
            velocity = float(midi["velocity"])
            duration = float(midi["duration"])
            vec = np.asarray([note, velocity, duration], dtype=np.float32)
            return np.tile(vec[None, :], (batch, 1))

        arr = np.asarray(midi, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError(f"MIDI vector must have shape (3,), got {arr.shape}")
            return np.tile(arr[None, :], (batch, 1))

        if arr.ndim == 2 and arr.shape[1] == 3:
            if arr.shape[0] != batch:
                raise ValueError(
                    f"MIDI batch size mismatch: expected {batch}, got {arr.shape[0]}"
                )
            return arr

        raise ValueError(f"Unsupported MIDI shape: {arr.shape}")

    def _midi_to_learnable(
        self,
        midi: Union[np.ndarray, Mapping[str, float]],
    ) -> np.ndarray:
        batch = 1
        if isinstance(midi, np.ndarray) and midi.ndim == 2:
            batch = midi.shape[0]

        midi_arr = self._normalize_midi_input(midi, batch)

        note = midi_arr[:, 0]
        velocity = midi_arr[:, 1]
        duration = midi_arr[:, 2]

        note_cls = np.clip(
            np.round(note - self.midi_cfg.note_min),
            0,
            self.midi_cfg.note_classes - 1,
        ).astype(np.int64)
        vel_cls = np.clip(
            np.round(velocity),
            0,
            self.midi_cfg.velocity_classes - 1,
        ).astype(np.int64)

        dur_den = max(self.midi_cfg.duration_max - self.midi_cfg.duration_min, 1e-8)
        dur_ratio = np.clip((duration - self.midi_cfg.duration_min) / dur_den, 0.0, 1.0)
        dur_cls = np.clip(
            np.round(dur_ratio * (self.midi_cfg.duration_classes - 1)),
            0,
            self.midi_cfg.duration_classes - 1,
        ).astype(np.int64)

        out = np.zeros((batch, self.midi_learnable_size), dtype=np.float32)
        for bi in range(batch):
            out[bi, note_cls[bi]] = 1.0
            out[bi, self.midi_cfg.note_classes + vel_cls[bi]] = 1.0
            out[
                bi,
                self.midi_cfg.note_classes + self.midi_cfg.velocity_classes + dur_cls[bi],
            ] = 1.0

        return out[0] if batch == 1 else out

    def _midi_from_learnable(self, midi_part: np.ndarray) -> np.ndarray:
        midi_part = np.asarray(midi_part, dtype=np.float32)
        if midi_part.ndim == 1:
            midi_part = midi_part[None, :]

        expected = self.midi_learnable_size
        if midi_part.shape[1] != expected:
            raise ValueError(
                f"Expected MIDI learnable size {expected}, got {midi_part.shape[1]}"
            )

        note_slice = slice(0, self.midi_cfg.note_classes)
        vel_slice = slice(note_slice.stop, note_slice.stop + self.midi_cfg.velocity_classes)
        dur_slice = slice(vel_slice.stop, vel_slice.stop + self.midi_cfg.duration_classes)

        note_cls = np.argmax(midi_part[:, note_slice], axis=1).astype(np.float32)
        vel_cls = np.argmax(midi_part[:, vel_slice], axis=1).astype(np.float32)
        dur_cls = np.argmax(midi_part[:, dur_slice], axis=1).astype(np.float32)

        notes = self.midi_cfg.note_min + note_cls
        velocities = vel_cls
        if self.midi_cfg.duration_classes <= 1:
            durations = np.full((midi_part.shape[0],), self.midi_cfg.duration_min, dtype=np.float32)
        else:
            ratios = dur_cls / float(self.midi_cfg.duration_classes - 1)
            durations = self.midi_cfg.duration_min + ratios * (
                self.midi_cfg.duration_max - self.midi_cfg.duration_min
            )
        return np.stack([notes, velocities, durations], axis=1).astype(np.float32)
