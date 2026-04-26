"""Convert Surge XT parameters to a continuous + categorical-logit space for flow models.

The space follows the current DDSynth-RL frozen Surge schema semantics:

- `mode: num` parameters remain scalar (optionally scaled from [0, 1] -> [-1, 1]).
- `mode: cat` parameters expand to one-hot logits with their native cardinality.
- `mode: null` parameters are not learned and fall back to frozen defaults on decode.
- MIDI is appended as 3 continuous scalars: note, velocity, duration (all normalized in [0, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.project_paths import resolve_project_path

MIDI_FIELD_NAMES = ("MIDI_NOTE", "MIDI_VELOCITY", "MIDI_DURATION")


@dataclass(frozen=True)
class _ParamSlice:
    full_index: int
    start: int
    end: int
    kind: str  # "num" | "cat"
    cardinality: int
    default: float
    raw_values: tuple[float, ...] | None


class SurgeDenoiseSpace:
    """Build a denoising space for Surge XT full parameters + normalized MIDI."""

    def __init__(
        self,
        config_path: Path | str = "configs/data/surge_params_summary.yaml",
        scale_to_unit: bool = True,
    ) -> None:
        self.scale_to_unit = bool(scale_to_unit)
        self.config_path = resolve_project_path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Surge FM config not found: {self.config_path}")

        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self._init_from_schema(raw, source=str(self.config_path))

    @classmethod
    def from_schema(
        cls,
        schema: Mapping[str, Any],
        *,
        scale_to_unit: bool = True,
    ) -> "SurgeDenoiseSpace":
        inst = cls.__new__(cls)
        inst.scale_to_unit = bool(scale_to_unit)
        inst.config_path = None
        inst._init_from_schema(schema, source="dataset metadata")
        return inst

    def _init_from_schema(self, raw: Any, *, source: str) -> None:
        if not isinstance(raw, Mapping) or "parameters" not in raw:
            raise ValueError(f"Surge schema must be a mapping with `parameters`: {source}")
        params = raw.get("parameters")
        if not isinstance(params, list):
            raise ValueError(f"Surge schema `parameters` must be a list: {source}")

        numeric_entries = [entry for entry in params if isinstance(entry, Mapping) and isinstance(entry.get("index"), int)]
        numeric_entries.sort(key=lambda entry: int(entry["index"]))
        if not numeric_entries:
            raise ValueError(f"Surge schema has no numeric parameter entries: {source}")

        # Frozen H5/backend vectors are stored in the compact schema order, not
        # in raw plugin-index space. The schema may have holes in the original
        # plugin indices (e.g. missing 29/31/135), so FM must use the compact
        # position here or every parameter after the first hole is misaligned.
        self.full_param_len = len(numeric_entries)
        self.midi_names = MIDI_FIELD_NAMES

        slices: List[_ParamSlice] = []
        defaults: Dict[int, float] = {}
        cursor = 0
        for compact_index, entry in enumerate(numeric_entries):
            mode = entry.get("mode")
            default_raw = entry.get("default", 0.0)
            default = float(default_raw) if default_raw is not None else 0.0
            defaults[compact_index] = default

            if mode is None:
                continue

            mode_str = str(mode).lower()
            if mode_str == "num":
                slices.append(
                    _ParamSlice(
                        full_index=compact_index,
                        start=cursor,
                        end=cursor + 1,
                        kind="num",
                        cardinality=1,
                        default=default,
                        raw_values=None,
                    )
                )
                cursor += 1
                continue

            if mode_str == "cat":
                cardinality = int(entry.get("cardinality", 0) or 0)
                if cardinality <= 1:
                    raise ValueError(
                        f"Categorical Surge parameter '{entry.get('name')}' needs cardinality>1: {source}"
                    )
                raw_values = entry.get("raw_values")
                parsed_raw_values = None
                if isinstance(raw_values, list) and len(raw_values) == cardinality:
                    parsed_raw_values = tuple(float(v) for v in raw_values)
                slices.append(
                    _ParamSlice(
                        full_index=compact_index,
                        start=cursor,
                        end=cursor + cardinality,
                        kind="cat",
                        cardinality=cardinality,
                        default=default,
                        raw_values=parsed_raw_values,
                    )
                )
                cursor += cardinality
                continue

            raise ValueError(
                f"Unexpected Surge mode '{mode}' for compact index {compact_index}: {source}"
            )

        self.slices = slices
        self.defaults = defaults
        self._midi_start = cursor
        self.total_dim = cursor + 3
        self.midi_start = self._midi_start
        self.midi_dim = 3

    def encode(self, full_params, midi_norm) -> torch.Tensor | np.ndarray:
        params_is_torch = torch.is_tensor(full_params)
        midi_is_torch = torch.is_tensor(midi_norm)

        if params_is_torch or midi_is_torch:
            if not params_is_torch:
                full_params = torch.as_tensor(full_params, dtype=torch.float32, device=midi_norm.device if midi_is_torch else None)
            if not midi_is_torch:
                midi_norm = torch.as_tensor(midi_norm, dtype=torch.float32, device=full_params.device)
            if full_params.ndim == 1:
                full_params = full_params.unsqueeze(0)
            if midi_norm.ndim == 1:
                midi_norm = midi_norm.unsqueeze(0)
            if full_params.shape[0] != midi_norm.shape[0]:
                raise ValueError(f"full_params batch {full_params.shape} and midi batch {midi_norm.shape} mismatch")
            out = full_params.new_zeros(full_params.shape[:-1] + (self.total_dim,))
            for sl in self.slices:
                values = full_params[..., sl.full_index]
                if sl.kind == "num":
                    if self.scale_to_unit:
                        values = values * 2.0 - 1.0
                    out[..., sl.start] = values
                else:
                    cls = self._categorical_class_torch(values, sl)
                    one_hot = F.one_hot(cls, num_classes=sl.cardinality).to(out.dtype)
                    out[..., sl.start : sl.end] = one_hot
            midi_vals = midi_norm
            if self.scale_to_unit:
                midi_vals = midi_vals * 2.0 - 1.0
            out[..., self._midi_start : self._midi_start + 3] = midi_vals
            return out

        full_params = np.asarray(full_params, dtype=np.float32)
        midi_norm = np.asarray(midi_norm, dtype=np.float32)
        if full_params.ndim == 1:
            full_params = full_params[None, :]
        if midi_norm.ndim == 1:
            midi_norm = midi_norm[None, :]
        if full_params.shape[0] != midi_norm.shape[0]:
            raise ValueError(f"full_params batch {full_params.shape} and midi batch {midi_norm.shape} mismatch")
        out = np.zeros(full_params.shape[:-1] + (self.total_dim,), dtype=np.float32)
        for sl in self.slices:
            values = full_params[..., sl.full_index]
            if sl.kind == "num":
                out[..., sl.start] = values * 2.0 - 1.0 if self.scale_to_unit else values
            else:
                cls = self._categorical_class_numpy(values, sl)
                one_hot = np.zeros(full_params.shape[:-1] + (sl.cardinality,), dtype=np.float32)
                rows = np.arange(cls.shape[0])
                one_hot[rows, cls] = 1.0
                out[..., sl.start : sl.end] = one_hot
        midi_vals = midi_norm * 2.0 - 1.0 if self.scale_to_unit else midi_norm
        out[..., self._midi_start : self._midi_start + 3] = midi_vals
        return out

    def decode(self, denoised) -> torch.Tensor | np.ndarray:
        is_torch = torch.is_tensor(denoised)
        arr = denoised if is_torch else np.asarray(denoised, dtype=np.float32)
        flat_shape = arr.shape[:-1]
        if is_torch:
            full = torch.zeros(flat_shape + (self.full_param_len,), device=arr.device, dtype=arr.dtype)
            midi = torch.zeros(flat_shape + (3,), device=arr.device, dtype=arr.dtype)
        else:
            full = np.zeros(flat_shape + (self.full_param_len,), dtype=np.float32)
            midi = np.zeros(flat_shape + (3,), dtype=np.float32)

        for idx, val in self.defaults.items():
            full[..., idx] = val

        for sl in self.slices:
            if sl.kind == "num":
                vals = arr[..., sl.start]
                if self.scale_to_unit:
                    vals = (vals + 1.0) * 0.5
                full[..., sl.full_index] = torch.clamp(vals, 0.0, 1.0) if is_torch else np.clip(vals, 0.0, 1.0)
            else:
                logits = arr[..., sl.start : sl.end]
                cls = torch.argmax(logits, dim=-1) if is_torch else np.argmax(logits, axis=-1)
                if sl.raw_values:
                    raw = torch.as_tensor(sl.raw_values, device=arr.device, dtype=arr.dtype) if is_torch else np.asarray(sl.raw_values, dtype=np.float32)
                    full[..., sl.full_index] = raw[cls]
                else:
                    if is_torch:
                        cls = cls.to(dtype=arr.dtype)
                        val = cls / float(sl.cardinality - 1)
                    else:
                        val = cls.astype(np.float32) / float(sl.cardinality - 1)
                    full[..., sl.full_index] = val

        midi_vals = arr[..., self._midi_start : self._midi_start + 3]
        if self.scale_to_unit:
            midi_vals = (midi_vals + 1.0) * 0.5
        midi[...] = torch.clamp(midi_vals, 0.0, 1.0) if is_torch else np.clip(midi_vals, 0.0, 1.0)

        return (
            torch.cat([full, midi], dim=-1)
            if is_torch
            else np.concatenate([full, midi], axis=-1)
        )

    def decode_full_and_midi_norm(self, denoised) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        decoded = self.decode(denoised)
        return decoded[..., : self.full_param_len], decoded[..., self.full_param_len :]

    @staticmethod
    def _categorical_class_torch(values: torch.Tensor, sl: _ParamSlice) -> torch.Tensor:
        if sl.raw_values:
            raw = torch.as_tensor(sl.raw_values, device=values.device, dtype=values.dtype)
            dists = torch.abs(values.unsqueeze(-1) - raw.unsqueeze(0))
            return torch.argmin(dists, dim=-1).to(dtype=torch.long)
        return torch.clamp(
            torch.round(torch.clamp(values, 0.0, 1.0) * float(sl.cardinality - 1)),
            0,
            sl.cardinality - 1,
        ).to(dtype=torch.long)

    @staticmethod
    def _categorical_class_numpy(values: np.ndarray, sl: _ParamSlice) -> np.ndarray:
        if sl.raw_values:
            raw = np.asarray(sl.raw_values, dtype=np.float32)
            dists = np.abs(values[:, None] - raw[None, :])
            return np.argmin(dists, axis=1).astype(np.int64)
        return np.clip(
            np.round(np.clip(values, 0.0, 1.0) * float(sl.cardinality - 1)),
            0,
            sl.cardinality - 1,
        ).astype(np.int64)
