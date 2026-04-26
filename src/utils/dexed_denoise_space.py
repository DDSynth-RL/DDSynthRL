"""Convert Dexed parameters to a continuous + categorical-logit space for denoising/flow models.

This is adapted from Synth-Matching's `DexedDenoiseSpace`, which itself mirrors the
flow-matching setup used in synth-permutations:

- Continuous parameters stay as scalars (optionally scaled from [0, 1] -> [-1, 1]).
- Categorical parameters expand into one-hot logits with length `cardinality`.
- Extra (string) indices (e.g. MIDI) are appended after the numeric Dexed indices and
  are always treated as continuous scalars.

The YAML config is expected to be a dict with a `parameters` list. Each entry contains:
    - index: int for Dexed params, or str for auxiliary fields (e.g. MIDI_NOTE)
    - mode: "num" | "cat" | null
    - cardinality: required for "cat"
    - default: optional default used for frozen params during decoding
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.project_paths import resolve_project_path


@dataclass(frozen=True)
class _ParamSlice:
    full_index: int
    start: int
    end: int
    kind: str  # "num" | "cat"
    cardinality: int
    default: float


class DexedDenoiseSpace:
    """Build a denoising space for Dexed parameters + auxiliary continuous fields (e.g. MIDI)."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        scale_to_unit: bool = True,
    ) -> None:
        self.scale_to_unit = bool(scale_to_unit)
        default_cfg = resolve_project_path("configs/data/dexed_params_fm.yaml")
        self.config_path = (
            resolve_project_path(config_path) if config_path is not None else default_cfg
        )

        if not self.config_path.exists():
            raise FileNotFoundError(f"Dexed FM config not found: {self.config_path}")

        entries = self._load_param_entries(self.config_path)

        numeric_entries = [e for e in entries if isinstance(e.get("index"), int)]
        numeric_indices = [int(e["index"]) for e in numeric_entries]
        max_numeric = max(numeric_indices) if numeric_indices else -1
        string_entries = [e for e in entries if isinstance(e.get("index"), str)]
        self.full_param_len = max_numeric + 1
        self.midi_names = tuple(str(e["index"]) for e in string_entries)

        index_to_pos: Dict[Union[int, str], int] = {}
        for e in numeric_entries:
            index_to_pos[int(e["index"])] = int(e["index"])
        for offset, e in enumerate(string_entries):
            index_to_pos[str(e["index"])] = max_numeric + 1 + offset

        self.index_to_pos = index_to_pos
        self.string_indices = [str(e["index"]) for e in string_entries]
        self.param_len = (max_numeric + 1) + len(self.string_indices)

        slices: List[_ParamSlice] = []
        cursor = 0
        defaults: Dict[int, float] = {}

        for entry in entries:
            idx_raw = entry.get("index")
            if not isinstance(idx_raw, (int, str)):
                continue

            pos = index_to_pos[idx_raw]
            mode = entry.get("mode")
            raw_default = entry.get("default", 0.0)
            default_val = float(raw_default) if raw_default is not None else 0.0
            defaults[pos] = default_val

            if mode is None:
                continue

            # Treat all string indices as continuous.
            force_num = isinstance(idx_raw, str)
            mode_str = str(mode).lower()
            if mode_str == "num" or force_num:
                start, end = cursor, cursor + 1
                cursor = end
                slices.append(
                    _ParamSlice(
                        full_index=pos,
                        start=start,
                        end=end,
                        kind="num",
                        cardinality=1,
                        default=default_val,
                    )
                )
            elif mode_str == "cat":
                cardinal_raw = entry.get("cardinality", 0)
                cardinal = int(cardinal_raw) if cardinal_raw is not None else 0
                if cardinal <= 1:
                    raise ValueError(
                        f"Categorical parameter '{entry.get('name')}' needs cardinality>1"
                    )
                start, end = cursor, cursor + cardinal
                cursor = end
                slices.append(
                    _ParamSlice(
                        full_index=pos,
                        start=start,
                        end=end,
                        kind="cat",
                        cardinality=cardinal,
                        default=default_val,
                    )
                )
            else:
                raise ValueError(f"Unexpected mode '{mode}' for index {idx_raw}")

        self.slices = slices
        self.defaults = defaults
        self.total_dim = cursor
        self.midi_dim = len(self.string_indices)
        self.midi_start = self.total_dim - self.midi_dim

    # ------------------------------------------------------------------ #
    def encode(self, full_params, midi=None) -> torch.Tensor | np.ndarray:
        """Map a full Dexed preset (+ optional midi) into denoise space (shape [..., total_dim])."""
        if midi is None and isinstance(full_params, (tuple, list)) and len(full_params) == 2:
            full_params, midi = full_params

        if midi is not None:
            arr = self._merge_params_midi(full_params, midi)
            is_torch = torch.is_tensor(arr)
        else:
            is_torch = torch.is_tensor(full_params)
            arr = full_params
            if not is_torch:
                arr = np.asarray(full_params, dtype=np.float32)
            if arr.shape[-1] != self.param_len:
                raise ValueError(f"Expected param_len={self.param_len}, got shape {arr.shape}")
            is_torch = torch.is_tensor(arr)

        flat_shape = arr.shape[:-1]
        out_shape = flat_shape + (self.total_dim,)
        out = arr.new_zeros(out_shape) if is_torch else np.zeros(out_shape, dtype=np.float32)

        for sl in self.slices:
            values = arr[..., sl.full_index]
            if sl.kind == "num":
                if self.scale_to_unit:
                    values = values * 2.0 - 1.0
                out[..., sl.start] = values
            else:
                cls = values * float(sl.cardinality - 1)
                cls = torch.round(cls).long() if is_torch else np.rint(cls).astype(np.int64)
                cls = torch.clamp(cls, 0, sl.cardinality - 1) if is_torch else np.clip(cls, 0, sl.cardinality - 1)
                if is_torch:
                    one_hot = F.one_hot(cls, num_classes=sl.cardinality).to(out.dtype)
                    out[..., sl.start : sl.end] = one_hot
                else:
                    one_hot = np.zeros(out_shape[:-1] + (sl.cardinality,), dtype=np.float32)
                    idx = tuple(np.indices(cls.shape)) + (cls,)
                    one_hot[idx] = 1.0
                    out[..., sl.start : sl.end] = one_hot

        return out

    def decode(self, denoised) -> torch.Tensor | np.ndarray:
        """Map a denoised vector (shape [..., total_dim]) back to full Dexed params (+ midi) in [0, 1]."""
        is_torch = torch.is_tensor(denoised)
        arr = denoised if is_torch else np.asarray(denoised, dtype=np.float32)

        flat_shape = arr.shape[:-1]
        if is_torch:
            out = torch.zeros(flat_shape + (self.param_len,), device=arr.device, dtype=arr.dtype)
        else:
            out = np.zeros(flat_shape + (self.param_len,), dtype=np.float32)

        for idx, val in self.defaults.items():
            out[..., idx] = val

        for sl in self.slices:
            if sl.kind == "num":
                vals = arr[..., sl.start]
                if self.scale_to_unit:
                    vals = (vals + 1.0) * 0.5
                out[..., sl.full_index] = torch.clamp(vals, 0.0, 1.0) if is_torch else np.clip(vals, 0.0, 1.0)
            else:
                logits = arr[..., sl.start : sl.end]
                cls = torch.argmax(logits, dim=-1) if is_torch else np.argmax(logits, axis=-1)
                if is_torch:
                    cls = cls.to(dtype=out.dtype)
                    val = cls / float(sl.cardinality - 1)
                else:
                    val = cls.astype(np.float32) / float(sl.cardinality - 1)
                out[..., sl.full_index] = val

        return out

    def decode_full_and_midi_norm(
        self,
        denoised,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        decoded = self.decode(denoised)
        return decoded[..., : self.full_param_len], decoded[..., self.full_param_len :]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_param_entries(path: Path) -> List[dict]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "parameters" not in data:
            raise ValueError(f"Config must be a dict with a 'parameters' list: {path}")
        params = data["parameters"]
        if not isinstance(params, list):
            raise ValueError(f"'parameters' must be a list in {path}")
        entries: List[dict] = []
        for entry in params:
            if not isinstance(entry, dict):
                continue
            if entry.get("index") is None:
                continue
            entries.append(entry)
        # Keep numeric indices sorted; preserve file order for string indices.
        entries.sort(key=lambda e: (0, int(e["index"])) if isinstance(e["index"], int) else (1, 0))
        return entries

    def _merge_params_midi(self, params, midi):
        params_is_torch = torch.is_tensor(params)
        midi_is_torch = torch.is_tensor(midi)

        if params_is_torch or midi_is_torch:
            if not params_is_torch:
                params = torch.as_tensor(params, dtype=torch.float32, device=midi.device if midi_is_torch else None)
            if not midi_is_torch:
                midi = torch.as_tensor(midi, dtype=torch.float32, device=params.device)
            if params.ndim == 1:
                params = params.unsqueeze(0)
            if midi.ndim == 1:
                midi = midi.unsqueeze(0)
            if params.shape[0] != midi.shape[0]:
                raise ValueError(f"params batch {params.shape} and midi batch {midi.shape} mismatch")

            out = torch.zeros(params.shape[:-1] + (self.param_len,), device=params.device, dtype=params.dtype)
            out[..., : params.shape[-1]] = params
            for i, key in enumerate(self.string_indices):
                pos = self.index_to_pos[key]
                out[..., pos] = midi[..., i]
            return out

        params = np.asarray(params, dtype=np.float32)
        midi = np.asarray(midi, dtype=np.float32)
        if params.ndim == 1:
            params = params[None, :]
        if midi.ndim == 1:
            midi = midi[None, :]
        if params.shape[0] != midi.shape[0]:
            raise ValueError(f"params batch {params.shape} and midi batch {midi.shape} mismatch")

        out = np.zeros(params.shape[:-1] + (self.param_len,), dtype=np.float32)
        out[..., : params.shape[-1]] = params
        for i, key in enumerate(self.string_indices):
            pos = self.index_to_pos[key]
            out[..., pos] = midi[..., i]
        return out
