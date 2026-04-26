from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

MIDI_TOKEN_NAMES: Tuple[str, str, str] = ("MIDI_NOTE", "MIDI_VELOCITY", "MIDI_DURATION")
_SPECIAL_EOS_TOKEN = "EOS"
_MIDI_TOKEN_TO_COLUMN = {name: idx for idx, name in enumerate(MIDI_TOKEN_NAMES)}

TensorLike = Union[torch.Tensor, np.ndarray]


@dataclass(frozen=True)
class TokenFieldSpec:
    name: str
    kind: str
    cardinality: int
    mode: Optional[str]
    learnable_indices: Tuple[int, ...]
    full_index: Optional[int] = None

    @property
    def is_midi(self) -> bool:
        return self.kind == "midi"

    @property
    def is_param(self) -> bool:
        return self.kind == "param"

    @property
    def is_special(self) -> bool:
        return self.kind == "special"


class SynthTokenSpace:
    """Shared token-space adapter for synth-conditioned training.

    This module sits between the H5 dataset and model families:
    - AR / DD consume discrete token ids
    - validation render / GRPO decode token outputs back to synth full params + MIDI

    The helper already defines the synth-specific learnable space. This class
    makes the token view of that space explicit and algorithm-friendly.
    The continuous/learnable passthrough methods here are convenience helpers,
    not a substitute for a future synth denoise-space module.
    """

    def __init__(self, helper: Any, order: Sequence[str]) -> None:
        if helper is None:
            raise ValueError("SynthTokenSpace requires a parameter helper.")
        if not order:
            raise ValueError("SynthTokenSpace order must be a non-empty sequence.")

        self.helper = helper
        self.synth_learnable_size = int(helper.preset_helper.learnable_preset_size)
        self.total_learnable_size = int(helper.learnable_preset_size)
        self.midi_cardinalities = (
            int(helper.midi_cfg.note_classes),
            int(helper.midi_cfg.velocity_classes),
            int(helper.midi_cfg.duration_classes),
        )

        param_meta = tuple(helper.get_param_meta())
        self._param_meta_by_upper: Dict[str, Any] = {
            str(meta.name).upper(): meta for meta in param_meta
        }
        self._canonical_param_order = tuple(
            meta.name for meta in param_meta if tuple(int(i) for i in meta.learnable_indices)
        )

        self.order = tuple(self._resolve_name(name) for name in order)
        duplicates = self._find_duplicates(self.order)
        if duplicates:
            raise ValueError(
                f"SynthTokenSpace order contains duplicate tokens: {', '.join(duplicates)}"
            )

        self.fields = tuple(self._build_field(name) for name in self.order)
        self.cardinalities = tuple(field.cardinality for field in self.fields)
        self.max_cardinality = max(self.cardinalities)
        self.seq_len = len(self.fields)

    @classmethod
    def from_helper(
        cls,
        helper: Any,
        order: Optional[Sequence[str]] = None,
        *,
        include_eos: bool = False,
    ) -> "SynthTokenSpace":
        resolved_order = tuple(order) if order is not None else cls.canonical_order_from_helper(
            helper, include_eos=include_eos
        )
        return cls(helper=helper, order=resolved_order)

    @staticmethod
    def canonical_order_from_helper(helper: Any, *, include_eos: bool = False) -> Tuple[str, ...]:
        order = list(MIDI_TOKEN_NAMES)
        for meta in helper.get_param_meta():
            if tuple(int(i) for i in meta.learnable_indices):
                order.append(str(meta.name))
        if include_eos:
            order.append(_SPECIAL_EOS_TOKEN)
        return tuple(order)

    def build_token_targets(
        self,
        target_params: TensorLike,
        midi_classes: TensorLike,
    ) -> torch.Tensor:
        learnable, learnable_squeezed = self._as_2d_float_tensor(target_params, "target_params")
        midi, midi_squeezed = self._as_2d_long_tensor(midi_classes, "midi_classes")

        if learnable.shape[1] != self.total_learnable_size:
            raise ValueError(
                f"Expected target_params width {self.total_learnable_size}, got {learnable.shape[1]}"
            )
        if midi.shape[1] != 3:
            raise ValueError(f"Expected midi_classes width 3, got {midi.shape[1]}")
        if learnable.shape[0] != midi.shape[0]:
            raise ValueError(
                f"Batch size mismatch between target_params ({learnable.shape[0]}) "
                f"and midi_classes ({midi.shape[0]})"
            )

        seq = []
        for field in self.fields:
            if field.is_special:
                token = torch.zeros((learnable.shape[0],), dtype=torch.long, device=learnable.device)
            elif field.is_midi:
                token = midi[:, _MIDI_TOKEN_TO_COLUMN[field.name]].long()
            else:
                indices = list(field.learnable_indices)
                if len(indices) == 1 and field.mode == "num":
                    if field.cardinality <= 1:
                        token = torch.zeros((learnable.shape[0],), dtype=torch.long, device=learnable.device)
                    else:
                        scalar = learnable[:, indices[0]]
                        token = torch.round(scalar * float(field.cardinality - 1)).long()
                        if torch.any(token < 0) or torch.any(token >= field.cardinality):
                            raise ValueError(
                                f"Numeric token '{field.name}' produced out-of-range classes "
                                f"for cardinality {field.cardinality}"
                            )
                else:
                    token_slice = learnable[:, indices]
                    token = torch.argmax(token_slice, dim=-1).long()
            seq.append(token)

        tokens = torch.stack(seq, dim=1)
        if learnable_squeezed and midi_squeezed:
            return tokens[0]
        return tokens

    def build_token_targets_from_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "target_params" not in batch:
            raise KeyError("Batch is missing 'target_params'")
        if "midi_classes" not in batch:
            raise KeyError("Batch is missing 'midi_classes'")
        return self.build_token_targets(batch["target_params"], batch["midi_classes"])

    def build_continuous_targets(self, target_params: TensorLike) -> torch.Tensor:
        learnable, squeezed = self._as_2d_float_tensor(target_params, "target_params")
        if learnable.shape[1] != self.total_learnable_size:
            raise ValueError(
                f"Expected target_params width {self.total_learnable_size}, got {learnable.shape[1]}"
            )
        if squeezed:
            return learnable[0]
        return learnable

    def build_continuous_targets_from_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "target_params" not in batch:
            raise KeyError("Batch is missing 'target_params'")
        return self.build_continuous_targets(batch["target_params"])

    def token_ids_to_learnable(self, token_ids: TensorLike) -> np.ndarray:
        tokens, squeezed = self._as_2d_long_array(token_ids, "token_ids")
        if tokens.shape[1] != self.seq_len:
            raise ValueError(f"Expected token_ids width {self.seq_len}, got {tokens.shape[1]}")

        learnable = np.zeros((tokens.shape[0], self.total_learnable_size), dtype=np.float32)
        midi_note_start = self.synth_learnable_size
        midi_vel_start = midi_note_start + self.midi_cardinalities[0]
        midi_dur_start = midi_vel_start + self.midi_cardinalities[1]
        batch_index = np.arange(tokens.shape[0], dtype=np.int64)

        for token_idx, field in enumerate(self.fields):
            cls = tokens[:, token_idx]
            if np.any(cls < 0) or np.any(cls >= field.cardinality):
                raise ValueError(
                    f"Token '{field.name}' contains out-of-range classes for cardinality {field.cardinality}"
                )

            if field.is_special:
                continue

            if field.is_midi:
                if field.name == "MIDI_NOTE":
                    learnable[batch_index, midi_note_start + cls] = 1.0
                elif field.name == "MIDI_VELOCITY":
                    learnable[batch_index, midi_vel_start + cls] = 1.0
                elif field.name == "MIDI_DURATION":
                    learnable[batch_index, midi_dur_start + cls] = 1.0
                else:
                    raise ValueError(f"Unsupported MIDI token name: {field.name}")
                continue

            indices = tuple(int(i) for i in field.learnable_indices)
            if len(indices) == 1 and field.mode == "num":
                if field.cardinality <= 1:
                    learnable[:, indices[0]] = 0.0
                else:
                    learnable[:, indices[0]] = cls.astype(np.float32) / float(field.cardinality - 1)
                continue

            mapped = np.asarray(indices, dtype=np.int64)[cls]
            learnable[batch_index, mapped] = 1.0

        if squeezed:
            return learnable[0]
        return learnable

    def token_ids_to_full_and_midi(
        self,
        token_ids: TensorLike,
        *,
        apply_defaults: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        learnable = self.token_ids_to_learnable(token_ids)
        return self.helper.learnable_to_full(learnable, apply_defaults=apply_defaults)

    def continuous_to_full_and_midi(
        self,
        target_params: TensorLike,
        *,
        apply_defaults: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        learnable = np.asarray(target_params, dtype=np.float32)
        if learnable.ndim == 1 and learnable.shape[0] != self.total_learnable_size:
            raise ValueError(
                f"Expected continuous target width {self.total_learnable_size}, got {learnable.shape[0]}"
            )
        if learnable.ndim == 2 and learnable.shape[1] != self.total_learnable_size:
            raise ValueError(
                f"Expected continuous target width {self.total_learnable_size}, got {learnable.shape[1]}"
            )
        return self.helper.learnable_to_full(learnable, apply_defaults=apply_defaults)

    def _resolve_name(self, name: str) -> str:
        raw = str(name).strip()
        if not raw:
            raise ValueError("Target order contains an empty token name.")

        upper = raw.upper()
        if upper in _MIDI_TOKEN_TO_COLUMN or upper == _SPECIAL_EOS_TOKEN:
            return upper

        meta = self._param_meta_by_upper.get(upper)
        if meta is None:
            raise KeyError(f"Unknown target token '{raw}' for current synth schema")
        return str(meta.name)

    def _build_field(self, name: str) -> TokenFieldSpec:
        if name in _MIDI_TOKEN_TO_COLUMN:
            midi_idx = _MIDI_TOKEN_TO_COLUMN[name]
            return TokenFieldSpec(
                name=name,
                kind="midi",
                cardinality=int(self.midi_cardinalities[midi_idx]),
                mode="cat",
                learnable_indices=(),
                full_index=None,
            )

        if name == _SPECIAL_EOS_TOKEN:
            return TokenFieldSpec(
                name=name,
                kind="special",
                cardinality=1,
                mode=None,
                learnable_indices=(),
                full_index=None,
            )

        meta = self._param_meta_by_upper[name.upper()]
        indices = tuple(int(i) for i in meta.learnable_indices)
        if not indices:
            raise ValueError(f"Parameter '{meta.name}' is not learnable and cannot appear in target order")

        cardinality = int(meta.cardinality)
        if cardinality <= 0:
            raise ValueError(f"Parameter '{meta.name}' has invalid cardinality {cardinality}")

        return TokenFieldSpec(
            name=str(meta.name),
            kind="param",
            cardinality=cardinality,
            mode=meta.mode,
            learnable_indices=indices,
            full_index=int(meta.full_index),
        )

    @staticmethod
    def _find_duplicates(names: Sequence[str]) -> Tuple[str, ...]:
        seen = set()
        duplicates = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        return tuple(duplicates)

    @staticmethod
    def _as_2d_float_tensor(value: TensorLike, label: str) -> Tuple[torch.Tensor, bool]:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.float()
        squeezed = False
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        if tensor.ndim != 2:
            raise ValueError(f"{label} must have shape (D,) or (B, D), got {tuple(tensor.shape)}")
        return tensor, squeezed

    @staticmethod
    def _as_2d_long_tensor(value: TensorLike, label: str) -> Tuple[torch.Tensor, bool]:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.long()
        squeezed = False
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        if tensor.ndim != 2:
            raise ValueError(f"{label} must have shape (D,) or (B, D), got {tuple(tensor.shape)}")
        return tensor, squeezed

    @staticmethod
    def _as_2d_long_array(value: TensorLike, label: str) -> Tuple[np.ndarray, bool]:
        arr = np.asarray(value, dtype=np.int64)
        squeezed = False
        if arr.ndim == 1:
            arr = arr[None, :]
            squeezed = True
        if arr.ndim != 2:
            raise ValueError(f"{label} must have shape (L,) or (B, L), got {arr.shape}")
        return arr, squeezed
