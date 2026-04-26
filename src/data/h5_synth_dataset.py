"""Shared HDF5 dataset for synth parameter matching (Dexed + Surge XT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.synth_backends.dexed.dexed_bridge import DexedParameterHelper
from src.data.synth_backends.surge.surge_bridge import SurgeParameterHelper
from src.project_paths import resolve_project_path


def _to_path(path_like: Union[str, Path]) -> Path:
    return resolve_project_path(path_like)


class H5SynthDataset(Dataset):
    """HDF5-backed dataset with shared logic for Dexed and Surge XT.

    Expected file layout under ``dataset_root``:
    - ``{split}.h5`` containing datasets:
      - ``parameters``: (N, P) float parameter vectors in metadata order.
      - ``midi``: (N, 3) normalized MIDI [note, velocity, duration].
      - ``mel_spec``: (N, n_mels, frames) packed input feature tensor.
      - ``audio``: (N, T) optional unless ``read_audio``.
      - ``preset_id``: optional.
    - ``dataset_metadata.json`` with frozen ``parameter_schema``, ``midi_representation``,
      feature normalization policy, and audio metadata.
    - ``stats.npz`` required iff metadata declares ``use_saved_mean_and_variance: true``.
    """

    VALID_SYNTHS = {"dexed", "surge"}

    def __init__(
        self,
        dataset_root: Union[str, Path],
        split: str,
        read_audio: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = _to_path(dataset_root)
        self.split = str(split)

        self.read_audio = bool(read_audio)

        self.h5_path = self.dataset_root / f"{self.split}.h5"
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 split not found: {self.h5_path}")

        meta_path = self.dataset_root / "dataset_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        parameter_schema = meta.get("parameter_schema")
        if not isinstance(parameter_schema, Mapping):
            raise ValueError(
                "dataset_metadata.json must define a mapping `parameter_schema` so the dataset "
                "is self-describing and does not depend on external YAML."
            )
        metadata_synth = meta.get("synth")
        if metadata_synth is None:
            raise ValueError("dataset_metadata.json must declare `synth`.")
        self.synth = str(metadata_synth).lower()
        if self.synth == "surge_xt":
            self.synth = "surge"
        if self.synth not in self.VALID_SYNTHS:
            raise ValueError(
                f"Unsupported metadata synth '{metadata_synth}'. Expected one of {sorted(self.VALID_SYNTHS)}"
            )

        midi_keys = meta.get("midi_keys")
        if not isinstance(midi_keys, list) or len(midi_keys) != 3:
            raise ValueError("dataset_metadata.json must declare `midi_keys` as a 3-item list.")
        self.midi_keys = [str(x) for x in midi_keys]

        sample_rate = meta.get("sample_rate")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("dataset_metadata.json must declare a positive integer `sample_rate`.")
        self.sample_rate = int(sample_rate)
        midi_representation = meta.get("midi_representation")
        if midi_representation != "normalized":
            raise ValueError(
                "dataset_metadata.json must declare `midi_representation: normalized`. "
                f"Found: {midi_representation!r}"
            )
        use_saved_mean_and_variance = meta.get("use_saved_mean_and_variance")
        if not isinstance(use_saved_mean_and_variance, bool):
            raise ValueError(
                "dataset_metadata.json must declare boolean `use_saved_mean_and_variance`."
            )
        self.use_saved_mean_and_variance = use_saved_mean_and_variance
        min_rms = meta.get("min_rms")
        if isinstance(min_rms, bool) or not isinstance(min_rms, (int, float)) or float(min_rms) < 0.0:
            raise ValueError(
                "dataset_metadata.json must declare non-negative numeric `min_rms`."
            )
        self.dataset_min_rms = float(min_rms)

        self._file: Optional[h5py.File] = None
        self._length: Optional[int] = None

        if self.synth == "dexed":
            self._helper = DexedParameterHelper.from_schema(parameter_schema)
        else:
            self._helper = SurgeParameterHelper.from_schema(parameter_schema)

        self.param_names = list(self._helper.preset_helper.vst_param_names)

        helper_midi_cfg = {
            "note_min": int(self._helper.midi_cfg.note_min),
            "note_classes": int(self._helper.midi_cfg.note_classes),
            "velocity_classes": int(self._helper.midi_cfg.velocity_classes),
            "duration_min": float(self._helper.midi_cfg.duration_min),
            "duration_max": float(self._helper.midi_cfg.duration_max),
            "duration_classes": int(self._helper.midi_cfg.duration_classes),
        }
        self.note_min = helper_midi_cfg["note_min"]
        self.note_classes = helper_midi_cfg["note_classes"]
        self.velocity_classes = helper_midi_cfg["velocity_classes"]
        self.duration_min = helper_midi_cfg["duration_min"]
        self.duration_max = helper_midi_cfg["duration_max"]
        self.duration_classes = helper_midi_cfg["duration_classes"]

        self.backend_param_names = list(self._helper.preset_helper.vst_param_names)
        self.backend_param_count = len(self.backend_param_names)

        # Map metadata parameter names -> H5 column index.
        self._param_name_to_idx = {name: i for i, name in enumerate(self.param_names)}

        self.mel_mean: Optional[np.ndarray] = None
        self.mel_std: Optional[np.ndarray] = None
        if self.use_saved_mean_and_variance:
            stats_path = self.dataset_root / "stats.npz"
            if not stats_path.exists():
                raise FileNotFoundError(
                    "use_saved_mean_and_variance=True requires stats.npz to exist at "
                    f"{stats_path}"
                )
            stats = np.load(stats_path)
            self.mel_mean = stats.get("mean")
            self.mel_std = stats.get("std")
            if self.mel_mean is None or self.mel_std is None:
                raise KeyError(
                    f"stats.npz must contain both 'mean' and 'std' arrays: {stats_path}"
                )

        self._validate_structure()

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _validate_structure(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            required = ["parameters", "midi"]
            for key in required:
                if key not in f:
                    raise KeyError(f"Missing dataset '{key}' in {self.h5_path}")

            num_samples = int(f["parameters"].shape[0])
            param_dim = int(f["parameters"].shape[1])
            if not self.param_names:
                raise ValueError(
                    "dataset_metadata.json must define a non-empty frozen `parameter_schema`; "
                    "implicit or external parameter ordering is not supported."
                )
            if len(self.param_names) != param_dim:
                raise ValueError(
                    f"HDF5 parameters dim {param_dim} does not match frozen schema parameter count {len(self.param_names)}"
                )
            if len(set(self.param_names)) != len(self.param_names):
                raise ValueError("Frozen parameter_schema contains duplicate parameter names.")

            midi_shape = f["midi"].shape
            if len(midi_shape) != 2 or midi_shape[0] != num_samples or midi_shape[1] != 3:
                raise ValueError(f"Expected midi shape (N,3), got {midi_shape}")

            if "mel_spec" in f:
                mel_shape = f["mel_spec"].shape
                if mel_shape[0] != num_samples:
                    raise ValueError(f"mel_spec length {mel_shape[0]} does not match parameters length {num_samples}")
            else:
                raise KeyError("mel_spec dataset missing; packed H5 datasets must include mel_spec.")

            if self.read_audio:
                if "audio" not in f:
                    raise KeyError("audio dataset missing but read_audio=True requested.")
                audio_shape = f["audio"].shape
                if audio_shape[0] != num_samples:
                    raise ValueError(f"audio length {audio_shape[0]} does not match parameters length {num_samples}")

            self._length = num_samples

    def __len__(self) -> int:
        if self._length is None:
            h5 = self._ensure_file()
            self._length = int(h5["parameters"].shape[0])
        return int(self._length)

    def _midi_to_classes(self, midi_norm: np.ndarray) -> np.ndarray:
        note = int(np.clip(round(float(midi_norm[0]) * (self.note_classes - 1)), 0, self.note_classes - 1))
        velocity = int(
            np.clip(round(float(midi_norm[1]) * (self.velocity_classes - 1)), 0, self.velocity_classes - 1)
        )
        duration = int(
            np.clip(round(float(midi_norm[2]) * (self.duration_classes - 1)), 0, self.duration_classes - 1)
        )
        return np.asarray([note, velocity, duration], dtype=np.int64)

    def _midi_norm_to_absolute(self, midi_norm: np.ndarray) -> np.ndarray:
        note = self.note_min + float(midi_norm[0]) * float(max(self.note_classes - 1, 1))
        velocity = float(midi_norm[1]) * float(max(self.velocity_classes - 1, 1))
        duration = self.duration_min + float(midi_norm[2]) * float(max(self.duration_max - self.duration_min, 1e-8))
        return np.asarray([note, velocity, duration], dtype=np.float32)

    def _coerce_midi_norm(self, midi_vec: np.ndarray) -> np.ndarray:
        midi_vec = np.asarray(midi_vec, dtype=np.float32)
        if not np.isfinite(midi_vec).all():
            raise ValueError(f"Normalized MIDI contains NaN/Inf values: {midi_vec!r}")
        if np.any(midi_vec < -1e-6) or np.any(midi_vec > 1.0 + 1e-6):
            raise ValueError(
                "H5 midi dataset is declared normalized but contains values outside [0, 1]: "
                f"{midi_vec!r}"
            )
        return midi_vec.astype(np.float32, copy=False)

    def _reorder_params_to_backend(self, param_vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(param_vec, dtype=np.float32)
        full = np.zeros((self.backend_param_count,), dtype=np.float32)

        missing_names = []
        for idx, name in enumerate(self.backend_param_names):
            src_idx = self._param_name_to_idx.get(name)

            if src_idx is None:
                missing_names.append(name)
                continue

            if src_idx < len(vec):
                full[idx] = float(vec[src_idx])

        if missing_names:
            sample = ", ".join(missing_names[:8])
            raise KeyError(
                f"Missing {len(missing_names)} backend parameters in metadata/H5 order mapping (examples: {sample})"
            )

        return full

    @staticmethod
    def _apply_saved_stats(
        mel: torch.Tensor,
        mean_np: Optional[np.ndarray],
        std_np: Optional[np.ndarray],
    ) -> torch.Tensor:
        if mean_np is None or std_np is None:
            return mel

        mean = torch.from_numpy(np.asarray(mean_np, dtype=np.float32))
        std = torch.from_numpy(np.asarray(std_np, dtype=np.float32))
        if mean.ndim + 1 == mel.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (mel - mean) / (std + 1e-8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5 = self._ensure_file()

        param_vec = np.asarray(h5["parameters"][idx], dtype=np.float32)
        midi_raw = np.asarray(h5["midi"][idx], dtype=np.float32)
        midi_norm = self._coerce_midi_norm(midi_raw)

        audio_tensor: Optional[torch.Tensor] = None
        if self.read_audio and "audio" in h5:
            audio = np.asarray(h5["audio"][idx], dtype=np.float32)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

        mel_np = np.asarray(h5["mel_spec"][idx], dtype=np.float32)
        mel = torch.from_numpy(mel_np).unsqueeze(0) if mel_np.ndim == 2 else torch.from_numpy(mel_np)
        mel = self._apply_saved_stats(mel, self.mel_mean, self.mel_std)

        full_backend = self._reorder_params_to_backend(param_vec)
        midi_abs = self._midi_norm_to_absolute(midi_norm)
        learnable = self._helper.full_to_learnable(full_backend, midi=midi_abs)

        sample: Dict[str, torch.Tensor] = {
            "mel_spec": mel.float(),
            "full_parameters_h5": torch.from_numpy(param_vec).float(),
            "full_parameters_backend": torch.from_numpy(full_backend).float(),
            "target_params": torch.from_numpy(np.asarray(learnable, dtype=np.float32)).float(),
            "midi_norm": torch.from_numpy(midi_norm).float(),
            "midi_classes": torch.from_numpy(self._midi_to_classes(midi_norm)),
        }
        if audio_tensor is not None:
            sample["audio"] = audio_tensor.float()
        return sample

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
