"""Shared Lightning DataModule for frozen-schema H5 synth datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import lightning as L
from torch.utils.data import DataLoader

from src.data.h5_synth_dataset import H5SynthDataset


class H5SynthDataModule(L.LightningDataModule):
    """Shared DataModule for Dexed and Surge XT HDF5 datasets.

    Expected config shape passed through ``**kwargs``:
    - ``root``: dataset directory containing ``train.h5/val.h5/test.h5``.
    - optional ``read_audio`` and optional val/test ``read_audio`` overrides.
    - ``training`` mapping with:
      - ``batch_size``
      - ``num_workers``
      - optional ``pin_memory``, ``drop_last``, ``persistent_workers``
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.cfg = dict(kwargs)
        self.train_dataset: Optional[H5SynthDataset] = None
        self.val_dataset: Optional[H5SynthDataset] = None
        self.test_dataset: Optional[H5SynthDataset] = None

    def _assert_no_external_schema_overrides(self) -> None:
        forbidden = (
            "summary_path",
            "dexed_summary_path",
            "dexed_config",
            "surge_summary_path",
            "surge_config",
            "synth",
            "midi",
            "midi_is_normalized",
            "spec",
            "compute_mel",
            "val_compute_mel",
            "test_compute_mel",
            "use_saved_mean_and_variance",
            "min_rms",
            "batch_size",
            "num_workers",
            "pin_memory",
            "drop_last",
            "persistent_workers",
        )
        present = [key for key in forbidden if key in self.cfg]
        if present:
            raise ValueError(
                "H5SynthDataModule does not accept external schema, MIDI, or feature overrides. "
                "Dataset semantics must come from frozen dataset_metadata.json only. "
                f"Remove: {', '.join(present)}"
            )

    def _require_mapping(self, key: str) -> Mapping[str, Any]:
        value = self.cfg.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"H5SynthDataModule requires `{key}` to be a mapping.")
        return value

    def _build_dataset(self, split: str, read_audio: bool) -> H5SynthDataset:
        root = self.cfg.get("root")
        if root is None:
            raise ValueError("H5SynthDataModule requires `root` in config.")

        self._assert_no_external_schema_overrides()

        return H5SynthDataset(
            dataset_root=Path(str(root)),
            split=split,
            read_audio=read_audio,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_split = str(self.cfg.get("train_split", "train"))
        val_split = str(self.cfg.get("val_split", "val"))
        test_split = str(self.cfg.get("test_split", "test"))

        read_audio = bool(self.cfg.get("read_audio", False))
        val_read_audio = bool(self.cfg.get("val_read_audio", read_audio))
        test_read_audio = bool(self.cfg.get("test_read_audio", val_read_audio))

        if stage in (None, "fit", "train"):
            self.train_dataset = self._build_dataset(
                split=train_split,
                read_audio=read_audio,
            )

        if stage in (None, "fit", "validate"):
            self.val_dataset = self._build_dataset(
                split=val_split,
                read_audio=val_read_audio,
            )

        if stage in (None, "test"):
            self.test_dataset = self._build_dataset(
                split=test_split,
                read_audio=test_read_audio,
            )

    def _loader_kwargs(self, shuffle: bool) -> dict[str, Any]:
        training_cfg = self._require_mapping("training")
        missing = [key for key in ("batch_size", "num_workers") if key not in training_cfg]
        if missing:
            raise ValueError(f"H5SynthDataModule training config missing required keys: {missing}")

        batch_size = int(training_cfg["batch_size"])
        num_workers = int(training_cfg["num_workers"])
        pin_memory = bool(training_cfg.get("pin_memory", True))
        drop_last = bool(training_cfg.get("drop_last", False))
        persistent_workers = bool(training_cfg.get("persistent_workers", num_workers > 0))

        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": shuffle,
            "pin_memory": pin_memory,
            "drop_last": drop_last if shuffle else False,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader().")
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') or setup('validate') before requesting val_dataloader().")
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test_dataloader().")
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))

    def teardown(self, stage: Optional[str] = None) -> None:
        for ds in (self.train_dataset, self.val_dataset, self.test_dataset):
            if ds is not None and hasattr(ds, "close"):
                ds.close()
