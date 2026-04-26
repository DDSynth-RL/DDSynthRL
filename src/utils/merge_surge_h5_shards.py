"""Merge Surge shard H5 files into final train/val/test datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from src.project_paths import project_relative_string
from src.utils.generate_surge_h5_dataset import (
    MelStatsTracker,
    _build_generation_context,
    _build_metadata,
    _load_dataset_recipe,
    _resolve_path,
    _shard_layout_for_split,
)

logger = logging.getLogger(__name__)


def _require_attr(h5: h5py.File, key: str) -> object:
    if key not in h5.attrs:
        raise ValueError(f"Shard file {h5.filename} is missing required attr `{key}`")
    return h5.attrs[key]


def _copy_split_from_shards(
    *,
    split_name: str,
    shard_root: Path,
    shard_layout: list[tuple[int, int]],
    common_fingerprint: str,
    out_path: Path,
    stats_tracker: Optional[MelStatsTracker],
    copy_batch_size: int,
) -> int:
    first_path = shard_root / split_name / f"{split_name}_000.h5"
    if not first_path.exists():
        raise FileNotFoundError(f"Missing first shard for split `{split_name}`: {first_path}")

    with h5py.File(first_path, "r") as first_h5:
        if str(_require_attr(first_h5, "soundmgm_format")) != "surge_h5_shard_v1":
            raise ValueError(f"{first_path} is not a Surge shard H5")
        audio_len = int(first_h5["audio"].shape[1])
        mel_bins = int(first_h5["mel_spec"].shape[1])
        mel_frames = int(first_h5["mel_spec"].shape[2])
        num_params = int(first_h5["parameters"].shape[1])
        sample_rate = int(first_h5["audio"].attrs["sample_rate"])
        target_duration = float(first_h5["audio"].attrs["target_duration"])

    total_count = int(sum(int(count) for _, count in shard_layout))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with h5py.File(out_path, "w") as out_h5:
        audio_ds = out_h5.create_dataset("audio", shape=(total_count, audio_len), dtype=np.float16, compression=None)
        mel_ds = out_h5.create_dataset(
            "mel_spec", shape=(total_count, mel_bins, mel_frames), dtype=np.float32, compression=None
        )
        param_ds = out_h5.create_dataset(
            "parameters", shape=(total_count, num_params), dtype=np.float32, compression=None
        )
        midi_ds = out_h5.create_dataset("midi", shape=(total_count, 3), dtype=np.float32, compression=None)
        preset_ds = out_h5.create_dataset(
            "preset_id",
            shape=(total_count,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression=None,
        )
        audio_ds.attrs["sample_rate"] = sample_rate
        audio_ds.attrs["target_duration"] = target_duration

        for shard_index, (global_start_index, expected_count) in enumerate(shard_layout):
            shard_path = shard_root / split_name / f"{split_name}_{shard_index:03d}.h5"
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard for split `{split_name}`: {shard_path}")

            with h5py.File(shard_path, "r") as shard_h5:
                split_attr = str(_require_attr(shard_h5, "split_name"))
                if split_attr != split_name:
                    raise ValueError(f"{shard_path} split_name={split_attr!r}, expected {split_name!r}")
                if int(_require_attr(shard_h5, "shard_index")) != shard_index:
                    raise ValueError(f"{shard_path} has wrong shard_index")
                if int(_require_attr(shard_h5, "shard_count")) != len(shard_layout):
                    raise ValueError(f"{shard_path} has wrong shard_count")
                if int(_require_attr(shard_h5, "global_start_index")) != int(global_start_index):
                    raise ValueError(f"{shard_path} has wrong global_start_index")
                if int(_require_attr(shard_h5, "num_samples")) != int(expected_count):
                    raise ValueError(f"{shard_path} has wrong num_samples")
                if str(_require_attr(shard_h5, "common_fingerprint")) != common_fingerprint:
                    raise ValueError(f"{shard_path} common_fingerprint mismatch")
                if bool(_require_attr(shard_h5, "write_complete")) is not True:
                    raise ValueError(f"{shard_path} is not marked write_complete=true")
                if int(_require_attr(shard_h5, "written_samples")) != int(expected_count):
                    raise ValueError(
                        f"{shard_path} written_samples mismatch: "
                        f"{int(_require_attr(shard_h5, 'written_samples'))} != {expected_count}"
                    )

                if shard_h5["audio"].shape != (expected_count, audio_len):
                    raise ValueError(f"{shard_path} has unexpected audio shape {shard_h5['audio'].shape}")
                if shard_h5["mel_spec"].shape != (expected_count, mel_bins, mel_frames):
                    raise ValueError(f"{shard_path} has unexpected mel shape {shard_h5['mel_spec'].shape}")
                if shard_h5["parameters"].shape != (expected_count, num_params):
                    raise ValueError(f"{shard_path} has unexpected parameter shape {shard_h5['parameters'].shape}")
                if shard_h5["midi"].shape != (expected_count, 3):
                    raise ValueError(f"{shard_path} has unexpected MIDI shape {shard_h5['midi'].shape}")
                if int(shard_h5["audio"].attrs["sample_rate"]) != sample_rate:
                    raise ValueError(f"{shard_path} has mismatched sample_rate")
                if float(shard_h5["audio"].attrs["target_duration"]) != target_duration:
                    raise ValueError(f"{shard_path} has mismatched target_duration")

                shard_audio = shard_h5["audio"]
                shard_mel = shard_h5["mel_spec"]
                shard_params = shard_h5["parameters"]
                shard_midi = shard_h5["midi"]
                shard_preset = shard_h5["preset_id"]

                for start in range(0, expected_count, copy_batch_size):
                    end = min(start + copy_batch_size, expected_count)
                    batch_len = end - start
                    audio_ds[written : written + batch_len, :] = shard_audio[start:end, :]
                    mel_batch = shard_mel[start:end, :, :]
                    mel_ds[written : written + batch_len, :, :] = mel_batch
                    param_ds[written : written + batch_len, :] = shard_params[start:end, :]
                    midi_ds[written : written + batch_len, :] = shard_midi[start:end, :]
                    preset_ds[written : written + batch_len] = shard_preset[start:end]
                    if stats_tracker is not None:
                        for mel in mel_batch:
                            stats_tracker.update(np.asarray(mel, dtype=np.float32))
                    written += batch_len

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Surge shard H5 files into final train/val/test splits.")
    parser.add_argument("--config", type=str, required=True, help="Dataset recipe YAML describing generation.")
    parser.add_argument(
        "--copy-batch-size",
        type=int,
        default=32,
        help="How many samples to copy from each shard H5 at once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if int(args.copy_batch_size) < 1:
        raise ValueError("--copy-batch-size must be >= 1")

    recipe_path = _resolve_path(args.config)
    recipe = _load_dataset_recipe(recipe_path)
    output_root = recipe.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = recipe.sharding.shard_root

    helper, sampling_priors, _midi_cfg, splits, mel_frames, _common_payload, common_fingerprint = (
        _build_generation_context(recipe, recipe_path)
    )

    logger.info("Shard root: %s", shard_root)
    logger.info("Output root: %s", output_root)

    train_stats = MelStatsTracker()
    for split_name in ("train", "val", "test"):
        split_total = int(splits[split_name])
        shard_layout = _shard_layout_for_split(recipe, split_name, split_total)
        out_path = output_root / f"{split_name}.h5"
        logger.info(
            "Merging %s split from %d shards (%d samples) -> %s",
            split_name,
            len(shard_layout),
            split_total,
            out_path,
        )
        written = _copy_split_from_shards(
            split_name=split_name,
            shard_root=shard_root,
            shard_layout=shard_layout,
            common_fingerprint=common_fingerprint,
            out_path=out_path,
            stats_tracker=train_stats if split_name == "train" else None,
            copy_batch_size=int(args.copy_batch_size),
        )
        logger.info("%s: wrote %d samples", split_name, written)

    mean, std = train_stats.finalize()
    np.savez(output_root / "stats.npz", mean=mean, std=std)
    logger.info("Saved mel stats -> %s", output_root / "stats.npz")

    metadata = _build_metadata(
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
        splits=splits,
        train_ratio=float(recipe.train_ratio),
        val_ratio=float(recipe.val_ratio),
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
        sharding={
            "shard_root": project_relative_string(recipe.sharding.shard_root),
            "train_shards": int(recipe.sharding.train_shards),
            "val_shards": int(recipe.sharding.val_shards),
            "test_shards": int(recipe.sharding.test_shards),
            "common_fingerprint": common_fingerprint,
        },
    )
    with (output_root / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata -> %s", output_root / "dataset_metadata.json")


if __name__ == "__main__":
    main()
