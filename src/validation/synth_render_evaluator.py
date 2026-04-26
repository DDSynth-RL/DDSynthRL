from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from src.data.synth_backends.dexed.dexed_renderer import SubprocessDexedRenderer
from src.data.synth_backends.surge.surge_renderer import SurgePedalboardRenderer
from src.project_paths import resolve_project_path
from src.utils.audio_metrics import (
    ClapCosineDistance,
    LibrosaMFCCTransform,
    WmfccMetric,
    compute_extra_audio_metrics,
)


class SynthRenderEvaluator:
    """Render synth predictions and score them against target audio."""

    def __init__(
        self,
        *,
        synth: str,
        helper: Any,
        renderer_cfg: Mapping[str, Any],
        metrics_cfg: Mapping[str, Any],
        dataset_sample_rate: int,
        dataset_target_duration: float,
        device: torch.device | str | None = None,
    ) -> None:
        self.synth = str(synth).lower()
        if self.synth not in {"dexed", "surge"}:
            raise ValueError(f"Unsupported synth for render evaluation: {synth!r}")

        self.helper = helper
        self.backend_param_names = tuple(str(name) for name in helper.preset_helper.vst_param_names)
        self.renderer_cfg = dict(renderer_cfg)
        self.metrics_cfg = dict(metrics_cfg)

        self.sample_rate = int(self.renderer_cfg.get("sample_rate", dataset_sample_rate))
        if self.sample_rate != int(dataset_sample_rate):
            raise ValueError(
                "renderer.sample_rate must match frozen dataset sample_rate. "
                f"Got renderer={self.sample_rate}, dataset={dataset_sample_rate}"
            )

        self.target_duration = float(self.renderer_cfg.get("target_duration", dataset_target_duration))
        if abs(self.target_duration - float(dataset_target_duration)) > 1e-8:
            raise ValueError(
                "renderer.target_duration must match frozen dataset target_duration. "
                f"Got renderer={self.target_duration}, dataset={dataset_target_duration}"
            )

        self.fadeout = float(self.renderer_cfg.get("fadeout", 0.1))
        self.release_ratio = float(self.renderer_cfg.get("release_ratio", 0.5))
        if self.release_ratio <= 0.0:
            raise ValueError(f"renderer.release_ratio must be positive, got {self.release_ratio!r}")

        self.renderer = self._build_renderer()
        self.wmfcc_metric = (
            WmfccMetric(sample_rate=self.sample_rate) if bool(self.metrics_cfg.get("wmfcc", False)) else None
        )
        self.mfcc13 = (
            LibrosaMFCCTransform(sample_rate=self.sample_rate, n_mfcc=13)
            if bool(self.metrics_cfg.get("mfcc13", False))
            else None
        )
        self.mfcc40 = (
            LibrosaMFCCTransform(sample_rate=self.sample_rate, n_mfcc=40)
            if bool(self.metrics_cfg.get("mfcc40", False))
            else None
        )
        self.clap_metric = (
            ClapCosineDistance(device=device) if bool(self.metrics_cfg.get("clap", False)) else None
        )

    @property
    def target_num_samples(self) -> int:
        return int(round(self.sample_rate * self.target_duration))

    def render_from_full_and_midi(
        self,
        full_parameters: np.ndarray | torch.Tensor,
        midi_absolute: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        full_np = self._to_2d_float_numpy(full_parameters, "full_parameters")
        midi_np = self._to_2d_float_numpy(midi_absolute, "midi_absolute")
        if full_np.shape[0] != midi_np.shape[0]:
            raise ValueError(
                f"Batch mismatch between full_parameters ({full_np.shape[0]}) and midi_absolute ({midi_np.shape[0]})"
            )
        if midi_np.shape[1] != 3:
            raise ValueError(f"Expected midi_absolute width 3, got {midi_np.shape[1]}")

        return self._render_batch(full_np, midi_np)

    def render_from_full_and_midi_resilient(
        self,
        full_parameters: np.ndarray | torch.Tensor,
        midi_absolute: np.ndarray | torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        full_np = self._to_2d_float_numpy(full_parameters, "full_parameters")
        midi_np = self._to_2d_float_numpy(midi_absolute, "midi_absolute")
        if full_np.shape[0] != midi_np.shape[0]:
            raise ValueError(
                f"Batch mismatch between full_parameters ({full_np.shape[0]}) and midi_absolute ({midi_np.shape[0]})"
            )
        if midi_np.shape[1] != 3:
            raise ValueError(f"Expected midi_absolute width 3, got {midi_np.shape[1]}")

        try:
            return self._render_batch(full_np, midi_np), np.arange(full_np.shape[0], dtype=np.int64)
        except Exception:
            rendered: list[np.ndarray] = []
            kept: list[int] = []
            for idx, (full_vec, midi_vec) in enumerate(zip(full_np, midi_np)):
                try:
                    rendered.append(self._render_one(full_vec, midi_vec))
                    kept.append(idx)
                except Exception:
                    continue
            if not rendered:
                raise
            return torch.from_numpy(np.stack(rendered)).float(), np.asarray(kept, dtype=np.int64)

    def compute_metrics(
        self,
        pred_audio: torch.Tensor | np.ndarray,
        target_audio: torch.Tensor | np.ndarray,
    ) -> Dict[str, float]:
        pred_t = self._to_audio_tensor(pred_audio)
        target_t = self._to_audio_tensor(target_audio)
        if pred_t.shape != target_t.shape:
            raise ValueError(f"Audio shape mismatch: pred {tuple(pred_t.shape)} vs target {tuple(target_t.shape)}")

        metrics: Dict[str, float] = {}
        if self.wmfcc_metric is not None:
            metrics["wmfcc"] = float(self.wmfcc_metric(pred_t, target_t).item())
        if self.mfcc13 is not None:
            pred_mfcc13 = self.mfcc13(pred_t)
            targ_mfcc13 = self.mfcc13(target_t)
            metrics["mfcc13"] = float(torch.mean(torch.abs(pred_mfcc13 - targ_mfcc13)).item())
        if self.mfcc40 is not None:
            pred_mfcc40 = self.mfcc40(pred_t)
            targ_mfcc40 = self.mfcc40(target_t)
            metrics["mfcc40"] = float(torch.mean(torch.abs(pred_mfcc40 - targ_mfcc40)).item())

        extra = compute_extra_audio_metrics(
            pred_t,
            target_t,
            self.sample_rate,
            mss=bool(self.metrics_cfg.get("mss", False)),
            sot=bool(self.metrics_cfg.get("sot", False)),
            rms=bool(self.metrics_cfg.get("rms", False)),
        )
        metrics.update(extra)

        if self.clap_metric is not None:
            metrics["clap"] = float(self.clap_metric(pred_t, target_t, sample_rate=self.sample_rate))
        return metrics

    def close(self) -> None:
        renderer = getattr(self, "renderer", None)
        if renderer is not None and hasattr(renderer, "close"):
            renderer.close()
        self.renderer = None

    def _build_renderer(self):
        if self.synth == "dexed":
            dexed_cfg = dict(self.renderer_cfg.get("dexed", {}))
            synth_path = resolve_project_path(dexed_cfg.get("synth_path", "synth/Dexed.vst3"))
            return SubprocessDexedRenderer(
                synth_path=str(synth_path),
                sample_rate=self.sample_rate,
                block_size=int(dexed_cfg.get("block_size", 512)),
                fadeout_seconds=self.fadeout,
                convert_to_mono=True,
                normalize_audio=False,
                note_on_delay=float(dexed_cfg.get("note_on_delay", 0.01)),
            )

        surge_cfg = dict(self.renderer_cfg.get("surge", {}))
        plugin_path = resolve_project_path(surge_cfg.get("plugin_path", "synth/Surge XT.vst3"))
        preset_value = surge_cfg.get("preset_path", "presets/surge-base.vstpreset")
        preset_path = resolve_project_path(preset_value) if preset_value is not None else None
        return SurgePedalboardRenderer(
            plugin_path=str(plugin_path),
            preset_path=(str(preset_path) if preset_path is not None else None),
            sample_rate=self.sample_rate,
            block_size=int(surge_cfg.get("block_size", 2048)),
            channels=int(surge_cfg.get("channels", 2)),
            fadeout_seconds=self.fadeout,
            convert_to_mono=True,
            normalize_audio=False,
            note_on_delay=float(surge_cfg.get("note_on_delay", 0.01)),
            reset_between_renders=bool(surge_cfg.get("reset_between_renders", True)),
            runtime_flush_seconds=float(surge_cfg.get("runtime_flush_seconds", 1.0)),
            preset_load_flush_seconds=float(surge_cfg.get("preset_load_flush_seconds", 0.0)),
            post_param_flush_seconds=float(surge_cfg.get("post_param_flush_seconds", 0.0)),
            post_render_flush_seconds=float(surge_cfg.get("post_render_flush_seconds", 0.0)),
        )

    def _surge_params_from_full(self, full_vec: np.ndarray) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.backend_param_names, np.asarray(full_vec, dtype=np.float32))
        }

    def _render_batch(self, full_np: np.ndarray, midi_np: np.ndarray) -> torch.Tensor:
        rendered: list[np.ndarray] = []
        if self.synth == "dexed":
            for full_vec, midi_vec in zip(full_np, midi_np):
                rendered.append(self._render_one(full_vec, midi_vec))
        else:
            params_batch = [self._surge_params_from_full(full_vec) for full_vec in full_np]
            midi_batch = []
            for midi_vec in midi_np:
                duration = max(float(midi_vec[2]), 0.01)
                midi_batch.append(
                    {
                        "note": int(round(float(midi_vec[0]))),
                        "velocity": int(round(float(midi_vec[1]))),
                        "duration": duration,
                        "release": max(duration * self.release_ratio, 0.01),
                    }
                )
            audio_batch = self.renderer.render_batch(params_batch, midi_batch)
            for audio in audio_batch:
                rendered.append(self._prepare_render_output(audio))
        return torch.from_numpy(np.stack(rendered)).float()

    def _render_one(self, full_vec: np.ndarray, midi_vec: np.ndarray) -> np.ndarray:
        note = int(round(float(midi_vec[0])))
        velocity = int(round(float(midi_vec[1])))
        duration = max(float(midi_vec[2]), 0.01)
        release = max(duration * self.release_ratio, 0.01)

        if self.synth == "dexed":
            self.renderer.configure_midi(
                note=note,
                velocity=velocity,
                sustain=duration,
                release=release,
            )
            audio = self.renderer.render_single(full_vec)
            return self._prepare_render_output(audio)

        audio = self.renderer.render_single(
            params=self._surge_params_from_full(full_vec),
            midi={
                "note": note,
                "velocity": velocity,
                "duration": duration,
                "release": release,
            },
        )
        return self._prepare_render_output(audio)

    def _prepare_render_output(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        if torch.is_tensor(audio):
            arr = audio.detach().cpu().numpy()
        else:
            arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.mean(axis=0)
        elif arr.ndim != 1:
            arr = arr.reshape(-1)
        arr = np.asarray(arr, dtype=np.float32)
        target_len = self.target_num_samples
        if arr.shape[-1] < target_len:
            arr = np.pad(arr, (0, target_len - arr.shape[-1]))
        elif arr.shape[-1] > target_len:
            arr = arr[:target_len]
        return arr.astype(np.float32, copy=False)

    def _to_audio_tensor(self, audio: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = audio if torch.is_tensor(audio) else torch.from_numpy(np.asarray(audio, dtype=np.float32))
        tensor = tensor.detach().cpu().float()
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor[:, 0, :]
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(f"Expected audio tensor shape (B,T) or (B,1,T), got {tuple(tensor.shape)}")

        target_len = self.target_num_samples
        if tensor.shape[-1] < target_len:
            pad = target_len - tensor.shape[-1]
            tensor = torch.cat([tensor, torch.zeros(tensor.shape[0], pad, dtype=tensor.dtype)], dim=-1)
        elif tensor.shape[-1] > target_len:
            tensor = tensor[..., :target_len]
        return tensor

    @staticmethod
    def _to_2d_float_numpy(value: np.ndarray | torch.Tensor, label: str) -> np.ndarray:
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"{label} must have shape (D,) or (B,D), got {arr.shape}")
        return arr
