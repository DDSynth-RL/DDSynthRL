"""Surge XT audio rendering via Pedalboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from src.project_paths import find_project_root, resolve_project_path

_PROJECT_ROOT = find_project_root(Path(__file__).resolve())
_DEFAULT_SURGE_PRESET_REL_PATH = Path("presets/surge-base.vstpreset")


def _resolve_path(path: Union[str, Path]) -> Path:
    return resolve_project_path(path)


class SurgePedalboardRenderer:
    """Render Surge XT parameter dictionaries into waveform audio."""

    def __init__(
        self,
        plugin_path: Union[str, Path],
        preset_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 44_100,
        block_size: int = 2048,
        channels: int = 2,
        fadeout_seconds: float = 0.1,
        convert_to_mono: bool = True,
        normalize_audio: bool = False,
        note_on_delay: float = 0.01,
        strict_parameter_check: bool = True,
        reset_between_renders: bool = True,
        runtime_flush_seconds: float = 1.0,
        preset_load_flush_seconds: float = 0.0,
        post_param_flush_seconds: float = 0.0,
        post_render_flush_seconds: float = 0.0,
    ) -> None:
        if not os.environ.get("DISPLAY"):
            os.environ["DISPLAY"] = "none"

        try:
            from pedalboard import VST3Plugin  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SurgePedalboardRenderer requires `pedalboard`. Install it in the current environment."
            ) from exc

        self.plugin_path = _resolve_path(plugin_path)
        if not self.plugin_path.exists():
            raise FileNotFoundError(f"Surge XT plugin not found: {self.plugin_path}")

        if preset_path is None:
            default_preset = _PROJECT_ROOT / _DEFAULT_SURGE_PRESET_REL_PATH
            self.preset_path = default_preset if default_preset.exists() else None
        else:
            resolved = _resolve_path(preset_path)
            self.preset_path = resolved if resolved.exists() else None

        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.channels = int(channels)
        self.convert_to_mono = bool(convert_to_mono)
        self.normalize_audio = bool(normalize_audio)
        self.note_on_delay = float(note_on_delay)
        self.strict_parameter_check = bool(strict_parameter_check)
        self.reset_between_renders = bool(reset_between_renders)
        self.runtime_flush_seconds = float(runtime_flush_seconds)
        self.preset_load_flush_seconds = float(preset_load_flush_seconds)
        self.post_param_flush_seconds = float(post_param_flush_seconds)
        self.post_render_flush_seconds = float(post_render_flush_seconds)

        self.plugin = VST3Plugin(str(self.plugin_path))
        if not bool(getattr(self.plugin, "is_instrument", False)):
            raise RuntimeError(
                "Loaded Surge plugin is not recognized as an instrument by Pedalboard. "
                "Please pin `pedalboard==0.9.9` in this environment."
            )

        if self.preset_path is not None:
            self.plugin.load_preset(str(self.preset_path))

        self._prime_runtime_parameter_space()
        if self.preset_path is not None:
            self.plugin.load_preset(str(self.preset_path))
        self.plugin.reset()

        self._refresh_parameter_cache()

        fadeout_len = int(self.sample_rate * float(fadeout_seconds))
        self.fadeout_len = max(fadeout_len, 0)
        self.fadeout_window = (
            np.linspace(1.0, 0.0, self.fadeout_len, dtype=np.float32)
            if self.fadeout_len > 0
            else None
        )

    def _refresh_parameter_cache(self) -> None:
        self.param_names = set(self.plugin.parameters.keys())
        self.param_count = len(self.param_names)

    def _prime_runtime_parameter_space(self) -> None:
        # Pedalboard+Surge starts with a compact parameter namespace and expands
        # to the legacy/full one after the first process() call.
        duration = float(max(self.runtime_flush_seconds, 0.05))
        self.plugin.process([], duration, self.sample_rate, self.channels, self.block_size, True)
        self.plugin.reset()

    def set_params_by_name(self, params: Mapping[str, float], strict: Optional[bool] = None) -> None:
        if strict is None:
            strict = self.strict_parameter_check

        missing = []
        for name, value in params.items():
            if name not in self.plugin.parameters:
                missing.append(name)
                continue
            self.plugin.parameters[name].raw_value = float(np.clip(value, 0.0, 1.0))

        if strict and missing:
            sample = ", ".join(missing[:10])
            raise KeyError(
                f"Surge renderer missing {len(missing)} parameters from config (examples: {sample})"
            )

    def render_single(
        self,
        params: Mapping[str, float],
        midi: Mapping[str, float],
    ) -> np.ndarray:
        return self.render_batch([params], midi=[midi])[0]

    def render_batch(
        self,
        params_batch: Sequence[Mapping[str, float]],
        midi: Union[np.ndarray, Sequence[Mapping[str, float]]],
    ) -> np.ndarray:
        if len(params_batch) == 0:
            return np.zeros((0, 1, 0), dtype=np.float32)

        midi_batch = self._normalize_midi_batch(midi, len(params_batch))

        outputs = []
        for i, params in enumerate(params_batch):
            if self.reset_between_renders:
                self._reset_state()

            self.set_params_by_name(params)
            if self.post_param_flush_seconds > 0.0:
                self._flush_empty(self.post_param_flush_seconds)

            note = int(midi_batch[i, 0])
            velocity = int(midi_batch[i, 1])
            sustain = float(max(midi_batch[i, 2], 0.01))
            release = float(max(midi_batch[i, 3], 0.01))

            start = max(self.note_on_delay, 0.001)
            end = start + sustain
            total_duration = max(end + release, start + 0.05)
            midi_events = self._make_midi_events(note, velocity, start, end)

            audio = self.plugin.process(
                midi_events,
                total_duration,
                self.sample_rate,
                self.channels,
                self.block_size,
                True,
            )
            if self.post_render_flush_seconds > 0.0:
                self._flush_empty(self.post_render_flush_seconds)
            outputs.append(self._postprocess_audio(audio))

        return np.stack(outputs, axis=0)

    def _normalize_midi_batch(
        self,
        midi: Union[np.ndarray, Sequence[Mapping[str, float]]],
        batch_size: int,
    ) -> np.ndarray:
        # Layout per row: [note, velocity, sustain, release]
        if midi is None:
            raise ValueError("MIDI must be provided explicitly; renderer no longer supplies defaults.")

        if isinstance(midi, np.ndarray):
            arr = np.asarray(midi, dtype=np.float32)
            if arr.ndim == 1:
                if arr.shape[0] != 4:
                    raise ValueError(f"MIDI vector must have shape (4,), got {arr.shape}")
                return np.tile(arr[None, :], (batch_size, 1))

            if arr.ndim == 2:
                if arr.shape[0] != batch_size:
                    raise ValueError(
                        f"MIDI batch mismatch: expected {batch_size}, got {arr.shape[0]}"
                    )
                if arr.shape[1] == 4:
                    return arr.astype(np.float32)
                raise ValueError(f"MIDI batch width must be 4, got {arr.shape[1]}")

            raise ValueError(f"Unsupported MIDI array shape: {arr.shape}")

        if len(midi) != batch_size:
            raise ValueError(f"MIDI sequence mismatch: expected {batch_size}, got {len(midi)}")

        out = np.zeros((batch_size, 4), dtype=np.float32)
        for i, item in enumerate(midi):
            required = ("note", "velocity", "duration", "release")
            missing = [key for key in required if key not in item]
            if missing:
                raise ValueError(
                    f"MIDI mapping must define keys {required}; missing {missing}"
                )
            note = float(item["note"])
            velocity = float(item["velocity"])
            duration = float(item["duration"])
            release = float(item["release"])
            out[i] = np.asarray([note, velocity, duration, release], dtype=np.float32)

        return out

    def _reset_state(self) -> None:
        if self.preset_path is not None:
            self.plugin.load_preset(str(self.preset_path))
        if len(self.plugin.parameters) != self.param_count:
            self._prime_runtime_parameter_space()
            if self.preset_path is not None:
                self.plugin.load_preset(str(self.preset_path))
            self._refresh_parameter_cache()
        if self.preset_load_flush_seconds > 0.0:
            self._flush_empty(self.preset_load_flush_seconds)
        else:
            self.plugin.reset()

    def _flush_empty(self, duration_seconds: float) -> None:
        self.plugin.process(
            [],
            float(duration_seconds),
            self.sample_rate,
            self.channels,
            self.block_size,
            True,
        )
        self.plugin.reset()

    @staticmethod
    def _make_midi_events(note: int, velocity: int, note_on: float, note_off: float):
        note = int(np.clip(note, 0, 127))
        velocity = int(np.clip(velocity, 0, 127))
        on = bytes([0x90, note, velocity])
        off = bytes([0x80, note, 0])
        return ((on, float(note_on)), (off, float(note_off)))

    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if self.convert_to_mono and arr.ndim == 2:
            arr = np.mean(arr, axis=0, keepdims=True)

        if self.normalize_audio:
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            if peak > 0.0:
                arr = arr / peak

        if self.fadeout_window is not None and arr.shape[-1] >= self.fadeout_len:
            arr[..., -self.fadeout_len :] *= self.fadeout_window

        return arr.astype(np.float32, copy=False)

    def close(self) -> None:
        self.plugin = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
