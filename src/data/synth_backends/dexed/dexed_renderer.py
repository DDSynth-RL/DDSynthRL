"""Dexed audio rendering via DawDreamer.

This module exposes:
- ``DexedDawRenderer``: direct in-process renderer
- ``SubprocessDexedRenderer``: spawned-worker renderer that isolates native crashes
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


class DexedDawRenderer:
    """Render full Dexed parameter vectors into waveform audio."""

    def __init__(
        self,
        synth_path: str,
        sample_rate: int = 44_100,
        block_size: int = 512,
        fadeout_seconds: float = 0.1,
        convert_to_mono: bool = True,
        normalize_audio: bool = False,
        note_on_delay: float = 0.01,
    ) -> None:
        if not os.environ.get("DISPLAY"):
            # Prevent noisy X warnings on headless compute nodes.
            os.environ["DISPLAY"] = "none"

        try:
            import dawdreamer as daw  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "DexedDawRenderer requires `dawdreamer`. Install it in the current environment."
            ) from exc

        self.synth_path = Path(synth_path)
        if not self.synth_path.exists():
            raise FileNotFoundError(f"Dexed plugin not found: {self.synth_path}")

        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.convert_to_mono = bool(convert_to_mono)
        self.normalize_audio = bool(normalize_audio)
        self.note_on_delay = float(note_on_delay)

        self.engine = daw.RenderEngine(self.sample_rate, self.block_size)
        self.synth = self.engine.make_plugin_processor(self.synth_path.stem, str(self.synth_path))
        self.engine.load_graph([(self.synth, [])])

        fadeout_len = int(self.sample_rate * float(fadeout_seconds))
        self.fadeout_len = max(fadeout_len, 0)
        self.fadeout_window = (
            np.linspace(1.0, 0.0, self.fadeout_len, dtype=np.float32)
            if self.fadeout_len > 0
            else None
        )

        self.midi_note: Optional[int] = None
        self.midi_velocity: Optional[int] = None
        self.sustain_seconds: Optional[float] = None
        self.release_seconds: Optional[float] = None

    def configure_midi(self, note: int, velocity: int, sustain: float, release: float) -> None:
        self.midi_note = int(note)
        self.midi_velocity = int(velocity)
        self.sustain_seconds = float(sustain)
        self.release_seconds = float(release)

    def render_single(self, preset: np.ndarray) -> np.ndarray:
        """Render one full Dexed preset. Returns audio as `(C, T)` float32."""
        return self.render_batch(np.asarray(preset, dtype=np.float32)[np.newaxis, :])[0]

    def render_batch(self, presets: np.ndarray) -> np.ndarray:
        """Render a batch of full Dexed presets (values in [0, 1])."""
        presets = np.asarray(presets, dtype=np.float32)
        if presets.ndim == 1:
            presets = presets[None, :]

        if (
            self.midi_note is None
            or self.midi_velocity is None
            or self.sustain_seconds is None
            or self.release_seconds is None
        ):
            raise RuntimeError(
                "DexedRenderer MIDI is not configured. Call configure_midi(...) before rendering."
            )

        total_duration = self.sustain_seconds + self.release_seconds
        audios = []
        for preset in presets:
            self._set_parameters(preset)
            audios.append(self._render_once(total_duration))

        return np.stack(audios, axis=0)

    def _set_parameters(self, values: Sequence[float]) -> None:
        for idx, value in enumerate(values):
            self.synth.set_parameter(idx, float(value))

    def _render_once(self, duration: float) -> np.ndarray:
        # Flush stale MIDI/audio states before rendering the requested note.
        self.synth.add_midi_note(60, 0, 0.0, max(self.note_on_delay, 0.001))
        self.engine.render(max(self.note_on_delay, 0.001))
        self.synth.clear_midi()

        self.synth.add_midi_note(
            int(self.midi_note),
            int(self.midi_velocity),
            max(self.note_on_delay, 0.001),
            max(self.sustain_seconds, 0.001),
        )
        self.engine.render(max(float(duration), 0.001))
        self.synth.clear_midi()

        audio = np.asarray(self.engine.get_audio(), dtype=np.float32)
        if self.convert_to_mono:
            audio = np.mean(audio, axis=0, keepdims=True)

        if self.normalize_audio:
            peak = float(np.max(np.abs(audio)))
            if peak > 0.0:
                audio = audio / peak

        if self.fadeout_window is not None and audio.shape[-1] >= self.fadeout_len:
            audio[..., -self.fadeout_len :] *= self.fadeout_window

        return audio

    def close(self) -> None:
        # DawDreamer objects rely on Python GC; this method makes release explicit.
        try:
            self.synth.clear_midi()
        except Exception:
            pass
        self.synth = None
        self.engine = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


def _dexed_worker_main(conn: Any, renderer_kwargs: dict[str, Any]) -> None:
    """Spawned worker process that owns the native DawDreamer/Dexed state."""
    renderer = DexedDawRenderer(**renderer_kwargs)
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            cmd = msg[0]
            if cmd == "close":
                break
            if cmd != "render_single":
                conn.send(("error", f"Unknown worker command: {cmd}"))
                continue

            _, preset, midi_note, midi_velocity, sustain, release = msg
            try:
                renderer.configure_midi(
                    note=int(midi_note),
                    velocity=int(midi_velocity),
                    sustain=float(sustain),
                    release=float(release),
                )
                audio = renderer.render_single(np.asarray(preset, dtype=np.float32))
                conn.send(("ok", np.asarray(audio, dtype=np.float32)))
            except Exception as exc:  # pragma: no cover - surfaced to parent
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


class SubprocessDexedRenderer:
    """Dexed renderer isolated in a spawned subprocess to contain plugin crashes."""

    def __init__(
        self,
        synth_path: str,
        sample_rate: int = 44_100,
        block_size: int = 512,
        fadeout_seconds: float = 0.1,
        convert_to_mono: bool = True,
        normalize_audio: bool = False,
        note_on_delay: float = 0.01,
        *,
        max_requests_per_worker: int = 256,
        restart_retries: int = 1,
        start_method: str = "spawn",
    ) -> None:
        self._renderer_kwargs = {
            "synth_path": synth_path,
            "sample_rate": sample_rate,
            "block_size": block_size,
            "fadeout_seconds": fadeout_seconds,
            "convert_to_mono": convert_to_mono,
            "normalize_audio": normalize_audio,
            "note_on_delay": note_on_delay,
        }
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.max_requests_per_worker = int(max_requests_per_worker)
        self.restart_retries = int(restart_retries)
        self._ctx = mp.get_context(start_method)
        self._conn = None
        self._proc = None
        self._requests_since_restart = 0

        self.midi_note = 60
        self.midi_velocity = 100
        self.sustain_seconds = 3.0
        self.release_seconds = 1.0

        self._start_worker()
        atexit.register(self.close)

    def configure_midi(self, note: int, velocity: int, sustain: float, release: float) -> None:
        self.midi_note = int(note)
        self.midi_velocity = int(velocity)
        self.sustain_seconds = float(sustain)
        self.release_seconds = float(release)

    def render_single(self, preset: np.ndarray) -> np.ndarray:
        return self.render_batch(np.asarray(preset, dtype=np.float32)[np.newaxis, :])[0]

    def render_batch(self, presets: np.ndarray) -> np.ndarray:
        presets = np.asarray(presets, dtype=np.float32)
        if presets.ndim == 1:
            presets = presets[None, :]
        audios = [self._render_single_once(preset) for preset in presets]
        return np.stack(audios, axis=0)

    def close(self) -> None:
        conn = self._conn
        proc = self._proc
        self._conn = None
        self._proc = None
        self._requests_since_restart = 0

        if conn is not None:
            try:
                conn.send(("close",))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        if proc is not None and proc.is_alive():
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)

    def _start_worker(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_dexed_worker_main,
            args=(child_conn, self._renderer_kwargs),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        self._conn = parent_conn
        self._proc = proc
        self._requests_since_restart = 0

    def _restart_worker(self) -> None:
        self.close()
        self._start_worker()

    def _ensure_worker(self) -> None:
        if self._proc is None or self._conn is None or not self._proc.is_alive():
            self._restart_worker()
        elif self.max_requests_per_worker > 0 and self._requests_since_restart >= self.max_requests_per_worker:
            self._restart_worker()

    def _render_single_once(self, preset: np.ndarray) -> np.ndarray:
        last_error: Exception | None = None
        for _ in range(max(self.restart_retries, 0) + 1):
            self._ensure_worker()
            assert self._conn is not None

            try:
                self._conn.send(
                    (
                        "render_single",
                        np.asarray(preset, dtype=np.float32),
                        int(self.midi_note),
                        int(self.midi_velocity),
                        float(self.sustain_seconds),
                        float(self.release_seconds),
                    )
                )
                status, payload = self._conn.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                last_error = RuntimeError(f"Dexed render worker crashed: {exc}")
                self._restart_worker()
                continue

            self._requests_since_restart += 1
            if status == "ok":
                return np.asarray(payload, dtype=np.float32)

            last_error = RuntimeError(f"Dexed render worker error: {payload}")
            break

        if last_error is None:
            last_error = RuntimeError("Dexed render worker failed without an explicit error.")
        raise last_error

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
