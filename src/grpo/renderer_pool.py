from __future__ import annotations

import atexit
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.synth_backends.dexed.dexed_renderer import SubprocessDexedRenderer
from src.data.synth_backends.surge.surge_renderer import SurgePedalboardRenderer
from src.project_paths import resolve_project_path


def infer_synth_name(cfg: Any) -> str:
    root_value = getattr(getattr(cfg, "data", None), "root", None)
    if root_value is None:
        raise ValueError("Checkpoint config does not define data.root; cannot infer synth.")

    meta_path = resolve_project_path(root_value) / "dataset_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    synth = str(metadata.get("synth", "")).lower()
    if "surge" in synth:
        return "surge"
    if "dexed" in synth:
        return "dexed"
    raise ValueError(f"Unsupported synth metadata value '{metadata.get('synth')}' in {meta_path}")


class SubprocessDexedRendererPool:
    """Round-robin pool of isolated Dexed render workers."""

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
        num_workers: int = 1,
        max_requests_per_worker: int = 256,
        restart_retries: int = 1,
        start_method: str = "spawn",
    ) -> None:
        self._workers = [
            SubprocessDexedRenderer(
                synth_path=synth_path,
                sample_rate=sample_rate,
                block_size=block_size,
                fadeout_seconds=fadeout_seconds,
                convert_to_mono=convert_to_mono,
                normalize_audio=normalize_audio,
                note_on_delay=note_on_delay,
                max_requests_per_worker=max_requests_per_worker,
                restart_retries=restart_retries,
                start_method=start_method,
            )
            for _ in range(max(1, int(num_workers)))
        ]
        self._next_worker = 0
        atexit.register(self.close)

    def render_single(
        self,
        *,
        preset: np.ndarray,
        midi_note: int,
        midi_velocity: int,
        sustain: float,
        release: float,
    ) -> np.ndarray:
        worker = self._workers[self._next_worker % len(self._workers)]
        self._next_worker = (self._next_worker + 1) % len(self._workers)
        worker.configure_midi(
            note=int(midi_note),
            velocity=int(midi_velocity),
            sustain=float(sustain),
            release=float(release),
        )
        return np.asarray(worker.render_single(np.asarray(preset, dtype=np.float32)), dtype=np.float32)

    def render_jobs(self, jobs: Sequence[dict[str, Any]]) -> np.ndarray:
        if not jobs:
            return np.empty((0,), dtype=np.float32)

        partitions: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in self._workers]
        for idx, job in enumerate(jobs):
            partitions[idx % len(self._workers)].append((idx, job))

        results: list[np.ndarray | None] = [None] * len(jobs)

        def _run_partition(
            worker: SubprocessDexedRenderer,
            items: Sequence[tuple[int, dict[str, Any]]],
        ) -> list[tuple[int, np.ndarray]]:
            out: list[tuple[int, np.ndarray]] = []
            for idx, job in items:
                worker.configure_midi(
                    note=int(job["midi_note"]),
                    velocity=int(job["midi_velocity"]),
                    sustain=float(job["sustain"]),
                    release=float(job["release"]),
                )
                audio = worker.render_single(np.asarray(job["preset"], dtype=np.float32))
                out.append((idx, np.asarray(audio, dtype=np.float32)))
            return out

        if len(self._workers) == 1:
            for idx, audio in _run_partition(self._workers[0], partitions[0]):
                results[idx] = audio
        else:
            with cf.ThreadPoolExecutor(max_workers=len(self._workers)) as ex:
                futures = [
                    ex.submit(_run_partition, worker, items)
                    for worker, items in zip(self._workers, partitions)
                    if items
                ]
                for fut in cf.as_completed(futures):
                    for idx, audio in fut.result():
                        results[idx] = audio

        if any(audio is None for audio in results):
            raise RuntimeError("Dexed renderer pool returned incomplete results.")
        return np.stack([np.asarray(audio, dtype=np.float32) for audio in results], axis=0)

    def close(self) -> None:
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass


def _surge_worker_main(conn: Any, renderer_kwargs: dict[str, Any]) -> None:
    renderer = SurgePedalboardRenderer(**renderer_kwargs)
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

            _, params, midi = msg
            try:
                audio = renderer.render_single(params=dict(params), midi=dict(midi))
                conn.send(("ok", np.asarray(audio, dtype=np.float32)))
            except Exception as exc:  # pragma: no cover
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


class SubprocessSurgeRenderer:
    """Surge renderer isolated in a spawned subprocess."""

    def __init__(
        self,
        plugin_path: str,
        preset_path: str | None = None,
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
        *,
        max_requests_per_worker: int = 256,
        restart_retries: int = 1,
        start_method: str = "spawn",
    ) -> None:
        self._renderer_kwargs = {
            "plugin_path": plugin_path,
            "preset_path": preset_path,
            "sample_rate": sample_rate,
            "block_size": block_size,
            "channels": channels,
            "fadeout_seconds": fadeout_seconds,
            "convert_to_mono": convert_to_mono,
            "normalize_audio": normalize_audio,
            "note_on_delay": note_on_delay,
            "strict_parameter_check": strict_parameter_check,
            "reset_between_renders": reset_between_renders,
            "runtime_flush_seconds": runtime_flush_seconds,
            "preset_load_flush_seconds": preset_load_flush_seconds,
            "post_param_flush_seconds": post_param_flush_seconds,
            "post_render_flush_seconds": post_render_flush_seconds,
        }
        self.max_requests_per_worker = int(max_requests_per_worker)
        self.restart_retries = int(restart_retries)
        self._ctx = mp.get_context(start_method)
        self._conn = None
        self._proc = None
        self._requests_since_restart = 0
        self._start_worker()
        atexit.register(self.close)

    def render_single(self, *, params: Mapping[str, float], midi: Mapping[str, float]) -> np.ndarray:
        last_error: Exception | None = None
        for _ in range(max(self.restart_retries, 0) + 1):
            self._ensure_worker()
            assert self._conn is not None
            try:
                self._conn.send(("render_single", dict(params), dict(midi)))
                status, payload = self._conn.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                last_error = RuntimeError(f"Surge render worker crashed: {exc}")
                self._restart_worker()
                continue

            self._requests_since_restart += 1
            if status == "ok":
                return np.asarray(payload, dtype=np.float32)
            last_error = RuntimeError(f"Surge render worker error: {payload}")
            break

        if last_error is None:
            last_error = RuntimeError("Surge render worker failed without an explicit error.")
        raise last_error

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
            target=_surge_worker_main,
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

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class SubprocessSurgeRendererPool:
    """Round-robin pool of isolated Surge render workers."""

    def __init__(self, *, num_workers: int = 1, **renderer_kwargs: Any) -> None:
        self._workers = [
            SubprocessSurgeRenderer(**renderer_kwargs)
            for _ in range(max(1, int(num_workers)))
        ]
        self._next_worker = 0
        atexit.register(self.close)

    def render_single(self, *, params: Mapping[str, float], midi: Mapping[str, float]) -> np.ndarray:
        worker = self._workers[self._next_worker % len(self._workers)]
        self._next_worker = (self._next_worker + 1) % len(self._workers)
        return np.asarray(worker.render_single(params=params, midi=midi), dtype=np.float32)

    def render_jobs(self, jobs: Sequence[dict[str, Any]]) -> np.ndarray:
        if not jobs:
            return np.empty((0,), dtype=np.float32)

        partitions: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in self._workers]
        for idx, job in enumerate(jobs):
            partitions[idx % len(self._workers)].append((idx, job))

        results: list[np.ndarray | None] = [None] * len(jobs)

        def _run_partition(
            worker: SubprocessSurgeRenderer,
            items: Sequence[tuple[int, dict[str, Any]]],
        ) -> list[tuple[int, np.ndarray]]:
            out: list[tuple[int, np.ndarray]] = []
            for idx, job in items:
                audio = worker.render_single(params=dict(job["params"]), midi=dict(job["midi"]))
                out.append((idx, np.asarray(audio, dtype=np.float32)))
            return out

        if len(self._workers) == 1:
            for idx, audio in _run_partition(self._workers[0], partitions[0]):
                results[idx] = audio
        else:
            with cf.ThreadPoolExecutor(max_workers=len(self._workers)) as ex:
                futures = [
                    ex.submit(_run_partition, worker, items)
                    for worker, items in zip(self._workers, partitions)
                    if items
                ]
                for fut in cf.as_completed(futures):
                    for idx, audio in fut.result():
                        results[idx] = audio

        if any(audio is None for audio in results):
            raise RuntimeError("Surge renderer pool returned incomplete results.")
        return np.stack([np.asarray(audio, dtype=np.float32) for audio in results], axis=0)

    def close(self) -> None:
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass


def build_grpo_renderer_pool(
    *,
    cfg: Any,
    synth: str,
    num_workers: int,
) -> Any:
    renderer_cfg = getattr(getattr(cfg, "model", None), "renderer", None)
    if renderer_cfg is None:
        raise ValueError("Checkpoint config does not define model.renderer; cannot build GRPO renderer.")

    sample_rate = int(getattr(renderer_cfg, "sample_rate", 44_100))
    fadeout = float(getattr(renderer_cfg, "fadeout", 0.1))

    if synth == "dexed":
        dexed_cfg = dict(getattr(renderer_cfg, "dexed", {}) or {})
        synth_path = resolve_project_path(dexed_cfg.get("synth_path", "synth/Dexed.vst3"))
        return SubprocessDexedRendererPool(
            synth_path=str(synth_path),
            sample_rate=sample_rate,
            block_size=int(dexed_cfg.get("block_size", 512)),
            fadeout_seconds=fadeout,
            convert_to_mono=True,
            normalize_audio=False,
            note_on_delay=float(dexed_cfg.get("note_on_delay", 0.01)),
            num_workers=num_workers,
        )

    if synth != "surge":
        raise ValueError(f"Unsupported synth '{synth}' for GRPO renderer pool.")

    surge_cfg = dict(getattr(renderer_cfg, "surge", {}) or {})
    plugin_path = resolve_project_path(surge_cfg.get("plugin_path", "synth/Surge XT.vst3"))
    preset_value = surge_cfg.get("preset_path", "presets/surge-base.vstpreset")
    preset_path = resolve_project_path(preset_value) if preset_value is not None else None
    return SubprocessSurgeRendererPool(
        num_workers=num_workers,
        plugin_path=str(plugin_path),
        preset_path=(str(preset_path) if preset_path is not None else None),
        sample_rate=sample_rate,
        block_size=int(surge_cfg.get("block_size", 2048)),
        channels=int(surge_cfg.get("channels", 2)),
        fadeout_seconds=fadeout,
        convert_to_mono=True,
        normalize_audio=False,
        note_on_delay=float(surge_cfg.get("note_on_delay", 0.01)),
        strict_parameter_check=bool(surge_cfg.get("strict_parameter_check", True)),
        reset_between_renders=bool(surge_cfg.get("reset_between_renders", True)),
        runtime_flush_seconds=float(surge_cfg.get("runtime_flush_seconds", 1.0)),
        preset_load_flush_seconds=float(surge_cfg.get("preset_load_flush_seconds", 0.0)),
        post_param_flush_seconds=float(surge_cfg.get("post_param_flush_seconds", 0.0)),
        post_render_flush_seconds=float(surge_cfg.get("post_render_flush_seconds", 0.0)),
    )
