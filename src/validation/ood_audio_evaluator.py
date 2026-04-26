from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.project_paths import resolve_project_path
from src.utils.audio_metrics import trim_leading_silence_and_left_align
from src.validation.synth_render_evaluator import SynthRenderEvaluator


class OodAudioEvaluator:
    """Run OOD audio evaluation against an external WAV directory."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        dataset_metadata: Mapping[str, object],
        eval_cfg: Mapping[str, object],
        batch_size: int,
        render_evaluator: SynthRenderEvaluator,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_metadata = dict(dataset_metadata)
        self.eval_cfg = dict(eval_cfg)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"OOD evaluator batch_size must be positive, got {self.batch_size!r}")

        self.render_evaluator = render_evaluator
        self.audio_root = resolve_project_path(self.eval_cfg.get("audio_root", "dataset/nsynth/valid/audio"))
        self.max_batches = int(self.eval_cfg.get("batches", 0) or 0)
        self.seed = int(self.eval_cfg.get("seed", 1234))
        self.shuffle = bool(self.eval_cfg.get("shuffle", False))
        self.shuffle_once = bool(self.eval_cfg.get("shuffle_once", False))
        self.sample_size = int(self.eval_cfg.get("sample_size", 0) or 0)

        mel_cfg = self.dataset_metadata.get("mel")
        if not isinstance(mel_cfg, Mapping):
            raise ValueError("dataset_metadata.json must define `mel` for OOD evaluation.")
        self.n_mels = int(mel_cfg.get("n_mels", 128))
        self.window_seconds = float(mel_cfg.get("window_seconds", 0.025))
        self.frames_per_second = float(mel_cfg.get("frames_per_second", 100.0))
        self.window = str(mel_cfg.get("window", "hamming"))
        self.target_frames = int(self.dataset_metadata.get("mel_frames", 0) or 0)

        self.sample_rate = int(self.render_evaluator.sample_rate)
        self.target_duration = float(self.render_evaluator.target_duration)
        self.target_num_samples = int(round(self.sample_rate * self.target_duration))
        self.n_fft = max(int(round(self.sample_rate * self.window_seconds)), 1)
        self.hop_length = max(int(round(self.sample_rate / self.frames_per_second)), 1)

        self.use_saved_stats = bool(self.dataset_metadata.get("use_saved_mean_and_variance", False))
        self.mel_mean: Optional[torch.Tensor] = None
        self.mel_std: Optional[torch.Tensor] = None
        if self.use_saved_stats:
            stats_path = self.dataset_root / "stats.npz"
            if not stats_path.exists():
                raise FileNotFoundError(
                    "use_saved_mean_and_variance=true requires stats.npz for OOD evaluation: "
                    f"{stats_path}"
                )
            with np.load(stats_path) as stats:
                mean = stats.get("mean")
                std = stats.get("std")
                if mean is None or std is None:
                    raise KeyError(f"stats.npz must contain 'mean' and 'std': {stats_path}")
                self.mel_mean = torch.from_numpy(np.asarray(mean, dtype=np.float32))
                self.mel_std = torch.from_numpy(np.asarray(std, dtype=np.float32))

        self._cached_files: Optional[List[Path]] = None
        if self.max_batches > 0:
            if not self.audio_root.exists():
                raise FileNotFoundError(
                    "OOD evaluation is enabled but audio_root does not exist: "
                    f"{self.audio_root}"
                )
            if not self.audio_root.is_dir():
                raise NotADirectoryError(
                    "OOD evaluation audio_root must be a directory of WAV files: "
                    f"{self.audio_root}"
                )

    def run(
        self,
        *,
        predict_from_mel: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> Dict[str, float]:
        if self.max_batches <= 0:
            return {}

        files = self._get_eval_files()
        if not files:
            return {}

        metric_lists: Dict[str, List[float]] = {}
        total_batches = min(self.max_batches, (len(files) + self.batch_size - 1) // self.batch_size)
        for batch_idx in range(self.max_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            if start >= len(files):
                break
            print(f"val_nsynth: {batch_idx + 1}/{total_batches}", flush=True)
            batch_files = files[start:end]
            gt_audio = torch.cat([self._load_audio(path) for path in batch_files], dim=0)
            mel = torch.cat([self._audio_to_mel(audio) for audio in gt_audio], dim=0).to(device=device)
            full_pred, midi_pred = predict_from_mel(mel)
            try:
                pred_audio, kept = self.render_evaluator.render_from_full_and_midi_resilient(full_pred, midi_pred)
            except Exception:
                continue
            if kept.size != gt_audio.shape[0]:
                gt_audio = gt_audio[torch.as_tensor(kept, dtype=torch.long)]
            metric_pred_audio = trim_leading_silence_and_left_align(
                pred_audio,
                sample_rate=self.sample_rate,
            )
            metric_gt_audio = trim_leading_silence_and_left_align(
                gt_audio,
                sample_rate=self.sample_rate,
            )
            batch_metrics = self.render_evaluator.compute_metrics(metric_pred_audio, metric_gt_audio)
            for key, value in batch_metrics.items():
                if not np.isfinite(value):
                    continue
                log_key = f"{key}_mae" if key in {"mfcc13", "mfcc40"} else key
                metric_lists.setdefault(log_key, []).append(float(value))

        return {
            f"val_nsynth/{name}": float(np.mean(values))
            for name, values in sorted(metric_lists.items())
            if values
        }

    def _get_eval_files(self) -> List[Path]:
        if self._cached_files is not None:
            return self._cached_files

        files = sorted(self.audio_root.glob("*.wav"))
        if not files:
            raise FileNotFoundError(
                "OOD evaluation is enabled but no WAV files were found under "
                f"{self.audio_root}"
            )

        if self.shuffle_once or self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(files)
        needed = self.sample_size if self.sample_size > 0 else self.max_batches * self.batch_size
        if needed > 0:
            files = files[:needed]
        self._cached_files = files
        return self._cached_files

    def _load_audio(self, path: Path) -> torch.Tensor:
        import librosa

        audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.shape[-1] >= self.target_num_samples:
            audio = audio[: self.target_num_samples]
        else:
            audio = np.pad(audio, (0, self.target_num_samples - audio.shape[-1]))
        return torch.from_numpy(audio).unsqueeze(0)

    def _audio_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        import librosa

        audio = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
        ).astype(np.float32, copy=False)
        spec = librosa.power_to_db(spec, ref=np.max, top_db=80.0).astype(np.float32, copy=False)
        mel = torch.from_numpy(spec).unsqueeze(0)
        if self.target_frames > 0:
            frames = mel.shape[-1]
            if frames < self.target_frames:
                mel = F.pad(mel, (0, self.target_frames - frames))
            elif frames > self.target_frames:
                mel = mel[..., : self.target_frames]
        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean.to(dtype=mel.dtype)) / (self.mel_std.to(dtype=mel.dtype) + 1e-8)
        return mel.unsqueeze(0)
