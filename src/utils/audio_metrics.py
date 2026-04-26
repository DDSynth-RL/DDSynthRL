"""Audio reconstruction metrics shared across synth training modules."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def trim_leading_silence_and_left_align(
    audio: torch.Tensor | np.ndarray,
    *,
    sample_rate: int,
    threshold_dbfs: float = -60.0,
    frame_seconds: float = 0.05,
    hop_seconds: float = 0.01,
) -> torch.Tensor:
    """Trim leading silence per example, left-align, and preserve original length.

    This is intended for metric-time alignment only. Each waveform is shifted so
    that the first frame whose RMS exceeds `threshold_dbfs` becomes the new
    start; the tail is zero-padded to keep the original length unchanged.
    """

    tensor = audio if torch.is_tensor(audio) else torch.from_numpy(np.asarray(audio, dtype=np.float32))
    tensor = tensor.detach().cpu().float()

    squeeze = False
    has_channel = False
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
        has_channel = True
    elif tensor.ndim != 2:
        raise ValueError(
            "trim_leading_silence_and_left_align expects shape (T,), (B,T), or (B,1,T), "
            f"got {tuple(tensor.shape)}"
        )

    batch, total_samples = tensor.shape
    if total_samples == 0:
        return audio if torch.is_tensor(audio) else tensor

    frame_length = max(int(round(frame_seconds * float(sample_rate))), 1)
    hop_length = max(int(round(hop_seconds * float(sample_rate))), 1)
    if total_samples < frame_length:
        pad = frame_length - total_samples
        work = torch.nn.functional.pad(tensor, (0, pad))
    else:
        work = tensor

    frames = work.unfold(dimension=-1, size=frame_length, step=hop_length)
    rms = torch.sqrt(torch.clamp(frames.pow(2).mean(dim=-1), min=1e-12))
    rms_db = 20.0 * torch.log10(rms)
    active = rms_db > float(threshold_dbfs)

    shifted = torch.zeros_like(tensor)
    for idx in range(batch):
        active_idx = torch.nonzero(active[idx], as_tuple=False)
        if active_idx.numel() == 0:
            shifted[idx] = tensor[idx]
            continue
        start = int(active_idx[0, 0].item()) * hop_length
        if start <= 0:
            shifted[idx] = tensor[idx]
            continue
        if start >= total_samples:
            continue
        shifted[idx, : total_samples - start] = tensor[idx, start:]

    if has_channel:
        shifted = shifted.unsqueeze(1)
    if squeeze:
        shifted = shifted[0]
    return shifted


class WmfccMetric:
    """DTW-based wMFCC distance used in Synth-Matching/synth-permutations."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_mfcc_wmfcc: int = 20,
        n_mels_wmfcc: int = 128,
        win_sec: float = 0.05,
        hop_sec: float = 0.01,
    ) -> None:
        import librosa

        self.librosa = librosa
        self.sample_rate = int(sample_rate)
        self.n_mfcc_wmfcc = int(n_mfcc_wmfcc)
        self.n_mels_wmfcc = int(n_mels_wmfcc)
        self.win_length = int(win_sec * sample_rate)
        self.hop_length = int(hop_sec * sample_rate)

    def _mfcc_wmfcc(self, waveform: np.ndarray) -> np.ndarray:
        return self.librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc_wmfcc,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels_wmfcc,
        )

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_np = pred.detach().cpu().numpy()
        targ_np = target.detach().cpu().numpy()
        wmfcc_results: List[float] = []

        for tgt, pre in zip(targ_np, pred_np):
            tgt = np.asarray(tgt, dtype=np.float32)
            pre = np.asarray(pre, dtype=np.float32)

            mfcc_tgt = self._mfcc_wmfcc(tgt).reshape(-1, self.n_mfcc_wmfcc)
            mfcc_pre = self._mfcc_wmfcc(pre).reshape(-1, self.n_mfcc_wmfcc)

            def l1(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.mean(np.abs(a - b)))

            dist = self.librosa.sequence.dtw(
                X=mfcc_tgt.T,
                Y=mfcc_pre.T,
                metric=l1,
                backtrack=False,
            )
            cost = dist[-1, -1]
            max_len = max(mfcc_tgt.shape[0], mfcc_pre.shape[0], 1)
            wmfcc_results.append(float(cost / max_len))

        if not wmfcc_results:
            return torch.tensor(0.0, dtype=torch.float32)
        return torch.tensor(float(np.mean(wmfcc_results)), dtype=torch.float32)


class LibrosaMFCCTransform:
    """Librosa MFCC transform (win=50ms, hop=10ms)."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_mfcc: int = 13,
        win_sec: float = 0.05,
        hop_sec: float = 0.01,
    ) -> None:
        import librosa

        self.librosa = librosa
        self.sample_rate = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.win_length = int(win_sec * sample_rate)
        self.hop_length = int(hop_sec * sample_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        wav_np = waveform.detach().cpu().numpy()
        if wav_np.ndim == 1:
            wav_np = wav_np[None, :]
        mfcc_list = []
        for wav in wav_np:
            mfcc = self.librosa.feature.mfcc(
                y=np.asarray(wav, dtype=np.float32),
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.win_length,
                hop_length=self.hop_length,
            )
            mfcc_list.append(torch.from_numpy(mfcc))
        return torch.stack(mfcc_list).float()


class ClapCosineDistance:
    """Optional CLAP cosine distance metric."""

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        try:
            from fadtk.model_loader import CLAPModel
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CLAP metric requires `fadtk` and its dependencies. "
                "Install them explicitly before enabling validation.metrics.clap."
            ) from exc

        self.model = CLAPModel("2023")
        if device is not None:
            self.model.device = torch.device(device)
        self.model.load_model()
        self.sample_rate = int(self.model.sr)

    def _prepare_audio(self, audio: np.ndarray | torch.Tensor, *, sample_rate: int) -> np.ndarray:
        if torch.is_tensor(audio):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)

        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if int(sample_rate) != int(self.sample_rate):
            import librosa

            audio_np = librosa.resample(audio_np, orig_sr=int(sample_rate), target_sr=int(self.sample_rate))
        return np.clip(audio_np, -1.0, 1.0)

    def _embed_one(self, audio: np.ndarray | torch.Tensor, *, sample_rate: int) -> torch.Tensor:
        audio_np = self._prepare_audio(audio, sample_rate=sample_rate)
        emb = self.model.get_embedding(audio_np.reshape(1, -1))
        emb = np.asarray(emb, dtype=np.float32)
        vec = emb if emb.ndim == 1 else emb[0]
        return torch.from_numpy(vec).float()

    def __call__(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        *,
        sample_rate: int,
    ) -> float:
        pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
        target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)

        if pred_np.ndim == 1:
            pred_np = pred_np[None, :]
        if target_np.ndim == 1:
            target_np = target_np[None, :]
        if pred_np.shape[0] != target_np.shape[0]:
            raise ValueError(f"Batch mismatch: pred {pred_np.shape}, target {target_np.shape}")

        distances: List[torch.Tensor] = []
        for pre, tgt in zip(pred_np, target_np):
            pre_emb = self._embed_one(pre, sample_rate=sample_rate)
            tgt_emb = self._embed_one(tgt, sample_rate=sample_rate)
            dist = 1.0 - torch.nn.functional.cosine_similarity(pre_emb, tgt_emb, dim=0)
            distances.append(dist)

        if not distances:
            return float("nan")
        return float(torch.stack(distances).mean().item())


class CrepeEmbeddingDistance:
    """Compute similarity/distance between CREPE pitch embeddings."""

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        model: str = "tiny",
        metric: str = "cosine",
        return_similarity: bool | None = None,
        hop_length: int | None = None,
        batch_size: int | None = None,
        pad: bool = True,
    ) -> None:
        try:
            import torchcrepe.core as torchcrepe_core
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CREPE embedding metric requires `torchcrepe`. Install it in the current environment."
            ) from exc

        self.torchcrepe = torchcrepe_core
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = str(model)
        self.metric = str(metric).lower()
        if return_similarity is None:
            return_similarity = self.metric == "cosine"
        self.return_similarity = bool(return_similarity)
        self.hop_length = int(hop_length) if hop_length is not None else None
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.pad = bool(pad)

        if self.metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported CREPE metric '{metric}'. Use 'cosine' or 'l2'.")

    def _embed(self, audio: np.ndarray | torch.Tensor, *, sample_rate: int) -> torch.Tensor:
        if torch.is_tensor(audio):
            audio_t = audio
        else:
            audio_t = torch.from_numpy(np.asarray(audio, dtype=np.float32))

        if audio_t.ndim == 1:
            audio_t = audio_t.unsqueeze(0)
        elif audio_t.ndim > 2:
            audio_t = audio_t.reshape(audio_t.shape[0], -1)

        audio_t = audio_t.to(self.device, dtype=torch.float32)

        emb = self.torchcrepe.embed(
            audio_t,
            sample_rate=int(sample_rate),
            hop_length=self.hop_length,
            model=self.model,
            batch_size=self.batch_size,
            device=str(self.device),
            pad=self.pad,
        )
        emb = emb.mean(dim=1).mean(dim=1)
        return emb

    def __call__(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        *,
        sample_rate: int,
    ) -> float:
        pred_emb = self._embed(pred, sample_rate=sample_rate)
        targ_emb = self._embed(target, sample_rate=sample_rate)
        if pred_emb.shape != targ_emb.shape:
            raise ValueError(f"Embedding shape mismatch: {tuple(pred_emb.shape)} vs {tuple(targ_emb.shape)}")

        if self.metric == "cosine":
            sim = torch.nn.functional.cosine_similarity(pred_emb, targ_emb, dim=-1)
            val = sim if self.return_similarity else (1.0 - sim)
        else:
            dist = torch.norm(pred_emb - targ_emb, p=2, dim=-1)
            val = -dist if self.return_similarity else dist

        return float(val.mean().item())


_MSS_MEL_PARAMS: Sequence[Tuple[int, int, int]] = (
    (10, 5, 32),
    (25, 10, 64),
    (100, 50, 128),
)


def _to_mono_1d(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return audio.mean(axis=0)
        return audio.mean(axis=1)
    return audio.reshape(-1).astype(np.float32, copy=False)


def _compute_mel_specs(y: np.ndarray, sample_rate: float) -> List[np.ndarray]:
    import librosa

    y_mono = _to_mono_1d(y)
    mel_specs: List[np.ndarray] = []
    for window_ms, hop_ms, n_mels in _MSS_MEL_PARAMS:
        win_length = int(window_ms * sample_rate / 1000.0)
        hop_length = int(hop_ms * sample_rate / 1000.0)
        spec = librosa.feature.melspectrogram(
            y=y_mono,
            sr=int(sample_rate),
            n_mels=int(n_mels),
            n_fft=int(win_length),
            hop_length=int(hop_length),
            window="hann",
        )
        mel_specs.append(librosa.power_to_db(spec, ref=np.max))
    return mel_specs


def compute_mss(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    target_specs = _compute_mel_specs(target, sample_rate)
    pred_specs = _compute_mel_specs(pred, sample_rate)
    dist = 0.0
    for target_spec, pred_spec in zip(target_specs, pred_specs):
        dist += float(np.mean(np.abs(target_spec - pred_spec)))
    return float(dist / max(len(target_specs), 1))


def _get_stft_mag(y: np.ndarray, sample_rate: float) -> np.ndarray:
    import librosa

    y_mono = _to_mono_1d(y)
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.02 * sample_rate)
    stft = librosa.stft(
        y_mono,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    ).T
    return np.abs(stft)


def _batched_wasserstein_distance_np(hist1: np.ndarray, hist2: np.ndarray) -> np.ndarray:
    bin_width = 1.0 / float(hist1.shape[-1])
    cdf1 = np.cumsum(hist1, axis=-1)
    cdf2 = np.cumsum(hist2, axis=-1)
    return np.sum(np.abs(cdf1 - cdf2), axis=-1) * bin_width


def compute_sot(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    target_stft = _get_stft_mag(target, sample_rate)
    pred_stft = _get_stft_mag(pred, sample_rate)
    target_stft = target_stft / np.clip(target_stft.sum(axis=-1, keepdims=True), 1e-6, None)
    pred_stft = pred_stft / np.clip(pred_stft.sum(axis=-1, keepdims=True), 1e-6, None)
    dists = _batched_wasserstein_distance_np(target_stft, pred_stft)
    return float(dists.mean())


def compute_rms_env_cosine(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    import librosa

    target_mono = _to_mono_1d(target)
    pred_mono = _to_mono_1d(pred)

    win_length = int(0.05 * sample_rate)
    hop_length = int(0.025 * sample_rate)

    target_rms = librosa.feature.rms(y=target_mono, frame_length=win_length, hop_length=hop_length)
    pred_rms = librosa.feature.rms(y=pred_mono, frame_length=win_length, hop_length=hop_length)

    target_norm = float(np.linalg.norm(target_rms, ord=2, axis=-1).reshape(-1)[0])
    pred_norm = float(np.linalg.norm(pred_rms, ord=2, axis=-1).reshape(-1)[0])
    denom = target_norm * pred_norm
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1e-12
    cosine_sim = float(np.dot(target_rms[0], pred_rms[0]) / denom)
    return float(cosine_sim)


def compute_extra_audio_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    *,
    mss: bool = True,
    sot: bool = True,
    rms: bool = True,
) -> Dict[str, float]:
    pred_np = pred.detach().cpu().numpy()
    targ_np = target.detach().cpu().numpy()

    if pred_np.ndim == 3 and pred_np.shape[1] == 1:
        pred_np = pred_np[:, 0, :]
    if targ_np.ndim == 3 and targ_np.shape[1] == 1:
        targ_np = targ_np[:, 0, :]
    if pred_np.ndim == 1:
        pred_np = pred_np[None, :]
    if targ_np.ndim == 1:
        targ_np = targ_np[None, :]

    if pred_np.shape[0] != targ_np.shape[0]:
        raise ValueError(f"Batch mismatch: pred {pred_np.shape}, target {targ_np.shape}")

    out: Dict[str, float] = {}
    mss_list: List[float] = []
    sot_list: List[float] = []
    rms_list: List[float] = []
    for pre, tgt in zip(pred_np, targ_np):
        if mss:
            mss_list.append(compute_mss(tgt, pre, float(sample_rate)))
        if sot:
            sot_list.append(compute_sot(tgt, pre, float(sample_rate)))
        if rms:
            rms_list.append(compute_rms_env_cosine(tgt, pre, float(sample_rate)))

    if mss_list:
        out["mss"] = float(np.mean(mss_list))
    if sot_list:
        out["sot"] = float(np.mean(sot_list))
    if rms_list:
        out["rms"] = float(np.mean(rms_list))
    return out
