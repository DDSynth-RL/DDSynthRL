from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.data.h5_synth_dataset import H5SynthDataset
from src.project_paths import project_relative_string, resolve_project_path
from src.utils.audio_metrics import CrepeEmbeddingDistance
from src.validation.synth_render_evaluator import SynthRenderEvaluator

log = logging.getLogger(__name__)


def _load_checkpoint_safe(path: Path, *, trust_ckpt: bool) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
    except pickle.UnpicklingError:
        pass

    try:
        from omegaconf import DictConfig as OmegaDictConfig
        from omegaconf import ListConfig
        import torch.serialization as ts

        if hasattr(ts, "safe_globals"):
            with ts.safe_globals([OmegaDictConfig, ListConfig]):
                return torch.load(str(path), map_location="cpu", weights_only=True)
        if hasattr(ts, "add_safe_globals"):
            ts.add_safe_globals([OmegaDictConfig, ListConfig])
            return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        pass

    if trust_ckpt:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    raise pickle.UnpicklingError(
        "Checkpoint requires unsafe loading. Re-run with `--trust-ckpt` only if you trust the file."
    )


def _find_hydra_config(ckpt_path: Path) -> Optional[Path]:
    for parent in [ckpt_path.parent, *ckpt_path.parents]:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def _infer_synth_hint(cfg: DictConfig) -> str:
    root_value = str(getattr(getattr(cfg, "data", None), "root", "")).lower()
    if "surge" in root_value:
        return "surge"
    return "dexed"


def _sanitize_cfg(cfg: DictConfig, *, output_dir: Path) -> DictConfig:
    synth = _infer_synth_hint(cfg)

    if "paths" not in cfg:
        cfg.paths = OmegaConf.create({})
    cfg.paths.root_dir = "."
    cfg.paths.output_dir = project_relative_string(output_dir)

    if "data" in cfg:
        cfg.data.root = f"dataset/{synth}"
        cfg.data.token_order_path = f"configs/order/{synth}_autoregressive_order.yaml"

    if "model" in cfg:
        if "validation" in cfg.model:
            cfg.model.validation.render_batches = 0
            if "nsynth_eval" in cfg.model.validation:
                cfg.model.validation.nsynth_eval.enable = False

        if "renderer" not in cfg.model:
            cfg.model.renderer = OmegaConf.create({})
        if "dexed" not in cfg.model.renderer:
            cfg.model.renderer.dexed = OmegaConf.create({})
        if "surge" not in cfg.model.renderer:
            cfg.model.renderer.surge = OmegaConf.create({})
        cfg.model.renderer.dexed.synth_path = "synth/Dexed.vst3"
        cfg.model.renderer.surge.plugin_path = "synth/Surge XT.vst3"
        cfg.model.renderer.surge.preset_path = "presets/surge-base.vstpreset"

    return cfg


def _load_cfg_from_ckpt_or_hydra(ckpt: Dict[str, Any], ckpt_path: Path, *, output_dir: Path) -> DictConfig:
    for key in ("base_cfg", "hyper_parameters"):
        maybe_cfg = ckpt.get(key)
        if isinstance(maybe_cfg, dict) and "data" in maybe_cfg and "model" in maybe_cfg:
            return _sanitize_cfg(OmegaConf.create(maybe_cfg), output_dir=output_dir)

    hydra_cfg = _find_hydra_config(ckpt_path)
    if hydra_cfg is None:
        raise FileNotFoundError(
            "Could not recover base Hydra config from checkpoint or neighboring .hydra/config.yaml."
        )
    return _sanitize_cfg(OmegaConf.load(str(hydra_cfg)), output_dir=output_dir)


def _instantiate_model(cfg: DictConfig, ckpt_dir: Path) -> Tuple[str, torch.nn.Module]:
    target = str(getattr(cfg.model, "_target_", ""))
    if target.endswith("DiscreteDiffusionModule"):
        from src.models.discrete_diffusion_module import DiscreteDiffusionModule

        return "dd", DiscreteDiffusionModule(cfg, ckpt_dir=ckpt_dir)
    if target.endswith("AutoregressiveModule"):
        from src.models.autoregressive_module import AutoregressiveModule

        return "ar", AutoregressiveModule(cfg, ckpt_dir=ckpt_dir)
    if target.endswith("FlowMatchingModule"):
        from src.models.flow_matching_module import FlowMatchingModule

        return "fm", FlowMatchingModule(cfg, ckpt_dir=ckpt_dir)
    raise ValueError(f"Unsupported model target for standalone test: {target!r}")


def _load_state_dict_into_model(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint does not contain `state_dict`.")
    model.load_state_dict(state_dict, strict=False)


def _load_stats_and_frames(
    dataset_root: Path,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], Optional[int]]:
    mean_t: Optional[torch.Tensor] = None
    std_t: Optional[torch.Tensor] = None
    n_mels: Optional[int] = None
    frames: Optional[int] = None

    stats_path = dataset_root / "stats.npz"
    if stats_path.exists():
        with np.load(stats_path) as stats:
            mean = stats.get("mean")
            std = stats.get("std")
        if mean is not None and std is not None:
            mean_np = np.asarray(mean, dtype=np.float32)
            std_np = np.asarray(std, dtype=np.float32)
            if mean_np.ndim == 3:
                mean_np = mean_np[0]
            if std_np.ndim == 3:
                std_np = std_np[0]
            if mean_np.ndim == 2:
                n_mels = int(mean_np.shape[0])
                frames = int(mean_np.shape[-1])
            mean_t = torch.from_numpy(mean_np).unsqueeze(0) if mean_np.ndim == 2 else torch.from_numpy(mean_np)
            std_t = torch.from_numpy(std_np).unsqueeze(0) if std_np.ndim == 2 else torch.from_numpy(std_np)

    meta_path = dataset_root / "dataset_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("mel_frames") is not None:
            frames = int(meta["mel_frames"])
    return mean_t, std_t, n_mels, frames


def _load_audio_resampled(path: Path, *, target_sr: int, target_len: int) -> np.ndarray:
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    if int(sr) != int(target_sr):
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr))
    if audio.shape[-1] < target_len:
        audio = np.pad(audio, (0, target_len - audio.shape[-1]))
    else:
        audio = audio[:target_len]
    return np.asarray(audio, dtype=np.float32)


def _audio_to_mel(
    audio: np.ndarray,
    *,
    sample_rate: int,
    spec_cfg: Mapping[str, Any],
    target_frames: Optional[int],
    target_n_mels: Optional[int],
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    n_fft = int(spec_cfg.get("n_fft", 1102))
    hop_length = int(spec_cfg.get("hop_length", 441))
    n_mels = int(target_n_mels or spec_cfg.get("n_mels", 128))
    use_db = bool(spec_cfg.get("use_db", True))
    top_db = float(spec_cfg.get("top_db", 80.0))

    spec = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=int(sample_rate),
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    ).astype(np.float32, copy=False)
    if use_db:
        spec = librosa.power_to_db(spec, ref=np.max, top_db=top_db).astype(np.float32, copy=False)
    else:
        spec = np.log1p(spec).astype(np.float32, copy=False)

    spec_t = torch.from_numpy(spec).unsqueeze(0)
    if target_frames is not None:
        frames = int(spec_t.shape[-1])
        if frames < int(target_frames):
            spec_t = F.pad(spec_t, (0, int(target_frames) - frames))
        elif frames > int(target_frames):
            spec_t = spec_t[..., : int(target_frames)]

    if mean is not None and std is not None:
        mean_t = mean.to(dtype=spec_t.dtype)
        std_t = std.to(dtype=spec_t.dtype)
        if mean_t.ndim == 2:
            mean_t = mean_t.unsqueeze(0)
        if std_t.ndim == 2:
            std_t = std_t.unsqueeze(0)
        spec_t = (spec_t - mean_t) / (std_t + 1e-8)

    return spec_t


def _prepare_render_output(audio: np.ndarray | torch.Tensor, *, target_len: int) -> np.ndarray:
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
    if arr.shape[-1] < target_len:
        arr = np.pad(arr, (0, target_len - arr.shape[-1]))
    elif arr.shape[-1] > target_len:
        arr = arr[:target_len]
    return arr.astype(np.float32, copy=False)


def _list_wavs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_") or "sample"


def _is_better(metric: str, candidate: float, best: float) -> bool:
    name = str(metric).lower()
    if name in {"rms", "reward"}:
        return candidate > best
    return candidate < best


def _validate_sampling_args(*, epsilon: float, top_k: int, min_prob: float, best_of: int) -> None:
    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError(f"--epsilon must be in [0,1], got {epsilon}")
    if int(top_k) < 0:
        raise ValueError(f"--top-k must be >=0, got {top_k}")
    if not (0.0 <= float(min_prob) <= 1.0):
        raise ValueError(f"--min-prob must be in [0,1], got {min_prob}")
    if int(best_of) < 1:
        raise ValueError(f"--best-of must be >=1, got {best_of}")


def _validate_dd_pos_sampling_args(*, epsilon: float, top_k: int, min_prob: float) -> None:
    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError(f"--dd-pos-epsilon must be in [0,1], got {epsilon}")
    if int(top_k) < 0:
        raise ValueError(f"--dd-pos-top-k must be >=0, got {top_k}")
    if not (0.0 <= float(min_prob) <= 1.0):
        raise ValueError(f"--dd-pos-min-prob must be in [0,1], got {min_prob}")


def _fmt_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?:??"
    seconds_i = int(seconds)
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


class _Progress:
    def __init__(self, *, label: str, total: int, every: int, enabled: bool) -> None:
        self.label = str(label)
        self.total = int(total)
        self.every = max(int(every), 1)
        self.enabled = bool(enabled)
        self.start = time.perf_counter()

    def update(self, done: int, *, extra: str = "") -> None:
        if not self.enabled:
            return
        done = int(done)
        if self.total > 0:
            done = max(0, min(done, self.total))
        should_print = (done == 1) or (done % self.every == 0) or (self.total > 0 and done == self.total)
        if not should_print:
            return
        elapsed = time.perf_counter() - self.start
        rate = (done / elapsed) if elapsed > 0 else float("nan")
        eta = ((self.total - done) / rate) if self.total > 0 and np.isfinite(rate) and rate > 0 else float("nan")
        pct = (100.0 * done / self.total) if self.total > 0 else float("nan")
        suffix = f" | {extra}" if extra else ""
        print(
            f"[{self.label}] {done}/{self.total} ({pct:.1f}%) "
            f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)} rate={rate:.2f}/s{suffix}",
            flush=True,
        )


@dataclass(frozen=True)
class SamplingConfig:
    token_epsilon: float
    token_top_k: int
    token_min_prob: float
    pos_epsilon: float
    pos_top_k: int
    pos_min_prob: float
    dd_midi_first: bool
    dd_normalized_entropy: bool
    dd_normalized_confidence: bool
    force_greedy_midi_only: bool
    force_greedy_special: bool


_FORCE_GREEDY_MIDI_NAMES: Tuple[str, ...] = ("MIDI_NOTE", "MIDI_VELOCITY", "MIDI_DURATION")
_FORCE_GREEDY_NAMES: Tuple[str, ...] = ("MIDI_NOTE", "MIDI_VELOCITY", "MIDI_DURATION", "ALGORITHM", "FEEDBACK")
_DD_MIDI_FIRST_ORDER: Tuple[str, ...] = ("MIDI_NOTE", "MIDI_DURATION", "MIDI_VELOCITY")


def _is_special_force_greedy_token(name: str) -> bool:
    n = str(name).upper()
    if n in _FORCE_GREEDY_NAMES:
        return True
    if "FEEDBACK" in n or "LFO" in n:
        return True
    return False


def _should_force_greedy_token(name: str, *, force_greedy_special: bool, force_greedy_midi_only: bool) -> bool:
    if bool(force_greedy_midi_only):
        return str(name).upper() in _FORCE_GREEDY_MIDI_NAMES
    if bool(force_greedy_special):
        return _is_special_force_greedy_token(name)
    return False


def _sample_from_logits_1d(
    logits_1d: torch.Tensor,
    *,
    epsilon: float,
    top_k: int,
    min_prob: float,
    force_greedy: bool,
    generator: torch.Generator,
) -> int:
    if logits_1d.ndim != 1:
        raise ValueError(f"Expected 1D logits, got {tuple(logits_1d.shape)}")
    if force_greedy or float(epsilon) <= 0.0:
        return int(torch.argmax(logits_1d, dim=-1).item())

    if float(epsilon) >= 1.0:
        choose_sample = True
    else:
        choose_sample = bool(torch.rand((), generator=generator, device=logits_1d.device).item() < float(epsilon))
    if not choose_sample:
        return int(torch.argmax(logits_1d, dim=-1).item())

    probs_full = F.softmax(logits_1d, dim=-1)
    allowed = torch.arange(logits_1d.numel(), device=logits_1d.device, dtype=torch.long)

    if int(top_k) > 0 and int(allowed.numel()) > int(top_k):
        top_vals, top_idx = torch.topk(probs_full, k=int(top_k), dim=-1)
        allowed = allowed[top_idx]
        allowed_probs = top_vals
    else:
        allowed_probs = probs_full

    denom = allowed_probs.sum()
    if not torch.isfinite(denom) or float(denom.item()) <= 0.0:
        return int(torch.argmax(logits_1d, dim=-1).item())
    probs = allowed_probs / denom

    if float(min_prob) > 0.0:
        keep = probs >= float(min_prob)
        if not bool(keep.any().item()):
            return int(torch.argmax(logits_1d, dim=-1).item())
        allowed = allowed[keep]
        probs = probs[keep]
        denom2 = probs.sum()
        if not torch.isfinite(denom2) or float(denom2.item()) <= 0.0:
            return int(torch.argmax(logits_1d, dim=-1).item())
        probs = probs / denom2

    if int(allowed.numel()) == 1:
        return int(allowed[0].item())
    pick = torch.multinomial(probs, 1, generator=generator).item()
    return int(allowed[int(pick)].item())


def _pos_distribution_from_conf(
    conf_1d: torch.Tensor,
    locked_1d: torch.Tensor,
    *,
    top_k: int,
    min_prob: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    weights = conf_1d.to(dtype=torch.float32).clamp(min=0.0)
    weights = weights.masked_fill(locked_1d, 0.0)
    total = weights.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return None

    allowed = torch.nonzero(~locked_1d, as_tuple=False).squeeze(-1)
    if allowed.numel() == 0:
        return None
    if int(top_k) > 0 and int(allowed.numel()) > int(top_k):
        allowed_weights = weights[allowed]
        top_vals, top_idx = torch.topk(allowed_weights, k=int(top_k), dim=-1)
        allowed = allowed[top_idx]
        weights = top_vals
    else:
        weights = weights[allowed]

    probs = weights / torch.clamp(weights.sum(), min=1e-12)
    pmin = float(min_prob)
    if pmin > 0.0:
        keep = probs >= pmin
        if not bool(keep.any().item()):
            return None
        allowed = allowed[keep]
        probs = probs[keep]
        probs = probs / torch.clamp(probs.sum(), min=1e-12)
    return allowed, probs.to(dtype=conf_1d.dtype)


def _compute_confidence(
    probs: torch.Tensor,
    *,
    cardinals: torch.Tensor,
    use_normalized_entropy: bool,
    use_normalized_confidence: bool,
) -> torch.Tensor:
    if use_normalized_entropy:
        p = probs.to(dtype=torch.float32)
        entropy = -(p * torch.log(torch.clamp(p, min=1e-12))).sum(dim=-1)
        log_card = torch.log(torch.clamp(cardinals, min=1.0)).to(dtype=entropy.dtype).unsqueeze(0)
        safe = log_card > 0
        uncertainty = torch.zeros_like(entropy)
        uncertainty = torch.where(safe, entropy / torch.clamp(log_card, min=1e-12), uncertainty)
        return (1.0 - uncertainty).clamp(0.0, 1.0).to(dtype=probs.dtype)

    p_max = probs.max(dim=-1).values
    if use_normalized_confidence:
        baseline = 1.0 / torch.clamp(cardinals, min=1.0)
        denom = 1.0 - baseline
        p_max = p_max.to(dtype=baseline.dtype)
        conf = (p_max - baseline) / denom
        conf = torch.where(denom > 0, conf, torch.zeros_like(conf))
        return conf.to(dtype=probs.dtype)
    return p_max


def _dd_select_next_pos(
    *,
    dd_order: Sequence[str],
    locked: torch.Tensor,
    conf: torch.Tensor,
    dd_midi_first: bool,
    epsilon: float,
    top_k: int,
    min_prob: float,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = locked.shape[0]
    name_to_pos: Dict[str, int] = {}
    for i, name in enumerate(dd_order):
        up = str(name).upper()
        if up not in name_to_pos:
            name_to_pos[up] = int(i)

    chosen: List[int] = []
    for bi in range(batch):
        picked: Optional[int] = None
        if bool(dd_midi_first):
            for midi_name in _DD_MIDI_FIRST_ORDER:
                pos = name_to_pos.get(str(midi_name).upper())
                if pos is not None and not bool(locked[bi, pos].item()):
                    picked = int(pos)
                    break
        if picked is not None:
            chosen.append(picked)
            continue

        conf_row = conf[bi]
        locked_row = locked[bi]
        greedy = int(torch.argmax(conf_row.masked_fill(locked_row, -1e9)).item())
        if float(epsilon) <= 0.0:
            chosen.append(greedy)
            continue
        if float(epsilon) >= 1.0:
            choose_sample = True
        else:
            choose_sample = bool(torch.rand((), generator=generator, device=conf_row.device).item() < float(epsilon))
        if not choose_sample:
            chosen.append(greedy)
            continue
        dist = _pos_distribution_from_conf(conf_row, locked_row, top_k=int(top_k), min_prob=float(min_prob))
        if dist is None:
            chosen.append(greedy)
            continue
        allowed, probs = dist
        if int(allowed.numel()) == 1:
            chosen.append(int(allowed[0].item()))
            continue
        pick = int(torch.multinomial(probs, 1, generator=generator).item())
        chosen.append(int(allowed[pick].item()))
    return torch.tensor(chosen, dtype=torch.long, device=locked.device)


@torch.no_grad()
def _ar_decode_with_logits_epsilon(
    ar_model: torch.nn.Module,
    spectrogram: torch.Tensor,
    *,
    sampling: SamplingConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(getattr(ar_model, "model").parameters()).device
    spec = spectrogram.to(device=device)
    batch = spec.shape[0]
    seq_len = int(getattr(ar_model, "model").seq_len)
    max_card = int(getattr(ar_model, "model").max_card)

    token_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    out_logits = torch.zeros((batch, seq_len, max_card), dtype=torch.float32, device=device)
    out_logits = out_logits.to(dtype=next(getattr(ar_model, "model").parameters()).dtype)

    for t in range(seq_len):
        logits = getattr(ar_model, "model")(spec, token_ids)
        out_logits[:, t, :] = logits[:, t, :]
        cardinal = int(getattr(ar_model, "model").cardinals[t])
        if cardinal <= 1:
            token_ids[:, t] = 0
            continue

        step_logits = logits[:, t, :cardinal]
        token_name = str(getattr(ar_model, "model").order[t])
        force_greedy = _should_force_greedy_token(
            token_name,
            force_greedy_special=bool(sampling.force_greedy_special),
            force_greedy_midi_only=bool(sampling.force_greedy_midi_only),
        )
        chosen: List[int] = []
        for b in range(batch):
            chosen.append(
                _sample_from_logits_1d(
                    step_logits[b],
                    epsilon=float(sampling.token_epsilon),
                    top_k=min(int(sampling.token_top_k), cardinal) if int(sampling.token_top_k) > 0 else 0,
                    min_prob=float(sampling.token_min_prob),
                    force_greedy=force_greedy,
                    generator=generator,
                )
            )
        token_ids[:, t] = torch.tensor(chosen, dtype=torch.long, device=device)

    return token_ids, out_logits


@torch.no_grad()
def _dd_decode_with_logits_epsilon(
    dd_model: torch.nn.Module,
    spectrogram: torch.Tensor,
    *,
    sampling: SamplingConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if spectrogram.ndim != 4:
        raise ValueError(f"Expected spectrogram shape (B,1,F,T), got {tuple(spectrogram.shape)}")

    device = next(getattr(dd_model, "model").parameters()).device
    spec = spectrogram.to(device=device)
    batch = spec.shape[0]

    length = int(getattr(dd_model, "model").seq_len)
    horizon = int(getattr(dd_model, "diffusion_T"))
    if length != horizon:
        raise RuntimeError(f"Expected L==T for sampling, got L={length}, T={horizon}")

    x = torch.full((batch, length), int(getattr(dd_model, "mask_token_id")), dtype=torch.long, device=device)
    locked = torch.zeros((batch, length), dtype=torch.bool, device=device)
    out_logits = torch.zeros((batch, length, int(getattr(dd_model, "model").max_card)), dtype=torch.float32, device=device)
    out_logits = out_logits.to(dtype=next(getattr(dd_model, "model").parameters()).dtype)

    cardinals = torch.as_tensor(getattr(dd_model, "model").cardinals, device=device, dtype=torch.float32)
    getattr(dd_model, "model").eval()
    for step in range(length):
        t = torch.full((batch,), horizon - step, dtype=torch.long, device=device)
        logits = getattr(dd_model, "model")(spec, x, t)
        probs = F.softmax(logits, dim=-1)
        conf = _compute_confidence(
            probs,
            cardinals=cardinals,
            use_normalized_entropy=bool(sampling.dd_normalized_entropy),
            use_normalized_confidence=bool(sampling.dd_normalized_confidence)
            and not bool(sampling.dd_normalized_entropy),
        )
        j = _dd_select_next_pos(
            dd_order=list(getattr(dd_model, "model").order),
            locked=locked,
            conf=conf,
            dd_midi_first=bool(sampling.dd_midi_first),
            epsilon=float(sampling.pos_epsilon),
            top_k=int(sampling.pos_top_k),
            min_prob=float(sampling.pos_min_prob),
            generator=generator,
        )

        batch_idx = torch.arange(batch, device=device)
        out_logits[batch_idx, j, :] = logits[batch_idx, j, :]

        chosen_ids: List[int] = []
        for bi in range(batch):
            pos = int(j[bi].item())
            token_name = str(getattr(dd_model, "model").order[pos])
            force_greedy = _should_force_greedy_token(
                token_name,
                force_greedy_special=bool(sampling.force_greedy_special),
                force_greedy_midi_only=bool(sampling.force_greedy_midi_only),
            )
            cardinal = int(getattr(dd_model, "model").cardinals[pos])
            if cardinal <= 1:
                chosen_ids.append(0)
                continue
            step_logits = logits[bi, pos, :cardinal]
            chosen_ids.append(
                _sample_from_logits_1d(
                    step_logits,
                    epsilon=float(sampling.token_epsilon),
                    top_k=min(int(sampling.token_top_k), cardinal) if int(sampling.token_top_k) > 0 else 0,
                    min_prob=float(sampling.token_min_prob),
                    force_greedy=force_greedy,
                    generator=generator,
                )
            )
        token_id = torch.tensor(chosen_ids, dtype=torch.long, device=device)
        x[batch_idx, j] = token_id
        locked[batch_idx, j] = True

    return x, out_logits


def _reward_value_from_metrics(metrics: Mapping[str, float], reward_cfg: Mapping[str, Any]) -> float:
    weights = reward_cfg.get("weights", {})
    reward_sign = float(reward_cfg.get("sign", -1.0))

    def _weighted_value(name: str, *, negate: bool = False) -> float:
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            return 0.0
        value = float(metrics.get(name, float("nan")))
        if not math.isfinite(value):
            return 0.0
        return (-weight if negate else weight) * value

    reward_base = (
        _weighted_value("wmfcc")
        + _weighted_value("clap")
        + _weighted_value("crepe")
        + _weighted_value("mss")
        + _weighted_value("sot")
        + _weighted_value("rms", negate=True)
    )
    return float(reward_sign) * reward_base


def _selection_value_for_metric(metric: str, metrics: Mapping[str, float], failure_reward: float) -> float:
    name = str(metric).lower()
    if name == "reward":
        value = float(metrics.get("reward", failure_reward))
        return value if math.isfinite(value) else float(failure_reward)
    value = float(metrics.get(name, float("nan")))
    if math.isfinite(value):
        return value
    return float("-inf") if name == "rms" else float("inf")


def _to_single_batch(sample: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.unsqueeze(0) if value.ndim >= 1 else value.view(1)
        batch[key] = tensor.to(device=device)
    return batch


@torch.no_grad()
def _flow_sample_with_generator(
    model: torch.nn.Module,
    mel_spec: torch.Tensor,
    *,
    steps: int,
    cfg_strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    device = next(getattr(model, "vector_field").parameters()).device
    conditioning = mel_spec.to(device=device, dtype=torch.float32)
    noise = torch.randn(
        conditioning.shape[0],
        int(getattr(model, "denoise_space").total_dim),
        device=device,
        generator=generator,
    )
    return getattr(model, "_sample")(conditioning, noise, steps=int(steps), cfg_strength=float(cfg_strength))


def _predict_candidate(
    *,
    algo: str,
    model: torch.nn.Module,
    mel_spec: torch.Tensor,
    sampling: SamplingConfig,
    flow_steps: Optional[int],
    flow_cfg_strength: Optional[float],
    generator: torch.Generator,
    force_greedy: bool,
) -> Dict[str, Any]:
    if algo == "ar":
        if force_greedy:
            token_ids, logits = getattr(model, "_greedy_decode_with_logits")(mel_spec)
        else:
            token_ids, logits = _ar_decode_with_logits_epsilon(model, mel_spec, sampling=sampling, generator=generator)
        full, midi = getattr(model, "_tokens_to_full_and_midi")(token_ids)
        return {"full": full, "midi": midi, "token_ids": token_ids, "logits": logits, "kind": "greedy" if force_greedy else "sample"}

    if algo == "dd":
        if force_greedy:
            token_ids, logits = getattr(model, "_diffusion_decode")(mel_spec, return_logits=True)
        else:
            token_ids, logits = _dd_decode_with_logits_epsilon(model, mel_spec, sampling=sampling, generator=generator)
        full, midi = getattr(model, "_tokens_to_full_and_midi")(token_ids)
        return {"full": full, "midi": midi, "token_ids": token_ids, "logits": logits, "kind": "greedy" if force_greedy else "sample"}

    if algo == "fm":
        steps = int(flow_steps if flow_steps is not None else getattr(model, "test_sample_steps"))
        cfg_strength = float(
            flow_cfg_strength if flow_cfg_strength is not None else getattr(model, "test_cfg_strength")
        )
        pred_params = _flow_sample_with_generator(
            model,
            mel_spec,
            steps=steps,
            cfg_strength=cfg_strength,
            generator=generator,
        )
        full, midi = getattr(model, "_denoised_to_full_and_midi_absolute")(pred_params.to(dtype=torch.float32))
        return {
            "full": full,
            "midi": midi,
            "pred_params": pred_params,
            "kind": "sample",
            "flow_steps": steps,
            "flow_cfg_strength": cfg_strength,
        }

    raise ValueError(f"Unsupported algo {algo!r}")


def _compute_in_domain_losses(
    *,
    algo: str,
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    candidate: Mapping[str, Any],
) -> Dict[str, float]:
    if algo == "ar":
        targets = getattr(model, "build_targets")(batch)
        loss, param_loss, midi_loss = getattr(model, "model").loss(
            candidate["logits"],
            targets,
            temperature=float(getattr(model, "loss_cfg").get("temperature", 1.0)),
            label_smoothing=float(getattr(model, "loss_cfg").get("label_smoothing", 0.0)),
            midi_label_smoothing=float(getattr(model, "loss_cfg").get("midi_label_smoothing", 0.0)),
            midi_label_smoothing_apply=getattr(model, "loss_cfg").get("midi_label_smoothing_apply"),
            token_weights=getattr(model, "token_loss_weights", None),
            uncertainty_weighting=bool(getattr(model, "loss_cfg").get("uncertainty_weighting", False)),
            return_components=True,
        )
        return {
            "loss": float(loss.detach().cpu().item()),
            "param_loss": float(param_loss.detach().cpu().item()),
            "midi_loss": float(midi_loss.detach().cpu().item()),
            "param_mse": float("nan"),
        }

    if algo == "dd":
        targets = getattr(model, "build_targets")(batch)
        mask = torch.ones_like(targets, dtype=torch.bool, device=targets.device)
        loss, param_loss, midi_loss = getattr(model, "model").loss(
            candidate["logits"],
            targets,
            mask,
            temperature=float(getattr(model, "loss_cfg").get("temperature", 1.0)),
            label_smoothing=float(getattr(model, "loss_cfg").get("label_smoothing", 0.0)),
            midi_label_smoothing=float(getattr(model, "loss_cfg").get("midi_label_smoothing", 0.0)),
            midi_label_smoothing_apply=getattr(model, "loss_cfg").get("midi_label_smoothing_apply"),
            token_weights=getattr(model, "token_loss_weights", None),
            uncertainty_weighting=bool(getattr(model, "loss_cfg").get("uncertainty_weighting", False)),
            return_components=True,
        )
        return {
            "loss": float(loss.detach().cpu().item()),
            "param_loss": float(param_loss.detach().cpu().item()),
            "midi_loss": float(midi_loss.detach().cpu().item()),
            "param_mse": float("nan"),
        }

    if algo == "fm":
        gt_params = getattr(model, "_encode_targets")(batch)
        pred_params = candidate["pred_params"].to(device=gt_params.device, dtype=gt_params.dtype)
        diff = pred_params - gt_params
        param_idx = torch.as_tensor(getattr(model, "param_indices"), device=diff.device, dtype=torch.long)
        midi_idx = torch.as_tensor(getattr(model, "midi_indices"), device=diff.device, dtype=torch.long)
        param_loss = diff[..., param_idx].square().mean() if param_idx.numel() else diff.new_zeros(())
        midi_loss = diff[..., midi_idx].square().mean() if midi_idx.numel() else diff.new_zeros(())
        param_mse = diff.square().mean()
        return {
            "loss": float((param_loss + midi_loss).detach().cpu().item()),
            "param_loss": float(param_loss.detach().cpu().item()),
            "midi_loss": float(midi_loss.detach().cpu().item()),
            "param_mse": float(param_mse.detach().cpu().item()),
        }

    raise ValueError(f"Unsupported algo {algo!r}")


def _audio_metrics_for_pair(
    *,
    render_evaluator: SynthRenderEvaluator,
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    crepe_metric: Optional[CrepeEmbeddingDistance],
    reward_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    metrics = render_evaluator.compute_metrics(
        torch.from_numpy(pred_audio).unsqueeze(0),
        torch.from_numpy(target_audio).unsqueeze(0),
    )
    for key in ("wmfcc", "mfcc13", "mfcc40", "clap", "mss", "sot", "rms"):
        metrics.setdefault(key, float("nan"))
    if crepe_metric is not None:
        metrics["crepe"] = float(crepe_metric(pred_audio, target_audio, sample_rate=sample_rate))
    else:
        metrics["crepe"] = float("nan")
    metrics["reward"] = _reward_value_from_metrics(metrics, reward_cfg)
    return {key: float(value) for key, value in metrics.items()}


def _maybe_write_audio(
    *,
    root: Path,
    split: str,
    sample_id: str,
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
) -> Tuple[str, str]:
    pred_path = root / "pred" / split / f"{sample_id}.wav"
    gt_path = root / "gt" / split / f"{sample_id}.wav"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(pred_path), pred_audio, sample_rate)
    sf.write(str(gt_path), target_audio, sample_rate)
    return project_relative_string(pred_path), project_relative_string(gt_path)


def _derive_run_name(ckpt_path: Path) -> str:
    stem = ckpt_path.stem
    parts: List[str] = []
    if ckpt_path.parent.name == "checkpoints":
        if len(ckpt_path.parents) >= 3:
            parts.extend([ckpt_path.parents[2].name, ckpt_path.parents[1].name, stem])
        else:
            parts.append(stem)
    else:
        parts.append(stem)
    return _slugify("_".join(parts))


def _path_for_report(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    try:
        return project_relative_string(path)
    except Exception:
        return str(path)


def _load_dataset_metadata(dataset_root: Path) -> Dict[str, Any]:
    meta_path = dataset_root / "dataset_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _split_from_h5_path(path: Path) -> Tuple[Path, str]:
    if path.suffix.lower() != ".h5":
        raise ValueError(f"Expected .h5 path, got {path}")
    return path.parent, path.stem


def _make_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _load_reward_cfg(project_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = OmegaConf.load(str(project_root / "configs" / "finetune" / "dd_grpo.yaml"))
    reward = OmegaConf.to_container(cfg.reward, resolve=True)
    if not isinstance(reward, dict):
        raise TypeError("configs/finetune/dd_grpo.yaml reward section must resolve to a mapping.")

    weights = dict(reward.get("weights", {}) or {})
    for key in ("wmfcc", "clap", "crepe", "mss", "sot", "rms"):
        override = getattr(args, f"reward_{key}")
        if override is not None:
            weights[key] = float(override)
    reward["weights"] = weights
    if args.reward_sign is not None:
        reward["sign"] = float(args.reward_sign)
    if args.reward_failure is not None:
        reward["failure"] = float(args.reward_failure)
    return reward


def _save_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone checkpoint evaluation for DDSynth-RL.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a DDSynth-RL checkpoint (.pt or .ckpt).")
    parser.add_argument(
        "--trust-ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow unsafe checkpoint loading fallback when needed.",
    )
    parser.add_argument(
        "--in-domain-h5",
        type=str,
        default=None,
        help="Optional explicit in-domain H5 path. Defaults to dataset/<synth>/test.h5 inferred from the checkpoint.",
    )
    parser.add_argument(
        "--ood-audio-root",
        type=str,
        default=None,
        help="Optional explicit OOD audio directory. Defaults to dataset/nsynth/test/audio.",
    )
    parser.add_argument("--max-in-domain", type=int, default=1, help="Max number of in-domain samples (0=all).")
    parser.add_argument("--max-ood", type=int, default=200, help="Max number of OOD samples (0=all).")
    parser.add_argument(
        "--no-shuffle-in-domain",
        action="store_true",
        help="Disable in-domain sample shuffling.",
    )
    parser.add_argument(
        "--no-shuffle-ood",
        action="store_true",
        help="Disable OOD audio shuffling.",
    )
    parser.add_argument("--epsilon", type=float, default=0.2, help="Token epsilon-greedy sampling rate.")
    parser.add_argument("--top-k", type=int, default=3, help="Token top-k sampling (0=disable).")
    parser.add_argument("--min-prob", type=float, default=0.1, help="Token minimum probability threshold.")
    parser.add_argument("--dd-pos-epsilon", type=float, default=0.2, help="DD position epsilon-greedy rate.")
    parser.add_argument("--dd-pos-top-k", type=int, default=3, help="DD position top-k sampling (0=disable).")
    parser.add_argument("--dd-pos-min-prob", type=float, default=0.1, help="DD position minimum probability.")
    parser.add_argument("--dd-midi-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dd-normalized-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dd-normalized-confidence", action=argparse.BooleanOptionalAction, default=False)
    force_group = parser.add_mutually_exclusive_group()
    force_group.add_argument("--force-greedy-special", action="store_true", help="Force greedy on special tokens.")
    force_group.add_argument("--force-greedy-midi-only", action="store_true", help="Force greedy only on MIDI tokens.")
    parser.add_argument("--best-of", type=int, default=10, help="Number of trials per sample.")
    parser.add_argument(
        "--include-greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a greedy DD/AR candidate when best-of uses stochastic sampling.",
    )
    parser.add_argument("--best-metric", type=str, default="reward", help="Selection metric for best-of.")
    parser.add_argument("--seed", type=int, default=1234, help="Shuffle and sampling seed.")
    parser.add_argument("--device", type=str, default=None, help='Torch device, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument(
        "--save-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save best prediction/target waveforms under render/<run_name>/pred and gt.",
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default="render",
        help="Base directory for metrics.csv, run_info.json, and optional audio output.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress printing.")
    parser.add_argument("--progress-every", type=int, default=10, help="Progress print cadence.")
    parser.add_argument("--flow-steps", type=int, default=None, help="Override FM sampling steps.")
    parser.add_argument("--flow-cfg-strength", type=float, default=None, help="Override FM CFG strength.")
    parser.add_argument("--reward-wmfcc", type=float, default=None)
    parser.add_argument("--reward-clap", type=float, default=None)
    parser.add_argument("--reward-crepe", type=float, default=None)
    parser.add_argument("--reward-mss", type=float, default=None)
    parser.add_argument("--reward-sot", type=float, default=None)
    parser.add_argument("--reward-rms", type=float, default=None)
    parser.add_argument("--reward-sign", type=float, default=None)
    parser.add_argument("--reward-failure", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    _validate_sampling_args(
        epsilon=float(args.epsilon),
        top_k=int(args.top_k),
        min_prob=float(args.min_prob),
        best_of=int(args.best_of),
    )
    _validate_dd_pos_sampling_args(
        epsilon=float(args.dd_pos_epsilon),
        top_k=int(args.dd_pos_top_k),
        min_prob=float(args.dd_pos_min_prob),
    )

    project_root = resolve_project_path(".")
    ckpt_path = resolve_project_path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    render_base = resolve_project_path(args.render_dir)
    run_name = _derive_run_name(ckpt_path)
    render_root = render_base / run_name
    render_root.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    ckpt = _load_checkpoint_safe(ckpt_path, trust_ckpt=bool(args.trust_ckpt))
    cfg = _load_cfg_from_ckpt_or_hydra(ckpt, ckpt_path, output_dir=render_root)
    algo, model = _instantiate_model(cfg, ckpt_dir=render_root / "checkpoints")
    _load_state_dict_into_model(model, ckpt)
    device = _make_device(args.device)
    model = model.to(device)
    model.eval()

    dataset_root = resolve_project_path(cfg.data.root)
    dataset_metadata = _load_dataset_metadata(dataset_root)
    synth = str(getattr(model, "synth", dataset_metadata.get("synth", ""))).lower()
    if synth == "surge_xt":
        synth = "surge"
    sample_rate = int(dataset_metadata["sample_rate"])
    target_duration = float(dataset_metadata["target_duration"])
    target_len = int(round(sample_rate * target_duration))
    mean_t, std_t, target_n_mels, target_frames = _load_stats_and_frames(dataset_root)

    reward_cfg = _load_reward_cfg(project_root, args)
    reward_weights = reward_cfg.get("weights", {}) if isinstance(reward_cfg, dict) else {}
    best_metric_name = str(args.best_metric).lower()
    metrics_cfg = {
        "wmfcc": True,
        "mfcc13": True,
        "mfcc40": True,
        "mss": float(reward_weights.get("mss", 0.0)) != 0.0 or best_metric_name == "mss",
        "sot": float(reward_weights.get("sot", 0.0)) != 0.0 or best_metric_name == "sot",
        "rms": float(reward_weights.get("rms", 0.0)) != 0.0 or best_metric_name == "rms",
        "clap": float(reward_weights.get("clap", 0.0)) != 0.0 or best_metric_name in {"clap", "reward"},
    }
    render_evaluator = SynthRenderEvaluator(
        synth=synth,
        helper=getattr(model, "helper"),
        renderer_cfg=OmegaConf.to_container(getattr(model, "renderer_cfg"), resolve=True),
        metrics_cfg=metrics_cfg,
        dataset_sample_rate=sample_rate,
        dataset_target_duration=target_duration,
        device=device,
    )
    crepe_metric = None
    try:
        reward_crepe_cfg = reward_cfg.get("crepe", {}) if isinstance(reward_cfg, dict) else {}
        need_crepe = float(reward_weights.get("crepe", 0.0)) != 0.0 or best_metric_name in {"crepe", "reward"}
        if need_crepe and bool(reward_crepe_cfg.get("enable", True)):
            crepe_metric = CrepeEmbeddingDistance(
                device=device,
                model=str(reward_crepe_cfg.get("model", "tiny")),
                metric=str(reward_crepe_cfg.get("metric", "cosine")),
                return_similarity=reward_crepe_cfg.get("return_similarity"),
                hop_length=reward_crepe_cfg.get("hop_length"),
                batch_size=reward_crepe_cfg.get("batch_size"),
                pad=bool(reward_crepe_cfg.get("pad", True)),
            )
    except Exception as exc:
        log.warning("CREPE metric unavailable: %s", exc)

    sampling = SamplingConfig(
        token_epsilon=float(args.epsilon),
        token_top_k=int(args.top_k),
        token_min_prob=float(args.min_prob),
        pos_epsilon=float(args.dd_pos_epsilon),
        pos_top_k=int(args.dd_pos_top_k),
        pos_min_prob=float(args.dd_pos_min_prob),
        dd_midi_first=bool(args.dd_midi_first),
        dd_normalized_entropy=bool(args.dd_normalized_entropy),
        dd_normalized_confidence=bool(args.dd_normalized_confidence),
        force_greedy_midi_only=bool(args.force_greedy_midi_only),
        force_greedy_special=bool(args.force_greedy_special),
    )

    in_domain_h5 = resolve_project_path(args.in_domain_h5) if args.in_domain_h5 else dataset_root / "test.h5"
    in_domain_root, in_domain_split = _split_from_h5_path(in_domain_h5)
    ood_root = resolve_project_path(args.ood_audio_root) if args.ood_audio_root else resolve_project_path(
        "dataset/nsynth/test/audio"
    )

    dataset = H5SynthDataset(in_domain_root, split=in_domain_split, read_audio=True)
    in_indices = list(range(len(dataset)))
    if not bool(args.no_shuffle_in_domain):
        random.Random(int(args.seed)).shuffle(in_indices)
    if int(args.max_in_domain) > 0:
        in_indices = in_indices[: int(args.max_in_domain)]

    ood_files = _list_wavs(ood_root)
    if not bool(args.no_shuffle_ood):
        random.Random(int(args.seed)).shuffle(ood_files)
    if int(args.max_ood) > 0:
        ood_files = ood_files[: int(args.max_ood)]

    rows: List[Dict[str, Any]] = []
    failure_reward = float(reward_cfg.get("failure", -1000.0))
    stochastic_enabled = algo in {"ar", "dd"} and (
        float(args.epsilon) > 0.0 or float(args.dd_pos_epsilon) > 0.0
    )

    in_progress = _Progress(
        label="in_domain",
        total=len(in_indices),
        every=int(args.progress_every),
        enabled=not bool(args.no_progress),
    )

    for done, idx in enumerate(in_indices, start=1):
        sample = dataset[idx]
        batch = _to_single_batch(sample, device)
        mel_spec = batch["mel_spec"]
        target_audio = _prepare_render_output(sample["audio"].squeeze(0).numpy(), target_len=target_len)
        sample_id = f"{int(idx):06d}"
        best_row: Optional[Dict[str, Any]] = None
        best_value = float("-inf") if str(args.best_metric).lower() in {"reward", "rms"} else float("inf")
        trial_records: List[Dict[str, Any]] = []
        num_trials = int(args.best_of)

        for trial in range(num_trials):
            generator = _make_torch_generator(device, int(args.seed) + int(idx) * 1009 + trial)
            candidate = _predict_candidate(
                algo=algo,
                model=model,
                mel_spec=mel_spec,
                sampling=sampling,
                flow_steps=args.flow_steps,
                flow_cfg_strength=args.flow_cfg_strength,
                generator=generator,
                force_greedy=False,
            )
            losses = _compute_in_domain_losses(algo=algo, model=model, batch=batch, candidate=candidate)
            pred_audio_path = ""
            gt_audio_path = ""
            try:
                pred_audio = render_evaluator.render_from_full_and_midi(candidate["full"], candidate["midi"])[0]
                pred_audio_np = _prepare_render_output(pred_audio, target_len=target_len)
                metrics = _audio_metrics_for_pair(
                    render_evaluator=render_evaluator,
                    pred_audio=pred_audio_np,
                    target_audio=target_audio,
                    sample_rate=sample_rate,
                    crepe_metric=crepe_metric,
                    reward_cfg=reward_cfg,
                )
                success = True
            except Exception as exc:
                pred_audio_np = np.zeros((target_len,), dtype=np.float32)
                metrics = {
                    "wmfcc": float("nan"),
                    "mfcc13": float("nan"),
                    "mfcc40": float("nan"),
                    "clap": float("nan"),
                    "crepe": float("nan"),
                    "mss": float("nan"),
                    "sot": float("nan"),
                    "rms": float("nan"),
                    "reward": failure_reward,
                }
                success = False
                log.warning("In-domain render failed for sample %s trial %d: %s", sample_id, trial, exc)

            selection_value = _selection_value_for_metric(args.best_metric, metrics, failure_reward)
            row = {
                "algo": algo,
                "synth": synth,
                "ckpt": str(ckpt_path),
                "split": "in_domain",
                "sample_id": sample_id,
                "source": f"{in_domain_h5}:{idx}",
                "trial": int(trial),
                "candidate_kind": str(candidate.get("kind", "sample")),
                "selection_metric": str(args.best_metric),
                "selection_value": float(selection_value),
                "best_of": int(args.best_of),
                "include_greedy": bool(args.include_greedy),
                "success": bool(success),
                **losses,
                **metrics,
                "pred_audio_path": pred_audio_path,
                "gt_audio_path": gt_audio_path,
            }
            trial_records.append(row)
            if best_row is None or _is_better(str(args.best_metric), float(selection_value), float(best_value)):
                best_row = row
                best_value = float(selection_value)
                if bool(args.save_audio):
                    pred_audio_path, gt_audio_path = _maybe_write_audio(
                        root=render_root,
                        split="in_domain",
                        sample_id=sample_id,
                        pred_audio=pred_audio_np,
                        target_audio=target_audio,
                        sample_rate=sample_rate,
                    )
                    best_row["pred_audio_path"] = pred_audio_path
                    best_row["gt_audio_path"] = gt_audio_path

        if bool(args.include_greedy) and stochastic_enabled:
            candidate = _predict_candidate(
                algo=algo,
                model=model,
                mel_spec=mel_spec,
                sampling=sampling,
                flow_steps=args.flow_steps,
                flow_cfg_strength=args.flow_cfg_strength,
                generator=_make_torch_generator(device, int(args.seed) + int(idx) * 1009 + 999999),
                force_greedy=True,
            )
            losses = _compute_in_domain_losses(algo=algo, model=model, batch=batch, candidate=candidate)
            pred_audio_path = ""
            gt_audio_path = ""
            try:
                pred_audio = render_evaluator.render_from_full_and_midi(candidate["full"], candidate["midi"])[0]
                pred_audio_np = _prepare_render_output(pred_audio, target_len=target_len)
                metrics = _audio_metrics_for_pair(
                    render_evaluator=render_evaluator,
                    pred_audio=pred_audio_np,
                    target_audio=target_audio,
                    sample_rate=sample_rate,
                    crepe_metric=crepe_metric,
                    reward_cfg=reward_cfg,
                )
                success = True
            except Exception as exc:
                pred_audio_np = np.zeros((target_len,), dtype=np.float32)
                metrics = {
                    "wmfcc": float("nan"),
                    "mfcc13": float("nan"),
                    "mfcc40": float("nan"),
                    "clap": float("nan"),
                    "crepe": float("nan"),
                    "mss": float("nan"),
                    "sot": float("nan"),
                    "rms": float("nan"),
                    "reward": failure_reward,
                }
                success = False
                log.warning("In-domain greedy render failed for sample %s: %s", sample_id, exc)

            selection_value = _selection_value_for_metric(args.best_metric, metrics, failure_reward)
            row = {
                "algo": algo,
                "synth": synth,
                "ckpt": str(ckpt_path),
                "split": "in_domain",
                "sample_id": sample_id,
                "source": f"{in_domain_h5}:{idx}",
                "trial": "greedy",
                "candidate_kind": "greedy",
                "selection_metric": str(args.best_metric),
                "selection_value": float(selection_value),
                "best_of": int(args.best_of),
                "include_greedy": bool(args.include_greedy),
                "success": bool(success),
                **losses,
                **metrics,
                "pred_audio_path": pred_audio_path,
                "gt_audio_path": gt_audio_path,
            }
            trial_records.append(row)
            if best_row is None or _is_better(str(args.best_metric), float(selection_value), float(best_value)):
                best_row = row
                best_value = float(selection_value)
                if bool(args.save_audio):
                    pred_audio_path, gt_audio_path = _maybe_write_audio(
                        root=render_root,
                        split="in_domain",
                        sample_id=sample_id,
                        pred_audio=pred_audio_np,
                        target_audio=target_audio,
                        sample_rate=sample_rate,
                    )
                    best_row["pred_audio_path"] = pred_audio_path
                    best_row["gt_audio_path"] = gt_audio_path

        if best_row is None:
            raise RuntimeError(f"No in-domain candidate was produced for sample {sample_id}.")
        rows.append(best_row)
        in_progress.update(done, extra=f"{args.best_metric}={best_row['selection_value']:.4f}")

    ood_progress = _Progress(
        label="ood",
        total=len(ood_files),
        every=int(args.progress_every),
        enabled=not bool(args.no_progress),
    )

    spec_cfg = OmegaConf.to_container(cfg.data.spec, resolve=True) if "spec" in cfg.data else {}
    for done, wav_path in enumerate(ood_files, start=1):
        audio = _load_audio_resampled(wav_path, target_sr=sample_rate, target_len=target_len)
        mel = _audio_to_mel(
            audio,
            sample_rate=sample_rate,
            spec_cfg=spec_cfg,
            target_frames=target_frames,
            target_n_mels=target_n_mels,
            mean=mean_t,
            std=std_t,
        ).unsqueeze(0).to(device=device)
        sample_id = _slugify(str(wav_path.relative_to(ood_root)).replace("/", "__"))
        best_row: Optional[Dict[str, Any]] = None
        best_value = float("-inf") if str(args.best_metric).lower() in {"reward", "rms"} else float("inf")

        num_trials = int(args.best_of)
        for trial in range(num_trials):
            generator = _make_torch_generator(device, int(args.seed) + done * 1009 + trial)
            candidate = _predict_candidate(
                algo=algo,
                model=model,
                mel_spec=mel,
                sampling=sampling,
                flow_steps=args.flow_steps,
                flow_cfg_strength=args.flow_cfg_strength,
                generator=generator,
                force_greedy=False,
            )
            pred_audio_path = ""
            gt_audio_path = ""
            try:
                pred_audio = render_evaluator.render_from_full_and_midi(candidate["full"], candidate["midi"])[0]
                pred_audio_np = _prepare_render_output(pred_audio, target_len=target_len)
                metrics = _audio_metrics_for_pair(
                    render_evaluator=render_evaluator,
                    pred_audio=pred_audio_np,
                    target_audio=audio,
                    sample_rate=sample_rate,
                    crepe_metric=crepe_metric,
                    reward_cfg=reward_cfg,
                )
                success = True
            except Exception as exc:
                pred_audio_np = np.zeros((target_len,), dtype=np.float32)
                metrics = {
                    "wmfcc": float("nan"),
                    "mfcc13": float("nan"),
                    "mfcc40": float("nan"),
                    "clap": float("nan"),
                    "crepe": float("nan"),
                    "mss": float("nan"),
                    "sot": float("nan"),
                    "rms": float("nan"),
                    "reward": failure_reward,
                }
                success = False
                log.warning("OOD render failed for %s trial %d: %s", wav_path, trial, exc)

            selection_value = _selection_value_for_metric(args.best_metric, metrics, failure_reward)
            row = {
                "algo": algo,
                "synth": synth,
                "ckpt": str(ckpt_path),
                "split": "ood",
                "sample_id": sample_id,
                "source": _path_for_report(wav_path),
                "trial": int(trial),
                "candidate_kind": str(candidate.get("kind", "sample")),
                "selection_metric": str(args.best_metric),
                "selection_value": float(selection_value),
                "best_of": int(args.best_of),
                "include_greedy": bool(args.include_greedy),
                "success": bool(success),
                "loss": float("nan"),
                "param_loss": float("nan"),
                "midi_loss": float("nan"),
                "param_mse": float("nan"),
                **metrics,
                "pred_audio_path": pred_audio_path,
                "gt_audio_path": gt_audio_path,
            }
            if best_row is None or _is_better(str(args.best_metric), float(selection_value), float(best_value)):
                best_row = row
                best_value = float(selection_value)
                if bool(args.save_audio):
                    pred_audio_path, gt_audio_path = _maybe_write_audio(
                        root=render_root,
                        split="ood",
                        sample_id=sample_id,
                        pred_audio=pred_audio_np,
                        target_audio=audio,
                        sample_rate=sample_rate,
                    )
                    best_row["pred_audio_path"] = pred_audio_path
                    best_row["gt_audio_path"] = gt_audio_path

        if bool(args.include_greedy) and stochastic_enabled:
            candidate = _predict_candidate(
                algo=algo,
                model=model,
                mel_spec=mel,
                sampling=sampling,
                flow_steps=args.flow_steps,
                flow_cfg_strength=args.flow_cfg_strength,
                generator=_make_torch_generator(device, int(args.seed) + done * 1009 + 999999),
                force_greedy=True,
            )
            pred_audio_path = ""
            gt_audio_path = ""
            try:
                pred_audio = render_evaluator.render_from_full_and_midi(candidate["full"], candidate["midi"])[0]
                pred_audio_np = _prepare_render_output(pred_audio, target_len=target_len)
                metrics = _audio_metrics_for_pair(
                    render_evaluator=render_evaluator,
                    pred_audio=pred_audio_np,
                    target_audio=audio,
                    sample_rate=sample_rate,
                    crepe_metric=crepe_metric,
                    reward_cfg=reward_cfg,
                )
                success = True
            except Exception as exc:
                pred_audio_np = np.zeros((target_len,), dtype=np.float32)
                metrics = {
                    "wmfcc": float("nan"),
                    "mfcc13": float("nan"),
                    "mfcc40": float("nan"),
                    "clap": float("nan"),
                    "crepe": float("nan"),
                    "mss": float("nan"),
                    "sot": float("nan"),
                    "rms": float("nan"),
                    "reward": failure_reward,
                }
                success = False
                log.warning("OOD greedy render failed for %s: %s", wav_path, exc)

            selection_value = _selection_value_for_metric(args.best_metric, metrics, failure_reward)
            row = {
                "algo": algo,
                "synth": synth,
                "ckpt": str(ckpt_path),
                "split": "ood",
                "sample_id": sample_id,
                "source": _path_for_report(wav_path),
                "trial": "greedy",
                "candidate_kind": "greedy",
                "selection_metric": str(args.best_metric),
                "selection_value": float(selection_value),
                "best_of": int(args.best_of),
                "include_greedy": bool(args.include_greedy),
                "success": bool(success),
                "loss": float("nan"),
                "param_loss": float("nan"),
                "midi_loss": float("nan"),
                "param_mse": float("nan"),
                **metrics,
                "pred_audio_path": pred_audio_path,
                "gt_audio_path": gt_audio_path,
            }
            if best_row is None or _is_better(str(args.best_metric), float(selection_value), float(best_value)):
                best_row = row
                best_value = float(selection_value)
                if bool(args.save_audio):
                    pred_audio_path, gt_audio_path = _maybe_write_audio(
                        root=render_root,
                        split="ood",
                        sample_id=sample_id,
                        pred_audio=pred_audio_np,
                        target_audio=audio,
                        sample_rate=sample_rate,
                    )
                    best_row["pred_audio_path"] = pred_audio_path
                    best_row["gt_audio_path"] = gt_audio_path

        if best_row is None:
            raise RuntimeError(f"No OOD candidate was produced for sample {wav_path}.")
        rows.append(best_row)
        ood_progress.update(done, extra=f"{args.best_metric}={best_row['selection_value']:.4f}")

    metrics_csv = render_root / "metrics.csv"
    _save_rows_csv(metrics_csv, rows)

    run_info = {
        "ckpt": str(ckpt_path),
        "algo": algo,
        "synth": synth,
        "render_root": _path_for_report(render_root),
        "in_domain_h5": _path_for_report(in_domain_h5),
        "ood_audio_root": _path_for_report(ood_root),
        "args": vars(args),
        "reward_cfg": reward_cfg,
        "counts": {
            "in_domain": int(len(in_indices)),
            "ood": int(len(ood_files)),
            "rows": int(len(rows)),
        },
    }
    (render_root / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")

    render_evaluator.close()
    dataset.close()
    log.info("Saved %d rows to %s", len(rows), metrics_csv)


if __name__ == "__main__":
    main()
