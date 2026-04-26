from __future__ import annotations

import csv
import json
import logging
import math
import pickle
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import hydra
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

from src.grpo import build_grpo_renderer_pool, infer_synth_name
from src.project_paths import project_relative_string, resolve_project_path
from src.utils.audio_metrics import (
    ClapCosineDistance,
    CrepeEmbeddingDistance,
    WmfccMetric,
    compute_extra_audio_metrics,
)
from src.utils.wandb_resume import infer_wandb_resume

log = logging.getLogger(__name__)


def _looks_like_path_key(key: str) -> bool:
    key = str(key)
    return key.endswith("_dir") or key.endswith("_path") or key in {"save_dir", "dir", "filename"}


def _resolve_path_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        resolved: Dict[str, Any] = {}
        for key, value in obj.items():
            item = _resolve_path_fields(value)
            if isinstance(item, str) and _looks_like_path_key(str(key)):
                item = str(resolve_project_path(item))
            resolved[str(key)] = item
        return resolved
    if isinstance(obj, list):
        return [_resolve_path_fields(item) for item in obj]
    return obj


def _instantiate_named_collection(cfg_section: DictConfig | None) -> list[Any]:
    if cfg_section is None:
        return []
    container = OmegaConf.to_container(cfg_section, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected config section to resolve to a mapping, got {type(container).__name__}")
    instances: list[Any] = []
    for _, item_cfg in container.items():
        if not isinstance(item_cfg, dict) or "_target_" not in item_cfg:
            continue
        instances.append(hydra.utils.instantiate(_resolve_path_fields(item_cfg)))
    return instances


def _apply_wandb_resume(cfg: DictConfig) -> None:
    resume_kwargs = infer_wandb_resume(
        cfg.get("logger"),
        cfg,
        resolve_project_path(cfg.get("ckpt_path")) if cfg.get("ckpt_path") else None,
    )
    if not resume_kwargs:
        return
    if "logger" not in cfg or "wandb" not in cfg.logger:
        return
    with open_dict(cfg.logger.wandb):
        for key, value in resume_kwargs.items():
            cfg.logger.wandb[key] = value


def _nanmean(values: List[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / float(len(finite)))


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
        "Checkpoint requires unsafe loading. Re-run with `trust_ckpt=true` only if you trust the file."
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


def _instantiate_dd_model(cfg: DictConfig, ckpt_dir: Path) -> torch.nn.Module:
    target = str(getattr(cfg.model, "_target_", ""))
    if not target.endswith("DiscreteDiffusionModule"):
        raise ValueError(f"Expected DiscreteDiffusionModule checkpoint, got model target '{target}'.")
    from src.models.discrete_diffusion_module import DiscreteDiffusionModule

    return DiscreteDiffusionModule(cfg, ckpt_dir=ckpt_dir)


def _load_state_dict_into_model(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint does not contain `state_dict`.")
    model.load_state_dict(state_dict, strict=False)


def _load_stats_and_frames(dataset_root: Path) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], Optional[int]]:
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


def _list_wavs(root: Path, *, max_files: int) -> List[Path]:
    files = sorted([p for p in root.rglob("*.wav") if p.is_file()])
    if int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
    except Exception:
        pass
    try:
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
    except Exception:
        pass
    try:
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


def _save_ckpt(
    path: Path,
    *,
    model: torch.nn.Module,
    step: int,
    cfg: DictConfig,
    dd_cfg: DictConfig,
    optimizer_state: Optional[Dict[str, Any]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "global_step": int(step),
        "finetune_cfg": OmegaConf.to_container(cfg, resolve=True),
        "base_cfg": OmegaConf.to_container(dd_cfg, resolve=True),
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if rng_state is not None:
        payload["rng_state"] = rng_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _update_best_reward_ckpts(
    *,
    best_list: List[Dict[str, Any]],
    new_reward: float,
    step: int,
    ckpt_dir: Path,
    model: torch.nn.Module,
    cfg: DictConfig,
    dd_cfg: DictConfig,
    optimizer_state: Optional[Dict[str, Any]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    keep: int = 3,
) -> List[Dict[str, Any]]:
    if not math.isfinite(new_reward):
        return best_list
    if best_list and len(best_list) >= keep:
        current_min = min(float(item.get("reward", float("-inf"))) for item in best_list)
        if float(new_reward) <= current_min:
            return best_list

    candidate_path = ckpt_dir / f"checkpoint_best_reward_candidate_step_{step:08d}.pt"
    _save_ckpt(
        candidate_path,
        model=model,
        step=step,
        cfg=cfg,
        dd_cfg=dd_cfg,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
    )
    best_list.append({"reward": float(new_reward), "step": int(step), "path": candidate_path})
    best_list.sort(key=lambda x: x["reward"], reverse=True)

    extras = best_list[keep:]
    best_list = best_list[:keep]
    for item in extras:
        path = item.get("path")
        if isinstance(path, Path) and path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    for rank, item in enumerate(best_list, start=1):
        dest = ckpt_dir / f"checkpoint_best_reward_{rank}_step_{int(item['step']):08d}.pt"
        src = item.get("path")
        if isinstance(src, Path) and src.exists() and src != dest:
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
            try:
                src.rename(dest)
                item["path"] = dest
            except Exception:
                pass
    return best_list


@dataclass(frozen=True)
class _SamplingCfg:
    token_epsilon: float
    token_top_k: int
    token_min_prob: float
    pos_epsilon: float
    pos_top_k: int
    pos_min_prob: float
    pos_temperature: float
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


def _apply_topk_minprob(probs_full: torch.Tensor, *, top_k: int, min_prob: float) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if probs_full.ndim != 1:
        raise ValueError(f"Expected 1D probs, got {tuple(probs_full.shape)}")
    device = probs_full.device
    allowed = torch.arange(probs_full.numel(), device=device, dtype=torch.long)

    if int(top_k) > 0 and int(allowed.numel()) > int(top_k):
        top_vals, top_idx = torch.topk(probs_full, k=int(top_k), dim=-1)
        allowed = allowed[top_idx]
        allowed_probs = top_vals
    else:
        allowed_probs = probs_full

    denom = allowed_probs.sum()
    if not torch.isfinite(denom) or float(denom.item()) <= 0.0:
        return None
    probs = allowed_probs / denom

    pmin = float(min_prob)
    if pmin > 0.0:
        keep = probs >= pmin
        if not bool(keep.any().item()):
            return None
        allowed = allowed[keep]
        probs = probs[keep]
        denom2 = probs.sum()
        if not torch.isfinite(denom2) or float(denom2.item()) <= 0.0:
            return None
        probs = probs / denom2
    return allowed, probs


def _epsilon_greedy_sample_1d(
    logits_1d: torch.Tensor,
    *,
    epsilon: float,
    top_k: int,
    min_prob: float,
    force_greedy: bool,
    generator: torch.Generator,
) -> Tuple[int, torch.Tensor]:
    greedy = int(torch.argmax(logits_1d, dim=-1).item())
    if bool(force_greedy) or float(epsilon) <= 0.0:
        return greedy, logits_1d.new_zeros(())

    if float(epsilon) >= 1.0:
        choose_sample = True
    else:
        choose_sample = bool(torch.rand((), generator=generator, device=logits_1d.device).item() < float(epsilon))

    probs_full = F.softmax(logits_1d, dim=-1)
    filtered = _apply_topk_minprob(probs_full, top_k=int(top_k), min_prob=float(min_prob))
    if filtered is None:
        return greedy, logits_1d.new_zeros(())
    allowed, probs = filtered

    if not choose_sample:
        action = greedy
    elif int(allowed.numel()) == 1:
        action = int(allowed[0].item())
    else:
        pick = int(torch.multinomial(probs, 1, generator=generator).item())
        action = int(allowed[pick].item())

    eps = float(epsilon)
    q_full = probs_full.new_zeros((probs_full.numel(),))
    q_full[allowed] = probs
    q_action = q_full[action]
    q_greedy = q_full[greedy]
    p = (1.0 - eps) + eps * q_greedy if action == greedy else eps * q_action
    return action, torch.log(torch.clamp(p, min=1e-12))


def _epsilon_greedy_logprob_1d(
    logits_1d: torch.Tensor,
    *,
    action: int,
    epsilon: float,
    top_k: int,
    min_prob: float,
    force_greedy: bool,
) -> torch.Tensor:
    greedy = int(torch.argmax(logits_1d, dim=-1).item())
    if bool(force_greedy) or float(epsilon) <= 0.0:
        return logits_1d.new_zeros(())

    probs_full = F.softmax(logits_1d, dim=-1)
    filtered = _apply_topk_minprob(probs_full, top_k=int(top_k), min_prob=float(min_prob))
    if filtered is None:
        return logits_1d.new_zeros(())
    allowed, probs = filtered

    eps = float(epsilon)
    q_full = probs_full.new_zeros((probs_full.numel(),))
    q_full[allowed] = probs
    q_action = q_full[int(action)]
    q_greedy = q_full[greedy]
    p = (1.0 - eps) + eps * q_greedy if int(action) == greedy else eps * q_action
    return torch.log(torch.clamp(p, min=1e-12))


def _pos_distribution_from_conf(
    conf_1d: torch.Tensor,
    locked_1d: torch.Tensor,
    *,
    top_k: int,
    min_prob: float,
    temperature: float,
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

    tau = float(temperature)
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError(f"pos.temperature must be finite > 0, got {temperature!r}")
    probs = torch.clamp(weights, min=1e-12).pow(1.0 / tau)
    denom = probs.sum()
    if not torch.isfinite(denom) or float(denom.item()) <= 0.0:
        return None
    probs = probs / denom

    pmin = float(min_prob)
    if pmin > 0.0:
        keep = probs >= pmin
        if not bool(keep.any().item()):
            return None
        allowed = allowed[keep]
        probs = probs[keep]
        denom2 = probs.sum()
        if not torch.isfinite(denom2) or float(denom2.item()) <= 0.0:
            return None
        probs = probs / denom2
    return allowed, probs.to(dtype=conf_1d.dtype)


def _dd_select_next_pos_with_logp(
    *,
    dd_order: Sequence[str],
    locked: torch.Tensor,
    conf: torch.Tensor,
    dd_midi_first: bool,
    epsilon: float,
    top_k: int,
    min_prob: float,
    temperature: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, length = locked.shape
    name_to_pos: Dict[str, int] = {}
    for i, name in enumerate(dd_order):
        up = str(name).upper()
        if up not in name_to_pos:
            name_to_pos[up] = int(i)

    chosen: List[int] = []
    logps: List[torch.Tensor] = []
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
            logps.append(conf.new_zeros(()))
            continue

        conf_row = conf[bi]
        locked_row = locked[bi]
        greedy = int(torch.argmax(conf_row.masked_fill(locked_row, -1e9)).item())
        if float(epsilon) <= 0.0:
            chosen.append(greedy)
            logps.append(conf.new_zeros(()))
            continue

        if float(epsilon) >= 1.0:
            choose_sample = True
        else:
            choose_sample = bool(torch.rand((), generator=generator, device=conf_row.device).item() < float(epsilon))

        dist = _pos_distribution_from_conf(
            conf_row,
            locked_row,
            top_k=int(top_k),
            min_prob=float(min_prob),
            temperature=float(temperature),
        )
        if dist is None:
            chosen.append(greedy)
            logps.append(conf.new_zeros(()))
            continue
        allowed, probs = dist
        if not choose_sample:
            action = greedy
        elif int(allowed.numel()) == 1:
            action = int(allowed[0].item())
        else:
            pick = int(torch.multinomial(probs, 1, generator=generator).item())
            action = int(allowed[pick].item())

        eps = float(epsilon)
        q_full = probs.new_zeros((length,))
        q_full[allowed] = probs
        q_action = q_full[action]
        q_greedy = q_full[greedy]
        p = (1.0 - eps) + eps * q_greedy if action == greedy else eps * q_action
        chosen.append(action)
        logps.append(torch.log(torch.clamp(p, min=1e-12)))

    return torch.tensor(chosen, dtype=torch.long, device=locked.device), torch.stack(logps, dim=0)


def _dd_pos_logprob_for_action(
    *,
    dd_order: Sequence[str],
    locked_row: torch.Tensor,
    conf_row: torch.Tensor,
    action: int,
    dd_midi_first: bool,
    epsilon: float,
    top_k: int,
    min_prob: float,
    temperature: float,
) -> torch.Tensor:
    length = int(conf_row.numel())
    if bool(dd_midi_first):
        name_to_pos: Dict[str, int] = {}
        for i, name in enumerate(dd_order):
            up = str(name).upper()
            if up not in name_to_pos:
                name_to_pos[up] = int(i)
        for midi_name in _DD_MIDI_FIRST_ORDER:
            pos = name_to_pos.get(str(midi_name).upper())
            if pos is not None and not bool(locked_row[pos].item()):
                return conf_row.new_zeros(())

    greedy = int(torch.argmax(conf_row.masked_fill(locked_row, -1e9)).item())
    if float(epsilon) <= 0.0:
        return conf_row.new_zeros(())

    dist = _pos_distribution_from_conf(
        conf_row,
        locked_row,
        top_k=int(top_k),
        min_prob=float(min_prob),
        temperature=float(temperature),
    )
    if dist is None:
        return conf_row.new_zeros(())
    allowed, probs = dist

    eps = float(epsilon)
    q_full = probs.new_zeros((length,))
    q_full[allowed] = probs
    q_action = q_full[int(action)]
    q_greedy = q_full[greedy]
    p = (1.0 - eps) + eps * q_greedy if int(action) == greedy else eps * q_action
    return torch.log(torch.clamp(p, min=1e-12))


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
        u = torch.zeros_like(entropy)
        u = torch.where(safe, entropy / torch.clamp(log_card, min=1e-12), u)
        return (1.0 - u).clamp(0.0, 1.0).to(dtype=probs.dtype)

    p_max = probs.max(dim=-1).values
    if use_normalized_confidence:
        baseline = 1.0 / torch.clamp(cardinals, min=1.0)
        denom = 1.0 - baseline
        p_max = p_max.to(dtype=baseline.dtype)
        conf = (p_max - baseline) / denom
        conf = torch.where(denom > 0, conf, torch.zeros_like(conf))
        return conf.to(dtype=probs.dtype)
    return p_max


def _dd_rollout_with_logps(
    *,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    spectrogram: torch.Tensor,
    sampling: _SamplingCfg,
    generator: torch.Generator,
    use_checkpointing: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dd = getattr(model, "model")
    dd_ref = getattr(ref_model, "model")
    device = next(dd.parameters()).device
    spec = spectrogram.to(device=device)
    batch = spec.shape[0]

    length = int(getattr(dd, "seq_len"))
    horizon = int(getattr(model, "diffusion_T"))
    if length != horizon:
        raise RuntimeError(f"Expected DD sequence length to equal diffusion horizon, got {length} vs {horizon}.")

    mask_id = int(getattr(model, "mask_token_id"))
    x = torch.full((batch, length), mask_id, dtype=torch.long, device=device)
    locked = torch.zeros((batch, length), dtype=torch.bool, device=device)

    cardinals = torch.as_tensor(getattr(dd, "cardinals"), device=device, dtype=torch.float32)
    cardinals_ref = torch.as_tensor(getattr(dd_ref, "cardinals"), device=device, dtype=torch.float32)
    logp_sum = spec.new_zeros((batch,), dtype=torch.float32)
    logp_ref_sum = spec.new_zeros((batch,), dtype=torch.float32)

    def _forward(module: torch.nn.Module, spec_in: torch.Tensor, x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        if not use_checkpointing:
            return module(spec_in, x_in, t_in)
        from torch.utils.checkpoint import checkpoint

        return checkpoint(module, spec_in, x_in.clone(), t_in.clone(), use_reentrant=False)

    dd.eval()
    dd_ref.eval()
    for step in range(length):
        locked_snapshot = locked.clone()
        t = torch.full((batch,), horizon - step, dtype=torch.long, device=device)
        logits = _forward(dd, spec, x, t)
        with torch.no_grad():
            logits_ref = dd_ref(spec, x, t)

        probs = F.softmax(logits, dim=-1)
        probs_ref = F.softmax(logits_ref, dim=-1)
        conf = _compute_confidence(
            probs,
            cardinals=cardinals,
            use_normalized_entropy=bool(sampling.dd_normalized_entropy),
            use_normalized_confidence=bool(sampling.dd_normalized_confidence) and not bool(sampling.dd_normalized_entropy),
        )
        conf_ref = _compute_confidence(
            probs_ref,
            cardinals=cardinals_ref,
            use_normalized_entropy=bool(sampling.dd_normalized_entropy),
            use_normalized_confidence=bool(sampling.dd_normalized_confidence) and not bool(sampling.dd_normalized_entropy),
        )

        j, logp_pos = _dd_select_next_pos_with_logp(
            dd_order=list(getattr(dd, "order")),
            locked=locked_snapshot,
            conf=conf,
            dd_midi_first=bool(sampling.dd_midi_first),
            epsilon=float(sampling.pos_epsilon),
            top_k=int(sampling.pos_top_k),
            min_prob=float(sampling.pos_min_prob),
            temperature=float(sampling.pos_temperature),
            generator=generator,
        )
        with torch.no_grad():
            logp_pos_ref = []
            order_ref = list(getattr(dd_ref, "order"))
            for bi in range(batch):
                logp_pos_ref.append(
                    _dd_pos_logprob_for_action(
                        dd_order=order_ref,
                        locked_row=locked_snapshot[bi],
                        conf_row=conf_ref[bi],
                        action=int(j[bi].item()),
                        dd_midi_first=bool(sampling.dd_midi_first),
                        epsilon=float(sampling.pos_epsilon),
                        top_k=int(sampling.pos_top_k),
                        min_prob=float(sampling.pos_min_prob),
                        temperature=float(sampling.pos_temperature),
                    )
                )
            logp_pos_ref = torch.stack(logp_pos_ref, dim=0)

        batch_idx = torch.arange(batch, device=device)
        chosen_ids: List[int] = []
        logp_tok_list: List[torch.Tensor] = []
        logp_tok_ref_list: List[torch.Tensor] = []
        for bi in range(batch):
            pos = int(j[bi].item())
            token_name = str(getattr(dd, "order")[pos])
            force_greedy = _should_force_greedy_token(
                token_name,
                force_greedy_special=bool(sampling.force_greedy_special),
                force_greedy_midi_only=bool(sampling.force_greedy_midi_only),
            )
            cardinal = int(getattr(dd, "cardinals")[pos])
            if cardinal <= 1:
                chosen_ids.append(0)
                logp_tok_list.append(spec.new_zeros(()))
                logp_tok_ref_list.append(spec.new_zeros(()))
                continue

            step_logits = logits[bi, pos, :cardinal]
            action, logp_tok = _epsilon_greedy_sample_1d(
                step_logits,
                epsilon=float(sampling.token_epsilon),
                top_k=min(int(sampling.token_top_k), cardinal) if int(sampling.token_top_k) > 0 else 0,
                min_prob=float(sampling.token_min_prob),
                force_greedy=force_greedy,
                generator=generator,
            )
            chosen_ids.append(int(action))
            logp_tok_list.append(logp_tok)

            with torch.no_grad():
                step_logits_ref = logits_ref[bi, pos, :cardinal]
                logp_tok_ref = _epsilon_greedy_logprob_1d(
                    step_logits_ref,
                    action=int(action),
                    epsilon=float(sampling.token_epsilon),
                    top_k=min(int(sampling.token_top_k), cardinal) if int(sampling.token_top_k) > 0 else 0,
                    min_prob=float(sampling.token_min_prob),
                    force_greedy=force_greedy,
                )
                logp_tok_ref_list.append(logp_tok_ref)

        token_id = torch.tensor(chosen_ids, dtype=torch.long, device=device)
        x[batch_idx, j] = token_id
        locked = locked_snapshot.clone()
        locked[batch_idx, j] = True
        logp_sum = logp_sum + logp_pos.to(dtype=logp_sum.dtype) + torch.stack(logp_tok_list, dim=0).to(dtype=logp_sum.dtype)
        logp_ref_sum = logp_ref_sum + logp_pos_ref.to(dtype=logp_ref_sum.dtype) + torch.stack(logp_tok_ref_list, dim=0).to(dtype=logp_ref_sum.dtype)

    return x, logp_sum, logp_ref_sum


def _surge_params_from_full(full_vec: np.ndarray, backend_param_names: Sequence[str]) -> Dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(backend_param_names, np.asarray(full_vec, dtype=np.float32))
    }


def _build_render_jobs(
    *,
    synth: str,
    full_params: np.ndarray,
    midi_absolute: np.ndarray,
    backend_param_names: Sequence[str],
    release_ratio: float,
) -> List[dict[str, Any]]:
    jobs: List[dict[str, Any]] = []
    for full_vec, midi_vec in zip(full_params, midi_absolute):
        note = int(round(float(midi_vec[0])))
        velocity = int(round(float(midi_vec[1])))
        duration = max(float(midi_vec[2]), 0.01)
        release = max(duration * float(release_ratio), 0.01)
        if synth == "dexed":
            jobs.append(
                {
                    "preset": np.asarray(full_vec, dtype=np.float32),
                    "midi_note": note,
                    "midi_velocity": velocity,
                    "sustain": duration,
                    "release": release,
                }
            )
        else:
            jobs.append(
                {
                    "params": _surge_params_from_full(full_vec, backend_param_names),
                    "midi": {
                        "note": note,
                        "velocity": velocity,
                        "duration": duration,
                        "release": release,
                    },
                }
            )
    return jobs


def _render_jobs_resilient(
    *,
    renderer: Any,
    jobs: Sequence[dict[str, Any]],
    target_len: int,
) -> Tuple[List[Optional[np.ndarray]], List[bool]]:
    outputs: List[Optional[np.ndarray]] = [None] * len(jobs)
    ok = [False] * len(jobs)
    try:
        batch_audio = renderer.render_jobs(jobs)
        for idx, audio in enumerate(batch_audio):
            outputs[idx] = _prepare_render_output(audio, target_len=target_len)
            ok[idx] = True
        return outputs, ok
    except Exception:
        pass

    for idx, job in enumerate(jobs):
        try:
            if "preset" in job:
                audio = renderer.render_single(
                    preset=np.asarray(job["preset"], dtype=np.float32),
                    midi_note=int(job["midi_note"]),
                    midi_velocity=int(job["midi_velocity"]),
                    sustain=float(job["sustain"]),
                    release=float(job["release"]),
                )
            else:
                audio = renderer.render_single(params=dict(job["params"]), midi=dict(job["midi"]))
            outputs[idx] = _prepare_render_output(audio, target_len=target_len)
            ok[idx] = True
        except Exception:
            continue
    return outputs, ok


def _compute_reward_components(
    *,
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    wmfcc_metric: WmfccMetric,
    clap_metric: ClapCosineDistance | None,
    crepe_metric: CrepeEmbeddingDistance | None,
    weights: Mapping[str, float],
    reward_sign: float,
) -> Dict[str, float]:
    pred_t = torch.from_numpy(pred_audio).unsqueeze(0)
    target_t = torch.from_numpy(target_audio).unsqueeze(0)
    metrics: Dict[str, float] = {
        "wmfcc": float(wmfcc_metric(pred_t, target_t).item()),
        "clap": 0.0,
        "crepe": float("nan"),
        "mss": float("nan"),
        "sot": float("nan"),
        "rms": float("nan"),
    }
    if clap_metric is not None:
        metrics["clap"] = float(clap_metric(pred_audio, target_audio, sample_rate=sample_rate))
    if crepe_metric is not None:
        metrics["crepe"] = float(crepe_metric(pred_audio, target_audio, sample_rate=sample_rate))

    extra = compute_extra_audio_metrics(
        pred_t,
        target_t,
        sample_rate,
        mss=float(weights.get("mss", 0.0)) != 0.0,
        sot=float(weights.get("sot", 0.0)) != 0.0,
        rms=float(weights.get("rms", 0.0)) != 0.0,
    )
    metrics.update(extra)

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
    metrics["reward"] = float(reward_sign) * reward_base
    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune/dd_grpo")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    output_dir = resolve_project_path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _apply_wandb_resume(cfg)

    loggers: List[Logger] = _instantiate_named_collection(cfg.get("logger"))
    if loggers:
        hparams = OmegaConf.to_container(cfg, resolve=True)
        for logger in loggers:
            try:
                logger.log_hyperparams(hparams)
            except Exception:
                pass

    if cfg.get("seed") is not None:
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    ckpt_path = resolve_project_path(cfg.ckpt_path)
    ref_ckpt_path = resolve_project_path(cfg.get("ref_ckpt_path") or cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not ref_ckpt_path.exists():
        raise FileNotFoundError(f"Reference checkpoint not found: {ref_ckpt_path}")

    log.info("Loading policy checkpoint from %s", ckpt_path)
    ckpt = _load_checkpoint_safe(ckpt_path, trust_ckpt=bool(cfg.get("trust_ckpt", True)))
    dd_cfg = _load_cfg_from_ckpt_or_hydra(ckpt, ckpt_path, output_dir=output_dir)
    dd_cfg = OmegaConf.create(OmegaConf.to_container(dd_cfg, resolve=True))
    model = _instantiate_dd_model(dd_cfg, ckpt_dir=ckpt_dir)
    _load_state_dict_into_model(model, ckpt)

    log.info("Loading reference checkpoint from %s", ref_ckpt_path)
    ref_ckpt = _load_checkpoint_safe(ref_ckpt_path, trust_ckpt=bool(cfg.get("trust_ckpt", True)))
    ref_dd_cfg = _load_cfg_from_ckpt_or_hydra(ref_ckpt, ref_ckpt_path, output_dir=output_dir)
    ref_dd_cfg = OmegaConf.create(OmegaConf.to_container(ref_dd_cfg, resolve=True))
    ref_model = _instantiate_dd_model(ref_dd_cfg, ckpt_dir=ckpt_dir)
    _load_state_dict_into_model(ref_model, ref_ckpt)

    synth = infer_synth_name(dd_cfg)
    renderer = build_grpo_renderer_pool(
        cfg=dd_cfg,
        synth=synth,
        num_workers=int(cfg.grpo.get("renderer_workers", 1) or 1),
    )
    log.info("Initialized %s GRPO render pool with %d worker(s).", synth, int(cfg.grpo.get("renderer_workers", 1)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    dataset_root = resolve_project_path(dd_cfg.data.root)
    mel_mean, mel_std, target_n_mels, target_frames = _load_stats_and_frames(dataset_root)
    sample_rate = int(model.dataset_sample_rate)
    target_duration = float(model.dataset_target_duration)
    target_len = int(sample_rate * target_duration)
    spec_cfg = OmegaConf.to_container(cfg.data.get("spec"), resolve=True) if cfg.data.get("spec") else {}
    weights_cfg = OmegaConf.to_container(cfg.reward.weights, resolve=True)
    release_ratio = float(dd_cfg.model.renderer.get("release_ratio", 0.5))
    backend_param_names = tuple(str(name) for name in model.helper.preset_helper.vst_param_names)

    train_files = _list_wavs(resolve_project_path(cfg.data.train_audio_root), max_files=int(cfg.data.max_train_files))
    val_files = _list_wavs(resolve_project_path(cfg.data.val_audio_root), max_files=int(cfg.data.max_val_files))
    if not train_files:
        raise FileNotFoundError("No training WAV files found for GRPO.")
    log.info("Loaded %d train prompts and %d val prompts for GRPO.", len(train_files), len(val_files))
    if cfg.data.get("shuffle", True):
        random.shuffle(train_files)
    if cfg.data.get("val_shuffle_once", True):
        rng = random.Random(int(cfg.data.get("val_seed", 1234)))
        rng.shuffle(val_files)

    wmfcc_metric = WmfccMetric(sample_rate=sample_rate)
    clap_metric = ClapCosineDistance(device=device) if float(weights_cfg.get("clap", 0.0)) != 0.0 else None
    crepe_metric = None
    if bool(cfg.reward.crepe.get("enable", True)) and float(weights_cfg.get("crepe", 0.0)) != 0.0:
        crepe_metric = CrepeEmbeddingDistance(
            device=device,
            model=str(cfg.reward.crepe.get("model", "tiny")),
            metric=str(cfg.reward.crepe.get("metric", "cosine")),
            return_similarity=cfg.reward.crepe.get("return_similarity"),
            hop_length=cfg.reward.crepe.get("hop_length"),
            batch_size=cfg.reward.crepe.get("batch_size"),
            pad=bool(cfg.reward.crepe.get("pad", True)),
        )

    sampling = _SamplingCfg(
        token_epsilon=float(cfg.sampling.token.epsilon),
        token_top_k=int(cfg.sampling.token.top_k),
        token_min_prob=float(cfg.sampling.token.min_prob),
        pos_epsilon=float(cfg.sampling.pos.epsilon),
        pos_top_k=int(cfg.sampling.pos.top_k),
        pos_min_prob=float(cfg.sampling.pos.min_prob),
        pos_temperature=float(cfg.sampling.pos.temperature),
        dd_midi_first=bool(cfg.sampling.dd_midi_first),
        dd_normalized_entropy=bool(cfg.sampling.dd_normalized_entropy),
        dd_normalized_confidence=bool(cfg.sampling.dd_normalized_confidence),
        force_greedy_midi_only=bool(cfg.sampling.force_greedy_midi_only),
        force_greedy_special=bool(cfg.sampling.force_greedy_special),
    )
    sampling_eval = replace(sampling, token_epsilon=0.0, pos_epsilon=0.0)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.grpo.optimizer.lr),
        betas=tuple(float(x) for x in cfg.grpo.optimizer.betas),
        eps=float(cfg.grpo.optimizer.eps),
        weight_decay=float(cfg.grpo.optimizer.weight_decay),
    )
    resume_opt = False
    if isinstance(ckpt, dict) and "optimizer_state" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
            resume_opt = True
            log.info("Loaded optimizer state from checkpoint.")
        except Exception as exc:
            log.warning("Failed to load optimizer state: %s", exc)
    if isinstance(ckpt, dict) and "rng_state" in ckpt:
        _restore_rng_state(ckpt["rng_state"])

    metrics_csv = output_dir / "metrics.csv"
    write_header = not metrics_csv.exists()
    csv_f = metrics_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_f,
        fieldnames=[
            "step",
            "wmfcc_mean",
            "clap_mean",
            "crepe_mean",
            "mss_mean",
            "sot_mean",
            "rms_mean",
            "val_wmfcc_mean",
            "val_clap_mean",
            "val_crepe_mean",
            "val_mss_mean",
            "val_sot_mean",
            "val_rms_mean",
            "val_reward_mean",
            "reward_mean",
            "reward_std",
            "adv_std_mean",
            "logp_mean",
            "kl_mean",
            "loss",
        ],
    )
    if write_header:
        writer.writeheader()
        csv_f.flush()

    def _log_metrics(metrics: Dict[str, float], *, step: int) -> None:
        for logger in loggers:
            try:
                logger.log_metrics(metrics, step=int(step))
            except Exception:
                pass

    def _run_val(step: int) -> Tuple[float, float, float, float, float, float, float]:
        eval_prompts = int(cfg.grpo.get("eval_prompts", 0) or 0)
        if eval_prompts <= 0 or not val_files:
            return (float("nan"),) * 7

        model.eval()
        eval_paths = val_files[: min(eval_prompts, len(val_files))]
        eval_metrics: Dict[str, List[float]] = {
            "wmfcc": [],
            "clap": [],
            "crepe": [],
            "mss": [],
            "sot": [],
            "rms": [],
            "reward": [],
        }
        with torch.no_grad():
            for wav_path in eval_paths:
                try:
                    audio_np = _load_audio_resampled(wav_path, target_sr=sample_rate, target_len=target_len)
                    mel = _audio_to_mel(
                        audio_np,
                        sample_rate=sample_rate,
                        spec_cfg=spec_cfg,
                        target_frames=target_frames,
                        target_n_mels=target_n_mels,
                        mean=mel_mean,
                        std=mel_std,
                    )
                    spec_1 = mel.unsqueeze(0).to(device=device)
                    gg = torch.Generator(device=device)
                    if cfg.get("seed") is not None:
                        gg.manual_seed(int(cfg.seed) + 10_000_000 + int(step))
                    tok, _, _ = _dd_rollout_with_logps(
                        model=model,
                        ref_model=ref_model,
                        spectrogram=spec_1,
                        sampling=sampling_eval,
                        generator=gg,
                        use_checkpointing=False,
                    )
                    full_t, midi_t = model._tokens_to_full_and_midi(tok.detach().cpu())
                    jobs = _build_render_jobs(
                        synth=synth,
                        full_params=full_t.numpy(),
                        midi_absolute=midi_t.numpy(),
                        backend_param_names=backend_param_names,
                        release_ratio=release_ratio,
                    )
                    rendered, ok = _render_jobs_resilient(renderer=renderer, jobs=jobs, target_len=target_len)
                    if not ok or not ok[0] or rendered[0] is None:
                        continue
                    metrics = _compute_reward_components(
                        pred_audio=rendered[0],
                        target_audio=audio_np,
                        sample_rate=sample_rate,
                        wmfcc_metric=wmfcc_metric,
                        clap_metric=clap_metric,
                        crepe_metric=crepe_metric,
                        weights=weights_cfg,
                        reward_sign=float(cfg.reward.sign),
                    )
                    for key in eval_metrics:
                        eval_metrics[key].append(float(metrics.get(key, float("nan"))))
                except Exception:
                    continue
        model.train()
        return (
            _nanmean(eval_metrics["wmfcc"]),
            _nanmean(eval_metrics["clap"]),
            _nanmean(eval_metrics["crepe"]),
            _nanmean(eval_metrics["mss"]),
            _nanmean(eval_metrics["sot"]),
            _nanmean(eval_metrics["rms"]),
            _nanmean(eval_metrics["reward"]),
        )

    resume_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
    start_step = resume_step if resume_opt else 0
    train_idx = 0
    updates = int(cfg.grpo.updates)
    batch_size = int(cfg.grpo.batch_size)
    group_size = int(cfg.grpo.group_size)
    eval_every = int(cfg.grpo.get("eval_every", 0) or 0)
    save_last_every = int(cfg.grpo.get("save_last_every", 0) or 0)
    best_reward_ckpts: List[Dict[str, Any]] = []
    latest_val = {k: float("nan") for k in ["wmfcc", "clap", "crepe", "mss", "sot", "rms", "reward"]}

    for step in range(start_step + 1, updates + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        batch_paths: List[Path] = []
        for _ in range(batch_size):
            if train_idx >= len(train_files):
                train_idx = 0
                if cfg.data.get("shuffle", True):
                    random.shuffle(train_files)
            batch_paths.append(train_files[train_idx])
            train_idx += 1

        mels: List[torch.Tensor] = []
        gt_audio: List[np.ndarray] = []
        for wav_path in batch_paths:
            audio_np = _load_audio_resampled(wav_path, target_sr=sample_rate, target_len=target_len)
            gt_audio.append(audio_np)
            mels.append(
                _audio_to_mel(
                    audio_np,
                    sample_rate=sample_rate,
                    spec_cfg=spec_cfg,
                    target_frames=target_frames,
                    target_n_mels=target_n_mels,
                    mean=mel_mean,
                    std=mel_std,
                )
            )

        spec_batch = torch.stack(mels, dim=0).to(device=device)
        spec_roll = spec_batch.repeat_interleave(group_size, dim=0)
        g = torch.Generator(device=device)
        if cfg.get("seed") is not None:
            g.manual_seed(int(cfg.seed) + int(step))

        token_ids, logp, logp_ref = _dd_rollout_with_logps(
            model=model,
            ref_model=ref_model,
            spectrogram=spec_roll,
            sampling=sampling,
            generator=g,
            use_checkpointing=bool(cfg.grpo.get("use_checkpointing", False)),
        )

        full_t, midi_t = model._tokens_to_full_and_midi(token_ids.detach().cpu())
        jobs = _build_render_jobs(
            synth=synth,
            full_params=full_t.numpy(),
            midi_absolute=midi_t.numpy(),
            backend_param_names=backend_param_names,
            release_ratio=release_ratio,
        )
        pred_audio_batch, ok_mask = _render_jobs_resilient(renderer=renderer, jobs=jobs, target_len=target_len)

        reward_values: List[float] = []
        metrics_buffer = {k: [] for k in ["wmfcc", "clap", "crepe", "mss", "sot", "rms"]}
        for i in range(batch_size * group_size):
            prompt_idx = i // group_size
            if not ok_mask[i] or pred_audio_batch[i] is None:
                for key in metrics_buffer:
                    metrics_buffer[key].append(float("nan"))
                reward_values.append(float(cfg.reward.failure))
                continue
            try:
                metrics = _compute_reward_components(
                    pred_audio=pred_audio_batch[i],
                    target_audio=gt_audio[prompt_idx],
                    sample_rate=sample_rate,
                    wmfcc_metric=wmfcc_metric,
                    clap_metric=clap_metric,
                    crepe_metric=crepe_metric,
                    weights=weights_cfg,
                    reward_sign=float(cfg.reward.sign),
                )
                for key in metrics_buffer:
                    metrics_buffer[key].append(float(metrics.get(key, float("nan"))))
                reward_values.append(float(metrics["reward"]))
            except Exception:
                for key in metrics_buffer:
                    metrics_buffer[key].append(float("nan"))
                reward_values.append(float(cfg.reward.failure))

        rewards_t = torch.tensor(reward_values, dtype=torch.float32, device=device).view(batch_size, group_size)
        mean_g = rewards_t.mean(dim=1, keepdim=True)
        std_g = rewards_t.std(dim=1, keepdim=True, unbiased=False)
        adv_flat = ((rewards_t - mean_g) / (std_g + 1e-6)).reshape(-1).detach()
        logp = logp.to(dtype=torch.float32)
        logp_ref = logp_ref.to(dtype=torch.float32).detach()
        kl = logp - logp_ref
        loss = -(adv_flat * logp).mean() + float(cfg.grpo.kl_coef) * kl.mean()
        loss.backward()

        clip = float(cfg.grpo.get("grad_clip_norm", 0.0) or 0.0)
        if clip > 0.0 and math.isfinite(clip):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step()

        eval_ran = False
        if eval_every > 0 and (step % eval_every == 0 or step == updates):
            (
                latest_val["wmfcc"],
                latest_val["clap"],
                latest_val["crepe"],
                latest_val["mss"],
                latest_val["sot"],
                latest_val["rms"],
                latest_val["reward"],
            ) = _run_val(step)
            eval_ran = True
            if math.isfinite(latest_val["reward"]):
                best_reward_ckpts = _update_best_reward_ckpts(
                    best_list=best_reward_ckpts,
                    new_reward=float(latest_val["reward"]),
                    step=step,
                    ckpt_dir=ckpt_dir,
                    model=model,
                    cfg=cfg,
                    dd_cfg=dd_cfg,
                    optimizer_state=opt.state_dict(),
                    rng_state=_capture_rng_state(),
                )

        row = {
            "step": int(step),
            "wmfcc_mean": _nanmean(metrics_buffer["wmfcc"]),
            "clap_mean": _nanmean(metrics_buffer["clap"]),
            "crepe_mean": _nanmean(metrics_buffer["crepe"]),
            "mss_mean": _nanmean(metrics_buffer["mss"]),
            "sot_mean": _nanmean(metrics_buffer["sot"]),
            "rms_mean": _nanmean(metrics_buffer["rms"]),
            "val_wmfcc_mean": float(latest_val["wmfcc"]),
            "val_clap_mean": float(latest_val["clap"]),
            "val_crepe_mean": float(latest_val["crepe"]),
            "val_mss_mean": float(latest_val["mss"]),
            "val_sot_mean": float(latest_val["sot"]),
            "val_rms_mean": float(latest_val["rms"]),
            "val_reward_mean": float(latest_val["reward"]),
            "reward_mean": float(rewards_t.mean().item()),
            "reward_std": float(rewards_t.std(unbiased=False).item()),
            "adv_std_mean": float(std_g.mean().item()),
            "logp_mean": float(logp.mean().item()),
            "kl_mean": float(kl.mean().item()),
            "loss": float(loss.detach().item()),
        }
        writer.writerow(row)
        csv_f.flush()

        if step == 1 or step % int(cfg.grpo.log_every) == 0:
            log.info(
                "step=%d loss=%.4f reward=%.3f wmfcc=%.3f clap=%.3f val_reward=%.3f kl=%.4f",
                step,
                row["loss"],
                row["reward_mean"],
                row["wmfcc_mean"],
                row["clap_mean"],
                row["val_reward_mean"],
                row["kl_mean"],
            )

        _log_metrics(
            {
                "train/wmfcc_mean": float(row["wmfcc_mean"]),
                "train/clap_mean": float(row["clap_mean"]),
                "train/crepe_mean": float(row["crepe_mean"]),
                "train/mss_mean": float(row["mss_mean"]),
                "train/sot_mean": float(row["sot_mean"]),
                "train/rms_mean": float(row["rms_mean"]),
                "train/reward_mean": float(row["reward_mean"]),
                "train/reward_std": float(row["reward_std"]),
                "train/loss": float(row["loss"]),
                "train/logp_mean": float(row["logp_mean"]),
                "train/kl_mean": float(row["kl_mean"]),
            },
            step=step,
        )
        if eval_ran:
            _log_metrics(
                {
                    "val/val_wmfcc_mean": float(row["val_wmfcc_mean"]),
                    "val/val_clap_mean": float(row["val_clap_mean"]),
                    "val/val_crepe_mean": float(row["val_crepe_mean"]),
                    "val/val_mss_mean": float(row["val_mss_mean"]),
                    "val/val_sot_mean": float(row["val_sot_mean"]),
                    "val/val_rms_mean": float(row["val_rms_mean"]),
                    "val/val_reward_mean": float(row["val_reward_mean"]),
                },
                step=step,
            )

        if save_last_every > 0 and (step % save_last_every == 0 or step == updates):
            _save_ckpt(
                ckpt_dir / "checkpoint_last.pt",
                model=model,
                step=step,
                cfg=cfg,
                dd_cfg=dd_cfg,
                optimizer_state=opt.state_dict(),
                rng_state=_capture_rng_state(),
            )

    csv_f.close()
    if hasattr(renderer, "close"):
        try:
            renderer.close()
        except Exception:
            pass
    for logger in loggers:
        try:
            logger.finalize("success")
        except Exception:
            pass


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
