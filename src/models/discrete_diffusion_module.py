from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import hydra
import lightning as L
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
import yaml
from omegaconf import DictConfig, OmegaConf

from src.data.synth_backends.dexed.dexed_bridge import DexedParameterHelper
from src.data.synth_backends.dexed.dexed_metadata import dexed_numerical_param_indices
from src.data.synth_backends.surge.surge_bridge import SurgeParameterHelper
from src.models.components.transformer_discrete_diffusion import DiscreteDiffusionParamTransformer
from src.models.synth_token_space import SynthTokenSpace
from src.project_paths import resolve_project_path
from src.validation.ood_audio_evaluator import OodAudioEvaluator
from src.validation.synth_render_evaluator import SynthRenderEvaluator


class SafeSequentialLR(lr_sched.SequentialLR):
    """SequentialLR that tolerates legacy checkpoints without `_schedulers`."""

    def load_state_dict(self, state_dict):
        if "_schedulers" not in state_dict:
            self._current_scheduler = len(self._schedulers) - 1
            self._schedulers[-1].load_state_dict(state_dict)
            self._last_epoch = state_dict.get("last_epoch", getattr(self, "last_epoch", -1))
            self._step_count = state_dict.get("_step_count", 0)
            self._last_lr = self._schedulers[-1].get_last_lr()
            for group, lr in zip(self.optimizer.param_groups, self._last_lr):
                group["lr"] = lr
            return
        super().load_state_dict(state_dict)


def _load_order(path: Path) -> List[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Token order file must contain a YAML list: {path}")
    return [str(name) for name in data if str(name).upper() != "EOS"]


def _load_helper_from_dataset_root(dataset_root: Path):
    meta_path = dataset_root / "dataset_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Frozen dataset metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    schema = meta.get("parameter_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("dataset_metadata.json must define `parameter_schema`.")
    synth = str(meta.get("synth", "")).lower()
    if synth == "surge_xt":
        synth = "surge"
    if synth == "dexed":
        return synth, DexedParameterHelper.from_schema(schema)
    if synth == "surge":
        return synth, SurgeParameterHelper.from_schema(schema)
    raise ValueError(f"Unsupported synth in dataset metadata: {meta.get('synth')!r}")


def _load_dataset_metadata(dataset_root: Path) -> Dict[str, Any]:
    meta_path = dataset_root / "dataset_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Frozen dataset metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"dataset_metadata.json must contain a JSON object: {meta_path}")
    return meta


class DiscreteDiffusionModule(L.LightningModule):
    """Stage-1 discrete diffusion training over a frozen synth token space."""

    def __init__(self, cfg: DictConfig, ckpt_dir: Path | None = None) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        model_cfg = cfg.model if "model" in cfg else cfg

        dataset_root = resolve_project_path(cfg.data.root)
        self.dataset_root = dataset_root
        self.dataset_metadata = _load_dataset_metadata(dataset_root)
        self.synth, self.helper = _load_helper_from_dataset_root(dataset_root)
        self.dataset_sample_rate = int(self.dataset_metadata["sample_rate"])
        self.dataset_target_duration = float(self.dataset_metadata["target_duration"])
        order_path_cfg = model_cfg.get("order_path") or cfg.data.get("token_order_path")
        order = (
            _load_order(resolve_project_path(order_path_cfg))
            if order_path_cfg
            else list(SynthTokenSpace.canonical_order_from_helper(self.helper, include_eos=False))
        )
        self.token_space = SynthTokenSpace.from_helper(self.helper, order=order)

        diffusion_cfg = getattr(model_cfg, "diffusion", {})
        diffusion_T_cfg = diffusion_cfg.get("T")
        self.diffusion_T = int(diffusion_T_cfg) if diffusion_T_cfg is not None else int(self.token_space.seq_len)
        self.masking_strategy = str(diffusion_cfg.get("masking", "bernoulli"))
        if self.masking_strategy not in {"bernoulli", "fixed_count"}:
            raise ValueError(
                "model.diffusion.masking must be one of {'bernoulli','fixed_count'}, "
                f"got {self.masking_strategy!r}"
            )

        self.model = DiscreteDiffusionParamTransformer(
            token_space=self.token_space,
            diffusion_T=self.diffusion_T,
            d_model=int(model_cfg.d_model),
            nhead=int(model_cfg.nhead),
            num_layers=int(model_cfg.num_layers),
            dim_feedforward=int(model_cfg.dim_feedforward),
            dropout=float(model_cfg.dropout),
            activation=str(model_cfg.activation),
            normalize_before=bool(model_cfg.normalize_before),
            gaussian_sigma=float(model_cfg.loss.get("gaussian_sigma", 0.02)),
            gaussian_smoothing_positions=self._gaussian_smoothing_positions(),
        )
        self.mask_token_id = int(self.model.mask_token_id)
        self.loss_cfg = model_cfg.loss
        self.validation_cfg = getattr(model_cfg, "validation", OmegaConf.create({}))
        self.renderer_cfg = getattr(model_cfg, "renderer", OmegaConf.create({}))
        self.log_per_param_loss = bool(self.validation_cfg.get("log_per_param_loss", False))
        self.dd_normalized_entropy = bool(self.validation_cfg.get("dd_normalized_entropy", False))
        self.dd_normalized_confidence = bool(self.validation_cfg.get("dd_normalized_confidence", False))
        self.metrics_cfg = self.validation_cfg.get("metrics", {}) if hasattr(self.validation_cfg, "get") else {}
        self.nsynth_eval_cfg = (
            self.validation_cfg.get("nsynth_eval", {}) if hasattr(self.validation_cfg, "get") else {}
        )
        self.nsynth_enable = bool(self.nsynth_eval_cfg.get("enable", False))
        self.render_batches = int(self.validation_cfg.get("render_batches", 0) or 0)
        self._enable_audio_metrics = bool(
            self.metrics_cfg.get("wmfcc")
            or self.metrics_cfg.get("mfcc13")
            or self.metrics_cfg.get("mfcc40")
            or self.metrics_cfg.get("mss")
            or self.metrics_cfg.get("sot")
            or self.metrics_cfg.get("rms")
            or self.metrics_cfg.get("clap")
        )
        if (self.render_batches > 0 or self.nsynth_enable) and not self._enable_audio_metrics:
            raise ValueError(
                "Render-backed validation requires at least one enabled audio metric "
                "under model.validation.metrics."
            )
        self.render_evaluator: Optional[SynthRenderEvaluator] = None
        if self.render_batches > 0 or self.nsynth_enable:
            self.render_evaluator = SynthRenderEvaluator(
                synth=self.synth,
                helper=self.helper,
                renderer_cfg=OmegaConf.to_container(self.renderer_cfg, resolve=True),
                metrics_cfg=OmegaConf.to_container(self.metrics_cfg, resolve=True),
                dataset_sample_rate=self.dataset_sample_rate,
                dataset_target_duration=self.dataset_target_duration,
            )
        self.ood_evaluator: Optional[OodAudioEvaluator] = None
        if self.nsynth_enable:
            if self.render_evaluator is None:
                raise RuntimeError("NSynth evaluation requires a render evaluator.")
            self.ood_evaluator = OodAudioEvaluator(
                dataset_root=self.dataset_root,
                dataset_metadata=self.dataset_metadata,
                eval_cfg=OmegaConf.to_container(self.nsynth_eval_cfg, resolve=True),
                batch_size=int(cfg.data.training.batch_size),
                render_evaluator=self.render_evaluator,
            )

        training_schedule = getattr(model_cfg, "training_schedule", OmegaConf.create({}))
        opt_cfg = getattr(model_cfg, "optimizer", OmegaConf.create({}))
        sched_cfg = getattr(model_cfg, "scheduler", OmegaConf.create({}))
        self.train_cfg = OmegaConf.create(
            {
                "max_steps": training_schedule.get("max_steps", 0),
                "weight_decay": opt_cfg.get("weight_decay", 0.0),
                "lr": opt_cfg.get("lr", 0.0),
                "optimizer": opt_cfg,
                "scheduler": sched_cfg,
            }
        )
        self._val_token_loss_sum: Dict[str, float] = {}
        self._val_token_loss_count: Dict[str, int] = {}
        self._val_greedy_token_loss_sum: Dict[str, float] = {}
        self._val_greedy_token_loss_count: Dict[str, int] = {}
        self._val_render_count = 0

        if ckpt_dir is None:
            ckpt_dir = resolve_project_path(cfg.paths.output_dir) / "checkpoints"
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._init_token_loss_weights()

    def _gaussian_smoothing_positions(self) -> Sequence[int]:
        positions: List[int] = []
        dexed_numeric = set(dexed_numerical_param_indices()) if self.synth == "dexed" else set()
        for idx, field in enumerate(self.token_space.fields):
            if field.is_midi or field.is_special or field.cardinality <= 1:
                continue
            if field.mode == "num":
                positions.append(idx)
                continue
            if self.synth == "dexed" and field.full_index is not None and field.full_index in dexed_numeric:
                positions.append(idx)
        return positions

    def _init_token_loss_weights(self) -> None:
        weight_cfg = self.cfg.get("weight") if hasattr(self.cfg, "get") else None
        order = list(self.token_space.order)
        name_to_idx = {name.upper(): i for i, name in enumerate(order)}

        default_weight = 1.0
        validate = True
        weights_map = {}
        if weight_cfg is not None:
            default_weight = float(weight_cfg.get("default", 1.0))
            validate = bool(weight_cfg.get("validate", True))
            raw_weights = weight_cfg.get("weights") or {}
            if raw_weights is None:
                weights_map = {}
            elif OmegaConf.is_config(raw_weights):
                weights_map = OmegaConf.to_container(raw_weights, resolve=True)
            elif isinstance(raw_weights, Mapping):
                weights_map = dict(raw_weights)
            else:
                raise TypeError(f"weight.weights must be a mapping, got {type(raw_weights).__name__}")

        if default_weight < 0.0:
            raise ValueError(f"weight.default must be non-negative, got {default_weight!r}")

        weights_list: List[float] = [default_weight for _ in order]
        unknown_keys: List[str] = []
        if isinstance(weights_map, dict):
            for raw_key, raw_value in weights_map.items():
                if raw_value is None:
                    continue
                key = str(raw_key)
                weight = float(raw_value)
                if weight < 0.0:
                    raise ValueError(f"Weight for token '{key}' must be non-negative, got {weight!r}")
                key_upper = key.upper()
                if key_upper == "EOS":
                    continue
                if key_upper in name_to_idx:
                    weights_list[name_to_idx[key_upper]] = weight
                else:
                    unknown_keys.append(key)

        if validate and unknown_keys:
            unknown = ", ".join(sorted(set(unknown_keys))[:25])
            raise ValueError(
                "Unknown keys in weight config. Only tokens in the active token order can be weighted. "
                f"Unknown: {unknown}"
            )

        self.register_buffer("token_loss_weights", torch.tensor(weights_list, dtype=torch.float32), persistent=False)

    def build_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.token_space.build_token_targets_from_batch(batch).to(batch["mel_spec"].device)

    def decode_midi(self, classes: torch.Tensor) -> torch.Tensor:
        cfg = self.helper.midi_cfg
        note = int(cfg.note_min) + classes[:, 0]
        velocity = classes[:, 1]
        duration = float(cfg.duration_min) + (
            classes[:, 2].float() / max(int(cfg.duration_classes) - 1, 1)
        ) * (float(cfg.duration_max) - float(cfg.duration_min))
        return torch.stack((note.float(), velocity.float(), duration.float()), dim=1)

    def _tokens_to_full_and_midi(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        full_np, midi_np = self.token_space.token_ids_to_full_and_midi(token_ids.detach().cpu().numpy())
        return (
            torch.from_numpy(full_np).float(),
            torch.from_numpy(midi_np).float(),
        )

    @torch.no_grad()
    def _predict_full_and_midi_from_mel(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = self._diffusion_decode(mel_spec)
        return self._tokens_to_full_and_midi(token_ids)

    def _make_noisy_tokens(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0.ndim != 2:
            raise ValueError(f"x0 must have shape (B, L), got {tuple(x0.shape)}")
        B, L = x0.shape
        if self.masking_strategy == "bernoulli":
            if self.diffusion_T <= 1:
                p_mask = torch.zeros_like(t, dtype=torch.float32)
            else:
                p_mask = (t.float() - 1.0) / float(self.diffusion_T - 1)
            mask = torch.rand((B, L), device=x0.device) < p_mask.unsqueeze(1)
        else:
            max_maskable = max(L - 1, 0)
            if self.diffusion_T <= 1:
                k = torch.zeros_like(t, dtype=torch.long)
            else:
                frac = (t.float() - 1.0) / float(self.diffusion_T - 1)
                k = torch.round(frac * float(max_maskable)).to(dtype=torch.long)
            k = torch.clamp(k, 0, max_maskable)
            max_k = int(k.max().item()) if B > 0 else 0
            if max_k <= 0:
                mask = torch.zeros((B, L), dtype=torch.bool, device=x0.device)
            else:
                perm = torch.rand((B, L), device=x0.device).argsort(dim=-1)
                prefix = perm[:, :max_k]
                take = torch.arange(max_k, device=x0.device).unsqueeze(0) < k.unsqueeze(1)
                rows = torch.arange(B, device=x0.device).unsqueeze(1).expand(B, max_k)[take]
                cols = prefix[take]
                mask = torch.zeros((B, L), dtype=torch.bool, device=x0.device)
                mask[rows, cols] = True
        x_t = x0.clone()
        x_t[mask] = int(self.mask_token_id)
        return x_t, mask

    def forward(self, mel_spec: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(mel_spec, x_t, t)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        x0 = self.build_targets(batch)
        B = x0.shape[0]
        t = torch.randint(1, self.diffusion_T + 1, (B,), device=x0.device, dtype=torch.long)
        x_t, mask = self._make_noisy_tokens(x0, t)
        logits = self.model(batch["mel_spec"], x_t, t)
        loss, param_loss, midi_loss = self.model.loss(
            logits,
            x0,
            mask,
            temperature=float(self.loss_cfg.get("temperature", 1.0)),
            label_smoothing=float(self.loss_cfg.get("label_smoothing", 0.0)),
            midi_label_smoothing=float(self.loss_cfg.get("midi_label_smoothing", 0.0)),
            midi_label_smoothing_apply=self.loss_cfg.get("midi_label_smoothing_apply"),
            token_weights=getattr(self, "token_loss_weights", None),
            uncertainty_weighting=bool(self.loss_cfg.get("uncertainty_weighting", False)),
            return_components=True,
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        self.log("train/param_loss", param_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("train/midi_loss", midi_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        if self.trainer is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, prog_bar=False, logger=True, batch_size=B)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._shared_eval_step(batch, stage="test")

    def _shared_eval_step(self, batch: Dict[str, torch.Tensor], *, stage: str):
        x0 = self.build_targets(batch)
        B = x0.shape[0]
        t = torch.randint(1, self.diffusion_T + 1, (B,), device=x0.device, dtype=torch.long)
        x_t, mask = self._make_noisy_tokens(x0, t)
        logits = self.model(batch["mel_spec"], x_t, t)
        outputs = self.model.loss(
            logits,
            x0,
            mask,
            temperature=float(self.loss_cfg.get("temperature", 1.0)),
            label_smoothing=float(self.loss_cfg.get("label_smoothing", 0.0)),
            midi_label_smoothing=float(self.loss_cfg.get("midi_label_smoothing", 0.0)),
            midi_label_smoothing_apply=self.loss_cfg.get("midi_label_smoothing_apply"),
            token_weights=getattr(self, "token_loss_weights", None),
            uncertainty_weighting=bool(self.loss_cfg.get("uncertainty_weighting", False)),
            return_components=True,
            return_midi_per_head=True,
            return_per_param_loss=self.log_per_param_loss and stage == "val",
        )

        token_loss_stats = None
        if self.log_per_param_loss and stage == "val":
            loss, param_loss, midi_loss, midi_head_losses, token_loss_stats = outputs
        else:
            loss, param_loss, midi_loss, midi_head_losses = outputs

        self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"), on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/param_loss", param_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/midi_loss", midi_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
        if midi_head_losses is not None and getattr(midi_head_losses, "numel", lambda: 0)() >= 3:
            self.log(f"{stage}/midi_note_loss", midi_head_losses[0], on_step=False, on_epoch=True, batch_size=B)
            self.log(f"{stage}/midi_velocity_loss", midi_head_losses[1], on_step=False, on_epoch=True, batch_size=B)
            self.log(f"{stage}/midi_duration_loss", midi_head_losses[2], on_step=False, on_epoch=True, batch_size=B)

        if self.log_per_param_loss and stage == "val" and token_loss_stats:
            for token_name, token_loss, token_count in token_loss_stats:
                if token_count <= 0 or not torch.isfinite(token_loss):
                    continue
                name = str(token_name)
                self._val_token_loss_sum[name] = self._val_token_loss_sum.get(name, 0.0) + float(
                    token_loss.detach().cpu().item()
                ) * int(token_count)
                self._val_token_loss_count[name] = self._val_token_loss_count.get(name, 0) + int(token_count)

        if stage == "val":
            self._maybe_log_render_validation(batch, x0)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_render_count = 0
        self._val_token_loss_sum.clear()
        self._val_token_loss_count.clear()
        self._val_greedy_token_loss_sum.clear()
        self._val_greedy_token_loss_count.clear()

    @torch.no_grad()
    def _maybe_log_render_validation(self, batch: Dict[str, torch.Tensor], x0: torch.Tensor) -> None:
        if self.render_evaluator is None or self._val_render_count >= self.render_batches:
            return
        if "audio" not in batch or batch["audio"] is None:
            raise ValueError(
                "Render validation requires audio targets in validation batches. "
                "Set data.val_read_audio=true for the active dataset config."
            )

        mel_spec = batch["mel_spec"]
        token_ids, greedy_logits = self._diffusion_decode(mel_spec, return_logits=True)
        greedy_mask = torch.ones_like(x0, dtype=torch.bool, device=x0.device)
        greedy_outputs = self.model.loss(
            greedy_logits,
            x0,
            greedy_mask,
            temperature=float(self.loss_cfg.get("temperature", 1.0)),
            label_smoothing=float(self.loss_cfg.get("label_smoothing", 0.0)),
            midi_label_smoothing=float(self.loss_cfg.get("midi_label_smoothing", 0.0)),
            midi_label_smoothing_apply=self.loss_cfg.get("midi_label_smoothing_apply"),
            token_weights=getattr(self, "token_loss_weights", None),
            uncertainty_weighting=bool(self.loss_cfg.get("uncertainty_weighting", False)),
            return_components=True,
            return_per_param_loss=self.log_per_param_loss,
        )

        greedy_token_loss_stats = None
        if self.log_per_param_loss:
            greedy_loss, greedy_param_loss, greedy_midi_loss, greedy_token_loss_stats = greedy_outputs
        else:
            greedy_loss, greedy_param_loss, greedy_midi_loss = greedy_outputs

        batch_size = x0.shape[0]
        self.log("val_greedy/loss", greedy_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(
            "val_greedy/param_loss",
            greedy_param_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_greedy/midi_loss",
            greedy_midi_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        if self.log_per_param_loss and greedy_token_loss_stats:
            for token_name, token_loss, token_count in greedy_token_loss_stats:
                if token_count <= 0 or not torch.isfinite(token_loss):
                    continue
                name = str(token_name)
                self._val_greedy_token_loss_sum[name] = self._val_greedy_token_loss_sum.get(name, 0.0) + float(
                    token_loss.detach().cpu().item()
                ) * int(token_count)
                self._val_greedy_token_loss_count[name] = self._val_greedy_token_loss_count.get(name, 0) + int(
                    token_count
                )

        full_pred, midi_pred = self._tokens_to_full_and_midi(token_ids)
        try:
            pred_audio, kept = self.render_evaluator.render_from_full_and_midi_resilient(full_pred, midi_pred)
        except Exception:
            self._val_render_count += 1
            return
        target_audio = batch["audio"]
        if kept.size != batch_size:
            target_audio = target_audio[torch.as_tensor(kept, device=target_audio.device, dtype=torch.long)]
        metrics = self.render_evaluator.compute_metrics(pred_audio, target_audio)
        for metric_name, metric_value in sorted(metrics.items()):
            self.log(
                f"val/{metric_name}",
                float(metric_value),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
        self._val_render_count += 1

    @torch.no_grad()
    def _diffusion_decode(self, mel_spec: torch.Tensor, *, return_logits: bool = False):
        if mel_spec.ndim != 4:
            raise ValueError(f"Expected mel_spec shape (B,1,F,T), got {tuple(mel_spec.shape)}")
        device = next(self.model.parameters()).device
        spec = mel_spec.to(device=device)
        batch = spec.shape[0]

        L = int(self.model.seq_len)
        T = int(self.diffusion_T)
        x = torch.full((batch, L), int(self.mask_token_id), dtype=torch.long, device=device)
        locked = torch.zeros((batch, L), dtype=torch.bool, device=device)
        greedy_logits = None
        if return_logits:
            greedy_logits = torch.zeros((batch, L, self.model.max_card), dtype=torch.float32, device=device)
            greedy_logits = greedy_logits.to(dtype=next(self.model.parameters()).dtype)

        if self.dd_normalized_entropy or self.dd_normalized_confidence:
            cardinals = torch.as_tensor(self.model.cardinals, device=device, dtype=torch.float32)
            cardinals = torch.clamp(cardinals, min=1.0)
            if self.dd_normalized_confidence and not self.dd_normalized_entropy:
                baseline = 1.0 / cardinals
                denom = 1.0 - baseline
            if self.dd_normalized_entropy:
                log_card = torch.log(cardinals)

        self.model.eval()
        for step in range(L):
            t = torch.full((batch,), T - step, dtype=torch.long, device=device)
            logits = self.model(spec, x, t)
            probs = F.softmax(logits, dim=-1)

            if self.dd_normalized_entropy:
                p = probs.to(dtype=torch.float32)
                entropy = -(p * torch.log(torch.clamp(p, min=1e-12))).sum(dim=-1)
                denom_ent = log_card.to(dtype=entropy.dtype).unsqueeze(0)
                safe = denom_ent > 0
                u = torch.zeros_like(entropy)
                u = torch.where(safe, entropy / torch.clamp(denom_ent, min=1e-12), u)
                confidence = (1.0 - u).clamp(0.0, 1.0).to(dtype=probs.dtype)
            else:
                p_max = probs.max(dim=-1).values
                if self.dd_normalized_confidence:
                    p_max = p_max.to(dtype=baseline.dtype)
                    confidence = (p_max - baseline) / denom
                    confidence = torch.where(denom > 0, confidence, torch.zeros_like(confidence))
                else:
                    confidence = p_max
            confidence = confidence.masked_fill(locked, -1e9)

            j = torch.argmax(confidence, dim=-1)
            batch_idx = torch.arange(batch, device=device)
            token_id = torch.argmax(probs[batch_idx, j], dim=-1)
            x[batch_idx, j] = token_id
            locked[batch_idx, j] = True
            if greedy_logits is not None:
                greedy_logits[batch_idx, j] = logits[batch_idx, j]

        if return_logits:
            return x, greedy_logits
        return x

    def on_validation_epoch_end(self) -> None:
        if self.log_per_param_loss:
            for token_name, total_loss in sorted(self._val_token_loss_sum.items()):
                count = self._val_token_loss_count.get(token_name, 0)
                if count <= 0:
                    continue
                mean_loss = total_loss / float(count)
                self.log(
                    f"val/token_loss/{token_name}",
                    mean_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            for token_name, total_loss in sorted(self._val_greedy_token_loss_sum.items()):
                count = self._val_greedy_token_loss_count.get(token_name, 0)
                if count <= 0:
                    continue
                mean_loss = total_loss / float(count)
                self.log(
                    f"val_greedy/token_loss/{token_name}",
                    mean_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
        if self.ood_evaluator is not None:
            device = next(self.model.parameters()).device
            nsynth_metrics = self.ood_evaluator.run(
                predict_from_mel=self._predict_full_and_midi_from_mel,
                device=device,
            )
            for metric_name, metric_value in sorted(nsynth_metrics.items()):
                self.log(
                    metric_name,
                    float(metric_value),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
        self._val_token_loss_sum.clear()
        self._val_token_loss_count.clear()
        self._val_greedy_token_loss_sum.clear()
        self._val_greedy_token_loss_count.clear()

    def teardown(self, stage: str | None) -> None:
        if self.render_evaluator is not None:
            self.render_evaluator.close()
            self.render_evaluator = None

    def configure_optimizers(self):
        opt_cfg = getattr(self.cfg.model, "optimizer", None)
        if opt_cfg is None and hasattr(self.train_cfg, "optimizer"):
            opt_cfg = self.train_cfg.optimizer
        if opt_cfg is not None:
            if opt_cfg.get("_partial_", False):
                opt_partial = hydra.utils.instantiate(opt_cfg)
                opt = opt_partial(self.parameters())
            else:
                opt = hydra.utils.instantiate(opt_cfg, self.parameters())
        else:
            opt = torch.optim.Adam(
                self.parameters(),
                lr=float(self.train_cfg.get("lr", 0.0)),
                weight_decay=float(self.train_cfg.get("weight_decay", 0.0)),
            )

        sched_cfg = getattr(self.cfg.model, "scheduler", None)
        if sched_cfg is None and hasattr(self.train_cfg, "scheduler"):
            sched_cfg = self.train_cfg.scheduler

        schedulers: list[dict] = []
        if sched_cfg and sched_cfg.get("type") == "cosine_warmup":
            warmup_steps = int(sched_cfg.get("warmup_steps", 0) or 0)
            start_factor = float(sched_cfg.get("start_factor", 0.1))
            eta_min = float(sched_cfg.get("eta_min", 0.0))
            warmup = lr_sched.LinearLR(opt, start_factor=start_factor, total_iters=warmup_steps)
            cosine = lr_sched.CosineAnnealingLR(
                opt,
                T_max=max(int(self.train_cfg.max_steps) - warmup_steps, 1),
                eta_min=eta_min,
            )
            sched = SafeSequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
            schedulers.append({"scheduler": sched, "interval": "step"})
        elif sched_cfg and sched_cfg.get("type") == "cosine":
            schedulers.append(
                {
                    "scheduler": lr_sched.CosineAnnealingLR(
                        opt,
                        T_max=max(int(self.train_cfg.max_steps), 1),
                        eta_min=float(sched_cfg.get("eta_min", 0.0)),
                    ),
                    "interval": "step",
                }
            )

        if schedulers:
            return [opt], schedulers
        return opt
