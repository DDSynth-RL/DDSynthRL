from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import hydra
import lightning as L
import torch
import torch.optim.lr_scheduler as lr_sched
import yaml
from omegaconf import DictConfig, OmegaConf

from src.data.synth_backends.dexed.dexed_bridge import DexedParameterHelper
from src.data.synth_backends.surge.surge_bridge import SurgeParameterHelper
from src.models.components.transformer_autoregressive import AutoregressiveParamTransformer
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
    return [str(name) for name in data]


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


class AutoregressiveModule(L.LightningModule):
    """Stage-1 autoregressive training over a frozen synth token space."""

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
            else list(SynthTokenSpace.canonical_order_from_helper(self.helper, include_eos=True))
        )
        self.token_space = SynthTokenSpace.from_helper(self.helper, order=order)
        self.model = AutoregressiveParamTransformer(
            token_space=self.token_space,
            synth=self.synth,
            d_model=int(model_cfg.d_model),
            nhead=int(model_cfg.nhead),
            num_layers=int(model_cfg.num_layers),
            dim_feedforward=int(model_cfg.dim_feedforward),
            dropout=float(model_cfg.dropout),
            activation=str(model_cfg.activation),
            normalize_before=bool(model_cfg.normalize_before),
            gaussian_sigma=float(model_cfg.loss.get("gaussian_sigma", 0.02)),
        )

        self.loss_cfg = model_cfg.loss
        self.validation_cfg = getattr(model_cfg, "validation", OmegaConf.create({}))
        self.renderer_cfg = getattr(model_cfg, "renderer", OmegaConf.create({}))
        self.log_per_param_loss = bool(self.validation_cfg.get("log_per_param_loss", False))
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
        return torch.from_numpy(full_np).float(), torch.from_numpy(midi_np).float()

    @torch.no_grad()
    def _predict_full_and_midi_from_mel(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = self._greedy_decode(mel_spec)
        return self._tokens_to_full_and_midi(token_ids)

    def forward(self, mel_spec: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(mel_spec, token_ids)

    @torch.no_grad()
    def _greedy_decode(self, mel_spec: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        spec = mel_spec.to(device=device)
        batch = spec.shape[0]
        seq_len = self.model.seq_len
        token_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
        self.model.eval()
        for t in range(seq_len):
            logits = self.model(spec, token_ids)
            cardinal = self.model.cardinals[t]
            if cardinal <= 1:
                token_ids[:, t] = 0
                continue
            token_ids[:, t] = torch.argmax(logits[:, t, :cardinal], dim=-1)
        return token_ids

    @torch.no_grad()
    def _greedy_decode_with_logits(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        spec = mel_spec.to(device=device)
        batch = spec.shape[0]
        seq_len = self.model.seq_len
        token_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
        greedy_logits = torch.zeros((batch, seq_len, self.model.max_card), dtype=torch.float32, device=device)
        greedy_logits = greedy_logits.to(dtype=next(self.model.parameters()).dtype)
        self.model.eval()
        for t in range(seq_len):
            logits = self.model(spec, token_ids)
            greedy_logits[:, t, :] = logits[:, t, :]
            cardinal = self.model.cardinals[t]
            if cardinal <= 1:
                token_ids[:, t] = 0
                continue
            token_ids[:, t] = torch.argmax(logits[:, t, :cardinal], dim=-1)
        return token_ids, greedy_logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        targets = self.build_targets(batch)
        logits = self.model(batch["mel_spec"], targets)
        loss, param_loss, midi_loss = self.model.loss(
            logits,
            targets,
            temperature=float(self.loss_cfg.get("temperature", 1.0)),
            label_smoothing=float(self.loss_cfg.get("label_smoothing", 0.0)),
            midi_label_smoothing=float(self.loss_cfg.get("midi_label_smoothing", 0.0)),
            midi_label_smoothing_apply=self.loss_cfg.get("midi_label_smoothing_apply"),
            token_weights=getattr(self, "token_loss_weights", None),
            uncertainty_weighting=bool(self.loss_cfg.get("uncertainty_weighting", False)),
            return_components=True,
        )
        batch_size = targets.shape[0]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/param_loss", param_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/midi_loss", midi_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size)
        if self.trainer is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, prog_bar=False, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        targets = self.build_targets(batch)
        logits = self.model(batch["mel_spec"], targets)
        outputs = self.model.loss(
            logits,
            targets,
            temperature=float(self.loss_cfg.get("temperature", 1.0)),
            label_smoothing=float(self.loss_cfg.get("label_smoothing", 0.0)),
            midi_label_smoothing=float(self.loss_cfg.get("midi_label_smoothing", 0.0)),
            midi_label_smoothing_apply=self.loss_cfg.get("midi_label_smoothing_apply"),
            token_weights=getattr(self, "token_loss_weights", None),
            uncertainty_weighting=bool(self.loss_cfg.get("uncertainty_weighting", False)),
            return_components=True,
            return_midi_per_head=True,
            return_per_param_loss=self.log_per_param_loss,
        )
        token_loss_stats = None
        if self.log_per_param_loss:
            loss, param_loss, midi_loss, midi_head_losses, token_loss_stats = outputs
        else:
            loss, param_loss, midi_loss, midi_head_losses = outputs

        batch_size = targets.shape[0]
        if self.metrics_cfg.get("loss", True):
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.metrics_cfg.get("param_loss", True):
            self.log("val/param_loss", param_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.metrics_cfg.get("midi_loss", True):
            self.log("val/midi_loss", midi_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        if midi_head_losses is not None and midi_head_losses.numel() >= 3:
            if self.metrics_cfg.get("midi_note_loss", False):
                self.log("val/midi_note_loss", midi_head_losses[0], on_step=False, on_epoch=True, batch_size=batch_size)
            if self.metrics_cfg.get("midi_velocity_loss", False):
                self.log(
                    "val/midi_velocity_loss",
                    midi_head_losses[1],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            if self.metrics_cfg.get("midi_duration_loss", False):
                self.log(
                    "val/midi_duration_loss",
                    midi_head_losses[2],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )

        if self.log_per_param_loss and token_loss_stats:
            for token_name, token_loss, token_count in token_loss_stats:
                if token_count <= 0 or not torch.isfinite(token_loss):
                    continue
                name = str(token_name)
                self._val_token_loss_sum[name] = self._val_token_loss_sum.get(name, 0.0) + float(
                    token_loss.detach().cpu().item()
                ) * int(token_count)
                self._val_token_loss_count[name] = self._val_token_loss_count.get(name, 0) + int(token_count)

        if self._enable_audio_metrics:
            self._maybe_log_render_validation(batch, targets)

    def on_validation_epoch_start(self) -> None:
        self._val_render_count = 0

    def _maybe_log_render_validation(self, batch: Dict[str, torch.Tensor], targets: torch.Tensor) -> None:
        if self.render_evaluator is None or self._val_render_count >= self.render_batches:
            return
        if "audio" not in batch or batch["audio"] is None:
            raise ValueError(
                "Render validation requires audio targets in validation batches. "
                "Set data.val_read_audio=true for the active dataset config."
            )

        token_ids, greedy_logits = self._greedy_decode_with_logits(batch["mel_spec"])
        greedy_outputs = self.model.loss(
            greedy_logits,
            targets,
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

        batch_size = targets.shape[0]
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

    def on_validation_epoch_end(self) -> None:
        if self.log_per_param_loss:
            for token_name, total_loss in sorted(self._val_token_loss_sum.items()):
                count = self._val_token_loss_count.get(token_name, 0)
                if count <= 0:
                    continue
                self.log(
                    f"val/token_loss/{token_name}",
                    total_loss / float(count),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            for token_name, total_loss in sorted(self._val_greedy_token_loss_sum.items()):
                count = self._val_greedy_token_loss_count.get(token_name, 0)
                if count <= 0:
                    continue
                self.log(
                    f"val_greedy/token_loss/{token_name}",
                    total_loss / float(count),
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
