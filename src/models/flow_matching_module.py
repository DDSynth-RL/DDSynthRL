from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import hydra
import lightning as L
import torch
import torch.optim.lr_scheduler as lr_sched
from omegaconf import DictConfig, OmegaConf

from src.data.synth_backends.dexed.dexed_bridge import DexedParameterHelper
from src.data.synth_backends.surge.surge_bridge import SurgeParameterHelper
from src.project_paths import resolve_project_path
from src.utils.dexed_denoise_space import DexedDenoiseSpace
from src.utils.surge_denoise_space import SurgeDenoiseSpace
from src.validation.ood_audio_evaluator import OodAudioEvaluator
from src.validation.synth_render_evaluator import SynthRenderEvaluator


def _late_curve(x: torch.Tensor, a: float) -> torch.Tensor:
    if a == 0.0:
        return x
    return (1 - torch.exp(-a * x)) / (1 - math.exp(-a))


def _cosine_curve(x: torch.Tensor) -> torch.Tensor:
    return 0.5 + 0.5 * torch.cos(torch.pi * (1 + x))


def call_with_cfg(
    f,
    x: torch.Tensor,
    t: torch.Tensor,
    conditioning: torch.Tensor,
    cfg_strength: float,
) -> torch.Tensor:
    y_c = f(x, t, conditioning)
    y_u = f(x, t, None)
    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_with_cfg(
    f,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    conditioning: torch.Tensor,
    cfg_strength: float,
) -> torch.Tensor:
    step = partial(call_with_cfg, f, conditioning=conditioning, cfg_strength=cfg_strength)
    k1 = step(x, t)
    k2 = step(x + dt * k1 / 2, t + dt / 2)
    k3 = step(x + dt * k2 / 2, t + dt / 2)
    k4 = step(x + dt * k3, t + dt)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


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


def _load_dataset_metadata(dataset_root: Path) -> Dict[str, Any]:
    meta_path = dataset_root / "dataset_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Frozen dataset metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"dataset_metadata.json must contain a JSON object: {meta_path}")
    return meta


def _load_helper_from_dataset_root(dataset_root: Path):
    meta = _load_dataset_metadata(dataset_root)
    synth = str(meta.get("synth", "")).lower()
    if synth == "surge_xt":
        synth = "surge"
    schema = meta.get("parameter_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("dataset_metadata.json must define `parameter_schema`.")
    if synth == "dexed":
        return synth, DexedParameterHelper.from_schema(schema)
    if synth == "surge":
        return synth, SurgeParameterHelper.from_schema(schema)
    raise ValueError(f"Flow-matching currently supports Dexed and Surge, found synth={meta.get('synth')!r}")


class FlowMatchingModule(L.LightningModule):
    """Stage-1 flow-matching training over synth-specific continuous denoise spaces."""

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

        encoder = hydra.utils.instantiate(model_cfg.encoder)
        vector_field = hydra.utils.instantiate(model_cfg.vector_field)
        denoise_space_cfg = getattr(model_cfg, "denoise_space", None)
        denoise_space = (
            cast(DexedDenoiseSpace | SurgeDenoiseSpace, hydra.utils.instantiate(denoise_space_cfg))
            if denoise_space_cfg is not None
            else (
                DexedDenoiseSpace()
                if self.synth == "dexed"
                else SurgeDenoiseSpace.from_schema(self.dataset_metadata["parameter_schema"])
            )
        )

        self.encoder = cast(torch.nn.Module, encoder)
        self.vector_field = cast(torch.nn.Module, vector_field)
        self.denoise_space = denoise_space

        projection = getattr(self.vector_field, "projection", None)
        assignment = getattr(projection, "assignment", None)
        if assignment is not None and hasattr(assignment, "shape"):
            expected_dim = int(assignment.shape[1])
            if int(self.denoise_space.total_dim) != expected_dim:
                raise ValueError(
                    f"Denoise space dim {self.denoise_space.total_dim} != "
                    f"vector_field.projection.num_params {expected_dim}."
                )

        midi_start = getattr(self.denoise_space, "midi_start", None)
        midi_dim = int(getattr(self.denoise_space, "midi_dim", 0))
        if midi_start is None:
            raise ValueError(
                f"{type(self.denoise_space).__name__} must expose `midi_start` for shared flow-matching."
            )
        midi_start = int(midi_start)
        self.param_indices = list(range(midi_start))
        self.midi_indices = list(range(midi_start, midi_start + midi_dim))

        self.p_time = str(model_cfg.get("p_time", "uniform"))
        self.w_time = str(model_cfg.get("w_time", "none"))
        self.coupling = str(model_cfg.get("coupling", "uniform"))
        self.oversample_ot = float(model_cfg.get("oversample_ot", 1.0))
        self.rectified_sigma_min = float(model_cfg.get("rectified_sigma_min", 0.0))
        self.sample_schedule = str(model_cfg.get("sample_schedule", "linear"))
        self.late_sample_schedule_curve = float(model_cfg.get("late_sample_schedule_curve", 2.0))
        self.cfg_dropout_rate = float(model_cfg.get("cfg_dropout_rate", 0.1))
        self.validation_sample_steps = int(model_cfg.get("validation_sample_steps", 50))
        self.validation_cfg_strength = float(model_cfg.get("validation_cfg_strength", 2.0))
        self.test_sample_steps = int(model_cfg.get("test_sample_steps", 200))
        self.test_cfg_strength = float(model_cfg.get("test_cfg_strength", 2.0))
        self.uniform_loss_weight = bool(model_cfg.get("uniform_loss_weight", True))
        self.warmup_steps = int(model_cfg.get("warmup_steps", 0))
        self.compile_flag = bool(model_cfg.get("compile", False))

        self.optimizer_cfg = getattr(model_cfg, "optimizer", OmegaConf.create({}))
        self.scheduler_cfg = getattr(model_cfg, "scheduler", None)
        self.training_schedule_cfg = getattr(model_cfg, "training_schedule", OmegaConf.create({}))

        self.validation_cfg = getattr(model_cfg, "validation", OmegaConf.create({}))
        self.metrics_cfg = (
            self.validation_cfg.get("metrics", {}) if hasattr(self.validation_cfg, "get") else {}
        )
        self.nsynth_eval_cfg = (
            self.validation_cfg.get("nsynth_eval", {}) if hasattr(self.validation_cfg, "get") else {}
        )
        self.render_batches = int(self.validation_cfg.get("render_batches", 0) or 0)
        self.nsynth_enable = bool(self.nsynth_eval_cfg.get("enable", False))
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

        self.renderer_cfg = getattr(model_cfg, "renderer", OmegaConf.create({}))
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

        self._val_render_count = 0
        self.note_min = int(self.helper.midi_cfg.note_min)
        self.note_classes = int(self.helper.midi_cfg.note_classes)
        self.velocity_classes = int(self.helper.midi_cfg.velocity_classes)
        self.duration_min = float(self.helper.midi_cfg.duration_min)
        self.duration_max = float(self.helper.midi_cfg.duration_max)
        self.full_param_len = len(self.helper.preset_helper.vst_param_names)

        if ckpt_dir is None:
            ckpt_dir = resolve_project_path(cfg.paths.output_dir) / "checkpoints"
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_start(self) -> None:
        self._val_render_count = 0

    def setup(self, stage: str) -> None:
        if self.compile_flag and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.vector_field = torch.compile(self.vector_field)

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        if self.p_time == "uniform":
            return torch.rand(n, 1, device=device)
        if self.p_time == "bias_later":
            return _late_curve(torch.rand(n, 1, device=device), 1.0)
        if self.p_time == "lognormal":
            return torch.randn(n, 1, device=device).sigmoid()
        if self.p_time == "beta":
            dist = torch.distributions.Beta(2.5, 1.0)
            return dist.sample((n, 1)).to(device)
        if self.p_time == "extreme_beta":
            dist = torch.distributions.Beta(10.0, 1.5)
            return dist.sample((n, 1)).to(device)
        raise ValueError(f"Unknown p_time {self.p_time!r}")

    def _weight_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.w_time == "none":
            return torch.ones_like(t)
        half_snr = torch.log(t / (1 - t))
        weighting = torch.exp(-half_snr) + 1
        if self.w_time == "flatten":
            return weighting.pow(-2)
        if self.w_time == "reverse":
            return weighting.pow(-4)
        raise ValueError(f"Unknown w_time {self.w_time!r}")

    def _basic_sample(self, params: torch.Tensor, oversample: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if oversample == 1.0:
            x0 = torch.randn_like(params)
        elif oversample < 1.0:
            raise ValueError(f"oversample must be >= 1.0, got {oversample}")
        else:
            n = int(oversample * params.shape[0])
            x0 = torch.randn(n, *params.shape[1:], device=params.device)
        return x0, params

    @torch.no_grad()
    def _ot_sample(
        self,
        params: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from scipy.optimize import linear_sum_assignment

        x0, x1 = self._basic_sample(params, oversample=self.oversample_ot)
        costs = torch.cdist(x0, x1)
        row_ind, col_ind = linear_sum_assignment(costs.detach().cpu().numpy())
        row_ind = torch.from_numpy(row_ind).to(params.device)
        col_ind = torch.from_numpy(col_ind).to(params.device)
        x0 = x0[row_ind]
        x1 = x1[col_ind]
        conditioning = conditioning[col_ind]
        return x0, x1, conditioning

    def _sample_x0_and_x1(
        self,
        params: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.coupling == "uniform":
            x0, x1 = self._basic_sample(params)
            return x0, x1, conditioning
        if self.coupling == "ot":
            return self._ot_sample(params, conditioning)
        raise ValueError(f"Unknown coupling {self.coupling!r}")

    def _rectified_probability_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return x0 * (1 - t) * (1 - self.rectified_sigma_min) + x1 * t

    @staticmethod
    def _rectified_vector_field(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def _warp_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.sample_schedule == "linear":
            return t
        if self.sample_schedule == "cosine":
            return _cosine_curve(t)
        if self.sample_schedule == "late":
            return _late_curve(t, self.late_sample_schedule_curve)
        raise ValueError(f"Unknown sample_schedule {self.sample_schedule!r}")

    def _encode_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.denoise_space.encode(batch["full_parameters_backend"], batch["midi_norm"])
        if not torch.is_tensor(encoded):
            encoded = torch.from_numpy(encoded)
        return encoded.to(device=batch["mel_spec"].device, dtype=torch.float32)

    def _midi_norm_to_absolute(self, midi_norm: torch.Tensor) -> torch.Tensor:
        note = self.note_min + midi_norm[:, 0] * float(max(self.note_classes - 1, 1))
        velocity = midi_norm[:, 1] * float(max(self.velocity_classes - 1, 1))
        duration = self.duration_min + midi_norm[:, 2] * (self.duration_max - self.duration_min)
        return torch.stack((note, velocity, duration), dim=1)

    def _denoised_to_full_and_midi_absolute(
        self,
        denoised: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full, midi_norm = self.denoise_space.decode_full_and_midi_norm(denoised.float())
        if not torch.is_tensor(full):
            full = torch.from_numpy(full)
        if not torch.is_tensor(midi_norm):
            midi_norm = torch.from_numpy(midi_norm)
        full = full.to(dtype=torch.float32, device=denoised.device)
        midi_norm = midi_norm.to(dtype=torch.float32, device=denoised.device)
        midi_absolute = self._midi_norm_to_absolute(midi_norm)
        return full, midi_absolute

    @torch.no_grad()
    def _predict_full_and_midi_from_mel(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.vector_field.parameters()).device
        conditioning = mel_spec.to(device=device, dtype=torch.float32)
        noise = torch.randn(conditioning.shape[0], int(self.denoise_space.total_dim), device=device)
        pred_params = self._sample(
            conditioning,
            noise,
            steps=self.validation_sample_steps,
            cfg_strength=self.validation_cfg_strength,
        )
        return self._denoised_to_full_and_midi_absolute(pred_params.to(device=device, dtype=torch.float32))

    def _sample(
        self,
        conditioning: Optional[torch.Tensor],
        noise: torch.Tensor,
        *,
        steps: int,
        cfg_strength: float,
    ) -> torch.Tensor:
        encoded_conditioning = self.encoder(conditioning) if conditioning is not None else None
        t = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt = 1.0 / steps
        sample = noise
        for _ in range(steps):
            warped_t = self._warp_time(t)
            warped_t_plus_dt = self._warp_time(t + dt)
            warped_dt = warped_t_plus_dt - warped_t
            sample = rk4_with_cfg(
                self.vector_field,
                sample,
                warped_t,
                warped_dt,
                encoded_conditioning,
                cfg_strength,
            )
            t = t + dt
        return sample

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        conditioning = batch["mel_spec"]
        params = self._encode_targets(batch)
        encoded_conditioning = self.encoder(conditioning)
        dropped_conditioning = self.vector_field.apply_dropout(encoded_conditioning, self.cfg_dropout_rate)

        with torch.no_grad():
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t).squeeze(-1)
            x0, x1, dropped_conditioning = self._sample_x0_and_x1(params, dropped_conditioning)
            x_t = self._rectified_probability_path(x0, x1, t)
            target = self._rectified_vector_field(x0, x1)

        prediction = self.vector_field(x_t, t, dropped_conditioning)
        diff = prediction - target

        param_idx = torch.as_tensor(self.param_indices, device=prediction.device)
        midi_idx = torch.as_tensor(self.midi_indices, device=prediction.device)

        if self.uniform_loss_weight:
            all_loss = diff.square().mean(dim=-1)
            loss = (all_loss * w).mean()
            param_loss = (diff[..., param_idx].square().mean(dim=-1) * w).mean()
            if midi_idx.numel():
                midi_loss = (diff[..., midi_idx].square().mean(dim=-1) * w).mean()
            else:
                midi_loss = torch.zeros_like(param_loss)
        else:
            param_loss = (diff[..., param_idx].square().mean(dim=-1) * w).mean()
            if midi_idx.numel():
                midi_loss = (diff[..., midi_idx].square().mean(dim=-1) * w).mean()
            else:
                midi_loss = torch.zeros_like(param_loss)
            loss = param_loss + midi_loss

        penalty = self.vector_field.penalty() if hasattr(self.vector_field, "penalty") else None
        return loss, penalty, param_loss, midi_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, penalty, param_loss, midi_loss = self._train_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/param_loss", param_loss, on_step=True, prog_bar=False)
        self.log("train/midi_loss", midi_loss, on_step=True, prog_bar=False)
        if penalty is not None:
            self.log("train/penalty", penalty, on_step=True, on_epoch=True, prog_bar=True)
            return loss + penalty
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        device = next(self.vector_field.parameters()).device
        gt_params = self._encode_targets(batch)
        conditioning = batch["mel_spec"].to(device=device, dtype=torch.float32)
        noise = torch.randn(gt_params.shape[0], int(self.denoise_space.total_dim), device=device)
        pred_params = self._sample(
            conditioning,
            noise,
            steps=self.validation_sample_steps,
            cfg_strength=self.validation_cfg_strength,
        )
        param_mse = (pred_params - gt_params).square().mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=gt_params.shape[0])

        if (
            self.render_evaluator is None
            or "audio" not in batch
            or batch["audio"] is None
            or self._val_render_count >= self.render_batches
        ):
            return

        full_pred, midi_pred = self._denoised_to_full_and_midi_absolute(
            pred_params.to(device=device, dtype=torch.float32)
        )

        try:
            pred_audio, kept = self.render_evaluator.render_from_full_and_midi_resilient(full_pred, midi_pred)
        except Exception:
            return

        target_audio = batch["audio"].squeeze(1).float()
        if kept.size != target_audio.shape[0]:
            target_audio = target_audio[torch.as_tensor(kept, dtype=torch.long, device=target_audio.device)]
        batch_metrics = self.render_evaluator.compute_metrics(pred_audio, target_audio)
        for name, value in batch_metrics.items():
            if math.isfinite(float(value)):
                self.log(
                    f"val/{name}",
                    float(value),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=pred_audio.shape[0],
                )
        self._val_render_count += 1

    def on_validation_epoch_end(self) -> None:
        if not self.nsynth_enable or self.ood_evaluator is None:
            return

        device = next(self.vector_field.parameters()).device
        metrics = self.ood_evaluator.run(
            predict_from_mel=self._predict_full_and_midi_from_mel,
            device=device,
        )
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        device = next(self.vector_field.parameters()).device
        gt_params = self._encode_targets(batch)
        conditioning = batch["mel_spec"].to(device=device, dtype=torch.float32)
        noise = torch.randn(gt_params.shape[0], int(self.denoise_space.total_dim), device=device)
        pred_params = self._sample(
            conditioning,
            noise,
            steps=self.test_sample_steps,
            cfg_strength=self.test_cfg_strength,
        )
        param_mse = (pred_params - gt_params).square().mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=gt_params.shape[0])
        return param_mse

    def configure_optimizers(self):
        opt_cfg = self.optimizer_cfg
        if opt_cfg and opt_cfg.get("_partial_", False):
            opt_partial = hydra.utils.instantiate(opt_cfg)
            optimizer = opt_partial(self.parameters())
        else:
            optimizer = hydra.utils.instantiate(opt_cfg, self.parameters())

        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer) if self.scheduler_cfg else None
        warmup_scheduler = None
        if self.warmup_steps > 0:
            warmup_scheduler = lr_sched.LinearLR(
                optimizer,
                start_factor=1e-10,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )

        if warmup_scheduler is not None and scheduler is None:
            scheduler = warmup_scheduler
        elif warmup_scheduler is not None and scheduler is not None:
            scheduler = SafeSequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.warmup_steps],
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
