from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch import Callback, LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

from src.project_paths import resolve_project_path
from src.utils.wandb_resume import infer_wandb_resume


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="dd_train_dexed")
def main(cfg: DictConfig) -> Optional[float]:
    if cfg.get("seed"):
        L.seed_everything(int(cfg.seed), workers=True)

    _apply_wandb_resume(cfg)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    ckpt_dir = resolve_project_path(cfg.paths.output_dir) / "checkpoints"
    model_target = cfg.model.get("_target_")
    if not model_target:
        raise ValueError("cfg.model must define a _target_.")
    model_cls = hydra.utils.get_class(model_target)
    model: LightningModule = model_cls(cfg=cfg, ckpt_dir=ckpt_dir)
    callbacks: List[Callback] = _instantiate_named_collection(cfg.get("callbacks"))
    loggers: List[Logger] = _instantiate_named_collection(cfg.get("logger"))

    trainer_kwargs = {}
    sched_cfg = getattr(cfg.model, "training_schedule", None)
    if sched_cfg:
        trainer_kwargs.update(
            {
                "max_steps": sched_cfg.get("max_steps"),
                "log_every_n_steps": sched_cfg.get("log_every_n_steps"),
                "val_check_interval": sched_cfg.get("val_check_interval"),
                "check_val_every_n_epoch": sched_cfg.get("check_val_every_n_epoch"),
            }
        )

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        default_root_dir=str(resolve_project_path(cfg.paths.output_dir)),
        callbacks=callbacks,
        logger=loggers if loggers else None,
        **{k: v for k, v in trainer_kwargs.items() if v is not None},
    )
    if loggers:
        hparams = OmegaConf.to_container(cfg, resolve=True)
        for logger in trainer.loggers:
            if hasattr(logger, "log_hyperparams"):
                logger.log_hyperparams(hparams)

    if cfg.get("train", True):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    if cfg.get("test", False):
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    metric_name = cfg.get("optimized_metric")
    if metric_name:
        metric_value = trainer.callback_metrics.get(metric_name)
        if metric_value is not None:
            return float(metric_value)
    return None


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
