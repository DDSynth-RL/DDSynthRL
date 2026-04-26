from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

_DELETED_RUN_MARKER = "previously created and deleted"


def _parse_run_id(run_dir: Path) -> Optional[str]:
    name = run_dir.name
    if not name.startswith("run-"):
        return None
    parts = name.split("-")
    if len(parts) < 3:
        return None
    run_id = parts[-1].strip()
    return run_id or None


def _find_run_dir(output_dir: Path) -> Optional[Path]:
    wandb_dir = output_dir / "wandb"
    if not wandb_dir.exists():
        return None

    latest = wandb_dir / "latest-run"
    if latest.exists():
        try:
            resolved = latest.resolve()
        except OSError:
            resolved = latest
        if resolved.exists() and resolved.is_dir():
            return resolved

    candidates = sorted(
        [path for path in wandb_dir.glob("run-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_previous_experiment_name(output_dir: Path) -> Optional[str]:
    cfg_path = output_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = OmegaConf.load(str(cfg_path))
    except Exception:
        return None
    name = OmegaConf.select(cfg, "experiment.name")
    return str(name) if name is not None else None


def _current_experiment_name(cfg: DictConfig) -> Optional[str]:
    name = OmegaConf.select(cfg, "experiment.name")
    return str(name) if name is not None else None


def _log_contains_deleted_marker(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return _DELETED_RUN_MARKER in text


def _run_id_is_known_deleted(experiment_root: Path, run_id: str) -> bool:
    for run_dir in experiment_root.glob("*/wandb/run-*"):
        if not run_dir.is_dir():
            continue
        if _parse_run_id(run_dir) != run_id:
            continue
        if _log_contains_deleted_marker(run_dir / "logs" / "debug-internal.log"):
            return True
        if _log_contains_deleted_marker(run_dir / "logs" / "debug.log"):
            return True
    return False


def infer_wandb_resume(logger_cfg: DictConfig | None, cfg: DictConfig, ckpt_path: Optional[Path]) -> Dict[str, Any]:
    if logger_cfg is None or ckpt_path is None:
        return {}
    if not ckpt_path.exists():
        return {}

    explicit_id = OmegaConf.select(logger_cfg, "wandb.id")
    explicit_resume = OmegaConf.select(logger_cfg, "wandb.resume")
    if explicit_id is not None or explicit_resume is not None:
        return {}

    output_dir = ckpt_path.parent.parent
    previous_name = _load_previous_experiment_name(output_dir)
    current_name = _current_experiment_name(cfg)
    if previous_name is not None and current_name is not None and previous_name != current_name:
        return {}

    run_dir = _find_run_dir(output_dir)
    if run_dir is None:
        return {}
    run_id = _parse_run_id(run_dir)
    if run_id is None:
        return {}
    if _run_id_is_known_deleted(output_dir.parent, run_id):
        return {}
    return {"id": run_id, "resume": "allow"}
