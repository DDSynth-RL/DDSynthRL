from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from lightning import Callback, LightningModule, Trainer


class MultiMetricTopKCheckpointCallback(Callback):
    """Save `checkpoint_last.pt` and keep top-k checkpoints for multiple metrics."""

    def __init__(
        self,
        ckpt_dir: Path,
        monitors: Sequence[str],
        modes: Sequence[str] | None = None,
        save_top_k: int = 5,
        save_last: bool = True,
    ) -> None:
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.monitors = [str(m) for m in monitors]
        if not self.monitors:
            raise ValueError("monitors must be a non-empty list of metric names.")

        self.modes = ["min"] * len(self.monitors) if modes is None else [str(m) for m in modes]
        if len(self.modes) != len(self.monitors):
            raise ValueError("modes must have the same length as monitors.")
        for mode in self.modes:
            if mode not in {"min", "max"}:
                raise ValueError(f"Unsupported mode '{mode}'. Use 'min' or 'max'.")

        self.save_top_k = int(save_top_k)
        self.save_last = bool(save_last)
        self._topk: Dict[str, List[Tuple[float, Path]]] = {m: [] for m in self.monitors}

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitors=tuple(self.monitors),
            modes=tuple(self.modes),
            save_top_k=int(self.save_top_k),
            save_last=bool(self.save_last),
        )

    @staticmethod
    def _sanitize(name: str) -> str:
        return str(name).replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")

    def state_dict(self) -> Dict:
        return {
            "topk": {
                monitor: [{"score": score, "path": str(path)} for score, path in items]
                for monitor, items in self._topk.items()
            }
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        topk = state_dict.get("topk")
        if not isinstance(topk, dict):
            return
        restored: Dict[str, List[Tuple[float, Path]]] = {}
        for monitor, items in topk.items():
            if not isinstance(items, list):
                continue
            entries: List[Tuple[float, Path]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    score = float(item.get("score"))
                except Exception:
                    continue
                path = item.get("path")
                if not path:
                    continue
                entries.append((score, Path(path)))
            restored[str(monitor)] = entries
        for monitor in self.monitors:
            self._topk[monitor] = restored.get(monitor, [])

    def _sort_topk(self, monitor: str, mode: str) -> None:
        items = self._topk.get(monitor, [])
        items.sort(key=lambda p: p[0], reverse=(mode == "max"))
        self._topk[monitor] = items

    def _is_better(self, score: float, other: float, mode: str) -> bool:
        return score < other if mode == "min" else score > other

    def _maybe_save_topk(self, trainer: Trainer, monitor: str, mode: str, score: float) -> None:
        if self.save_top_k <= 0:
            return

        items = self._topk.get(monitor, [])
        if items:
            self._sort_topk(monitor, mode)
            items = self._topk[monitor]

        should_save = len(items) < self.save_top_k
        if not should_save and items:
            should_save = self._is_better(score, items[-1][0], mode)
        if not should_save:
            return

        safe = self._sanitize(monitor)
        step = int(getattr(trainer, "global_step", 0) or 0)
        path = self.ckpt_dir / f"checkpoint_best_{safe}_step_{step:08d}_metric_{score:.6f}.pt"
        trainer.save_checkpoint(path)

        items.append((score, path))
        self._topk[monitor] = items
        self._sort_topk(monitor, mode)

        items = self._topk[monitor]
        for _, prune_path in items[self.save_top_k :]:
            try:
                prune_path.unlink()
            except OSError:
                pass
        self._topk[monitor] = items[: self.save_top_k]

    def _get_metric(self, trainer: Trainer, name: str):
        metrics = {}
        metrics.update(getattr(trainer, "callback_metrics", {}) or {})
        metrics.update(getattr(trainer, "logged_metrics", {}) or {})
        metrics.update(getattr(trainer, "progress_bar_metrics", {}) or {})
        return metrics.get(name)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return

        if self.save_last:
            trainer.save_checkpoint(self.ckpt_dir / "checkpoint_last.pt")

        for monitor, mode in zip(self.monitors, self.modes):
            value = self._get_metric(trainer, monitor)
            if value is None:
                continue
            try:
                score = float(value.item()) if hasattr(value, "item") else float(value)
            except Exception:
                continue
            if not math.isfinite(score):
                continue
            self._maybe_save_topk(trainer, monitor=monitor, mode=mode, score=score)
