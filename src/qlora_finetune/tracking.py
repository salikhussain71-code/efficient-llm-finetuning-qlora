"""Experiment tracking: always a CSV log, optionally Weights & Biases."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class TrackingUnavailable(RuntimeError):
    """Raised when Weights & Biases is requested but cannot be used."""


class Tracker:
    """Collects metrics rows, writes ``train_log.csv`` and mirrors them to W&B when enabled."""

    def __init__(self, out_dir: str | Path, report_to: str = "none", project: str = "qlora-samsum",
                 run_name: str | None = None, config: dict | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.rows: list[dict] = []
        self.report_to = report_to
        self._wandb = None
        if report_to == "wandb":
            try:
                import wandb
            except ImportError as err:
                raise TrackingUnavailable("wandb is not installed: pip install wandb") from err
            self._wandb = wandb
            self.run = wandb.init(project=project, name=run_name, config=config or {}, dir=str(self.out_dir))
        elif report_to != "none":
            raise ValueError("report_to must be 'none' or 'wandb'")

    def log(self, metrics: dict, step: int) -> None:
        self.rows.append({"step": step, **metrics})
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def summary(self, values: dict) -> None:
        if self._wandb is not None:
            self._wandb.run.summary.update(values)

    def flush(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "train_log.csv"
        pd.DataFrame(self.rows).to_csv(path, index=False)
        return path

    def finish(self) -> None:
        self.flush()
        if self._wandb is not None:
            self._wandb.finish()
