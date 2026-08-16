"""Metric tracking: W&B when reachable, always a local JSONL mirror and curve plots.

The JSONL mirror is written first and unconditionally, so a run whose W&B connection dies
mid-training still yields complete curves from ``plot_curves``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRACKER_NAME = "rdan_wandb"
METRICS_FILE = "metrics.jsonl"
CURVES_DIR = "curves"
STEP_METRIC = "system/step"
# Curves worth a standalone plot, grouped by panel title.
CURVE_PANELS = {
    "reward": ("reward/selected_mean", "reward/valid_rate", "reward/process_quality_mean"),
    "advantage": ("advantage/mean", "advantage/std", "advantage/zero_rate"),
    "policy": ("rdan/response_token_clipfrac", "actor/pg_loss", "actor/entropy"),
    "length": ("length/mean", "length/cap_hit_rate"),
    "judge": ("judge/failure_rate", "judge/latency_p95", "judge/cost_usd"),
}
_SECRET = re.compile(r"(?:sk-or-v1-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|[0-9a-f]{40})")
_SECRET_KEYS = frozenset({"api_key", "apikey", "authorization", "password", "secret", "token", "credential"})


def register_tracker() -> type[RdanTracker]:
    """Register the RDAN tracker under ROLL's tracker registry."""

    from roll.utils import tracking

    tracking.tracker_registry[TRACKER_NAME] = RdanTracker
    return RdanTracker


class RdanTracker:
    """Mirror every metric row to disk and forward it to W&B when available."""

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        # ROLL builds trackers with the pipeline config plus tracker_kwargs and passes no
        # directory, so the run's own logging_dir is what keeps runs from sharing a mirror.
        log_dir = Path(kwargs.pop("log_dir", None) or config.get("logging_dir") or "output")
        self.metrics_path = log_dir / METRICS_FILE
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.run = _start_wandb(redact(dict(config)), log_dir, kwargs)

    def log(self, values: Mapping[str, Any], step: int | None = None, **kwargs: Any) -> None:
        """Append one metric row locally, then forward it to W&B."""

        row = {name: value for name, value in values.items() if isinstance(value, (int, float, str))}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if self.run is not None:
            # Make W&B's own step index the pipeline step, so every panel plots correctly on
            # the default axis. A custom x-axis needs define_metric to associate each metric
            # with it, and a "*" glob there also captures the axis metric itself, which leaves
            # the axis self-referential and every panel empty.
            if step is None and isinstance(row.get(STEP_METRIC), (int, float)):
                step = int(row[STEP_METRIC])
            # commit=True is required whenever step is explicit: W&B otherwise holds the row
            # open until a later step arrives, so the dashboard trails by one step and the
            # final step of a run never lands at all.
            self.run.log(dict(row), step=step, commit=True, **kwargs)

    def finish(self) -> None:
        """Write the curve plots and close the W&B run."""

        plot_curves(self.metrics_path)
        if self.run is not None:
            self.run.finish()


def plot_curves(metrics_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Render the metric panels from a metrics JSONL file into PNGs."""

    path = Path(metrics_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if STEP_METRIC in row]
    if not rows:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_dir) if output_dir else path.parent / CURVES_DIR
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for panel, names in CURVE_PANELS.items():
        series = {
            name: [(row[STEP_METRIC], row[name]) for row in rows if isinstance(row.get(name), (int, float))]
            for name in names
        }
        series = {name: points for name, points in series.items() if points}
        if not series:
            continue
        figure, axes = plt.subplots(len(series), 1, figsize=(9, 2.6 * len(series)), sharex=True, squeeze=False)
        for axis, (name, points) in zip(axes[:, 0], series.items(), strict=True):
            axis.plot([point[0] for point in points], [point[1] for point in points], linewidth=1.4)
            axis.set_ylabel(name.split("/")[-1])
            axis.set_title(name, fontsize=9, loc="left")
            axis.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("training step")
        figure.tight_layout()
        destination = target / f"{panel}.png"
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        written.append(destination)
    return written


def redact(value: Any) -> Any:
    """Replace credential-shaped values so they never reach a tracker or a log file."""

    if isinstance(value, Mapping):
        return {key: "[REDACTED]" if str(key).lower() in _SECRET_KEYS else redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET.sub("[REDACTED]", value)
    return value


def _start_wandb(config: Mapping[str, Any], log_dir: Path, kwargs: dict[str, Any]) -> Any:
    """Start a W&B run, falling back to offline mode and then to disk-only tracking."""

    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
    settings = kwargs.pop("settings", {"console": "off"})
    kwargs.pop("api_key", None)
    try:
        run = wandb.init(dir=str(log_dir), settings=settings, resume="allow", **kwargs)
    except Exception as error:  # noqa: BLE001 - tracking must never end a training run
        print(f"W&B unavailable ({type(error).__name__}: {error}); metrics go to {log_dir / METRICS_FILE} only")
        return None
    run.config.update(dict(config), allow_val_change=True)
    return run
