"""One RDAN optimizer transaction over a rewarded rollout batch."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from roll.distributed.scheduler.protocol import DataProto

from rdan_grpo.bridge import assess_batch, inject_advantages
from rdan_grpo.scalar import ScalarMethod, ScalarOutput

REQUIRED_TENSORS = (
    "input_ids",
    "attention_mask",
    "response_mask",
    "rdan_scores",
    "rdan_rubric_mask",
    "rdan_eval_mask",
    "rdan_hard_mask",
)
REQUIRED_METADATA = ("prompt", "rubrics", "source", "ground_truth", "rdan_prompt_key")


@dataclass(frozen=True)
class TrainStepResult:
    """What one completed optimizer transaction produced."""

    method: ScalarMethod
    prompt_count: int
    response_count: int
    scalar: ScalarOutput
    metrics: Mapping[str, float]
    peak_memory_fraction: float


def run_train_step(
    *,
    actor_train: Any,
    rewarded_batch: DataProto,
    group_size: int,
    method: ScalarMethod,
    quality_weight: float | None,
    observe_memory: Any,
) -> TrainStepResult:
    """Score, advantage, and train once from a rewarded batch."""

    prompt_count = _validate_batch(rewarded_batch, group_size)
    scalar = _prepare_batch(rewarded_batch, group_size, method, quality_weight)
    _bind_log_probs(actor_train, rewarded_batch)
    output = actor_train.train_step(rewarded_batch, blocking=True)
    metrics = _train_metrics(output)
    return TrainStepResult(
        method=method,
        prompt_count=prompt_count,
        response_count=len(rewarded_batch),
        scalar=scalar,
        metrics=metrics,
        peak_memory_fraction=_peak_memory_fraction(observe_memory()),
    )


def _prepare_batch(
    batch: DataProto,
    group_size: int,
    method: ScalarMethod,
    quality_weight: float | None,
) -> ScalarOutput:
    """Build the RDAN advantage and expand it over the shifted response mask."""

    shifted = batch.batch["response_mask"][:, 1:].to(torch.bool)
    batch.batch["final_response_mask"] = shifted.clone()
    scalar = assess_batch(batch, method, group_size, quality_weight)
    inject_advantages(batch, scalar, shifted)
    expected = scalar.scalar_advantage.unsqueeze(-1).to(batch.batch["advantages"]) * shifted.to(
        batch.batch["advantages"].dtype
    )
    if not torch.equal(batch.batch["advantages"], expected):
        raise RuntimeError("training advantage differs from the RDAN scalar advantage")
    return scalar


def _validate_batch(data: DataProto, group_size: int) -> int:
    """Check the scheduler produced complete, prompt-major, contiguous response groups."""

    if data.batch is None or len(data) == 0 or len(data) % group_size:
        raise RuntimeError(f"rewarded batch must be a non-empty multiple of group_size {group_size}")
    missing = sorted(set(REQUIRED_TENSORS) - set(data.batch.keys()))
    missing.extend(name for name in REQUIRED_METADATA if name not in data.non_tensor_batch)
    if missing:
        raise RuntimeError(f"rewarded batch is missing fields: {', '.join(missing)}")
    keys = list(data.non_tensor_batch["rdan_prompt_key"])
    seen: set[Any] = set()
    for start in range(0, len(data), group_size):
        key = keys[start]
        if any(value != key for value in keys[start : start + group_size]):
            raise RuntimeError("rewarded batch is not grouped prompt-major by rdan_prompt_key")
        if key in seen:
            raise RuntimeError(f"rewarded batch repeats prompt group {key!r}")
        seen.add(key)
    response_mask = data.batch["response_mask"]
    if data.batch["input_ids"].shape != response_mask.shape:
        raise RuntimeError("rewarded batch input_ids and response_mask are misaligned")
    if not bool(response_mask.to(torch.bool).any(dim=-1).all()):
        raise RuntimeError("every rewarded response must contain at least one response token")
    return len(data) // group_size


def _bind_log_probs(actor_train: Any, data: DataProto) -> None:
    """Recompute the behaviour policy log probs the PPO ratio is measured against."""

    output = actor_train.compute_log_probs(data.clone(), blocking=True)
    log_probs = output.batch.get("log_probs") if output.batch is not None else None
    expected = (len(data), data.batch["response_mask"].shape[1] - 1)
    if not isinstance(log_probs, torch.Tensor) or log_probs.shape != expected:
        raise RuntimeError(f"actor log probs must have shape {expected}")
    data.batch["old_log_probs"] = log_probs.clone()
    data.batch["ref_log_probs"] = log_probs.clone()


def _train_metrics(output: Any) -> dict[str, float]:
    metrics = getattr(output, "meta_info", {}).get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise RuntimeError("actor train step returned no metrics")
    return {name: value for name, value in _flatten(metrics) if math.isfinite(value)}


def _flatten(metrics: Mapping[str, Any], prefix: str = "") -> list[tuple[str, float]]:
    flat: list[tuple[str, float]] = []
    for name, value in metrics.items():
        key = f"{prefix}{name}"
        if isinstance(value, Mapping):
            flat.extend(_flatten(value, f"{key}/"))
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            flat.append((key, float(value.item())))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat.append((key, float(value)))
        elif isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (str, bytes)):
            values = [float(item) for item in np.asarray(value).ravel().tolist()]
            if values:
                flat.append((key, sum(values) / len(values)))
    return flat


def _peak_memory_fraction(values: Sequence[Mapping[str, Any]]) -> float:
    """Highest peak-to-capacity GPU memory ratio across ranks, reported not enforced."""

    fractions = [row["peak_bytes"] / row["total_bytes"] for row in values if row.get("total_bytes")]
    return max(fractions) if fractions else 0.0
