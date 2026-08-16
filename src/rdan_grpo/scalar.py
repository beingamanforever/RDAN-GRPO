"""Offline scalar reward and advantage construction for ROLL adapters."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch
from torch import Tensor

from rdan_grpo.advantages import group_advantages, quality_advantages
from rdan_grpo.rewards import RubricRewards, extract_quality, score_rubrics

ScalarMethod = Literal["rl_aon", "rl_csr", "rl_mix", "rdan_scalar", "rtt_papo_response"]
QUALITY_METHODS: frozenset[ScalarMethod] = frozenset({"rdan_scalar", "rtt_papo_response"})
DiagnosticValue = bool | int | float


@dataclass(frozen=True)
class ScalarOutput:
    """Fail-closed scalar channels ready for a training adapter."""

    method: ScalarMethod
    mix_weight: float | None
    quality_weight: float | None
    raw_aon: Tensor
    raw_csr: Tensor
    raw_signed_csr: Tensor
    response_valid: Tensor
    hard_pass: Tensor
    raw_quality: Tensor
    quality_valid: Tensor
    quality_eligible: Tensor
    selected_raw_reward: Tensor
    response_advantage: Tensor
    quality_advantage: Tensor
    scalar_advantage: Tensor
    training_ready: bool
    diagnostics: Mapping[str, DiagnosticValue]


def validate_groups(prompt_keys: Sequence[Hashable], group_size: int) -> None:
    """Require complete contiguous groups and exactly one group per prompt key."""

    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if not prompt_keys:
        raise ValueError("prompt_keys must not be empty")
    if len(prompt_keys) % group_size:
        raise ValueError("response count must be divisible by group_size")

    seen: set[Hashable] = set()
    for start in range(0, len(prompt_keys), group_size):
        key = prompt_keys[start]
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError("prompt keys must be hashable") from exc
        if any(candidate != key for candidate in prompt_keys[start : start + group_size]):
            raise ValueError(f"prompt group at response {start} is not contiguous and complete")
        if key in seen:
            raise ValueError(f"prompt key {key!r} appears in more than one group")
        seen.add(key)


def build_scalar_output(
    method: ScalarMethod,
    prompt_keys: Sequence[Hashable],
    scores: Tensor,
    rubric_mask: Tensor,
    eval_mask: Tensor,
    hard_mask: Tensor,
    group_size: int = 8,
    mix_weight: float | None = None,
    quality_weight: float | None = None,
) -> ScalarOutput:
    """Build one scalar advantage per response without framework side effects."""

    if method not in ("rl_aon", "rl_csr", "rl_mix", "rdan_scalar", "rtt_papo_response"):
        raise ValueError(f"unsupported scalar method: {method}")
    if scores.ndim != 2 or len(prompt_keys) != scores.shape[0]:
        raise ValueError("prompt_keys must contain one key per response")
    validate_groups(prompt_keys, group_size)
    mix, quality_scale = _method_weights(method, mix_weight, quality_weight)

    all_rewards = score_rubrics(scores, rubric_mask, eval_mask)
    quality = extract_quality(scores, rubric_mask, eval_mask, hard_mask)
    rewards = all_rewards
    hard_pass = quality.hard_pass
    quality_eligible = quality.eligible
    if method in QUALITY_METHODS:
        rewards = score_rubrics(scores, rubric_mask & hard_mask, eval_mask)
    if method == "rtt_papo_response":
        no_hard = ~(rubric_mask & hard_mask).any(dim=-1)
        one = torch.ones_like(rewards.aon)
        rewards = RubricRewards(
            aon=torch.where(no_hard, one, rewards.aon),
            csr=torch.where(no_hard, one, rewards.csr),
            signed_csr=torch.where(no_hard, one, rewards.signed_csr),
            valid=rewards.valid | no_hard,
        )
        hard_pass = quality.hard_pass | no_hard
        quality_eligible = hard_pass & quality.quality_valid
    response_valid = rewards.valid & all_rewards.valid
    selected = _select_reward(method, rewards.aon, rewards.csr, mix)
    # PAPO normalizes the outcome advantage over the whole group and the process advantage
    # over the correct subset only, so A_out sees every response and invalid ones are zeroed
    # after composition rather than excluded from the group statistics.
    response_advantage = group_advantages(selected, group_size)
    quality_advantage = torch.zeros_like(response_advantage)
    scalar_advantage = response_advantage
    if method in QUALITY_METHODS:
        quality_advantage = quality_advantages(quality.quality, quality_eligible, group_size)
        scalar_advantage = scalar_advantage + quality_scale * quality_advantage
    scalar_advantage = torch.where(response_valid, scalar_advantage, 0.0)

    float_outputs = (
        rewards.aon,
        rewards.csr,
        rewards.signed_csr,
        quality.quality,
        selected,
        response_advantage,
        quality_advantage,
        scalar_advantage,
    )
    if not all(bool(torch.isfinite(value).all()) for value in float_outputs):
        raise ValueError("scalar output contains non-finite values")

    valid_count = int(response_valid.sum().item())
    response_count = scores.shape[0]
    training_ready = valid_count == response_count
    diagnostics: Mapping[str, DiagnosticValue] = MappingProxyType(
        {
            "response_count": response_count,
            "group_count": response_count // group_size,
            "invalid_response_count": response_count - valid_count,
            "response_valid_rate": valid_count / response_count,
            "hard_pass_rate": float(hard_pass.float().mean().item()),
            "quality_valid_rate": float(quality.quality_valid.float().mean().item()),
            "quality_eligible_rate": float(quality_eligible.float().mean().item()),
            "finite": True,
            "training_ready": training_ready,
        }
    )
    return ScalarOutput(
        method=method,
        mix_weight=mix if method == "rl_mix" else None,
        quality_weight=quality_scale if method in QUALITY_METHODS else None,
        raw_aon=rewards.aon,
        raw_csr=rewards.csr,
        raw_signed_csr=rewards.signed_csr,
        response_valid=response_valid,
        hard_pass=hard_pass,
        raw_quality=quality.quality,
        quality_valid=quality.quality_valid,
        quality_eligible=quality_eligible,
        selected_raw_reward=selected,
        response_advantage=response_advantage,
        quality_advantage=quality_advantage,
        scalar_advantage=scalar_advantage,
        training_ready=training_ready,
        diagnostics=diagnostics,
    )


def _method_weight(name: str, value: float | None, *, required: bool, maximum: float | None = None) -> float:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (maximum is not None and value > maximum):
        interval = "[0, 1]" if maximum == 1 else "non-negative"
        raise ValueError(f"{name} must be {interval}")
    return float(value)


def _method_weights(
    method: ScalarMethod,
    mix_weight: float | None,
    quality_weight: float | None,
) -> tuple[float, float]:
    if method != "rl_mix" and mix_weight is not None:
        raise ValueError(f"mix_weight is not valid for {method}")
    if method not in QUALITY_METHODS and quality_weight is not None:
        raise ValueError(f"quality_weight is not valid for {method}")
    mix = _method_weight("mix_weight", mix_weight, required=method == "rl_mix", maximum=1.0)
    quality = _method_weight("quality_weight", quality_weight, required=method in QUALITY_METHODS)
    return mix, quality


def _select_reward(method: ScalarMethod, aon: Tensor, csr: Tensor, mix_weight: float) -> Tensor:
    if method == "rl_csr":
        return csr.clone()
    if method == "rl_mix":
        return mix_weight * aon + (1.0 - mix_weight) * csr
    return aon.clone()
