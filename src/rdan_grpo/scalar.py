"""RDAN-GRPO advantage construction from rubric scores.

RDAN decouples two normalizations over each group of responses to one prompt, following PAPO:

    A_out  = normalize(outcome reward, over responses with a well-defined outcome)
    A_proc = normalize(soft rubric quality, over hard-passing responses only)
    A      = A_out + quality_weight * A_proc

``rl_aon`` and ``rl_csr`` are the RTT reward-union baselines: one outcome channel over all
rubrics, no process channel.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from rdan_grpo.advantages import group_advantages
from rdan_grpo.rewards import RubricRewards, extract_quality, score_rubrics

ScalarMethod = Literal["rdan", "rl_aon", "rl_csr"]
METHODS: frozenset[str] = frozenset({"rdan", "rl_aon", "rl_csr"})
QUALITY_METHODS: frozenset[str] = frozenset({"rdan"})


@dataclass(frozen=True)
class ScalarOutput:
    """One scalar advantage per response, with the channels that produced it."""

    method: ScalarMethod
    quality_weight: float | None
    raw_aon: Tensor
    raw_csr: Tensor
    response_valid: Tensor
    hard_pass: Tensor
    raw_quality: Tensor
    quality_valid: Tensor
    quality_eligible: Tensor
    selected_raw_reward: Tensor
    response_advantage: Tensor
    quality_advantage: Tensor
    scalar_advantage: Tensor
    diagnostics: Mapping[str, float]


def validate_groups(prompt_keys: Sequence[Hashable], group_size: int) -> None:
    """Require complete contiguous groups and exactly one group per prompt key."""

    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if not prompt_keys or len(prompt_keys) % group_size:
        raise ValueError("response count must be a positive multiple of group_size")
    seen: set[Hashable] = set()
    for start in range(0, len(prompt_keys), group_size):
        key = prompt_keys[start]
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
    quality_weight: float | None = None,
) -> ScalarOutput:
    """Build one scalar advantage per response without framework side effects."""

    if method not in METHODS:
        raise ValueError(f"unsupported scalar method: {method}")
    if scores.ndim != 2 or len(prompt_keys) != scores.shape[0]:
        raise ValueError("prompt_keys must contain one key per response")
    validate_groups(prompt_keys, group_size)
    weight = _quality_weight(method, quality_weight)

    quality = extract_quality(scores, rubric_mask, eval_mask, hard_mask)
    if method in QUALITY_METHODS:
        # The outcome channel is the hard rubrics alone, so a judge failure costs a response
        # its process credit but never its outcome reward or its place in the group.
        rewards = score_rubrics(scores, rubric_mask & hard_mask, eval_mask)
        no_hard = ~(rubric_mask & hard_mask).any(dim=-1)
        satisfied = torch.ones_like(rewards.aon)
        rewards = RubricRewards(
            aon=torch.where(no_hard, satisfied, rewards.aon),
            csr=torch.where(no_hard, satisfied, rewards.csr),
            valid=rewards.valid | no_hard,
        )
        hard_pass = quality.hard_pass | no_hard
        quality_eligible = hard_pass & quality.quality_valid
        selected = rewards.aon
    else:
        rewards = score_rubrics(scores, rubric_mask, eval_mask)
        hard_pass = quality.hard_pass
        quality_eligible = quality.eligible
        selected = rewards.csr if method == "rl_csr" else rewards.aon

    response_valid = rewards.valid
    response_advantage = group_advantages(selected, group_size, valid=response_valid)
    quality_advantage = torch.zeros_like(response_advantage)
    scalar_advantage = response_advantage
    if method in QUALITY_METHODS:
        quality_advantage = group_advantages(quality.quality, group_size, valid=quality_eligible)
        scalar_advantage = scalar_advantage + weight * quality_advantage
    scalar_advantage = torch.where(response_valid, scalar_advantage, 0.0)

    outputs = (rewards.aon, rewards.csr, quality.quality, selected, response_advantage, quality_advantage)
    if not all(bool(torch.isfinite(value).all()) for value in outputs):
        raise ValueError("scalar output contains non-finite values")

    count = scores.shape[0]
    diagnostics = {
        "response_count": float(count),
        "group_count": float(count // group_size),
        "response_valid_rate": float(response_valid.float().mean().item()),
        "hard_pass_rate": float(hard_pass.float().mean().item()),
        "quality_valid_rate": float(quality.quality_valid.float().mean().item()),
        "quality_eligible_rate": float(quality_eligible.float().mean().item()),
    }
    return ScalarOutput(
        method=method,
        quality_weight=weight if method in QUALITY_METHODS else None,
        raw_aon=rewards.aon,
        raw_csr=rewards.csr,
        response_valid=response_valid,
        hard_pass=hard_pass,
        raw_quality=quality.quality,
        quality_valid=quality.quality_valid,
        quality_eligible=quality_eligible,
        selected_raw_reward=selected,
        response_advantage=response_advantage,
        quality_advantage=quality_advantage,
        scalar_advantage=scalar_advantage,
        diagnostics=diagnostics,
    )


def _quality_weight(method: ScalarMethod, value: float | None) -> float:
    if method not in QUALITY_METHODS:
        if value is not None:
            raise ValueError(f"quality_weight is not valid for {method}")
        return 0.0
    if value is None:
        raise ValueError("quality_weight is required for rdan")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("quality_weight must be a non-negative finite number")
    return float(value)
