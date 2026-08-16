"""Advantage normalization and composition."""

from typing import Literal

import torch
from torch import Tensor

QualityMode = Literal["off", "raw", "full_group", "hard_valid"]


def group_advantages(
    rewards: Tensor,
    group_size: int,
    eps: float = 1e-6,
    valid: Tensor | None = None,
) -> Tensor:
    """Normalize scalar rewards in contiguous response groups using sample standard deviation."""

    groups = _groups(rewards, group_size)
    if valid is not None:
        if valid.dtype != torch.bool or valid.shape != rewards.shape:
            raise ValueError("valid must be boolean and match rewards")
        selected = valid.reshape_as(groups)
        count = selected.sum(dim=-1, keepdim=True)
        mask = selected.to(rewards.dtype)
        mean = (groups * mask).sum(dim=-1, keepdim=True) / count.clamp_min(1)
        var = ((groups - mean).square() * mask).sum(dim=-1, keepdim=True) / (count - 1).clamp_min(1)
        std = var.sqrt()
        out = torch.where((count >= 2) & (std > eps) & selected, (groups - mean) / (std + eps), 0.0)
        return out.reshape_as(rewards)
    if group_size < 2:
        return torch.zeros_like(rewards)
    mean = groups.mean(dim=-1, keepdim=True)
    std = groups.std(dim=-1, keepdim=True)
    out = torch.where(std > eps, (groups - mean) / (std + eps), 0.0)
    return out.reshape_as(rewards)



def quality_advantages(
    quality: Tensor,
    hard_valid: Tensor,
    group_size: int,
    mode: QualityMode = "hard_valid",
    eps: float = 1e-6,
) -> Tensor:
    """Build a scalar quality advantage with optional conditional group normalization.

    ``hard_valid`` identifies responses whose evaluator output is valid and whose hard rubrics pass.
    Conditional groups with fewer than two selected values or zero variance return zero.
    """

    groups = _groups(quality, group_size)
    if hard_valid.dtype != torch.bool or hard_valid.shape != quality.shape:
        raise ValueError("hard_valid must be boolean and match quality")
    if mode == "off":
        return torch.zeros_like(quality)
    if mode == "raw":
        return quality.clone()
    if mode == "full_group":
        return group_advantages(quality, group_size, eps)
    if mode != "hard_valid":
        raise ValueError(f"unsupported quality mode: {mode}")

    selected = hard_valid.reshape_as(groups)
    count = selected.sum(dim=-1, keepdim=True)
    mask = selected.to(quality.dtype)
    mean = (groups * mask).sum(dim=-1, keepdim=True) / count.clamp_min(1)
    var = ((groups - mean).square() * mask).sum(dim=-1, keepdim=True) / (count - 1).clamp_min(1)
    std = var.sqrt()
    use = (count >= 2) & (std > eps)
    out = torch.where(use & selected, (groups - mean) / (std + eps), 0.0)
    return out.reshape_as(quality)



def _groups(values: Tensor, group_size: int) -> Tensor:
    if not values.is_floating_point() or values.ndim != 1:
        raise ValueError("values must be a floating tensor with shape [responses]")
    if group_size <= 0 or values.numel() % group_size:
        raise ValueError("group_size must be positive and divide the response count")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    return values.reshape(-1, group_size)
