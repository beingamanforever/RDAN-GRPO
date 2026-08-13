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


def token_advantages(
    values: Tensor,
    rubric_mask: Tensor,
    token_mask: Tensor,
    valid: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Normalize each response-rubric token sequence, then average active rubrics.

    ``values`` contains reward-weighted token relevance with shape ``[responses, rubrics, tokens]``.
    Statistics use only unpadded tokens and population standard deviation.
    """

    if not values.is_floating_point() or values.ndim != 3:
        raise ValueError("values must be a floating tensor with shape [responses, rubrics, tokens]")
    if rubric_mask.dtype != torch.bool or rubric_mask.shape != values.shape[:2]:
        raise ValueError("rubric_mask must be boolean with shape [responses, rubrics]")
    if token_mask.dtype != torch.bool or token_mask.shape != (values.shape[0], values.shape[2]):
        raise ValueError("token_mask must be boolean with shape [responses, tokens]")
    if valid.dtype != torch.bool or valid.shape != (values.shape[0],):
        raise ValueError("valid must be boolean with shape [responses]")
    active_rubric = rubric_mask & valid.unsqueeze(-1)
    active = active_rubric.unsqueeze(-1) & token_mask.unsqueeze(1)
    if not bool((torch.isfinite(values) | ~active).all()):
        raise ValueError("active token values must be finite")

    safe = torch.where(active, values, 0.0)
    mask = active.to(values.dtype)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = safe.sum(dim=-1, keepdim=True) / count
    var = ((safe - mean).square() * mask).sum(dim=-1, keepdim=True) / count
    std = var.sqrt()
    norm = torch.where((std > eps) & active, (values - mean) / (std + eps), 0.0)
    rubrics = active_rubric.sum(dim=-1, keepdim=True).clamp_min(1)
    return norm.sum(dim=1) / rubrics * token_mask.to(values.dtype)


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


def compose_advantages(
    response: Tensor,
    token: Tensor,
    quality: Tensor,
    token_mask: Tensor,
    beta: float = 1.0,
    quality_weight: float = 1.0,
) -> Tensor:
    """Compose ``A_res + beta * A_tok + quality_weight * A_qual`` on response tokens."""

    if response.ndim != 1 or quality.shape != response.shape:
        raise ValueError("response and quality must have matching shape [responses]")
    if token.ndim != 2 or token.shape[0] != response.shape[0]:
        raise ValueError("token must have shape [responses, tokens]")
    if token_mask.dtype != torch.bool or token_mask.shape != token.shape:
        raise ValueError("token_mask must be boolean and match token")
    if not response.is_floating_point() or not token.is_floating_point() or not quality.is_floating_point():
        raise ValueError("advantages must be floating tensors")
    out = response.unsqueeze(-1) + beta * token + quality_weight * quality.unsqueeze(-1)
    return out * token_mask.to(out.dtype)


def _groups(values: Tensor, group_size: int) -> Tensor:
    if not values.is_floating_point() or values.ndim != 1:
        raise ValueError("values must be a floating tensor with shape [responses]")
    if group_size <= 0 or values.numel() % group_size:
        raise ValueError("group_size must be positive and divide the response count")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    return values.reshape(-1, group_size)
