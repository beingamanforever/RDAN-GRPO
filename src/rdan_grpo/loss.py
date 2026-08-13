"""Policy optimization losses."""

import torch
from torch import Tensor


def clipped_actor_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    token_mask: Tensor,
    clip_low: float,
    clip_high: float | None = None,
) -> Tensor:
    """Return RTT's asymmetric PPO loss with per-sequence token means."""

    if log_probs.shape != old_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError("log_probs, old_log_probs, and advantages must have matching shapes")
    if token_mask.dtype != torch.bool or token_mask.shape != log_probs.shape:
        raise ValueError("token_mask must be boolean and match log_probs")
    if (
        not log_probs.is_floating_point()
        or not old_log_probs.is_floating_point()
        or not advantages.is_floating_point()
    ):
        raise ValueError("loss inputs must be floating tensors")
    if clip_low < 0 or (clip_high is not None and clip_high < 0):
        raise ValueError("clip bounds must be non-negative")
    if not bool(token_mask.any()):
        raise ValueError("token_mask must select at least one token")
    active = token_mask
    finite = torch.isfinite(log_probs) & torch.isfinite(old_log_probs) & torch.isfinite(advantages)
    if not bool((finite | ~active).all()):
        raise ValueError("active loss inputs must be finite")

    high = clip_low if clip_high is None else clip_high
    log_ratio = torch.where(active, log_probs - old_log_probs, 0.0)
    ratio = torch.exp(log_ratio)
    if not bool((torch.isfinite(ratio) | ~active).all()):
        raise ValueError("active importance ratios must be finite")
    advantage = torch.where(active, advantages.detach(), 0.0)
    raw = ratio * advantage
    clipped = ratio.clamp(1 - clip_low, 1 + high) * advantage
    if not bool((torch.isfinite(raw) & torch.isfinite(clipped) | ~active).all()):
        raise ValueError("active policy surrogates must be finite")
    loss = torch.where(active, -torch.minimum(raw, clipped), 0.0)
    count = active.sum(dim=-1)
    sequence_loss = loss.sum(dim=-1) / count.clamp_min(1)
    out = sequence_loss[count > 0].mean()
    if not bool(torch.isfinite(out)):
        raise ValueError("actor loss is not finite")
    return out
