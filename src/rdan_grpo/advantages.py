"""Group-relative advantage normalization.

RDAN normalizes two channels over the same groups but different subsets: the outcome channel
over responses with a well-defined outcome, the process channel over hard-passing responses
only. Both are the same masked standardization, so they share one implementation.
"""

import torch
from torch import Tensor


def group_advantages(
    values: Tensor,
    group_size: int,
    valid: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Standardize values within contiguous groups over the selected subset.

    Unselected responses receive zero, as do groups with fewer than two selected values or no
    spread, since neither carries a usable learning signal.
    """

    groups = _groups(values, group_size)
    if valid is None:
        selected = torch.ones_like(groups, dtype=torch.bool)
    elif valid.dtype != torch.bool or valid.shape != values.shape:
        raise ValueError("valid must be boolean and match values")
    else:
        selected = valid.reshape_as(groups)

    mask = selected.to(values.dtype)
    count = selected.sum(dim=-1, keepdim=True)
    mean = (groups * mask).sum(dim=-1, keepdim=True) / count.clamp_min(1)
    # Sample standard deviation over the selected subset, matching GRPO's group normalization.
    variance = ((groups - mean).square() * mask).sum(dim=-1, keepdim=True) / (count - 1).clamp_min(1)
    std = variance.sqrt()
    out = torch.where((count >= 2) & (std > eps) & selected, (groups - mean) / (std + eps), 0.0)
    return out.reshape_as(values)


def _groups(values: Tensor, group_size: int) -> Tensor:
    if not values.is_floating_point() or values.ndim != 1:
        raise ValueError("values must be a floating tensor with shape [responses]")
    if group_size <= 0 or values.numel() % group_size:
        raise ValueError("group_size must be positive and divide the response count")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    return values.reshape(-1, group_size)
