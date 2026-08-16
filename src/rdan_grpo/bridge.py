"""Adapter between ROLL's batch protocol and the RDAN advantage computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

import torch
from torch import Tensor

from rdan_grpo.scalar import ScalarMethod, ScalarOutput, build_scalar_output

# Written by the reward worker, read here. Everything else on the batch is ROLL's.
REWARD_FIELDS = ("rdan_scores", "rdan_rubric_mask", "rdan_eval_mask", "rdan_hard_mask")
# Written here for the training curves; none of these feed back into the loss.
ADVANTAGE_FIELDS = (
    "rdan_raw_aon",
    "rdan_raw_csr",
    "rdan_selected_reward",
    "rdan_response_advantage",
    "rdan_raw_quality",
    "rdan_quality_eligible",
    "rdan_quality_advantage",
    "rdan_scalar_advantage",
    "rdan_response_valid",
    "rdan_hard_pass",
)


def make_advantage_fn(
    method: ScalarMethod,
    group_size: int,
    quality_weight: float | None = None,
) -> Callable[..., Any]:
    """Build a drop-in replacement for ROLL's ``compute_advantage`` boundary."""

    def compute_advantage(data: Any, *, response_mask: Tensor | None = None, **_: Any) -> Any:
        output = assess_batch(data, method, group_size, quality_weight)
        mask = response_mask if response_mask is not None else data.batch.get("final_response_mask")
        if not isinstance(mask, Tensor):
            raise ValueError("ROLL data is missing final_response_mask")
        return inject_advantages(data, output, mask.to(dtype=torch.bool))

    return compute_advantage


def assess_batch(
    data: Any,
    method: ScalarMethod,
    group_size: int,
    quality_weight: float | None = None,
) -> ScalarOutput:
    """Read the reward worker's fields off a ROLL batch and build the scalar advantage."""

    batch = getattr(data, "batch", None)
    non_tensor = getattr(data, "non_tensor_batch", None)
    if not isinstance(batch, MutableMapping) or not isinstance(non_tensor, Mapping):
        raise ValueError("ROLL data must expose batch and non_tensor_batch mappings")
    missing = [name for name in REWARD_FIELDS if name not in batch]
    if "rdan_prompt_key" not in non_tensor:
        missing.append("rdan_prompt_key")
    if missing:
        raise ValueError(f"ROLL reward output is missing fields: {', '.join(missing)}")
    return build_scalar_output(
        method,
        list(non_tensor["rdan_prompt_key"]),
        batch["rdan_scores"].float(),
        batch["rdan_rubric_mask"].bool(),
        batch["rdan_eval_mask"].bool(),
        batch["rdan_hard_mask"].bool(),
        group_size=group_size,
        quality_weight=quality_weight,
    )


def inject_advantages(data: Any, output: ScalarOutput, response_mask: Tensor) -> Any:
    """Expand the scalar advantage over response tokens without renormalizing."""

    if response_mask.dtype != torch.bool or response_mask.ndim != 2:
        raise ValueError("response_mask must be boolean with shape [responses, tokens]")
    if response_mask.shape[0] != output.scalar_advantage.numel():
        raise ValueError("response_mask must contain one row per scalar advantage")
    if not bool(response_mask.any(dim=-1).all()):
        raise ValueError("every response must contain at least one active token")

    batch = data if isinstance(data, MutableMapping) else data.batch
    scalar = output.scalar_advantage.to(device=response_mask.device)
    token_advantage = scalar.unsqueeze(-1) * response_mask.to(scalar.dtype)
    channels = {
        "rdan_raw_aon": output.raw_aon,
        "rdan_raw_csr": output.raw_csr,
        "rdan_selected_reward": output.selected_raw_reward,
        "rdan_response_advantage": output.response_advantage,
        "rdan_raw_quality": output.raw_quality,
        "rdan_quality_eligible": output.quality_eligible,
        "rdan_quality_advantage": output.quality_advantage,
        "rdan_scalar_advantage": output.scalar_advantage,
        "rdan_response_valid": output.response_valid,
        "rdan_hard_pass": output.hard_pass,
    }
    batch.update({name: value.to(response_mask.device).clone() for name, value in channels.items()})
    batch["advantages"] = token_advantage.clone()
    # Response-only RDAN has no value model, so returns are the advantages.
    batch["returns"] = token_advantage.clone()
    return data
