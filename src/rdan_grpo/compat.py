"""Make the vendored RTT/ROLL checkout importable and repair the gaps RDAN depends on."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RTT_ROOT_ENV = "RTT_ROOT"
_SEED_PATCH_OWNER = "rdan-grpo:vllm-sampling-seed"


def rtt_root(explicit: str | Path | None = None) -> Path:
    """Resolve the RTT checkout that supplies ROLL, from the argument or the environment."""

    value = explicit or os.environ.get(RTT_ROOT_ENV)
    if not value:
        raise ValueError(f"--rtt-root or {RTT_ROOT_ENV} is required to locate the ROLL runtime")
    root = Path(value).resolve()
    if not (root / "roll").is_dir():
        raise ValueError(f"{root} does not contain a roll package")
    return root


def install_rtt_runtime(explicit: str | Path | None = None) -> Path:
    """Put the RTT checkout on the import path and supply the helper its rlvr package lacks."""

    root = rtt_root(explicit)
    for entry in (root, root / "mcore_adapter" / "src"):
        text = str(entry)
        if entry.is_dir() and text not in sys.path:
            sys.path.insert(0, text)

    # roll.pipeline.rlvr.rlvr_pipeline imports this name from roll.pipeline.rlvr.utils, which
    # never defines it, so the module is unimportable without the shim. RDAN does not use
    # ROLL's reward-system dump, and the config that would enable it is never set.
    from roll.pipeline.rlvr import utils

    if not hasattr(utils, "dump_batch_to_reward_system"):
        utils.dump_batch_to_reward_system = _reject_reward_system_dump
    return root


def _reject_reward_system_dump(batch: Any, tokenizer: Any) -> None:
    """Stand in for ROLL's missing hook, refusing only if a run actually enables it."""

    del tokenizer
    if getattr(batch, "meta_info", {}).get("reward_system_config"):
        raise RuntimeError("ROLL reward-system logging is unavailable in this checkout")


def install_vllm_sampling_seed_compat() -> None:
    """Forward the request seed that ROLL's vLLM adapter drops, so rollouts are reproducible."""

    from roll.distributed.strategy import vllm_strategy

    original = vllm_strategy.create_sampling_params_for_vllm
    if getattr(original, "__rdan_owner__", None) == _SEED_PATCH_OWNER:
        return

    def create_sampling_params_for_vllm(gen_kwargs: Mapping[str, Any]) -> Any:
        seed = gen_kwargs.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise RuntimeError("vLLM generation requires a non-negative deterministic seed")
        return vllm_strategy.SamplingParams(
            max_tokens=gen_kwargs["max_new_tokens"],
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            top_k=gen_kwargs["top_k"],
            stop_token_ids=gen_kwargs["eos_token_id"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            n=gen_kwargs["num_return_sequences"],
            stop=gen_kwargs["stop_strings"],
            logprobs=gen_kwargs.get("logprobs", 0),
            output_kind=gen_kwargs.get("output_kind", vllm_strategy.RequestOutputKind.FINAL_ONLY),
            include_stop_str_in_output=gen_kwargs.get("include_stop_str_in_output", True),
            seed=seed,
        )

    create_sampling_params_for_vllm.__rdan_owner__ = _SEED_PATCH_OWNER
    vllm_strategy.create_sampling_params_for_vllm = create_sampling_params_for_vllm
