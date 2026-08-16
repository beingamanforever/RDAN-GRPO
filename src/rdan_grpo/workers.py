"""ROLL workers for FSDP2 response training and deterministic vLLM rollout."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import InferWorker
from roll.pipeline.rlvr.actor_worker import ActorWorker

_COUNTERS = "_rdan_counters"
_CLIP = "_rdan_clip_fractions"
_STEP = "_rdan_generation_step"
_ORDINAL = "_rdan_generation_ordinal"
CLIP_METRIC = "rdan/response_token_clipfrac"


class ResponseActorWorker(ActorWorker):
    """Train through ROLL RLVR while tracking optimizer state for checkpoint and resume."""

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Record the PPO clipping fraction alongside the actor loss."""

        loss, metrics = super().loss_func(data, output_tensor)
        value = _clip_fraction(self, data, output_tensor)
        metrics[CLIP_METRIC] = value
        getattr(self, _CLIP).append(value)
        return loss, metrics

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto) -> DataProto:
        """Count optimizer steps and skip, rather than abort, on a non-finite gradient.

        A non-finite gradient is a normal event over thousands of RL updates. Skipping the
        update and recording it keeps the run alive; ``rdan/skipped_updates`` shows how often
        it happens so a genuine divergence is still visible on the curves.
        """

        strategy = getattr(self, "strategy", None)
        optimizer = getattr(strategy, "optimizer", None)
        scheduler = getattr(strategy, "scheduler", None)
        if optimizer is None or scheduler is None:
            raise RuntimeError("response training requires an initialized optimizer and scheduler")
        counters = _counters(self)
        setattr(self, _CLIP, [])
        optimizer_step, scheduler_step = optimizer.step, scheduler.step

        def handle_optimizer_step(*args: Any, **kwargs: Any) -> Any:
            if not _all_ranks_grad_finite(strategy):
                counters["skipped_updates"] += 1
                optimizer.zero_grad(set_to_none=True)
                return None
            result = optimizer_step(*args, **kwargs)
            counters["optimizer_steps"] += 1
            return result

        def handle_scheduler_step(*args: Any, **kwargs: Any) -> Any:
            result = scheduler_step(*args, **kwargs)
            counters["scheduler_steps"] += 1
            return result

        optimizer.step, scheduler.step = handle_optimizer_step, handle_scheduler_step
        try:
            output = super().train_step(data)
        finally:
            optimizer.step, scheduler.step = optimizer_step, scheduler_step
        output.meta_info["rdan_counters"] = {"rank": _rank(self), **counters}
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_counters(self) -> dict[str, int]:
        """Return rank-local optimizer, scheduler, and skipped-update counters."""

        return {"rank": _rank(self), **_counters(self)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_save(self, checkpoint_dir: str, pipeline_step: int) -> dict[str, Any]:
        """Save sharded DCP state, Hugging Face weights, and the counter sidecar."""

        root = Path(checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        strategy = self.strategy
        manager = getattr(strategy, "checkpoint_manager", None)
        strategy.checkpoint_manager = None
        try:
            # ROLL writes DCP shards plus rank-0 safetensors and the tokenizer in one call.
            strategy.save_checkpoint(
                save_dir=str(root),
                global_step=max(pipeline_step - 1, 0),
                ckpt_id=f"step-{pipeline_step:06d}",
                is_last_step=True,
            )
        finally:
            strategy.checkpoint_manager = manager
        _counter_path(root, _rank(self)).write_text(
            json.dumps({"pipeline_step": pipeline_step, "counters": _counters(self)}, sort_keys=True),
            encoding="utf-8",
        )
        return {"rank": _rank(self), "pipeline_step": pipeline_step}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_load(self, checkpoint_dir: str) -> dict[str, Any]:
        """Restore sharded DCP state and the counter sidecar."""

        root = Path(checkpoint_dir)
        metadata = json.loads(_counter_path(root, _rank(self)).read_text(encoding="utf-8"))
        strategy = self.strategy
        strategy.load_states()
        try:
            _prime_optimizer_state(strategy)
            strategy._load_checkpoint_with_dcp(checkpoint_dir=strategy._get_dcp_checkpoint_dir(str(root)))
        finally:
            strategy.offload_states()
        _require_restored_moments(strategy)
        setattr(self, _COUNTERS, dict(metadata["counters"]))
        return {"rank": _rank(self), "pipeline_step": int(metadata["pipeline_step"])}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_cuda_peak(self) -> None:
        """Reset rank-local CUDA peak memory statistics."""

        torch.cuda.reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_release_cache(self) -> dict[str, int]:
        """Return cached-but-unused GPU blocks to the driver.

        Offloading moves tensors to CPU but leaves the freed segments in PyTorch's caching
        allocator, which still owns the physical pages. vLLM maps its KV pool with the CUDA
        virtual memory API and fails outright when those pages are unavailable, so the actor
        must release them before the rollout engine wakes on the same device.
        """

        return _release_cache(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_cuda_memory(self) -> dict[str, int]:
        """Return rank-local CUDA peak and capacity bytes."""

        return _cuda_memory(self)


class ResponseVLLMInferWorker(InferWorker):
    """Generate vLLM rollouts under a seed derived from the pipeline step."""

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def generate(self, data: DataProto) -> DataProto:
        """Seed each generation call distinctly so repeated calls within a step differ."""

        from rdan_grpo.compat import install_vllm_sampling_seed_compat

        install_vllm_sampling_seed_compat()
        step = data.meta_info.get("global_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise RuntimeError(f"vLLM generation requires a positive integer global_step, received {step!r}")
        ordinal = getattr(self, _ORDINAL, -1) + 1 if step == getattr(self, _STEP, 0) else 0
        rank = _rank(self)

        request = data.clone()
        generation_config = copy.deepcopy(request.meta_info.get("generation_config"))
        if not isinstance(generation_config, dict):
            raise RuntimeError("vLLM generation config is unavailable")
        generation_config["seed"] = _generation_seed(_base_seed(self), step, ordinal, rank)
        request.meta_info["generation_config"] = generation_config

        output = await super().generate(request)
        output.non_tensor_batch["generation_id"] = np.asarray(
            [f"gen-{step:06d}-r{rank}-c{ordinal:04d}-{index:012d}" for index in range(len(output))],
            dtype=object,
        )
        setattr(self, _STEP, step)
        setattr(self, _ORDINAL, ordinal)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_cuda_peak(self) -> None:
        """Reset rank-local CUDA peak memory statistics."""

        torch.cuda.reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_release_cache(self) -> dict[str, int]:
        """Return cached-but-unused GPU blocks so the colocated trainer can use them."""

        return _release_cache(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_cuda_memory(self) -> dict[str, int]:
        """Return rank-local CUDA peak and capacity bytes."""

        return _cuda_memory(self)


def _release_cache(worker: Any) -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    return {
        "rank": _rank(worker),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "allocated_bytes": torch.cuda.memory_allocated(device),
    }


def _counters(worker: Any) -> dict[str, int]:
    counters = getattr(worker, _COUNTERS, None)
    if counters is None:
        counters = {"optimizer_steps": 0, "scheduler_steps": 0, "skipped_updates": 0}
        setattr(worker, _COUNTERS, counters)
    return counters


def _counter_path(root: Path, rank: int) -> Path:
    return root / f"rdan-counters-rank-{rank}.json"


def _clip_fraction(worker: Any, data: DataProto, output_tensor: torch.Tensor) -> float:
    """Fraction of active response tokens whose importance ratio left the clip range."""

    batch = data.batch
    mask = batch["final_response_mask"].to(torch.bool)
    old_log_probs = batch["old_log_probs"]
    with torch.no_grad():
        log_probs = worker.strategy.op_compute_log_probs(
            logits=output_tensor.detach(),
            input_ids=batch["input_ids"],
            attention_mask=batch["response_mask"],
        )
    log_ratio = log_probs - old_log_probs
    if getattr(worker.pipeline_config, "importance_sampling", None) == "seq":
        mean = (log_ratio * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        ratio = mean.exp().unsqueeze(-1).expand_as(log_ratio)
    else:
        ratio = log_ratio.exp()
    low, high = _clip_bounds(worker.pipeline_config)
    clipped = ((ratio < 1 - low) | (ratio > 1 + high)) & mask
    return float(clipped.sum().div(mask.sum().clamp_min(1)).item())


def _clip_bounds(config: Any) -> tuple[float, float]:
    if getattr(config, "use_pg_clip_range", None) is True:
        return float(config.pg_clip_low), float(config.pg_clip_high)
    return float(config.pg_clip), float(config.pg_clip)


def _all_ranks_grad_finite(strategy: Any) -> bool:
    """Return whether every rank sees finite gradients, so all ranks skip or step together."""

    grads = [
        parameter.grad
        for parameter in strategy.model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite = bool(grads) and all(_grad_is_finite(gradient) for gradient in grads)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return finite
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    signal = torch.tensor(int(finite), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(signal, op=torch.distributed.ReduceOp.MIN)
    return bool(signal.item())


def _grad_is_finite(gradient: torch.Tensor) -> bool:
    to_local = getattr(gradient, "to_local", None)
    local = to_local() if callable(to_local) else gradient
    return bool(torch.isfinite(local).all().item())


def _prime_optimizer_state(strategy: Any) -> None:
    """Populate optimizer.state before a DCP load so moments have somewhere to land.

    A freshly built optimizer has never stepped, so its state_dict has an empty 'state'
    subtree. dcp.load plans against that template and would silently restore param_groups
    only, restarting Adam from zero moments. get_optimizer_state_dict initializes the
    moments first (DTensor-safe under FSDP2), giving the load real targets.
    """

    optimizer = getattr(strategy, "optimizer", None)
    if getattr(strategy, "model", None) is None or not isinstance(optimizer, torch.optim.Optimizer):
        return
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    get_optimizer_state_dict(strategy.model, optimizer)


def _require_restored_moments(strategy: Any) -> None:
    """Fail loudly if a resume silently produced an optimizer with no moment state."""

    optimizer = getattr(strategy, "optimizer", None)
    if isinstance(optimizer, torch.optim.Optimizer) and not optimizer.state:
        raise RuntimeError("resume restored no optimizer moment state")


def _base_seed(worker: Any) -> int:
    seed = getattr(getattr(worker, "pipeline_config", None), "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimeError("response training requires a non-negative pipeline seed")
    return seed


def _generation_seed(base_seed: int, step: int, ordinal: int, rank: int) -> int:
    value = f"{base_seed}:{step}:{ordinal}:{rank}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _rank(worker: Any) -> int:
    rank_info = getattr(worker, "rank_info", None)
    rank = getattr(rank_info, "dp_rank", getattr(worker, "rank", None))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise RuntimeError("response worker requires a non-negative DP rank")
    return rank


def _cuda_memory(worker: Any) -> dict[str, int]:
    device = torch.cuda.current_device()
    return {
        "rank": _rank(worker),
        "peak_bytes": max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)),
        "total_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
