"""Production RTT workers for response-level FSDP2 and Hugging Face training."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import InferWorker
from roll.pipeline.rlvr.actor_worker import ActorWorker

from rdan_grpo.fsdp_hf_receipt import FSDPHFReceiptError
from rdan_grpo.roll_fsdp_hf_receipt import (
    begin_fsdp_hf_receipt,
    begin_hf_infer_receipt,
    finish_hf_infer_receipt,
    get_fsdp_actor_receipt,
    reset_fsdp_hf_receipt,
    run_receipted_fsdp_hf_update,
)
from rdan_grpo.roll_response_config import UPDATES_PER_STEP
from rdan_grpo.roll_same_backend import SynchronousHFInferWorker

_COUNTER_ATTR = "_rdan_response_counters"
_CLIP_ATTR = "_rdan_response_clip_fractions"
_TRAIN_STATE_ATTR = "_rdan_response_train_state"
_GENERATION_COUNT_ATTR = "_rdan_generation_count"
_VLLM_GENERATION_STEP_ATTR = "_rdan_vllm_generation_step"
_VLLM_GENERATION_ORDINAL_ATTR = "_rdan_vllm_generation_ordinal"
RESPONSE_CLIP_METRIC = "rdan/response_token_clipfrac"


class ResponseActorWorker(ActorWorker):
    """Train through RTT RLVR while observing exact optimizer and receipt state."""

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Record the PPO clipping fraction emitted by the actor loss."""

        loss, metrics = super().loss_func(data, output_tensor)
        value = _response_clip_fraction(self, data, output_tensor)
        metrics[RESPONSE_CLIP_METRIC] = value
        getattr(self, _CLIP_ATTR).append(value)
        return loss, metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_begin_response_receipt(self, transaction_id: str) -> None:
        """Begin a named actor-to-inference weight receipt transaction."""

        begin_fsdp_hf_receipt(self, transaction_id, _rank(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_get_response_receipt(self) -> dict[str, Any]:
        """Return the completed actor side of the active weight receipt."""

        return get_fsdp_actor_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_response_receipt(self) -> dict[str, Any]:
        """Validate and clear the completed actor weight receipt."""

        return reset_fsdp_hf_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_train_counters(self) -> dict[str, int]:
        """Return exact rank-local optimizer and scheduler counters."""

        return {"rank": _rank(self), **_counters(self)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_training_state(self) -> dict[str, int | bool]:
        """Return rank-local optimizer safety state for the current step."""

        counters = _counters(self)
        state = _train_state(self)
        return {
            "rank": _rank(self),
            "optimizer_step": counters["optimizer_steps"],
            "scheduler_step": counters["scheduler_steps"],
            **state,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_save_dcp(self, checkpoint_dir: str, pipeline_step: int) -> dict[str, Any]:
        """Save rank-local distributed checkpoint state and exact counters."""

        return _save_dcp(self, checkpoint_dir, pipeline_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_load_dcp(self, checkpoint_dir: str) -> dict[str, Any]:
        """Restore rank-local distributed checkpoint state and counters."""

        return _load_dcp(self, checkpoint_dir)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_cuda_peak(self) -> None:
        """Reset rank-local CUDA peak memory statistics."""

        torch.cuda.reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_cuda_memory(self) -> dict[str, int]:
        """Return rank-local CUDA peak and capacity bytes."""

        return _cuda_memory(self)

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto) -> DataProto:
        """Count successful exact optimizer and scheduler calls around RTT training."""

        strategy = getattr(self, "strategy", None)
        optimizer = getattr(strategy, "optimizer", None)
        scheduler = getattr(strategy, "scheduler", None)
        if optimizer is None or scheduler is None:
            raise RuntimeError("response training requires initialized optimizer and scheduler")
        counters = _counters(self)
        state = _train_state(self)
        state.update(grad_finite=True, update_skipped=False)
        setattr(self, _CLIP_ATTR, [])
        optimizer_step = optimizer.step
        scheduler_step = scheduler.step
        handle_optimizer_step, handle_scheduler_step = _step_handlers(
            self,
            strategy,
            optimizer,
            counters,
            state,
            optimizer_step,
            scheduler_step,
        )
        optimizer.step = handle_optimizer_step
        scheduler.step = handle_scheduler_step
        try:
            output = super().train_step(data)
        finally:
            optimizer.step = optimizer_step
            scheduler.step = scheduler_step
        if counters["optimizer_steps"] != counters["scheduler_steps"]:
            raise RuntimeError("optimizer and scheduler counters differ after RTT training")
        if getattr(self, _CLIP_ATTR):
            raise RuntimeError("response training left clipping evidence without an optimizer boundary")
        output.meta_info["response_train_evidence"] = {"rank": _rank(self), **counters}
        return output

    def start_model_update(self, model_update_name: str) -> DataProto:
        """Run a named FSDP model update under backend-specific receipt tracking."""

        updater = getattr(getattr(self, "strategy", None), "weight_updaters", {}).get(model_update_name)
        strategy_name = getattr(
            getattr(getattr(updater, "infer_worker_config", None), "strategy_args", None),
            "strategy_name",
            None,
        )
        if strategy_name == "vllm":
            from rdan_grpo.roll_weight_receipt import run_receipted_fsdp_vllm_update

            return run_receipted_fsdp_vllm_update(
                self,
                model_update_name,
                lambda: super(ResponseActorWorker, self).start_model_update(model_update_name),
            )
        if strategy_name != "hf_infer":
            raise RuntimeError(f"response training does not support inference strategy {strategy_name}")
        return run_receipted_fsdp_hf_update(
            self,
            model_update_name,
            lambda: super(ResponseActorWorker, self).start_model_update(model_update_name),
        )


class ResponseInferWorker(SynchronousHFInferWorker):
    """Generate prompt-major microbatches and expose resumable inference state."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_begin_response_receipt(self, transaction_id: str) -> None:
        """Begin a named inference-side weight receipt transaction."""

        begin_hf_infer_receipt(self, transaction_id, _rank(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_finish_response_receipt(self) -> dict[str, Any]:
        """Finish and return the active inference-side weight receipt."""

        return finish_hf_infer_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_response_receipt(self) -> dict[str, Any]:
        """Validate and clear the completed inference-side receipt."""

        return reset_fsdp_hf_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_save_rng(self) -> dict[str, torch.Tensor]:
        """Snapshot exact CPU and CUDA inference RNG state."""

        if not torch.cuda.is_available():
            raise RuntimeError("response inference RNG requires CUDA")
        return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state().clone()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_load_rng(self, state: Mapping[str, torch.Tensor]) -> None:
        """Restore exact CPU and CUDA inference RNG state."""

        if not isinstance(state, Mapping) or set(state) != {"cpu", "cuda"}:
            raise RuntimeError("response inference RNG state is invalid")
        if any(not isinstance(state[name], torch.Tensor) or state[name].dtype != torch.uint8 for name in state):
            raise RuntimeError("response inference RNG tensors are invalid")
        torch.set_rng_state(state["cpu"].cpu())
        torch.cuda.set_rng_state(state["cuda"].cpu())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_cuda_peak(self) -> None:
        """Reset rank-local CUDA peak memory statistics."""

        torch.cuda.reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_cuda_memory(self) -> dict[str, int]:
        """Return rank-local CUDA peak and capacity bytes."""

        return _cuda_memory(self)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def generate(self, data: DataProto) -> DataProto:
        """Generate contiguous rank-local prompt microbatches in prompt-major order."""

        size = getattr(self.worker_config, "infer_batch_size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RuntimeError("response inference requires a positive infer_batch_size")
        generate = super().generate
        output = (
            generate(data)
            if len(data) <= size
            else DataProto.concat([generate(data[start : start + size]) for start in range(0, len(data), size)])
        )
        step = data.meta_info.get("global_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise RuntimeError("response inference requires a positive global step")
        start = getattr(self, _GENERATION_COUNT_ATTR, 0)
        rank = _rank(self)
        output.non_tensor_batch["generation_id"] = np.asarray(
            [f"gen-{step:06d}-r{rank}-{index:012d}" for index in range(start, start + len(output))],
            dtype=object,
        )
        setattr(self, _GENERATION_COUNT_ATTR, start + len(output))
        return output


class ResponseVLLMInferWorker(InferWorker):
    """Generate continuously batched vLLM rollouts with deterministic resume state."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def rdan_begin_response_receipt(self, transaction_id: str) -> None:
        """Begin one identity-paired receipt on the TP1 vLLM engine."""

        from rdan_grpo.roll_weight_receipt import begin_infer_weight_receipt

        await begin_infer_weight_receipt(self, transaction_id, _rank(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def rdan_finish_response_receipt(self) -> dict[str, Any]:
        """Finish and return the active TP1 vLLM loader receipt."""

        from rdan_grpo.roll_weight_receipt import get_infer_weight_receipt

        return _vllm_response_receipt(await get_infer_weight_receipt(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def rdan_reset_response_receipt(self) -> dict[str, Any]:
        """Validate and clear the completed TP1 vLLM loader receipt."""

        from rdan_grpo.roll_weight_receipt import reset_infer_weight_receipt

        return _vllm_response_receipt(await reset_infer_weight_receipt(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_save_rng(self) -> dict[str, int]:
        """Save deterministic vLLM generation progress for exact-step restart."""

        step = getattr(self, _VLLM_GENERATION_STEP_ATTR, 0)
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise RuntimeError("vLLM generation progress is unavailable")
        return {
            "schema_version": 1,
            "rank": _rank(self),
            "base_seed": _base_seed(self),
            "last_generation_step": step,
            "last_generation_ordinal": getattr(self, _VLLM_GENERATION_ORDINAL_ATTR, 0),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_load_rng(self, state: Mapping[str, int]) -> None:
        """Restore deterministic vLLM progress before the next pipeline step."""

        if not isinstance(state, Mapping) or set(state) != {
            "schema_version",
            "rank",
            "base_seed",
            "last_generation_step",
            "last_generation_ordinal",
        }:
            raise RuntimeError("vLLM generation progress is invalid")
        expected = {"schema_version": 1, "rank": _rank(self), "base_seed": _base_seed(self)}
        if any(state.get(name) != value for name, value in expected.items()):
            raise RuntimeError("vLLM generation progress identity differs")
        step = state.get("last_generation_step")
        ordinal = state.get("last_generation_ordinal")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 1
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
        ):
            raise RuntimeError("vLLM generation progress step is invalid")
        setattr(self, _VLLM_GENERATION_STEP_ATTR, step)
        setattr(self, _VLLM_GENERATION_ORDINAL_ATTR, ordinal)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_cuda_peak(self) -> None:
        """Reset rank-local CUDA peak memory statistics."""

        torch.cuda.reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_cuda_memory(self) -> dict[str, int]:
        """Return rank-local CUDA peak and capacity bytes."""

        return _cuda_memory(self)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def generate(self, data: DataProto) -> DataProto:
        """Generate one seeded pipeline step and attach stable rank-local IDs."""

        if self.worker_config.strategy_args.strategy_name != "vllm":
            raise RuntimeError("ResponseVLLMInferWorker requires the vLLM strategy")
        step = data.meta_info.get("global_step")
        previous_step = getattr(self, _VLLM_GENERATION_STEP_ATTR, 0)
        previous_ordinal = getattr(self, _VLLM_GENERATION_ORDINAL_ATTR, -1)
        if isinstance(step, bool) or not isinstance(step, int):
            raise RuntimeError(f"vLLM generation requires an integer global_step, received {step!r}")
        if step not in {previous_step, previous_step + 1}:
            raise RuntimeError(f"vLLM generation step {step} is not contiguous with restored progress {previous_step}")
        if step == 0 or (step == previous_step and previous_step == 0):
            raise RuntimeError("vLLM generation requires a positive pipeline step")
        ordinal = previous_ordinal + 1 if step == previous_step else 0
        from rdan_grpo.roll_compat import install_vllm_sampling_seed_compat

        install_vllm_sampling_seed_compat()
        request = data.clone()
        generation_config = copy.deepcopy(request.meta_info.get("generation_config"))
        if not isinstance(generation_config, dict):
            raise RuntimeError("vLLM generation config is unavailable")
        generation_config["seed"] = _generation_seed(_base_seed(self), step, ordinal, _rank(self))
        request.meta_info["generation_config"] = generation_config
        output = await super().generate(request)
        metrics = _vllm_engine_metrics(getattr(self, "strategy", None))
        rank = _rank(self)
        # Rank-tagged non_tensor columns cannot carry these: check_consistency requires every
        # column to be dtype object and exactly batch length, and concat keeps only the keys
        # every rank supplied. Concat likewise keeps rank zero's meta_info, so the curves
        # describe engine zero and _rank identifies which engine reported them.
        output.meta_info["vllm_metrics"] = {"vllm/rank": float(rank), **metrics}
        output.non_tensor_batch["generation_id"] = np.asarray(
            [f"gen-{step:06d}-r{rank}-c{ordinal:04d}-{index:012d}" for index in range(len(output))],
            dtype=object,
        )
        setattr(self, _VLLM_GENERATION_STEP_ATTR, step)
        setattr(self, _VLLM_GENERATION_ORDINAL_ATTR, ordinal)
        return output


_VLLM_METRICS_STATUS_READER_MISSING = 0.0
_VLLM_METRICS_STATUS_READER_RAISED = 1.0
_VLLM_METRICS_STATUS_READER_EMPTY = 2.0
_VLLM_METRICS_STATUS_POPULATED = 3.0


def _vllm_engine_metrics(strategy: Any) -> dict[str, Any]:
    """Return the rollout engine metrics vLLM aggregated since the previous generation.

    ``vllm/metrics_available`` reflects whether any metric was actually returned, so an
    empty read never reports the same value as a populated one. ``vllm/metrics_status``
    additionally distinguishes a missing reader, a reader that raised, an empty read, and
    a populated read as four distinct failure modes, each with its own numeric code so a
    reader failure is never collapsed into a healthy-looking flat curve.
    """

    reader = getattr(strategy, "get_metrics", None)
    if reader is None:
        return {
            "vllm/metrics_available": 0.0,
            "vllm/metrics_status": _VLLM_METRICS_STATUS_READER_MISSING,
            "vllm/metrics_count": 0.0,
        }
    try:
        metrics = reader() or {}
    except Exception as error:
        return {
            "vllm/metrics_available": 0.0,
            "vllm/metrics_status": _VLLM_METRICS_STATUS_READER_RAISED,
            "vllm/metrics_count": 0.0,
            "vllm/metrics_error": type(error).__name__,
        }
    values = {name: float(value) for name, value in metrics.items() if isinstance(value, (int, float))}
    return {
        "vllm/metrics_available": float(bool(values)),
        "vllm/metrics_status": _VLLM_METRICS_STATUS_POPULATED if values else _VLLM_METRICS_STATUS_READER_EMPTY,
        "vllm/metrics_count": float(len(values)),
        **values,
    }


def _vllm_response_receipt(value: Any) -> dict[str, Any]:
    values = value if isinstance(value, list) else [value]
    if len(values) != 1 or not isinstance(values[0], Mapping):
        raise RuntimeError("response receipt requires exactly one vLLM TP rank")
    snapshot = values[0]
    loader = snapshot.get("loader")
    if not isinstance(loader, Mapping) or loader.get("loaded") is not True:
        raise RuntimeError("response receipt requires a completed vLLM loader transaction")
    return {
        name: snapshot.get(name)
        for name in (
            "transaction_id",
            "side",
            "rank",
            "paired_rank",
            "accelerator_name",
            "stream_started",
            "stream_complete",
            "items",
            "tensor_count",
            "total_bytes",
            "manifest_sha256",
            "loader",
        )
    } | {"backend": "vllm"}


def _base_seed(worker: Any) -> int:
    seed = getattr(getattr(worker, "pipeline_config", None), "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimeError("response training requires a nonnegative pipeline seed")
    return seed


def _generation_seed(base_seed: int, step: int, ordinal: int, rank: int) -> int:
    value = f"{base_seed}:{step}:{ordinal}:{rank}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _step_handlers(
    worker: Any,
    strategy: Any,
    optimizer: Any,
    counters: dict[str, int],
    state: dict[str, bool],
    optimizer_step: Callable[..., Any],
    scheduler_step: Callable[..., Any],
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    def handle_optimizer_step(*args: Any, **kwargs: Any) -> Any:
        """Validate and count one optimizer step."""

        fractions = getattr(worker, _CLIP_ATTR)
        if not fractions:
            raise RuntimeError("response training did not report PPO clipping evidence")
        if not _all_ranks_grad_finite(strategy):
            _block_update(counters, state, optimizer, finite=False)
            raise RuntimeError("response training blocked a non-finite gradient update")
        if _all_ranks_fully_clipped(fractions):
            _block_update(counters, state, optimizer, finite=True)
            raise RuntimeError("response training blocked a fully clipped optimizer update")
        result = optimizer_step(*args, **kwargs)
        fractions.clear()
        counters["optimizer_steps"] += 1
        counters["finite_steps"] += 1
        return result

    def handle_scheduler_step(*args: Any, **kwargs: Any) -> Any:
        """Advance the scheduler only after a successful optimizer step."""

        if counters["optimizer_steps"] <= counters["scheduler_steps"]:
            counters["skipped_optimizer_steps"] += 1
            state.update(grad_finite=False, update_skipped=True)
            raise RuntimeError("RTT attempted to advance the scheduler without a successful optimizer step")
        result = scheduler_step(*args, **kwargs)
        counters["scheduler_steps"] += 1
        return result

    return handle_optimizer_step, handle_scheduler_step


def _response_clip_fraction(worker: Any, data: DataProto, output_tensor: torch.Tensor) -> float:
    batch = getattr(data, "batch", None)
    if not isinstance(batch, Mapping):
        raise RuntimeError("response clipping requires the training tensor batch")
    mask = batch.get("final_response_mask")
    old_log_probs = batch.get("old_log_probs")
    log_probs = _current_log_probs(worker, batch, output_tensor)
    if not _aligned_clip_tensors(log_probs, old_log_probs, mask):
        raise RuntimeError("response clipping tensors are missing or misaligned")
    active = mask.to(torch.bool)
    ratio = _importance_ratio(worker, log_probs, old_log_probs, active)
    low, high = _clip_bounds(worker)
    clipped = ((ratio < 1 - low) | (ratio > 1 + high)) & active
    return float(clipped.sum().div(active.sum()).item())


def _current_log_probs(worker: Any, batch: Mapping[str, Any], output_tensor: torch.Tensor) -> torch.Tensor:
    input_ids = batch.get("input_ids")
    response_mask = batch.get("response_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
        raise RuntimeError("response clipping requires input IDs and response mask")
    with torch.no_grad():
        return worker.strategy.op_compute_log_probs(
            logits=output_tensor.detach(),
            input_ids=input_ids,
            attention_mask=response_mask,
        )


def _aligned_clip_tensors(log_probs: Any, old_log_probs: Any, mask: Any) -> bool:
    return bool(
        isinstance(log_probs, torch.Tensor)
        and isinstance(old_log_probs, torch.Tensor)
        and isinstance(mask, torch.Tensor)
        and log_probs.shape == old_log_probs.shape == mask.shape
        and mask.to(torch.bool).any()
        and torch.isfinite(log_probs).all()
        and torch.isfinite(old_log_probs).all()
    )


def _importance_ratio(
    worker: Any,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_ratio = log_probs - old_log_probs
    mode = getattr(worker.pipeline_config, "importance_sampling", None)
    if mode == "token":
        return log_ratio.exp()
    if mode == "seq":
        mean = (log_ratio * mask).sum(dim=-1) / mask.sum(dim=-1)
        return mean.exp().unsqueeze(-1).expand_as(log_ratio)
    raise RuntimeError("response clipping requires token or sequence importance sampling")


def _clip_bounds(worker: Any) -> tuple[float, float]:
    config = worker.pipeline_config
    ranged = getattr(config, "use_pg_clip_range", None)
    low = getattr(config, "pg_clip_low", None) if ranged is True else getattr(config, "pg_clip", None)
    high = getattr(config, "pg_clip_high", None) if ranged is True else getattr(config, "pg_clip", None)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in (low, high)):
        raise RuntimeError("response clipping bounds are invalid")
    return float(low), float(high)


def _block_update(counters: dict[str, int], state: dict[str, bool], optimizer: Any, *, finite: bool) -> None:
    counters["skipped_optimizer_steps"] += 1
    state.update(grad_finite=finite, update_skipped=True)
    optimizer.zero_grad(set_to_none=True)


def _counters(worker: Any) -> dict[str, int]:
    counters = getattr(worker, _COUNTER_ATTR, None)
    if counters is None:
        counters = {
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "finite_steps": 0,
            "skipped_optimizer_steps": 0,
        }
        setattr(worker, _COUNTER_ATTR, counters)
    return counters


def _train_state(worker: Any) -> dict[str, bool]:
    state = getattr(worker, _TRAIN_STATE_ATTR, None)
    if state is None:
        state = {"grad_finite": True, "update_skipped": False}
        setattr(worker, _TRAIN_STATE_ATTR, state)
    return state


def _all_ranks_fully_clipped(values: list[float]) -> bool:
    clipped = all(value >= 1.0 for value in values)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return clipped
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    signal = torch.tensor(int(clipped), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(signal, op=torch.distributed.ReduceOp.MIN)
    return bool(signal.item())


def _all_ranks_grad_finite(strategy: Any) -> bool:
    model = getattr(strategy, "model", None)
    if model is None or not callable(getattr(model, "parameters", None)):
        raise RuntimeError("response training requires initialized model parameters")
    grads = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None
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


def _save_dcp(worker: Any, checkpoint_dir: str, pipeline_step: int) -> dict[str, Any]:
    root = _checkpoint_dir(checkpoint_dir, create=True)
    if isinstance(pipeline_step, bool) or not isinstance(pipeline_step, int) or pipeline_step < 0:
        raise RuntimeError("response checkpoint pipeline step is invalid")
    counters = _counters(worker)
    if not _valid_checkpoint_state(pipeline_step, counters):
        raise RuntimeError("response checkpoint counters are inconsistent with pipeline step")
    path = _counter_path(root, _rank(worker))
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    strategy = worker.strategy
    manager = getattr(strategy, "checkpoint_manager", None)
    strategy.checkpoint_manager = None
    try:
        strategy.save_checkpoint(
            save_dir=str(root),
            global_step=max(pipeline_step - 1, 0),
            ckpt_id=f"response-step-{pipeline_step:06d}",
            is_last_step=True,
        )
    finally:
        strategy.checkpoint_manager = manager
    metadata = {"pipeline_step": pipeline_step, "counters": counters}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"checkpoint_dir": str(root), "pipeline_step": pipeline_step, "rank": _rank(worker)}


def _load_dcp(worker: Any, checkpoint_dir: str) -> dict[str, Any]:
    root = _checkpoint_dir(checkpoint_dir, create=False)
    path = _counter_path(root, _rank(worker))
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("response checkpoint counter evidence is unavailable") from error
    counters = metadata.get("counters") if isinstance(metadata, Mapping) else None
    pipeline_step = metadata.get("pipeline_step") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != {"pipeline_step", "counters"}
        or isinstance(pipeline_step, bool)
        or not isinstance(pipeline_step, int)
        or pipeline_step < 0
        or not _valid_checkpoint_state(pipeline_step, counters)
    ):
        raise RuntimeError("response checkpoint counter evidence is invalid")
    strategy = worker.strategy
    dcp_dir = strategy._get_dcp_checkpoint_dir(str(root))
    strategy.load_states()
    try:
        _prime_optimizer_state(strategy)
        strategy._load_checkpoint_with_dcp(checkpoint_dir=dcp_dir)
    finally:
        strategy.offload_states()
    _validate_restored_optimizer_state(strategy, counters["optimizer_steps"])
    setattr(worker, _COUNTER_ATTR, dict(counters))
    setattr(worker, _TRAIN_STATE_ATTR, {"grad_finite": True, "update_skipped": False})
    return {"checkpoint_dir": str(root), "pipeline_step": pipeline_step, "rank": _rank(worker)}


def _prime_optimizer_state(strategy: Any) -> None:
    """Populate optimizer.state before DCP load so it has FQNs to load moments into.

    A freshly initialized optimizer has never called .step(), so optimizer.state_dict() ==
    {'state': {}, 'param_groups': [...]}. dcp.load builds its read plan from that template,
    so with an empty 'state' subtree it has no target key for the saved exp_avg/exp_avg_sq/
    step and silently restores param_groups only, restarting Adam from zero moments.
    get_optimizer_state_dict's _init_optim_state side effect (DTensor-safe under FSDP2)
    populates optimizer.state with zeroed moments first, giving dcp.load somewhere to read
    into. Only a genuine torch.optim.Optimizer exposes this machinery; anything else (never
    produced outside tests) is left untouched rather than crashed on.
    """

    optimizer = getattr(strategy, "optimizer", None)
    model = getattr(strategy, "model", None)
    if model is None or not isinstance(optimizer, torch.optim.Optimizer):
        return
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    get_optimizer_state_dict(model, optimizer)


def _validate_restored_optimizer_state(strategy: Any, expected_optimizer_steps: int) -> None:
    """Raise a distinct message for each way a DCP resume can lose optimizer moments.

    'No moment state at all' (DCP skipped the subtree) and 'moments present but the step
    disagrees with the sidecar' (checkpoint/sidecar mismatch) are different failure modes
    and must not share a message.
    """

    optimizer = getattr(strategy, "optimizer", None)
    if not isinstance(optimizer, torch.optim.Optimizer):
        return
    state = optimizer.state
    if not state:
        raise RuntimeError("response checkpoint restored no optimizer moment state")
    for moment in state.values():
        if int(moment["step"]) != expected_optimizer_steps:
            raise RuntimeError("response checkpoint optimizer step disagrees with the counter sidecar")


def _checkpoint_dir(value: str, *, create: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("response checkpoint directory is invalid")
    path = Path(value)
    if path.is_symlink():
        raise RuntimeError("response checkpoint directory cannot be a symlink")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError("response checkpoint directory is unavailable")
    return path.resolve()


def _counter_path(root: Path, rank: int) -> Path:
    return root / f"rdan-response-counters-rank-{rank}.json"


def _valid_counters(value: Any) -> bool:
    keys = {"optimizer_steps", "scheduler_steps", "finite_steps", "skipped_optimizer_steps"}
    valid = (
        isinstance(value, Mapping)
        and set(value) == keys
        and all(
            isinstance(value[name], int) and not isinstance(value[name], bool) and value[name] >= 0 for name in keys
        )
    )
    return bool(
        valid
        and value["optimizer_steps"] == value["scheduler_steps"] == value["finite_steps"]
        and value["skipped_optimizer_steps"] == 0
    )


def _valid_checkpoint_state(pipeline_step: int, counters: Any) -> bool:
    return bool(_valid_counters(counters) and counters["optimizer_steps"] == pipeline_step * UPDATES_PER_STEP)


def _rank(worker: Any) -> int:
    rank_info = getattr(worker, "rank_info", None)
    rank = getattr(rank_info, "dp_rank", getattr(worker, "rank", None))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in {0, 1}:
        raise FSDPHFReceiptError("response worker requires actor or inference DP rank 0 or 1")
    return rank


def _cuda_memory(worker: Any) -> dict[str, int]:
    device = torch.cuda.current_device()
    return {
        "rank": _rank(worker),
        "peak_bytes": max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)),
        "total_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
