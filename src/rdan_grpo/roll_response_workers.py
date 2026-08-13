"""Production RTT workers for response-level FSDP2 and Hugging Face training."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
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


class ResponseActorWorker(ActorWorker):
    """Train through RTT RLVR while observing exact optimizer and receipt state."""

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        loss, metrics = super().loss_func(data, output_tensor)
        value = metrics.get("actor/ppo_ratio_clipfrac")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise RuntimeError("response training requires a finite PPO clipping fraction")
        getattr(self, _CLIP_ATTR).append(float(value))
        return loss, metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_begin_response_receipt(self, transaction_id: str) -> None:
        begin_fsdp_hf_receipt(self, transaction_id, _rank(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_get_response_receipt(self) -> dict[str, Any]:
        return get_fsdp_actor_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_response_receipt(self) -> dict[str, Any]:
        return reset_fsdp_hf_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_train_counters(self) -> dict[str, int]:
        return {"rank": _rank(self), **_counters(self)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_training_state(self) -> dict[str, int | bool]:
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
        return _save_dcp(self, checkpoint_dir, pipeline_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_load_dcp(self, checkpoint_dir: str) -> dict[str, Any]:
        return _load_dcp(self, checkpoint_dir)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_cuda_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_cuda_memory(self) -> dict[str, int]:
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

        def observed_optimizer_step(*args: Any, **kwargs: Any) -> Any:
            fractions = getattr(self, _CLIP_ATTR)
            if not fractions:
                raise RuntimeError("response training did not report PPO clipping evidence")
            if not _all_ranks_grad_finite(strategy):
                counters["skipped_optimizer_steps"] += 1
                state.update(grad_finite=False, update_skipped=True)
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("response training blocked a non-finite gradient update")
            if _all_ranks_fully_clipped(fractions):
                counters["skipped_optimizer_steps"] += 1
                state.update(update_skipped=True)
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("response training blocked a fully clipped optimizer update")
            result = optimizer_step(*args, **kwargs)
            fractions.clear()
            counters["optimizer_steps"] += 1
            counters["finite_steps"] += 1
            return result

        def observed_scheduler_step(*args: Any, **kwargs: Any) -> Any:
            if counters["optimizer_steps"] <= counters["scheduler_steps"]:
                counters["skipped_optimizer_steps"] += 1
                state.update(grad_finite=False, update_skipped=True)
                raise RuntimeError("RTT attempted to advance the scheduler without a successful optimizer step")
            result = scheduler_step(*args, **kwargs)
            counters["scheduler_steps"] += 1
            return result

        optimizer.step = observed_optimizer_step
        scheduler.step = observed_scheduler_step
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
        return run_receipted_fsdp_hf_update(
            self,
            model_update_name,
            lambda: super(ResponseActorWorker, self).start_model_update(model_update_name),
        )


class ResponseInferWorker(SynchronousHFInferWorker):
    """Generate prompt-major microbatches and expose resumable inference state."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_begin_response_receipt(self, transaction_id: str) -> None:
        begin_hf_infer_receipt(self, transaction_id, _rank(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_finish_response_receipt(self) -> dict[str, Any]:
        return finish_hf_infer_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_response_receipt(self) -> dict[str, Any]:
        return reset_fsdp_hf_receipt(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_save_rng(self) -> dict[str, torch.Tensor]:
        if not torch.cuda.is_available():
            raise RuntimeError("response inference RNG requires CUDA")
        return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state().clone()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_load_rng(self, state: Mapping[str, torch.Tensor]) -> None:
        if not isinstance(state, Mapping) or set(state) != {"cpu", "cuda"}:
            raise RuntimeError("response inference RNG state is invalid")
        if any(not isinstance(state[name], torch.Tensor) or state[name].dtype != torch.uint8 for name in state):
            raise RuntimeError("response inference RNG tensors are invalid")
        torch.set_rng_state(state["cpu"].cpu())
        torch.cuda.set_rng_state(state["cuda"].cpu())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reset_cuda_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_cuda_memory(self) -> dict[str, int]:
        return _cuda_memory(self)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def generate(self, data: DataProto) -> DataProto:
        """Generate contiguous rank-local prompt microbatches in prompt-major order."""

        size = getattr(self.worker_config, "infer_batch_size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RuntimeError("response inference requires a positive infer_batch_size")
        if len(data) <= size:
            return super().generate(data)
        generate = super().generate
        outputs = [generate(data[start : start + size]) for start in range(0, len(data), size)]
        return DataProto.concat(outputs)


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
    worker.strategy.save_checkpoint(
        save_dir=str(root),
        global_step=max(pipeline_step - 1, 0),
        ckpt_id=f"response-step-{pipeline_step:06d}",
        is_last_step=True,
    )
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
        strategy._load_checkpoint_with_dcp(checkpoint_dir=dcp_dir)
    finally:
        strategy.offload_states()
    setattr(worker, _COUNTER_ATTR, dict(counters))
    setattr(worker, _TRAIN_STATE_ATTR, {"grad_finite": True, "update_skipped": False})
    return {"checkpoint_dir": str(root), "pipeline_step": pipeline_step, "rank": _rank(worker)}


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
