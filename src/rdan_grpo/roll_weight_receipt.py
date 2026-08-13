"""ROLL and vLLM hooks for the revision-gated weight receipt diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import vllm
from roll.third_party.megatron import model_update as rtt_model_update
from roll.third_party.vllm.worker import WorkerV1

from rdan_grpo.weight_receipt import TensorStreamReceipt, WeightReceiptError

_RECEIPT_ATTR = "_rdan_weight_receipt"
_UPDATER_ATTR = "_rdan_receipt_updater"


class ReceiptWorkerV1(WorkerV1):
    """Diagnostic vLLM worker that records bytes consumed by its real loader."""

    def custom_init_worker(self, *args: Any, **kwargs: Any) -> None:
        super().custom_init_worker(*args, **kwargs)
        if vllm.__version__ != "0.10.2":
            raise RuntimeError(f"weight receipt requires vLLM 0.10.2, got {vllm.__version__}")
        setattr(self, _RECEIPT_ATTR, None)

    def rdan_begin_weight_receipt(self, transaction_id: str, infer_rank: int, actor_rank: int) -> None:
        current = getattr(self, _RECEIPT_ATTR, None)
        if current is not None:
            raise WeightReceiptError("vLLM weight receipt transaction is already active")
        receipt = TensorStreamReceipt(
            transaction_id,
            "infer",
            infer_rank,
            actor_rank,
            accelerator_name=torch.cuda.get_device_name(),
        )
        self.reload_model()
        receipt.set_internal_before(self.model_runner.model.named_parameters())
        setattr(self, _RECEIPT_ATTR, receipt)

    def rdan_get_weight_receipt(self) -> dict[str, Any]:
        receipt = _required_receipt(self)
        if (
            receipt.loader_calls > 0
            and receipt.loader_calls == receipt.loader_successes
            and receipt.loader_segments_started == receipt.loader_segments_completed
            and not receipt.loader_failed
        ):
            receipt.finish_stream()
        state = receipt.snapshot()["loader"]
        if state["loaded"]:
            receipt.set_internal_after(self.model_runner.model.named_parameters())
        return receipt.snapshot()

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> None:
        receipt = _required_receipt(self)
        self.reload_model()
        try:
            super().load_weights(receipt.wrap_loader_segment(weights))
        except Exception:
            receipt.mark_loader_failure()
            raise
        receipt.mark_loader_success()


def bind_actor_weight_updater(worker: Any, model_update_name: str) -> None:
    """Bind only the colocated updater created by RTT setup_model_update."""

    updater = worker.strategy.weight_updaters.get(model_update_name)
    if updater is None:
        raise WeightReceiptError("RTT did not create the actor weight updater")
    if updater.is_colocated is not True:
        raise WeightReceiptError("weight receipt requires colocated model update")
    if updater.infer_worker_config.num_gpus_per_worker != 1:
        raise WeightReceiptError("weight receipt requires vLLM TP1")
    setattr(worker, _UPDATER_ATTR, updater)


def begin_actor_weight_receipt(worker: Any, transaction_id: str, infer_rank: int) -> None:
    if getattr(worker, _UPDATER_ATTR, None) is None:
        raise WeightReceiptError("actor weight updater is not receipt-bound")
    if getattr(worker, _RECEIPT_ATTR, None) is not None:
        raise WeightReceiptError("actor weight receipt transaction is already active")
    rank = int(worker.rank)
    if rank != infer_rank:
        raise WeightReceiptError(f"weight receipt requires identity pairing, got actor {rank} to infer {infer_rank}")
    setattr(
        worker,
        _RECEIPT_ATTR,
        TensorStreamReceipt(
            transaction_id,
            "actor",
            rank,
            infer_rank,
            accelerator_name=torch.cuda.get_device_name(),
        ),
    )


def get_actor_weight_receipt(worker: Any) -> dict[str, Any]:
    return _required_receipt(worker).snapshot()


def run_receipted_actor_update(worker: Any, model_update_name: str, update: Callable[[], Any]) -> Any:
    """Patch the pinned RTT generator only for one actor update and always restore it."""

    receipt = _required_receipt(worker)
    updater = getattr(worker, _UPDATER_ATTR, None)
    if updater is None or updater.model_update_name != model_update_name:
        raise WeightReceiptError("actor receipt is bound to a different model updater")
    original = rtt_model_update.gather_all_hf_weights
    if getattr(original, "__rdan_receipt_owner__", None) is not None:
        raise WeightReceiptError("conflicting RTT weight generator wrapper")

    def gather_all_hf_weights(*args: Any, **kwargs: Any) -> Iterable[Any]:
        receipt.stream_started = True
        for batch in original(*args, **kwargs):
            receipt.record_batch(_named_weights(batch))
            yield batch
        receipt.finish_stream()

    gather_all_hf_weights.__rdan_receipt_owner__ = "rdan-grpo"
    rtt_model_update.gather_all_hf_weights = gather_all_hf_weights
    try:
        return update()
    finally:
        rtt_model_update.gather_all_hf_weights = original


async def begin_infer_weight_receipt(worker: Any, transaction_id: str, actor_rank: int) -> Any:
    return await _collective_rpc(
        worker,
        "rdan_begin_weight_receipt",
        args=(transaction_id, int(worker.rank), actor_rank),
    )


async def get_infer_weight_receipt(worker: Any) -> Any:
    return await _collective_rpc(worker, "rdan_get_weight_receipt")


async def _collective_rpc(worker: Any, method: str, args: tuple[Any, ...] = ()) -> Any:
    model = getattr(getattr(worker, "strategy", None), "model", None)
    engine_core = getattr(model, "engine_core", None)
    rpc = getattr(engine_core, "collective_rpc_async", None)
    if not callable(rpc):
        raise WeightReceiptError("weight receipt requires the vLLM V1 collective RPC boundary")
    return await rpc(method=method, args=args)


def _required_receipt(owner: Any) -> TensorStreamReceipt:
    receipt = getattr(owner, _RECEIPT_ATTR, None)
    if not isinstance(receipt, TensorStreamReceipt):
        raise WeightReceiptError("weight receipt transaction was not begun")
    return receipt


def _named_weights(batch: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(batch, Mapping):
        return batch.items()
    return batch
