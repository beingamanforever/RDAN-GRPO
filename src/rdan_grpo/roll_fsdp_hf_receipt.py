"""ROLL hooks for the revision-gated FSDP2 to HF weight receipt."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from roll.third_party.fsdp2 import model_update as fsdp2_model_update

from rdan_grpo.fsdp_hf_receipt import (
    FSDPHFReceiptError,
    FSDPHFStreamReceipt,
    FSDPHFTransaction,
)

_RECEIPT_ATTR = "_rdan_fsdp_hf_receipt"


def begin_fsdp_hf_receipt(worker: Any, transaction_id: str, infer_rank: int) -> None:
    """Begin one identity-paired actor receipt before the real model update."""

    if getattr(worker, _RECEIPT_ATTR, None) is not None:
        raise FSDPHFReceiptError("FSDP2 actor receipt transaction is already active")
    actor_rank = _worker_rank(worker)
    transaction = FSDPHFTransaction(transaction_id, actor_rank, infer_rank)
    setattr(
        worker,
        _RECEIPT_ATTR,
        FSDPHFStreamReceipt(transaction, "actor", accelerator_name=torch.cuda.get_device_name()),
    )


def run_receipted_fsdp_hf_update(worker: Any, model_update_name: str, update: Callable[[], Any]) -> Any:
    """Wrap the real gather_fsdp2_weights stream for exactly one update."""

    receipt = _required_receipt(worker, "actor")
    updater = getattr(getattr(worker, "strategy", None), "weight_updaters", {}).get(model_update_name)
    if updater is None:
        raise FSDPHFReceiptError("RTT did not create the FSDP2 weight updater")
    if getattr(updater, "model_update_name", None) != model_update_name:
        raise FSDPHFReceiptError("actor receipt is bound to a different model updater")
    if getattr(updater, "is_lora", None) is not False:
        raise FSDPHFReceiptError("FSDP2 to HF receipt requires full model weights")
    if getattr(updater, "is_colocated", None) is not True:
        raise FSDPHFReceiptError("FSDP2 to HF receipt requires colocated model update")
    infer_config = getattr(updater, "infer_worker_config", None)
    strategy_name = getattr(getattr(infer_config, "strategy_args", None), "strategy_name", None)
    if strategy_name != "hf_infer" or getattr(infer_config, "num_gpus_per_worker", None) != 1:
        raise FSDPHFReceiptError("FSDP2 to HF receipt requires paired HF TP1 inference")
    original = fsdp2_model_update.gather_fsdp2_weights
    if getattr(original, "__rdan_receipt_owner__", None) is not None:
        raise FSDPHFReceiptError("conflicting RTT FSDP2 weight generator wrapper")

    def gather_fsdp2_weights(*args: Any, **kwargs: Any) -> Any:
        receipt.open_actor_stream()
        return receipt.wrap_actor_batches(original(*args, **kwargs))

    gather_fsdp2_weights.__rdan_receipt_owner__ = "rdan-grpo"
    fsdp2_model_update.gather_fsdp2_weights = gather_fsdp2_weights
    try:
        return update()
    finally:
        fsdp2_model_update.gather_fsdp2_weights = original


def get_fsdp_actor_receipt(worker: Any) -> dict[str, Any]:
    """Return the actor-side receipt without changing transaction state."""

    return _required_receipt(worker, "actor").snapshot()


def begin_hf_infer_receipt(worker: Any, transaction_id: str, actor_rank: int) -> None:
    """Begin one HF receiver receipt before the paired real model update."""

    if getattr(worker, _RECEIPT_ATTR, None) is not None:
        raise FSDPHFReceiptError("HF infer receipt transaction is already active")
    infer_rank = _worker_rank(worker)
    strategy = getattr(worker, "strategy", None)
    if getattr(strategy, "strategy_name", None) != "hf_infer":
        raise FSDPHFReceiptError("FSDP2 to HF receipt requires the HF inference strategy")
    transaction = FSDPHFTransaction(transaction_id, actor_rank, infer_rank)
    setattr(
        worker,
        _RECEIPT_ATTR,
        FSDPHFStreamReceipt(transaction, "infer", accelerator_name=torch.cuda.get_device_name()),
    )


def finish_hf_infer_receipt(worker: Any) -> dict[str, Any]:
    """Hash final HF named parameters after the real model update returns."""

    receipt = _required_receipt(worker, "infer")
    model = getattr(getattr(worker, "strategy", None), "model", None)
    if model is None or not callable(getattr(model, "named_parameters", None)):
        raise FSDPHFReceiptError("HF inference model does not expose named parameters")
    receipt.finish_infer(model.named_parameters())
    return receipt.snapshot()


def reset_fsdp_hf_receipt(worker: Any) -> dict[str, Any]:
    """Clear one completed named transaction so the next phase can be receipted."""

    receipt = getattr(worker, _RECEIPT_ATTR, None)
    if not isinstance(receipt, FSDPHFStreamReceipt):
        raise FSDPHFReceiptError("receipt transaction was not begun")
    snapshot = receipt.snapshot()
    transaction = snapshot.get("transaction", {})
    if snapshot.get("stream_complete") is not True or transaction != {"calls": 1, "complete": True}:
        raise FSDPHFReceiptError("only one completed receipt transaction can be reset")
    delattr(worker, _RECEIPT_ATTR)
    return snapshot


def _required_receipt(worker: Any, side: str) -> FSDPHFStreamReceipt:
    receipt = getattr(worker, _RECEIPT_ATTR, None)
    if not isinstance(receipt, FSDPHFStreamReceipt) or receipt.side != side:
        raise FSDPHFReceiptError(f"{side} receipt transaction was not begun")
    return receipt


def _worker_rank(worker: Any) -> int:
    rank_info = getattr(worker, "rank_info", None)
    rank = getattr(rank_info, "dp_rank", getattr(worker, "rank", None))
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise FSDPHFReceiptError("worker DP rank is unavailable")
    return rank
