"""Production FSDP2 and Hugging Face byte receipts for response-level training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rdan_grpo.fsdp_hf_receipt import MODEL, MODEL_REVISION, RTT_REVISION, canonical_sha256, manifest_summary
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY

PHASES = {"initial", "post_update", "resume_initial"}
MODEL_KEYS = {
    "model",
    "revision",
    "snapshot_sha256",
    "tokenizer_files_sha256",
    "chat_template_sha256",
}
RUNTIME_KEYS = {
    "resolved_config_sha256",
    "production_train_config_sha256",
    "response_data_manifest_sha256",
    "response_data_output_sha256",
    "rtt_revision",
    *GENERATION_SOURCE_IDENTITY,
}
COUNTER_KEYS = {"rank", "optimizer_steps", "scheduler_steps", "finite_steps", "skipped_optimizer_steps"}
ITEM_KEYS = {"index", "name", "shape", "dtype", "nbytes", "sha256"}


class ResponseReceiptError(ValueError):
    """Raised when production byte or training-step evidence is incomplete."""


def build_response_receipt(
    actor_receipts: Sequence[Mapping[str, Any]],
    infer_receipts: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    pipeline_step: int,
    actor_counters: Sequence[Mapping[str, Any]],
    resolved_config_sha256: str,
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    method: str,
    fixed_weight: float,
) -> dict[str, Any]:
    """Build one passing production receipt from exact DP2 bytes and counters."""

    if phase not in PHASES:
        raise ResponseReceiptError("response receipt phase is invalid")
    if isinstance(pipeline_step, bool) or not isinstance(pipeline_step, int) or pipeline_step < 0:
        raise ResponseReceiptError("response receipt pipeline step is invalid")
    actors = [_receipt(value, "actor") for value in actor_receipts]
    infers = [_receipt(value, "infer") for value in infer_receipts]
    transaction_id = _validate_pairs(actors, infers)
    counters = _validate_counters(actor_counters, phase, pipeline_step)
    runtime = _runtime(runtime_identity, resolved_config_sha256)
    model = _model(model_identity)
    if not isinstance(method, str) or not method or method.strip() != method:
        raise ResponseReceiptError("response receipt method is invalid")
    if isinstance(fixed_weight, bool) or not isinstance(fixed_weight, (int, float)):
        raise ResponseReceiptError("response receipt fixed weight is invalid")
    weight = float(fixed_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ResponseReceiptError("response receipt fixed weight is invalid")
    artifact = {
        "schema_version": 1,
        "id": f"qwen_response_receipt_{phase}_v1",
        "status": "receipt_passed",
        "phase": phase,
        "transaction_id": transaction_id,
        "pipeline_step": pipeline_step,
        "method": method,
        "fixed_weight": weight,
        "topology": {"actor_dp": 2, "infer_dp": 2, "pairs": [[0, 0], [1, 1]]},
        "runtime": runtime,
        "model": model,
        "optimizer_updates": counters[0]["optimizer_steps"],
        "actor_counters": counters,
        "actor_receipts": actors,
        "infer_receipts": infers,
    }
    artifact["receipt_manifest_sha256"] = canonical_sha256({"actor_receipts": actors, "infer_receipts": infers})
    return artifact


def _receipt(value: Mapping[str, Any], side: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseReceiptError("response receipt worker evidence is invalid")
    receipt = dict(value)
    transaction = receipt.get("transaction")
    items = receipt.get("items")
    if (
        receipt.get("side") != side
        or receipt.get("stream_started") is not True
        or receipt.get("stream_complete") is not True
        or transaction != {"calls": 1, "complete": True}
        or not isinstance(items, list)
        or not items
    ):
        raise ResponseReceiptError("response receipt worker transaction is incomplete")
    for index, item in enumerate(items):
        if (
            not isinstance(item, Mapping)
            or set(item) != ITEM_KEYS
            or item.get("index") != index
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("shape"), list)
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in item["shape"])
            or not isinstance(item.get("dtype"), str)
            or isinstance(item.get("nbytes"), bool)
            or not isinstance(item.get("nbytes"), int)
            or item["nbytes"] < 0
            or not _sha256(item.get("sha256"))
        ):
            raise ResponseReceiptError("response receipt tensor manifest is invalid")
    if any(receipt.get(key) != value for key, value in manifest_summary(items).items()):
        raise ResponseReceiptError("response receipt tensor summary is invalid")
    return receipt


def _validate_pairs(actors: list[dict[str, Any]], infers: list[dict[str, Any]]) -> str:
    if len(actors) != 2 or len(infers) != 2:
        raise ResponseReceiptError("response receipt requires exact actor and inference DP2")
    actors.sort(key=lambda value: value.get("rank", -1))
    infers.sort(key=lambda value: value.get("rank", -1))
    if [value.get("rank") for value in actors] != [0, 1] or [value.get("rank") for value in infers] != [0, 1]:
        raise ResponseReceiptError("response receipt ranks are invalid")
    transaction_ids = {value.get("transaction_id") for value in [*actors, *infers]}
    transaction_id = next(iter(transaction_ids)) if len(transaction_ids) == 1 else None
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ResponseReceiptError("response receipt transaction IDs differ")
    for rank, (actor, infer) in enumerate(zip(actors, infers, strict=True)):
        if actor.get("paired_rank") != rank or infer.get("paired_rank") != rank:
            raise ResponseReceiptError("response receipt DP ranks are not identity paired")
        if actor["items"] != infer["items"]:
            raise ResponseReceiptError("response receipt actor and inference bytes differ")
    if actors[0]["items"] != actors[1]["items"] or infers[0]["items"] != infers[1]["items"]:
        raise ResponseReceiptError("response receipt replicas differ")
    return transaction_id


def _validate_counters(
    values: Sequence[Mapping[str, Any]],
    phase: str,
    pipeline_step: int,
) -> list[dict[str, int]]:
    if len(values) != 2:
        raise ResponseReceiptError("response receipt requires counters for actor DP2")
    counters = [dict(value) for value in values]
    counters.sort(key=lambda value: value.get("rank", -1))
    if [value.get("rank") for value in counters] != [0, 1] or any(set(value) != COUNTER_KEYS for value in counters):
        raise ResponseReceiptError("response receipt actor counters are invalid")
    for value in counters:
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 0
            for name in COUNTER_KEYS
        ):
            raise ResponseReceiptError("response receipt actor counters are invalid")
        if value["optimizer_steps"] != value["scheduler_steps"] or value["skipped_optimizer_steps"] != 0:
            raise ResponseReceiptError("optimizer and scheduler counters differ")
        if value["finite_steps"] != value["optimizer_steps"]:
            raise ResponseReceiptError("response receipt finite-step evidence differs")
    if counters[0] != {**counters[1], "rank": 0}:
        raise ResponseReceiptError("response receipt counters differ across replicas")
    updates = counters[0]["optimizer_steps"]
    if phase == "initial" and (pipeline_step != 0 or updates != 0):
        raise ResponseReceiptError("initial response receipt requires zero training state")
    if phase != "initial" and (pipeline_step < 1 or updates < 1):
        raise ResponseReceiptError("trained response receipt requires nonzero training state")
    if phase != "initial" and updates < pipeline_step:
        raise ResponseReceiptError("trained response receipt optimizer updates trail the pipeline step")
    return counters


def _runtime(value: Mapping[str, Any], resolved_config_sha256: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != RUNTIME_KEYS:
        raise ResponseReceiptError("response receipt runtime identity is invalid")
    runtime = dict(value)
    if runtime.get("resolved_config_sha256") != resolved_config_sha256 or not _sha256(resolved_config_sha256):
        raise ResponseReceiptError("response receipt resolved config identity differs")
    if runtime.get("rtt_revision") != RTT_REVISION or any(
        runtime.get(key) != expected for key, expected in GENERATION_SOURCE_IDENTITY.items()
    ):
        raise ResponseReceiptError("response receipt runtime identity differs")
    for key in ("production_train_config_sha256", "response_data_manifest_sha256", "response_data_output_sha256"):
        if not _sha256(runtime.get(key)):
            raise ResponseReceiptError("response receipt runtime artifact identity is invalid")
    return runtime


def _model(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != MODEL_KEYS:
        raise ResponseReceiptError("response receipt model identity is invalid")
    model = dict(value)
    if model.get("model") != MODEL or model.get("revision") != MODEL_REVISION:
        raise ResponseReceiptError("response receipt model identity differs")
    if any(not _sha256(model.get(key)) for key in MODEL_KEYS - {"model", "revision"}):
        raise ResponseReceiptError("response receipt model hashes are invalid")
    return model


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
