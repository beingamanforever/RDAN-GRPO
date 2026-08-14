"""One response-only optimizer transaction at the pinned RTT batch boundary."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from roll.distributed.scheduler.protocol import DataProto

from rdan_grpo.fsdp_hf_receipt import canonical_sha256, manifest_summary
from rdan_grpo.roll_bridge import PreflightCertificate, make_roll_compute_advantage
from rdan_grpo.roll_response_config import UPDATES_PER_STEP
from rdan_grpo.roll_scalar import QUALITY_METHODS, ScalarMethod

MAX_MEMORY_FRACTION = 0.92
PAIR_RANKS = ((0, 0), (1, 1))
REWARD_METADATA = ("prompt", "rubrics", "source", "ground_truth", "rdan_prompt_key")
TOKEN_FIELDS = (
    "old_log_probs",
    "ref_log_probs",
    "advantages",
    "returns",
    "final_response_mask",
)
TRAINING_STATE_KEYS = {"rank", "optimizer_step", "scheduler_step", "grad_finite", "update_skipped"}


@dataclass(frozen=True)
class ResponseTrainResult:
    """Evidence for one completed response-only optimizer transaction."""

    method: ScalarMethod
    prompt_count: int
    response_count: int
    optimizer_updates: int
    scheduler_steps: int
    initial_transaction_id: str
    post_transaction_id: str
    metrics: Mapping[str, Any]
    peak_memory_fraction: float
    promotion_ready: bool


def run_response_train_step(
    *,
    pipeline_config: Any,
    actor_train: Any,
    actor_infer: Any,
    rewarded_batch: DataProto,
    certificate: PreflightCertificate | Mapping[str, Any],
    initial_receipt: Mapping[str, Any],
    transfer_after_update: Callable[[], Mapping[str, Any]],
    observe_training_state: Callable[[], Sequence[Mapping[str, Any]]],
    observe_post_transaction_memory: Callable[[], Sequence[Mapping[str, Any]]],
    method: ScalarMethod,
    quality_weight: float | None = None,
    mix_weight: float | None = None,
) -> ResponseTrainResult:
    """Train once from a rewarded batch returned by RTT's synchronous scheduler."""

    returns = _validate_topology(pipeline_config, actor_train, actor_infer)
    prompt_count = _validate_rewarded_batch(rewarded_batch, returns)
    state_before = _training_state(observe_training_state(), "before")
    initial_id, initial_pipeline_step, initial_inventory = _initial_receipt(initial_receipt, state_before)
    full_response_mask = _prepare_training_batch(
        rewarded_batch,
        certificate,
        returns,
        method=method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )
    _bind_log_probs(actor_train, rewarded_batch)
    _validate_token_fields(rewarded_batch, full_response_mask)
    metrics, state_after, optimizer_delta, scheduler_delta = _execute_actor_update(
        actor_train, rewarded_batch, full_response_mask, state_before, observe_training_state
    )
    post_id = _validate_post_receipt(
        transfer_after_update(), state_after, initial_id, initial_pipeline_step, initial_inventory
    )
    peak_fraction = _validate_post_transaction_memory(observe_post_transaction_memory())
    return _train_result(
        method,
        rewarded_batch,
        prompt_count,
        optimizer_delta,
        scheduler_delta,
        initial_id,
        post_id,
        metrics,
        peak_fraction,
    )


def _initial_receipt(
    receipt: Mapping[str, Any], state: Mapping[int, Mapping[str, Any]]
) -> tuple[str, int, tuple[Mapping[str, Any], ...]]:
    validated = _validate_receipt(receipt, _shared_counter(state, "optimizer_step"))
    _require_fp32_initial_actor_receipt(receipt)
    return validated


def _execute_actor_update(
    actor_train: Any,
    batch: DataProto,
    response_mask: torch.Tensor,
    state_before: Mapping[int, Mapping[str, Any]],
    observe_training_state: Callable[[], Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[int, Mapping[str, Any]], int, int]:
    metrics = dict(_train_metrics(actor_train.train_step(batch, blocking=True))) | _rollout_metrics(batch)
    if not torch.equal(batch.batch["response_mask"], response_mask):
        raise RuntimeError("actor training changed the full response mask")
    state_after = _training_state(observe_training_state(), "after")
    optimizer_delta, scheduler_delta = _validate_state_delta(state_before, state_after)
    return metrics, state_after, optimizer_delta, scheduler_delta


def _train_result(
    method: ScalarMethod,
    batch: DataProto,
    prompt_count: int,
    optimizer_updates: int,
    scheduler_steps: int,
    initial_id: str,
    post_id: str,
    metrics: Mapping[str, Any],
    peak_memory_fraction: float,
) -> ResponseTrainResult:
    return ResponseTrainResult(
        method,
        prompt_count,
        len(batch),
        optimizer_updates,
        scheduler_steps,
        initial_id,
        post_id,
        MappingProxyType(dict(metrics)),
        peak_memory_fraction,
        True,
    )


def _prepare_training_batch(
    batch: DataProto,
    certificate: PreflightCertificate | Mapping[str, Any],
    returns: int,
    *,
    method: ScalarMethod,
    quality_weight: float | None,
    mix_weight: float | None,
) -> torch.Tensor:
    full_response_mask = batch.batch["response_mask"].clone()
    shifted_mask = full_response_mask[:, 1:].to(torch.bool)
    batch.batch["final_response_mask"] = shifted_mask.clone()
    adapter = make_roll_compute_advantage(
        certificate,
        method=method,
        group_size=returns,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )
    adapter(batch, response_mask=shifted_mask)
    _bind_method_evidence(batch, method=method, quality_weight=quality_weight, mix_weight=mix_weight)
    _pre_update_reward_gate(batch, returns, method)
    return full_response_mask


def _validate_post_receipt(
    receipt: Mapping[str, Any],
    state_after: Mapping[int, Mapping[str, Any]],
    initial_id: str,
    initial_pipeline_step: int,
    initial_inventory: Sequence[Mapping[str, Any]],
) -> str:
    post_id, post_pipeline_step, post_inventory = _validate_receipt(
        receipt, _shared_counter(state_after, "optimizer_step")
    )
    if post_id == initial_id:
        raise RuntimeError("post-update receipt must use a distinct transaction")
    if post_pipeline_step != initial_pipeline_step + 1:
        raise RuntimeError("post-update receipt did not advance exactly one pipeline transaction")
    _validate_parameter_update(initial_inventory, post_inventory)
    return post_id


def _validate_topology(config: Any, actor_train: Any, actor_infer: Any) -> int:
    expected = (
        (getattr(config, "actor_train", None), actor_train, "fsdp2_train", "actor_train"),
        (getattr(config, "actor_infer", None), actor_infer, "vllm", "actor_infer"),
    )
    for worker_config, cluster, strategy_name, name in expected:
        strategy = getattr(getattr(worker_config, "strategy_args", None), "strategy_name", None)
        if strategy != strategy_name:
            raise RuntimeError(f"response training requires {name} strategy {strategy_name}")
        if (
            list(getattr(worker_config, "device_mapping", ())) != [0, 1]
            or getattr(worker_config, "num_gpus_per_worker", None) != 1
            or getattr(worker_config, "world_size", None) != 2
        ):
            raise RuntimeError(f"response training requires {name} DP2 colocated on devices [0, 1]")
        _validate_cluster(cluster, name)
    if (
        bool(getattr(config, "async_pipeline", False))
        or getattr(config, "async_generation_ratio", None) != 0
        or getattr(config, "generate_opt_level", None) != 0
    ):
        raise RuntimeError("response training requires synchronous generate_opt_level=0")
    if getattr(config.actor_infer, "max_concurrency", 0) <= 1:
        raise RuntimeError("response training requires concurrent vLLM inference")
    if getattr(config, "enable_reference", None) is not False:
        raise RuntimeError("response training requires the reference model to be disabled")
    if getattr(config, "enable_old_logprobs_recompute", None) is not True:
        raise RuntimeError("response training requires explicit old log-probability recomputation")
    generating_args = getattr(config.actor_infer, "generating_args", None)
    returns = getattr(generating_args, "num_return_sequences", None)
    if isinstance(returns, bool) or not isinstance(returns, int) or returns <= 1:
        raise RuntimeError("response training requires grouped num_return_sequences")
    return returns


def _validate_cluster(cluster: Any, name: str) -> None:
    if getattr(cluster, "dp_size", None) != 2:
        raise RuntimeError(f"response training requires live {name} DP2")
    ranks = getattr(cluster, "worker_rank_info", None)
    if not isinstance(ranks, Sequence) or len(ranks) != 2:
        raise RuntimeError(f"response training requires two live {name} workers")
    for rank in ranks:
        if (
            getattr(rank, "dp_size", None) != 2
            or getattr(rank, "tp_size", None) != 1
            or getattr(rank, "pp_size", None) != 1
            or getattr(rank, "cp_size", 1) != 1
        ):
            raise RuntimeError(f"response training requires live {name} DP2 TP1 PP1 CP1")


def _validate_rewarded_batch(data: DataProto, returns: int) -> int:
    if not isinstance(data, DataProto) or data.batch is None or len(data) <= 0:
        raise RuntimeError("response training requires a non-empty scheduler-produced reward batch")
    required = {
        "origin_prompt_id",
        "input_ids",
        "attention_mask",
        "response_mask",
        "rdan_scores",
        "rdan_rubric_mask",
        "rdan_eval_mask",
        "rdan_hard_mask",
    }
    missing = sorted(required - set(data.batch.keys()))
    missing.extend(name for name in REWARD_METADATA if name not in data.non_tensor_batch)
    if missing:
        raise RuntimeError(f"scheduler reward batch is missing fields: {', '.join(missing)}")
    for name in REWARD_METADATA:
        values = data.non_tensor_batch[name]
        if not isinstance(values, np.ndarray) or values.dtype != object or len(values) != len(data):
            raise RuntimeError(f"scheduler reward metadata {name} has invalid boundaries")
    prompt_ids = _values(data.batch["origin_prompt_id"])
    prompt_keys = _values(data.non_tensor_batch["rdan_prompt_key"])
    if len(prompt_ids) != len(data) or len(prompt_keys) != len(data) or len(data) % returns:
        raise RuntimeError("scheduler reward batch has invalid grouped response boundaries")
    seen_keys: set[Any] = set()
    for start in range(0, len(data), returns):
        stop = start + returns
        prompt_id = prompt_ids[start]
        prompt_key = prompt_keys[start]
        try:
            repeated_group = prompt_key in seen_keys
        except TypeError as error:
            raise RuntimeError("scheduler reward prompt keys must be hashable") from error
        if repeated_group:
            raise RuntimeError("scheduler reward batch repeats a prompt group")
        if any(value != prompt_id for value in prompt_ids[start:stop]) or any(
            value != prompt_key for value in prompt_keys[start:stop]
        ):
            raise RuntimeError("scheduler reward batch changed prompt-major return order")
        for name in REWARD_METADATA[:-1]:
            if not _group_equal(data.non_tensor_batch[name][start:stop]):
                raise RuntimeError(f"scheduler reward metadata {name} is not repeated with its prompt")
        try:
            seen_keys.add(prompt_key)
        except TypeError as error:
            raise RuntimeError("scheduler reward prompt keys must be hashable") from error
    response_mask = data.batch["response_mask"]
    input_ids = data.batch["input_ids"]
    attention_mask = data.batch["attention_mask"]
    if (
        not isinstance(response_mask, torch.Tensor)
        or response_mask.ndim != 2
        or not isinstance(input_ids, torch.Tensor)
        or input_ids.shape != response_mask.shape
        or not isinstance(attention_mask, torch.Tensor)
        or attention_mask.shape != response_mask.shape
        or not bool(response_mask.to(torch.bool).any(dim=-1).all())
    ):
        raise RuntimeError("scheduler reward batch has invalid full-sequence token boundaries")
    return len(data) // returns


def _group_equal(values: Sequence[Any]) -> bool:
    first = values[0]
    for value in values[1:]:
        if isinstance(first, np.ndarray) or isinstance(value, np.ndarray):
            if not np.array_equal(first, value):
                return False
        elif value != first:
            return False
    return True


def _values(values: Any) -> list[Any]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        return []
    return [value.item() if hasattr(value, "item") else value for value in values]


def _bind_log_probs(actor_train: Any, data: DataProto) -> None:
    output = actor_train.compute_log_probs(data.clone(), blocking=True)
    log_probs = output.batch.get("log_probs") if isinstance(output, DataProto) and output.batch is not None else None
    expected_tokens = data.batch["response_mask"].shape[1] - 1
    _require_token_tensor(log_probs, len(data), expected_tokens, "actor old_log_probs")
    data.batch["old_log_probs"] = log_probs.clone()
    data.batch["ref_log_probs"] = log_probs.clone()


def _bind_method_evidence(
    data: DataProto,
    *,
    method: ScalarMethod,
    quality_weight: float | None,
    mix_weight: float | None,
) -> None:
    required = (
        "rdan_raw_aon",
        "rdan_raw_csr",
        "rdan_raw_signed_csr",
        "rdan_selected_reward",
        "rdan_response_advantage",
        "rdan_raw_quality",
        "rdan_quality_eligible",
        "rdan_quality_advantage",
        "rdan_scalar_advantage",
        "rdan_response_valid",
    )
    if data.batch is None or any(name not in data.batch for name in required):
        raise RuntimeError("method adapter did not preserve decomposed response evidence")
    if any(data.batch[name].shape != (len(data),) for name in required):
        raise RuntimeError("decomposed response evidence has invalid boundaries")
    weight = quality_weight if method in QUALITY_METHODS else mix_weight if method == "rl_mix" else None
    data.meta_info["rdan_method"] = method
    data.meta_info["rdan_method_weight"] = weight


def _pre_update_reward_gate(data: DataProto, group_size: int, method: ScalarMethod) -> None:
    selected = data.batch.get("rdan_selected_reward")
    quality = data.batch.get("rdan_raw_quality")
    eligible = data.batch.get("rdan_quality_eligible")
    if (
        not isinstance(selected, torch.Tensor)
        or not isinstance(quality, torch.Tensor)
        or not isinstance(eligible, torch.Tensor)
        or selected.ndim != 1
        or selected.shape != quality.shape
        or selected.shape != eligible.shape
        or selected.numel() == 0
        or selected.numel() % group_size
    ):
        raise RuntimeError("pre-update reward evidence is invalid")
    selected_groups = selected.detach().float().reshape(-1, group_size)
    quality_groups = quality.detach().float().reshape(-1, group_size)
    eligible_groups = eligible.detach().bool().reshape(-1, group_size)
    if not bool(torch.isfinite(selected_groups).all()) or not bool(torch.isfinite(quality_groups).all()):
        raise RuntimeError("pre-update reward evidence is non-finite")
    response_active = selected_groups.var(dim=-1, unbiased=False) > 1e-8
    if not bool(response_active.any()):
        raise RuntimeError("pre-update gate requires useful within-group selected-reward variance")
    eligible_count = eligible_groups.sum(dim=-1)
    quality_mean = (quality_groups * eligible_groups).sum(dim=-1) / eligible_count.clamp_min(1)
    quality_var = ((quality_groups - quality_mean.unsqueeze(-1)).square() * eligible_groups).sum(dim=-1)
    quality_var = quality_var / eligible_count.clamp_min(1)
    quality_active = (eligible_count >= 2) & (quality_var > 1e-8)
    quality_rate = float(quality_active.float().mean().item())
    if method in QUALITY_METHODS and quality_rate < 0.1:
        raise RuntimeError("pre-update gate requires quality active group rate at least 0.1")
    data.meta_info["rdan_pre_update_gate"] = {
        "response_active_group_rate": float(response_active.float().mean().item()),
        "quality_active_group_rate": quality_rate,
    }


def _validate_token_fields(data: DataProto, full_response_mask: torch.Tensor) -> None:
    rows, tokens = full_response_mask.shape[0], full_response_mask.shape[1] - 1
    shifted = full_response_mask[:, 1:].to(torch.bool)
    for name in TOKEN_FIELDS:
        _require_token_tensor(data.batch.get(name), rows, tokens, name)
    if not torch.equal(data.batch["final_response_mask"].to(torch.bool), shifted):
        raise RuntimeError("final response mask does not match the shifted full response mask")
    if not torch.equal(data.batch["old_log_probs"], data.batch["ref_log_probs"]):
        raise RuntimeError("disabled reference must clone explicit actor old log-probabilities")
    if not torch.equal(data.batch["advantages"], data.batch["returns"]):
        raise RuntimeError("response-only returns differ from advantages")
    if bool((data.batch["advantages"].masked_select(~shifted) != 0).any()):
        raise RuntimeError("response-only advantages escaped the shifted response mask")
    scalar = data.batch.get("rdan_scalar_advantage")
    expected = scalar.unsqueeze(-1).to(data.batch["advantages"]) * shifted.to(data.batch["advantages"].dtype)
    if not torch.equal(data.batch["advantages"], expected) or not bool(expected.abs().gt(0).any()):
        raise RuntimeError("final scalar advantage differs from the training advantage")


def _require_token_tensor(value: Any, rows: int, tokens: int, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != (rows, tokens) or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} must be finite with shape [responses, sequence_length - 1]")


def _training_state(values: Sequence[Mapping[str, Any]], phase: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise RuntimeError(f"{phase} training state must contain both actor ranks")
    states: dict[int, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping) or set(value) != TRAINING_STATE_KEYS:
            raise RuntimeError(f"{phase} training state entry is invalid")
        rank = value.get("rank")
        optimizer_step = value.get("optimizer_step")
        scheduler_step = value.get("scheduler_step")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in {0, 1}
            or rank in states
            or isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
            or isinstance(scheduler_step, bool)
            or not isinstance(scheduler_step, int)
            or scheduler_step < 0
            or not isinstance(value.get("grad_finite"), bool)
            or not isinstance(value.get("update_skipped"), bool)
        ):
            raise RuntimeError(f"{phase} training state is invalid")
        if phase == "after" and (value.get("grad_finite") is not True or value.get("update_skipped") is not False):
            raise RuntimeError("after training state records a skipped/nonfinite update")
        states[rank] = value
    return states


def _shared_counter(states: Mapping[int, Mapping[str, Any]], name: str) -> int:
    values = {int(state[name]) for state in states.values()}
    if len(values) != 1:
        raise RuntimeError(f"actor ranks disagree on {name}")
    return values.pop()


def _validate_state_delta(
    before: Mapping[int, Mapping[str, Any]],
    after: Mapping[int, Mapping[str, Any]],
) -> tuple[int, int]:
    optimizer_deltas = [after[rank]["optimizer_step"] - before[rank]["optimizer_step"] for rank in (0, 1)]
    scheduler_deltas = [after[rank]["scheduler_step"] - before[rank]["scheduler_step"] for rank in (0, 1)]
    if (
        len(set(optimizer_deltas)) != 1
        or len(set(scheduler_deltas)) != 1
        or optimizer_deltas[0] != UPDATES_PER_STEP
        or scheduler_deltas[0] != UPDATES_PER_STEP
        or optimizer_deltas[0] != scheduler_deltas[0]
    ):
        raise RuntimeError("optimizer and scheduler state did not follow the frozen two-update cadence")
    return optimizer_deltas[0], scheduler_deltas[0]


def _validate_receipt(
    receipt: Mapping[str, Any], optimizer_updates: int
) -> tuple[str, int, tuple[Mapping[str, Any], ...]]:
    if not isinstance(receipt, Mapping) or receipt.get("status") != "receipt_passed":
        raise RuntimeError("paired receipt did not pass")
    observed_updates = receipt.get("optimizer_updates")
    if (
        isinstance(observed_updates, bool)
        or not isinstance(observed_updates, int)
        or observed_updates != optimizer_updates
    ):
        raise RuntimeError("paired receipt optimizer state is invalid")
    pipeline_step = receipt.get("pipeline_step")
    if (
        isinstance(pipeline_step, bool)
        or not isinstance(pipeline_step, int)
        or pipeline_step < 0
        or (pipeline_step == 0) != (optimizer_updates == 0)
        or optimizer_updates != pipeline_step * UPDATES_PER_STEP
    ):
        raise RuntimeError("paired receipt pipeline and optimizer state are inconsistent")
    transaction_id = receipt.get("transaction_id")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise RuntimeError("paired receipt transaction is missing")
    actors = _receipt_side(receipt.get("actor_receipts"), "actor", transaction_id)
    infers = _receipt_side(receipt.get("infer_receipts"), "infer", transaction_id)
    for actor_rank, infer_rank in PAIR_RANKS:
        actor = actors[actor_rank]
        infer = infers[infer_rank]
        if actor["paired_rank"] != infer_rank or infer["paired_rank"] != actor_rank:
            raise RuntimeError("paired receipt ranks differ")
        if actor["items"] != infer["items"]:
            raise RuntimeError("paired actor and inference receipt bytes differ")
    if actors[0]["items"] != actors[1]["items"] or infers[0]["items"] != infers[1]["items"]:
        raise RuntimeError("receipt replicas contain different model bytes")
    expected_manifest = canonical_sha256(
        {"actor_receipts": list(receipt["actor_receipts"]), "infer_receipts": list(receipt["infer_receipts"])}
    )
    if receipt.get("receipt_manifest_sha256") != expected_manifest:
        raise RuntimeError("paired receipt manifest digest is invalid")
    return transaction_id, pipeline_step, tuple(actors[0]["items"])


def _validate_parameter_update(initial: Sequence[Mapping[str, Any]], post: Sequence[Mapping[str, Any]]) -> None:
    initial_metadata = [(item["name"], item["shape"], item["dtype"], item["nbytes"]) for item in initial]
    post_metadata = [(item["name"], item["shape"], item["dtype"], item["nbytes"]) for item in post]
    if initial_metadata != post_metadata:
        raise RuntimeError("post-update trainable parameter inventory differs from the initial receipt")
    # A 1e-6 optimizer update can round away in bf16, so byte mutation is sound only after fp32 master proof.
    if not any(before["sha256"] != after["sha256"] for before, after in zip(initial, post, strict=True)):
        raise RuntimeError("optimizer updates did not change any trainable parameter bytes")


def _require_fp32_initial_actor_receipt(receipt: Mapping[str, Any]) -> None:
    actors = receipt.get("actor_receipts")
    if not isinstance(actors, Sequence) or isinstance(actors, (str, bytes)) or len(actors) != 2:
        raise RuntimeError("initial actor receipt does not prove fp32 master parameters")
    for actor in actors:
        transport = actor.get("transport") if isinstance(actor, Mapping) else None
        if not isinstance(transport, Mapping) or transport.get("source_dtypes") != ["torch.float32"]:
            raise RuntimeError("initial actor receipt does not prove fp32 master parameters")


def _receipt_side(values: Any, side: str, transaction_id: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise RuntimeError(f"paired receipt requires two {side} manifests")
    result: dict[int, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"paired {side} receipt is invalid")
        rank = value.get("rank")
        items = value.get("items")
        if (
            value.get("side") != side
            or value.get("transaction_id") != transaction_id
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in {0, 1}
            or rank in result
            or isinstance(value.get("paired_rank"), bool)
            or not isinstance(value.get("paired_rank"), int)
            or value.get("paired_rank") not in {0, 1}
            or "A100" not in str(value.get("accelerator_name"))
            or value.get("stream_started") is not True
            or value.get("stream_complete") is not True
            or not isinstance(items, list)
            or not items
        ):
            raise RuntimeError(f"paired {side} receipt is incomplete")
        _validate_receipt_backend(value, side)
        _validate_manifest_items(items)
        summary = manifest_summary(items)
        if any(value.get(name) != expected for name, expected in summary.items()):
            raise RuntimeError(f"paired {side} receipt manifest summary is invalid")
        result[rank] = value
    return result


def _validate_receipt_backend(value: Mapping[str, Any], side: str) -> None:
    if side == "actor":
        if value.get("transaction") != {"calls": 1, "complete": True}:
            raise RuntimeError("paired actor receipt transaction is incomplete")
        return
    backend = value.get("backend", "hf_infer")
    if backend == "hf_infer":
        if value.get("transaction") != {"calls": 1, "complete": True} or "loader" in value:
            raise RuntimeError("paired HF inference receipt transaction is incomplete")
        return
    if backend != "vllm" or "transaction" in value or "transport" in value:
        raise RuntimeError("paired vLLM inference receipt provenance is invalid")
    loader = value.get("loader")
    keys = {"calls", "successes", "failed", "segments_started", "segments_completed", "loaded"}
    if not isinstance(loader, Mapping) or set(loader) != keys:
        raise RuntimeError("paired vLLM inference receipt loader evidence is invalid")
    calls = loader.get("calls")
    if (
        isinstance(calls, bool)
        or not isinstance(calls, int)
        or calls < 1
        or loader.get("successes") != calls
        or loader.get("segments_started") != calls
        or loader.get("segments_completed") != calls
        or loader.get("failed") is not False
        or loader.get("loaded") is not True
    ):
        raise RuntimeError("paired vLLM inference receipt loader transaction is incomplete")


def _validate_manifest_items(items: Sequence[Any]) -> None:
    for index, item in enumerate(items):
        if (
            not isinstance(item, Mapping)
            or item.get("index") != index
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("shape"), list)
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in item["shape"])
            or not isinstance(item.get("dtype"), str)
            or not item["dtype"].startswith("torch.")
            or isinstance(item.get("nbytes"), bool)
            or not isinstance(item.get("nbytes"), int)
            or item["nbytes"] < 0
            or not _sha256(item.get("sha256"))
        ):
            raise RuntimeError("paired receipt tensor manifest is invalid")
    names = [item["name"] for item in items]
    if len(names) != len(set(names)):
        raise RuntimeError("paired receipt tensor names are duplicated")


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_post_transaction_memory(values: Sequence[Mapping[str, Any]]) -> float:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise RuntimeError("post-transaction memory must contain both GPUs")
    fractions: list[float] = []
    ranks: set[int] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise RuntimeError("post-transaction GPU memory observation is invalid")
        rank = value.get("rank")
        peak = value.get("peak_bytes")
        total = value.get("total_bytes")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in {0, 1}
            or rank in ranks
            or isinstance(peak, bool)
            or isinstance(total, bool)
            or not isinstance(peak, (int, float))
            or not isinstance(total, (int, float))
            or not math.isfinite(float(peak))
            or not math.isfinite(float(total))
            or peak < 0
            or total <= 0
            or peak > total
        ):
            raise RuntimeError("post-transaction GPU memory observation is invalid")
        ranks.add(rank)
        fractions.append(float(peak) / float(total))
    maximum = max(fractions)
    if maximum > MAX_MEMORY_FRACTION:
        raise RuntimeError("post-transaction peak GPU memory exceeds 92 percent")
    return maximum


def _rollout_metrics(batch: DataProto) -> dict[str, float]:
    """Return the rollout engine metrics the vLLM worker attached to this batch."""

    values = getattr(batch, "meta_info", {}).get("vllm_metrics")
    if not isinstance(values, Mapping):
        return {}
    return {name: float(value) for name, value in values.items() if isinstance(value, (int, float))}


def _train_metrics(output: Any) -> Mapping[str, Any]:
    metrics = getattr(output, "meta_info", {}).get("metrics")
    if not isinstance(metrics, Mapping) or not metrics or not _finite_metric(metrics):
        raise RuntimeError("actor train step returned invalid metrics")
    return metrics


def _finite_metric(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(isinstance(key, str) and _finite_metric(item) for key, item in value.items())
    if isinstance(value, torch.Tensor):
        return value.numel() > 0 and bool(torch.isfinite(value).all())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value) and all(_finite_metric(item) for item in value)
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
