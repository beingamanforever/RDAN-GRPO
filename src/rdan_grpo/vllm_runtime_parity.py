"""Diagnostic-only production vLLM parity evidence for response training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from rdan_grpo.runtime_parity import ParityObservation, RuntimeIdentity, write_artifact

COMPARISON_POLICY = "diagnostic_only_actor_recompute_authoritative"
INFER_LOGPROBS_SOURCE = "observed_vllm_generation"
MIN_RESPONSES = 32


class VLLMParityError(ValueError):
    """Raised when vLLM parity cannot prove its non-numeric launch invariants."""


class VLLMParityBoundary(Protocol):
    """No-update boundary implemented by the live FSDP2 and vLLM pipeline."""

    def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> ParityObservation:
        """Generate with vLLM and recompute the same token boundaries on FSDP2."""


def run_vllm_runtime_parity(
    boundary: VLLMParityBoundary,
    identity: RuntimeIdentity,
    *,
    pipeline_config: Any,
    parity_config_sha256: str,
    parity_resolved_config_sha256: str,
    production_config_sha256: str,
    production_resolved_config_sha256: str,
    rtt_revision: str,
    weight_receipt: Mapping[str, str],
    responses: int = MIN_RESPONSES,
    artifact_id: str = "qwen_vllm_runtime_parity_v1",
) -> dict[str, Any]:
    """Build passing one-time vLLM evidence without applying a drift threshold."""

    if isinstance(responses, bool) or not isinstance(responses, int) or responses < MIN_RESPONSES:
        raise VLLMParityError(f"vLLM parity requires at least {MIN_RESPONSES} responses")
    generation_config = _generation_config(pipeline_config)
    backend = _backend(
        pipeline_config,
        parity_config_sha256,
        parity_resolved_config_sha256,
        production_config_sha256,
        production_resolved_config_sha256,
        rtt_revision,
    )
    receipt = _receipt_link(weight_receipt, parity_resolved_config_sha256)
    evidence = _assess(boundary.collect_parity(responses, generation_config), responses)
    return {
        "schema_version": 1,
        "id": artifact_id,
        "status": "parity_passed",
        "comparison_policy": COMPARISON_POLICY,
        "model": {
            "model": identity.model,
            "revision": identity.revision,
            "snapshot_sha256": identity.snapshot_sha256,
        },
        "tokenizer": {
            "model": identity.model,
            "revision": identity.revision,
            "files_sha256": identity.tokenizer_files_sha256,
        },
        "chat_template": {
            "source": "pinned_tokenizer",
            "enable_thinking": False,
            "sha256": identity.chat_template_sha256,
        },
        "runtime_backend": backend,
        "weight_receipt": receipt,
        "diagnostic": evidence,
    }


def validate_vllm_runtime_parity(
    artifact: Any,
    *,
    artifact_id: str,
    model: str,
    revision: str,
    rtt_revision: str,
    parity_config_sha256: str,
    production_config_sha256: str,
) -> Mapping[str, Any]:
    """Revalidate one serialized vLLM parity artifact at a launch boundary."""

    if not isinstance(artifact, Mapping):
        raise VLLMParityError("vLLM parity artifact must be an object")
    expected_keys = {
        "schema_version",
        "id",
        "status",
        "comparison_policy",
        "model",
        "tokenizer",
        "chat_template",
        "runtime_backend",
        "weight_receipt",
        "diagnostic",
    }
    if set(artifact) != expected_keys:
        raise VLLMParityError("vLLM parity artifact keys are invalid")
    model_identity = _mapping(artifact["model"], "model")
    tokenizer = _mapping(artifact["tokenizer"], "tokenizer")
    template = _mapping(artifact["chat_template"], "chat template")
    backend = _mapping(artifact["runtime_backend"], "runtime backend")
    receipt = _mapping(artifact["weight_receipt"], "weight receipt")
    diagnostic = _mapping(artifact["diagnostic"], "diagnostic")
    valid = (
        artifact["schema_version"] == 1
        and artifact["id"] == artifact_id
        and artifact["status"] == "parity_passed"
        and artifact["comparison_policy"] == COMPARISON_POLICY
        and model_identity
        == {"model": model, "revision": revision, "snapshot_sha256": model_identity.get("snapshot_sha256")}
        and _sha256(model_identity.get("snapshot_sha256"))
        and tokenizer == {"model": model, "revision": revision, "files_sha256": tokenizer.get("files_sha256")}
        and _sha256(tokenizer.get("files_sha256"))
        and template == {"source": "pinned_tokenizer", "enable_thinking": False, "sha256": template.get("sha256")}
        and _sha256(template.get("sha256"))
        and _valid_backend(backend, rtt_revision, parity_config_sha256, production_config_sha256)
        and _valid_receipt(receipt, backend)
        and _valid_diagnostic(diagnostic)
    )
    if not valid:
        raise VLLMParityError("vLLM runtime parity artifact is invalid")
    return artifact


def load_vllm_runtime_parity(path: str | Path, **expected: str) -> Mapping[str, Any]:
    """Load and validate one regular non-symlink vLLM parity artifact."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise VLLMParityError("vLLM parity artifact must be a regular file")
    try:
        artifact = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VLLMParityError("vLLM parity artifact cannot be parsed") from error
    return validate_vllm_runtime_parity(artifact, **expected)


def write_vllm_runtime_parity(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Create a validated immutable vLLM parity artifact."""

    write_artifact(path, artifact)


def _generation_config(pipeline_config: Any) -> dict[str, Any]:
    if bool(getattr(pipeline_config, "async_pipeline", False)):
        raise VLLMParityError("vLLM parity requires async_pipeline=false")
    if getattr(pipeline_config, "async_generation_ratio", None) != 0:
        raise VLLMParityError("vLLM parity requires async_generation_ratio=0")
    actor_train = getattr(pipeline_config, "actor_train", None)
    actor_infer = getattr(pipeline_config, "actor_infer", None)
    if _strategy(actor_train) != "fsdp2_train" or _strategy(actor_infer) != "vllm":
        raise VLLMParityError("vLLM parity requires fsdp2_train to vllm")
    generating_args = getattr(actor_infer, "generating_args", None)
    if generating_args is None or not callable(getattr(generating_args, "to_dict", None)):
        raise VLLMParityError("vLLM generation arguments are missing")
    config = dict(generating_args.to_dict())
    returns = config.get("num_return_sequences")
    if isinstance(returns, bool) or not isinstance(returns, int) or returns < 1:
        raise VLLMParityError("num_return_sequences must be a positive integer")
    config["logprobs"] = 1
    return config


def _backend(
    pipeline_config: Any,
    parity_config_sha256: str,
    parity_resolved_config_sha256: str,
    production_config_sha256: str,
    production_resolved_config_sha256: str,
    rtt_revision: str,
) -> dict[str, str]:
    values = {
        "parity_config_sha256": parity_config_sha256,
        "parity_resolved_config_sha256": parity_resolved_config_sha256,
        "production_config_sha256": production_config_sha256,
        "production_resolved_config_sha256": production_resolved_config_sha256,
        "actor_train_strategy": _strategy(pipeline_config.actor_train),
        "actor_infer_strategy": _strategy(pipeline_config.actor_infer),
        "transformer_impl": str(pipeline_config.actor_train.strategy_args.strategy_config.get("transformer_impl")),
        "rtt_revision": rtt_revision,
    }
    if any(not _sha256(values[name]) for name in values if name.endswith("sha256")):
        raise VLLMParityError("vLLM parity config identity is invalid")
    if values["actor_train_strategy"] != "fsdp2_train" or values["actor_infer_strategy"] != "vllm":
        raise VLLMParityError("vLLM parity backend identity is invalid")
    if values["transformer_impl"] != "huggingface" or not rtt_revision:
        raise VLLMParityError("vLLM parity runtime identity is invalid")
    return values


def _receipt_link(value: Mapping[str, str], resolved_config_sha256: str) -> dict[str, str]:
    receipt = dict(value)
    if not _valid_receipt(receipt, {"parity_resolved_config_sha256": resolved_config_sha256}):
        raise VLLMParityError("vLLM parity receipt linkage is invalid")
    return receipt


def _assess(observation: ParityObservation, responses: int) -> dict[str, Any]:
    tensors = (
        observation.input_ids,
        observation.attention_mask,
        observation.response_mask,
        observation.infer_logprobs,
        observation.actor_logprobs,
        observation.actor_input_ids,
        observation.actor_attention_mask,
        observation.actor_response_mask,
    )
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in tensors):
        raise VLLMParityError("vLLM parity tensors must be rank-2")
    if observation.input_ids.shape[0] < responses:
        raise VLLMParityError("vLLM parity returned fewer responses than requested")
    if not (
        torch.equal(observation.input_ids, observation.actor_input_ids)
        and torch.equal(observation.attention_mask, observation.actor_attention_mask)
        and torch.equal(observation.response_mask, observation.actor_response_mask)
    ):
        raise VLLMParityError("vLLM and actor token boundaries differ")
    mask = observation.response_mask[:, 1:].bool()
    if observation.infer_logprobs.shape != mask.shape or observation.actor_logprobs.shape != mask.shape:
        raise VLLMParityError("vLLM and actor logprobs have different token boundaries")
    if not bool(mask.any()) or not bool(torch.isfinite(observation.infer_logprobs[mask]).all()):
        raise VLLMParityError("vLLM sampled logprobs are empty or non-finite")
    if not bool(torch.isfinite(observation.actor_logprobs[mask]).all()):
        raise VLLMParityError("actor recomputed logprobs are non-finite")
    if observation.infer_logprobs_source != INFER_LOGPROBS_SOURCE:
        raise VLLMParityError("vLLM sampled logprobs were not observed at the rollout boundary")
    if observation.actor_train_recomputed is not True or observation.actor_boundary_observed is not True:
        raise VLLMParityError("actor recomputation boundary was not observed")
    if observation.optimizer_updates != 0:
        raise VLLMParityError("vLLM parity must not perform optimizer updates")
    inferred = observation.infer_logprobs[mask].detach().double().cpu()
    actor = observation.actor_logprobs[mask].detach().double().cpu()
    difference = inferred - actor
    absolute = difference.abs()
    return {
        "prompt_response_tokens_sha256": _token_hash(observation),
        "responses": int(observation.input_ids.shape[0]),
        "optimizer_updates": 0,
        "infer_logprobs_source": INFER_LOGPROBS_SOURCE,
        "actor_train_recomputed": True,
        "actor_boundary_observed": True,
        "exact_token_boundaries": True,
        "sampled_logprobs_finite": True,
        "compared_tokens": int(absolute.numel()),
        "source_mean_logprob": float(inferred.mean().item()),
        "target_mean_logprob": float(actor.mean().item()),
        "signed_mean_difference": float(difference.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
    }


def _token_hash(observation: ParityObservation) -> str:
    digest = hashlib.sha256()
    for tensor in (observation.input_ids, observation.attention_mask, observation.response_mask):
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _valid_backend(
    backend: Mapping[str, Any],
    rtt_revision: str,
    parity_config_sha256: str,
    production_config_sha256: str,
) -> bool:
    keys = {
        "parity_config_sha256",
        "parity_resolved_config_sha256",
        "production_config_sha256",
        "production_resolved_config_sha256",
        "actor_train_strategy",
        "actor_infer_strategy",
        "transformer_impl",
        "rtt_revision",
    }
    return bool(
        set(backend) == keys
        and backend.get("parity_config_sha256") == parity_config_sha256
        and backend.get("production_config_sha256") == production_config_sha256
        and all(_sha256(backend.get(key)) for key in keys if key.endswith("sha256"))
        and backend.get("actor_train_strategy") == "fsdp2_train"
        and backend.get("actor_infer_strategy") == "vllm"
        and backend.get("transformer_impl") == "huggingface"
        and backend.get("rtt_revision") == rtt_revision
    )


def _valid_receipt(receipt: Mapping[str, Any], backend: Mapping[str, Any]) -> bool:
    return bool(
        set(receipt) == {"transaction_id", "artifact_sha256", "resolved_config_sha256"}
        and isinstance(receipt.get("transaction_id"), str)
        and bool(receipt["transaction_id"])
        and _sha256(receipt.get("artifact_sha256"))
        and receipt.get("resolved_config_sha256") == backend.get("parity_resolved_config_sha256")
    )


def _valid_diagnostic(value: Mapping[str, Any]) -> bool:
    keys = {
        "prompt_response_tokens_sha256",
        "responses",
        "optimizer_updates",
        "infer_logprobs_source",
        "actor_train_recomputed",
        "actor_boundary_observed",
        "exact_token_boundaries",
        "sampled_logprobs_finite",
        "compared_tokens",
        "source_mean_logprob",
        "target_mean_logprob",
        "signed_mean_difference",
        "rmse",
        "max_abs_error",
        "mean_abs_error",
    }
    numeric = {
        "source_mean_logprob",
        "target_mean_logprob",
        "signed_mean_difference",
        "rmse",
        "max_abs_error",
        "mean_abs_error",
    }
    return bool(
        set(value) == keys
        and _sha256(value.get("prompt_response_tokens_sha256"))
        and isinstance(value.get("responses"), int)
        and not isinstance(value["responses"], bool)
        and value["responses"] >= MIN_RESPONSES
        and value.get("optimizer_updates") == 0
        and value.get("infer_logprobs_source") == INFER_LOGPROBS_SOURCE
        and value.get("actor_train_recomputed") is True
        and value.get("actor_boundary_observed") is True
        and value.get("exact_token_boundaries") is True
        and value.get("sampled_logprobs_finite") is True
        and isinstance(value.get("compared_tokens"), int)
        and not isinstance(value["compared_tokens"], bool)
        and value["compared_tokens"] > 0
        and all(
            isinstance(value.get(name), (int, float))
            and not isinstance(value[name], bool)
            and math.isfinite(value[name])
            for name in numeric
        )
        and value["max_abs_error"] >= 0
        and value["mean_abs_error"] >= 0
        and value["rmse"] >= 0
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VLLMParityError(f"vLLM parity {name} must be an object")
    return value


def _strategy(worker: Any) -> str:
    return str(getattr(getattr(worker, "strategy_args", None), "strategy_name", ""))


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
