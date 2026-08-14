"""Fail-closed runtime parity evidence for the pinned Qwen ROLL topology."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch

MAX_ABS_ERROR = 1e-3
MEAN_ABS_ERROR = 1e-4
MIN_RESPONSES = 32
MAX_WORST_TOKEN_EVIDENCE = 16
FAILURE_CODES = {"alignment_failed", "receipt_failed", "receipt_linkage_failed", "threshold_exceeded"}
TRANSFORMERS_VERSION = "4.57.0"
GENERATION_SOURCE_SHA256 = {
    "generation_get_logits_processor_sha256": "9f4d47d0e175dccb2c5b463435e39d738f4c8c69b47f8cfd5891c3d8b20b85b5",
    "generation_sample_sha256": "676dcf123496b831d13024402537120a5a5dcb16133a342705892ac8bd2291d6",
}
GENERATION_SOURCE_IDENTITY = {
    "transformers_version": TRANSFORMERS_VERSION,
    **GENERATION_SOURCE_SHA256,
}


@dataclass(frozen=True)
class BackendProfile:
    """Exact caller-boundary identity for one supported parity topology."""

    actor_train_strategy: str
    actor_infer_strategy: str
    transformer_impl: str
    infer_logprobs_source: str
    generation_logprobs: bool


MEGATRON_VLLM_PROFILE = BackendProfile(
    actor_train_strategy="megatron_train",
    actor_infer_strategy="vllm",
    transformer_impl="local",
    infer_logprobs_source="observed_rollout_engine",
    generation_logprobs=True,
)
FSDP2_HF_PROFILE = BackendProfile(
    actor_train_strategy="fsdp2_train",
    actor_infer_strategy="hf_infer",
    transformer_impl="huggingface",
    infer_logprobs_source="observed_hf_generation",
    generation_logprobs=False,
)


class ParityError(ValueError):
    """Raised when runtime evidence cannot prove exact-boundary parity."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "alignment_failed",
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class RuntimeIdentity:
    """Immutable model, tokenizer, and template identity."""

    model: str
    revision: str
    snapshot_sha256: str
    tokenizer_files_sha256: str
    chat_template_sha256: str


@dataclass(frozen=True)
class ParityObservation:
    """Values observed at the rollout and actor-train caller boundary."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    infer_logprobs: torch.Tensor | None
    actor_logprobs: torch.Tensor
    actor_input_ids: torch.Tensor
    actor_attention_mask: torch.Tensor
    actor_response_mask: torch.Tensor
    infer_logprobs_source: str
    actor_train_recomputed: bool
    actor_boundary_observed: bool
    optimizer_updates: int
    worker_ids: tuple[str, ...] | None = None


class ParityBoundary(Protocol):
    """Caller boundary implemented by the live ROLL parity pipeline."""

    def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> ParityObservation:
        """Collect rollout evidence and recompute actor-train logprobs without training."""


def verify_transformers_generation_boundary() -> dict[str, str]:
    """Require the exact Transformers generation source used by streaming capture."""

    try:
        import transformers
        from transformers.generation.utils import GenerationMixin
    except (ImportError, AttributeError) as error:
        raise ParityError("pinned Transformers generation source is unavailable") from error
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ParityError("Transformers generation version drift detected")
    observed = {}
    for method, key in (
        (GenerationMixin._get_logits_processor, "generation_get_logits_processor_sha256"),
        (GenerationMixin._sample, "generation_sample_sha256"),
    ):
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError) as error:
            raise ParityError("pinned Transformers generation source is unavailable") from error
        observed[key] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if observed != GENERATION_SOURCE_SHA256:
        raise ParityError("Transformers generation source drift detected")
    return dict(GENERATION_SOURCE_IDENTITY)


def build_generation_config(
    pipeline_config: Any,
    backend_profile: BackendProfile = MEGATRON_VLLM_PROFILE,
) -> dict[str, Any]:
    """Build an immutable generation configuration for one exact backend profile."""

    if bool(getattr(pipeline_config, "async_pipeline", False)):
        raise ParityError("runtime parity requires async_pipeline=false")
    if getattr(pipeline_config, "async_generation_ratio", None) != 0:
        raise ParityError("runtime parity requires async_generation_ratio=0")
    allowed_levels = {0} if backend_profile == FSDP2_HF_PROFILE else {0, 1}
    if getattr(pipeline_config, "generate_opt_level", None) not in allowed_levels:
        raise ParityError("runtime parity requires a supported synchronous generate level")
    actor_infer = getattr(pipeline_config, "actor_infer", None)
    strategy_args = getattr(actor_infer, "strategy_args", None)
    if getattr(strategy_args, "strategy_name", None) != backend_profile.actor_infer_strategy:
        raise ParityError(f"runtime parity requires the {backend_profile.actor_infer_strategy} rollout strategy")
    generating_args = getattr(actor_infer, "generating_args", None)
    if generating_args is None or not callable(getattr(generating_args, "to_dict", None)):
        raise ParityError("actor_infer generating arguments are missing")
    config = dict(generating_args.to_dict())
    returns = config.get("num_return_sequences")
    if not isinstance(returns, int) or isinstance(returns, bool) or returns < 1:
        raise ParityError("num_return_sequences must be a positive integer")
    if backend_profile.generation_logprobs:
        config["logprobs"] = 1
    return config


def run_runtime_parity(
    boundary: ParityBoundary,
    identity: RuntimeIdentity,
    *,
    pipeline_config: Any,
    train_config_sha256: str,
    resolved_config_sha256: str,
    rtt_revision: str,
    weight_receipt: Mapping[str, str],
    production_train_config_sha256: str | None = None,
    production_resolved_config_sha256: str | None = None,
    preflight_train_config_sha256: str | None = None,
    preflight_resolved_config_sha256: str | None = None,
    responses: int = MIN_RESPONSES,
    artifact_id: str = "qwen_runtime_parity_v1",
    failure_output: str | Path | None = None,
    backend_profile: BackendProfile = MEGATRON_VLLM_PROFILE,
) -> dict[str, Any]:
    """Return passing evidence or atomically seal failure evidence before raising."""

    if not isinstance(responses, int) or isinstance(responses, bool) or responses < MIN_RESPONSES:
        raise ParityError(f"runtime parity requires at least {MIN_RESPONSES} responses")
    generation_config = build_generation_config(pipeline_config, backend_profile)
    _validate_identity(identity)
    config_hashes = (
        production_train_config_sha256,
        production_resolved_config_sha256,
        preflight_train_config_sha256,
        preflight_resolved_config_sha256,
    )
    backend = _backend_identity(
        pipeline_config,
        train_config_sha256,
        resolved_config_sha256,
        rtt_revision,
        backend_profile,
        *config_hashes,
    )
    receipt_linkage = _validate_receipt_linkage(weight_receipt, resolved_config_sha256)
    artifact_context = identity, backend, receipt_linkage, artifact_id, failure_output
    observation = _collect_parity_observation(boundary, responses, generation_config, *artifact_context)
    evidence = _assess_parity_observation(observation, responses, backend_profile, *artifact_context)
    return _parity_artifact(identity, backend, receipt_linkage, evidence, artifact_id)


def _collect_parity_observation(
    boundary: ParityBoundary,
    responses: int,
    generation_config: Mapping[str, Any],
    identity: RuntimeIdentity,
    backend: Mapping[str, str],
    receipt_linkage: Mapping[str, str],
    artifact_id: str,
    failure_output: str | Path | None,
) -> ParityObservation:
    try:
        return boundary.collect_parity(responses, generation_config)
    except Exception as error:
        _write_parity_failure(failure_output, identity, backend, receipt_linkage, None, error, artifact_id)
        raise


def _assess_parity_observation(
    observation: ParityObservation,
    responses: int,
    backend_profile: BackendProfile,
    identity: RuntimeIdentity,
    backend: Mapping[str, str],
    receipt_linkage: Mapping[str, str],
    artifact_id: str,
    failure_output: str | Path | None,
) -> dict[str, Any]:
    try:
        if (
            not isinstance(observation.input_ids, torch.Tensor)
            or observation.input_ids.ndim != 2
            or observation.input_ids.shape[0] < responses
        ):
            raise ParityError("runtime parity returned fewer responses than requested")
        return _assess(observation, backend_profile)
    except ParityError as error:
        _write_parity_failure(failure_output, identity, backend, receipt_linkage, observation, error, artifact_id)
        raise


def _write_parity_failure(
    failure_output: str | Path | None,
    identity: RuntimeIdentity,
    backend: Mapping[str, str],
    receipt_linkage: Mapping[str, str],
    observation: ParityObservation | None,
    error: Exception,
    artifact_id: str,
) -> None:
    if failure_output is None:
        return
    write_artifact(
        failure_output,
        _failure_artifact(identity, backend, receipt_linkage, observation, error, artifact_id),
    )


def _parity_artifact(
    identity: RuntimeIdentity,
    backend: Mapping[str, str],
    receipt_linkage: Mapping[str, str],
    evidence: Mapping[str, Any],
    artifact_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "id": artifact_id,
        "status": "parity_passed",
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
        "weight_receipt": receipt_linkage,
        "rollout_logprob_evidence": evidence,
    }


def build_runtime_identity(
    snapshot: str | Path,
    tokenizer: Any,
    *,
    model: str,
    revision: str,
) -> RuntimeIdentity:
    """Hash the exact local snapshot, tokenizer files, and tokenizer chat template."""

    root = Path(snapshot).resolve()
    if not root.is_dir():
        raise ParityError(f"model snapshot is not a directory: {root}")
    if root.name != revision:
        raise ParityError("model snapshot directory must be named with the pinned revision")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, (str, dict)) or not template:
        raise ParityError("pinned tokenizer chat template is missing")
    tokenizer_files = _tokenizer_files(root)
    return RuntimeIdentity(
        model=model,
        revision=revision,
        snapshot_sha256=_tree_hash(root, list(_all_files(root))),
        tokenizer_files_sha256=_tree_hash(root, tokenizer_files),
        chat_template_sha256=_json_hash(template),
    )


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Atomically create an immutable JSON artifact without replacing an existing file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(artifact), sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _assess(observation: ParityObservation, backend_profile: BackendProfile) -> dict[str, Any]:
    if observation.infer_logprobs is None:
        raise ParityError("rollout engine did not return token logprobs")
    if observation.infer_logprobs_source != backend_profile.infer_logprobs_source:
        raise ParityError("rollout logprobs came from a fallback source")
    if observation.actor_train_recomputed is not True:
        raise ParityError("actor-train logprobs were not independently recomputed")
    if observation.actor_boundary_observed is not True:
        raise ParityError("actor-train token boundaries were not observed inside the forward callback")
    if observation.optimizer_updates != 0:
        raise ParityError("runtime parity observed a nonzero optimizer update count")

    _validate_boundaries(observation)
    mask = observation.response_mask[:, 1:].bool()
    inferred = observation.infer_logprobs
    recomputed = observation.actor_logprobs
    expected_shape = mask.shape
    if inferred.shape != expected_shape or recomputed.shape != expected_shape:
        raise ParityError("rollout and actor-train logprobs do not match response boundaries")
    inferred_values = inferred.detach().float().cpu()[mask.cpu()]
    recomputed_values = recomputed.detach().float().cpu()[mask.cpu()]
    if inferred_values.numel() == 0:
        raise ParityError("runtime parity has no response tokens to compare")
    if not bool(torch.isfinite(inferred_values).all()) or not bool(torch.isfinite(recomputed_values).all()):
        raise ParityError("runtime parity logprobs must be finite")
    errors = (inferred_values - recomputed_values).abs()
    max_error = float(errors.max().item())
    mean_error = float(errors.mean().item())
    if max_error > MAX_ABS_ERROR or mean_error > MEAN_ABS_ERROR:
        diagnostics = _failure_diagnostics(
            observation, inferred.detach().float().cpu(), recomputed.detach().float().cpu(), mask.cpu()
        )
        payload = json.dumps(diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False)
        raise ParityError(
            f"runtime parity exceeds thresholds: diagnostics={payload}",
            code="threshold_exceeded",
            diagnostics=diagnostics,
        )
    return {
        "prompt_response_tokens_sha256": _prompt_response_hash(observation),
        "responses": int(observation.input_ids.shape[0]),
        "optimizer_updates": 0,
        "infer_logprobs_source": backend_profile.infer_logprobs_source,
        "actor_train_recomputed": True,
        "actor_boundary_observed": True,
        "compared_tokens": int(errors.numel()),
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "thresholds": {
            "max_abs_error_at_most": MAX_ABS_ERROR,
            "mean_abs_error_at_most": MEAN_ABS_ERROR,
        },
    }


def _failure_artifact(
    identity: RuntimeIdentity,
    backend: Mapping[str, str],
    weight_receipt: Mapping[str, str],
    observation: ParityObservation | None,
    error: Exception,
    artifact_id: str,
) -> dict[str, Any]:
    diagnostics = {}
    if observation is not None:
        diagnostics = (
            error.diagnostics
            if isinstance(error, ParityError) and error.diagnostics is not None
            else _available_diagnostics(observation)
        )
    aggregate_keys = (
        "absolute_error_fractions",
        "absolute_error_percentiles",
        "actor_mean_logprob",
        "actor_shift_mean_abs_error",
        "infer_mean_logprob",
        "max_abs_error",
        "mean_abs_error",
        "rmse",
        "signed_mean_difference",
    )
    comparison = {
        "returned_responses": _response_count(observation) if observation is not None else None,
        "compared_responses": diagnostics.get("compared_responses", 0),
        "compared_tokens": diagnostics.get("compared_tokens", 0),
        "optimizer_updates": _optimizer_updates(observation),
        "thresholds": {
            "max_abs_error_at_most": MAX_ABS_ERROR,
            "mean_abs_error_at_most": MEAN_ABS_ERROR,
        },
        "aggregate_error_stats": {key: diagnostics[key] for key in aggregate_keys if key in diagnostics},
        "response_position_aggregates": diagnostics.get("response_position_bins", {}),
        "worst_token_evidence": diagnostics.get("worst_token_evidence", []),
    }
    if "worker_aggregates" in diagnostics:
        comparison["worker_aggregates"] = diagnostics["worker_aggregates"]
    return {
        "schema_version": 2,
        "id": f"{artifact_id}_failure",
        "status": "parity_failed",
        "failure": {
            "code": _safe_failure_code(error),
            "type": _safe_error_type(error),
        },
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
        "runtime_backend": dict(backend),
        "weight_receipt": dict(weight_receipt),
        "comparison": comparison,
    }


def _available_diagnostics(observation: ParityObservation) -> dict[str, Any]:
    inferred = observation.infer_logprobs
    actor = observation.actor_logprobs
    response_mask = observation.response_mask
    if (
        not isinstance(inferred, torch.Tensor)
        or not isinstance(actor, torch.Tensor)
        or not isinstance(response_mask, torch.Tensor)
        or response_mask.ndim != 2
    ):
        return {}
    mask = response_mask[:, 1:].detach().cpu().bool()
    inferred = inferred.detach().float().cpu()
    actor = actor.detach().float().cpu()
    if inferred.shape != mask.shape or actor.shape != mask.shape or not bool(mask.any()):
        return {}
    values = (inferred[mask], actor[mask])
    if not all(bool(torch.isfinite(value).all()) for value in values):
        return {}
    return _failure_diagnostics(observation, inferred, actor, mask)


def _safe_failure_code(error: Exception) -> str:
    code = error.code if isinstance(error, ParityError) else None
    return code if code in FAILURE_CODES else "alignment_failed"


def _safe_error_type(error: Exception) -> str:
    name = type(error).__name__
    return name if len(name) <= 64 and name.isascii() and name.isidentifier() else "Exception"


def _failure_diagnostics(
    observation: ParityObservation,
    inferred: torch.Tensor,
    actor: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    inferred_values = inferred[mask].double()
    actor_values = actor[mask].double()
    signed_errors = inferred_values - actor_values
    absolute_errors = signed_errors.abs()
    positions = mask.long().cumsum(dim=1) - 1
    adjacent = mask[:, :-1] & mask[:, 1:]
    bins = {
        "0": positions == 0,
        "1": positions == 1,
        "2": positions == 2,
        "3": positions == 3,
        "4-15": (positions >= 4) & (positions <= 15),
        "16-63": (positions >= 16) & (positions <= 63),
        "64-255": (positions >= 64) & (positions <= 255),
        "256+": positions >= 256,
    }
    return {
        "absolute_error_fractions": {
            "<=1e-3": float((absolute_errors <= 1e-3).double().mean().item()),
            "<=1e-2": float((absolute_errors <= 1e-2).double().mean().item()),
            "<=5e-2": float((absolute_errors <= 5e-2).double().mean().item()),
            "<=1e-1": float((absolute_errors <= 1e-1).double().mean().item()),
        },
        "absolute_error_percentiles": {
            name: float(torch.quantile(absolute_errors, quantile).item())
            for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
        },
        "actor_mean_logprob": float(actor_values.mean().item()),
        "actor_shift_mean_abs_error": {
            "-1": _mean_abs_error(inferred[:, 1:][adjacent], actor[:, :-1][adjacent]),
            "+1": _mean_abs_error(inferred[:, :-1][adjacent], actor[:, 1:][adjacent]),
        },
        "compared_tokens": int(absolute_errors.numel()),
        "compared_responses": int(mask.any(dim=1).sum().item()),
        "first_response_token": _error_stats(absolute_errors[positions[mask] == 0]),
        "infer_mean_logprob": float(inferred_values.mean().item()),
        "later_response_tokens": _error_stats(absolute_errors[positions[mask] > 0]),
        "max_abs_error": float(absolute_errors.max().item()),
        "mean_abs_error": float(absolute_errors.mean().item()),
        "response_position_bins": {
            name: _error_stats(absolute_errors[selected[mask]]) for name, selected in bins.items()
        },
        "rmse": float(signed_errors.square().mean().sqrt().item()),
        "signed_mean_difference": float(signed_errors.mean().item()),
        "thresholds": {
            "max_abs_error_at_most": MAX_ABS_ERROR,
            "mean_abs_error_at_most": MEAN_ABS_ERROR,
        },
        "worst_token_evidence": _worst_token_evidence(observation, inferred, actor, mask),
        **_worker_aggregates(observation, absolute_errors, mask),
    }


def _worst_token_evidence(
    observation: ParityObservation,
    inferred: torch.Tensor,
    actor: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, int | float | str]]:
    if (
        not isinstance(observation.input_ids, torch.Tensor)
        or not isinstance(observation.response_mask, torch.Tensor)
        or observation.input_ids.shape != observation.response_mask.shape
        or observation.input_ids.ndim != 2
        or observation.input_ids.shape[0] != mask.shape[0]
    ):
        return []
    coordinates = mask.nonzero(as_tuple=False)
    response_positions = (mask.long().cumsum(dim=1) - 1)[mask]
    inferred_values = inferred[mask].double()
    actor_values = actor[mask].double()
    signed_errors = inferred_values - actor_values
    order = sorted(
        range(len(coordinates)),
        key=lambda index: (
            -float(abs(signed_errors[index].item())),
            int(coordinates[index, 0].item()),
            int(response_positions[index].item()),
        ),
    )[:MAX_WORST_TOKEN_EVIDENCE]
    response_hashes = _response_hashes(observation)
    return [
        {
            "response_sha256": response_hashes[int(coordinates[index, 0].item())],
            "response_position": int(response_positions[index].item()),
            "infer_logprob": float(inferred_values[index].item()),
            "actor_logprob": float(actor_values[index].item()),
            "signed_difference": float(signed_errors[index].item()),
            "abs_error": float(abs(signed_errors[index].item())),
        }
        for index in order
    ]


def _worker_aggregates(
    observation: ParityObservation,
    absolute_errors: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, list[dict[str, int | float | str]]]:
    worker_ids = observation.worker_ids
    if (
        worker_ids is None
        or len(worker_ids) != mask.shape[0]
        or any(not isinstance(value, str) or not value for value in worker_ids)
    ):
        return {}
    aggregates = []
    response_rows = torch.arange(mask.shape[0]).unsqueeze(1).expand_as(mask)[mask]
    for worker_id in sorted(set(worker_ids)):
        rows = torch.tensor([value == worker_id for value in worker_ids], dtype=torch.bool)
        aggregates.append(
            {
                "worker_sha256": _json_hash(worker_id),
                **_error_stats(absolute_errors[rows[response_rows]]),
            }
        )
    return {"worker_aggregates": aggregates}


def _error_stats(errors: torch.Tensor) -> dict[str, int | float | None]:
    if errors.numel() == 0:
        return {"count": 0, "mean_abs_error": None, "max_abs_error": None}
    return {
        "count": int(errors.numel()),
        "mean_abs_error": float(errors.mean().item()),
        "max_abs_error": float(errors.max().item()),
    }


def _mean_abs_error(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() == 0:
        return None
    return float((left.double() - right.double()).abs().mean().item())


def _validate_boundaries(observation: ParityObservation) -> None:
    tensors = {
        "input_ids": observation.input_ids,
        "attention_mask": observation.attention_mask,
        "response_mask": observation.response_mask,
        "actor_input_ids": observation.actor_input_ids,
        "actor_attention_mask": observation.actor_attention_mask,
        "actor_response_mask": observation.actor_response_mask,
    }
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 for tensor in tensors.values()):
        raise ParityError("runtime parity token boundaries must be rank-two tensors")
    shape = observation.input_ids.shape
    if shape[0] < MIN_RESPONSES or any(tensor.shape != shape for tensor in tensors.values()):
        raise ParityError("runtime parity requires at least 32 aligned response boundaries")
    if observation.worker_ids is not None and (
        len(observation.worker_ids) != shape[0]
        or any(not isinstance(worker_id, str) or not worker_id for worker_id in observation.worker_ids)
    ):
        raise ParityError("runtime parity worker identities do not match response boundaries")
    for original, recomputed in (
        (observation.input_ids, observation.actor_input_ids),
        (observation.attention_mask, observation.actor_attention_mask),
        (observation.response_mask, observation.actor_response_mask),
    ):
        if not torch.equal(original.detach().cpu(), recomputed.detach().cpu()):
            raise ParityError("actor-train recomputation changed prompt or response boundaries")
    attention = observation.attention_mask.detach().cpu().bool()
    response = observation.response_mask.detach().cpu().bool()
    for name, tensor in (
        ("attention_mask", observation.attention_mask),
        ("response_mask", observation.response_mask),
    ):
        values = tensor.detach().cpu()
        if not bool(((values == 0) | (values == 1)).all()):
            raise ParityError(f"{name} must be binary")
    if bool((response & ~attention).any()):
        raise ParityError("response mask includes padded tokens")
    for valid, response_row in zip(attention, response, strict=True):
        flags = response_row[valid]
        if not bool(flags.any()):
            raise ParityError("every parity response must contain at least one token")
        first = int(torch.nonzero(flags, as_tuple=False)[0].item())
        if first == 0 or bool(flags[:first].any()) or not bool(flags[first:].all()):
            raise ParityError("response tokens must form one contiguous suffix")


def _prompt_response_hash(observation: ParityObservation) -> str:
    rows = []
    inputs = observation.input_ids.detach().cpu()
    attention = observation.attention_mask.detach().cpu().bool()
    responses = observation.response_mask.detach().cpu().bool()
    for token_ids, valid, response in zip(inputs, attention, responses, strict=True):
        rows.append(
            {
                "prompt": token_ids[valid & ~response].tolist(),
                "response": token_ids[response].tolist(),
            }
        )
    return _json_hash(rows)


def _response_hashes(observation: ParityObservation) -> list[str]:
    inputs = observation.input_ids.detach().cpu()
    responses = observation.response_mask.detach().cpu().bool()
    return [_json_hash(token_ids[response].tolist()) for token_ids, response in zip(inputs, responses, strict=True)]


def _response_count(observation: ParityObservation) -> int | None:
    inputs = observation.input_ids
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        return None
    return int(inputs.shape[0])


def _optimizer_updates(observation: ParityObservation | None) -> int | None:
    if observation is None:
        return None
    updates = observation.optimizer_updates
    if not isinstance(updates, int) or isinstance(updates, bool):
        return None
    return updates


def _validate_identity(identity: RuntimeIdentity) -> None:
    if not identity.model or len(identity.revision) != 40:
        raise ParityError("model identity is incomplete")
    for digest in (
        identity.snapshot_sha256,
        identity.tokenizer_files_sha256,
        identity.chat_template_sha256,
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ParityError("runtime identity contains an invalid SHA-256 digest")


def _backend_identity(
    pipeline_config: Any,
    train_config_sha256: str,
    resolved_config_sha256: str,
    rtt_revision: str,
    backend_profile: BackendProfile,
    production_train_config_sha256: str | None,
    production_resolved_config_sha256: str | None,
    preflight_train_config_sha256: str | None,
    preflight_resolved_config_sha256: str | None,
) -> dict[str, str]:
    train_args = getattr(getattr(pipeline_config, "actor_train", None), "strategy_args", None)
    infer_args = getattr(getattr(pipeline_config, "actor_infer", None), "strategy_args", None)
    transformer_impl = _transformer_impl(getattr(train_args, "strategy_config", None))
    identity: dict[str, str] = {
        "train_config_sha256": train_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "actor_train_strategy": getattr(train_args, "strategy_name", None),
        "actor_infer_strategy": getattr(infer_args, "strategy_name", None),
        "transformer_impl": transformer_impl,
        "rtt_revision": rtt_revision,
    }
    _validate_backend_values(identity, backend_profile)
    if backend_profile == FSDP2_HF_PROFILE:
        identity.update(
            _same_backend_config_hashes(
                production_train_config_sha256,
                production_resolved_config_sha256,
                preflight_train_config_sha256,
                preflight_resolved_config_sha256,
            )
        )
        identity.update(verify_transformers_generation_boundary())
    return identity


def _transformer_impl(train_config: Any) -> Any:
    if isinstance(train_config, Mapping):
        return train_config.get("transformer_impl")
    return getattr(train_config, "transformer_impl", None)


def _validate_backend_values(identity: Mapping[str, str], backend_profile: BackendProfile) -> None:
    if not all(isinstance(value, str) and value for value in identity.values()):
        raise ParityError("runtime backend identity is incomplete")
    for name, digest in {
        "train config": identity["train_config_sha256"],
        "resolved config": identity["resolved_config_sha256"],
    }.items():
        if not _is_sha256(digest):
            raise ParityError(f"{name} contains an invalid SHA-256 digest")
    if identity["actor_train_strategy"] != backend_profile.actor_train_strategy:
        raise ParityError(f"runtime parity requires the {backend_profile.actor_train_strategy} actor-train strategy")
    if identity["actor_infer_strategy"] != backend_profile.actor_infer_strategy:
        raise ParityError(f"runtime parity requires the {backend_profile.actor_infer_strategy} actor-infer strategy")
    if identity["transformer_impl"] != backend_profile.transformer_impl:
        raise ParityError(f"runtime parity requires transformer_impl={backend_profile.transformer_impl}")
    if not _git_revision(identity["rtt_revision"]):
        raise ParityError("RTT revision must be a full Git revision")


def _same_backend_config_hashes(
    production_train_config_sha256: str | None,
    production_resolved_config_sha256: str | None,
    preflight_train_config_sha256: str | None,
    preflight_resolved_config_sha256: str | None,
) -> dict[str, str]:
    hashes = {
        "production_train_config_sha256": production_train_config_sha256,
        "production_resolved_config_sha256": production_resolved_config_sha256,
        "preflight_train_config_sha256": preflight_train_config_sha256,
        "preflight_resolved_config_sha256": preflight_resolved_config_sha256,
    }
    if not all(_is_sha256(digest) for digest in hashes.values()):
        raise ParityError("same-backend parity requires exact raw and resolved production and preflight digests")
    return {name: digest for name, digest in hashes.items() if isinstance(digest, str)}


def _git_revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_receipt_linkage(linkage: Mapping[str, str], resolved_config_sha256: str) -> dict[str, str]:
    expected_keys = {"transaction_id", "artifact_sha256", "resolved_config_sha256"}
    if not isinstance(linkage, Mapping) or set(linkage) != expected_keys:
        raise ParityError("runtime parity requires exact weight receipt linkage")
    transaction_id = linkage.get("transaction_id")
    artifact_sha256 = linkage.get("artifact_sha256")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ParityError("weight receipt transaction ID is invalid")
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise ParityError("weight receipt artifact digest is invalid")
    if linkage.get("resolved_config_sha256") != resolved_config_sha256:
        raise ParityError("weight receipt and parity resolved config digests differ")
    return dict(linkage)


def _tokenizer_files(root: Path) -> list[Path]:
    names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
    files = [path for path in _all_files(root) if path.name in names]
    if not any(path.name == "tokenizer_config.json" for path in files):
        raise ParityError("snapshot is missing tokenizer_config.json")
    return files


def _all_files(root: Path) -> tuple[Path, ...]:
    files = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files:
        raise ParityError("model snapshot contains no files")
    return files


def _tree_hash(root: Path, files: list[Path]) -> str:
    entries = []
    for path in files:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        entries.append({"path": path.relative_to(root).as_posix(), "bytes": size, "sha256": digest.hexdigest()})
    return _json_hash(entries)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()
