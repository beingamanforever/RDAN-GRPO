"""Fail-closed scalar RDAN preflight and ROLL batch injection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Hashable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from rdan_grpo.roll_scalar import QUALITY_METHODS, ScalarMethod, ScalarOutput, build_scalar_output

SHA256_LEN = 64
Metric = bool | int | float


@dataclass(frozen=True)
class BatchAssessment:
    """Scalar output plus batch-local validity and activity evidence."""

    output: ScalarOutput
    method: ScalarMethod
    prompt_keys: tuple[Hashable, ...]
    group_size: int
    quality_weight: float | None
    mix_weight: float | None
    batch_valid: bool
    reasons: tuple[str, ...]
    response_active_groups: int
    quality_active_groups: int


@dataclass(frozen=True)
class PreflightCertificate:
    """Immutable compact evidence that scalar training may start."""

    certificate_id: str
    ready: bool
    method: ScalarMethod
    quality_weight: float | None
    mix_weight: float | None
    config_sha256: str
    source_sha256: Mapping[str, str]
    metrics: Mapping[str, Metric]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible certificate payload."""

        payload = {
            "schema_version": 2,
            "certificate_id": self.certificate_id,
            "ready": self.ready,
            "method": self.method,
            "config_sha256": self.config_sha256,
            "source_sha256": dict(self.source_sha256),
            "metrics": dict(self.metrics),
            "reasons": list(self.reasons),
        }
        if self.quality_weight is not None:
            payload["quality_weight"] = self.quality_weight
        if self.mix_weight is not None:
            payload["mix_weight"] = self.mix_weight
        return payload


def assess_scalar_batch(
    prompt_keys: Sequence[Hashable],
    scores: Tensor,
    rubric_mask: Tensor,
    eval_mask: Tensor,
    hard_mask: Tensor,
    *,
    method: ScalarMethod = "rdan_scalar",
    unsupported_hard: Tensor | None = None,
    judge_failed: Tensor | None = None,
    group_size: int = 8,
    quality_weight: float | None = None,
    mix_weight: float | None = None,
    eps: float = 1e-6,
) -> BatchAssessment:
    """Assess one no-update batch while preserving both scalar channels."""

    quality_weight, mix_weight = _resolve_method_parameters(method, quality_weight, mix_weight)
    unsupported = _response_mask("unsupported_hard", unsupported_hard, scores.shape[0], scores.device)
    judge_fail = _response_mask("judge_failed", judge_failed, scores.shape[0], scores.device)
    output = build_scalar_output(
        method,
        prompt_keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        group_size=group_size,
        mix_weight=mix_weight,
        quality_weight=quality_weight,
    )

    hard_present = (rubric_mask & hard_mask).any(dim=-1)
    soft_only = ~hard_present
    quality_leak = (~output.hard_pass | judge_fail) & (
        output.quality_eligible | (output.quality_advantage.abs() > eps)
    )
    reasons: list[str] = []
    if not output.training_ready:
        reasons.append("invalid_evaluator_output")
    if bool(unsupported.any()):
        reasons.append("unsupported_hard_route")
    if bool(judge_fail.any()):
        reasons.append("judge_failure")
    if method == "rdan_scalar" and bool(soft_only.any()):
        reasons.append("soft_only_response")
    if method in QUALITY_METHODS and bool(quality_leak.any()):
        reasons.append("invalid_quality_credit")

    response_active, quality_active = _active_groups(output, group_size, eps)
    return BatchAssessment(
        output=output,
        method=method,
        prompt_keys=tuple(prompt_keys),
        group_size=group_size,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
        batch_valid=not reasons,
        reasons=tuple(reasons),
        response_active_groups=response_active,
        quality_active_groups=quality_active,
    )


def build_preflight_certificate(
    batches: Sequence[BatchAssessment],
    *,
    method: ScalarMethod = "rdan_scalar",
    config_sha256: str,
    source_sha256: Mapping[str, str],
    optimizer_updates: int,
    quality_weight: float | None = None,
    mix_weight: float | None = None,
    min_prompts: int = 256,
    max_prompts: int = 1024,
    min_quality_active_rate: float = 0.1,
) -> PreflightCertificate:
    """Aggregate no-update batches into a deterministic readiness certificate."""

    if not batches:
        raise ValueError("batches must not be empty")
    quality_weight, mix_weight = _resolve_method_parameters(method, quality_weight, mix_weight)
    if any(batch.method != method for batch in batches):
        raise ValueError("method does not match the assessed batches")
    if any(batch.quality_weight != quality_weight or batch.mix_weight != mix_weight for batch in batches):
        raise ValueError("method parameter does not match the assessed batches")
    _check_sha256("config_sha256", config_sha256)
    sources = _validated_sources(source_sha256)
    if isinstance(optimizer_updates, bool) or not isinstance(optimizer_updates, int) or optimizer_updates < 0:
        raise ValueError("optimizer_updates must be a non-negative integer")
    if min_prompts <= 0 or max_prompts < min_prompts:
        raise ValueError("prompt bounds are invalid")
    if not 0 <= min_quality_active_rate <= 1:
        raise ValueError("min_quality_active_rate must be in [0, 1]")

    group_sizes = {batch.group_size for batch in batches}
    prompt_keys = tuple(key for batch in batches for key in batch.prompt_keys[:: batch.group_size])
    group_count = len(prompt_keys)
    response_count = sum(batch.output.scalar_advantage.numel() for batch in batches)
    response_active = sum(batch.response_active_groups for batch in batches)
    quality_active = sum(batch.quality_active_groups for batch in batches)
    quality_active_rate = quality_active / group_count
    response_active_rate = response_active / group_count
    finite = all(bool(torch.isfinite(batch.output.scalar_advantage).all()) for batch in batches)

    reasons = {reason for batch in batches for reason in batch.reasons}
    if group_sizes != {8}:
        reasons.add("group_size_not_8")
    if not min_prompts <= group_count <= max_prompts:
        reasons.add("prompt_count_out_of_range")
    if len(set(prompt_keys)) != group_count:
        reasons.add("duplicate_prompt_group")
    if optimizer_updates != 0:
        reasons.add("optimizer_update_observed")
    if not all(batch.batch_valid for batch in batches):
        reasons.add("invalid_batch")
    if response_active == 0:
        reasons.add("zero_response_variance")
    if method in QUALITY_METHODS and quality_active_rate < min_quality_active_rate:
        reasons.add("low_quality_active_group_rate")
    if not finite:
        reasons.add("nonfinite_scalar_output")

    metrics: dict[str, Metric] = {
        "prompt_count": group_count,
        "response_count": response_count,
        "group_size": next(iter(group_sizes)) if len(group_sizes) == 1 else 0,
        "optimizer_updates": optimizer_updates,
        "batch_count": len(batches),
        "valid_batch_count": sum(batch.batch_valid for batch in batches),
        "response_active_group_count": response_active,
        "response_active_group_rate": response_active_rate,
        "quality_active_group_count": quality_active,
        "quality_active_group_rate": quality_active_rate,
        "finite": finite,
    }
    ordered_reasons = tuple(sorted(reasons))
    body: dict[str, Any] = {
        "schema_version": 2,
        "ready": not ordered_reasons,
        "method": method,
        "config_sha256": config_sha256,
        "source_sha256": dict(sources),
        "metrics": metrics,
        "reasons": list(ordered_reasons),
    }
    if quality_weight is not None:
        body["quality_weight"] = quality_weight
    if mix_weight is not None:
        body["mix_weight"] = mix_weight
    certificate_id = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
    return PreflightCertificate(
        certificate_id=certificate_id,
        ready=not ordered_reasons,
        method=method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
        config_sha256=config_sha256,
        source_sha256=sources,
        metrics=MappingProxyType(metrics),
        reasons=ordered_reasons,
    )


def inject_roll_advantages(
    data: MutableMapping[str, Tensor] | Any,
    output: ScalarOutput,
    response_mask: Tensor,
    certificate: PreflightCertificate | Mapping[str, Any],
) -> MutableMapping[str, Tensor] | Any:
    """Expand a certified scalar advantage over response tokens without renormalizing."""

    if not _certificate_ready(certificate):
        raise ValueError("a ready scalar preflight certificate is required")
    _require_method_binding(
        certificate,
        output.method,
        quality_weight=output.quality_weight,
        mix_weight=output.mix_weight,
    )
    if not output.training_ready:
        raise ValueError("scalar output is not batch-valid")
    if response_mask.dtype != torch.bool or response_mask.ndim != 2:
        raise ValueError("response_mask must be boolean with shape [responses, tokens]")
    if response_mask.shape[0] != output.scalar_advantage.numel():
        raise ValueError("response_mask must contain one row per scalar advantage")
    if not bool(response_mask.any(dim=-1).all()):
        raise ValueError("every response must contain at least one active token")

    batch = data if isinstance(data, MutableMapping) else getattr(data, "batch", None)
    if not isinstance(batch, MutableMapping):
        raise ValueError("data must be a mutable tensor mapping or expose one as .batch")
    scalar = output.scalar_advantage.to(device=response_mask.device)
    token_advantage = scalar.unsqueeze(-1) * response_mask.to(scalar.dtype)
    evidence = {
        "rdan_raw_aon": output.raw_aon,
        "rdan_raw_csr": output.raw_csr,
        "rdan_raw_signed_csr": output.raw_signed_csr,
        "rdan_selected_reward": output.selected_raw_reward,
        "rdan_response_advantage": output.response_advantage,
        "rdan_raw_quality": output.raw_quality,
        "rdan_quality_eligible": output.quality_eligible,
        "rdan_quality_advantage": output.quality_advantage,
        "rdan_scalar_advantage": output.scalar_advantage,
        "rdan_response_valid": output.response_valid,
    }
    batch.update({name: value.to(response_mask.device).clone() for name, value in evidence.items()})
    batch["raw_advantages"] = token_advantage.clone()
    batch["advantages"] = token_advantage
    return data


def make_roll_compute_advantage(
    certificate: PreflightCertificate | Mapping[str, Any],
    *,
    method: ScalarMethod = "rdan_scalar",
    group_size: int = 8,
    quality_weight: float | None = None,
    mix_weight: float | None = None,
) -> Callable[..., Any]:
    """Build a drop-in replacement for ROLL's pre-train advantage boundary."""

    if not _certificate_ready(certificate):
        raise ValueError("a ready scalar preflight certificate is required")
    quality_weight, mix_weight = _resolve_method_parameters(method, quality_weight, mix_weight)
    _require_method_binding(
        certificate,
        method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )

    def compute_scalar_advantage(data: Any, *, response_mask: Tensor | None = None, **_: Any) -> Any:
        batch = getattr(data, "batch", None)
        non_tensor = getattr(data, "non_tensor_batch", None)
        if not isinstance(batch, MutableMapping) or not isinstance(non_tensor, Mapping):
            raise ValueError("ROLL data must expose batch and non_tensor_batch mappings")
        required = ("rdan_scores", "rdan_rubric_mask", "rdan_eval_mask", "rdan_hard_mask")
        missing = [name for name in required if name not in batch]
        if "rdan_prompt_key" not in non_tensor:
            missing.append("rdan_prompt_key")
        if missing:
            raise ValueError(f"ROLL reward output is missing scalar fields: {', '.join(missing)}")
        mask = response_mask if response_mask is not None else batch.get("final_response_mask")
        if not isinstance(mask, Tensor):
            raise ValueError("ROLL data is missing final_response_mask")
        mask = mask.to(dtype=torch.bool)
        assessment = assess_scalar_batch(
            list(non_tensor["rdan_prompt_key"]),
            batch["rdan_scores"].float(),
            batch["rdan_rubric_mask"].bool(),
            batch["rdan_eval_mask"].bool(),
            batch["rdan_hard_mask"].bool(),
            method=method,
            unsupported_hard=batch.get("rdan_unsupported_hard"),
            judge_failed=batch.get("rdan_judge_failed"),
            group_size=group_size,
            quality_weight=quality_weight,
            mix_weight=mix_weight,
        )
        if not assessment.batch_valid:
            raise ValueError(f"scalar ROLL batch failed closed: {', '.join(assessment.reasons)}")
        inject_roll_advantages(data, assessment.output, mask, certificate)
        batch["returns"] = batch["advantages"].clone()
        return data

    compute_scalar_advantage.__name__ = f"compute_{method}_advantage"
    return compute_scalar_advantage


def install_roll_adapter(
    certificate: PreflightCertificate | Mapping[str, Any],
    *,
    method: ScalarMethod = "rdan_scalar",
    group_size: int = 8,
    quality_weight: float | None = None,
    mix_weight: float | None = None,
) -> Callable[..., Any]:
    """Install the scalar boundary into the pinned RTT response-level ROLL pipeline."""

    from roll.pipeline.rlvr import rubircs_pipeline

    adapter = make_roll_compute_advantage(
        certificate,
        method=method,
        group_size=group_size,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )
    rubircs_pipeline.compute_advantage = adapter
    return adapter


def write_certificate(certificate: PreflightCertificate, path: str | Path) -> None:
    """Write a certificate once and refuse to replace existing evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(certificate.as_dict()) + "\n")


def require_train_certificate(
    path: str | Path,
    *,
    method: ScalarMethod = "rdan_scalar",
    config_sha256: str | None = None,
    source_sha256: Mapping[str, str] | None = None,
    quality_weight: float | None = None,
    mix_weight: float | None = None,
) -> dict[str, Any]:
    """Load and verify the readiness gate required before any optimizer launch."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"scalar preflight certificate not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or payload.get("ready") is not True
        or payload.get("method") != method
    ):
        raise ValueError("scalar preflight certificate is not ready")
    if set(payload) != _certificate_keys(method):
        raise ValueError("scalar preflight certificate schema does not match its method")
    quality_weight, mix_weight = _resolve_method_parameters(method, quality_weight, mix_weight)
    body = {key: value for key, value in payload.items() if key != "certificate_id"}
    expected_id = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
    if payload.get("certificate_id") != expected_id:
        raise ValueError("scalar preflight certificate digest does not match its contents")
    _require_method_binding(
        payload,
        method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )
    if config_sha256 is not None and payload.get("config_sha256") != config_sha256:
        raise ValueError("scalar preflight certificate config hash does not match")
    if source_sha256 is not None:
        recorded = payload.get("source_sha256")
        if not isinstance(recorded, dict) or recorded != dict(source_sha256):
            raise ValueError("scalar preflight certificate source hashes do not match")
    return payload


def _require_method_binding(
    certificate: PreflightCertificate | Mapping[str, Any],
    method: ScalarMethod,
    *,
    quality_weight: float | None,
    mix_weight: float | None,
) -> None:
    recorded_method = (
        certificate.method if isinstance(certificate, PreflightCertificate) else certificate.get("method")
    )
    if recorded_method != method:
        raise ValueError("scalar preflight certificate method does not match")
    recorded_quality = (
        certificate.quality_weight
        if isinstance(certificate, PreflightCertificate)
        else certificate.get("quality_weight")
    )
    recorded_mix = (
        certificate.mix_weight if isinstance(certificate, PreflightCertificate) else certificate.get("mix_weight")
    )
    if recorded_quality != quality_weight:
        raise ValueError("scalar preflight certificate quality weight does not match")
    if recorded_mix != mix_weight:
        raise ValueError("scalar preflight certificate mix weight does not match")


def _resolve_method_parameters(
    method: ScalarMethod,
    quality_weight: float | None,
    mix_weight: float | None,
) -> tuple[float | None, float | None]:
    if method not in ("rl_aon", "rl_csr", "rl_mix", "rdan_scalar", "rtt_papo_response"):
        raise ValueError(f"unsupported scalar method: {method}")
    if method in QUALITY_METHODS and quality_weight is None:
        quality_weight = 0.5
    if method not in QUALITY_METHODS and quality_weight is not None:
        raise ValueError(f"quality_weight is not valid for {method}")
    if method != "rl_mix" and mix_weight is not None:
        raise ValueError(f"mix_weight is not valid for {method}")
    if method == "rl_mix" and mix_weight is None:
        raise ValueError("mix_weight is required")
    if quality_weight is not None:
        _check_method_weight("quality_weight", quality_weight)
    if mix_weight is not None:
        _check_method_weight("mix_weight", mix_weight, maximum=1.0)
    return quality_weight, mix_weight


def _check_method_weight(name: str, value: float, *, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (maximum is not None and value > maximum):
        interval = "[0, 1]" if maximum == 1 else "non-negative"
        raise ValueError(f"{name} must be {interval}")


def sha256_file(path: str | Path) -> str:
    """Hash one source file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_groups(output: ScalarOutput, group_size: int, eps: float) -> tuple[int, int]:
    selected = output.selected_raw_reward.reshape(-1, group_size)
    response_active = int((selected.std(dim=-1) > eps).sum().item())
    if output.method not in QUALITY_METHODS:
        return response_active, 0
    quality = output.raw_quality.reshape(-1, group_size)
    eligible = output.quality_eligible.reshape(-1, group_size)
    count = eligible.sum(dim=-1)
    mean = (quality * eligible).sum(dim=-1) / count.clamp_min(1)
    variance = ((quality - mean.unsqueeze(-1)).square() * eligible).sum(dim=-1) / (count - 1).clamp_min(1)
    quality_active = int(((count >= 2) & (variance.sqrt() > eps)).sum().item())
    return response_active, quality_active


def attach_roll_reward_fields(data: Any, output: Any, infos: Sequence[Mapping[str, Any]]) -> Any:
    """Attach fail-closed scalar evidence to one ROLL reward-worker output."""

    scores = output.batch["rubric_scores_list"].float()
    active = scores != -100
    if len(infos) != int(active.sum().item()):
        raise ValueError("ROLL rubric evaluator evidence count does not match active rubrics")
    hard = torch.zeros_like(active)
    evaluated = active.clone()
    judge_failed = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    unsupported = torch.zeros_like(judge_failed)
    sources = list(data.non_tensor_batch["source"])
    ground_truth = list(data.non_tensor_batch["ground_truth"])
    info_index = 0
    for row, count in enumerate(active.sum(dim=-1).tolist()):
        truth = _json_object(ground_truth[row])
        for rubric_index in range(int(count)):
            info = infos[info_index]
            info_index += 1
            method = str(info.get("method", ""))
            hard[row, rubric_index] = method.startswith("code_")
            if method == "llm_judge" and not str(info.get("llm_response", "")).strip():
                evaluated[row, rubric_index] = False
                judge_failed[row] = True
            if method == "llm_judge" and _expected_hard_route(str(sources[row]), truth, rubric_index):
                unsupported[row] = True

    prompt_values = data.non_tensor_batch.get("id")
    if prompt_values is None:
        prompt_values = [
            hashlib.sha256(str(prompt).encode()).hexdigest() for prompt in data.non_tensor_batch["prompt"]
        ]
    data.non_tensor_batch["rdan_prompt_key"] = prompt_values
    output.batch["rdan_scores"] = torch.where(active, scores, torch.zeros_like(scores))
    output.batch["rdan_rubric_mask"] = active
    output.batch["rdan_eval_mask"] = evaluated
    output.batch["rdan_hard_mask"] = hard
    output.batch["rdan_judge_failed"] = judge_failed
    output.batch["rdan_unsupported_hard"] = unsupported
    return output


def _expected_hard_route(source: str, truth: Mapping[str, Any], rubric_index: int) -> bool:
    if source in {"type1", "type2", "type3"}:
        return True
    if source != "type4":
        return False
    checkers = truth.get("checker", [])
    return (
        isinstance(checkers, Sequence)
        and rubric_index < len(checkers)
        and str(checkers[rubric_index]).startswith("[rule]")
    )


def _json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _response_mask(name: str, value: Tensor | None, size: int, device: torch.device) -> Tensor:
    if value is None:
        return torch.zeros(size, dtype=torch.bool, device=device)
    if value.dtype != torch.bool or value.shape != (size,):
        raise ValueError(f"{name} must be boolean with shape [responses]")
    return value.to(device=device)


def _validated_sources(source_sha256: Mapping[str, str]) -> Mapping[str, str]:
    if not source_sha256:
        raise ValueError("source_sha256 must not be empty")
    sources: dict[str, str] = {}
    for name, digest in sorted(source_sha256.items()):
        if not isinstance(name, str) or not name or any(char.isspace() for char in name):
            raise ValueError("source hash names must be non-empty and contain no whitespace")
        _check_sha256(f"source_sha256[{name!r}]", digest)
        sources[name] = digest
    return MappingProxyType(sources)


def _check_sha256(name: str, digest: str) -> None:
    if not isinstance(digest, str):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    if len(digest) != SHA256_LEN or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _certificate_ready(certificate: PreflightCertificate | Mapping[str, Any]) -> bool:
    if isinstance(certificate, PreflightCertificate):
        return certificate.ready and certificate.method in (
            "rl_aon",
            "rl_csr",
            "rl_mix",
            "rdan_scalar",
            "rtt_papo_response",
        )
    if certificate.get("ready") is not True or certificate.get("method") not in (
        "rl_aon",
        "rl_csr",
        "rl_mix",
        "rdan_scalar",
        "rtt_papo_response",
    ):
        return False
    if set(certificate) != _certificate_keys(certificate["method"]):
        return False
    certificate_id = certificate.get("certificate_id")
    if not isinstance(certificate_id, str):
        return False
    body = {key: value for key, value in certificate.items() if key != "certificate_id"}
    return certificate_id == hashlib.sha256(_canonical_json(body).encode()).hexdigest()


def _certificate_keys(method: ScalarMethod) -> set[str]:
    keys = {
        "schema_version",
        "certificate_id",
        "ready",
        "method",
        "config_sha256",
        "source_sha256",
        "metrics",
        "reasons",
    }
    if method in QUALITY_METHODS:
        keys.add("quality_weight")
    if method == "rl_mix":
        keys.add("mix_weight")
    return keys


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
