"""Strict runner-issued evidence for the response recovery and pilot lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rdan_grpo.roll_response_checkpoint import ArtifactIdentity, CheckpointIdentity, load_checkpoint
from rdan_grpo.roll_response_receipt import build_response_receipt

SCHEMA_VERSION = 1
_CERTIFICATE_KEYS = {
    "schema_version",
    "id",
    "status",
    "stage",
    "completed_step",
    "start_mode",
    "identity",
    "runtime_identity",
    "model_identity",
    "checkpoint",
    "step_artifacts",
    "predecessor",
    "resume_checkpoint",
}
_ARTIFACT_FILES = {
    "diagnostics.json",
    "groups.jsonl",
    "metrics.json",
    "receipts/initial.json",
    "receipts/post-update.json",
    "responses.jsonl",
}
_RESPONSE_KEYS = {
    "response_index",
    "prompt_key",
    "generation_id",
    "prompt",
    "source",
    "ground_truth",
    "rubrics",
    "response_tokens",
    "response_text",
    "response_length",
    "reward",
    "rubric_outcomes",
    "failures",
}
_REWARD_KEYS = {
    "raw_aon",
    "raw_csr",
    "raw_signed_csr",
    "selected_reward",
    "response_advantage",
    "raw_quality",
    "quality_eligible",
    "quality_advantage",
    "scalar_advantage",
    "response_valid",
}
_RUBRIC_KEYS = {"scores", "rubric_mask", "eval_mask", "hard_mask", "evidence"}
_FAILURE_KEYS = {"judge_failed", "unsupported_hard"}
_GROUP_KEYS = {
    "group_index",
    "prompt_key",
    "selected_rewards",
    "selected_reward_variance",
    "quality_eligible_count",
    "conditional_quality_variance",
}
_RUBRIC_BASE_KEYS = {"id", "category", "description", "weight"}
_RUBRIC_ROUTE_KEYS = {"verifier", "function", "parameters"}
_EVIDENCE_KEYS = {
    "rubric_index",
    "rubric_id",
    "rubric_description_sha256",
    "score",
    "evaluator_route",
    "reason",
    "generation_id",
    "request_sha256",
    "judge_provenance",
    "judge_failed",
    "evaluator_failed",
    "reward_lane",
    "judge_role",
    "fallback_reason",
}
_JUDGE_PROVENANCE_KEYS = {
    "generation_id",
    "selected_endpoint",
    "provider",
    "model",
    "finish_reason",
    "service_tier",
    "schema_id",
    "rubric_ids",
    "tokens",
    "reasoning_effort",
    "latency_ms",
    "cost",
    "generation_metadata_polls",
    "error",
    "request_sha256",
}
_REDACTED_TOKENS = "[REDACTED]"
_NUMERIC_REWARDS = _REWARD_KEYS - {"quality_eligible", "response_valid"}
_BOOLEAN_REWARDS = {"quality_eligible", "response_valid"}
_OUTCOME_TOKEN = object()
_ExpectedIdentity = tuple[CheckpointIdentity, Mapping[str, Any], Mapping[str, Any]]
_CertificateContract = tuple[
    str,
    int,
    str,
    range,
    CheckpointIdentity,
    Mapping[str, Any],
    Mapping[str, Any],
]


class LifecycleCertificateError(ValueError):
    """Raised when lifecycle evidence is incomplete, forged, or stale."""


@dataclass
class CompletedResponseRun:
    """Opaque single-use evidence returned only by a successful pipeline run."""

    _token: object
    identity: CheckpointIdentity
    runtime_identity: Mapping[str, Any]
    model_identity: Mapping[str, Any]
    checkpoints: tuple[Path, ...]
    artifact_root: Path
    predecessor: Path | None
    resume_checkpoint: Path | None
    _consumed: bool = False

    def _consume(self) -> None:
        if self._token is not _OUTCOME_TOKEN or self._consumed:
            raise LifecycleCertificateError("lifecycle runner outcome is invalid or already consumed")
        self._consumed = True


def _complete_response_run(
    *,
    identity: CheckpointIdentity,
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    checkpoints: Sequence[Path],
    artifact_root: Path,
    predecessor: Path | None,
    resume_checkpoint: Path | None,
) -> CompletedResponseRun:
    """Seal the successful pipeline result before returning to its caller."""

    return CompletedResponseRun(
        _OUTCOME_TOKEN,
        identity,
        dict(runtime_identity),
        dict(model_identity),
        tuple(checkpoints),
        artifact_root,
        predecessor,
        resume_checkpoint,
    )


def issue_lifecycle_certificate(
    output: str | Path,
    *,
    stage: str,
    outcome: CompletedResponseRun,
) -> Path:
    """Validate a completed runner outcome and atomically issue its certificate."""

    if not isinstance(outcome, CompletedResponseRun):
        raise LifecycleCertificateError("lifecycle issuance requires a completed runner outcome")
    outcome._consume()
    identity = outcome.identity
    expected_step, start_mode, artifact_steps = _stage_contract(stage)
    if not outcome.checkpoints or outcome.checkpoints[-1].name != f"step-{expected_step:06d}":
        raise LifecycleCertificateError("successful lifecycle run did not promote its final checkpoint")
    checkpoint_path, checkpoint_digest = _checkpoint_evidence(outcome.checkpoints[-1], identity, expected_step)
    artifacts = [
        _step_artifact_evidence(outcome.artifact_root, step, final=step == expected_step) for step in artifact_steps
    ]
    predecessor_value = None
    if outcome.predecessor is not None:
        predecessor_path, predecessor_payload = _load_certificate(outcome.predecessor)
        predecessor_value = _certificate_link(predecessor_path, predecessor_payload)
    resume_value = None
    if outcome.resume_checkpoint is not None:
        resume_path, resume_digest = _checkpoint_evidence(outcome.resume_checkpoint, identity, 1)
        resume_value = {"path": str(resume_path), "manifest_sha256": resume_digest}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "id": f"qwen_response_{stage}_certificate_v1",
        "status": "lifecycle_passed",
        "stage": stage,
        "completed_step": expected_step,
        "start_mode": start_mode,
        "identity": asdict(identity),
        "runtime_identity": dict(outcome.runtime_identity),
        "model_identity": dict(outcome.model_identity),
        "checkpoint": {"path": str(checkpoint_path), "manifest_sha256": checkpoint_digest},
        "step_artifacts": artifacts,
        "predecessor": predecessor_value,
        "resume_checkpoint": resume_value,
    }
    expected = (identity, outcome.runtime_identity, outcome.model_identity)
    _validate_certificate(Path(output).expanduser().absolute(), payload, expected)
    path = _atomic_write(output, payload)
    _validate_certificate(path, _load_certificate(path)[1], expected)
    return path


def validate_lifecycle_certificate(
    path: str | Path,
    *,
    expected_stage: str,
    expected_identity: CheckpointIdentity,
    expected_runtime_identity: Mapping[str, Any],
    expected_model_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Revalidate a certificate and its complete retained artifact chain."""

    certificate_path, payload = _load_certificate(path)
    if payload.get("stage") != expected_stage:
        raise LifecycleCertificateError(f"expected {expected_stage} lifecycle certificate")
    return _validate_certificate(
        certificate_path, payload, (expected_identity, expected_runtime_identity, expected_model_identity)
    )


def _validate_certificate(
    path: Path,
    payload: Mapping[str, Any],
    expected: _ExpectedIdentity | None,
) -> Mapping[str, Any]:
    stage, expected_step, start_mode, artifact_steps, identity, runtime_identity, model_identity = (
        _certificate_contract(payload, expected)
    )
    checkpoint_path = _certificate_checkpoint(payload, identity, expected_step)
    artifacts = _certificate_artifacts(payload, artifact_steps, expected_step)
    _validate_receipt_sequence(
        [Path(item["path"]) for item in artifacts],
        start_mode,
        identity,
        runtime_identity,
        model_identity,
    )
    _validate_lifecycle_start(path, payload, stage, identity, artifacts, runtime_identity, model_identity)
    _validate_final_checkpoint_link(checkpoint_path, artifacts[-1])
    return payload


def _certificate_contract(
    payload: Mapping[str, Any],
    expected: _ExpectedIdentity | None,
) -> _CertificateContract:
    if set(payload) != _CERTIFICATE_KEYS:
        raise LifecycleCertificateError("lifecycle certificate schema is invalid")
    stage = payload.get("stage")
    if not isinstance(stage, str):
        raise LifecycleCertificateError("lifecycle certificate stage is invalid")
    expected_step, start_mode, artifact_steps = _stage_contract(stage)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("id") != f"qwen_response_{stage}_certificate_v1"
        or payload.get("status") != "lifecycle_passed"
        or payload.get("completed_step") != expected_step
        or payload.get("start_mode") != start_mode
    ):
        raise LifecycleCertificateError("lifecycle certificate contract is invalid")
    identity_value = payload.get("identity")
    if not isinstance(identity_value, Mapping):
        raise LifecycleCertificateError("lifecycle certificate identity is invalid")
    identity = expected[0] if expected else _checkpoint_identity(identity_value)
    if identity_value != asdict(identity):
        raise LifecycleCertificateError("lifecycle certificate identity differs")
    runtime_identity = payload.get("runtime_identity")
    model_identity = payload.get("model_identity")
    if (
        not isinstance(runtime_identity, Mapping)
        or not isinstance(model_identity, Mapping)
        or (expected is not None and (runtime_identity != expected[1] or model_identity != expected[2]))
    ):
        raise LifecycleCertificateError("lifecycle runtime or model identity differs")
    return stage, expected_step, start_mode, artifact_steps, identity, runtime_identity, model_identity


def _certificate_checkpoint(payload: Mapping[str, Any], identity: CheckpointIdentity, expected_step: int) -> Path:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "manifest_sha256"}:
        raise LifecycleCertificateError("lifecycle checkpoint evidence is invalid")
    checkpoint_path, checkpoint_digest = _checkpoint_evidence(checkpoint.get("path"), identity, expected_step)
    if checkpoint != {"path": str(checkpoint_path), "manifest_sha256": checkpoint_digest}:
        raise LifecycleCertificateError("lifecycle checkpoint manifest digest differs")
    return checkpoint_path


def _certificate_artifacts(
    payload: Mapping[str, Any],
    artifact_steps: range,
    expected_step: int,
) -> list[Mapping[str, Any]]:
    artifacts = payload.get("step_artifacts")
    if not isinstance(artifacts, list) or any(not isinstance(item, Mapping) for item in artifacts):
        raise LifecycleCertificateError("lifecycle step artifact sequence is incomplete")
    observed_steps = [item.get("step") for item in artifacts]
    if observed_steps != list(artifact_steps):
        raise LifecycleCertificateError("lifecycle step artifact sequence is incomplete")
    observed = [
        _step_artifact_evidence(Path(item["path"]).parent, step, final=step == expected_step)
        for item, step in zip(artifacts, artifact_steps)
    ]
    if artifacts != observed:
        raise LifecycleCertificateError("lifecycle step artifact evidence differs")
    generation_hashes = [item["generation_sha256"] for item in artifacts]
    if len(generation_hashes) != len(set(generation_hashes)):
        raise LifecycleCertificateError("lifecycle step generations are not fresh")
    return artifacts


def _validate_lifecycle_start(
    path: Path,
    payload: Mapping[str, Any],
    stage: str,
    identity: CheckpointIdentity,
    artifacts: Sequence[Mapping[str, Any]],
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> None:
    predecessor = payload.get("predecessor")
    resume = payload.get("resume_checkpoint")
    if stage == "recovery_step_1":
        if predecessor is not None or resume is not None:
            raise LifecycleCertificateError("step-1 recovery must be a fresh lifecycle root")
        return
    predecessor_path, predecessor_payload = _validated_predecessor(path, predecessor, stage, payload["identity"])
    if (
        predecessor_payload["runtime_identity"] != runtime_identity
        or predecessor_payload["model_identity"] != model_identity
    ):
        raise LifecycleCertificateError("lifecycle stage runtime or model identities differ")
    if stage == "recovery_step_2":
        _validate_recovery_resume(resume, identity, artifacts[0], predecessor_payload)
    elif resume is not None:
        raise LifecycleCertificateError("pilot certificate does not prove a fresh base start")


def _validated_predecessor(
    certificate_path: Path,
    predecessor: Any,
    stage: str,
    identity: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any]]:
    path, payload = _predecessor(predecessor, certificate_path)
    expected_stage = "recovery_step_1" if stage == "recovery_step_2" else "recovery_step_2"
    if payload.get("stage") != expected_stage:
        raise LifecycleCertificateError("lifecycle predecessor substitution detected")
    _validate_certificate(path, payload, None)
    _require_shared_identity(payload["identity"], identity, distinct=stage == "pilot_step_20")
    return path, payload


def _validate_recovery_resume(
    resume: Any,
    identity: CheckpointIdentity,
    artifact: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> None:
    if resume != predecessor["checkpoint"]:
        raise LifecycleCertificateError("recovery resume checkpoint is not the certified predecessor")
    resume_path, resume_digest = _checkpoint_evidence(resume["path"], identity, 1)
    if resume != {"path": str(resume_path), "manifest_sha256": resume_digest}:
        raise LifecycleCertificateError("recovery resume checkpoint digest differs")
    predecessor_artifact = predecessor["step_artifacts"][-1]
    if artifact["generation_sha256"] == predecessor_artifact["generation_sha256"]:
        raise LifecycleCertificateError("post-resume generation artifact is not fresh")
    previous_post = _json(Path(predecessor_artifact["path"]) / "receipts/post-update.json")
    current_initial = _json(Path(artifact["path"]) / "receipts/initial.json")
    if (
        (current_initial.get("phase"), current_initial.get("pipeline_step")) != ("resume_initial", 1)
        or _weight_manifest(current_initial) != _weight_manifest(previous_post)
        or current_initial.get("optimizer_updates") != previous_post.get("optimizer_updates")
    ):
        raise LifecycleCertificateError("recovery resume receipt does not prove restored predecessor bytes")


def _validate_final_checkpoint_link(checkpoint_path: Path, final_artifact: Mapping[str, Any]) -> None:
    checkpoint_step = _json(checkpoint_path / "artifacts/step.json")
    artifact_manifest = _json(Path(final_artifact["path"]) / "manifest.json")
    if artifact_manifest["checkpoint"] != {"path": str(checkpoint_path), "status": "pending_local_promotion"}:
        raise LifecycleCertificateError("final step artifact does not link the promoted checkpoint")
    if checkpoint_step != {
        "path": final_artifact["path"],
        "manifest_sha256": final_artifact["manifest_sha256"],
    }:
        raise LifecycleCertificateError("promoted checkpoint does not link the final step artifact")


def _stage_contract(stage: Any) -> tuple[int, str, range]:
    if stage == "recovery_step_1":
        return 1, "fresh_base", range(1, 2)
    if stage == "recovery_step_2":
        return 2, "restored_step_1", range(2, 3)
    if stage == "pilot_step_20":
        return 20, "fresh_base", range(1, 21)
    raise LifecycleCertificateError("lifecycle certificate stage is invalid")


def _checkpoint_evidence(value: Any, identity: CheckpointIdentity, step: int) -> tuple[Path, str]:
    path = _real_directory(value, "lifecycle checkpoint")
    manifest = load_checkpoint(path, identity=identity)
    if manifest["completed_step"] != step:
        raise LifecycleCertificateError(f"lifecycle checkpoint must be promoted at step {step}")
    return path, _file_sha256(path / "manifest.json")


def _step_artifact_evidence(root_value: str | Path, step: int, *, final: bool) -> dict[str, Any]:
    root = _real_directory(root_value, "step artifact root")
    path = _real_directory(root / f"step-{step:06d}", "step artifact")
    manifest_path = path / "manifest.json"
    manifest = _canonical_json_object(manifest_path, "step artifact manifest")
    _validate_artifact_manifest(manifest, step, final)
    _validate_artifact_inventory(path, manifest.get("inventory"))
    _validate_artifact_contents(path, step)
    responses = _jsonl(path / "responses.jsonl", "response artifact")
    groups = _jsonl(path / "groups.jsonl", "group artifact")
    _validate_step_records(responses, groups)
    return _artifact_evidence(path, manifest_path, step, responses)


def _validate_artifact_manifest(manifest: Mapping[str, Any], step: int, final: bool) -> None:
    if set(manifest) != {"schema_version", "status", "step", "checkpoint", "inventory"}:
        raise LifecycleCertificateError("step artifact manifest schema is invalid")
    checkpoint = manifest.get("checkpoint")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "sealed"
        or manifest.get("step") != step
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"path", "status"}
        or checkpoint.get("status") != ("pending_local_promotion" if final else "not_scheduled")
    ):
        raise LifecycleCertificateError("step artifact manifest contract is invalid")


def _validate_artifact_inventory(path: Path, inventory: Any) -> None:
    inventory_paths = (
        {item.get("path") for item in inventory if isinstance(item, Mapping)} if isinstance(inventory, list) else set()
    )
    if not isinstance(inventory, list) or inventory_paths != _ARTIFACT_FILES:
        raise LifecycleCertificateError("step artifact inventory schema is incomplete")
    if [item.get("path") for item in inventory] != sorted(_ARTIFACT_FILES):
        raise LifecycleCertificateError("step artifact inventory order is invalid")
    for item in inventory:
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise LifecycleCertificateError("step artifact inventory entry is invalid")
        artifact = _regular_file(path / item["path"], "step artifact file")
        if item["size"] != artifact.stat().st_size or item["sha256"] != _file_sha256(artifact):
            raise LifecycleCertificateError("step artifact inventory digest differs")
    observed = {child.relative_to(path).as_posix() for child in path.rglob("*") if child.is_file()}
    if observed != {*_ARTIFACT_FILES, "manifest.json"} or any(child.is_symlink() for child in path.rglob("*")):
        raise LifecycleCertificateError("step artifact contains missing or unowned files")


def _validate_artifact_contents(path: Path, step: int) -> None:
    _validate_receipt(path / "receipts/initial.json")
    post = _validate_receipt(path / "receipts/post-update.json")
    if post.get("phase") != "post_update" or post.get("pipeline_step") != step:
        raise LifecycleCertificateError("step artifact post-update receipt linkage is invalid")
    _validate_metrics(_json(path / "metrics.json"), "step metrics")
    _validate_metrics(_json(path / "diagnostics.json"), "step diagnostics")


def _artifact_evidence(
    path: Path,
    manifest_path: Path,
    step: int,
    responses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    generation_sha = _sha256_json([row["generation_id"] for row in responses])
    return {
        "step": step,
        "path": str(path),
        "manifest_sha256": _file_sha256(manifest_path),
        "responses_sha256": _file_sha256(path / "responses.jsonl"),
        "generation_sha256": generation_sha,
        "initial_receipt_sha256": _file_sha256(path / "receipts/initial.json"),
        "post_receipt_sha256": _file_sha256(path / "receipts/post-update.json"),
    }


def _validate_receipt(path: Path) -> Mapping[str, Any]:
    receipt = _canonical_json_object(path, "response receipt")
    try:
        rebuilt = build_response_receipt(
            receipt["actor_receipts"],
            receipt["infer_receipts"],
            phase=receipt["phase"],
            pipeline_step=receipt["pipeline_step"],
            actor_counters=receipt["actor_counters"],
            resolved_config_sha256=receipt["runtime"]["resolved_config_sha256"],
            runtime_identity=receipt["runtime"],
            model_identity=receipt["model"],
            method=receipt["method"],
            fixed_weight=receipt["fixed_weight"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LifecycleCertificateError("response receipt schema is invalid") from error
    if receipt != rebuilt:
        raise LifecycleCertificateError("response receipt schema or digest differs")
    return receipt


def _weight_manifest(receipt: Mapping[str, Any]) -> Any:
    try:
        return receipt["actor_receipts"][0]["manifest_sha256"]
    except (KeyError, IndexError, TypeError) as error:
        raise LifecycleCertificateError("response receipt weight manifest is invalid") from error


def _validate_receipt_sequence(
    paths: Sequence[Path],
    start_mode: str,
    identity: CheckpointIdentity,
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> None:
    previous = None
    for index, path in enumerate(paths):
        initial = _json(path / "receipts/initial.json")
        post = _json(path / "receipts/post-update.json")
        _receipt_matches_identity(initial, identity, runtime_identity, model_identity)
        _receipt_matches_identity(post, identity, runtime_identity, model_identity)
        if index == 0:
            if start_mode == "fresh_base" and (initial.get("phase"), initial.get("pipeline_step")) != ("initial", 0):
                raise LifecycleCertificateError("fresh run does not start from an initial base receipt")
        elif initial != previous:
            raise LifecycleCertificateError("step receipts do not form a continuous transaction chain")
        previous = post


def _receipt_matches_identity(
    receipt: Mapping[str, Any],
    identity: CheckpointIdentity,
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> None:
    runtime = receipt.get("runtime")
    model = receipt.get("model")
    if (
        runtime != runtime_identity
        or model != model_identity
        or receipt.get("method") != identity.method
        or receipt.get("fixed_weight") != identity.method_weight
    ):
        raise LifecycleCertificateError("response receipt differs from the lifecycle identity")


def _validate_responses(rows: Sequence[Mapping[str, Any]]) -> None:
    generation_ids = []
    for index, row in enumerate(rows):
        _validate_response_schema(row, index)
        _validate_response_tokens(row)
        _validate_response_reward(row)
        _validate_response_rubrics(row)
        generation_ids.append(str(row["generation_id"]))
    if len(generation_ids) != len(set(generation_ids)):
        raise LifecycleCertificateError("response generation IDs are not unique")


def _validate_response_schema(row: Mapping[str, Any], index: int) -> None:
    if (
        set(row) != _RESPONSE_KEYS
        or row.get("response_index") != index
        or not _valid_generation_id(row.get("generation_id"))
        or not _positive_int(row.get("response_length"))
        or not isinstance(row.get("reward"), Mapping)
        or set(row["reward"]) != _REWARD_KEYS
        or not isinstance(row.get("rubric_outcomes"), Mapping)
        or set(row["rubric_outcomes"]) != _RUBRIC_KEYS
        or not isinstance(row.get("failures"), Mapping)
        or set(row["failures"]) != _FAILURE_KEYS
        or any(not isinstance(row.get(key), str) or not row[key] for key in ("prompt", "source", "response_text"))
        or not _strict_json_object(row.get("ground_truth"))
    ):
        raise LifecycleCertificateError("response artifact row schema is invalid")


def _validate_response_tokens(row: Mapping[str, Any]) -> None:
    tokens = row.get("response_tokens")
    if tokens == _REDACTED_TOKENS:
        return
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
        or row["response_length"] != len(tokens)
    ):
        raise LifecycleCertificateError("response token evidence is invalid")


def _validate_response_reward(row: Mapping[str, Any]) -> None:
    reward = row["reward"]
    failures = row["failures"]
    if any(not _finite_number(reward[key]) for key in _NUMERIC_REWARDS) or any(
        not isinstance(reward[key], bool) for key in _BOOLEAN_REWARDS
    ):
        raise LifecycleCertificateError("response reward evidence is invalid")
    if any(not isinstance(failures[key], bool) for key in _FAILURE_KEYS):
        raise LifecycleCertificateError("response failure evidence is invalid")


def _validate_response_rubrics(row: Mapping[str, Any]) -> None:
    outcomes = row["rubric_outcomes"]
    arrays = [outcomes[key] for key in ("scores", "rubric_mask", "eval_mask", "hard_mask")]
    rubrics = row.get("rubrics")
    evidence = outcomes["evidence"]
    if (
        not isinstance(rubrics, list)
        or not rubrics
        or any(not isinstance(value, list) for value in arrays)
        or not arrays[0]
        or len({len(value) for value in arrays}) != 1
        or any(not _finite_number(value) for value in arrays[0])
        or any(not isinstance(value, bool) for array in arrays[1:] for value in array)
        or any(active and not rubric for array in arrays[2:] for active, rubric in zip(array, arrays[1], strict=True))
        or len(rubrics) != sum(arrays[1])
        or not isinstance(evidence, list)
        or len(evidence) != sum(arrays[1])
        or any(not _valid_rubric(rubric, index) for index, rubric in enumerate(rubrics))
        or any(
            not _valid_evidence(item, rubrics[index], arrays[0][index], index) for index, item in enumerate(evidence)
        )
    ):
        raise LifecycleCertificateError("response rubric evidence is invalid")


def _valid_generation_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool) and bool(str(value))


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _validate_groups(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        rewards = row.get("selected_rewards")
        if (
            set(row) != _GROUP_KEYS
            or row.get("group_index") != index
            or isinstance(row.get("group_index"), bool)
            or not _valid_prompt_key(row.get("prompt_key"))
            or not isinstance(rewards, list)
            or not rewards
            or any(not _finite_number(value) for value in rewards)
            or not _finite_number(row.get("selected_reward_variance"), minimum=0)
            or not _nonnegative_int(row.get("quality_eligible_count"))
            or row["quality_eligible_count"] > len(rewards)
            or not _finite_number(row.get("conditional_quality_variance"), minimum=0)
        ):
            raise LifecycleCertificateError("group artifact row schema is invalid")


def _validate_step_records(responses: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> None:
    _validate_responses(responses)
    _validate_groups(groups)
    sizes = {len(group["selected_rewards"]) for group in groups}
    prompt_keys = [group.get("prompt_key") for group in groups]
    if (
        len(sizes) != 1
        or len(responses) != len(groups) * next(iter(sizes))
        or len(set(prompt_keys)) != len(prompt_keys)
    ):
        raise LifecycleCertificateError("response group size is inconsistent")
    size = next(iter(sizes))
    for group, start in zip(groups, range(0, len(responses), size), strict=True):
        members = responses[start : start + size]
        selected = [response["reward"]["selected_reward"] for response in members]
        eligible = [response["reward"]["quality_eligible"] for response in members]
        quality = [response["reward"]["raw_quality"] for response in members if response["reward"]["quality_eligible"]]
        if (
            any(response.get("prompt_key") != group["prompt_key"] for response in members)
            or group["selected_rewards"] != selected
            or group["quality_eligible_count"] != sum(eligible)
            or not math.isclose(group["selected_reward_variance"], _variance(selected), rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(group["conditional_quality_variance"], _variance(quality), rel_tol=1e-6, abs_tol=1e-8)
        ):
            raise LifecycleCertificateError("response group evidence is inconsistent")


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    try:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and (minimum is None or value >= minimum)
        )
    except OverflowError:
        return False


def _valid_rubric(value: Any, index: int) -> bool:
    if not isinstance(value, Mapping) or set(value) not in (
        _RUBRIC_BASE_KEYS,
        _RUBRIC_BASE_KEYS | _RUBRIC_ROUTE_KEYS,
    ):
        return False
    if (
        value.get("id") != index + 1
        or isinstance(value.get("id"), bool)
        or not isinstance(value.get("category"), str)
        or not isinstance(value.get("description"), str)
        or not value["description"].strip()
        or not _finite_number(value.get("weight"), minimum=0)
        or value["weight"] == 0
    ):
        return False
    return set(value) == _RUBRIC_BASE_KEYS or (
        isinstance(value.get("verifier"), str)
        and isinstance(value.get("function"), str)
        and isinstance(value.get("parameters"), Mapping)
        and (bool(value["verifier"]) or not value["function"])
    )


def _valid_evidence(value: Any, rubric: Mapping[str, Any], score: Any, index: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_KEYS:
        return False
    route = value.get("evaluator_route")
    judge = route == "llm_judge"
    return (
        value.get("rubric_index") == index
        and not isinstance(value.get("rubric_index"), bool)
        and value.get("rubric_id") == rubric["id"]
        and not isinstance(value.get("rubric_id"), bool)
        and value.get("rubric_description_sha256") == hashlib.sha256(rubric["description"].encode()).hexdigest()
        and _finite_number(value.get("score"))
        and value["score"] == score
        and isinstance(route, str)
        and bool(route)
        and isinstance(value.get("reason"), str)
        and isinstance(value.get("judge_failed"), bool)
        and isinstance(value.get("evaluator_failed"), bool)
        and isinstance(value.get("reward_lane"), str)
        and bool(value["reward_lane"])
        and _optional_string(value.get("judge_role"))
        and _optional_string(value.get("fallback_reason"))
        and ((judge and _valid_judge_evidence(value)) or (not judge and _empty_judge_evidence(value)))
    )


def _valid_judge_evidence(value: Mapping[str, Any]) -> bool:
    provenance = value.get("judge_provenance")
    return (
        isinstance(value.get("generation_id"), str)
        and bool(value["generation_id"])
        and _sha256(value.get("request_sha256"))
        and isinstance(provenance, Mapping)
        and set(provenance) == _JUDGE_PROVENANCE_KEYS
        and provenance.get("generation_id") == value["generation_id"]
        and provenance.get("request_sha256") == value["request_sha256"]
    )


def _empty_judge_evidence(value: Mapping[str, Any]) -> bool:
    return all(value.get(key) is None for key in ("generation_id", "request_sha256", "judge_provenance"))


def _optional_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _strict_json_object(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _valid_prompt_key(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool) and bool(str(value))


def _variance(values: Sequence[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _validate_metrics(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise LifecycleCertificateError(f"{name} schema is invalid")


def _predecessor(value: Any, certificate_path: Path) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "stage", "checkpoint_manifest_sha256"}:
        raise LifecycleCertificateError("lifecycle predecessor link is invalid")
    path, payload = _load_certificate(value.get("path"))
    if path == certificate_path or value != _certificate_link(path, payload):
        raise LifecycleCertificateError("lifecycle predecessor link digest differs")
    return path, payload


def _certificate_link(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "stage": payload["stage"],
        "checkpoint_manifest_sha256": payload["checkpoint"]["manifest_sha256"],
    }


def _require_shared_identity(left: Mapping[str, Any], right: Mapping[str, Any], *, distinct: bool) -> None:
    left_common = {key: value for key, value in left.items() if key != "wandb"}
    right_common = {key: value for key, value in right.items() if key != "wandb"}
    if left_common != right_common or distinct == (left.get("wandb") == right.get("wandb")):
        raise LifecycleCertificateError("lifecycle stage identities are not linked or distinct")


def _checkpoint_identity(value: Mapping[str, Any]) -> CheckpointIdentity:
    try:
        certificate = value["certificate"]
        data = value["data"]
        return CheckpointIdentity(
            planned_horizon=value["planned_horizon"],
            method=value["method"],
            method_weight=value["method_weight"],
            resolved_config_sha256=value["resolved_config_sha256"],
            certificate=ArtifactIdentity(certificate["id"], certificate["sha256"]),
            data=ArtifactIdentity(data["id"], data["sha256"]),
            revisions=value["revisions"],
            base_checkpoint_sha256=value["base_checkpoint_sha256"],
            wandb=value["wandb"],
        )
    except (KeyError, TypeError) as error:
        raise LifecycleCertificateError("lifecycle checkpoint identity schema is invalid") from error


def _load_certificate(value: str | Path) -> tuple[Path, Mapping[str, Any]]:
    path = _regular_file(value, "lifecycle certificate")
    return path, _canonical_json_object(path, "lifecycle certificate")


def _canonical_json_object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        body = path.read_bytes()
        value = json.loads(body)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LifecycleCertificateError(f"cannot read {name}") from error
    if not isinstance(value, Mapping) or body != _canonical_json(value):
        raise LifecycleCertificateError(f"{name} must be a canonical JSON object")
    return value


def _json(path: Path) -> Mapping[str, Any]:
    return _canonical_json_object(_regular_file(path, "lifecycle JSON artifact"), "lifecycle JSON artifact")


def _jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    artifact = _regular_file(path, name)
    rows = []
    try:
        for line in artifact.read_bytes().splitlines(keepends=True):
            value = json.loads(line)
            if not isinstance(value, Mapping) or line != _canonical_json(value):
                raise LifecycleCertificateError(f"{name} row is not canonical JSON")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LifecycleCertificateError(f"cannot read {name}") from error
    if not rows:
        raise LifecycleCertificateError(f"{name} must not be empty")
    return rows


def _atomic_write(value: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(value).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise LifecycleCertificateError("lifecycle certificate parent must be canonical")
    lock = path.parent / f".{path.name}.lock"
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    lock_fd = None
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        if path.exists() or path.is_symlink():
            raise LifecycleCertificateError("lifecycle certificate already exists")
        descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        temp.unlink()
        _fsync_directory(path.parent)
    except OSError as error:
        raise LifecycleCertificateError(f"cannot issue lifecycle certificate: {error}") from error
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if temp.exists():
            temp.unlink()
        if lock_fd is not None and lock.exists():
            lock.unlink()
            _fsync_directory(path.parent)
    return path


def _real_directory(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise LifecycleCertificateError(f"{name} path is invalid")
    path = Path(value).expanduser().absolute()
    try:
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
            raise LifecycleCertificateError(f"{name} must be a canonical real directory")
    except OSError as error:
        raise LifecycleCertificateError(f"cannot inspect {name}") from error
    return path


def _regular_file(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise LifecycleCertificateError(f"{name} path is invalid")
    path = Path(value).expanduser().absolute()
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.resolve(strict=True) != path:
            raise LifecycleCertificateError(f"{name} must be a canonical regular file")
    except OSError as error:
        raise LifecycleCertificateError(f"cannot inspect {name}") from error
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).removesuffix(b"\n")).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
