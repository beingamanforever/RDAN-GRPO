#!/usr/bin/env python3
"""Run the receipted response-only production training gate."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdan_grpo.program import MODEL_NAME, MODEL_REVISION, require_launch_gate
from rdan_grpo.response_identity import (
    canonical_resolved_config_sha256,
    response_data_identity,
    response_source_hashes,
)
from rdan_grpo.response_identity import file_sha256 as _file_sha256
from rdan_grpo.response_pilot_lifecycle import (
    CompletedResponseRun,
    issue_lifecycle_certificate,
    validate_lifecycle_certificate,
)
from rdan_grpo.response_readiness import validate_response_readiness
from rdan_grpo.roll_bridge import require_train_certificate
from rdan_grpo.roll_compat import RTT_REVISION, install_rtt_compat
from rdan_grpo.roll_response_checkpoint import ArtifactIdentity, CheckpointIdentity, load_checkpoint
from rdan_grpo.roll_response_config import ResponseConfig, load_response_rlvr_config
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY, build_runtime_identity
from rdan_grpo.wandb_tracking import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    deterministic_run_id,
    register_wandb_tracker,
)

METHOD_NAMES = {
    "rtt_papo_response": "rtt-papo-response",
    "rdan_scalar": "rdan-scalar",
    "rl_csr": "rl-csr",
    "rl_aon": "rl-aon",
    "rl_mix": "rl-mix",
}
CURRENT_LAUNCH_METHOD = "rtt_papo_response"


class _LaunchPaths(NamedTuple):
    config: Path
    runtime: Path
    certificate: Path
    preflight: Path
    evaluator_rows: Path
    program: Path
    rtt_root: Path
    snapshot: Path
    resume: Path | None
    recovery_evidence: Path | None
    pilot_evidence: Path | None


class _LaunchEvidence(NamedTuple):
    code_revision: str
    production_payload: dict[str, Any]
    production_hash: str
    certificate: Mapping[str, Any]
    data_identity: Mapping[str, Any]
    parity: Mapping[str, Any]


def main() -> int:
    """Validate immutable inputs, build the custom pipeline, and run the requested gate."""

    args = _parse_args()
    _require_current_launch_method(args.method)
    paths = _launch_paths(args)
    evidence = _launch_evidence(args, paths)
    run_dir = _prepare_directory(args.run_dir, "run directory")
    checkpoint_root = _prepare_directory(args.checkpoint_root, "checkpoint root")
    pipeline_config, response_config, model_identity = _load_pipeline_config(args, paths, evidence)
    runtime_identity = _response_runtime_identity(
        response_config,
        evidence.data_identity,
        evidence.production_hash,
    )
    identities = _prepare_lifecycle(
        args, paths, evidence, pipeline_config, response_config, model_identity, runtime_identity, run_dir
    )
    identity = identities[args.stage]
    _bind_checkpoint_identity(run_dir, identity, paths.resume is not None)
    completed = _run_pipeline(
        args,
        paths,
        evidence,
        pipeline_config,
        response_config,
        model_identity,
        runtime_identity,
        identity,
        checkpoint_root,
        run_dir,
    )
    _print_outcome(args, completed, run_dir)
    return 0


def _launch_paths(args: argparse.Namespace) -> _LaunchPaths:
    return _LaunchPaths(
        config=_regular_file(args.config, "config"),
        runtime=_regular_file(args.runtime_parity, "runtime parity artifact"),
        certificate=_regular_file(args.certificate, "preflight certificate"),
        preflight=_regular_file(args.preflight_config, "preflight config"),
        evaluator_rows=_regular_file(args.evaluator_rows, "evaluator rows"),
        program=_regular_file(args.program, "experiment program"),
        rtt_root=_real_directory(args.rtt_root, "RTT root"),
        snapshot=_snapshot(args.snapshot),
        resume=_checkpoint_path(args.resume_checkpoint, "resume checkpoint"),
        recovery_evidence=_optional_regular_file(args.recovery_evidence, "recovery lifecycle certificate"),
        pilot_evidence=_optional_regular_file(args.pilot_evidence, "pilot lifecycle certificate"),
    )


def _launch_evidence(args: argparse.Namespace, paths: _LaunchPaths) -> _LaunchEvidence:
    _bind_snapshot_environment(paths.snapshot)
    code_revision = _git_revision(REPO_ROOT)
    _require_response_readiness(args, paths.program, code_revision)
    require_launch_gate(paths.program, paths.config, paths.certificate, paths.rtt_root)
    data_identity = response_data_identity(paths.program)
    production_payload = _compose_config(paths.config)
    preflight_payload = _compose_config(paths.preflight)
    production_hash = _file_sha256(paths.config)
    preflight_hash = _file_sha256(paths.preflight)
    production_resolved_hash = canonical_resolved_config_sha256(production_payload)
    preflight_resolved_hash = canonical_resolved_config_sha256(preflight_payload)
    parity = _runtime_parity(
        paths.runtime,
        production_hash,
        production_resolved_hash,
        preflight_hash,
        preflight_resolved_hash,
    )
    source_hashes = _response_source_identity(
        paths,
        production_resolved_hash,
        preflight_resolved_hash,
    )
    certificate = require_train_certificate(
        paths.certificate,
        method=args.method,
        config_sha256=preflight_hash,
        source_sha256=source_hashes,
        quality_weight=args.quality_weight,
        mix_weight=args.mix_weight,
    )
    _require_secrets()
    return _LaunchEvidence(code_revision, production_payload, production_hash, certificate, data_identity, parity)


def _response_source_identity(
    paths: _LaunchPaths,
    production_resolved_hash: str,
    preflight_resolved_hash: str,
) -> dict[str, str]:
    hashes = response_source_hashes(
        REPO_ROOT / "src/rdan_grpo",
        evaluator_rows=paths.evaluator_rows,
        train_config=paths.config,
        preflight_config=paths.preflight,
        program=paths.program,
        train_resolved_config_sha256=production_resolved_hash,
        preflight_resolved_config_sha256=preflight_resolved_hash,
    )
    if _file_sha256(paths.runtime) != hashes["runtime_parity"]:
        raise ValueError("runtime parity artifact differs from the frozen experiment program")
    return hashes


def _require_secrets() -> None:
    for name in ("OPENROUTER_API_KEY", "WANDB_API_KEY"):
        if not os.environ.get(name, "").strip():
            raise ValueError(f"{name} must be set in the environment")


def _load_pipeline_config(
    args: argparse.Namespace,
    paths: _LaunchPaths,
    evidence: _LaunchEvidence,
) -> tuple[Any, ResponseConfig, dict[str, str]]:
    rtt_root = paths.rtt_root
    sys.path.insert(0, str(rtt_root))
    install_rtt_compat(rtt_root)
    register_wandb_tracker(rtt_root)

    from roll.datasets.chat_template import register_chat_template
    from roll.models.model_providers import default_tokenizer_provider
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(
        tokenizer: Any,
        conversation: Any,
        tools: Any = None,
        documents: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Render Qwen3 messages with thinking disabled for training."""

        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    pipeline_config = load_response_rlvr_config(rtt_root, RLVRConfig, evidence.production_payload)
    response_config = pipeline_config.rdan_response
    _validate_pipeline_config(args, pipeline_config, response_config, paths.snapshot)
    model_identity = _model_identity(evidence.parity)
    observed = build_runtime_identity(
        paths.snapshot,
        default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args),
        model=MODEL_NAME,
        revision=MODEL_REVISION,
    )
    if observed.__dict__ != model_identity:
        raise ValueError("current snapshot, tokenizer, or chat template differs from runtime parity")
    return pipeline_config, response_config, model_identity


def _validate_pipeline_config(
    args: argparse.Namespace,
    pipeline_config: Any,
    response_config: ResponseConfig,
    snapshot: Path,
) -> None:
    _validate_method(args, response_config)
    if args.planned_horizon != pipeline_config.max_steps:
        raise ValueError("planned horizon must match the frozen production config")
    if args.stop_after_step is not None and args.stop_after_step > args.planned_horizon:
        raise ValueError("stop_after_step cannot exceed the planned horizon")
    _validate_stage(args)
    _validate_snapshot_config(pipeline_config, snapshot)


def _prepare_lifecycle(
    args: argparse.Namespace,
    paths: _LaunchPaths,
    evidence: _LaunchEvidence,
    pipeline_config: Any,
    response_config: ResponseConfig,
    model_identity: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, CheckpointIdentity]:
    tracking = _stage_tracking(args, paths, evidence, pipeline_config, response_config, model_identity, run_dir)
    pipeline_config.track_with = "rdan_wandb"
    pipeline_config.tracker_kwargs = tracking[args.stage][0]
    identities = {
        stage: _checkpoint_identity(
            args=args,
            response=response_config,
            certificate=evidence.certificate,
            certificate_path=paths.certificate,
            data_identity=evidence.data_identity,
            code_revision=evidence.code_revision,
            model_identity=model_identity,
            wandb_identity=value[1],
        )
        for stage, value in tracking.items()
    }
    _require_pilot_sequence(
        args,
        identities=identities,
        runtime_identity=runtime_identity,
        model_identity=model_identity,
        resume=paths.resume,
        recovery_evidence=paths.recovery_evidence,
        pilot_evidence=paths.pilot_evidence,
    )
    return identities


def _stage_tracking(
    args: argparse.Namespace,
    paths: _LaunchPaths,
    evidence: _LaunchEvidence,
    pipeline_config: Any,
    response_config: ResponseConfig,
    model_identity: Mapping[str, str],
    run_dir: Path,
) -> dict[str, tuple[dict[str, Any], dict[str, str]]]:
    return {
        stage: _tracking(
            pipeline_config,
            response_config,
            evidence.certificate,
            model_identity,
            str(evidence.data_identity["manifest_sha256"]),
            evidence.code_revision,
            run_dir,
            paths.resume is not None and stage == args.stage,
            stage,
        )
        for stage in ("recovery", "pilot", "train")
    }


def _run_pipeline(
    args: argparse.Namespace,
    paths: _LaunchPaths,
    evidence: _LaunchEvidence,
    pipeline_config: Any,
    response_config: ResponseConfig,
    model_identity: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
    identity: CheckpointIdentity,
    checkpoint_root: Path,
    run_dir: Path,
) -> CompletedResponseRun:
    import ray
    from roll.distributed.scheduler.initialize import init

    from rdan_grpo.roll_response_pipeline import build_response_training_pipeline

    init()
    try:
        pipeline = build_response_training_pipeline(
            pipeline_config,
            response_config=response_config,
            certificate=evidence.certificate,
            runtime_identity=runtime_identity,
            model_identity=model_identity,
            checkpoint_identity=identity,
            checkpoint_root=checkpoint_root,
            run_root=run_dir,
            artifact_root=run_dir / "artifacts",
            stop_after_step=args.stop_after_step,
            resume_checkpoint=paths.resume,
            lifecycle_predecessor=paths.recovery_evidence,
        )
        return pipeline.run()
    finally:
        ray.shutdown()


def _print_outcome(args: argparse.Namespace, completed: CompletedResponseRun, run_dir: Path) -> None:
    lifecycle = _issue_lifecycle_outcome(args, completed=completed, run_dir=run_dir)
    print(
        json.dumps(
            {
                "checkpoints": [str(path) for path in completed.checkpoints],
                "lifecycle_certificate": str(lifecycle) if lifecycle is not None else None,
            },
            sort_keys=True,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rtt-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--preflight-config", type=Path, required=True)
    parser.add_argument("--evaluator-rows", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--runtime-parity", type=Path, required=True)
    parser.add_argument("--readiness-receipt", type=Path, required=True)
    parser.add_argument("--readiness-bootstrap", type=Path, required=True)
    parser.add_argument(
        "--readiness-evidence",
        type=Path,
        action="append",
        required=True,
        help="repeat in judge calibration, runtime parity, no-update order",
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_NAMES), required=True)
    parser.add_argument("--quality-weight", type=float)
    parser.add_argument("--mix-weight", type=float)
    parser.add_argument("--planned-horizon", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--recovery-evidence", type=Path)
    parser.add_argument("--pilot-evidence", type=Path)
    parser.add_argument("--stage", choices=("recovery", "pilot", "train"), required=True)
    return parser.parse_args()


def _validate_method(args: argparse.Namespace, config: ResponseConfig) -> None:
    observed = (args.method, args.quality_weight, args.mix_weight)
    expected = (config.method, config.quality_weight, config.mix_weight)
    if observed != expected:
        raise ValueError("CLI method and weights must exactly match the production sidecar")


def _require_response_readiness(args: argparse.Namespace, program_path: Path, code_revision: str) -> Mapping[str, Any]:
    receipt = _regular_file(args.readiness_receipt, "response readiness receipt")
    bootstrap = _regular_file(args.readiness_bootstrap, "response readiness bootstrap")
    evidence = tuple(_regular_file(path, "response readiness evidence") for path in args.readiness_evidence)
    return validate_response_readiness(
        receipt,
        program_path,
        bootstrap,
        evidence,
        rdan_revision=code_revision,
    )


def _require_current_launch_method(method: str) -> None:
    if method != CURRENT_LAUNCH_METHOD:
        raise ValueError(
            f"the frozen launch gate authorizes only {CURRENT_LAUNCH_METHOD}; "
            f"{method} requires its later method-scoped lifecycle freeze"
        )


def _validate_stage(args: argparse.Namespace) -> None:
    if args.stage == "recovery" and args.stop_after_step not in {1, 2}:
        raise ValueError("recovery stage requires stop_after_step 1 or 2")
    if args.stage == "pilot" and args.stop_after_step != 20:
        raise ValueError("pilot stage requires stop_after_step 20")
    if args.stage == "train" and args.stop_after_step is not None:
        raise ValueError("train stage must run the complete frozen horizon")


def _checkpoint_identity(
    *,
    args: argparse.Namespace,
    response: ResponseConfig,
    certificate: Mapping[str, Any],
    certificate_path: Path,
    data_identity: Mapping[str, Any],
    code_revision: str,
    model_identity: Mapping[str, str],
    wandb_identity: Mapping[str, str],
) -> CheckpointIdentity:
    return CheckpointIdentity(
        planned_horizon=args.planned_horizon,
        method=args.method,
        method_weight=(
            args.quality_weight if args.method in {"rtt_papo_response", "rdan_scalar"} else args.mix_weight
        ),
        resolved_config_sha256=response.resolved_config_sha256,
        certificate=ArtifactIdentity(id=certificate["certificate_id"], sha256=_file_sha256(certificate_path)),
        data=ArtifactIdentity(
            id=str(data_identity["artifact_id"]),
            sha256=str(data_identity["manifest_sha256"]),
        ),
        revisions={"code": code_revision, "rtt": RTT_REVISION, "model": MODEL_REVISION},
        base_checkpoint_sha256=model_identity["snapshot_sha256"],
        wandb=wandb_identity,
    )


def _response_runtime_identity(
    response: ResponseConfig,
    data_identity: Mapping[str, Any],
    production_config_sha256: str,
) -> dict[str, Any]:
    return {
        "resolved_config_sha256": response.resolved_config_sha256,
        "production_train_config_sha256": production_config_sha256,
        "response_data_manifest_sha256": data_identity["manifest_sha256"],
        "response_data_output_sha256": data_identity["output_sha256"],
        "rtt_revision": RTT_REVISION,
        **GENERATION_SOURCE_IDENTITY,
    }


def _require_pilot_sequence(
    args: argparse.Namespace,
    *,
    identities: Mapping[str, CheckpointIdentity],
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    resume: Path | None,
    recovery_evidence: Path | None,
    pilot_evidence: Path | None,
) -> None:
    _validate_stage(args)
    _validate_stage_identities(identities)
    if args.stage == "recovery":
        _require_recovery_sequence(
            args,
            identity=identities["recovery"],
            runtime_identity=runtime_identity,
            model_identity=model_identity,
            resume=resume,
            recovery_evidence=recovery_evidence,
            pilot_evidence=pilot_evidence,
        )
        return
    if args.stage == "pilot":
        _require_pilot_gate(
            identity=identities["recovery"],
            runtime_identity=runtime_identity,
            model_identity=model_identity,
            resume=resume,
            recovery_evidence=recovery_evidence,
            pilot_evidence=pilot_evidence,
        )
        return
    _require_train_gate(
        identity=identities["train"],
        pilot_identity=identities["pilot"],
        runtime_identity=runtime_identity,
        model_identity=model_identity,
        resume=resume,
        recovery_evidence=recovery_evidence,
        pilot_evidence=pilot_evidence,
    )


def _require_recovery_sequence(
    args: argparse.Namespace,
    *,
    identity: CheckpointIdentity,
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    resume: Path | None,
    recovery_evidence: Path | None,
    pilot_evidence: Path | None,
) -> None:
    if pilot_evidence is not None:
        raise ValueError("recovery stage does not accept pilot lifecycle evidence")
    if args.stop_after_step == 1 and resume is None and recovery_evidence is None:
        return
    if args.stop_after_step != 2 or resume is None or recovery_evidence is None:
        raise ValueError("recovery must run fresh to step 1, then resume with its step-1 lifecycle certificate")
    _require_checkpoint(resume, identity, 1, "recovery resume")
    evidence = validate_lifecycle_certificate(
        recovery_evidence,
        expected_stage="recovery_step_1",
        expected_identity=identity,
        expected_runtime_identity=runtime_identity,
        expected_model_identity=model_identity,
    )
    if evidence["checkpoint"]["path"] != str(resume):
        raise ValueError("recovery resume differs from the certified step-1 checkpoint")


def _require_pilot_gate(
    *,
    identity: CheckpointIdentity,
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    resume: Path | None,
    recovery_evidence: Path | None,
    pilot_evidence: Path | None,
) -> None:
    if resume is not None or pilot_evidence is not None:
        raise ValueError("pilot must start fresh from the pinned base")
    if recovery_evidence is None:
        raise ValueError("pilot requires the strict step-2 recovery lifecycle certificate")
    validate_lifecycle_certificate(
        recovery_evidence,
        expected_stage="recovery_step_2",
        expected_identity=identity,
        expected_runtime_identity=runtime_identity,
        expected_model_identity=model_identity,
    )


def _require_train_gate(
    *,
    identity: CheckpointIdentity,
    pilot_identity: CheckpointIdentity,
    runtime_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    resume: Path | None,
    recovery_evidence: Path | None,
    pilot_evidence: Path | None,
) -> None:
    if recovery_evidence is not None:
        raise ValueError("train stage accepts pilot evidence only")
    if pilot_evidence is None:
        raise ValueError("full training requires the strict step-20 pilot lifecycle certificate")
    validate_lifecycle_certificate(
        pilot_evidence,
        expected_stage="pilot_step_20",
        expected_identity=pilot_identity,
        expected_runtime_identity=runtime_identity,
        expected_model_identity=model_identity,
    )
    if resume is not None:
        manifest = _require_checkpoint(resume, identity, None, "train resume")
        if not 1 <= manifest["completed_step"] < identity.planned_horizon:
            raise ValueError("train resume must advance an incomplete full-run checkpoint")


def _issue_lifecycle_outcome(
    args: argparse.Namespace,
    *,
    completed: CompletedResponseRun,
    run_dir: Path,
) -> Path | None:
    if args.stage == "train":
        return None
    stage = f"recovery_step_{args.stop_after_step}" if args.stage == "recovery" else "pilot_step_20"
    return issue_lifecycle_certificate(
        run_dir / f"{stage}.json",
        stage=stage,
        outcome=completed,
    )


def _validate_stage_identities(identities: Mapping[str, CheckpointIdentity]) -> None:
    if set(identities) != {"recovery", "pilot", "train"}:
        raise ValueError("checkpoint lifecycle identities are incomplete")
    values = list(identities.values())
    common = [{key: value for key, value in asdict(identity).items() if key != "wandb"} for identity in values]
    if common[1:] != common[:-1]:
        raise ValueError("checkpoint lifecycle immutable identities differ")
    run_ids = {identity.wandb.get("run_id") for identity in values}
    if len(run_ids) != len(values):
        raise ValueError("recovery, pilot, and train checkpoint identities must be distinct")


def _require_checkpoint(
    path: Path,
    identity: CheckpointIdentity,
    completed_step: int | None,
    name: str,
) -> Mapping[str, Any]:
    manifest = load_checkpoint(path, identity=identity)
    if completed_step is not None and manifest["completed_step"] != completed_step:
        raise ValueError(f"{name} must be a promoted step-{completed_step} checkpoint")
    return manifest


def _runtime_parity(
    path: Path,
    production_hash: str,
    production_resolved_hash: str,
    preflight_hash: str,
    preflight_resolved_hash: str,
) -> Mapping[str, Any]:
    artifact = _json_object(path, "runtime parity artifact")
    backend = artifact.get("runtime_backend")
    if artifact.get("status") != "parity_passed" or not isinstance(backend, Mapping):
        raise ValueError("runtime parity artifact is not passing")
    expected_configs = {
        "production_train_config_sha256": production_hash,
        "production_resolved_config_sha256": production_resolved_hash,
        "preflight_train_config_sha256": preflight_hash,
        "preflight_resolved_config_sha256": preflight_resolved_hash,
    }
    if any(backend.get(key) != value for key, value in expected_configs.items()):
        raise ValueError("runtime parity artifact is linked to different composed launch configs")
    required = {
        "actor_train_strategy": "fsdp2_train",
        "actor_infer_strategy": "hf_infer",
        "transformer_impl": "huggingface",
        "rtt_revision": RTT_REVISION,
    }
    if any(backend.get(key) != value for key, value in required.items()):
        raise ValueError("runtime parity artifact backend differs from production")
    return artifact


def _compose_config(path: Path) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        config = compose(config_name=path.stem)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must resolve to an object")
    return payload


def _model_identity(parity: Mapping[str, Any]) -> dict[str, str]:
    model = parity.get("model")
    tokenizer = parity.get("tokenizer")
    template = parity.get("chat_template")
    if not all(isinstance(value, Mapping) for value in (model, tokenizer, template)):
        raise ValueError("runtime parity artifact model identity is incomplete")
    identity = {
        "model": model.get("model"),
        "revision": model.get("revision"),
        "snapshot_sha256": model.get("snapshot_sha256"),
        "tokenizer_files_sha256": tokenizer.get("files_sha256"),
        "chat_template_sha256": template.get("sha256"),
    }
    if identity["model"] != MODEL_NAME or identity["revision"] != MODEL_REVISION:
        raise ValueError("runtime parity artifact model differs from the pinned base")
    if any(not _sha256(value) for key, value in identity.items() if key not in {"model", "revision"}):
        raise ValueError("runtime parity artifact model hashes are invalid")
    return identity  # type: ignore[return-value]


def _tracking(
    config: Any,
    response: ResponseConfig,
    certificate: Mapping[str, Any],
    model: Mapping[str, str],
    data_sha256: str,
    code_revision: str,
    run_dir: Path,
    resume: bool,
    stage: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    method = METHOD_NAMES[response.method]
    wandb_stage = "resume" if stage == "recovery" else stage
    source_hashes = certificate.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("preflight certificate source identity is missing")
    metadata = {
        "kind": "train",
        "method": method,
        "seed": config.seed,
        "stage": wandb_stage,
        "resolved_config_sha256": response.resolved_config_sha256,
        "model_revision": MODEL_REVISION,
        "data_sha256": data_sha256,
        "code_revision": code_revision,
        "checkpoint_sha256": model["snapshot_sha256"],
    }
    run_id = deterministic_run_id(metadata)
    group = f"qwen-{method}"
    name = f"qwen-{method}-{wandb_stage}-s{config.seed}"
    kwargs = {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "group": group,
        "name": name,
        "job_type": "train",
        "id": run_id,
        "resume": "must" if resume else "allow",
        "tags": [method, "response-only", wandb_stage],
        "notes": f"Receipted response-only {wandb_stage}",
        "log_dir": str(run_dir),
        "settings": {"console": "off"},
        "metadata": metadata,
    }
    identity = {"entity": WANDB_ENTITY, "project": WANDB_PROJECT, "run_id": run_id, "name": name, "group": group}
    return kwargs, identity


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    return path.resolve()


def _real_directory(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a real directory")
    return path.resolve()


def _prepare_directory(path: Path, name: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return _real_directory(path, name)


def _checkpoint_path(path: Path | None, name: str) -> Path | None:
    return None if path is None else _real_directory(path, name)


def _optional_regular_file(path: Path | None, name: str) -> Path | None:
    return None if path is None else _regular_file(path, name)


def _snapshot(path: Path) -> Path:
    snapshot = _real_directory(path, "snapshot")
    if snapshot.name != MODEL_REVISION:
        raise ValueError("snapshot directory must be named with the pinned model revision")
    return snapshot


def _bind_snapshot_environment(snapshot: Path) -> None:
    configured = os.environ.get("RDAN_MODEL_SNAPSHOT")
    if configured and Path(configured).resolve() != snapshot:
        raise ValueError("RDAN_MODEL_SNAPSHOT differs from --snapshot")
    os.environ["RDAN_MODEL_SNAPSHOT"] = str(snapshot)


def _bind_checkpoint_identity(run_dir: Path, identity: CheckpointIdentity, resume: bool) -> Path:
    path = run_dir / "checkpoint-identity.json"
    body = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if resume:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != body:
            raise ValueError("resume run directory differs from the checkpoint identity")
        return path
    try:
        with path.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError("fresh run directory already contains checkpoint identity") from error
    return path


def _validate_snapshot_config(config: Any, snapshot: Path) -> None:
    for name in ("actor_train", "actor_infer"):
        configured = getattr(config, name).model_args.model_name_or_path
        if not isinstance(configured, str) or Path(configured).resolve() != snapshot:
            raise ValueError(f"{name} does not use the pinned local snapshot")


def _json_object(path: Path, name: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _git_revision(path: Path) -> str:
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("response training requires a clean Git worktree")
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
