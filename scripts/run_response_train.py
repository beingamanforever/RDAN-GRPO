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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdan_grpo.program import MODEL_NAME, MODEL_REVISION, require_launch_gate
from rdan_grpo.response_identity import file_sha256 as _file_sha256
from rdan_grpo.response_identity import response_data_identity, response_source_hashes
from rdan_grpo.roll_bridge import require_train_certificate
from rdan_grpo.roll_compat import RTT_REVISION, install_rtt_compat
from rdan_grpo.roll_response_checkpoint import ArtifactIdentity, CheckpointIdentity
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


def main() -> int:
    """Validate immutable inputs, build the custom pipeline, and run the requested gate."""

    args = _parse_args()
    _require_current_launch_method(args.method)
    config_path = _regular_file(args.config, "config")
    runtime_path = _regular_file(args.runtime_parity, "runtime parity artifact")
    certificate_path = _regular_file(args.certificate, "preflight certificate")
    preflight_path = _regular_file(args.preflight_config, "preflight config")
    evaluator_rows = _regular_file(args.evaluator_rows, "evaluator rows")
    program_path = _regular_file(args.program, "experiment program")
    rtt_root = _real_directory(args.rtt_root, "RTT root")
    snapshot = _snapshot(args.snapshot)
    resume = _resume_path(args.resume_checkpoint)
    _bind_snapshot_environment(snapshot)
    require_launch_gate(program_path, config_path, certificate_path, rtt_root)
    code_revision = _git_revision(REPO_ROOT)
    data_identity = response_data_identity(program_path)
    production_hash = _file_sha256(config_path)
    parity = _runtime_parity(runtime_path, production_hash)
    source_hashes = response_source_hashes(
        REPO_ROOT / "src/rdan_grpo",
        evaluator_rows=evaluator_rows,
        train_config=config_path,
        preflight_config=preflight_path,
        program=program_path,
    )
    if _file_sha256(runtime_path) != source_hashes["runtime_parity"]:
        raise ValueError("runtime parity artifact differs from the frozen experiment program")
    certificate = require_train_certificate(
        certificate_path,
        method=args.method,
        config_sha256=_file_sha256(preflight_path),
        source_sha256=source_hashes,
        quality_weight=args.quality_weight,
        mix_weight=args.mix_weight,
    )
    for name in ("OPENROUTER_API_KEY", "WANDB_API_KEY"):
        if not os.environ.get(name, "").strip():
            raise ValueError(f"{name} must be set in the environment")
    run_dir = _prepare_directory(args.run_dir, "run directory")
    checkpoint_root = _prepare_directory(args.checkpoint_root, "checkpoint root")

    sys.path.insert(0, str(rtt_root))
    install_rtt_compat(rtt_root)
    register_wandb_tracker(rtt_root)

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from roll.datasets.chat_template import register_chat_template
    from roll.distributed.scheduler.initialize import init
    from roll.models.model_providers import default_tokenizer_provider
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(tokenizer, conversation, tools=None, documents=None, **kwargs):
        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        hydra_config = compose(config_name=config_path.stem)
    payload = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("response training config must resolve to an object")
    pipeline_config = load_response_rlvr_config(rtt_root, RLVRConfig, payload)
    response_config = pipeline_config.rdan_response
    _validate_method(args, response_config)
    if args.planned_horizon != pipeline_config.max_steps:
        raise ValueError("planned horizon must match the frozen production config")
    if args.stop_after_step is not None and args.stop_after_step > args.planned_horizon:
        raise ValueError("stop_after_step cannot exceed the planned horizon")
    _validate_stage(args)
    _validate_snapshot_config(pipeline_config, snapshot)

    model_identity = _model_identity(parity)
    observed_identity = build_runtime_identity(
        snapshot,
        default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args),
        model=MODEL_NAME,
        revision=MODEL_REVISION,
    )
    if observed_identity.__dict__ != model_identity:
        raise ValueError("current snapshot, tokenizer, or chat template differs from runtime parity")
    runtime_identity = {
        "resolved_config_sha256": response_config.resolved_config_sha256,
        "production_train_config_sha256": production_hash,
        "response_data_manifest_sha256": data_identity["manifest_sha256"],
        "response_data_output_sha256": data_identity["output_sha256"],
        "rtt_revision": RTT_REVISION,
        **GENERATION_SOURCE_IDENTITY,
    }
    tracker, wandb_identity = _tracking(
        pipeline_config,
        response_config,
        certificate,
        model_identity,
        str(data_identity["manifest_sha256"]),
        code_revision,
        run_dir,
        resume is not None,
        args.stage,
    )
    pipeline_config.track_with = "rdan_wandb"
    pipeline_config.tracker_kwargs = tracker
    identity = CheckpointIdentity(
        planned_horizon=args.planned_horizon,
        method=args.method,
        method_weight=(
            args.quality_weight if args.method in {"rtt_papo_response", "rdan_scalar"} else args.mix_weight
        ),
        resolved_config_sha256=response_config.resolved_config_sha256,
        certificate=ArtifactIdentity(id=certificate["certificate_id"], sha256=_file_sha256(certificate_path)),
        data=ArtifactIdentity(
            id=str(data_identity["artifact_id"]),
            sha256=str(data_identity["manifest_sha256"]),
        ),
        revisions={
            "code": code_revision,
            "rtt": RTT_REVISION,
            "model": MODEL_REVISION,
        },
        base_checkpoint_sha256=model_identity["snapshot_sha256"],
        wandb=wandb_identity,
    )
    _bind_checkpoint_identity(run_dir, identity, resume is not None)

    import ray

    from rdan_grpo.roll_response_pipeline import build_response_training_pipeline

    init()
    try:
        pipeline = build_response_training_pipeline(
            pipeline_config,
            response_config=response_config,
            certificate=certificate,
            runtime_identity=runtime_identity,
            model_identity=model_identity,
            checkpoint_identity=identity,
            checkpoint_root=checkpoint_root,
            run_root=run_dir,
            artifact_root=run_dir / "artifacts",
            stop_after_step=args.stop_after_step,
            resume_checkpoint=resume,
        )
        checkpoints = pipeline.run()
    finally:
        ray.shutdown()
    print(json.dumps({"checkpoints": [str(path) for path in checkpoints]}, sort_keys=True))
    return 0


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
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_NAMES), required=True)
    parser.add_argument("--quality-weight", type=float)
    parser.add_argument("--mix-weight", type=float)
    parser.add_argument("--planned-horizon", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--stage", choices=("pilot", "train"), required=True)
    return parser.parse_args()


def _validate_method(args: argparse.Namespace, config: ResponseConfig) -> None:
    observed = (args.method, args.quality_weight, args.mix_weight)
    expected = (config.method, config.quality_weight, config.mix_weight)
    if observed != expected:
        raise ValueError("CLI method and weights must exactly match the production sidecar")


def _require_current_launch_method(method: str) -> None:
    if method != CURRENT_LAUNCH_METHOD:
        raise ValueError(
            f"the frozen launch gate authorizes only {CURRENT_LAUNCH_METHOD}; "
            f"{method} requires its later method-scoped lifecycle freeze"
        )


def _validate_stage(args: argparse.Namespace) -> None:
    if args.stage == "pilot":
        if args.stop_after_step is None or not 1 <= args.stop_after_step <= 20:
            raise ValueError("pilot stage requires stop_after_step in [1, 20]")
    elif args.stop_after_step is not None:
        raise ValueError("train stage must run the complete frozen horizon")


def _runtime_parity(path: Path, production_hash: str) -> Mapping[str, Any]:
    artifact = _json_object(path, "runtime parity artifact")
    backend = artifact.get("runtime_backend")
    if artifact.get("status") != "parity_passed" or not isinstance(backend, Mapping):
        raise ValueError("runtime parity artifact is not passing")
    if backend.get("production_train_config_sha256") != production_hash:
        raise ValueError("runtime parity artifact is linked to a different production config")
    required = {
        "actor_train_strategy": "fsdp2_train",
        "actor_infer_strategy": "hf_infer",
        "transformer_impl": "huggingface",
        "rtt_revision": RTT_REVISION,
    }
    if any(backend.get(key) != value for key, value in required.items()):
        raise ValueError("runtime parity artifact backend differs from production")
    return artifact


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
    source_hashes = certificate.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("preflight certificate source identity is missing")
    metadata = {
        "kind": "train",
        "method": method,
        "seed": config.seed,
        "stage": stage,
        "resolved_config_sha256": response.resolved_config_sha256,
        "model_revision": MODEL_REVISION,
        "data_sha256": data_sha256,
        "code_revision": code_revision,
        "checkpoint_sha256": model["snapshot_sha256"],
    }
    run_id = deterministic_run_id(metadata)
    group = f"qwen-{method}"
    name = f"qwen-{method}-{stage}-s{config.seed}"
    kwargs = {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "group": group,
        "name": name,
        "job_type": "train",
        "id": run_id,
        "resume": "must" if resume else "allow",
        "tags": [method, "response-only", stage],
        "notes": f"Receipted response-only {stage}",
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


def _resume_path(path: Path | None) -> Path | None:
    return None if path is None else _real_directory(path, "resume checkpoint")


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
