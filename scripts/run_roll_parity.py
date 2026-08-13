#!/usr/bin/env python3
"""Prove synchronous ROLL/vLLM token-logprob parity without an optimizer step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from rdan_grpo.program import MODEL_NAME, MODEL_REVISION
from rdan_grpo.runtime_parity import ParityError, build_runtime_identity, run_runtime_parity, write_artifact
from rdan_grpo.weight_receipt import (
    RECEIPT_WORKER_EXTENSION,
    WeightReceiptError,
    build_receipt_link,
    canonical_sha256,
    validate_parity_receipt_pair,
    verify_rtt_checkout,
)

RTT_ROOT_ENV = "RTT_ROOT"


def main() -> int:
    """Construct the pinned live topology and atomically seal parity evidence."""

    args = _parse_args()
    outputs = (args.output, args.failure_output, args.weight_receipt_output)
    if len({output.resolve() for output in outputs}) != len(outputs):
        raise ValueError("parity success, failure, and weight receipt outputs must be different paths")
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    rtt_root = args.rtt_root or _required_path(RTT_ROOT_ENV)
    rtt_boundary_sha256 = verify_rtt_checkout(rtt_root)
    sys.path.insert(0, str(rtt_root.resolve()))

    from rdan_grpo.roll_compat import RTT_REVISION, install_rtt_compat

    install_rtt_compat(rtt_root)

    import ray
    from dacite import from_dict
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from roll.datasets.chat_template import register_chat_template
    from roll.distributed.scheduler.initialize import init
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    from rdan_grpo.roll_live import ObservedActorWorker, ObservedLogprobInferWorker, RuntimeParityPipeline

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(tokenizer, conversation, tools=None, documents=None, **kwargs):
        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    config_path = args.config.resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        config = compose(config_name=config_path.stem)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ParityError("ROLL parity config must resolve to an object")
    resolved_config_sha256 = canonical_sha256(payload)
    pipeline_config = from_dict(data_class=RLVRConfig, data=payload)
    _validate_snapshot_config(pipeline_config, args.snapshot)
    if pipeline_config.global_template != "qwen3_nothinking":
        raise ParityError("runtime parity requires the non-thinking Qwen chat template")
    worker_extension = pipeline_config.actor_infer.strategy_args.strategy_config.get("worker_extension_cls")
    if worker_extension != RECEIPT_WORKER_EXTENSION:
        raise ParityError("runtime parity requires the dedicated weight receipt worker extension")
    pipeline_config.actor_train.worker_cls = ObservedActorWorker
    pipeline_config.actor_infer.worker_cls = ObservedLogprobInferWorker
    pipeline_config.track_with = "stdout"
    pipeline_config.tracker_kwargs = {}

    init()
    try:
        pipeline = RuntimeParityPipeline(pipeline_config)
        identity = build_runtime_identity(
            args.snapshot,
            pipeline.tokenizer,
            model=MODEL_NAME,
            revision=MODEL_REVISION,
        )
        pipeline.seal_weight_receipt(
            args.weight_receipt_output,
            model_identity=identity,
            resolved_config_sha256=resolved_config_sha256,
            rtt_revision=RTT_REVISION,
            rtt_boundary_sha256=rtt_boundary_sha256,
        )
        receipt_link = build_receipt_link(args.weight_receipt_output, resolved_config_sha256)
        artifact = run_runtime_parity(
            pipeline,
            identity,
            pipeline_config=pipeline_config,
            train_config_sha256=_file_sha256(config_path),
            resolved_config_sha256=resolved_config_sha256,
            rtt_revision=RTT_REVISION,
            weight_receipt=receipt_link,
            responses=args.responses,
            failure_output=args.failure_output,
        )
        validate_parity_receipt_pair(artifact, args.weight_receipt_output)
        write_artifact(args.output, artifact)
    finally:
        ray.shutdown()
    print(json.dumps(artifact, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="exact synchronous ROLL training YAML")
    parser.add_argument("--snapshot", type=Path, required=True, help="local pinned Hugging Face snapshot directory")
    parser.add_argument("--output", type=Path, required=True, help="new lifecycle parity artifact")
    parser.add_argument("--failure-output", type=Path, required=True, help="new immutable failure evidence artifact")
    parser.add_argument(
        "--weight-receipt-output",
        type=Path,
        required=True,
        help="new immutable pre-generation weight receipt artifact",
    )
    parser.add_argument("--rtt-root", type=Path, help=f"pinned RTT checkout, otherwise ${RTT_ROOT_ENV}")
    parser.add_argument("--responses", type=int, default=32)
    return parser.parse_args()


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ParityError(f"{name} must point to the pinned RTT checkout")
    return Path(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_snapshot_config(pipeline_config: object, snapshot: Path) -> None:
    expected = snapshot.resolve()
    for role in ("actor_train", "actor_infer"):
        worker = getattr(pipeline_config, role, None)
        model_args = getattr(worker, "model_args", None)
        configured = getattr(model_args, "model_name_or_path", None)
        if not isinstance(configured, str) or Path(configured).resolve() != expected:
            raise ParityError(f"{role} must load the exact pinned local snapshot")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ParityError, RuntimeError, ValueError, WeightReceiptError) as exc:
        print(f"runtime parity failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
