#!/usr/bin/env python3
"""Prove receipt-first FSDP2 and Hugging Face parity without training."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdan_grpo.fsdp_hf_receipt import (
    RTT_REVISION,
    FSDPHFReceiptError,
    canonical_sha256,
    verify_fsdp_hf_checkout,
)
from rdan_grpo.program import MODEL_NAME, MODEL_REVISION
from rdan_grpo.runtime_parity import (
    FSDP2_HF_PROFILE,
    ParityError,
    RuntimeIdentity,
    build_runtime_identity,
    run_runtime_parity,
    verify_transformers_generation_boundary,
    write_artifact,
)

RTT_ROOT_ENV = "RTT_ROOT"


class _ReceiptFailureBoundary:
    def __init__(self, code: str = "receipt_failed") -> None:
        self.code = code

    def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> Any:
        del responses, generation_config
        raise ParityError("weight receipt failed before generation", code=self.code)


def main() -> int:
    """Verify immutable inputs, seal a receipt, then seal one parity outcome."""

    args = _parse_args()
    _validate_paths(args)
    rtt_root = args.rtt_root or _required_path(RTT_ROOT_ENV)
    rtt_boundary_sha256 = verify_fsdp_hf_checkout(rtt_root)
    generation_source_identity = verify_transformers_generation_boundary()
    config_path = _regular_file(args.config, "config")
    production_config_path = _regular_file(args.production_config, "production config")
    if production_config_path == config_path:
        raise ParityError("diagnostic and production configs must be different files")
    snapshot = _snapshot(args.snapshot)
    sys.path.insert(0, str(rtt_root.resolve()))

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from roll.datasets.chat_template import register_chat_template
    from roll.distributed.scheduler.initialize import init
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    from rdan_grpo.roll_compat import install_rtt_compat, load_sync_hf_rlvr_config
    from rdan_grpo.roll_same_backend_live import (
        build_fsdp_hf_receipt_link,
        build_same_backend_pipeline,
        raw_fsdp_hf_receipt_link,
    )

    install_rtt_compat(rtt_root)

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(tokenizer, conversation, tools=None, documents=None, **kwargs):
        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        config = compose(config_name=config_path.stem)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ParityError("same-backend parity config must resolve to an object")
    resolved_config_sha256 = canonical_sha256(payload)
    pipeline_config = load_sync_hf_rlvr_config(rtt_root, RLVRConfig, payload)
    _validate_snapshot_config(pipeline_config, snapshot)
    with initialize_config_dir(version_base=None, config_dir=str(production_config_path.parent)):
        production_config = compose(config_name=production_config_path.stem)
    production_payload = OmegaConf.to_container(production_config, resolve=True)
    _validate_production_config(production_payload, snapshot)
    pipeline_config.track_with = "stdout"
    pipeline_config.tracker_kwargs = {}

    import ray

    try:
        init()
        pipeline = build_same_backend_pipeline(pipeline_config)
        identity = build_runtime_identity(
            snapshot,
            pipeline.tokenizer,
            model=MODEL_NAME,
            revision=MODEL_REVISION,
        )
        try:
            pipeline.seal_weight_receipt(
                args.weight_receipt_output,
                model_identity=identity,
                resolved_config_sha256=resolved_config_sha256,
                rtt_revision=RTT_REVISION,
                rtt_boundary_sha256=rtt_boundary_sha256,
                generation_source_identity=generation_source_identity,
            )
        except (FSDPHFReceiptError, RuntimeError, ValueError):
            receipt_link = raw_fsdp_hf_receipt_link(args.weight_receipt_output, resolved_config_sha256)
            _seal_receipt_failure(
                pipeline_config,
                identity,
                config_path,
                resolved_config_sha256,
                receipt_link,
                production_config_path,
                args,
            )
            raise
        try:
            receipt_link = build_fsdp_hf_receipt_link(args.weight_receipt_output, resolved_config_sha256)
        except FSDPHFReceiptError:
            receipt_link = raw_fsdp_hf_receipt_link(args.weight_receipt_output, resolved_config_sha256)
            _seal_receipt_failure(
                pipeline_config,
                identity,
                config_path,
                resolved_config_sha256,
                receipt_link,
                production_config_path,
                args,
                code="receipt_linkage_failed",
            )
            raise
        artifact = run_runtime_parity(
            pipeline,
            identity,
            pipeline_config=pipeline_config,
            train_config_sha256=_file_sha256(config_path),
            resolved_config_sha256=resolved_config_sha256,
            rtt_revision=RTT_REVISION,
            weight_receipt=receipt_link,
            production_train_config_sha256=_file_sha256(production_config_path),
            responses=args.responses,
            failure_output=args.failure_output,
            backend_profile=FSDP2_HF_PROFILE,
        )
        if artifact["weight_receipt"] != receipt_link:
            raise ParityError("parity artifact changed the sealed receipt linkage")
        _seal_success(args, artifact)
    finally:
        ray.shutdown()
    print(json.dumps(artifact, sort_keys=True))
    return 0


def _seal_receipt_failure(
    pipeline_config: Any,
    identity: RuntimeIdentity,
    config_path: Path,
    resolved_config_sha256: str,
    receipt_link: Mapping[str, str],
    production_config_path: Path,
    args: argparse.Namespace,
    code: str = "receipt_failed",
) -> None:
    if args.output.exists() or not args.weight_receipt_output.is_file():
        raise RuntimeError("receipt failure cannot preserve exclusive parity evidence")
    try:
        run_runtime_parity(
            _ReceiptFailureBoundary(code),
            identity,
            pipeline_config=pipeline_config,
            train_config_sha256=_file_sha256(config_path),
            resolved_config_sha256=resolved_config_sha256,
            rtt_revision=RTT_REVISION,
            weight_receipt=receipt_link,
            production_train_config_sha256=_file_sha256(production_config_path),
            responses=args.responses,
            failure_output=args.failure_output,
            backend_profile=FSDP2_HF_PROFILE,
        )
    except ParityError:
        return
    raise RuntimeError("receipt failure did not seal parity failure evidence")


def _seal_success(args: argparse.Namespace, artifact: Mapping[str, Any]) -> None:
    if args.failure_output.exists() or not args.weight_receipt_output.is_file():
        raise RuntimeError("parity success requires one receipt and no failure artifact")
    write_artifact(args.output, artifact)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--weight-receipt-output", type=Path, required=True)
    parser.add_argument("--rtt-root", type=Path, help=f"pinned RTT checkout, otherwise ${RTT_ROOT_ENV}")
    parser.add_argument("--responses", type=int, default=32)
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    outputs = (args.output, args.failure_output, args.weight_receipt_output)
    resolved = [path.resolve() for path in outputs]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("parity success, failure, and receipt outputs must not overlap")
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        _reject_symlink_ancestors(output)


def _reject_symlink_ancestors(path: Path) -> None:
    current = path.absolute().parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"output parent must not be a symlink: {current}")
        current = current.parent


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ParityError(f"{name} must be a regular file")
    return path.resolve()


def _snapshot(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ParityError("snapshot must be a real local directory")
    resolved = path.resolve()
    if resolved.name != MODEL_REVISION:
        raise ParityError("snapshot directory must be named with the pinned model revision")
    return resolved


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


def _validate_snapshot_config(pipeline_config: Any, snapshot: Path) -> None:
    for role in ("actor_train", "actor_infer"):
        worker = getattr(pipeline_config, role, None)
        configured = getattr(getattr(worker, "model_args", None), "model_name_or_path", None)
        if not isinstance(configured, str) or Path(configured).resolve() != snapshot:
            raise ParityError(f"{role} must load the exact pinned local snapshot")


def _validate_production_config(payload: Any, snapshot: Path) -> None:
    if not isinstance(payload, Mapping):
        raise ParityError("same-backend production config must resolve to an object")
    actor_train = payload.get("actor_train")
    actor_infer = payload.get("actor_infer")
    if not isinstance(actor_train, Mapping) or not isinstance(actor_infer, Mapping):
        raise ParityError("same-backend production config is missing actor topology")
    diagnostic_workers = {
        "rdan_grpo.roll_same_backend_live.ReceiptedFSDP2ActorWorker",
        "rdan_grpo.roll_same_backend_live.ReceiptedSynchronousHFInferWorker",
    }
    if actor_train.get("worker_cls") in diagnostic_workers or actor_infer.get("worker_cls") in diagnostic_workers:
        raise ParityError("same-backend production config cannot use diagnostic-only workers")
    for name, worker, strategy in (
        ("actor_train", actor_train, "fsdp2_train"),
        ("actor_infer", actor_infer, "hf_infer"),
    ):
        strategy_args = worker.get("strategy_args")
        model_args = worker.get("model_args")
        if not isinstance(strategy_args, Mapping) or strategy_args.get("strategy_name") != strategy:
            raise ParityError(f"same-backend production config requires {name} strategy {strategy}")
        if strategy_args.get("strategy_config", {}).get("transformer_impl") != "huggingface":
            raise ParityError(f"same-backend production config requires {name} Hugging Face implementation")
        if worker.get("device_mapping") != [0, 1]:
            raise ParityError(f"same-backend production config requires {name} device mapping [0, 1]")
        if worker.get("world_size") != 2 or worker.get("num_gpus_per_worker") != 1:
            raise ParityError(f"same-backend production config requires {name} DP2 topology")
        configured = model_args.get("model_name_or_path") if isinstance(model_args, Mapping) else None
        if not isinstance(configured, str) or Path(configured).resolve() != snapshot:
            raise ParityError(f"same-backend production config requires {name} pinned snapshot")
    if payload.get("async_pipeline") is not False or payload.get("async_generation_ratio") != 0:
        raise ParityError("same-backend production config requires synchronous generation")
    if payload.get("generate_opt_level") != 0:
        raise ParityError("same-backend production config requires generate_opt_level=0")
    max_steps = payload.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 1:
        raise ParityError("same-backend production config must enable optimizer training steps")
    if not isinstance(payload.get("rewards"), Mapping) or not payload["rewards"]:
        raise ParityError("same-backend production config requires nonempty rewards")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, FSDPHFReceiptError, ParityError, RuntimeError, ValueError) as error:
        print(f"same-backend parity failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
