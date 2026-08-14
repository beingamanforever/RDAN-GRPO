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
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdan_grpo.fsdp_hf_receipt import (
    RTT_REVISION,
    FSDPHFReceiptError,
    verify_fsdp_hf_checkout,
)
from rdan_grpo.program import MODEL_NAME, MODEL_REVISION
from rdan_grpo.response_identity import canonical_resolved_config_sha256, clean_repository_revision
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
ROLLOUT_STRATEGIES = ("hf_infer", "vllm")


class _ParityPaths(NamedTuple):
    rtt_root: Path
    config: Path
    production_config: Path
    preflight_config: Path
    snapshot: Path


class _ParityConfigs(NamedTuple):
    pipeline: Any
    resolved_sha256: str
    production_resolved_sha256: str
    preflight_resolved_sha256: str


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
    paths = _parity_paths(args, rtt_root)
    configs = _load_parity_configs(paths)
    artifact = _run_parity(
        args,
        paths,
        configs,
        rtt_boundary_sha256,
        generation_source_identity,
    )
    print(json.dumps(artifact, sort_keys=True))
    return 0


def _parity_paths(args: argparse.Namespace, rtt_root: Path) -> _ParityPaths:
    config = _regular_file(args.config, "config")
    production = _regular_file(args.production_config, "production config")
    if production == config:
        raise ParityError("diagnostic and production configs must be different files")
    return _ParityPaths(
        rtt_root=rtt_root,
        config=config,
        production_config=production,
        preflight_config=_regular_file(args.preflight_config, "preflight config"),
        snapshot=_snapshot(args.snapshot),
    )


def _load_parity_configs(paths: _ParityPaths) -> _ParityConfigs:
    sys.path.insert(0, str(paths.rtt_root.resolve()))
    from roll.datasets.chat_template import register_chat_template
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    from rdan_grpo.roll_compat import install_rtt_compat, load_sync_hf_rlvr_config

    install_rtt_compat(paths.rtt_root)

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(
        tokenizer: Any,
        conversation: Any,
        tools: Any = None,
        documents: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Render Qwen3 messages with thinking disabled for parity."""

        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    payload = _compose_config(paths.config, "same-backend parity")
    production = _compose_config(paths.production_config, "same-backend production")
    preflight = _compose_config(paths.preflight_config, "same-backend preflight")
    _validate_production_config(production, paths.snapshot)
    pipeline = load_sync_hf_rlvr_config(paths.rtt_root, RLVRConfig, payload)
    _validate_snapshot_config(pipeline, paths.snapshot)
    pipeline.track_with = "stdout"
    pipeline.tracker_kwargs = {}
    return _ParityConfigs(
        pipeline,
        canonical_resolved_config_sha256(payload),
        canonical_resolved_config_sha256(production),
        canonical_resolved_config_sha256(preflight),
    )


def _compose_config(path: Path, name: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        config = compose(config_name=path.stem)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ParityError(f"{name} config must resolve to an object")
    return payload


def _run_parity(
    args: argparse.Namespace,
    paths: _ParityPaths,
    configs: _ParityConfigs,
    rtt_boundary_sha256: str,
    generation_source_identity: Mapping[str, str],
) -> Mapping[str, Any]:
    clean_repository_revision(REPO_ROOT)
    import ray
    from roll.distributed.scheduler.initialize import init

    from rdan_grpo.roll_same_backend_live import build_same_backend_pipeline

    try:
        init()
        pipeline = build_same_backend_pipeline(configs.pipeline)
        identity = build_runtime_identity(
            paths.snapshot,
            pipeline.tokenizer,
            model=MODEL_NAME,
            revision=MODEL_REVISION,
        )
        receipt_link = _seal_and_link_receipt(
            pipeline, identity, args, paths, configs, rtt_boundary_sha256, generation_source_identity
        )
        artifact = run_runtime_parity(
            pipeline,
            identity,
            pipeline_config=configs.pipeline,
            train_config_sha256=_file_sha256(paths.config),
            resolved_config_sha256=configs.resolved_sha256,
            rtt_revision=RTT_REVISION,
            weight_receipt=receipt_link,
            production_train_config_sha256=_file_sha256(paths.production_config),
            production_resolved_config_sha256=configs.production_resolved_sha256,
            preflight_train_config_sha256=_file_sha256(paths.preflight_config),
            preflight_resolved_config_sha256=configs.preflight_resolved_sha256,
            responses=args.responses,
            failure_output=args.failure_output,
            backend_profile=FSDP2_HF_PROFILE,
        )
        if artifact["weight_receipt"] != receipt_link:
            raise ParityError("parity artifact changed the sealed receipt linkage")
        _seal_success(args, artifact)
    finally:
        ray.shutdown()
    return artifact


def _seal_and_link_receipt(
    pipeline: Any,
    identity: RuntimeIdentity,
    args: argparse.Namespace,
    paths: _ParityPaths,
    configs: _ParityConfigs,
    rtt_boundary_sha256: str,
    generation_source_identity: Mapping[str, str],
) -> Mapping[str, str]:
    from rdan_grpo.roll_same_backend_live import build_fsdp_hf_receipt_link, raw_fsdp_hf_receipt_link

    try:
        pipeline.seal_weight_receipt(
            args.weight_receipt_output,
            model_identity=identity,
            resolved_config_sha256=configs.resolved_sha256,
            rtt_revision=RTT_REVISION,
            rtt_boundary_sha256=rtt_boundary_sha256,
            generation_source_identity=generation_source_identity,
        )
    except (FSDPHFReceiptError, RuntimeError, ValueError):
        receipt = raw_fsdp_hf_receipt_link(args.weight_receipt_output, configs.resolved_sha256)
        _seal_failure_from_context(configs, identity, paths, receipt, args)
        raise
    try:
        return build_fsdp_hf_receipt_link(args.weight_receipt_output, configs.resolved_sha256)
    except FSDPHFReceiptError:
        receipt = raw_fsdp_hf_receipt_link(args.weight_receipt_output, configs.resolved_sha256)
        _seal_failure_from_context(configs, identity, paths, receipt, args, code="receipt_linkage_failed")
        raise


def _seal_failure_from_context(
    configs: _ParityConfigs,
    identity: RuntimeIdentity,
    paths: _ParityPaths,
    receipt: Mapping[str, str],
    args: argparse.Namespace,
    code: str = "receipt_failed",
) -> None:
    _seal_receipt_failure(
        configs.pipeline,
        identity,
        paths.config,
        configs.resolved_sha256,
        receipt,
        paths.production_config,
        configs.production_resolved_sha256,
        paths.preflight_config,
        configs.preflight_resolved_sha256,
        args,
        code=code,
    )


def _seal_receipt_failure(
    pipeline_config: Any,
    identity: RuntimeIdentity,
    config_path: Path,
    resolved_config_sha256: str,
    receipt_link: Mapping[str, str],
    production_config_path: Path,
    production_resolved_config_sha256: str,
    preflight_config_path: Path,
    preflight_resolved_config_sha256: str,
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
            production_resolved_config_sha256=production_resolved_config_sha256,
            preflight_train_config_sha256=_file_sha256(preflight_config_path),
            preflight_resolved_config_sha256=preflight_resolved_config_sha256,
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
    parser.add_argument("--preflight-config", type=Path, required=True)
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
    _validate_production_strategies(actor_train, actor_infer)
    for name, worker in (("actor_train", actor_train), ("actor_infer", actor_infer)):
        model_args = worker.get("model_args")
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


def _validate_production_strategies(actor_train: Mapping[str, Any], actor_infer: Mapping[str, Any]) -> None:
    """Require the FSDP2 trainer this gate measures and any supported rollout backend."""

    train_args = actor_train.get("strategy_args")
    if not isinstance(train_args, Mapping) or train_args.get("strategy_name") != "fsdp2_train":
        raise ParityError("same-backend production config requires actor_train strategy fsdp2_train")
    if train_args.get("strategy_config", {}).get("transformer_impl") != "huggingface":
        raise ParityError("same-backend production config requires actor_train Hugging Face implementation")
    infer_args = actor_infer.get("strategy_args")
    if not isinstance(infer_args, Mapping) or infer_args.get("strategy_name") not in ROLLOUT_STRATEGIES:
        raise ParityError(f"same-backend production config requires an actor_infer strategy in {ROLLOUT_STRATEGIES}")
    if infer_args["strategy_name"] != "hf_infer":
        return
    if infer_args.get("strategy_config", {}).get("transformer_impl") != "huggingface":
        raise ParityError("same-backend production config requires actor_infer Hugging Face implementation")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, FSDPHFReceiptError, ParityError, RuntimeError, ValueError) as error:
        print(f"same-backend parity failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
