#!/usr/bin/env python3
"""Seal one no-update FSDP2 to production vLLM parity artifact."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.fsdp_hf_receipt import RTT_REVISION, verify_fsdp_hf_checkout
from rdan_grpo.program import MODEL_NAME, MODEL_REVISION, check_program
from rdan_grpo.response_identity import (
    canonical_resolved_config_sha256,
    clean_repository_revision,
    response_data_identity,
)
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY, build_runtime_identity, write_artifact
from rdan_grpo.vllm_runtime_parity import (
    VLLMParityError,
    run_vllm_runtime_parity,
    validate_vllm_runtime_parity,
    write_vllm_runtime_parity,
)


def main() -> int:
    """Load the exact production topology, seal its receipt, and compare logprobs."""

    args = _parse_args()
    _validate_outputs(args)
    rtt_root = _directory(args.rtt_root or _required_path("RTT_ROOT"), "RTT root")
    verify_fsdp_hf_checkout(rtt_root)
    clean_repository_revision(ROOT)
    sys.path.insert(0, str(rtt_root))

    from rdan_grpo.roll_compat import install_rtt_compat

    install_rtt_compat(rtt_root)

    from roll.datasets.chat_template import register_chat_template
    from roll.distributed.scheduler.initialize import init
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    from rdan_grpo.roll_response_config import ResponseConfig
    from rdan_grpo.roll_vllm_parity_live import build_receipt_link, build_vllm_parity_pipeline

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(
        tokenizer: Any,
        conversation: Any,
        tools: Any = None,
        documents: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Render Qwen messages with thinking disabled."""

        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    program = check_program(_file(args.program, "program"))
    config = _file(args.config, "vLLM parity config")
    production = _file(args.production_config, "production config")
    _validate_program_configs(program, config, production)
    snapshot = _snapshot(args.snapshot)
    payload = _compose(config)
    production_payload = _compose(production)
    from dacite import from_dict

    config_payload = copy.deepcopy(payload)
    if config_payload.get("rewards", object()) is not None:
        raise VLLMParityError("vLLM parity config requires rewards=null")
    config_payload["rewards"] = {}
    for worker in ("actor_train", "actor_infer"):
        config_payload[worker]["device_mapping"] = _device_mapping_literal(
            config_payload[worker].get("device_mapping")
        )
    response_payload = config_payload.pop("rdan_response", None)
    if not isinstance(response_payload, dict):
        raise VLLMParityError("vLLM parity response sidecar is missing")
    pipeline_config = from_dict(data_class=RLVRConfig, data=config_payload)
    pipeline_config.rdan_response = ResponseConfig(
        method=response_payload.get("method"),
        quality_weight=response_payload.get("quality_weight"),
        mix_weight=response_payload.get("mix_weight"),
        resolved_config_sha256=canonical_resolved_config_sha256(payload),
    )
    pipeline_config.track_with = "stdout"
    pipeline_config.tracker_kwargs = {}
    parity_resolved = canonical_resolved_config_sha256(payload)
    production_resolved = canonical_resolved_config_sha256(production_payload)
    data = response_data_identity(args.program)
    init()
    import ray

    try:
        pipeline = build_vllm_parity_pipeline(pipeline_config)
        identity = build_runtime_identity(snapshot, pipeline.tokenizer, model=MODEL_NAME, revision=MODEL_REVISION)
        model_identity = {
            "model": identity.model,
            "revision": identity.revision,
            "snapshot_sha256": identity.snapshot_sha256,
            "tokenizer_files_sha256": identity.tokenizer_files_sha256,
            "chat_template_sha256": identity.chat_template_sha256,
        }
        runtime_identity = {
            "resolved_config_sha256": parity_resolved,
            "production_train_config_sha256": _sha256(production),
            "response_data_manifest_sha256": data["manifest_sha256"],
            "response_data_output_sha256": data["output_sha256"],
            "rtt_revision": RTT_REVISION,
            **GENERATION_SOURCE_IDENTITY,
        }
        pipeline.seal_weight_receipt(
            args.weight_receipt_output,
            runtime_identity=runtime_identity,
            model_identity=model_identity,
            resolved_config_sha256=parity_resolved,
        )
        receipt = build_receipt_link(args.weight_receipt_output, parity_resolved)
        artifact = run_vllm_runtime_parity(
            pipeline,
            identity,
            pipeline_config=pipeline_config,
            parity_config_sha256=_sha256(config),
            parity_resolved_config_sha256=parity_resolved,
            production_config_sha256=_sha256(production),
            production_resolved_config_sha256=production_resolved,
            rtt_revision=RTT_REVISION,
            weight_receipt=receipt,
            responses=args.responses,
        )
        reference = program.program["lifecycle_artifacts"]["vllm_runtime_parity"]
        validate_vllm_runtime_parity(
            artifact,
            artifact_id="qwen_vllm_runtime_parity_v1"
            if reference["artifact_id"] == "pending"
            else reference["artifact_id"],
            model=MODEL_NAME,
            revision=MODEL_REVISION,
            rtt_revision=RTT_REVISION,
            parity_config_sha256=_sha256(config),
            production_config_sha256=_sha256(production),
        )
        write_vllm_runtime_parity(args.output, artifact)
    except Exception as error:
        write_artifact(
            args.failure_output,
            {
                "schema_version": 1,
                "status": "parity_failed",
                "comparison_policy": "diagnostic_only_actor_recompute_authoritative",
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise
    finally:
        ray.shutdown()
    print(json.dumps(artifact, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--rtt-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--weight-receipt-output", type=Path, required=True)
    parser.add_argument("--responses", type=int, default=32)
    return parser.parse_args()


def _validate_outputs(args: argparse.Namespace) -> None:
    outputs = (args.output, args.failure_output, args.weight_receipt_output)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("vLLM parity output paths must differ")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)


def _device_mapping_literal(value: Any) -> str:
    """Return the string literal RTT evaluates when it parses a worker device mapping."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return repr(value)
    raise VLLMParityError("vLLM parity config requires a worker device mapping")


def _validate_program_configs(program: Any, parity: Path, production: Path) -> None:
    configs = program.program["same_backend_configs"]
    expected_parity = (program.repo_root / configs["vllm_diagnostic"]["path"]).resolve()
    expected_production = (program.repo_root / configs["production"]["path"]).resolve()
    if parity != expected_parity or _sha256(parity) != configs["vllm_diagnostic"]["sha256"]:
        raise VLLMParityError("vLLM parity config differs from the frozen program")
    if production != expected_production or _sha256(production) != configs["production"]["sha256"]:
        raise VLLMParityError("production config differs from the frozen program")


def _compose(path: Path) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        config = compose(config_name=path.stem)
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise VLLMParityError("vLLM parity config must resolve to an object")
    return value


def _file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise VLLMParityError(f"{name} must be a regular file")
    return path.resolve()


def _directory(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise VLLMParityError(f"{name} must be a real directory")
    return path.resolve()


def _snapshot(path: Path) -> Path:
    target = _directory(path, "snapshot")
    if target.name != MODEL_REVISION:
        raise VLLMParityError("snapshot directory must use the pinned revision")
    return target


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise VLLMParityError(f"{name} must be set")
    return Path(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"vLLM runtime parity failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
