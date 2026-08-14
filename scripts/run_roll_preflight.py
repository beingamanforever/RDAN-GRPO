#!/usr/bin/env python3
"""Create a compact scalar RDAN no-update certificate from evaluator outputs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.response_identity import (
    canonical_resolved_config_sha256,
    clean_repository_revision,
    lifecycle_source_hashes,
    response_source_hashes,
)
from rdan_grpo.roll_bridge import (
    assess_scalar_batch,
    build_preflight_certificate,
    sha256_file,
    write_certificate,
)

RTT_ROOT_ENV = "RTT_ROOT"
LIVE_QUALITY_WEIGHT = 0.5


def main() -> int:
    """Validate opaque evaluator rows and write one immutable certificate."""

    args = _parse_args()
    clean_repository_revision(ROOT)
    if args.live_rollout:
        return _live_rollout(args)
    for name in ("input", "config", "train_config", "program", "output"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name} is required for preflight")
    rows = _load_rows(args.input)
    keys = [row["prompt_key"] for row in rows]
    assessment = assess_scalar_batch(
        keys,
        _tensor(rows, "scores", torch.float32),
        _tensor(rows, "rubric_mask", torch.bool),
        _tensor(rows, "eval_mask", torch.bool),
        _tensor(rows, "hard_mask", torch.bool),
        method=args.method,
        unsupported_hard=_vector(rows, "unsupported_hard"),
        judge_failed=_vector(rows, "judge_failed"),
        group_size=args.group_size,
        quality_weight=args.quality_weight,
        mix_weight=args.mix_weight,
    )
    source_dir = Path(__file__).resolve().parents[1] / "src" / "rdan_grpo"
    train_resolved_hash = canonical_resolved_config_sha256(_compose_config(args.train_config))
    preflight_resolved_hash = canonical_resolved_config_sha256(_compose_config(args.config))
    certificate = build_preflight_certificate(
        [assessment],
        method=args.method,
        config_sha256=sha256_file(args.config),
        source_sha256=response_source_hashes(
            source_dir,
            evaluator_rows=args.input,
            train_config=args.train_config,
            preflight_config=args.config,
            program=args.program,
            train_resolved_config_sha256=train_resolved_hash,
            preflight_resolved_config_sha256=preflight_resolved_hash,
        ),
        optimizer_updates=args.optimizer_updates,
        quality_weight=args.quality_weight,
        mix_weight=args.mix_weight,
    )
    write_certificate(certificate, args.output)
    print(json.dumps(certificate.as_dict(), sort_keys=True))
    return 0 if certificate.ready else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL with opaque prompt keys and rubric outcomes")
    parser.add_argument("--config", type=Path, help="frozen scalar preflight configuration")
    parser.add_argument("--train-config", type=Path, help="exact gated ROLL training configuration")
    parser.add_argument("--output", type=Path, help="new certificate path")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument(
        "--method",
        choices=("rtt_papo_response", "rdan_scalar", "rl_csr", "rl_aon", "rl_mix"),
        default="rtt_papo_response",
    )
    parser.add_argument("--quality-weight", type=float)
    parser.add_argument("--mix-weight", type=float)
    parser.add_argument("--optimizer-updates", type=int, default=0)
    parser.add_argument("--live-rollout", action="store_true", help="run actor-infer plus reward no-update preflight")
    parser.add_argument("--program", type=Path, help="fully checked experiment program")
    parser.add_argument("--restricted-output", type=Path, help="restricted raw rollout artifact path")
    return parser.parse_args()


def _live_rollout(args: argparse.Namespace) -> int:
    _validate_live_method(args)
    rtt_root = _validate_live_inputs(args)
    pipeline_config = _load_live_pipeline(args, rtt_root)
    return _run_live_pipeline(args, pipeline_config)


def _validate_live_inputs(args: argparse.Namespace) -> str:
    required = (args.config, args.train_config, args.program, args.output, args.restricted_output)
    if any(value is None for value in required):
        raise ValueError("--config, --train-config, --program, --output, and --restricted-output are required")
    from rdan_grpo.program import check_program

    bundle = check_program(args.program)
    if sha256_file(args.train_config) != bundle.program["launch_train_config"]["sha256"]:
        raise ValueError("live rollout train config differs from the experiment program")
    if sha256_file(args.config) != bundle.program["launch_train_config"]["preflight_sha256"]:
        raise ValueError("live rollout config differs from the experiment program")
    lifecycle_source_hashes(args.program)
    parity = bundle.lifecycle_artifacts.get("runtime_parity")
    backend = parity.get("runtime_backend") if isinstance(parity, dict) else None
    expected = {
        "production_resolved_config_sha256": canonical_resolved_config_sha256(_compose_config(args.train_config)),
        "preflight_resolved_config_sha256": canonical_resolved_config_sha256(_compose_config(args.config)),
    }
    if not isinstance(backend, dict) or any(backend.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime parity artifact differs from the composed preflight launch configs")
    for name in ("OPENROUTER_API_KEY",):
        if not os.environ.get(name):
            raise ValueError(f"{name} must be set in the environment")
    rtt_root = os.environ.get(RTT_ROOT_ENV)
    if not rtt_root:
        raise ValueError(f"{RTT_ROOT_ENV} must be set to the pinned RTT checkout")
    return rtt_root


def _load_live_pipeline(args: argparse.Namespace, rtt_root: str) -> Any:
    from rdan_grpo.roll_compat import install_rtt_compat

    install_rtt_compat(rtt_root)

    from roll.datasets.chat_template import register_chat_template
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    from rdan_grpo.roll_response_config import load_response_preflight_config

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(
        tokenizer: Any,
        conversation: Any,
        tools: Any = None,
        documents: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Render Qwen3 messages with thinking disabled for preflight."""

        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    return load_response_preflight_config(rtt_root, RLVRConfig, _compose_config(args.config))


def _run_live_pipeline(args: argparse.Namespace, pipeline_config: Any) -> int:
    from roll.distributed.scheduler.initialize import init

    from rdan_grpo.roll_live import ScalarPreflightPipeline, run_live_preflight, seal_live_batch

    init()
    batch = run_live_preflight(ScalarPreflightPipeline, pipeline_config)
    assessment = seal_live_batch(
        batch,
        evaluator_path=args.output,
        restricted_path=args.restricted_output,
        method=args.method,
        quality_weight=args.quality_weight,
        mix_weight=args.mix_weight,
    )
    if not assessment.batch_valid:
        raise ValueError(f"live scalar batch failed closed: {', '.join(assessment.reasons)}")
    return 0


def _validate_live_method(args: argparse.Namespace) -> None:
    if args.method in {"rtt_papo_response", "rdan_scalar"}:
        if args.mix_weight is not None or args.quality_weight not in (None, LIVE_QUALITY_WEIGHT):
            raise ValueError(f"live rollout quality method requires weight {LIVE_QUALITY_WEIGHT}")
        args.quality_weight = LIVE_QUALITY_WEIGHT
        return
    if args.method in {"rl_csr", "rl_aon"}:
        if args.quality_weight is not None or args.mix_weight is not None:
            raise ValueError("live rollout AON and CSR methods do not accept weights")
        return
    if args.quality_weight is not None or args.mix_weight is None:
        raise ValueError("live rollout response mix requires only mix_weight")


def _compose_config(path: Path) -> dict[str, object]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    resolved = path.resolve()
    with initialize_config_dir(version_base=None, config_dir=str(resolved.parent)):
        config = compose(config_name=resolved.stem)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must resolve to an object")
    return payload


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError("input must contain evaluator rows")
    return rows


def _tensor(rows: list[dict[str, object]], name: str, dtype: torch.dtype) -> torch.Tensor:
    try:
        return torch.tensor([row[name] for row in rows], dtype=dtype)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} values") from exc


def _vector(rows: list[dict[str, object]], name: str) -> torch.Tensor:
    values = [row.get(name, False) for row in rows]
    if any(not isinstance(value, bool) for value in values):
        raise ValueError(f"{name} values must be boolean")
    return torch.tensor(values, dtype=torch.bool)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
