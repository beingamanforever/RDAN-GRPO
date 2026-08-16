#!/usr/bin/env python3
"""Launch RDAN-GRPO training: vLLM rollout, rubric reward, PAPO advantage, FSDP2 update."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIG = ROOT / "configs/roll/qwen_rtt_papo_response_train.yaml"
DEFAULT_DATA = ROOT / "data/hybrid.jsonl"
RTT_ROOT_ENV = "RTT_ROOT"
REQUIRED_SECRETS = ("OPENROUTER_API_KEY", "WANDB_API_KEY")


def main() -> int:
    """Build the pipeline from one config and run it to the requested step."""

    args = _parse_args()
    rtt_root = _rtt_root(args.rtt_root)
    _require_secrets()

    config = _build_config(args.config, rtt_root)
    pipeline = _build_pipeline(config, args)
    completed = pipeline.run()
    print(f"completed_step={completed.completed_step} checkpoints={len(completed.checkpoints)}")
    return 0


def _build_pipeline(config: Any, args: argparse.Namespace) -> Any:
    from roll.distributed.scheduler.initialize import init

    from rdan_grpo.checkpoint import ArtifactIdentity, CheckpointIdentity
    from rdan_grpo.pipeline import build_response_training_pipeline

    response = config.rdan_response
    horizon = args.steps or config.max_steps
    identity = CheckpointIdentity(
        planned_horizon=horizon,
        method=response.method,
        method_weight=response.quality_weight if response.quality_weight is not None else response.mix_weight,
        resolved_config_sha256=response.resolved_config_sha256,
        certificate=None,
        data=ArtifactIdentity(id=DEFAULT_DATA.name, sha256=_file_sha256(DEFAULT_DATA)),
        revisions=_revisions(),
        base_checkpoint_sha256=response.resolved_config_sha256,
        wandb=_wandb_run(config, horizon),
    )
    init()
    return build_response_training_pipeline(
        config,
        response_config=response,
        certificate=None,
        checkpoint_identity=identity,
        checkpoint_root=args.checkpoint_root,
        stop_after_step=args.steps,
        resume_checkpoint=args.resume,
    )


def _build_config(path: Path, rtt_root: Path) -> Any:
    from rdan_grpo.compat import install_rtt_compat

    install_rtt_compat(rtt_root)

    from roll.datasets.chat_template import register_chat_template
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    from rdan_grpo.config import load_response_rlvr_config
    from rdan_grpo.tracking import register_wandb_tracker

    register_wandb_tracker(rtt_root)

    @register_chat_template("qwen3_nothinking")
    def qwen3_nothinking(
        tokenizer: Any,
        conversation: Any,
        tools: Any = None,
        documents: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Render Qwen3 messages with thinking disabled."""

        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

    return load_response_rlvr_config(rtt_root, RLVRConfig, _compose(path))


def _compose(path: Path) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    resolved = path.resolve()
    with initialize_config_dir(version_base=None, config_dir=str(resolved.parent)):
        payload = OmegaConf.to_container(compose(config_name=resolved.stem), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must resolve to an object")
    return payload


def _wandb_run(config: Any, horizon: int) -> dict[str, str]:
    from rdan_grpo.tracking import WANDB_ENTITY, WANDB_PROJECT

    return {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "group": config.rdan_response.method,
        "name": f"{config.exp_name}-step{horizon}",
        "job_type": "train",
    }


def _revisions() -> dict[str, str]:
    from rdan_grpo.compat import RTT_REVISION

    return {"rtt": RTT_REVISION}


def _rtt_root(value: Path | None) -> Path:
    root = value or (Path(os.environ[RTT_ROOT_ENV]) if os.environ.get(RTT_ROOT_ENV) else None)
    if root is None:
        raise ValueError(f"--rtt-root or {RTT_ROOT_ENV} is required")
    return root.resolve()


def _require_secrets() -> None:
    for name in REQUIRED_SECRETS:
        if not os.environ.get(name, "").strip():
            raise ValueError(f"{name} must be set in the environment")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rtt-root", type=Path)
    parser.add_argument("--steps", type=int, help="stop after this step instead of the config horizon")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "output/checkpoints")
    parser.add_argument("--resume", type=Path, help="checkpoint directory to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
