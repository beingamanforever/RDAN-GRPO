#!/usr/bin/env python3
"""Launch RDAN-GRPO training: vLLM rollout, rubric reward, decoupled advantage, FSDP2 update."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIG_DIR = ROOT / "configs/train"


def main() -> int:
    """Build the pipeline from one config and run it to the requested step."""

    args = _parse_args()
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise ValueError("OPENROUTER_API_KEY must be set to score soft rubrics")

    config = build_config(args.config, args.rtt_root, args.overrides)
    pipeline = _build_pipeline(config, args)
    completed = pipeline.run()
    print(f"completed_step={completed.completed_step} checkpoints={len(completed.checkpoints)}")
    return 0


def build_config(name: str, rtt_root: Path | None, overrides: list[str]) -> Any:
    """Compose the Hydra config and construct ROLL's RLVR config object."""

    from rdan_grpo.compat import install_rtt_runtime

    install_rtt_runtime(rtt_root)

    from roll.datasets.chat_template import register_chat_template
    from roll.pipeline.rlvr.rubric_config import RLVRConfig

    from rdan_grpo.config import load_config
    from rdan_grpo.tracking import register_tracker

    register_tracker()

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

    return load_config(RLVRConfig, _compose(name, overrides))


def _build_pipeline(config: Any, args: argparse.Namespace) -> Any:
    from roll.distributed.scheduler.initialize import init

    from rdan_grpo.pipeline import build_pipeline

    init()
    return build_pipeline(
        config,
        checkpoint_root=args.checkpoint_root or ROOT / "output" / config.exp_name / "checkpoints",
        stop_after_step=args.steps,
        resume=args.resume,
    )


def _compose(name: str, overrides: list[str]) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        payload = OmegaConf.to_container(compose(config_name=name, overrides=overrides), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must resolve to an object")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="rdan", help="config name under configs/train")
    parser.add_argument("--rtt-root", type=Path, help="RTT checkout supplying ROLL (or set RTT_ROOT)")
    parser.add_argument("--steps", type=int, help="stop after this step instead of the config horizon")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume from the latest checkpoint, or from the given checkpoint directory",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. num_gpus=4 max_steps=100")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
