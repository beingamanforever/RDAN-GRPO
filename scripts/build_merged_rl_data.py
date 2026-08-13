#!/usr/bin/env python3
"""Build the pinned English RubricHub and HIR RL corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_data_module():
    path = ROOT / "src/rdan_grpo/rubrichub_data.py"
    spec = importlib.util.spec_from_file_location("rdan_grpo_rubrichub_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load data module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DEFAULT_CONFIG = ROOT / "configs/data/rubrichub_instruction_following.json"


def main() -> None:
    """Generate a language certificate or build the deterministic merge."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    language = subparsers.add_parser("language-certificate")
    language.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    language.add_argument("--model", type=Path, required=True)
    language.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--language-certificate", type=Path)
    build.add_argument("--checker-certificate", type=Path)
    build.add_argument("--tokenizer-certificate", type=Path)

    args = parser.parse_args()
    data = _load_data_module()
    if args.command == "language-certificate":
        payload = data.build_language_certificate(args.config, args.model, args.output, repo_root=ROOT)
        result = {"output": str(args.output.resolve()), "records": len(payload["results"])}
    else:
        payload = data.build_merged_rl_data(
            args.config,
            repo_root=ROOT,
            language_certificate=args.language_certificate,
            checker_certificate=args.checker_certificate,
            tokenizer_certificate=args.tokenizer_certificate,
        )
        result = {
            "manifest": payload["id"],
            "outputs": payload["outputs"],
            "rubrichub_eligible_rows": payload["rl_eligibility"]["rubrichub_eligible_rows"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
