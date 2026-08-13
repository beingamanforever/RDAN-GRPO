#!/usr/bin/env python3
"""Generate and freeze reconstructed SFT and DPO control data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.control_data import (  # noqa: E402
    CONFIG,
    HIR_MANIFEST,
    ControlDataError,
    freeze_control_data,
    run_candidate_stage,
    run_teacher_stage,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=ROOT / "data/HIR_trainv1.jsonl")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--hir-manifest", type=Path, default=HIR_MANIFEST)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    teacher = commands.add_parser("teacher", help="append OpenRouter Luna teacher outputs")
    _common(teacher)
    teacher.add_argument("--output", type=Path, required=True, help="raw JSONL path outside Git")
    teacher.add_argument("--endpoint", default=None, help=argparse.SUPPRESS)
    teacher.add_argument("--row-id", type=int, action="append")

    candidates = commands.add_parser("candidates", help="append local Qwen rejected candidates")
    _common(candidates)
    candidates.add_argument("--output", type=Path, required=True, help="raw JSONL path outside Git")
    candidates.add_argument("--api-base", required=True)
    candidates.add_argument("--server-manifest", type=Path, required=True)
    candidates.add_argument("--row-id", type=int, action="append")

    freeze = commands.add_parser("freeze", help="freeze externally evaluated SFT and DPO JSONL")
    _common(freeze)
    freeze.add_argument("--teacher-raw", type=Path, required=True)
    freeze.add_argument("--candidate-raw", type=Path, required=True)
    freeze.add_argument("--evidence", type=Path, required=True)
    freeze.add_argument("--sft-output", type=Path, required=True)
    freeze.add_argument("--dpo-output", type=Path, required=True)
    freeze.add_argument("--sft-manifest", type=Path, required=True)
    freeze.add_argument("--dpo-manifest", type=Path, required=True)
    return parser


def main() -> None:
    """Execute one resumable generation stage or one immutable freeze."""

    parser = _parser()
    args = parser.parse_args()
    try:
        if args.stage == "teacher":
            result = run_teacher_stage(
                args.source,
                args.output,
                row_ids=args.row_id,
                config_path=args.config,
                hir_manifest_path=args.hir_manifest,
                endpoint=args.endpoint,
            )
            print(json.dumps(result.__dict__, sort_keys=True, separators=(",", ":")))
        elif args.stage == "candidates":
            result = run_candidate_stage(
                args.source,
                args.output,
                args.api_base,
                args.server_manifest,
                row_ids=args.row_id,
                config_path=args.config,
                hir_manifest_path=args.hir_manifest,
            )
            print(json.dumps(result.__dict__, sort_keys=True, separators=(",", ":")))
        else:
            sft, dpo = freeze_control_data(
                args.source,
                args.teacher_raw,
                args.candidate_raw,
                args.evidence,
                args.sft_output,
                args.dpo_output,
                args.sft_manifest,
                args.dpo_manifest,
                config_path=args.config,
                hir_manifest_path=args.hir_manifest,
            )
            print(json.dumps({"sft": sft["data"], "dpo": dpo["data"]}, sort_keys=True, separators=(",", ":")))
    except ControlDataError as error:
        parser.exit(1, f"control data blocked: {error}\n")


if __name__ == "__main__":
    main()
