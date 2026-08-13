#!/usr/bin/env python3
"""Validate the Qwen-first experiment and provisioning contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.program import ProgramContractError, check_program, require_launch_gate  # noqa: E402


def main() -> None:
    """Validate a program config and report its derived execution counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", nargs="?", type=Path, default=ROOT / "configs/program/qwen_first.json")
    parser.add_argument("--launch-config", type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--rtt-root", type=Path)
    args = parser.parse_args()
    try:
        launch_values = (args.launch_config, args.certificate, args.rtt_root)
        if any(launch_values) and not all(launch_values):
            parser.error("--launch-config, --certificate, and --rtt-root must be provided together")
        bundle = (
            require_launch_gate(args.program, args.launch_config, args.certificate, args.rtt_root)
            if all(launch_values)
            else check_program(args.program)
        )
    except ProgramContractError as error:
        raise SystemExit(f"program contract invalid: {error}") from error
    counts = bundle.program["counts"]
    print(
        f"verified {bundle.program['id']}: "
        f"{counts['trainable_runs']} trainable runs, {counts['evaluation_suites']} evaluation suites"
    )


if __name__ == "__main__":
    main()
