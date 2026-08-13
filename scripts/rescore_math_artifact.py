#!/usr/bin/env python3
"""Rescore sealed MATH-500 responses with the pinned verifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.ood import OODError, rescore_math_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--math-verify-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/ood_qwen.json")
    args = parser.parse_args()
    try:
        output = rescore_math_artifact(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            data_path=args.data,
            math_verify_root=args.math_verify_root,
            config_path=args.config,
            argv=sys.argv,
        )
    except OODError as error:
        parser.exit(1, f"MATH-500 rescore failed: {error}\n")
    print(f"rescored MATH-500 artifact: {output}")


if __name__ == "__main__":
    main()
