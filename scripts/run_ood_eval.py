#!/usr/bin/env python3
"""Evaluate one frozen Qwen checkpoint on a pinned OOD benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.ood import OODError, run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("math_500", "gpqa", "mmlu_pro"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--server-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--math-verify-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/ood_qwen.json")
    args = parser.parse_args()
    try:
        output = run_evaluation(
            benchmark=args.benchmark,
            data_path=args.data,
            api_base=args.api_base,
            server_manifest=args.server_manifest,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            config_path=args.config,
            math_verify_root=args.math_verify_root,
            argv=sys.argv,
        )
    except OODError as error:
        parser.exit(1, f"OOD evaluation failed: {error}\n")
    print(f"completed OOD evaluation: {output}")


if __name__ == "__main__":
    main()
