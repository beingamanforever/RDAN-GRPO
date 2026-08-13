#!/usr/bin/env python3
"""Evaluate the pinned Qwen base model through a local vLLM server and RTT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.baseline import EvaluationError, run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("ifeval", "ifbench", "muldimif"), required=True)
    parser.add_argument("--rtt-root", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--server-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    try:
        output = run_evaluation(
            args.benchmark,
            args.rtt_root,
            args.api_base,
            args.server_manifest,
            args.output_dir,
            args.concurrency,
            argv=sys.argv,
        )
    except EvaluationError as error:
        parser.exit(1, f"base evaluation failed: {error}\n")
    print(f"completed base evaluation: {output}")


if __name__ == "__main__":
    main()
