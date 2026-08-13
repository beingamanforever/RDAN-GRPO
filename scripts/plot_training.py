#!/usr/bin/env python3
"""Regenerate deterministic training figures from metric JSONL runs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.metrics import plot_runs


def main() -> None:
    """Validate metric runs and write configured offline figures."""

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/logging/qwen_metrics.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot_runs(args.runs, args.output, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
