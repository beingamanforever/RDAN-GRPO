#!/usr/bin/env python3
"""Merge generated soft rubrics into the training set.

Generated rubrics are appended after a row's existing ones and marked soft through an explicit
`hard_mask`. Order matters: checkers are looked up by position, so anything appended must come
last and must be marked soft, or a checker lookup runs off the end of its array.

Every added rubric carries `generated: true`, so results on expanded data stay separable from
results on the original data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.rewards import MAX_RUBRICS, hard_mask  # noqa: E402


def main() -> int:
    """Write the expanded dataset and report how much of it the process channel now reaches."""

    args = _parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated = {
        record["id"]: record["soft_rubrics"]
        for record in (
            json.loads(line) for line in args.generated.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }

    expanded = [_expand(row, generated.get(row["id"])) for row in rows]
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expanded),
        encoding="utf-8",
    )

    print(f"wrote {len(expanded)} rows to {args.out}")
    _report("before", rows)
    _report("after ", expanded)
    return 0


def _expand(row: dict[str, Any], rubrics: list[str] | None) -> dict[str, Any]:
    """Append the generated rubrics to one row, or return it unchanged when it has none."""

    existing = row.get("rubrics") or []
    if not rubrics:
        return row
    if len(existing) + len(rubrics) > MAX_RUBRICS:
        rubrics = rubrics[: MAX_RUBRICS - len(existing)]
    if not rubrics:
        return row

    added = [
        {
            "id": len(existing) + offset + 1,
            "category": "",
            "description": text,
            "weight": 1,
            "verifier": "",
            "function": "",
            "parameters": "{}",
            "generated": True,
        }
        for offset, text in enumerate(rubrics)
    ]
    truth = row.get("ground_truth") or {}
    if isinstance(truth, str):
        truth = json.loads(truth)
    return {
        **row,
        "rubrics": [*existing, *added],
        "ground_truth": {**truth, "hard_mask": [True] * len(existing) + [False] * len(added)},
    }


def _report(label: str, rows: list[dict[str, Any]]) -> None:
    """Print how many rows carry a judged rubric, which bounds the process channel's reach."""

    with_soft = sum(1 for row in rows if _soft_count(row) > 0)
    total_soft = sum(_soft_count(row) for row in rows)
    share = with_soft / len(rows) * 100
    print(f"{label}: {with_soft:6d}/{len(rows)} rows with soft rubrics ({share:5.1f}%), {total_soft} soft rubrics")


def _soft_count(row: dict[str, Any]) -> int:
    """Count judged rubrics exactly as the reward worker classifies them."""

    rubrics = row.get("rubrics") or []
    if not rubrics:
        return 0
    truth = row.get("ground_truth") or {}
    if isinstance(truth, str):
        truth = json.loads(truth)
    return sum(1 for hard in hard_mask(row["source"], truth, len(rubrics)) if not hard)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/hybrid.jsonl")
    parser.add_argument("--generated", type=Path, default=ROOT / "data/soft_rubrics_generated.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/hybrid_expanded.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"dataset expansion failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
