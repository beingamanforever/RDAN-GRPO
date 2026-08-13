#!/usr/bin/env python3
"""Reproduce the HIR transformation used by Rubrics-To-Tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/data/hir.json"
SOURCE = ROOT / "data/HIR_trainv1.jsonl"
OUTPUT = ROOT / "data/HIR_trainv1_rubrics_processed.jsonl"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _transform(row: dict[str, Any]) -> dict[str, Any]:
    prompt = " ".join([row["prompt"], *row["criteria"]])
    rubrics = [
        {"id": index, "category": "", "description": criterion, "weight": 1}
        for index, criterion in enumerate(row["criteria"], 1)
    ]
    identity = {key: row[key] for key in row if key in {"source", "id"}}
    return identity | {
        "prompt": prompt,
        "question": row["prompt"],
        "rubrics": rubrics,
        "tag": "llm_judge",
        "difficulty": 0,
        "ground_truth": row["ground_truth"],
        "messages": [{"role": "user", "content": prompt}],
    }


def _write(source: Path, output: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with source.open(encoding="utf-8") as input_file, output.open("wb") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            encoded = (json.dumps(_transform(json.loads(line)), ensure_ascii=False) + "\n").encode()
            output_file.write(encoded)
            digest.update(encoded)
            count += 1
    return count, digest.hexdigest()


def prepare(source: Path, output: Path, records: int, sha256: str) -> None:
    """Transform HIR atomically and verify exact parity with RTT."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as file:
        temp = Path(file.name)
    try:
        actual_records, actual_hash = _write(source, temp)
        if (actual_records, actual_hash) != (records, sha256):
            raise ValueError(f"processed mismatch: records={actual_records}, sha256={actual_hash}")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    """Prepare HIR using the pinned RTT transformation contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    expected = _load_manifest(args.manifest)["rtt_processed"]
    prepare(args.source, args.output, expected["records"], expected["sha256"])
    print(f"verified {args.output}: {expected['records']} records, sha256={expected['sha256']}")


if __name__ == "__main__":
    main()
