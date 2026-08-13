#!/usr/bin/env python3
"""Audit the pinned HIR-16K hard and soft rubric taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/hir_taxonomy.json"
SOURCE = ROOT / "data/HIR_trainv1.jsonl"
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.hir import audit_hir  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Verify the pinned input identity, inferred counts, and taxonomy digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()

    config = _load_json(args.config)
    actual_source_hash = _sha256(args.source)
    expected_source_hash = config["source"]["sha256"]
    if actual_source_hash != expected_source_hash:
        raise ValueError(f"source mismatch: sha256={actual_source_hash}")

    with args.source.open(encoding="utf-8") as file:
        audit = audit_hir(file)
    actual = asdict(audit)
    actual_digest = actual.pop("digest")
    if actual != config["expected"]:
        raise ValueError(f"taxonomy count mismatch: {actual}")
    if actual_digest != config["taxonomy_digest"]["sha256"]:
        raise ValueError(f"taxonomy digest mismatch: sha256={actual_digest}")

    sources = ",".join(f"{source}:{counts['rows']}" for source, counts in audit.sources.items())
    print(
        f"verified {args.source}: rows={audit.rows}, criteria={audit.criteria}, hard={audit.hard}, soft={audit.soft}, "
        f"sources={sources}, type4_mixed={audit.type4_rows['mixed']}, "
        f"type4_soft_only={audit.type4_rows['soft_only']}, source_sha256={actual_source_hash}, "
        f"taxonomy_sha256={audit.digest}"
    )


if __name__ == "__main__":
    main()
