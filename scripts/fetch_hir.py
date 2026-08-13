#!/usr/bin/env python3
"""Fetch the pinned HIR-16K source artifact and verify its identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/data/hir.json"
OUTPUT = ROOT / "data/HIR_trainv1.jsonl"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def fetch(url: str, output: Path, size: int, sha256: str) -> None:
    """Download one artifact atomically and reject unexpected bytes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as file:
        temp = Path(file.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        actual_size, actual_hash = _digest(temp)
        if (actual_size, actual_hash) != (size, sha256):
            raise ValueError(f"artifact mismatch: bytes={actual_size}, sha256={actual_hash}")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    """Fetch HIR-16K using the repository manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = _load_manifest(args.manifest)["source"]
    fetch(source["url"], args.output, source["bytes"], source["sha256"])
    print(f"verified {args.output}: {source['records']} records, sha256={source['sha256']}")


if __name__ == "__main__":
    main()
