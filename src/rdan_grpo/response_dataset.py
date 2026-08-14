"""Safe JSONL loading for heterogeneous response-training records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import datasets


def load_response_dataset(
    file_names: str | Path | Sequence[str | Path],
    *,
    dataset_dir: str | Path = ".",
) -> datasets.Dataset:
    """Load response JSONL after serializing heterogeneous objects for Arrow."""

    names = [file_names] if isinstance(file_names, (str, Path)) else file_names
    root = Path(dataset_dir)
    rows: list[dict[str, Any]] = []
    for name in names:
        path = Path(name)
        path = path if path.is_absolute() else root / path
        _load_rows(path, rows)
    if not rows:
        raise ValueError("response dataset is empty")
    return datasets.Dataset.from_list(rows)


def canonical_json(value: Any) -> str:
    """Serialize a JSON value deterministically without changing its content."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.suffix not in {".json", ".jsonl"} or path.is_symlink() or not path.is_file():
        raise ValueError(f"response dataset must be a regular JSONL file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid response dataset JSON at {path}:{line_number}") from error
            rows.append(_serialize_response_row(row, path, line_number))


def _serialize_response_row(row: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"response dataset row must be an object at {path}:{line_number}")
    required = ("id", "prompt", "rubrics", "source", "ground_truth")
    if any(field not in row for field in required):
        raise ValueError(f"response dataset row is missing required fields at {path}:{line_number}")
    if isinstance(row["id"], bool) or not isinstance(row["id"], (int, str)) or not str(row["id"]):
        raise ValueError(f"response dataset id is invalid at {path}:{line_number}")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip() or not isinstance(row["source"], str):
        raise ValueError(f"response dataset prompt or source is invalid at {path}:{line_number}")
    rubrics = row["rubrics"]
    if not isinstance(rubrics, list) or not rubrics or any(not isinstance(rubric, dict) for rubric in rubrics):
        raise ValueError(f"response dataset rubrics are invalid at {path}:{line_number}")
    truth = row["ground_truth"]
    if isinstance(truth, str):
        try:
            truth = json.loads(truth)
        except json.JSONDecodeError as error:
            raise ValueError(f"response dataset ground_truth is invalid at {path}:{line_number}") from error
    if not isinstance(truth, dict):
        raise ValueError(f"response dataset ground_truth must be an object at {path}:{line_number}")
    normalized = dict(row)
    normalized["id"] = str(row["id"])
    normalized["rubrics"] = canonical_json(rubrics)
    normalized["ground_truth"] = canonical_json(truth)
    return normalized
