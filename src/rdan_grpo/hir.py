"""Fail-closed hard and soft rubric inference for HIR-16K."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

SOURCES = ("type1", "type2", "type3", "type4")


@dataclass(frozen=True)
class HirAudit:
    """Deterministic HIR taxonomy counts and digest."""

    rows: int
    criteria: int
    hard: int
    soft: int
    sources: dict[str, dict[str, int]]
    type4_rows: dict[str, int]
    digest: str


def classify_hir_row(row: dict[str, Any]) -> tuple[bool, ...]:
    """Return the hard-rubric mask for one valid raw HIR row."""
    if not isinstance(row, dict):
        raise ValueError("HIR row must be an object")
    source = row.get("source")
    if source not in SOURCES:
        raise ValueError(f"unknown HIR source: {source!r}")
    criteria = row.get("criteria")
    if not isinstance(criteria, list) or not criteria or not all(isinstance(item, str) and item for item in criteria):
        raise ValueError("criteria must be a non-empty list of strings")
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError("ground_truth must be an object")
    if source in {"type1", "type2"}:
        instruction_ids = ground_truth.get("instruction_id_list")
        kwargs = ground_truth.get("kwargs")
        if not _aligned(instruction_ids, criteria) or not _aligned(kwargs, criteria):
            raise ValueError(f"{source} instruction_id_list and kwargs must align one-to-one with criteria")
        if not all(isinstance(item, str) and item for item in instruction_ids):
            raise ValueError(f"{source} instruction_id_list entries must be non-empty strings")
        if not all(isinstance(item, dict) for item in kwargs):
            raise ValueError(f"{source} kwargs entries must be objects")
        return (True,) * len(criteria)
    if source == "type3":
        constraints = ground_truth.get("constraints")
        if not _aligned(constraints, criteria):
            raise ValueError("type3 constraints must align one-to-one with criteria")
        if not all(
            isinstance(item, list) and len(item) == 3 and all(isinstance(value, str) and value for value in item)
            for item in constraints
        ):
            raise ValueError("type3 constraints must contain three non-empty string fields")
        return (True,) * len(criteria)

    checkers = ground_truth.get("checker")
    functions = ground_truth.get("functions")
    if not _aligned(checkers, criteria) or not _aligned(functions, criteria):
        raise ValueError("type4 checker and functions must align one-to-one with criteria")
    if not all(isinstance(function, str) and function for function in functions):
        raise ValueError("type4 function entries must be non-empty strings")

    hard_mask = []
    for checker in checkers:
        if not isinstance(checker, str):
            raise ValueError("type4 checker entries must be strings")
        if checker.startswith("[rule]"):
            hard_mask.append(True)
        elif checker.startswith("[llm]"):
            hard_mask.append(False)
        else:
            raise ValueError(f"unknown type4 checker prefix: {checker!r}")
    return tuple(hard_mask)


def _aligned(values: Any, criteria: list[str]) -> bool:
    return isinstance(values, list) and len(values) == len(criteria)


def audit_hir(lines: Iterable[str]) -> HirAudit:
    """Classify a raw HIR JSONL stream and return deterministic audit totals."""
    sources = {source: {"rows": 0, "criteria": 0, "hard": 0, "soft": 0} for source in SOURCES}
    type4_rows = {"hard_only": 0, "mixed": 0, "soft_only": 0}
    digest = hashlib.sha256()
    seen_ids: set[int | str] = set()
    rows = criteria = hard = soft = 0

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"line {line_number}: empty JSONL record")
        try:
            row = json.loads(line)
            hard_mask = classify_hir_row(row)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"line {line_number}: {error}") from error

        row_id = row.get("id")
        if isinstance(row_id, bool) or not isinstance(row_id, (int, str)):
            raise ValueError(f"line {line_number}: id must be an integer or string")
        if row_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate id {row_id!r}")
        seen_ids.add(row_id)

        source = row["source"]
        criterion_count = len(hard_mask)
        hard_count = sum(hard_mask)
        soft_count = criterion_count - hard_count
        totals = sources[source]
        totals["rows"] += 1
        totals["criteria"] += criterion_count
        totals["hard"] += hard_count
        totals["soft"] += soft_count
        rows += 1
        criteria += criterion_count
        hard += hard_count
        soft += soft_count

        if source == "type4":
            kind = "hard_only" if soft_count == 0 else "soft_only" if hard_count == 0 else "mixed"
            type4_rows[kind] += 1

        encoded = json.dumps([row_id, source, [int(value) for value in hard_mask]], separators=(",", ":"))
        digest.update(f"{encoded}\n".encode())

    return HirAudit(rows, criteria, hard, soft, sources, type4_rows, digest.hexdigest())
