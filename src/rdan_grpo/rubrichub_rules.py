"""Fail-closed deterministic RubricHub rule routes."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

CERTIFIED_FUNCTIONS = frozenset({"CommaChecker", "LetterFrequencyChecker"})
RELATIONS = frozenset({"at least", "less than"})


class RubricHubRuleError(ValueError):
    """Raised when a RubricHub rule request violates the certified contract."""


@dataclass(frozen=True)
class RuleResult:
    """One deterministic rule outcome with fail-closed validity."""

    valid: bool
    passed: bool
    error: str | None


def evaluate_rubrichub_rule(function: Any, response: Any, parameters: Any) -> RuleResult:
    """Evaluate one certified route and return a fail-closed result."""

    try:
        if not isinstance(function, str) or function not in CERTIFIED_FUNCTIONS:
            raise RubricHubRuleError("uncertified_rule_route")
        if not isinstance(response, str):
            raise RubricHubRuleError("response_must_be_string")
        checker = _CHECKERS[function]
        return RuleResult(True, checker(response, parameters), None)
    except Exception as error:
        message = str(error) if isinstance(error, RubricHubRuleError) else "rule_evaluation_failed"
        return RuleResult(False, False, message)


def comma_checker(response: str, parameters: Any) -> bool:
    """Return whether the response contains no ASCII comma."""

    _empty_parameters(parameters)
    return "," not in response


def letter_frequency_checker(response: str, parameters: Any) -> bool:
    """Check one ASCII letter count against the certified relation."""

    normalized = normalize_letter_parameters(parameters)
    count = Counter(response.lower())[normalized["letter"]]
    if normalized["let_relation"] == "less than":
        return count < normalized["let_frequency"]
    return count >= normalized["let_frequency"]


def normalize_letter_parameters(parameters: Any) -> dict[str, str | int]:
    """Validate and normalize LetterFrequencyChecker parameters."""

    if not isinstance(parameters, Mapping) or set(parameters) != {"letter", "let_frequency", "let_relation"}:
        raise RubricHubRuleError("letter_parameters_schema_invalid")
    letter = parameters["letter"]
    if (
        not isinstance(letter, str)
        or len(letter) != 1
        or letter not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):
        raise RubricHubRuleError("letter_must_be_one_ascii_alphabetic_character")
    relation = parameters["let_relation"]
    if not isinstance(relation, str) or relation not in RELATIONS:
        raise RubricHubRuleError("let_relation_invalid")
    frequency = _frequency(parameters["let_frequency"])
    return {"letter": letter.lower(), "let_frequency": frequency, "let_relation": relation}


def implementation_sha256() -> str:
    """Return the exact implementation file hash."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verify_rule_certificate(path: str | Path, source: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a compact checker certificate against this implementation."""

    certificate_path = Path(path)
    try:
        payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RubricHubRuleError("checker_certificate_unreadable") from error
    expected_source = {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "file": source["file"],
        "sha256": source["sha256"],
        "records": source["records"],
    }
    if payload.get("schema_version") != 1 or payload.get("status") != "certified":
        raise RubricHubRuleError("checker_certificate_status_invalid")
    if payload.get("source") != expected_source:
        raise RubricHubRuleError("checker_certificate_source_invalid")
    routes = payload.get("routes")
    if (
        not isinstance(routes, list)
        or len(routes) != len(CERTIFIED_FUNCTIONS)
        or not all(isinstance(route, dict) for route in routes)
        or {route.get("function") for route in routes} != set(CERTIFIED_FUNCTIONS)
    ):
        raise RubricHubRuleError("checker_certificate_routes_invalid")
    digest = implementation_sha256()
    if any(route.get("implementation_sha256") != digest for route in routes):
        raise RubricHubRuleError("checker_certificate_implementation_stale")
    return payload


def _frequency(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RubricHubRuleError("let_frequency_must_be_numeric_integer")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise RubricHubRuleError("let_frequency_must_be_finite_integer")
    frequency = int(numeric)
    if not 1 <= frequency <= 20:
        raise RubricHubRuleError("let_frequency_out_of_range")
    return frequency


def _empty_parameters(parameters: Any) -> None:
    if not isinstance(parameters, Mapping) or parameters:
        raise RubricHubRuleError("comma_parameters_must_be_empty")


_CHECKERS: dict[str, Callable[[str, Any], bool]] = {
    "CommaChecker": comma_checker,
    "LetterFrequencyChecker": letter_frequency_checker,
}
