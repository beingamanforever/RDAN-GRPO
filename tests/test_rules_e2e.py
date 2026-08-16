import json
from pathlib import Path

import pytest

from rdan_grpo.rules import (
    CERTIFIED_FUNCTIONS,
    RubricHubRuleError,
    evaluate_rubrichub_rule,
    normalize_letter_parameters,
    verify_rule_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/rubrichub_instruction_following.json"
RULE_CERTIFICATE = ROOT / "configs/artifacts/rubrichub_rule_certificate.json"


def test_rule_caller_matches_strict_boundary_and_fails_closed() -> None:
    assert normalize_letter_parameters({"letter": "A", "let_frequency": 2.0, "let_relation": "at least"}) == {
        "letter": "a",
        "let_frequency": 2,
        "let_relation": "at least",
    }
    assert evaluate_rubrichub_rule(
        "LetterFrequencyChecker",
        "A cab",
        {"letter": "a", "let_frequency": 2, "let_relation": "at least"},
    ).passed
    assert not evaluate_rubrichub_rule(
        "LetterFrequencyChecker",
        "A cab",
        {"letter": "a", "let_frequency": 2, "let_relation": "less than"},
    ).passed
    assert evaluate_rubrichub_rule("CommaChecker", "", {}).passed
    assert not evaluate_rubrichub_rule("CommaChecker", "<think>hidden, text</think>answer", {}).passed

    malformed = [
        ("LetterFrequencyChecker", "a", {"letter": "é", "let_frequency": 1, "let_relation": "at least"}),
        ("LetterFrequencyChecker", "a", {"letter": "a", "let_frequency": True, "let_relation": "at least"}),
        ("LetterFrequencyChecker", "a", {"letter": "a", "let_frequency": 21, "let_relation": "at least"}),
        ("CommaChecker", "plain", {"unused": 1}),
        ("UnknownChecker", "plain", {}),
        ("CommaChecker", None, {}),
    ]
    for function, response, parameters in malformed:
        result = evaluate_rubrichub_rule(function, response, parameters)
        assert result.valid is False
        assert result.passed is False
        assert result.error
        assert result == evaluate_rubrichub_rule(function, response, parameters)


def test_frozen_rule_certificate_matches_implementation_and_rejects_tampering(tmp_path: Path) -> None:
    source = json.loads(CONFIG.read_text(encoding="utf-8"))["source"]
    certified = verify_rule_certificate(RULE_CERTIFICATE, source)
    assert {route["function"] for route in certified["routes"]} == set(CERTIFIED_FUNCTIONS)

    checker = json.loads(RULE_CERTIFICATE.read_text(encoding="utf-8"))
    checker["routes"][0]["implementation_sha256"] = "0" * 64
    tampered_checker = tmp_path / "checker.json"
    tampered_checker.write_text(json.dumps(checker), encoding="utf-8")
    with pytest.raises(RubricHubRuleError, match="stale"):
        verify_rule_certificate(tampered_checker, source)

    dropped = json.loads(RULE_CERTIFICATE.read_text(encoding="utf-8"))
    dropped["routes"] = dropped["routes"][:1]
    tampered_routes = tmp_path / "routes.json"
    tampered_routes.write_text(json.dumps(dropped), encoding="utf-8")
    with pytest.raises(RubricHubRuleError, match="routes_invalid"):
        verify_rule_certificate(tampered_routes, source)
