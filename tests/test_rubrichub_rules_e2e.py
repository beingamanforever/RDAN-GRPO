import json
import subprocess
import sys
from pathlib import Path

import pytest

from rdan_grpo.rubrichub_rules import (
    RubricHubRuleError,
    evaluate_rubrichub_rule,
    normalize_letter_parameters,
    verify_rule_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/rubrichub_instruction_following.json"
RULE_CERTIFICATE = ROOT / "configs/artifacts/rubrichub_rule_certificate.json"
TOKEN_CERTIFICATE = ROOT / "configs/artifacts/rubrichub_tokenizer_certificate.json"
RULE_EVIDENCE = ROOT / "data/rubrichub-source/certificates/rubrichub_rule_evidence.jsonl"
TOKEN_EVIDENCE = ROOT / "data/rubrichub-source/certificates/rubrichub_tokenizer_evidence.jsonl"
LANGUAGE_CERTIFICATE = (
    ROOT / "data/rubrichub-source/language-id/rubrichub_instruction_following_english_certificate.json"
)
MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
    / "cdbee75f17c01a7cc42f958dc650907174af0554"
)


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


def test_frozen_certificates_pass_scripts_and_reject_tampering(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/certify_rubrichub_rules.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/certify_rubrichub_tokenizer.py", "--check"],
        cwd=ROOT,
        check=True,
    )

    source = json.loads(CONFIG.read_text(encoding="utf-8"))["source"]
    checker = json.loads(RULE_CERTIFICATE.read_text(encoding="utf-8"))
    checker["routes"][0]["implementation_sha256"] = "0" * 64
    tampered_checker = tmp_path / "checker.json"
    tampered_checker.write_text(json.dumps(checker), encoding="utf-8")
    with pytest.raises(RubricHubRuleError, match="stale"):
        verify_rule_certificate(tampered_checker, source)

    tampered_evidence = tmp_path / "tokenizer.jsonl"
    lines = TOKEN_EVIDENCE.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["accepted"] = not row["accepted"]
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    tampered_evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/certify_rubrichub_tokenizer.py",
            "--check",
            "--certificate",
            str(TOKEN_CERTIFICATE),
            "--evidence",
            str(tampered_evidence),
            "--checker-certificate",
            str(RULE_CERTIFICATE),
            "--language-certificate",
            str(LANGUAGE_CERTIFICATE),
            "--model-path",
            str(MODEL_PATH),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "tampered" in result.stderr
    assert RULE_EVIDENCE.is_file()
