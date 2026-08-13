import json
from pathlib import Path

import pytest

from rdan_grpo.evaluator_cert import EvaluatorCertificationError, verify_type4_certificate
from rdan_grpo.safe_rule import UnsafeRuleError, evaluate_rule, function_sha256, validate_rule
from rdan_grpo.scalar_data import (
    ScalarDataError,
    build_rtt_hir_dataset,
    build_scalar_dataset,
    verify_hir_tokenizer_gate,
    verify_scalar_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scalar_data_is_byte_reproducible_and_exactly_partitioned(tmp_path: Path) -> None:
    output = tmp_path / "scalar.jsonl"
    certificate = json.loads((ROOT / "configs/artifacts/hir_type4_rule_certificate.json").read_text())
    tokenizer = verify_hir_tokenizer_gate(
        ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json",
        ROOT / "data/HIR_trainv1_rubrics_processed.jsonl",
        ROOT,
    )
    manifest = build_scalar_dataset(
        ROOT / "data/HIR_trainv1.jsonl",
        ROOT / "data/HIR_trainv1_rubrics_processed.jsonl",
        ROOT / "configs/artifacts/hir_route_resolution.json",
        output,
        certified_families={("type4", "embedded_check_following")},
        excluded_type4_hashes=set(certificate["blocked_function_hashes"]),
        accepted_row_ids=tokenizer.accepted_row_ids,
    )
    frozen = json.loads((ROOT / "configs/artifacts/hir_scalar_certified_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == frozen["counts"]
    assert manifest["derived"]["sha256"] == frozen["derived"]["sha256"]
    assert output.read_bytes() == (ROOT / "data/HIR_trainv1_rdan_scalar_certified.jsonl").read_bytes()
    assert manifest["counts"] == {
        "source_rows": 16_968,
        "candidate_rows": 5_700,
        "included_rows": 5_699,
        "excluded_rows": 11_269,
        "soft_only_rows": 643,
        "uncertified_family_rows": 9_956,
        "unsafe_type4_rows": 19,
        "tokenizer_excluded_rows": 1,
        "unsupported_hard_identities": 799,
        "retained_hard_identities": 12_755,
        "excluded_hard_identities": 63_701,
        "all_hard_identities": 76_456,
    }
    assert len(manifest["row_ids"]["preflight_first_256"]) == 256


def test_certified_subset_and_linked_files_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = ROOT / "configs/artifacts/hir_scalar_certified_manifest.json"
    manifest = verify_scalar_dataset(frozen)
    assert manifest["counts"]["included_rows"] == 5_699
    assert manifest["counts"]["retained_hard_identities"] == 12_755

    certificate = json.loads((ROOT / "configs/artifacts/hir_type4_rule_certificate.json").read_text())
    tokenizer = verify_hir_tokenizer_gate(
        ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json",
        ROOT / "data/HIR_trainv1_rubrics_processed.jsonl",
        ROOT,
    )
    rerun = build_scalar_dataset(
        ROOT / "data/HIR_trainv1.jsonl",
        ROOT / "data/HIR_trainv1_rubrics_processed.jsonl",
        ROOT / "configs/artifacts/hir_route_resolution.json",
        tmp_path / "certified.jsonl",
        certified_families={("type4", "embedded_check_following")},
        excluded_type4_hashes=set(certificate["blocked_function_hashes"]),
        accepted_row_ids=tokenizer.accepted_row_ids,
    )
    assert rerun["derived"]["sha256"] == manifest["derived"]["sha256"]

    copy = tmp_path / "manifest.json"
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    payload["derived"]["sha256"] = "0" * 64
    copy.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(ROOT)
    with pytest.raises(ScalarDataError, match="missing or tampered"):
        verify_scalar_dataset(copy)


def test_full_rtt_hir_is_distinct_and_byte_reproducible(tmp_path: Path) -> None:
    certificate_path = ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json"
    processed = ROOT / "data/HIR_trainv1_rubrics_processed.jsonl"
    tokenizer = verify_hir_tokenizer_gate(certificate_path, processed, ROOT)
    output = tmp_path / "rtt.jsonl"
    manifest = build_rtt_hir_dataset(processed, output, tokenizer, certificate_path)
    frozen = json.loads((ROOT / "configs/artifacts/qwen_rtt_hir_data_manifest.json").read_text())
    assert manifest["counts"] == frozen["counts"]
    assert manifest["derived"]["sha256"] == frozen["derived"]["sha256"]
    assert manifest["row_ids"] == frozen["row_ids"]
    assert manifest["scope"] == "full_rtt_compatible_not_authoritative"
    assert manifest["counts"] == {
        "processed_rows": 16_968,
        "effective_rows": 16_962,
        "tokenizer_excluded_rows": 6,
    }
    assert output.read_bytes() == (ROOT / "data/HIR_trainv1_rtt_qwen.jsonl").read_bytes()


def test_restricted_rule_rejects_code_and_uncertified_hashes() -> None:
    code = "import re\n\ndef check_following(instruction, response):\n    return bool(re.search(r'x', response))\n"
    digest = function_sha256(code)
    assert evaluate_rule(code, "instruction", "x", allowed_hashes=[digest]).value is True
    assert evaluate_rule(code, "instruction", "x", allowed_hashes=[]).error == "uncertified_rule_hash"
    with pytest.raises(UnsafeRuleError, match="dunder"):
        validate_rule("def check_following(instruction, response):\n    return response.__class__\n")
    with pytest.raises(UnsafeRuleError, match="unapproved import"):
        validate_rule("import os\n\ndef check_following(instruction, response):\n    return True\n")


def test_type4_evidence_certificate_is_loaded_and_hashed(tmp_path: Path) -> None:
    evidence = ROOT / "configs/artifacts/hir_type4_rule_evidence.jsonl"
    certificate = ROOT / "configs/artifacts/hir_type4_rule_certificate.json"
    payload = verify_type4_certificate(evidence, certificate)
    assert payload["status"] == "certified_subset"
    assert payload["counts"] == {
        "function_instances": 12_814,
        "unique_function_hashes": 4_159,
        "frozen_probes_per_function": 2,
        "failures": 19,
    }
    assert len(payload["function_hashes"]) == 4_140
    assert len(payload["blocked_function_hashes"]) == 19
    tampered = tmp_path / "evidence.jsonl"
    tampered.write_bytes(evidence.read_bytes() + b"{}\n")
    with pytest.raises(EvaluatorCertificationError, match="tampered|blocked"):
        verify_type4_certificate(tampered, certificate)
