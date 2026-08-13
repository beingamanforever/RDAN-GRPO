import hashlib
import json
from pathlib import Path

import pytest
import yaml

from rdan_grpo.evaluator_cert import EvaluatorCertificationError, verify_type4_certificate
from rdan_grpo.scalar_data import inspect_scalar_gate, verify_hir_tokenizer_gate

ROOT = Path(__file__).resolve().parents[1]


def test_scalar_lineage_dataset_path_matches_roll_loader() -> None:
    config = yaml.safe_load((ROOT / "configs/roll/qwen_scalar_train.yaml").read_text(encoding="utf-8"))
    data = config["actor_train"]["data_args"]
    path = ROOT / data["dataset_dir"] / data["file_name"][0]
    assert path.resolve() == (ROOT / "data/HIR_trainv1_rdan_scalar_certified.jsonl").resolve()
    assert path.is_file()


def test_scalar_gate_matches_effective_roll_partition() -> None:
    gate = inspect_scalar_gate(
        ROOT,
        ROOT / "configs/roll/qwen_scalar_train.yaml",
        ROOT / "configs/artifacts/hir_scalar_certified_manifest.json",
    )
    assert gate.manifest["data"]["records"] == 5_699
    preprocessing = gate.manifest["preprocessing"]
    assert (
        preprocessing["implementation"]["sha256"]
        == hashlib.sha256((ROOT / "scripts/run_roll_parity.py").read_bytes()).hexdigest()
    )
    assert preprocessing["source_records"] == 16_968
    assert preprocessing["candidate_records"] == 5_700
    assert preprocessing["effective_records"] == 5_699
    assert preprocessing["largest_included_input_tokens"] == 2_008
    assert [row["row_id"] for row in preprocessing["excluded"]] == [11_279]
    implemented = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in gate.implemented}
    excluded = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in gate.excluded}
    assert len(implemented) == 12_755
    assert len(excluded) == 63_701
    assert implemented.isdisjoint(excluded)
    assert gate.manifest["scope"]["implemented_hard_identities_by_source"] == {
        "type1": 0,
        "type2": 0,
        "type3": 0,
        "type4": 12_755,
    }


def test_hir_tokenizer_certificate_is_reproducible() -> None:
    gate = verify_hir_tokenizer_gate(
        ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json",
        ROOT / "data/HIR_trainv1_rubrics_processed.jsonl",
        ROOT,
    )
    assert len(gate.accepted_row_ids) == 16_962
    assert [row["row_id"] for row in gate.rejected_rows] == [9_604, 9_776, 9_854, 9_943, 10_531, 11_279]


def test_type4_evidence_rejects_changed_repeat_or_mutation(tmp_path: Path) -> None:
    source = ROOT / "configs/artifacts/hir_type4_rule_evidence.jsonl"
    rows = source.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["probes"][0]["repeat_output"] = not first["probes"][0]["output"]
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    tampered = tmp_path / "evidence.jsonl"
    tampered.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(EvaluatorCertificationError, match="tampered|blocked"):
        verify_type4_certificate(tampered, ROOT / "configs/artifacts/hir_type4_rule_certificate.json")
