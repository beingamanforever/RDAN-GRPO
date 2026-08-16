import json
from pathlib import Path

import pytest
import yaml

from rdan_grpo.evaluator import EvaluatorCertificationError, verify_type4_certificate
from rdan_grpo.scalar_data import verify_hir_tokenizer_gate

ROOT = Path(__file__).resolve().parents[1]


def test_scalar_lineage_dataset_path_matches_roll_loader() -> None:
    config = yaml.safe_load((ROOT / "configs/roll/qwen_scalar_train.yaml").read_text(encoding="utf-8"))
    data = config["actor_train"]["data_args"]
    path = ROOT / data["dataset_dir"] / data["file_name"][0]
    assert path.resolve() == (ROOT / "data/HIR_trainv1_rdan_scalar_certified.jsonl").resolve()
    assert path.is_file()


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
