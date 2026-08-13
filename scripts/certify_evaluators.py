#!/usr/bin/env python3
"""Certify safe type4 rules and seal the authoritative evaluator gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.evaluator_cert import (  # noqa: E402
    EvaluatorCertificationError,
    blocked_family_artifact,
    certify_type4,
    seal_type4_evidence,
)


def main() -> None:
    """Generate source-grounded evidence and its compact certificates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-existing", action="store_true")
    args = parser.parse_args()
    evidence = ROOT / "configs/artifacts/hir_type4_rule_evidence.jsonl"
    certificate_path = ROOT / "configs/artifacts/hir_type4_rule_certificate.json"
    certificate = (
        seal_type4_evidence(evidence, certificate_path, 12_814)
        if args.seal_existing
        else certify_type4(ROOT / "data/HIR_trainv1.jsonl", evidence, certificate_path)
    )
    gate = blocked_family_artifact(certificate)
    gate_path = ROOT / "configs/artifacts/hir_authoritative_evaluator_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except EvaluatorCertificationError as error:
        raise SystemExit(f"evaluator certification blocked: {error}") from error
