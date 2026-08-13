#!/usr/bin/env python3
"""Build and verify the pinned whole-row scalar HIR dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.evaluator_cert import verify_type4_certificate  # noqa: E402
from rdan_grpo.scalar_data import (  # noqa: E402
    ScalarDataError,
    build_rtt_hir_dataset,
    build_scalar_dataset,
    verify_hir_tokenizer_gate,
)


def main() -> None:
    """Build the exact derived bytes and write a compact manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/HIR_trainv1.jsonl")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/HIR_trainv1_rubrics_processed.jsonl")
    parser.add_argument("--resolution", type=Path, default=ROOT / "configs/artifacts/hir_route_resolution.json")
    parser.add_argument("--scope", choices=("authoritative", "rtt-full"), default="authoritative")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--tokenizer-certificate",
        type=Path,
        default=ROOT / "configs/artifacts/hir_qwen_tokenizer_certificate.json",
    )
    args = parser.parse_args()
    output = args.output or (
        ROOT / "data/HIR_trainv1_rdan_scalar_certified.jsonl"
        if args.scope == "authoritative"
        else ROOT / "data/HIR_trainv1_rtt_qwen.jsonl"
    )
    manifest_path = args.manifest or (
        ROOT / "configs/artifacts/hir_scalar_certified_manifest.json"
        if args.scope == "authoritative"
        else ROOT / "configs/artifacts/qwen_rtt_hir_data_manifest.json"
    )
    gate = verify_hir_tokenizer_gate(args.tokenizer_certificate, args.processed, ROOT)
    if args.scope == "rtt-full":
        manifest = build_rtt_hir_dataset(args.processed, output, gate, args.tokenizer_certificate)
        encoded = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        manifest_path.write_text(encoded, encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return

    certificate_path = ROOT / "configs/artifacts/hir_type4_rule_certificate.json"
    evidence_path = ROOT / "configs/artifacts/hir_type4_rule_evidence.jsonl"
    certificate = verify_type4_certificate(evidence_path, certificate_path)
    blocked_hashes = set(certificate["blocked_function_hashes"])
    certification_paths = {
        "data_manifest": ROOT / "configs/data/hir.json",
        "taxonomy": ROOT / "configs/data/hir_taxonomy.json",
        "evaluator_gate": ROOT / "configs/artifacts/hir_authoritative_evaluator_gate.json",
        "type4_certificate": certificate_path,
        "type4_evidence": evidence_path,
        "tokenizer_certificate": args.tokenizer_certificate,
    }
    manifest = build_scalar_dataset(
        args.source,
        args.processed,
        args.resolution,
        output,
        {("type4", "embedded_check_following")},
        blocked_hashes,
        gate.accepted_row_ids,
        certification_paths,
    )
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(encoded, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except ScalarDataError as error:
        raise SystemExit(f"scalar data build blocked: {error}") from error
