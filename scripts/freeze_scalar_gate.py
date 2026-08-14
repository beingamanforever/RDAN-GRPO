#!/usr/bin/env python3
"""Freeze the safe scalar data and evaluator launch evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.evaluator_cert import EvaluatorCertificationError, scalar_evaluator_certificate  # noqa: E402
from rdan_grpo.scalar_data import ScalarDataError, inspect_scalar_gate  # noqa: E402


def main() -> None:
    """Generate or byte-check the scalar gate artifacts and program pins."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed bytes differ from regeneration")
    args = parser.parse_args()
    paths = {
        "program": ROOT / "configs/program/qwen_first.json",
        "lineage_train": ROOT / "configs/roll/qwen_scalar_train.yaml",
        "launch_train": ROOT / "configs/roll/qwen_rtt_papo_response_train.yaml",
        "preflight": ROOT / "configs/roll/qwen_rtt_papo_response_preflight.yaml",
        "same_backend_diagnostic": ROOT / "configs/roll/qwen_rtt_papo_response_parity.yaml",
        "vllm_diagnostic": ROOT / "configs/roll/qwen_rtt_papo_response_vllm_parity.yaml",
        "response_data": ROOT / "configs/artifacts/qwen_merged_rl_data_manifest.json",
        "certified": ROOT / "configs/artifacts/hir_scalar_certified_manifest.json",
        "scalar": ROOT / "configs/artifacts/qwen_scalar_data_manifest.json",
        "implementation": ROOT / "configs/artifacts/hir_route_implementation.json",
        "exclusions": ROOT / "configs/artifacts/hir_hard_route_exclusions.json",
        "certificate": ROOT / "configs/artifacts/hir_evaluator_certificate.json",
    }
    gate = inspect_scalar_gate(ROOT, paths["lineage_train"], paths["certified"])
    scalar_bytes = _encode(gate.manifest)
    scalar_sha = _sha256_bytes(scalar_bytes)
    program = _load(paths["program"])
    resolution_sha = program["hard_route_policy"]["route_resolution"]["sha256"]
    implementation = {
        "schema_version": 1,
        "id": "hir_route_implementation_v1",
        "status": "implemented_not_certified",
        "route_resolution_sha256": resolution_sha,
        "scalar_data_manifest_sha256": scalar_sha,
        "counts": {"type4": len(gate.implemented), "total": len(gate.implemented)},
        "identities": list(gate.implemented),
    }
    implementation_bytes = _encode(implementation)
    implementation_sha = _sha256_bytes(implementation_bytes)
    exclusion_counts = {source: 0 for source in ("type1", "type2", "type3", "type4")}
    for identity in gate.excluded:
        exclusion_counts[identity["source"]] += 1
    exclusions = {
        "schema_version": 1,
        "id": "hir_hard_route_exclusions_v1",
        "status": "excluded",
        "route_resolution_sha256": resolution_sha,
        "scalar_data_manifest_sha256": scalar_sha,
        "counts": {**exclusion_counts, "total": len(gate.excluded)},
        "identities": list(gate.excluded),
    }
    certificate = scalar_evaluator_certificate(
        ROOT,
        gate.implemented,
        gate.function_hashes,
        resolution_sha,
        implementation_sha,
        scalar_sha,
    )
    outputs = {
        paths["scalar"]: scalar_bytes,
        paths["implementation"]: implementation_bytes,
        paths["exclusions"]: _encode(exclusions),
        paths["certificate"]: _encode(certificate),
    }
    policy = program["hard_route_policy"]
    _freeze(policy["implementation_manifest"], implementation["id"], _sha256_bytes(outputs[paths["implementation"]]))
    _freeze(policy["exclusion_manifest"], exclusions["id"], _sha256_bytes(outputs[paths["exclusions"]]))
    _freeze(policy["evaluator_certificate"], certificate["id"], _sha256_bytes(outputs[paths["certificate"]]))
    scalar_ref = program["lifecycle_artifacts"]["scalar_data"]
    _freeze(scalar_ref, gate.manifest["id"], scalar_sha)
    response_data = _load(paths["response_data"])
    if (
        response_data.get("status") != "eligible"
        or response_data.get("outputs", {}).get("merged_eligible", {}).get("records") != 18_096
    ):
        raise ValueError("response data manifest is not the eligible 18,096-row hybrid corpus")
    response_ref = program["lifecycle_artifacts"]["response_data"]
    _freeze(response_ref, response_data["id"], _sha256(paths["response_data"]))
    launch_sha = _sha256(paths["launch_train"])
    program["launch_train_config"]["path"] = "configs/roll/qwen_rtt_papo_response_train.yaml"
    program["launch_train_config"]["sha256"] = launch_sha
    program["launch_train_config"]["preflight_sha256"] = _sha256(paths["preflight"])
    program["same_backend_configs"]["diagnostic"]["sha256"] = _sha256(paths["same_backend_diagnostic"])
    program["same_backend_configs"]["vllm_diagnostic"] = {
        "path": "configs/roll/qwen_rtt_papo_response_vllm_parity.yaml",
        "sha256": _sha256(paths["vllm_diagnostic"]),
    }
    program["same_backend_configs"]["production"] = {
        "path": "configs/roll/qwen_rtt_papo_response_train.yaml",
        "sha256": launch_sha,
        "status": "frozen",
    }
    program["lifecycle_artifacts"].setdefault(
        "vllm_runtime_parity",
        {
            "status": "pending",
            "path": "configs/artifacts/qwen_vllm_runtime_parity.json",
            "artifact_id": "pending",
            "sha256": "pending",
        },
    )
    program["readiness"]["scalar_training"] = "ready"
    outputs[paths["program"]] = _encode(program)
    if args.check:
        changed = [
            str(path.relative_to(ROOT))
            for path, body in outputs.items()
            if not path.is_file() or path.read_bytes() != body
        ]
        if changed:
            raise SystemExit(f"scalar gate regeneration differs: {', '.join(changed)}")
    else:
        for path, body in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
    print(
        json.dumps(
            {
                "input_rows": gate.manifest["data"]["records"],
                "effective_rows": gate.manifest["preprocessing"]["effective_records"],
                "implemented": len(gate.implemented),
                "excluded": len(gate.excluded),
                "function_hashes": len(gate.function_hashes),
                "checked": args.check,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _freeze(reference: dict[str, Any], artifact_id: str, sha256: str) -> None:
    id_key = "artifact_id"
    if "manifest_id" in reference:
        id_key = "manifest_id"
    elif "certificate_id" in reference:
        id_key = "certificate_id"
    reference.update(status="frozen", **{id_key: artifact_id}, sha256=sha256)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _encode(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    try:
        main()
    except (EvaluatorCertificationError, ScalarDataError, OSError, ValueError) as error:
        raise SystemExit(f"scalar gate blocked: {error}") from error
