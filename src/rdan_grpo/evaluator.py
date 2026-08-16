"""Source-grounded certification for restricted HIR type4 evaluators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from rdan_grpo.rule_sandbox import evaluate_rule, function_sha256, mutate_returns, validate_rule

PROBES = (
    ("Frozen instruction.", "Alpha beta.\n\nGamma delta."),
    ("Second frozen instruction.", "ONE, two, three!"),
)


class EvaluatorCertificationError(ValueError):
    """Raised when evaluator evidence is incomplete, invalid, or tampered."""


def certify_type4(source_path: str | Path, evidence_path: str | Path, certificate_path: str | Path) -> dict[str, Any]:
    """Exercise each unique pinned type4 rule twice and detect a return-negation mutation."""

    functions: dict[str, str] = {}
    instances = 0
    for line_number, line in enumerate(Path(source_path).open(encoding="utf-8"), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluatorCertificationError(f"source line {line_number} is invalid JSON") from error
        if row.get("source") != "type4":
            continue
        truth = row.get("ground_truth", {})
        for checker, code in zip(truth.get("checker", []), truth.get("functions", []), strict=True):
            if isinstance(checker, str) and checker.startswith("[rule]"):
                digest = function_sha256(code)
                if digest in functions and functions[digest] != code:
                    raise EvaluatorCertificationError("type4 SHA-256 collision")
                functions[digest] = code
                instances += 1
    rows = []
    for digest, code in sorted(functions.items()):
        validate_rule(code)
        mutated = mutate_returns(code)
        mutated_digest = function_sha256(mutated)
        probe_rows = []
        failure = None
        for instruction, response in PROBES:
            first = evaluate_rule(code, instruction, response, allowed_hashes=[digest])
            second = evaluate_rule(code, instruction, response, allowed_hashes=[digest])
            mutation = evaluate_rule(mutated, instruction, response, allowed_hashes=[mutated_digest])
            if not first.valid or first != second:
                failure = "non_deterministic_or_exception"
            elif not mutation.valid or mutation.value == first.value:
                failure = "mutation_not_detected"
            probe_rows.append(
                {
                    "input_sha256": _digest([instruction, response]),
                    "output": first.value,
                    "repeat_output": second.value,
                    "mutated_output": mutation.value,
                }
            )
        rows.append(
            {
                "function_sha256": digest,
                "parse_allowlisted": True,
                "probes": probe_rows,
                "failure": failure,
            }
        )
    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")
    certificate = _certificate_from_evidence(evidence_path, instances)
    Path(certificate_path).write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if certificate["status"] not in {"certified", "certified_subset"}:
        raise EvaluatorCertificationError("one or more type4 evaluators failed certification")
    return certificate


def verify_type4_certificate(evidence_path: str | Path, certificate_path: str | Path) -> dict[str, Any]:
    """Recompute all evidence counts and fail closed on missing, tampered, or incomplete evidence."""

    evidence_path, certificate_path = Path(evidence_path), Path(certificate_path)
    if not evidence_path.is_file() or not certificate_path.is_file():
        raise EvaluatorCertificationError("type4 evidence or certificate is missing")
    try:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluatorCertificationError("type4 certificate is invalid JSON") from error
    actual = _certificate_from_evidence(evidence_path, certificate.get("counts", {}).get("function_instances"))
    evidence = certificate.get("evidence")
    if isinstance(evidence, dict) and evidence.get("path") == "configs/artifacts/hir_type4_rule_evidence.jsonl":
        actual["evidence"]["path"] = evidence["path"]
    if certificate != actual or certificate.get("status") not in {"certified", "certified_subset"}:
        raise EvaluatorCertificationError("type4 certificate is tampered, incomplete, or blocked")
    return certificate


def seal_type4_evidence(
    evidence_path: str | Path, certificate_path: str | Path, function_instances: int
) -> dict[str, Any]:
    """Seal an already generated evidence stream without rerunning its isolated probes."""

    certificate = _certificate_from_evidence(Path(evidence_path), function_instances)
    Path(certificate_path).write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate


def blocked_family_artifact(type4_certificate: dict[str, Any]) -> dict[str, Any]:
    """Record the only certified family and conservatively block families without parity evidence."""

    blocked = {
        "type1": "official benchmark tests are not linked by parity evidence to every RTT reward route",
        "type2": "official benchmark tests are not linked by parity evidence to every RTT reward route",
        "type3": "no labeled family test corpus or reward-route parity evidence is present in pinned RTT",
    }
    return {
        "schema_version": 1,
        "id": "hir_authoritative_evaluator_gate_v1",
        "status": "certified_subset",
        "certified_families": [{"source": "type4", "route": "embedded_check_following"}],
        "blocked_sources": blocked,
        "type4_certificate_sha256": _digest(type4_certificate),
        "counts": {"certified_families": 1, "blocked_families": 43},
    }


def scalar_evaluator_certificate(
    repo_root: str | Path,
    identities: Iterable[dict[str, Any]],
    function_hashes: set[str] | frozenset[str],
    route_resolution_sha256: str,
    implementation_sha256: str,
    scalar_manifest_sha256: str,
) -> dict[str, Any]:
    """Build a certificate linked to the exact passing type4 evidence bytes."""

    root = Path(repo_root).resolve()
    evidence_path = root / "configs/artifacts/hir_type4_rule_evidence.jsonl"
    type4_path = root / "configs/artifacts/hir_type4_rule_certificate.json"
    gate_path = root / "configs/artifacts/hir_authoritative_evaluator_gate.json"
    type4 = verify_type4_certificate(evidence_path, type4_path)
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluatorCertificationError(f"cannot load authoritative evaluator gate: {error}") from error
    if gate != blocked_family_artifact(type4):
        raise EvaluatorCertificationError("authoritative evaluator gate is tampered or mis-cross-linked")
    passing = set(type4["function_hashes"])
    if len(function_hashes) != 4_138 or not function_hashes <= passing:
        raise EvaluatorCertificationError("scalar functions are not an exact passing type4 subset")
    ordered = sorted(identities, key=_canonical)
    if len(ordered) != 12_755 or ordered != list(identities):
        raise EvaluatorCertificationError("scalar evaluator identities must be sorted and complete")
    return {
        "schema_version": 1,
        "id": "hir_evaluator_certificate_v1",
        "status": "evaluator_certified",
        "route_resolution_sha256": route_resolution_sha256,
        "implementation_manifest_sha256": implementation_sha256,
        "scalar_data_manifest_sha256": scalar_manifest_sha256,
        "identities": ordered,
        "evidence": {
            "authoritative_gate": {
                "path": "configs/artifacts/hir_authoritative_evaluator_gate.json",
                "sha256": _sha256(gate_path),
            },
            "type4_certificate": {
                "path": "configs/artifacts/hir_type4_rule_certificate.json",
                "sha256": _sha256(type4_path),
            },
            "type4_evidence": {
                "path": "configs/artifacts/hir_type4_rule_evidence.jsonl",
                "sha256": _sha256(evidence_path),
            },
        },
        "derived": {
            "implemented_identities": len(ordered),
            "function_hashes": len(function_hashes),
            "probes_per_function": len(PROBES),
            "repeat_checks": len(function_hashes) * len(PROBES),
            "mutation_checks": len(function_hashes) * len(PROBES),
            "failures": 0,
        },
    }


def _certificate_from_evidence(path: Path, function_instances: Any) -> dict[str, Any]:
    rows = []
    try:
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise EvaluatorCertificationError(f"evidence line {line_number} is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluatorCertificationError(f"cannot load type4 evidence: {error}") from error
    hashes = [row.get("function_sha256") for row in rows]
    passed = [row.get("function_sha256") for row in rows if row.get("failure") is None]
    blocked = [row.get("function_sha256") for row in rows if row.get("failure") is not None]
    complete = (
        len(rows) == 4_159
        and all(isinstance(digest, str) and len(digest) == 64 for digest in hashes)
        and hashes == sorted(set(hashes))
        and all(_valid_evidence_row(row) for row in rows)
    )
    if not isinstance(function_instances, int) or isinstance(function_instances, bool):
        complete = False
    return {
        "schema_version": 1,
        "id": "hir_type4_rule_certificate_v1",
        "status": (
            "certified"
            if complete and function_instances == 12_814 and not blocked
            else "certified_subset"
            if complete and function_instances == 12_814 and passed
            else "blocked"
        ),
        "evaluator": {
            "path": "src/rdan_grpo/rule_sandbox.py",
            "sha256": _sha256(Path(__file__).with_name("rule_sandbox.py")),
        },
        "evidence": {"path": _display_path(path), "bytes": path.stat().st_size, "sha256": _sha256(path)},
        "counts": {
            "function_instances": function_instances,
            "unique_function_hashes": len(rows),
            "frozen_probes_per_function": len(PROBES),
            "failures": sum(row.get("failure") is not None for row in rows),
        },
        "function_hashes": passed,
        "blocked_function_hashes": blocked,
    }


def _valid_evidence_row(row: dict[str, Any]) -> bool:
    if set(row) != {"failure", "function_sha256", "parse_allowlisted", "probes"}:
        return False
    if row["parse_allowlisted"] is not True or row["failure"] not in {
        None,
        "non_deterministic_or_exception",
        "mutation_not_detected",
    }:
        return False
    probes = row["probes"]
    if not isinstance(probes, list) or len(probes) != len(PROBES):
        return False
    for probe, (instruction, response) in zip(probes, PROBES, strict=True):
        if not isinstance(probe, dict) or set(probe) != {
            "input_sha256",
            "output",
            "repeat_output",
            "mutated_output",
        }:
            return False
        if probe["input_sha256"] != _digest([instruction, response]):
            return False
        if row["failure"] is None:
            if not all(isinstance(probe[key], bool) for key in ("output", "repeat_output", "mutated_output")):
                return False
            if probe["repeat_output"] != probe["output"] or probe["mutated_output"] == probe["output"]:
                return False
    return True


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())
