#!/usr/bin/env python3
"""Certify the first deterministic RubricHub rule subset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load data module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_data = _load_module("rdan_grpo_rubrichub_data", ROOT / "src/rdan_grpo/rubrichub_data.py")
_rules = _load_module("rdan_grpo_rubrichub_rules", ROOT / "src/rdan_grpo/rubrichub_rules.py")
source_row_hash = _data.source_row_hash
CERTIFIED_FUNCTIONS = _rules.CERTIFIED_FUNCTIONS
RubricHubRuleError = _rules.RubricHubRuleError
evaluate_rubrichub_rule = _rules.evaluate_rubrichub_rule
implementation_sha256 = _rules.implementation_sha256
verify_rule_certificate = _rules.verify_rule_certificate

RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
EXPECTED_CANDIDATES = 1_142


def main() -> None:
    """Build or verify the compact route certificate and full evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/rubrichub_instruction_following.json")
    parser.add_argument(
        "--language-certificate",
        type=Path,
        default=ROOT / "data/rubrichub-source/language-id/rubrichub_instruction_following_english_certificate.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "data/rubrichub-source/certificates/rubrichub_rule_evidence.jsonl",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=ROOT / "configs/artifacts/rubrichub_rule_certificate.json",
    )
    parser.add_argument("--rtt-root", type=Path, default=ROOT.parent / "Rubrics-To-Tokens")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = _read_json(args.config)
    if args.check:
        certificate = verify_certificate(args.certificate, args.evidence, config["source"], args.rtt_root)
    else:
        certificate = build_certificate(
            config,
            args.language_certificate,
            args.evidence,
            args.certificate,
            args.rtt_root,
        )
    print(json.dumps(_summary(certificate), sort_keys=True, separators=(",", ":")))


def build_certificate(
    config: dict[str, Any],
    language_path: Path,
    evidence_path: Path,
    certificate_path: Path,
    rtt_root: Path,
) -> dict[str, Any]:
    """Generate the two-route certification artifacts."""

    source = config["source"]
    parquet_path = _resolve(ROOT, source["path"])
    _verify_file(parquet_path, source["bytes"], source["sha256"], "RubricHub parquet")
    language = _load_language_certificate(language_path, source)
    reference = _load_reference(rtt_root, import_module=True)
    probes = _probes(reference["module"])
    candidates = _candidate_rows(parquet_path, language, source["records"])
    if len(candidates) != EXPECTED_CANDIDATES:
        raise RubricHubRuleError(
            f"candidate count differs from frozen English two-route gate: {len(candidates)} != {EXPECTED_CANDIDATES}"
        )

    records = [*probes, *({"kind": "candidate", **candidate} for candidate in candidates)]
    evidence_ref = _write_jsonl(evidence_path, records)
    implementation = implementation_sha256()
    routes = []
    for function in sorted(CERTIFIED_FUNCTIONS):
        route_probes = [probe for probe in probes if probe["function"] == function]
        routes.append(
            {
                "function": function,
                "implementation_sha256": implementation,
                "probe_count": len(route_probes),
                "probe_cases_sha256": _digest([probe["case"] for probe in route_probes]),
                "reference_parity_cases": sum(probe["reference_parity"] is True for probe in route_probes),
                "malformed_cases": sum(probe["valid"] is False for probe in route_probes),
                "status": "certified",
            }
        )
    certificate = {
        "schema_version": 1,
        "id": "rubrichub_rules_letter_frequency_comma_v1",
        "status": "certified",
        "scope": "certified_route_subset",
        "source": _certificate_source(source),
        "response_policy": {
            "input": "verbatim_decoded_response",
            "think_tag_filtering": False,
            "rollout_template": "qwen3_nothinking",
            "enable_thinking": False,
        },
        "routes": routes,
        "candidate_selection": {
            "policy": "strict_english_and_full_rule_function_set_subset_of_certified_routes",
            "frozen_expected_rows": EXPECTED_CANDIDATES,
            "rows": len(candidates),
            "source_indices": [candidate["source_index"] for candidate in candidates],
            "source_indices_sha256": _digest([candidate["source_index"] for candidate in candidates]),
            "source_row_hashes_sha256": _digest([candidate["source_row_sha256"] for candidate in candidates]),
            "language_certificate": _file_ref(language_path),
        },
        "implementation": {
            "path": "src/rdan_grpo/rubrichub_rules.py",
            "sha256": implementation,
        },
        "reference": {
            "repository": "TURLEing/Rubrics-To-Tokens",
            "revision": RTT_REVISION,
            "path": "Benchmark/instruction_following_eval/instructions.py",
            "source_sha256": reference["source_sha256"],
            "class_sha256": reference["class_sha256"],
        },
        "generator": {
            "path": "scripts/certify_rubrichub_rules.py",
            "sha256": _sha256(Path(__file__)),
        },
        "evidence": evidence_ref,
        "counts": {
            "probe_records": len(probes),
            "candidate_records": len(candidates),
            "total_records": len(records),
        },
    }
    _write_json(certificate_path, certificate)
    return verify_certificate(certificate_path, evidence_path, source, rtt_root)


def verify_certificate(
    certificate_path: Path,
    evidence_path: Path,
    source: dict[str, Any],
    rtt_root: Path,
) -> dict[str, Any]:
    """Fail closed when any certificate identity or evidence record changed."""

    certificate = verify_rule_certificate(certificate_path, source)
    if certificate.get("generator", {}).get("sha256") != _sha256(Path(__file__)):
        raise RubricHubRuleError("checker_certificate_generator_stale")
    reference = _load_reference(rtt_root)
    frozen_reference = certificate.get("reference", {})
    if frozen_reference.get("revision") != RTT_REVISION or any(
        frozen_reference.get(key) != reference[key] for key in ("source_sha256", "class_sha256")
    ):
        raise RubricHubRuleError("checker_certificate_reference_stale")
    evidence = certificate.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("sha256") != _sha256(evidence_path):
        raise RubricHubRuleError("checker_certificate_evidence_tampered")
    if evidence.get("bytes") != evidence_path.stat().st_size:
        raise RubricHubRuleError("checker_certificate_evidence_size_changed")
    lines = evidence_path.read_bytes().splitlines()
    if evidence.get("records") != len(lines) or certificate.get("counts", {}).get("total_records") != len(lines):
        raise RubricHubRuleError("checker_certificate_evidence_count_changed")
    return certificate


def _probes(reference: Any) -> list[dict[str, Any]]:
    cases = [
        (
            "LetterFrequencyChecker",
            "less_positive",
            "A cab",
            {"letter": "a", "let_frequency": 3, "let_relation": "less than"},
        ),
        (
            "LetterFrequencyChecker",
            "less_negative_equality",
            "A cabana",
            {"letter": "a", "let_frequency": 4, "let_relation": "less than"},
        ),
        ("LetterFrequencyChecker", "less_empty", "", {"letter": "z", "let_frequency": 1, "let_relation": "less than"}),
        (
            "LetterFrequencyChecker",
            "at_least_positive_equality",
            "A cab",
            {"letter": "a", "let_frequency": 2.0, "let_relation": "at least"},
        ),
        (
            "LetterFrequencyChecker",
            "at_least_negative",
            "A cab",
            {"letter": "a", "let_frequency": 3, "let_relation": "at least"},
        ),
        (
            "LetterFrequencyChecker",
            "at_least_empty",
            "",
            {"letter": "z", "let_frequency": 1, "let_relation": "at least"},
        ),
        ("CommaChecker", "positive", "No comma here.", {}),
        ("CommaChecker", "negative", "One, comma.", {}),
        ("CommaChecker", "empty", "", {}),
        ("CommaChecker", "unicode_comma", "No ASCII comma，here.", {}),
    ]
    malformed = [
        ("LetterFrequencyChecker", "missing_key", "a", {"letter": "a", "let_frequency": 1}),
        (
            "LetterFrequencyChecker",
            "extra_key",
            "a",
            {"letter": "a", "let_frequency": 1, "let_relation": "at least", "x": 1},
        ),
        (
            "LetterFrequencyChecker",
            "non_ascii_letter",
            "é",
            {"letter": "é", "let_frequency": 1, "let_relation": "at least"},
        ),
        (
            "LetterFrequencyChecker",
            "fractional_frequency",
            "a",
            {"letter": "a", "let_frequency": 1.5, "let_relation": "at least"},
        ),
        (
            "LetterFrequencyChecker",
            "nonfinite_frequency",
            "a",
            {"letter": "a", "let_frequency": float("inf"), "let_relation": "at least"},
        ),
        (
            "LetterFrequencyChecker",
            "zero_frequency",
            "",
            {"letter": "a", "let_frequency": 0, "let_relation": "at least"},
        ),
        (
            "LetterFrequencyChecker",
            "invalid_relation",
            "a",
            {"letter": "a", "let_frequency": 1, "let_relation": "more than"},
        ),
        ("CommaChecker", "nonempty_parameters", "plain", {"unused": 1}),
        ("CommaChecker", "parameters_not_mapping", "plain", None),
        ("CommaChecker", "response_not_string", None, {}),
        ("UnknownChecker", "uncertified_route", "plain", {}),
    ]
    records: list[dict[str, Any]] = []
    for function, case, response, parameters in [*cases, *malformed]:
        first = evaluate_rubrichub_rule(function, response, parameters)
        repeat = evaluate_rubrichub_rule(function, response, parameters)
        if first != repeat:
            raise RubricHubRuleError(f"nondeterministic rule result: {function}/{case}")
        parity = None
        if first.valid:
            parity = _reference_result(reference, function, response, parameters)
            if first.passed != parity:
                raise RubricHubRuleError(f"RTT reference parity failed: {function}/{case}")
        records.append(
            {
                "kind": "probe",
                "function": function,
                "case": case,
                "response": response,
                "parameters": _evidence_value(parameters),
                "valid": first.valid,
                "passed": first.passed,
                "error": first.error,
                "repeat_equal": first == repeat,
                "reference_parity": parity,
            }
        )
    return records


def _reference_result(reference: Any, function: str, response: str, parameters: dict[str, Any]) -> bool:
    checker_type = getattr(reference, function)
    checker = checker_type("rubrichub_certificate")
    if function == "LetterFrequencyChecker":
        checker.build_description(**parameters)
    else:
        checker.build_description()
    return bool(checker.check_following(response))


def _candidate_rows(
    parquet_path: Path,
    language: dict[int, dict[str, Any]],
    expected_rows: int,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RubricHubRuleError("pyarrow is required to scan the pinned RubricHub parquet") from error
    candidates = []
    seen = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches():
        for row in batch.to_pylist():
            index = seen
            seen += 1
            language_result = language.get(index)
            if language_result is None or not _strict_english(language_result):
                continue
            routes = _rule_routes(row)
            functions = [route["function"] for route in routes]
            if functions and set(functions).issubset(CERTIFIED_FUNCTIONS):
                for route in routes:
                    rule_result = evaluate_rubrichub_rule(route["function"], "", route["parameters"])
                    if not rule_result.valid:
                        raise RubricHubRuleError(
                            f"certified route parameters are invalid at source row {index}: {rule_result.error}"
                        )
                digest = source_row_hash(row)
                if language_result.get("source_row_sha256") != digest:
                    raise RubricHubRuleError(f"language certificate row {index} is stale")
                candidates.append(
                    {
                        "source_index": index,
                        "source_row_sha256": digest,
                        "functions": functions,
                        "route_parameters_sha256": _digest([route["parameters"] for route in routes]),
                    }
                )
    if seen != expected_rows or len(language) != expected_rows:
        raise RubricHubRuleError("candidate selection did not cover every pinned source row")
    return candidates


def _strict_english(result: dict[str, Any]) -> bool:
    return result.get("language") == "en" and result.get("mixed") is False and result.get("reason_flags") == []


def _rule_routes(row: dict[str, Any]) -> list[dict[str, Any]]:
    rubrics = row.get("reward_model", {}).get("rubrics")
    if not isinstance(rubrics, list):
        raise RubricHubRuleError("RubricHub source rubrics are malformed")
    routes = []
    for rubric in rubrics:
        tags = rubric.get("tags") if isinstance(rubric, dict) else None
        if not isinstance(tags, dict):
            raise RubricHubRuleError("RubricHub source tags are malformed")
        if tags.get("verifier") == "rule":
            function = tags.get("function")
            if not isinstance(function, str) or not function:
                raise RubricHubRuleError("RubricHub rule function is malformed")
            parameters = tags.get("parameters")
            if parameters is None:
                parameters = {}
            if not isinstance(parameters, dict):
                raise RubricHubRuleError("RubricHub rule parameters are malformed")
            routes.append(
                {
                    "function": function,
                    "parameters": {key: value for key, value in parameters.items() if value is not None},
                }
            )
    return routes


def _load_language_certificate(path: Path, source: dict[str, Any]) -> dict[int, dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("source") != _certificate_source(source) or payload.get("status") != "frozen":
        raise RubricHubRuleError("strict English certificate source or status is invalid")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != source["records"]:
        raise RubricHubRuleError("strict English certificate coverage is invalid")
    indexed = {}
    for result in results:
        index = result.get("source_index") if isinstance(result, dict) else None
        if not isinstance(index, int) or index in indexed:
            raise RubricHubRuleError("strict English certificate index is invalid")
        indexed[index] = result
    return indexed


def _load_reference(rtt_root: Path, *, import_module: bool = False) -> dict[str, Any]:
    root = rtt_root.resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != RTT_REVISION:
        raise RubricHubRuleError("RTT checkout differs from the pinned revision")
    source_path = root / "Benchmark/instruction_following_eval/instructions.py"
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    class_source = {
        node.name: ast.get_source_segment(source_text, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in CERTIFIED_FUNCTIONS
    }
    if set(class_source) != set(CERTIFIED_FUNCTIONS) or any(value is None for value in class_source.values()):
        raise RubricHubRuleError("certified RTT checker source cannot be located")
    classes = {
        function: hashlib.sha256(class_source[function].encode()).hexdigest()
        for function in sorted(CERTIFIED_FUNCTIONS)
    }
    reference = {
        "source_sha256": _sha256(source_path),
        "class_sha256": classes,
    }
    if import_module:
        benchmark = root / "Benchmark"
        sys.path.insert(0, str(benchmark))
        try:
            reference["module"] = importlib.import_module("instruction_following_eval.instructions")
        finally:
            sys.path.remove(str(benchmark))
    return reference


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    digest = hashlib.sha256()
    count = 0
    size = 0
    try:
        with os.fdopen(handle, "wb") as stream:
            for record in records:
                line = _canonical(record) + b"\n"
                stream.write(line)
                digest.update(line)
                count += 1
                size += len(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {
        "path": "data/rubrichub-source/certificates/rubrichub_rule_evidence.jsonl",
        "records": count,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RubricHubRuleError(f"cannot read JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RubricHubRuleError(f"JSON root must be an object: {path}")
    return payload


def _certificate_source(source: dict[str, Any]) -> dict[str, Any]:
    return {key: source[key] for key in ("dataset", "revision", "file", "sha256", "records")}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_file(path: Path, size: int, digest: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
        raise RubricHubRuleError(f"{label} differs from its frozen pin")


def _file_ref(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _evidence_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": "positive_infinity" if value > 0 else "negative_infinity"}
    if isinstance(value, dict):
        return {key: _evidence_value(item) for key, item in value.items()}
    return value


def _summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": certificate["id"],
        "status": certificate["status"],
        "routes": [route["function"] for route in certificate["routes"]],
        "candidate_rows": certificate["candidate_selection"]["rows"],
        "evidence_sha256": certificate["evidence"]["sha256"],
    }


if __name__ == "__main__":
    try:
        main()
    except (RubricHubRuleError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"RubricHub rule certification blocked: {error}") from error
