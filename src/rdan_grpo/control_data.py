"""Reproducible generation and freezing for reconstructed SFT and DPO controls."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from rdan_grpo import baseline
from rdan_grpo import program as program_contract
from rdan_grpo.hir import classify_hir_row

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/baselines/qwen_control_data.json"
HIR_MANIFEST = ROOT / "configs/data/hir.json"
MAX_HTTP_BYTES = 64 * 1024 * 1024
PINNED_SOURCE = {
    "dataset": "sastpg/HIR-16K",
    "revision": "2a95f69eb56cc47edc16a45f939cde479673a4cb",
    "path": "HIR_trainv1.jsonl",
    "bytes": 53_147_812,
    "records": 16_968,
    "sha256": "465a01c19dc29e2c8d1cf183ccf3135872f7ec94ef10b20b7eb35603164c183b",
    "row_ids_sha256": "c65b58ecd4458858153bf4fdd37b83e7a99dba6e04c8fe2258b67ad933e42497",
}
PINNED_TEACHER_MODEL = "openai/gpt-5.6-luna"
PINNED_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PINNED_QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


class ControlDataError(RuntimeError):
    """Raised when generation or freezing violates the reconstructed-data contract."""


@dataclass(frozen=True)
class StageSummary:
    """Current append-only stage state after one generation pass."""

    selected_rows: int
    successful_rows: int
    failed_rows: int
    recorded_failures: int
    raw_sha256: str


@dataclass(frozen=True)
class _SourceRow:
    position: int
    row_id: int
    prompt: str
    digest: str
    source: str
    criteria: tuple[str, ...]
    hard_mask: tuple[bool, ...]


class _RequestFailure(Exception):
    def __init__(self, kind: str, retryable: bool) -> None:
        super().__init__(kind)
        self.kind = kind
        self.retryable = retryable


def run_teacher_stage(
    source_path: str | Path,
    output_path: str | Path,
    *,
    row_ids: Iterable[int] | None = None,
    config_path: str | Path = CONFIG,
    hir_manifest_path: str | Path = HIR_MANIFEST,
    endpoint: str | None = None,
    command: Sequence[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> StageSummary:
    """Append independent Luna teacher results, retrying only bounded transient failures."""

    source_path, output_path = Path(source_path), Path(output_path)
    _require_raw_path(output_path)
    config, config_hash = _load_config(Path(config_path), Path(hir_manifest_path))
    rows, source_identity = _load_source(source_path, config["source"])
    repo_root = _repo_root(Path(config_path))
    selected = _certified_selection(rows, row_ids, config, repo_root)
    teacher = config["teacher"]
    request_parameters = teacher["request"]
    endpoint = _http_url(endpoint or teacher["endpoint"], exact_path="/api/v1/chat/completions")
    if endpoint != teacher["endpoint"] and urlsplit(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ControlDataError("teacher endpoint overrides are restricted to loopback test servers")
    catalog_url = teacher["catalog_url"]
    if endpoint != teacher["endpoint"]:
        parts = urlsplit(endpoint)
        catalog_url = urlunsplit((parts.scheme, parts.netloc, "/api/v1/models", "", ""))
    identity = _stage_identity(
        "teacher",
        source_identity,
        selected,
        config_hash,
        {
            "endpoint_sha256": _digest(endpoint),
            "public_alias": teacher["public_alias"],
            "model": teacher["model"],
            "revision": teacher["revision"],
            "response_models": teacher["response_models"],
            "request": request_parameters,
            "retry": teacher["retry"],
            "timeout_seconds": teacher["timeout_seconds"],
        },
        command,
    )
    api_key = os.environ.get(teacher["api_key_env"])
    if not api_key:
        raise ControlDataError("OPENROUTER_API_KEY is required in the environment")
    _preflight_teacher(catalog_url, teacher)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled",
    }
    maximum_attempts = teacher["retry"]["maximum_attempts"]
    backoff = teacher["retry"]["backoff_seconds"]

    with _locked(output_path):
        _ensure_identity(output_path, identity)
        existing = _load_stage(output_path, "teacher", selected, _teacher_payloads(selected, teacher))
        for row in selected:
            attempts = existing.get(row.row_id, [])
            if attempts and attempts[-1]["status"] == "ok":
                continue
            payload = _teacher_payload(row, teacher)
            content = None
            failure = _RequestFailure("unattempted", False)
            used_attempts = 0
            for request_attempt in range(1, maximum_attempts + 1):
                used_attempts = request_attempt
                try:
                    response = _request_json(endpoint, payload, teacher["timeout_seconds"], headers)
                    content = _teacher_content(response, teacher)
                    failure = _RequestFailure("none", False)
                    break
                except _RequestFailure as error:
                    failure = error
                    if not error.retryable or request_attempt == maximum_attempts:
                        break
                    sleep(backoff[request_attempt - 1])
            record = _stage_record("teacher", row, len(attempts) + 1, payload, content, failure, used_attempts)
            _append_jsonl(output_path, record)
            existing.setdefault(row.row_id, []).append(record)
        return _stage_summary(output_path, existing, len(selected))


def run_candidate_stage(
    source_path: str | Path,
    output_path: str | Path,
    api_base: str,
    server_manifest_path: str | Path,
    *,
    row_ids: Iterable[int] | None = None,
    config_path: str | Path = CONFIG,
    hir_manifest_path: str | Path = HIR_MANIFEST,
    command: Sequence[str] | None = None,
) -> StageSummary:
    """Append eight pinned local-Qwen candidates after verifying the live server identity."""

    source_path, output_path = Path(source_path), Path(output_path)
    server_manifest_path = Path(server_manifest_path)
    _require_raw_path(output_path)
    config, config_hash = _load_config(Path(config_path), Path(hir_manifest_path))
    rows, source_identity = _load_source(source_path, config["source"])
    selected = _certified_selection(rows, row_ids, config, _repo_root(Path(config_path)))
    candidate = config["candidates"]
    expected_model = {
        "name": candidate["model"],
        "revision": candidate["revision"],
        "served_name": candidate["served_name"],
    }
    try:
        server_manifest, server_hash = baseline._load_server_manifest(server_manifest_path, expected_model)
        normalized_api = baseline._api_base(api_base)
        if urlsplit(normalized_api).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ControlDataError("local Qwen API base must use a loopback host")
        baseline._verify_model(
            normalized_api,
            candidate["served_name"],
            Path(server_manifest["model"]["snapshot_path"]),
            candidate["timeout_seconds"],
        )
    except baseline.EvaluationError as error:
        raise ControlDataError(f"local Qwen server identity failed: {error}") from error
    identity = _stage_identity(
        "candidates",
        source_identity,
        selected,
        config_hash,
        {
            "api_base_sha256": _digest(normalized_api),
            "model": candidate["model"],
            "revision": candidate["revision"],
            "served_name": candidate["served_name"],
            "request": candidate["request"],
            "server_manifest_sha256": server_hash,
            "timeout_seconds": candidate["timeout_seconds"],
        },
        command,
    )
    payloads = {row.row_id: _candidate_payload(row, candidate) for row in selected}
    with _locked(output_path):
        _ensure_identity(output_path, identity)
        existing = _load_stage(output_path, "candidates", selected, payloads)
        endpoint = f"{normalized_api}/chat/completions"
        for row in selected:
            attempts = existing.get(row.row_id, [])
            if attempts and attempts[-1]["status"] == "ok":
                continue
            payload = payloads[row.row_id]
            contents = None
            try:
                response = _request_json(endpoint, payload, candidate["timeout_seconds"])
                contents = _candidate_contents(response, candidate)
                failure = _RequestFailure("none", False)
            except _RequestFailure as error:
                failure = error
            record = _stage_record("candidates", row, len(attempts) + 1, payload, contents, failure, 1)
            _append_jsonl(output_path, record)
            existing.setdefault(row.row_id, []).append(record)
        return _stage_summary(output_path, existing, len(selected))


def freeze_control_data(
    source_path: str | Path,
    teacher_path: str | Path,
    candidate_path: str | Path,
    evidence_path: str | Path,
    sft_path: str | Path,
    dpo_path: str | Path,
    sft_manifest_path: str | Path,
    dpo_manifest_path: str | Path,
    *,
    config_path: str | Path = CONFIG,
    hir_manifest_path: str | Path = HIR_MANIFEST,
    command: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze SFT and DPO JSONL only from complete externally evaluated outputs."""

    source_path, teacher_path, candidate_path = map(Path, (source_path, teacher_path, candidate_path))
    evidence_path = Path(evidence_path)
    outputs = tuple(map(Path, (sft_path, dpo_path, sft_manifest_path, dpo_manifest_path)))
    if len(set(outputs)) != len(outputs):
        raise ControlDataError("freeze output paths must be distinct")
    _require_raw_path(teacher_path)
    _require_raw_path(candidate_path)
    _require_raw_path(evidence_path)
    config, config_hash = _load_config(Path(config_path), Path(hir_manifest_path))
    repo_root = _repo_root(Path(config_path))
    judge_contract = _load_validated_judge_contract(repo_root)
    rows, source_identity = _load_source(source_path, config["source"])
    teacher_identity = _read_identity(teacher_path)
    candidate_identity = _read_identity(candidate_path)
    _matching_stage_identities(teacher_identity, candidate_identity, source_identity, config_hash, config)
    selected = _selected_from_identity(rows, teacher_identity)
    teacher_attempts = _load_stage(teacher_path, "teacher", selected, _teacher_payloads(selected, config["teacher"]))
    candidate_payloads = {row.row_id: _candidate_payload(row, config["candidates"]) for row in selected}
    candidate_attempts = _load_stage(candidate_path, "candidates", selected, candidate_payloads)
    teachers = _successful_values(teacher_attempts, selected, "teacher")
    candidates = _successful_values(candidate_attempts, selected, "candidates")
    expected = {
        (row.row_id, output["sha256"])
        for row in selected
        for output in [teachers[row.row_id], *candidates[row.row_id]]
    }
    if len(expected) != len(selected) * 9:
        raise ControlDataError("teacher and candidate outputs contain an identical pair")
    evidence, evidence_manifest, dev_split = _load_evidence_manifest(
        evidence_path, expected, selected, config, repo_root, judge_contract
    )

    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    seen_teacher_digests: set[str] = set()
    for row in selected:
        teacher = teachers[row.row_id]
        outputs_for_row = [teacher, *candidates[row.row_id]]
        for output in outputs_for_row:
            _reject_leakage(row, output["content"], evidence[(row.row_id, output["sha256"])])
        teacher_evidence = evidence[(row.row_id, teacher["sha256"])]
        if teacher_evidence["hard"]["pass"] and teacher["sha256"] not in seen_teacher_digests:
            sft_rows.append({"row_id": row.row_id, "prompt": row.prompt, "output": teacher["content"]})
            seen_teacher_digests.add(teacher["sha256"])
        ranked = sorted(
            outputs_for_row,
            key=lambda output: _preference_key(output, evidence[(row.row_id, output["sha256"])]),
        )
        chosen, rejected = ranked[0], ranked[-1]
        if not evidence[(row.row_id, chosen["sha256"])]["hard"]["pass"]:
            raise ControlDataError(f"row {row.row_id}: no candidate has an authoritative hard pass")
        if chosen["sha256"] == rejected["sha256"]:
            raise ControlDataError(f"row {row.row_id}: chosen and rejected outputs are identical")
        dpo_rows.append(
            {
                "row_id": row.row_id,
                "prompt": row.prompt,
                "chosen": chosen["content"],
                "rejected": rejected["content"],
            }
        )

    sft_body = _jsonl_bytes(sft_rows)
    dpo_body = _jsonl_bytes(dpo_rows)
    common = {
        "evidence_sha256": evidence_manifest["sha256"],
        "teacher": {"model_id": config["teacher"]["model"], "revision": config["teacher"]["revision"]},
    }
    sft_manifest = _baseline_manifest(
        "sft",
        common,
        repo_root,
        Path(sft_path),
        sft_body,
        sft_rows,
        dev_split,
        config,
    )
    dpo_manifest = _baseline_manifest(
        "dpo",
        common,
        repo_root,
        Path(dpo_path),
        dpo_body,
        dpo_rows,
        dev_split,
        config,
    )
    payloads = {
        Path(sft_path): sft_body,
        Path(dpo_path): dpo_body,
        Path(sft_manifest_path): _pretty_json_bytes(sft_manifest),
        Path(dpo_manifest_path): _pretty_json_bytes(dpo_manifest),
    }
    _publish_immutable(payloads)
    return sft_manifest, dpo_manifest


def _load_config(path: Path, hir_manifest_path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        config = _json_loads(raw)
        hir_manifest = _json_loads(hir_manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlDataError(f"cannot load control-data configuration: {error}") from error
    expected_teacher = {
        "max_tokens": 4096,
        "seed": 240520,
        "reasoning": {"effort": "medium", "exclude": True},
        "provider": {
            "order": ["openai"],
            "only": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": False,
        },
    }
    expected_candidates = {
        "temperature": 0.99,
        "top_p": 0.99,
        "top_k": 100,
        "max_tokens": 4096,
        "seed": 240520,
        "n": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        valid = (
            set(config) == {"schema_version", "id", "source", "source_sha256", "teacher", "candidates", "evidence"}
            and set(config["source"]) == set(PINNED_SOURCE)
            and set(config["source_sha256"]) == {"hir_source", "hir_processed", "taxonomy"}
            and set(config["teacher"])
            == {
                "endpoint",
                "catalog_url",
                "public_alias",
                "model",
                "revision",
                "response_models",
                "request",
                "required_parameters",
                "timeout_seconds",
                "retry",
                "api_key_env",
            }
            and set(config["teacher"]["request"]) == {"max_tokens", "seed", "reasoning", "provider"}
            and set(config["teacher"]["request"]["reasoning"]) == {"effort", "exclude"}
            and set(config["teacher"]["request"]["provider"])
            == {"order", "only", "allow_fallbacks", "require_parameters", "data_collection", "zdr"}
            and set(config["teacher"]["retry"]) == {"maximum_attempts", "backoff_seconds"}
            and set(config["candidates"]) == {"model", "revision", "served_name", "request", "timeout_seconds"}
            and set(config["candidates"]["request"])
            == {"temperature", "top_p", "top_k", "max_tokens", "seed", "n", "chat_template_kwargs"}
            and set(config["candidates"]["request"]["chat_template_kwargs"]) == {"enable_thinking"}
            and set(config["evidence"])
            == {"authoritative_evaluator", "evaluator_implementation", "judge_calibration", "leakage_detector"}
            and set(config["evidence"]["authoritative_evaluator"]) == {"path", "id", "sha256"}
            and set(config["evidence"]["evaluator_implementation"]) == {"path", "id", "sha256"}
            and set(config["evidence"]["judge_calibration"]) == {"path", "id"}
            and set(config["evidence"]["leakage_detector"]) == {"path", "id"}
            and set(hir_manifest)
            == {"schema_version", "id", "dataset", "revision", "license", "source", "rtt_processed"}
            and set(hir_manifest["source"]) == {"path", "url", "bytes", "records", "sha256"}
            and set(hir_manifest["rtt_processed"]) == {"repository", "revision", "path", "records", "sha256"}
            and config["schema_version"] == 1
            and config["id"] == "qwen_reconstructed_control_data_v1"
            and config["source"] == PINNED_SOURCE
            and config["source_sha256"]
            == {
                "hir_source": PINNED_SOURCE["sha256"],
                "hir_processed": "d6690a29cd4f24a3627dd8d48e78953191d0c97ad6acb92cdaf2bf5f1b67568a",
                "taxonomy": program_contract.HIR_TAXONOMY_SHA256,
            }
            and hir_manifest["dataset"] == PINNED_SOURCE["dataset"]
            and hir_manifest["revision"] == PINNED_SOURCE["revision"]
            and all(
                hir_manifest["source"][key] == PINNED_SOURCE[key] for key in ("path", "bytes", "records", "sha256")
            )
            and config["teacher"]["public_alias"] == PINNED_TEACHER_MODEL
            and config["teacher"]["model"] == "openai/gpt-5.6-luna-20260709"
            and config["teacher"]["revision"] == "openai/gpt-5.6-luna-20260709"
            and config["teacher"]["endpoint"] == "https://openrouter.ai/api/v1/chat/completions"
            and config["teacher"]["catalog_url"] == "https://openrouter.ai/api/v1/models"
            and config["teacher"]["response_models"] == ["openai/gpt-5.6-luna-20260709"]
            and config["teacher"]["request"] == expected_teacher
            and config["teacher"]["required_parameters"] == ["max_tokens", "reasoning", "seed"]
            and config["teacher"]["api_key_env"] == "OPENROUTER_API_KEY"
            and config["teacher"]["retry"] == {"maximum_attempts": 3, "backoff_seconds": [1, 2]}
            and config["teacher"]["timeout_seconds"] == 600
            and config["candidates"]["model"] == PINNED_QWEN_MODEL
            and config["candidates"]["revision"] == PINNED_QWEN_REVISION
            and config["candidates"]["served_name"] == "qwen3-4b-instruct-2507"
            and config["candidates"]["request"] == expected_candidates
            and config["candidates"]["timeout_seconds"] == 600
            and config["evidence"]["authoritative_evaluator"]
            == {
                "path": "configs/artifacts/hir_evaluator_certificate.json",
                "id": "hir_evaluator_certificate_v1",
                "sha256": "24cf5aa6d8cac4d45c8103f6555f0293721a2ac8120f489ff93bce0d9951d516",
            }
            and config["evidence"]["evaluator_implementation"]
            == {
                "path": "configs/artifacts/hir_route_implementation.json",
                "id": "hir_route_implementation_v1",
                "sha256": "87a387b91fbba2f44c933931bfe40dab1c56127560d0a24717ec9195a0857e01",
            }
            and config["evidence"]["judge_calibration"]
            == {
                "path": "configs/artifacts/qwen_judge_calibration.json",
                "id": "qwen_judge_calibration_v1",
            }
            and config["evidence"]["leakage_detector"]
            == {
                "path": "src/rdan_grpo/control_data.py",
                "id": "control_data_exact_leakage_v1",
            }
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ControlDataError("control-data configuration differs from the pinned contract")
    return config, hashlib.sha256(raw).hexdigest()


def _load_source(path: Path, expected: Mapping[str, Any]) -> tuple[list[_SourceRow], dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ControlDataError(f"cannot read HIR source: {error}") from error
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ControlDataError("HIR source bytes do not match configs/data/hir.json")
    rows = []
    ids = []
    seen: set[int] = set()
    try:
        for position, line in enumerate(raw.splitlines()):
            if not line:
                raise ControlDataError(f"blank HIR source row at position {position}")
            value = _json_loads(line)
            row_id = value.get("id") if isinstance(value, dict) else None
            prompt = value.get("prompt") if isinstance(value, dict) else None
            criteria = value.get("criteria") if isinstance(value, dict) else None
            source = value.get("source") if isinstance(value, dict) else None
            if (
                not isinstance(row_id, int)
                or isinstance(row_id, bool)
                or row_id in seen
                or not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(criteria, list)
                or not criteria
                or not all(isinstance(item, str) and item.strip() for item in criteria)
            ):
                raise ControlDataError(f"invalid or duplicate HIR source identity at position {position}")
            try:
                hard_mask = classify_hir_row(value)
            except ValueError as error:
                raise ControlDataError(f"invalid HIR rubric identity at position {position}: {error}") from error
            seen.add(row_id)
            ids.append(row_id)
            instruction = f"{prompt.strip()}\n\nRequirements:\n" + "\n".join(f"- {item.strip()}" for item in criteria)
            rows.append(_SourceRow(position, row_id, instruction, _digest(value), source, tuple(criteria), hard_mask))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlDataError(f"cannot parse HIR source: {error}") from error
    if len(rows) != expected["records"] or _digest(ids) != expected["row_ids_sha256"]:
        raise ControlDataError("HIR source row count, unique IDs, or frozen order differs")
    return rows, {
        "dataset": expected["dataset"],
        "revision": expected["revision"],
        "path": path.name,
        "bytes": len(raw),
        "records": len(rows),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_ids_sha256": _digest(ids),
    }


def _select_rows(rows: list[_SourceRow], requested: Iterable[int] | None) -> list[_SourceRow]:
    if requested is None:
        return rows
    values = list(requested)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values) or len(values) != len(
        set(values)
    ):
        raise ControlDataError("selected row IDs must be unique integers")
    requested_set = set(values)
    selected = [row for row in rows if row.row_id in requested_set]
    if len(selected) != len(values) or not selected:
        raise ControlDataError("selected row IDs contain missing IDs or are empty")
    return selected


def _repo_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name != "baselines" or resolved.parent.parent.name != "configs":
        raise ControlDataError("control-data config must be under configs/baselines")
    return resolved.parents[2]


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ControlDataError(f"{label} escapes the repository root") from error
    return path


def _certified_selection(
    rows: list[_SourceRow],
    requested: Iterable[int] | None,
    config: Mapping[str, Any],
    repo_root: Path,
) -> list[_SourceRow]:
    certificate_ref = config["evidence"]["authoritative_evaluator"]
    implementation_ref = config["evidence"]["evaluator_implementation"]
    certificate_path = _under(repo_root, certificate_ref["path"], "evaluator certificate")
    implementation_path = _under(repo_root, implementation_ref["path"], "evaluator implementation")
    certificate = _verified_json(certificate_path, certificate_ref["sha256"], certificate_ref["id"])
    implementation = _verified_json(implementation_path, implementation_ref["sha256"], implementation_ref["id"])
    if (
        certificate.get("status") != "evaluator_certified"
        or certificate.get("implementation_manifest_sha256") != implementation_ref["sha256"]
        or implementation.get("status") != "implemented_not_certified"
    ):
        raise ControlDataError("authoritative evaluator certificate linkage is invalid")
    identities = certificate.get("identities")
    if not isinstance(identities, list):
        raise ControlDataError("authoritative evaluator identities are missing")
    certified = {
        (value.get("row_id"), value.get("rubric_index"), value.get("source"), value.get("route"))
        for value in identities
        if isinstance(value, dict)
    }
    eligible = []
    for row in rows:
        hard = [
            (row.row_id, index, row.source, "embedded_check_following")
            for index, is_hard in enumerate(row.hard_mask)
            if is_hard
        ]
        if hard and not all(row.hard_mask) and all(identity in certified for identity in hard):
            eligible.append(row)
    if requested is None:
        if not eligible:
            raise ControlDataError("authoritative evaluator certifies no source rows")
        return eligible
    selected = _select_rows(rows, requested)
    eligible_ids = {row.row_id for row in eligible}
    unsupported = [row.row_id for row in selected if row.row_id not in eligible_ids]
    if unsupported:
        raise ControlDataError(f"selected rows have unsupported hard routes: {unsupported}")
    return selected


def _verified_json(path: Path, expected_sha256: str, expected_id: str) -> dict[str, Any]:
    try:
        body = path.read_bytes()
        value = _json_loads(body)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlDataError(f"cannot load frozen certificate {path}: {error}") from error
    if (
        hashlib.sha256(body).hexdigest() != expected_sha256
        or not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("id") != expected_id
    ):
        raise ControlDataError(f"frozen certificate is missing, forged, or changed: {path}")
    return value


def _preflight_teacher(catalog_url: str, teacher: Mapping[str, Any]) -> None:
    try:
        catalog = _get_json(catalog_url, teacher["timeout_seconds"])
    except _RequestFailure as error:
        raise ControlDataError(f"teacher supported-parameter preflight failed: {error.kind}") from error
    data = catalog.get("data")
    models = data if isinstance(data, list) else []
    matches = [
        model
        for model in models
        if isinstance(model, dict)
        and model.get("id") == teacher["public_alias"]
        and model.get("canonical_slug") == teacher["model"]
    ]
    supported = matches[0].get("supported_parameters") if len(matches) == 1 else None
    if not isinstance(supported, list) or not set(teacher["required_parameters"]) <= set(supported):
        raise ControlDataError("teacher model does not support every pinned request parameter")


def _teacher_payload(row: _SourceRow, teacher: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": teacher["model"],
        "messages": [{"role": "user", "content": row.prompt}],
        **teacher["request"],
    }


def _teacher_payloads(rows: list[_SourceRow], teacher: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {row.row_id: _teacher_payload(row, teacher) for row in rows}


def _candidate_payload(row: _SourceRow, candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": candidate["served_name"],
        "messages": [{"role": "user", "content": row.prompt}],
        **candidate["request"],
    }


def _teacher_content(response: Mapping[str, Any], teacher: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    model = response.get("model")
    metadata = response.get("openrouter_metadata")
    endpoints = metadata.get("endpoints") if isinstance(metadata, dict) else None
    available = endpoints.get("available") if isinstance(endpoints, dict) else None
    selected = [value for value in available or [] if isinstance(value, dict) and value.get("selected") is True]
    if (
        model not in teacher["response_models"]
        or not isinstance(metadata, dict)
        or metadata.get("requested") != teacher["model"]
        or metadata.get("attempt") != 1
        or len(selected) != 1
        or str(selected[0].get("provider", "")).lower() != "openai"
        or selected[0].get("model") not in teacher["response_models"]
    ):
        raise _RequestFailure("teacher_identity_mismatch", False)
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise _RequestFailure("invalid_teacher_choices", False)
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if choices[0].get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
        raise _RequestFailure("empty_teacher_content", False)
    return content.strip()


def _candidate_contents(response: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    if response.get("model") != candidate["served_name"]:
        raise _RequestFailure("candidate_identity_mismatch", False)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != candidate["request"]["n"]:
        raise _RequestFailure("invalid_candidate_count", False)
    indexed: dict[int, str] = {}
    for choice in choices:
        index = choice.get("index") if isinstance(choice, dict) else None
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(index, int) or isinstance(index, bool) or index in indexed:
            raise _RequestFailure("invalid_candidate_index", False)
        if choice.get("finish_reason") != "stop":
            raise _RequestFailure("candidate_incomplete", False)
        if not isinstance(content, str) or not content.strip():
            raise _RequestFailure("empty_candidate_content", False)
        indexed[index] = content.strip()
    if set(indexed) != set(range(candidate["request"]["n"])):
        raise _RequestFailure("invalid_candidate_indices", False)
    contents = [indexed[index] for index in range(candidate["request"]["n"])]
    if len({_digest(content) for content in contents}) != len(contents):
        raise _RequestFailure("duplicate_candidates", False)
    return contents


def _request_json(
    url: str,
    payload: Mapping[str, Any],
    timeout: int,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        headers=dict(headers or {"Content-Type": "application/json"}),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise _RequestFailure(f"http_{response.status}", response.status == 429 or response.status >= 500)
            body = response.read(MAX_HTTP_BYTES + 1)
    except HTTPError as error:
        raise _RequestFailure(f"http_{error.code}", error.code == 429 or error.code >= 500) from error
    except (TimeoutError, URLError, OSError) as error:
        raise _RequestFailure("transport_error", True) from error
    if len(body) > MAX_HTTP_BYTES:
        raise _RequestFailure("response_too_large", False)
    try:
        value = _json_loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _RequestFailure("invalid_json_response", False) from error
    if not isinstance(value, dict):
        raise _RequestFailure("invalid_json_response", False)
    return value


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as response:
            if response.status != 200:
                raise _RequestFailure(f"http_{response.status}", response.status == 429 or response.status >= 500)
            body = response.read(MAX_HTTP_BYTES + 1)
    except HTTPError as error:
        raise _RequestFailure(f"http_{error.code}", error.code == 429 or error.code >= 500) from error
    except (TimeoutError, URLError, OSError) as error:
        raise _RequestFailure("transport_error", True) from error
    if len(body) > MAX_HTTP_BYTES:
        raise _RequestFailure("response_too_large", False)
    try:
        value = _json_loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _RequestFailure("invalid_json_response", False) from error
    if not isinstance(value, dict):
        raise _RequestFailure("invalid_json_response", False)
    return value


def _stage_record(
    stage: str,
    row: _SourceRow,
    attempt: int,
    payload: Mapping[str, Any],
    value: str | list[str] | None,
    failure: _RequestFailure,
    network_attempts: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "position": row.position,
        "row_id": row.row_id,
        "source_sha256": row.digest,
        "request_sha256": _digest(payload),
        "attempt": attempt,
        "network_attempts": network_attempts,
    }
    if value is None:
        return record | {"status": "failure", "error": failure.kind}
    if isinstance(value, str):
        return record | {"status": "ok", "content": value, "content_sha256": _digest(value)}
    return record | {
        "status": "ok",
        "candidates": [
            {"index": index, "content": content, "content_sha256": _digest(content)}
            for index, content in enumerate(value)
        ],
    }


def _load_stage(
    path: Path,
    stage: str,
    rows: list[_SourceRow],
    payloads: Mapping[int, Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    expected = {row.row_id: row for row in rows}
    attempts: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        return attempts
    try:
        lines = path.read_bytes().splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line:
                raise ControlDataError(f"blank {stage} raw row at line {line_number}")
            record = _json_loads(line)
            row_id = record.get("row_id") if isinstance(record, dict) else None
            row = expected.get(row_id)
            prior = attempts.setdefault(row_id, []) if row is not None else []
            common_valid = (
                isinstance(record, dict)
                and record.get("schema_version") == 1
                and record.get("stage") == stage
                and row is not None
                and record.get("position") == row.position
                and record.get("source_sha256") == row.digest
                and record.get("request_sha256") == _digest(payloads[row_id])
                and record.get("attempt") == len(prior) + 1
                and isinstance(record.get("network_attempts"), int)
                and not isinstance(record.get("network_attempts"), bool)
                and record["network_attempts"] > 0
                and record.get("status") in {"ok", "failure"}
                and not (prior and prior[-1].get("status") == "ok")
            )
            if not common_valid or not _valid_stage_value(record, stage):
                raise ControlDataError(f"invalid, duplicate, or inconsistent {stage} identity at line {line_number}")
            prior.append(record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlDataError(f"cannot load {stage} raw output: {error}") from error
    return attempts


def _valid_stage_value(record: Mapping[str, Any], stage: str) -> bool:
    if record.get("status") == "failure":
        return isinstance(record.get("error"), str) and bool(record["error"])
    if stage == "teacher":
        content = record.get("content")
        return isinstance(content, str) and bool(content.strip()) and record.get("content_sha256") == _digest(content)
    candidates = record.get("candidates")
    return (
        isinstance(candidates, list)
        and len(candidates) == 8
        and all(
            isinstance(value, dict)
            and value.get("index") == index
            and isinstance(value.get("content"), str)
            and bool(value["content"].strip())
            and value.get("content_sha256") == _digest(value["content"])
            for index, value in enumerate(candidates)
        )
        and len({value["content_sha256"] for value in candidates}) == 8
    )


def _successful_values(
    attempts: Mapping[int, list[dict[str, Any]]], rows: list[_SourceRow], stage: str
) -> dict[int, Any]:
    values = {}
    for row in rows:
        records = attempts.get(row.row_id)
        if not records or records[-1]["status"] != "ok":
            raise ControlDataError(f"row {row.row_id}: failed or missing {stage} call")
        latest = records[-1]
        if stage == "teacher":
            values[row.row_id] = {"content": latest["content"], "sha256": latest["content_sha256"]}
        else:
            values[row.row_id] = [
                {"content": candidate["content"], "sha256": candidate["content_sha256"]}
                for candidate in latest["candidates"]
            ]
    return values


def _load_evidence_manifest(
    path: Path,
    expected: set[tuple[int, str]],
    rows: list[_SourceRow],
    config: Mapping[str, Any],
    repo_root: Path,
    judge_contract: dict[str, Any],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any], dict[str, Any]]:
    try:
        manifest_bytes = path.read_bytes()
        manifest = _json_loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlDataError(f"cannot load frozen evidence manifest: {error}") from error
    required = {"schema_version", "id", "status", "source", "certificates", "dev_split", "soft_quality", "data"}
    if not isinstance(manifest, dict) or set(manifest) != required or manifest.get("status") != "frozen":
        raise ControlDataError("evidence manifest is not a frozen exact-schema artifact")
    selected_ids = [row.row_id for row in rows]
    source = manifest["source"]
    if source != {
        "sha256": PINNED_SOURCE["sha256"],
        "row_ids": selected_ids,
        "row_ids_sha256": program_contract._json_sha256({"row_ids": selected_ids}),
    }:
        raise ControlDataError("evidence manifest source rows are forged or reordered")
    certificates = manifest["certificates"]
    if not isinstance(certificates, dict) or set(certificates) != {
        "authoritative_evaluator",
        "evaluator_implementation",
        "judge_calibration",
        "leakage_detector",
    }:
        raise ControlDataError("evidence certificate linkage is incomplete")
    _validate_evidence_certificates(certificates, config, repo_root, judge_contract)
    dev_split = _validate_dev_split(manifest["dev_split"], repo_root)
    if set(selected_ids) & set(dev_split["row_ids"]):
        raise ControlDataError("selected control rows leak into the frozen dev split")
    scale = manifest["soft_quality"]
    if (
        not isinstance(scale, dict)
        or set(scale) != {"minimum", "maximum", "higher_is_better"}
        or not _finite_number(scale["minimum"])
        or not _finite_number(scale["maximum"])
        or scale["minimum"] >= scale["maximum"]
        or scale["higher_is_better"] is not True
    ):
        raise ControlDataError("soft-quality evidence has a missing or mixed scale")
    data = manifest["data"]
    if not isinstance(data, dict) or set(data) != {"path", "bytes", "sha256", "records"}:
        raise ControlDataError("evidence data identity is invalid")
    relative = data["path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ControlDataError("evidence data path must be relative to its manifest")
    data_path = (path.parent / relative).resolve()
    try:
        data_path.relative_to(path.parent.resolve())
        body = data_path.read_bytes()
    except (ValueError, OSError) as error:
        raise ControlDataError("evidence data is missing or escapes its manifest directory") from error
    if len(body) != data["bytes"] or hashlib.sha256(body).hexdigest() != data["sha256"]:
        raise ControlDataError("evidence data bytes do not match the frozen manifest")
    evidence: dict[tuple[int, str], dict[str, Any]] = {}
    row_map = {row.row_id: row for row in rows}
    try:
        lines = body.splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line:
                raise ControlDataError(f"blank evidence row at line {line_number}")
            value = _json_loads(line)
            key = (value.get("row_id"), value.get("output_sha256")) if isinstance(value, dict) else (None, None)
            source_row = row_map.get(key[0])
            if (
                key in evidence
                or key not in expected
                or source_row is None
                or not _valid_evidence(value, source_row, scale)
            ):
                raise ControlDataError(f"duplicate, unexpected, or incomplete evidence at line {line_number}")
            evidence[key] = value
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlDataError(f"cannot parse external evidence: {error}") from error
    if data["records"] != len(lines) or set(evidence) != expected:
        raise ControlDataError("external hard-pass or soft-quality evidence is missing")
    manifest["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return evidence, manifest, dev_split


def _validate_evidence_certificates(
    certificates: Mapping[str, Any],
    config: Mapping[str, Any],
    repo_root: Path,
    judge_contract: dict[str, Any],
) -> None:
    for name in ("authoritative_evaluator", "evaluator_implementation"):
        if certificates[name] != config["evidence"][name]:
            raise ControlDataError(f"{name} certificate pin is forged or changed")
        reference = certificates[name]
        _verified_json(_under(repo_root, reference["path"], name), reference["sha256"], reference["id"])
    judge = certificates["judge_calibration"]
    expected_judge = config["evidence"]["judge_calibration"]
    if (
        not isinstance(judge, dict)
        or set(judge) != {"path", "id", "sha256"}
        or judge.get("path") != expected_judge["path"]
        or judge.get("id") != expected_judge["id"]
    ):
        raise ControlDataError("judge calibration certificate linkage is forged")
    judge_path = _under(repo_root, judge["path"], "judge calibration")
    judge_artifact = _verified_json(judge_path, judge["sha256"], judge["id"])
    try:
        program_contract._validate_judge_calibration(
            judge_artifact,
            {"artifact_id": judge["id"], "sha256": judge["sha256"]},
            repo_root,
            judge_contract,
        )
    except program_contract.ProgramContractError as error:
        raise ControlDataError(f"judge calibration certificate is invalid: {error}") from error
    leakage = certificates["leakage_detector"]
    expected_leakage = config["evidence"]["leakage_detector"]
    if (
        not isinstance(leakage, dict)
        or set(leakage) != {"path", "id", "sha256"}
        or leakage.get("path") != expected_leakage["path"]
        or leakage.get("id") != expected_leakage["id"]
        or _path_identity(_under(repo_root, leakage["path"], "leakage detector"))["sha256"] != leakage["sha256"]
    ):
        raise ControlDataError("leakage detector certificate linkage is forged")


def _load_validated_judge_contract(repo_root: Path) -> dict[str, Any]:
    judge_path = _under(repo_root, "configs/judges/openrouter_luna.json", "judge config")
    try:
        judge = program_contract._load_json(judge_path)
        prompt_reference = judge.get("prompt")
        if not isinstance(prompt_reference, Mapping) or not isinstance(prompt_reference.get("path"), str):
            raise program_contract.ProgramContractError("judge prompt reference is invalid")
        prompt_path = _under(judge_path.parent, prompt_reference["path"], "judge prompt")
        prompt = prompt_path.read_text(encoding="utf-8")
        program_contract._validate_judge(judge, prompt)
    except (OSError, program_contract.ProgramContractError) as error:
        raise ControlDataError(f"judge contract is invalid: {error}") from error
    return judge


def _validate_dev_split(value: Any, repo_root: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "id", "sha256", "row_ids"}:
        raise ControlDataError("dev split linkage is incomplete")
    path = _under(repo_root, value["path"], "dev split")
    artifact = _verified_json(path, value["sha256"], value["id"])
    try:
        program_contract._validate_dev_split_artifact(
            {"manifest_id": value["id"], "sha256": value["sha256"]}, artifact, value["sha256"]
        )
    except program_contract.ProgramContractError as error:
        raise ControlDataError(f"dev split certificate is invalid: {error}") from error
    if value["row_ids"] != artifact["row_ids"]:
        raise ControlDataError("dev split row IDs differ from its frozen certificate")
    return value


def _valid_evidence(row: Mapping[str, Any], source: _SourceRow, scale: Mapping[str, Any]) -> bool:
    if set(row) != {"row_id", "source_sha256", "output_sha256", "rubrics", "hard", "soft", "leakage"}:
        return False
    rubrics = row.get("rubrics")
    expected_rubrics = _rubric_identities(source)
    hard = row.get("hard")
    soft = row.get("soft")
    leakage = row.get("leakage")
    hard_count = sum(source.hard_mask)
    soft_count = len(source.hard_mask) - hard_count
    hard_scores = hard.get("rubric_passes") if isinstance(hard, dict) else None
    soft_scores = soft.get("rubric_scores") if isinstance(soft, dict) else None
    quality = soft.get("quality") if isinstance(soft, dict) else None
    return (
        row.get("row_id") == source.row_id
        and row.get("source_sha256") == source.digest
        and isinstance(row.get("output_sha256"), str)
        and rubrics == expected_rubrics
        and isinstance(hard, dict)
        and set(hard) == {"pass", "rubric_passes"}
        and isinstance(hard_scores, list)
        and len(hard_scores) == hard_count
        and all(isinstance(value, bool) for value in hard_scores)
        and hard.get("pass") is all(hard_scores)
        and isinstance(soft, dict)
        and set(soft) == {"valid", "quality", "rubric_scores"}
        and soft.get("valid") is True
        and isinstance(soft_scores, list)
        and len(soft_scores) == soft_count
        and all(_in_scale(value, scale) for value in soft_scores)
        and _in_scale(quality, scale)
        and leakage == {"prompt": False, "reference": False}
    )


def _rubric_identities(row: _SourceRow) -> list[dict[str, Any]]:
    return [
        {
            "rubric_index": index,
            "kind": "hard" if hard else "soft",
            "criterion_sha256": hashlib.sha256(criterion.encode()).hexdigest(),
            "route": "embedded_check_following" if hard else None,
        }
        for index, (criterion, hard) in enumerate(zip(row.criteria, row.hard_mask, strict=True))
    ]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _in_scale(value: Any, scale: Mapping[str, Any]) -> bool:
    return _finite_number(value) and scale["minimum"] <= value <= scale["maximum"]


def _reject_leakage(row: _SourceRow, content: str, evidence: Mapping[str, Any]) -> None:
    prompt = _normalize(row.prompt.split("\n\nRequirements:\n", 1)[0])
    if prompt and prompt in _normalize(content):
        raise ControlDataError(f"row {row.row_id}: output contains prompt leakage")
    if evidence["leakage"]["prompt"] or evidence["leakage"]["reference"]:
        raise ControlDataError(f"row {row.row_id}: external evidence reports prompt or reference leakage")


def _preference_key(output: Mapping[str, str], evidence: Mapping[str, Any]) -> tuple[int, float, str]:
    return (-int(evidence["hard"]["pass"]), -float(evidence["soft"]["quality"]), output["sha256"])


def _stage_identity(
    stage: str,
    source: Mapping[str, Any],
    selected: list[_SourceRow],
    config_hash: str,
    request: Mapping[str, Any],
    command: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "source": dict(source),
        "selection": {
            "records": len(selected),
            "row_ids_sha256": _digest([row.row_id for row in selected]),
            "positions_sha256": _digest([row.position for row in selected]),
        },
        "selected_positions": [row.position for row in selected],
        "config_sha256": config_hash,
        "request": dict(request),
        "implementation_sha256": _implementation_sha(),
        "command": _command_identity(command),
    }


def _matching_stage_identities(
    teacher: Mapping[str, Any],
    candidates: Mapping[str, Any],
    source: Mapping[str, Any],
    config_hash: str,
    config: Mapping[str, Any],
) -> None:
    teacher_request = teacher.get("request")
    candidate_request = candidates.get("request")
    if (
        teacher.get("stage") != "teacher"
        or candidates.get("stage") != "candidates"
        or teacher.get("source") != source
        or candidates.get("source") != source
        or teacher.get("selection") != candidates.get("selection")
        or teacher.get("config_sha256") != config_hash
        or candidates.get("config_sha256") != config_hash
        or teacher.get("implementation_sha256") != _implementation_sha()
        or candidates.get("implementation_sha256") != _implementation_sha()
        or not isinstance(teacher_request, dict)
        or teacher_request.get("model") != config["teacher"]["model"]
        or teacher_request.get("revision") != config["teacher"]["revision"]
        or teacher_request.get("response_models") != config["teacher"]["response_models"]
        or teacher_request.get("request") != config["teacher"]["request"]
        or teacher_request.get("retry") != config["teacher"]["retry"]
        or teacher_request.get("timeout_seconds") != config["teacher"]["timeout_seconds"]
        or not isinstance(candidate_request, dict)
        or candidate_request.get("model") != config["candidates"]["model"]
        or candidate_request.get("revision") != config["candidates"]["revision"]
        or candidate_request.get("served_name") != config["candidates"]["served_name"]
        or candidate_request.get("request") != config["candidates"]["request"]
        or candidate_request.get("timeout_seconds") != config["candidates"]["timeout_seconds"]
    ):
        raise ControlDataError("teacher and candidate stage identities are inconsistent")


def _selected_from_identity(rows: list[_SourceRow], identity: Mapping[str, Any]) -> list[_SourceRow]:
    selection = identity.get("selection")
    if not isinstance(selection, dict):
        raise ControlDataError("stage selection identity is missing")
    count = selection.get("records")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= len(rows):
        raise ControlDataError("stage selection count is invalid")
    matches = [
        rows[index : index + count]
        for index in range(len(rows) - count + 1)
        if _digest([row.row_id for row in rows[index : index + count]]) == selection.get("row_ids_sha256")
        and _digest([row.position for row in rows[index : index + count]]) == selection.get("positions_sha256")
    ]
    if len(matches) != 1:
        selected = [row for row in rows if row.position in _positions_from_raw_identity(identity)]
        if (
            len(selected) != count
            or _digest([row.row_id for row in selected]) != selection.get("row_ids_sha256")
            or _digest([row.position for row in selected]) != selection.get("positions_sha256")
        ):
            raise ControlDataError("stage selection cannot be reconstructed")
        return selected
    return matches[0]


def _positions_from_raw_identity(identity: Mapping[str, Any]) -> set[int]:
    positions = identity.get("selected_positions")
    if not isinstance(positions, list) or not all(isinstance(value, int) for value in positions):
        return set()
    return set(positions)


def _baseline_manifest(
    name: str,
    common: Mapping[str, Any],
    repo_root: Path,
    path: Path,
    body: bytes,
    rows: list[dict[str, Any]],
    dev_split: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ControlDataError(f"frozen {name} data must be under the repository root") from error
    fields = ["row_id", "prompt", "output"] if name == "sft" else ["row_id", "prompt", "chosen", "rejected"]
    row_ids = [row["row_id"] for row in rows]
    data: dict[str, Any] = {
        "path": relative,
        "sha256": hashlib.sha256(body).hexdigest(),
        "schema": {"format": "jsonl", "fields": fields},
        "records": len(rows),
        "row_ids": row_ids,
        "row_ids_sha256": program_contract._json_sha256({"row_ids": row_ids}),
        "output_digests": [program_contract._json_sha256(row) for row in rows],
    }
    if name == "dpo":
        data["pairs"] = [
            {
                "row_id": row["row_id"],
                "chosen_sha256": hashlib.sha256(row["chosen"].encode()).hexdigest(),
                "rejected_sha256": hashlib.sha256(row["rejected"].encode()).hexdigest(),
            }
            for row in rows
        ]
    return {
        "schema_version": 1,
        "id": f"{name}_data_v1-{common['evidence_sha256']}",
        "baseline_id": f"{name}_reconstructed",
        "source_sha256": config["source_sha256"],
        "dev_split_manifest": {
            "id": dev_split["id"],
            "sha256": dev_split["sha256"],
            "row_ids": dev_split["row_ids"],
        },
        "data": data,
        "teacher": common["teacher"],
    }


def _path_identity(path: Path) -> dict[str, Any]:
    try:
        body = path.read_bytes()
    except OSError as error:
        raise ControlDataError(f"cannot hash raw input: {error}") from error
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _stage_summary(path: Path, attempts: Mapping[int, list[dict[str, Any]]], count: int) -> StageSummary:
    latest = [records[-1] for records in attempts.values() if records]
    return StageSummary(
        selected_rows=count,
        successful_rows=sum(record["status"] == "ok" for record in latest),
        failed_rows=sum(record["status"] == "failure" for record in latest),
        recorded_failures=sum(record["status"] == "failure" for rows in attempts.values() for record in rows),
        raw_sha256=_path_identity(path)["sha256"] if path.exists() else hashlib.sha256(b"").hexdigest(),
    )


def _read_identity(path: Path) -> dict[str, Any]:
    identity_path = _identity_path(path)
    try:
        value = _json_loads(identity_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlDataError(f"cannot load stage identity: {error}") from error
    if not isinstance(value, dict):
        raise ControlDataError("stage identity must be a JSON object")
    return value


def _ensure_identity(path: Path, identity: Mapping[str, Any]) -> None:
    identity_path = _identity_path(path)
    if identity_path.exists():
        if _read_identity(path) != identity:
            raise ControlDataError("existing stage identity does not match this invocation")
        return
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(identity_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if _read_identity(path) != identity:
            raise ControlDataError("existing stage identity does not match this invocation")
        return
    with os.fdopen(descriptor, "wb") as file:
        file.write(_pretty_json_bytes(identity))
        file.flush()
        os.fsync(file.fileno())


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControlDataError(f"another process owns the stage lock: {lock_path}") from error
        yield


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    initial_size = os.fstat(descriptor).st_size
    try:
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("JSONL append made no progress")
            written += count
        os.fsync(descriptor)
    except BaseException:
        os.ftruncate(descriptor, initial_size)
        os.fsync(descriptor)
        raise
    finally:
        os.close(descriptor)


def _publish_immutable(payloads: Mapping[Path, bytes]) -> None:
    for path in payloads:
        if path.exists() or path.is_symlink():
            raise ControlDataError(f"freeze output already exists: {path}")
    temporary: list[Path] = []
    complete = False
    try:
        for path, body in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
            descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as file:
                file.write(body)
                file.flush()
                os.fsync(file.fileno())
            temporary.append(temp)
        for temp, path in zip(temporary, payloads, strict=True):
            os.link(temp, path)
        complete = True
    except OSError as error:
        raise ControlDataError(f"cannot publish immutable freeze outputs: {error}") from error
    finally:
        if not complete:
            for temp, path in reversed(list(zip(temporary, payloads))):
                try:
                    path_stat = path.stat()
                    temp_stat = temp.stat()
                    if not path.is_symlink() and (path_stat.st_dev, path_stat.st_ino) == (
                        temp_stat.st_dev,
                        temp_stat.st_ino,
                    ):
                        path.unlink()
                except FileNotFoundError:
                    pass
        for temp in temporary:
            temp.unlink(missing_ok=True)


def _require_raw_path(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    for parent in (resolved.parent, *resolved.parent.parents):
        if (parent / ".git").exists():
            raise ControlDataError(f"raw artifacts must be outside Git: {resolved}")


def _http_url(value: str, *, exact_path: str | None = None) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or (exact_path is not None and parts.path.rstrip("/") != exact_path.rstrip("/"))
    ):
        raise ControlDataError("endpoint must be an HTTP URL without credentials, query, or fragment")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _identity_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.identity.json")


def _command_identity(command: Sequence[str] | None) -> dict[str, Any]:
    values = list(command or sys.argv)
    return {"program": Path(values[0]).name if values else None, "argv_sha256": _digest(values)}


def _implementation_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_loads(value: str | bytes) -> Any:
    return json.loads(value, object_pairs_hook=_unique_object)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n" for row in rows)


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
