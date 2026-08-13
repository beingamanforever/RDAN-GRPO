"""Deterministic whole-row partitioning for the scalar HIR dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from rdan_grpo.hir import classify_hir_row

MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOKENIZER_JSON_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_CONFIG_SHA256 = "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
TEMPLATE_IMPLEMENTATION_SHA256 = "ceeb36b3c17ffa55c6f82f19aeae75684ca5c4625cebec84cb90e35ea88ecacc"
TOKENIZER_CERTIFICATE_ID = "hir_qwen3_4b_instruct_2507_tokenizer_v1"


class ScalarDataError(ValueError):
    """Raised when pinned HIR inputs or their partition are inconsistent."""


@dataclass(frozen=True)
class ScalarGate:
    """Evidence derived from the exact scalar train configuration and data."""

    manifest: dict[str, Any]
    implemented: tuple[dict[str, Any], ...]
    excluded: tuple[dict[str, Any], ...]
    function_hashes: frozenset[str]


@dataclass(frozen=True)
class HirTokenizerGate:
    """Frozen Qwen eligibility for every processed HIR row."""

    certificate: dict[str, Any]
    accepted_row_ids: frozenset[int | str]
    rejected_rows: tuple[dict[str, Any], ...]


def verify_hir_tokenizer_gate(
    certificate_path: str | Path,
    source_ref_path: str | Path,
    root: str | Path | None = None,
) -> HirTokenizerGate:
    """Verify the full HIR Qwen token evidence and return accepted row IDs."""

    repo_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    certificate_path = Path(certificate_path).resolve()
    certificate = _json(certificate_path)
    if (
        certificate.get("schema_version") != 1
        or certificate.get("id") != TOKENIZER_CERTIFICATE_ID
        or certificate.get("status") != "frozen"
    ):
        raise ScalarDataError("HIR tokenizer certificate status is invalid")
    source_path = _under(repo_root, repo_root / source_ref_path, "HIR tokenizer source")
    source_ref = certificate.get("source")
    if not isinstance(source_ref, dict) or source_ref != _file_reference(source_path, len(_rows(source_path))):
        raise ScalarDataError("HIR tokenizer source identity changed")
    evidence_ref = certificate.get("evidence")
    if not isinstance(evidence_ref, dict):
        raise ScalarDataError("HIR tokenizer evidence reference is missing")
    evidence_path = _under(repo_root, repo_root / str(evidence_ref.get("path")), "HIR tokenizer evidence")
    evidence = _jsonl(evidence_path)
    if evidence_ref != _file_reference(evidence_path, len(evidence)):
        raise ScalarDataError("HIR tokenizer evidence was tampered")
    sources = list(_rows(source_path).values())
    if len(sources) != 16_968 or len(evidence) != len(sources):
        raise ScalarDataError("HIR tokenizer source or evidence count changed")
    accepted = []
    rejected = []
    for source, item in zip(sources, evidence, strict=True):
        row_id = source["id"]
        tokens = item.get("input_tokens")
        is_accepted = item.get("accepted")
        expected_acceptance = isinstance(tokens, int) and not isinstance(tokens, bool) and 5 < tokens <= 2_048
        if (
            item.get("row_id") != row_id
            or item.get("source") != source.get("source")
            or item.get("messages_sha256") != _digest(source.get("messages"))
            or not isinstance(is_accepted, bool)
            or is_accepted != expected_acceptance
        ):
            raise ScalarDataError(f"row {row_id!r}: HIR tokenizer evidence changed")
        (accepted if is_accepted else rejected).append(row_id if is_accepted else item)
    results = certificate.get("results")
    if not isinstance(results, dict) or (
        results.get("input_rows") != len(sources)
        or results.get("input_row_ids_sha256") != _digest([row["id"] for row in sources])
        or results.get("accepted") != len(accepted)
        or results.get("accepted_row_ids_sha256") != _digest(accepted)
        or results.get("rejected") != len(rejected)
        or results.get("rejected_rows") != rejected
        or len(accepted) != 16_962
        or len(rejected) != 6
    ):
        raise ScalarDataError("HIR tokenizer result inventory changed")
    return HirTokenizerGate(certificate, frozenset(accepted), tuple(rejected))


def inspect_scalar_gate(
    repo_root: str | Path,
    train_config_path: str | Path,
    certified_manifest_path: str | Path,
) -> ScalarGate:
    """Recompute the safe scalar partition selected by a ROLL train config."""

    root = Path(repo_root).resolve()
    train_path = Path(train_config_path).resolve()
    certified_path = Path(certified_manifest_path).resolve()
    certified = verify_scalar_dataset(certified_path, root)
    train = _yaml(train_path)
    try:
        names = train["actor_train"]["data_args"]["file_name"]
    except (KeyError, TypeError) as error:
        raise ScalarDataError("train config is missing actor_train.data_args.file_name") from error
    if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
        raise ScalarDataError("train config must select exactly one scalar JSONL")
    data_path = _under(root, root / names[0], "train dataset")
    derived = certified.get("derived")
    if not isinstance(derived, dict) or derived.get("path") != names[0]:
        raise ScalarDataError("train config dataset does not match the certified subset manifest")
    source_ref = certified.get("sources", {}).get("hir")
    if not isinstance(source_ref, dict):
        raise ScalarDataError("certified subset manifest has no HIR source")
    source_path = _under(root, root / str(source_ref.get("path")), "HIR source")
    source_rows = _rows(source_path)
    train_rows = _rows(data_path)
    selected = set(train_rows)
    if not selected or not selected <= set(source_rows):
        raise ScalarDataError("scalar dataset contains missing or no source rows")
    prompt_limit = train.get("prompt_length")
    if prompt_limit != 2_048:
        raise ScalarDataError("train prompt length differs from the measured ROLL preprocessing gate")
    tokenizer_ref = certified.get("sources", {}).get("tokenizer_certificate")
    if not isinstance(tokenizer_ref, dict):
        raise ScalarDataError("certified subset manifest has no tokenizer certificate")
    tokenizer_path = _under(root, root / str(tokenizer_ref.get("path")), "HIR tokenizer certificate")
    processed_ref = certified.get("sources", {}).get("rtt_processed")
    if not isinstance(processed_ref, dict):
        raise ScalarDataError("certified subset manifest has no RTT processed source")
    tokenizer = verify_hir_tokenizer_gate(tokenizer_path, source_ref_path=Path(processed_ref["path"]), root=root)
    if selected & {row["row_id"] for row in tokenizer.rejected_rows}:
        raise ScalarDataError("scalar dataset contains a Qwen-ineligible HIR row")
    template_path = root / "scripts/run_roll_parity.py"
    template_sha256 = _sha256(template_path)
    if template_sha256 != TEMPLATE_IMPLEMENTATION_SHA256:
        raise ScalarDataError("measured qwen3_nothinking preprocessing implementation changed")

    implemented: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    function_hashes: set[str] = set()
    for row_id, source in source_rows.items():
        hard_mask = classify_hir_row(source)
        target = implemented if row_id in selected else excluded
        if row_id in selected:
            _validate_train_row(train_rows[row_id], source, hard_mask)
        for index, hard in enumerate(hard_mask):
            if not hard:
                continue
            target.append(
                {
                    "source": source["source"],
                    "row_id": row_id,
                    "rubric_index": index,
                    "route": _route(source, index),
                }
            )
            if row_id in selected and source["source"] == "type4":
                function_hashes.add(hashlib.sha256(source["ground_truth"]["functions"][index].encode()).hexdigest())
    implemented.sort(key=_canonical)
    excluded.sort(key=_canonical)
    row_ids = list(train_rows)
    counts = certified.get("counts")
    if not isinstance(counts, dict):
        raise ScalarDataError("certified subset partition metadata is missing")
    if (
        counts.get("included_rows") != len(row_ids)
        or counts.get("retained_hard_identities") != len(implemented)
        or counts.get("excluded_hard_identities") != len(excluded)
        or counts.get("all_hard_identities") != len(implemented) + len(excluded)
        or counts.get("candidate_rows") != len(row_ids) + counts.get("tokenizer_excluded_rows", -1)
    ):
        raise ScalarDataError("certified input subset partition counts differ")

    candidate_sources = {source_rows[row_id]["source"] for row_id in selected}
    if candidate_sources != {"type4"}:
        raise ScalarDataError("certified scalar candidates must remain the Type-4 partition")
    candidate_rejections = [
        row
        for row in tokenizer.rejected_rows
        if row["row_id"] not in selected and source_rows[row["row_id"]]["source"] in candidate_sources
    ]
    if len(candidate_rejections) != counts.get("tokenizer_excluded_rows"):
        raise ScalarDataError("certified scalar tokenizer exclusions differ")
    source_counts = {source: 0 for source in ("type1", "type2", "type3", "type4")}
    excluded_counts = dict(source_counts)
    for identity in implemented:
        source_counts[identity["source"]] += 1
    for identity in excluded:
        excluded_counts[identity["source"]] += 1

    manifest = {
        "schema_version": 1,
        "id": "qwen_scalar_data_v1",
        "status": "frozen",
        "train_config": {
            "path": _relative(root, train_path),
            "sha256": _sha256(train_path),
            "dataset_path": names[0],
        },
        "certified_subset_manifest": {
            "path": _relative(root, certified_path),
            "id": certified.get("id"),
            "sha256": _sha256(certified_path),
        },
        "data": {
            "path": names[0],
            "format": "jsonl",
            "schema": derived.get("schema"),
            "bytes": data_path.stat().st_size,
            "sha256": _sha256(data_path),
            "records": len(row_ids),
            "row_ids_sha256": _digest(row_ids),
        },
        "preprocessing": {
            "implementation": {
                "path": _relative(root, template_path),
                "sha256": template_sha256,
                "template": "qwen3_nothinking",
                "apply_chat_template": {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
                "tokenizer_add_special_tokens": True,
            },
            "model_revision": MODEL_REVISION,
            "tokenizer_files_sha256": {
                "tokenizer.json": TOKENIZER_JSON_SHA256,
                "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
            },
            "criterion": "5 < len(input_ids) <= prompt_length",
            "prompt_length": prompt_limit,
            "certificate": {
                "path": _relative(root, tokenizer_path),
                "id": tokenizer.certificate["id"],
                "sha256": _sha256(tokenizer_path),
            },
            "source_records": tokenizer.certificate["results"]["input_rows"],
            "candidate_records": counts["candidate_rows"],
            "effective_records": len(row_ids),
            "effective_row_ids_sha256": _digest(row_ids),
            "excluded": candidate_rejections,
            "largest_included_input_tokens": tokenizer.certificate["results"]["largest_accepted_input_tokens"],
        },
        "scope": {
            "sources": ["type1", "type2", "type3", "type4"],
            "requires_supported_hard_route": True,
            "requires_hard_criterion": True,
            "requires_certified_type4_function": True,
            "implemented_hard_identities": len(implemented),
            "implemented_hard_identities_by_source": source_counts,
            "excluded_hard_identities": len(excluded),
            "excluded_hard_identities_by_source": excluded_counts,
            "function_hashes": len(function_hashes),
        },
    }
    return ScalarGate(manifest, tuple(implemented), tuple(excluded), frozenset(function_hashes))


def build_scalar_dataset(
    source_path: str | Path,
    processed_path: str | Path,
    resolution_path: str | Path,
    output_path: str | Path,
    certified_families: set[tuple[str, str]] | None = None,
    excluded_type4_hashes: set[str] | None = None,
    accepted_row_ids: set[int | str] | frozenset[int | str] | None = None,
    certification_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Write the frozen scalar JSONL and return its compact integrity manifest."""

    source_path, processed_path = Path(source_path), Path(processed_path)
    resolution_path, output_path = Path(resolution_path), Path(output_path)
    resolution = _json(resolution_path)
    unsupported = resolution.get("unsupported_identities")
    if not isinstance(unsupported, list) or len(unsupported) != 799:
        raise ScalarDataError("route resolution must contain exactly 799 unsupported identities")
    unsupported_rows = {identity["row_id"] for identity in unsupported}
    source_rows = _rows(source_path)
    processed_rows = _rows(processed_path)
    if list(source_rows) != list(processed_rows):
        raise ScalarDataError("source and processed row IDs or order differ")

    included: list[int | str] = []
    excluded: list[int | str] = []
    retained_identities: list[dict[str, Any]] = []
    excluded_identities: list[dict[str, Any]] = []
    encoded_rows: list[bytes] = []
    soft_only_rows = 0
    uncertified_rows = 0
    candidate_rows = 0
    tokenizer_excluded_rows = 0
    for row_id, source in source_rows.items():
        hard_mask = classify_hir_row(source)
        soft_only = not any(hard_mask)
        routes = [_route(source, index) for index, hard in enumerate(hard_mask) if hard]
        uncertified = certified_families is not None and any(
            (source["source"], route) not in certified_families for route in routes
        )
        unsafe_type4 = source["source"] == "type4" and any(
            hard and hashlib.sha256(code.encode()).hexdigest() in (excluded_type4_hashes or set())
            for hard, code in zip(hard_mask, source["ground_truth"]["functions"], strict=True)
        )
        authority_excluded = row_id in unsupported_rows or soft_only or uncertified or unsafe_type4
        candidate_rows += int(not authority_excluded)
        tokenizer_excluded = not authority_excluded and accepted_row_ids is not None and row_id not in accepted_row_ids
        tokenizer_excluded_rows += int(tokenizer_excluded)
        remove = authority_excluded or tokenizer_excluded
        soft_only_rows += int(soft_only)
        uncertified_rows += int(uncertified and row_id not in unsupported_rows and not soft_only)
        (excluded if remove else included).append(row_id)
        target = excluded_identities if remove else retained_identities
        for rubric_index, hard in enumerate(hard_mask):
            if hard:
                target.append(
                    {
                        "source": source["source"],
                        "row_id": row_id,
                        "rubric_index": rubric_index,
                        "route": _route(source, rubric_index),
                    }
                )
        if not remove:
            processed = processed_rows[row_id]
            if processed.get("source") != source.get("source"):
                raise ScalarDataError(f"row {row_id!r}: processed source differs")
            if processed.get("ground_truth") != source.get("ground_truth"):
                raise ScalarDataError(f"row {row_id!r}: processed ground truth differs")
            if [rubric.get("description") for rubric in processed.get("rubrics", [])] != source.get("criteria"):
                raise ScalarDataError(f"row {row_id!r}: processed rubrics differ")
            encoded_rows.append((json.dumps(processed, ensure_ascii=False, separators=(",", ":")) + "\n").encode())

    included_sorted = _sorted_ids(included)
    excluded_sorted = _sorted_ids(excluded)
    retained_identities.sort(key=_canonical)
    excluded_identities.sort(key=_canonical)
    if certified_families is None:
        expected = (
            (15_675, 1_293, 15_675, 0, 72_296, 4_160)
            if excluded_type4_hashes is None
            else (15_656, 1_312, 15_656, 0, 72_238, 4_218)
            if accepted_row_ids is None
            else (15_650, 1_318, 15_656, 6, 72_207, 4_249)
        )
        actual = (
            len(included),
            len(excluded),
            candidate_rows,
            tokenizer_excluded_rows,
            len(retained_identities),
            len(excluded_identities),
        )
        if expected is not None and actual != expected:
            raise ScalarDataError(f"scalar partition counts differ from the frozen contract: {actual}")
    body = b"".join(encoded_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(body)
    return {
        "schema_version": 1,
        "id": "hir_trainv1_rdan_scalar_v1",
        "status": "frozen",
        "sources": {
            "hir": {
                "path": _display_path(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            },
            "rtt_processed": {
                "path": _display_path(processed_path),
                "bytes": processed_path.stat().st_size,
                "sha256": _sha256(processed_path),
            },
            "route_resolution": {"path": _display_path(resolution_path), "sha256": _sha256(resolution_path)},
            **{
                name: {"path": _display_path(path), "sha256": _sha256(path)}
                for name, path in (certification_paths or {}).items()
            },
        },
        "derived": {
            "path": _display_path(output_path),
            "format": "jsonl",
            "schema": sorted(next(iter(processed_rows.values())).keys()),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        },
        "counts": {
            "source_rows": len(source_rows),
            "candidate_rows": candidate_rows,
            "included_rows": len(included),
            "excluded_rows": len(excluded),
            "soft_only_rows": soft_only_rows,
            "uncertified_family_rows": uncertified_rows,
            "unsafe_type4_rows": sum(
                row_id not in included
                and row_id not in unsupported_rows
                and any(classify_hir_row(source))
                and source["source"] == "type4"
                for row_id, source in source_rows.items()
                if excluded_type4_hashes
                and source["source"] == "type4"
                and any(
                    hard and hashlib.sha256(code.encode()).hexdigest() in excluded_type4_hashes
                    for hard, code in zip(
                        classify_hir_row(source), source["ground_truth"].get("functions", []), strict=True
                    )
                )
            ),
            "tokenizer_excluded_rows": tokenizer_excluded_rows,
            "unsupported_hard_identities": len(unsupported),
            "retained_hard_identities": len(retained_identities),
            "excluded_hard_identities": len(excluded_identities),
            "all_hard_identities": len(retained_identities) + len(excluded_identities),
        },
        "row_ids": {
            "included_sha256": _digest(included_sorted),
            "excluded_sha256": _digest(excluded_sorted),
            "preflight_first_256": included[:256],
            "preflight_first_256_sha256": _digest(included[:256]),
        },
        "hard_identities": {
            "retained_sha256": _digest(retained_identities),
            "excluded_sha256": _digest(excluded_identities),
            "partition_sha256": _digest(retained_identities + excluded_identities),
        },
    }


def build_rtt_hir_dataset(
    processed_path: str | Path,
    output_path: str | Path,
    tokenizer_gate: HirTokenizerGate,
    certificate_path: str | Path,
) -> dict[str, Any]:
    """Write the full RTT-compatible HIR corpus after the frozen Qwen token gate."""

    processed_path, output_path = Path(processed_path), Path(output_path)
    rows = _rows(processed_path)
    if len(rows) != 16_968 or set(rows) != tokenizer_gate.accepted_row_ids | {
        row["row_id"] for row in tokenizer_gate.rejected_rows
    }:
        raise ScalarDataError("full RTT HIR inventory differs from the tokenizer certificate")
    encoded = [
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row_id, row in rows.items()
        if row_id in tokenizer_gate.accepted_row_ids
    ]
    body = b"".join(encoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(body)
    accepted_ids = [row_id for row_id in rows if row_id in tokenizer_gate.accepted_row_ids]
    if len(accepted_ids) != 16_962:
        raise ScalarDataError("full RTT HIR effective count differs from the frozen contract")
    return {
        "schema_version": 1,
        "id": "qwen_rtt_hir_data_v1",
        "status": "frozen",
        "scope": "full_rtt_compatible_not_authoritative",
        "sources": {
            "rtt_processed": _file_reference(processed_path, len(rows)),
            "tokenizer_certificate": {
                "path": _display_path(Path(certificate_path)),
                "sha256": _sha256(Path(certificate_path)),
                "id": tokenizer_gate.certificate["id"],
            },
        },
        "derived": {
            "path": _display_path(output_path),
            "format": "jsonl",
            "schema": sorted(next(iter(rows.values())).keys()),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "records": len(accepted_ids),
        },
        "counts": {
            "processed_rows": len(rows),
            "effective_rows": len(accepted_ids),
            "tokenizer_excluded_rows": len(tokenizer_gate.rejected_rows),
        },
        "row_ids": {
            "effective_sha256": _digest(accepted_ids),
            "excluded": list(tokenizer_gate.rejected_rows),
        },
    }


def verify_scalar_dataset(manifest_path: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load a compact manifest and reject any missing or tampered linked file."""

    manifest = _json(Path(manifest_path))
    if manifest.get("status") != "frozen" or manifest.get("id") != "hir_trainv1_rdan_scalar_v1":
        raise ScalarDataError("scalar manifest is not frozen")
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    for source in manifest["sources"].values():
        path = _under(root, root / source["path"], "scalar source")
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ScalarDataError(f"scalar source is missing or tampered: {path}")
    derived = manifest["derived"]
    path = _under(root, root / derived["path"], "derived scalar data")
    if not path.is_file() or path.stat().st_size != derived["bytes"] or _sha256(path) != derived["sha256"]:
        raise ScalarDataError("derived scalar data is missing or tampered")
    return manifest


def _validate_train_row(row: dict[str, Any], source: dict[str, Any], hard_mask: tuple[bool, ...]) -> None:
    if source.get("source") not in {"type1", "type2", "type3", "type4"} or not any(hard_mask):
        raise ScalarDataError(f"row {source.get('id')!r}: scalar gate requires a hard HIR rubric")
    if row.get("source") != source.get("source") or row.get("ground_truth") != source.get("ground_truth"):
        raise ScalarDataError(f"row {source.get('id')!r}: train data differs from source truth")
    if [rubric.get("description") for rubric in row.get("rubrics", [])] != source.get("criteria"):
        raise ScalarDataError(f"row {source.get('id')!r}: train rubrics differ from source criteria")


def _yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ScalarDataError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScalarDataError(f"{path} must contain a YAML object")
    return value


def _under(root: Path, path: Path, name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ScalarDataError(f"{name} escapes the repository root") from error
    return resolved


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError as error:
        raise ScalarDataError("scalar gate input escapes the repository root") from error


def _rows(path: Path) -> dict[int | str, dict[str, Any]]:
    rows: dict[int | str, dict[str, Any]] = {}
    try:
        lines = path.open(encoding="utf-8")
    except OSError as error:
        raise ScalarDataError(f"cannot read {path}: {error}") from error
    with lines:
        for line_number, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ScalarDataError(f"{path}:{line_number}: invalid JSON") from error
            row_id = row.get("id") if isinstance(row, dict) else None
            if isinstance(row_id, bool) or not isinstance(row_id, (int, str)) or row_id in rows:
                raise ScalarDataError(f"{path}:{line_number}: invalid or duplicate ID")
            rows[row_id] = row
    return rows


def _route(row: dict[str, Any], index: int) -> str:
    source, truth = row["source"], row["ground_truth"]
    if source in {"type1", "type2"}:
        return truth["instruction_id_list"][index]
    if source == "type3":
        return "_".join(truth["constraints"][index][:2])
    return "embedded_check_following"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScalarDataError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScalarDataError(f"{path} must contain an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ScalarDataError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise ScalarDataError(f"cannot load JSONL: {path}") from error
    return rows


def _file_reference(path: Path, records: int) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sorted_ids(values: Iterable[int | str]) -> list[int | str]:
    return sorted(values, key=_canonical)


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
