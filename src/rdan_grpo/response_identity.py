"""Immutable source identity for response-only training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ResponseIdentityError(ValueError):
    """Raised when response-training inputs differ from their frozen identity."""


_CONTAMINATION_POLICY = "all_training_fields_exact_casefold_word_5gram_jaccard_v1"
_BENCHMARK_FIELDS = {
    "IFEval": ["prompt"],
    "IFBench": ["prompt"],
    "MulDimIF": ["prompt"],
    "AdvancedIF": ["message_content", "role_preserving_transcript", "rubric"],
}


def response_source_hashes(
    source_dir: str | Path,
    *,
    evaluator_rows: str | Path,
    train_config: str | Path,
    preflight_config: str | Path,
    program: str | Path,
    train_resolved_config_sha256: str,
    preflight_resolved_config_sha256: str,
) -> dict[str, str]:
    """Hash the exact code, configs, data, and lifecycle evidence for preflight."""

    source = Path(source_dir).resolve()
    root = source.parents[1]
    files = _response_source_files(
        source,
        root,
        evaluator_rows=evaluator_rows,
        train_config=train_config,
        preflight_config=preflight_config,
    )
    for label, digest in (
        ("production resolved config", train_resolved_config_sha256),
        ("preflight resolved config", preflight_resolved_config_sha256),
    ):
        if not _sha256(digest):
            raise ResponseIdentityError(f"{label} hash is invalid")
    return {
        **{name: file_sha256(path) for name, path in files.items()},
        "train_resolved_config": train_resolved_config_sha256,
        "preflight_resolved_config": preflight_resolved_config_sha256,
        **lifecycle_source_hashes(program),
    }


def _response_source_files(
    source: Path,
    root: Path,
    *,
    evaluator_rows: str | Path,
    train_config: str | Path,
    preflight_config: str | Path,
) -> dict[str, Path]:
    return {
        "advantages": source / "advantages.py",
        "evaluator_cert": source / "evaluator_cert.py",
        "evaluator_rows": Path(evaluator_rows),
        "fsdp_hf_receipt": source / "fsdp_hf_receipt.py",
        "hir": source / "hir.py",
        "judge": source / "judge.py",
        "program": source / "program.py",
        "preflight_cli": root / "scripts" / "run_roll_preflight.py",
        "preflight_config": Path(preflight_config),
        "readiness_cli": root / "scripts" / "run_response_readiness.py",
        "response_checkpoint": source / "roll_response_checkpoint.py",
        "response_config": source / "roll_response_config.py",
        "response_identity": source / "response_identity.py",
        "response_pilot_lifecycle": source / "response_pilot_lifecycle.py",
        "response_pipeline": source / "roll_response_pipeline.py",
        "response_readiness": source / "response_readiness.py",
        "response_receipt": source / "roll_response_receipt.py",
        "response_sampling": source / "response_sampling.py",
        "response_train": source / "roll_response_train.py",
        "response_workers": source / "roll_response_workers.py",
        "rewards": source / "rewards.py",
        "rubrichub_rules": source / "rubrichub_rules.py",
        "roll_bridge": source / "roll_bridge.py",
        "roll_compat": source / "roll_compat.py",
        "roll_fsdp_hf_receipt": source / "roll_fsdp_hf_receipt.py",
        "roll_live": source / "roll_live.py",
        "roll_reward": source / "roll_reward.py",
        "roll_same_backend": source / "roll_same_backend.py",
        "roll_same_backend_live": source / "roll_same_backend_live.py",
        "roll_scalar": source / "roll_scalar.py",
        "roll_weight_receipt": source / "roll_weight_receipt.py",
        "runtime_parity": source / "runtime_parity.py",
        "safe_rule": source / "safe_rule.py",
        "same_backend_cli": root / "scripts" / "run_same_backend_parity.py",
        "scalar_data": source / "scalar_data.py",
        "train_cli": root / "scripts" / "run_response_train.py",
        "train_config": Path(train_config),
        "wandb_tracking": source / "wandb_tracking.py",
        "weight_receipt": source / "weight_receipt.py",
    }


def clean_repository_revision(path: str | Path, expected: str | None = None) -> str:
    """Return HEAD only when the containing Git worktree is clean and exact."""

    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    if status:
        raise ResponseIdentityError("response evidence requires a clean Git worktree")
    revision = _git(root, "rev-parse", "HEAD")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ResponseIdentityError("response evidence cannot resolve an exact Git revision")
    if expected is not None and revision != expected:
        raise ResponseIdentityError("response evidence revision differs from the clean checkout")
    return revision


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ResponseIdentityError("response evidence path is not inside a Git worktree")
    return result.stdout.strip()


def canonical_resolved_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash one fully composed and resolved Hydra config canonically."""

    try:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ResponseIdentityError(f"resolved config is not canonical JSON: {error}") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lifecycle_source_hashes(program_path: str | Path) -> dict[str, str]:
    """Hash all frozen lifecycle inputs that authorize a response run."""

    path = Path(program_path).resolve()
    root = _repo_root(path)
    program = _json_object(path, "experiment program")
    try:
        references = {
            "scalar_data_manifest": program["lifecycle_artifacts"]["scalar_data"],
            "evaluator_certificate": program["hard_route_policy"]["evaluator_certificate"],
            "judge_calibration": program["lifecycle_artifacts"]["judge_calibration"],
            "runtime_parity": program["lifecycle_artifacts"]["runtime_parity"],
        }
    except (KeyError, TypeError) as error:
        raise ResponseIdentityError(f"invalid experiment program: {error}") from error
    hashes = {name: _frozen_reference(root, reference, name)[1] for name, reference in references.items()}
    data = response_data_identity(path)
    return hashes | {
        "response_data_manifest": data["manifest_sha256"],
        "response_data_output": data["output_sha256"],
        "response_data_config": data["config_sha256"],
        "response_hir_manifest": data["hir_manifest_sha256"],
        "rubrichub_rule_certificate": data["rule_certificate_sha256"],
        "rubrichub_tokenizer_certificate": data["tokenizer_certificate_sha256"],
    }


def response_data_identity(program_path: str | Path) -> dict[str, str | int]:
    """Validate and return the frozen merged response dataset identity."""

    path = Path(program_path).resolve()
    root = _repo_root(path)
    program = _json_object(path, "experiment program")
    try:
        data_ref = program["lifecycle_artifacts"]["response_data"]
        scalar_ref = program["lifecycle_artifacts"]["scalar_data"]
    except (KeyError, TypeError) as error:
        raise ResponseIdentityError(f"invalid response data references: {error}") from error
    manifest_path, manifest_sha = _frozen_reference(root, data_ref, "response_data")
    _frozen_reference(root, scalar_ref, "scalar_data")
    manifest = _json_object(manifest_path, "response data manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("id") != data_ref.get("artifact_id")
        or manifest.get("status") != "eligible"
    ):
        raise ResponseIdentityError("response data manifest identity or status is invalid")

    config = _mapping(manifest.get("config"), "response data config")
    config_path, config_sha = _file_reference(root, config, "response data config")
    outputs = _mapping(manifest.get("outputs"), "response data outputs")
    output = _mapping(outputs.get("merged_eligible"), "merged eligible response data")
    output_path, output_sha = _file_reference(root, output, "merged eligible response data", require_size=True)
    records = output.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise ResponseIdentityError("merged eligible response data record count is invalid")

    eligibility = _mapping(manifest.get("rl_eligibility"), "response data eligibility")
    eligible = eligibility.get("rubrichub_eligible_rows")
    if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible <= 0:
        raise ResponseIdentityError("response data requires a nonempty certified RubricHub partition")
    quarantine = _mapping(manifest.get("benchmark_quarantine"), "response data benchmark quarantine")
    reports = quarantine.get("reports")
    _validate_quarantine_reports(reports)
    _reject_contaminated_output(output_path, records, reports)
    rule_path, rule_sha = _file_reference(
        root,
        _mapping(eligibility.get("checker_certificate"), "RubricHub rule certificate"),
        "RubricHub rule certificate",
    )
    token_path, token_sha = _file_reference(
        root,
        _mapping(eligibility.get("tokenizer_certificate"), "RubricHub tokenizer certificate"),
        "RubricHub tokenizer certificate",
    )
    hir = _mapping(_mapping(manifest.get("sources"), "response data sources").get("hir"), "HIR source")
    hir_manifest_path, hir_manifest_sha = _file_reference(
        root,
        {
            "path": hir.get("qwen_effective_manifest_path"),
            "sha256": hir.get("qwen_effective_manifest_sha256"),
        },
        "full Qwen HIR manifest",
    )
    hir_manifest = _json_object(hir_manifest_path, "full Qwen HIR manifest")
    qwen_records = hir.get("qwen_effective_records")
    if (
        hir_manifest.get("schema_version") != 1
        or hir_manifest.get("id") != "qwen_rtt_hir_data_v1"
        or hir_manifest.get("status") != "frozen"
        or hir_manifest.get("scope") != "full_rtt_compatible_not_authoritative"
        or hir_manifest.get("sources", {}).get("rtt_processed", {}).get("sha256") != hir.get("sha256")
        or hir_manifest.get("derived", {}).get("records") != qwen_records
        or hir_manifest.get("derived", {}).get("sha256") != hir.get("qwen_effective_data_sha256")
        or [item.get("row_id") for item in hir_manifest.get("row_ids", {}).get("excluded", [])]
        != hir.get("qwen_excluded_row_ids")
        or isinstance(qwen_records, bool)
        or not isinstance(qwen_records, int)
        or qwen_records <= 0
        or qwen_records + eligible != records
    ):
        raise ResponseIdentityError("response data counts or scalar-data linkage are invalid")
    return {
        "artifact_id": str(data_ref["artifact_id"]),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha,
        "output_path": output_path.as_posix(),
        "output_sha256": output_sha,
        "records": records,
        "config_path": config_path.as_posix(),
        "config_sha256": config_sha,
        "rule_certificate_path": rule_path.as_posix(),
        "rule_certificate_sha256": rule_sha,
        "tokenizer_certificate_path": token_path.as_posix(),
        "tokenizer_certificate_sha256": token_sha,
        "hir_manifest_path": hir_manifest_path.as_posix(),
        "hir_manifest_sha256": hir_manifest_sha,
    }


def _validate_quarantine_reports(value: Any) -> None:
    if not isinstance(value, list) or [report.get("id") for report in value if isinstance(report, Mapping)] != list(
        _BENCHMARK_FIELDS
    ):
        raise ResponseIdentityError("response data requires ordered quarantine for every instruction benchmark")
    training_counts: set[int] = set()
    for report in value:
        if not isinstance(report, Mapping) or report.get("status") == "unresolved_gate":
            raise ResponseIdentityError("response data benchmark quarantine is unresolved")
        benchmark = report.get("id")
        records = report.get("records")
        benchmark_fields = report.get("benchmark_field_count")
        training_fields = report.get("training_field_count")
        matches = [report.get(name) for name in ("exact_matches", "casefold_matches", "near_matches")]
        if (
            report.get("contamination_policy") != _CONTAMINATION_POLICY
            or report.get("near_match_method") != "word_5gram_jaccard_at_least_0.8"
            or report.get("comparison_fields") != _BENCHMARK_FIELDS[benchmark]
            or isinstance(records, bool)
            or not isinstance(records, int)
            or records <= 0
            or isinstance(benchmark_fields, bool)
            or not isinstance(benchmark_fields, int)
            or benchmark_fields < records
            or (benchmark != "AdvancedIF" and benchmark_fields != records)
            or isinstance(training_fields, bool)
            or not isinstance(training_fields, int)
            or training_fields <= 0
            or any(not isinstance(items, list) for items in matches)
        ):
            raise ResponseIdentityError(f"{benchmark} quarantine coverage or policy is invalid")
        training_counts.add(training_fields)
        for kind, items in zip(("exact", "casefold", "near"), matches, strict=True):
            for item in items:
                _validate_match(item, benchmark, kind)
    if len(training_counts) != 1:
        raise ResponseIdentityError("benchmark quarantine training coverage differs")


def _validate_match(value: Any, benchmark: str, kind: str) -> None:
    if not isinstance(value, Mapping):
        raise ResponseIdentityError(f"{benchmark} {kind} match evidence is malformed")
    source = value.get("training_source")
    source_id = value.get("training_id") if source == "hir" else value.get("source_index")
    score = value.get("score")
    if (
        source not in {"hir", "rubrichub"}
        or isinstance(source_id, bool)
        or not isinstance(source_id, (str, int))
        or not str(source_id)
        or value.get("training_field") not in {"prompt", "rubric_description"}
        or not _sha256(value.get("source_row_sha256"))
        or not _sha256(value.get("training_text_sha256"))
        or not _sha256(value.get("benchmark_text_sha256"))
        or isinstance(value.get("benchmark_index"), bool)
        or not isinstance(value.get("benchmark_index"), int)
        or value["benchmark_index"] < 0
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0.8 <= float(score) <= 1
        or (kind in {"exact", "casefold"} and float(score) != 1)
    ):
        raise ResponseIdentityError(f"{benchmark} {kind} match evidence is invalid")


def _reject_contaminated_output(path: Path, expected_records: int, reports: Any) -> None:
    contaminated_hir: set[str] = set()
    contaminated_rubrichub: set[int] = set()
    for report in reports:
        for name in ("exact_matches", "casefold_matches", "near_matches"):
            for item in report[name]:
                if item["training_source"] == "hir":
                    contaminated_hir.add(str(item["training_id"]))
                else:
                    contaminated_rubrichub.add(item["source_index"])
    count = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                if not isinstance(row, Mapping):
                    raise ResponseIdentityError("merged eligible response row is malformed")
                if str(row.get("id")) in contaminated_hir:
                    raise ResponseIdentityError("contaminated HIR row remained response-training eligible")
                truth = row.get("ground_truth")
                provenance = truth.get("source_provenance") if isinstance(truth, Mapping) else None
                source_index = provenance.get("source_index") if isinstance(provenance, Mapping) else None
                if source_index in contaminated_rubrichub:
                    raise ResponseIdentityError("contaminated RubricHub row remained response-training eligible")
    except ResponseIdentityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseIdentityError(f"cannot inspect merged eligible response data: {error}") from error
    if count != expected_records:
        raise ResponseIdentityError("merged eligible response row count differs from its manifest")


def file_sha256(path: str | Path) -> str:
    """Hash one regular, non-symlink file."""

    target = Path(path).resolve()
    if Path(path).is_symlink() or not target.is_file():
        raise ResponseIdentityError(f"response source must be a regular file: {path}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(program_path: Path) -> Path:
    try:
        root = program_path.parents[2]
    except IndexError as error:
        raise ResponseIdentityError("experiment program path is not under configs/program") from error
    if program_path.parent.name != "program" or program_path.parent.parent.name != "configs":
        raise ResponseIdentityError("experiment program path is not under configs/program")
    return root.resolve()


def _frozen_reference(root: Path, value: Any, label: str) -> tuple[Path, str]:
    reference = _mapping(value, label)
    if reference.get("status") != "frozen" or not _sha256(reference.get("sha256")):
        raise ResponseIdentityError(f"{label} must be frozen")
    path = _safe_path(root, reference.get("path"), label)
    digest = file_sha256(path)
    if digest != reference["sha256"]:
        raise ResponseIdentityError(f"{label} differs from its frozen hash")
    return path, digest


def _file_reference(
    root: Path,
    value: Mapping[str, Any],
    label: str,
    *,
    require_size: bool = False,
) -> tuple[Path, str]:
    if not _sha256(value.get("sha256")):
        raise ResponseIdentityError(f"{label} hash is invalid")
    path = _safe_path(root, value.get("path"), label)
    if require_size and (
        isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or path.stat().st_size != value["bytes"]
    ):
        raise ResponseIdentityError(f"{label} size differs from its frozen identity")
    digest = file_sha256(path)
    if digest != value["sha256"]:
        raise ResponseIdentityError(f"{label} differs from its frozen hash")
    return path, digest


def _safe_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ResponseIdentityError(f"{label} path is invalid")
    raw = root / value
    path = raw.resolve()
    if raw.is_symlink() or not path.is_relative_to(root) or not path.is_file():
        raise ResponseIdentityError(f"{label} path is absent or outside the repository")
    return path


def _json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ResponseIdentityError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseIdentityError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ResponseIdentityError(f"{label} must be an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseIdentityError(f"{label} must be an object")
    return value


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
