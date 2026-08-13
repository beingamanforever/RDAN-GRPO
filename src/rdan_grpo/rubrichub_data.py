"""Pinned RubricHub instruction-following conversion and HIR merge."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

BASE_TASK_PREFIX = "Does the response address the follow question? \\n\\n"
MERGED_ROW_SCHEMA = {
    "id": "string",
    "parameters_encoding": "canonical_json_object_string",
    "rubric_fields": ["id", "category", "description", "weight", "verifier", "function", "parameters"],
    "rubric_route_fields": ["rubric_index", "verifier", "function", "parameters", "criterion", "points"],
    "ground_truth_fields": [
        "checker",
        "functions",
        "instruction_id_list",
        "kwargs",
        "constraints",
        "constraint_pattern",
        "source_ground_truth",
        "style",
        "hard_mask",
        "rubric_routes",
        "source_provenance",
        "rl_eligible",
        "quarantine_reasons",
    ],
}


class RubricHubDataError(ValueError):
    """Raised when a source, certificate, or derived dataset is inconsistent."""


@dataclass(frozen=True)
class RowMeta:
    """Comparison metadata for one validated source row."""

    index: int
    source_hash: str
    prompt: str
    prompt_hash: str
    exact_key: str
    folded_key: str
    rubric_hash: str
    rubric_descriptions: tuple[str, ...]
    functions: tuple[str, ...]
    foreign_scripts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class LanguageResult:
    """Frozen detector result bound to one source row."""

    language: str
    confidence: float
    mixed: bool
    segments: tuple[str, ...]
    flags: tuple[str, ...]


@dataclass
class _JsonlWriter:
    path: Path
    temp_path: Path
    stream: Any
    digest: Any
    records: int = 0
    size: int = 0

    @classmethod
    def open(cls, path: Path) -> _JsonlWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        return cls(path, Path(name), os.fdopen(handle, "wb"), hashlib.sha256())

    def write_bytes(self, payload: bytes) -> None:
        self.stream.write(payload)
        self.digest.update(payload)
        self.records += 1
        self.size += len(payload)

    def write(self, payload: dict[str, Any]) -> None:
        self.write_bytes(_json_line(payload))

    def close(self) -> dict[str, Any]:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        return {
            "path": self.path.as_posix(),
            "format": "jsonl",
            "records": self.records,
            "bytes": self.size,
            "sha256": self.digest.hexdigest(),
        }

    def abort(self) -> None:
        if not self.stream.closed:
            self.stream.close()
        self.temp_path.unlink(missing_ok=True)


def source_row_hash(row: dict[str, Any]) -> str:
    """Return the stable canonical hash used to bind source certificates."""

    return hashlib.sha256(_canonical_bytes(row)).hexdigest()


def normalize_prompt(prompt: str, *, casefold: bool = False) -> str:
    """Normalize only for comparison without changing stored prompt text."""

    value = prompt.replace("\r\n", "\n").replace("\r", "\n")
    value = " ".join(unicodedata.normalize("NFKC", value).split())
    return value.casefold() if casefold else value


def build_merged_rl_data(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    language_certificate: str | Path | None = None,
    checker_certificate: str | Path | None = None,
    tokenizer_certificate: str | Path | None = None,
    output_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build English RubricHub archives and fail-closed RL-eligible merges."""

    config_path = Path(config_path).resolve()
    root = Path(repo_root).resolve() if repo_root else config_path.parents[2]
    config = _read_json(config_path)
    if config.get("row_schema") != MERGED_ROW_SCHEMA:
        raise RubricHubDataError("merged row schema differs from the supported uniform schema")
    source = config["source"]
    parquet_path = _resolve(root, source["path"])
    _verify_file(parquet_path, source["bytes"], source["sha256"], "RubricHub parquet")

    metas, function_counts = _scan_source(parquet_path, source)
    language_path = _required_path(language_certificate, config.get("language_certificate"), root)
    languages, language_ref = _load_language_certificate(language_path, metas, config, source)
    english, language_exclusions = _english_partition(metas, languages, config["language_gate"])
    _verify_documented_exclusions(language_exclusions, config["language_gate"])

    hir_path = _resolve(root, config["hir"]["path"])
    _verify_file(hir_path, config["hir"]["bytes"], config["hir"]["sha256"], "certified HIR JSONL")
    qwen_manifest_path = _resolve(root, config["hir"]["qwen_effective_manifest_path"])
    _verify_file(
        qwen_manifest_path,
        qwen_manifest_path.stat().st_size,
        config["hir"]["qwen_effective_manifest_sha256"],
        "Qwen effective HIR manifest",
    )
    qwen_manifest = _read_json(qwen_manifest_path)
    hir_source_ref, effective, excluded = _hir_qwen_inventory(qwen_manifest)
    if (
        hir_source_ref.get("sha256") != config["hir"]["sha256"]
        or (hir_source_ref.get("records") is not None and hir_source_ref["records"] != config["hir"]["records"])
        or effective != config["hir"]["qwen_effective_records"]
        or excluded != config["hir"]["qwen_excluded_row_ids"]
    ):
        raise RubricHubDataError("Qwen effective HIR manifest differs from the frozen data gate")
    hir_excluded = set(config["hir"]["qwen_excluded_row_ids"])
    hir_rows = list(_read_jsonl_bytes(hir_path))
    if len(hir_rows) != config["hir"]["records"]:
        raise RubricHubDataError("certified HIR row count differs from the frozen configuration")
    hir_exact = {normalize_prompt(row[0]["prompt"]): row[0]["id"] for row in hir_rows}
    hir_folded = {normalize_prompt(row[0]["prompt"], casefold=True): row[0]["id"] for row in hir_rows}

    decisions, duplicate_evidence, collision_evidence = _partition_duplicates(english, hir_exact, hir_folded)
    benchmark_report, benchmark_reasons, hir_benchmark_reasons = _benchmark_report(
        config["benchmarks"], root, english, hir_rows, hir_excluded
    )
    checker_path = _optional_path(checker_certificate, config.get("checker_certificate"), root)
    tokenizer_path = _optional_path(tokenizer_certificate, config.get("tokenizer_certificate"), root)
    supported, checker_rows, checker_ref = _load_checker_certificate(
        checker_path,
        source,
        root,
        metas,
        english,
        language_path,
        config.get("checker_reference_root"),
    )
    token_rows, tokenizer_ref = _load_tokenizer_certificate(
        tokenizer_path,
        metas,
        source,
        root,
        checker_path,
        language_path,
        checker_rows,
        config.get("tokenizer_gate"),
    )

    outputs = {key: _resolve(root, value) for key, value in config["outputs"].items()}
    if output_paths:
        outputs.update({key: Path(value).resolve() for key, value in output_paths.items()})
    required_outputs = {"rubrichub_archive", "rubrichub_eligible", "merged_archive", "merged_eligible", "manifest"}
    if set(outputs) != required_outputs or len(set(outputs.values())) != len(outputs):
        raise RubricHubDataError("output paths must contain five distinct required targets")

    writers = {key: _JsonlWriter.open(outputs[key]) for key in required_outputs - {"manifest"}}
    rubric_reasons: dict[int, list[str]] = defaultdict(list)
    unsupported_routes: Counter[str] = Counter()
    try:
        for hir, _ in hir_rows:
            eligible = hir.get("id") not in hir_excluded
            converted_hir = _uniform_row(
                hir,
                hir_eligible=eligible,
                hir_reasons=hir_benchmark_reasons.get(str(hir.get("id")), ()),
            )
            writers["merged_archive"].write(converted_hir)
            if converted_hir["ground_truth"]["rl_eligible"]:
                writers["merged_eligible"].write(converted_hir)

        for index, row in _iter_parquet_rows(parquet_path):
            if index not in decisions or decisions[index] == "duplicate_payload":
                continue
            meta = metas[index]
            reasons = list(benchmark_reasons.get(index, ()))
            if decisions[index] != "keep":
                reasons.append(decisions[index])
            missing = sorted(set(meta.functions) - supported)
            if missing:
                reasons.append("uncertified_rule_route")
                unsupported_routes.update(missing)
            token = token_rows.get(index)
            if token is None:
                reasons.append("missing_tokenizer_certificate")
            elif not token:
                reasons.append("prompt_tokenizer_gate_failed")
            reasons = sorted(set(reasons))
            rubric_reasons[index] = reasons
            converted = _uniform_row(_convert_row(index, row, meta, reasons))
            writers["rubrichub_archive"].write(converted)
            if not reasons:
                writers["rubrichub_eligible"].write(converted)
            if decisions[index] != "cross_source_collision":
                writers["merged_archive"].write(converted)
                if not reasons:
                    writers["merged_eligible"].write(converted)

        output_refs = {key: writer.close() for key, writer in writers.items()}
        if (
            config.get("require_nonempty_rubrichub_eligible") is True
            and not output_refs["rubrichub_eligible"]["records"]
        ):
            raise RubricHubDataError("production merge requires a nonempty certified RubricHub partition")
        manifest = _build_manifest(
            config_path=config_path,
            root=root,
            config=config,
            source=source,
            metas=metas,
            function_counts=function_counts,
            english=english,
            language_ref=language_ref,
            language_exclusions=language_exclusions,
            checker_ref=checker_ref,
            tokenizer_ref=tokenizer_ref,
            duplicate_evidence=duplicate_evidence,
            collision_evidence=collision_evidence,
            benchmark_report=benchmark_report,
            reasons=rubric_reasons,
            unsupported_routes=unsupported_routes,
            output_refs=output_refs,
        )
        manifest_temp = _write_json_temp(outputs["manifest"], manifest)
        for writer in writers.values():
            os.replace(writer.temp_path, writer.path)
        os.replace(manifest_temp, outputs["manifest"])
        return manifest
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise


def build_language_certificate(
    config_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Classify every pinned source prompt with the frozen fastText policy."""

    config_path = Path(config_path).resolve()
    root = Path(repo_root).resolve() if repo_root else config_path.parents[2]
    config = _read_json(config_path)
    source, gate = config["source"], config["language_gate"]
    parquet_path = _resolve(root, source["path"])
    model_path, output_path = Path(model_path).resolve(), Path(output_path).resolve()
    _verify_file(parquet_path, source["bytes"], source["sha256"], "RubricHub parquet")
    detector = gate["detector"]
    _verify_file(model_path, detector["model_bytes"], detector["model_sha256"], "fastText language model")
    if platform.python_version() != detector["python_version"]:
        raise RubricHubDataError("Python version differs from the frozen language detector runtime")
    if importlib.metadata.version("fasttext-wheel") != detector["package_version"]:
        raise RubricHubDataError("fasttext-wheel version differs from the frozen language detector runtime")
    if importlib.metadata.version("numpy") != config["language_gate"]["detector"]["numpy_version"]:
        raise RubricHubDataError("NumPy version differs from the frozen language detector runtime")
    try:
        import fasttext
    except ImportError as error:
        raise RubricHubDataError("fasttext-wheel is required to build the language certificate") from error
    model = fasttext.load_model(str(model_path))
    results: list[dict[str, Any]] = []
    for index, row in _iter_parquet_rows(parquet_path):
        prompt, _ = _validate_source_row(index, row)
        base_task = _base_task(row)
        training_text = _training_text(row)
        language, confidence = _predict_language(model, base_task)
        full_language, full_confidence = _predict_language(model, prompt)
        training_language, training_confidence = _predict_language(model, training_text)
        segments = []
        for segment in _language_segments(training_text, gate):
            segment_language, segment_confidence = _predict_language(model, segment)
            segments.append(
                {
                    "language": segment_language,
                    "confidence": segment_confidence,
                    "sha256": hashlib.sha256(segment.encode("utf-8")).hexdigest(),
                }
            )
        scripts = _non_latin_scripts(training_text)
        foreign_scripts = _foreign_language_scripts(training_text)
        letters = sum(character.isalpha() for character in training_text)
        non_latin_letters = sum(scripts.values())
        non_latin = non_latin_letters >= gate["non_latin_minimum_letters"] and (
            non_latin_letters / max(letters, 1) >= gate["non_latin_minimum_ratio"]
        )
        segment_non_english = any(
            segment["language"] != "en" and segment["confidence"] >= gate["segment_minimum_confidence"]
            for segment in segments
        )
        output_languages = _required_output_languages(row)
        non_english_output = any(language.casefold() not in {"en", "eng", "english"} for language in output_languages)
        flags = []
        if full_language != "en" or training_language != "en":
            flags.append("non_english_training_text")
        if non_latin:
            flags.append("non_latin_training_text")
        if foreign_scripts:
            flags.append("non_english_script")
        if segment_non_english:
            flags.append("mixed_language_training_text")
        if non_english_output:
            flags.append("non_english_output_requirement")
        mixed = bool(flags)
        results.append(
            {
                "source_index": index,
                "source_row_sha256": source_row_hash(row),
                "language": language,
                "confidence": confidence,
                "mixed": mixed,
                "full_prompt_language": full_language,
                "full_prompt_confidence": full_confidence,
                "training_text_language": training_language,
                "training_text_confidence": training_confidence,
                "base_task_sha256": hashlib.sha256(base_task.encode("utf-8")).hexdigest(),
                "training_text_sha256": hashlib.sha256(training_text.encode("utf-8")).hexdigest(),
                "non_latin_scripts": scripts,
                "required_output_languages": output_languages,
                "reason_flags": flags,
                "segment_languages": [segment["language"] for segment in segments],
                "segment_results": segments,
            }
        )
    if len(results) != source["records"]:
        raise RubricHubDataError("language detector did not cover every pinned source row")
    payload = {
        "schema_version": 1,
        "id": "rubrichub_instruction_following_english_fasttext_v1",
        "status": "frozen",
        "source": _certificate_source(source),
        "detector": detector,
        "policy": {
            key: gate[key]
            for key in (
                "required_base_task_language",
                "segment_minimum_confidence",
                "segment_minimum_characters",
                "segment_minimum_letters",
                "non_latin_minimum_letters",
                "non_latin_minimum_ratio",
            )
        },
        "results": results,
    }
    temp = _write_json_temp(output_path, payload)
    os.replace(temp, output_path)
    return payload


def _scan_source(path: Path, source: dict[str, Any]) -> tuple[list[RowMeta], Counter[str]]:
    metas: list[RowMeta] = []
    functions: Counter[str] = Counter()
    for index, row in _iter_parquet_rows(path):
        prompt, row_functions = _validate_source_row(index, row)
        digest = source_row_hash(row)
        exact = normalize_prompt(prompt)
        metas.append(
            RowMeta(
                index=index,
                source_hash=digest,
                prompt=prompt,
                prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                exact_key=exact,
                folded_key=exact.casefold(),
                rubric_hash=hashlib.sha256(_canonical_bytes(row["reward_model"]["rubrics"])).hexdigest(),
                rubric_descriptions=tuple(rubric["criterion"] for rubric in row["reward_model"]["rubrics"]),
                functions=tuple(row_functions),
                foreign_scripts=tuple(_foreign_language_scripts(_training_text(row)).items()),
            )
        )
        functions.update(row_functions)
    if len(metas) != source["records"]:
        raise RubricHubDataError(f"RubricHub row count differs: {len(metas)}")
    expected = set(source["rule_functions"])
    if set(functions) != expected or len(expected) != source["rule_function_count"]:
        raise RubricHubDataError("RubricHub rule function set differs from the frozen routes")
    return metas, functions


def _validate_source_row(index: int, row: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("data_source") != "Instruction_Following" or row.get("ability") != "Instruction_Following":
        raise RubricHubDataError(f"row {index}: source or ability is not Instruction_Following")
    messages = row.get("prompt")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
        or not messages[0]["content"].strip()
    ):
        raise RubricHubDataError(f"row {index}: prompt must contain one nonempty user message")
    reward = row.get("reward_model")
    extra = row.get("extra_info")
    if not isinstance(reward, dict) or not isinstance(extra, dict):
        raise RubricHubDataError(f"row {index}: reward mirrors are missing")
    if reward != extra.get("reward_model") or messages != extra.get("prompt"):
        raise RubricHubDataError(f"row {index}: reward_model or prompt mirror differs")
    rubrics = reward.get("rubrics")
    if not isinstance(rubrics, list) or not 2 <= len(rubrics) <= 6:
        raise RubricHubDataError(f"row {index}: expected 2-6 rubrics")
    summary = [{"criterion": rubric.get("criterion"), "points": rubric.get("points")} for rubric in rubrics]
    if summary != row.get("Rubrics"):
        raise RubricHubDataError(f"row {index}: Rubrics mirror differs")
    verifiers: Counter[str] = Counter()
    functions: list[str] = []
    for rubric_index, rubric in enumerate(rubrics):
        criterion, points, tags = rubric.get("criterion"), rubric.get("points"), rubric.get("tags")
        if not isinstance(criterion, str) or not criterion.strip():
            raise RubricHubDataError(f"row {index} rubric {rubric_index}: criterion is empty")
        if not isinstance(points, int) or isinstance(points, bool) or points <= 0:
            raise RubricHubDataError(f"row {index} rubric {rubric_index}: points must be a positive integer")
        if not isinstance(tags, dict) or (
            tags.get("parameters") is not None and not isinstance(tags["parameters"], dict)
        ):
            raise RubricHubDataError(f"row {index} rubric {rubric_index}: tags are malformed")
        verifier, function = tags.get("verifier"), tags.get("function")
        if verifier not in {"rule", "llm"} or not isinstance(function, str):
            raise RubricHubDataError(f"row {index} rubric {rubric_index}: verifier is invalid")
        if (verifier == "llm") != (function == ""):
            raise RubricHubDataError(f"row {index} rubric {rubric_index}: empty function must mean llm")
        verifiers[verifier] += 1
        if verifier == "rule":
            functions.append(function)
    if verifiers != Counter({"llm": 1, "rule": len(rubrics) - 1}):
        raise RubricHubDataError(f"row {index}: expected exactly one llm and 1-5 rule rubrics")
    _base_task(row)
    return messages[0]["content"], functions


def _english_partition(
    metas: list[RowMeta],
    results: dict[int, LanguageResult],
    gate: dict[str, Any],
) -> tuple[list[RowMeta], list[dict[str, Any]]]:
    accepted: list[RowMeta] = []
    excluded: list[dict[str, Any]] = []
    for meta in metas:
        result = results[meta.index]
        reason = ""
        if result.language != gate["required_base_task_language"]:
            reason = "non_english_base_task"
        elif "non_english_output_requirement" in result.flags:
            reason = "non_english_output_requirement"
        elif "non_english_training_text" in result.flags:
            reason = "non_english_training_text"
        elif meta.foreign_scripts:
            reason = "non_english_script"
        elif result.mixed:
            reason = "mixed_language_prompt"
        if reason:
            excluded.append(
                {
                    "source_index": meta.index,
                    "source_row_sha256": meta.source_hash,
                    "prompt_sha256": meta.prompt_hash,
                    "language": result.language,
                    "confidence": result.confidence,
                    "mixed": result.mixed,
                    "segment_languages": list(result.segments),
                    "foreign_scripts": dict(meta.foreign_scripts),
                    "reason": reason,
                }
            )
        else:
            accepted.append(meta)
    return accepted, excluded


def _verify_documented_exclusions(exclusions: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    indexed = {item["source_index"]: item for item in exclusions}
    for expected in gate.get("documented_exclusions", []):
        observed = indexed.get(expected.get("source_index"))
        if not observed or any(observed.get(key) != expected[key] for key in ("language", "confidence", "reason")):
            raise RubricHubDataError("documented language exclusion differs from the frozen certificate")


def _partition_duplicates(
    metas: list[RowMeta],
    hir_exact: dict[str, Any],
    hir_folded: dict[str, Any],
) -> tuple[dict[int, str], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = {meta.index: "keep" for meta in metas}
    duplicate_evidence: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    prompt_groups = _group(metas, "exact_key")
    for _, group in prompt_groups.items():
        if len(group) < 2:
            continue
        if len({meta.rubric_hash for meta in group}) == 1:
            keep = group[0]
            for duplicate in group[1:]:
                decisions[duplicate.index] = "duplicate_payload"
                duplicate_evidence.append(
                    {
                        "kept_source_index": keep.index,
                        "duplicate_source_index": duplicate.index,
                        "kept_source_row_sha256": keep.source_hash,
                        "duplicate_source_row_sha256": duplicate.source_hash,
                        "rubric_sha256": keep.rubric_hash,
                    }
                )
        else:
            for conflict in group:
                decisions[conflict.index] = "conflicting_prompt_payload"
            collisions.append(
                {
                    "kind": "rubrichub_conflicting_prompt_payload",
                    "source_indices": [meta.index for meta in group],
                    "rubric_sha256": [meta.rubric_hash for meta in group],
                }
            )
    folded_groups = _group([meta for meta in metas if decisions[meta.index] == "keep"], "folded_key")
    for _, group in folded_groups.items():
        exact_keys = {meta.exact_key for meta in group}
        if len(exact_keys) <= 1:
            continue
        indices = [meta.index for meta in group]
        for meta in group:
            decisions[meta.index] = "casefold_collision"
        collisions.append({"kind": "rubrichub_casefold", "source_indices": indices})
    for meta in metas:
        if decisions[meta.index] == "duplicate_payload":
            continue
        match = None
        if meta.exact_key in hir_exact:
            match = {"kind": "hir_exact", "hir_id": hir_exact[meta.exact_key]}
        elif meta.folded_key in hir_folded:
            match = {"kind": "hir_casefold", "hir_id": hir_folded[meta.folded_key]}
        if match:
            decisions[meta.index] = "cross_source_collision"
            collisions.append({**match, "source_index": meta.index, "source_row_sha256": meta.source_hash})
    return decisions, duplicate_evidence, collisions


def _benchmark_report(
    configs: list[dict[str, Any]],
    root: Path,
    metas: list[RowMeta],
    hir_rows: list[tuple[dict[str, Any], bytes]],
    hir_excluded: set[Any],
) -> tuple[list[dict[str, Any]], dict[int, list[str]], dict[str, list[str]]]:
    report: list[dict[str, Any]] = []
    reasons: dict[int, list[str]] = defaultdict(list)
    hir_reasons: dict[str, list[str]] = defaultdict(list)
    training_fields = list(_training_fields(metas, hir_rows, hir_excluded))
    for config in configs:
        path = _resolve(root, config["path"])
        _verify_file(path, config["bytes"], config["sha256"], f"{config['id']} benchmark")
        if config["id"] == "AdvancedIF":
            benchmark = _read_advancedif_fields(path, config)
            comparison_fields = ["message_content", "role_preserving_transcript", "rubric"]
        else:
            prompts = _read_benchmark_prompts(path, config)
            if len(prompts) != config["records"]:
                raise RubricHubDataError(f"{config['id']} benchmark row count differs")
            benchmark = [_benchmark_field(index, "prompt", 0, prompt) for index, prompt in enumerate(prompts)]
            comparison_fields = ["prompt"]
        item, item_reasons, item_hir_reasons = _complete_benchmark_report(
            path,
            config,
            benchmark,
            comparison_fields,
            training_fields,
        )
        report.append(item)
        for index, values in item_reasons.items():
            reasons[index].extend(values)
        for row_id, values in item_hir_reasons.items():
            hir_reasons[row_id].extend(values)
    return report, reasons, hir_reasons


def _complete_benchmark_report(
    path: Path,
    config: dict[str, Any],
    benchmark: list[dict[str, Any]],
    comparison_fields: list[str],
    fields: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, list[str]], dict[str, list[str]]]:
    if len({item["benchmark_index"] for item in benchmark}) != config["records"]:
        raise RubricHubDataError(f"{config['id']} benchmark row count differs")
    exact: dict[str, list[int]] = defaultdict(list)
    folded: dict[str, list[int]] = defaultdict(list)
    grams: list[set[str]] = []
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, item in enumerate(benchmark):
        exact[item["exact_key"]].append(index)
        folded[item["folded_key"]].append(index)
        item_grams = _word_grams(item["folded_key"])
        grams.append(item_grams)
        for gram in item_grams:
            inverted[gram].add(index)

    exact_matches: list[dict[str, Any]] = []
    casefold_matches: list[dict[str, Any]] = []
    near_matches: list[dict[str, Any]] = []
    reasons: dict[int, list[str]] = defaultdict(list)
    hir_reasons: dict[str, list[str]] = defaultdict(list)
    for field in fields:
        matched_exact = set(exact.get(field["exact_key"], ()))
        for benchmark_index in sorted(matched_exact):
            exact_matches.append({**_match_evidence(field, benchmark[benchmark_index]), "score": 1.0})
            _add_benchmark_reason(field, reasons, hir_reasons, f"benchmark_exact:{config['id']}")
        matched_folded = set(folded.get(field["folded_key"], ())) - matched_exact
        for benchmark_index in sorted(matched_folded):
            casefold_matches.append({**_match_evidence(field, benchmark[benchmark_index]), "score": 1.0})
            _add_benchmark_reason(field, reasons, hir_reasons, f"benchmark_casefold:{config['id']}")

        field_grams = _word_grams(field["folded_key"])
        candidate_counts: Counter[int] = Counter()
        for gram in field_grams:
            candidate_counts.update(inverted.get(gram, ()))
        for benchmark_index in sorted(candidate_counts):
            if benchmark_index in matched_exact or benchmark_index in matched_folded:
                continue
            other = grams[benchmark_index]
            required = max(1, int(0.8 * max(len(field_grams), len(other))))
            if candidate_counts[benchmark_index] < required:
                continue
            score = len(field_grams & other) / len(field_grams | other) if field_grams or other else 1.0
            if score >= 0.8:
                near_matches.append({**_match_evidence(field, benchmark[benchmark_index]), "score": round(score, 6)})
                _add_benchmark_reason(field, reasons, hir_reasons, f"benchmark_near:{config['id']}")
    return (
        {
            "id": config["id"],
            "path": config["path"],
            "records": config["records"],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "comparison_fields": comparison_fields,
            "benchmark_field_count": len(benchmark),
            "training_field_count": len(fields),
            "contamination_policy": "all_training_fields_exact_casefold_word_5gram_jaccard_v1",
            "exact_matches": exact_matches,
            "casefold_matches": casefold_matches,
            "near_matches": near_matches,
            "near_match_method": "word_5gram_jaccard_at_least_0.8",
        },
        reasons,
        hir_reasons,
    )


def _read_advancedif_fields(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("format") != "csv" or config.get("prompt_field") != "conversation_history":
        raise RubricHubDataError("AdvancedIF benchmark configuration is unsupported")
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise RubricHubDataError(f"cannot read AdvancedIF CSV: {path}") from error
    if len(rows) != config["records"]:
        raise RubricHubDataError("AdvancedIF benchmark row count differs")
    fields: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            messages = json.loads(row["conversation_history"])
            metadata = json.loads(row["prompt_metadata"])
            rubrics = json.loads(metadata["rubrics"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RubricHubDataError(f"AdvancedIF row {row_index} is malformed") from error
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in messages
            )
            or not isinstance(rubrics, list)
            or not rubrics
            or any(not isinstance(rubric, str) or not rubric.strip() for rubric in rubrics)
        ):
            raise RubricHubDataError(f"AdvancedIF row {row_index} is malformed")
        for message_index, message in enumerate(messages):
            fields.append(
                _benchmark_field(row_index, "message_content", message_index, message["content"], message["role"])
            )
        transcript = "\n".join(f"[{message['role']}]\n{message['content']}" for message in messages)
        fields.append(_benchmark_field(row_index, "role_preserving_transcript", 0, transcript))
        for rubric_index, rubric in enumerate(rubrics):
            fields.append(_benchmark_field(row_index, "rubric", rubric_index, rubric))
    return fields


def _benchmark_field(
    benchmark_index: int,
    field: str,
    field_index: int,
    text: str,
    role: str | None = None,
) -> dict[str, Any]:
    exact = normalize_prompt(text)
    return {
        "benchmark_index": benchmark_index,
        "benchmark_field": field,
        "benchmark_field_index": field_index,
        "benchmark_role": role,
        "benchmark_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "exact_key": exact,
        "folded_key": exact.casefold(),
    }


def _training_fields(
    metas: list[RowMeta],
    hir_rows: list[tuple[dict[str, Any], bytes]],
    hir_excluded: set[Any],
) -> Iterator[dict[str, Any]]:
    for row, _ in hir_rows:
        if row.get("id") in hir_excluded:
            continue
        source_hash = source_row_hash(row)
        yield _training_field("hir", str(row["id"]), None, source_hash, "prompt", 0, row["prompt"])
        for index, rubric in enumerate(row["rubrics"]):
            yield _training_field(
                "hir", str(row["id"]), None, source_hash, "rubric_description", index, rubric["description"]
            )
    for meta in metas:
        yield _training_field("rubrichub", None, meta, meta.source_hash, "prompt", 0, meta.prompt)
        for index, description in enumerate(meta.rubric_descriptions):
            yield _training_field("rubrichub", None, meta, meta.source_hash, "rubric_description", index, description)


def _training_field(
    source: str,
    row_id: str | None,
    meta: RowMeta | None,
    source_hash: str,
    field: str,
    field_index: int,
    text: str,
) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise RubricHubDataError(f"{source} training {field} is empty")
    exact = normalize_prompt(text)
    return {
        "training_source": source,
        "training_id": row_id,
        "source_index": meta.index if meta else None,
        "source_row_sha256": source_hash,
        "training_field": field,
        "training_field_index": field_index,
        "training_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "exact_key": exact,
        "folded_key": exact.casefold(),
    }


def _match_evidence(field: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {**field, **benchmark}.items()
        if key not in {"exact_key", "folded_key"} and value is not None
    }


def _add_benchmark_reason(
    field: dict[str, Any],
    reasons: dict[int, list[str]],
    hir_reasons: dict[str, list[str]],
    reason: str,
) -> None:
    if field["training_source"] == "rubrichub":
        reasons[field["source_index"]].append(reason)
    else:
        hir_reasons[field["training_id"]].append(reason)


def _convert_row(
    index: int,
    row: dict[str, Any],
    meta: RowMeta,
    reasons: list[str],
) -> dict[str, Any]:
    reward = row["reward_model"]
    rubrics: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    hard_mask: list[bool] = []
    for rubric_index, source_rubric in enumerate(reward["rubrics"]):
        tags = source_rubric["tags"]
        parameters = {key: value for key, value in (tags["parameters"] or {}).items() if value is not None}
        is_rule = tags["verifier"] == "rule"
        rubrics.append(
            {
                "id": rubric_index + 1,
                "category": tags["verifier"],
                "description": source_rubric["criterion"],
                "weight": source_rubric["points"],
                "verifier": tags["verifier"],
                "function": tags["function"],
                "parameters": parameters,
            }
        )
        routes.append(
            {
                "rubric_index": rubric_index,
                "verifier": tags["verifier"],
                "function": tags["function"],
                "parameters": parameters,
                "criterion": source_rubric["criterion"],
                "points": source_rubric["points"],
            }
        )
        hard_mask.append(is_rule)
    provenance = {
        "dataset": "sojuL/RubricHub_v1",
        "revision": "3837d55971473a872e84879c88f708b8da3ec2ef",
        "file": "RuRL/rurbichub_v1_Instruction_Following.parquet",
        "source_index": index,
        "source_row_sha256": meta.source_hash,
        "prompt_sha256": meta.prompt_hash,
    }
    return {
        "id": f"rubrichub-if-{index:05d}-{meta.source_hash[:12]}",
        "source": "rubrichub_instruction_following",
        "prompt": meta.prompt,
        "question": meta.prompt,
        "rubrics": rubrics,
        "tag": "llm_judge",
        "difficulty": 0,
        "ground_truth": {
            "source_ground_truth": reward["ground_truth"],
            "style": reward["style"],
            "hard_mask": hard_mask,
            "rubric_routes": routes,
            "source_provenance": provenance,
            "rl_eligible": not reasons,
            "quarantine_reasons": reasons,
        },
        "messages": row["prompt"],
    }


def _uniform_row(
    row: dict[str, Any],
    *,
    hir_eligible: bool | None = None,
    hir_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    source = row.get("source")
    is_rubrichub = source == "rubrichub_instruction_following"
    if is_rubrichub == (hir_eligible is not None):
        raise RubricHubDataError("uniform row eligibility does not match its source")

    rubric_fields = MERGED_ROW_SCHEMA["rubric_fields"]
    rubrics = row.get("rubrics")
    if not isinstance(rubrics, list):
        raise RubricHubDataError("uniform row rubrics must be a list")
    normalized_rubrics: list[dict[str, Any]] = []
    required = set(rubric_fields[:4])
    allowed = set(rubric_fields)
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict) or not required <= set(rubric) or not set(rubric) <= allowed:
            raise RubricHubDataError(f"uniform row rubric {index} has an invalid schema")
        values = {**rubric, "verifier": rubric.get("verifier", ""), "function": rubric.get("function", "")}
        values["parameters"] = _canonical_object_string(rubric.get("parameters", {}), f"rubric {index} parameters")
        normalized_rubrics.append({field: values[field] for field in rubric_fields})

    truth = row.get("ground_truth")
    if not isinstance(truth, dict) or not set(truth) <= set(MERGED_ROW_SCHEMA["ground_truth_fields"]):
        raise RubricHubDataError("uniform row ground truth has an invalid schema")
    if is_rubrichub:
        truth_values = {
            **truth,
            "checker": [],
            "functions": [],
            "instruction_id_list": [],
            "kwargs": [],
            "constraints": [],
            "constraint_pattern": [],
        }
    else:
        expected = {
            "type1": {"instruction_id_list", "kwargs"},
            "type2": {"instruction_id_list", "kwargs"},
            "type3": {"constraints", "constraint_pattern"},
            "type4": {"checker", "functions"},
        }
        if source not in expected or set(truth) != expected[source]:
            raise RubricHubDataError("HIR ground truth differs from its source schema")
        if source in {"type1", "type2"}:
            ids, kwargs = truth["instruction_id_list"], truth["kwargs"]
            valid = (
                isinstance(ids, list)
                and len(ids) == len(normalized_rubrics)
                and all(isinstance(value, str) and value for value in ids)
                and isinstance(kwargs, list)
                and len(kwargs) == len(normalized_rubrics)
                and all(isinstance(value, dict) for value in kwargs)
            )
        elif source == "type3":
            constraints = truth["constraints"]
            valid = isinstance(constraints, list) and len(constraints) == len(normalized_rubrics)
        else:
            checker, functions = truth["checker"], truth["functions"]
            valid = (
                isinstance(checker, list)
                and len(checker) == len(normalized_rubrics)
                and isinstance(functions, list)
                and len(functions) == len(normalized_rubrics)
            )
        if not valid:
            raise RubricHubDataError("HIR evaluator inventory differs from its rubrics")
        quarantine = ([] if hir_eligible else ["qwen_effective_exclusion"]) + sorted(set(hir_reasons))
        truth_values = {
            "checker": truth.get("checker", []),
            "functions": truth.get("functions", []),
            "instruction_id_list": truth.get("instruction_id_list", []),
            "kwargs": truth.get("kwargs", []),
            "constraints": truth.get("constraints", []),
            "constraint_pattern": truth.get("constraint_pattern", []),
            "source_ground_truth": "",
            "style": "",
            "hard_mask": (
                [True] * len(normalized_rubrics)
                if source in {"type1", "type2", "type3"}
                else [isinstance(item, str) and item.startswith("[rule]") for item in truth["checker"]]
            ),
            "rubric_routes": [
                {
                    "rubric_index": index,
                    "verifier": "",
                    "function": "",
                    "parameters": "{}",
                    "criterion": "",
                    "points": 0,
                }
                for index in range(len(normalized_rubrics))
            ],
            "source_provenance": {
                "dataset": "",
                "revision": "",
                "file": "",
                "source_index": -1,
                "source_row_sha256": "",
                "prompt_sha256": "",
            },
            "rl_eligible": hir_eligible and not quarantine,
            "quarantine_reasons": quarantine,
        }
    route_fields = MERGED_ROW_SCHEMA["rubric_route_fields"]
    routes = truth_values["rubric_routes"]
    if not isinstance(routes, list):
        raise RubricHubDataError("uniform row rubric routes must be a list")
    normalized_routes: list[dict[str, Any]] = []
    for index, route in enumerate(routes):
        if not isinstance(route, dict) or set(route) != set(route_fields):
            raise RubricHubDataError(f"uniform row rubric route {index} has an invalid schema")
        values = {
            **route,
            "parameters": _canonical_object_string(route["parameters"], f"rubric route {index} parameters"),
        }
        normalized_routes.append({field: values[field] for field in route_fields})
    truth_values["rubric_routes"] = normalized_routes
    normalized_truth = {field: truth_values[field] for field in MERGED_ROW_SCHEMA["ground_truth_fields"]}
    return {**row, "id": str(row["id"]), "rubrics": normalized_rubrics, "ground_truth": normalized_truth}


def _canonical_object_string(value: Any, label: str) -> str:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise RubricHubDataError(f"{label} is not canonical JSON") from error
        if not isinstance(decoded, dict) or value != _canonical_bytes(decoded).decode("utf-8"):
            raise RubricHubDataError(f"{label} is not a canonical JSON object")
        return value
    if not isinstance(value, dict):
        raise RubricHubDataError(f"{label} must be an object")
    try:
        return _canonical_bytes(value).decode("utf-8")
    except (TypeError, ValueError) as error:
        raise RubricHubDataError(f"{label} is not strict JSON") from error


def _load_language_certificate(
    path: Path,
    metas: list[RowMeta],
    config: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[int, LanguageResult], dict[str, Any]]:
    payload = _read_json(path)
    _verify_certificate_source(payload, source, "language")
    detector = payload.get("detector")
    expected = config["language_gate"]["detector"]
    if not isinstance(detector, dict) or any(detector.get(key) != value for key, value in expected.items()):
        raise RubricHubDataError("language certificate detector metadata differs from the frozen gate")
    expected_policy = {
        key: config["language_gate"][key]
        for key in (
            "required_base_task_language",
            "segment_minimum_confidence",
            "segment_minimum_characters",
            "segment_minimum_letters",
            "non_latin_minimum_letters",
            "non_latin_minimum_ratio",
        )
    }
    if payload.get("policy") != expected_policy:
        raise RubricHubDataError("language certificate policy differs from the frozen gate")
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != len(metas):
        raise RubricHubDataError("language certificate must cover every source row exactly once")
    results: dict[int, LanguageResult] = {}
    for item in rows:
        index = item.get("source_index")
        if not isinstance(index, int) or not 0 <= index < len(metas) or index in results:
            raise RubricHubDataError("language certificate contains an invalid or duplicate source index")
        if item.get("source_row_sha256") != metas[index].source_hash:
            raise RubricHubDataError(f"language certificate row {index} is stale")
        confidence = item.get("confidence")
        segments = item.get("segment_languages")
        flags = item.get("reason_flags")
        if (
            not isinstance(item.get("language"), str)
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
            or not isinstance(item.get("mixed"), bool)
            or not isinstance(segments, list)
            or not all(isinstance(value, str) and value for value in segments)
            or not isinstance(flags, list)
            or not all(isinstance(value, str) and value for value in flags)
        ):
            raise RubricHubDataError(f"language certificate row {index} is malformed")
        results[index] = LanguageResult(
            item["language"], float(confidence), item["mixed"], tuple(segments), tuple(flags)
        )
    return results, _certificate_ref(path, payload)


def _load_checker_certificate(
    path: Path | None,
    source: dict[str, Any],
    root: Path,
    metas: list[RowMeta],
    english: list[RowMeta],
    language_path: Path,
    reference_root: Any,
) -> tuple[set[str], set[int], dict[str, Any]]:
    if path is None:
        return set(), set(), {"status": "missing", "effect": "zero_rubrichub_rows_eligible"}
    payload = _read_json(path)
    _verify_certificate_source(payload, source, "checker")
    if payload.get("status") != "certified":
        raise RubricHubDataError("checker certificate status must be certified")
    routes = payload.get("routes")
    if not isinstance(routes, list):
        raise RubricHubDataError("checker certificate routes are missing")
    supported: set[str] = set()
    implementation = payload.get("implementation")
    if not isinstance(implementation, dict):
        raise RubricHubDataError("checker certificate implementation identity is missing")
    implementation_path = _resolve(root, implementation.get("path", ""))
    _verify_digest(implementation_path, implementation.get("sha256"), "checker implementation")
    generator = payload.get("generator")
    if not isinstance(generator, dict):
        raise RubricHubDataError("checker certificate generator identity is missing")
    generator_path = _resolve(root, generator.get("path", ""))
    _verify_digest(generator_path, generator.get("sha256"), "checker certificate generator")
    for route in routes:
        if not isinstance(route, dict):
            raise RubricHubDataError("checker certificate contains a malformed route")
        function, digest = route.get("function"), route.get("implementation_sha256")
        if function in supported or function not in source["rule_functions"] or digest != implementation["sha256"]:
            raise RubricHubDataError("checker certificate contains an invalid route")
        supported.add(function)
    _verify_checker_reference(payload.get("reference"), root, supported, reference_root)
    selection = payload.get("candidate_selection")
    if not isinstance(selection, dict):
        raise RubricHubDataError("checker certificate candidate selection is missing")
    expected = sorted(meta.index for meta in english if meta.functions and set(meta.functions).issubset(supported))
    indices = selection.get("source_indices")
    if indices != expected or selection.get("rows") != len(expected):
        raise RubricHubDataError("checker certificate candidate inventory differs from exact source rows")
    if selection.get("source_indices_sha256") != _value_sha256(expected):
        raise RubricHubDataError("checker certificate candidate index digest differs")
    hashes = [metas[index].source_hash for index in expected]
    if selection.get("source_row_hashes_sha256") != _value_sha256(hashes):
        raise RubricHubDataError("checker certificate candidate source hashes differ")
    _verify_file_ref(selection.get("language_certificate"), language_path, "checker language certificate")
    counts = payload.get("counts")
    if (
        not isinstance(counts, dict)
        or counts.get("candidate_records") != len(expected)
        or counts.get("probe_records") != 21
        or counts.get("total_records") != len(expected) + 21
    ):
        raise RubricHubDataError("checker certificate evidence counts differ")
    evidence_rows = _verify_evidence(
        payload.get("evidence"),
        root,
        counts["total_records"],
        "checker",
        read_rows=True,
    )
    probes = [item for item in evidence_rows if item.get("kind") == "probe"]
    candidates = [item for item in evidence_rows if item.get("kind") == "candidate"]
    if len(probes) != 21 or any(item.get("repeat_equal") is not True for item in probes):
        raise RubricHubDataError("checker probe evidence is incomplete or nondeterministic")
    expected_candidates = [
        {
            "source_index": index,
            "source_row_sha256": metas[index].source_hash,
            "functions": list(metas[index].functions),
        }
        for index in expected
    ]
    if [
        {key: item.get(key) for key in ("source_index", "source_row_sha256", "functions")} for item in candidates
    ] != expected_candidates:
        raise RubricHubDataError("checker evidence candidate rows differ from exact source rows")
    return supported, set(expected), _certificate_ref(path, payload)


def _load_tokenizer_certificate(
    path: Path | None,
    metas: list[RowMeta],
    source: dict[str, Any],
    root: Path,
    checker_path: Path | None,
    language_path: Path,
    checker_rows: set[int],
    frozen_policy: Any,
) -> tuple[dict[int, bool], dict[str, Any]]:
    if path is None:
        return {}, {"status": "missing", "effect": "zero_rubrichub_rows_eligible"}
    if checker_path is None or not checker_rows:
        raise RubricHubDataError("tokenizer certificate requires a nonempty checker-certified subset")
    payload = _read_json(path)
    _verify_certificate_source(payload, source, "tokenizer")
    if payload.get("status") != "frozen" or not isinstance(frozen_policy, dict):
        raise RubricHubDataError("tokenizer certificate status or frozen policy is invalid")
    if payload.get("tokenizer") != frozen_policy:
        raise RubricHubDataError("tokenizer certificate differs from the frozen policy")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("transformers") != frozen_policy.get("transformers_version"):
        raise RubricHubDataError("tokenizer certificate runtime differs from the frozen policy")
    generator = payload.get("generator")
    if not isinstance(generator, dict):
        raise RubricHubDataError("tokenizer certificate generator identity is missing")
    _verify_digest(
        _resolve(root, generator.get("path", "")),
        generator.get("sha256"),
        "tokenizer certificate generator",
    )
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise RubricHubDataError("tokenizer certificate selection is missing")
    _verify_file_ref(selection.get("checker_certificate"), checker_path, "tokenizer checker certificate")
    _verify_file_ref(selection.get("language_certificate"), language_path, "tokenizer language certificate")
    expected = sorted(checker_rows)
    if selection.get("candidate_rows") != len(expected) or selection.get("candidate_indices_sha256") != _value_sha256(
        expected
    ):
        raise RubricHubDataError("tokenizer candidate index inventory differs")
    if selection.get("candidate_source_hashes_sha256") != _value_sha256(
        [metas[index].source_hash for index in expected]
    ):
        raise RubricHubDataError("tokenizer candidate source hashes differ")
    evidence_rows = _verify_evidence(payload.get("evidence"), root, len(expected), "tokenizer", read_rows=True)
    accepted: dict[int, bool] = {}
    for item in evidence_rows:
        index = item.get("source_index")
        if not isinstance(index, int) or index not in checker_rows or index in accepted:
            raise RubricHubDataError("tokenizer certificate contains an invalid row")
        if item.get("source_row_sha256") != metas[index].source_hash or not isinstance(item.get("accepted"), bool):
            raise RubricHubDataError(f"tokenizer certificate row {index} is stale or malformed")
        tokens = item.get("input_tokens")
        expected_acceptance = isinstance(tokens, int) and not isinstance(tokens, bool) and 5 < tokens <= 2_048
        if item["accepted"] != expected_acceptance:
            raise RubricHubDataError(f"tokenizer certificate row {index} violates the frozen length policy")
        accepted[index] = item["accepted"]
    if sorted(accepted) != expected:
        raise RubricHubDataError("tokenizer evidence does not cover every checker candidate")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise RubricHubDataError("tokenizer compact result summary is missing")
    accepted_indices = [index for index in expected if accepted[index]]
    rejected_rows = [item for item in evidence_rows if item["accepted"] is False]
    if (
        results.get("candidates") != len(expected)
        or results.get("accepted") != len(accepted_indices)
        or results.get("rejected") != len(rejected_rows)
        or results.get("accepted_source_indices") != accepted_indices
        or results.get("accepted_source_indices_sha256") != _value_sha256(accepted_indices)
        or results.get("rejected_rows") != rejected_rows
    ):
        raise RubricHubDataError("tokenizer compact accepted or rejected inventory differs from evidence")
    return accepted, _certificate_ref(path, payload)


def _verify_certificate_source(payload: dict[str, Any], source: dict[str, Any], kind: str) -> None:
    expected = _certificate_source(source)
    if payload.get("schema_version") != 1 or payload.get("source") != expected:
        raise RubricHubDataError(f"{kind} certificate source differs from the frozen dataset")


def _certificate_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "file": source["file"],
        "sha256": source["sha256"],
        "records": source["records"],
    }


def _predict_language(model: Any, text: str) -> tuple[str, float]:
    labels, probabilities = model.predict(" ".join(text.split()), k=1)
    if not labels or not probabilities or not labels[0].startswith("__label__"):
        return "unknown", 0.0
    probability = min(1.0, max(0.0, float(probabilities[0])))
    return labels[0].removeprefix("__label__"), round(probability, 8)


def _language_segments(prompt: str, gate: dict[str, Any]) -> list[str]:
    pieces = re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", prompt)
    segments = [
        piece.strip()
        for piece in pieces
        if len(piece.strip()) >= gate["segment_minimum_characters"]
        and sum(character.isalpha() for character in piece) >= gate["segment_minimum_letters"]
    ]
    words = prompt.split()
    for start in range(0, max(len(words) - 39, 0), 20):
        window = " ".join(words[start : start + 40])
        if len(window) >= gate["segment_minimum_characters"]:
            segments.append(window)
    return list(dict.fromkeys(segments))


def _base_task(row: dict[str, Any]) -> str:
    criteria = [
        rubric["criterion"] for rubric in row["reward_model"]["rubrics"] if rubric["tags"]["verifier"] == "llm"
    ]
    if len(criteria) != 1 or not criteria[0].startswith(BASE_TASK_PREFIX):
        raise RubricHubDataError("LLM criterion does not contain the frozen base-task prefix")
    task = criteria[0][len(BASE_TASK_PREFIX) :]
    if not task.strip():
        raise RubricHubDataError("LLM criterion contains an empty base task")
    return task


def _training_text(row: dict[str, Any]) -> str:
    texts = [row["prompt"][0]["content"]]
    reward = row["reward_model"]
    texts.extend(_text_values(reward.get("ground_truth")))
    texts.extend(_text_values(reward.get("style")))
    for rubric in reward["rubrics"]:
        texts.append(rubric["criterion"])
        parameters = rubric["tags"]["parameters"] or {}
        texts.extend(_text_values(parameters))
    return "\n".join(texts)


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _text_values(item)]
    if isinstance(value, dict):
        return [text for key in sorted(value) for text in _text_values(value[key])]
    return []


def _required_output_languages(row: dict[str, Any]) -> list[str]:
    languages: list[str] = []
    for rubric in row["reward_model"]["rubrics"]:
        if rubric["tags"]["function"] != "ResponseLanguageChecker":
            continue
        language = (rubric["tags"]["parameters"] or {}).get("language")
        if not isinstance(language, str) or not language.strip():
            raise RubricHubDataError("ResponseLanguageChecker has no explicit language parameter")
        languages.append(language.strip())
    return languages


def _non_latin_scripts(text: str) -> dict[str, int]:
    scripts: Counter[str] = Counter()
    for character in text:
        if not character.isalpha():
            continue
        decomposed_letters = [value for value in unicodedata.normalize("NFKD", character) if value.isalpha()]
        if decomposed_letters and all("LATIN" in unicodedata.name(value, "") for value in decomposed_letters):
            continue
        if unicodedata.category(character) == "Lm":
            continue
        name = unicodedata.name(character, "UNKNOWN")
        script = next(
            (
                candidate
                for candidate in (
                    "CJK",
                    "HIRAGANA",
                    "KATAKANA",
                    "HANGUL",
                    "CYRILLIC",
                    "ARABIC",
                    "DEVANAGARI",
                    "HEBREW",
                    "THAI",
                    "GREEK",
                    "BENGALI",
                    "TAMIL",
                    "TELUGU",
                    "GUJARATI",
                    "GURMUKHI",
                    "ARMENIAN",
                    "GEORGIAN",
                    "ETHIOPIC",
                    "LAO",
                    "MYANMAR",
                    "KHMER",
                    "SINHALA",
                    "MALAYALAM",
                    "KANNADA",
                    "ORIYA",
                )
                if candidate in name
            ),
            "OTHER",
        )
        scripts[script] += 1
    return dict(sorted(scripts.items()))


def _foreign_language_scripts(text: str) -> dict[str, int]:
    scripts = _non_latin_scripts(text)
    if "GREEK" in scripts and not _has_greek_word_run(text):
        del scripts["GREEK"]
    return scripts


def _has_greek_word_run(text: str) -> bool:
    run = 0
    for character in text:
        if character.isalpha() and "GREEK" in unicodedata.name(character, ""):
            run += 1
            if run >= 2:
                return True
        elif not unicodedata.combining(character):
            run = 0
    return False


def _build_manifest(**values: Any) -> dict[str, Any]:
    config_path, root, config, source = (
        values["config_path"],
        values["root"],
        values["config"],
        values["source"],
    )
    metas, english = values["metas"], values["english"]
    reasons: dict[int, list[str]] = values["reasons"]
    reason_counts = Counter(reason for row_reasons in reasons.values() for reason in row_reasons)
    language_exclusions = values["language_exclusions"]
    language_ref = dict(values["language_ref"])
    language_ref["path"] = _relative(root, Path(language_ref["path"]))
    checker_ref = dict(values["checker_ref"])
    tokenizer_ref = dict(values["tokenizer_ref"])
    if "path" in checker_ref:
        checker_ref["path"] = _relative(root, Path(checker_ref["path"]))
    if "path" in tokenizer_ref:
        tokenizer_ref["path"] = _relative(root, Path(tokenizer_ref["path"]))
    outputs = {}
    for key, reference in values["output_refs"].items():
        outputs[key] = {**reference, "path": _relative(root, Path(reference["path"]))}
    return {
        "schema_version": 1,
        "id": "qwen_hir_rubrichub_if_hybrid_v1",
        "status": "archive_ready_rl_fail_closed" if not outputs["rubrichub_eligible"]["records"] else "eligible",
        "config": {"path": _relative(root, config_path), "sha256": _sha256(config_path)},
        "row_schema": config["row_schema"],
        "sources": {
            "hir": config["hir"],
            "rubrichub": {
                **source,
                "observed_rule_function_counts": dict(sorted(values["function_counts"].items())),
            },
        },
        "language_gate": {
            "certificate": language_ref,
            "accepted_english_rows": len(english),
            "excluded_rows": len(language_exclusions),
            "exclusion_reason_counts": dict(sorted(Counter(item["reason"] for item in language_exclusions).items())),
            "excluded_source_indices_sha256": hashlib.sha256(
                _canonical_bytes([item["source_index"] for item in language_exclusions])
            ).hexdigest(),
            "excluded_source_hashes_sha256": hashlib.sha256(
                _canonical_bytes([item["source_row_sha256"] for item in language_exclusions])
            ).hexdigest(),
            "documented_exclusions": config["language_gate"].get("documented_exclusions", []),
        },
        "deduplication": {
            "normalization": "NFKC, CRLF to LF, collapsed whitespace; casefold tracked separately",
            "source_rows": len(metas),
            "payload_duplicates": values["duplicate_evidence"],
            "collisions": values["collision_evidence"],
            "source_priority": "HIR before RubricHub",
        },
        "benchmark_quarantine": {
            "reports": values["benchmark_report"],
            "lineage_note": (
                "RubricHub instruction-following data derives from IFTRAIN and shares constraint lineage "
                "with IFEval and IFBench."
            ),
        },
        "rl_eligibility": {
            "checker_certificate": checker_ref,
            "tokenizer_certificate": tokenizer_ref,
            "rubrichub_eligible_rows": outputs["rubrichub_eligible"]["records"],
            "quarantine_reason_counts": dict(sorted(reason_counts.items())),
            "unsupported_route_counts": dict(sorted(values["unsupported_routes"].items())),
            "policy": "every rule route certified, tokenizer accepted, and no dedup or benchmark collision",
        },
        "outputs": outputs,
    }


def _hir_qwen_inventory(manifest: dict[str, Any]) -> tuple[dict[str, Any], Any, list[Any]]:
    """Read the full-corpus Qwen gate while retaining synthetic legacy fixtures."""

    if manifest.get("id") == "qwen_rtt_hir_data_v1":
        source = manifest.get("sources", {}).get("rtt_processed", {})
        effective = manifest.get("derived", {}).get("records")
        excluded = [item.get("row_id") for item in manifest.get("row_ids", {}).get("excluded", [])]
        return source, effective, excluded
    source = manifest.get("data", {})
    preprocessing = manifest.get("preprocessing", {})
    source = {**source, "records": source.get("records", preprocessing.get("input_records"))}
    excluded = [item.get("row_id") for item in preprocessing.get("excluded", [])]
    return source, preprocessing.get("effective_records"), excluded


def _read_benchmark_prompts(path: Path, config: dict[str, Any]) -> list[str]:
    if config["format"] == "jsonl":
        rows = [row for row, _ in _read_jsonl_bytes(path)]
    elif config["format"] == "json":
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RubricHubDataError(f"cannot read JSON: {path}") from error
        if not isinstance(rows, list):
            raise RubricHubDataError(f"{config['id']} benchmark must be a JSON list")
    elif config["format"] == "csv":
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error) as error:
            raise RubricHubDataError(f"cannot read CSV: {path}") from error
    else:
        raise RubricHubDataError(f"{config['id']} benchmark format is unsupported")
    prompts: list[str] = []
    for row in rows:
        value: Any = row
        for part in config["prompt_field"].split("."):
            value = value[int(part)] if part.isdigit() else value[part]
        if config["format"] == "csv" and config["prompt_field"] == "conversation_history":
            try:
                messages = json.loads(value)
            except (TypeError, json.JSONDecodeError) as error:
                raise RubricHubDataError(f"{config['id']} benchmark contains invalid conversation JSON") from error
            if (
                not isinstance(messages, list)
                or not messages
                or any(
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                    for message in messages
                )
            ):
                raise RubricHubDataError(f"{config['id']} benchmark contains an invalid conversation")
            value = "\n".join(message["content"] for message in messages)
        if not isinstance(value, str) or not value.strip():
            raise RubricHubDataError(f"{config['id']} benchmark contains an empty prompt")
        prompts.append(value)
    return prompts


def _iter_parquet_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RubricHubDataError("pyarrow is required to convert RubricHub parquet") from error
    index = 0
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=2_048):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def _read_jsonl_bytes(path: Path) -> Iterator[tuple[dict[str, Any], bytes]]:
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                raise RubricHubDataError(f"{path}: blank line {line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise RubricHubDataError(f"{path}: invalid JSON on line {line_number}") from error
            if not isinstance(row, dict):
                raise RubricHubDataError(f"{path}: row {line_number} is not an object")
            yield row, raw if raw.endswith(b"\n") else raw + b"\n"


def _group(metas: Iterable[RowMeta], field: str) -> dict[str, list[RowMeta]]:
    groups: dict[str, list[RowMeta]] = defaultdict(list)
    for meta in metas:
        groups[getattr(meta, field)].append(meta)
    return groups


def _word_grams(value: str) -> set[str]:
    words = value.split()
    width = min(5, len(words))
    return {" ".join(words[index : index + width]) for index in range(len(words) - width + 1)} if width else set()


def _required_path(cli: str | Path | None, configured: Any, root: Path) -> Path:
    path = _optional_path(cli, configured, root)
    if path is None:
        raise RubricHubDataError("a frozen English language certificate is required before conversion")
    return path


def _optional_path(cli: str | Path | None, configured: Any, root: Path) -> Path | None:
    value = cli if cli is not None else configured
    return _resolve(root, value) if value else None


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), root)


def _verify_file(path: Path, size: int, digest: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
        raise RubricHubDataError(f"{label} is missing or differs from its frozen pin")


def _verify_digest(path: Path, digest: Any, label: str) -> None:
    if not _is_sha256(digest) or not path.is_file() or _sha256(path) != digest:
        raise RubricHubDataError(f"{label} is missing or differs from its frozen identity")


def _verify_checker_reference(reference: Any, root: Path, functions: set[str], configured_root: Any) -> None:
    if not isinstance(reference, dict) or reference.get("repository") != "TURLEing/Rubrics-To-Tokens":
        raise RubricHubDataError("checker RTT reference identity is malformed")
    checkout = _resolve(root, configured_root) if configured_root else root.parent / "Rubrics-To-Tokens"
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RubricHubDataError("checker RTT reference checkout is unavailable") from error
    if revision != reference.get("revision"):
        raise RubricHubDataError("checker RTT reference revision differs")
    source_path = checkout / str(reference.get("path", ""))
    _verify_digest(source_path, reference.get("source_sha256"), "checker RTT reference source")
    source = source_path.read_text(encoding="utf-8")
    classes = {
        node.name: hashlib.sha256((ast.get_source_segment(source, node) or "").encode()).hexdigest()
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name in functions
    }
    if classes != reference.get("class_sha256"):
        raise RubricHubDataError("checker RTT reference class sources differ")


def _verify_file_ref(reference: Any, path: Path, label: str) -> None:
    if not isinstance(reference, dict) or set(reference) != {"bytes", "sha256"}:
        raise RubricHubDataError(f"{label} reference is malformed")
    _verify_file(path, reference["bytes"], reference["sha256"], label)


def _verify_evidence(
    reference: Any,
    root: Path,
    records: int,
    label: str,
    *,
    read_rows: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(reference, dict) or set(reference) != {"path", "records", "bytes", "sha256"}:
        raise RubricHubDataError(f"{label} evidence reference is malformed")
    path = _resolve(root, reference["path"])
    _verify_file(path, reference["bytes"], reference["sha256"], f"{label} evidence")
    rows = [row for row, _ in _read_jsonl_bytes(path)]
    if reference["records"] != records or len(rows) != records:
        raise RubricHubDataError(f"{label} evidence row count differs")
    return rows if read_rows else []


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _certificate_ref(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "id": payload.get("id"),
        "sha256": _sha256(path),
        "status": payload.get("status", "frozen"),
    }


def _write_json_temp(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return temp
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RubricHubDataError(f"cannot read JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RubricHubDataError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _json_line(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
