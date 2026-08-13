import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")
datasets = pytest.importorskip("datasets")

from rdan_grpo.rubrichub_data import (  # noqa: E402
    MERGED_ROW_SCHEMA,
    RubricHubDataError,
    _foreign_language_scripts,
    _predict_language,
    _training_text,
    build_language_certificate,
    build_merged_rl_data,
    source_row_hash,
)

ROOT = Path(__file__).resolve().parents[1]


def test_english_rubrichub_merge_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    prompts = [
        "Explain why rainbows form. Use exactly two paragraphs.",
        "Explain why rainbows form. Use exactly two paragraphs.",
        "Test a small idea. Use exactly two paragraphs.",
        "test a small idea. Use exactly two paragraphs.",
        "解释彩虹为什么形成。 Use exactly two paragraphs.",
        "Expliquez pourquoi les arcs-en-ciel se forment. Use exactly two paragraphs.",
        "Benchmark prompt. Use exactly two paragraphs.",
        "HIR prompt. Use exactly two paragraphs.",
        "Explain photosynthesis. Answer in French.",
    ]
    base_tasks = [
        "Explain why rainbows form.",
        "Explain why rainbows form.",
        "Test a small idea.",
        "test a small idea.",
        "解释彩虹为什么形成。",
        "Expliquez pourquoi les arcs-en-ciel se forment.",
        "Benchmark prompt.",
        "HIR prompt.",
        "Explain photosynthesis.",
    ]
    rows = [_source_row(prompt, task) for prompt, task in zip(prompts, base_tasks, strict=True)]
    rows[-1] = _source_row(prompts[-1], base_tasks[-1], output_language="French")
    source_path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), source_path)
    observed = parquet.read_table(source_path).to_pylist()

    hir_path = tmp_path / "hir.jsonl"
    hir_rows = [_hir_row(10, "HIR prompt. Use exactly two paragraphs."), _hir_row(11, "Eligible HIR prompt.")]
    _write_jsonl(hir_path, hir_rows)
    qwen_manifest = {
        "data": {"sha256": _sha256(hir_path)},
        "preprocessing": {"effective_records": 1, "excluded": [{"row_id": 10}]},
    }
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(json.dumps(qwen_manifest), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.jsonl"
    _write_jsonl(benchmark_path, [{"prompt": "Benchmark prompt. Use exactly two paragraphs."}])

    config = _config(tmp_path, source_path, hir_path, qwen_path, benchmark_path, len(rows))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    source_ref = {
        "dataset": config["source"]["dataset"],
        "revision": config["source"]["revision"],
        "file": config["source"]["file"],
        "sha256": config["source"]["sha256"],
        "records": len(rows),
    }
    languages = ["en", "en", "en", "en", "zh", "fr", "en", "en", "en"]
    certificate = {
        "schema_version": 1,
        "id": "synthetic-language",
        "status": "frozen",
        "source": source_ref,
        "detector": config["language_gate"]["detector"],
        "policy": {
            key: config["language_gate"][key]
            for key in (
                "required_base_task_language",
                "segment_minimum_confidence",
                "segment_minimum_characters",
                "segment_minimum_letters",
                "non_latin_minimum_letters",
                "non_latin_minimum_ratio",
            )
        },
        "results": [
            {
                "source_index": index,
                "source_row_sha256": source_row_hash(row),
                "language": language,
                "confidence": 0.99,
                "mixed": language != "en" or index == 8,
                "segment_languages": [language],
                "reason_flags": (
                    ["non_english_output_requirement"]
                    if index == 8
                    else ["non_english_training_text"]
                    if language != "en"
                    else []
                ),
            }
            for index, (row, language) in enumerate(zip(observed, languages, strict=True))
        ],
    }
    language_path = tmp_path / "language.json"
    language_path.write_text(json.dumps(certificate, sort_keys=True), encoding="utf-8")

    first = build_merged_rl_data(config_path, repo_root=tmp_path)
    first_bytes = {key: Path(path).read_bytes() for key, path in config["outputs"].items()}
    second = build_merged_rl_data(config_path, repo_root=tmp_path)

    assert first == second
    assert first_bytes == {key: Path(path).read_bytes() for key, path in config["outputs"].items()}
    assert first["language_gate"]["accepted_english_rows"] == 6
    assert first["language_gate"]["excluded_rows"] == 3
    assert first["language_gate"]["exclusion_reason_counts"]["non_english_output_requirement"] == 1
    assert first["deduplication"]["payload_duplicates"][0]["kept_source_index"] == 0
    assert {item["kind"] for item in first["deduplication"]["collisions"]} == {
        "hir_exact",
        "rubrichub_casefold",
    }
    assert first["outputs"]["rubrichub_archive"]["records"] == 5
    assert first["outputs"]["rubrichub_eligible"]["records"] == 0
    assert first["outputs"]["merged_archive"]["records"] == 6
    assert first["outputs"]["merged_eligible"]["records"] == 1
    archive = [json.loads(line) for line in (tmp_path / "rubrichub.jsonl").read_text().splitlines()]
    assert all("解释" not in row["prompt"] and "Expliquez" not in row["prompt"] for row in archive)
    assert json.loads(archive[0]["ground_truth"]["rubric_routes"][0]["parameters"]) == {"num_paragraphs": 2.0}
    assert first["benchmark_quarantine"]["reports"][0]["exact_matches"][0]["source_index"] == 6
    assert first["rl_eligibility"]["rubrichub_eligible_rows"] == 0


def test_malformed_rubrichub_schema_fails_closed(tmp_path: Path) -> None:
    row = _source_row("English task. Use exactly two paragraphs.", "English task.")
    row["extra_info"]["prompt"] = [{"content": "changed", "role": "user"}]
    source_path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.Table.from_pylist([row]), source_path)
    observed = parquet.read_table(source_path).to_pylist()
    hir_path = tmp_path / "hir.jsonl"
    _write_jsonl(hir_path, [_hir_row(1, "HIR")])
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(
        json.dumps({"data": {"sha256": _sha256(hir_path)}, "preprocessing": {"effective_records": 1, "excluded": []}}),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "bench.jsonl"
    _write_jsonl(benchmark_path, [{"prompt": "benchmark"}])
    config = _config(tmp_path, source_path, hir_path, qwen_path, benchmark_path, 1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RubricHubDataError, match="mirror differs"):
        build_merged_rl_data(config_path, repo_root=tmp_path, language_certificate=tmp_path / "missing.json")
    assert observed[0]["prompt"] != observed[0]["extra_info"]["prompt"]


def test_short_non_english_script_is_excluded_even_when_certificate_calls_it_english(tmp_path: Path) -> None:
    rows = [
        _source_row("Explain a small idea. Ref: 中", "Explain a small idea."),
        _source_row("Explain another idea with x and π.", "Explain another idea with x and π."),
    ]
    source_path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), source_path)
    observed = parquet.read_table(source_path).to_pylist()
    hir_path = tmp_path / "hir.jsonl"
    hir_rows = [_hir_row(10, "Excluded HIR"), _hir_row(11, "Eligible HIR")]
    _write_jsonl(hir_path, hir_rows)
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(
        json.dumps(
            {
                "data": {"sha256": _sha256(hir_path)},
                "preprocessing": {"effective_records": 1, "excluded": [{"row_id": 10}]},
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "bench.jsonl"
    _write_jsonl(benchmark_path, [{"prompt": "benchmark"}])
    config = _config(tmp_path, source_path, hir_path, qwen_path, benchmark_path, len(rows))
    language_path = tmp_path / "language.json"
    language_path.write_text(
        json.dumps(_language_certificate(config, observed, ["en", "en"]), sort_keys=True),
        encoding="utf-8",
    )
    config["language_certificate"] = str(language_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    manifest = build_merged_rl_data(config_path, repo_root=tmp_path)

    assert manifest["language_gate"]["accepted_english_rows"] == 1
    assert manifest["language_gate"]["exclusion_reason_counts"] == {"non_english_script": 1}
    archive = [json.loads(line) for line in (tmp_path / "rubrichub.jsonl").read_text().splitlines()]
    assert [row["prompt"] for row in archive] == ["Explain another idea with x and π."]
    merged = [json.loads(line) for line in (tmp_path / "merged.jsonl").read_text().splitlines()]
    assert [row["id"] for row in merged[:2]] == ["10", "11"]


@pytest.mark.parametrize(
    ("field", "text", "script"),
    [
        ("style", "да", "CYRILLIC"),
        ("ground_truth", "日", "CJK"),
        ("criterion", "あ", "HIRAGANA"),
        ("parameter", "안", "HANGUL"),
        ("style", "Ꭰ", "OTHER"),
    ],
)
def test_all_human_readable_fields_are_checked_for_short_foreign_scripts(
    field: str,
    text: str,
    script: str,
) -> None:
    row = _source_row("Explain an English idea.", "Explain an English idea.")
    reward = row["reward_model"]
    if field == "style":
        reward["style"] = text
    elif field == "ground_truth":
        reward["ground_truth"] = text
    elif field == "criterion":
        reward["rubrics"][0]["criterion"] += text
    else:
        reward["rubrics"][0]["tags"]["parameters"]["keyword"] = text

    scripts = _foreign_language_scripts(_training_text(row))
    assert set(scripts) == {script}
    assert scripts[script] >= 1


def test_math_letters_and_phonetic_marks_are_not_treated_as_foreign_language() -> None:
    assert _foreign_language_scripts("Let 𝔸 be a set and π be a constant. Pronounce /ˈtest/.") == {}


def test_greek_words_are_treated_as_foreign_language() -> None:
    assert _foreign_language_scripts("Αυτή είναι μια ελληνική πρόταση.") == {"GREEK": 27}


def test_advancedif_exact_casefold_and_near_matches_quarantine_all_training_fields(tmp_path: Path) -> None:
    hir_prompt = "Write an exact HIR answer with seven distinct useful words."
    rh_prompt = "Write an exact RubricHub answer with seven distinct useful words."
    hir_case = "Respect This HIR Criterion With Enough Distinct Words For Matching"
    hir_near = "Explain this HIR rubric using several precise distinct words for reliable comparison"
    rh_case = "Respect This RubricHub Criterion With Enough Distinct Words For Matching"
    rh_near = "Does the response address the follow question? \\n\\nExplain this RubricHub task using precise words"
    rows = [
        _source_row(rh_prompt, "Explain this RubricHub task using precise words", rule_function="CommaChecker"),
        _source_row("A clean RubricHub training prompt.", "Explain a clean task", rule_function="CommaChecker"),
    ]
    rows[0]["reward_model"]["rubrics"][0]["criterion"] = rh_case
    rows[0]["reward_model"]["rubrics"][1]["criterion"] = rh_near
    rows[0]["Rubrics"] = [
        {"criterion": item["criterion"], "points": item["points"]} for item in rows[0]["reward_model"]["rubrics"]
    ]
    source_path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), source_path)
    observed = parquet.read_table(source_path).to_pylist()

    hir_path = tmp_path / "hir.jsonl"
    excluded = _hir_row(10, "Excluded HIR")
    hir = _hir_row(11, hir_prompt)
    hir["rubrics"] = [
        {"id": 1, "category": "", "description": hir_case, "weight": 1},
        {"id": 2, "category": "", "description": hir_near, "weight": 1},
    ]
    hir["ground_truth"] = {"checker": ["judge", "judge"], "functions": ["soft", "soft"]}
    _write_jsonl(hir_path, [excluded, hir])
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(
        json.dumps(
            {
                "data": {"sha256": _sha256(hir_path)},
                "preprocessing": {"effective_records": 1, "excluded": [{"row_id": 10}]},
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "bench.jsonl"
    _write_jsonl(benchmark_path, [{"prompt": "unrelated synthetic benchmark prompt"}])
    advanced_path = tmp_path / "advanced.csv"
    _write_advancedif(
        advanced_path,
        [
            {"role": "user", "content": hir_prompt},
            {"role": "user", "content": rh_prompt},
        ],
        [
            hir_case.swapcase(),
            f"{hir_near} now",
            rh_case.swapcase(),
            f"{rh_near} now",
        ],
    )

    config = _config(tmp_path, source_path, hir_path, qwen_path, benchmark_path, len(rows))
    config["benchmarks"].append(
        {
            "id": "AdvancedIF",
            "path": str(advanced_path),
            "format": "csv",
            "prompt_field": "conversation_history",
            "records": 1,
            "bytes": advanced_path.stat().st_size,
            "sha256": _sha256(advanced_path),
        }
    )
    language_path = tmp_path / "language.json"
    language_path.write_text(
        json.dumps(_language_certificate(config, observed, ["en", "en"]), sort_keys=True),
        encoding="utf-8",
    )
    config["language_certificate"] = str(language_path)
    config["checker_reference_root"] = str(ROOT.parent / "Rubrics-To-Tokens")
    checker_path, tokenizer_path = _compact_certificates(tmp_path, config, observed, language_path)
    config["checker_certificate"] = str(checker_path)
    config["tokenizer_certificate"] = str(tokenizer_path)
    config["require_nonempty_rubrichub_eligible"] = True
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    manifest = build_merged_rl_data(config_path, repo_root=tmp_path)

    report = next(item for item in manifest["benchmark_quarantine"]["reports"] if item["id"] == "AdvancedIF")
    for kind in ("exact_matches", "casefold_matches", "near_matches"):
        assert {item["training_source"] for item in report[kind]} == {"hir", "rubrichub"}
    assert report["comparison_fields"] == ["message_content", "role_preserving_transcript", "rubric"]
    assert report["benchmark_field_count"] == 7
    assert report["training_field_count"] == 9
    assert report["contamination_policy"] == "all_training_fields_exact_casefold_word_5gram_jaccard_v1"
    merged = [json.loads(line) for line in (tmp_path / "merged.jsonl").read_text().splitlines()]
    quarantined = {
        row["id"]: row["ground_truth"]
        for row in merged
        if row["id"] in {"11"} or row["id"].startswith("rubrichub-if-00000")
    }
    assert set(quarantined["11"]["quarantine_reasons"]) == {
        "benchmark_casefold:AdvancedIF",
        "benchmark_exact:AdvancedIF",
        "benchmark_near:AdvancedIF",
    }
    rh_truth = next(value for key, value in quarantined.items() if key != "11")
    assert set(rh_truth["quarantine_reasons"]) == {
        "benchmark_casefold:AdvancedIF",
        "benchmark_exact:AdvancedIF",
        "benchmark_near:AdvancedIF",
    }
    assert not quarantined["11"]["rl_eligible"]
    assert not rh_truth["rl_eligible"]
    eligible = [json.loads(line) for line in (tmp_path / "merged_eligible.jsonl").read_text().splitlines()]
    assert [row["prompt"] for row in eligible] == ["A clean RubricHub training prompt."]


def test_standard_benchmark_quarantines_all_training_fields(tmp_path: Path) -> None:
    hir_prompt = "Write one exact HIR response with enough distinct words for matching."
    hir_case = "Honor This HIR Rubric With Enough Distinct Words For Exact Matching"
    hir_near = "Explain this HIR rubric using several precise distinct words for reliable comparison"
    rh_prompt = "Write one exact RubricHub response with enough distinct words for matching."
    rh_case = "Honor This RubricHub Rubric With Enough Distinct Words For Exact Matching"
    rows = [
        _source_row(rh_prompt, "Explain one RubricHub task", rule_function="CommaChecker"),
        _source_row("A clean RubricHub training prompt.", "Explain a clean task", rule_function="CommaChecker"),
    ]
    rows[0]["reward_model"]["rubrics"][0]["criterion"] = rh_case
    rh_near = rows[0]["reward_model"]["rubrics"][1]["criterion"]
    rows[0]["Rubrics"] = [
        {"criterion": item["criterion"], "points": item["points"]} for item in rows[0]["reward_model"]["rubrics"]
    ]
    source_path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), source_path)
    observed = parquet.read_table(source_path).to_pylist()

    hir_path = tmp_path / "hir.jsonl"
    excluded = _hir_row(10, "Excluded HIR")
    hir = _hir_row(11, hir_prompt)
    hir["rubrics"] = [
        {"id": 1, "category": "", "description": hir_case, "weight": 1},
        {"id": 2, "category": "", "description": hir_near, "weight": 1},
    ]
    hir["ground_truth"] = {"checker": ["judge", "judge"], "functions": ["soft", "soft"]}
    _write_jsonl(hir_path, [excluded, hir])
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(
        json.dumps(
            {
                "data": {"sha256": _sha256(hir_path)},
                "preprocessing": {"effective_records": 1, "excluded": [{"row_id": 10}]},
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_prompts = [
        hir_prompt,
        hir_case.swapcase(),
        f"{hir_near} now",
        rh_prompt,
        rh_case.swapcase(),
        f"{rh_near} now",
    ]
    _write_jsonl(benchmark_path, [{"prompt": prompt} for prompt in benchmark_prompts])

    config = _config(tmp_path, source_path, hir_path, qwen_path, benchmark_path, len(rows))
    config["benchmarks"][0]["id"] = "IFEval"
    config["benchmarks"][0]["records"] = len(benchmark_prompts)
    language_path = tmp_path / "language.json"
    language_path.write_text(
        json.dumps(_language_certificate(config, observed, ["en", "en"]), sort_keys=True), encoding="utf-8"
    )
    config["language_certificate"] = str(language_path)
    config["checker_reference_root"] = str(ROOT.parent / "Rubrics-To-Tokens")
    checker_path, tokenizer_path = _compact_certificates(tmp_path, config, observed, language_path)
    config["checker_certificate"] = str(checker_path)
    config["tokenizer_certificate"] = str(tokenizer_path)
    config["require_nonempty_rubrichub_eligible"] = True
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    manifest = build_merged_rl_data(config_path, repo_root=tmp_path)

    report = manifest["benchmark_quarantine"]["reports"][0]
    for kind in ("exact_matches", "casefold_matches", "near_matches"):
        assert {item["training_source"] for item in report[kind]} == {"hir", "rubrichub"}
        assert all("source_row_sha256" in item and "benchmark_text_sha256" in item for item in report[kind])
    assert report["comparison_fields"] == ["prompt"]
    assert report["benchmark_field_count"] == 6
    assert report["training_field_count"] == 9
    assert report["contamination_policy"] == "all_training_fields_exact_casefold_word_5gram_jaccard_v1"
    merged = [json.loads(line) for line in (tmp_path / "merged.jsonl").read_text().splitlines()]
    quarantined = [row for row in merged if row["id"] == "11" or row["id"].startswith("rubrichub-if-00000")]
    expected = {"benchmark_exact:IFEval", "benchmark_casefold:IFEval", "benchmark_near:IFEval"}
    assert all(set(row["ground_truth"]["quarantine_reasons"]) == expected for row in quarantined)
    eligible = [json.loads(line) for line in (tmp_path / "merged_eligible.jsonl").read_text().splitlines()]
    assert [row["prompt"] for row in eligible] == ["A clean RubricHub training prompt."]


def test_compact_certificates_enable_only_exact_source_rows_and_reject_tampering(tmp_path: Path) -> None:
    rows = [
        _source_row(
            "Write a brief explanation without commas.",
            "Explain one idea.",
            rule_function="CommaChecker",
            rule_parameters={"forbidden": [","]},
        ),
        _source_row(
            "Write another explanation without commas.",
            "Explain another idea.",
            rule_function="CommaChecker",
            rule_parameters={"case_sensitive": True, "limit": 0},
        ),
    ]
    source_path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), source_path)
    observed = parquet.read_table(source_path).to_pylist()
    hir_path = tmp_path / "hir.jsonl"
    _write_jsonl(hir_path, [_hir_row(10, "Excluded HIR"), _hir_row(11, "HIR")])
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(
        json.dumps(
            {
                "data": {"sha256": _sha256(hir_path)},
                "preprocessing": {"effective_records": 1, "excluded": [{"row_id": 10}]},
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "bench.jsonl"
    _write_jsonl(benchmark_path, [{"prompt": "benchmark"}])
    config = _config(tmp_path, source_path, hir_path, qwen_path, benchmark_path, len(rows))
    language_path = tmp_path / "language.json"
    language_path.write_text(
        json.dumps(_language_certificate(config, observed, ["en", "en"]), sort_keys=True),
        encoding="utf-8",
    )
    config["language_certificate"] = str(language_path)
    config["checker_reference_root"] = str(ROOT.parent / "Rubrics-To-Tokens")
    checker_path, tokenizer_path = _compact_certificates(tmp_path, config, observed, language_path)
    config["checker_certificate"] = str(checker_path)
    config["tokenizer_certificate"] = str(tokenizer_path)
    config["require_nonempty_rubrichub_eligible"] = True
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    manifest = build_merged_rl_data(config_path, repo_root=tmp_path)
    assert manifest["rl_eligibility"]["rubrichub_eligible_rows"] == 2
    assert manifest["outputs"]["merged_eligible"]["records"] == 3
    assert manifest["row_schema"] == MERGED_ROW_SCHEMA

    merged_path = tmp_path / "merged_eligible.jsonl"
    merged_rows = [json.loads(line) for line in merged_path.read_text().splitlines()]
    assert all(isinstance(item["id"], str) for item in merged_rows)
    assert all(
        list(rubric) == MERGED_ROW_SCHEMA["rubric_fields"] for item in merged_rows for rubric in item["rubrics"]
    )
    assert all(list(item["ground_truth"]) == MERGED_ROW_SCHEMA["ground_truth_fields"] for item in merged_rows)
    assert merged_rows[0]["source"] == "type4"
    by_source = {item["source"]: item for item in merged_rows}
    assert by_source["type4"]["ground_truth"]["checker"] == ["judge"]
    assert by_source["type4"]["ground_truth"]["functions"] == ["soft"]
    rubrichub = [item for item in merged_rows if item["source"] == "rubrichub_instruction_following"]
    assert [json.loads(item["rubrics"][0]["parameters"]) for item in rubrichub] == [
        {"forbidden": [","]},
        {"case_sensitive": True, "limit": 0.0},
    ]
    assert all(
        item["rubrics"][0]["parameters"] == item["ground_truth"]["rubric_routes"][0]["parameters"]
        for item in rubrichub
    )
    assert all(item["ground_truth"]["rubric_routes"][0]["function"] == "CommaChecker" for item in rubrichub)

    loaded = datasets.load_dataset(
        "json",
        data_files=[str(merged_path)],
        cache_dir=str(tmp_path / "datasets-cache"),
        chunksize=512,
    )["train"]
    assert len(loaded) == 3
    assert loaded.features["id"].dtype == "string"
    assert all(isinstance(rubric["parameters"], str) for item in loaded for rubric in item["rubrics"])
    assert all(
        isinstance(route["parameters"], str) for item in loaded for route in item["ground_truth"]["rubric_routes"]
    )

    token = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    token["selection"]["candidate_source_hashes_sha256"] = "0" * 64
    tokenizer_path.write_text(json.dumps(token, sort_keys=True), encoding="utf-8")
    with pytest.raises(RubricHubDataError, match="candidate source hashes"):
        build_merged_rl_data(config_path, repo_root=tmp_path)


def test_fasttext_probability_roundoff_is_clamped() -> None:
    class Model:
        def predict(self, text: str, k: int) -> tuple[list[str], list[float]]:
            return ["__label__en"], [1.0001]

    assert _predict_language(Model(), "English") == ("en", 1.0)


def test_language_model_hash_pin_fails_closed(tmp_path: Path) -> None:
    row = _source_row("English task. Use exactly two paragraphs.", "English task.")
    source_path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.Table.from_pylist([row]), source_path)
    hir_path = tmp_path / "hir.jsonl"
    _write_jsonl(hir_path, [_hir_row(1, "HIR")])
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(
        json.dumps({"data": {"sha256": _sha256(hir_path)}, "preprocessing": {"effective_records": 1, "excluded": []}}),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "bench.jsonl"
    _write_jsonl(benchmark_path, [{"prompt": "benchmark"}])
    config = _config(tmp_path, source_path, hir_path, qwen_path, benchmark_path, 1)
    config["language_gate"]["detector"]["model_sha256"] = "missing-final-hex"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")
    with pytest.raises(RubricHubDataError, match="language model.*frozen pin"):
        build_language_certificate(config_path, model, tmp_path / "language.json", repo_root=tmp_path)


def _source_row(
    prompt: str,
    base_task: str,
    output_language: str | None = None,
    *,
    rule_function: str = "ParagraphBasicChecker",
    rule_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = (
        rule_parameters
        if rule_parameters is not None
        else {"num_paragraphs": 2.0, "keyword": None}
        if rule_function == "ParagraphBasicChecker"
        else {"unused": None}
    )
    rubrics = [
        {
            "criterion": "paragraphs:two",
            "points": 10,
            "tags": {"function": rule_function, "parameters": params, "verifier": "rule"},
        },
        {
            "criterion": f"Does the response address the follow question? \\n\\n{base_task}",
            "points": 10,
            "tags": {"function": "", "parameters": params, "verifier": "llm"},
        },
    ]
    if output_language:
        rubrics.insert(
            1,
            {
                "criterion": f"language:{output_language}",
                "points": 10,
                "tags": {
                    "function": "ResponseLanguageChecker",
                    "parameters": {"language": output_language},
                    "verifier": "rule",
                },
            },
        )
    messages = [{"content": prompt, "role": "user"}]
    reward = {"ground_truth": "", "rubrics": rubrics, "style": "rubric"}
    return {
        "prompt": messages,
        "data_source": "Instruction_Following",
        "ability": "Instruction_Following",
        "reward_model": reward,
        "extra_info": {"prompt": messages, "reward_model": reward},
        "Rubrics": [{"criterion": item["criterion"], "points": item["points"]} for item in rubrics],
    }


def _hir_row(row_id: int, prompt: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": "type4",
        "prompt": prompt,
        "question": prompt,
        "rubrics": [{"id": 1, "category": "", "description": "Answer the question.", "weight": 1}],
        "tag": "llm_judge",
        "difficulty": 0,
        "ground_truth": {"checker": ["judge"], "functions": ["soft"]},
        "messages": [{"role": "user", "content": prompt}],
    }


def _config(
    root: Path,
    source: Path,
    hir: Path,
    qwen: Path,
    benchmark: Path,
    source_records: int,
) -> dict[str, Any]:
    functions = sorted(
        {
            rubric["tags"]["function"]
            for row in parquet.read_table(source).to_pylist()
            for rubric in row["reward_model"]["rubrics"]
            if rubric["tags"]["function"]
        }
    )
    detector = {
        "name": "synthetic",
        "package": "synthetic",
        "package_version": "1",
        "python_version": "1",
        "numpy_version": "1",
        "model_url": "https://example.invalid/model",
        "model_bytes": 1,
        "model_sha256": "0" * 64,
        "model_license": "test",
        "policy_version": "test-v1",
    }
    return {
        "row_schema": MERGED_ROW_SCHEMA,
        "source": {
            "id": "synthetic",
            "dataset": "synthetic",
            "url": "https://example.invalid",
            "revision": "rev",
            "file": "source.parquet",
            "path": str(source),
            "license": "test",
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "records": source_records,
            "rule_function_count": len(functions),
            "rule_functions": functions,
        },
        "hir": {
            "path": str(hir),
            "manifest_path": "unused",
            "manifest_sha256": "0" * 64,
            "bytes": hir.stat().st_size,
            "sha256": _sha256(hir),
            "records": 2 if source_records > 1 else 1,
            "qwen_effective_manifest_path": str(qwen),
            "qwen_effective_manifest_sha256": _sha256(qwen),
            "qwen_effective_records": 1,
            "qwen_excluded_row_ids": [10] if source_records > 1 else [],
        },
        "language_gate": {
            "required_base_task_language": "en",
            "segment_minimum_confidence": 0.8,
            "segment_minimum_characters": 80,
            "segment_minimum_letters": 30,
            "non_latin_minimum_letters": 8,
            "non_latin_minimum_ratio": 0.05,
            "detector": detector,
        },
        "language_certificate": str(root / "language.json"),
        "checker_certificate": None,
        "tokenizer_certificate": None,
        "benchmarks": [
            {
                "id": "SyntheticBench",
                "path": str(benchmark),
                "format": "jsonl",
                "prompt_field": "prompt",
                "records": 1,
                "bytes": benchmark.stat().st_size,
                "sha256": _sha256(benchmark),
            }
        ],
        "outputs": {
            "rubrichub_archive": str(root / "rubrichub.jsonl"),
            "rubrichub_eligible": str(root / "rubrichub_eligible.jsonl"),
            "merged_archive": str(root / "merged.jsonl"),
            "merged_eligible": str(root / "merged_eligible.jsonl"),
            "manifest": str(root / "manifest.json"),
        },
    }


def _language_certificate(config: dict[str, Any], rows: list[dict[str, Any]], languages: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "synthetic-language",
        "status": "frozen",
        "source": _certificate_source(config),
        "detector": config["language_gate"]["detector"],
        "policy": {
            key: config["language_gate"][key]
            for key in (
                "required_base_task_language",
                "segment_minimum_confidence",
                "segment_minimum_characters",
                "segment_minimum_letters",
                "non_latin_minimum_letters",
                "non_latin_minimum_ratio",
            )
        },
        "results": [
            {
                "source_index": index,
                "source_row_sha256": source_row_hash(row),
                "language": language,
                "confidence": 0.99,
                "mixed": language != "en",
                "segment_languages": [language],
                "reason_flags": [] if language == "en" else ["non_english_training_text"],
            }
            for index, (row, language) in enumerate(zip(rows, languages, strict=True))
        ],
    }


def _compact_certificates(
    root: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    language_path: Path,
) -> tuple[Path, Path]:
    implementation = ROOT / "src/rdan_grpo/rubrichub_rules.py"
    checker_generator = ROOT / "scripts/certify_rubrichub_rules.py"
    tokenizer_generator = ROOT / "scripts/certify_rubrichub_tokenizer.py"
    rtt_root = ROOT.parent / "Rubrics-To-Tokens"
    source_path = rtt_root / "Benchmark/instruction_following_eval/instructions.py"
    reference = json.loads((ROOT / "configs/artifacts/rubrichub_rule_certificate.json").read_text())["reference"]
    reference = {
        **reference,
        "class_sha256": {"CommaChecker": reference["class_sha256"]["CommaChecker"]},
    }
    indices = list(range(len(rows)))
    row_hashes = [source_row_hash(row) for row in rows]
    rule_evidence = root / "rules.jsonl"
    probes = [
        {
            "kind": "probe",
            "function": "CommaChecker",
            "case": f"case-{index}",
            "repeat_equal": True,
            "valid": True,
            "passed": True,
            "reference_parity": True,
        }
        for index in range(21)
    ]
    candidates = [
        {
            "kind": "candidate",
            "source_index": index,
            "source_row_sha256": row_hashes[index],
            "functions": ["CommaChecker"],
            "route_parameters_sha256": _value_sha256(
                [
                    {key: value for key, value in rubric["tags"]["parameters"].items() if value is not None}
                    for rubric in row["reward_model"]["rubrics"]
                    if rubric["tags"]["verifier"] == "rule"
                ]
            ),
        }
        for index, row in enumerate(rows)
    ]
    _write_jsonl(rule_evidence, [*probes, *candidates])
    checker_path = root / "checker.json"
    checker = {
        "schema_version": 1,
        "id": "synthetic-checker",
        "status": "certified",
        "source": _certificate_source(config),
        "routes": [
            {
                "function": "CommaChecker",
                "implementation_sha256": _sha256(implementation),
                "status": "certified",
            }
        ],
        "implementation": {"path": str(implementation), "sha256": _sha256(implementation)},
        "generator": {"path": str(checker_generator), "sha256": _sha256(checker_generator)},
        "reference": reference,
        "candidate_selection": {
            "rows": len(indices),
            "source_indices": indices,
            "source_indices_sha256": _value_sha256(indices),
            "source_row_hashes_sha256": _value_sha256(row_hashes),
            "language_certificate": _file_ref(language_path),
        },
        "evidence": {"path": str(rule_evidence), "records": len(probes) + len(candidates), **_file_ref(rule_evidence)},
        "counts": {
            "probe_records": len(probes),
            "candidate_records": len(candidates),
            "total_records": len(probes) + len(candidates),
        },
    }
    checker_path.write_text(json.dumps(checker, sort_keys=True), encoding="utf-8")

    tokenizer_policy = {
        "model": "synthetic-qwen",
        "revision": "model-rev",
        "files_sha256": {"tokenizer.json": "1" * 64, "tokenizer_config.json": "2" * 64},
        "transformers_version": "4.57.0",
        "chat_template": {
            "messages": "one_user_message",
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "acceptance": "5 < input_tokens <= 2048",
    }
    config["tokenizer_gate"] = tokenizer_policy
    token_evidence = root / "tokens.jsonl"
    token_rows = [
        {
            "source_index": index,
            "source_row_sha256": row_hashes[index],
            "messages_sha256": "3" * 64,
            "input_ids_sha256": "4" * 64,
            "input_tokens": 10,
            "accepted": True,
            "reason": "accepted",
        }
        for index in indices
    ]
    _write_jsonl(token_evidence, token_rows)
    tokenizer_path = root / "tokenizer.json"
    tokenizer = {
        "schema_version": 1,
        "id": "synthetic-tokenizer",
        "status": "frozen",
        "source": _certificate_source(config),
        "selection": {
            "checker_certificate": _file_ref(checker_path),
            "language_certificate": _file_ref(language_path),
            "candidate_rows": len(indices),
            "candidate_indices_sha256": _value_sha256(indices),
            "candidate_source_hashes_sha256": _value_sha256(row_hashes),
        },
        "tokenizer": tokenizer_policy,
        "runtime": {"transformers": "4.57.0"},
        "generator": {"path": str(tokenizer_generator), "sha256": _sha256(tokenizer_generator)},
        "evidence": {"path": str(token_evidence), "records": len(token_rows), **_file_ref(token_evidence)},
        "results": {
            "candidates": len(indices),
            "accepted": len(indices),
            "rejected": 0,
            "accepted_source_indices": indices,
            "accepted_source_indices_sha256": _value_sha256(indices),
            "rejected_rows": [],
            "largest_accepted_input_tokens": 10,
        },
    }
    tokenizer_path.write_text(json.dumps(tokenizer, sort_keys=True), encoding="utf-8")
    assert source_path.is_file()
    return checker_path, tokenizer_path


def _certificate_source(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    return {key: source[key] for key in ("dataset", "revision", "file", "sha256", "records")}


def _file_ref(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _value_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _write_advancedif(path: Path, messages: list[dict[str, str]], rubrics: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["conversation_history", "benchmark_name", "prompt_metadata"])
        writer.writeheader()
        writer.writerow(
            {
                "conversation_history": json.dumps(messages),
                "benchmark_name": "test",
                "prompt_metadata": json.dumps({"rubrics": json.dumps(rubrics)}),
            }
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
