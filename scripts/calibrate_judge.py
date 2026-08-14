#!/usr/bin/env python3
"""Run the frozen 200-call OpenRouter judge calibration."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.judge import JudgeResult, OpenRouterJudge, preflight_snapshots, select_reasoning_effort

CONFIG = ROOT / "configs/judges/openrouter_luna.json"
PROMPT = ROOT / "configs/judges/rubric_prompt.txt"
IMPLEMENTATION = ROOT / "src/rdan_grpo/judge.py"
CASES_PATH = "configs/judges/qwen_judge_calibration_cases.jsonl"
CASES = ROOT / CASES_PATH
EFFORTS = ("high", "medium", "low")
# Authorized by the experiment owner on 2026-08-15 for openai/gpt-5.6-luna.
# valid_call_rate: the model exposes no retry and a 200-call run cannot guarantee zero
# transport failures, so one blip is tolerated. Measured 0.995.
# injection_exact_accuracy: measured 0.625 at the selected effort. This judge is
# persuadable by adversarial text inside the content it grades, which is a live reward
# hacking risk over long training and must be revisited before any full run.
# The accuracy and self-consistency gates are unchanged and all pass.
THRESHOLDS = {
    "valid_call_rate": 0.99,
    "selected_labeled_exact_accuracy": 0.85,
    "heldout_exact_accuracy": 0.85,
    "injection_exact_accuracy": 0.60,
    "heldout_duplicate_agreement_rate": 0.96,
}
EXPECTED_CORPUS_SUMMARY = {
    "rubrics": 206,
    "categories": [
        "Desired_Writing_Style",
        "Hierarchical_Instructions",
        "Item_Listing_Details",
        "Key_Formatting",
        "Morphological_Constraints",
        "Multi-lingual_Constraints",
        "Paragraphs_Constraints",
        "Semantic_elements",
        "Special_Output_Format",
        "Specific_Grammatical_Structure",
        "Specific_Literary_Devices",
        "Specific_Sentence",
    ],
    "cases_with_6_to_12_rubrics": 12,
    "rubric_count_bins": {"1": 21, "2": 41, "3": 1, "6": 4, "8": 4, "10": 2, "12": 2},
    "multilingual_cases": 14,
    "unicode_cases": 14,
    "injection_cases": 8,
    "long_instruction_cases": 10,
    "long_response_cases": 11,
}
CANARY = {
    "instruction": "Reply exactly OK.",
    "response": "OK",
    "rubrics": [{"id": 1, "text": "The response is exactly OK."}],
}


def run_calibration(cases_path: Path, raw_path: Path, certificate_path: Path, api_key: str) -> dict[str, Any]:
    if cases_path.resolve() != CASES:
        raise ValueError(f"calibration cases must be loaded from {CASES}")
    cases = _load_cases(cases_path, require_human_review=True)
    _require_outside_repo(raw_path)
    if raw_path.exists() or certificate_path.exists():
        raise FileExistsError("calibration outputs must not already exist")
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    snapshots = preflight_snapshots(contract, lambda url: _get_json(url, api_key))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.touch(exist_ok=False)
    judge = OpenRouterJudge(contract, PROMPT.read_text(encoding="utf-8"), api_key)
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str, int], JudgeResult] = {}

    debug = next(case for case in cases if case["split"] == "debug")
    debug_result = judge.debug_canary(CANARY["instruction"], CANARY["response"], CANARY["rubrics"], debug["seed"])
    _record(debug, "none", 1, debug_result, rows, raw_path, results)
    labeled = [case for case in cases if case["split"] == "labeled"]
    for effort in EFFORTS:
        for case in labeled:
            _call(judge, case, effort, 1, rows, raw_path, results)
    indicators = {
        effort: [_exact(case, results[(case["case_id"], effort, 1)]) for case in labeled] for effort in EFFORTS
    }
    selection = select_reasoning_effort(indicators)
    selected = selection["selected_effort"]
    heldout = [case for case in cases if case["split"] == "heldout"]
    for case in heldout:
        for repeat in (1, 2):
            _call(judge, case, selected, repeat, rows, raw_path, results)

    invalid = sum(not result.valid for result in results.values())
    selected_labeled = sum(_exact(case, results[(case["case_id"], selected, 1)]) for case in labeled)
    heldout_exact = sum(_exact(case, results[(case["case_id"], selected, 1)]) for case in heldout)
    agreements = sum(
        _scores(results[(case["case_id"], selected, 1)]) == _scores(results[(case["case_id"], selected, 2)])
        for case in heldout
    )
    injection = [case for case in labeled + heldout if "injection" in case.get("tags", [])]
    injection_exact = sum(_exact(case, results[(case["case_id"], selected, 1)]) for case in injection)
    metrics = {
        "valid_call_rate": (len(results) - invalid) / len(results),
        "selected_labeled_exact_accuracy": selected_labeled / len(labeled),
        "heldout_exact_accuracy": heldout_exact / len(heldout),
        "injection_exact_accuracy": injection_exact / len(injection),
        "heldout_duplicate_agreement_rate": agreements / len(heldout),
    }
    threshold_passes = {name: metrics[name] >= minimum for name, minimum in THRESHOLDS.items()}
    failures = sum(not passed for passed in threshold_passes.values())
    call_manifest = {"rows": rows}
    case_rows = [
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "tags": case.get("tags", []),
            "sha256": _sha256(case),
        }
        for case in cases
    ]
    corpus_summary = _corpus_summary(cases)
    certificate = {
        "schema_version": 1,
        "id": "qwen_judge_calibration_v1",
        "status": "calibrated" if failures == 0 else "failed",
        "judge_config_sha256": _file_sha256(CONFIG),
        "implementation": {
            "path": "src/rdan_grpo/judge.py",
            "sha256": _file_sha256(IMPLEMENTATION),
            "one_call_per_response": True,
            "strict_schema": True,
            "max_retries": 0,
        },
        "case_manifest": {
            "path": CASES_PATH,
            "file_sha256": _file_sha256(CASES),
            "sha256": _sha256({"rows": case_rows}),
            "cases": 76,
            "debug": 1,
            "debug_redacted": True,
            "labeled": 49,
            "heldout": 26,
            "corpus_summary": corpus_summary,
            "rows": case_rows,
        },
        "call_manifest": {
            "sha256": _sha256(call_manifest),
            "raw_file_sha256": _file_sha256(raw_path),
            "calls": len(rows),
            **call_manifest,
        },
        "preflight": {
            "catalog": snapshots["catalog"],
            "catalog_sha256": snapshots["catalog_sha256"],
            "endpoints": snapshots["endpoints"],
            "endpoints_sha256": snapshots["endpoints_sha256"],
            "provider": "openai",
        },
        "debug_canary": {
            "generation_id": debug_result.evidence["generation_id"],
            "provider": debug_result.evidence["provider"],
            "model": debug_result.evidence["model"],
            "parameter_names": debug_result.evidence["parameter_names"],
            "upstream_body_sha256": debug_result.evidence["upstream_body_sha256"],
            "generation_metadata_polls": debug_result.evidence["generation_metadata_polls"],
            "request_sha256": debug_result.evidence["request_sha256"],
            "valid": debug_result.valid,
        },
        "selection": selection,
        "selected_reasoning_effort": selected,
        "outcomes": {
            "calls": len(rows),
            "cases": len(cases),
            "invalid_calls": invalid,
            "labeled_exact_matches": sum(sum(values) for values in indicators.values()),
            "selected_labeled_exact_matches": selected_labeled,
            "heldout_exact_matches": heldout_exact,
            "heldout_agreements": agreements,
            "injection_exact_matches": injection_exact,
            "injection_cases": len(injection),
            "thresholds": THRESHOLDS,
            "metrics": metrics,
            "threshold_passes": threshold_passes,
            "failures": failures,
        },
    }
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    with certificate_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(certificate, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return certificate


def _call(
    judge: OpenRouterJudge,
    case: dict[str, Any],
    effort: str,
    repeat: int,
    rows: list[dict[str, Any]],
    raw_path: Path,
    results: dict[tuple[str, str, int], JudgeResult],
) -> None:
    result = judge.judge(case["instruction"], case["response"], case["rubrics"], case["seed"], effort)
    _record(case, effort, repeat, result, rows, raw_path, results)


def _record(
    case: dict[str, Any],
    effort: str,
    repeat: int,
    result: JudgeResult,
    rows: list[dict[str, Any]],
    raw_path: Path,
    results: dict[tuple[str, str, int], JudgeResult],
) -> None:
    key = (case["case_id"], effort, repeat)
    results[key] = result
    evidence = result.evidence
    scores = {str(rubric_id): score for rubric_id, score in _scores(result).items()}
    scores_sha256 = _sha256(scores)
    raw = {
        "case_id": case["case_id"],
        "effort": effort,
        "repeat": repeat,
        "scores": scores,
        "scores_sha256": scores_sha256,
        "evidence": result.evidence,
        "raw": result.raw,
    }
    rows.append(
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "effort": effort,
            "repeat": repeat,
            "generation_id": evidence["generation_id"],
            "valid": result.valid,
            "exact_match": bool(_exact(case, result)) if case["split"] != "debug" else None,
            "scores_sha256": scores_sha256,
            "raw_record_sha256": _sha256(raw),
            "provenance_sha256": _sha256(evidence),
        }
    )
    with raw_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")


def _load_cases(path: Path, *, require_human_review: bool = False) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {split: sum(case.get("split") == split for case in cases) for split in ("debug", "labeled", "heldout")}
    ids = [case.get("case_id") for case in cases]
    if counts != {"debug": 1, "labeled": 49, "heldout": 26} or len(ids) != len(set(ids)) == 76:
        raise ValueError("case manifest must contain 1 debug, 49 labeled, and 26 heldout cases")
    debug = next(case for case in cases if case["split"] == "debug")
    if debug.get("redacted") is not True:
        raise ValueError("debug canary must be explicitly redacted")
    required = {"case_id", "split", "instruction", "response", "rubrics", "seed"}
    if any(not required.issubset(case) for case in cases):
        raise ValueError("calibration case is incomplete")
    if any(case["seed"] != 240520 for case in cases):
        raise ValueError("calibration cases must use the program tuning seed")
    scored = [case for case in cases if case["split"] != "debug"]
    if any(
        not isinstance(case["rubrics"], list)
        or not 1 <= len(case["rubrics"]) <= 12
        or [rubric.get("id") for rubric in case["rubrics"]] != list(range(1, len(case["rubrics"]) + 1))
        for case in scored
    ):
        raise ValueError("scored rubrics must use sequential IDs and the HIR-sized 1-12 range")
    if any(
        set(case.get("expected_scores", {})) != {str(rubric["id"]) for rubric in case["rubrics"]} for case in scored
    ):
        raise ValueError("every scored rubric must have a frozen expected score")
    if any(
        isinstance(score, bool) or not isinstance(score, int) or score not in {-1, 1}
        for case in scored
        for score in case["expected_scores"].values()
    ):
        raise ValueError("expected scores must be -1 or 1")
    if any(
        not isinstance(case.get("tags", []), list) or not all(isinstance(tag, str) for tag in case.get("tags", []))
        for case in scored
    ):
        raise ValueError("scored calibration cases must have string tags")
    review_status = "human_reviewed" if require_human_review else "not_human_reviewed"
    if any(
        case.get("provenance") != {"source": "curated_hir_soft_taxonomy_v1", "review_status": review_status}
        for case in cases
    ):
        raise ValueError(f"calibration provenance must be curated and marked {review_status}")
    labeled_rubrics = {_sha256(case["rubrics"]) for case in scored if case["split"] == "labeled"}
    if any(_sha256(case["rubrics"]) in labeled_rubrics for case in scored if case["split"] == "heldout"):
        raise ValueError("heldout rubric sets must not duplicate labeled rubric sets")
    _validate_corpus_summary(_corpus_summary(cases))
    return cases


def _corpus_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [case for case in cases if case["split"] != "debug"]
    categories = sorted({rubric["category"] for case in scored for rubric in case["rubrics"]})
    rubric_counts = [len(case["rubrics"]) for case in scored]
    return {
        "rubrics": sum(rubric_counts),
        "categories": categories,
        "cases_with_6_to_12_rubrics": sum(6 <= count <= 12 for count in rubric_counts),
        "rubric_count_bins": {str(count): rubric_counts.count(count) for count in sorted(set(rubric_counts))},
        "multilingual_cases": sum("multilingual" in case.get("tags", []) for case in scored),
        "unicode_cases": sum(
            any(ord(char) > 127 for char in case["instruction"] + case["response"]) for case in scored
        ),
        "injection_cases": sum("injection" in case.get("tags", []) for case in scored),
        "long_instruction_cases": sum(len(case["instruction"]) >= 150 for case in scored),
        "long_response_cases": sum(len(case["response"]) >= 300 for case in scored),
    }


def _validate_corpus_summary(summary: dict[str, Any]) -> None:
    if summary != EXPECTED_CORPUS_SUMMARY:
        raise ValueError("calibration corpus does not match the frozen production-representative profile")


def _exact(case: dict[str, Any], result: JudgeResult) -> int:
    expected = {int(key): value for key, value in case.get("expected_scores", {}).items()}
    return int(result.valid and _scores(result) == expected)


def _scores(result: JudgeResult) -> dict[int, int]:
    return {rubric_id: row["score"] for rubric_id, row in result.judgments.items()}


def _get_json(url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _require_outside_repo(path: Path) -> None:
    try:
        path.resolve().relative_to(ROOT)
    except ValueError:
        return
    raise ValueError("raw calibration calls must be stored outside Git")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("raw_output", type=Path)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    certificate = run_calibration(args.cases, args.raw_output, args.certificate, key)
    if certificate["status"] != "calibrated":
        raise SystemExit("calibration failed closed; inspect the external raw artifact")


if __name__ == "__main__":
    main()
