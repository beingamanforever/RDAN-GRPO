import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from rdan_grpo import program as program_contract
from rdan_grpo import scalar_data
from rdan_grpo.program import (
    CONFIRMATION_SEEDS,
    HIR_SOURCE_SHA256,
    HIR_TAXONOMY_SHA256,
    ProgramContractError,
    build_judge_request,
    check_program,
    require_launch_gate,
    resolve_baseline_training,
)
from rdan_grpo.response_identity import ResponseIdentityError, response_data_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _render_tmp_data_paths_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    original = scalar_data._display_path

    def display(path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT):
            parts = resolved.parts
            if "data" in parts:
                return Path(*parts[parts.index("data") :]).as_posix()
        return original(path)

    monkeypatch.setattr(scalar_data, "_display_path", display)


def _program():
    return check_program(ROOT / "configs/program/qwen_first.json")


def _contract_copy(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    for name in (
        "HIR_trainv1.jsonl",
        "HIR_trainv1_rubrics_processed.jsonl",
        "HIR_trainv1_rtt_qwen.jsonl",
        "HIR_trainv1_rdan_scalar_certified.jsonl",
        "hir-certificates/hir_qwen_tokenizer_evidence.jsonl",
        "qwen_hir_rubrichub_if_hybrid.jsonl",
    ):
        target = tmp_path / f"data/{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(ROOT / f"data/{name}", target)
    (tmp_path / "scripts").mkdir()
    os.link(ROOT / "scripts/run_roll_parity.py", tmp_path / "scripts/run_roll_parity.py")
    return tmp_path / "configs/program/qwen_first.json"


def _refresh_response_manifest(program_path: Path, manifest: dict) -> None:
    manifest_path = program_path.parent.parent / "artifacts/qwen_merged_rl_data_manifest.json"
    manifest_sha = _write(manifest_path, manifest)
    program = _read(program_path)
    program["lifecycle_artifacts"]["response_data"]["sha256"] = manifest_sha
    _write(program_path, program)


def test_response_data_rejects_incomplete_benchmark_quarantine(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    manifest_path = program_path.parent.parent / "artifacts/qwen_merged_rl_data_manifest.json"
    manifest = _read(manifest_path)
    manifest["benchmark_quarantine"]["reports"][0]["contamination_policy"] = "prompt_exact_only"
    _refresh_response_manifest(program_path, manifest)

    with pytest.raises(ResponseIdentityError, match="coverage or policy"):
        response_data_identity(program_path)


def test_response_data_rejects_contaminated_eligible_row(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    manifest_path = program_path.parent.parent / "artifacts/qwen_merged_rl_data_manifest.json"
    manifest = _read(manifest_path)
    output = tmp_path / manifest["outputs"]["merged_eligible"]["path"]
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    manifest["benchmark_quarantine"]["reports"][0]["exact_matches"].append(
        {
            "training_source": "hir",
            "training_id": str(first["id"]),
            "source_row_sha256": "1" * 64,
            "training_field": "prompt",
            "training_field_index": 0,
            "training_text_sha256": "2" * 64,
            "benchmark_index": 0,
            "benchmark_field": "prompt",
            "benchmark_field_index": 0,
            "benchmark_text_sha256": "3" * 64,
            "score": 1.0,
        }
    )
    _refresh_response_manifest(program_path, manifest)

    with pytest.raises(ResponseIdentityError, match="contaminated HIR"):
        response_data_identity(program_path)


def _freeze_same_backend_production(program_path: Path) -> str:
    program = _read(program_path)
    path = program_path.parent.parent / "roll/qwen_rtt_papo_response_train.yaml"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    program["same_backend_configs"]["production"] = {
        "status": "frozen",
        "path": "configs/roll/qwen_rtt_papo_response_train.yaml",
        "sha256": digest,
    }
    program["launch_train_config"]["path"] = "configs/roll/qwen_rtt_papo_response_train.yaml"
    program["launch_train_config"]["sha256"] = digest
    _write(program_path, program)
    return digest


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> str:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def _freeze_baselines(program_path: Path) -> None:
    config_root = program_path.parent.parent
    program = _read(program_path)
    for name, relative in program["baseline_configs"].items():
        config_path = (program_path.parent / relative).resolve()
        config = _read(config_path)
        manifest_id = f"{name}_data_v1"
        config["data"].update(
            status="frozen",
            manifest_id=manifest_id,
            teacher={**config["data"]["teacher"], "model_id": "teacher/model", "revision": "1" * 40},
        )
        data_path = config_root.parent / f"data/{name}.jsonl"
        data_path.parent.mkdir(exist_ok=True)
        row = (
            {"row_id": 1, "prompt": "p", "chosen": "c", "rejected": "r"}
            if name == "dpo"
            else {"row_id": 1, "prompt": "p", "output": "o"}
        )
        data_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        row_ids = [1]
        dev_ids = [2]
        output_digests = [program_contract._json_sha256(row)]
        data = {
            "path": f"data/{name}.jsonl",
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "schema": {
                "format": "jsonl",
                "fields": ["row_id", "prompt", "output"]
                if name == "sft"
                else ["row_id", "prompt", "chosen", "rejected"],
            },
            "records": 1,
            "row_ids": row_ids,
            "row_ids_sha256": hashlib.sha256(
                json.dumps({"row_ids": row_ids}, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "output_digests": output_digests,
        }
        if name == "dpo":
            data["pairs"] = [
                {
                    "row_id": 1,
                    "chosen_sha256": hashlib.sha256(b"c").hexdigest(),
                    "rejected_sha256": hashlib.sha256(b"r").hexdigest(),
                }
            ]
        artifact = {
            "schema_version": 1,
            "id": manifest_id,
            "baseline_id": config["id"],
            "source_sha256": {
                "hir_source": HIR_SOURCE_SHA256,
                "hir_processed": "d6690a29cd4f24a3627dd8d48e78953191d0c97ad6acb92cdaf2bf5f1b67568a",
                "taxonomy": HIR_TAXONOMY_SHA256,
            },
            "dev_split_manifest": {"id": "dev_v1", "sha256": "5" * 64, "row_ids": dev_ids},
            "data": data,
            "teacher": {"model_id": "teacher/model", "revision": "1" * 40},
        }
        config["data"]["sha256"] = _write(config_root / f"artifacts/{name}_data_manifest.json", artifact)
        config["readiness"] = "ready"
        _write(config_path, config)
    program["readiness"]["baselines"] = "ready"
    _write(program_path, program)


def _freeze_selection(program_path: Path) -> None:
    artifact_root = program_path.parent.parent / "artifacts"
    program = _read(program_path)
    selection = program["selection"]
    row_ids = list(range(1, 10))
    dev = {
        "schema_version": 1,
        "id": "qwen_dev_v1",
        "split": "dev",
        "source_data_sha256": HIR_SOURCE_SHA256,
        "taxonomy_sha256": HIR_TAXONOMY_SHA256,
        "records": len(row_ids),
        "row_ids": row_ids,
        "row_ids_sha256": hashlib.sha256(
            json.dumps({"row_ids": row_ids}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    dev_hash = _write(artifact_root / "qwen_dev_split.json", dev)
    selection["dev_split"].update(status="frozen", manifest_id=dev["id"], sha256=dev_hash)
    selected = {
        "schema_version": 1,
        "id": "qwen_selection_v1",
        "dev_split_manifest": {"id": dev["id"], "sha256": dev_hash},
        "candidate_metrics": {
            name: [{"candidate": candidate, "score": index / 10} for index, candidate in enumerate(values)]
            for name, values in selection["candidates"].items()
        },
        "selected_values": {name: values[-1] for name, values in selection["candidates"].items()},
        "selection_rule": "maximum_score_then_lowest_candidate",
        "created_at_utc": "2026-08-13T00:00:00Z",
    }
    selected_hash = _write(artifact_root / "qwen_selection.json", selected)
    selection["immutable_artifact"].update(
        status="frozen",
        artifact_id=selected["id"],
        sha256=selected_hash,
    )
    program["readiness"].update(candidate_tuning="ready", confirmation="ready")
    _write(program_path, program)


def _seal_route_partition(program_path: Path) -> None:
    check_program(program_path)


def _freeze_lifecycle(program_path: Path, name: str, artifact: dict) -> Path:
    program = _read(program_path)
    path = program_path.parent.parent.parent / program["lifecycle_artifacts"][name]["path"]
    digest = _write(path, artifact)
    program["lifecycle_artifacts"][name].update(status="frozen", artifact_id=artifact["id"], sha256=digest)
    if name == "judge_calibration":
        program["readiness"]["judge"] = "ready"
    _write(program_path, program)
    return path


def _calibration_artifact(config_root: Path) -> dict:
    implementation = config_root.parent / "src/rdan_grpo/judge.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "src/rdan_grpo/judge.py", implementation)
    judge = config_root / "judges/openrouter_luna.json"
    cases_path = config_root / "judges/qwen_judge_calibration_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    for case in cases:
        case["provenance"]["review_status"] = "human_reviewed"
    cases_path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    by_split = {split: [case for case in cases if case["split"] == split] for split in ("debug", "labeled", "heldout")}
    rows = [
        {
            "case_id": by_split["debug"][0]["case_id"],
            "split": "debug",
            "effort": "none",
            "repeat": 1,
            "generation_id": "gen-debug-1",
            "valid": True,
            "exact_match": None,
            "scores_sha256": program_contract._json_sha256({}),
            "raw_record_sha256": "9" * 64,
            "provenance_sha256": "1" * 64,
        }
    ]
    rows.extend(
        {
            "case_id": case["case_id"],
            "split": "labeled",
            "effort": effort,
            "repeat": 1,
            "generation_id": f"gen-{case['case_id']}-{effort}",
            "valid": True,
            "exact_match": True,
            "scores_sha256": program_contract._json_sha256(case["expected_scores"]),
            "raw_record_sha256": "9" * 64,
            "provenance_sha256": "2" * 64,
        }
        for effort in ("none", "low", "medium")
        for case in by_split["labeled"]
    )
    rows.extend(
        {
            "case_id": case["case_id"],
            "split": "heldout",
            "effort": "none",
            "repeat": repeat,
            "generation_id": f"gen-{case['case_id']}-{repeat}",
            "valid": True,
            "exact_match": True,
            "scores_sha256": program_contract._json_sha256(case["expected_scores"]),
            "raw_record_sha256": "9" * 64,
            "provenance_sha256": "3" * 64,
        }
        for case in by_split["heldout"]
        for repeat in (1, 2)
    )
    case_rows = [
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "tags": case.get("tags", []),
            "sha256": program_contract._json_sha256(case),
        }
        for case in cases
    ]
    artifact = {
        "schema_version": 1,
        "id": "qwen_judge_calibration_v1",
        "status": "calibrated",
        "judge_config_sha256": hashlib.sha256(judge.read_bytes()).hexdigest(),
        "implementation": {
            "path": "src/rdan_grpo/judge.py",
            "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
            "one_call_per_response": True,
            "strict_schema": True,
            "max_retries": 0,
        },
        "case_manifest": {
            "path": "configs/judges/qwen_judge_calibration_cases.jsonl",
            "file_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "sha256": program_contract._json_sha256({"rows": case_rows}),
            "cases": 76,
            "debug": 1,
            "debug_redacted": True,
            "labeled": 49,
            "heldout": 26,
            "corpus_summary": {
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
            },
            "rows": case_rows,
        },
        "call_manifest": {
            "sha256": program_contract._json_sha256({"rows": rows}),
            "raw_file_sha256": "9" * 64,
            "calls": 200,
            "rows": rows,
        },
        "preflight": {
            "catalog": {
                "id": "openai/gpt-5.6-luna",
                "canonical_slug": "openai/gpt-5.6-luna-20260709",
                "supported_parameters": ["max_tokens", "reasoning_effort", "response_format", "seed"],
            },
            "catalog_sha256": "pending",
            "endpoints": {
                "id": "openai/gpt-5.6-luna",
                "provider": "openai",
                "model": "openai/gpt-5.6-luna",
                "tag": "openai",
                "supported_parameters": ["max_tokens", "reasoning_effort", "response_format", "seed"],
                "status": 0,
            },
            "endpoints_sha256": "pending",
            "provider": "openai",
        },
        "debug_canary": {
            "generation_id": "gen-debug-1",
            "provider": "OpenAI",
            "model": "openai/gpt-5.6-luna",
            "parameter_names": ["max_tokens", "reasoning_effort", "response_format", "seed"],
            "upstream_body_sha256": "7" * 64,
            "generation_metadata_polls": 1,
            "request_sha256": "8" * 64,
            "valid": True,
        },
        "selection": program_contract.select_reasoning_effort(
            {effort: [1] * 49 for effort in ("none", "low", "medium")}
        ),
        "selected_reasoning_effort": "none",
        "outcomes": {
            "calls": 200,
            "cases": 76,
            "invalid_calls": 0,
            "labeled_exact_matches": 147,
            "selected_labeled_exact_matches": 49,
            "heldout_exact_matches": 26,
            "heldout_agreements": 26,
            "injection_exact_matches": 8,
            "injection_cases": 8,
            "thresholds": {
                "valid_call_rate": 1.0,
                "selected_labeled_exact_accuracy": 0.85,
                "heldout_exact_accuracy": 0.85,
                "injection_exact_accuracy": 1.0,
                "heldout_duplicate_agreement_rate": 0.96,
            },
            "metrics": {
                "valid_call_rate": 1.0,
                "selected_labeled_exact_accuracy": 1.0,
                "heldout_exact_accuracy": 1.0,
                "injection_exact_accuracy": 1.0,
                "heldout_duplicate_agreement_rate": 1.0,
            },
            "threshold_passes": {
                "valid_call_rate": True,
                "selected_labeled_exact_accuracy": True,
                "heldout_exact_accuracy": True,
                "injection_exact_accuracy": True,
                "heldout_duplicate_agreement_rate": True,
            },
            "failures": 0,
        },
    }
    artifact["preflight"]["catalog_sha256"] = program_contract._json_sha256(artifact["preflight"]["catalog"])
    artifact["preflight"]["endpoints_sha256"] = program_contract._json_sha256(artifact["preflight"]["endpoints"])
    return artifact


def _parity_artifact(root: Path = ROOT, production_sha256: str = "c" * 64) -> dict:
    return {
        "schema_version": 2,
        "id": "qwen_runtime_parity_v1",
        "status": "parity_passed",
        "model": {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": program_contract.MODEL_REVISION,
            "snapshot_sha256": "6" * 64,
        },
        "tokenizer": {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": program_contract.MODEL_REVISION,
            "files_sha256": "7" * 64,
        },
        "chat_template": {"source": "pinned_tokenizer", "enable_thinking": False, "sha256": "8" * 64},
        "runtime_backend": {
            "train_config_sha256": hashlib.sha256(
                (root / "configs/roll/qwen_rtt_papo_response_parity.yaml").read_bytes()
            ).hexdigest(),
            "production_train_config_sha256": production_sha256,
            "resolved_config_sha256": "a" * 64,
            "actor_train_strategy": "fsdp2_train",
            "actor_infer_strategy": "hf_infer",
            "transformer_impl": "huggingface",
            "rtt_revision": program_contract.RTT_REVISION,
            **program_contract.GENERATION_SOURCE_IDENTITY,
        },
        "weight_receipt": {
            "transaction_id": "program-test-transaction",
            "artifact_sha256": "b" * 64,
            "resolved_config_sha256": "a" * 64,
        },
        "rollout_logprob_evidence": {
            "prompt_response_tokens_sha256": "9" * 64,
            "responses": 32,
            "optimizer_updates": 0,
            "infer_logprobs_source": "observed_hf_generation",
            "actor_train_recomputed": True,
            "actor_boundary_observed": True,
            "compared_tokens": 512,
            "max_abs_error": 0.0005,
            "mean_abs_error": 0.00005,
            "thresholds": {"max_abs_error_at_most": 0.001, "mean_abs_error_at_most": 0.0001},
        },
    }


def _no_update_artifact(bundle) -> tuple[dict, dict]:
    response_data = program_contract.response_data_identity(bundle.repo_root / "configs/program/qwen_first.json")
    metrics = {
        "prompt_count": 256,
        "response_count": 2048,
        "group_size": 8,
        "optimizer_updates": 0,
        "batch_count": 4,
        "valid_batch_count": 4,
        "response_active_group_count": 64,
        "response_active_group_rate": 0.25,
        "quality_active_group_count": 32,
        "quality_active_group_rate": 0.125,
        "finite": True,
    }
    body = {
        "schema_version": 2,
        "ready": True,
        "method": "rtt_papo_response",
        "quality_weight": 0.5,
        "config_sha256": bundle.program["launch_train_config"]["preflight_sha256"],
        "source_sha256": {
            "train_config": bundle.program["launch_train_config"]["sha256"],
            "scalar_data_manifest": bundle.program["lifecycle_artifacts"]["scalar_data"]["sha256"],
            "response_data_manifest": response_data["manifest_sha256"],
            "response_data_output": response_data["output_sha256"],
            "response_data_config": response_data["config_sha256"],
            "response_hir_manifest": response_data["hir_manifest_sha256"],
            "rubrichub_rule_certificate": response_data["rule_certificate_sha256"],
            "rubrichub_tokenizer_certificate": response_data["tokenizer_certificate_sha256"],
            "evaluator_certificate": bundle.program["hard_route_policy"]["evaluator_certificate"]["sha256"],
            "judge_calibration": bundle.program["lifecycle_artifacts"]["judge_calibration"]["sha256"],
            "runtime_parity": bundle.program["lifecycle_artifacts"]["runtime_parity"]["sha256"],
        },
        "metrics": metrics,
        "reasons": [],
    }
    certificate_id = program_contract._json_sha256(body)
    return {"certificate_id": certificate_id, **body}, {"artifact_id": certificate_id}


def _refresh_no_update(artifact: dict, reference: dict) -> None:
    body = {key: value for key, value in artifact.items() if key != "certificate_id"}
    artifact["certificate_id"] = reference["artifact_id"] = program_contract._json_sha256(body)


def _token_label_artifact(root: Path) -> tuple[dict, dict, dict[str, bool]]:
    rows = []
    for row_id in (1, 2, 3):
        content = {"row_id": row_id, "input_ids": [10, 11], "labels": [0, 1], "offsets": [[0, 1], [1, 2]]}
        rows.append({"sample_sha256": program_contract._json_sha256(content), **content})
    data_path = root / "labels/token.jsonl"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    sample_ids = [row["sample_sha256"] for row in rows]
    split_names = ("train", "validation", "test")
    splits = {
        name: {
            "sample_ids": [sample_ids[index]],
            "sample_ids_sha256": program_contract._json_sha256({"sample_ids": [sample_ids[index]]}),
            "records": 1,
        }
        for index, name in enumerate(split_names)
    }
    refs = {
        "scalar_data": {"sha256": "1" * 64},
        "token_labels": {"artifact_id": "qwen_token_labels_v1", "sha256": "2" * 64},
    }
    artifact = {
        "schema_version": 1,
        "id": "qwen_token_labels_v1",
        "status": "frozen",
        "provenance": {
            "kind": "reconstructed_rtt_token_relevance",
            "source_repository": "https://github.com/TURLEing/Rubrics-To-Tokens",
            "source_revision": program_contract.RTT_REVISION,
            "hir_source_sha256": program_contract.HIR_SOURCE_SHA256,
            "scalar_data_manifest_sha256": refs["scalar_data"]["sha256"],
            "generator_sha256": "3" * 64,
        },
        "tokenizer": {
            "model": program_contract.MODEL_NAME,
            "revision": program_contract.MODEL_REVISION,
            "files_sha256": "4" * 64,
        },
        "data": {
            "path": "labels/token.jsonl",
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "format": "jsonl",
            "records": 3,
            "tokens": 6,
            "sample_ids_sha256": program_contract._json_sha256({"sample_ids": sample_ids}),
        },
        "splits": splits,
        "alignment": {"records_checked": 3, "tokens_checked": 6, "failures": 0},
    }
    return artifact, refs, {"scalar_data": True, "no_update": True, "token_labels": True}


def test_current_program_freezes_scalar_gate_but_keeps_launch_closed() -> None:
    bundle = _program()
    assert bundle.program["readiness"]["candidate_tuning"] == "blocked_until_frozen_dev_split"
    assert bundle.program["readiness"]["confirmation"] == "blocked_until_immutable_selection_artifact"
    assert bundle.program["readiness"]["baselines"] == "blocked_until_frozen_data_manifests"
    assert bundle.program["readiness"]["scalar_training"] == "ready"
    assert bundle.program["readiness"]["launch"] == "blocked_until_all_launch_artifacts_frozen"
    assert all(config["readiness"] == "blocked_until_frozen_data_manifest" for config in bundle.baselines.values())


def test_repo_backed_frozen_manifests_enable_lifecycle_states(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    _freeze_baselines(program_path)
    _freeze_selection(program_path)
    _seal_route_partition(program_path)

    bundle = check_program(program_path)
    assert {
        key: bundle.program["readiness"][key]
        for key in ("candidate_tuning", "confirmation", "baselines", "scalar_training")
    } == {key: "ready" for key in ("candidate_tuning", "confirmation", "baselines", "scalar_training")}


def test_frozen_artifact_must_exist_and_match_exact_bytes(tmp_path: Path) -> None:
    missing_path = _contract_copy(tmp_path / "missing")
    missing = _read(missing_path)
    baseline_path = missing_path.parent.parent / "baselines/sft_reconstructed.json"
    baseline = _read(baseline_path)
    baseline["data"].update(status="frozen", manifest_id="sft_data_v1", sha256="1" * 64)
    baseline["data"]["teacher"].update(model_id="teacher/model", revision="1" * 40)
    baseline["readiness"] = "ready"
    _write(baseline_path, baseline)
    missing["readiness"]["baselines"] = "ready"
    _write(missing_path, missing)
    with pytest.raises(ProgramContractError, match="cannot hash"):
        check_program(missing_path)

    tampered_path = _contract_copy(tmp_path / "tampered")
    _freeze_baselines(tampered_path)
    artifact_path = tampered_path.parent.parent / "artifacts/sft_data_manifest.json"
    artifact_path.write_text(artifact_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ProgramContractError, match="data hash mismatch"):
        check_program(tampered_path)


def test_selection_cross_links_are_enforced(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    _freeze_selection(program_path)
    artifact_path = program_path.parent.parent / "artifacts/qwen_selection.json"
    artifact = _read(artifact_path)
    artifact["dev_split_manifest"]["id"] = "wrong_dev"
    artifact_hash = _write(artifact_path, artifact)
    program = _read(program_path)
    program["selection"]["immutable_artifact"]["sha256"] = artifact_hash
    _write(program_path, program)

    with pytest.raises(ProgramContractError, match="mis-cross-linked"):
        check_program(program_path)


def test_route_counts_cannot_unlock_or_replace_exact_partition(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    program = _read(program_path)
    program["hard_route_policy"]["implemented_count"] = 799
    program["readiness"]["scalar_training"] = "ready"
    _write(program_path, program)
    with pytest.raises(ProgramContractError, match="hard routes keys are invalid"):
        check_program(program_path)

    partition_path = _contract_copy(tmp_path / "partition")
    _seal_route_partition(partition_path)
    exclusion_path = partition_path.parent.parent / "artifacts/hir_hard_route_exclusions.json"
    exclusion = _read(exclusion_path)
    exclusion["identities"] = exclusion["identities"][1:]
    exclusion_hash = _write(exclusion_path, exclusion)
    program = _read(partition_path)
    program["hard_route_policy"]["exclusion_manifest"]["sha256"] = exclusion_hash
    _write(partition_path, program)
    with pytest.raises(ProgramContractError, match="exactly cover all 76,456"):
        check_program(partition_path)


def test_dev_split_rejects_nonexistent_hir_id(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    _freeze_selection(program_path)
    dev_path = program_path.parent.parent / "artifacts/qwen_dev_split.json"
    dev = _read(dev_path)
    dev["row_ids"][-1] = 8_920
    dev["row_ids_sha256"] = hashlib.sha256(
        json.dumps({"row_ids": dev["row_ids"]}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    program = _read(program_path)
    program["selection"]["dev_split"]["sha256"] = _write(dev_path, dev)
    _write(program_path, program)

    with pytest.raises(ProgramContractError, match="dev split artifact is invalid"):
        check_program(program_path)


@pytest.mark.parametrize("scores,selected", [([0.9, 0.1, 0.2], 0.25), ([0.9, 0.9, 0.2], 0.25)])
def test_selection_rejects_wrong_maximum_or_tie_break(tmp_path: Path, scores: list[float], selected: float) -> None:
    program_path = _contract_copy(tmp_path)
    _freeze_selection(program_path)
    artifact_path = program_path.parent.parent / "artifacts/qwen_selection.json"
    artifact = _read(artifact_path)
    for row, score in zip(artifact["candidate_metrics"]["rl_mix"], scores, strict=True):
        row["score"] = score
    artifact["selected_values"]["rl_mix"] = 0.5 if selected == 0.25 else selected
    program = _read(program_path)
    program["selection"]["immutable_artifact"]["sha256"] = _write(artifact_path, artifact)
    _write(program_path, program)

    with pytest.raises(ProgramContractError, match="does not follow maximum_score_then_lowest_candidate"):
        check_program(program_path)


def test_evaluator_evidence_missing_or_tampered_fails(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    _seal_route_partition(program_path)
    evidence = program_path.parent.parent / "artifacts/hir_type4_rule_evidence.jsonl"
    evidence.unlink()
    with pytest.raises(ProgramContractError, match="evaluator certificate evidence is invalid"):
        check_program(program_path)

    program_path = _contract_copy(tmp_path / "tampered")
    _seal_route_partition(program_path)
    evidence = program_path.parent.parent / "artifacts/hir_type4_rule_evidence.jsonl"
    evidence.write_text(evidence.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ProgramContractError, match="evaluator certificate evidence is invalid"):
        check_program(program_path)


def test_evaluator_evidence_incomplete_or_failing_fails() -> None:
    identity = {"source": "type1", "row_id": 1, "rubric_index": 0, "route": "language:response_language"}
    encoded = program_contract._identity_set([identity], "fixture")
    incomplete = {
        "schema_version": 1,
        "certificate_id": "cert",
        "kind": "smoke",
        "coverage_unit": "family",
        "outcomes": [],
    }
    with pytest.raises(ProgramContractError, match="non-empty"):
        program_contract._validate_evaluator_evidence(incomplete, "cert", "smoke", encoded)

    failing = {
        **incomplete,
        "coverage_unit": "identity",
        "outcomes": [{"identity": identity, "passed": False}],
    }
    covered, failures = program_contract._validate_evaluator_evidence(failing, "cert", "smoke", encoded)
    assert covered == encoded and failures == 1


def test_pending_launch_gate_fails_before_rtt_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(program_contract.subprocess, "run", lambda *args, **kwargs: pytest.fail("RTT inspected"))
    with pytest.raises(ProgramContractError, match="pending lifecycle artifacts"):
        require_launch_gate(
            ROOT / "configs/program/qwen_first.json",
            ROOT / "configs/roll/qwen_rtt_papo_response_train.yaml",
            ROOT / "configs/artifacts/qwen_no_update_certificate.json",
            ROOT.parent / "Rubrics-To-Tokens",
        )


def test_frozen_judge_calibration_and_runtime_parity_are_byte_checked(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    config_root = program_path.parent.parent
    _freeze_lifecycle(program_path, "judge_calibration", _calibration_artifact(config_root))
    production_sha256 = _freeze_same_backend_production(program_path)
    parity_path = _freeze_lifecycle(program_path, "runtime_parity", _parity_artifact(tmp_path, production_sha256))

    bundle = check_program(program_path)
    assert bundle.program["readiness"]["judge"] == "ready"
    assert bundle.program["readiness"]["launch"] == "blocked_until_all_launch_artifacts_frozen"

    parity_path.write_text(parity_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ProgramContractError, match="runtime_parity hash mismatch"):
        check_program(program_path)


def test_same_backend_runtime_parity_freezes_and_passes_program_check(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    production_sha256 = _freeze_same_backend_production(program_path)
    parity = _parity_artifact(tmp_path, production_sha256)
    _freeze_lifecycle(program_path, "runtime_parity", parity)

    bundle = check_program(program_path)

    assert bundle.lifecycle_artifacts["runtime_parity"] == parity
    assert bundle.program["readiness"]["launch"] == "blocked_until_all_launch_artifacts_frozen"


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda artifact, root: artifact["runtime_backend"].update(transformers_version="4.57.1"), "parity"),
        (
            lambda artifact, root: artifact["runtime_backend"].update(generation_get_logits_processor_sha256="0" * 64),
            "parity",
        ),
        (lambda artifact, root: artifact["runtime_backend"].update(generation_sample_sha256="0" * 64), "parity"),
        (lambda artifact, root: artifact["runtime_backend"].update(actor_infer_strategy="vllm"), "parity"),
        (lambda artifact, root: artifact["weight_receipt"].update(resolved_config_sha256="0" * 64), "parity"),
        (
            lambda artifact, root: (root / "configs/roll/qwen_rtt_papo_response_parity.yaml").write_text(
                (root / "configs/roll/qwen_rtt_papo_response_parity.yaml").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            ),
            "diagnostic config pin",
        ),
    ],
    ids=["version", "logits-processor", "sample", "strategy", "receipt", "config"],
)
def test_same_backend_runtime_parity_rejects_lifecycle_tampering(
    tmp_path: Path,
    tamper: object,
    message: str,
) -> None:
    program_path = _contract_copy(tmp_path)
    production_sha256 = _freeze_same_backend_production(program_path)
    parity = _parity_artifact(tmp_path, production_sha256)
    tamper(parity, tmp_path)  # type: ignore[operator]
    _freeze_lifecycle(program_path, "runtime_parity", parity)

    with pytest.raises(ProgramContractError, match=message):
        check_program(program_path)


def test_judge_calibration_rejects_forged_case_manifest_hash(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    artifact = _calibration_artifact(program_path.parent.parent)
    artifact["case_manifest"]["rows"][1]["sha256"] = "0" * 64
    artifact["case_manifest"]["sha256"] = program_contract._json_sha256({"rows": artifact["case_manifest"]["rows"]})
    _freeze_lifecycle(program_path, "judge_calibration", artifact)

    with pytest.raises(ProgramContractError, match="calibration artifact is invalid"):
        check_program(program_path)


def test_judge_calibration_recomputes_duplicate_agreements_from_rows(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    artifact = _calibration_artifact(program_path.parent.parent)
    artifact["outcomes"]["heldout_agreements"] = 25
    artifact["outcomes"]["metrics"]["heldout_duplicate_agreement_rate"] = 25 / 26
    _freeze_lifecycle(program_path, "judge_calibration", artifact)

    with pytest.raises(ProgramContractError, match="calibration artifact is invalid"):
        check_program(program_path)


def test_judge_calibration_rejects_unreviewed_case_provenance(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    artifact = _calibration_artifact(program_path.parent.parent)
    cases_path = program_path.parent.parent / "judges/qwen_judge_calibration_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    for case in cases:
        case["provenance"]["review_status"] = "not_human_reviewed"
    cases_path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    artifact["case_manifest"]["file_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    artifact["case_manifest"]["rows"] = [
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "tags": case.get("tags", []),
            "sha256": program_contract._json_sha256(case),
        }
        for case in cases
    ]
    artifact["case_manifest"]["sha256"] = program_contract._json_sha256({"rows": artifact["case_manifest"]["rows"]})
    _freeze_lifecycle(program_path, "judge_calibration", artifact)

    with pytest.raises(ProgramContractError, match="human-reviewed provenance"):
        check_program(program_path)


def test_missing_calibration_and_failing_parity_cannot_unlock(tmp_path: Path) -> None:
    missing_path = _contract_copy(tmp_path / "missing")
    artifact = _calibration_artifact(missing_path.parent.parent)
    program = _read(missing_path)
    program["lifecycle_artifacts"]["judge_calibration"].update(
        status="frozen", artifact_id=artifact["id"], sha256="9" * 64
    )
    program["readiness"]["judge"] = "ready"
    _write(missing_path, program)
    with pytest.raises(ProgramContractError, match="cannot hash"):
        check_program(missing_path)

    failing_path = _contract_copy(tmp_path / "failing")
    production_sha256 = _freeze_same_backend_production(failing_path)
    parity = _parity_artifact(failing_path.parent.parent.parent, production_sha256)
    parity["rollout_logprob_evidence"]["infer_logprobs_source"] = "old_log_probs_fallback"
    _freeze_lifecycle(failing_path, "runtime_parity", parity)
    with pytest.raises(ProgramContractError, match="parity artifact is invalid"):
        check_program(failing_path)


def test_calibrated_reasoning_effort_must_match_live_request(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    artifact = _calibration_artifact(program_path.parent.parent)
    artifact["selected_reasoning_effort"] = "medium"
    _freeze_lifecycle(program_path, "judge_calibration", artifact)
    with pytest.raises(ProgramContractError, match="calibration artifact is invalid"):
        check_program(program_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["metrics"].update(optimizer_updates=1),
        lambda value: value["metrics"].update(response_count=2047),
        lambda value: value["metrics"].update(valid_batch_count=3),
        lambda value: value["metrics"].update(response_active_group_count=0, response_active_group_rate=0.0),
        lambda value: value["metrics"].update(quality_active_group_count=0, quality_active_group_rate=0.0),
        lambda value: value["metrics"].update(response_active_group_rate=float("nan")),
        lambda value: value["metrics"].update(finite=False),
        lambda value: value.update(quality_weight=0.25),
        lambda value: value.update(reasons=["hidden_failure"]),
        lambda value: value["metrics"].update(extra=1),
    ],
)
def test_no_update_certificate_requires_exact_ready_semantics(mutate: object) -> None:
    bundle = _program()
    artifact, reference = _no_update_artifact(bundle)
    program_contract._validate_no_update_artifact(artifact, reference, bundle)
    mutate(artifact)  # type: ignore[operator]
    _refresh_no_update(artifact, reference)
    with pytest.raises(ProgramContractError, match="no-update"):
        program_contract._validate_no_update_artifact(artifact, reference, bundle)


def test_token_labels_validate_provenance_alignment_and_split_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "external"
    root.mkdir()
    monkeypatch.setenv(program_contract.ARTIFACT_ROOT_ENV, str(root))
    artifact, refs, frozen = _token_label_artifact(root)
    program_contract._validate_token_labels(artifact, refs, frozen, ROOT)

    overlap = copy.deepcopy(artifact)
    overlap["splits"]["validation"] = copy.deepcopy(overlap["splits"]["train"])
    with pytest.raises(ProgramContractError, match="split partition"):
        program_contract._validate_token_labels(overlap, refs, frozen, ROOT)

    unlinked = copy.deepcopy(artifact)
    unlinked["provenance"]["scalar_data_manifest_sha256"] = "9" * 64
    with pytest.raises(ProgramContractError, match="provenance"):
        program_contract._validate_token_labels(unlinked, refs, frozen, ROOT)

    data_path = root / artifact["data"]["path"]
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["labels"].pop()
    data_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    misaligned = copy.deepcopy(artifact)
    misaligned["data"]["sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    with pytest.raises(ProgramContractError, match="alignment"):
        program_contract._validate_token_labels(misaligned, refs, frozen, ROOT)


def test_discriminator_checkpoint_hashes_external_bytes_and_requires_label_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "external"
    checkpoint = root / "discriminator"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv(program_contract.ARTIFACT_ROOT_ENV, str(root))
    refs = {
        "token_labels": {"sha256": "1" * 64},
        "discriminator_checkpoint": {"artifact_id": "disc_v1"},
    }
    artifact = {
        "schema_version": 1,
        "id": "disc_v1",
        "status": "trained",
        "token_labels_sha256": refs["token_labels"]["sha256"],
        "checkpoint": {
            "path": "discriminator",
            "sha256": program_contract._artifact_bytes_sha256(checkpoint),
        },
    }
    frozen = {"token_labels": True}
    program_contract._validate_discriminator_checkpoint(artifact, refs, frozen, ROOT)
    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ProgramContractError, match="checkpoint is invalid"):
        program_contract._validate_discriminator_checkpoint(artifact, refs, frozen, ROOT)
    artifact["checkpoint"]["sha256"] = program_contract._artifact_bytes_sha256(checkpoint)
    with pytest.raises(ProgramContractError, match="blocked until token labels"):
        program_contract._validate_discriminator_checkpoint(artifact, refs, {"token_labels": False}, ROOT)


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": {"wandb_api_key": "value"}},
        {"nested": [{"hf_token": "value"}]},
        {"value": "hf_" + "a" * 32},
        {"value": "wandb_v1_" + "a" * 32},
    ],
)
def test_recursive_secret_rejection_covers_provider_credentials(payload: dict) -> None:
    with pytest.raises(ProgramContractError, match="credential|secret-looking"):
        program_contract._reject_secrets(payload, "payload")


def test_secret_rejection_does_not_treat_hashes_as_credentials() -> None:
    program_contract._reject_secrets({"sha256": "a" * 64, "nested": [{"revision": "b" * 40}]}, "payload")


def test_gpqa_ready_state_requires_frozen_access_artifact(tmp_path: Path) -> None:
    program_path = _contract_copy(tmp_path)
    program = _read(program_path)
    program["readiness"]["gpqa"] = "ready"
    program["benchmarks"][5]["execution_state"] = "runnable_frozen_access"
    _write(program_path, program)
    with pytest.raises(ProgramContractError, match="GPQA"):
        check_program(program_path)

    artifact = {
        "schema_version": 1,
        "id": "gpqa_access_v1",
        "status": "frozen",
        "dataset": {
            "dataset": "Idavidrein/gpqa",
            "config": "gpqa_main",
            "split": "train",
            "revision": "1" * 40,
            "records": 448,
            "filename": "gpqa_main.csv",
            "sha256": "2" * 64,
        },
        "access": {"provider": "huggingface", "gated": True, "verified": True},
    }
    _freeze_lifecycle(program_path, "gpqa_access", artifact)
    check_program(program_path)


def test_scalar_dataset_manifest_rejects_changed_effective_partition_or_bytes() -> None:
    bundle = _program()
    artifact = copy.deepcopy(bundle.lifecycle_artifacts["scalar_data"])
    reference = bundle.program["lifecycle_artifacts"]["scalar_data"]
    artifact["preprocessing"]["effective_records"] += 1
    with pytest.raises(ProgramContractError, match="not linked to its lineage YAML"):
        program_contract._validate_scalar_data_manifest(artifact, reference, ROOT)

    artifact = copy.deepcopy(bundle.lifecycle_artifacts["scalar_data"])
    artifact["data"]["sha256"] = "0" * 64
    with pytest.raises(ProgramContractError, match="not linked to its lineage YAML"):
        program_contract._validate_scalar_data_manifest(artifact, reference, ROOT)


def test_baseline_training_resolves_three_isolated_confirmation_seeds() -> None:
    bundle = _program()
    for baseline in bundle.baselines.values():
        resolved = [resolve_baseline_training(baseline, seed) for seed in CONFIRMATION_SEEDS]
        assert [config["seed"] for config in resolved] == [240521, 240522, 240523]
        resolved[0]["qlora"]["rank"] = 1
        assert resolved[1]["qlora"]["rank"] == baseline["training"]["qlora"]["rank"] == 64
        assert baseline["training"]["seed"] == {"source": "program_confirmation_seed"}


@pytest.mark.parametrize("name", ["sft", "dpo"])
def test_frozen_baseline_accepts_only_coupled_luna_provider_revision(name: str) -> None:
    data = copy.deepcopy(_program().baselines[name]["data"])
    data.update(status="frozen", manifest_id=f"{name}_data_v1", sha256="a" * 64)
    data["teacher"].update(
        model_id=program_contract.LUNA_REVISION,
        revision=program_contract.LUNA_REVISION,
    )
    assert program_contract._validate_baseline_data(data, name) is True

    wrong_revision = copy.deepcopy(data)
    wrong_revision["teacher"]["revision"] = "openai/gpt-5.6-luna"
    with pytest.raises(ProgramContractError, match="teacher must be pinned"):
        program_contract._validate_baseline_data(wrong_revision, name)

    wrong_model = copy.deepcopy(data)
    wrong_model["teacher"]["model_id"] = "openai/gpt-5.6-luna"
    with pytest.raises(ProgramContractError, match="teacher must be pinned"):
        program_contract._validate_baseline_data(wrong_model, name)


def test_judge_substitution_preserves_placeholder_text_from_untrusted_fields() -> None:
    bundle = _program()
    request = build_judge_request(
        bundle.judge,
        bundle.judge_prompt,
        "Keep {{response}} literal.",
        "Keep {{rubrics_json}} and {{instruction}} literal.",
        [{"id": 1, "text": "Keep {{instruction}} literal."}],
        240520,
    )
    content = request["messages"][0]["content"]
    assert "Keep {{response}} literal." in content
    assert "Keep {{rubrics_json}} and {{instruction}} literal." in content
    assert '"text":"Keep {{instruction}} literal."' in content
