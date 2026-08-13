from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rdan_grpo import metrics
from rdan_grpo.metrics import MetricError, derive_stop, load_records, plot_runs, validate_records, write_records

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/logging/qwen_metrics.json"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(seed: int, run_id: str, beta: float) -> dict[str, object]:
    return {
        "project": "rdan-grpo",
        "run_id": run_id,
        "method": "rdan_scalar",
        "stage": "pilot",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "1" * 40,
        "tokenizer_revision": "2" * 40,
        "data_revision": "3" * 40,
        "code_revision": "4" * 40,
        "config_sha256": _hash(f"config:{beta}"),
        "protocol_sha256": "6" * 64,
        "seed": seed,
    }


def _ratio(numerator: float, denominator: float = 8) -> dict[str, float]:
    return {"numerator": numerator, "denominator": denominator}


def _record(seed: int, step: int, beta: float) -> dict[str, object]:
    rows = []
    correct_values = (False, True, True, False, True, True, False, True)
    for index, correct in enumerate(correct_values):
        group_index = index // 4
        hard_valid = index != 0
        quality = (0.2 + 0.02 * step) if hard_valid and correct else 0.0
        quality_advantage = (0.1 * ((index % 3) - 1)) if hard_valid and correct else 0.0
        response_advantage = 0.15 * (index - 3.5) + 0.01 * seed
        rows.append(
            {
                "sample_sha256": _hash(f"{seed}:{step}:{index}"),
                "group_sha256": _hash(f"{seed}:{step}:group:{group_index}"),
                "response_index": index % 4,
                "group_size": 4,
                "group_complete": True,
                "hard_valid": hard_valid,
                "correct": correct,
                "evaluation": {
                    "evaluator_revision": "7" * 64,
                    "unsupported_hard": False,
                    "judge_failed": False,
                    "invalid_output": False,
                },
                "rubrics": [
                    {
                        "rubric_sha256": _hash(f"rubric:{index}"),
                        "rubric_type": "hard",
                        "route": "deterministic",
                        "outcome": "pass" if correct else "fail",
                        "score": float(correct),
                        "evaluator_revision": "7" * 64,
                    }
                ],
                "reward": {
                    "aon": float(correct),
                    "csr": 0.5 + 0.02 * index,
                    "signed_csr": 0.04 * index,
                    "quality": quality,
                    "mix": 0.6 + 0.02 * step,
                    "selected": 0.55 + 0.02 * step,
                },
                "advantage": {
                    "aon": response_advantage,
                    "csr": response_advantage / 2,
                    "mix": response_advantage * 0.75,
                    "rdan": response_advantage + quality_advantage,
                    "response": response_advantage,
                    "quality": quality_advantage,
                    "token": 0.02 * (index - 3.5),
                },
                "generation": {"token_count": 3, "cap_hit": False, "aborted": False, "stop_reason": "eos"},
                "token": {"mask": [True, True, False], "relevance": [0.2, 0.7, 0.0]},
            }
        )
    return {
        "schema_version": 2,
        "kind": "train",
        "identity": _identity(seed, f"rdan-s{seed}-b{int(beta * 100)}", beta),
        "step": step,
        "wall_time_s": float(step * 10),
        "rows": rows,
        "stats": {
            "entropy": _ratio(24 - step),
            "kl": _ratio(0.8 + step / 10),
            "clip": _ratio(step / 2),
            "length": _ratio(800 + 8 * step),
            "throughput": _ratio(8000 + 10 * step, 10),
            "gpu_hours": _ratio(20 * step, 3600),
        },
        "eval": {
            "ifeval": _ratio(4 + step / 2),
            "ifbench": _ratio(3 + step / 2),
            "muldimif": _ratio(2 + step / 2),
            "olympiad_avg4": _ratio(1 + step / 2),
        },
        "hparams": {"token_beta": beta},
        "evidence": {"calibration_checked": 100, "calibration_failed": 0, "nonfinite_count": 0},
    }


def _run(seed: int, beta: float) -> list[dict[str, object]]:
    return [_record(seed, step, beta) for step in range(4)]


def test_metric_pipeline_writes_validates_and_plots_deterministically(tmp_path: Path) -> None:
    paths = []
    for seed, beta in ((11, 0.0), (12, 0.0), (11, 0.5), (12, 0.5)):
        path = tmp_path / f"seed-{seed}-beta-{beta}.jsonl"
        write_records(path, _run(seed, beta))
        assert len(load_records(path)) == 4
        paths.append(path)

    outputs = []
    for name, ordered_paths in (("first", paths), ("second", list(reversed(paths)))):
        output = tmp_path / name
        command = [
            sys.executable,
            str(ROOT / "scripts/plot_training.py"),
            *map(str, ordered_paths),
            "--config",
            str(CONFIG),
            "--output",
            str(output),
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        outputs.append(output)

    first_manifest = (outputs[0] / "sha256_manifest.json").read_bytes()
    second_manifest = (outputs[1] / "sha256_manifest.json").read_bytes()
    assert first_manifest == second_manifest
    manifest = json.loads(first_manifest)
    assert len(manifest["files"]) == 15
    assert len(manifest["inputs"]) == 4
    assert len(manifest["config_sha256"]) == 64
    assert manifest["paper_reference"]["papo_fig9_process_active_final"] == 0.4844
    assert all((outputs[0] / item["path"]).stat().st_size == item["bytes"] for item in manifest["files"])
    assert all(
        hashlib.sha256((outputs[0] / item["path"]).read_bytes()).hexdigest() == item["sha256"]
        for item in manifest["files"]
    )
    sensitivity = (outputs[0] / "rtt_fig3_sensitivity.svg").read_text(encoding="utf-8")
    assert "IFEval + IFBench + MulDimIF macro average" in sensitivity
    assert "response-only beta=0" in sensitivity
    training = (outputs[0] / "rtt_fig5_training.svg").read_text(encoding="utf-8")
    assert "rollout_accuracy" in training
    assert "reward_mix" not in training
    assert "raw sample SD" in training
    reward_hacking = (outputs[0] / "papo_fig2_reward_hacking.svg").read_text(encoding="utf-8")
    assert "eval/olympiad_avg4" in reward_hacking


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record["stats"]["entropy"].update(numerator=float("nan")), "must be finite"),
        (lambda record: record["stats"]["kl"].update(denominator=0), "must be >"),
        (lambda record: record["rows"][0].update(extra="value"), "keys differ"),
        (lambda record: record["rows"][0]["generation"].update(token_count=2), "token arrays must align"),
        (lambda record: record.update(secret="value"), "keys differ"),
    ],
)
def test_metric_contract_fails_closed(mutate: object, message: str) -> None:
    record = _record(11, 0, 0.0)
    mutate(record)  # type: ignore[operator]
    with pytest.raises(MetricError, match=message):
        validate_records([record])


def test_metric_contract_rejects_identity_change() -> None:
    records = _run(11, 0.0)
    changed = copy.deepcopy(records[1])
    changed["identity"]["seed"] = 12  # type: ignore[index]
    records[1] = changed
    with pytest.raises(MetricError, match="identity changed"):
        validate_records(records)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(kind="eval"),
        lambda record: record["hparams"].update(token_beta=0.75),
    ],
)
def test_metric_contract_rejects_kind_or_hparam_drift(mutation: object) -> None:
    records = _run(11, 0.0)
    mutation(records[1])  # type: ignore[operator]
    with pytest.raises(MetricError, match="changed within run"):
        validate_records(records)


def test_group_completeness_is_validated() -> None:
    record = _record(11, 0, 0.0)
    record["rows"].pop()  # type: ignore[union-attr]
    with pytest.raises(MetricError, match="complete group row count"):
        validate_records([record])


def test_stop_indicators_are_derived_from_raw_evidence() -> None:
    record = _record(11, 0, 0.0)
    assert derive_stop(record)["halted"] is False
    row = record["rows"][0]  # type: ignore[index]
    row["evaluation"]["judge_failed"] = True
    row["reward"]["quality"] = 0.2
    indicators = derive_stop(record)
    assert indicators["evaluator_invalid"] is True
    assert indicators["credit_leakage"] is True
    assert indicators["halted"] is True


@pytest.mark.parametrize(("field", "value"), [("reward", -0.2), ("advantage", -0.2)])
def test_negative_quality_credit_on_ineligible_rows_is_leakage(field: str, value: float) -> None:
    record = _record(11, 0, 0.0)
    record["rows"][0][field]["quality"] = value  # type: ignore[index]
    assert derive_stop(record)["credit_leakage"] is True


def test_token_lane_requires_nonuniform_relevance() -> None:
    record = _record(11, 0, 0.5)
    for row in record["rows"]:  # type: ignore[union-attr]
        row["token"]["relevance"] = [0.5, 0.5, 0.5]
    assert derive_stop(record)["token_invalid"] is True


def test_papo_group_activity_and_process_sd_use_paper_definitions() -> None:
    record = _record(11, 0, 0.0)
    assert metrics._metric(record, "process_active_group_ratio", 1e-6) == 1.0
    nonzero = [
        row["advantage"]["quality"]
        for row in record["rows"]  # type: ignore[union-attr]
        if abs(row["advantage"]["quality"]) > 1e-6
    ]
    assert metrics._metric(record, "process_adv_nonzero_std", 1e-6) == pytest.approx(metrics.stdev(nonzero))


def test_duplicate_seed_is_not_aggregated(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_records(first, _run(11, 0.0))
    duplicate = copy.deepcopy(_run(11, 0.0))
    for record in duplicate:
        record["identity"]["run_id"] = "duplicate"
    write_records(second, duplicate)
    with pytest.raises(MetricError, match="duplicate seed"):
        plot_runs([first, second], tmp_path / "figures", CONFIG)


def test_plotter_rejects_unexpected_output_files(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_records(path, _run(11, 0.0))
    output = tmp_path / "figures"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(MetricError, match="unexpected files"):
        plot_runs([path], output, CONFIG)


def test_single_seed_has_no_aggregate_uncertainty(tmp_path: Path) -> None:
    path = tmp_path / "single.jsonl"
    write_records(path, _run(11, 0.0))
    output = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plot_training.py"),
            str(path),
            "--config",
            str(CONFIG),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    svg = (output / "rtt_fig5_training.svg").read_text(encoding="utf-8")
    assert "mean raw" not in svg
    assert "raw sample SD" not in svg
