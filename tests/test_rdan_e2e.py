"""End-to-end tests for the RDAN-GRPO chain.

Expected advantages are recomputed here in plain Python from the raw rubric outcomes, so a
passing test never just echoes the implementation back at itself. Tests that need ROLL are
skipped where ROLL cannot be imported and run on the training host.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest
import torch

from rdan_grpo.advantages import group_advantages
from rdan_grpo.checkpoint import (
    latest_checkpoint,
    promote_checkpoint,
    promoted_checkpoints,
    prune_checkpoints,
    read_state,
    stage_checkpoint,
)
from rdan_grpo.config import ResponseConfig, load_config, updates_per_step
from rdan_grpo.judge import JudgeRequest, OpenRouterJudge, aggregate_stats
from rdan_grpo.rewards import extract_quality, score_rubrics
from rdan_grpo.rules import evaluate_python_rule, evaluate_rubrichub_rule
from rdan_grpo.scalar import build_scalar_output
from rdan_grpo.tracking import RdanTracker, plot_curves, redact

ROOT = Path(__file__).resolve().parents[1]
GROUP_SIZE = 4
QUALITY_WEIGHT = 1.0


# --------------------------------------------------------------------------------------
# Plain-Python references


def reference_standardize(values: list[float], selected: list[bool]) -> list[float]:
    """Standardize the selected subset of one group, mirroring GRPO group normalization."""

    chosen = [value for value, keep in zip(values, selected, strict=True) if keep]
    if len(chosen) < 2:
        return [0.0] * len(values)
    mean = statistics.fmean(chosen)
    std = statistics.stdev(chosen)
    if std <= 1e-6:
        return [0.0] * len(values)
    return [(value - mean) / (std + 1e-6) if keep else 0.0 for value, keep in zip(values, selected, strict=True)]


def reference_rdan(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Recompute the RDAN channels for one group from raw per-rubric outcomes."""

    hard_pass, outcome, outcome_valid, quality, quality_ok = [], [], [], [], []
    for row in rows:
        hard = [score for score, is_hard, seen in zip(row["scores"], row["hard"], row["seen"], strict=True) if is_hard]
        hard_seen = [seen for is_hard, seen in zip(row["hard"], row["seen"], strict=True) if is_hard]
        soft = [
            score for score, is_hard, seen in zip(row["scores"], row["hard"], row["seen"], strict=True) if not is_hard
        ]
        soft_seen = [seen for is_hard, seen in zip(row["hard"], row["seen"], strict=True) if not is_hard]

        has_hard = bool(hard)
        hard_complete = all(hard_seen)
        passed = (not has_hard) or (hard_complete and all(score == 1 for score in hard))
        hard_pass.append(passed)
        outcome.append(1.0 if passed else 0.0)
        outcome_valid.append((not has_hard) or hard_complete)

        soft_complete = bool(soft) and all(soft_seen)
        quality_ok.append(soft_complete and passed)
        quality.append(statistics.fmean([(score + 1) / 2 for score in soft]) if soft_complete else 0.0)

    outcome_advantage = reference_standardize(outcome, outcome_valid)
    quality_advantage = reference_standardize(quality, quality_ok)
    total = [
        (out + QUALITY_WEIGHT * proc) if valid else 0.0
        for out, proc, valid in zip(outcome_advantage, quality_advantage, outcome_valid, strict=True)
    ]
    return {
        "outcome": outcome,
        "outcome_advantage": outcome_advantage,
        "quality": quality,
        "quality_advantage": quality_advantage,
        "total": total,
        "hard_pass": hard_pass,
        "quality_eligible": quality_ok,
    }


def build_tensors(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    width = max(len(row["scores"]) for row in rows)
    scores, rubric, evaluated, hard = [], [], [], []
    for row in rows:
        pad = width - len(row["scores"])
        scores.append(row["scores"] + [0.0] * pad)
        rubric.append([True] * len(row["scores"]) + [False] * pad)
        evaluated.append(row["seen"] + [False] * pad)
        hard.append(row["hard"] + [False] * pad)
    return {
        "scores": torch.tensor(scores, dtype=torch.float32),
        "rubric_mask": torch.tensor(rubric, dtype=torch.bool),
        "eval_mask": torch.tensor(evaluated, dtype=torch.bool),
        "hard_mask": torch.tensor(hard, dtype=torch.bool),
    }


def make_row(hard_scores: list[float], soft_scores: list[float], judge_failed: bool = False) -> dict[str, Any]:
    """One response: signed hard outcomes, signed soft judgments, and whether the judge failed."""

    return {
        "scores": hard_scores + ([0.0] * len(soft_scores) if judge_failed else soft_scores),
        "hard": [True] * len(hard_scores) + [False] * len(soft_scores),
        "seen": [True] * len(hard_scores) + [not judge_failed] * len(soft_scores),
    }


# --------------------------------------------------------------------------------------
# Reward and advantage


def test_rdan_advantage_matches_independent_reference() -> None:
    rows = [
        make_row([1.0, 1.0], [1.0, 0.0]),
        make_row([1.0, -1.0], [1.0, 1.0]),
        make_row([1.0, 1.0], [-1.0, 0.0]),
        make_row([-1.0, 1.0], [0.0, 0.0]),
    ]
    expected = reference_rdan(rows)
    tensors = build_tensors(rows)
    output = build_scalar_output(
        "rdan", ["p"] * GROUP_SIZE, **tensors, group_size=GROUP_SIZE, quality_weight=QUALITY_WEIGHT
    )

    assert output.selected_raw_reward.tolist() == pytest.approx(expected["outcome"])
    assert output.raw_quality.tolist() == pytest.approx(expected["quality"])
    assert output.response_advantage.tolist() == pytest.approx(expected["outcome_advantage"], abs=1e-5)
    assert output.quality_advantage.tolist() == pytest.approx(expected["quality_advantage"], abs=1e-5)
    assert output.scalar_advantage.tolist() == pytest.approx(expected["total"], abs=1e-5)
    assert output.hard_pass.tolist() == expected["hard_pass"]
    assert output.quality_eligible.tolist() == expected["quality_eligible"]


def test_process_advantage_normalizes_over_hard_passing_responses_only() -> None:
    """PAPO's correct-subset rule: a failing response never earns process credit."""

    rows = [
        make_row([1.0], [1.0]),
        make_row([1.0], [-1.0]),
        make_row([-1.0], [1.0]),
        make_row([-1.0], [1.0]),
    ]
    tensors = build_tensors(rows)
    output = build_scalar_output(
        "rdan", ["p"] * GROUP_SIZE, **tensors, group_size=GROUP_SIZE, quality_weight=QUALITY_WEIGHT
    )

    assert output.quality_eligible.tolist() == [True, True, False, False]
    # The two hard-failing responses scored maximum soft quality and still get zero process credit.
    assert output.quality_advantage[2:].abs().max().item() == 0.0
    assert output.quality_advantage[0].item() > 0 > output.quality_advantage[1].item()


def test_judge_failure_keeps_outcome_reward_and_group_membership() -> None:
    """A dead judge call costs one response its process credit, not the whole step."""

    rows = [
        make_row([1.0], [1.0]),
        make_row([1.0], [1.0], judge_failed=True),
        make_row([-1.0], [1.0]),
        make_row([1.0], [-1.0]),
    ]
    tensors = build_tensors(rows)
    output = build_scalar_output(
        "rdan", ["p"] * GROUP_SIZE, **tensors, group_size=GROUP_SIZE, quality_weight=QUALITY_WEIGHT
    )

    assert output.response_valid.tolist() == [True] * GROUP_SIZE
    assert output.raw_aon.tolist() == [1.0, 1.0, 0.0, 1.0]
    assert output.quality_valid.tolist() == [True, False, True, True]
    assert output.quality_eligible.tolist() == [True, False, False, True]
    assert output.quality_advantage[1].item() == 0.0
    # The outcome channel still separates the group, so the step keeps a usable gradient.
    assert output.scalar_advantage.abs().sum().item() > 0


def test_checker_failure_removes_a_response_from_group_statistics() -> None:
    rows = [make_row([1.0], [1.0]), make_row([1.0], [1.0]), make_row([-1.0], [1.0]), make_row([1.0], [1.0])]
    rows[2]["seen"][0] = False  # the hard checker could not decide
    tensors = build_tensors(rows)
    output = build_scalar_output(
        "rdan", ["p"] * GROUP_SIZE, **tensors, group_size=GROUP_SIZE, quality_weight=QUALITY_WEIGHT
    )

    assert output.response_valid.tolist() == [True, True, False, True]
    assert output.scalar_advantage[2].item() == 0.0


@pytest.mark.parametrize(
    ("method", "expected"),
    [("rl_aon", [1.0, 0.0, 0.0, 0.0]), ("rl_csr", [1.0, 0.75, 0.5, 0.25])],
)
def test_baselines_score_every_rubric_in_one_outcome_channel(method: str, expected: list[float]) -> None:
    rows = [
        make_row([1.0, 1.0], [1.0, 1.0]),
        make_row([1.0, 1.0], [1.0, -1.0]),
        make_row([1.0, -1.0], [1.0, -1.0]),
        make_row([1.0, -1.0], [-1.0, -1.0]),
    ]
    tensors = build_tensors(rows)
    output = build_scalar_output(method, ["p"] * GROUP_SIZE, **tensors, group_size=GROUP_SIZE)

    assert output.selected_raw_reward.tolist() == pytest.approx(expected)
    assert output.quality_advantage.abs().max().item() == 0.0
    assert output.scalar_advantage.tolist() == pytest.approx(
        reference_standardize(expected, [True] * GROUP_SIZE), abs=1e-5
    )


def test_uniform_group_yields_no_learning_signal() -> None:
    rows = [make_row([1.0], [1.0]) for _ in range(GROUP_SIZE)]
    tensors = build_tensors(rows)
    output = build_scalar_output(
        "rdan", ["p"] * GROUP_SIZE, **tensors, group_size=GROUP_SIZE, quality_weight=QUALITY_WEIGHT
    )

    assert output.scalar_advantage.abs().max().item() == 0.0


def test_group_advantages_rejects_misaligned_validity_mask() -> None:
    with pytest.raises(ValueError, match="valid must be boolean"):
        group_advantages(torch.zeros(4), 4, valid=torch.zeros(3, dtype=torch.bool))


def test_soft_only_response_carries_no_outcome_signal_but_full_process_signal() -> None:
    rows = [make_row([], [1.0]), make_row([], [-1.0]), make_row([], [1.0]), make_row([], [0.0])]
    tensors = build_tensors(rows)
    output = build_scalar_output(
        "rdan", ["p"] * GROUP_SIZE, **tensors, group_size=GROUP_SIZE, quality_weight=QUALITY_WEIGHT
    )

    assert output.raw_aon.tolist() == [1.0] * GROUP_SIZE
    assert output.response_advantage.abs().max().item() == 0.0
    assert output.quality_advantage.abs().sum().item() > 0


def test_score_rubrics_and_quality_reject_malformed_inputs() -> None:
    scores = torch.zeros(2, 3)
    wrong = torch.zeros(2, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="rubric_mask must be boolean"):
        score_rubrics(scores, wrong, wrong.bool())
    with pytest.raises(ValueError, match="hard_mask must be boolean"):
        extract_quality(scores, wrong.bool(), wrong.bool(), wrong)


# --------------------------------------------------------------------------------------
# Judge


class FakeCompletion:
    def __init__(self, payload: Any, prompt_tokens: int = 100, completion_tokens: int = 40) -> None:
        content = payload if isinstance(payload, str) else json.dumps(payload)
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]
        self.usage = type("Usage", (), {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})()


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeClient:
    """Minimal stand-in for the OpenAI client, scripted per call."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        if not self.responses:
            raise AssertionError("the judge made more calls than the test scripted")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def judge_config() -> dict[str, Any]:
    config = json.loads((ROOT / "configs/judges/openrouter.json").read_text())
    config["prompt"] = (ROOT / "configs/judges/rubric_prompt.txt").read_text()
    return config


def judgment(ids: list[int], score: float = 1) -> dict[str, Any]:
    return {"rubrics": [{"id": value, "score": score, "reason": "evidence"} for value in ids]}


def make_request(ids: list[int]) -> JudgeRequest:
    return JudgeRequest("instruction", "response", tuple({"id": value, "text": "rubric"} for value in ids))


def test_judge_scores_map_onto_the_signed_rubric_scale() -> None:
    client = FakeClient([FakeCompletion(judgment([1, 2], score=0.5))])
    judge = OpenRouterJudge(judge_config(), "key", client=client)

    result = judge.judge(make_request([1, 2]))

    assert result.valid
    # A judge score of 0.5 is the midpoint of the signed rubric scale, so it maps to 0.0.
    assert [row["score"] for row in result.judgments.values()] == [0.0, 0.0]
    price = judge_config()["price_per_million"]
    stats = aggregate_stats([judge.drain_stats()])
    assert stats["judge/cost_usd"] == pytest.approx((100 * price["prompt"] + 40 * price["completion"]) / 1e6)


def test_judge_request_disables_reasoning_and_pins_decoding() -> None:
    """Reasoning tokens dominate judge cost and add nothing to a rubric classification."""

    client = FakeClient([FakeCompletion(judgment([1]))])
    judge = OpenRouterJudge(judge_config(), "key", client=client)

    judge.judge(make_request([1]))

    payload = client.calls[0]
    assert payload["extra_body"]["reasoning"] == {"enabled": False}
    assert payload["temperature"] == 0.0
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_judge_batch_runs_concurrently_and_preserves_request_order() -> None:
    client = FakeClient([FakeCompletion(judgment([index])) for index in range(1, 5)])
    judge = OpenRouterJudge(judge_config(), "key", client=client)

    results = judge.judge_batch([make_request([index]) for index in range(1, 5)])

    assert [sorted(result.judgments) for result in results] == [[1], [2], [3], [4]]
    assert all(result.valid for result in results)


def test_judge_retries_a_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rdan_grpo.judge.time.sleep", lambda _: None)
    client = FakeClient([FakeStatusError(429), FakeCompletion(judgment([1]))])
    judge = OpenRouterJudge(judge_config(), "key", client=client)

    result = judge.judge(make_request([1]))

    assert result.valid and result.attempts == 2
    assert aggregate_stats([judge.drain_stats()])["judge/retry_rate"] == 1.0


def test_judge_does_not_retry_a_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rdan_grpo.judge.time.sleep", lambda _: None)
    client = FakeClient([FakeStatusError(400), FakeCompletion(judgment([1]))])
    judge = OpenRouterJudge(judge_config(), "key", client=client)

    result = judge.judge(make_request([1]))

    assert not result.valid and len(client.calls) == 1


def test_judge_gives_up_after_max_attempts_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rdan_grpo.judge.time.sleep", lambda _: None)
    config = judge_config()
    client = FakeClient([FakeStatusError(503) for _ in range(config["max_attempts"])])
    judge = OpenRouterJudge(config, "key", client=client)

    result = judge.judge(make_request([1]))

    assert not result.valid and result.judgments == {}
    assert len(client.calls) == config["max_attempts"]
    assert aggregate_stats([judge.drain_stats()])["judge/failure_rate"] == 1.0


@pytest.mark.parametrize(
    "payload",
    [
        {"rubrics": [{"id": 9, "score": 1, "reason": "wrong id"}]},
        {"rubrics": [{"id": 1, "score": 0.7, "reason": "off-scale"}]},
        {"rubrics": [{"id": 1, "score": 1, "reason": ""}]},
        {"rubrics": []},
        "not json at all",
    ],
)
def test_judge_rejects_output_with_nothing_usable(payload: Any) -> None:
    judge = OpenRouterJudge(judge_config(), "key", client=FakeClient([FakeCompletion(payload)]))

    result = judge.judge(make_request([1]))

    assert not result.valid and result.error.startswith("schema_")


def test_judge_keeps_the_well_formed_subset_when_a_row_is_mangled() -> None:
    """A single bad row must not forfeit the whole response's process signal."""

    payload = {
        "rubrics": [
            {"id": 1, "score": 1, "reason": "satisfied"},
            {"id": 2, "score": 0.7, "reason": "off-scale, unusable"},
            {"id": 3, "score": 0, "reason": "violated"},
        ]
    }
    judge = OpenRouterJudge(judge_config(), "key", client=FakeClient([FakeCompletion(payload)]))

    result = judge.judge(make_request([1, 2, 3]))

    assert result.valid
    assert sorted(result.judgments) == [1, 3]
    assert result.judgments[1]["score"] == 1.0 and result.judgments[3]["score"] == -1.0


def test_judge_ignores_ids_it_was_not_asked_about_and_duplicates() -> None:
    payload = {
        "rubrics": [
            {"id": 1, "score": 1, "reason": "first"},
            {"id": 1, "score": 0, "reason": "duplicate, ignored"},
            {"id": 7, "score": 1, "reason": "never requested"},
        ]
    }
    judge = OpenRouterJudge(judge_config(), "key", client=FakeClient([FakeCompletion(payload)]))

    result = judge.judge(make_request([1, 2]))

    assert result.valid and sorted(result.judgments) == [1]
    assert result.judgments[1]["reason"] == "first"


def test_judge_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterJudge(judge_config(), "", client=FakeClient([]))


def test_step_stats_combine_counters_across_reward_workers() -> None:
    """Rates must be recomputed from summed counters, never averaged per worker."""

    workers = [
        {
            "calls": 10,
            "failures": 1,
            "retries": 0,
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "latencies": [1.0],
            "errors": {"APITimeoutError": 1},
            "cost_usd": 0.01,
        },
        {
            "calls": 90,
            "failures": 0,
            "retries": 9,
            "prompt_tokens": 900,
            "completion_tokens": 90,
            "latencies": [3.0],
            "errors": {},
            "cost_usd": 0.09,
        },
    ]

    stats = aggregate_stats(workers)

    assert stats["judge/calls"] == 100.0
    # A naive per-worker average of failure rate would give 0.05, not the true 0.01.
    assert stats["judge/failure_rate"] == pytest.approx(0.01)
    assert stats["judge/retry_rate"] == pytest.approx(0.09)
    assert stats["judge/cost_usd"] == pytest.approx(0.10)
    assert stats["judge/error_APITimeoutError"] == 1.0


def test_step_stats_are_empty_when_no_response_needed_a_judge() -> None:
    assert aggregate_stats([{"calls": 0}, {"calls": 0}]) == {}


# --------------------------------------------------------------------------------------
# Deterministic checkers


def test_rubrichub_letter_and_comma_checkers() -> None:
    letter = {"letter": "i", "let_frequency": 3, "let_relation": "less than"}
    assert evaluate_rubrichub_rule("LetterFrequencyChecker", "abc", letter).passed
    assert not evaluate_rubrichub_rule("LetterFrequencyChecker", "iii", letter).passed
    assert evaluate_rubrichub_rule("CommaChecker", "no comma here", {}).passed
    assert not evaluate_rubrichub_rule("CommaChecker", "one, two", {}).passed


def test_rubrichub_rule_fails_soft_on_an_unsupported_route() -> None:
    result = evaluate_rubrichub_rule("NotImplementedChecker", "text", {})

    assert not result.valid and result.error == "unsupported_rule_route"


def test_python_rule_executes_in_isolation() -> None:
    code = "def check_following(instruction, response):\n    return len(response.split()) > 3\n"

    assert evaluate_python_rule(code, "instruction", "one two three four").passed
    assert not evaluate_python_rule(code, "instruction", "too short").passed


def test_python_rule_rejects_dunder_access() -> None:
    unsafe = "def check_following(instruction, response):\n    return response.__class__ is str\n"

    result = evaluate_python_rule(unsafe, "i", "r")

    assert not result.valid and result.error == "rule_source_uses_dunder"


def test_python_rule_survives_an_infinite_loop() -> None:
    looping = "def check_following(instruction, response):\n    while True:\n        pass\n"

    result = evaluate_python_rule(looping, "i", "r", timeout_seconds=1.0)

    assert not result.valid and result.error == "timeout"


def test_python_rule_that_raises_is_reported_invalid_not_passed() -> None:
    code = "def check_following(instruction, response):\n    return 1 / 0\n"

    result = evaluate_python_rule(code, "i", "r")

    assert not result.valid and not result.passed


# --------------------------------------------------------------------------------------
# Checkpoint lifecycle


def test_checkpoint_promotes_atomically_and_resumes(tmp_path: Path) -> None:
    staging = stage_checkpoint(tmp_path, 20)
    (staging / "actor").mkdir()
    (staging / "actor/model.safetensors").write_bytes(b"weights")
    promoted = promote_checkpoint(staging, {"completed_step": 20, "method": "rdan"})

    assert promoted.name == "step-000020"
    assert not staging.exists()
    assert read_state(promoted)["completed_step"] == 20
    assert latest_checkpoint(tmp_path) == promoted


def test_abandoned_staging_is_never_resumable(tmp_path: Path) -> None:
    stage_checkpoint(tmp_path, 20)  # crash before promotion

    assert latest_checkpoint(tmp_path) is None
    assert promoted_checkpoints(tmp_path) == []


def test_retention_keeps_recent_checkpoints_and_milestones(tmp_path: Path) -> None:
    for step in (20, 40, 60, 80, 100, 120, 140):
        staging = stage_checkpoint(tmp_path, step)
        (staging / "actor/dcp").mkdir(parents=True)
        (staging / "actor/dcp/shard.bin").write_bytes(b"optimizer")
        (staging / "actor/model.safetensors").write_bytes(b"weights")
        promote_checkpoint(staging, {"completed_step": step})

    prune_checkpoints(tmp_path, keep_recent=2, keep_every=100)

    assert [path.name for path in promoted_checkpoints(tmp_path)] == [
        "step-000100",
        "step-000120",
        "step-000140",
    ]
    # The milestone keeps its weights for evaluation but sheds the optimizer state, which is
    # an order of magnitude larger and only useful for resuming from that exact step.
    milestone = tmp_path / "step-000100"
    assert (milestone / "actor/model.safetensors").is_file()
    assert not (milestone / "actor/dcp").exists()
    for recent in ("step-000120", "step-000140"):
        assert (tmp_path / recent / "actor/dcp/shard.bin").is_file()


def test_pruning_clears_abandoned_staging_directories(tmp_path: Path) -> None:
    promote_checkpoint(stage_checkpoint(tmp_path, 20), {"completed_step": 20})
    stage_checkpoint(tmp_path, 40)

    prune_checkpoints(tmp_path, keep_recent=2, keep_every=100)

    assert [path.name for path in tmp_path.iterdir()] == ["step-000020"]


# --------------------------------------------------------------------------------------
# Tracking


def test_tracker_mirrors_metrics_to_disk_and_plots_curves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rdan_grpo.tracking._start_wandb", lambda *_, **__: None)
    tracker = RdanTracker({"exp": "test"}, log_dir=tmp_path)

    for step in range(1, 6):
        tracker.log({"system/step": float(step), "reward/selected_mean": step / 10, "judge/failure_rate": 0.0})
    tracker.finish()

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["system/step"] for row in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert (tmp_path / "curves/reward.png").is_file()
    assert (tmp_path / "curves/judge.png").is_file()


def test_curves_can_be_rebuilt_from_the_mirror_alone(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(json.dumps({"system/step": float(step), "advantage/mean": step * 0.1}) for step in range(1, 4))
    )

    written = plot_curves(metrics, tmp_path / "out")

    assert [path.name for path in written] == ["advantage.png"]


def test_redaction_removes_credential_shaped_values() -> None:
    redacted = redact({"api_key": "sk-or-v1-" + "a" * 32, "note": "token sk-or-v1-" + "b" * 32})

    assert redacted["api_key"] == "[REDACTED]"
    assert "sk-or-v1" not in redacted["note"]


# --------------------------------------------------------------------------------------
# Config


def base_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "rollout_batch_size": 64,
        "num_return_sequences_in_group": 8,
        "actor_train": {
            "worker_cls": "rdan_grpo.workers.ResponseActorWorker",
            "device_mapping": 2,
            "world_size": 2,
        },
        "actor_infer": {
            "worker_cls": "rdan_grpo.workers.ResponseVLLMInferWorker",
            "device_mapping": 2,
            "world_size": 2,
        },
        "rewards": {"llm_judge": {"worker_cls": "rdan_grpo.reward_worker.RubricRewardWorker", "device_mapping": None}},
        "rdan_response": {"method": "rdan", "quality_weight": 1.0},
    }
    payload.update(overrides)
    return payload


class FakeRLVRConfig:
    """Stand-in for ROLL's config dataclass so config loading is testable without ROLL."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def load_fake(payload: dict[str, Any]) -> Any:
    return load_config(FakeRLVRConfig, payload)


def test_config_expands_a_gpu_count_into_a_device_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dacite.from_dict", lambda data_class, data: FakeRLVRConfig(**data), raising=False)
    payload = base_payload()

    config = load_fake(payload)

    assert config.actor_train["device_mapping"] == "[0, 1]"
    assert config.rdan_response == ResponseConfig(method="rdan", quality_weight=1.0)
    # The caller's payload must survive construction unchanged.
    assert payload["actor_train"]["device_mapping"] == 2


def test_config_rejects_a_quality_weight_on_a_baseline_method() -> None:
    payload = base_payload(rdan_response={"method": "rl_aon", "quality_weight": 1.0})

    with pytest.raises(ValueError, match="quality_weight is not valid"):
        load_fake(payload)


def test_config_rejects_a_device_mapping_that_disagrees_with_world_size() -> None:
    payload = base_payload()
    payload["actor_train"]["world_size"] = 4

    with pytest.raises(ValueError, match="device_mapping must cover"):
        load_fake(payload)


def test_updates_per_step_follows_the_actor_global_batch() -> None:
    config = FakeRLVRConfig(
        rollout_batch_size=64,
        num_return_sequences_in_group=8,
        actor_train=FakeRLVRConfig(
            world_size=2,
            training_args=FakeRLVRConfig(per_device_train_batch_size=1, gradient_accumulation_steps=128),
        ),
    )

    assert updates_per_step(config) == 2

    config.actor_train.world_size = 4
    assert updates_per_step(config) == 1

    config.actor_train.training_args.gradient_accumulation_steps = 100
    with pytest.raises(ValueError, match="must divide the actor global batch"):
        updates_per_step(config)


# --------------------------------------------------------------------------------------
# Reward worker over real dataset rows (needs ROLL)


@pytest.fixture(scope="module")
def reward_worker_module() -> Any:
    pytest.importorskip("ray", reason="ROLL runtime is only available on the training host")
    from rdan_grpo import reward_worker

    return reward_worker


@pytest.fixture(scope="module")
def dataset_rows() -> dict[str, dict[str, Any]]:
    """One real row per source, so the checker routing is exercised on production data."""

    path = ROOT / "data/hybrid.jsonl"
    if not path.is_file():
        pytest.skip("training data is not present")
    picked: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            picked.setdefault(row["source"], row)
            if len(picked) == 5:
                break
    return picked


def test_every_source_routes_its_hard_rubrics_without_falling_back_to_the_judge(
    reward_worker_module: Any, dataset_rows: dict[str, dict[str, Any]]
) -> None:
    for source, row in dataset_rows.items():
        truth = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
        rubrics = json.loads(row["rubrics"]) if isinstance(row["rubrics"], str) else row["rubrics"]

        result = reward_worker_module._evaluate_hard_rubrics(
            row["prompt"], "A plain response.", rubrics, source, truth
        )

        hard_count = sum(result["hard"][: result["active"]])
        assert result["active"] == len(rubrics)
        assert hard_count + len(result["judge_rubrics"]) == len(rubrics), source
        assert result["checker_failures"] == 0, f"{source} could not evaluate a hard rubric"


def test_soft_rubric_ids_are_one_based_positions(reward_worker_module: Any) -> None:
    rubrics = [{"description": "first"}, {"description": "second"}]
    truth = {"checker": ["soft", "soft"]}

    result = reward_worker_module._evaluate_hard_rubrics("p", "r", rubrics, "type4", truth)

    assert [rubric["id"] for rubric in result["judge_rubrics"]] == [1, 2]
