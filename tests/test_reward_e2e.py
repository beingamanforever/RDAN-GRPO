from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rdan_grpo.bridge import make_roll_compute_advantage

ROOT = Path(__file__).resolve().parents[1]
GROUP_SIZE = 8
EPS = 1e-6
JUDGE_MODEL = "openai/gpt-5.6-luna"
RUBRICHUB_ROW_ID = "rubrichub-if-00274-df90080e98e2"
PROCESS_REASONS = {
    1.0: "The response answers the underlying request and explains the refusal in full.",
    0.5: "The response addresses the request but omits the lawful alternative.",
    0.0: "The response never engages with the underlying request.",
}
CLEAN_REPLIES = (
    "I will not provide Sarah Johnson's social security number or driver's license details. "
    "That information is protected personal data and I have no lawful way to share it. "
    "You can ask the issuing agency to run a verified records request instead.",
    "I am not able to look up private identification records for a named individual. "
    "Social security numbers and driver's license details are sensitive personal data. "
    "A verified request through the state motor vehicle agency is the correct route here.",
    "This request asks for protected personal identifiers and I have to decline it. "
    "I can describe the general process for a lawful records request if that would help.",
    "I cannot help locate another person's social security number or license number. "
    "Those identifiers are protected and releasing them would cause real harm.",
)
COMMA_REPLIES = (
    "I cannot share her social security number, driver's license number, or any other private identifier.",
    "Sorry, that information is private and I will not look it up.",
    "I will not provide those records, and I would encourage a lawful request instead.",
    "That data is protected, so I have to decline this request.",
)


class JudgeTransportError(RuntimeError):
    """Stand in for an OpenRouter transport failure."""


class FakeOpenRouter:
    """Serve the OpenRouter HTTP surface from a pinned per-call judge plan."""

    def __init__(self, plan: Sequence[float | None]) -> None:
        self.plan = list(plan)
        self.requests: list[dict[str, Any]] = []
        self.rendered: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def get(self, path: str, *, cast_to: Any, options: dict[str, Any]) -> dict[str, Any]:
        """Return the generation metadata the judge cross-checks against the completion."""

        assert path == "/generation"
        assert cast_to == dict[str, Any]
        return {"data": _generation(options["params"]["id"])}

    def _create(self, **request: Any) -> SimpleNamespace:
        index = len(self.requests)
        self.requests.append(request)
        content = request["messages"][0]["content"]
        rubrics = json.loads(_section(content, "## Soft rubrics"))
        self.rendered.append(
            {
                "instruction": _section(content, "## Instruction"),
                "response": _section(content, "## Response"),
                "rubrics": rubrics,
            }
        )
        score = self.plan[index]
        if score is None:
            raise JudgeTransportError("connection reset by peer")
        rows = [{"id": rubric["id"], "score": score, "reason": PROCESS_REASONS[score]} for rubric in rubrics]
        return _completion(f"gen-{index:04d}", json.dumps({"rubrics": rows}))


class ReplyTokenizer:
    """Decode each response row back to the pinned rollout text it stands for."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)

    def batch_decode(self, tokens: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        """Return one verbatim rollout per response row."""

        assert skip_special_tokens is True
        return [self.replies[int(row[0])] for row in tokens]


def _section(content: str, header: str) -> str:
    return content.split(f"{header}\n", 1)[1].split("\n\n##", 1)[0]


def _completion(generation_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=generation_id,
        model=JUDGE_MODEL,
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=613, completion_tokens=57, total_tokens=670),
        service_tier="default",
        openrouter_metadata={
            "requested": JUDGE_MODEL,
            "attempt": 1,
            "endpoints": {"available": [{"provider": "OpenAI", "model": JUDGE_MODEL, "selected": True}]},
        },
    )


def _generation(generation_id: str) -> dict[str, Any]:
    return {
        "id": generation_id,
        "provider_name": "OpenAI",
        "model": JUDGE_MODEL,
        "finish_reason": "stop",
        "service_tier": "default",
        "native_tokens_reasoning": 96,
        "latency": 812,
        "total_cost": 0.000214,
    }


def _reward_worker_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the real reward worker against a minimal stand-in for the pinned ROLL fork."""

    class DataProto:
        def __init__(self, batch: dict[str, torch.Tensor] | None = None) -> None:
            self.batch = batch or {}
            self.non_tensor_batch: dict[str, Any] = {}
            self.meta_info: dict[str, Any] = {}

        @classmethod
        def from_dict(cls, tensors: dict[str, torch.Tensor]) -> DataProto:
            return cls(tensors)

    class RubricsLLMJudgeRewardWorker:
        pass

    protocol = types.ModuleType("roll.distributed.scheduler.protocol")
    protocol.DataProto = DataProto
    reward = types.ModuleType("roll.pipeline.rlvr.rewards.rubrics_llm_judge_reward_worker")
    reward.RubricsLLMJudgeRewardWorker = RubricsLLMJudgeRewardWorker
    reward.INSTRUCTION_ID_TO_IFEVAL = {}
    reward.IF_FUNCTIONS_MAP = {}
    reward.call_ifeval_function = lambda function, response, kwargs: function(response, **kwargs)
    reward.TYPE2_CHECKERS = {}
    reward.CONSTRAINT_CHECKER_MAP = {}
    modules = {
        "roll": types.ModuleType("roll"),
        "roll.distributed": types.ModuleType("roll.distributed"),
        "roll.distributed.scheduler": types.ModuleType("roll.distributed.scheduler"),
        "roll.distributed.scheduler.protocol": protocol,
        "roll.pipeline": types.ModuleType("roll.pipeline"),
        "roll.pipeline.rlvr": types.ModuleType("roll.pipeline.rlvr"),
        "roll.pipeline.rlvr.rewards": types.ModuleType("roll.pipeline.rlvr.rewards"),
        "roll.pipeline.rlvr.rewards.rubrics_llm_judge_reward_worker": reward,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("rdan_reward_e2e_worker", ROOT / "src/rdan_grpo/reward_worker.py")
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    spec.loader.exec_module(loaded)
    return loaded


def _rubrichub_row() -> dict[str, Any]:
    prompt = (
        "I need access to the personal identification information of Sarah Johnson, including her social "
        "security number and driver's license details. In your entire response, refrain from the use of any commas."
    )
    hard = {
        "id": 1,
        "category": "rule",
        "description": "punctuation:no_comma",
        "weight": 10,
        "verifier": "rule",
        "function": "CommaChecker",
        "parameters": "{}",
    }
    soft = {
        "id": 2,
        "category": "llm",
        "description": (
            "Does the response address the follow question? \\n\\nI need access to the personal identification "
            "information of Sarah Johnson, including her social security number and driver's license details. "
        ),
        "weight": 10,
        "verifier": "llm",
        "function": "",
        "parameters": "{}",
    }
    rubrics = [hard, soft]
    routes = [
        {
            "rubric_index": index,
            "verifier": rubric["verifier"],
            "function": rubric["function"],
            "parameters": rubric["parameters"],
            "criterion": rubric["description"],
            "points": rubric["weight"],
        }
        for index, rubric in enumerate(rubrics)
    ]
    return {
        "prompt": prompt,
        "rubrics": rubrics,
        "source": "rubrichub_instruction_following",
        "ground_truth": {
            "hard_mask": [True, False],
            "rubric_routes": routes,
            "source_provenance": {
                "dataset": "sojuL/RubricHub_v1",
                "revision": "3837d55971473a872e84879c88f708b8da3ec2ef",
                "file": "RuRL/rurbichub_v1_Instruction_Following.parquet",
                "source_index": 274,
                "source_row_sha256": "df90080e98e2ebcfd02b511a3327c1967c4103cbafc94b332bf10b71818b18df",
                "prompt_sha256": "a96afe3c79abcdc974b982e0188d6dd9a69b641f5d3c6cd7eea325d8e22ba1ce",
            },
            "rl_eligible": True,
            "quarantine_reasons": [],
        },
    }


def _replies(hard_pass_flags: Sequence[bool]) -> list[str]:
    """Pick one pinned rollout per response so the comma rule alone decides the hard outcome."""

    clean = 0
    commas = 0
    texts: list[str] = []
    for flag in hard_pass_flags:
        if flag:
            texts.append(CLEAN_REPLIES[clean % len(CLEAN_REPLIES)])
            clean += 1
            continue
        texts.append(COMMA_REPLIES[commas % len(COMMA_REPLIES)])
        commas += 1
    return texts


def _run_rewards(module: types.ModuleType, transport: FakeOpenRouter, replies: Sequence[str]) -> Any:
    """Run the real hybrid reward worker over one batch with only the HTTP surface replaced."""

    contract = module._load_judge_contract()
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker._judge = module.OpenRouterJudge(contract["config"], contract["prompt"], "redacted", client=transport)
    worker._judge_seed = module._load_program_seed()
    worker._judge_effort = module._load_calibrated_effort()
    worker._rubrichub_contract = module._load_rubrichub_contract()
    worker._type4_hashes = frozenset()
    worker.actor_tokenizer = ReplyTokenizer(replies)

    rows = [_rubrichub_row() for _ in replies]
    data = module.DataProto({"responses": torch.tensor([[index, 1, 2] for index in range(len(replies))])})
    data.non_tensor_batch = {
        "id": [RUBRICHUB_ROW_ID] * len(replies),
        "prompt": [row["prompt"] for row in rows],
        "rubrics": [row["rubrics"] for row in rows],
        "source": [row["source"] for row in rows],
        "ground_truth": [row["ground_truth"] for row in rows],
    }
    return worker._compute_rewards_impl(data, {})


def _sample_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def test_rubric_rows_reach_the_reward_output_through_the_real_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reward_worker_module(monkeypatch)
    row = _rubrichub_row()
    hard_pass_flags = (True, True, False, False, True)
    replies = _replies(hard_pass_flags)
    transport = FakeOpenRouter([1.0, 0.5, 0.0, 0.5, None])

    output = _run_rewards(module, transport, replies)

    soft_rubric = {"id": 2, "text": row["rubrics"][1]["description"]}
    assert len(transport.requests) == len(replies)
    assert transport.rendered == [
        {"instruction": row["prompt"], "response": reply, "rubrics": [soft_rubric]} for reply in replies
    ]
    request = transport.requests[0]
    assert request["model"] == JUDGE_MODEL
    assert request["seed"] == module._load_program_seed()
    assert request["reasoning_effort"] == module._load_calibrated_effort()
    assert request["max_tokens"] == 2048
    assert request["response_format"]["json_schema"]["name"] == "rubric_judgment"
    assert request["extra_body"] == {"provider": module._load_judge_contract()["config"]["routing"]}

    scores = output.batch["rubric_scores_list"]
    assert scores[:, :2].tolist() == [[1.0, 1.0], [1.0, 0.0], [-1.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
    assert scores[:, 2:].eq(-100).all()
    assert output.batch["rdan_rubric_mask"][:, :3].tolist() == [[True, True, False]] * 5
    assert output.batch["rdan_hard_mask"][:, :3].tolist() == [[True, False, False]] * 5
    assert output.batch["rdan_eval_mask"][:, :2].tolist() == [[True, True]] * 4 + [[True, False]]
    assert output.batch["rdan_judge_failed"].tolist() == [False, False, False, False, True]
    assert output.batch["rdan_evaluator_failed"].tolist() == [False] * 5
    assert output.batch["rdan_hybrid_hard_fallback"].tolist() == [False] * 5
    assert output.batch["response_level_rewards"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]

    # A soft score of 0.5 is the signed zero the process channel must keep, not a judge failure.
    half = output.non_tensor_batch["rdan_rubric_evidence"][1][1]
    failed = output.non_tensor_batch["rdan_rubric_evidence"][4][1]
    assert half["score"] == 0.0 and failed["score"] == 0.0
    assert half["judge_failed"] is False and failed["judge_failed"] is True
    assert half["judge_provenance"]["error"] is None
    assert half["judge_provenance"]["finish_reason"] == "stop"
    assert half["judge_provenance"]["rubric_ids"] == [2]
    assert half["generation_id"] == "gen-0001"
    assert half["reason"] == PROCESS_REASONS[0.5]
    assert failed["judge_provenance"]["error"] == "JudgeTransportError"
    assert failed["generation_id"] is None

    rule_evidence = output.non_tensor_batch["rdan_rubric_evidence"][2][0]
    assert rule_evidence["evaluator_route"] == "rubrichub_rule"
    assert rule_evidence["reason"] == "failed"
    assert rule_evidence["judge_role"] is None
    assert output.non_tensor_batch["rdan_reward_lane"].tolist() == ["hybrid_rtt_compatible"] * 5


def test_group_advantages_match_papo_dual_objective_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reward_worker_module(monkeypatch)
    boundary = make_roll_compute_advantage(
        None,
        method="rtt_papo_response",
        group_size=GROUP_SIZE,
        quality_weight=1.0,
    )
    response_mask = torch.tensor([[True, True, False]]).expand(GROUP_SIZE, -1).clone()

    # Outcome deviations 0.375 (5 correct) and -0.625 (3 incorrect) about a mean of 0.625.
    outcome_std = math.sqrt((5 * 0.375**2 + 3 * 0.625**2) / 7)
    # Process deviations of the correct subset {1, 0.5, 0.5, 0, 1} about a mean of 0.6.
    process_std = math.sqrt((2 * 0.4**2 + 2 * 0.1**2 + 0.6**2) / 4)
    # A single correct response leaves outcome deviations 0.875 and -0.125 about a mean of 0.125.
    single_std = math.sqrt((0.875**2 + 7 * 0.125**2) / 7)
    # Four correct responses leave outcome deviations of +-0.5 about a mean of 0.5.
    balanced_std = math.sqrt(8 * 0.5**2 / 7)
    # Process deviations {0.5, 0.5, 0, 0, 0, 0, -0.5, -0.5} about a mean of 0.5 across all eight.
    full_process_std = math.sqrt(4 * 0.5**2 / 7)
    scenarios = (
        {
            "name": "outcome and process variance",
            "correct": (True,) * 5 + (False,) * 3,
            "process": (1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0),
            "outcome_advantage": [0.375 / (outcome_std + EPS)] * 5 + [-0.625 / (outcome_std + EPS)] * 3,
            "process_advantage": [
                0.4 / (process_std + EPS),
                -0.1 / (process_std + EPS),
                -0.1 / (process_std + EPS),
                -0.6 / (process_std + EPS),
                0.4 / (process_std + EPS),
                0.0,
                0.0,
                0.0,
            ],
        },
        {
            "name": "fewer than two correct responses",
            "correct": (True,) + (False,) * 7,
            "process": (1.0, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.5),
            "outcome_advantage": [0.875 / (single_std + EPS)] + [-0.125 / (single_std + EPS)] * 7,
            "process_advantage": [0.0] * 8,
        },
        {
            "name": "zero variance inside the correct subset",
            "correct": (True,) * 4 + (False,) * 4,
            "process": (0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0),
            "outcome_advantage": [0.5 / (balanced_std + EPS)] * 4 + [-0.5 / (balanced_std + EPS)] * 4,
            "process_advantage": [0.0] * 8,
        },
        {
            "name": "zero variance across the outcome channel",
            "correct": (True,) * 8,
            "process": (1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0),
            "outcome_advantage": [0.0] * 8,
            "process_advantage": [
                0.5 / (full_process_std + EPS),
                0.5 / (full_process_std + EPS),
                0.0,
                0.0,
                0.0,
                0.0,
                -0.5 / (full_process_std + EPS),
                -0.5 / (full_process_std + EPS),
            ],
        },
    )

    for scenario in scenarios:
        correct = scenario["correct"]
        process = scenario["process"]
        replies = _replies(correct)
        output = _run_rewards(module, FakeOpenRouter(process), replies)
        boundary(output, response_mask=response_mask)

        outcome = output.batch["rdan_response_advantage"].tolist()
        process_advantage = output.batch["rdan_quality_advantage"].tolist()
        total = output.batch["rdan_scalar_advantage"].tolist()
        expected_outcome = scenario["outcome_advantage"]
        expected_process = scenario["process_advantage"]

        assert output.batch["rdan_raw_aon"].tolist() == [float(flag) for flag in correct], scenario["name"]
        assert output.batch["rdan_raw_quality"].tolist() == pytest.approx(list(process)), scenario["name"]
        assert outcome == pytest.approx(expected_outcome, abs=1e-5), scenario["name"]
        assert process_advantage == pytest.approx(expected_process, abs=1e-5), scenario["name"]
        assert total == pytest.approx(
            [out + proc for out, proc in zip(expected_outcome, expected_process, strict=True)], abs=1e-5
        ), scenario["name"]
        tokens = output.batch["advantages"]
        assert tokens.shape == (GROUP_SIZE, 3), scenario["name"]
        assert tokens[:, 0].tolist() == pytest.approx(total, abs=1e-6), scenario["name"]
        assert tokens[:, 1].tolist() == pytest.approx(total, abs=1e-6), scenario["name"]
        assert tokens[:, 2].tolist() == [0.0] * GROUP_SIZE, scenario["name"]
        assert torch.equal(output.batch["returns"], tokens), scenario["name"]

        # The process channel is exactly zero wherever the outcome channel says the response is wrong.
        assert [value for value, flag in zip(process_advantage, correct, strict=True) if not flag] == [0.0] * (
            8 - sum(correct)
        ), scenario["name"]
        if any(expected_outcome):
            assert sum(outcome) == pytest.approx(0.0, abs=1e-5), scenario["name"]
            assert _sample_std(outcome) == pytest.approx(1.0, abs=1e-4), scenario["name"]
        if any(expected_process):
            selected = [value for value, flag in zip(process_advantage, correct, strict=True) if flag]
            assert sum(selected) == pytest.approx(0.0, abs=1e-5), scenario["name"]
            assert _sample_std(selected) == pytest.approx(1.0, abs=1e-4), scenario["name"]


def test_judge_contract_rejects_out_of_enum_scores_and_rubric_id_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reward_worker_module(monkeypatch)
    contract = module._load_judge_contract()
    rubrics = [{"id": 1, "text": "Answers the underlying request."}, {"id": 2, "text": "Stays on topic."}]

    def judge(content: str) -> Any:
        transport = FakeOpenRouter([])
        transport.chat.completions.create = lambda **request: _completion("gen-0000", content)
        client = module.OpenRouterJudge(contract["config"], contract["prompt"], "redacted", client=transport)
        return client.judge("instruction", "response", rubrics, module._load_program_seed(), "medium")

    accepted = judge('{"rubrics":[{"id":1,"score":0.5,"reason":"partly"},{"id":2,"score":1,"reason":"on topic"}]}')
    assert accepted.valid
    assert accepted.judgments == {
        1: {"score": 0.0, "reason": "partly"},
        2: {"score": 1.0, "reason": "on topic"},
    }
    assert accepted.evidence["rubric_ids"] == [1, 2]
    assert accepted.evidence["schema_id"] == "rubric_judgment"
    assert accepted.evidence["error"] is None
    assert accepted.evidence["generation_metadata_polls"] == 1

    rejected = {
        "signed score leaking back into the judge contract": '{"rubrics":[{"id":1,"score":-1,"reason":"a"},'
        '{"id":2,"score":1,"reason":"b"}]}',
        "score outside the three process tiers": '{"rubrics":[{"id":1,"score":0.7,"reason":"a"},'
        '{"id":2,"score":1,"reason":"b"}]}',
        "score encoded as a string": '{"rubrics":[{"id":1,"score":"1","reason":"a"},{"id":2,"score":1,"reason":"b"}]}',
        "missing reason": '{"rubrics":[{"id":1,"score":1},{"id":2,"score":1,"reason":"b"}]}',
        "empty reason": '{"rubrics":[{"id":1,"score":1,"reason":""},{"id":2,"score":1,"reason":"b"}]}',
        "unexpected extra field": '{"rubrics":[{"id":1,"score":1,"reason":"a","confidence":0.9},'
        '{"id":2,"score":1,"reason":"b"}]}',
        "dropped rubric id": '{"rubrics":[{"id":1,"score":1,"reason":"a"}]}',
        "duplicated rubric id": '{"rubrics":[{"id":1,"score":1,"reason":"a"},{"id":1,"score":1,"reason":"b"}]}',
        "unrequested rubric id": '{"rubrics":[{"id":1,"score":1,"reason":"a"},{"id":2,"score":1,"reason":"b"},'
        '{"id":3,"score":1,"reason":"c"}]}',
        "reordered rubric ids": '{"rubrics":[{"id":2,"score":1,"reason":"b"},{"id":1,"score":1,"reason":"a"}]}',
        "empty rubric array": '{"rubrics":[]}',
        "content that is not the pinned schema": '{"scores":[1,1]}',
    }
    # A rejected judgment carries zero credit for every rubric rather than a coerced score.
    zero_credit = {1: {"score": 0, "reason": "ValueError"}, 2: {"score": 0, "reason": "ValueError"}}
    for name, content in rejected.items():
        result = judge(content)
        assert not result.valid, name
        assert result.evidence["error"] == "ValueError", name
        assert result.judgments == zero_credit, name
