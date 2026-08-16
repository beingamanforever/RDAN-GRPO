from __future__ import annotations

import importlib.util
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    class DataProto:
        def __init__(self, batch: dict[str, torch.Tensor] | None = None):
            self.batch = batch or {}
            self.non_tensor_batch: dict[str, object] = {}
            self.meta_info: dict[str, object] = {}

        @classmethod
        def from_dict(cls, tensors: dict[str, torch.Tensor]) -> DataProto:
            return cls(tensors)

    class RewardWorker:
        def _evaluate_type1_rubric(
            self, response: str, instruction_id: str, kwargs: dict[str, object]
        ) -> float | None:
            del response, kwargs
            return 1.0 if instruction_id == "supported" else None

        def _evaluate_type2_rubric(
            self, response: str, instruction_id: str, kwargs: dict[str, object]
        ) -> float | None:
            del response, instruction_id, kwargs
            return None

        def _evaluate_type3_rubric(self, response: str, constraint: list[object]) -> float | None:
            del response, constraint
            return None

        def _evaluate_rubric(self, *args: object) -> tuple[float, dict[str, object]]:
            del args
            return -1.0, {"method": "llm_judge", "llm_response": "fallback"}

    protocol = types.ModuleType("roll.distributed.scheduler.protocol")
    protocol.DataProto = DataProto
    reward = types.ModuleType("roll.pipeline.rlvr.rewards.rubrics_llm_judge_reward_worker")
    reward.RubricsLLMJudgeRewardWorker = RewardWorker
    reward.INSTRUCTION_ID_TO_IFEVAL = {"supported": ("supported_fn", {})}
    reward.IF_FUNCTIONS_MAP = {"supported_fn": lambda response, **kwargs: bool(response and kwargs == {})}
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
    path = ROOT / "src/rdan_grpo/reward_worker.py"
    spec = importlib.util.spec_from_file_location("test_roll_reward_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _hir_row(*, hard: bool = True, soft: bool = True) -> dict[str, object]:
    rubrics: list[dict[str, object]] = []
    checkers: list[str] = []
    functions: list[str] = []
    if hard:
        rubrics.append({"id": 1, "category": "", "description": "Use exactly one sentence.", "weight": 1})
        checkers.append("[rule] sentence count")
        functions.append("def check_following(instruction, response):\n    return True")
    if soft:
        rubric_id = len(rubrics) + 1
        rubrics.append({"id": rubric_id, "category": "", "description": "Be useful.", "weight": 1})
        checkers.append("[llm] usefulness")
        functions.append("")
    return {
        "prompt": "Answer the request.",
        "rubrics": rubrics,
        "source": "type4",
        "ground_truth": {"checker": checkers, "functions": functions},
    }


def _reward_data(module: types.ModuleType, row: dict[str, object], response: str = "answer") -> object:
    class Tokenizer:
        def batch_decode(self, tokens: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
            assert skip_special_tokens is True
            assert tokens.shape == (1, 3)
            return [response]

    data = module.DataProto({"responses": torch.tensor([[1, 2, 3]])})
    data.non_tensor_batch = {
        "id": ["row-1"],
        "prompt": [row["prompt"]],
        "rubrics": [row["rubrics"]],
        "source": [row["source"]],
        "ground_truth": [row["ground_truth"]],
        "rdan_prompt_key": ["row-1"],
    }
    return data, Tokenizer()


def _rubrichub_row() -> dict[str, object]:
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


def test_real_hir_soft_rubric_matches_calibrated_request_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    with (ROOT / "data/HIR_trainv1_rdan_scalar_certified.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            hard = module._hard_mask(row["source"], row["ground_truth"], len(row["rubrics"]))
            if False in hard:
                index = hard.index(False)
                break
        else:
            raise AssertionError("mixed HIR fixture not found")

    rubric = row["rubrics"][index]
    with (ROOT / "data/HIR_trainv1.jsonl").open(encoding="utf-8") as handle:
        source = next(json.loads(line) for line in handle if json.loads(line)["id"] == row["id"])
    assert rubric["description"] == source["criteria"][index]
    assert module._soft_rubric(rubric, index) == {"id": index + 1, "text": rubric["description"]}
    with pytest.raises(ValueError, match="description"):
        module._soft_rubric({**rubric, "description": {"nested": "invalid"}}, index)


def test_rubric_evidence_is_strict_redacted_json_without_reasoning_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    rubric = {"id": 1, "description": "Be correct."}
    info = {
        "method": "llm_judge",
        "reason": "credential sk-or-v1-abcdefghijklmnopqrstuvwxyz",
        "generation_id": "gen-1",
        "request_sha256": "a" * 64,
        "provider": "OpenAI",
        "model": "openai/gpt-5.6-luna",
        "selected_endpoint": {"provider_name": "OpenAI"},
        "finish_reason": "stop",
        "schema_id": "rubric_scores_v1",
        "reasoning_effort": "low",
        "latency_ms": 25.0,
        "cost": 0.01,
        "generation_metadata_polls": 1,
        "error": None,
        "tokens": {"prompt": 10, "completion": 4},
        "quality_valid": True,
        "reasoning_details": {"private": "must not persist"},
    }

    evidence = module._rubric_evidence(info, rubric, 0, 1.0)

    assert evidence["generation_id"] == "gen-1"
    assert evidence["request_sha256"] == "a" * 64
    assert evidence["reason"] == "[REDACTED]"
    assert evidence["judge_provenance"] == {
        "generation_id": "gen-1",
        "selected_endpoint": {"provider_name": "OpenAI"},
        "provider": "OpenAI",
        "model": "openai/gpt-5.6-luna",
        "finish_reason": "stop",
        "service_tier": None,
        "schema_id": "rubric_scores_v1",
        "rubric_ids": None,
        "tokens": {"prompt": 10, "completion": 4},
        "reasoning_effort": "low",
        "latency_ms": 25.0,
        "cost": 0.01,
        "generation_metadata_polls": 1,
        "error": None,
        "request_sha256": "a" * 64,
    }
    assert "reasoning_details" not in evidence
    json.dumps(evidence, allow_nan=False)


def test_rubrichub_reward_uses_verbatim_response_and_one_soft_judgment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    row = _rubrichub_row()
    response = "<think>hidden, text</think>answer"
    decoded: list[str] = []
    calls: list[tuple[str, str, list[dict[str, object]]]] = []

    class Tokenizer:
        def batch_decode(self, tokens: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
            assert skip_special_tokens is True
            assert tokens.shape == (1, 3)
            decoded.append(response)
            return [response]

    worker = module.ScalarRubricRewardWorker.__new__(module.ScalarRubricRewardWorker)
    worker.actor_tokenizer = Tokenizer()
    worker._rubrichub_contract = module._load_rubrichub_contract()

    def judge(
        prompt: str, answer: str, rubrics: list[dict[str, object]]
    ) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
        calls.append((prompt, answer, rubrics))
        return {2: {"score": 1.0, "reason": "meets the task"}}, {"quality_valid": True}

    worker._judge_soft = judge
    data = module.DataProto({"responses": torch.tensor([[1, 2, 3]])})
    data.non_tensor_batch = {
        "id": ["rubrichub-if-00274-df90080e98e2"],
        "prompt": [row["prompt"]],
        "rubrics": [row["rubrics"]],
        "source": [row["source"]],
        "ground_truth": [row["ground_truth"]],
        "rdan_prompt_key": ["rubrichub-if-00274-df90080e98e2"],
    }

    output = worker._compute_rewards_impl(data, {})

    assert decoded == [response]
    assert calls == [
        (
            row["prompt"],
            response,
            [{"id": 2, "text": row["rubrics"][1]["description"]}],
        )
    ]
    assert output.batch["rubric_scores_list"][0, :2].tolist() == [-1.0, 1.0]
    assert output.batch["response_level_rewards"].tolist() == [0.0]
    assert output.batch["rdan_hard_mask"][0, :2].tolist() == [True, False]
    assert [info["method"] for info in worker._rdan_infos] == ["rubrichub_rule", "llm_judge"]
    evidence = output.non_tensor_batch["rdan_rubric_evidence"][0]
    assert evidence[0]["evaluator_route"] == "rubrichub_rule"
    assert evidence[0]["reason"] == "failed"


def test_hybrid_worker_uses_supported_rtt_hard_evaluator_without_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    row = {
        "prompt": "Answer in the requested format.",
        "rubrics": [{"id": 1, "description": "Use the supported format."}],
        "source": "type1",
        "ground_truth": {"instruction_id_list": ["supported"], "kwargs": [{}]},
    }
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer

    def unexpected(*args: object) -> object:
        raise AssertionError(f"judge must not be called: {args}")

    worker._judge_batch = unexpected
    output = worker._compute_rewards_impl(data, {})

    assert output.batch["rubric_scores_list"][0, 0].item() == 1.0
    assert output.batch["rdan_hard_mask"][0, 0].item() is True
    assert output.batch["rdan_hybrid_hard_fallback"].tolist() == [False]
    assert output.non_tensor_batch["rdan_reward_lane"].tolist() == ["hybrid_rtt_compatible"]
    assert output.non_tensor_batch["rdan_rubric_evidence"][0][0]["evaluator_route"] == "code_type1"


def test_hybrid_worker_batches_hard_fallback_and_soft_quality_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    row = _hir_row()
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer
    calls: list[list[dict[str, object]]] = []

    def local(*args: object) -> tuple[None, None, str]:
        del args
        return None, None, "rule_sandbox:hash_not_allowed"

    def judge(
        prompt: str, response: str, rubrics: list[dict[str, object]]
    ) -> tuple[dict[int, dict[str, object]], dict[str, object], bool]:
        assert prompt == row["prompt"]
        assert response == "answer"
        calls.append(rubrics)
        return (
            {
                1: {"score": 1.0, "reason": "constraint met"},
                2: {"score": -1.0, "reason": "not useful"},
            },
            {"generation_id": "gen-1", "request_sha256": "a" * 64},
            True,
        )

    worker._evaluate_hir_hard = local
    worker._judge_batch = judge
    output = worker._compute_rewards_impl(data, {})

    assert calls == [
        [
            {"id": 1, "text": "Use exactly one sentence."},
            {"id": 2, "text": "Be useful."},
        ]
    ]
    assert output.batch["rubric_scores_list"][0, :2].tolist() == [1.0, -1.0]
    assert output.batch["rdan_hard_mask"][0, :2].tolist() == [True, False]
    assert output.batch["rdan_eval_mask"][0, :2].tolist() == [True, True]
    assert output.batch["rdan_hybrid_hard_fallback"].tolist() == [True]
    assert output.batch["rdan_unsupported_hard"].tolist() == [False]
    evidence = output.non_tensor_batch["rdan_rubric_evidence"][0]
    assert [item["judge_role"] for item in evidence] == ["hard_fallback", "soft_quality"]
    assert evidence[0]["fallback_reason"] == "rule_sandbox:hash_not_allowed"
    assert all(item["reward_lane"] == "hybrid_rtt_compatible" for item in evidence)


def test_hybrid_worker_marks_judge_failure_invalid_without_positive_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    row = _hir_row()
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer
    worker._evaluate_hir_hard = lambda *args: (None, None, "unsupported_route:type4")
    worker._judge_batch = lambda *args: (
        {1: {"score": 0.0, "reason": "TimeoutError"}, 2: {"score": 0.0, "reason": "TimeoutError"}},
        {"error": "TimeoutError", "request_sha256": "b" * 64},
        False,
    )

    output = worker._compute_rewards_impl(data, {})

    assert output.batch["rubric_scores_list"][0, :2].tolist() == [0.0, 0.0]
    assert output.batch["rdan_eval_mask"][0, :2].tolist() == [False, False]
    assert output.batch["rdan_judge_failed"].tolist() == [True]
    assert output.batch["response_level_rewards"].tolist() == [0.0]
    assert all(item["judge_failed"] for item in output.non_tensor_batch["rdan_rubric_evidence"][0])


def test_hybrid_worker_rejects_malformed_successful_judgment(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    row = _hir_row(hard=False)
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer
    worker._judge_batch = lambda *args: (
        {1: {"score": 1.0, "reason": "useful", "extra": True}},
        {"generation_id": "gen-bad"},
        True,
    )

    output = worker._compute_rewards_impl(data, {})

    assert output.batch["rubric_scores_list"][0, 0].item() == 0.0
    assert output.batch["rdan_eval_mask"][0, 0].item() is False
    assert output.batch["rdan_judge_failed"].tolist() == [True]
    assert output.batch["response_level_rewards"].tolist() == [0.0]
    evidence = output.non_tensor_batch["rdan_rubric_evidence"][0][0]
    assert evidence["judge_provenance"]["error"] == "malformed_judgments"


@pytest.mark.parametrize("signed", [-1.0, 0.0, 1.0])
def test_hybrid_worker_keeps_every_process_tier(monkeypatch: pytest.MonkeyPatch, signed: float) -> None:
    module = _module(monkeypatch)
    row = _hir_row(hard=False)
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer
    worker._judge_batch = lambda *args: (
        {1: {"score": signed, "reason": "useful"}},
        {"generation_id": "gen-1"},
        True,
    )

    output = worker._compute_rewards_impl(data, {})

    assert output.batch["rubric_scores_list"][0, 0].item() == signed
    assert output.batch["rdan_eval_mask"][0, 0].item() is True
    assert output.batch["rdan_judge_failed"].tolist() == [False]


def test_hybrid_worker_rejects_malformed_hir_metadata_before_judging(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    row = {
        "prompt": "Answer in the requested format.",
        "rubrics": [{"id": 1, "description": "Use the format."}],
        "source": "type1",
        "ground_truth": {"instruction_id_list": [], "kwargs": []},
    }
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer

    def unexpected(*args: object) -> object:
        raise AssertionError(f"judge must not be called: {args}")

    worker._judge_batch = unexpected
    with pytest.raises(ValueError, match="metadata is malformed"):
        worker._compute_rewards_impl(data, {})


def test_hybrid_soft_only_outcome_gate_passes_when_quality_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    row = _hir_row(hard=False)
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer
    worker._judge_batch = lambda *args: (
        {1: {"score": 1.0, "reason": "useful"}},
        {"generation_id": "gen-2", "request_sha256": "c" * 64},
        True,
    )

    output = worker._compute_rewards_impl(data, {})

    assert output.batch["rdan_hard_mask"][0, 0].item() is False
    assert output.batch["rdan_eval_mask"][0, 0].item() is True
    assert output.batch["response_level_rewards"].tolist() == [1.0]
    evidence = output.non_tensor_batch["rdan_rubric_evidence"][0][0]
    assert evidence["judge_role"] == "soft_quality"
    assert evidence["reward_lane"] == "hybrid_rtt_compatible"


def test_hybrid_rubrichub_hard_route_stays_local(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    row = _rubrichub_row()
    data, tokenizer = _reward_data(module, row, response="answer")
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer
    worker._rubrichub_contract = module._load_rubrichub_contract()
    calls: list[list[dict[str, object]]] = []

    def judge(
        prompt: str, response: str, rubrics: list[dict[str, object]]
    ) -> tuple[dict[int, dict[str, object]], dict[str, object], bool]:
        del prompt, response
        calls.append(rubrics)
        return {2: {"score": 1.0, "reason": "useful"}}, {"generation_id": "gen-rh"}, True

    worker._judge_batch = judge
    output = worker._compute_rewards_impl(data, {})

    assert calls == [[{"id": 2, "text": row["rubrics"][1]["description"]}]]
    assert [info["method"] for info in worker._rdan_infos] == ["rubrichub_rule", "llm_judge"]
    evidence = output.non_tensor_batch["rdan_rubric_evidence"][0]
    assert evidence[0]["reward_lane"] == "hybrid_rtt_compatible"
    assert evidence[0]["judge_role"] is None
    assert evidence[1]["judge_role"] == "soft_quality"


def test_type4_safe_rule_failure_stays_fail_closed_in_hybrid_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    result = types.SimpleNamespace(valid=False, value=False, error="timeout")
    monkeypatch.setattr(module, "evaluate_rule", lambda *args, **kwargs: result)
    row = _hir_row(soft=False)
    data, tokenizer = _reward_data(module, row)
    hybrid = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    hybrid.actor_tokenizer = tokenizer
    hybrid._type4_hashes = frozenset({"a" * 64})
    hybrid._judge_batch = lambda *args: pytest.fail("deterministic failure reached Luna")

    output = hybrid._compute_rewards_impl(data, {})

    evidence = output.non_tensor_batch["rdan_rubric_evidence"][0][0]
    assert evidence["judge_role"] is None
    assert evidence["evaluator_failed"] is True
    assert output.batch["rdan_eval_mask"][0, 0].item() is False
    assert output.batch["response_level_rewards"].tolist() == [0.0]

    strict = module.ScalarRubricRewardWorker.__new__(module.ScalarRubricRewardWorker)
    strict.actor_tokenizer = tokenizer
    strict._type4_hashes = frozenset({"a" * 64})
    with pytest.raises(RuntimeError, match="failed closed"):
        strict._compute_rewards_impl(data, {})


@pytest.mark.parametrize("source", ["type1", "type2", "type3"])
def test_supported_hir_evaluator_failure_cannot_reach_luna(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    module = _module(monkeypatch)

    def fail(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise RuntimeError("checker failed")

    if source == "type1":
        module.IF_FUNCTIONS_MAP["supported_fn"] = fail
        truth = {"instruction_id_list": ["supported"], "kwargs": [{}]}
    elif source == "type2":
        module.TYPE2_CHECKERS["supported"] = fail
        truth = {"instruction_id_list": ["supported"], "kwargs": [{}]}
    else:
        module.CONSTRAINT_CHECKER_MAP["Format_Table"] = types.SimpleNamespace(check=fail)
        truth = {"constraints": [["Format", "Table", "Use a table."]]}
    row = {
        "prompt": "Answer in the requested format.",
        "rubrics": [{"id": 1, "description": "Use the supported format."}],
        "source": source,
        "ground_truth": truth,
    }
    data, tokenizer = _reward_data(module, row)
    worker = module.RTTCompatibleRubricRewardWorker.__new__(module.RTTCompatibleRubricRewardWorker)
    worker.actor_tokenizer = tokenizer
    worker._judge_batch = lambda *args: pytest.fail("deterministic failure reached Luna")

    output = worker._compute_rewards_impl(data, {})

    assert output.batch["response_level_rewards"].tolist() == [0.0]
    assert output.batch["rdan_eval_mask"][0, 0].item() is False
    evidence = output.non_tensor_batch["rdan_rubric_evidence"][0][0]
    assert evidence["evaluator_failed"] is True
    assert evidence["judge_role"] is None


def test_strict_worker_still_rejects_rtt_hard_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    row = {
        "prompt": "Answer in the requested format.",
        "rubrics": [{"id": 1, "description": "Use an unsupported format."}],
        "source": "type1",
        "ground_truth": {"instruction_id_list": ["unsupported"], "kwargs": [{}]},
    }
    data, tokenizer = _reward_data(module, row)
    worker = module.ScalarRubricRewardWorker.__new__(module.ScalarRubricRewardWorker)
    worker.actor_tokenizer = tokenizer

    with pytest.raises(RuntimeError, match="hard evaluator failed closed"):
        worker._compute_rewards_impl(data, {})


def test_rubrichub_row_contract_rejects_tampering_and_unknown_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    contract = module._load_rubrichub_contract()
    row = _rubrichub_row()

    def validate(candidate: dict[str, object]) -> None:
        truth = candidate["ground_truth"]
        rubrics = candidate["rubrics"]
        mask = module._hard_mask(candidate["source"], truth, len(rubrics))
        module._validate_rubrichub_row(candidate["prompt"], rubrics, truth, mask, contract)

    validate(row)
    tokenizer = contract["tokenizer_certificate"]
    assert len(tokenizer["accepted_source_indices"]) == 1_134
    assert 4_979 not in tokenizer["accepted_source_indices"]

    tokenizer_rejected = deepcopy(row)
    tokenizer_rejected["ground_truth"]["source_provenance"]["source_index"] = 4_979
    tokenizer_rejected["ground_truth"]["source_provenance"]["source_row_sha256"] = contract["rows"][4_979][
        "source_row_sha256"
    ]
    with pytest.raises(ValueError, match="not tokenizer-certified"):
        validate(tokenizer_rejected)

    ineligible = deepcopy(row)
    ineligible["ground_truth"]["rl_eligible"] = False
    with pytest.raises(ValueError, match="not certified"):
        validate(ineligible)

    mask_tamper = deepcopy(row)
    mask_tamper["ground_truth"]["hard_mask"][0] = False
    with pytest.raises(ValueError, match="differs from the frozen rubric"):
        validate(mask_tamper)

    route_tamper = deepcopy(row)
    route_tamper["ground_truth"]["rubric_routes"][0]["parameters"] = '{"unused":1}'
    with pytest.raises(ValueError, match="differs from the frozen rubric"):
        validate(route_tamper)

    noncanonical = deepcopy(row)
    noncanonical["rubrics"][0]["parameters"] = "{ }"
    noncanonical["ground_truth"]["rubric_routes"][0]["parameters"] = "{ }"
    with pytest.raises(ValueError, match="canonical JSON object"):
        validate(noncanonical)

    provenance_tamper = deepcopy(row)
    provenance_tamper["ground_truth"]["source_provenance"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="frozen source"):
        validate(provenance_tamper)

    source_row_tamper = deepcopy(row)
    source_row_tamper["ground_truth"]["source_provenance"]["source_row_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="certified subset"):
        validate(source_row_tamper)

    prompt_tamper = deepcopy(row)
    prompt_tamper["prompt"] += " changed"
    with pytest.raises(ValueError, match="tokenizer certificate"):
        validate(prompt_tamper)

    route_name_tamper = deepcopy(row)
    route_name_tamper["rubrics"][0]["function"] = "__import__('os').system"
    route_name_tamper["ground_truth"]["rubric_routes"][0]["function"] = "__import__('os').system"
    with pytest.raises(ValueError, match="certified source row"):
        validate(route_name_tamper)

    with pytest.raises(ValueError, match="unsupported reward source"):
        module._hard_mask("unknown", {}, 1)


def test_rubrichub_certificate_and_evidence_are_verified_at_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module(monkeypatch)
    contract = module._load_rubrichub_contract()
    rule = json.loads(module.RUBRICHUB_RULE_CERTIFICATE.read_text(encoding="utf-8"))
    rule["candidate_selection"]["source_indices"] = rule["candidate_selection"]["source_indices"][1:]
    with pytest.raises(RuntimeError, match="inventory is inconsistent"):
        module._load_rubrichub_evidence(rule, contract["functions"])

    tokenizer = json.loads(module.RUBRICHUB_TOKENIZER_CERTIFICATE.read_text(encoding="utf-8"))
    wrong_checker = deepcopy(tokenizer)
    wrong_checker["selection"]["checker_certificate"]["sha256"] = "0" * 64
    tampered = tmp_path / "rubrichub_tokenizer_wrong_checker.json"
    tampered.write_text(json.dumps(wrong_checker), encoding="utf-8")
    monkeypatch.setattr(module, "RUBRICHUB_TOKENIZER_CERTIFICATE", tampered)
    config = json.loads(module.RUBRICHUB_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="selection is invalid"):
        module._load_rubrichub_tokenizer(config, contract["source"], contract["rows"])

    tokenizer["results"]["accepted_source_indices"] = tokenizer["results"]["accepted_source_indices"][1:]
    tampered = tmp_path / "rubrichub_tokenizer_certificate.json"
    tampered.write_text(json.dumps(tokenizer), encoding="utf-8")
    monkeypatch.setattr(module, "RUBRICHUB_TOKENIZER_CERTIFICATE", tampered)
    with pytest.raises(RuntimeError, match="inventory is inconsistent"):
        module._load_rubrichub_tokenizer(config, contract["source"], contract["rows"])
