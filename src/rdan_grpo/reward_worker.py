"""ROLL reward worker: deterministic hard rubrics locally, soft rubrics through one judge call.

Hard rubrics run in-process; soft rubrics for the whole batch are judged concurrently in a
single fan-out. Both channels fail soft: a checker or judge failure clears that rubric's
``eval_mask`` bit so the advantage layer withholds credit, and the step still trains.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.rewards.rubrics_llm_judge_reward_worker import (
    CONSTRAINT_CHECKER_MAP,
    IF_FUNCTIONS_MAP,
    INSTRUCTION_ID_TO_IFEVAL,
    TYPE2_CHECKERS,
    RubricsLLMJudgeRewardWorker,
    call_ifeval_function,
)

from rdan_grpo.judge import JudgeRequest, OpenRouterJudge, aggregate_stats, load_judge_config
from rdan_grpo.rules import evaluate_python_rule, evaluate_rubrichub_rule

ROOT = Path(__file__).resolve().parents[2]
JUDGE_CONFIG = ROOT / "configs/judges/openrouter.json"
RUBRICHUB_SOURCE = "rubrichub_instruction_following"
# Fixed rubric axis: every batch is one dense [responses, MAX_RUBRICS] tensor.
MAX_RUBRICS = 20
INACTIVE = -100.0


class RubricRewardWorker(RubricsLLMJudgeRewardWorker):
    """Score each response's rubrics and publish the RDAN reward fields on the batch."""

    def __init__(self, worker_config: Any):
        super().__init__(worker_config)
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        config = load_judge_config(JUDGE_CONFIG)
        if self.judge_api_url:
            config["base_url"] = self.judge_api_url
        self.judge = OpenRouterJudge(config, api_key)
        self._rubric_counts = _empty_counts()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_reward_stats(self) -> dict[str, Any]:
        """Return and reset this worker's judge and checker counters for one step.

        ROLL's scheduler replaces the batch meta_info metrics after concatenation, so reward
        health has to be pulled from the workers rather than ridden in on the batch.
        """

        counts, self._rubric_counts = self._rubric_counts, _empty_counts()
        return {**self.judge.drain_stats(), **counts}

    def _compute_rewards_impl(self, data: DataProto, metrics: dict[str, Any]) -> DataProto:
        responses = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        rows = [
            _evaluate_hard_rubrics(prompt, response, rubrics, str(source), _ground_truth(truth))
            for prompt, response, rubrics, source, truth in zip(
                data.non_tensor_batch["prompt"],
                responses,
                data.non_tensor_batch["rubrics"],
                data.non_tensor_batch["source"],
                data.non_tensor_batch["ground_truth"],
                strict=True,
            )
        ]

        pending = [index for index, row in enumerate(rows) if row["judge_rubrics"]]
        requests = [
            JudgeRequest(
                str(data.non_tensor_batch["prompt"][index]), responses[index], tuple(rows[index]["judge_rubrics"])
            )
            for index in pending
        ]
        for index, result in zip(pending, self.judge.judge_batch(requests), strict=True):
            _apply_judgments(rows[index], result)

        self._rubric_counts = _accumulate_counts(self._rubric_counts, rows)
        return _build_output(data, rows, metrics)


def _evaluate_hard_rubrics(
    prompt: Any,
    response: str,
    rubrics: Any,
    source: str,
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Score every hard rubric now and collect the soft ones for the judge fan-out."""

    if not isinstance(rubrics, list) or not 1 <= len(rubrics) <= MAX_RUBRICS:
        raise ValueError(f"rubrics must contain between 1 and {MAX_RUBRICS} entries")
    hard_mask = _hard_mask(source, truth, len(rubrics))
    routes = _rubrichub_routes(truth, len(rubrics)) if source == RUBRICHUB_SOURCE else None

    scores = [INACTIVE] * MAX_RUBRICS
    evaluated = [False] * MAX_RUBRICS
    hard = [False] * MAX_RUBRICS
    judge_rubrics: list[dict[str, Any]] = []
    checker_failures = 0
    for index, rubric in enumerate(rubrics):
        if not hard_mask[index]:
            judge_rubrics.append(_soft_rubric(rubric, index))
            scores[index] = 0.0
            continue
        hard[index] = True
        passed = _run_checker(prompt, response, source, truth, index, routes)
        if passed is None:
            scores[index] = 0.0
            checker_failures += 1
            continue
        scores[index] = 1.0 if passed else -1.0
        evaluated[index] = True
    return {
        "active": len(rubrics),
        "scores": scores,
        "evaluated": evaluated,
        "hard": hard,
        "judge_rubrics": judge_rubrics,
        "checker_failures": checker_failures,
        "judge_failed": False,
    }


def _run_checker(
    prompt: Any,
    response: str,
    source: str,
    truth: Mapping[str, Any],
    index: int,
    routes: list[Mapping[str, Any]] | None,
) -> bool | None:
    """Return the hard rubric verdict, or None when the checker could not decide."""

    try:
        if routes is not None:
            result = evaluate_rubrichub_rule(routes[index]["function"], response, routes[index]["parameters"])
            return result.passed if result.valid else None
        if source in {"type1", "type2"}:
            instruction_id = truth["instruction_id_list"][index]
            kwargs = truth["kwargs"][index]
            if source == "type2":
                checker = TYPE2_CHECKERS.get(instruction_id)
                return None if checker is None else bool(checker(response, kwargs))
            route = INSTRUCTION_ID_TO_IFEVAL.get(instruction_id)
            if route is None:
                return None
            function_name, remap = route
            function = IF_FUNCTIONS_MAP.get(function_name)
            if function is None:
                return None
            return bool(call_ifeval_function(function, response, {remap.get(k, k): v for k, v in kwargs.items()}))
        if source == "type3":
            constraint = truth["constraints"][index]
            checker = CONSTRAINT_CHECKER_MAP.get(f"{constraint[0]}_{constraint[1]}")
            return None if checker is None else bool(checker.check(constraint[2], response))
        if source == "type4":
            result = evaluate_python_rule(truth["functions"][index], str(prompt), response)
            return result.passed if result.valid else None
    except Exception:  # noqa: BLE001 - a checker that raised simply did not decide
        return None
    return None


def _apply_judgments(row: dict[str, Any], result: Any) -> None:
    """Write back every rubric the judge scored, leaving any it skipped unevaluated."""

    if not result.valid:
        row["judge_failed"] = True
        return
    for rubric in row["judge_rubrics"]:
        judgment = result.judgments.get(rubric["id"])
        if judgment is None:
            # Withhold credit for this rubric only; the rest of the response still counts.
            row["judge_failed"] = True
            continue
        index = rubric["id"] - 1
        row["scores"][index] = float(judgment["score"])
        row["evaluated"][index] = True


def _build_output(data: DataProto, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> DataProto:
    scores = torch.tensor([row["scores"] for row in rows], dtype=torch.float32)
    active = torch.zeros_like(scores, dtype=torch.bool)
    for index, row in enumerate(rows):
        active[index, : row["active"]] = True
    evaluated = torch.tensor([row["evaluated"] for row in rows], dtype=torch.bool) & active
    hard = torch.tensor([row["hard"] for row in rows], dtype=torch.bool) & active
    outcome = ((scores == 1) | ~(hard & active)).all(-1) & (evaluated | ~(hard & active)).all(-1)

    output = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.zeros_like(data.batch["responses"], dtype=torch.float16),
            "response_level_rewards": outcome.to(torch.float16),
            "scores": outcome.to(torch.float16),
            "rdan_scores": torch.where(active, scores, torch.zeros_like(scores)),
            "rdan_rubric_mask": active,
            "rdan_eval_mask": evaluated,
            "rdan_hard_mask": hard,
        }
    )
    output.meta_info = {"metrics": metrics}
    output.non_tensor_batch["rdan_prompt_key"] = np.asarray(_prompt_keys(data), dtype=object)
    return output


def _empty_counts() -> dict[str, int]:
    return {"responses": 0, "rubrics": 0, "checker_failures": 0, "judged_responses": 0, "judge_failed_responses": 0}


def _accumulate_counts(counts: dict[str, int], rows: list[dict[str, Any]]) -> dict[str, int]:
    """Add one batch's checker and judge outcomes to this worker's running totals."""

    return {
        "responses": counts["responses"] + len(rows),
        "rubrics": counts["rubrics"] + sum(row["active"] for row in rows),
        "checker_failures": counts["checker_failures"] + sum(row["checker_failures"] for row in rows),
        "judged_responses": counts["judged_responses"] + sum(1 for row in rows if row["judge_rubrics"]),
        "judge_failed_responses": counts["judge_failed_responses"] + sum(1 for row in rows if row["judge_failed"]),
    }


def aggregate_reward_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Combine per-worker reward counters from one step into checker and judge health metrics."""

    totals = {name: sum(int(row.get(name, 0)) for row in rows) for name in _empty_counts()}
    if not totals["responses"]:
        return aggregate_stats(rows)
    return {
        **aggregate_stats(rows),
        "reward/checker_failure_rate": totals["checker_failures"] / max(totals["rubrics"], 1),
        "reward/judged_response_rate": totals["judged_responses"] / totals["responses"],
        "reward/judge_failed_rate": totals["judge_failed_responses"] / max(totals["judged_responses"], 1),
    }


def _prompt_keys(data: DataProto) -> list[str]:
    """Group key per response: the dataset id when it survives preprocessing, else the prompt."""

    ids = data.non_tensor_batch.get("id")
    if ids is not None:
        return [str(value) for value in ids]
    return [hashlib.sha256(str(prompt).encode()).hexdigest() for prompt in data.non_tensor_batch["prompt"]]


def _hard_mask(source: str, truth: Mapping[str, Any], count: int) -> list[bool]:
    if source in {"type1", "type2", "type3"}:
        return [True] * count
    if source == "type4":
        checkers = truth.get("checker")
        if not isinstance(checkers, list) or len(checkers) != count:
            raise ValueError("type4 checker metadata does not match the rubric count")
        return [isinstance(checker, str) and checker.startswith("[rule]") for checker in checkers]
    if source == RUBRICHUB_SOURCE:
        mask = truth.get("hard_mask")
        if not isinstance(mask, list) or len(mask) != count or any(not isinstance(value, bool) for value in mask):
            raise ValueError("RubricHub hard mask is malformed")
        return list(mask)
    raise ValueError(f"unsupported reward source: {source}")


def _rubrichub_routes(truth: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    """Decode the per-rubric rule routes, whose parameters are canonical JSON strings."""

    routes = truth.get("rubric_routes")
    if not isinstance(routes, list) or len(routes) != count:
        raise ValueError("RubricHub rubric routes do not match the rubric count")
    decoded = []
    for route in routes:
        parameters = route.get("parameters")
        decoded.append({**route, "parameters": json.loads(parameters) if isinstance(parameters, str) else parameters})
    return decoded


def _soft_rubric(rubric: Any, index: int) -> dict[str, Any]:
    if not isinstance(rubric, dict):
        raise ValueError("soft rubric must be an object")
    description = rubric.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("soft rubric description must be a non-empty string")
    return {"id": index + 1, "text": description}


def _ground_truth(value: Any) -> Mapping[str, Any]:
    truth = json.loads(value) if isinstance(value, str) else value
    if not isinstance(truth, dict):
        raise ValueError("ground_truth must be an object")
    return truth
