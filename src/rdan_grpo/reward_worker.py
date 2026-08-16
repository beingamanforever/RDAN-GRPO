"""ROLL reward worker for response and conditional quality channels."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.rewards.rubrics_llm_judge_reward_worker import (
    CONSTRAINT_CHECKER_MAP,
    IF_FUNCTIONS_MAP,
    INSTRUCTION_ID_TO_IFEVAL,
    TYPE2_CHECKERS,
    RubricsLLMJudgeRewardWorker,
    call_ifeval_function,
)

from rdan_grpo.evaluator import verify_type4_certificate
from rdan_grpo.judge import SIGNED_PROCESS_SCORES, OpenRouterJudge
from rdan_grpo.bridge import attach_roll_reward_fields
from rdan_grpo.rules import evaluate_rubrichub_rule, verify_rule_certificate
from rdan_grpo.rule_sandbox import evaluate_rule
from rdan_grpo.scalar_data import verify_scalar_dataset
from rdan_grpo.tracking import redact_secrets

ROOT = Path(__file__).resolve().parents[2]
JUDGE_CONFIG = ROOT / "configs/judges/openrouter_luna.json"
JUDGE_PROMPT = ROOT / "configs/judges/rubric_prompt.txt"
TYPE4_CERTIFICATE = ROOT / "configs/artifacts/hir_type4_rule_certificate.json"
TYPE4_EVIDENCE = ROOT / "configs/artifacts/hir_type4_rule_evidence.jsonl"
SCALAR_MANIFEST = ROOT / "configs/artifacts/hir_scalar_certified_manifest.json"
TUNING_SEED = 240520
JUDGE_CERTIFICATE = ROOT / "configs/artifacts/qwen_judge_calibration.json"
RUBRICHUB_CONFIG = ROOT / "configs/data/rubrichub_instruction_following.json"
RUBRICHUB_RULE_CERTIFICATE = ROOT / "configs/artifacts/rubrichub_rule_certificate.json"
RUBRICHUB_TOKENIZER_CERTIFICATE = ROOT / "configs/artifacts/rubrichub_tokenizer_certificate.json"
RUBRICHUB_SOURCE = "rubrichub_instruction_following"


class ScalarRubricRewardWorker(RubricsLLMJudgeRewardWorker):
    """Evaluate hard rules locally and all soft rubrics once per response."""

    def __init__(self, worker_config: Any):
        super().__init__(worker_config)
        self._judge_contract = _load_judge_contract()
        self._judge_effort = _load_calibrated_effort()
        self._judge_seed = _load_program_seed()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key or not self.judge_api_url:
            raise ValueError("OPENROUTER_API_KEY and judge_api_url are required")
        self._judge = OpenRouterJudge(
            self._judge_contract["config"],
            self._judge_contract["prompt"],
            key,
            base_url=self.judge_api_url,
        )
        self._type4_hashes = _load_type4_hashes()
        self._rubrichub_contract = _load_rubrichub_contract()
        verify_scalar_dataset(SCALAR_MANIFEST)

    def _compute_rewards_impl(self, data: Any, metrics: dict[str, Any]) -> Any:
        self._rdan_infos: list[dict[str, Any]] = []
        responses = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        rubric_rows: list[list[float]] = []
        hard_rows: list[list[bool]] = []
        evidence_rows: list[list[dict[str, Any]]] = []
        for prompt, response, rubrics, source, raw_truth in zip(
            data.non_tensor_batch["prompt"],
            responses,
            data.non_tensor_batch["rubrics"],
            data.non_tensor_batch["source"],
            data.non_tensor_batch["ground_truth"],
            strict=True,
        ):
            truth = json.loads(raw_truth) if isinstance(raw_truth, str) else raw_truth
            if not isinstance(truth, dict):
                raise ValueError("ground_truth must be an object")
            source = str(source)
            hard_mask = _hard_mask(source, truth, len(rubrics))
            routes = (
                _validate_rubrichub_row(prompt, rubrics, truth, hard_mask, self._rubrichub_contract)
                if source == RUBRICHUB_SOURCE
                else None
            )
            scores = [-100.0] * 20
            infos: list[dict[str, Any] | None] = [None] * min(len(rubrics), 20)
            soft: list[dict[str, Any]] = []
            for index, (rubric, hard) in enumerate(zip(rubrics[:20], hard_mask[:20], strict=True)):
                if hard:
                    if routes is not None:
                        score, info = _evaluate_rubrichub_hard(response, rubric, routes[index])
                    else:
                        score, info = super()._evaluate_rubric(
                            prompt, response, response, rubric, source, truth, index
                        )
                        if info.get("method") == "llm_judge":
                            raise RuntimeError(f"hard evaluator failed closed for {source} rubric {index}")
                    scores[index] = float(score)
                    infos[index] = dict(info)
                else:
                    soft.append(_soft_rubric(rubric, index))
            judgments, evidence = self._judge_soft(prompt, response, soft) if soft else ({}, {})
            for index, (rubric, hard) in enumerate(zip(rubrics[:20], hard_mask[:20], strict=True)):
                if hard:
                    continue
                rubric_id = index + 1
                scores[index] = float(judgments[rubric_id]["score"])
                infos[index] = {
                    "score": scores[index],
                    "method": "llm_judge",
                    "rubric": rubric,
                    "rubric_id": rubric_id,
                    "llm_response": "structured" if evidence["quality_valid"] else "",
                    "reason": judgments[rubric_id]["reason"],
                    **evidence,
                }
            if any(info is None for info in infos):
                raise RuntimeError("rubric evaluator evidence is incomplete")
            self._rdan_infos.extend(info for info in infos if info is not None)
            evidence_rows.append(
                [
                    _rubric_evidence(info, rubric, index, scores[index])
                    for index, (info, rubric) in enumerate(zip(infos, rubrics[:20], strict=True))
                    if info is not None
                ]
            )
            rubric_rows.append(scores)
            hard_rows.append(hard_mask[:20] + [False] * (20 - len(hard_mask[:20])))
        score_tensor = torch.tensor(rubric_rows, dtype=torch.float16)
        hard_tensor = torch.tensor(hard_rows, dtype=torch.bool)
        response_scores = ((score_tensor == 1) | ~hard_tensor).all(-1).to(torch.float16)
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.zeros_like(data.batch["responses"], dtype=torch.float16),
                "response_level_rewards": response_scores,
                "scores": response_scores,
                "rubric_scores_list": score_tensor,
            }
        )
        output.meta_info = {"metrics": metrics}
        output = attach_roll_reward_fields(data, output, self._rdan_infos)
        output.batch["rdan_hard_mask"] = hard_tensor.to(output.batch["rdan_hard_mask"].device)
        output.non_tensor_batch["rdan_prompt_key"] = np.asarray(data.non_tensor_batch["rdan_prompt_key"], dtype=object)
        evidence_array = np.empty(len(evidence_rows), dtype=object)
        evidence_array[:] = evidence_rows
        output.non_tensor_batch["rdan_rubric_evidence"] = evidence_array
        return output

    def _evaluate_type4_rule(self, prompt_text: str, response_text: str, function_code: str) -> float:
        result = evaluate_rule(
            function_code,
            prompt_text,
            response_text,
            allowed_hashes=self._type4_hashes,
            timeout_seconds=1.0,
        )
        if not result.valid:
            raise RuntimeError(f"type4 evaluator failed closed: {result.error}")
        return 1.0 if result.value else -1.0

    def _judge_soft(
        self, instruction: str, response: str, rubrics: list[dict[str, Any]]
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        if not rubrics:
            return {}, {}
        result = self._judge.judge(instruction, response, rubrics, self._judge_seed, self._judge_effort)
        return result.judgments, {**result.evidence, "quality_valid": result.valid}


class RTTCompatibleRubricRewardWorker(RubricsLLMJudgeRewardWorker):
    """Run the non-authoritative RTT-compatible hybrid reward lane."""

    def __init__(self, worker_config: Any):
        super().__init__(worker_config)
        self._judge_contract = _load_judge_contract()
        self._judge_effort = _load_calibrated_effort()
        self._judge_seed = _load_program_seed()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key or not self.judge_api_url:
            raise ValueError("OPENROUTER_API_KEY and judge_api_url are required")
        self._judge = OpenRouterJudge(
            self._judge_contract["config"],
            self._judge_contract["prompt"],
            key,
            base_url=self.judge_api_url,
        )
        self._type4_hashes = _load_type4_hashes()
        self._rubrichub_contract = _load_rubrichub_contract()

    def _compute_rewards_impl(self, data: Any, metrics: dict[str, Any]) -> Any:
        self._rdan_infos: list[dict[str, Any]] = []
        responses = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        rubric_rows: list[list[float]] = []
        hard_rows: list[list[bool]] = []
        eval_rows: list[list[bool]] = []
        fallback_rows: list[bool] = []
        judge_failed_rows: list[bool] = []
        evaluator_failed_rows: list[bool] = []
        evidence_rows: list[list[dict[str, Any]]] = []
        for prompt, response, rubrics, source, raw_truth in zip(
            data.non_tensor_batch["prompt"],
            responses,
            data.non_tensor_batch["rubrics"],
            data.non_tensor_batch["source"],
            data.non_tensor_batch["ground_truth"],
            strict=True,
        ):
            if not isinstance(rubrics, list) or not rubrics or len(rubrics) > 20:
                raise ValueError("rubrics must contain between 1 and 20 entries")
            truth = json.loads(raw_truth) if isinstance(raw_truth, str) else raw_truth
            if not isinstance(truth, dict):
                raise ValueError("ground_truth must be an object")
            row = self._evaluate_hybrid_row(str(prompt), response, rubrics, str(source), truth)
            rubric_rows.append(row["scores"])
            hard_rows.append(row["hard_mask"])
            eval_rows.append(row["eval_mask"])
            fallback_rows.append(row["hard_fallback"])
            judge_failed_rows.append(row["judge_failed"])
            evaluator_failed_rows.append(row["evaluator_failed"])
            evidence_rows.append(row["evidence"])

        score_tensor = torch.tensor(rubric_rows, dtype=torch.float16)
        hard_tensor = torch.tensor(hard_rows, dtype=torch.bool)
        eval_tensor = torch.tensor(eval_rows, dtype=torch.bool)
        active = score_tensor != -100
        hard_pass = ((score_tensor == 1) | ~hard_tensor).all(-1)
        response_scores = (hard_pass & (eval_tensor | ~active).all(-1)).to(torch.float16)
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.zeros_like(data.batch["responses"], dtype=torch.float16),
                "response_level_rewards": response_scores,
                "scores": response_scores,
                "rubric_scores_list": score_tensor,
            }
        )
        output.meta_info = {"metrics": metrics, "reward_lane": "hybrid_rtt_compatible"}
        output = attach_roll_reward_fields(data, output, self._rdan_infos)
        device = output.batch["rdan_scores"].device
        output.batch["rdan_hard_mask"] = hard_tensor.to(device)
        output.batch["rdan_eval_mask"] = eval_tensor.to(device)
        output.batch["rdan_unsupported_hard"] = torch.zeros(len(rubric_rows), dtype=torch.bool, device=device)
        output.batch["rdan_judge_failed"] = torch.tensor(judge_failed_rows, dtype=torch.bool, device=device)
        output.batch["rdan_hybrid_hard_fallback"] = torch.tensor(fallback_rows, dtype=torch.bool, device=device)
        output.batch["rdan_evaluator_failed"] = torch.tensor(evaluator_failed_rows, dtype=torch.bool, device=device)
        output.non_tensor_batch["rdan_prompt_key"] = np.asarray(data.non_tensor_batch["rdan_prompt_key"], dtype=object)
        output.non_tensor_batch["rdan_reward_lane"] = np.asarray(
            ["hybrid_rtt_compatible"] * len(rubric_rows), dtype=object
        )
        evidence_array = np.empty(len(evidence_rows), dtype=object)
        evidence_array[:] = evidence_rows
        output.non_tensor_batch["rdan_rubric_evidence"] = evidence_array
        return output

    def _evaluate_hybrid_row(
        self,
        prompt: str,
        response: str,
        rubrics: list[dict[str, Any]],
        source: str,
        truth: dict[str, Any],
    ) -> dict[str, Any]:
        hard_mask = _hard_mask(source, truth, len(rubrics))
        if source != RUBRICHUB_SOURCE:
            _validate_hir_metadata(source, truth, len(rubrics))
        routes = (
            _validate_rubrichub_row(prompt, rubrics, truth, hard_mask, self._rubrichub_contract)
            if source == RUBRICHUB_SOURCE
            else None
        )
        scores = [-100.0] * 20
        evaluated = [False] * 20
        infos: list[dict[str, Any] | None] = [None] * len(rubrics)
        judge_rubrics: list[dict[str, Any]] = []
        judge_roles: dict[int, tuple[str, str | None]] = {}
        evaluator_failed = False
        for index, (rubric, hard) in enumerate(zip(rubrics, hard_mask, strict=True)):
            if not hard:
                judge_rubrics.append(_soft_rubric(rubric, index))
                judge_roles[index] = ("soft_quality", None)
                continue
            if routes is not None:
                try:
                    score, info = _evaluate_rubrichub_hard(response, rubric, routes[index])
                    info = {
                        **info,
                        "reward_lane": "hybrid_rtt_compatible",
                        "evaluator_failed": False,
                    }
                except Exception as error:
                    score, info = 0.0, _failed_local_info(rubric, "rubrichub_rule", error)
                    evaluator_failed = True
                scores[index] = float(score)
                evaluated[index] = info["evaluator_failed"] is not True
                infos[index] = info
                continue
            score, info, fallback_reason = self._evaluate_hir_hard(response, rubric, source, truth, index, prompt)
            if score is None:
                if info is not None:
                    scores[index] = 0.0
                    evaluated[index] = False
                    infos[index] = info
                    evaluator_failed = True
                else:
                    judge_rubrics.append(_soft_rubric(rubric, index))
                    judge_roles[index] = ("hard_fallback", fallback_reason)
            else:
                scores[index] = float(score)
                evaluated[index] = True
                infos[index] = info

        judge_failed = False
        if judge_rubrics:
            try:
                judgments, provenance, valid = self._judge_batch(prompt, response, judge_rubrics)
            except Exception as error:
                judgments = {}
                provenance = {"error": type(error).__name__}
                valid = False
            expected_ids = [rubric["id"] for rubric in judge_rubrics]
            if valid and not _valid_judgments(judgments, expected_ids):
                provenance = {**provenance, "error": "malformed_judgments"}
                valid = False
            judge_failed = not valid
            for index, (role, fallback_reason) in judge_roles.items():
                rubric_id = index + 1
                judgment = judgments.get(rubric_id, {"score": 0.0, "reason": "missing_judgment"})
                score = float(judgment["score"]) if valid else 0.0
                scores[index] = score
                evaluated[index] = valid
                infos[index] = {
                    "score": score,
                    "method": "llm_judge",
                    "rubric": rubrics[index],
                    "rubric_id": rubric_id,
                    "llm_response": "structured" if valid else "",
                    "reason": str(judgment["reason"]),
                    "quality_valid": valid,
                    "eval_valid": valid,
                    "judge_role": role,
                    "fallback_reason": fallback_reason,
                    "reward_lane": "hybrid_rtt_compatible",
                    "evaluator_failed": not valid,
                    **provenance,
                }
        if any(info is None for info in infos):
            raise RuntimeError("hybrid rubric evaluator evidence is incomplete")
        complete_infos = [info for info in infos if info is not None]
        self._rdan_infos.extend(complete_infos)
        evidence = [
            _rubric_evidence(info, rubric, index, scores[index])
            for index, (info, rubric) in enumerate(zip(complete_infos, rubrics, strict=True))
        ]
        return {
            "scores": scores,
            "hard_mask": hard_mask + [False] * (20 - len(hard_mask)),
            "eval_mask": evaluated,
            "hard_fallback": any(role == "hard_fallback" for role, _ in judge_roles.values()),
            "judge_failed": judge_failed,
            "evaluator_failed": evaluator_failed,
            "evidence": evidence,
        }

    def _evaluate_hir_hard(
        self,
        response: str,
        rubric: dict[str, Any],
        source: str,
        truth: dict[str, Any],
        index: int,
        prompt: str,
    ) -> tuple[float | None, dict[str, Any] | None, str | None]:
        try:
            if source in {"type1", "type2"}:
                instruction_id = truth["instruction_id_list"][index]
                kwargs = truth["kwargs"][index]
                if source == "type1":
                    route_info = INSTRUCTION_ID_TO_IFEVAL.get(instruction_id)
                    if route_info is None:
                        return None, None, f"unsupported_route:{source}"
                    function_name, remap = route_info
                    function = IF_FUNCTIONS_MAP.get(function_name)
                    if function is None:
                        raise RuntimeError(f"missing deterministic function: {function_name}")
                    parameters = {remap.get(key, key): value for key, value in kwargs.items()}
                    score = 1.0 if call_ifeval_function(function, response, parameters) else -1.0
                else:
                    checker = TYPE2_CHECKERS.get(instruction_id)
                    if checker is None:
                        return None, None, f"unsupported_route:{source}"
                    score = 1.0 if checker(response, kwargs) else -1.0
                route = f"code_{source}"
                detail = {"instruction_id": instruction_id}
            elif source == "type3":
                constraint = truth["constraints"][index]
                checker_key = f"{constraint[0]}_{constraint[1]}"
                checker = CONSTRAINT_CHECKER_MAP.get(checker_key)
                if checker is None:
                    return None, None, f"unsupported_route:{source}"
                score = 1.0 if checker.check(constraint[2], response) else -1.0
                route = "code_type3"
                detail = {"constraint_type": checker_key}
            elif source == "type4":
                code = truth["functions"][index]
                result = evaluate_rule(code, prompt, response, allowed_hashes=self._type4_hashes, timeout_seconds=1.0)
                if not result.valid:
                    error = RuntimeError(f"rule_sandbox:{result.error or 'invalid'}")
                    return None, _failed_local_info(rubric, "code_type4_rule", error), None
                score = 1.0 if result.value else -1.0
                route = "code_type4_rule"
                detail = {"checker": truth["checker"][index]}
            else:
                return None, None, f"unsupported_source:{source}"
        except Exception as error:
            return None, _failed_local_info(rubric, f"code_{source}", error), None
        info = {
            "score": float(score),
            "method": route,
            "rubric": rubric,
            "reason": "passed" if score == 1 else "failed",
            "reward_lane": "hybrid_rtt_compatible",
            "evaluator_failed": False,
            **detail,
        }
        return float(score), info, None

    def _judge_batch(
        self, instruction: str, response: str, rubrics: list[dict[str, Any]]
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any], bool]:
        result = self._judge.judge(instruction, response, rubrics, self._judge_seed, self._judge_effort)
        return result.judgments, result.evidence, result.valid


def _failed_local_info(rubric: Any, method: str, error: Exception) -> dict[str, Any]:
    return {
        "score": 0.0,
        "method": method,
        "rubric": rubric,
        "reason": f"deterministic_error:{type(error).__name__}",
        "reward_lane": "hybrid_rtt_compatible",
        "evaluator_failed": True,
    }


def _valid_judgments(judgments: Any, expected_ids: list[int]) -> bool:
    if not isinstance(judgments, dict) or list(judgments) != expected_ids:
        return False
    return all(
        isinstance(row, dict)
        and set(row) == {"score", "reason"}
        and row["score"] in SIGNED_PROCESS_SCORES
        and isinstance(row["reason"], str)
        and bool(row["reason"])
        for row in judgments.values()
    )


def _validate_hir_metadata(source: str, truth: Mapping[str, Any], count: int) -> None:
    if source in {"type1", "type2"}:
        instruction_ids = truth.get("instruction_id_list")
        kwargs = truth.get("kwargs")
        if (
            not isinstance(instruction_ids, list)
            or len(instruction_ids) != count
            or any(not isinstance(value, str) or not value for value in instruction_ids)
            or not isinstance(kwargs, list)
            or len(kwargs) != count
            or any(not isinstance(value, dict) for value in kwargs)
        ):
            raise ValueError(f"{source} evaluator metadata is malformed")
        return
    if source == "type3":
        constraints = truth.get("constraints")
        if (
            not isinstance(constraints, list)
            or len(constraints) != count
            or any(not isinstance(value, (list, tuple)) or len(value) < 3 for value in constraints)
        ):
            raise ValueError("type3 evaluator metadata is malformed")
        return
    if source == "type4":
        checkers = truth.get("checker")
        functions = truth.get("functions")
        if (
            not isinstance(checkers, list)
            or len(checkers) != count
            or any(not isinstance(value, str) or not value for value in checkers)
            or not isinstance(functions, list)
            or len(functions) != count
            or any(not isinstance(value, str) for value in functions)
            or any(checker.startswith("[rule]") and not function for checker, function in zip(checkers, functions))
        ):
            raise ValueError("type4 evaluator metadata is malformed")
        return
    raise ValueError(f"unsupported reward source: {source}")


def _soft_rubric(rubric: Any, index: int) -> dict[str, Any]:
    if not isinstance(rubric, dict):
        raise ValueError("soft rubric must be an object")
    rubric_id = rubric.get("id")
    description = rubric.get("description")
    if isinstance(rubric_id, bool) or not isinstance(rubric_id, int) or rubric_id != index + 1:
        raise ValueError("soft rubric id must match its one-based position")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("soft rubric description must be a non-empty string")
    return {"id": rubric_id, "text": description}


def _rubric_evidence(info: Mapping[str, Any], rubric: Any, index: int, score: float) -> dict[str, Any]:
    calibrated = _soft_rubric(rubric, index)
    method = str(info.get("method", ""))
    judge_fields = (
        "generation_id",
        "selected_endpoint",
        "provider",
        "model",
        "finish_reason",
        "service_tier",
        "schema_id",
        "rubric_ids",
        "tokens",
        "reasoning_effort",
        "latency_ms",
        "cost",
        "generation_metadata_polls",
        "error",
        "request_sha256",
    )
    evidence = {
        "rubric_index": index,
        "rubric_id": calibrated["id"],
        "rubric_description_sha256": hashlib.sha256(calibrated["text"].encode()).hexdigest(),
        "score": float(score),
        "evaluator_route": method,
        "reason": str(info.get("reason", "")),
        "generation_id": info.get("generation_id") if method == "llm_judge" else None,
        "request_sha256": info.get("request_sha256") if method == "llm_judge" else None,
        "judge_provenance": {name: info.get(name) for name in judge_fields} if method == "llm_judge" else None,
        "judge_failed": method == "llm_judge" and info.get("quality_valid") is not True,
        "evaluator_failed": info.get("evaluator_failed") is True,
        "reward_lane": info.get("reward_lane", "authoritative_strict"),
        "judge_role": info.get("judge_role"),
        "fallback_reason": info.get("fallback_reason"),
    }
    safe = redact_secrets(evidence)
    try:
        return json.loads(json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("rubric evidence is not strict JSON") from error


def _hard_mask(source: str, truth: dict[str, Any], count: int) -> list[bool]:
    if source in {"type1", "type2", "type3"}:
        return [True] * count
    if source == "type4":
        checkers = truth.get("checker")
        if not isinstance(checkers, list) or len(checkers) != count:
            raise ValueError("hard rubric mask cannot be derived from ground truth")
        return [isinstance(checker, str) and checker.startswith("[rule]") for checker in checkers]
    if source == RUBRICHUB_SOURCE:
        if truth.get("rl_eligible") is not True:
            raise ValueError("RubricHub row is not certified for RL")
        if truth.get("quarantine_reasons") != []:
            raise ValueError("RubricHub row has quarantine reasons")
        mask = truth.get("hard_mask")
        if not isinstance(mask, list) or len(mask) != count or any(not isinstance(value, bool) for value in mask):
            raise ValueError("RubricHub hard mask is malformed")
        return list(mask)
    raise ValueError(f"unsupported reward source: {source}")


def _validate_rubrichub_row(
    prompt: Any,
    rubrics: Any,
    truth: Mapping[str, Any],
    hard_mask: list[bool],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(prompt, str):
        raise ValueError("RubricHub prompt must be a string")
    if not isinstance(rubrics, list) or not rubrics or len(rubrics) > 20:
        raise ValueError("RubricHub rubrics must contain between 1 and 20 entries")
    routes = truth.get("rubric_routes")
    if not isinstance(routes, list) or len(routes) != len(rubrics):
        raise ValueError("RubricHub rubric routes are malformed")

    validated: list[dict[str, Any]] = []
    for index, (rubric, route, hard) in enumerate(zip(rubrics, routes, hard_mask, strict=True)):
        if not isinstance(rubric, dict) or not isinstance(route, dict):
            raise ValueError("RubricHub rubric route is malformed")
        rubric_parameters = _parameter_object(rubric.get("parameters"), f"RubricHub rubric {index}")
        route_parameters = _parameter_object(route.get("parameters"), f"RubricHub rubric route {index}")
        expected = {
            "rubric_index": index,
            "verifier": "rule" if hard else "llm",
            "function": rubric.get("function"),
            "parameters": rubric_parameters,
            "criterion": rubric.get("description"),
            "points": rubric.get("weight"),
        }
        decoded_route = {**route, "parameters": route_parameters}
        if not _same_json(decoded_route, expected):
            raise ValueError(f"RubricHub rubric route {index} differs from the frozen rubric")
        if rubric.get("id") != index + 1 or rubric.get("verifier") != expected["verifier"]:
            raise ValueError(f"RubricHub rubric {index} metadata is malformed")
        if rubric.get("category") != expected["verifier"]:
            raise ValueError(f"RubricHub rubric {index} category is malformed")
        validated.append(decoded_route)

    provenance = truth.get("source_provenance")
    source = contract["source"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "dataset",
        "revision",
        "file",
        "source_index",
        "source_row_sha256",
        "prompt_sha256",
    }:
        raise ValueError("RubricHub source provenance is malformed")
    if any(provenance.get(key) != source[key] for key in ("dataset", "revision", "file")):
        raise ValueError("RubricHub source provenance differs from the frozen source")
    index = provenance.get("source_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("RubricHub source index is malformed")
    row = contract["rows"].get(index)
    if row is None or provenance.get("source_row_sha256") != row["source_row_sha256"]:
        raise ValueError("RubricHub source row is not in the certified subset")
    tokenizer = contract["tokenizer_certificate"]
    if index not in tokenizer["accepted_source_indices"]:
        raise ValueError("RubricHub source row is not tokenizer-certified")
    token_row = tokenizer["rows"][index]
    messages = [{"role": "user", "content": prompt}]
    if token_row["messages_sha256"] != _json_sha256(messages):
        raise ValueError("RubricHub prompt differs from the tokenizer certificate")
    if provenance.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError("RubricHub prompt differs from its frozen provenance")

    hard_routes = [route for route, hard in zip(validated, hard_mask, strict=True) if hard]
    functions = [route["function"] for route in hard_routes]
    parameters = [route["parameters"] for route in hard_routes]
    if functions != row["functions"] or _json_sha256(parameters) != row["route_parameters_sha256"]:
        raise ValueError("RubricHub hard routes differ from the certified source row")
    if any(function not in contract["functions"] for function in functions):
        raise ValueError("RubricHub row contains an uncertified rule route")
    return validated


def _parameter_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} parameters must use the canonical JSON string encoding")
    try:
        parameters = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} parameters are invalid JSON") from error
    if not isinstance(parameters, dict) or value != json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ):
        raise ValueError(f"{label} parameters are not a canonical JSON object")
    return parameters


def _evaluate_rubrichub_hard(
    response: str, rubric: dict[str, Any], route: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    result = evaluate_rubrichub_rule(route.get("function"), response, route.get("parameters"))
    if not result.valid:
        raise RuntimeError(f"RubricHub rule failed closed: {result.error}")
    score = 1.0 if result.passed else -1.0
    return score, {
        "score": score,
        "method": "rubrichub_rule",
        "function": route["function"],
        "parameters_sha256": _json_sha256(route["parameters"]),
        "rubric": rubric,
        "reason": "passed" if result.passed else "failed",
    }


def _load_judge_contract() -> dict[str, Any]:
    config = json.loads(JUDGE_CONFIG.read_text(encoding="utf-8"))
    prompt = JUDGE_PROMPT.read_text(encoding="utf-8")
    if hashlib.sha256(prompt.encode()).hexdigest() != config["prompt"]["sha256"]:
        raise RuntimeError("judge prompt hash differs from frozen config")
    return {"config": config, "prompt": prompt}


def _load_type4_hashes() -> frozenset[str]:
    payload = verify_type4_certificate(TYPE4_EVIDENCE, TYPE4_CERTIFICATE)
    hashes = payload.get("function_hashes")
    if payload.get("status") not in {"certified", "certified_subset"} or not isinstance(hashes, list) or not hashes:
        raise RuntimeError("type4 evaluator certificate is absent or blocked")
    if hashes != sorted(set(hashes)) or any(len(value) != 64 for value in hashes):
        raise RuntimeError("type4 evaluator certificate hash inventory is invalid")
    return frozenset(hashes)


def _load_rubrichub_contract() -> dict[str, Any]:
    config = _read_json_object(RUBRICHUB_CONFIG, "RubricHub data config")
    source = config.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("RubricHub data source config is malformed")
    if (ROOT / str(config.get("checker_certificate"))).resolve() != RUBRICHUB_RULE_CERTIFICATE.resolve():
        raise RuntimeError("RubricHub checker certificate path is not frozen")
    if (ROOT / str(config.get("tokenizer_certificate"))).resolve() != RUBRICHUB_TOKENIZER_CERTIFICATE.resolve():
        raise RuntimeError("RubricHub tokenizer certificate path is not frozen")
    certificate = verify_rule_certificate(RUBRICHUB_RULE_CERTIFICATE, source)
    if certificate.get("response_policy") != {
        "input": "verbatim_decoded_response",
        "think_tag_filtering": False,
        "rollout_template": "qwen3_nothinking",
        "enable_thinking": False,
    }:
        raise RuntimeError("RubricHub rule response policy is invalid")
    routes = certificate.get("routes")
    if not isinstance(routes, list) or any(route.get("status") != "certified" for route in routes):
        raise RuntimeError("RubricHub rule routes are not certified")
    functions = frozenset(route["function"] for route in routes)
    rows = _load_rubrichub_evidence(certificate, functions)
    source_contract = {key: source[key] for key in ("dataset", "revision", "file", "sha256", "records")}
    tokenizer = _load_rubrichub_tokenizer(config, source_contract, rows)
    return {
        "source": source_contract,
        "functions": functions,
        "rows": rows,
        "tokenizer_certificate": tokenizer,
    }


def _load_rubrichub_evidence(certificate: Mapping[str, Any], functions: frozenset[str]) -> dict[int, dict[str, Any]]:
    reference = certificate.get("evidence")
    if not isinstance(reference, dict):
        raise RuntimeError("RubricHub rule evidence reference is malformed")
    relative = reference.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("RubricHub rule evidence path is malformed")
    path = (ROOT / relative).resolve()
    evidence_root = (ROOT / "data/rubrichub-source/certificates").resolve()
    if not path.is_relative_to(evidence_root):
        raise RuntimeError("RubricHub rule evidence path leaves the frozen artifact directory")
    if (
        not path.is_file()
        or isinstance(reference.get("bytes"), bool)
        or not isinstance(reference.get("bytes"), int)
        or path.stat().st_size != reference["bytes"]
        or _sha256_file(path) != reference.get("sha256")
    ):
        raise RuntimeError("RubricHub rule evidence differs from its certificate")

    rows: dict[int, dict[str, Any]] = {}
    records = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                records += 1
                if item.get("kind") != "candidate":
                    continue
                if set(item) != {
                    "kind",
                    "source_index",
                    "source_row_sha256",
                    "functions",
                    "route_parameters_sha256",
                }:
                    raise RuntimeError("RubricHub candidate evidence is malformed")
                index = item.get("source_index")
                item_functions = item.get("functions")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index in rows
                    or not _is_sha256(item.get("source_row_sha256"))
                    or not isinstance(item_functions, list)
                    or not item_functions
                    or any(function not in functions for function in item_functions)
                    or not _is_sha256(item.get("route_parameters_sha256"))
                ):
                    raise RuntimeError("RubricHub candidate evidence is invalid")
                rows[index] = item
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("RubricHub rule evidence is unreadable") from error

    selection = certificate.get("candidate_selection")
    counts = certificate.get("counts")
    indices = sorted(rows)
    hashes = [rows[index]["source_row_sha256"] for index in indices]
    if (
        not isinstance(selection, dict)
        or not isinstance(counts, dict)
        or records != reference.get("records")
        or records != counts.get("total_records")
        or len(rows) != counts.get("candidate_records")
        or len(rows) != selection.get("rows")
        or indices != selection.get("source_indices")
        or _json_sha256(indices) != selection.get("source_indices_sha256")
        or _json_sha256(hashes) != selection.get("source_row_hashes_sha256")
    ):
        raise RuntimeError("RubricHub rule evidence inventory is inconsistent")
    return rows


def _load_rubrichub_tokenizer(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    rule_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    certificate = _read_json_object(RUBRICHUB_TOKENIZER_CERTIFICATE, "RubricHub tokenizer certificate")
    runtime = certificate.get("runtime")
    if (
        certificate.get("schema_version") != 1
        or certificate.get("status") != "frozen"
        or certificate.get("source") != source
        or certificate.get("tokenizer") != config.get("tokenizer_gate")
        or not isinstance(runtime, dict)
        or runtime.get("transformers") != config.get("tokenizer_gate", {}).get("transformers_version")
    ):
        raise RuntimeError("RubricHub tokenizer certificate contract is invalid")
    selection = certificate.get("selection")
    indices = sorted(rule_rows)
    source_hashes = [rule_rows[index]["source_row_sha256"] for index in indices]
    checker_ref = {
        "bytes": RUBRICHUB_RULE_CERTIFICATE.stat().st_size,
        "sha256": _sha256_file(RUBRICHUB_RULE_CERTIFICATE),
    }
    if (
        not isinstance(selection, dict)
        or selection.get("checker_certificate") != checker_ref
        or selection.get("candidate_rows") != len(indices)
        or selection.get("candidate_indices_sha256") != _json_sha256(indices)
        or selection.get("candidate_source_hashes_sha256") != _json_sha256(source_hashes)
    ):
        raise RuntimeError("RubricHub tokenizer certificate selection is invalid")

    reference = certificate.get("evidence")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise RuntimeError("RubricHub tokenizer evidence reference is malformed")
    path = (ROOT / reference["path"]).resolve()
    evidence_root = (ROOT / "data/rubrichub-source/certificates").resolve()
    if not path.is_relative_to(evidence_root):
        raise RuntimeError("RubricHub tokenizer evidence path leaves the frozen artifact directory")
    if (
        not path.is_file()
        or isinstance(reference.get("bytes"), bool)
        or not isinstance(reference.get("bytes"), int)
        or path.stat().st_size != reference["bytes"]
        or _sha256_file(path) != reference.get("sha256")
    ):
        raise RuntimeError("RubricHub tokenizer evidence differs from its certificate")

    rows: dict[int, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if set(item) != {
                    "accepted",
                    "input_ids_sha256",
                    "input_tokens",
                    "messages_sha256",
                    "reason",
                    "source_index",
                    "source_row_sha256",
                }:
                    raise RuntimeError("RubricHub tokenizer evidence row is malformed")
                index = item.get("source_index")
                tokens = item.get("input_tokens")
                accepted = isinstance(tokens, int) and not isinstance(tokens, bool) and 5 < tokens <= 2_048
                reason = (
                    "accepted"
                    if accepted
                    else "input_tokens_too_short"
                    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens <= 5
                    else "input_tokens_exceed_prompt_length"
                )
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or isinstance(tokens, bool)
                    or not isinstance(tokens, int)
                    or tokens < 0
                    or index not in rule_rows
                    or index in rows
                    or item.get("source_row_sha256") != rule_rows[index]["source_row_sha256"]
                    or item.get("accepted") is not accepted
                    or item.get("reason") != reason
                    or not _is_sha256(item.get("messages_sha256"))
                    or not _is_sha256(item.get("input_ids_sha256"))
                ):
                    raise RuntimeError("RubricHub tokenizer evidence row is invalid")
                rows[index] = item
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("RubricHub tokenizer evidence is unreadable") from error

    accepted = [index for index in indices if rows.get(index, {}).get("accepted") is True]
    rejected = [rows[index] for index in indices if rows.get(index, {}).get("accepted") is False]
    results = certificate.get("results")
    if (
        sorted(rows) != indices
        or len(rows) != reference.get("records")
        or not isinstance(results, dict)
        or results.get("candidates") != len(indices)
        or results.get("accepted") != len(accepted)
        or results.get("rejected") != len(rejected)
        or results.get("frozen_expected_candidates") != len(indices)
        or results.get("frozen_expected_accepted") != len(accepted)
        or results.get("accepted_source_indices") != accepted
        or results.get("accepted_source_indices_sha256") != _json_sha256(accepted)
        or not _same_json(results.get("rejected_rows"), rejected)
        or results.get("largest_accepted_input_tokens") != max(rows[index]["input_tokens"] for index in accepted)
    ):
        raise RuntimeError("RubricHub tokenizer certificate inventory is inconsistent")
    return {
        "sha256": _sha256_file(RUBRICHUB_TOKENIZER_CERTIFICATE),
        "accepted_source_indices": frozenset(accepted),
        "rows": rows,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be an object")
    return payload


def _same_json(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_calibrated_effort() -> str:
    payload = json.loads(JUDGE_CERTIFICATE.read_text(encoding="utf-8"))
    effort = payload.get("selected_reasoning_effort")
    if payload.get("status") != "calibrated" or effort not in {"none", "low", "medium"}:
        raise RuntimeError("judge calibration certificate is absent or invalid")
    return effort


def _load_program_seed() -> int:
    """Return the fixed tuning seed the judge sampler is pinned to."""

    return TUNING_SEED
