"""Strict OpenRouter judge boundary with auditable fail-closed provenance."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]
PROCESS_SCORES = (0, 0.5, 1)
SIGNED_PROCESS_SCORES = (-1.0, 0.0, 1.0)
# Ordered by preference: the first non-inferior candidate becomes the selected effort.
EFFORT_CANDIDATES = ("high", "medium", "low")
# OpenRouter forwards this model on OpenAI's Responses API, which renames the request
# parameters. Seed is not forwarded upstream at all, so the canary cannot assert it.
UPSTREAM_PARAMETERS = ("max_output_tokens", "reasoning.effort", "text.format")


@dataclass(frozen=True)
class JudgeResult:
    judgments: dict[int, JsonObject]
    evidence: JsonObject
    valid: bool
    raw: JsonObject | None = None


def build_request(
    contract: Mapping[str, Any],
    prompt: str,
    instruction: str,
    response: str,
    rubrics: Sequence[Mapping[str, Any]],
    seed: int,
    reasoning_effort: str,
) -> JsonObject:
    values = {
        "instruction": instruction,
        "response": response,
        "rubrics_json": json.dumps(rubrics, ensure_ascii=False, separators=(",", ":")),
    }
    content = re.sub(r"{{(instruction|response|rubrics_json)}}", lambda match: values[match.group(1)], prompt)
    request = contract["request"]
    return {
        "model": contract["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": request["max_tokens"],
        "reasoning_effort": reasoning_effort,
        "response_format": contract["response_format"],
        "seed": seed,
    }


class OpenRouterJudge:
    """Reuse one no-retry OpenAI client and validate both metadata sources."""

    def __init__(
        self,
        contract: Mapping[str, Any],
        prompt: str,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"X-OpenRouter-Metadata": "enabled"},
                max_retries=0,
            )
        self.contract = contract
        self.prompt = prompt
        self.client = client
        self.sleep = sleep
        self.poll_attempts, self.poll_interval_seconds = _generation_poll_contract(contract)

    def judge(
        self,
        instruction: str,
        response: str,
        rubrics: Sequence[Mapping[str, Any]],
        seed: int,
        reasoning_effort: str,
    ) -> JudgeResult:
        ids = [rubric.get("id") for rubric in rubrics]
        request = build_request(self.contract, self.prompt, instruction, response, rubrics, seed, reasoning_effort)
        request_hash = _sha256(request)
        started = time.monotonic()
        try:
            completion = self.client.chat.completions.create(
                **request,
                extra_body={"provider": self.contract["routing"]},
            )
            latency_ms = round((time.monotonic() - started) * 1000, 3)
            return self._validate(completion, ids, reasoning_effort, request_hash, latency_ms)
        except Exception as exc:
            return _invalid(ids, reasoning_effort, request_hash, type(exc).__name__)

    def debug_canary(
        self,
        instruction: str,
        response: str,
        rubrics: Sequence[Mapping[str, Any]],
        seed: int,
    ) -> JudgeResult:
        request = build_request(self.contract, self.prompt, instruction, response, rubrics, seed, "none")
        request_hash = _sha256(request)
        try:
            stream = self.client.chat.completions.create(
                **request,
                stream=True,
                extra_body={
                    "provider": self.contract["routing"],
                    "debug": {"echo_upstream_body": True},
                },
            )
            chunks = list(stream)
            first = chunks[0] if chunks else None
            body = _field(_field(first, "debug"), "echo_upstream_body")
            generation_id = _field(first, "id")
            generation, metadata_polls = _generation_data(
                self.client,
                generation_id,
                self.sleep,
                self.poll_attempts,
                self.poll_interval_seconds,
            )
            parameter_names = sorted(name for name in UPSTREAM_PARAMETERS if _dotted(body, name) is not None)
            expected_names = sorted(UPSTREAM_PARAMETERS)
            schema_format = _dotted(body, "text.format")
            direct_models = {
                self.contract["model"],
                self.contract["expected_canonical_slug"],
                self.contract["model"].removeprefix("openai/"),
                self.contract["expected_canonical_slug"].removeprefix("openai/"),
            }
            if (
                _field(first, "choices") != []
                or not isinstance(body, Mapping)
                or body.get("model") not in direct_models
                or body.get("stream") is not True
                or _dotted(body, "max_output_tokens") != self.contract["request"]["max_tokens"]
                or _dotted(body, "reasoning.effort") != "none"
                or not isinstance(schema_format, Mapping)
                or schema_format.get("name") != self.contract["response_format"]["json_schema"]["name"]
                or schema_format.get("strict") is not True
                or parameter_names != expected_names
                or _field(generation, "id") != generation_id
                or _field(generation, "provider_name") != "OpenAI"
            ):
                raise ValueError("OpenRouter debug canary failed")
            evidence = {
                "generation_id": generation_id,
                "provider": "OpenAI",
                "model": body["model"],
                "parameter_names": parameter_names,
                "upstream_body_sha256": _sha256(body),
                "generation_metadata_polls": metadata_polls,
                "request_sha256": request_hash,
                "error": None,
            }
            return JudgeResult({}, evidence, True, {"debug_upstream_body": dict(body), "generation": dict(generation)})
        except Exception as exc:
            evidence = {
                "generation_id": None,
                "provider": None,
                "model": None,
                "parameter_names": [],
                "upstream_body_sha256": None,
                "generation_metadata_polls": getattr(exc, "metadata_polls", None),
                "request_sha256": request_hash,
                "error": type(exc).__name__,
            }
            return JudgeResult({}, evidence, False)

    def _validate(
        self,
        completion: Any,
        ids: list[Any],
        effort: str,
        request_hash: str,
        latency_ms: float,
    ) -> JudgeResult:
        generation_id = _field(completion, "id")
        model = _field(completion, "model")
        choices = _field(completion, "choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        finish_reason = _field(choice, "finish_reason")
        message = _field(choice, "message")
        content = _field(message, "content")
        metadata = _field(completion, "openrouter_metadata")
        selected = _selected_endpoint(metadata)
        usage = _field(completion, "usage")
        try:
            generation, metadata_polls = _generation_data(
                self.client,
                generation_id,
                self.sleep,
                self.poll_attempts,
                self.poll_interval_seconds,
            )
            generation_error = None
        except Exception as exc:
            generation = {}
            metadata_polls = getattr(exc, "metadata_polls", None)
            generation_error = type(exc).__name__
        evidence = {
            "generation_id": generation_id,
            "selected_endpoint": selected,
            "provider": _field(generation, "provider_name"),
            "model": model,
            "finish_reason": finish_reason,
            "service_tier": _field(completion, "service_tier"),
            "schema_id": self.contract["response_format"]["json_schema"]["name"],
            "rubric_ids": ids,
            "tokens": {
                "prompt": _field(usage, "prompt_tokens"),
                "completion": _field(usage, "completion_tokens"),
                "total": _field(usage, "total_tokens"),
                "reasoning": _field(generation, "native_tokens_reasoning"),
            },
            "reasoning_effort": effort,
            "latency_ms": _field(generation, "latency") or latency_ms,
            "cost": _field(generation, "total_cost"),
            "generation_metadata_polls": metadata_polls,
            "error": generation_error,
            "request_sha256": request_hash,
        }
        if generation_error:
            return JudgeResult(_zero(ids, generation_error), evidence, False, _raw(completion, generation))
        try:
            _validate_provenance(self.contract, evidence, metadata, generation)
            rows = json.loads(content).get("rubrics") if isinstance(content, str) else None
            judgments = _validate_rows(rows, ids)
        except Exception as exc:
            evidence["error"] = type(exc).__name__
            return JudgeResult(_zero(ids, evidence["error"]), evidence, False, _raw(completion, generation))
        return JudgeResult(judgments, evidence, True, _raw(completion, generation))


def preflight_snapshots(contract: Mapping[str, Any], get_json: Callable[[str], Mapping[str, Any]]) -> JsonObject:
    """Verify the free catalog and endpoint inventories before any generation."""

    catalog_data = get_json(str(contract["catalog_url"])).get("data")
    models = catalog_data if isinstance(catalog_data, list) else [catalog_data]
    model = next((row for row in models if _field(row, "id") == contract["model"]), None)
    supported = set(_field(model, "supported_parameters") or [])
    if _field(model, "canonical_slug") != contract["expected_canonical_slug"] or not set(
        contract["required_parameters"]
    ).issubset(supported):
        raise ValueError("OpenRouter catalog gate failed")
    endpoint_data = get_json(str(contract["endpoints_url"])).get("data")
    endpoints = _field(endpoint_data, "endpoints")
    direct = [row for row in endpoints or [] if str(_field(row, "provider_name")).lower() == "openai"]
    if _field(endpoint_data, "id") != contract["model"] or not direct:
        raise ValueError("OpenRouter OpenAI endpoint gate failed")
    required = set(contract["required_parameters"])
    eligible = [row for row in direct if _field(row, "tag") == "openai"]
    if len(eligible) != 1:
        raise ValueError("OpenRouter base OpenAI endpoint count failed")
    base = eligible[0]
    if (
        _field(base, "model_id") != contract["model"]
        or _field(base, "status") != 0
        or not required.issubset(set(_field(base, "supported_parameters") or []))
    ):
        raise ValueError("OpenRouter OpenAI endpoint capabilities failed")
    catalog = {
        "id": _field(model, "id"),
        "canonical_slug": _field(model, "canonical_slug"),
        "supported_parameters": sorted(supported),
    }
    endpoint = {
        "id": _field(endpoint_data, "id"),
        "provider": "openai",
        "model": str(_field(base, "model_id")),
        "tag": "openai",
        "supported_parameters": sorted(set(_field(base, "supported_parameters"))),
        "status": 0,
    }
    return {
        "catalog": catalog,
        "catalog_sha256": _sha256(catalog),
        "endpoints": endpoint,
        "endpoints_sha256": _sha256(endpoint),
    }


def calibration_plan(case_ids: Mapping[str, Sequence[str]], selected_effort: str) -> list[tuple[str, str, int]]:
    if {name: len(values) for name, values in case_ids.items()} != {"debug": 1, "labeled": 49, "heldout": 26}:
        raise ValueError("calibration requires exactly 1 debug, 49 labeled, and 26 heldout cases")
    plan = [(case_ids["debug"][0], "none", 1)]
    plan.extend((case_id, effort, 1) for effort in EFFORT_CANDIDATES for case_id in case_ids["labeled"])
    plan.extend((case_id, selected_effort, repeat) for case_id in case_ids["heldout"] for repeat in (1, 2))
    if len(plan) != 200:
        raise AssertionError("calibration plan must contain exactly 200 calls")
    return plan


def select_reasoning_effort(
    indicators: Mapping[str, Sequence[int]],
    *,
    samples: int = 10_000,
    seed: int = 240520,
    margin: float = -0.02,
    pinned: str | None = None,
) -> JsonObject:
    efforts = EFFORT_CANDIDATES
    if pinned is not None and pinned not in efforts:
        raise ValueError("pinned reasoning effort is not a calibration candidate")
    if set(indicators) != set(efforts) or samples <= 0:
        raise ValueError("paired calibration indicators or bootstrap sample count are invalid")
    rows = {effort: list(indicators[effort]) for effort in efforts}
    if any(len(values) != 49 or any(value not in {0, 1} for value in values) for values in rows.values()):
        raise ValueError("each effort requires 49 binary paired exact-match indicators")
    point_scores = {effort: sum(rows[effort]) / 49 for effort in efforts}
    reference = max(efforts, key=lambda effort: (point_scores[effort], -efforts.index(effort)))
    rng = random.Random(seed)
    deltas = {effort: [] for effort in efforts}
    for _ in range(samples):
        indices = [rng.randrange(49) for _ in range(49)]
        for effort in efforts:
            deltas[effort].append(sum(rows[effort][index] - rows[reference][index] for index in indices) / 49)
    intervals = {
        effort: {
            "lower": _percentile(values, 0.025),
            "upper": _percentile(values, 0.975),
        }
        for effort, values in deltas.items()
    }
    qualifying = [effort for effort in efforts if intervals[effort]["lower"] >= margin]
    return {
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "noninferiority_margin": margin,
        "point_scores": point_scores,
        "reference_effort": reference,
        "paired_difference_ci95": intervals,
        "qualifying_efforts": qualifying,
        "pinned_effort": pinned,
        "selected_effort": pinned or qualifying[0],
    }


def _generation_poll_contract(contract: Mapping[str, Any]) -> tuple[int, float]:
    poll = contract.get("generation_metadata_poll")
    if not isinstance(poll, Mapping) or set(poll) != {"attempts", "interval_seconds"}:
        raise ValueError("generation metadata poll keys are invalid")
    attempts = poll["attempts"]
    interval = poll["interval_seconds"]
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts <= 0:
        raise ValueError("generation metadata poll attempts are invalid")
    if not isinstance(interval, (int, float)) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("generation metadata poll interval is invalid")
    return attempts, float(interval)


def _generation_data(
    client: Any,
    generation_id: Any,
    sleep: Callable[[float], None],
    attempts: int,
    interval_seconds: float,
) -> tuple[Mapping[str, Any], int]:
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError("generation ID is absent")
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            sleep(interval_seconds)
        try:
            payload = client.get("/generation", cast_to=JsonObject, options={"params": {"id": generation_id}})
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404 and attempt < attempts:
                continue
            setattr(exc, "metadata_polls", attempt)
            raise
        data = _field(payload, "data")
        if not isinstance(data, Mapping):
            error = ValueError("generation metadata is absent")
            setattr(error, "metadata_polls", attempt)
            raise error
        return data, attempt
    raise AssertionError("metadata poll loop did not terminate")


def _selected_endpoint(metadata: Any) -> JsonObject | None:
    endpoints = _field(_field(metadata, "endpoints"), "available")
    selected = [row for row in endpoints or [] if _field(row, "selected") is True]
    if len(selected) != 1:
        return None
    return {"provider": _field(selected[0], "provider"), "model": _field(selected[0], "model")}


def _validate_provenance(
    contract: Mapping[str, Any], evidence: Mapping[str, Any], metadata: Any, generation: Any
) -> None:
    selected = evidence["selected_endpoint"]
    allowed = contract["preflight"]["response_model_must_be_one_of"]
    if (
        not isinstance(selected, Mapping)
        or str(selected.get("provider", "")).lower() != "openai"
        or selected.get("model") not in allowed
        or _field(metadata, "requested") != contract["model"]
        or _field(metadata, "attempt") != 1
        or evidence["provider"] != "OpenAI"
        or evidence["model"] not in allowed
        or _field(generation, "id") != evidence["generation_id"]
        or _field(generation, "model") != selected.get("model")
        or _field(generation, "finish_reason") != evidence["finish_reason"]
        or evidence["finish_reason"] != "stop"
        or evidence["service_tier"] not in {None, "default"}
    ):
        raise ValueError("judge provenance cross-check failed")


def _validate_rows(rows: Any, ids: list[Any]) -> dict[int, JsonObject]:
    if not isinstance(rows, list) or [row.get("id") for row in rows if isinstance(row, dict)] != ids:
        raise ValueError("judge rubric IDs differ from request")
    if len(ids) != len(set(ids)) or any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
        raise ValueError("judge rubric IDs are invalid")
    result: dict[int, JsonObject] = {}
    for row in rows:
        if set(row) != {"id", "score", "reason"} or row["score"] not in PROCESS_SCORES:
            raise ValueError("judge output schema is invalid")
        if not isinstance(row["reason"], str) or not row["reason"]:
            raise ValueError("judge reason is empty")
        result[row["id"]] = {"score": signed_process_score(row["score"]), "reason": row["reason"]}
    return result


def _dotted(value: Any, path: str) -> Any:
    """Read a dotted key path out of the echoed upstream request body."""

    for name in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def signed_process_score(score: float) -> float:
    """Map a PAPO process score in {0, 0.5, 1} onto the signed rubric scale."""

    return 2.0 * float(score) - 1.0


def _invalid(ids: list[Any], effort: str, request_hash: str, error: str) -> JudgeResult:
    evidence = {
        "generation_id": None,
        "selected_endpoint": None,
        "provider": None,
        "model": None,
        "finish_reason": None,
        "service_tier": None,
        "schema_id": "rubric_judgment",
        "rubric_ids": ids,
        "tokens": {"prompt": None, "completion": None, "total": None, "reasoning": None},
        "reasoning_effort": effort,
        "latency_ms": None,
        "cost": None,
        "generation_metadata_polls": None,
        "error": error,
        "request_sha256": request_hash,
    }
    return JudgeResult(_zero(ids, error), evidence, False)


def _zero(ids: Sequence[Any], error: str) -> dict[int, JsonObject]:
    return {
        value: {"score": 0, "reason": error} for value in ids if isinstance(value, int) and not isinstance(value, bool)
    }


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    result = getattr(value, name, None)
    if result is not None:
        return result
    extra = getattr(value, "model_extra", None)
    return extra.get(name) if isinstance(extra, Mapping) else None


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _raw(completion: Any, generation: Any) -> JsonObject:
    if hasattr(completion, "model_dump"):
        response = completion.model_dump(mode="json")
    elif isinstance(completion, Mapping):
        response = dict(completion)
    else:
        response = {"repr": repr(completion)}
    return {"response": response, "generation": dict(generation)}
