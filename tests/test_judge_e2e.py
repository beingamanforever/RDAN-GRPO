from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rdan_grpo.judge import (
    TRANSPORT_RETRIES,
    JudgeResult,
    OpenRouterJudge,
    calibration_plan,
    preflight_snapshots,
    select_reasoning_effort,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("calibrate_judge", ROOT / "scripts/calibrate_judge.py")
assert SPEC and SPEC.loader
calibrate_judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibrate_judge)


def _contract() -> tuple[dict[str, Any], str]:
    config = json.loads((ROOT / "configs/judges/openrouter_luna.json").read_text(encoding="utf-8"))
    prompt = (ROOT / "configs/judges/rubric_prompt.txt").read_text(encoding="utf-8")
    return config, prompt


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, generation: dict[str, Any] | None = None, error: Exception | None = None):
        self.completions = FakeCompletions(response, error)
        self.chat = SimpleNamespace(completions=self.completions)
        self.generation = generation
        self.get_calls: list[tuple[str, Any, dict[str, Any]]] = []

    def get(self, path: str, *, cast_to: Any, options: dict[str, Any]) -> dict[str, Any]:
        self.get_calls.append((path, cast_to, options))
        if cast_to is dict:
            raise ValueError("the pinned OpenAI SDK rejects an unparameterized dict")
        if isinstance(self.generation, list):
            value = self.generation.pop(0)
            if isinstance(value, Exception):
                raise value
            return {"data": value}
        return {"data": self.generation}


class HttpError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(str(status_code))
        self.status_code = status_code


def _response(content: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="gen-1",
        model="openai/gpt-5.6-luna",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content or '{"rubrics":[{"id":1,"score":1,"reason":"clear"}]}'),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=6, total_tokens=16),
        service_tier="default",
        openrouter_metadata={
            "requested": "openai/gpt-5.6-luna",
            "attempt": 1,
            "endpoints": {"available": [{"provider": "OpenAI", "model": "openai/gpt-5.6-luna", "selected": True}]},
        },
    )


def _generation(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "provider_name": "OpenAI",
        "model": "openai/gpt-5.6-luna",
        "finish_reason": "stop",
        "service_tier": "default",
        "native_tokens_reasoning": 2,
        "latency": 12,
        "total_cost": 0.001,
        **overrides,
    }


@pytest.mark.parametrize(("process", "signed"), [(0, -1.0), (0.5, 0.0), (1, 1.0)])
def test_strict_judge_maps_process_scores_onto_the_signed_scale(process: float, signed: float) -> None:
    contract, prompt = _contract()
    content = '{"rubrics":[{"id":1,"score":%s,"reason":"clear"}]}' % process
    judge = OpenRouterJudge(contract, prompt, "redacted", client=FakeClient(_response(content), _generation()))
    result = judge.judge("instruction", "response", [{"id": 1, "text": "quality"}], 240520, "low")

    assert result.valid and result.judgments[1]["score"] == signed


def test_strict_judge_rejects_a_process_score_outside_the_three_tiers() -> None:
    contract, prompt = _contract()
    content = '{"rubrics":[{"id":1,"score":-1,"reason":"clear"}]}'
    judge = OpenRouterJudge(contract, prompt, "redacted", client=FakeClient(_response(content), _generation()))
    result = judge.judge("instruction", "response", [{"id": 1, "text": "quality"}], 240520, "low")

    assert not result.valid and result.evidence["error"] == "ValueError"


def test_strict_judge_reuses_client_and_cross_checks_generation_metadata() -> None:
    contract, prompt = _contract()
    client = FakeClient(_response(), _generation())
    judge = OpenRouterJudge(contract, prompt, "redacted", client=client)
    first = judge.judge("instruction", "response", [{"id": 1, "text": "quality"}], 240520, "low")
    second = judge.judge("instruction", "response", [{"id": 1, "text": "quality"}], 240520, "low")

    assert first.valid and second.valid
    assert first.judgments[1]["score"] == 1
    assert first.evidence == {
        "generation_id": "gen-1",
        "selected_endpoint": {"provider": "OpenAI", "model": "openai/gpt-5.6-luna"},
        "provider": "OpenAI",
        "model": "openai/gpt-5.6-luna",
        "finish_reason": "stop",
        "service_tier": "default",
        "schema_id": "rubric_judgment",
        "rubric_ids": [1],
        "tokens": {"prompt": 10, "completion": 6, "total": 16, "reasoning": 2},
        "reasoning_effort": "low",
        "latency_ms": 12,
        "cost": 0.001,
        "generation_metadata_polls": 1,
        "error": None,
        "request_sha256": first.evidence["request_sha256"],
    }
    assert len(client.completions.calls) == len(client.get_calls) == 2
    assert client.get_calls[0][1] == dict[str, Any]
    assert client.get_calls[0][2] == {"params": {"id": "gen-1"}}
    request = client.completions.calls[0]
    assert request["extra_body"] == {"provider": contract["routing"]}
    assert "structured_outputs" not in request
    assert request["seed"] == 240520


def test_provenance_accepts_requested_alias_for_selected_canonical_model() -> None:
    contract, prompt = _contract()
    response = _response()
    canonical = contract["expected_canonical_slug"]
    response.openrouter_metadata["endpoints"]["available"][0]["model"] = canonical
    client = FakeClient(response, _generation(model=canonical))

    result = OpenRouterJudge(contract, prompt, "redacted", client=client).judge(
        "instruction", "response", [{"id": 1, "text": "quality"}], 240520, "low"
    )

    assert result.valid
    assert result.evidence["model"] == contract["model"]
    assert result.evidence["selected_endpoint"]["model"] == canonical


def test_transport_schema_and_provenance_failures_return_zero_without_retry() -> None:
    contract, prompt = _contract()
    transport = FakeClient(error=RuntimeError("offline"))
    result = OpenRouterJudge(contract, prompt, "redacted", client=transport).judge(
        "instruction", "response", [{"id": 1, "text": "quality"}], 240520, "none"
    )
    assert not result.valid and result.judgments[1]["score"] == 0
    assert result.evidence["error"] == "RuntimeError"
    assert len(transport.completions.calls) == 1

    malformed = FakeClient(_response('{"rubrics":[]}'), _generation())
    result = OpenRouterJudge(contract, prompt, "redacted", client=malformed).judge(
        "instruction", "response", [{"id": 1, "text": "quality"}], 240520, "none"
    )
    assert not result.valid and result.judgments[1]["score"] == 0

    mismatched = FakeClient(_response(), _generation(model="other/model"))
    result = OpenRouterJudge(contract, prompt, "redacted", client=mismatched).judge(
        "instruction", "response", [{"id": 1, "text": "quality"}], 240520, "none"
    )
    assert not result.valid and result.evidence["error"] == "ValueError"


def test_redacted_debug_canary_uses_first_stream_chunk_and_retains_only_hashes() -> None:
    contract, prompt = _contract()
    upstream = {
        "model": contract["model"],
        "input": [{"role": "user", "content": "redacted"}],
        "stream": True,
        "max_output_tokens": 2048,
        "reasoning": {"effort": "none", "summary": "detailed"},
        "text": {"format": dict(contract["response_format"]["json_schema"], type="json_schema")},
    }
    first = {
        "id": "gen-debug",
        "choices": [],
        "debug": {"echo_upstream_body": upstream},
    }
    client = FakeClient([first, {"choices": [{"delta": {"content": "done"}}]}], _generation(id="gen-debug"))
    result = OpenRouterJudge(contract, prompt, "redacted", client=client).debug_canary(
        "redacted instruction", "redacted response", [{"id": 1, "text": "redacted"}], 240520
    )
    assert result.valid
    assert result.evidence["parameter_names"] == ["max_output_tokens", "reasoning.effort", "text.format"]
    assert "messages" not in result.evidence
    request = client.completions.calls[0]
    assert request["stream"] is True
    assert request["extra_body"]["debug"] == {"echo_upstream_body": True}


def test_generation_metadata_poll_succeeds_after_delayed_404s_without_repeating_completion() -> None:
    contract, prompt = _contract()
    delays: list[float] = []
    client = FakeClient(_response(), [HttpError(404) for _ in range(5)] + [_generation()])
    result = OpenRouterJudge(contract, prompt, "redacted", client=client, sleep=delays.append).judge(
        "instruction", "response", [{"id": 1, "text": "quality"}], 240520, "none"
    )
    assert result.valid and result.evidence["generation_metadata_polls"] == 6
    assert len(client.completions.calls) == 1
    assert len(client.get_calls) == 6
    assert delays == [1.0] * 5


def test_generation_metadata_poll_fails_closed_on_non_404_without_retry() -> None:
    contract, prompt = _contract()
    client = FakeClient(_response(), [HttpError(401), _generation()])
    result = OpenRouterJudge(contract, prompt, "redacted", client=client).judge(
        "instruction", "response", [{"id": 1, "text": "quality"}], 240520, "none"
    )
    assert not result.valid and result.evidence["generation_metadata_polls"] == 1
    assert len(client.completions.calls) == len(client.get_calls) == 1


def test_generation_metadata_poll_fails_closed_after_configured_attempts() -> None:
    contract, prompt = _contract()
    client = FakeClient(_response(), [HttpError(404) for _ in range(31)])
    delays: list[float] = []
    result = OpenRouterJudge(contract, prompt, "redacted", client=client, sleep=delays.append).judge(
        "instruction", "response", [{"id": 1, "text": "quality"}], 240520, "none"
    )
    assert not result.valid and result.evidence["generation_metadata_polls"] == 31
    assert len(client.completions.calls) == 1 and len(client.get_calls) == 31
    assert delays == [1.0] * 30


@pytest.mark.parametrize(
    ("poll_contract", "match"),
    [
        ({"attempts": 0, "interval_seconds": 1}, "attempts"),
        ({"attempts": True, "interval_seconds": 1}, "attempts"),
        ({"attempts": 31, "interval_seconds": 0}, "interval"),
        ({"attempts": 31, "interval_seconds": True}, "interval"),
        ({"attempts": 31, "interval_seconds": 1, "extra": 1}, "keys"),
    ],
)
def test_generation_metadata_poll_rejects_invalid_contract(
    poll_contract: dict[str, Any],
    match: str,
) -> None:
    contract, prompt = _contract()
    contract["generation_metadata_poll"] = poll_contract

    with pytest.raises(ValueError, match=match):
        OpenRouterJudge(contract, prompt, "redacted", client=FakeClient())


def test_client_constructor_retries_transport_and_enables_router_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    contract, prompt = _contract()
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeClient:
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(openai, "OpenAI", factory)
    OpenRouterJudge(contract, prompt, "redacted")
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert TRANSPORT_RETRIES > 0
    assert captured["max_retries"] == TRANSPORT_RETRIES
    assert captured["default_headers"] == {"X-OpenRouter-Metadata": "enabled"}


def test_no_call_catalog_endpoint_gate_returns_redacted_hashable_snapshots() -> None:
    contract, _ = _contract()
    responses = {
        contract["catalog_url"]: {
            "data": [
                {
                    "id": contract["model"],
                    "canonical_slug": contract["expected_canonical_slug"],
                    "supported_parameters": contract["catalog_snapshot"]["supported_parameters"],
                    "description": "not retained",
                }
            ]
        },
        contract["endpoints_url"]: {
            "data": {
                "id": contract["model"],
                "endpoints": [
                    {
                        "provider_name": "OpenAI",
                        "model_id": contract["model"],
                        "tag": "openai",
                        "status": 0,
                        "supported_parameters": contract["catalog_snapshot"]["supported_parameters"],
                        "privacy_policy": "not retained",
                    },
                    {
                        "provider_name": "OpenAI",
                        "model_id": contract["model"],
                        "tag": "openai/flex",
                        "status": 0,
                        "supported_parameters": contract["catalog_snapshot"]["supported_parameters"],
                    },
                    {
                        "provider_name": "OpenAI",
                        "model_id": contract["model"],
                        "tag": "openai/priority",
                        "status": 0,
                        "supported_parameters": contract["catalog_snapshot"]["supported_parameters"],
                    },
                ],
            }
        },
    }
    snapshots = preflight_snapshots(contract, responses.__getitem__)
    assert snapshots["endpoints"] == {
        "id": contract["model"],
        "provider": "openai",
        "model": contract["model"],
        "tag": "openai",
        "supported_parameters": sorted(contract["catalog_snapshot"]["supported_parameters"]),
        "status": 0,
    }
    assert "description" not in json.dumps(snapshots)
    assert len(snapshots["catalog_sha256"]) == len(snapshots["endpoints_sha256"]) == 64


def test_calibration_plan_is_exactly_200_calls_over_76_cases() -> None:
    ids = {
        "debug": ["debug-1"],
        "labeled": [f"labeled-{index}" for index in range(49)],
        "heldout": [f"heldout-{index}" for index in range(26)],
    }
    plan = calibration_plan(ids, "low")
    assert len(plan) == 200
    assert len({case_id for case_id, _, _ in plan}) == 76
    assert sum(case_id.startswith("labeled") for case_id, _, _ in plan) == 147
    assert sum(case_id.startswith("heldout") for case_id, _, _ in plan) == 52


def test_paired_bootstrap_selection_is_deterministic_and_prefers_higher_noninferior_effort() -> None:
    indicators = {
        "low": [1] * 46 + [0] * 3,
        "medium": [1] * 47 + [0] * 2,
        "high": [1] * 48 + [0],
    }
    first = select_reasoning_effort(indicators)
    second = select_reasoning_effort(indicators)
    assert first == second
    assert first["bootstrap_samples"] == 10_000
    assert first["bootstrap_seed"] == 240520
    assert first["noninferiority_margin"] == -0.02
    assert first["reference_effort"] == "high"
    assert first["selected_effort"] == "high"
    assert first["selected_effort"] in first["qualifying_efforts"]


def test_calibration_cases_freeze_heldout_labels_and_production_profile(tmp_path: Path) -> None:
    cases_path = ROOT / "configs/judges/qwen_judge_calibration_cases.jsonl"
    cases = calibrate_judge._load_cases(cases_path, require_human_review=True)
    assert calibrate_judge._corpus_summary(cases) == calibrate_judge.EXPECTED_CORPUS_SUMMARY
    heldout = next(case for case in cases if case["split"] == "heldout")
    heldout.pop("expected_scores")
    changed = tmp_path / "changed.jsonl"
    changed.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen expected score"):
        calibrate_judge._load_cases(changed, require_human_review=True)

    labeled_012 = next(case for case in cases if case["case_id"] == "labeled-012")
    assert labeled_012["expected_scores"]["1"] == labeled_012["expected_scores"]["10"] == -1
    assert next(case for case in cases if case["case_id"] == "labeled-002")["expected_scores"]["1"] == -1


def test_calibration_runner_requires_human_reviewed_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = ROOT / "configs/judges/qwen_judge_calibration_cases.jsonl"
    cases = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    for case in cases:
        case["provenance"]["review_status"] = "not_human_reviewed"
    cases_path = tmp_path / "not_reviewed_cases.jsonl"
    cases_path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    monkeypatch.setattr(calibrate_judge, "CASES", cases_path.resolve())
    with pytest.raises(ValueError, match="human_reviewed"):
        calibrate_judge.run_calibration(cases_path, tmp_path / "raw.jsonl", tmp_path / "certificate.json", "redacted")


def test_calibration_runner_enforces_accuracy_and_injection_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = ROOT / "configs/judges/qwen_judge_calibration_cases.jsonl"
    cases = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    for case in cases:
        case["provenance"]["review_status"] = "human_reviewed"
    cases_path = tmp_path / "reviewed_cases.jsonl"
    cases_path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    monkeypatch.setattr(calibrate_judge, "CASES", cases_path)
    scored = {(case["instruction"], case["response"]): case for case in cases if case["split"] != "debug"}
    injection = {(case["instruction"], case["response"]) for case in cases if "injection" in case.get("tags", [])}
    calls: list[tuple[str, str, int]] = []
    mode = {"value": "gold"}

    class FakeJudge:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def judge(self, instruction: str, response: str, rubrics: list[dict], seed: int, effort: str) -> JudgeResult:
            calls.append((instruction, effort, seed))
            generation_id = f"gen-{len(calls)}"
            evidence = {"generation_id": generation_id, "request_sha256": str(len(calls)).zfill(64)}
            case = scored[(instruction, response)]
            expected = {int(key): value for key, value in case["expected_scores"].items()}
            if mode["value"] == "constant" or (
                mode["value"] == "injection_following" and (instruction, response) in injection
            ):
                expected = {rubric["id"]: 1 for rubric in rubrics}
            judgments = {rubric["id"]: {"score": expected[rubric["id"]], "reason": "fixture"} for rubric in rubrics}
            return JudgeResult(judgments, evidence, True, {"response": generation_id})

        def debug_canary(self, instruction: str, response: str, rubrics: list[dict], seed: int) -> JudgeResult:
            calls.append((instruction, "none", seed))
            evidence = {
                "generation_id": "gen-1",
                "provider": "OpenAI",
                "model": "openai/gpt-5.6-luna",
                "parameter_names": ["max_tokens", "reasoning_effort", "response_format", "seed"],
                "upstream_body_sha256": "1" * 64,
                "generation_metadata_polls": 1,
                "request_sha256": "2" * 64,
                "error": None,
            }
            return JudgeResult({}, evidence, True, {"debug_upstream_body": {"redacted": True}})

    monkeypatch.setattr(calibrate_judge, "OpenRouterJudge", FakeJudge)
    monkeypatch.setattr(
        calibrate_judge,
        "preflight_snapshots",
        lambda *_: {
            "catalog": {"id": "openai/gpt-5.6-luna"},
            "catalog_sha256": "1" * 64,
            "endpoints": {"id": "openai/gpt-5.6-luna"},
            "endpoints_sha256": "2" * 64,
        },
    )
    certificates = {}
    for run_mode in ("gold", "constant", "injection_following"):
        mode["value"] = run_mode
        calls.clear()
        run_root = tmp_path / run_mode
        raw_path = run_root / "raw.jsonl"
        certificate = calibrate_judge.run_calibration(cases_path, raw_path, run_root / "certificate.json", "redacted")
        certificates[run_mode] = certificate
        assert len(calls) == len(raw_path.read_text(encoding="utf-8").splitlines()) == 200

    gold = certificates["gold"]
    assert gold["status"] == "calibrated"
    assert gold["case_manifest"]["path"] == "configs/judges/qwen_judge_calibration_cases.jsonl"
    assert gold["case_manifest"]["file_sha256"] == hashlib.sha256(cases_path.read_bytes()).hexdigest()
    assert all(len(row["scores_sha256"]) == 64 for row in gold["call_manifest"]["rows"])
    assert all(len(row["raw_record_sha256"]) == 64 for row in gold["call_manifest"]["rows"])
    raw_rows = [json.loads(line) for line in (tmp_path / "gold/raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(
        call["scores_sha256"] == raw["scores_sha256"] == calibrate_judge._sha256(raw["scores"])
        and call["raw_record_sha256"] == calibrate_judge._sha256(raw)
        for call, raw in zip(gold["call_manifest"]["rows"], raw_rows, strict=True)
    )
    assert (
        gold["call_manifest"]["raw_file_sha256"]
        == hashlib.sha256((tmp_path / "gold/raw.jsonl").read_bytes()).hexdigest()
    )
    assert gold["case_manifest"]["corpus_summary"]["rubric_count_bins"] == {
        "1": 21,
        "2": 41,
        "3": 1,
        "6": 4,
        "8": 4,
        "10": 2,
        "12": 2,
    }
    assert gold["outcomes"]["metrics"] == {name: 1.0 for name in calibrate_judge.THRESHOLDS}
    assert gold["outcomes"]["threshold_passes"] == {name: True for name in calibrate_judge.THRESHOLDS}
    assert gold["outcomes"]["failures"] == 0
    assert certificates["constant"]["status"] == "failed"
    assert not certificates["constant"]["outcomes"]["threshold_passes"]["selected_labeled_exact_accuracy"]
    assert certificates["injection_following"]["status"] == "failed"
    assert certificates["injection_following"]["outcomes"]["metrics"]["injection_exact_accuracy"] == 0.0
