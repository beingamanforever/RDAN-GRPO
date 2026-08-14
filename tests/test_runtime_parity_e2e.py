from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from rdan_grpo import program, runtime_parity
from rdan_grpo.runtime_parity import (
    FSDP2_HF_PROFILE,
    GENERATION_SOURCE_SHA256,
    MAX_ABS_ERROR,
    MEAN_ABS_ERROR,
    ParityError,
    ParityObservation,
    RuntimeIdentity,
    run_runtime_parity,
    verify_transformers_generation_boundary,
    write_artifact,
)

TRAIN_CONFIG_SHA256 = "4" * 64
RESOLVED_CONFIG_SHA256 = "5" * 64
WEIGHT_RECEIPT = {
    "transaction_id": "parity-transaction",
    "artifact_sha256": "6" * 64,
    "resolved_config_sha256": RESOLVED_CONFIG_SHA256,
}
ROOT = Path(__file__).resolve().parents[1]


class _GeneratingArgs:
    def __init__(self) -> None:
        self.values = {"max_new_tokens": 32, "num_return_sequences": 8, "temperature": 0.9}

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


class _FaithfulRollBoundary:
    def __init__(self, observation: ParityObservation) -> None:
        self.observation = observation
        self.requests: list[tuple[int, dict[str, Any]]] = []

    def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> ParityObservation:
        self.requests.append((responses, dict(generation_config)))
        return self.observation


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        async_pipeline=False,
        async_generation_ratio=0,
        generate_opt_level=1,
        actor_train=SimpleNamespace(
            strategy_args=SimpleNamespace(
                strategy_name="megatron_train",
                strategy_config={"transformer_impl": "local"},
            ),
        ),
        actor_infer=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name="vllm"),
            generating_args=_GeneratingArgs(),
        ),
    )


def _same_backend_config() -> SimpleNamespace:
    return SimpleNamespace(
        async_pipeline=False,
        async_generation_ratio=0,
        generate_opt_level=0,
        actor_train=SimpleNamespace(
            strategy_args=SimpleNamespace(
                strategy_name="fsdp2_train",
                strategy_config={"transformer_impl": "huggingface"},
            ),
        ),
        actor_infer=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name="hf_infer"),
            generating_args=SimpleNamespace(
                to_dict=lambda: {
                    "do_sample": True,
                    "max_new_tokens": 32,
                    "num_beams": 1,
                    "num_return_sequences": 8,
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                }
            ),
        ),
    )


def _observation() -> ParityObservation:
    rows = 32
    input_ids = torch.arange(rows * 8, dtype=torch.long).reshape(rows, 8)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long).repeat(rows, 1)
    response_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.long).repeat(rows, 1)
    infer_logprobs = torch.zeros((rows, 7), dtype=torch.float32)
    actor_logprobs = infer_logprobs.clone()
    actor_logprobs[response_mask[:, 1:].bool()] = 5e-5
    return ParityObservation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        infer_logprobs=infer_logprobs,
        actor_logprobs=actor_logprobs,
        actor_input_ids=input_ids.clone(),
        actor_attention_mask=attention_mask.clone(),
        actor_response_mask=response_mask.clone(),
        infer_logprobs_source="observed_rollout_engine",
        actor_train_recomputed=True,
        actor_boundary_observed=True,
        optimizer_updates=0,
    )


def _run(
    boundary: _FaithfulRollBoundary,
    identity: RuntimeIdentity,
    config: SimpleNamespace | None = None,
    failure_output: Path | None = None,
) -> dict:
    return run_runtime_parity(
        boundary,
        identity,
        pipeline_config=config or _config(),
        train_config_sha256=TRAIN_CONFIG_SHA256,
        resolved_config_sha256=RESOLVED_CONFIG_SHA256,
        rtt_revision=program.RTT_REVISION,
        weight_receipt=WEIGHT_RECEIPT,
        failure_output=failure_output,
    )


def test_runtime_parity_caller_boundary_and_lifecycle_contract(tmp_path: Path) -> None:
    config = _config()
    observation = _observation()
    boundary = _FaithfulRollBoundary(observation)
    identity = RuntimeIdentity(
        model=program.MODEL_NAME,
        revision=program.MODEL_REVISION,
        snapshot_sha256="1" * 64,
        tokenizer_files_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )

    failure_output = tmp_path / "failure.json"
    artifact = _run(boundary, identity, config, failure_output)
    assert not failure_output.exists()
    assert boundary.requests == [
        (
            32,
            {
                "max_new_tokens": 32,
                "num_return_sequences": 8,
                "temperature": 0.9,
                "logprobs": 1,
            },
        )
    ]
    assert config.actor_infer.generating_args.values == {
        "max_new_tokens": 32,
        "num_return_sequences": 8,
        "temperature": 0.9,
    }
    evidence = artifact["rollout_logprob_evidence"]
    assert evidence["responses"] == 32
    assert evidence["compared_tokens"] == 96
    assert evidence["optimizer_updates"] == 0
    assert evidence["infer_logprobs_source"] == "observed_rollout_engine"
    assert evidence["actor_train_recomputed"] is True
    assert evidence["actor_boundary_observed"] is True
    assert "blocking_surface" not in evidence
    assert "diagnostic_surface" not in evidence
    assert "surface_comparisons" not in evidence
    assert artifact["runtime_backend"] == {
        "train_config_sha256": TRAIN_CONFIG_SHA256,
        "resolved_config_sha256": RESOLVED_CONFIG_SHA256,
        "actor_train_strategy": "megatron_train",
        "actor_infer_strategy": "vllm",
        "transformer_impl": "local",
        "rtt_revision": program.RTT_REVISION,
    }
    assert artifact["weight_receipt"] == WEIGHT_RECEIPT
    reference = {"artifact_id": "qwen_runtime_parity_v1"}
    with pytest.raises(program.ProgramContractError, match="runtime parity backend keys are invalid"):
        program._validate_runtime_parity(artifact, reference, ROOT)

    output = tmp_path / "parity.json"
    write_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    with pytest.raises(FileExistsError):
        write_artifact(output, artifact)

    failures = (
        replace(observation, infer_logprobs=None),
        replace(observation, infer_logprobs_source="old_log_probs_fallback"),
        replace(observation, optimizer_updates=1),
        replace(observation, actor_boundary_observed=False),
        replace(observation, actor_input_ids=observation.actor_input_ids + 1),
        replace(observation, actor_response_mask=torch.zeros_like(observation.response_mask)),
    )
    for invalid in failures:
        with pytest.raises(ParityError):
            _run(_FaithfulRollBoundary(invalid), identity)


def test_same_backend_profile_is_explicit_and_old_profile_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_identity = {
        "transformers_version": "4.57.0",
        **GENERATION_SOURCE_SHA256,
    }
    monkeypatch.setattr(
        "rdan_grpo.runtime_parity.verify_transformers_generation_boundary",
        lambda: dict(source_identity),
    )
    identity = RuntimeIdentity(
        model=program.MODEL_NAME,
        revision=program.MODEL_REVISION,
        snapshot_sha256="1" * 64,
        tokenizer_files_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )
    observation = replace(
        _observation(),
        infer_logprobs_source="observed_hf_generation",
        infer_full_logprobs=_observation().actor_logprobs.clone(),
    )
    boundary = _FaithfulRollBoundary(observation)
    parity_config_sha256 = hashlib.sha256(
        (ROOT / "configs/roll/qwen_scalar_same_backend_parity.yaml").read_bytes()
    ).hexdigest()
    artifact = run_runtime_parity(
        boundary,
        identity,
        pipeline_config=_same_backend_config(),
        train_config_sha256=parity_config_sha256,
        resolved_config_sha256=RESOLVED_CONFIG_SHA256,
        rtt_revision=program.RTT_REVISION,
        weight_receipt=WEIGHT_RECEIPT,
        production_train_config_sha256="d" * 64,
        production_resolved_config_sha256="e" * 64,
        preflight_train_config_sha256="f" * 64,
        preflight_resolved_config_sha256="0" * 64,
        backend_profile=FSDP2_HF_PROFILE,
    )

    assert boundary.requests == [
        (
            32,
            {
                "do_sample": True,
                "max_new_tokens": 32,
                "num_beams": 1,
                "num_return_sequences": 8,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
        )
    ]
    assert artifact["runtime_backend"] == {
        "train_config_sha256": parity_config_sha256,
        "production_train_config_sha256": "d" * 64,
        "production_resolved_config_sha256": "e" * 64,
        "preflight_train_config_sha256": "f" * 64,
        "preflight_resolved_config_sha256": "0" * 64,
        "resolved_config_sha256": RESOLVED_CONFIG_SHA256,
        "actor_train_strategy": "fsdp2_train",
        "actor_infer_strategy": "hf_infer",
        "transformer_impl": "huggingface",
        "rtt_revision": program.RTT_REVISION,
        **source_identity,
    }
    evidence = artifact["rollout_logprob_evidence"]
    assert evidence["infer_logprobs_source"] == "observed_hf_generation"
    assert evidence["blocking_surface"] == "infer_full_vs_actor_full"
    assert evidence["diagnostic_surface"] == "generation_vs_infer_full"
    assert set(evidence["surface_comparisons"]) == {
        "generation_vs_infer_full",
        "infer_full_vs_actor_full",
    }
    configs = {
        "diagnostic": {"sha256": parity_config_sha256},
        "production": {"status": "frozen", "sha256": "d" * 64},
    }
    program._validate_runtime_parity(artifact, {"artifact_id": "qwen_runtime_parity_v1"}, configs)
    with pytest.raises(ParityError, match="fallback source"):
        run_runtime_parity(
            _FaithfulRollBoundary(replace(observation, infer_logprobs_source="observed_rollout_engine")),
            identity,
            pipeline_config=_same_backend_config(),
            train_config_sha256=TRAIN_CONFIG_SHA256,
            resolved_config_sha256=RESOLVED_CONFIG_SHA256,
            rtt_revision=program.RTT_REVISION,
            weight_receipt=WEIGHT_RECEIPT,
            production_train_config_sha256="d" * 64,
            production_resolved_config_sha256="e" * 64,
            preflight_train_config_sha256="f" * 64,
            preflight_resolved_config_sha256="0" * 64,
            backend_profile=FSDP2_HF_PROFILE,
        )
    assert _run(_FaithfulRollBoundary(_observation()), identity)["runtime_backend"]["actor_infer_strategy"] == "vllm"


def test_transformers_generation_boundary_rejects_version_and_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GenerationMixin:
        def _get_logits_processor(self) -> None:
            return None

        def _sample(self) -> None:
            return None

    transformers = type("Transformers", (), {"__version__": "4.57.0"})
    generation = type("Generation", (), {"GenerationMixin": GenerationMixin})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.generation.utils", generation)
    sources = {
        GenerationMixin._get_logits_processor: "get logits processor\n",
        GenerationMixin._sample: "sample\n",
    }
    monkeypatch.setattr("rdan_grpo.runtime_parity.inspect.getsource", lambda function: sources[function])

    with pytest.raises(ParityError, match="source drift"):
        verify_transformers_generation_boundary()

    transformers.__version__ = "4.57.1"
    with pytest.raises(ParityError, match="version drift"):
        verify_transformers_generation_boundary()


def test_collection_exception_is_sealed_without_message_leakage(tmp_path: Path) -> None:
    secret = "do-not-write-this-secret"
    sentinel = {
        "compared_tokens": 999,
        "worst_token_evidence": [{"response_sha256": secret}],
        "worker_aggregates": [{"worker_sha256": secret}],
    }

    original = ParityError(secret, code="threshold_exceeded", diagnostics=sentinel)

    class BrokenBoundary:
        def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> ParityObservation:
            del responses, generation_config
            raise original

    identity = RuntimeIdentity(
        model=program.MODEL_NAME,
        revision=program.MODEL_REVISION,
        snapshot_sha256="1" * 64,
        tokenizer_files_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )
    output = tmp_path / "failure.json"

    with pytest.raises(ParityError, match=secret) as raised:
        _run(BrokenBoundary(), identity, failure_output=output)  # type: ignore[arg-type]

    assert raised.value is original
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["failure"] == {"code": "threshold_exceeded", "type": "ParityError"}
    assert artifact["comparison"]["compared_tokens"] == 0
    assert artifact["comparison"]["worst_token_evidence"] == []
    assert "worker_aggregates" not in artifact["comparison"]
    assert secret not in json.dumps(artifact, sort_keys=True)


def test_threshold_failure_emits_aggregate_alignment_diagnostics() -> None:
    observation = _observation()
    mask = observation.response_mask[:, 1:].bool()
    observation.infer_logprobs[mask] = torch.tensor([-1.0, -2.0, -3.0]).repeat(32)
    observation.actor_logprobs[mask] = torch.tensor([-9.0, -1.0, -2.0]).repeat(32)
    identity = RuntimeIdentity(
        model=program.MODEL_NAME,
        revision=program.MODEL_REVISION,
        snapshot_sha256="1" * 64,
        tokenizer_files_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )

    with pytest.raises(ParityError, match="runtime parity exceeds thresholds") as raised:
        _run(_FaithfulRollBoundary(observation), identity)

    message = str(raised.value)
    payload = message.split("diagnostics=", 1)[1]
    diagnostics = json.loads(payload)
    assert payload == json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
    assert diagnostics["compared_tokens"] == 96
    assert diagnostics["infer_mean_logprob"] == pytest.approx(-2.0)
    assert diagnostics["actor_mean_logprob"] == pytest.approx(-4.0)
    assert diagnostics["signed_mean_difference"] == pytest.approx(2.0)
    assert diagnostics["rmse"] == pytest.approx((66 / 3) ** 0.5)
    assert set(diagnostics["absolute_error_percentiles"]) == {"p50", "p90", "p95", "p99"}
    assert set(diagnostics["absolute_error_fractions"]) == {"<=1e-3", "<=1e-2", "<=5e-2", "<=1e-1"}
    assert diagnostics["first_response_token"] == {
        "count": 32,
        "mean_abs_error": 8.0,
        "max_abs_error": 8.0,
    }
    assert diagnostics["later_response_tokens"] == {
        "count": 64,
        "mean_abs_error": 1.0,
        "max_abs_error": 1.0,
    }
    assert list(diagnostics["response_position_bins"]) == ["0", "1", "16-63", "2", "256+", "3", "4-15", "64-255"]
    assert diagnostics["response_position_bins"]["0"]["count"] == 32
    assert diagnostics["response_position_bins"]["1"]["count"] == 32
    assert diagnostics["response_position_bins"]["2"]["count"] == 32
    assert diagnostics["response_position_bins"]["3"]["count"] == 0
    assert diagnostics["actor_shift_mean_abs_error"]["+1"] == 0.0
    assert diagnostics["actor_shift_mean_abs_error"]["-1"] > 0.0
    assert diagnostics["thresholds"] == {
        "max_abs_error_at_most": 1e-3,
        "mean_abs_error_at_most": 1e-4,
    }
    assert MAX_ABS_ERROR == 1e-3
    assert MEAN_ABS_ERROR == 1e-4
    assert all(
        field not in message for field in ("input_ids", "token_ids", "prompt", "infer_logprobs", "actor_logprobs")
    )


def test_same_backend_threshold_selects_full_forward_and_reports_generation_diagnostic() -> None:
    observation = _observation()
    mask = observation.response_mask[:, 1:].bool()
    generation = observation.infer_logprobs.clone()
    generation[mask] = 1.0
    infer_full = observation.actor_logprobs.clone()
    passing = replace(
        observation,
        infer_logprobs=generation,
        infer_logprobs_source="observed_hf_generation",
        infer_full_logprobs=infer_full,
    )

    evidence = runtime_parity._assess(passing, FSDP2_HF_PROFILE)

    assert evidence["blocking_surface"] == "infer_full_vs_actor_full"
    assert evidence["diagnostic_surface"] == "generation_vs_infer_full"
    assert evidence["max_abs_error"] == 0.0
    assert evidence["mean_abs_error"] == 0.0
    assert evidence["surface_comparisons"]["generation_vs_infer_full"]["mean_abs_error"] == pytest.approx(0.99995)
    assert evidence["surface_comparisons"]["infer_full_vs_actor_full"]["mean_abs_error"] == 0.0

    failing_full = infer_full.clone()
    failing_full[mask] = 1.0
    failing = replace(passing, infer_full_logprobs=failing_full)
    with pytest.raises(ParityError, match="runtime parity exceeds thresholds") as raised:
        runtime_parity._assess(failing, FSDP2_HF_PROFILE)

    diagnostics = raised.value.diagnostics
    assert diagnostics is not None
    assert diagnostics["blocking_surface"] == "infer_full_vs_actor_full"
    assert diagnostics["diagnostic_surface"] == "generation_vs_infer_full"
    surfaces = diagnostics["surface_comparisons"]
    assert surfaces["generation_vs_infer_full"]["mean_abs_error"] == 0.0
    assert surfaces["infer_full_vs_actor_full"]["mean_abs_error"] == pytest.approx(0.99995)
    assert diagnostics["mean_abs_error"] == pytest.approx(0.99995)
    assert diagnostics["thresholds"] == {
        "max_abs_error_at_most": MAX_ABS_ERROR,
        "mean_abs_error_at_most": MEAN_ABS_ERROR,
    }


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "missing"),
        ("invalid", "malformed"),
        (torch.zeros((32, 6)), "shape_mismatch"),
        (torch.full((32, 7), float("nan")), "non_finite"),
    ],
)
def test_same_backend_full_surface_fails_closed_when_unavailable(
    value: object,
    reason: str,
) -> None:
    observation = replace(
        _observation(),
        infer_logprobs_source="observed_hf_generation",
        infer_full_logprobs=value,
    )

    with pytest.raises(ParityError, match="blocking surface infer_full_vs_actor_full is unavailable") as raised:
        runtime_parity._assess(observation, FSDP2_HF_PROFILE)

    assert raised.value.diagnostics is not None
    assert raised.value.diagnostics["blocking_surface"] == "infer_full_vs_actor_full"
    assert raised.value.diagnostics["diagnostic_surface"] == "generation_vs_infer_full"
    assert raised.value.diagnostics["surface_diagnostic"] == {
        "status": "blocking_surface_unavailable",
        "reason": reason,
    }
    assert "surface_comparisons" not in raised.value.diagnostics


def test_same_backend_computation_failure_reason_is_allowlisted() -> None:
    observation = replace(
        _observation(),
        infer_logprobs_source="observed_hf_generation",
        infer_full_unavailable_reason="computation_failed",
    )
    with pytest.raises(ParityError, match="blocking surface infer_full_vs_actor_full is unavailable") as raised:
        runtime_parity._assess(observation, FSDP2_HF_PROFILE)

    assert raised.value.diagnostics is not None
    assert raised.value.diagnostics["surface_diagnostic"] == {
        "status": "blocking_surface_unavailable",
        "reason": "computation_failed",
    }


def test_failure_artifact_is_atomic_bounded_and_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observation = _observation()
    mask = observation.response_mask[:, 1:].bool()
    observation.actor_logprobs[mask] = 0.5
    identity = RuntimeIdentity(
        model=program.MODEL_NAME,
        revision=program.MODEL_REVISION,
        snapshot_sha256="1" * 64,
        tokenizer_files_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )
    boundary = _FaithfulRollBoundary(observation)
    boundary.prompt_text = "raw-prompt-secret"
    boundary.response_text = "raw-response-secret"
    boundary.credentials = "credential-secret"
    monkeypatch.setenv("PARITY_TEST_SECRET", "environment-secret")
    failure_output = tmp_path / "failure.json"
    lifecycle_output = tmp_path / "parity.json"
    real_link = os.link

    def observed_link(source: str | Path, target: str | Path) -> None:
        assert Path(target) == failure_output
        assert not failure_output.exists()
        json.loads(Path(source).read_text(encoding="utf-8"))
        real_link(source, target)

    monkeypatch.setattr(os, "link", observed_link)
    with pytest.raises(ParityError, match="runtime parity exceeds thresholds"):
        _run(boundary, identity, failure_output=failure_output)

    assert not lifecycle_output.exists()
    artifact = json.loads(failure_output.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 2
    assert artifact["id"] == "qwen_runtime_parity_v1_failure"
    assert artifact["status"] == "parity_failed"
    assert artifact["failure"]["code"] == "threshold_exceeded"
    assert artifact["runtime_backend"]["train_config_sha256"] == TRAIN_CONFIG_SHA256
    assert artifact["runtime_backend"]["resolved_config_sha256"] == RESOLVED_CONFIG_SHA256
    assert artifact["runtime_backend"]["rtt_revision"] == program.RTT_REVISION
    assert artifact["weight_receipt"] == WEIGHT_RECEIPT
    assert artifact["model"]["snapshot_sha256"] == "1" * 64
    comparison = artifact["comparison"]
    assert comparison["returned_responses"] == 32
    assert comparison["compared_responses"] == 32
    assert comparison["compared_tokens"] == 96
    assert comparison["thresholds"] == {
        "max_abs_error_at_most": 1e-3,
        "mean_abs_error_at_most": 1e-4,
    }
    assert comparison["aggregate_error_stats"]["max_abs_error"] == 0.5
    assert comparison["response_position_aggregates"]["0"]["count"] == 32
    assert len(comparison["worst_token_evidence"]) == 16
    assert all(len(item["response_sha256"]) == 64 for item in comparison["worst_token_evidence"])
    serialized = json.dumps(artifact, sort_keys=True)
    assert all(
        secret not in serialized
        for secret in ("raw-prompt-secret", "raw-response-secret", "credential-secret", "environment-secret")
    )
    assert "input_ids" not in serialized
    assert "token_ids" not in serialized


def test_failure_artifact_refuses_overwrite_and_records_alignment_failure(tmp_path: Path) -> None:
    observation = _observation()
    identity = RuntimeIdentity(
        model=program.MODEL_NAME,
        revision=program.MODEL_REVISION,
        snapshot_sha256="1" * 64,
        tokenizer_files_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )
    output = tmp_path / "failure.json"
    invalid = replace(
        observation,
        actor_input_ids=observation.actor_input_ids + 1,
        worker_ids=tuple("worker-a" if index < 16 else "worker-b" for index in range(32)),
    )

    with pytest.raises(ParityError, match="changed prompt or response boundaries"):
        _run(_FaithfulRollBoundary(invalid), identity, failure_output=output)

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["failure"]["code"] == "alignment_failed"
    assert artifact["comparison"]["compared_responses"] == 32
    assert artifact["comparison"]["compared_tokens"] == 96
    worker_aggregates = artifact["comparison"]["worker_aggregates"]
    assert [aggregate["count"] for aggregate in worker_aggregates] == [48, 48]
    assert all(len(aggregate["worker_sha256"]) == 64 for aggregate in worker_aggregates)

    with pytest.raises(FileExistsError):
        _run(_FaithfulRollBoundary(invalid), identity, failure_output=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact


@pytest.mark.parametrize("existing_name", ["success.json", "failure.json", "receipt.json"])
def test_cli_refuses_existing_success_or_failure_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    spec = importlib.util.spec_from_file_location("run_roll_parity", ROOT / "scripts/run_roll_parity.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    success = tmp_path / "success.json"
    failure = tmp_path / "failure.json"
    receipt = tmp_path / "receipt.json"
    (tmp_path / existing_name).write_text("preserve-me\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_roll_parity.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--output",
            str(success),
            "--failure-output",
            str(failure),
            "--weight-receipt-output",
            str(receipt),
        ],
    )

    with pytest.raises(FileExistsError, match=existing_name):
        script.main()

    assert (tmp_path / existing_name).read_text(encoding="utf-8") == "preserve-me\n"
    outputs = {"success.json": success, "failure.json": failure, "receipt.json": receipt}
    assert all(path.exists() is (name == existing_name) for name, path in outputs.items())


@pytest.mark.skipif(importlib.util.find_spec("roll") is None, reason="pinned RTT ROLL is not installed")
def test_rewardless_parity_constructor_and_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from rdan_grpo.roll_compat import install_rtt_compat

    rtt_root = os.environ.get("RTT_ROOT")
    if not rtt_root:
        pytest.skip("RTT_ROOT is required for the pinned ROLL constructor test")
    install_rtt_compat(Path(rtt_root))
    from roll.pipeline.rlvr import rubircs_pipeline

    from rdan_grpo import roll_live

    events: dict[str, list[Any]] = {"clusters": [], "downloads": [], "initializes": [], "scheduler": []}

    class Dataset:
        column_names = ["domain"]

        def __len__(self) -> int:
            return 4

        def map(self, *args: Any, **kwargs: Any) -> Dataset:
            return self

        def filter(self, *args: Any, **kwargs: Any) -> Dataset:
            return self

    class Cluster:
        def __init__(self, *, name: str, worker_cls: type, resource_manager: object, worker_config: object) -> None:
            self.name = name
            events["clusters"].append((name, worker_cls))

        def initialize(self, *, pipeline_config: object, blocking: bool) -> list[str]:
            events["initializes"].append(self.name)
            return [self.name]

    class RemoteCall:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
            events["scheduler"].append((self.name, kwargs))
            return self.name, kwargs

    class Scheduler:
        set_scheduler = RemoteCall("set_scheduler")

    class RemoteScheduler:
        def options(self, **kwargs: Any) -> RemoteScheduler:
            return self

        def remote(self, **kwargs: Any) -> Scheduler:
            return Scheduler()

    def base_init(self: object, config: object) -> None:
        self.pipeline_config = config
        self.resource_manager = object()
        self.state = SimpleNamespace(kv={}, step=0)

    monkeypatch.setattr(roll_live.BasePipeline, "__init__", base_init)
    monkeypatch.setattr(
        roll_live.BasePipeline,
        "download_models",
        lambda self, *clusters: events["downloads"].append([cluster.name for cluster in clusters]),
    )
    monkeypatch.setattr(roll_live.BasePipeline, "set_model_update_pair", lambda self, **kwargs: None)
    monkeypatch.setattr(roll_live.BasePipeline, "set_checkpoint_clusters", lambda self, *clusters: None)
    monkeypatch.setattr(roll_live, "Cluster", Cluster)
    monkeypatch.setattr(roll_live, "default_tokenizer_provider", lambda **kwargs: object())
    monkeypatch.setattr(
        roll_live,
        "load_response_dataset",
        lambda file_names, *, dataset_dir: Dataset(),
    )
    monkeypatch.setattr(rubircs_pipeline, "get_encode_function", lambda *args: object())
    monkeypatch.setattr(rubircs_pipeline, "preprocess_dataset", lambda dataset, *args, **kwargs: dataset)
    monkeypatch.setattr(roll_live.ray, "remote", lambda cls: RemoteScheduler())
    monkeypatch.setattr(roll_live.ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(
        roll_live.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node"),
    )
    monkeypatch.setattr(roll_live, "NodeAffinitySchedulingStrategy", lambda **kwargs: kwargs)

    def config(actor_cls: type = roll_live.ObservedActorWorker) -> SimpleNamespace:
        reward_config = {"llm_judge": object()}
        data_args = SimpleNamespace(
            domain_interleave_probs={"llm_judge": 1.0},
            file_name=["data/qwen_hir_rubrichub_if_rl_eligible.jsonl"],
            dataset_dir=".",
            preprocessing_num_workers=1,
            template="qwen3_nothinking",
        )
        actor_train = SimpleNamespace(
            name="actor_train",
            worker_cls=actor_cls,
            model_args=object(),
            data_args=data_args,
            model_update_frequency=1,
        )
        actor_infer = SimpleNamespace(
            name="actor_infer",
            worker_cls=roll_live.ObservedLogprobInferWorker,
            model_args=object(),
            strategy_args=SimpleNamespace(
                strategy_config={"worker_extension_cls": roll_live.RECEIPT_WORKER_EXTENSION}
            ),
        )
        result = SimpleNamespace(
            actor_train=actor_train,
            actor_infer=actor_infer,
            global_template="qwen3_nothinking",
            max_steps=20,
            prompt_length=64,
            tag_2_domain={},
            rewards=reward_config,
        )
        result.set_max_steps = lambda max_steps: events.setdefault("max_steps", []).append(max_steps)
        return result

    pipeline_config = config()
    rewards_before = dict(pipeline_config.rewards)
    pipeline = roll_live.RuntimeParityPipeline(pipeline_config)
    assert events["clusters"] == [
        ("actor_train", roll_live.ObservedActorWorker),
        ("actor_infer", roll_live.ObservedLogprobInferWorker),
    ]
    assert events["downloads"] == [["actor_train", "actor_infer"]]
    assert events["initializes"] == ["actor_infer", "actor_train"]
    assert events["max_steps"] == [20]
    scheduler_kwargs = events["scheduler"][0][1]
    assert set(scheduler_kwargs["reward_clusters"]) == {"llm_judge"}
    assert scheduler_kwargs["reward_clusters"]["llm_judge"].workers == ()
    assert pipeline.rewards == {}
    assert pipeline_config.rewards == rewards_before

    batch = roll_live.DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((8, 6), dtype=torch.long),
            "attention_mask": torch.ones((8, 6), dtype=torch.long),
            "response_mask": torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long).repeat(8, 1),
            "infer_logprobs": torch.zeros((8, 5)),
        }
    )
    actor_output = roll_live.DataProto.from_dict(
        tensors={
            "log_probs": torch.zeros((8, 5)),
            "actor_input_ids": batch.batch["input_ids"].clone(),
            "actor_attention_mask": batch.batch["attention_mask"].clone(),
            "actor_response_mask": batch.batch["response_mask"].clone(),
        }
    )

    class BoundaryCall:
        def __init__(self, name: str, result: Any) -> None:
            self.name = name
            self.result = result

        def remote(self, **kwargs: Any) -> Any:
            events["scheduler"].append((self.name, kwargs))
            return self.result

    boundary_scheduler = SimpleNamespace(
        get_batch=BoundaryCall("get_batch", batch),
        shutdown=BoundaryCall("shutdown", None),
    )
    pipeline.generate_schedulers = {"llm_judge": boundary_scheduler}
    pipeline.pipeline_config = SimpleNamespace(rpc_timeout=10)
    pipeline.state = SimpleNamespace(step=0)
    pipeline.actor_train = SimpleNamespace(
        offload_states=lambda **kwargs: None,
        compute_log_probs=lambda data, **kwargs: actor_output,
    )
    pipeline.actor_infer = SimpleNamespace(load_states=lambda **kwargs: None, offload_states=lambda **kwargs: None)
    pipeline.model_update = lambda step: events.setdefault("model_updates", []).append(step)
    pipeline._weight_receipt_passed = True
    observation = pipeline.collect_parity(8, {"num_return_sequences": 8, "logprobs": 1})
    get_batch_event = next(item for item in events["scheduler"] if item[0] == "get_batch")
    assert get_batch_event[1]["data"].meta_info["skip_rewards"] is True
    assert any(name == "shutdown" for name, _ in events["scheduler"])
    assert "model_updates" not in events
    assert observation.optimizer_updates == 0

    with pytest.raises(ValueError, match="ObservedActorWorker"):
        roll_live.RuntimeParityPipeline(config(type("TamperedActor", (), {})))
    assert len(events["clusters"]) == 2
