from __future__ import annotations

import copy
import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rdan_grpo.runtime_parity import ParityObservation, RuntimeIdentity
from rdan_grpo.vllm_runtime_parity import (
    COMPARISON_POLICY,
    INFER_LOGPROBS_SOURCE,
    VLLMParityError,
    run_vllm_runtime_parity,
    validate_vllm_runtime_parity,
)

SHA = "a" * 64
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "b" * 40
RTT_REVISION = "c" * 40
ROOT = Path(__file__).resolve().parents[1]


class _GeneratingArgs:
    def to_dict(self) -> dict[str, object]:
        return {"num_return_sequences": 8, "temperature": 0.99}


class _Boundary:
    def __init__(self, observation: ParityObservation) -> None:
        self.observation = observation

    def collect_parity(self, responses: int, generation_config: dict[str, object]) -> ParityObservation:
        assert responses == 32
        assert generation_config["logprobs"] == 1
        return self.observation


def _config() -> SimpleNamespace:
    actor_train = SimpleNamespace(
        strategy_args=SimpleNamespace(
            strategy_name="fsdp2_train",
            strategy_config={"transformer_impl": "huggingface"},
        )
    )
    actor_infer = SimpleNamespace(
        strategy_args=SimpleNamespace(strategy_name="vllm", strategy_config={}),
        generating_args=_GeneratingArgs(),
    )
    return SimpleNamespace(
        async_pipeline=False,
        async_generation_ratio=0,
        actor_train=actor_train,
        actor_infer=actor_infer,
    )


def _observation() -> ParityObservation:
    tokens = torch.arange(32 * 5).reshape(32, 5)
    attention = torch.ones_like(tokens)
    response = torch.tensor([[0, 0, 1, 1, 1]] * 32)
    infer = torch.full((32, 4), -0.5)
    actor = torch.full((32, 4), -0.51)
    return ParityObservation(
        input_ids=tokens,
        attention_mask=attention,
        response_mask=response,
        infer_logprobs=infer,
        actor_logprobs=actor,
        actor_input_ids=tokens.clone(),
        actor_attention_mask=attention.clone(),
        actor_response_mask=response.clone(),
        infer_logprobs_source=INFER_LOGPROBS_SOURCE,
        actor_train_recomputed=True,
        actor_boundary_observed=True,
        optimizer_updates=0,
    )


def _artifact(observation: ParityObservation | None = None) -> dict[str, object]:
    identity = RuntimeIdentity(MODEL, REVISION, "1" * 64, "2" * 64, "3" * 64)
    return run_vllm_runtime_parity(
        _Boundary(observation or _observation()),
        identity,
        pipeline_config=_config(),
        parity_config_sha256="4" * 64,
        parity_resolved_config_sha256="5" * 64,
        production_config_sha256="6" * 64,
        production_resolved_config_sha256="7" * 64,
        rtt_revision=RTT_REVISION,
        weight_receipt={
            "transaction_id": "receipt-1",
            "artifact_sha256": "8" * 64,
            "resolved_config_sha256": "5" * 64,
        },
    )


def _validate(artifact: dict[str, object]) -> object:
    return validate_vllm_runtime_parity(
        artifact,
        artifact_id="qwen_vllm_runtime_parity_v1",
        model=MODEL,
        revision=REVISION,
        rtt_revision=RTT_REVISION,
        parity_config_sha256="4" * 64,
        production_config_sha256="6" * 64,
    )


def _runner() -> object:
    path = ROOT / "scripts/run_vllm_response_parity.py"
    spec = importlib.util.spec_from_file_location("test_run_vllm_response_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vllm_drift_is_diagnostic_without_a_threshold() -> None:
    artifact = _artifact()
    diagnostic = artifact["diagnostic"]

    assert artifact["comparison_policy"] == COMPARISON_POLICY
    assert isinstance(diagnostic, dict)
    assert diagnostic["mean_abs_error"] == pytest.approx(0.01)
    assert "threshold" not in str(artifact).lower()
    assert _validate(artifact) == artifact


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: replace(value, optimizer_updates=1), "optimizer"),
        (lambda value: replace(value, actor_train_recomputed=False), "recomputation"),
        (lambda value: replace(value, actor_boundary_observed=False), "recomputation"),
        (lambda value: replace(value, actor_input_ids=value.actor_input_ids + 1), "boundaries"),
        (
            lambda value: replace(
                value,
                infer_logprobs=value.infer_logprobs.index_put(
                    (torch.tensor([0]), torch.tensor([2])), torch.tensor([float("nan")])
                ),
            ),
            "non-finite",
        ),
    ],
)
def test_vllm_parity_fails_closed_before_artifact_creation(mutation: object, message: str) -> None:
    observation = mutation(_observation())  # type: ignore[operator]

    with pytest.raises(VLLMParityError, match=message):
        _artifact(observation)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(comparison_policy="thresholded"),
        lambda value: value["runtime_backend"].update(production_config_sha256=SHA),
        lambda value: value["weight_receipt"].update(transaction_id=""),
        lambda value: value["diagnostic"].update(optimizer_updates=1),
        lambda value: value["diagnostic"].update(sampled_logprobs_finite=False),
        lambda value: value["diagnostic"].update(mean_abs_error=float("nan")),
        lambda value: value["diagnostic"].update(thresholds={}),
    ],
)
def test_serialized_vllm_artifact_rejects_tampering(mutation: object) -> None:
    artifact = copy.deepcopy(_artifact())
    mutation(artifact)  # type: ignore[operator]

    with pytest.raises(VLLMParityError, match="invalid"):
        _validate(artifact)


def test_live_runner_requires_exact_program_config_paths(tmp_path: Path) -> None:
    module = _runner()
    parity = tmp_path / "parity.yaml"
    production = tmp_path / "production.yaml"
    parity.write_text("parity\n", encoding="utf-8")
    production.write_text("production\n", encoding="utf-8")
    program = SimpleNamespace(
        repo_root=tmp_path,
        program={
            "same_backend_configs": {
                "vllm_diagnostic": {
                    "path": parity.name,
                    "sha256": hashlib.sha256(parity.read_bytes()).hexdigest(),
                },
                "production": {
                    "path": production.name,
                    "sha256": hashlib.sha256(production.read_bytes()).hexdigest(),
                },
            }
        },
    )
    module._validate_program_configs(program, parity, production)

    other = tmp_path / "other.yaml"
    other.write_bytes(parity.read_bytes())
    with pytest.raises(VLLMParityError, match="frozen program"):
        module._validate_program_configs(program, other, production)


def test_live_parity_offloads_vllm_before_actor_recomputation() -> None:
    source = (ROOT / "src/rdan_grpo/roll_vllm_parity_live.py").read_text(encoding="utf-8")
    collect = source[source.index("def collect_parity(") : source.index("def build_vllm_parity_pipeline(")]

    assert collect.index("self.actor_train.offload_states") < collect.index("self.actor_infer.load_states")
    assert collect.index("self.actor_infer.offload_states") < collect.index("self.actor_train.load_states")
    assert collect.index("self.actor_train.load_states") < collect.index("self.actor_train.compute_log_probs")
