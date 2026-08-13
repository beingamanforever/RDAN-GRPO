from __future__ import annotations

import copy
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    tracking = types.ModuleType("rdan_grpo.wandb_tracking")
    tracking.canonical_config_sha256 = lambda value: "f" * 64
    monkeypatch.setitem(sys.modules, "rdan_grpo.wandb_tracking", tracking)
    path = ROOT / "src/rdan_grpo/roll_response_config.py"
    spec = importlib.util.spec_from_file_location("test_response_config_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _payload(method: str = "rtt_papo_response") -> dict[str, Any]:
    quality = 0.5 if method in {"rtt_papo_response", "rdan_scalar"} else None
    hybrid = method in {"rtt_papo_response", "rl_csr", "rl_aon"}
    data_path = "data/qwen_hir_rubrichub_if_hybrid.jsonl" if hybrid else "data/qwen_hir_rubrichub_if_rl_eligible.jsonl"
    reward_worker = (
        "rdan_grpo.roll_reward.RTTCompatibleRubricRewardWorker"
        if hybrid
        else "rdan_grpo.roll_reward.ScalarRubricRewardWorker"
    )
    return {
        "async_pipeline": False,
        "async_generation_ratio": 0,
        "generate_opt_level": 0,
        "enable_reference": False,
        "enable_old_logprobs_recompute": True,
        "rollout_batch_size": 64,
        "num_return_sequences_in_group": 8,
        "max_steps": 500,
        "save_steps": 20,
        "actor_train": {
            **_worker("ResponseActorWorker", "fsdp2_train"),
            "data_args": {"file_name": [data_path]},
            "training_args": {
                "learning_rate": 1e-6,
                "weight_decay": 0,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 32,
                "warmup_steps": 20,
                "num_train_epochs": 50,
            },
        },
        "actor_infer": {**_worker("ResponseInferWorker", "hf_infer"), "max_concurrency": 1},
        "rewards": {
            "llm_judge": {
                "worker_cls": reward_worker,
                "device_mapping": None,
                "judge_model_type": "api",
            }
        },
        "rdan_response": {"method": method, "quality_weight": quality, "mix_weight": None},
    }


def _worker(name: str, strategy: str) -> dict[str, Any]:
    return {
        "worker_cls": f"rdan_grpo.roll_response_workers.{name}",
        "device_mapping": [0, 1],
        "world_size": 2,
        "num_gpus_per_worker": 1,
        "strategy_args": {"strategy_name": strategy, "strategy_config": {"transformer_impl": "huggingface"}},
        "model_args": {"dtype": "bf16", "attn_implementation": "sdpa"},
    }


@dataclass
class FakeConfig:
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("actor_train", "actor_infer"):
            worker = self.payload[name]
            mappings = {"[0, 1]": [0, 1], "[]": []}
            value = worker["device_mapping"]
            device_mapping = mappings.get(value, value) if isinstance(value, str) else value
            setattr(
                self,
                name,
                SimpleNamespace(
                    **{
                        key: value
                        for key, value in worker.items()
                        if key not in {"strategy_args", "device_mapping", "data_args"}
                    },
                    device_mapping=device_mapping,
                    strategy_args=SimpleNamespace(**worker["strategy_args"]),
                    data_args=SimpleNamespace(**worker["data_args"]) if "data_args" in worker else None,
                ),
            )
        self.rewards = self.payload["rewards"]
        self.max_steps = self.payload["max_steps"]
        self.track_with = self.payload.get("track_with")
        if "validation" in self.payload:
            validation = self.payload["validation"]
            self.validation = SimpleNamespace(
                generating_args=SimpleNamespace(**validation["generating_args"]),
                data_args=SimpleNamespace(**validation["data_args"]),
            )

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def test_production_constructor_uses_only_minimal_hf_surrogate(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    payload = _payload()
    original = copy.deepcopy(payload)
    observed: dict[str, Any] = {}
    monkeypatch.setattr(module, "_verify_rtt", lambda root: observed.update(root=root))

    def construct(config_cls: type, surrogate: dict[str, Any]) -> FakeConfig:
        observed["surrogate"] = copy.deepcopy(surrogate)
        assert surrogate["actor_infer"]["strategy_args"]["strategy_name"] == "vllm"
        assert surrogate["rewards"] == payload["rewards"]
        return FakeConfig(surrogate)

    monkeypatch.setattr(module, "_construct", construct)
    config = module.load_response_rlvr_config("/pinned/rtt", FakeConfig, payload)

    assert payload == original
    assert config.actor_train.worker_cls == module.ACTOR_WORKER_PATH
    assert config.actor_infer.worker_cls == module.INFER_WORKER_PATH
    assert config.actor_infer.strategy_args.strategy_name == "hf_infer"
    assert config.rewards == payload["rewards"]
    assert config.rdan_response == module.ResponseConfig("rtt_papo_response", 0.5, None, "f" * 64)
    assert "rdan_response" not in observed["surrogate"]


def test_preflight_constructor_restores_zero_update_hf_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    payload = _payload()
    payload.update(max_steps=0, track_with="stdout")
    payload["actor_train"]["device_mapping"] = "[]"
    payload["validation"] = {
        "generating_args": {"num_return_sequences": 8},
        "data_args": {"file_name": ["data/qwen_hir_rubrichub_if_hybrid.jsonl"]},
    }
    original = copy.deepcopy(payload)
    monkeypatch.setattr(module, "_verify_rtt", lambda root: None)
    monkeypatch.setattr(module, "_construct", lambda config_cls, value: FakeConfig(value))

    config = module.load_response_preflight_config("/pinned/rtt", FakeConfig, payload)

    assert payload == original
    assert config.actor_train.device_mapping == []
    assert config.actor_infer.device_mapping == [0, 1]
    assert config.actor_infer.strategy_args.strategy_name == "hf_infer"
    assert config.max_steps == 0
    assert config.track_with == "stdout"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(rewards={}),
        lambda value: value["actor_train"].update(worker_cls="roll.pipeline.rlvr.actor_worker.ActorWorker"),
        lambda value: value["actor_infer"]["strategy_args"].update(strategy_name="vllm"),
        lambda value: value.update(enable_old_logprobs_recompute=False),
        lambda value: value["actor_train"]["training_args"].update(gradient_accumulation_steps=2),
        lambda value: value["actor_train"]["data_args"].update(
            file_name=["data/qwen_hir_rubrichub_if_rl_eligible.jsonl"]
        ),
        lambda value: value["rewards"]["llm_judge"].update(
            worker_cls="rdan_grpo.roll_reward.ScalarRubricRewardWorker"
        ),
    ],
)
def test_production_constructor_rejects_stock_or_weakened_profiles(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    module = _module(monkeypatch)
    payload = _payload()
    mutate(payload)
    monkeypatch.setattr(module, "_verify_rtt", lambda root: None)
    with pytest.raises(ValueError, match="response config"):
        module.load_response_rlvr_config("/pinned/rtt", FakeConfig, payload)


@pytest.mark.parametrize(
    ("path", "method", "quality"),
    [
        ("qwen_rtt_papo_response_train.yaml", "rtt_papo_response", 0.5),
        ("qwen_rl_csr_train.yaml", "rl_csr", None),
        ("qwen_rl_aon_train.yaml", "rl_aon", None),
    ],
)
def test_method_train_yaml_uses_exact_objective_sidecar(path: str, method: str, quality: float | None) -> None:
    payload = _compose_method_yaml(path)
    assert payload["actor_train"]["worker_cls"].endswith("ResponseActorWorker")
    assert payload["actor_infer"]["worker_cls"].endswith("ResponseInferWorker")
    assert payload["actor_train"]["data_args"]["file_name"] == ["data/qwen_hir_rubrichub_if_hybrid.jsonl"]
    assert payload["actor_train"]["training_args"] == {
        "learning_rate": 1e-6,
        "weight_decay": 0,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 32,
        "warmup_steps": 20,
        "num_train_epochs": 50,
    }
    assert payload["rewards"]["llm_judge"]["worker_cls"].endswith("RTTCompatibleRubricRewardWorker")
    assert payload["rewards"]["llm_judge"]["judge_model_type"] == "api"
    assert payload["rdan_response"] == {"method": method, "quality_weight": quality, "mix_weight": None}
    assert "api_key" not in payload["rewards"]["llm_judge"]


@pytest.mark.parametrize(
    "path",
    [
        "qwen_rtt_papo_response_preflight.yaml",
        "qwen_rl_csr_preflight.yaml",
        "qwen_rl_aon_preflight.yaml",
    ],
)
def test_method_preflight_yaml_is_zero_update_over_hybrid_data(path: str) -> None:
    payload = _compose_method_yaml(path)
    assert payload["max_steps"] == 0
    assert payload["track_with"] == "stdout"
    assert payload["actor_train"]["device_mapping"] == "[]"
    assert payload["validation"]["generating_args"]["num_return_sequences"] == 8
    assert payload["validation"]["data_args"]["file_name"] == ["data/qwen_hir_rubrichub_if_hybrid.jsonl"]


@pytest.mark.parametrize("suffix", ["train", "preflight"])
def test_method_configs_differ_only_by_objective_and_run_names(suffix: str) -> None:
    payloads = [
        _compose_method_yaml(f"qwen_{method}_{suffix}.yaml") for method in ("rtt_papo_response", "rl_csr", "rl_aon")
    ]
    normalized = [_without_method_fields(payload) for payload in payloads]
    assert normalized[1:] == normalized[:1] * 2


@pytest.mark.parametrize("method", ["rtt_papo_response", "rl_csr", "rl_aon"])
@pytest.mark.parametrize("mismatch", ["data", "worker"])
def test_method_binding_rejects_cross_wiring(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    mismatch: str,
) -> None:
    module = _module(monkeypatch)
    monkeypatch.setattr(module, "_verify_rtt", lambda root: None)
    payload = _payload(method)
    if mismatch == "data":
        payload["actor_train"]["data_args"]["file_name"] = ["data/qwen_hir_rubrichub_if_rl_eligible.jsonl"]
    else:
        payload["rewards"]["llm_judge"]["worker_cls"] = "rdan_grpo.roll_reward.ScalarRubricRewardWorker"
    with pytest.raises(ValueError, match="requires"):
        module.load_response_rlvr_config("/pinned/rtt", FakeConfig, payload)


def _compose_method_yaml(path: str) -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / "configs/roll" / path).read_text())
    defaults = payload.pop("defaults", [])
    result: dict[str, Any] = {}
    for item in defaults:
        if item == "_self_":
            continue
        parent = item if isinstance(item, str) else next(iter(item.values()))
        result = _merge(result, _compose_method_yaml(f"{parent}.yaml"))
    return _merge(result, payload)


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _without_method_fields(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    for key in ("checkpoint_config", "exp_name", "logging_dir", "output_dir", "rdan_response", "rollout_dump_dir"):
        result.pop(key, None)
    return result
