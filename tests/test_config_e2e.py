"""End-to-end proof that the shipped method configs compose and construct."""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from rdan_grpo import config as config_module

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/roll"
RTT_ROOT = Path(__file__).resolve().parents[2] / "Rubrics-To-Tokens"
MODEL_SNAPSHOT = "/pinned/qwen3-4b"
METHOD_CONFIGS = {
    "rtt_papo_response": ("qwen_rtt_papo_response_train.yaml", 1.0),
    "rl_csr": ("qwen_rl_csr_train.yaml", None),
    "rl_aon": ("qwen_rl_aon_train.yaml", None),
}
RUN_IDENTITY_KEYS = frozenset({"exp_name", "logging_dir", "output_dir", "rollout_dump_dir", "rdan_response"})
CROSS_WIRINGS = {
    "reward-worker": lambda payload: payload["rewards"]["llm_judge"].update(
        worker_cls=config_module.SCALAR_REWARD_WORKER_PATH
    ),
    "dataset": lambda payload: payload["actor_train"]["data_args"].update(file_name=[config_module.SCALAR_DATA_PATH]),
    "objective": lambda payload: payload["rdan_response"].update(method="rdan_scalar", quality_weight=0.5),
}


@pytest.mark.parametrize("method", METHOD_CONFIGS)
def test_shipped_method_config_builds_its_objective(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    path, quality = METHOD_CONFIGS[method]

    config = _build(monkeypatch, _compose(path))

    assert config.rdan_response == config_module.ResponseConfig(
        method=method,
        quality_weight=quality,
        mix_weight=None,
        resolved_config_sha256=config_module.canonical_config_sha256(config.to_dict()),
    )
    assert config.actor_train.worker_cls == config_module.ACTOR_WORKER_PATH
    assert config.actor_infer.worker_cls == config_module.INFER_WORKER_PATH
    assert config.actor_infer.strategy_args.strategy_name == "vllm"
    assert config.rewards["llm_judge"]["worker_cls"] == config_module.HYBRID_REWARD_WORKER_PATH
    assert config.actor_train.data_args.file_name == [config_module.HYBRID_DATA_PATH]


def test_method_configs_differ_only_by_objective_and_run_naming() -> None:
    payloads = {method: _compose(path) for method, (path, _) in METHOD_CONFIGS.items()}

    shared = [
        {key: value for key, value in payload.items() if key not in RUN_IDENTITY_KEYS} for payload in payloads.values()
    ]
    assert shared[1:] == shared[:1] * (len(payloads) - 1)
    for method, payload in payloads.items():
        run_name = payload["exp_name"]
        assert payload["rdan_response"]["method"] == method
        assert method.replace("_", "-") in run_name
        assert all(run_name in payload[key] for key in ("logging_dir", "output_dir", "rollout_dump_dir"))
    assert len({payload["exp_name"] for payload in payloads.values()}) == len(payloads)


@pytest.mark.parametrize("cross_wiring", CROSS_WIRINGS)
@pytest.mark.parametrize("method", METHOD_CONFIGS)
def test_cross_wired_method_is_rejected(monkeypatch: pytest.MonkeyPatch, method: str, cross_wiring: str) -> None:
    payload = _compose(METHOD_CONFIGS[method][0])
    CROSS_WIRINGS[cross_wiring](payload)

    # Match the method-binding message so an unrelated profile error cannot satisfy the expectation.
    with pytest.raises(ValueError, match=r"response config method \w+ requires"):
        _build(monkeypatch, payload)


def _build(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> Any:
    """Run the production constructor against a stand-in for the fork's RLVRConfig."""

    monkeypatch.setattr(config_module, "_construct", lambda config_cls, data: config_cls(data))
    original = copy.deepcopy(payload)
    config = config_module.load_response_rlvr_config(RTT_ROOT, _ConstructedConfig, payload)
    assert payload == original
    return config


class _ConstructedConfig:
    """Attribute view of a constructed RLVR config, matching how the fork parses device mappings."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        for name in ("actor_train", "actor_infer"):
            worker = dict(payload[name])
            worker["device_mapping"] = ast.literal_eval(worker["device_mapping"])
            setattr(self, name, _namespace(worker))
        self.rewards = payload["rewards"]

    def to_dict(self) -> dict[str, Any]:
        """Return the payload the fork would serialize into the run receipt."""

        return self._payload


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


def _compose(path: str) -> dict[str, Any]:
    """Resolve one method config the way scripts/train.py composes it through hydra."""

    os.environ.setdefault("RDAN_MODEL_SNAPSHOT", MODEL_SNAPSHOT)
    payload = _merge_defaults(path)
    payload.pop("hydra", None)
    return _resolve(payload, payload)


def _merge_defaults(path: str) -> dict[str, Any]:
    payload = yaml.safe_load((CONFIG_DIR / path).read_text())
    result: dict[str, Any] = {}
    for item in payload.pop("defaults", []):
        if item == "_self_":
            continue
        parent = item if isinstance(item, str) else next(iter(item.values()))
        result = _merge(result, _merge_defaults(f"{parent}.yaml"))
    return _merge(result, payload)


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        nested = isinstance(value, dict) and isinstance(result.get(key), dict)
        result[key] = _merge(result[key], value) if nested else copy.deepcopy(value)
    return result


def _resolve(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, root) for item in value]
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        return value
    reference = value[2:-1]
    if reference.startswith("oc.env:"):
        return os.environ[reference.split(":", 1)[1]]
    return _resolve(root[reference], root)
