"""Construct the ROLL RLVR config from a Hydra payload plus the RDAN sidecar."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from rdan_grpo.scalar import METHODS, QUALITY_METHODS

ACTOR_WORKER_PATH = "rdan_grpo.workers.ResponseActorWorker"
INFER_WORKER_PATH = "rdan_grpo.workers.ResponseVLLMInferWorker"
REWARD_WORKER_PATH = "rdan_grpo.reward_worker.RubricRewardWorker"
SIDECAR_KEY = "rdan_response"


@dataclass(frozen=True)
class ResponseConfig:
    """RDAN-specific fields that sit outside ROLL's own config schema."""

    method: str
    quality_weight: float | None


def load_config(config_cls: type, payload: Mapping[str, Any]) -> Any:
    """Build the ROLL config object and attach the validated RDAN sidecar."""

    production = deepcopy(payload)
    response = _response_config(production)
    _drop_null_reward_device_mapping(production)
    _validate_workers(production)
    for name in ("actor_train", "actor_infer"):
        production[name]["device_mapping"] = _device_mapping(production[name]["device_mapping"])

    from dacite import from_dict

    config = from_dict(data_class=config_cls, data=production)
    config.rdan_response = response
    return config


def updates_per_step(config: Any) -> int:
    """Optimizer updates one rollout batch produces, given the actor's global batch."""

    training = config.actor_train.training_args
    global_batch = training.per_device_train_batch_size * training.gradient_accumulation_steps
    global_batch *= config.actor_train.world_size
    responses = config.rollout_batch_size * config.num_return_sequences_in_group
    if global_batch <= 0 or responses % global_batch:
        raise ValueError(
            f"rollout responses ({responses}) must divide the actor global batch ({global_batch}); "
            "adjust gradient_accumulation_steps or rollout_batch_size"
        )
    return responses // global_batch


def _response_config(payload: dict[str, Any]) -> ResponseConfig:
    sidecar = payload.pop(SIDECAR_KEY, None)
    if not isinstance(sidecar, Mapping) or "method" not in sidecar:
        raise ValueError(f"config requires a {SIDECAR_KEY} sidecar with a method")
    method = str(sidecar["method"])
    weight = sidecar.get("quality_weight")
    if method not in METHODS:
        raise ValueError(f"unsupported method {method}; expected one of {sorted(METHODS)}")
    if method in QUALITY_METHODS:
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise ValueError("quality_weight must be a non-negative number for rdan")
    elif weight is not None:
        raise ValueError(f"quality_weight is not valid for {method}")
    return ResponseConfig(method=method, quality_weight=None if weight is None else float(weight))


def _validate_workers(payload: Mapping[str, Any]) -> None:
    """Check only what a wrong value would silently corrupt rather than crash."""

    for name, expected in (("actor_train", ACTOR_WORKER_PATH), ("actor_infer", INFER_WORKER_PATH)):
        worker = payload.get(name)
        if not isinstance(worker, Mapping) or worker.get("worker_cls") != expected:
            raise ValueError(f"config requires {name}.worker_cls={expected}")
        if len(_device_list(worker.get("device_mapping"))) != worker.get("world_size"):
            raise ValueError(f"{name}.device_mapping must cover exactly world_size devices")
    rewards = payload.get("rewards")
    if not isinstance(rewards, Mapping) or not rewards:
        raise ValueError("config requires at least one reward worker")
    for name, reward in rewards.items():
        if reward.get("worker_cls") != REWARD_WORKER_PATH:
            raise ValueError(f"reward {name} requires worker_cls={REWARD_WORKER_PATH}")


def _device_list(value: Any) -> list[int]:
    """Read a device mapping written either as a GPU count or as an explicit device list."""

    if isinstance(value, int) and not isinstance(value, bool):
        return list(range(value))
    if isinstance(value, (list, tuple)):
        return [int(device) for device in value]
    raise ValueError(f"device_mapping must be a GPU count or a device list, received {value!r}")


def _device_mapping(value: Any) -> str:
    """ROLL parses device_mapping from a Python expression string, not a list."""

    return repr(_device_list(value))


def _drop_null_reward_device_mapping(payload: dict[str, Any]) -> None:
    """ROLL types device_mapping without Optional, so an explicit null fails construction.

    Removing the key selects ROLL's own CPU-only default, which is what the judge workers need.
    """

    for worker in payload.get("rewards", {}).values():
        if isinstance(worker, dict) and worker.get("device_mapping", False) is None:
            worker.pop("device_mapping")
