"""Exact production config construction for response-only RTT training."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rdan_grpo.roll_compat import RTT_BASE_CONFIG_SHA256, RTT_REVISION
from rdan_grpo.wandb_tracking import canonical_config_sha256

ACTOR_WORKER_PATH = "rdan_grpo.roll_response_workers.ResponseActorWorker"
INFER_WORKER_PATH = "rdan_grpo.roll_response_workers.ResponseVLLMInferWorker"
HF_INFER_WORKER_PATH = "rdan_grpo.roll_response_workers.ResponseInferWorker"
VLLM_RECEIPT_WORKER_PATH = "rdan_grpo.roll_weight_receipt.ReceiptWorkerV1"
HYBRID_REWARD_WORKER_PATH = "rdan_grpo.roll_reward.RTTCompatibleRubricRewardWorker"
SCALAR_REWARD_WORKER_PATH = "rdan_grpo.roll_reward.ScalarRubricRewardWorker"
HYBRID_DATA_PATH = "data/qwen_hir_rubrichub_if_hybrid.jsonl"
SCALAR_DATA_PATH = "data/qwen_hir_rubrichub_if_rl_eligible.jsonl"
SIDECAR_KEY = "rdan_response"
HYBRID_METHODS = frozenset({"rtt_papo_response", "rl_csr", "rl_aon"})
METHODS = HYBRID_METHODS | {"rdan_scalar", "rl_mix"}
UPDATES_PER_STEP = 2
FROZEN_PROFILE = {
    "num_gpus_per_node": 2,
    "prompt_length": 2048,
    "response_length": 4096,
    "rollout_batch_size": 64,
    "num_return_sequences_in_group": 8,
    "ppo_epochs": 1,
    "pg_clip": 0.2,
    "pg_clip_low": 0.2,
    "pg_clip_high": 0.27,
    "use_pg_clip_range": True,
    "importance_sampling": "token",
    "init_kl_coef": 0,
    "enable_reference": False,
    "add_token_level_kl": False,
    "save_steps": 20,
}
TRAINING_ARGS = {
    "learning_rate": 1.0e-6,
    "weight_decay": 0,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 128,
    "warmup_steps": 20,
    "num_train_epochs": 50,
}
GENERATION_ARGS = {
    "do_sample": True,
    "max_new_tokens": 4096,
    "num_beams": 1,
    "num_return_sequences": 8,
    "temperature": 0.99,
    "top_k": 100,
    "top_p": 0.99,
}
VLLM_STRATEGY_CONFIG = {
    "gpu_memory_utilization": 0.8,
    "block_size": 16,
    "max_model_len": 8000,
    "load_format": "auto",
    "sleep_level": 1,
    "worker_extension_cls": VLLM_RECEIPT_WORKER_PATH,
}


@dataclass(frozen=True)
class ResponseConfig:
    """Production-only fields intentionally kept outside RTT's RLVRConfig."""

    method: str
    quality_weight: float | None
    mix_weight: float | None
    resolved_config_sha256: str

    @property
    def fixed_weight(self) -> float:
        """Return the scalar recorded by the current receipt schema."""

        return float(self.quality_weight or self.mix_weight or 0.0)


def load_response_rlvr_config(rtt_root: str | Path, config_cls: type, payload: Mapping[str, Any]) -> Any:
    """Construct the pinned production config through RTT's minimal vLLM surrogate."""

    original = deepcopy(payload)
    production = deepcopy(payload)
    response = _response_config(production)
    _validate_payload(production, response)
    _verify_rtt(Path(rtt_root).resolve())

    production["actor_train"]["device_mapping"] = repr([0, 1])
    production["actor_infer"]["device_mapping"] = repr([0, 1])
    _drop_cpu_device_mapping(production)
    config = _construct(config_cls, production)
    response = replace(response, resolved_config_sha256=canonical_config_sha256(config.to_dict()))
    config.rdan_response = response
    _validate_config(config)
    if payload != original:
        raise RuntimeError("response config construction mutated the caller payload")
    return config


def load_response_preflight_config(rtt_root: str | Path, config_cls: type, payload: Mapping[str, Any]) -> Any:
    """Construct the no-update HF rollout config through RTT's vLLM parser boundary."""

    original = deepcopy(payload)
    preflight = deepcopy(payload)
    response = _response_config(preflight)
    _validate_preflight_payload(preflight, response)
    _verify_rtt(Path(rtt_root).resolve())

    surrogate = deepcopy(preflight)
    surrogate["actor_infer"]["device_mapping"] = repr([0, 1])
    surrogate["actor_infer"]["strategy_args"]["strategy_name"] = "vllm"
    surrogate["actor_infer"]["strategy_args"]["strategy_config"] = deepcopy(VLLM_STRATEGY_CONFIG)
    _drop_cpu_device_mapping(surrogate)
    config = _construct(config_cls, surrogate)
    config.actor_infer.strategy_args.strategy_name = "hf_infer"
    config.actor_infer.strategy_args.strategy_config = {"transformer_impl": "huggingface", "max_model_len": 8000}
    config.actor_infer.max_concurrency = 1
    response = replace(response, resolved_config_sha256=canonical_config_sha256(config.to_dict()))
    config.rdan_response = response
    _validate_preflight_config(config)
    if payload != original:
        raise RuntimeError("response preflight config construction mutated the caller payload")
    return config


def _response_config(payload: dict[str, Any]) -> ResponseConfig:
    sidecar = payload.pop(SIDECAR_KEY, None)
    if not isinstance(sidecar, Mapping) or set(sidecar) != {"method", "quality_weight", "mix_weight"}:
        raise ValueError("response config requires an exact rdan_response sidecar")
    method = sidecar.get("method")
    quality = sidecar.get("quality_weight")
    mix = sidecar.get("mix_weight")
    if method not in METHODS:
        raise ValueError("response config method is unsupported")
    if method in ("rdan_scalar", "rtt_papo_response"):
        _weight(quality, "quality_weight", required=True)
    elif quality is not None:
        raise ValueError(f"quality_weight is not valid for {method}")
    if method == "rl_mix":
        _weight(mix, "mix_weight", required=True)
    elif mix is not None:
        raise ValueError(f"mix_weight is not valid for {method}")
    return ResponseConfig(
        method=str(method),
        quality_weight=None if quality is None else float(quality),
        mix_weight=None if mix is None else float(mix),
        resolved_config_sha256="0" * 64,
    )


def _weight(value: Any, name: str, *, required: bool) -> None:
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError(f"response config {name} must be in [0, 1]")


def _validate_payload(payload: Mapping[str, Any], response: ResponseConfig) -> None:
    actor = _worker(payload, "actor_train", ACTOR_WORKER_PATH, "fsdp2_train")
    infer = _worker(payload, "actor_infer", INFER_WORKER_PATH, "vllm")
    if actor.get("device_mapping") != [0, 1] or infer.get("device_mapping") != [0, 1]:
        raise ValueError("response config requires colocated device mappings [0, 1]")
    if any(worker.get("world_size") != 2 or worker.get("num_gpus_per_worker") != 1 for worker in (actor, infer)):
        raise ValueError("response config requires actor and inference DP2")
    rewards = payload.get("rewards")
    if not isinstance(rewards, Mapping) or set(rewards) != {"llm_judge"}:
        raise ValueError("response config requires the production llm_judge reward")
    reward = rewards["llm_judge"]
    expected_worker, expected_data = _method_profile(response.method)
    if not isinstance(reward, Mapping) or reward.get("worker_cls") != expected_worker:
        raise ValueError(f"response config method {response.method} requires {expected_worker}")
    if _data_files(actor) != [expected_data]:
        raise ValueError(f"response config method {response.method} requires data {expected_data}")
    if reward.get("device_mapping") is not None or reward.get("judge_model_type") != "api":
        raise ValueError("response reward workers must be CPU/API only")
    required = {
        "async_pipeline": False,
        "async_generation_ratio": 0,
        "generate_opt_level": 0,
        "enable_old_logprobs_recompute": True,
    }
    if any(payload.get(name) != value for name, value in required.items()):
        raise ValueError("response config differs from the frozen synchronous 500-step profile")
    if infer.get("max_concurrency") != 32:
        raise ValueError("response config requires actor_infer.max_concurrency=32")
    _validate_recipe(payload, actor, infer, max_steps=500)


def _validate_preflight_payload(payload: Mapping[str, Any], response: ResponseConfig) -> None:
    actor = _worker(payload, "actor_train", ACTOR_WORKER_PATH, "fsdp2_train")
    infer = _worker(payload, "actor_infer", HF_INFER_WORKER_PATH, "hf_infer")
    if actor.get("device_mapping") != "[]" or infer.get("device_mapping") != [0, 1]:
        raise ValueError("response preflight requires no actor devices and HF inference on [0, 1]")
    if infer.get("world_size") != 2 or infer.get("num_gpus_per_worker") != 1:
        raise ValueError("response preflight requires inference DP2")
    if payload.get("max_steps") != 0 or payload.get("track_with") != "stdout":
        raise ValueError("response preflight requires zero updates and stdout tracking")
    _validate_recipe(payload, actor, infer, max_steps=0)
    rewards = payload.get("rewards")
    if not isinstance(rewards, Mapping) or set(rewards) != {"llm_judge"}:
        raise ValueError("response preflight requires the production llm_judge reward")
    reward = rewards["llm_judge"]
    expected_worker, expected_data = _method_profile(response.method)
    if not isinstance(reward, Mapping) or reward.get("worker_cls") != expected_worker:
        raise ValueError(f"response preflight method {response.method} requires {expected_worker}")
    if _data_files(actor) != [expected_data]:
        raise ValueError(f"response preflight method {response.method} requires data {expected_data}")
    validation = payload.get("validation")
    generating = validation.get("generating_args") if isinstance(validation, Mapping) else None
    data = validation.get("data_args") if isinstance(validation, Mapping) else None
    _validate_generation(generating, "response preflight validation")
    if (
        not isinstance(generating, Mapping)
        or generating.get("num_return_sequences") != 8
        or not isinstance(data, Mapping)
        or data.get("file_name") != [expected_data]
    ):
        raise ValueError(f"response preflight requires eight responses over data {expected_data}")


def _validate_recipe(
    payload: Mapping[str, Any], actor: Mapping[str, Any], infer: Mapping[str, Any], *, max_steps: int
) -> None:
    expected = {**FROZEN_PROFILE, "max_steps": max_steps}
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("response config differs from the frozen RL recipe")
    if actor.get("training_args") != TRAINING_ARGS:
        raise ValueError("response config differs from the RTT global optimizer batch recipe")
    strategy = infer.get("strategy_args")
    strategy_config = strategy.get("strategy_config") if isinstance(strategy, Mapping) else None
    if not isinstance(strategy_config, Mapping) or strategy_config.get("max_model_len") != 8000:
        raise ValueError("response config differs from the frozen inference topology")
    if strategy.get("strategy_name") == "vllm" and dict(strategy_config) != VLLM_STRATEGY_CONFIG:
        raise ValueError("response config differs from the frozen vLLM topology")
    _validate_generation(infer.get("generating_args"), "response config")
    responses = payload["rollout_batch_size"] * payload["num_return_sequences_in_group"]
    optimizer_batch = TRAINING_ARGS["per_device_train_batch_size"] * TRAINING_ARGS["gradient_accumulation_steps"] * 2
    if responses != optimizer_batch * UPDATES_PER_STEP:
        raise ValueError("response config differs from the frozen optimizer update cadence")


def _validate_generation(value: Any, name: str) -> None:
    expected = GENERATION_ARGS
    observed = value if isinstance(value, Mapping) else {key: getattr(value, key, None) for key in expected}
    if any(observed.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError(f"{name} differs from the frozen RTT generation recipe")


def _method_profile(method: str) -> tuple[str, str]:
    if method in HYBRID_METHODS:
        return HYBRID_REWARD_WORKER_PATH, HYBRID_DATA_PATH
    return SCALAR_REWARD_WORKER_PATH, SCALAR_DATA_PATH


def _data_files(worker: Mapping[str, Any]) -> Any:
    data = worker.get("data_args")
    return data.get("file_name") if isinstance(data, Mapping) else None


def _worker(payload: Mapping[str, Any], name: str, worker_cls: str, strategy: str) -> Mapping[str, Any]:
    worker = payload.get(name)
    if not isinstance(worker, Mapping) or worker.get("worker_cls") != worker_cls:
        raise ValueError(f"response config requires {name}.worker_cls={worker_cls}")
    strategy_args = worker.get("strategy_args")
    if not isinstance(strategy_args, Mapping) or strategy_args.get("strategy_name") != strategy:
        raise ValueError(f"response config requires {name} strategy {strategy}")
    strategy_config = strategy_args.get("strategy_config")
    if not isinstance(strategy_config, Mapping):
        raise ValueError(f"response config requires {name}.strategy_args.strategy_config")
    if strategy != "vllm" and strategy_config.get("transformer_impl") != "huggingface":
        raise ValueError(f"response config requires {name} Hugging Face implementation")
    if strategy == "vllm" and strategy_config.get("transformer_impl") is not None:
        raise ValueError(f"response config requires {name} native vLLM implementation")
    model_args = worker.get("model_args")
    if (
        not isinstance(model_args, Mapping)
        or model_args.get("dtype") != "bf16"
        or model_args.get("attn_implementation") != "sdpa"
    ):
        raise ValueError(f"response config requires {name} bf16 SDPA")
    return worker


def _drop_cpu_device_mapping(payload: dict[str, Any]) -> None:
    """Omit a null reward device mapping so RTT applies its cpu-only default.

    RTT documents None as cpu-only but types the field without Optional, so an explicit
    null fails construction while an absent key uses the default.
    """

    for worker in payload.get("rewards", {}).values():
        if isinstance(worker, dict) and worker.get("device_mapping", False) is None:
            worker.pop("device_mapping")


def _construct(config_cls: type, payload: Mapping[str, Any]) -> Any:
    from dacite import from_dict

    return from_dict(data_class=config_cls, data=payload)


def _verify_rtt(root: Path) -> None:
    import subprocess

    top_level = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    base_config = root / "roll/configs/base_config.py"
    digest = hashlib.sha256(base_config.read_bytes()).hexdigest()
    if Path(top_level).resolve() != root or revision != RTT_REVISION or status or digest != RTT_BASE_CONFIG_SHA256:
        raise RuntimeError("response config requires the exact clean pinned RTT checkout")


def _validate_config(config: Any) -> None:
    for name, worker_cls, strategy in (
        ("actor_train", ACTOR_WORKER_PATH, "fsdp2_train"),
        ("actor_infer", INFER_WORKER_PATH, "vllm"),
    ):
        worker = getattr(config, name)
        if (
            worker.worker_cls != worker_cls
            or worker.strategy_args.strategy_name != strategy
            or worker.device_mapping != [0, 1]
            or worker.world_size != 2
        ):
            raise RuntimeError(f"constructed response config changed {name}")
    rewards = getattr(config, "rewards", None)
    if not isinstance(rewards, Mapping) or set(rewards) != {"llm_judge"}:
        raise RuntimeError("constructed response config changed production rewards")
    expected_worker, expected_data = _method_profile(config.rdan_response.method)
    if rewards["llm_judge"].get("worker_cls") != expected_worker:
        raise RuntimeError("constructed response config changed the method reward worker")
    if config.actor_train.data_args.file_name != [expected_data]:
        raise RuntimeError("constructed response config changed the method dataset")
    _validate_generation(config.actor_infer.generating_args, "constructed response config")


def _validate_preflight_config(config: Any) -> None:
    if (
        config.actor_train.worker_cls != ACTOR_WORKER_PATH
        or config.actor_train.strategy_args.strategy_name != "fsdp2_train"
        or config.actor_train.device_mapping != []
        or config.actor_infer.worker_cls != HF_INFER_WORKER_PATH
        or config.actor_infer.strategy_args.strategy_name != "hf_infer"
        or config.actor_infer.device_mapping != [0, 1]
        or config.actor_infer.world_size != 2
        or config.actor_infer.max_concurrency != 1
    ):
        raise RuntimeError("constructed response preflight changed worker topology")
    if config.max_steps != 0 or config.track_with != "stdout":
        raise RuntimeError("constructed response preflight can perform updates or external tracking")
    if config.validation.generating_args.num_return_sequences != 8:
        raise RuntimeError("constructed response preflight changed group size")
    _validate_generation(config.actor_infer.generating_args, "constructed response preflight actor inference")
    _validate_generation(config.validation.generating_args, "constructed response preflight validation")
    expected_worker, expected_data = _method_profile(config.rdan_response.method)
    if config.rewards["llm_judge"].get("worker_cls") != expected_worker:
        raise RuntimeError("constructed response preflight changed the method reward worker")
    if config.actor_train.data_args.file_name != [expected_data] or config.validation.data_args.file_name != [
        expected_data
    ]:
        raise RuntimeError("constructed response preflight changed the method dataset")
