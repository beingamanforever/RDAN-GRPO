"""End-to-end response training from the production config to a promoted checkpoint.

These tests run the real config loader, the real ``ResponseTrainingPipeline``, the real
optimizer transaction in ``train_step``, the real ORM/PRM advantage math, the real actor
worker optimizer accounting, and the real checkpoint promotion and resume path. Only the
boundaries that need a GPU, a Ray cluster, a model, or the network are substituted: the
Ray ``Cluster`` and scheduler actors, the tokenizer, the dataset load, and the rollout
engine itself.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib.util
import os
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch
import yaml

from rdan_grpo.checkpoint import ArtifactIdentity, CheckpointIdentity, load_checkpoint
from rdan_grpo.config import ACTOR_WORKER_PATH, INFER_WORKER_PATH, UPDATES_PER_STEP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CONFIG_DIR = ROOT / "configs/roll"
CONFIG_NAME = "qwen_rtt_papo_response_train"
MODEL_SNAPSHOT = "/snapshots/qwen3-4b-instruct-2507"
DOMAIN = "llm_judge"

WORLD_SIZE = 2
GROUP_SIZE = 8
PROMPT_COUNT = 4
RESPONSE_COUNT = PROMPT_COUNT * GROUP_SIZE
SEQUENCE_LENGTH = 6
TOKEN_COUNT = SEQUENCE_LENGTH - 1
LOG_PROB = -0.25
TOTAL_BYTES = 80 * 1024**3
PEAK_BYTES = 48 * 1024**3

# RTT's RLVRConfig supplies these; the production yaml never spells them out.
RUNTIME_DEFAULTS = {"rpc_timeout": 3600, "reward_system_config": None, "tag_2_domain": {DOMAIN: DOMAIN}}
MODEL_UPDATE_FREQUENCY = 1


# --------------------------------------------------------------------------------------
# Framework surrogates
# --------------------------------------------------------------------------------------


class FakeDataProto:
    """Stand-in for RTT's ``DataProto`` transport object."""

    def __init__(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        non_tensor_batch: dict[str, Any] | None = None,
        meta_info: dict[str, Any] | None = None,
    ) -> None:
        self.batch = {} if batch is None else batch
        self.non_tensor_batch = {} if non_tensor_batch is None else non_tensor_batch
        self.meta_info = {} if meta_info is None else meta_info

    def __len__(self) -> int:
        if self.batch:
            return int(next(iter(self.batch.values())).shape[0])
        return len(next(iter(self.non_tensor_batch.values())))

    def clone(self) -> FakeDataProto:
        """Copy the batch the way RTT clones a proto before a worker call."""

        return FakeDataProto(
            {name: value.clone() for name, value in self.batch.items()},
            {name: copy.copy(value) for name, value in self.non_tensor_batch.items()},
            dict(self.meta_info),
        )


class FakeLrScheduler:
    """Minimal learning-rate scheduler the real optimizer accounting can wrap."""

    def __init__(self) -> None:
        self.calls = 0

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Advance the scheduler exactly like RTT's own scheduler call."""

        self.calls += 1


class FakeFsdpStrategy:
    """Rank-local FSDP2 strategy surrogate with a real optimizer and DCP load semantics."""

    def __init__(self, rank: int) -> None:
        torch.manual_seed(1000 + rank)
        self.rank = rank
        self.model = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.0e-6)
        self.scheduler = FakeLrScheduler()
        self.checkpoint_manager = SimpleNamespace()

    def op_compute_log_probs(
        self, *, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: Any
    ) -> torch.Tensor:
        """Return the per-token log-probabilities the actor loss compares against."""

        return torch.full((input_ids.shape[0], input_ids.shape[1] - 1), LOG_PROB)

    def save_checkpoint(self, *, save_dir: str, global_step: int, ckpt_id: str, is_last_step: bool) -> None:
        """Write the rank-local distributed checkpoint shard."""

        target = Path(self._get_dcp_checkpoint_dir(save_dir))
        target.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        torch.save(payload, target / f"rank-{self.rank}.pt")

    def _get_dcp_checkpoint_dir(self, save_dir: str) -> str:
        return str(Path(save_dir) / "dcp")

    def load_states(self) -> None:
        """Bring the rank-local shard back onto its device before a load."""

    def offload_states(self) -> None:
        """Release the rank-local shard after a load."""

    def _load_checkpoint_with_dcp(self, *, checkpoint_dir: str) -> None:
        """Restore a shard the way ``dcp.load`` does, including its silent-skip behaviour.

        ``dcp.load`` builds its read plan from the live ``optimizer.state_dict()``. An
        optimizer that has never stepped exposes an empty ``state`` subtree, so there is no
        key to read the saved moments into and the moments are dropped without any error.
        """

        payload = torch.load(Path(checkpoint_dir) / f"rank-{self.rank}.pt", weights_only=False)
        self.model.load_state_dict(payload["model"])
        if not self.optimizer.state:
            return
        self.optimizer.load_state_dict(payload["optimizer"])


class FakeRltActorWorker:
    """Stand-in for RTT's ``ActorWorker``: drives the optimizer the real wrapper counts."""

    def loss_func(self, data: FakeDataProto, output_tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the base PPO loss and metrics the response worker extends."""

        return torch.tensor(0.5), {"actor/total_loss": 0.5}

    def train_step(self, data: FakeDataProto) -> FakeDataProto:
        """Run the configured optimizer updates for one pipeline step."""

        metrics: dict[str, Any] = {}
        logits = torch.zeros(len(data), SEQUENCE_LENGTH, 8)
        for _ in range(UPDATES_PER_STEP):
            for parameter in self.strategy.model.parameters():
                parameter.grad = torch.full_like(parameter, 0.01)
            _, metrics = self.loss_func(data, logits)
            self.strategy.optimizer.step()
            self.strategy.scheduler.step()
        return FakeDataProto(meta_info={"metrics": metrics})


class FakeVllmInferWorker:
    """Stand-in for RTT's ``InferWorker`` rollout base class."""

    async def generate(self, data: FakeDataProto) -> FakeDataProto:
        """Return one generated batch for the seeded request."""

        return FakeDataProto(
            {"input_ids": torch.ones(2, SEQUENCE_LENGTH, dtype=torch.long)}, meta_info=dict(data.meta_info)
        )


def _base_pipeline_cls(events: list[str]) -> type:
    """Build the ``BasePipeline`` surrogate that records the weight-transfer boundary."""

    class FakeBasePipeline:
        """Stand-in for RTT's ``BasePipeline`` runtime scaffolding."""

        model_update_groups: list[Any] = []
        checkpoint_clusters: list[Any] = []

        def __init__(self, pipeline_config: Any) -> None:
            self.pipeline_config = pipeline_config
            self.resource_manager = SimpleNamespace(name="resource-manager")
            self.state = SimpleNamespace(step=0, log_history=[])
            self.tracker = FakeTracker(events)
            self.model_update_pairs: list[tuple[str, str, int]] = []
            self.transfers: list[int] = []
            self.downloaded: list[Any] = []

        def download_models(self, *clusters: Any) -> None:
            """Record the clusters RTT would fetch model weights for."""

            self.downloaded.extend(clusters)

        def set_model_update_pair(self, *, src_cluster: Any, tgt_cluster: Any, frequency: int) -> None:
            """Bind the FSDP2 actor to the vLLM engine it pushes weights into."""

            self.model_update_pairs.append((src_cluster.role, tgt_cluster.role, frequency))

        def model_update(self, pipeline_step: int) -> None:
            """Push trained weights into the rollout engine."""

            self.transfers.append(pipeline_step)
            events.append(f"transfer:{pipeline_step}")

    return FakeBasePipeline


class FakeTracker:
    """Stand-in for the W&B tracker attached by ``BasePipeline``."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.logged: list[dict[str, Any]] = []
        self.finished = 0

    def log(self, *, values: Mapping[str, Any], step: int | None) -> None:
        """Record one metric payload exactly as the pipeline emits it."""

        assert step is None
        self.logged.append(dict(values))
        self.events.append("wandb-log")

    def finish(self) -> None:
        """Close the run."""

        self.finished += 1
        self.events.append("wandb-finish")


class Remote:
    """Stand-in for one Ray actor method handle, optionally recording an event."""

    def __init__(self, call: Callable[..., Any], events: list[str] | None = None, label: str = "") -> None:
        self._call = call
        self._events = events
        self._label = label

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the underlying method eagerly."""

        if self._events is not None and self._label:
            self._events.append(self._label)
        return self._call(*args, **kwargs)


# --------------------------------------------------------------------------------------
# Cluster and scheduler surrogates
# --------------------------------------------------------------------------------------


class FakeCluster:
    """Shared Ray cluster surrogate recording the offload lifecycle as ordered events."""

    def __init__(self, role: str, events: list[str]) -> None:
        self.role = role
        self.events = events

    def initialize(self, *, pipeline_config: Any, blocking: bool) -> list[Any]:
        """Start the cluster workers."""

        return []

    def load_states(self, *, blocking: bool) -> None:
        """Move this cluster onto its GPUs."""

        self.events.append(f"load:{self.role}")

    def offload_states(self, *, blocking: bool) -> None:
        """Move this cluster off its GPUs."""

        self.events.append(f"offload:{self.role}")

    def rdan_reset_cuda_peak(self, *, blocking: bool) -> None:
        """Reset the per-step peak memory statistics."""

        self.events.append(f"reset-peak:{self.role}")


class FakeActorTrainCluster(FakeCluster):
    """Stand-in for the FSDP2 actor cluster, driving real ``ResponseActorWorker`` ranks."""

    def __init__(self, runtime: Runtime, events: list[str], pipeline_config: Any) -> None:
        super().__init__("actor_train", events)
        self.dp_size = WORLD_SIZE
        self.worker_rank_info = [_rank_info() for _ in range(WORLD_SIZE)]
        self.workers = [_actor_worker(runtime, pipeline_config, rank) for rank in range(WORLD_SIZE)]
        self.train_calls = 0
        self.log_prob_calls = 0
        self.trained_batch: FakeDataProto | None = None

    def rdan_training_state(self, *, blocking: bool) -> list[dict[str, Any]]:
        """Return per-rank optimizer safety state from the real worker method."""

        self.events.append("training-state")
        return [worker.rdan_training_state() for worker in self.workers]

    def rdan_cuda_memory(self, *, blocking: bool) -> list[dict[str, int]]:
        """Return per-rank GPU memory after the transaction."""

        self.events.append("cuda-memory")
        return [{"rank": rank, "peak_bytes": PEAK_BYTES, "total_bytes": TOTAL_BYTES} for rank in range(WORLD_SIZE)]

    def compute_log_probs(self, data: FakeDataProto, *, blocking: bool) -> FakeDataProto:
        """Recompute explicit old log-probabilities for the rollout batch."""

        self.events.append("log-probs")
        self.log_prob_calls += 1
        return FakeDataProto({"log_probs": torch.full((len(data), TOKEN_COUNT), LOG_PROB)})

    def train_step(self, data: FakeDataProto, *, blocking: bool) -> FakeDataProto:
        """Run one real optimizer transaction on every rank."""

        self.events.append("train")
        self.train_calls += 1
        self.trained_batch = data
        outputs = [worker.train_step(data) for worker in self.workers]
        return outputs[0]

    def rdan_train_counters(self, *, blocking: bool) -> list[dict[str, int]]:
        """Return the exact per-rank optimizer and scheduler counters."""

        self.events.append("train-counters")
        return [worker.rdan_train_counters() for worker in self.workers]

    def rdan_save_dcp(self, checkpoint_dir: str, pipeline_step: int, *, blocking: bool) -> list[dict[str, Any]]:
        """Save every rank shard through the real worker save path."""

        self.events.append(f"save-dcp:{pipeline_step}")
        return [worker.rdan_save_dcp(checkpoint_dir, pipeline_step) for worker in self.workers]

    def rdan_load_dcp(self, checkpoint_dir: str, *, blocking: bool) -> list[dict[str, Any]]:
        """Restore every rank shard through the real worker load path."""

        self.events.append("load-dcp")
        return [worker.rdan_load_dcp(checkpoint_dir) for worker in self.workers]

    def optimizer_steps(self) -> list[int]:
        """Return the Adam step recorded inside each rank's live optimizer moments."""

        state = [worker.strategy.optimizer.state for worker in self.workers]
        return [int(next(iter(value.values()))["step"]) if value else 0 for value in state]


class FakeInferCluster(FakeCluster):
    """Stand-in for the vLLM rollout cluster, holding real ``ResponseVLLMInferWorker`` ranks."""

    def __init__(self, runtime: Runtime, events: list[str], pipeline_config: Any) -> None:
        super().__init__("actor_infer", events)
        self.dp_size = WORLD_SIZE
        self.worker_rank_info = [_rank_info() for _ in range(WORLD_SIZE)]
        self.ranks = [_infer_worker(runtime, pipeline_config, rank) for rank in range(WORLD_SIZE)]
        self.workers = [
            SimpleNamespace(
                rdan_save_rng=Remote(worker.rdan_save_rng, events, f"save-rng:infer-{rank}"),
                rdan_load_rng=Remote(worker.rdan_load_rng, events, f"load-rng:infer-{rank}"),
            )
            for rank, worker in enumerate(self.ranks)
        ]

    def record_generation(self, step: int) -> None:
        """Advance the deterministic rollout progress one pipeline step.

        Stands in for the real ``ResponseVLLMInferWorker.generate`` body, whose step guard
        is exercised directly in ``test_vllm_generation_step_guard_rejects_broken_progress``.
        """

        for worker in self.ranks:
            worker._rdan_vllm_generation_step = step
            worker._rdan_vllm_generation_ordinal = 0


class FakeScheduler:
    """Stand-in for RTT's ``DynamicSamplingScheduler`` Ray actor."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.requests: list[FakeDataProto] = []
        self.batch_sizes: list[int] = []
        self.restored_state: Any = None
        self.infer_cluster: Any = None
        self.set_scheduler = Remote(self._set_scheduler)
        self.get_batch_opt_level_0 = Remote(self._get_batch)
        self.get_scheduler_state = Remote(self._get_state)
        self.shutdown = Remote(self._shutdown)

    def _set_scheduler(self, *, actor_cluster: Any, dataset: Any, state: Any, **kwargs: Any) -> None:
        self.infer_cluster = actor_cluster
        self.dataset = dataset
        self.restored_state = state

    def _get_batch(self, *, data: FakeDataProto, batch_size: int) -> FakeDataProto:
        step = data.meta_info["global_step"]
        self.requests.append(data)
        self.batch_sizes.append(batch_size)
        self.events.append(f"rollout:{step}")
        self.infer_cluster.record_generation(step)
        return _rewarded_batch()

    def _get_state(self) -> dict[str, Any]:
        self.events.append("scheduler-state")
        return {"dataset_iter_count": len(self.requests)}

    def _shutdown(self) -> None:
        self.events.append("scheduler-shutdown")


# --------------------------------------------------------------------------------------
# Runtime loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Runtime:
    """The production modules loaded against the framework surrogates."""

    pipeline: ModuleType
    workers: ModuleType
    config: ModuleType
    events: list[str]


def _load_runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    """Import the production modules with RTT and Ray replaced by surrogates."""

    events: list[str] = []
    modules = _roll_stub_modules(events)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    workers = _load_module(monkeypatch, "workers", register=False)
    train_step = _load_module(monkeypatch, "train_step", register=True)
    assert train_step.DataProto is FakeDataProto
    pipeline = _load_module(monkeypatch, "pipeline", register=False)
    import rdan_grpo.config as config_module

    return Runtime(pipeline=pipeline, workers=workers, config=config_module, events=events)


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str, *, register: bool) -> ModuleType:
    path = ROOT / f"src/rdan_grpo/{name}.py"
    spec_name = f"rdan_grpo.{name}" if register else f"test_training_e2e_{name}"
    spec = importlib.util.spec_from_file_location(spec_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec_name, module)
    spec.loader.exec_module(module)
    return module


def _roll_stub_modules(events: list[str]) -> dict[str, ModuleType]:
    ray = types.ModuleType("ray")
    ray.get = lambda value, **kwargs: value
    ray.remote = lambda value: SimpleNamespace(
        options=lambda **kwargs: SimpleNamespace(remote=lambda **inner: FakeScheduler(events))
    )
    ray.get_runtime_context = lambda: SimpleNamespace(get_node_id=lambda: "node-0")
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    scheduling.NodeAffinitySchedulingStrategy = lambda **kwargs: SimpleNamespace(**kwargs)
    decorator = types.ModuleType("roll.distributed.scheduler.decorator")
    decorator.Dispatch = SimpleNamespace(ONE_TO_ALL=1, DP_MP_DISPATCH_FIRST=2, DP_MP_COMPUTE=3)
    decorator.register = lambda **kwargs: lambda function: function
    protocol = types.ModuleType("roll.distributed.scheduler.protocol")
    protocol.DataProto = FakeDataProto
    modules: dict[str, ModuleType] = {
        "ray": ray,
        "ray.util": types.ModuleType("ray.util"),
        "ray.util.scheduling_strategies": scheduling,
        "roll.distributed.scheduler.decorator": decorator,
        "roll.distributed.scheduler.protocol": protocol,
    }
    attributes = {
        "roll.datasets.collator": {"DataCollatorWithPaddingForPaddedKeys": object},
        "roll.distributed.executor.cluster": {"Cluster": object},
        "roll.distributed.scheduler.generate_scheduler": {"DynamicSamplingScheduler": object},
        "roll.models.model_providers": {"default_tokenizer_provider": lambda **kwargs: None},
        "roll.pipeline.base_pipeline": {"BasePipeline": _base_pipeline_cls(events)},
        "roll.pipeline.base_worker": {"InferWorker": FakeVllmInferWorker},
        "roll.pipeline.rlvr.actor_worker": {"ActorWorker": FakeRltActorWorker},
        "roll.utils.worker_state": {"WorkerState": _worker_state_cls(events)},
    }
    for name, values in attributes.items():
        module = types.ModuleType(name)
        for key, value in values.items():
            setattr(module, key, value)
        modules[name] = module
    for name in list(modules):
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            modules.setdefault(parent, types.ModuleType(parent))
    return modules


def _worker_state_cls(events: list[str]) -> type:
    """Build the driver RNG surrogate that writes the artifact the manifest links."""

    class FakeWorkerState:
        """Stand-in for RTT's ``WorkerState`` driver RNG helpers."""

        @staticmethod
        def save_rng_state(path: str, name: str) -> None:
            """Persist the driver RNG state into the checkpoint stage."""

            events.append(f"save-rng:{name}")
            torch.save(torch.get_rng_state(), Path(path) / f"rng_state_{name}.pth")

        @staticmethod
        def load_rng_state(path: str, name: str) -> None:
            """Restore the driver RNG state from a promoted checkpoint."""

            events.append(f"load-rng:{name}")
            torch.set_rng_state(torch.load(Path(path) / f"rng_state_{name}.pth", weights_only=False))

    return FakeWorkerState


def _rank_info() -> SimpleNamespace:
    return SimpleNamespace(dp_size=WORLD_SIZE, tp_size=1, pp_size=1, cp_size=1)


def _actor_worker(runtime: Runtime, pipeline_config: Any, rank: int) -> Any:
    worker = runtime.workers.ResponseActorWorker.__new__(runtime.workers.ResponseActorWorker)
    worker.rank_info = SimpleNamespace(dp_rank=rank)
    worker.strategy = FakeFsdpStrategy(rank)
    worker.pipeline_config = pipeline_config
    return worker


def _infer_worker(runtime: Runtime, pipeline_config: Any, rank: int) -> Any:
    worker = runtime.workers.ResponseVLLMInferWorker.__new__(runtime.workers.ResponseVLLMInferWorker)
    worker.rank_info = SimpleNamespace(dp_rank=rank)
    worker.pipeline_config = pipeline_config
    worker.worker_config = SimpleNamespace(strategy_args=SimpleNamespace(strategy_name="vllm"))
    worker.strategy = SimpleNamespace(get_metrics=lambda: {"vllm/num_requests": 4})
    return worker


# --------------------------------------------------------------------------------------
# Production config composition
# --------------------------------------------------------------------------------------


class ConfigNode:
    """Attribute view over one composed configuration object."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)
        for key, value in payload.items():
            setattr(self, key, _config_value(value))

    def to_dict(self) -> dict[str, Any]:
        """Return the exact payload this object was constructed from."""

        return copy.deepcopy(self._payload)


class SurrogateRlvrConfig(ConfigNode):
    """Stand-in for the pinned fork's ``RLVRConfig`` dataclass tree.

    ``dacite`` and the fork itself are not importable here, so this reproduces the parts of
    the constructed config the pipeline reads, including the ``device_mapping`` string that
    the fork's ``__post_init__`` evaluates back into a device list.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__({**RUNTIME_DEFAULTS, **payload})
        self.rewards = {name: ConfigNode(worker) for name, worker in payload["rewards"].items()}
        for name in ("actor_train", "actor_infer"):
            worker = getattr(self, name)
            worker.name = name
            worker.device_mapping = _device_mapping(payload[name]["device_mapping"])
        self.actor_train.model_update_frequency = MODEL_UPDATE_FREQUENCY

    def set_max_steps(self, *, max_steps: int) -> None:
        """Apply the resolved training horizon the way the fork's config does."""

        self.max_steps = max_steps


def _config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode(value)
    if isinstance(value, list):
        return [_config_value(item) for item in value]
    return value


def _device_mapping(value: Any) -> list[int]:
    return ast.literal_eval(value) if isinstance(value, str) else value


def _compose_config_payload(name: str = CONFIG_NAME) -> dict[str, Any]:
    """Compose one production yaml the way Hydra does, then resolve its interpolations."""

    payload = _merge_defaults(name)
    payload.pop("hydra", None)
    return _resolve(payload, payload)


def _merge_defaults(name: str) -> dict[str, Any]:
    payload = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    merged: dict[str, Any] = {}
    for entry in payload.pop("defaults", []):
        if entry == "_self_":
            continue
        parent = entry if isinstance(entry, str) else next(iter(entry.values()))
        merged = _merge(merged, _merge_defaults(parent))
    return _merge(merged, payload)


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve(value: Any, root: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: _resolve(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, root) for item in value]
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        return value
    reference = value[2:-1]
    if reference.startswith("oc.env:"):
        return os.environ[reference.removeprefix("oc.env:")]
    resolved: Any = root
    for part in reference.split("."):
        resolved = resolved[part]
    return _resolve(resolved, root)


def _build_config(runtime: Runtime, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run the real config loader over the composed production yaml."""

    monkeypatch.setenv("RDAN_MODEL_SNAPSHOT", MODEL_SNAPSHOT)
    # Neutralize the pinned-checkout git probe when the loader still carries one, and build
    # the surrogate config directly because dacite and the fork are not importable here.
    monkeypatch.setattr(runtime.config, "_verify_rtt", lambda root: None, raising=False)
    monkeypatch.setattr(runtime.config, "_construct", lambda config_cls, payload: config_cls(payload))
    return runtime.config.load_response_rlvr_config(
        ROOT / "reference/rtt", SurrogateRlvrConfig, _compose_config_payload()
    )


def _identity(config: Any) -> CheckpointIdentity:
    response = config.rdan_response
    return CheckpointIdentity(
        planned_horizon=config.max_steps,
        method=response.method,
        method_weight=response.quality_weight,
        resolved_config_sha256=response.resolved_config_sha256,
        certificate=None,
        data=ArtifactIdentity(id="hybrid.jsonl", sha256="1" * 64),
        revisions={"code": "a" * 40, "rtt": "b" * 40, "model": "c" * 40},
        base_checkpoint_sha256="d" * 64,
        wandb={
            "entity": "RDAN-GRPO",
            "project": "rdan-grpo-qwen3-4b",
            "run_id": "run-e2e",
            "name": config.exp_name,
            "group": response.method,
        },
    )


# --------------------------------------------------------------------------------------
# Rollout batch
# --------------------------------------------------------------------------------------


def _rewarded_batch() -> FakeDataProto:
    """Build the grouped rubric-reward batch the RTT scheduler returns for one rollout.

    Column zero is the binary hard rubric that drives ORM; column one is the soft
    judge rubric that drives PRM. Half of every group passes the hard rubric, so both
    the outcome variance and the correct-subset quality variance are non-degenerate.
    """

    prompt_index = torch.arange(PROMPT_COUNT).repeat_interleave(GROUP_SIZE)
    position = torch.arange(RESPONSE_COUNT) % GROUP_SIZE
    hard = torch.where(position < GROUP_SIZE // 2, 1.0, -1.0)
    soft = torch.tensor([-1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0]).repeat(PROMPT_COUNT)
    scores = torch.stack((hard, soft), dim=1)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0]]).repeat(RESPONSE_COUNT, 1)
    batch = {
        "origin_prompt_id": prompt_index,
        "input_ids": torch.ones(RESPONSE_COUNT, SEQUENCE_LENGTH, dtype=torch.long),
        "attention_mask": torch.ones(RESPONSE_COUNT, SEQUENCE_LENGTH, dtype=torch.long),
        "response_mask": mask,
        "rdan_scores": scores,
        "rdan_rubric_mask": torch.ones_like(scores, dtype=torch.bool),
        "rdan_eval_mask": torch.ones_like(scores, dtype=torch.bool),
        "rdan_hard_mask": torch.tensor([True, False]).expand_as(scores).clone(),
        "rdan_unsupported_hard": torch.zeros(RESPONSE_COUNT, dtype=torch.bool),
        "rdan_judge_failed": torch.zeros(RESPONSE_COUNT, dtype=torch.bool),
    }
    keys = [f"prompt-{int(index)}" for index in prompt_index]
    rubrics = np.empty(RESPONSE_COUNT, dtype=object)
    rubrics[:] = [[{"description": "Answer in English.", "weight": 1}] for _ in range(RESPONSE_COUNT)]
    non_tensor = {
        "prompt": np.asarray([f"prompt text {int(index)}" for index in prompt_index], dtype=object),
        "rubrics": rubrics,
        "source": np.asarray(["rubrichub_instruction_following"] * RESPONSE_COUNT, dtype=object),
        "ground_truth": np.asarray([{"hard_mask": [True, False]}] * RESPONSE_COUNT, dtype=object),
        "rdan_prompt_key": np.asarray(keys, dtype=object),
    }
    metrics = {"vllm/rank": 0.0, "vllm/metrics_available": 1.0, "vllm/num_requests": 4.0}
    return FakeDataProto(batch, non_tensor, {"vllm_metrics": metrics})


# --------------------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Harness:
    """One constructed pipeline and the surrogates it was wired to."""

    pipeline: Any
    actor_train: FakeActorTrainCluster
    actor_infer: FakeInferCluster
    scheduler: FakeScheduler
    tracker: FakeTracker
    dataset_args: list[Any]


def _build_pipeline(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
    config: Any,
    identity: CheckpointIdentity,
    checkpoint_root: Path,
    *,
    stop_after_step: int,
    resume: Path | None = None,
) -> Harness:
    """Construct the real pipeline against the cluster, tokenizer, and dataset surrogates."""

    dataset_args: list[Any] = []
    clusters: dict[str, Any] = {}

    def cluster(*, name: str, worker_cls: str, resource_manager: Any, worker_config: Any) -> Any:
        if worker_cls == ACTOR_WORKER_PATH:
            clusters["actor_train"] = FakeActorTrainCluster(runtime, runtime.events, config)
            return clusters["actor_train"]
        if worker_cls == INFER_WORKER_PATH:
            clusters["actor_infer"] = FakeInferCluster(runtime, runtime.events, config)
            return clusters["actor_infer"]
        return FakeCluster(DOMAIN, runtime.events)

    def load_domain_dataset(pipeline_config: Any, tokenizer: Any) -> tuple[str, Any]:
        dataset_args.append(pipeline_config.actor_train.data_args)
        return DOMAIN, SimpleNamespace(name="response-dataset")

    monkeypatch.setattr(runtime.pipeline, "Cluster", cluster)
    monkeypatch.setattr(runtime.pipeline, "_load_domain_dataset", load_domain_dataset)
    monkeypatch.setattr(runtime.pipeline, "default_tokenizer_provider", lambda **kwargs: SimpleNamespace())
    pipeline = runtime.pipeline.build_response_training_pipeline(
        config,
        response_config=config.rdan_response,
        certificate=None,
        checkpoint_identity=identity,
        checkpoint_root=checkpoint_root,
        stop_after_step=stop_after_step,
        resume_checkpoint=resume,
    )
    return Harness(
        pipeline=pipeline,
        actor_train=clusters["actor_train"],
        actor_infer=clusters["actor_infer"],
        scheduler=pipeline.scheduler,
        tracker=pipeline.tracker,
        dataset_args=dataset_args,
    )


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def test_one_step_runs_rollout_reward_advantage_update_transfer_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _load_runtime(monkeypatch)
    config = _build_config(runtime, monkeypatch)
    identity = _identity(config)
    root = tmp_path / "checkpoints"
    harness = _build_pipeline(runtime, monkeypatch, config, identity, root, stop_after_step=1)

    completed = harness.pipeline.run()

    # The production yaml really did reach the pipeline.
    assert config.rdan_response.method == "rtt_papo_response"
    assert config.rdan_response.quality_weight == 1.0
    assert harness.dataset_args[0].file_name == ["data/hybrid.jsonl"]
    assert harness.pipeline.model_update_pairs == [("actor_train", "actor_infer", MODEL_UPDATE_FREQUENCY)]

    # Every boundary ran once, in order, with the weight transfer after the update.
    assert runtime.events == [
        "transfer:0",
        "reset-peak:actor_train",
        "reset-peak:actor_infer",
        "offload:actor_train",
        "load:actor_infer",
        f"load:{DOMAIN}",
        "rollout:1",
        "offload:actor_infer",
        f"offload:{DOMAIN}",
        "load:actor_train",
        "training-state",
        "log-probs",
        "train",
        "training-state",
        "cuda-memory",
        "transfer:1",
        "save-dcp:1",
        "save-rng:driver",
        "save-rng:infer-0",
        "save-rng:infer-1",
        "scheduler-state",
        "train-counters",
        "wandb-log",
        "scheduler-shutdown",
        "wandb-finish",
    ]
    assert harness.pipeline.transfers == [0, 1]
    assert runtime.events.index("train") < runtime.events.index("transfer:1")
    assert runtime.events.index("transfer:1") < runtime.events.index("save-dcp:1")
    assert harness.actor_train.train_calls == harness.actor_train.log_prob_calls == 1

    # The rollout was requested for a positive contiguous pipeline step.
    request = harness.scheduler.requests[0]
    assert request.meta_info["global_step"] == 1
    assert request.meta_info["is_offload_states"] is False
    assert request.meta_info["generation_config"] == config.actor_infer.generating_args.to_dict()
    assert harness.scheduler.batch_sizes == [config.rollout_batch_size]

    # Rubric rewards and both advantage channels reached the training batch.
    trained = harness.actor_train.trained_batch
    assert trained is not None
    evidence = trained.batch
    for name in (
        "rdan_selected_reward",
        "rdan_response_advantage",
        "rdan_raw_quality",
        "rdan_quality_eligible",
        "rdan_quality_advantage",
        "rdan_scalar_advantage",
        "rdan_response_valid",
    ):
        assert evidence[name].shape == (RESPONSE_COUNT,)
    assert trained.meta_info["rdan_method"] == "rtt_papo_response"
    assert trained.meta_info["rdan_method_weight"] == 1.0

    # ORM is the binary outcome normalized over all eight responses of the group.
    outcome = _outcome()
    assert torch.equal(evidence["rdan_selected_reward"], outcome)
    response_groups = evidence["rdan_response_advantage"].reshape(-1, GROUP_SIZE)
    assert torch.allclose(response_groups.sum(dim=-1), torch.zeros(PROMPT_COUNT), atol=1e-5)
    assert bool((response_groups[:, : GROUP_SIZE // 2] > 0).all())
    assert bool((response_groups[:, GROUP_SIZE // 2 :] < 0).all())

    # PRM is normalized over the correct subset only and is exactly zero elsewhere.
    correct = outcome.reshape(-1, GROUP_SIZE).bool()
    eligible = evidence["rdan_quality_eligible"].reshape(-1, GROUP_SIZE)
    quality_groups = evidence["rdan_quality_advantage"].reshape(-1, GROUP_SIZE)
    assert torch.equal(eligible, correct)
    assert bool(quality_groups[~correct].eq(0).all())
    assert bool(quality_groups[correct].abs().max() > 0)
    assert torch.allclose(quality_groups.sum(dim=-1), torch.zeros(PROMPT_COUNT), atol=1e-5)

    # A_total = A_out + A_proc, unweighted.
    expected_total = evidence["rdan_response_advantage"] + evidence["rdan_quality_advantage"]
    assert torch.allclose(evidence["rdan_scalar_advantage"], expected_total)

    # The token advantage is the scalar advantage broadcast over the shifted response mask.
    shifted = evidence["response_mask"][:, 1:].bool()
    broadcast = evidence["rdan_scalar_advantage"].unsqueeze(-1) * shifted.to(torch.float32)
    assert torch.equal(evidence["advantages"], evidence["returns"])
    assert torch.equal(evidence["advantages"], broadcast)
    assert torch.equal(evidence["old_log_probs"], evidence["ref_log_probs"])
    assert torch.equal(evidence["final_response_mask"], shifted)

    # Both ranks took the configured optimizer updates and the checkpoint recorded them.
    assert harness.actor_train.optimizer_steps() == [UPDATES_PER_STEP] * WORLD_SIZE
    assert completed.completed_step == 1
    assert completed.checkpoints == (root / "step-000001",)
    manifest = load_checkpoint(root / "step-000001", identity=identity)
    assert manifest["completed_step"] == 1 and manifest["next_step"] == 2
    assert manifest["optimizer_counters"] == manifest["scheduler_counters"] == {"0": 2, "1": 2}
    assert manifest["scheduler_state"] == {"dataset_iter_count": 1}
    assert manifest["group_diagnostics"]["group_count"] == PROMPT_COUNT
    assert manifest["group_diagnostics"]["response_active_group_rate"] == 1.0
    assert manifest["group_diagnostics"]["quality_active_group_rate"] >= 0.1
    assert manifest["clipping_fraction"] == 0.0

    # W&B received the step metrics, including both reward channels and the rollout curves.
    assert harness.tracker.finished == 1
    logged = harness.tracker.logged
    assert len(logged) == 1
    assert logged[0]["system/step"] == 1
    assert logged[0]["rdan/response_token_clipfrac"] == 0.0
    assert logged[0]["vllm/num_requests"] == 4.0
    assert logged[0]["reward/outcome_advantage_mean"] == pytest.approx(0.0, abs=1e-6)
    assert logged[0]["advantage/positive_rate"] > 0
    assert logged[0]["length/mean"] == 3.0
    assert harness.pipeline.state.log_history == logged


def test_resume_restores_optimizer_state_and_continues_at_the_next_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _load_runtime(monkeypatch)
    config = _build_config(runtime, monkeypatch)
    identity = _identity(config)
    root = tmp_path / "checkpoints"
    first = _build_pipeline(runtime, monkeypatch, config, identity, root, stop_after_step=1)
    first.pipeline.run()
    checkpoint = root / "step-000001"
    saved_rng = first.actor_infer.ranks[0].rdan_save_rng()
    runtime.events.clear()

    resumed = _build_pipeline(runtime, monkeypatch, config, identity, root, stop_after_step=2, resume=checkpoint)

    # The checkpoint was reloaded before any rollout, optimizer moments included.
    assert resumed.pipeline.completed_step == 1
    assert resumed.actor_train.optimizer_steps() == [UPDATES_PER_STEP] * WORLD_SIZE
    assert [worker.rdan_train_counters()["optimizer_steps"] for worker in resumed.actor_train.workers] == [2, 2]
    assert resumed.scheduler.restored_state == {"dataset_iter_count": 1}
    assert resumed.actor_infer.ranks[0].rdan_save_rng() == saved_rng
    assert runtime.events[:4] == ["load-dcp", "load-rng:driver", "load-rng:infer-0", "load-rng:infer-1"]

    completed = resumed.pipeline.run()

    assert runtime.events[4] == "transfer:1"
    assert resumed.scheduler.requests[0].meta_info["global_step"] == 2
    assert resumed.pipeline.transfers == [1, 2]
    assert completed.completed_step == 2
    assert completed.checkpoints == (root / "step-000002",)
    assert resumed.actor_train.optimizer_steps() == [2 * UPDATES_PER_STEP] * WORLD_SIZE
    manifest = load_checkpoint(root / "step-000002", identity=identity)
    assert manifest["completed_step"] == 2
    assert manifest["optimizer_counters"] == {"0": 4, "1": 4}
    assert resumed.tracker.logged[0] == {"system/resumed_from_step": 1}
    assert resumed.tracker.logged[1]["system/step"] == 2

    # Priming is load bearing: without it DCP restores no moments and the resume fails.
    monkeypatch.setattr(runtime.workers, "_prime_optimizer_state", lambda strategy: None)
    with pytest.raises(RuntimeError, match="restored no optimizer moment state"):
        _build_pipeline(runtime, monkeypatch, config, identity, root, stop_after_step=2, resume=checkpoint)


def test_vllm_generation_step_guard_rejects_broken_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _load_runtime(monkeypatch)
    compat = types.ModuleType("rdan_grpo.compat")
    compat.install_vllm_sampling_seed_compat = lambda: None
    monkeypatch.setitem(sys.modules, "rdan_grpo.compat", compat)
    worker = _infer_worker(runtime, SimpleNamespace(seed=240520), rank=0)

    def generate(step: Any) -> Any:
        request = FakeDataProto(meta_info={"global_step": step, "generation_config": {"temperature": 0.99}})
        return asyncio.run(worker.generate(request))

    with pytest.raises(RuntimeError, match="positive pipeline step"):
        generate(0)
    output = generate(1)
    assert output.non_tensor_batch["generation_id"][0].startswith("gen-000001-r0-c0000-")
    assert output.meta_info["vllm_metrics"]["vllm/rank"] == 0.0
    assert output.meta_info["vllm_metrics"]["vllm/num_requests"] == 4.0
    with pytest.raises(RuntimeError, match="not contiguous with restored progress 1"):
        generate(3)
    with pytest.raises(RuntimeError, match="integer global_step"):
        generate(None)
    assert worker.rdan_save_rng()["last_generation_step"] == 1


def _outcome() -> torch.Tensor:
    """Return the binary hard-rubric outcome reward implied by the rollout batch."""

    position = torch.arange(RESPONSE_COUNT) % GROUP_SIZE
    return (position < GROUP_SIZE // 2).float()
