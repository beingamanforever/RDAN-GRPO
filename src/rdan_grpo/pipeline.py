"""Custom response-only production pipeline for the pinned RTT checkout."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.worker_state import WorkerState

from rdan_grpo.dataset import load_response_dataset as _load_jsonl_response_dataset
from rdan_grpo.checkpoint import (
    CheckpointIdentity,
    CheckpointState,
    create_checkpoint_stage,
    load_checkpoint,
    promote_checkpoint,
)
from rdan_grpo.config import ACTOR_WORKER_PATH, INFER_WORKER_PATH, ResponseConfig
from rdan_grpo.train_step import ResponseTrainResult, run_response_train_step
from rdan_grpo.scalar import QUALITY_METHODS

_StateObserver = Callable[[], Sequence[Mapping[str, Any]]]

# Promoted checkpoints kept on disk after each successful save. Two is sufficient for
# resume: a resumed run only ever needs its own immediate predecessor checkpoint.
CHECKPOINT_RETENTION_COUNT = 2

# checkpoint requires a non-empty receipt_links mapping pointing at a file
# staged inside the checkpoint, so the post-update per-rank training state is written and
# linked there. It is also the evidence that says which optimizer step these weights are.
TRAINING_STATE_ARTIFACT = "state/training.json"


@dataclass(frozen=True)
class CompletedResponseRun:
    """Result of a response training run that reached its requested step."""

    completed_step: int
    checkpoints: tuple[Path, ...]


class ResponseTrainingPipeline(BasePipeline):
    """Run response-only optimizer transactions with local checkpointing."""

    def __init__(
        self,
        pipeline_config: Any,
        *,
        response_config: ResponseConfig,
        certificate: Mapping[str, Any] | None,
        checkpoint_identity: CheckpointIdentity,
        checkpoint_root: str | Path,
        stop_after_step: int | None = None,
        resume_checkpoint: str | Path | None = None,
    ) -> None:
        _validate_inputs(pipeline_config, response_config, checkpoint_identity, stop_after_step)
        BasePipeline.model_update_groups = []
        BasePipeline.checkpoint_clusters = []
        super().__init__(pipeline_config)
        self._initialize_run_contract(
            response_config=response_config,
            certificate=certificate,
            checkpoint_identity=checkpoint_identity,
            checkpoint_root=checkpoint_root,
            stop_after_step=stop_after_step,
            resume_checkpoint=resume_checkpoint,
        )
        pipeline_config.set_max_steps(max_steps=pipeline_config.max_steps)
        self._initialize_runtime(pipeline_config)
        if self._resume_manifest:
            self._restore_checkpoint()

    def _initialize_run_contract(
        self,
        *,
        response_config: ResponseConfig,
        certificate: Mapping[str, Any] | None,
        checkpoint_identity: CheckpointIdentity,
        checkpoint_root: str | Path,
        stop_after_step: int | None,
        resume_checkpoint: str | Path | None,
    ) -> None:
        self.model_update_groups = []
        self.checkpoint_clusters = []
        self.response_config = response_config
        self.certificate = None if certificate is None else dict(certificate)
        self.checkpoint_identity = checkpoint_identity
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.stop_after_step = stop_after_step or checkpoint_identity.planned_horizon
        self._resume_path = Path(resume_checkpoint).resolve() if resume_checkpoint is not None else None
        self._resume_manifest = (
            load_checkpoint(self._resume_path, identity=checkpoint_identity) if self._resume_path is not None else None
        )
        self.completed_step = int(self._resume_manifest["completed_step"]) if self._resume_manifest else 0
        self._start_step = self.completed_step + 1
        if self.stop_after_step <= self.completed_step:
            raise ValueError("stop_after_step must advance beyond the resumed checkpoint")
        self.state.step = self.completed_step

    def _initialize_runtime(self, pipeline_config: Any) -> None:
        self.tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args)
        domain, dataset = _load_domain_dataset(pipeline_config, self.tokenizer)
        self._initialize_clusters(pipeline_config)
        self.scheduler = self._initialize_scheduler(pipeline_config, domain, dataset)
        self.domain = domain
        self._initialize_workers(pipeline_config)

    def _initialize_clusters(self, pipeline_config: Any) -> None:
        self.actor_train = _cluster(pipeline_config.actor_train, self.resource_manager)
        self.actor_infer = _cluster(pipeline_config.actor_infer, self.resource_manager)
        self.rewards = {
            name: Cluster(
                name=f"reward-{name}",
                worker_cls=worker.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker,
            )
            for name, worker in pipeline_config.rewards.items()
        }
        self.download_models(self.actor_train, self.actor_infer, *self.rewards.values())

    def _initialize_scheduler(self, pipeline_config: Any, domain: str, dataset: Any) -> Any:
        scheduler_state = self._resume_manifest["scheduler_state"] if self._resume_manifest else None
        scheduler = (
            ray.remote(DynamicSamplingScheduler)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            )
            .remote(pipeline_config=pipeline_config)
        )
        ray.get(
            scheduler.set_scheduler.remote(
                actor_cluster=self.actor_infer,
                reward_clusters={domain: self.rewards[domain]},
                dataset=dataset,
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs={"max_length": pipeline_config.prompt_length, "padding": "max_length"},
                state=scheduler_state,
            )
        )
        return scheduler

    def _initialize_workers(self, pipeline_config: Any) -> None:
        ray.get(self.actor_infer.initialize(pipeline_config=pipeline_config, blocking=False))
        reward_refs = []
        for reward in self.rewards.values():
            reward_refs.extend(reward.initialize(pipeline_config=pipeline_config, blocking=False))
        ray.get(reward_refs)
        ray.get(self.actor_train.initialize(pipeline_config=pipeline_config, blocking=False))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=pipeline_config.actor_train.model_update_frequency,
        )

    @torch.no_grad()
    def run(self) -> CompletedResponseRun:
        """Train through the requested step and return the promoted checkpoints."""

        primary_error: BaseException | None = None
        promoted: list[Path] = []
        try:
            promoted = self._run_steps()
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error = _cleanup(self.scheduler, self.tracker)
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error
        return CompletedResponseRun(completed_step=self.completed_step, checkpoints=tuple(promoted))

    def _run_steps(self) -> list[Path]:
        promoted: list[Path] = []
        # The inference engine starts from its own copy of the base weights, so the trained
        # actor must be pushed across before the first rollout of a fresh or resumed run.
        self._transfer(self.completed_step)
        for step in range(self.completed_step + 1, self.stop_after_step + 1):
            rewarded, result, observations = self._run_step(step)
            checkpoint = self._save_step(step, rewarded, result, observations)
            if checkpoint is not None:
                promoted.append(checkpoint)
            self._record_step(step, rewarded, result)
        return promoted

    def _run_step(self, step: int) -> tuple[DataProto, ResponseTrainResult, dict[str, Any]]:
        self.actor_train.rdan_reset_cuda_peak(blocking=True)
        self.actor_infer.rdan_reset_cuda_peak(blocking=True)
        rewarded = self._generate(step)
        observations: dict[str, Any] = {}
        training_state, memory = self._step_observers(observations)
        result = run_response_train_step(
            pipeline_config=self.pipeline_config,
            actor_train=self.actor_train,
            actor_infer=self.actor_infer,
            rewarded_batch=rewarded,
            certificate=self.certificate,
            observe_training_state=training_state,
            observe_post_transaction_memory=memory,
            method=self.response_config.method,
            quality_weight=self.response_config.quality_weight,
            mix_weight=self.response_config.mix_weight,
        )
        # The rollout engine holds the pre-update weights until this push, so skipping it
        # would train on fresh weights while generating from stale ones for the whole run.
        self._transfer(step)
        return rewarded, result, observations

    def _step_observers(
        self,
        observations: dict[str, Any],
    ) -> tuple[_StateObserver, _StateObserver]:
        def training_state() -> Sequence[Mapping[str, Any]]:
            """Capture actor training state at the update boundary."""

            values = self.actor_train.rdan_training_state(blocking=True)
            observations["training_state"] = values
            return values

        def memory() -> Sequence[Mapping[str, Any]]:
            """Capture actor CUDA memory after the transaction."""

            values = self.actor_train.rdan_cuda_memory(blocking=True)
            observations["memory"] = values
            return values

        return training_state, memory

    def _record_step(self, step: int, rewarded: DataProto, result: ResponseTrainResult) -> None:
        self.completed_step = step
        self.state.step = step
        diagnostics = _group_diagnostics(rewarded, self.pipeline_config.num_return_sequences_in_group)
        response_length_cap = self.pipeline_config.actor_infer.generating_args.max_new_tokens
        metrics = {
            **_number_metrics(result.metrics),
            "reward/within_group_selected_variance_mean": diagnostics["selected_reward_variance_mean"],
            "reward/response_active_group_rate": diagnostics["response_active_group_rate"],
            "reward/quality_active_group_rate": diagnostics["quality_active_group_rate"],
            **_reward_curve_metrics(rewarded, response_length_cap),
            "system/peak_memory_fraction": result.peak_memory_fraction,
            "system/step": step,
        }
        self.state.log_history.append(metrics)
        if self._resume_manifest is not None and step == self._start_step:
            # Make the re-executed region attributable: steps _start_step..completed-before-crash
            # legitimately re-run from the restored optimizer/scheduler/RNG state, so they appear
            # twice on the system/step axis. Record the boundary rather than leaving it ambiguous.
            self.tracker.log(values={"system/resumed_from_step": self._start_step - 1}, step=None)
        # step=None keeps W&B's own log index monotonic across a resume (it would otherwise
        # discard every re-logged point for steps _start_step..completed-before-crash, since
        # W&B requires a strictly increasing step). metrics["system/step"] already carries the
        # true pipeline step for the x-axis.
        # TODO: tracking.py (owned by another change in this pass) should call
        # run.define_metric("system/step") and run.define_metric("*", step_metric="system/step")
        # at init so dashboards use it as the x-axis automatically instead of by manual configuration.
        self.tracker.log(values=metrics, step=None)

    def _generate(self, step: int) -> DataProto:
        request = DataProto(
            meta_info={
                "global_step": step,
                "generation_config": self.pipeline_config.actor_infer.generating_args.to_dict(),
                "is_offload_states": False,
                "reward_system_config": self.pipeline_config.reward_system_config,
            }
        )
        self.actor_train.offload_states(blocking=True)
        self.actor_infer.load_states(blocking=True)
        for reward in self.rewards.values():
            reward.load_states(blocking=True)
        try:
            return ray.get(
                self.scheduler.get_batch_opt_level_0.remote(
                    data=request,
                    batch_size=self.pipeline_config.rollout_batch_size,
                ),
                timeout=self.pipeline_config.rpc_timeout,
            )
        finally:
            self.actor_infer.offload_states(blocking=True)
            for reward in self.rewards.values():
                reward.offload_states(blocking=True)
            self.actor_train.load_states(blocking=True)

    def _transfer(self, pipeline_step: int) -> None:
        """Synchronize trained actor weights into the vLLM inference engine."""

        self.model_update(pipeline_step)

    def _save_step(
        self,
        step: int,
        rewarded: DataProto,
        result: ResponseTrainResult,
        observations: Mapping[str, Any],
    ) -> Path | None:
        metrics = _number_metrics(result.metrics)
        diagnostics = _group_diagnostics(rewarded, self.pipeline_config.num_return_sequences_in_group)
        clipping = _clipping_fraction(metrics)
        self._validate_step_promotion(diagnostics, clipping)
        if step % self.pipeline_config.save_steps and step != self.stop_after_step:
            return None
        return self._save_checkpoint(step, metrics, diagnostics, clipping, observations)

    def _validate_step_promotion(self, diagnostics: Mapping[str, int | float], clipping: float) -> None:
        if clipping >= 1:
            raise RuntimeError("response checkpoint rejects 100 percent clipping")
        if self.response_config.method in QUALITY_METHODS and diagnostics["quality_active_group_rate"] < 0.1:
            raise RuntimeError("quality method checkpoint requires quality active group rate at least 0.1")

    def _save_checkpoint(
        self,
        step: int,
        metrics: Mapping[str, float],
        diagnostics: Mapping[str, int | float],
        clipping: float,
        observations: Mapping[str, Any],
    ) -> Path:
        stage = create_checkpoint_stage(self.checkpoint_root, step)
        scheduler_state = self._write_checkpoint_payload(stage, step, observations)
        counters = self.actor_train.rdan_train_counters(blocking=True)
        state = _checkpoint_state(step, counters, scheduler_state, metrics, diagnostics, clipping, observations)
        artifacts = [path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()]
        promoted = promote_checkpoint(stage, identity=self.checkpoint_identity, state=state, artifacts=artifacts)
        _prune_checkpoints(self.checkpoint_root, self._resume_path)
        return promoted

    def _write_checkpoint_payload(self, stage: Path, step: int, observations: Mapping[str, Any]) -> Mapping[str, Any]:
        actor_dir = stage / "actor"
        self.actor_train.rdan_save_dcp(str(actor_dir), step, blocking=True)
        rng_dir = stage / "rng"
        rng_dir.mkdir()
        WorkerState.save_rng_state(str(rng_dir), "driver")
        infer_states = ray.get([worker.rdan_save_rng.remote() for worker in self.actor_infer.workers])
        for rank, state in enumerate(infer_states):
            torch.save(state, rng_dir / f"infer-rank-{rank}.pt")
        scheduler_state = ray.get(self.scheduler.get_scheduler_state.remote())
        _write_json(stage / "scheduler/state.json", scheduler_state)
        _write_json(stage / TRAINING_STATE_ARTIFACT, {"ranks": list(observations["training_state"])})
        return scheduler_state

    def _restore_checkpoint(self) -> None:
        assert self._resume_path is not None and self._resume_manifest is not None
        self.actor_train.rdan_load_dcp(str(self._resume_path / "actor"), blocking=True)
        WorkerState.load_rng_state(str(self._resume_path / "rng"), "driver")
        states = [torch.load(self._resume_path / f"rng/infer-rank-{rank}.pt", weights_only=False) for rank in range(2)]
        ray.get([worker.rdan_load_rng.remote(states[rank]) for rank, worker in enumerate(self.actor_infer.workers)])


def _checkpoint_state(
    step: int,
    counters: Sequence[Mapping[str, int]],
    scheduler_state: Mapping[str, Any],
    metrics: Mapping[str, float],
    diagnostics: Mapping[str, int | float],
    clipping: float,
    observations: Mapping[str, Any],
) -> CheckpointState:
    optimizer = {value["rank"]: value["optimizer_steps"] for value in counters}
    scheduler = {value["rank"]: value["scheduler_steps"] for value in counters}
    memory = {value["rank"]: value["peak_bytes"] for value in observations["memory"]}
    return CheckpointState(
        completed_step=step,
        optimizer_counters=optimizer,
        scheduler_counters=scheduler,
        scheduler_state=scheduler_state,
        rng_artifacts={
            "driver": "rng/rng_state_driver.pth",
            "infer_0": "rng/infer-rank-0.pt",
            "infer_1": "rng/infer-rank-1.pt",
        },
        metrics=metrics,
        peak_memory=memory,
        reward_variance=diagnostics["selected_reward_variance_mean"],
        group_diagnostics=diagnostics,
        clipping_fraction=clipping,
        receipt_links={"training_state": TRAINING_STATE_ARTIFACT},
    )


def _reward_curve_metrics(rewarded: DataProto, response_length_cap: int) -> dict[str, float]:
    """Surface reward, advantage, and response-length training curves already on the batch.

    Every value here reshapes a tensor the reward/advantage adapter already computed
    (bridge.py / scalar.py) into a scalar training curve; no reward math is
    recomputed here. ORM (outcome) values come from rdan_selected_reward and its advantage
    component (rdan_response_advantage); PRM (process) values come from rdan_raw_quality,
    restricted to quality-eligible rows, and its advantage component (rdan_quality_advantage).
    """

    selected = rewarded.batch["rdan_selected_reward"].detach().float()
    outcome_advantage = rewarded.batch["rdan_response_advantage"].detach().float()
    quality = rewarded.batch["rdan_raw_quality"].detach().float()
    quality_eligible = rewarded.batch["rdan_quality_eligible"].detach().bool()
    quality_advantage = rewarded.batch["rdan_quality_advantage"].detach().float()
    scalar_advantage = rewarded.batch["rdan_scalar_advantage"].detach().float()
    valid = rewarded.batch["rdan_response_valid"].detach().bool()
    lengths = rewarded.batch["response_mask"].detach().sum(dim=-1).float()
    eligible_quality = quality[quality_eligible]

    metrics = {
        "reward/selected_mean": float(selected.mean().item()),
        "reward/selected_std": float(selected.std(unbiased=False).item()),
        "reward/selected_min": float(selected.min().item()),
        "reward/selected_max": float(selected.max().item()),
        "reward/valid_rate": float(valid.float().mean().item()),
        "reward/outcome_advantage_mean": float(outcome_advantage.mean().item()),
        "reward/outcome_advantage_std": float(outcome_advantage.std(unbiased=False).item()),
        "reward/process_quality_mean": float(eligible_quality.mean().item()) if eligible_quality.numel() else 0.0,
        "reward/process_quality_std": (
            float(eligible_quality.std(unbiased=False).item()) if eligible_quality.numel() >= 2 else 0.0
        ),
        "reward/process_advantage_mean": float(quality_advantage.mean().item()),
        "reward/process_advantage_std": float(quality_advantage.std(unbiased=False).item()),
        "advantage/mean": float(scalar_advantage.mean().item()),
        "advantage/std": float(scalar_advantage.std(unbiased=False).item()),
        "advantage/zero_rate": float((scalar_advantage.abs() <= 1e-8).float().mean().item()),
        "advantage/positive_rate": float((scalar_advantage > 1e-8).float().mean().item()),
        "length/mean": float(lengths.mean().item()),
        "length/max": float(lengths.max().item()),
        "length/cap_hit_rate": float((lengths >= response_length_cap).float().mean().item()),
    }
    _validate_finite_metrics(metrics, "reward curve")
    return metrics


def _validate_finite_metrics(metrics: Mapping[str, float], name: str) -> None:
    values = torch.tensor(list(metrics.values()), dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"response {name} metrics are not finite")


def _promoted_step_dirs(checkpoint_root: Path) -> list[Path]:
    """Return promoted step-NNNNNN checkpoint directories, oldest first."""

    prefix = "step-"
    return sorted(
        (
            entry
            for entry in checkpoint_root.iterdir()
            if entry.is_dir()
            and not entry.is_symlink()
            and entry.name.startswith(prefix)
            and len(entry.name) == len(prefix) + 6
            and entry.name[len(prefix) :].isdigit()
        ),
        key=lambda path: path.name,
    )


def _prune_checkpoints(checkpoint_root: Path, resume_path: Path | None) -> None:
    """Delete promoted checkpoints beyond the retention window and stale quarantine stages.

    Runs only after this step's own checkpoint has been promoted, so a failure here never
    risks the checkpoint just sealed. The checkpoint this run was resumed from, if any, is
    always kept regardless of age, since a crash immediately afterwards may still need it.
    """

    promoted = _promoted_step_dirs(checkpoint_root)
    stale = promoted[:-CHECKPOINT_RETENTION_COUNT] if len(promoted) > CHECKPOINT_RETENTION_COUNT else []
    resolved_resume = resume_path.resolve() if resume_path is not None else None
    for path in stale:
        if resolved_resume is not None and path.resolve() == resolved_resume:
            continue
        shutil.rmtree(path)
    for entry in checkpoint_root.iterdir():
        if entry.is_dir() and not entry.is_symlink() and entry.name.startswith(".quarantined-step-"):
            shutil.rmtree(entry)


def build_response_training_pipeline(config: Any, **kwargs: Any) -> ResponseTrainingPipeline:
    """Validate production response worker paths and construct the pipeline."""

    expected = (
        (config.actor_train, ACTOR_WORKER_PATH, "actor_train"),
        (config.actor_infer, INFER_WORKER_PATH, "actor_infer"),
    )
    for worker, path, name in expected:
        if worker.worker_cls != path:
            raise ValueError(f"response training requires {name}.worker_cls={path}")
    if config.actor_train.strategy_args.strategy_name != "fsdp2_train":
        raise ValueError("response training requires actor_train strategy fsdp2_train")
    if config.actor_infer.strategy_args.strategy_name != "vllm":
        raise ValueError("response training requires actor_infer strategy vllm")
    return ResponseTrainingPipeline(config, **kwargs)


def _cluster(worker: Any, resource_manager: Any) -> Cluster:
    return Cluster(
        name=worker.name,
        worker_cls=worker.worker_cls,
        resource_manager=resource_manager,
        worker_config=worker,
    )


def _load_domain_dataset(config: Any, tokenizer: Any) -> tuple[str, Any]:
    ratios = config.actor_train.data_args.domain_interleave_probs
    if not isinstance(ratios, Mapping) or len(ratios) != 1 or next(iter(ratios.values())) != 1.0:
        raise ValueError("response training requires exactly one full-weight domain")
    domain = next(iter(ratios))
    get_encode_function, preprocess_dataset, update_dataset_domain = _rlvr_dataset_helpers()
    dataset = _load_response_dataset(config.actor_train.data_args)
    template = config.global_template or config.actor_train.data_args.template
    dataset = preprocess_dataset(
        dataset,
        config.prompt_length,
        get_encode_function(template, tokenizer, config.actor_train.data_args),
        data_args=config.actor_train.data_args,
    )
    dataset = dataset.map(
        partial(update_dataset_domain, config.tag_2_domain),
        num_proc=config.actor_train.data_args.preprocessing_num_workers,
        desc="update_dataset_domain",
        load_from_cache_file=False,
    )
    selected = dataset.filter(
        lambda example, expected: example["domain"] == expected,
        num_proc=config.actor_train.data_args.preprocessing_num_workers,
        fn_kwargs={"expected": domain},
    )
    if len(selected) <= config.rollout_batch_size:
        raise ValueError("response training domain is too small for one rollout batch")
    return domain, selected.with_transform(_restore_rubrics)


def _rlvr_dataset_helpers() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Import RTT's dataset helpers only once the compat hook has run.

    RTT's rlvr pipeline imports a symbol its own utils module never defines, which the
    compat hook supplies. Ray workers import this module to unpickle scheduler callables
    without running that hook, so importing it at module scope kills every worker.
    """

    from roll.pipeline.rlvr.rlvr_pipeline import get_encode_function, preprocess_dataset, update_dataset_domain

    return get_encode_function, preprocess_dataset, update_dataset_domain


def _load_response_dataset(data_args: Any) -> Any:
    """Load response JSONL without exposing heterogeneous objects to Arrow."""

    return _load_jsonl_response_dataset(
        data_args.file_name,
        dataset_dir=getattr(data_args, "dataset_dir", "."),
    )


def _restore_rubrics(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    restored = dict(batch)
    restored["rubrics"] = [json.loads(value) for value in batch["rubrics"]]
    return restored


def _validate_inputs(config: Any, response: ResponseConfig, identity: CheckpointIdentity, stop: int | None) -> None:
    if not isinstance(response, ResponseConfig) or getattr(config, "rdan_response", None) != response:
        raise ValueError("response config sidecar is not bound to the RTT config")
    if config.resume_from_checkpoint:
        raise ValueError("response training forbids RTT stock resume_from_checkpoint")
    if identity.planned_horizon != config.max_steps or identity.method != response.method:
        raise ValueError("checkpoint identity differs from the response config")
    expected_weight = response.quality_weight if response.method in QUALITY_METHODS else response.mix_weight
    if identity.method_weight != expected_weight or identity.resolved_config_sha256 != response.resolved_config_sha256:
        raise ValueError("checkpoint identity differs from response method or config bytes")
    if stop is not None and (
        isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= stop <= identity.planned_horizon
    ):
        raise ValueError("stop_after_step must be within the planned horizon")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_value(value), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
        return values
    if isinstance(value, np.generic):
        return value.item()
    return value


def _number_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, Mapping):
            metrics.update({f"{name}/{key}": item for key, item in _number_metrics(value).items()})
            continue
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"training metric {name} is invalid")
        metrics[name] = float(tensor.mean().item())
    if not metrics:
        raise RuntimeError("training metrics are empty")
    return metrics


def _group_diagnostics(batch: DataProto, group_size: int) -> dict[str, int | float]:
    selected = batch.batch.get("rdan_selected_reward")
    quality = batch.batch.get("rdan_raw_quality")
    eligible = batch.batch.get("rdan_quality_eligible")
    if (
        not isinstance(selected, torch.Tensor)
        or not isinstance(quality, torch.Tensor)
        or not isinstance(eligible, torch.Tensor)
        or selected.ndim != 1
        or quality.ndim != 1
        or eligible.ndim != 1
        or selected.shape != quality.shape
        or eligible.shape != selected.shape
        or selected.numel() == 0
        or selected.numel() % group_size
    ):
        raise RuntimeError("response checkpoint requires grouped reward evidence")
    _validate_group_keys(batch, group_size)
    selected_groups = selected.detach().float().reshape(-1, group_size)
    quality_groups = quality.detach().float().reshape(-1, group_size)
    eligible_groups = eligible.detach().bool().reshape(-1, group_size)
    if not bool(torch.isfinite(selected_groups).all()) or not bool(torch.isfinite(quality_groups).all()):
        raise RuntimeError("response group rewards are not finite")
    selected_variance = selected_groups.var(dim=-1, unbiased=False)
    response_active = selected_variance > 1e-8
    eligible_count = eligible_groups.sum(dim=-1)
    quality_mean = (quality_groups * eligible_groups).sum(dim=-1) / eligible_count.clamp_min(1)
    quality_variance = ((quality_groups - quality_mean.unsqueeze(-1)).square() * eligible_groups).sum(
        dim=-1
    ) / eligible_count.clamp_min(1)
    quality_active = (eligible_count >= 2) & (quality_variance > 1e-8)
    group_count = selected_groups.shape[0]
    response_count = int(response_active.sum().item())
    if response_count == 0:
        raise RuntimeError("response checkpoint requires useful within-group selected-reward variance")
    return {
        "group_count": group_count,
        "response_active_group_count": response_count,
        "response_active_group_rate": response_count / group_count,
        "quality_active_group_count": int(quality_active.sum().item()),
        "quality_active_group_rate": float(quality_active.float().mean().item()),
        "selected_reward_variance_mean": float(selected_variance.mean().item()),
    }


def _validate_group_keys(batch: DataProto, group_size: int) -> None:
    values = batch.non_tensor_batch.get("rdan_prompt_key")
    if values is None or len(values) != len(batch):
        raise RuntimeError("response checkpoint requires one prompt key per response")
    seen: set[Any] = set()
    for start in range(0, len(batch), group_size):
        group = [_row_value(values, index) for index in range(start, start + group_size)]
        key = group[0]
        try:
            repeated = key in seen
        except TypeError as error:
            raise RuntimeError("response checkpoint prompt keys must be hashable") from error
        if repeated or any(value != key for value in group[1:]):
            raise RuntimeError("response checkpoint prompt groups are interleaved or repeated")
        seen.add(key)


def _clipping_fraction(metrics: Mapping[str, float]) -> float:
    value = metrics.get("rdan/response_token_clipfrac")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise RuntimeError("response checkpoint requires a valid response-token clipping fraction")
    return float(value)


def _cleanup(scheduler: Any, tracker: Any) -> BaseException | None:
    errors: list[BaseException] = []
    try:
        ray.get(scheduler.shutdown.remote())
    except BaseException as error:
        errors.append(error)
    try:
        tracker.finish()
    except BaseException as error:
        errors.append(error)
    return errors[0] if errors else None


def _row_value(values: Any, index: int) -> Any:
    value = values[index]
    return value.item() if isinstance(value, np.generic) else value
