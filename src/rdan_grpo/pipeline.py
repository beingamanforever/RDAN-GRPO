"""RDAN-GRPO training pipeline: rollout, rubric reward, decoupled advantage, FSDP2 update."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from rdan_grpo.checkpoint import (
    latest_checkpoint,
    promote_checkpoint,
    prune_checkpoints,
    read_state,
    stage_checkpoint,
)
from rdan_grpo.config import ResponseConfig, updates_per_step
from rdan_grpo.dataset import load_response_dataset
from rdan_grpo.train_step import TrainStepResult, run_train_step

# Recent checkpoints exist so a crash can resume; milestone checkpoints exist so intermediate
# Hugging Face weights survive for evaluation after the run.
KEEP_RECENT_CHECKPOINTS = 2
KEEP_EVERY_STEPS = 100


@dataclass(frozen=True)
class CompletedRun:
    """Result of a training run that reached its requested step."""

    completed_step: int
    checkpoints: tuple[Path, ...]


class RdanTrainingPipeline(BasePipeline):
    """Run RDAN-GRPO steps with local checkpointing and resume."""

    def __init__(
        self,
        pipeline_config: Any,
        *,
        response_config: ResponseConfig,
        checkpoint_root: str | Path,
        stop_after_step: int | None = None,
        resume: str | Path | bool = False,
    ) -> None:
        BasePipeline.model_update_groups = []
        BasePipeline.checkpoint_clusters = []
        super().__init__(pipeline_config)
        self.model_update_groups = []
        self.checkpoint_clusters = []
        self.response_config = response_config
        self.group_size = pipeline_config.num_return_sequences_in_group
        self.updates_per_step = updates_per_step(pipeline_config)
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.stop_after_step = stop_after_step or pipeline_config.max_steps

        self.resume_path = _resume_path(resume, self.checkpoint_root)
        self.completed_step = int(read_state(self.resume_path)["completed_step"]) if self.resume_path else 0
        if self.stop_after_step <= self.completed_step:
            raise ValueError(
                f"stop_after_step {self.stop_after_step} must advance beyond resumed step {self.completed_step}"
            )
        self.state.step = self.completed_step

        pipeline_config.set_max_steps(max_steps=pipeline_config.max_steps)
        self._initialize_runtime(pipeline_config)
        if self.resume_path:
            self._restore(self.resume_path)

    def _initialize_runtime(self, config: Any) -> None:
        self.tokenizer = default_tokenizer_provider(model_args=config.actor_train.model_args)
        self.domain, dataset = _load_dataset(config, self.tokenizer)
        self.actor_train = _cluster(config.actor_train.name, config.actor_train, self.resource_manager)
        self.actor_infer = _cluster(config.actor_infer.name, config.actor_infer, self.resource_manager)
        self.rewards = {
            name: _cluster(f"reward-{name}", worker, self.resource_manager) for name, worker in config.rewards.items()
        }
        self.download_models(self.actor_train, self.actor_infer, *self.rewards.values())
        self.scheduler = self._build_scheduler(config, dataset)

        ray.get(self.actor_infer.initialize(pipeline_config=config, blocking=False))
        reward_refs: list[Any] = []
        for reward in self.rewards.values():
            reward_refs.extend(reward.initialize(pipeline_config=config, blocking=False))
        ray.get(reward_refs)
        ray.get(self.actor_train.initialize(pipeline_config=config, blocking=False))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=config.actor_train.model_update_frequency,
        )

    def _build_scheduler(self, config: Any, dataset: Any) -> Any:
        state = read_state(self.resume_path)["scheduler"] if self.resume_path else None
        scheduler = (
            ray.remote(DynamicSamplingScheduler)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            )
            .remote(pipeline_config=config)
        )
        ray.get(
            scheduler.set_scheduler.remote(
                actor_cluster=self.actor_infer,
                reward_clusters={self.domain: self.rewards[self.domain]},
                dataset=dataset,
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs={"max_length": config.prompt_length, "padding": "max_length"},
                state=state,
            )
        )
        return scheduler

    @torch.no_grad()
    def run(self) -> CompletedRun:
        """Train through the requested step and return the promoted checkpoints."""

        promoted: list[Path] = []
        failed = False
        try:
            # The rollout engine starts from its own copy of the weights, so the trained actor
            # must be pushed across before the first rollout of a fresh or resumed run.
            self.model_update(self.completed_step)
            for step in range(self.completed_step + 1, self.stop_after_step + 1):
                rewarded, result = self._run_step(step)
                self.completed_step = step
                self.state.step = step
                metrics = self._step_metrics(step, rewarded, result)
                if self._should_save(step):
                    promoted.append(self._save(step, metrics))
                self.tracker.log(values=metrics, step=None)
        except BaseException:
            failed = True
            raise
        finally:
            # Shut down either way, but never let a cleanup error replace the training error
            # that caused it.
            shutdown_error = _shutdown(self.scheduler, self.tracker)
            if shutdown_error is not None and not failed:
                raise shutdown_error
        return CompletedRun(completed_step=self.completed_step, checkpoints=tuple(promoted))

    def _run_step(self, step: int) -> tuple[DataProto, TrainStepResult]:
        self.actor_train.rdan_reset_cuda_peak(blocking=True)
        self.actor_infer.rdan_reset_cuda_peak(blocking=True)
        rewarded = self._generate(step)
        result = run_train_step(
            actor_train=self.actor_train,
            rewarded_batch=rewarded,
            group_size=self.group_size,
            method=self.response_config.method,
            quality_weight=self.response_config.quality_weight,
            observe_memory=lambda: self.actor_train.rdan_cuda_memory(blocking=True),
        )
        # The rollout engine holds pre-update weights until this push; skipping it would train
        # on fresh weights while generating from stale ones for the rest of the run.
        self.model_update(step)
        return rewarded, result

    def _generate(self, step: int) -> DataProto:
        request = DataProto(
            meta_info={
                "global_step": step,
                "generation_config": self.pipeline_config.actor_infer.generating_args.to_dict(),
                "is_offload_states": False,
            }
        )
        self.actor_train.offload_states(blocking=True)
        self.actor_infer.load_states(blocking=True)
        for reward in self.rewards.values():
            reward.load_states(blocking=True)
        try:
            return ray.get(
                self.scheduler.get_batch_opt_level_0.remote(
                    data=request, batch_size=self.pipeline_config.rollout_batch_size
                ),
                timeout=self.pipeline_config.rpc_timeout,
            )
        finally:
            self.actor_infer.offload_states(blocking=True)
            for reward in self.rewards.values():
                reward.offload_states(blocking=True)
            self.actor_train.load_states(blocking=True)

    def _reward_stats(self) -> dict[str, float]:
        """Pull judge and checker health off the reward workers for this step."""

        from rdan_grpo.reward_worker import aggregate_reward_stats

        rows = [row for cluster in self.rewards.values() for row in cluster.rdan_reward_stats(blocking=True)]
        return aggregate_reward_stats(rows)

    def _step_metrics(self, step: int, rewarded: DataProto, result: TrainStepResult) -> dict[str, float]:
        scalar = result.scalar
        lengths = rewarded.batch["response_mask"].detach().sum(dim=-1).float()
        length_cap = self.pipeline_config.actor_infer.generating_args.max_new_tokens
        eligible_quality = scalar.raw_quality[scalar.quality_eligible]
        advantage = scalar.scalar_advantage
        metrics = {
            **result.metrics,
            **_scheduler_metrics(rewarded),
            **self._reward_stats(),
            "reward/selected_mean": float(scalar.selected_raw_reward.mean()),
            "reward/selected_std": float(scalar.selected_raw_reward.std(unbiased=False)),
            "reward/valid_rate": scalar.diagnostics["response_valid_rate"],
            "reward/hard_pass_rate": scalar.diagnostics["hard_pass_rate"],
            "reward/quality_eligible_rate": scalar.diagnostics["quality_eligible_rate"],
            "reward/outcome_advantage_std": float(scalar.response_advantage.std(unbiased=False)),
            "reward/process_quality_mean": float(eligible_quality.mean()) if eligible_quality.numel() else 0.0,
            "reward/process_advantage_std": float(scalar.quality_advantage.std(unbiased=False)),
            "advantage/mean": float(advantage.mean()),
            "advantage/std": float(advantage.std(unbiased=False)),
            "advantage/zero_rate": float((advantage.abs() <= 1e-8).float().mean()),
            "length/mean": float(lengths.mean()),
            "length/cap_hit_rate": float((lengths >= length_cap).float().mean()),
            "system/peak_memory_fraction": result.peak_memory_fraction,
            "system/prompts": float(result.prompt_count),
            "system/responses": float(result.response_count),
            "system/step": float(step),
        }
        return {name: value for name, value in metrics.items() if isinstance(value, (int, float))}

    def _should_save(self, step: int) -> bool:
        return step % self.pipeline_config.save_steps == 0 or step == self.stop_after_step

    def _save(self, step: int, metrics: Mapping[str, float]) -> Path:
        staging = stage_checkpoint(self.checkpoint_root, step)
        # ROLL's FSDP2 save writes DCP shards plus rank-0 safetensors and the tokenizer here,
        # so actor/ is directly loadable with from_pretrained for evaluation.
        self.actor_train.rdan_save(str(staging / "actor"), step, blocking=True)
        WorkerState.save_rng_state(str(staging / "rng"), "driver")
        scheduler_state = ray.get(self.scheduler.get_scheduler_state.remote())
        counters = self.actor_train.rdan_counters(blocking=True)
        promoted = promote_checkpoint(
            staging,
            {
                "completed_step": step,
                "method": self.response_config.method,
                "quality_weight": self.response_config.quality_weight,
                "updates_per_step": self.updates_per_step,
                "counters": list(counters),
                "scheduler": scheduler_state,
                "metrics": dict(metrics),
            },
        )
        prune_checkpoints(self.checkpoint_root, KEEP_RECENT_CHECKPOINTS, KEEP_EVERY_STEPS)
        return promoted

    def _restore(self, checkpoint: Path) -> None:
        self.actor_train.rdan_load(str(checkpoint / "actor"), blocking=True)
        WorkerState.load_rng_state(str(checkpoint / "rng"), "driver")


def build_pipeline(config: Any, **kwargs: Any) -> RdanTrainingPipeline:
    """Construct the training pipeline from a fully resolved config."""

    return RdanTrainingPipeline(config, response_config=config.rdan_response, **kwargs)


def _shutdown(scheduler: Any, tracker: Any) -> BaseException | None:
    """Stop the scheduler and close tracking, attempting both regardless of the first result."""

    errors: list[BaseException] = []
    for close in (lambda: ray.get(scheduler.shutdown.remote()), tracker.finish):
        try:
            close()
        except BaseException as error:  # noqa: BLE001 - collect, the caller decides
            errors.append(error)
    return errors[0] if errors else None


def _scheduler_metrics(rewarded: DataProto) -> dict[str, float]:
    """Forward the sampling metrics the scheduler stamps onto the assembled batch."""

    values = getattr(rewarded, "meta_info", {}).get("metrics")
    if not isinstance(values, Mapping):
        return {}
    return {name: float(value) for name, value in values.items() if isinstance(value, (int, float))}


def _resume_path(resume: str | Path | bool, root: Path) -> Path | None:
    if resume is False or resume is None:
        return None
    if resume is True:
        return latest_checkpoint(root)
    path = Path(resume).resolve()
    if not path.is_dir():
        raise ValueError(f"resume checkpoint does not exist: {path}")
    return path


def _cluster(name: str, worker_config: Any, resource_manager: Any) -> Cluster:
    return Cluster(
        name=name,
        worker_cls=worker_config.worker_cls,
        resource_manager=resource_manager,
        worker_config=worker_config,
    )


def _load_dataset(config: Any, tokenizer: Any) -> tuple[str, Any]:
    """Load the single training domain and encode it with ROLL's RLVR helpers."""

    probabilities = config.actor_train.data_args.domain_interleave_probs
    if len(probabilities) != 1:
        raise ValueError("RDAN training expects exactly one reward domain")
    domain = next(iter(probabilities))
    if domain not in config.rewards:
        raise ValueError(f"domain {domain} has no reward worker")

    # Imported lazily: this module reaches ROLL code that only works once the compat shim
    # has installed the helper RTT's own utils module never defines.
    from functools import partial

    from roll.pipeline.rlvr.rlvr_pipeline import get_encode_function, preprocess_dataset, update_dataset_domain

    # ROLL's DataArguments has no directory field: file_name is resolved against the
    # working directory the run was launched from.
    data_args = config.actor_train.data_args
    dataset = load_response_dataset(data_args.file_name)
    dataset = preprocess_dataset(
        dataset,
        config.prompt_length,
        get_encode_function(config.global_template or data_args.template, tokenizer, data_args),
        data_args=data_args,
    )
    dataset = dataset.map(
        partial(update_dataset_domain, config.tag_2_domain),
        num_proc=data_args.preprocessing_num_workers,
        desc="assign reward domain",
        load_from_cache_file=False,
    )
    selected = dataset.filter(
        lambda example, expected: example["domain"] == expected,
        num_proc=data_args.preprocessing_num_workers,
        fn_kwargs={"expected": domain},
    )
    if len(selected) <= config.rollout_batch_size:
        raise ValueError(f"domain {domain} has {len(selected)} rows, too few for one rollout batch")
    return domain, selected.with_transform(_restore_rubrics)


def _restore_rubrics(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Decode the rubric lists that were serialized to JSON strings for Arrow storage."""

    return {**batch, "rubrics": [json.loads(value) for value in batch["rubrics"]]}
