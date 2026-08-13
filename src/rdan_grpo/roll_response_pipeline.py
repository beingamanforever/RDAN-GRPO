"""Custom response-only production pipeline for the pinned RTT checkout."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import datasets
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
from roll.pipeline.rlvr.rlvr_pipeline import get_encode_function, preprocess_dataset, update_dataset_domain
from roll.utils.worker_state import WorkerState

from rdan_grpo.roll_response_checkpoint import (
    CheckpointIdentity,
    CheckpointState,
    create_checkpoint_stage,
    load_checkpoint,
    promote_checkpoint,
)
from rdan_grpo.roll_response_config import ACTOR_WORKER_PATH, INFER_WORKER_PATH, ResponseConfig
from rdan_grpo.roll_response_receipt import build_response_receipt
from rdan_grpo.roll_response_train import ResponseTrainResult, run_response_train_step
from rdan_grpo.roll_scalar import QUALITY_METHODS
from rdan_grpo.wandb_tracking import redact_secrets


class ResponseTrainingPipeline(BasePipeline):
    """Run receipted response-only optimizer transactions without RTT checkpointing."""

    def __init__(
        self,
        pipeline_config: Any,
        *,
        response_config: ResponseConfig,
        certificate: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
        model_identity: Mapping[str, Any],
        checkpoint_identity: CheckpointIdentity,
        checkpoint_root: str | Path,
        run_root: str | Path,
        artifact_root: str | Path,
        stop_after_step: int | None = None,
        resume_checkpoint: str | Path | None = None,
    ) -> None:
        _validate_inputs(pipeline_config, response_config, checkpoint_identity, stop_after_step)
        BasePipeline.model_update_groups = []
        BasePipeline.checkpoint_clusters = []
        super().__init__(pipeline_config)
        self.model_update_groups = []
        self.checkpoint_clusters = []
        self.response_config = response_config
        self.certificate = dict(certificate)
        self.runtime_identity = dict(runtime_identity)
        self.model_identity = dict(model_identity)
        self.checkpoint_identity = checkpoint_identity
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.artifact_root = _artifact_root(run_root, artifact_root, self.checkpoint_root)
        self.stop_after_step = stop_after_step or checkpoint_identity.planned_horizon
        self._resume_path = Path(resume_checkpoint).resolve() if resume_checkpoint is not None else None
        self._resume_manifest = (
            load_checkpoint(self._resume_path, identity=checkpoint_identity) if self._resume_path is not None else None
        )
        self.completed_step = int(self._resume_manifest["completed_step"]) if self._resume_manifest else 0
        if self.stop_after_step <= self.completed_step:
            raise ValueError("stop_after_step must advance beyond the resumed checkpoint")
        self.state.step = self.completed_step
        pipeline_config.set_max_steps(max_steps=pipeline_config.max_steps)

        self.tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args)
        domain, dataset = _load_domain_dataset(pipeline_config, self.tokenizer)
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

        scheduler_state = self._resume_manifest["scheduler_state"] if self._resume_manifest else None
        self.scheduler = (
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
            self.scheduler.set_scheduler.remote(
                actor_cluster=self.actor_infer,
                reward_clusters={domain: self.rewards[domain]},
                dataset=dataset,
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs={"max_length": pipeline_config.prompt_length, "padding": "max_length"},
                state=scheduler_state,
            )
        )
        self.domain = domain

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
        if self._resume_manifest:
            self._restore_checkpoint()

    @torch.no_grad()
    def run(self) -> list[Path]:
        """Run through the requested gate and return atomically promoted checkpoints."""

        promoted: list[Path] = []
        primary_error: BaseException | None = None
        try:
            initial_phase = "resume_initial" if self._resume_manifest else "initial"
            receipt = self._transfer(initial_phase, self.completed_step)
            for step in range(self.completed_step + 1, self.stop_after_step + 1):
                self.actor_train.rdan_reset_cuda_peak(blocking=True)
                self.actor_infer.rdan_reset_cuda_peak(blocking=True)
                rewarded = self._generate(step)
                observations: dict[str, Any] = {}

                def training_state() -> Sequence[Mapping[str, Any]]:
                    values = self.actor_train.rdan_training_state(blocking=True)
                    observations["training_state"] = values
                    return values

                def transfer() -> Mapping[str, Any]:
                    value = self._transfer("post_update", step)
                    observations["post_receipt"] = value
                    return value

                def memory() -> Sequence[Mapping[str, Any]]:
                    values = self.actor_train.rdan_cuda_memory(blocking=True)
                    observations["memory"] = values
                    return values

                result = run_response_train_step(
                    pipeline_config=self.pipeline_config,
                    actor_train=self.actor_train,
                    actor_infer=self.actor_infer,
                    rewarded_batch=rewarded,
                    certificate=self.certificate,
                    initial_receipt=receipt,
                    transfer_after_update=transfer,
                    observe_training_state=training_state,
                    observe_post_transaction_memory=memory,
                    method=self.response_config.method,
                    quality_weight=self.response_config.quality_weight,
                    mix_weight=self.response_config.mix_weight,
                )
                checkpoint = self._save_step(step, rewarded, receipt, result, observations)
                if checkpoint is not None:
                    promoted.append(checkpoint)
                self.completed_step = step
                self.state.step = step
                diagnostics = _group_diagnostics(rewarded, self.pipeline_config.num_return_sequences_in_group)
                metrics = {
                    **_number_metrics(result.metrics),
                    "reward/within_group_selected_variance_mean": diagnostics["selected_reward_variance_mean"],
                    "reward/response_active_group_rate": diagnostics["response_active_group_rate"],
                    "reward/quality_active_group_rate": diagnostics["quality_active_group_rate"],
                    "system/peak_memory_fraction": result.peak_memory_fraction,
                    "system/step": step,
                }
                self.state.log_history.append(metrics)
                self.tracker.log(values=metrics, step=step)
                receipt = observations["post_receipt"]
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error = _cleanup(self.scheduler, self.tracker)
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error
        return promoted

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

    def _transfer(self, phase: str, pipeline_step: int) -> dict[str, Any]:
        transaction_id = uuid.uuid4().hex
        ray.get(
            [worker.rdan_begin_response_receipt.remote(transaction_id) for worker in self.actor_train.workers]
            + [worker.rdan_begin_response_receipt.remote(transaction_id) for worker in self.actor_infer.workers]
        )
        self.model_update(pipeline_step)
        actors = ray.get([worker.rdan_get_response_receipt.remote() for worker in self.actor_train.workers])
        infers = ray.get([worker.rdan_finish_response_receipt.remote() for worker in self.actor_infer.workers])
        counters = self.actor_train.rdan_train_counters(blocking=True)
        receipt = build_response_receipt(
            actors,
            infers,
            phase=phase,
            pipeline_step=pipeline_step,
            actor_counters=counters,
            resolved_config_sha256=self.response_config.resolved_config_sha256,
            runtime_identity=self.runtime_identity,
            model_identity=self.model_identity,
            method=self.response_config.method,
            fixed_weight=self.response_config.fixed_weight,
        )
        self.actor_train.rdan_reset_response_receipt(blocking=True)
        self.actor_infer.rdan_reset_response_receipt(blocking=True)
        return receipt

    def _save_step(
        self,
        step: int,
        rewarded: DataProto,
        initial_receipt: Mapping[str, Any],
        result: ResponseTrainResult,
        observations: Mapping[str, Any],
    ) -> Path | None:
        if result.promotion_ready is not True:
            raise RuntimeError("response transaction is not ready for checkpoint promotion")
        metrics = _number_metrics(result.metrics)
        diagnostics = _group_diagnostics(rewarded, self.pipeline_config.num_return_sequences_in_group)
        clipping = _clipping_fraction(metrics)
        if clipping >= 1:
            raise RuntimeError("response checkpoint rejects 100 percent clipping")
        if self.response_config.method in QUALITY_METHODS and diagnostics["quality_active_group_rate"] < 0.1:
            raise RuntimeError("quality method checkpoint requires quality active group rate at least 0.1")
        save_checkpoint = step % self.pipeline_config.save_steps == 0 or step == self.stop_after_step
        artifact_dir = _seal_step_artifact(
            self.artifact_root,
            step=step,
            checkpoint_root=self.checkpoint_root,
            save_checkpoint=save_checkpoint,
            rewarded=rewarded,
            tokenizer=self.tokenizer,
            metrics={**metrics, "system/peak_memory_fraction": result.peak_memory_fraction},
            diagnostics=diagnostics,
            initial_receipt=initial_receipt,
            post_receipt=observations["post_receipt"],
        )
        checkpoint: Path | None = None
        if save_checkpoint:
            checkpoint = self._save_checkpoint(
                step,
                artifact_dir,
                initial_receipt,
                metrics,
                diagnostics,
                clipping,
                observations,
            )
        self.tracker.log_artifact(
            artifact_dir,
            name=f"qwen-{self.response_config.method.replace('_', '-')}-step-{step:06d}",
            artifact_type="training-step",
            aliases=(f"step-{step:06d}",),
            metadata={
                "step": step,
                "artifact_manifest_sha256": _file_sha256(artifact_dir / "manifest.json"),
                "checkpoint": checkpoint.name if checkpoint is not None else None,
                **diagnostics,
            },
        )
        return checkpoint

    def _save_checkpoint(
        self,
        step: int,
        artifact_dir: Path,
        initial_receipt: Mapping[str, Any],
        metrics: Mapping[str, float],
        diagnostics: Mapping[str, int | float],
        clipping: float,
        observations: Mapping[str, Any],
    ) -> Path:
        stage = create_checkpoint_stage(self.checkpoint_root, step)
        _write_json(
            stage / "artifacts/step.json",
            {
                "path": str(artifact_dir),
                "manifest_sha256": _file_sha256(artifact_dir / "manifest.json"),
            },
        )
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
        _write_json(stage / "receipts/initial.json", initial_receipt)
        _write_json(stage / "receipts/post-update.json", observations["post_receipt"])
        _write_json(stage / "metrics/step.json", metrics)
        counters = self.actor_train.rdan_train_counters(blocking=True)
        optimizer = {value["rank"]: value["optimizer_steps"] for value in counters}
        scheduler = {value["rank"]: value["scheduler_steps"] for value in counters}
        memory = {value["rank"]: value["peak_bytes"] for value in observations["memory"]}
        state = CheckpointState(
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
            receipt_links={"initial": "receipts/initial.json", "post_update": "receipts/post-update.json"},
        )
        artifacts = [path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()]
        return promote_checkpoint(stage, identity=self.checkpoint_identity, state=state, artifacts=artifacts)

    def _restore_checkpoint(self) -> None:
        assert self._resume_path is not None and self._resume_manifest is not None
        self.actor_train.rdan_load_dcp(str(self._resume_path / "actor"), blocking=True)
        WorkerState.load_rng_state(str(self._resume_path / "rng"), "driver")
        states = [torch.load(self._resume_path / f"rng/infer-rank-{rank}.pt", weights_only=False) for rank in range(2)]
        ray.get([worker.rdan_load_rng.remote(states[rank]) for rank, worker in enumerate(self.actor_infer.workers)])


def build_response_training_pipeline(config: Any, **kwargs: Any) -> ResponseTrainingPipeline:
    """Validate production response worker paths and construct the pipeline."""

    expected = (
        (config.actor_train, ACTOR_WORKER_PATH, "actor_train"),
        (config.actor_infer, INFER_WORKER_PATH, "actor_infer"),
    )
    for worker, path, name in expected:
        if worker.worker_cls != path:
            raise ValueError(f"response training requires {name}.worker_cls={path}")
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


def _load_response_dataset(data_args: Any) -> datasets.Dataset:
    """Load response JSONL without exposing heterogeneous objects to Arrow."""

    names = data_args.file_name if isinstance(data_args.file_name, list) else [data_args.file_name]
    dataset_dir = Path(getattr(data_args, "dataset_dir", "."))
    rows: list[dict[str, Any]] = []
    for name in names:
        path = Path(name)
        path = path if path.is_absolute() else dataset_dir / path
        if path.suffix not in {".json", ".jsonl"} or path.is_symlink() or not path.is_file():
            raise ValueError(f"response dataset must be a regular JSONL file: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid response dataset JSON at {path}:{line_number}") from error
                rows.append(_serialize_response_row(row, path, line_number))
    if not rows:
        raise ValueError("response dataset is empty")
    return datasets.Dataset.from_list(rows)


def _serialize_response_row(row: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"response dataset row must be an object at {path}:{line_number}")
    required = ("id", "prompt", "rubrics", "source", "ground_truth")
    if any(field not in row for field in required):
        raise ValueError(f"response dataset row is missing required fields at {path}:{line_number}")
    if isinstance(row["id"], bool) or not isinstance(row["id"], (int, str)) or not str(row["id"]):
        raise ValueError(f"response dataset id is invalid at {path}:{line_number}")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip() or not isinstance(row["source"], str):
        raise ValueError(f"response dataset prompt or source is invalid at {path}:{line_number}")
    rubrics = row["rubrics"]
    if not isinstance(rubrics, list) or not rubrics or any(not isinstance(rubric, dict) for rubric in rubrics):
        raise ValueError(f"response dataset rubrics are invalid at {path}:{line_number}")
    truth = row["ground_truth"]
    if isinstance(truth, str):
        try:
            truth = json.loads(truth)
        except json.JSONDecodeError as error:
            raise ValueError(f"response dataset ground_truth is invalid at {path}:{line_number}") from error
    if not isinstance(truth, dict):
        raise ValueError(f"response dataset ground_truth must be an object at {path}:{line_number}")
    normalized = dict(row)
    normalized["id"] = str(row["id"])
    normalized["rubrics"] = _canonical_json(rubrics)
    normalized["ground_truth"] = _canonical_json(truth)
    return normalized


def _restore_rubrics(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    restored = dict(batch)
    restored["rubrics"] = [json.loads(value) for value in batch["rubrics"]]
    return restored


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _artifact_root(run_value: str | Path, value: str | Path, checkpoint_root: Path) -> Path:
    run_path = Path(run_value)
    run_root = run_path.resolve()
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    if (
        run_path.is_symlink()
        or not run_root.is_dir()
        or path.is_symlink()
        or not resolved.is_dir()
        or resolved == run_root
        or not resolved.is_relative_to(run_root)
        or resolved == checkpoint_root
        or resolved.is_relative_to(checkpoint_root)
    ):
        raise ValueError("step artifacts require a real run-owned directory outside checkpoint root")
    return resolved


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
    for name in ("actor/ppo_ratio_clipfrac", "actor/clipfrac"):
        if name in metrics and 0 <= metrics[name] <= 1:
            return metrics[name]
    raise RuntimeError("response checkpoint requires a valid actor clipping fraction")


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


def _seal_step_artifact(
    root: Path,
    *,
    step: int,
    checkpoint_root: Path,
    save_checkpoint: bool,
    rewarded: DataProto,
    tokenizer: Any,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, int | float],
    initial_receipt: Mapping[str, Any],
    post_receipt: Mapping[str, Any],
) -> Path:
    stage = root / f".incomplete-step-{step:06d}"
    target = root / f"step-{step:06d}"
    if stage.is_symlink() or target.is_symlink():
        raise RuntimeError("step artifact paths must not be symlinks")
    _quarantine_artifact(stage, root, step)
    _quarantine_artifact(target, root, step)
    stage.mkdir(mode=0o700)
    try:
        _write_jsonl(stage / "responses.jsonl", _response_records(rewarded, tokenizer))
        _write_jsonl(stage / "groups.jsonl", _group_records(rewarded, diagnostics))
        _write_json(stage / "metrics.json", metrics)
        _write_json(stage / "diagnostics.json", diagnostics)
        _write_json(stage / "receipts/initial.json", redact_secrets(initial_receipt))
        _write_json(stage / "receipts/post-update.json", redact_secrets(post_receipt))
        files = [path for path in stage.rglob("*") if path.is_file()]
        inventory = [
            {
                "path": path.relative_to(stage).as_posix(),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in sorted(files)
        ]
        _write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "status": "sealed",
                "step": step,
                "checkpoint": {
                    "path": str(checkpoint_root / f"step-{step:06d}"),
                    "status": "pending_local_promotion" if save_checkpoint else "not_scheduled",
                },
                "inventory": inventory,
            },
        )
        os.rename(stage, target)
        _fsync_directory(root)
    except BaseException:
        if stage.exists():
            _fsync_directory(stage)
        raise
    return target


def _quarantine_artifact(path: Path, root: Path, step: int) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise RuntimeError("stale step artifact must be a real directory")
    attempt = 1
    while True:
        target = root / f".quarantined-step-{step:06d}-attempt-{attempt:06d}"
        if not target.exists() and not target.is_symlink():
            os.rename(path, target)
            _fsync_directory(root)
            return
        attempt += 1


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("step artifact JSONL must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            json.dump(_json_value(redact_secrets(row)), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _response_records(batch: DataProto, tokenizer: Any) -> list[dict[str, Any]]:
    required = (
        "rdan_raw_aon",
        "rdan_raw_csr",
        "rdan_raw_signed_csr",
        "rdan_selected_reward",
        "rdan_response_advantage",
        "rdan_raw_quality",
        "rdan_quality_eligible",
        "rdan_quality_advantage",
        "rdan_scalar_advantage",
        "rdan_response_valid",
        "rdan_scores",
        "rdan_rubric_mask",
        "rdan_eval_mask",
        "rdan_hard_mask",
        "rdan_judge_failed",
        "rdan_unsupported_hard",
    )
    missing = [name for name in required if name not in batch.batch]
    metadata = ("prompt", "rubrics", "source", "ground_truth", "rdan_prompt_key", "rdan_rubric_evidence")
    missing.extend(name for name in metadata if name not in batch.non_tensor_batch)
    if missing:
        raise RuntimeError(f"step artifact is missing response evidence: {', '.join(sorted(missing))}")
    token_rows = _response_tokens(batch)
    records = []
    for index, tokens in enumerate(token_rows):
        evidence = _row_value(batch.non_tensor_batch["rdan_rubric_evidence"], index)
        active_rubrics = int(batch.batch["rdan_rubric_mask"][index].sum().item())
        if (
            not isinstance(evidence, list)
            or len(evidence) != active_rubrics
            or any(not isinstance(item, Mapping) for item in evidence)
        ):
            raise RuntimeError("rubric evidence must match every active response rubric")
        response_text = tokenizer.decode(tokens, skip_special_tokens=True)
        safe_text = redact_secrets(response_text)
        safe_tokens: list[int] | str = tokens if safe_text == response_text else "[REDACTED]"
        records.append(
            {
                "response_index": index,
                "prompt_key": _row_value(batch.non_tensor_batch["rdan_prompt_key"], index),
                "generation_id": _generation_id(batch, index),
                "prompt": _row_value(batch.non_tensor_batch["prompt"], index),
                "source": _row_value(batch.non_tensor_batch["source"], index),
                "ground_truth": _row_value(batch.non_tensor_batch["ground_truth"], index),
                "rubrics": _row_value(batch.non_tensor_batch["rubrics"], index),
                "response_tokens": safe_tokens,
                "response_text": safe_text,
                "response_length": len(tokens),
                "reward": {
                    name.removeprefix("rdan_"): _tensor_row(batch.batch[name], index) for name in required[:10]
                },
                "rubric_outcomes": {
                    "scores": _tensor_row(batch.batch["rdan_scores"], index),
                    "rubric_mask": _tensor_row(batch.batch["rdan_rubric_mask"], index),
                    "eval_mask": _tensor_row(batch.batch["rdan_eval_mask"], index),
                    "hard_mask": _tensor_row(batch.batch["rdan_hard_mask"], index),
                    "evidence": evidence,
                },
                "failures": {
                    "judge_failed": bool(_tensor_row(batch.batch["rdan_judge_failed"], index)),
                    "unsupported_hard": bool(_tensor_row(batch.batch["rdan_unsupported_hard"], index)),
                },
            }
        )
    return records


def _response_tokens(batch: DataProto) -> list[list[int]]:
    input_ids = batch.batch.get("input_ids")
    mask = batch.batch.get("response_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(mask, torch.Tensor) or input_ids.shape != mask.shape:
        raise RuntimeError("step artifact requires aligned input_ids and response_mask")
    return [
        [int(token) for token in row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()]
        for row_ids, row_mask in zip(input_ids, mask, strict=True)
    ]


def _group_records(batch: DataProto, diagnostics: Mapping[str, int | float]) -> list[dict[str, Any]]:
    count = int(diagnostics["group_count"])
    size = len(batch) // count
    selected = batch.batch["rdan_selected_reward"].detach().float().reshape(count, size)
    quality = batch.batch["rdan_raw_quality"].detach().float().reshape(count, size)
    eligible = batch.batch["rdan_quality_eligible"].detach().bool().reshape(count, size)
    rows = []
    for group in range(count):
        active_quality = quality[group][eligible[group]]
        rows.append(
            {
                "group_index": group,
                "prompt_key": _row_value(batch.non_tensor_batch["rdan_prompt_key"], group * size),
                "selected_rewards": selected[group].tolist(),
                "selected_reward_variance": float(selected[group].var(unbiased=False).item()),
                "quality_eligible_count": int(eligible[group].sum().item()),
                "conditional_quality_variance": (
                    float(active_quality.var(unbiased=False).item()) if active_quality.numel() >= 2 else 0.0
                ),
            }
        )
    return rows


def _generation_id(batch: DataProto, index: int) -> Any:
    for name in ("generation_id", "request_id", "response_id"):
        values = batch.non_tensor_batch.get(name)
        if values is not None and len(values) == len(batch):
            return _row_value(values, index)
        values = batch.meta_info.get(name) or batch.meta_info.get(f"{name}s")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) == len(batch):
            return _row_value(values, index)
    return None


def _tensor_row(value: torch.Tensor, index: int) -> Any:
    return _json_value(value[index])


def _row_value(values: Any, index: int) -> Any:
    value = values[index]
    return value.item() if isinstance(value, np.generic) else value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
