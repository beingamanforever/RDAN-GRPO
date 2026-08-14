"""Live no-update scalar preflight on the pinned RTT rollout pipeline."""

# ruff: noqa: E402

from __future__ import annotations

import copy
import json
import math
import os
import uuid
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import datasets as hf_datasets
import ray
import torch

from rdan_grpo.roll_compat import install_rtt_compat

# Ray workers must install the pinned compatibility hook before importing ROLL pipelines.
_rtt_root = os.environ.get("RTT_ROOT")
if _rtt_root:
    install_rtt_compat(_rtt_root)

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import InferWorker
from roll.pipeline.rlvr import rlvr_rollout_pipeline as rtt_rollout_pipeline
from roll.pipeline.rlvr.actor_worker import ActorWorker
from roll.pipeline.rlvr.rlvr_rollout_pipeline import RLVRRolloutPipeline
from roll.platforms import current_platform
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import concatenate_input_and_output, postprocess_generate
from roll.utils.offload_states import OffloadStateType
from torch.nn.utils.rnn import pad_sequence

from rdan_grpo.response_dataset import load_response_dataset
from rdan_grpo.response_sampling import balanced_preflight_indices
from rdan_grpo.roll_bridge import assess_scalar_batch
from rdan_grpo.roll_scalar import ScalarMethod
from rdan_grpo.runtime_parity import ParityObservation
from rdan_grpo.weight_receipt import (
    RECEIPT_WORKER_EXTENSION,
    build_weight_receipt_artifact,
)
from rdan_grpo.weight_receipt import (
    seal_weight_receipt as write_weight_receipt,
)


class ObservedLogprobInferWorker(InferWorker):
    """Parity-only vLLM worker that preserves observed token logprobs."""

    async def generate_request(self, data: DataProto) -> DataProto:
        """Reject any request whose vLLM output lacks aligned sampled-token logprobs."""

        generation_config = data.meta_info.get("generation_config")
        if not isinstance(generation_config, dict) or generation_config.get("logprobs") != 1:
            raise RuntimeError("parity rollout requires logprobs=1")
        output = await super().generate_request(data)
        token_ids = output.meta_info.get("output_token_ids")
        logprobs = output.meta_info.get("output_logprobs")
        if not isinstance(token_ids, list) or not isinstance(logprobs, list):
            raise RuntimeError("vLLM parity response is missing token logprobs")
        if len(token_ids) != len(logprobs) or any(
            len(tokens) != len(scores) for tokens, scores in zip(token_ids, logprobs, strict=True)
        ):
            raise RuntimeError("vLLM parity token and logprob boundaries differ")
        return output

    async def rdan_begin_weight_receipt(self, transaction_id: str, actor_rank: int) -> Any:
        """Begin the paired transaction on this outer worker and its vLLM engine."""

        from rdan_grpo.roll_weight_receipt import begin_infer_weight_receipt

        return await begin_infer_weight_receipt(self, transaction_id, actor_rank)

    async def rdan_get_weight_receipt(self) -> Any:
        """Fetch the loader receipt through the outer worker boundary."""

        from rdan_grpo.roll_weight_receipt import get_infer_weight_receipt

        return await get_infer_weight_receipt(self)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def generate(self, data: DataProto) -> DataProto:
        """Generate through vLLM and attach its sampled-token logprobs."""

        generation_config = copy.deepcopy(data.meta_info.get("generation_config"))
        if not isinstance(generation_config, dict) or generation_config.get("logprobs") != 1:
            raise RuntimeError("parity rollout requires logprobs=1")
        if self.worker_config.strategy_args.strategy_name != "vllm":
            raise RuntimeError("parity rollout requires the vLLM strategy")
        generation_config["eos_token_id"] = list(
            dict.fromkeys(
                token_id
                for token_id in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
                if token_id is not None
            )
        )
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        data = data.to(current_platform.device_type)
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

        await self.strategy.load_states()
        try:
            token_ids: list[list[int]] = []
            logprobs: list[list[float]] = []
            for index in range(len(data)):
                request = data[index : index + 1]
                request.meta_info = {
                    **request.meta_info,
                    "generation_config": generation_config,
                    "request_id": f"parity-{self.worker_name}-{uuid.uuid4().hex}",
                }
                output = await self.strategy.generate_request(data=request)
                finish_reasons = output.meta_info.get("finish_reasons")
                request_tokens = output.meta_info.get("output_token_ids")
                request_logprobs = output.meta_info.get("output_logprobs")
                if (
                    not isinstance(finish_reasons, list)
                    or any(reason == "abort" for reason in finish_reasons)
                    or len(finish_reasons) != generation_config["num_return_sequences"]
                ):
                    raise RuntimeError("parity rollout did not finish")
                if not isinstance(request_tokens, list) or not isinstance(request_logprobs, list):
                    raise RuntimeError("vLLM parity response is missing token logprobs")
                if len(request_tokens) != len(request_logprobs) or any(
                    len(tokens) != len(scores) for tokens, scores in zip(request_tokens, request_logprobs, strict=True)
                ):
                    raise RuntimeError("vLLM parity token and logprob boundaries differ")
                token_ids.extend(request_tokens)
                logprobs.extend(request_logprobs)

            expected = len(data) * generation_config["num_return_sequences"]
            if len(token_ids) != expected:
                raise RuntimeError(f"parity rollout expected {expected} responses, received {len(token_ids)}")
            output_tokens = [torch.tensor(tokens, device=data.batch.device) for tokens in token_ids]
            padded = pad_sequence(
                output_tokens,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            output_tensor = concatenate_input_and_output(
                input_ids=data.batch["input_ids"],
                output_ids=padded,
                num_return_sequences=generation_config["num_return_sequences"],
            )
            result = postprocess_generate(
                prompts=data,
                output=output_tensor,
                num_return_sequences=generation_config["num_return_sequences"],
                sequence_length=self.pipeline_config.sequence_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                output_logprobs=logprobs,
            )
            result.meta_info = {
                **data.meta_info,
                "infer_logprobs_source": "observed_rollout_engine",
            }
            return result.to("cpu")
        finally:
            data.to("cpu")
            if data.meta_info.get("is_offload_states", True):
                await self.strategy.offload_states()


class ObservedActorWorker(ActorWorker):
    """Parity-only actor that returns tensors observed by its forward callback."""

    def setup_model_update(self, infer_cluster: Any, model_update_name: str) -> None:
        from rdan_grpo.roll_weight_receipt import bind_actor_weight_updater

        super().setup_model_update(infer_cluster=infer_cluster, model_update_name=model_update_name)
        bind_actor_weight_updater(self, model_update_name)

    def rdan_begin_weight_receipt(self, transaction_id: str, infer_rank: int) -> None:
        from rdan_grpo.roll_weight_receipt import begin_actor_weight_receipt

        begin_actor_weight_receipt(self, transaction_id, infer_rank)

    def rdan_get_weight_receipt(self) -> dict[str, Any]:
        from rdan_grpo.roll_weight_receipt import get_actor_weight_receipt

        return get_actor_weight_receipt(self)

    def start_model_update(self, model_update_name: str) -> DataProto:
        from rdan_grpo.roll_weight_receipt import run_receipted_actor_update

        return run_receipted_actor_update(
            self,
            model_update_name,
            lambda: super(ObservedActorWorker, self).start_model_update(model_update_name),
        )

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def compute_log_probs(self, data: DataProto) -> DataProto:
        """Recompute logprobs and return the exact actor-seen token boundaries."""

        metrics: dict[str, Any] = {}
        is_offload_states = data.meta_info.get("is_offload_states", True)
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = self.strategy.get_data_input(data)
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results = self.strategy.forward_step(batch=data, forward_func=self.forward_func_log_probs)
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            required = {
                "log_probs",
                "entropy",
                "actor_input_ids",
                "actor_attention_mask",
                "actor_response_mask",
            }
            if set(results) < required:
                raise RuntimeError(f"actor parity result is missing fields: {sorted(required - set(results))}")
            output = DataProto.from_dict(tensors={name: results[name] for name in required}).to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_log_probs(
        self, data: DataProto, output_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Capture the token tensors passed to the actor logprob callback."""

        log_probs, results = super().forward_func_log_probs(data, output_tensor)
        results.update(
            actor_input_ids=data.batch["input_ids"].clone().detach(),
            actor_attention_mask=data.batch["attention_mask"].clone().detach(),
            actor_response_mask=data.batch["response_mask"].clone().detach(),
        )
        return log_probs, results


class RuntimeParityPipeline(BasePipeline):
    """Construct the production actor topology but never call a training step."""

    def __init__(self, pipeline_config: Any):
        if pipeline_config.actor_train.worker_cls is not ObservedActorWorker:
            raise ValueError("runtime parity requires ObservedActorWorker")
        if pipeline_config.actor_infer.worker_cls is not ObservedLogprobInferWorker:
            raise ValueError("runtime parity requires ObservedLogprobInferWorker")
        domain_ratios = pipeline_config.actor_train.data_args.domain_interleave_probs
        if not isinstance(domain_ratios, Mapping) or len(domain_ratios) != 1:
            raise ValueError("runtime parity requires exactly one training domain")
        domain = next(iter(domain_ratios))

        BasePipeline.__init__(self, pipeline_config)
        self.model_update_groups = []
        self.checkpoint_clusters = []
        self.rewards: dict[str, Any] = {}
        strategy_config = pipeline_config.actor_infer.strategy_args.strategy_config
        self._weight_receipt_required = strategy_config.get("worker_extension_cls") == RECEIPT_WORKER_EXTENSION
        self._weight_receipt_passed = False
        self.tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args)
        dataset = _load_runtime_parity_dataset(pipeline_config, self.tokenizer)
        if "domain" not in dataset.column_names:
            raise ValueError("runtime parity dataset must contain a domain field")
        domain_dataset = dataset.filter(
            lambda example, expected: example["domain"] == expected,
            num_proc=pipeline_config.actor_train.data_args.preprocessing_num_workers,
            fn_kwargs={"expected": domain},
        )
        if not domain_dataset:
            raise ValueError(f"runtime parity domain dataset {domain} has no data")
        if pipeline_config.max_steps <= 0:
            raise ValueError("runtime parity requires a positive production max_steps")
        pipeline_config.set_max_steps(max_steps=pipeline_config.max_steps)

        self.actor_train = Cluster(
            name=pipeline_config.actor_train.name,
            worker_cls=pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=pipeline_config.actor_train,
        )
        self.actor_infer = Cluster(
            name=pipeline_config.actor_infer.name,
            worker_cls=pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=pipeline_config.actor_infer,
        )
        self.download_models(self.actor_train, self.actor_infer)

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
        empty_reward = SimpleNamespace(workers=())
        if empty_reward.workers or self.rewards:
            raise RuntimeError("runtime parity requires empty reward workers")
        ray.get(
            scheduler.set_scheduler.remote(
                actor_cluster=self.actor_infer,
                reward_clusters={domain: empty_reward},
                dataset=domain_dataset,
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs={"max_length": pipeline_config.prompt_length, "padding": "max_length"},
                state=self.state.kv.get(f"scheduler_state_{domain}"),
            )
        )
        self.generate_schedulers = {domain: scheduler}

        ray.get(self.actor_infer.initialize(pipeline_config=pipeline_config, blocking=False))
        ray.get(self.actor_train.initialize(pipeline_config=pipeline_config, blocking=False))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=pipeline_config.actor_train.model_update_frequency,
        )
        self.set_checkpoint_clusters(self.actor_train)

    def seal_weight_receipt(
        self,
        output: str | Path,
        *,
        model_identity: Any,
        resolved_config_sha256: str,
        rtt_revision: str,
        rtt_boundary_sha256: Mapping[str, str],
    ) -> dict[str, Any]:
        """Run exactly one step-zero update and seal its transport and loader receipts."""

        if not self._weight_receipt_required:
            raise RuntimeError("runtime parity config is missing the receipt worker extension")
        _validate_receipt_topology(self.actor_train, self.actor_infer)
        transaction_id = uuid.uuid4().hex
        actor_receipts: list[Mapping[str, Any]] = []
        infer_receipts: list[Mapping[str, Any]] = []
        update_error: str | None = None
        state_before = self.state.step
        try:
            ray.get(
                [
                    worker.rdan_begin_weight_receipt.remote(transaction_id, rank)
                    for rank, worker in enumerate(self.actor_train.workers)
                ]
            )
            self.actor_train.offload_states(blocking=True)
            ray.get(
                [
                    worker.rdan_begin_weight_receipt.remote(transaction_id, rank)
                    for rank, worker in enumerate(self.actor_infer.workers)
                ]
            )
            self.model_update(0)
            if self.state.step != state_before:
                raise RuntimeError("weight receipt changed the pipeline training step")
        except Exception as error:
            update_error = type(error).__name__
        try:
            actor_receipts = ray.get([worker.rdan_get_weight_receipt.remote() for worker in self.actor_train.workers])
            infer_results = ray.get([worker.rdan_get_weight_receipt.remote() for worker in self.actor_infer.workers])
            infer_receipts = _flatten_infer_receipts(infer_results)
        except Exception as error:
            update_error = update_error or f"receipt_fetch_{type(error).__name__}"
        artifact = build_weight_receipt_artifact(
            actor_receipts,
            infer_receipts,
            model_identity=model_identity,
            resolved_config_sha256=resolved_config_sha256,
            rtt_revision=rtt_revision,
            rtt_boundary_sha256=rtt_boundary_sha256,
            transaction_id=transaction_id,
            update_error=update_error,
        )
        write_weight_receipt(output, artifact)
        self._weight_receipt_passed = True
        return artifact

    @torch.no_grad()
    def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> ParityObservation:
        """Generate once and independently recompute logprobs on actor-train."""

        returns = generation_config.get("num_return_sequences")
        if not isinstance(returns, int) or returns < 1:
            raise ValueError("runtime parity requires a positive return count")
        prompt_count = math.ceil(responses / returns)
        if len(self.generate_schedulers) != 1:
            raise ValueError("runtime parity requires exactly one training domain")
        scheduler = next(iter(self.generate_schedulers.values()))
        state_before = self.state.step
        request = DataProto(
            meta_info={
                "global_step": 0,
                "generation_config": dict(generation_config),
                "is_offload_states": False,
                "skip_rewards": True,
            }
        )

        self.actor_train.offload_states(blocking=True)
        if self._weight_receipt_required:
            if not self._weight_receipt_passed:
                raise RuntimeError("weight receipt must pass before generation")
        else:
            self.model_update(0)
        self.actor_infer.load_states(blocking=True)
        try:
            batch = ray.get(
                scheduler.get_batch.remote(data=request, global_step=0, batch_size=prompt_count),
                timeout=self.pipeline_config.rpc_timeout,
            )
        finally:
            self.actor_infer.offload_states(blocking=True)
            ray.get(scheduler.shutdown.remote())
        if "infer_logprobs" not in batch.batch:
            raise RuntimeError("runtime parity batch is missing infer_logprobs")

        rollout_ids = batch.batch["input_ids"].clone()
        rollout_attention = batch.batch["attention_mask"].clone()
        rollout_response = batch.batch["response_mask"].clone()
        recomputed = self.actor_train.compute_log_probs(batch.clone(), blocking=True)
        required = {
            "log_probs",
            "actor_input_ids",
            "actor_attention_mask",
            "actor_response_mask",
        }
        if recomputed.batch is None or set(recomputed.batch.keys()) < required:
            raise RuntimeError("runtime parity actor did not return observed token boundaries")
        if self.state.step != state_before:
            raise RuntimeError("runtime parity changed the pipeline training step")
        return ParityObservation(
            input_ids=rollout_ids,
            attention_mask=rollout_attention,
            response_mask=rollout_response,
            infer_logprobs=batch.batch["infer_logprobs"].clone(),
            actor_logprobs=recomputed.batch["log_probs"].clone(),
            actor_input_ids=recomputed.batch["actor_input_ids"].clone(),
            actor_attention_mask=recomputed.batch["actor_attention_mask"].clone(),
            actor_response_mask=recomputed.batch["actor_response_mask"].clone(),
            infer_logprobs_source="observed_rollout_engine",
            actor_train_recomputed=True,
            actor_boundary_observed=True,
            optimizer_updates=0,
        )


def _load_runtime_parity_dataset(config: Any, tokenizer: Any) -> Any:
    from roll.pipeline.rlvr.rubircs_pipeline import get_encode_function, preprocess_dataset, update_dataset_domain

    data_args = config.actor_train.data_args
    dataset = load_response_dataset(
        data_args.file_name,
        dataset_dir=getattr(data_args, "dataset_dir", "."),
    )
    template = config.global_template or data_args.template
    dataset = preprocess_dataset(
        dataset,
        config.prompt_length,
        get_encode_function(template, tokenizer, data_args),
        data_args=data_args,
    )
    return dataset.map(
        partial(update_dataset_domain, config.tag_2_domain),
        num_proc=data_args.preprocessing_num_workers,
        desc="update_dataset_domain",
        load_from_cache_file=False,
    )


def _flatten_infer_receipts(results: list[Any]) -> list[Mapping[str, Any]]:
    receipts: list[Mapping[str, Any]] = []
    for result in results:
        values = result if isinstance(result, list) else [result]
        if len(values) != 1 or not isinstance(values[0], Mapping):
            raise RuntimeError("weight receipt requires exactly one vLLM TP rank per infer worker")
        receipts.append(values[0])
    return receipts


def _validate_receipt_topology(actor_train: Any, actor_infer: Any) -> None:
    if len(actor_train.workers) != 2 or len(actor_infer.workers) != 2:
        raise RuntimeError("weight receipt requires two actor and two infer workers")
    for name, cluster in (("actor", actor_train), ("infer", actor_infer)):
        ranks = cluster.worker_rank_info
        if len(ranks) != 2 or any(rank.tp_size != 1 or rank.pp_size != 1 or rank.dp_size != 2 for rank in ranks):
            raise RuntimeError(f"weight receipt requires {name} DP2 TP1 PP1")
        if list(cluster.worker_config.device_mapping) != [0, 1]:
            raise RuntimeError(f"weight receipt requires {name} device mapping [0, 1]")


class ScalarPreflightPipeline(RLVRRolloutPipeline):
    """Construct only actor inference and reward workers for a frozen rollout sample."""

    prompt_count = 256
    group_size = 8

    def __init__(self, pipeline_config: Any):
        original_datasets = rtt_rollout_pipeline.datasets
        original_scheduler = rtt_rollout_pipeline.DynamicSamplingScheduler
        rtt_rollout_pipeline.datasets = _ResponseDatasetProxy(pipeline_config.validation.data_args)
        rtt_rollout_pipeline.DynamicSamplingScheduler = _ResponseDynamicSamplingScheduler
        try:
            super().__init__(pipeline_config)
        finally:
            rtt_rollout_pipeline.datasets = original_datasets
            rtt_rollout_pipeline.DynamicSamplingScheduler = original_scheduler
        self.val_dataset = self.val_dataset.with_transform(_restore_rubrics)
        if hasattr(self, "actor_train"):
            raise RuntimeError("no-update preflight constructed an actor training cluster")
        if len(self.val_dataset) < self.prompt_count:
            raise ValueError("preflight validation dataset contains fewer than 256 prompts")
        indices = balanced_preflight_indices(self.val_dataset["source"], self.prompt_count)
        self.val_dataset = self.val_dataset.select(indices)
        returns = self.pipeline_config.validation.generating_args.num_return_sequences
        if returns != self.group_size:
            raise ValueError("preflight requires exactly eight responses per prompt")

    @torch.no_grad()
    def collect_scalar_batch(self) -> DataProto:
        """Run one live rollout with no actor training cluster or optimizer calls."""

        request = DataProto(
            meta_info={
                "is_offload_states": False,
                "generation_config": self.pipeline_config.validation.generating_args.to_dict(),
            }
        )
        self.actor_infer.load_states()
        for reward in self.rewards.values():
            reward.load_states()
        try:
            batch = ray.get(
                self.val_generate_scheduler.get_batch.remote(
                    data=request,
                    global_step=0,
                    batch_size=self.prompt_count,
                ),
                timeout=self.pipeline_config.rpc_timeout,
            )
        finally:
            self.actor_infer.offload_states()
            for reward in self.rewards.values():
                reward.offload_states()
            ray.get(self.val_generate_scheduler.shutdown.remote())
        if hasattr(self, "actor_train"):
            raise RuntimeError("no-update preflight acquired an actor training cluster")
        expected = self.prompt_count * self.group_size
        if len(batch) != expected:
            raise ValueError(f"preflight expected {expected} responses, received {len(batch)}")
        return batch


class _ResponseDatasetProxy:
    """Load only the configured response JSON files through the canonical schema."""

    def __init__(self, data_args: Any) -> None:
        self._data_args = data_args
        self._data_files = _normalize_data_files(data_args.file_name)

    def load_dataset(
        self,
        dataset_type: str,
        *,
        data_files: Any,
        **kwargs: Any,
    ) -> hf_datasets.DatasetDict:
        """Return the canonical response dataset only for the exact pinned call."""

        if dataset_type != "json" or kwargs or _normalize_data_files(data_files) != self._data_files:
            raise ValueError("preflight dataset loader received an unconfigured request")
        dataset = load_response_dataset(
            self._data_args.file_name,
            dataset_dir=getattr(self._data_args, "dataset_dir", "."),
        )
        return hf_datasets.DatasetDict({"train": dataset})


class _ResponseDynamicSamplingScheduler(DynamicSamplingScheduler):
    """Restore rubric objects only after the pinned RTT dataset maps complete."""

    async def set_scheduler(
        self,
        actor_cluster: Any,
        reward_clusters: dict[str, Any],
        dataset: hf_datasets.Dataset,
        collect_fn_cls: Any,
        collect_fn_kwargs: dict[str, Any],
        state: dict[str, Any] | None = None,
        is_val: bool = False,
    ) -> None:
        """Pass a decoded-on-access dataset to the validation scheduler."""

        if not is_val:
            raise ValueError("response dataset scheduler is validation-only")
        await super().set_scheduler(
            actor_cluster=actor_cluster,
            reward_clusters=reward_clusters,
            dataset=dataset.with_transform(_restore_rubrics),
            collect_fn_cls=collect_fn_cls,
            collect_fn_kwargs=collect_fn_kwargs,
            state=state,
            is_val=is_val,
        )


def _normalize_data_files(data_files: Any) -> tuple[str, ...]:
    values = (data_files,) if isinstance(data_files, (str, Path)) else tuple(data_files)
    if not values or any(not isinstance(value, (str, Path)) for value in values):
        raise ValueError("preflight data files must be configured paths")
    return tuple(str(value) for value in values)


def _restore_rubrics(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    if "rubrics" not in batch:
        return batch
    restored = dict(batch)
    restored["rubrics"] = [json.loads(value) for value in batch["rubrics"]]
    return restored


def seal_live_batch(
    batch: Any,
    *,
    evaluator_path: str | Path,
    restricted_path: str | Path,
    method: ScalarMethod = "rdan_scalar",
    quality_weight: float | None = None,
    mix_weight: float | None = None,
) -> Any:
    """Seal compact evaluator rows and response token IDs in separate immutable files."""

    fields = ("rdan_scores", "rdan_rubric_mask", "rdan_eval_mask", "rdan_hard_mask")
    missing = [name for name in fields if name not in batch.batch]
    keys = batch.non_tensor_batch.get("rdan_prompt_key")
    if keys is None:
        missing.append("rdan_prompt_key")
    if missing:
        raise ValueError(f"live rollout is missing scalar fields: {', '.join(missing)}")
    assessment = assess_scalar_batch(
        list(keys),
        batch.batch["rdan_scores"].float().cpu(),
        batch.batch["rdan_rubric_mask"].bool().cpu(),
        batch.batch["rdan_eval_mask"].bool().cpu(),
        batch.batch["rdan_hard_mask"].bool().cpu(),
        unsupported_hard=batch.batch.get("rdan_unsupported_hard", None),
        judge_failed=batch.batch.get("rdan_judge_failed", None),
        group_size=8,
        method=method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )
    evaluator_rows = []
    for index, key in enumerate(keys):
        evaluator_rows.append(
            {
                "prompt_key": str(key),
                "scores": batch.batch["rdan_scores"][index].float().cpu().tolist(),
                "rubric_mask": batch.batch["rdan_rubric_mask"][index].bool().cpu().tolist(),
                "eval_mask": batch.batch["rdan_eval_mask"][index].bool().cpu().tolist(),
                "hard_mask": batch.batch["rdan_hard_mask"][index].bool().cpu().tolist(),
                "unsupported_hard": bool(batch.batch["rdan_unsupported_hard"][index]),
                "judge_failed": bool(batch.batch["rdan_judge_failed"][index]),
            }
        )
    restricted_rows = [
        {
            "prompt_key": str(key),
            "response_token_ids": batch.batch["responses"][index].cpu().tolist(),
        }
        for index, key in enumerate(keys)
    ]
    _write_jsonl_once(evaluator_path, evaluator_rows, 0o644)
    _write_jsonl_once(restricted_path, restricted_rows, 0o600)
    return assessment


def run_live_preflight(pipeline_cls: type, pipeline_config: Any) -> Any:
    """Construct the no-update pipeline and collect exactly one evidence batch."""

    if not isinstance(pipeline_cls, type) or not issubclass(pipeline_cls, ScalarPreflightPipeline):
        raise TypeError("live preflight requires ScalarPreflightPipeline")
    actor_train = getattr(pipeline_config, "actor_train", None)
    if actor_train is None or getattr(actor_train, "device_mapping", None) != []:
        raise ValueError("live preflight requires actor_train.device_mapping to resolve to an empty list")
    if getattr(pipeline_config, "max_steps", None) != 0:
        raise ValueError("live preflight requires max_steps=0")
    pipeline = pipeline_cls(pipeline_config)
    if hasattr(pipeline, "actor_train"):
        raise RuntimeError("live preflight constructed actor_train")
    return pipeline.collect_scalar_batch()


def _write_jsonl_once(path: str | Path, rows: list[dict[str, Any]], mode: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
