"""Live no-update FSDP2 to production vLLM parity boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import ray
import torch
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline

from rdan_grpo.response_dataset import load_response_dataset
from rdan_grpo.roll_live import ObservedLogprobInferWorker
from rdan_grpo.roll_response_receipt import build_response_receipt
from rdan_grpo.roll_response_workers import (
    ResponseActorWorker,
    ResponseVLLMInferWorker,
    _base_seed,
    _generation_seed,
    _rank,
)
from rdan_grpo.roll_same_backend import ObservedFSDP2ActorWorker
from rdan_grpo.runtime_parity import ParityObservation, write_artifact
from rdan_grpo.vllm_runtime_parity import INFER_LOGPROBS_SOURCE

ACTOR_WORKER_PATH = "rdan_grpo.roll_vllm_parity_live.ObservedResponseActorWorker"
INFER_WORKER_PATH = "rdan_grpo.roll_vllm_parity_live.ObservedResponseVLLMInferWorker"


class ObservedResponseActorWorker(ObservedFSDP2ActorWorker, ResponseActorWorker):
    """Reuse the production actor receipt while exposing exact FSDP2 boundaries."""


class ObservedResponseVLLMInferWorker(ObservedLogprobInferWorker, ResponseVLLMInferWorker):
    """Reuse production vLLM loading while preserving sampled-token logprobs."""

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def generate(self, data: DataProto) -> DataProto:
        """Generate through the observed vLLM request path and name its source."""

        from rdan_grpo.roll_compat import install_vllm_sampling_seed_compat

        install_vllm_sampling_seed_compat()
        request = data.clone()
        generation_config = copy.deepcopy(request.meta_info.get("generation_config"))
        if not isinstance(generation_config, dict):
            raise RuntimeError("vLLM parity generation config is unavailable")
        generation_config["seed"] = _generation_seed(_base_seed(self), 1, 0, _rank(self))
        request.meta_info["generation_config"] = generation_config
        output = await ObservedLogprobInferWorker.generate(self, request)
        if output.batch is None or "infer_logprobs" not in output.batch:
            raise RuntimeError("vLLM parity generation is missing sampled-token logprobs")
        output.meta_info["infer_logprobs_source"] = INFER_LOGPROBS_SOURCE
        return output


class VLLMRuntimeParityPipeline(BasePipeline):
    """Construct production DP2 actor and vLLM roles without rewards or training."""

    def __init__(self, pipeline_config: Any, response_config: Any) -> None:
        _validate_config(pipeline_config)
        super().__init__(pipeline_config)
        self.model_update_groups = []
        self.checkpoint_clusters = []
        self.response_config = response_config
        self._receipt_passed = False
        self._generation_started = False
        self.tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args)
        self.dataset = _load_dataset(pipeline_config, self.tokenizer)
        self.actor_train = _cluster(pipeline_config.actor_train, self.resource_manager)
        self.actor_infer = _cluster(pipeline_config.actor_infer, self.resource_manager)
        self.download_models(self.actor_train, self.actor_infer)
        ray.get(self.actor_infer.initialize(pipeline_config=pipeline_config, blocking=False))
        ray.get(self.actor_train.initialize(pipeline_config=pipeline_config, blocking=False))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=pipeline_config.actor_train.model_update_frequency,
        )
        _validate_topology(self.actor_train, self.actor_infer)

    def seal_weight_receipt(
        self,
        output: str | Path,
        *,
        runtime_identity: Mapping[str, Any],
        model_identity: Mapping[str, Any],
        resolved_config_sha256: str,
    ) -> dict[str, Any]:
        """Seal the production response receipt before any generation begins."""

        if self._receipt_passed or self._generation_started:
            raise RuntimeError("vLLM parity receipt must be the first model transaction")
        transaction_id = uuid.uuid4().hex
        ray.get(
            [
                worker.rdan_begin_response_receipt.remote(transaction_id)
                for worker in [*self.actor_train.workers, *self.actor_infer.workers]
            ]
        )
        self.model_update(0)
        actor_receipts = ray.get([worker.rdan_get_response_receipt.remote() for worker in self.actor_train.workers])
        infer_receipts = ray.get([worker.rdan_finish_response_receipt.remote() for worker in self.actor_infer.workers])
        counters = ray.get([worker.rdan_train_counters.remote() for worker in self.actor_train.workers])
        artifact = build_response_receipt(
            actor_receipts,
            infer_receipts,
            phase="initial",
            pipeline_step=0,
            actor_counters=counters,
            resolved_config_sha256=resolved_config_sha256,
            runtime_identity=runtime_identity,
            model_identity=model_identity,
            method=self.response_config.method,
            fixed_weight=float(self.response_config.quality_weight),
        )
        write_artifact(output, artifact)
        self._receipt_passed = True
        return artifact

    @torch.no_grad()
    def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> ParityObservation:
        """Generate once on vLLM and recompute the identical boundaries on FSDP2."""

        if not self._receipt_passed or self._generation_started:
            raise RuntimeError("vLLM parity requires one passing pre-generation receipt")
        returns = generation_config.get("num_return_sequences")
        if isinstance(returns, bool) or not isinstance(returns, int) or returns < 1:
            raise ValueError("vLLM parity requires a positive return count")
        prompts = math.ceil(responses / returns)
        if prompts * returns != responses:
            raise ValueError("vLLM parity response count must be divisible by return count")
        data = _prompt_batch(self.dataset, self.tokenizer, self.pipeline_config.prompt_length, prompts)
        data.meta_info = {
            "generation_config": dict(generation_config),
            "global_step": 0,
            "is_offload_states": False,
            "optimizer_updates": 0,
            "pipeline_steps": 0,
        }
        self._generation_started = True
        state_before = self.state.step
        self.actor_train.offload_states(blocking=True)
        self.actor_infer.load_states(blocking=True)
        try:
            generated = self.actor_infer.generate(data=data, blocking=True)
        finally:
            self.actor_infer.offload_states(blocking=True)
        if len(generated) != responses or generated.meta_info.get("infer_logprobs_source") != INFER_LOGPROBS_SOURCE:
            raise RuntimeError("vLLM parity generation boundary is incomplete")
        required = {"input_ids", "attention_mask", "response_mask", "infer_logprobs"}
        if generated.batch is None or set(generated.batch) < required:
            raise RuntimeError("vLLM parity generation is missing exact token boundaries")
        self.actor_train.load_states(blocking=True)
        try:
            recomputed = self.actor_train.compute_log_probs(generated.clone(), blocking=True)
        finally:
            self.actor_train.offload_states(blocking=True)
        actor_fields = {"log_probs", "actor_input_ids", "actor_attention_mask", "actor_response_mask"}
        if recomputed.batch is None or set(recomputed.batch) < actor_fields:
            raise RuntimeError("vLLM parity actor recomputation is incomplete")
        if recomputed.meta_info.get("actor_boundary_observed") is not True or self.state.step != state_before:
            raise RuntimeError("vLLM parity changed or missed the actor boundary")
        counters = ray.get([worker.rdan_train_counters.remote() for worker in self.actor_train.workers])
        if any(value.get("optimizer_steps") != 0 for value in counters):
            raise RuntimeError("vLLM parity performed an optimizer update")
        return ParityObservation(
            input_ids=generated.batch["input_ids"].clone(),
            attention_mask=generated.batch["attention_mask"].clone(),
            response_mask=generated.batch["response_mask"].clone(),
            infer_logprobs=generated.batch["infer_logprobs"].clone(),
            actor_logprobs=recomputed.batch["log_probs"].clone(),
            actor_input_ids=recomputed.batch["actor_input_ids"].clone(),
            actor_attention_mask=recomputed.batch["actor_attention_mask"].clone(),
            actor_response_mask=recomputed.batch["actor_response_mask"].clone(),
            infer_logprobs_source=INFER_LOGPROBS_SOURCE,
            actor_train_recomputed=True,
            actor_boundary_observed=True,
            optimizer_updates=0,
        )


def build_vllm_parity_pipeline(config: Any) -> VLLMRuntimeParityPipeline:
    """Resolve the two parity-only subclasses and construct their pipeline."""

    expected = (
        (config.actor_train, ACTOR_WORKER_PATH, ObservedResponseActorWorker, "actor_train"),
        (config.actor_infer, INFER_WORKER_PATH, ObservedResponseVLLMInferWorker, "actor_infer"),
    )
    for worker, path, worker_cls, name in expected:
        if worker.worker_cls != path:
            raise ValueError(f"vLLM parity requires {name}.worker_cls={path}")
        worker.worker_cls = worker_cls
    return VLLMRuntimeParityPipeline(config, config.rdan_response)


def build_receipt_link(path: str | Path, resolved_config_sha256: str) -> dict[str, str]:
    """Link one immutable zero-update production response receipt."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("vLLM parity receipt must be a regular file")
    payload = target.read_bytes()
    try:
        artifact = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("vLLM parity receipt cannot be parsed") from error
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("status") != "receipt_passed"
        or artifact.get("phase") != "initial"
        or artifact.get("pipeline_step") != 0
        or artifact.get("optimizer_updates") != 0
        or artifact.get("runtime", {}).get("resolved_config_sha256") != resolved_config_sha256
        or not isinstance(artifact.get("transaction_id"), str)
        or not artifact["transaction_id"]
    ):
        raise RuntimeError("vLLM parity receipt cannot authorize generation")
    return {
        "transaction_id": artifact["transaction_id"],
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "resolved_config_sha256": resolved_config_sha256,
    }


def _validate_config(config: Any) -> None:
    expected = (
        (config.actor_train, "fsdp2_train", ObservedResponseActorWorker, "actor_train"),
        (config.actor_infer, "vllm", ObservedResponseVLLMInferWorker, "actor_infer"),
    )
    for worker, strategy, worker_cls, name in expected:
        if worker.strategy_args.strategy_name != strategy or worker.worker_cls is not worker_cls:
            raise ValueError(f"vLLM parity {name} topology is invalid")
        if list(worker.device_mapping) != [0, 1] or worker.world_size != 2 or worker.num_gpus_per_worker != 1:
            raise ValueError(f"vLLM parity {name} requires DP2 on devices [0, 1]")
    if config.async_pipeline or config.async_generation_ratio != 0 or config.rewards != {}:
        raise ValueError("vLLM parity must be synchronous and reward-free")
    if config.track_with != "stdout" or config.tracker_kwargs != {}:
        raise ValueError("vLLM parity only permits stdout tracking")


def _validate_topology(actor: Any, infer: Any) -> None:
    for name, cluster in (("actor", actor), ("infer", infer)):
        ranks = cluster.worker_rank_info
        if len(ranks) != 2 or any(rank.dp_size != 2 or rank.tp_size != 1 or rank.pp_size != 1 for rank in ranks):
            raise RuntimeError(f"vLLM parity requires {name} DP2 TP1 PP1")


def _cluster(worker: Any, resource_manager: Any) -> Cluster:
    return Cluster(
        name=worker.name,
        worker_cls=worker.worker_cls,
        resource_manager=resource_manager,
        worker_config=worker,
    )


def _load_dataset(config: Any, tokenizer: Any) -> Any:
    from roll.pipeline.rlvr.rubircs_pipeline import get_encode_function, preprocess_dataset, update_dataset_domain

    data_args = config.actor_train.data_args
    data = load_response_dataset(data_args.file_name, dataset_dir=getattr(data_args, "dataset_dir", "."))
    template = config.global_template or data_args.template
    data = preprocess_dataset(
        data, config.prompt_length, get_encode_function(template, tokenizer, data_args), data_args
    )
    data = data.map(
        partial(update_dataset_domain, config.tag_2_domain),
        num_proc=data_args.preprocessing_num_workers,
        desc="update_dataset_domain",
        load_from_cache_file=False,
    )
    if len(data) < 4:
        raise ValueError("vLLM parity dataset cannot produce 32 responses")
    return data


def _prompt_batch(dataset: Any, tokenizer: Any, prompt_length: int, count: int) -> DataProto:
    collator = DataCollatorWithPaddingForPaddedKeys(
        tokenizer=tokenizer, max_length=prompt_length, padding="max_length"
    )
    return DataProto.from_single_dict(collator([dataset[index] for index in range(count)]))
