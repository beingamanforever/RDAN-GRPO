"""Live receipt-first FSDP2 and Hugging Face parity diagnostic for RTT."""

from __future__ import annotations

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

from rdan_grpo.fsdp_hf_receipt import (
    FSDPHFReceiptError,
    build_fsdp_hf_receipt_artifact,
    seal_fsdp_hf_receipt,
)
from rdan_grpo.response_dataset import load_response_dataset
from rdan_grpo.roll_fsdp_hf_receipt import (
    begin_fsdp_hf_receipt,
    begin_hf_infer_receipt,
    finish_hf_infer_receipt,
    get_fsdp_actor_receipt,
    run_receipted_fsdp_hf_update,
)
from rdan_grpo.roll_same_backend import ObservedFSDP2ActorWorker, SynchronousHFInferWorker
from rdan_grpo.runtime_parity import ParityObservation

ACTOR_WORKER_PATH = "rdan_grpo.roll_same_backend_live.ReceiptedFSDP2ActorWorker"
INFER_WORKER_PATH = "rdan_grpo.roll_same_backend_live.ReceiptedSynchronousHFInferWorker"


class ReceiptedFSDP2ActorWorker(ObservedFSDP2ActorWorker):
    """Observe actor boundaries and receipt exactly one real FSDP2 update."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_begin_fsdp_hf_receipt(self, transaction_id: str, infer_rank: int) -> None:
        """Begin one identity-paired actor transaction."""

        begin_fsdp_hf_receipt(self, transaction_id, infer_rank)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_get_fsdp_hf_receipt(self) -> dict[str, Any]:
        """Return the current actor receipt."""

        return get_fsdp_actor_receipt(self)

    def start_model_update(self, model_update_name: str) -> DataProto:
        """Wrap the exact FSDP2 gather stream consumed by one real update."""

        return run_receipted_fsdp_hf_update(
            self,
            model_update_name,
            lambda: super(ReceiptedFSDP2ActorWorker, self).start_model_update(model_update_name),
        )


class ReceiptedSynchronousHFInferWorker(SynchronousHFInferWorker):
    """Generate synchronously and receipt final HF parameters after update."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_begin_fsdp_hf_receipt(self, transaction_id: str, actor_rank: int) -> None:
        """Begin one identity-paired HF receiver transaction."""

        begin_hf_infer_receipt(self, transaction_id, actor_rank)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rdan_finish_fsdp_hf_receipt(self) -> dict[str, Any]:
        """Hash final named parameters after the real update returns."""

        return finish_hf_infer_receipt(self)


class SameBackendParityPipeline(BasePipeline):
    """Construct only colocated DP2 actor and synchronous HF inference roles."""

    def __init__(self, pipeline_config: Any):
        _validate_pipeline_config(pipeline_config)
        super().__init__(pipeline_config)
        self.model_update_groups = []
        self.checkpoint_clusters = []
        self._receipt_passed = False
        self._generation_started = False
        self._optimizer_updates = 0
        self._pipeline_steps = 0
        self.tokenizer = default_tokenizer_provider(model_args=pipeline_config.actor_train.model_args)
        self.dataset = _load_dataset(pipeline_config, self.tokenizer)
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
        ray.get(self.actor_infer.initialize(pipeline_config=pipeline_config, blocking=False))
        ray.get(self.actor_train.initialize(pipeline_config=pipeline_config, blocking=False))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=pipeline_config.actor_train.model_update_frequency,
        )
        _validate_live_topology(self.actor_train, self.actor_infer)

    def seal_weight_receipt(
        self,
        output: str | Path,
        *,
        model_identity: Any,
        resolved_config_sha256: str,
        rtt_revision: str,
        rtt_boundary_sha256: Mapping[str, str],
        generation_source_identity: Mapping[str, str],
    ) -> dict[str, Any]:
        """Run one real step-zero update and seal final actor and HF bytes."""

        if self._receipt_passed or self._generation_started:
            raise FSDPHFReceiptError("weight receipt must be the first and only model transaction")
        transaction_id = uuid.uuid4().hex
        actor_receipts: list[Mapping[str, Any]] = []
        infer_receipts: list[Mapping[str, Any]] = []
        update_error: dict[str, str] | None = None
        try:
            ray.get(
                [
                    worker.rdan_begin_fsdp_hf_receipt.remote(transaction_id, rank)
                    for rank, worker in enumerate(self.actor_train.workers)
                ]
            )
            ray.get(
                [
                    worker.rdan_begin_fsdp_hf_receipt.remote(transaction_id, rank)
                    for rank, worker in enumerate(self.actor_infer.workers)
                ]
            )
            self.model_update(0)
            actor_receipts = ray.get([worker.rdan_get_fsdp_hf_receipt.remote() for worker in self.actor_train.workers])
            infer_receipts = ray.get(
                [worker.rdan_finish_fsdp_hf_receipt.remote() for worker in self.actor_infer.workers]
            )
        except Exception as error:
            update_error = _classified_exception(error)
            actor_receipts = _available_actor_receipts(self.actor_train.workers)
            infer_receipts = _available_infer_receipts(self.actor_infer.workers)
        artifact = build_fsdp_hf_receipt_artifact(
            actor_receipts,
            infer_receipts,
            model_identity=model_identity,
            resolved_config_sha256=resolved_config_sha256,
            rtt_revision=rtt_revision,
            rtt_boundary_sha256=rtt_boundary_sha256,
            generation_source_identity=generation_source_identity,
            transaction_id=transaction_id,
            optimizer_updates=self._optimizer_updates,
            pipeline_steps=self._pipeline_steps,
            generation_started_before_seal=self._generation_started,
            update_error=update_error,
        )
        seal_fsdp_hf_receipt(output, artifact)
        self._receipt_passed = True
        return artifact

    @torch.no_grad()
    def collect_parity(self, responses: int, generation_config: Mapping[str, Any]) -> ParityObservation:
        """Generate after receipt seal and recompute exact RTT boundaries on FSDP2."""

        if not self._receipt_passed:
            raise RuntimeError("weight receipt must pass before generation")
        if self._generation_started:
            raise RuntimeError("same-backend parity permits exactly one generation batch")
        returns = generation_config.get("num_return_sequences")
        if isinstance(returns, bool) or not isinstance(returns, int) or returns <= 0:
            raise ValueError("same-backend parity requires a positive return count")
        prompt_count = math.ceil(responses / returns)
        if prompt_count * returns != responses:
            raise ValueError("response count must be divisible by num_return_sequences")
        data = _prompt_batch(self.dataset, self.tokenizer, self.pipeline_config.prompt_length, prompt_count)
        data.meta_info = {
            "generation_config": dict(generation_config),
            "is_offload_states": False,
            "optimizer_updates": 0,
            "pipeline_steps": 0,
        }
        state_before = self.state.step
        self._generation_started = True
        self.actor_train.offload_states(blocking=True)
        self.actor_infer.load_states(blocking=True)
        try:
            generated = self.actor_infer.generate(data=data, blocking=True)
        finally:
            self.actor_infer.offload_states(blocking=True)
        if len(generated) != responses:
            raise RuntimeError(f"same-backend parity expected {responses} responses, received {len(generated)}")
        if generated.meta_info.get("infer_logprobs_source") != "observed_hf_generation":
            raise RuntimeError("same-backend generation did not expose observed HF logprobs")
        required = {"input_ids", "attention_mask", "response_mask", "infer_logprobs"}
        if generated.batch is None or set(generated.batch.keys()) < required:
            raise RuntimeError("same-backend generation is missing exact postprocessed boundaries")
        recomputed = self.actor_train.compute_log_probs(generated.clone(), blocking=True)
        actor_fields = {
            "log_probs",
            "actor_input_ids",
            "actor_attention_mask",
            "actor_response_mask",
        }
        if recomputed.batch is None or set(recomputed.batch.keys()) < actor_fields:
            raise RuntimeError("same-backend actor did not return observed token boundaries")
        if recomputed.meta_info.get("actor_boundary_observed") is not True:
            raise RuntimeError("same-backend actor callback boundary was not observed")
        if self.state.step != state_before or self._optimizer_updates != 0 or self._pipeline_steps != 0:
            raise RuntimeError("same-backend parity changed the training state")
        return ParityObservation(
            input_ids=generated.batch["input_ids"].clone(),
            attention_mask=generated.batch["attention_mask"].clone(),
            response_mask=generated.batch["response_mask"].clone(),
            infer_logprobs=generated.batch["infer_logprobs"].clone(),
            actor_logprobs=recomputed.batch["log_probs"].clone(),
            actor_input_ids=recomputed.batch["actor_input_ids"].clone(),
            actor_attention_mask=recomputed.batch["actor_attention_mask"].clone(),
            actor_response_mask=recomputed.batch["actor_response_mask"].clone(),
            infer_logprobs_source="observed_hf_generation",
            actor_train_recomputed=True,
            actor_boundary_observed=True,
            optimizer_updates=0,
        )


def resolve_same_backend_workers(config: Any) -> None:
    """Resolve only the two revision-gated worker paths accepted by the adapter."""

    expected = (
        (config.actor_train, ACTOR_WORKER_PATH, ReceiptedFSDP2ActorWorker, "actor_train"),
        (config.actor_infer, INFER_WORKER_PATH, ReceiptedSynchronousHFInferWorker, "actor_infer"),
    )
    for worker, path, worker_cls, name in expected:
        if worker.worker_cls != path:
            raise ValueError(f"same-backend parity requires {name}.worker_cls={path}")
        worker.worker_cls = worker_cls


def build_same_backend_pipeline(config: Any) -> SameBackendParityPipeline:
    """Resolve the pinned worker paths immediately before pipeline construction."""

    resolve_same_backend_workers(config)
    return SameBackendParityPipeline(config)


def build_fsdp_hf_receipt_link(path: str | Path, resolved_config_sha256: str) -> dict[str, str]:
    """Link parity to one sealed passing FSDP2 to HF receipt."""

    target = Path(path)
    payload = target.read_bytes()
    try:
        artifact = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FSDPHFReceiptError("cannot parse FSDP2 to HF receipt") from error
    runtime = artifact.get("runtime") if isinstance(artifact, Mapping) else None
    model = artifact.get("model") if isinstance(artifact, Mapping) else None
    if (
        artifact.get("id") != "qwen_a100_fsdp2_hf_weight_receipt_v1"
        or artifact.get("status") != "receipt_passed"
        or not isinstance(runtime, Mapping)
        or not isinstance(model, Mapping)
        or runtime.get("resolved_config_sha256") != resolved_config_sha256
        or not isinstance(artifact.get("transaction_id"), str)
        or not artifact["transaction_id"]
    ):
        raise FSDPHFReceiptError("FSDP2 to HF receipt cannot authorize generation")
    try:
        rebuilt = build_fsdp_hf_receipt_artifact(
            artifact["actor_receipts"],
            artifact["infer_receipts"],
            model_identity=model,
            resolved_config_sha256=resolved_config_sha256,
            rtt_revision=runtime["rtt_revision"],
            rtt_boundary_sha256=runtime["rtt_boundary_sha256"],
            generation_source_identity={
                key: runtime[key]
                for key in (
                    "transformers_version",
                    "generation_get_logits_processor_sha256",
                    "generation_sample_sha256",
                )
            },
            transaction_id=artifact["transaction_id"],
            optimizer_updates=artifact["optimizer_updates"],
            pipeline_steps=artifact["pipeline_steps"],
            generation_started_before_seal=artifact["generation_started_before_seal"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FSDPHFReceiptError("FSDP2 to HF receipt is malformed") from error
    if artifact != rebuilt:
        raise FSDPHFReceiptError("FSDP2 to HF receipt evidence is invalid")
    return {
        "transaction_id": artifact["transaction_id"],
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "resolved_config_sha256": resolved_config_sha256,
    }


def raw_fsdp_hf_receipt_link(path: str | Path, resolved_config_sha256: str) -> dict[str, str]:
    """Link failure evidence to a sealed receipt without authorizing generation."""

    payload = Path(path).read_bytes()
    artifact = json.loads(payload)
    transaction_id = artifact.get("transaction_id") if isinstance(artifact, Mapping) else None
    runtime = artifact.get("runtime") if isinstance(artifact, Mapping) else None
    if (
        artifact.get("id") != "qwen_a100_fsdp2_hf_weight_receipt_v1"
        or artifact.get("status") not in {"receipt_passed", "receipt_failed"}
        or not isinstance(transaction_id, str)
        or not transaction_id
        or not isinstance(runtime, Mapping)
        or runtime.get("resolved_config_sha256") != resolved_config_sha256
    ):
        raise FSDPHFReceiptError("failed receipt linkage is invalid")
    return {
        "transaction_id": transaction_id,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "resolved_config_sha256": resolved_config_sha256,
    }


def _validate_pipeline_config(config: Any) -> None:
    expected = (
        (config.actor_train, "fsdp2_train", ReceiptedFSDP2ActorWorker, "actor_train"),
        (config.actor_infer, "hf_infer", ReceiptedSynchronousHFInferWorker, "actor_infer"),
    )
    for worker, strategy, worker_cls, name in expected:
        if getattr(getattr(worker, "strategy_args", None), "strategy_name", None) != strategy:
            raise ValueError(f"same-backend parity requires {name} strategy {strategy}")
        if worker.worker_cls is not worker_cls:
            raise ValueError(f"same-backend parity requires {worker_cls.__name__}")
        if list(worker.device_mapping) != [0, 1] or worker.num_gpus_per_worker != 1 or worker.world_size != 2:
            raise ValueError(f"same-backend parity requires {name} DP2 on devices [0, 1]")
    if config.actor_infer.max_concurrency != 1:
        raise ValueError("same-backend parity requires actor_infer.max_concurrency=1")
    if config.async_pipeline or config.async_generation_ratio != 0 or config.generate_opt_level != 0:
        raise ValueError("same-backend parity requires the synchronous generate_opt_level=0 path")
    if config.global_template != "qwen3_nothinking":
        raise ValueError("same-backend parity requires the non-thinking Qwen template")
    if config.rewards != {}:
        raise ValueError("same-backend parity must not construct reward workers")
    if config.track_with != "stdout" or config.tracker_kwargs != {}:
        raise ValueError("same-backend parity only permits stdout tracking")
    if isinstance(config.max_steps, bool) or not isinstance(config.max_steps, int) or config.max_steps <= 0:
        raise ValueError("same-backend parity requires a positive diagnostic max_steps")


def _validate_live_topology(actor_train: Any, actor_infer: Any) -> None:
    for name, cluster in (("actor", actor_train), ("infer", actor_infer)):
        ranks = cluster.worker_rank_info
        if len(ranks) != 2 or any(rank.dp_size != 2 or rank.tp_size != 1 or rank.pp_size != 1 for rank in ranks):
            raise RuntimeError(f"same-backend parity requires {name} DP2 TP1 PP1")
        if list(cluster.worker_config.device_mapping) != [0, 1]:
            raise RuntimeError(f"same-backend parity requires {name} device mapping [0, 1]")


def _load_dataset(config: Any, tokenizer: Any) -> Any:
    from roll.pipeline.rlvr.rubircs_pipeline import get_encode_function, preprocess_dataset, update_dataset_domain

    data_args = config.actor_train.data_args
    data = load_response_dataset(
        data_args.file_name,
        dataset_dir=getattr(data_args, "dataset_dir", "."),
    )
    template = config.global_template or config.actor_train.data_args.template
    encode = get_encode_function(template, tokenizer, config.actor_train.data_args)
    data = preprocess_dataset(data, config.prompt_length, encode, config.actor_train.data_args)
    data = data.map(
        partial(update_dataset_domain, config.tag_2_domain),
        num_proc=config.actor_train.data_args.preprocessing_num_workers,
        desc="update_dataset_domain",
        load_from_cache_file=False,
    )
    if len(data) < math.ceil(32 / config.actor_infer.generating_args.num_return_sequences):
        raise ValueError("same-backend parity dataset cannot produce 32 responses")
    return data


def _prompt_batch(dataset: Any, tokenizer: Any, prompt_length: int, count: int) -> DataProto:
    rows = [dataset[index] for index in range(count)]
    collator = DataCollatorWithPaddingForPaddedKeys(
        tokenizer=tokenizer,
        max_length=prompt_length,
        padding="max_length",
    )
    return DataProto.from_single_dict(collator(rows))


def _available_actor_receipts(workers: list[Any]) -> list[Mapping[str, Any]]:
    return _available_receipts(workers, "rdan_get_fsdp_hf_receipt")


def _available_infer_receipts(workers: list[Any]) -> list[Mapping[str, Any]]:
    return _available_receipts(workers, "rdan_finish_fsdp_hf_receipt")


def _available_receipts(workers: list[Any], method: str) -> list[Mapping[str, Any]]:
    receipts: list[Mapping[str, Any]] = []
    for worker in workers:
        try:
            value = ray.get(getattr(worker, method).remote())
        except Exception:
            continue
        if isinstance(value, Mapping):
            receipts.append(value)
    return receipts


def _classified_exception(error: Exception) -> dict[str, str]:
    cause = getattr(error, "cause", None)
    if not isinstance(cause, BaseException):
        cause = error.__cause__ if isinstance(error.__cause__, BaseException) else error
    message = str(cause).lower()
    error_type = type(cause).__name__
    if error_type == "OutOfMemoryError" or ("cuda" in message and "out of memory" in message):
        code = "cuda_out_of_memory"
    elif "cuda ipc" in message or "cuda_ipc" in message:
        code = "cuda_ipc_failure"
    elif any(token in message for token in ("deserialize", "deserialization", "unpickl", "forkingpickler")):
        code = "deserialization_failure"
    elif "shape" in message and any(token in message for token in ("mismatch", "must match", "size")):
        code = "shape_mismatch"
    elif "device" in message and any(token in message for token in ("mismatch", "expected", "same device")):
        code = "device_mismatch"
    elif "dtype" in message and any(token in message for token in ("mismatch", "expected", "same dtype")):
        code = "dtype_mismatch"
    elif any(token in message for token in ("copy_", "copy parameter", "parameter copy")):
        code = "parameter_copy_failure"
    else:
        code = "unclassified"
    return {"type": error_type, "code": code}
