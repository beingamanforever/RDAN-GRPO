"""Synchronous Hugging Face rollout and actor observation workers for RTT."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.models.model_providers import default_actor_model_provider
from roll.pipeline.rlvr.actor_worker import ActorWorker
from roll.platforms import current_platform
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import postprocess_generate
from roll.utils.offload_states import OffloadStateType

from rdan_grpo.runtime_parity import verify_transformers_generation_boundary

_ACTOR_FIELDS = (
    "actor_input_ids",
    "actor_attention_mask",
    "actor_response_mask",
)
_SAME_BACKEND_INVARIANTS = {
    "do_sample": True,
    "num_beams": 1,
}
_CONFIGURED_SAMPLE_FIELDS = ("temperature", "top_p", "top_k")
_FINAL_PROCESSOR_PROFILE = {
    "min_p": None,
    "typical_p": 1.0,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "watermarking_config": None,
    "renormalize_logits": False,
}


class SynchronousHFInferWorker(Worker):
    """Run RTT rollout generation through the synchronous Hugging Face model."""

    def __init__(self, worker_config: Any) -> None:
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config: Any) -> None:
        """Initialize the exact RTT Hugging Face inference strategy synchronously."""

        Worker.initialize(self, pipeline_config)
        _require_sync_hf_worker(self)
        self.strategy = create_strategy(worker=self)
        if getattr(self.strategy, "strategy_name", None) != "hf_infer":
            raise RuntimeError("same-backend rollout requires the RTT hf_infer strategy")
        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer
        self.offload_states()
        current_platform.init()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_states(self, include: Sequence[Any] | None = None, non_blocking: bool = False) -> None:
        """Move the standard Hugging Face model to its configured inference devices."""

        del include, non_blocking
        _load_standard_hf_model(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_states(self, include: Sequence[Any] | None = None, non_blocking: bool = False) -> None:
        """Move standard Hugging Face model parameters to CPU and clear device caches."""

        del non_blocking
        if include is None or OffloadStateType.model_params in include:
            _offload_standard_hf_model(self)
        current_platform.empty_cache()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def generate(self, data: DataProto) -> DataProto:
        """Generate responses and preserve selected-token log-probabilities."""

        verify_transformers_generation_boundary()
        _require_sync_hf_worker(self)
        config = _generation_config(self, data)
        data = data.to(current_platform.device_type)
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
        self.load_states()
        try:
            sequences, logprobs = _generate_with_scores(self.strategy.model, data, config)
            result = postprocess_generate(
                prompts=data,
                output=sequences,
                num_return_sequences=config["num_return_sequences"],
                sequence_length=self.pipeline_config.sequence_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                output_logprobs=logprobs,
            )
            result.meta_info = {
                **data.meta_info,
                "infer_logprobs_source": "observed_hf_generation",
            }
            return result.to("cpu")
        finally:
            data.to("cpu")
            if data.meta_info.get("is_offload_states", True):
                self.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_full_log_probs(self, data: DataProto) -> DataProto:
        """Compute full-sequence HF log-probabilities without changing state."""

        _require_zero_update_markers(data, "HF inference observation")
        _require_sync_hf_worker(self)
        data = self.strategy.get_data_input(data)
        data = data.to(current_platform.device_type)
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
        try:
            with torch.no_grad():
                results = self.strategy.forward_step(batch=data, forward_func=self._forward_func_full_log_probs)
            log_probs = None if results is None else results.get("log_probs")
            if not isinstance(log_probs, torch.Tensor):
                raise RuntimeError("HF inference observation is missing full-sequence log-probabilities")
            return DataProto.from_dict(tensors={"log_probs": log_probs}).to("cpu")
        finally:
            data.to("cpu")

    def _forward_func_full_log_probs(
        self,
        data: DataProto,
        output_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor,
            input_ids=data.batch["input_ids"],
            attention_mask=data.batch["response_mask"],
        )
        return log_probs, {"log_probs": log_probs.clone().detach()}

    def generate_request(self, data: DataProto) -> DataProto:
        """Reject the asynchronous request path."""

        raise RuntimeError("same-backend rollout only supports synchronous batch generation")

    def abort_requests(self, request_ids: Sequence[str]) -> None:
        """Reject request cancellation because asynchronous generation is disabled."""

        raise RuntimeError("same-backend rollout does not support asynchronous requests")


class ObservedFSDP2ActorWorker(ActorWorker):
    """Expose token boundaries observed by the no-update FSDP2 actor forward."""

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def compute_log_probs(self, data: DataProto) -> DataProto:
        """Recompute actor log-probabilities without permitting optimizer state."""

        _require_zero_update_state(self, data)
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
            required = ("log_probs", "entropy", *_ACTOR_FIELDS)
            missing = [name for name in required if name not in results]
            if missing:
                raise RuntimeError(f"actor observation is missing fields: {missing}")
            _require_observed_boundaries(data, results)
            _require_empty_optimizer(self)
            output = DataProto.from_dict(tensors={name: results[name] for name in required}).to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics, "actor_boundary_observed": True}
        return output

    def forward_func_log_probs(
        self,
        data: DataProto,
        output_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Capture copies of the exact token tensors passed to the actor callback."""

        boundaries = {
            "actor_input_ids": data.batch["input_ids"].clone().detach(),
            "actor_attention_mask": data.batch["attention_mask"].clone().detach(),
            "actor_response_mask": data.batch["response_mask"].clone().detach(),
        }
        log_probs, results = super().forward_func_log_probs(data, output_tensor)
        results.update(boundaries)
        return log_probs, results


class _StreamingLogprobs:
    """Capture sampled-token log-probabilities with bounded full-vocabulary state."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] <= 0:
            raise RuntimeError("streaming log-probabilities require non-empty rank-2 input IDs")
        self._input_ids = input_ids.detach().clone()
        self._scores: torch.Tensor | None = None
        self._selected: list[torch.Tensor] = []
        self._vocab_size: int | None = None
        self._called = False
        self._finalized = False

    @property
    def retained_full_vocab_elements(self) -> int:
        """Return the number of retained full-vocabulary scalar values."""

        return 0 if self._scores is None else self._scores.numel()

    @property
    def selected_elements(self) -> int:
        """Return the number of retained selected-token scalar values."""

        return sum(values.numel() for values in self._selected)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Record the preceding sampled token and retain the current final scores."""

        self._require_call(input_ids, scores)
        if self._called:
            self._selected.append(self._select(input_ids[:, -1]))
            self._scores = None
        self._input_ids = input_ids.detach().clone()
        self._scores = scores.detach().to(dtype=torch.float32)
        self._vocab_size = scores.shape[1]
        self._called = True
        return scores

    def finalize(self, sequences: torch.Tensor) -> torch.Tensor:
        """Record the final sampled token and release all retained score state."""

        try:
            if self._finalized or not self._called or self._scores is None:
                raise RuntimeError("streaming log-probabilities cannot be finalized in the current state")
            if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                raise RuntimeError("generated sequences must be rank-2 tensors")
            if sequences.shape != (self._input_ids.shape[0], self._input_ids.shape[1] + 1):
                raise RuntimeError("generated sequences and streaming step count have different boundaries")
            if not torch.equal(sequences[:, :-1], self._input_ids):
                raise RuntimeError("generated sequence row order changed before finalization")
            self._selected.append(self._select(sequences[:, -1]))
            transition = torch.stack(self._selected, dim=1)
            if not bool(torch.isfinite(transition).all()):
                raise RuntimeError("selected token scores contain non-finite values")
            return transition
        finally:
            self._scores = None
            self._selected.clear()
            self._finalized = True

    def clear(self) -> None:
        """Release retained score state after an interrupted generation."""

        self._scores = None
        self._selected.clear()
        self._finalized = True

    def _require_call(self, input_ids: torch.Tensor, scores: torch.Tensor) -> None:
        if self._finalized:
            raise RuntimeError("streaming log-probability call sequence continued after finalization")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise RuntimeError("streaming input IDs must be rank-2 tensors")
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
            raise RuntimeError("streaming token scores must be rank-2 tensors")
        if scores.shape[0] != self._input_ids.shape[0] or input_ids.shape[0] != self._input_ids.shape[0]:
            raise RuntimeError("streaming token score rows changed")
        if not scores.is_floating_point():
            raise RuntimeError("streaming token scores must be floating point")
        if self._vocab_size is not None and scores.shape[1] != self._vocab_size:
            raise RuntimeError("streaming token score vocabulary changed")
        expected_length = self._input_ids.shape[1] + int(self._called)
        if input_ids.shape[1] != expected_length:
            raise RuntimeError("streaming log-probability call sequence skipped or repeated a step")
        prefix = input_ids[:, :-1] if self._called else input_ids
        if not torch.equal(prefix, self._input_ids):
            raise RuntimeError("streaming generation row order changed")
        if input_ids.device != scores.device:
            raise RuntimeError("streaming input IDs and token scores must share a device")

    def _select(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._scores is None:
            raise RuntimeError("streaming token scores are missing")
        try:
            selected = self._scores.gather(1, token_ids[:, None]).squeeze(1)
        except RuntimeError as error:
            raise RuntimeError("sampled token IDs are outside the captured vocabulary") from error
        return selected - torch.logsumexp(self._scores, dim=-1)


def _require_sync_hf_worker(worker: Any) -> None:
    pipeline = getattr(worker, "pipeline_config", None)
    if pipeline is None:
        raise RuntimeError("same-backend rollout requires an initialized pipeline")
    if bool(getattr(pipeline, "async_pipeline", False)):
        raise RuntimeError("same-backend rollout requires async_pipeline=false")
    if getattr(pipeline, "async_generation_ratio", None) != 0:
        raise RuntimeError("same-backend rollout requires async_generation_ratio=0")
    if getattr(pipeline, "generate_opt_level", None) != 0:
        raise RuntimeError("same-backend rollout requires generate_opt_level=0")
    strategy = getattr(getattr(worker, "worker_config", None), "strategy_args", None)
    if getattr(strategy, "strategy_name", None) != "hf_infer":
        raise RuntimeError("same-backend rollout requires actor_infer strategy hf_infer")
    model_args = getattr(getattr(worker, "worker_config", None), "model_args", None)
    if getattr(model_args, "model_type", None) == "trl":
        raise RuntimeError("same-backend rollout does not support TRL models")


def _load_standard_hf_model(worker: Any) -> None:
    model = _standard_hf_model(worker)
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        model.to(current_platform.device_type)
        return
    for layer_name, device_id in device_map.items():
        device = device_id if isinstance(device_id, torch.device) else f"{current_platform.device_type}:{device_id}"
        model.get_submodule(layer_name).to(device)


def _offload_standard_hf_model(worker: Any) -> None:
    model = _standard_hf_model(worker)
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        model.to("cpu")
        return
    for layer_name in device_map:
        model.get_submodule(layer_name).to("cpu")


def _standard_hf_model(worker: Any) -> Any:
    _require_sync_hf_worker(worker)
    model = getattr(getattr(worker, "strategy", None), "model", None)
    classes = getattr(type(model), "__mro__", ())
    trl_class = any(
        candidate.__name__ == "AutoModelForCausalLMWithValueHead"
        or candidate.__module__ == "trl"
        or candidate.__module__.startswith("trl.")
        for candidate in classes
    )
    if trl_class or (hasattr(model, "pretrained_model") and hasattr(model, "v_head")):
        raise RuntimeError("same-backend rollout does not support TRL model wrappers")
    if model is None or not callable(getattr(model, "to", None)):
        raise RuntimeError("same-backend rollout requires a standard Hugging Face model")
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None and not isinstance(device_map, Mapping):
        raise RuntimeError("same-backend rollout requires a valid Hugging Face device map")
    return model


def _generation_config(worker: Any, data: DataProto) -> dict[str, Any]:
    expected = _configured_sample_profile(worker)
    supplied = data.meta_info.get("generation_config")
    if supplied is None:
        supplied = worker.worker_config.generating_args.to_dict()
    if not isinstance(supplied, Mapping):
        raise RuntimeError("same-backend rollout requires a generation config mapping")
    config = copy.deepcopy(dict(supplied))
    if "logits_processor" in config:
        raise RuntimeError("caller-supplied logits_processor is not allowed")
    for name, value in expected.items():
        _require_profile_value(name, config.get(name), value)
    for name, expected in _FINAL_PROCESSOR_PROFILE.items():
        _require_profile_value(name, config.get(name, expected), expected)
        config[name] = expected
    returns = config.get("num_return_sequences")
    if isinstance(returns, bool) or not isinstance(returns, int) or returns <= 0:
        raise RuntimeError("num_return_sequences must be a positive integer")
    if "return_dict_in_generate" in config and config["return_dict_in_generate"] is not True:
        raise RuntimeError("same-backend rollout requires return_dict_in_generate=true")
    config["return_dict_in_generate"] = True
    for name in ("output_scores", "output_logits"):
        if name in config and config[name] is not False:
            raise RuntimeError(f"same-backend rollout requires {name}=false")
        config[name] = False
    config["eos_token_id"] = list(
        dict.fromkeys(
            token_id
            for token_id in (worker.tokenizer.eos_token_id, worker.tokenizer.pad_token_id)
            if token_id is not None
        )
    )
    config["pad_token_id"] = worker.tokenizer.pad_token_id
    return config


def _configured_sample_profile(worker: Any) -> dict[str, bool | int | float]:
    generating_args = getattr(getattr(worker, "worker_config", None), "generating_args", None)
    configured = generating_args.to_dict() if hasattr(generating_args, "to_dict") else None
    if not isinstance(configured, Mapping):
        raise RuntimeError("same-backend rollout requires configured generation arguments")
    expected: dict[str, bool | int | float] = {}
    for name, required in _SAME_BACKEND_INVARIANTS.items():
        _require_profile_value(name, configured.get(name), required)
        expected[name] = required
    for name in _CONFIGURED_SAMPLE_FIELDS:
        value = configured.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"same-backend rollout requires configured {name}")
        expected[name] = value
    return expected


def _require_profile_value(name: str, value: Any, expected: bool | int | float | None) -> None:
    if expected is None:
        valid = value is None
    elif isinstance(expected, bool):
        valid = value is expected
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        valid = False
    else:
        valid = math.isfinite(float(value)) and float(value) == float(expected)
    if not valid:
        raise RuntimeError(f"same-backend rollout requires {name}={expected!r}")


def _generate_with_scores(
    model: Any,
    data: DataProto,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, list[list[float]]]:
    input_ids = data.batch["input_ids"]
    attention_mask = data.batch["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise RuntimeError("rollout input IDs and attention masks must be aligned rank-2 tensors")
    forward_args = data.meta_info.get("forward_args", {})
    if not isinstance(forward_args, Mapping):
        raise RuntimeError("rollout forward_args must be a mapping")
    if "logits_processor" in forward_args or "logits_processor" in config:
        raise RuntimeError("caller-supplied logits_processor is not allowed")
    expected_rows = input_ids.shape[0] * config["num_return_sequences"]
    prompt_length = input_ids.shape[1]
    repeated = input_ids.repeat_interleave(config["num_return_sequences"], dim=0)
    recorder = _StreamingLogprobs(repeated)
    model.eval()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_processor=[recorder],
                **dict(forward_args),
                **dict(config),
            )
    except BaseException:
        recorder.clear()
        raise
    sequences = getattr(output, "sequences", None)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        recorder.clear()
        raise RuntimeError("Hugging Face generation did not return rank-2 sequences")
    transition = recorder.finalize(sequences)
    _require_sequence_boundaries(sequences, input_ids, expected_rows, prompt_length, transition)
    logprobs = _trim_logprobs(
        sequences[:, prompt_length:],
        transition,
        config["pad_token_id"],
        config["eos_token_id"],
    )
    return sequences, logprobs


def _require_sequence_boundaries(
    sequences: torch.Tensor,
    input_ids: torch.Tensor,
    expected_rows: int,
    prompt_length: int,
    transition: torch.Tensor,
) -> None:
    if (
        transition.ndim != 2
        or sequences.shape[0] != expected_rows
        or transition.shape != (expected_rows, sequences.shape[1] - prompt_length)
    ):
        raise RuntimeError("generated sequences and token scores have different boundaries")
    repeated = input_ids.repeat_interleave(expected_rows // input_ids.shape[0], dim=0)
    if not torch.equal(sequences[:, :prompt_length], repeated):
        raise RuntimeError("generated prompt order or input token bytes changed")


def _trim_logprobs(
    response_ids: torch.Tensor,
    transition: torch.Tensor,
    pad_token_id: int,
    eos_token_ids: Sequence[int],
) -> list[list[float]]:
    valid = response_ids.ne(pad_token_id)
    if bool(((~valid).cumsum(dim=1).bool() & valid).any()):
        raise RuntimeError("generated responses are not right padded")
    eos_ids = [token_id for token_id in eos_token_ids if token_id != pad_token_id]
    if eos_ids:
        eos = torch.zeros_like(valid)
        for token_id in eos_ids:
            eos |= response_ids.eq(token_id)
        after_eos = torch.cat([torch.zeros_like(eos[:, :1]), eos[:, :-1].cumsum(dim=1).bool()], dim=1)
        if bool((after_eos & valid).any()):
            raise RuntimeError("generated responses contain tokens after EOS")
    lengths = valid.sum(dim=1).tolist()
    if any(length <= 0 or length > transition.shape[1] for length in lengths):
        raise RuntimeError("generated response scores are missing or shorter than the response")
    selected = [transition[index, :length] for index, length in enumerate(lengths)]
    if any(not bool(torch.isfinite(values).all()) for values in selected):
        raise RuntimeError("selected token scores contain non-finite values")
    return [values.float().cpu().tolist() for values in selected]


def _require_zero_update_state(worker: Any, data: DataProto) -> None:
    _require_zero_update_markers(data, "actor observation")
    _require_empty_optimizer(worker)


def _require_zero_update_markers(data: DataProto, label: str) -> None:
    for name in ("optimizer_updates", "pipeline_steps"):
        value = data.meta_info.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise RuntimeError(f"{label} requires {name}=0")


def _require_empty_optimizer(worker: Any) -> None:
    optimizer = getattr(getattr(worker, "strategy", None), "optimizer", None)
    state = getattr(optimizer, "state", None)
    if optimizer is None or state is None or len(state) != 0:
        raise RuntimeError("actor observation requires an initialized optimizer with empty state")


def _require_observed_boundaries(data: DataProto, results: Mapping[str, Any]) -> None:
    expected = {
        "actor_input_ids": data.batch["input_ids"],
        "actor_attention_mask": data.batch["attention_mask"],
        "actor_response_mask": data.batch["response_mask"],
    }
    for name, tensor in expected.items():
        observed = results[name]
        if (
            not isinstance(observed, torch.Tensor)
            or observed.dtype != tensor.dtype
            or observed.shape != tensor.shape
            or not torch.equal(observed, tensor)
        ):
            raise RuntimeError(f"actor callback boundary changed {name}")
