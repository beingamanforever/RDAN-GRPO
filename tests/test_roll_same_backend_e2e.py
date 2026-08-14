from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


class FakeData:
    def __init__(self, batch: dict[str, torch.Tensor], meta_info: dict[str, Any] | None = None) -> None:
        self.batch = batch
        self.meta_info = meta_info or {}
        self.moves: list[str] = []

    def to(self, device: str) -> FakeData:
        self.moves.append(device)
        return self

    @classmethod
    def from_dict(cls, tensors: dict[str, torch.Tensor]) -> FakeData:
        return cls(tensors)


class FakeWorker:
    def __init__(self, worker_config: Any) -> None:
        self.worker_config = worker_config

    def initialize(self, pipeline_config: Any) -> None:
        self.pipeline_config = pipeline_config


class FakeActorWorker(FakeWorker):
    def forward_func_log_probs(
        self,
        data: FakeData,
        output_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length = data.batch["input_ids"].shape
        values = torch.full((batch, length - 1), -0.5)
        return values, {"log_probs": values, "entropy": torch.ones_like(values)}


def _load_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    decorator = types.ModuleType("roll.distributed.scheduler.decorator")
    decorator.Dispatch = SimpleNamespace(ONE_TO_ALL=1, DP_MP_COMPUTE=2, DP_MP_DISPATCH_FIRST=3)
    decorator.register = lambda **_kwargs: lambda function: function
    protocol = types.ModuleType("roll.distributed.scheduler.protocol")
    protocol.DataProto = FakeData
    worker = types.ModuleType("roll.distributed.executor.worker")
    worker.Worker = FakeWorker
    factory = types.ModuleType("roll.distributed.strategy.factory")
    factory.create_strategy = lambda worker: worker.strategy
    providers = types.ModuleType("roll.models.model_providers")
    providers.default_actor_model_provider = object()
    actor = types.ModuleType("roll.pipeline.rlvr.actor_worker")
    actor.ActorWorker = FakeActorWorker
    platform = types.ModuleType("roll.platforms")
    platform.current_platform = SimpleNamespace(device_type="cuda", init=lambda: None)

    @contextmanager
    def offload_manager(**_kwargs: Any):
        yield

    contexts = types.ModuleType("roll.utils.context_managers")
    contexts.state_offload_manger = offload_manager
    functionals = types.ModuleType("roll.utils.functionals")
    functionals.postprocess_generate = lambda **kwargs: FakeData(
        {
            "input_ids": kwargs["output"].clone(),
            "infer_logprobs": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(scores) for scores in kwargs["output_logprobs"]],
                batch_first=True,
            ),
        }
    )
    states = types.ModuleType("roll.utils.offload_states")
    states.OffloadStateType = SimpleNamespace(model_params="model_params")
    modules = {
        "roll": types.ModuleType("roll"),
        "roll.distributed": types.ModuleType("roll.distributed"),
        "roll.distributed.executor": types.ModuleType("roll.distributed.executor"),
        "roll.distributed.executor.worker": worker,
        "roll.distributed.scheduler": types.ModuleType("roll.distributed.scheduler"),
        "roll.distributed.scheduler.decorator": decorator,
        "roll.distributed.scheduler.protocol": protocol,
        "roll.distributed.strategy": types.ModuleType("roll.distributed.strategy"),
        "roll.distributed.strategy.factory": factory,
        "roll.models": types.ModuleType("roll.models"),
        "roll.models.model_providers": providers,
        "roll.pipeline": types.ModuleType("roll.pipeline"),
        "roll.pipeline.rlvr": types.ModuleType("roll.pipeline.rlvr"),
        "roll.pipeline.rlvr.actor_worker": actor,
        "roll.platforms": platform,
        "roll.utils": types.ModuleType("roll.utils"),
        "roll.utils.context_managers": contexts,
        "roll.utils.functionals": functionals,
        "roll.utils.offload_states": states,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/rdan_grpo/roll_same_backend.py"
    spec = importlib.util.spec_from_file_location("test_roll_same_backend", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class FakeModel:
    def __init__(self, sequences: torch.Tensor, step_scores: list[torch.Tensor]) -> None:
        self.sequences = sequences
        self.step_scores = step_scores
        self.generate_calls: list[dict[str, Any]] = []
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1

    def generate(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        prompt_length = kwargs["input_ids"].shape[1]
        input_ids = kwargs["input_ids"].repeat_interleave(kwargs["num_return_sequences"], dim=0)
        for index, scores in enumerate(self.step_scores):
            processed = scores
            for processor in kwargs["logits_processor"]:
                processed = processor(input_ids, processed)
            assert processed is scores
            input_ids = torch.cat([input_ids, self.sequences[:, prompt_length + index, None]], dim=1)
        return SimpleNamespace(sequences=self.sequences.clone())


def _infer_worker(module: types.ModuleType, model: FakeModel, worker_class: type[Any] | None = None) -> Any:
    worker_type = worker_class or module.SynchronousHFInferWorker
    worker = worker_type.__new__(worker_type)
    worker.pipeline_config = SimpleNamespace(
        async_pipeline=False,
        async_generation_ratio=0,
        generate_opt_level=0,
        sequence_length=6,
    )
    worker.worker_config = SimpleNamespace(
        strategy_args=SimpleNamespace(strategy_name="hf_infer"),
        infer_batch_size=2,
        generating_args=SimpleNamespace(to_dict=lambda: _generation_config()),
    )
    worker.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    worker.strategy = SimpleNamespace(
        model=model,
        load_states=lambda: setattr(worker, "loads", worker.loads + 1),
        offload_states=lambda: setattr(worker, "offloads", worker.offloads + 1),
    )
    worker.loads = 0
    worker.offloads = 0
    return worker


def _generation_config(**overrides: Any) -> dict[str, Any]:
    return {
        "do_sample": True,
        "temperature": 0.99,
        "top_p": 0.99,
        "top_k": 100,
        "num_beams": 1,
        "num_return_sequences": 2,
        "max_new_tokens": 3,
        **overrides,
    }


def _load_response_workers(monkeypatch: pytest.MonkeyPatch, same_backend: types.ModuleType) -> types.ModuleType:
    receipt = types.ModuleType("rdan_grpo.roll_fsdp_hf_receipt")
    for name in (
        "begin_fsdp_hf_receipt",
        "begin_hf_infer_receipt",
        "finish_hf_infer_receipt",
        "get_fsdp_actor_receipt",
        "reset_fsdp_hf_receipt",
        "run_receipted_fsdp_hf_update",
    ):
        setattr(receipt, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "rdan_grpo.roll_fsdp_hf_receipt", receipt)
    monkeypatch.setitem(sys.modules, "rdan_grpo.roll_same_backend", same_backend)
    path = ROOT / "src/rdan_grpo/roll_response_workers.py"
    spec = importlib.util.spec_from_file_location("test_response_sampling_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose_response_generation() -> dict[str, Any]:
    config_dir = ROOT / "configs/roll"
    child = yaml.safe_load((config_dir / "qwen_rtt_papo_response_train.yaml").read_text(encoding="utf-8"))
    parent_name = child["defaults"][0]
    parent = yaml.safe_load((config_dir / f"{parent_name}.yaml").read_text(encoding="utf-8"))
    generation = {
        **parent["actor_infer"]["generating_args"],
        **child["actor_infer"]["generating_args"],
    }
    generation["max_new_tokens"] = child.get("response_length", parent["response_length"])
    generation["num_return_sequences"] = child.get(
        "num_return_sequences_in_group", parent["num_return_sequences_in_group"]
    )
    return generation


def test_response_worker_uses_composed_sampling_profile_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    workers = _load_response_workers(monkeypatch, module)
    configured = _compose_response_generation()
    worker = _infer_worker(
        module,
        FakeModel(torch.ones(8, 6, dtype=torch.long), []),
        workers.ResponseInferWorker,
    )
    worker.worker_config.generating_args = SimpleNamespace(to_dict=lambda: dict(configured))

    observed = module._generation_config(worker, FakeData({}, {"generation_config": dict(configured)}))

    assert observed["do_sample"] is True
    assert observed["temperature"] == 0.99
    assert observed["top_p"] == 0.99
    assert observed["top_k"] == 100
    drifted = dict(configured, top_k=0)
    with pytest.raises(RuntimeError, match="top_k"):
        module._generation_config(worker, FakeData({}, {"generation_config": drifted}))


def test_sync_hf_generation_preserves_prompt_return_and_early_eos_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "verify_transformers_generation_boundary", lambda: {})
    sequences = torch.tensor(
        [
            [0, 11, 12, 31, 2, 0],
            [0, 11, 12, 32, 33, 2],
            [21, 22, 23, 41, 2, 0],
            [21, 22, 23, 42, 43, 2],
        ]
    )
    generator = torch.Generator().manual_seed(17)
    step_scores = [torch.randn(4, 64, generator=generator) for _ in range(3)]
    model = FakeModel(sequences, step_scores)
    worker = _infer_worker(module, model)
    data = FakeData(
        {
            "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        },
        {"generation_config": _generation_config(), "is_offload_states": True},
    )

    output = worker.generate(data)

    expected = torch.stack(
        [
            torch.log_softmax(scores.float(), dim=-1).gather(1, sequences[:, 3 + index, None]).squeeze(1)
            for index, scores in enumerate(step_scores)
        ],
        dim=1,
    )
    expected[[0, 2], 2] = 0.0

    assert torch.equal(output.batch["input_ids"], sequences)
    assert torch.allclose(
        output.batch["infer_logprobs"],
        expected,
        atol=1e-6,
        rtol=1e-6,
    )
    assert output.meta_info["infer_logprobs_source"] == "observed_hf_generation"
    assert worker.loads == worker.offloads == 1
    call = model.generate_calls[0]
    assert call["return_dict_in_generate"] is True
    assert call["output_scores"] is False
    assert len(call["logits_processor"]) == 1
    assert call["input_ids"] is data.batch["input_ids"]
    assert call["attention_mask"] is data.batch["attention_mask"]
    assert call["eos_token_id"] == [2, 0]
    assert model.eval_calls == 1
    assert not inspect.iscoroutinefunction(worker.generate)


def test_inference_worker_rejects_generation_source_drift_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    model = FakeModel(torch.ones((2, 4), dtype=torch.long), [torch.randn(2, 16)])
    worker = _infer_worker(module, model)
    data = FakeData(
        {"input_ids": torch.ones((1, 3), dtype=torch.long), "attention_mask": torch.ones((1, 3), dtype=torch.long)},
        {"generation_config": _generation_config(num_return_sequences=2, max_new_tokens=1)},
    )
    monkeypatch.setattr(
        module,
        "verify_transformers_generation_boundary",
        lambda: (_ for _ in ()).throw(RuntimeError("generation source drift")),
    )

    with pytest.raises(RuntimeError, match="source drift"):
        worker.generate(data)

    assert worker.loads == worker.offloads == 0
    assert model.generate_calls == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("do_sample", False),
        ("temperature", 0.9),
        ("top_p", 0.95),
        ("top_k", 1),
        ("num_beams", 2),
        ("min_p", 0.1),
        ("typical_p", 0.95),
        ("epsilon_cutoff", 0.1),
        ("eta_cutoff", 0.1),
        ("watermarking_config", {"greenlist_ratio": 0.25}),
        ("renormalize_logits", True),
    ],
)
def test_sampling_profile_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Any,
) -> None:
    module = _load_module(monkeypatch)
    worker = _infer_worker(module, FakeModel(torch.ones(4, 6, dtype=torch.long), []))
    data = FakeData({}, {"generation_config": _generation_config(**{name: value})})
    with pytest.raises(RuntimeError, match=name):
        module._generation_config(worker, data)


def test_async_paths_and_pipeline_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    worker = _infer_worker(module, FakeModel(torch.ones(4, 6, dtype=torch.long), []))
    assert not inspect.iscoroutinefunction(worker.initialize)
    assert not inspect.iscoroutinefunction(worker.generate)
    assert not inspect.iscoroutinefunction(worker.generate_request)
    with pytest.raises(RuntimeError, match="synchronous"):
        worker.generate_request(FakeData({}))
    with pytest.raises(RuntimeError, match="asynchronous"):
        worker.abort_requests(["request"])
    worker.pipeline_config.async_pipeline = True
    with pytest.raises(RuntimeError, match="async_pipeline=false"):
        module._require_sync_hf_worker(worker)


@pytest.mark.parametrize("failure", ["short_scores", "prompt_order", "interior_pad", "token_after_eos"])
def test_generation_boundary_failures_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = _load_module(monkeypatch)
    prompts = torch.tensor([[0, 11, 12], [21, 22, 23]])
    sequences = prompts.repeat_interleave(2, dim=0)
    sequences = torch.cat([sequences, torch.tensor([[31, 2, 0], [32, 33, 2], [41, 2, 0], [42, 43, 2]])], dim=1)
    model = FakeModel(sequences, [torch.randn(4, 64) for _ in range(3)])
    data = FakeData(
        {"input_ids": prompts, "attention_mask": prompts.ne(0).long()},
        {"generation_config": _generation_config()},
    )
    if failure == "short_scores":
        model.step_scores.pop()
    elif failure == "prompt_order":
        model.sequences[[0, 2]] = model.sequences[[2, 0]]
    elif failure == "interior_pad":
        model.sequences[0, 3:] = torch.tensor([31, 0, 2])
    else:
        model.sequences[0, 3:] = torch.tensor([31, 2, 33])
    with pytest.raises(RuntimeError):
        module._generate_with_scores(model, data, _generation_config(pad_token_id=0, eos_token_id=[2, 0]))


@pytest.mark.parametrize("source", ["config", "forward_args"])
def test_caller_logits_processors_are_rejected(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    module = _load_module(monkeypatch)
    prompts = torch.tensor([[11, 12]])
    sequences = torch.tensor([[11, 12, 31]])
    model = FakeModel(sequences, [torch.randn(1, 64)])
    config = _generation_config(num_return_sequences=1, max_new_tokens=1)
    meta_info: dict[str, Any] = {"generation_config": config}
    if source == "config":
        config["logits_processor"] = []
    else:
        meta_info["forward_args"] = {"logits_processor": []}
    data = FakeData({"input_ids": prompts, "attention_mask": torch.ones_like(prompts)}, meta_info)
    with pytest.raises(RuntimeError, match="logits_processor"):
        if source == "config":
            module._generation_config(_infer_worker(module, model), data)
        else:
            module._generate_with_scores(model, data, _generation_config(num_return_sequences=1, pad_token_id=0))


def test_streaming_recorder_validates_calls_and_bounds_retained_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    rows, vocab, steps = 3, 257, 4096
    prompt = torch.tensor([[11, 12], [21, 22], [31, 32]])
    recorder = module._StreamingLogprobs(prompt)
    input_ids = prompt
    expected_by_step = []
    for step in range(steps):
        scores = torch.randn(rows, vocab)
        assert recorder(input_ids, scores) is scores
        assert recorder.retained_full_vocab_elements == rows * vocab
        assert recorder.selected_elements == rows * step
        token = torch.tensor([(step + 3) % vocab, (step + 5) % vocab, (step + 7) % vocab])
        expected_by_step.append(torch.log_softmax(scores, dim=-1).gather(1, token[:, None]).squeeze(1))
        input_ids = torch.cat([input_ids, token[:, None]], dim=1)

    transition = recorder.finalize(input_ids)

    expected = torch.stack(expected_by_step, dim=1)
    assert torch.allclose(transition, expected, atol=1e-6, rtol=1e-6)
    assert recorder.retained_full_vocab_elements == 0
    assert recorder.selected_elements == 0
    assert transition.numel() == rows * steps


@pytest.mark.parametrize("failure", ["same_length", "row_order", "score_rows", "nonfinite"])
def test_streaming_recorder_rejects_malformed_calls(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    module = _load_module(monkeypatch)
    prompt = torch.tensor([[11, 12], [21, 22]])
    recorder = module._StreamingLogprobs(prompt)
    recorder(prompt, torch.randn(2, 32))
    next_ids = torch.cat([prompt, torch.tensor([[3], [4]])], dim=1)
    if failure == "same_length":
        with pytest.raises(RuntimeError, match="call sequence"):
            recorder(prompt, torch.randn(2, 32))
    elif failure == "row_order":
        with pytest.raises(RuntimeError, match="row order"):
            recorder(next_ids.flip(0), torch.randn(2, 32))
    elif failure == "score_rows":
        with pytest.raises(RuntimeError, match="score rows"):
            recorder(next_ids, torch.randn(3, 32))
    else:
        scores = torch.randn(2, 32)
        scores[0, 3] = float("nan")
        recorder.clear()
        recorder = module._StreamingLogprobs(prompt)
        recorder(prompt, scores)
        with pytest.raises(RuntimeError, match="non-finite"):
            recorder.finalize(next_ids)


def _actor_worker(module: types.ModuleType, optimizer_state: dict[str, Any] | None = None) -> Any:
    worker = module.ObservedFSDP2ActorWorker.__new__(module.ObservedFSDP2ActorWorker)
    worker.strategy = SimpleNamespace(optimizer=SimpleNamespace(state=optimizer_state or {}))
    return worker


def test_actor_callback_records_independent_byte_identical_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    worker = _actor_worker(module)
    data = FakeData(
        {
            "input_ids": torch.tensor([[11, 12, 31, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "response_mask": torch.tensor([[0, 0, 1, 1]]),
        }
    )
    originals = {name: tensor.clone() for name, tensor in data.batch.items()}
    _, results = worker.forward_func_log_probs(data, torch.zeros(1, 4, 64))
    for source, observed in (
        ("input_ids", "actor_input_ids"),
        ("attention_mask", "actor_attention_mask"),
        ("response_mask", "actor_response_mask"),
    ):
        assert torch.equal(results[observed], originals[source])
        assert results[observed].data_ptr() != data.batch[source].data_ptr()
    data.batch["input_ids"][0, 0] = 99
    assert torch.equal(results["actor_input_ids"], originals["input_ids"])


def test_actor_compute_returns_observed_boundaries_without_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    worker = _actor_worker(module)
    worker.cluster_name = "actor_train"
    worker.worker_config = SimpleNamespace(infer_batch_size=1)

    def forward_step(batch: FakeData, forward_func: Any) -> dict[str, torch.Tensor]:
        _, results = forward_func(batch, torch.zeros(1, 4, 64))
        return results

    worker.strategy.get_data_input = lambda data: data
    worker.strategy.forward_step = forward_step
    data = FakeData(
        {
            "input_ids": torch.tensor([[11, 12, 31, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "response_mask": torch.tensor([[0, 0, 1, 1]]),
        },
        {"optimizer_updates": 0, "pipeline_steps": 0, "is_offload_states": False},
    )

    output = worker.compute_log_probs(data)

    assert output.meta_info["actor_boundary_observed"] is True
    assert output.meta_info["metrics"] == {}
    assert torch.equal(output.batch["actor_input_ids"], data.batch["input_ids"])
    assert torch.equal(output.batch["actor_attention_mask"], data.batch["attention_mask"])
    assert torch.equal(output.batch["actor_response_mask"], data.batch["response_mask"])
    assert worker.strategy.optimizer.state == {}


@pytest.mark.parametrize(
    ("meta_info", "optimizer_state", "message"),
    [
        ({"optimizer_updates": 1, "pipeline_steps": 0}, {}, "optimizer_updates=0"),
        ({"optimizer_updates": 0, "pipeline_steps": 1}, {}, "pipeline_steps=0"),
        ({"optimizer_updates": 0, "pipeline_steps": 0}, {"weight": {}}, "empty state"),
    ],
)
def test_actor_pre_parity_state_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    meta_info: dict[str, int],
    optimizer_state: dict[str, Any],
    message: str,
) -> None:
    module = _load_module(monkeypatch)
    worker = _actor_worker(module, optimizer_state)
    with pytest.raises(RuntimeError, match=message):
        module._require_zero_update_state(worker, FakeData({}, meta_info))


def test_actor_zero_update_state_and_boundary_mismatch_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    worker = _actor_worker(module)
    data = FakeData(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "response_mask": torch.tensor([[0, 1]]),
        },
        {"optimizer_updates": 0, "pipeline_steps": 0},
    )
    module._require_zero_update_state(worker, data)
    results = {
        "actor_input_ids": data.batch["input_ids"].clone(),
        "actor_attention_mask": data.batch["attention_mask"].clone(),
        "actor_response_mask": torch.tensor([[1, 0]]),
    }
    with pytest.raises(RuntimeError, match="actor_response_mask"):
        module._require_observed_boundaries(data, results)
    results["actor_response_mask"] = data.batch["response_mask"].to(torch.bool)
    with pytest.raises(RuntimeError, match="actor_response_mask"):
        module._require_observed_boundaries(data, results)
