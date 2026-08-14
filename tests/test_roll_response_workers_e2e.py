from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from rdan_grpo.fsdp_hf_receipt import FSDPHFReceiptError, FSDPHFStreamReceipt, FSDPHFTransaction

ROOT = Path(__file__).resolve().parents[1]


class FakeData:
    def __init__(
        self,
        values: torch.Tensor,
        meta_info: dict[str, Any] | None = None,
        non_tensor_batch: dict[str, Any] | None = None,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.values = values
        self.meta_info = meta_info or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.batch = batch or _clip_batch(values, self.meta_info)

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, item: slice) -> FakeData:
        return FakeData(self.values[item], dict(self.meta_info))

    @staticmethod
    def concat(values: list[FakeData]) -> FakeData:
        keys = set().union(*(value.non_tensor_batch for value in values))
        non_tensor = {key: np.concatenate([value.non_tensor_batch[key] for value in values]) for key in keys}
        return FakeData(torch.cat([value.values for value in values]), dict(values[0].meta_info), non_tensor)


class FakeActorWorker:
    def loss_func(self, data: FakeData, output_tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        del output_tensor
        return data.values.float().mean(), {"actor/ppo_ratio_clipfrac": 0.0}

    def train_step(self, data: FakeData) -> FakeData:
        default = (
            torch.ones_like(data.values, dtype=torch.float32)
            if data.meta_info.get("clip") == 1.0
            else torch.zeros_like(data.values, dtype=torch.float32)
        )
        logits = data.meta_info.get("current_log_probs", default)
        self.loss_func(data, logits)
        if data.meta_info.get("optimizer", True):
            self.strategy.optimizer.step()
        self.strategy.scheduler.step()
        return FakeData(data.values, {"metrics": {}})

    def start_model_update(self, model_update_name: str) -> str:
        return f"updated:{model_update_name}"


class FakeInferWorker:
    calls: list[list[int]]

    def generate(self, data: FakeData) -> FakeData:
        self.calls.append(data.values.tolist())
        return FakeData(data.values.repeat_interleave(2), {"source": "hf"})


def _load_workers(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    decorator = types.ModuleType("roll.distributed.scheduler.decorator")
    decorator.Dispatch = SimpleNamespace(ONE_TO_ALL=1, DP_MP_DISPATCH_FIRST=2, DP_MP_COMPUTE=3)
    decorator.register = lambda **_kwargs: lambda function: function
    protocol = types.ModuleType("roll.distributed.scheduler.protocol")
    protocol.DataProto = FakeData
    actor = types.ModuleType("roll.pipeline.rlvr.actor_worker")
    actor.ActorWorker = FakeActorWorker
    receipt = types.ModuleType("rdan_grpo.roll_fsdp_hf_receipt")
    events: list[tuple[str, tuple[Any, ...]]] = []
    for name in (
        "begin_fsdp_hf_receipt",
        "begin_hf_infer_receipt",
        "finish_hf_infer_receipt",
        "get_fsdp_actor_receipt",
        "reset_fsdp_hf_receipt",
        "run_receipted_fsdp_hf_update",
    ):
        setattr(receipt, name, lambda *args, _name=name, **kwargs: events.append((_name, args)) or {"name": _name})
    same = types.ModuleType("rdan_grpo.roll_same_backend")
    same.SynchronousHFInferWorker = FakeInferWorker
    for name, module in {
        "roll": types.ModuleType("roll"),
        "roll.distributed": types.ModuleType("roll.distributed"),
        "roll.distributed.scheduler": types.ModuleType("roll.distributed.scheduler"),
        "roll.distributed.scheduler.decorator": decorator,
        "roll.distributed.scheduler.protocol": protocol,
        "roll.pipeline": types.ModuleType("roll.pipeline"),
        "roll.pipeline.rlvr": types.ModuleType("roll.pipeline.rlvr"),
        "roll.pipeline.rlvr.actor_worker": actor,
        "rdan_grpo.roll_fsdp_hf_receipt": receipt,
        "rdan_grpo.roll_same_backend": same,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/rdan_grpo/roll_response_workers.py"
    spec = importlib.util.spec_from_file_location("test_roll_response_workers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._test_events = events
    return module


def _load_receipt_hook(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    update = types.ModuleType("roll.third_party.fsdp2.model_update")
    update.gather_fsdp2_weights = lambda: iter(())
    fsdp2 = types.ModuleType("roll.third_party.fsdp2")
    fsdp2.model_update = update
    for name, module in {
        "roll": types.ModuleType("roll"),
        "roll.third_party": types.ModuleType("roll.third_party"),
        "roll.third_party.fsdp2": fsdp2,
        "roll.third_party.fsdp2.model_update": update,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/rdan_grpo/roll_fsdp_hf_receipt.py"
    spec = importlib.util.spec_from_file_location("test_roll_response_receipt_hook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stepper:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none


def _actor(module: types.ModuleType, rank: int = 0) -> Any:
    worker = module.ResponseActorWorker()
    worker.rank_info = SimpleNamespace(dp_rank=rank)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(1.0)
    worker.strategy = SimpleNamespace(
        optimizer=Stepper(),
        scheduler=Stepper(),
        model=SimpleNamespace(parameters=lambda: [parameter]),
        op_compute_log_probs=lambda **kwargs: kwargs["logits"].reshape(1, -1).float(),
    )
    worker.pipeline_config = SimpleNamespace(
        importance_sampling="token",
        use_pg_clip_range=True,
        pg_clip=0.2,
        pg_clip_low=0.2,
        pg_clip_high=0.27,
    )
    return worker


def _clip_batch(values: torch.Tensor, meta_info: dict[str, Any]) -> dict[str, torch.Tensor]:
    current = values.reshape(1, -1).float()
    if meta_info.get("clip") == 1.0:
        current = torch.full_like(current, 1.0)
    mask = torch.ones_like(current, dtype=torch.bool)
    return {
        "input_ids": torch.zeros((1, current.shape[1] + 1), dtype=torch.long),
        "response_mask": torch.cat([torch.zeros((1, 1), dtype=torch.bool), mask], dim=1),
        "final_response_mask": mask,
        "old_log_probs": torch.zeros_like(current),
    }


def test_actor_counts_exact_successful_calls_and_blocks_scheduler_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    worker = _actor(module)

    output = worker.train_step(FakeData(torch.tensor([1]), {"optimizer": True}))

    expected = {
        "rank": 0,
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "finite_steps": 1,
        "skipped_optimizer_steps": 0,
    }
    assert output.meta_info["response_train_evidence"] == expected
    assert worker.rdan_train_counters() == expected
    assert worker.rdan_training_state() == {
        "rank": 0,
        "optimizer_step": 1,
        "scheduler_step": 1,
        "grad_finite": True,
        "update_skipped": False,
    }
    assert worker.strategy.optimizer.calls == worker.strategy.scheduler.calls == 1

    with pytest.raises(RuntimeError, match="without a successful optimizer"):
        worker.train_step(FakeData(torch.tensor([2]), {"optimizer": False}))
    assert worker.strategy.scheduler.calls == 1
    assert worker.rdan_train_counters()["skipped_optimizer_steps"] == 1
    assert worker.rdan_training_state() == {
        "rank": 0,
        "optimizer_step": 1,
        "scheduler_step": 1,
        "grad_finite": False,
        "update_skipped": True,
    }


def test_actor_blocks_full_clipping_before_optimizer_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    worker = _actor(module)

    with pytest.raises(RuntimeError, match="fully clipped optimizer update"):
        worker.train_step(FakeData(torch.tensor([1]), {"clip": 1.0}))

    assert worker.strategy.optimizer.calls == worker.strategy.scheduler.calls == 0
    assert worker.rdan_train_counters()["skipped_optimizer_steps"] == 1
    assert worker.rdan_training_state()["update_skipped"] is True


def test_actor_ignores_padding_when_every_response_token_is_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    worker = _actor(module)
    mask = torch.tensor([[True, False, False, False]])
    data = FakeData(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        {"current_log_probs": torch.tensor([1.0, 0.0, 0.0, 0.0])},
        batch={
            "input_ids": torch.zeros((1, 5), dtype=torch.long),
            "response_mask": torch.tensor([[False, True, False, False, False]]),
            "final_response_mask": mask,
            "old_log_probs": torch.zeros((1, 4)),
        },
    )

    with pytest.raises(RuntimeError, match="fully clipped optimizer update"):
        worker.train_step(data)

    assert worker.strategy.optimizer.calls == 0


def test_actor_blocks_nonfinite_gradient_before_optimizer_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    worker = _actor(module)
    parameter = next(iter(worker.strategy.model.parameters()))
    parameter.grad = torch.tensor(float("nan"))

    with pytest.raises(RuntimeError, match="non-finite gradient update"):
        worker.train_step(FakeData(torch.tensor([1])))

    assert worker.strategy.optimizer.calls == worker.strategy.scheduler.calls == 0
    assert worker.rdan_train_counters()["skipped_optimizer_steps"] == 1
    assert worker.rdan_training_state() == {
        "rank": 0,
        "optimizer_step": 0,
        "scheduler_step": 0,
        "grad_finite": False,
        "update_skipped": True,
    }


def test_production_receipt_rpcs_and_model_update_use_named_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    actors = [_actor(module, rank) for rank in range(2)]
    infers = [module.ResponseInferWorker() for _ in range(2)]
    for rank, infer in enumerate(infers):
        infer.rank_info = SimpleNamespace(dp_rank=rank)

    for worker in [*actors, *infers]:
        worker.rdan_begin_response_receipt("tx")
    assert [event[1][2] for event in module._test_events[:4]] == [0, 1, 0, 1]
    with pytest.raises(TypeError):
        actors[0].rdan_begin_response_receipt("tx", 1)

    actor = actors[0]
    infer = infers[0]
    assert actor.rdan_get_response_receipt() == {"name": "get_fsdp_actor_receipt"}
    assert actor.rdan_reset_response_receipt() == {"name": "reset_fsdp_hf_receipt"}
    assert infer.rdan_finish_response_receipt() == {"name": "finish_hf_infer_receipt"}
    assert infer.rdan_reset_response_receipt() == {"name": "reset_fsdp_hf_receipt"}
    assert actor.start_model_update("actor-to-infer") == {"name": "run_receipted_fsdp_hf_update"}
    assert [event[0] for event in module._test_events] == [
        "begin_fsdp_hf_receipt",
        "begin_fsdp_hf_receipt",
        "begin_hf_infer_receipt",
        "begin_hf_infer_receipt",
        "get_fsdp_actor_receipt",
        "reset_fsdp_hf_receipt",
        "finish_hf_infer_receipt",
        "reset_fsdp_hf_receipt",
        "run_receipted_fsdp_hf_update",
    ]


def test_receipt_reset_requires_one_completed_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_receipt_hook(monkeypatch)
    worker = SimpleNamespace()
    receipt = FSDPHFStreamReceipt(FSDPHFTransaction("tx", 0, 0), "actor", "A100")
    setattr(worker, module._RECEIPT_ATTR, receipt)

    with pytest.raises(FSDPHFReceiptError, match="completed"):
        module.reset_fsdp_hf_receipt(worker)

    receipt.open_actor_stream()
    list(receipt.wrap_actor_batches([[("weight", torch.ones(1))]]))
    snapshot = module.reset_fsdp_hf_receipt(worker)

    assert snapshot["transaction"] == {"calls": 1, "complete": True}
    assert not hasattr(worker, module._RECEIPT_ATTR)
    with pytest.raises(FSDPHFReceiptError, match="not begun"):
        module.reset_fsdp_hf_receipt(worker)


def test_inference_microbatches_remain_prompt_major(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    worker = module.ResponseInferWorker()
    worker.worker_config = SimpleNamespace(infer_batch_size=2)
    worker.calls = []

    worker.rank_info = SimpleNamespace(dp_rank=1)
    output = worker.generate(FakeData(torch.tensor([0, 1, 2, 3, 4]), {"global_step": 7}))

    assert worker.calls == [[0, 1], [2, 3], [4]]
    assert output.values.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert output.non_tensor_batch["generation_id"].tolist() == [f"gen-000007-r1-{index:012d}" for index in range(10)]


def test_scheduler_concatenation_preserves_inference_generation_id_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_workers(monkeypatch)
    outputs = []
    for rank, values in enumerate(([10, 11], [20, 21])):
        worker = module.ResponseInferWorker()
        worker.worker_config = SimpleNamespace(infer_batch_size=1)
        worker.rank_info = SimpleNamespace(dp_rank=rank)
        worker.calls = []
        outputs.append(worker.generate(FakeData(torch.tensor(values), {"global_step": 1})))

    merged = FakeData.concat(outputs)

    assert merged.values.tolist() == [10, 10, 11, 11, 20, 20, 21, 21]
    assert merged.non_tensor_batch["generation_id"].tolist() == [
        "gen-000001-r0-000000000000",
        "gen-000001-r0-000000000001",
        "gen-000001-r0-000000000002",
        "gen-000001-r0-000000000003",
        "gen-000001-r1-000000000000",
        "gen-000001-r1-000000000001",
        "gen-000001-r1-000000000002",
        "gen-000001-r1-000000000003",
    ]


def test_actor_dcp_round_trip_restores_exact_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    worker = _actor(module)
    events: list[str] = []
    worker.strategy = SimpleNamespace(
        optimizer=Stepper(),
        scheduler=Stepper(),
        model=worker.strategy.model,
        _get_dcp_checkpoint_dir=lambda root: str(Path(root) / "dcp"),
        save_checkpoint=lambda **kwargs: events.append(
            f"save:{kwargs['global_step']}:{kwargs['ckpt_id']}:{kwargs['is_last_step']}"
        ),
        _load_checkpoint_with_dcp=lambda **kwargs: events.append("load"),
        load_states=lambda: events.append("load_states"),
        offload_states=lambda: events.append("offload_states"),
    )
    worker._rdan_response_counters = {
        "optimizer_steps": 4,
        "scheduler_steps": 4,
        "finite_steps": 4,
        "skipped_optimizer_steps": 0,
    }

    saved = worker.rdan_save_dcp(str(tmp_path / "checkpoint"), 2)
    worker._rdan_response_counters["optimizer_steps"] = 99
    loaded = worker.rdan_load_dcp(str(tmp_path / "checkpoint"))

    expected = {"checkpoint_dir": str((tmp_path / "checkpoint").resolve()), "pipeline_step": 2, "rank": 0}
    assert saved == loaded == expected
    assert worker.rdan_train_counters() == {
        "rank": 0,
        "optimizer_steps": 4,
        "scheduler_steps": 4,
        "finite_steps": 4,
        "skipped_optimizer_steps": 0,
    }
    assert worker.rdan_training_state() == {
        "rank": 0,
        "optimizer_step": 4,
        "scheduler_step": 4,
        "grad_finite": True,
        "update_skipped": False,
    }
    assert events == [
        "save:1:response-step-000002:True",
        "load_states",
        "load",
        "offload_states",
    ]


def test_actor_dcp_rejects_pipeline_and_optimizer_count_unit_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_workers(monkeypatch)
    worker = _actor(module)
    worker._rdan_response_counters = {
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "finite_steps": 1,
        "skipped_optimizer_steps": 0,
    }

    with pytest.raises(RuntimeError, match="inconsistent with pipeline step"):
        worker.rdan_save_dcp(str(tmp_path / "checkpoint"), 2)


def test_rng_and_cuda_memory_rpcs_use_exact_torch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_workers(monkeypatch)
    worker = module.ResponseInferWorker()
    cpu = torch.tensor([1, 2], dtype=torch.uint8)
    cuda = torch.tensor([3, 4], dtype=torch.uint8)
    loaded: list[tuple[str, torch.Tensor]] = []
    monkeypatch.setattr(module.torch, "get_rng_state", lambda: cpu)
    monkeypatch.setattr(module.torch, "set_rng_state", lambda value: loaded.append(("cpu", value)))
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "get_rng_state", lambda: cuda)
    monkeypatch.setattr(module.torch.cuda, "set_rng_state", lambda value: loaded.append(("cuda", value)))
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(module.torch.cuda, "max_memory_allocated", lambda device: 3)
    monkeypatch.setattr(module.torch.cuda, "max_memory_reserved", lambda device: 4)
    monkeypatch.setattr(module.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=5))
    resets: list[bool] = []
    monkeypatch.setattr(module.torch.cuda, "reset_peak_memory_stats", lambda: resets.append(True))

    state = worker.rdan_save_rng()
    worker.rdan_load_rng(state)
    worker.rdan_reset_cuda_peak()
    worker.rank_info = SimpleNamespace(dp_rank=1)

    assert [name for name, _ in loaded] == ["cpu", "cuda"]
    assert torch.equal(loaded[0][1], cpu)
    assert torch.equal(loaded[1][1], cuda)
    assert resets == [True]
    assert worker.rdan_cuda_memory() == {
        "rank": 1,
        "peak_bytes": 4,
        "total_bytes": 5,
    }
