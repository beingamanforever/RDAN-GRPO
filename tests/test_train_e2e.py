from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from rdan_grpo.bridge import PreflightCertificate

ROOT = Path(__file__).resolve().parents[1]
RETURNS = 8


class FakeData:
    def __init__(
        self,
        batch: dict[str, torch.Tensor],
        non_tensor_batch: dict[str, Any] | None = None,
        meta_info: dict[str, Any] | None = None,
    ) -> None:
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self) -> int:
        if self.batch:
            return next(iter(self.batch.values())).shape[0]
        return len(next(iter(self.non_tensor_batch.values())))

    def clone(self) -> FakeData:
        return FakeData(
            {key: value.clone() for key, value in self.batch.items()},
            {key: value.copy() for key, value in self.non_tensor_batch.items()},
            dict(self.meta_info),
        )


def _load_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    protocol = types.ModuleType("roll.distributed.scheduler.protocol")
    protocol.DataProto = FakeData
    for name, module in {
        "roll": types.ModuleType("roll"),
        "roll.distributed": types.ModuleType("roll.distributed"),
        "roll.distributed.scheduler": types.ModuleType("roll.distributed.scheduler"),
        "roll.distributed.scheduler.protocol": protocol,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/rdan_grpo/train_step.py"
    spec = importlib.util.spec_from_file_location("test_roll_response_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _ranks() -> list[Any]:
    return [SimpleNamespace(dp_size=2, tp_size=1, pp_size=1, cp_size=1) for _ in range(2)]


def _config() -> Any:
    return SimpleNamespace(
        actor_train=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name="fsdp2_train"),
            device_mapping=[0, 1],
            num_gpus_per_worker=1,
            world_size=2,
            training_args=SimpleNamespace(per_device_train_batch_size=4, gradient_accumulation_steps=2),
        ),
        actor_infer=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name="vllm"),
            device_mapping=[0, 1],
            num_gpus_per_worker=1,
            world_size=2,
            max_concurrency=32,
            generating_args=SimpleNamespace(num_return_sequences=RETURNS),
        ),
        async_pipeline=False,
        async_generation_ratio=0,
        generate_opt_level=0,
        enable_reference=False,
        enable_old_logprobs_recompute=True,
    )


class FakeInfer:
    def __init__(self) -> None:
        self.dp_size = 2
        self.worker_rank_info = _ranks()


class FakeActor:
    def __init__(self, *, skip: bool = False, finite: bool = True, rank_drift: bool = False) -> None:
        self.dp_size = 2
        self.worker_rank_info = _ranks()
        self.skip = skip
        self.finite = finite
        self.rank_drift = rank_drift
        self.optimizer_step = [0, 0]
        self.scheduler_step = [0, 0]
        self.compute_calls = 0
        self.train_calls = 0
        self.trained: FakeData | None = None
        self.metrics: dict[str, Any] = {"actor/total_loss": [0.4, 0.5]}

    def compute_log_probs(self, data: FakeData, *, blocking: bool) -> FakeData:
        assert blocking is True
        self.compute_calls += 1
        rows, tokens = data.batch["response_mask"].shape
        return FakeData({"log_probs": torch.full((rows, tokens - 1), -0.25)})

    def train_step(self, data: FakeData, *, blocking: bool) -> FakeData:
        assert blocking is True
        self.train_calls += 1
        self.trained = data
        if not self.skip:
            self.optimizer_step = [
                value + (3 if self.rank_drift and rank == 1 else 2) for rank, value in enumerate(self.optimizer_step)
            ]
        self.scheduler_step = [value + 2 for value in self.scheduler_step]
        return FakeData({}, meta_info={"metrics": self.metrics})

    def state(self) -> list[dict[str, Any]]:
        return [
            {
                "rank": rank,
                "optimizer_step": self.optimizer_step[rank],
                "scheduler_step": self.scheduler_step[rank],
                "grad_finite": self.finite if self.train_calls else True,
                "update_skipped": self.skip and self.train_calls > 0,
            }
            for rank in range(2)
        ]


def _scheduler_rewarded_batch() -> FakeData:
    prompt_ids = torch.tensor([0, 1, 2, 3])
    prompt_keys = np.asarray(["p0", "p1", "p2", "p3"], dtype=object)
    rubrics = np.empty(4, dtype=object)
    rubrics[:] = [[{"text": "r"}] for _ in range(4)]
    request_metadata = {
        "prompt": np.asarray(["zero", "one", "two", "three"], dtype=object),
        "rubrics": rubrics,
        "source": np.asarray(["type1"] * 4, dtype=object),
        "ground_truth": np.asarray([{"answer": index} for index in range(4)], dtype=object),
        "rdan_prompt_key": prompt_keys,
    }
    repeated_ids = prompt_ids.repeat_interleave(RETURNS)
    repeated_metadata = {
        key: np.repeat(value, RETURNS, axis=0).astype(object) for key, value in request_metadata.items()
    }
    rows = len(repeated_ids)
    sequence_length = 6
    response_mask = torch.tensor([[0, 0, 1, 1, 0, 0]]).repeat(rows, 1)
    batch = {
        "origin_prompt_id": repeated_ids,
        "input_ids": torch.ones(rows, sequence_length, dtype=torch.long),
        "attention_mask": torch.ones(rows, sequence_length, dtype=torch.long),
        "response_mask": response_mask,
    }
    row = torch.arange(rows) % RETURNS
    passed = row < RETURNS // 2
    soft = torch.tensor([-1.0, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5]).repeat(rows // RETURNS)
    scores = torch.stack((torch.where(passed, 1.0, -1.0), soft), dim=1)
    batch.update(
        {
            "rdan_scores": scores,
            "rdan_rubric_mask": torch.ones_like(scores, dtype=torch.bool),
            "rdan_eval_mask": torch.ones_like(scores, dtype=torch.bool),
            "rdan_hard_mask": torch.tensor([True, False]).expand_as(scores).clone(),
            "rdan_unsupported_hard": torch.zeros(rows, dtype=torch.bool),
            "rdan_judge_failed": torch.zeros(rows, dtype=torch.bool),
        }
    )
    return FakeData(batch, repeated_metadata)


def _certificate(method: str, quality_weight: float | None, mix_weight: float | None) -> PreflightCertificate:
    return PreflightCertificate(
        certificate_id="c" * 64,
        ready=True,
        method=method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
        config_sha256="a" * 64,
        source_sha256=MappingProxyType({"dataset": "b" * 64}),
        metrics=MappingProxyType({"finite": True}),
        reasons=(),
    )


def _run(
    module: types.ModuleType,
    method: str,
    *,
    actor: FakeActor | None = None,
    batch: FakeData | None = None,
    memory: list[dict[str, Any]] | None = None,
) -> tuple[Any, FakeActor, FakeData, list[str]]:
    actor = actor or FakeActor()
    rewarded = batch or _scheduler_rewarded_batch()
    quality = 0.5 if method in {"rdan_scalar", "rtt_papo_response"} else None
    mix = 0.5 if method == "rl_mix" else None
    events: list[str] = []

    def state() -> list[dict[str, Any]]:
        events.append("state")
        return actor.state()

    def observe_memory() -> list[dict[str, Any]]:
        events.append("memory")
        return memory or [
            {"rank": 0, "peak_bytes": 70, "total_bytes": 80},
            {"rank": 1, "peak_bytes": 71, "total_bytes": 80},
        ]

    result = module.run_response_train_step(
        pipeline_config=_config(),
        actor_train=actor,
        actor_infer=FakeInfer(),
        rewarded_batch=rewarded,
        certificate=_certificate(method, quality, mix),
        observe_training_state=state,
        observe_post_transaction_memory=observe_memory,
        method=method,
        quality_weight=quality,
        mix_weight=mix,
    )
    return result, actor, rewarded, events


@pytest.mark.parametrize("method", ["rl_aon", "rl_csr", "rl_mix", "rdan_scalar", "rtt_papo_response"])
def test_scheduler_batch_trains_with_shifted_rtt_contract(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    module = _load_module(monkeypatch)
    result, actor, batch, events = _run(module, method)

    assert result.method == method
    assert result.prompt_count == 4
    assert result.response_count == 32
    assert result.optimizer_updates == result.scheduler_steps == 2
    assert result.peak_memory_fraction == 71 / 80
    assert actor.compute_calls == actor.train_calls == 1
    assert events == ["state", "state", "memory"]
    assert actor.trained is batch
    assert actor.trained is not None
    token_shape = (32, batch.batch["response_mask"].shape[1] - 1)
    for name in module.TOKEN_FIELDS:
        assert batch.batch[name].shape == token_shape
    assert torch.equal(batch.batch["final_response_mask"], batch.batch["response_mask"][:, 1:].bool())
    assert torch.equal(batch.batch["old_log_probs"], batch.batch["ref_log_probs"])
    assert torch.equal(batch.batch["advantages"], batch.batch["returns"])
    expected = batch.batch["rdan_scalar_advantage"].unsqueeze(-1) * batch.batch["response_mask"][:, 1:]
    assert torch.equal(batch.batch["advantages"], expected)
    assert batch.non_tensor_batch["prompt"].tolist()[:RETURNS] == ["zero"] * RETURNS


@pytest.mark.parametrize("missing", ["prompt", "rubrics", "source", "ground_truth"])
def test_missing_scheduler_metadata_fails_before_actor_work(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()
    batch = _scheduler_rewarded_batch()
    batch.non_tensor_batch.pop(missing)

    with pytest.raises(RuntimeError, match="missing fields"):
        _run(module, "rl_aon", actor=actor, batch=batch)
    assert actor.compute_calls == actor.train_calls == 0


def test_prompt_metadata_union_must_follow_prompt_major_order(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()
    batch = _scheduler_rewarded_batch()
    batch.non_tensor_batch["prompt"][RETURNS] = "wrong"

    with pytest.raises(RuntimeError, match="metadata prompt"):
        _run(module, "rl_aon", actor=actor, batch=batch)
    assert actor.compute_calls == actor.train_calls == 0


def test_scheduler_local_prompt_ids_may_repeat_across_unique_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    batch = _scheduler_rewarded_batch()
    batch.batch["origin_prompt_id"][2 * RETURNS : 3 * RETURNS] = 0

    result, _, _, _ = _run(module, "rl_aon", batch=batch)

    assert result.prompt_count == 4


@pytest.mark.parametrize(
    "actor,match",
    [
        (FakeActor(skip=True), "skipped/nonfinite"),
        (FakeActor(finite=False), "skipped/nonfinite"),
        (FakeActor(rank_drift=True), "ranks disagree on the optimizer or scheduler step delta"),
    ],
)
def test_observed_optimizer_and_scheduler_state_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    actor: FakeActor,
    match: str,
) -> None:
    module = _load_module(monkeypatch)

    with pytest.raises(RuntimeError, match=match):
        _run(module, "rl_aon", actor=actor)
    assert actor.compute_calls == actor.train_calls == 1


def test_memory_is_observed_after_the_update_and_blocks_unsafe_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()
    memory = [
        {"rank": 0, "peak_bytes": 70, "total_bytes": 80},
        {"rank": 1, "peak_bytes": 75, "total_bytes": 80},
    ]

    with pytest.raises(RuntimeError, match="post-transaction peak GPU memory"):
        _run(module, "rl_aon", actor=actor, memory=memory)
    assert actor.optimizer_step == [2, 2]


def test_inference_logprobs_are_not_part_of_the_training_batch_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()

    _, actor, batch, _ = _run(module, "rl_aon", actor=actor)

    assert "infer_logprobs" not in batch.batch
    assert actor.compute_calls == actor.train_calls == 1
