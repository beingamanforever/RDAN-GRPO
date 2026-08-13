from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from rdan_grpo.fsdp_hf_receipt import MODEL, MODEL_REVISION, RTT_REVISION
from rdan_grpo.roll_bridge import PreflightCertificate
from rdan_grpo.roll_response_receipt import build_response_receipt
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY

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
    path = ROOT / "src/rdan_grpo/roll_response_train.py"
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
            strategy_args=SimpleNamespace(strategy_name="hf_infer"),
            device_mapping=[0, 1],
            num_gpus_per_worker=1,
            world_size=2,
            max_concurrency=1,
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


def _scheduler_rewarded_batch(*, bad_infer_shape: bool = False) -> FakeData:
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
    infer_tokens = sequence_length - (2 if bad_infer_shape else 1)
    batch = {
        "origin_prompt_id": repeated_ids,
        "input_ids": torch.ones(rows, sequence_length, dtype=torch.long),
        "attention_mask": torch.ones(rows, sequence_length, dtype=torch.long),
        "response_mask": response_mask,
        "infer_logprobs": torch.full((rows, infer_tokens), -0.25),
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
    return FakeData(batch, repeated_metadata, {"infer_logprobs_source": "observed_hf_generation"})


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _receipt(
    updates: int,
    transaction_id: str,
    *,
    phase: str | None = None,
    pipeline_step: int | None = None,
    mismatch: bool = False,
) -> dict[str, Any]:
    items = [
        {
            "index": 0,
            "name": "model.weight",
            "shape": [2, 2],
            "dtype": "torch.float32",
            "nbytes": 16,
            "sha256": "a" * 64,
        }
    ]
    actor_receipts = []
    infer_receipts = []
    for rank in range(2):
        actor_receipts.append(_rank_receipt(transaction_id, "actor", rank, [dict(items[0])]))
        infer_receipts.append(_rank_receipt(transaction_id, "infer", rank, [dict(items[0])]))
    receipt = build_response_receipt(
        actor_receipts,
        infer_receipts,
        phase=phase or ("initial" if updates == 0 else "post_update"),
        pipeline_step=(0 if updates == 0 else 1) if pipeline_step is None else pipeline_step,
        actor_counters=[
            {
                "rank": rank,
                "optimizer_steps": updates,
                "scheduler_steps": updates,
                "finite_steps": updates,
                "skipped_optimizer_steps": 0,
            }
            for rank in range(2)
        ],
        resolved_config_sha256="4" * 64,
        runtime_identity={
            "resolved_config_sha256": "4" * 64,
            "production_train_config_sha256": "5" * 64,
            "rtt_revision": RTT_REVISION,
            **GENERATION_SOURCE_IDENTITY,
        },
        model_identity={
            "model": MODEL,
            "revision": MODEL_REVISION,
            "snapshot_sha256": "1" * 64,
            "tokenizer_files_sha256": "2" * 64,
            "chat_template_sha256": "3" * 64,
        },
        method="rdan_scalar",
        fixed_weight=0.5,
    )
    if mismatch:
        receipt = copy.deepcopy(receipt)
        receipt["infer_receipts"][1]["items"][0]["sha256"] = "f" * 64
        receipt["infer_receipts"][1]["manifest_sha256"] = _canonical_sha256(receipt["infer_receipts"][1]["items"])
        receipt["receipt_manifest_sha256"] = _canonical_sha256(
            {
                "actor_receipts": receipt["actor_receipts"],
                "infer_receipts": receipt["infer_receipts"],
            }
        )
    return receipt


def _rank_receipt(transaction_id: str, side: str, rank: int, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "transaction_id": transaction_id,
        "side": side,
        "rank": rank,
        "paired_rank": rank,
        "accelerator_name": "NVIDIA A100-SXM4-80GB",
        "stream_started": True,
        "stream_complete": True,
        "items": items,
        "tensor_count": len(items),
        "total_bytes": sum(item["nbytes"] for item in items),
        "manifest_sha256": _canonical_sha256(items),
        "transaction": {"calls": 1, "complete": True},
    }


def _run(
    module: types.ModuleType,
    method: str,
    *,
    actor: FakeActor | None = None,
    batch: FakeData | None = None,
    initial: dict[str, Any] | None = None,
    post: dict[str, Any] | None = None,
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

    def transfer() -> dict[str, Any]:
        events.append("transfer")
        return post or _receipt(2, "post")

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
        initial_receipt=initial or _receipt(0, "initial"),
        transfer_after_update=transfer,
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
    assert result.initial_transaction_id == "initial"
    assert result.post_transaction_id == "post"
    assert result.peak_memory_fraction == 71 / 80
    assert result.promotion_ready is True
    assert actor.compute_calls == actor.train_calls == 1
    assert events == ["state", "state", "transfer", "memory"]
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


def test_built_response_receipts_pass_the_training_caller_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    initial = _receipt(0, "built-initial")
    post = _receipt(2, "built-post")

    result, _, _, _ = _run(module, "rdan_scalar", initial=initial, post=post)

    assert initial["optimizer_updates"] == 0
    assert post["optimizer_updates"] == 2
    assert result.initial_transaction_id == "built-initial"
    assert result.post_transaction_id == "built-post"


def test_receipt_pipeline_transactions_and_optimizer_updates_use_distinct_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()
    actor.optimizer_step = [30, 30]
    actor.scheduler_step = [30, 30]
    initial = _receipt(30, "resume-initial", phase="resume_initial", pipeline_step=15)
    post = _receipt(32, "post", pipeline_step=16)

    result, _, _, _ = _run(module, "rdan_scalar", actor=actor, initial=initial, post=post)

    assert result.optimizer_updates == 2
    assert initial["pipeline_step"] == 15
    assert initial["optimizer_updates"] == 30
    assert post["pipeline_step"] == 16
    assert post["optimizer_updates"] == 32


def test_receipts_must_advance_one_pipeline_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()
    actor.optimizer_step = [30, 30]
    actor.scheduler_step = [30, 30]
    initial = _receipt(30, "resume-initial", phase="resume_initial", pipeline_step=15)
    post = _receipt(32, "post", pipeline_step=17)

    with pytest.raises(RuntimeError, match="pipeline and optimizer state"):
        _run(module, "rdan_scalar", actor=actor, initial=initial, post=post)


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
    "initial",
    [
        {"status": "receipt_failed"},
        _receipt(0, "initial", mismatch=True),
        {**_receipt(0, "initial"), "receipt_manifest_sha256": "f" * 64},
        {**_receipt(0, "initial"), "pipeline_step": 1},
    ],
)
def test_real_initial_receipt_manifest_failure_prevents_actor_work(
    monkeypatch: pytest.MonkeyPatch,
    initial: dict[str, Any],
) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()

    with pytest.raises(RuntimeError, match="receipt"):
        _run(module, "rl_aon", actor=actor, initial=initial)
    assert actor.compute_calls == actor.train_calls == 0


@pytest.mark.parametrize(
    "actor,match",
    [
        (FakeActor(skip=True), "skipped/nonfinite"),
        (FakeActor(finite=False), "skipped/nonfinite"),
        (FakeActor(rank_drift=True), "frozen two-update cadence"),
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


def test_post_receipt_failure_blocks_promotion_after_observed_update(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()

    with pytest.raises(RuntimeError, match="receipt"):
        _run(module, "rl_aon", actor=actor, post=_receipt(2, "post", mismatch=True))
    assert actor.optimizer_step == [2, 2]


def test_post_receipt_rejects_more_pipeline_transactions_than_optimizer_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    post = {**_receipt(2, "post"), "pipeline_step": 2}

    with pytest.raises(RuntimeError, match="pipeline and optimizer state"):
        _run(module, "rl_aon", post=post)


def test_memory_is_observed_after_transfer_and_blocks_unsafe_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()
    memory = [
        {"rank": 0, "peak_bytes": 70, "total_bytes": 80},
        {"rank": 1, "peak_bytes": 75, "total_bytes": 80},
    ]

    with pytest.raises(RuntimeError, match="post-transaction peak GPU memory"):
        _run(module, "rl_aon", actor=actor, memory=memory)
    assert actor.optimizer_step == [2, 2]


def test_invalid_inference_logprob_boundary_fails_before_actor_work(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    actor = FakeActor()

    with pytest.raises(RuntimeError, match="infer_logprobs"):
        _run(module, "rl_aon", actor=actor, batch=_scheduler_rewarded_batch(bad_infer_shape=True))
    assert actor.compute_calls == actor.train_calls == 0
