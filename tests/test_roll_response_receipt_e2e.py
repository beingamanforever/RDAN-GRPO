from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch

from rdan_grpo.fsdp_hf_receipt import (
    MODEL,
    MODEL_REVISION,
    RTT_REVISION,
    FSDPHFStreamReceipt,
    FSDPHFTransaction,
)
from rdan_grpo.roll_response_receipt import ResponseReceiptError, build_response_receipt
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY

TRANSACTION = "response-receipt-transaction"
ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITY = {
    "model": MODEL,
    "revision": MODEL_REVISION,
    "snapshot_sha256": "1" * 64,
    "tokenizer_files_sha256": "2" * 64,
    "chat_template_sha256": "3" * 64,
}
RUNTIME_IDENTITY = {
    "resolved_config_sha256": "4" * 64,
    "production_train_config_sha256": "5" * 64,
    "response_data_manifest_sha256": "6" * 64,
    "response_data_output_sha256": "7" * 64,
    "rtt_revision": RTT_REVISION,
    **GENERATION_SOURCE_IDENTITY,
}


def _weights() -> list[tuple[str, torch.Tensor]]:
    return [("model.weight", torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16))]


def _receipt(side: str, rank: int) -> dict[str, Any]:
    receipt = FSDPHFStreamReceipt(
        FSDPHFTransaction(TRANSACTION, rank, rank),
        side,
        accelerator_name="NVIDIA A100-SXM4-80GB",
    )
    if side == "actor":
        receipt.open_actor_stream(torch.bfloat16)
        list(receipt.wrap_actor_batches([_weights()]))
    else:
        receipt.finish_infer(_weights())
    return receipt.snapshot()


def _vllm_receipt(rank: int, calls: int = 2) -> dict[str, Any]:
    receipt = _receipt("infer", rank)
    receipt.pop("transaction")
    receipt.pop("transport", None)
    receipt["backend"] = "vllm"
    receipt["loader"] = {
        "calls": calls,
        "successes": calls,
        "failed": False,
        "segments_started": calls,
        "segments_completed": calls,
        "loaded": True,
    }
    return receipt


def _counters(steps: int) -> list[dict[str, int]]:
    return [
        {
            "rank": rank,
            "optimizer_steps": steps,
            "scheduler_steps": steps,
            "finite_steps": steps,
            "skipped_optimizer_steps": 0,
        }
        for rank in range(2)
    ]


def _build(phase: str = "initial", pipeline_step: int = 0, steps: int = 0, **kwargs: Any) -> dict[str, Any]:
    return build_response_receipt(
        kwargs.pop("actors", [_receipt("actor", 0), _receipt("actor", 1)]),
        kwargs.pop("infers", [_receipt("infer", 0), _receipt("infer", 1)]),
        phase=phase,
        pipeline_step=pipeline_step,
        actor_counters=kwargs.pop("counters", _counters(steps)),
        resolved_config_sha256=kwargs.pop("resolved_config_sha256", "4" * 64),
        runtime_identity=kwargs.pop("runtime_identity", RUNTIME_IDENTITY),
        model_identity=kwargs.pop("model_identity", MODEL_IDENTITY),
        method=kwargs.pop("method", "rdan_scalar"),
        fixed_weight=kwargs.pop("fixed_weight", 0.5),
        **kwargs,
    )


def test_runner_runtime_identity_reaches_receipt_boundary_exactly() -> None:
    path = ROOT / "scripts/run_response_train.py"
    spec = importlib.util.spec_from_file_location("test_response_receipt_runner", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    response = runner.ResponseConfig("rtt_papo_response", 0.5, None, "4" * 64)
    runtime = runner._response_runtime_identity(
        response,
        {"manifest_sha256": "6" * 64, "output_sha256": "7" * 64},
        "5" * 64,
    )

    receipt = _build(runtime_identity=runtime)

    assert receipt["runtime"] == runtime == RUNTIME_IDENTITY


@pytest.mark.parametrize(
    ("phase", "pipeline_step", "steps"),
    [("initial", 0, 0), ("post_update", 1, 2), ("resume_initial", 3, 7)],
)
def test_production_response_receipt_preserves_exact_dp2_evidence(
    phase: str,
    pipeline_step: int,
    steps: int,
) -> None:
    artifact = _build(phase, pipeline_step, steps)

    assert artifact["status"] == "receipt_passed"
    assert artifact["phase"] == phase
    assert artifact["pipeline_step"] == pipeline_step
    assert artifact["runtime"] == RUNTIME_IDENTITY
    assert artifact["model"] == MODEL_IDENTITY
    assert artifact["method"] == "rdan_scalar"
    assert artifact["fixed_weight"] == 0.5
    assert artifact["optimizer_updates"] == steps
    assert len(artifact["receipt_manifest_sha256"]) == 64
    assert artifact["actor_counters"] == _counters(steps)
    assert artifact["actor_receipts"][0]["items"] == artifact["infer_receipts"][0]["items"]
    assert artifact["actor_receipts"][0]["transport"]["transport_dtype"] == "torch.bfloat16"


def test_production_response_receipt_accepts_observed_vllm_loader_transactions() -> None:
    artifact = _build(infers=[_vllm_receipt(0), _vllm_receipt(1)])

    assert artifact["infer_receipts"][0]["backend"] == "vllm"
    assert artifact["infer_receipts"][0]["loader"]["calls"] == 2
    assert "transaction" not in artifact["infer_receipts"][0]
    assert "transport" not in artifact["infer_receipts"][0]


@pytest.mark.parametrize(
    "change",
    [
        lambda receipt: receipt["loader"].update(successes=1),
        lambda receipt: receipt["loader"].update(segments_completed=1),
        lambda receipt: receipt["loader"].update(loaded=False),
        lambda receipt: receipt.update(transaction={"calls": 1, "complete": True}),
        lambda receipt: receipt.update(transport=None),
    ],
)
def test_response_receipt_rejects_incomplete_or_synthetic_vllm_evidence(change: object) -> None:
    infers = [_vllm_receipt(0), _vllm_receipt(1)]
    change(infers[0])  # type: ignore[operator]

    with pytest.raises(ResponseReceiptError, match="vLLM"):
        _build(infers=infers)


@pytest.mark.parametrize("field", ["sha256", "name", "dtype", "shape", "nbytes"])
def test_response_receipt_rejects_pair_or_replica_byte_drift(field: str) -> None:
    infers = [_receipt("infer", 0), _receipt("infer", 1)]
    values = {"sha256": "0" * 64, "name": "changed", "dtype": "torch.float16", "shape": [4], "nbytes": 9}
    infers[0]["items"][0][field] = values[field]

    with pytest.raises(ResponseReceiptError, match="bytes differ|manifest|summary"):
        _build(infers=infers)


def test_response_receipt_rejects_rank_counter_and_phase_drift() -> None:
    swapped = [_receipt("infer", 0), _receipt("infer", 1)]
    swapped[0]["paired_rank"] = 1
    with pytest.raises(ResponseReceiptError, match="identity paired"):
        _build(infers=swapped)

    counters = _counters(1)
    counters[1]["scheduler_steps"] = 2
    with pytest.raises(ResponseReceiptError, match="counters differ|scheduler"):
        _build("post_update", 1, counters=counters)

    with pytest.raises(ResponseReceiptError, match="nonzero training state"):
        _build("post_update", 1, 0)
    with pytest.raises(ResponseReceiptError, match="zero training state"):
        _build("initial", 0, 1)
    with pytest.raises(ResponseReceiptError, match="pipeline step"):
        _build("post_update", 3, 2)
    with pytest.raises(ResponseReceiptError, match="nonzero training state"):
        _build("resume_initial", 0, 2)


def test_response_receipt_validates_and_binds_transport_provenance() -> None:
    baseline = _build()
    actors = [_receipt("actor", 0), _receipt("actor", 1)]
    for actor in actors:
        actor["transport"]["source_dtypes"] = ["torch.float32"]
    normalized = _build(actors=actors)

    assert normalized["actor_receipts"][0]["transport"]["source_dtypes"] == ["torch.float32"]
    assert normalized["receipt_manifest_sha256"] != baseline["receipt_manifest_sha256"]

    actors[1]["transport"]["source_dtypes"] = ["torch.bfloat16"]
    with pytest.raises(ResponseReceiptError, match="transport provenance differs"):
        _build(actors=actors)

    actors[0]["transport"]["normalization"] = "undisclosed_cast"
    with pytest.raises(ResponseReceiptError, match="transport provenance is invalid"):
        _build(actors=actors)


@pytest.mark.parametrize(
    "change",
    [
        lambda runtime, model: runtime.update(transformers_version="4.57.1"),
        lambda runtime, model: runtime.update(production_train_config_sha256="bad"),
        lambda runtime, model: runtime.update(response_data_manifest_sha256="bad"),
        lambda runtime, model: runtime.update(resolved_config_sha256="0" * 64),
        lambda runtime, model: model.update(revision="0" * 40),
        lambda runtime, model: model.update(snapshot_sha256="bad"),
    ],
)
def test_response_receipt_rejects_runtime_or_model_identity_drift(change: object) -> None:
    runtime = copy.deepcopy(RUNTIME_IDENTITY)
    model = copy.deepcopy(MODEL_IDENTITY)
    change(runtime, model)  # type: ignore[operator]

    with pytest.raises(ResponseReceiptError, match="identity|hash"):
        _build(runtime_identity=runtime, model_identity=model)
