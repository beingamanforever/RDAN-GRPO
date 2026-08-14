from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rdan_grpo.fsdp_hf_receipt import (
    MODEL,
    MODEL_REVISION,
    RTT_BOUNDARY_SHA256,
    RTT_REVISION,
    FSDPHFReceiptError,
    FSDPHFStreamReceipt,
    FSDPHFTransaction,
    build_fsdp_hf_receipt_artifact,
    seal_fsdp_hf_receipt,
    verify_fsdp_hf_boundary,
    verify_fsdp_hf_checkout,
)
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY, RuntimeIdentity

ROOT = Path(__file__).resolve().parents[1]
RTT_ROOT = ROOT.parent / "Rubrics-To-Tokens"
TRANSACTION = "fsdp-hf-transaction"
MODEL_IDENTITY = {
    "model": MODEL,
    "revision": MODEL_REVISION,
    "snapshot_sha256": "1" * 64,
    "tokenizer_files_sha256": "2" * 64,
    "chat_template_sha256": "3" * 64,
}


def _weights() -> list[tuple[str, torch.Tensor]]:
    return [
        ("model.embed_tokens.weight", torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)),
        ("model.norm.weight", torch.tensor([5, 6], dtype=torch.bfloat16)),
    ]


def _source_weights() -> list[tuple[str, torch.Tensor]]:
    return [(name, tensor.float()) for name, tensor in _weights()]


def _receipt(
    side: str, rank: int, paired_rank: int, weights: list[tuple[str, torch.Tensor]] | None = None
) -> dict[str, Any]:
    actor_rank, infer_rank = (rank, paired_rank) if side == "actor" else (paired_rank, rank)
    receipt = FSDPHFStreamReceipt(
        FSDPHFTransaction(TRANSACTION, actor_rank, infer_rank),
        side,
        accelerator_name="NVIDIA A100-SXM4-80GB",
    )
    values = weights or _weights()
    if side == "actor":
        receipt.open_actor_stream(torch.bfloat16)
        list(receipt.wrap_actor_batches([values]))
    else:
        receipt.finish_infer(values)
    return receipt.snapshot()


def _artifact(
    actors: list[dict[str, Any]] | None = None,
    infers: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return build_fsdp_hf_receipt_artifact(
        actors or [_receipt("actor", 0, 0), _receipt("actor", 1, 1)],
        infers or [_receipt("infer", 0, 0), _receipt("infer", 1, 1)],
        model_identity=kwargs.pop("model_identity", MODEL_IDENTITY),
        resolved_config_sha256=kwargs.pop("resolved_config_sha256", "a" * 64),
        rtt_revision=kwargs.pop("rtt_revision", RTT_REVISION),
        rtt_boundary_sha256=kwargs.pop("rtt_boundary_sha256", RTT_BOUNDARY_SHA256),
        generation_source_identity=kwargs.pop("generation_source_identity", GENERATION_SOURCE_IDENTITY),
        transaction_id=TRANSACTION,
        **kwargs,
    )


def _checks(artifact: dict[str, Any]) -> set[tuple[str, str | None]]:
    return {(failure["check"], failure.get("field")) for failure in artifact["failures"]}


def test_exact_fsdp2_to_hf_receipt_passes() -> None:
    artifact = _artifact()
    assert artifact["id"] == "qwen_a100_fsdp2_hf_weight_receipt_v1"
    assert artifact["status"] == "receipt_passed"
    assert artifact["optimizer_updates"] == artifact["pipeline_steps"] == 0
    assert artifact["generation_started_before_seal"] is False
    assert artifact["diagnostic_target"] == (
        "configured-dtype FSDP2 transport bytes versus final paired HF parameter bytes"
    )
    assert artifact["model"] == MODEL_IDENTITY
    assert {key: artifact["runtime"][key] for key in GENERATION_SOURCE_IDENTITY} == GENERATION_SOURCE_IDENTITY
    assert artifact["topology"]["pairs"] == [
        {"actor_rank": 0, "infer_rank": 0},
        {"actor_rank": 1, "infer_rank": 1},
    ]
    assert all(receipt["transaction"] == {"calls": 1, "complete": True} for receipt in artifact["actor_receipts"])
    assert artifact["actor_receipts"][0]["transport"] == {
        "source_dtypes": ["torch.bfloat16"],
        "transport_dtype": "torch.bfloat16",
        "normalization": "cast_to_configured_dtype_before_rtt_serialization",
    }
    serialized = json.dumps(artifact, sort_keys=True)
    assert "vllm" not in serialized.lower()
    assert all(field not in serialized for field in ("prompt", "response", "secret", "credential", "environment"))


def test_runtime_identity_serialization_passes_exactly() -> None:
    artifact = _artifact(model_identity=RuntimeIdentity(**MODEL_IDENTITY))
    assert artifact["status"] == "receipt_passed"
    assert artifact["model"] == MODEL_IDENTITY


def test_generation_source_identity_fails_closed() -> None:
    artifact = _artifact(generation_source_identity={**GENERATION_SOURCE_IDENTITY, "transformers_version": "4.57.1"})
    assert artifact["status"] == "receipt_failed"
    assert ("generation_source_identity", None) in _checks(artifact)


@pytest.mark.parametrize(
    "identity",
    [
        {},
        None,
        "malformed",
        {**MODEL_IDENTITY, "extra": "field"},
        {key: value for key, value in MODEL_IDENTITY.items() if key != "chat_template_sha256"},
        {**MODEL_IDENTITY, "model": "Qwen/other"},
        {**MODEL_IDENTITY, "revision": "0" * 40},
        {**MODEL_IDENTITY, "snapshot_sha256": "bad"},
        {**MODEL_IDENTITY, "tokenizer_files_sha256": "A" * 64},
        {**MODEL_IDENTITY, "chat_template_sha256": 3},
    ],
)
def test_model_identity_fails_closed(identity: Any) -> None:
    artifact = _artifact(model_identity=identity)
    assert artifact["status"] == "receipt_failed"
    assert ("model_identity", None) in _checks(artifact)
    assert set(artifact["model"]) <= set(MODEL_IDENTITY)


@pytest.mark.parametrize("field", ["name", "shape", "dtype", "nbytes", "sha256"])
def test_pair_metadata_and_bytes_fail_closed(field: str) -> None:
    infer = _receipt("infer", 0, 0)
    replacements = {
        "name": "raw.prefix.changed.weight",
        "shape": [4],
        "dtype": "torch.float16",
        "nbytes": 7,
        "sha256": "0" * 64,
    }
    infer["items"][0][field] = replacements[field]
    artifact = _artifact(infers=[infer, _receipt("infer", 1, 1)])
    assert ("pair", field) in _checks(artifact)


def test_order_rank_pair_transaction_and_replica_fail_closed() -> None:
    reordered = _receipt("infer", 0, 0)
    reordered["items"].reverse()
    assert ("pair", "order") in _checks(_artifact(infers=[reordered, _receipt("infer", 1, 1)]))

    missing = _artifact(actors=[_receipt("actor", 0, 0)])
    assert ("missing_rank", None) in _checks(missing)

    swapped = _receipt("actor", 0, 0)
    swapped["paired_rank"] = 1
    assert ("pair", None) in _checks(_artifact(actors=[swapped, _receipt("actor", 1, 1)]))

    incomplete = FSDPHFStreamReceipt(
        FSDPHFTransaction(TRANSACTION, 0, 0),
        "actor",
        accelerator_name="NVIDIA A100-SXM4-80GB",
    )
    incomplete.open_actor_stream(torch.bfloat16)
    stream = incomplete.wrap_actor_batches([_weights()])
    next(iter(stream))
    assert ("incomplete", None) in _checks(_artifact(actors=[incomplete.snapshot(), _receipt("actor", 1, 1)]))

    changed = _weights()
    changed[0][1].view(torch.int16)[0, 0] ^= 1
    cross_replica = _artifact(
        actors=[_receipt("actor", 0, 0), _receipt("actor", 1, 1, changed)],
        infers=[_receipt("infer", 0, 0), _receipt("infer", 1, 1, changed)],
    )
    assert ("actor_cross_replica", "sha256") in _checks(cross_replica)
    assert ("infer_cross_replica", "sha256") in _checks(cross_replica)


def test_transport_provenance_is_validated_cross_replica_and_manifest_bound() -> None:
    baseline = _artifact()
    actors = [_receipt("actor", 0, 0), _receipt("actor", 1, 1)]
    for actor in actors:
        actor["transport"]["source_dtypes"] = ["torch.float32"]
    normalized = _artifact(actors=actors)

    assert normalized["status"] == "receipt_passed"
    assert normalized["receipt_manifest_sha256"] != baseline["receipt_manifest_sha256"]

    actors[1]["transport"]["source_dtypes"] = ["torch.bfloat16"]
    inconsistent = _artifact(actors=actors)
    assert ("actor_cross_replica", "transport") in _checks(inconsistent)

    actors[0]["transport"]["transport_dtype"] = "torch.float16"
    invalid = _artifact(actors=actors)
    assert ("transport_provenance", None) in _checks(invalid)


def test_revision_boundary_config_accelerator_and_step_counts_are_gated() -> None:
    assert ("rtt_revision", None) in _checks(_artifact(rtt_revision="0" * 40))
    assert ("rtt_boundary_sha256", None) in _checks(_artifact(rtt_boundary_sha256={}))
    assert ("resolved_config_sha256", None) in _checks(_artifact(resolved_config_sha256="bad"))
    assert ("optimizer_updates", None) in _checks(_artifact(optimizer_updates=1))
    assert ("pipeline_steps", None) in _checks(_artifact(pipeline_steps=1))
    assert ("generation_started_before_seal", None) in _checks(_artifact(generation_started_before_seal=True))
    actor = _receipt("actor", 0, 0)
    actor["accelerator_name"] = "NVIDIA H100"
    assert ("accelerator", None) in _checks(_artifact(actors=[actor, _receipt("actor", 1, 1)]))


def _load_roll_hook(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, types.ModuleType]:
    fake_model_update = types.ModuleType("roll.third_party.fsdp2.model_update")
    fake_model_update.gather_fsdp2_weights = lambda *args, **kwargs: iter((_source_weights(),))
    fsdp2_package = types.ModuleType("roll.third_party.fsdp2")
    fsdp2_package.model_update = fake_model_update
    for name, module in {
        "roll": types.ModuleType("roll"),
        "roll.third_party": types.ModuleType("roll.third_party"),
        "roll.third_party.fsdp2": fsdp2_package,
        "roll.third_party.fsdp2.model_update": fake_model_update,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/rdan_grpo/roll_fsdp_hf_receipt.py"
    spec = importlib.util.spec_from_file_location("test_roll_fsdp_hf_receipt", path)
    assert spec is not None and spec.loader is not None
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    return hook, fake_model_update


def _updater(actor_dtype: str | None = "bf16", infer_dtype: str | None = "bf16") -> SimpleNamespace:
    return SimpleNamespace(
        model_update_name="update",
        is_lora=False,
        is_colocated=True,
        worker_config=SimpleNamespace(model_args=SimpleNamespace(dtype=actor_dtype)),
        infer_worker_config=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name="hf_infer"),
            num_gpus_per_worker=1,
            model_args=SimpleNamespace(dtype=infer_dtype),
        ),
    )


def test_real_gather_wrapper_and_final_hf_parameters_form_one_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    hook, model_update = _load_roll_hook(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA A100-SXM4-80GB")
    updater = _updater()
    actor = SimpleNamespace(
        rank=0,
        strategy=SimpleNamespace(weight_updaters={"update": updater}),
    )
    hook.begin_fsdp_hf_receipt(actor, TRANSACTION, 0)
    original = model_update.gather_fsdp2_weights

    def update() -> None:
        batches = list(model_update.gather_fsdp2_weights(object(), 10))
        assert [[name for name, _ in batch] for batch in batches] == [[name for name, _ in _weights()]]
        assert all(tensor.dtype == torch.bfloat16 for batch in batches for _, tensor in batch)

    hook.run_receipted_fsdp_hf_update(actor, "update", update)
    assert model_update.gather_fsdp2_weights is original
    actor_receipt = hook.get_fsdp_actor_receipt(actor)
    assert actor_receipt["stream_complete"] is True
    assert actor_receipt["transaction"]["calls"] == 1
    assert actor_receipt["transport"]["source_dtypes"] == ["torch.float32"]
    assert actor_receipt["transport"]["transport_dtype"] == "torch.bfloat16"

    infer = SimpleNamespace(
        rank=0,
        strategy=SimpleNamespace(
            strategy_name="hf_infer",
            model=SimpleNamespace(named_parameters=lambda: iter(_weights())),
        ),
    )
    hook.begin_hf_infer_receipt(infer, TRANSACTION, 0)
    infer_receipt = hook.finish_hf_infer_receipt(infer)
    assert infer_receipt["items"] == actor_receipt["items"]
    with pytest.raises(FSDPHFReceiptError, match="exactly once"):
        hook.finish_hf_infer_receipt(infer)


def test_gather_wrapper_restores_rtt_and_rejects_second_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    hook, model_update = _load_roll_hook(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA A100-SXM4-80GB")
    updater = _updater()
    worker = SimpleNamespace(rank=0, strategy=SimpleNamespace(weight_updaters={"update": updater}))
    hook.begin_fsdp_hf_receipt(worker, TRANSACTION, 0)
    original = model_update.gather_fsdp2_weights

    def update() -> None:
        list(model_update.gather_fsdp2_weights(object(), 10))
        list(model_update.gather_fsdp2_weights(object(), 10))

    with pytest.raises(FSDPHFReceiptError, match="exactly once"):
        hook.run_receipted_fsdp_hf_update(worker, "update", update)
    assert model_update.gather_fsdp2_weights is original


@pytest.mark.parametrize(
    ("actor_dtype", "infer_dtype", "message"),
    [
        ("float8", "float8", "unsupported"),
        (None, "bf16", "missing"),
        ("fp32", "bf16", "must match"),
    ],
)
def test_transport_dtype_contract_fails_closed_and_restores_rtt(
    monkeypatch: pytest.MonkeyPatch,
    actor_dtype: str | None,
    infer_dtype: str | None,
    message: str,
) -> None:
    hook, model_update = _load_roll_hook(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA A100-SXM4-80GB")
    worker = SimpleNamespace(
        rank=0,
        strategy=SimpleNamespace(weight_updaters={"update": _updater(actor_dtype, infer_dtype)}),
    )
    hook.begin_fsdp_hf_receipt(worker, TRANSACTION, 0)
    original = model_update.gather_fsdp2_weights

    with pytest.raises(FSDPHFReceiptError, match=message):
        hook.run_receipted_fsdp_hf_update(worker, "update", lambda: None)

    assert model_update.gather_fsdp2_weights is original


def test_failure_artifact_is_sealed_before_exception(tmp_path: Path) -> None:
    output = tmp_path / "fsdp-hf-receipt.json"
    artifact = _artifact(update_error={"type": "RuntimeError", "code": "unclassified"})
    with pytest.raises(FSDPHFReceiptError, match="model_update"):
        seal_fsdp_hf_receipt(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "receipt_failed"
    with pytest.raises(FileExistsError):
        seal_fsdp_hf_receipt(output, _artifact())


def test_failure_artifact_never_persists_arbitrary_exception_text() -> None:
    private_text = (
        "Basic cGFzc3dvcmQ= password=hunter2 https://secret.example.com/private "
        "2001:db8::1 secret.internal /root/secrets.env"
    )
    artifact = _artifact(
        update_error={
            "type": "BasicPasswordSecret",
            "code": "cuda_ipc_failure",
            "detail": private_text,
        }
    )

    assert artifact["failures"][0]["reason"] == {"type": "Exception", "code": "cuda_ipc_failure"}
    serialized = json.dumps(artifact)
    assert all(token not in serialized for token in private_text.split())


@pytest.mark.skipif(not RTT_ROOT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_pinned_rtt_checkout_and_all_boundary_bytes_are_exact(tmp_path: Path) -> None:
    assert verify_fsdp_hf_checkout(RTT_ROOT) == RTT_BOUNDARY_SHA256
    copy = tmp_path / "boundary"
    for relative in RTT_BOUNDARY_SHA256:
        target = copy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RTT_ROOT / relative, target)
    changed = next(iter(RTT_BOUNDARY_SHA256))
    target = copy / changed
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(FSDPHFReceiptError, match=changed):
        verify_fsdp_hf_boundary(copy)
