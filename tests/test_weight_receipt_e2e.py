from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rdan_grpo.weight_receipt import (
    RECEIPT_CLAIM,
    RECEIPT_NON_CLAIM,
    REQUIRED_VLLM_VERSION,
    RTT_BOUNDARY_SHA256,
    RTT_REVISION,
    TensorStreamReceipt,
    WeightReceiptError,
    build_receipt_link,
    build_weight_receipt_artifact,
    canonical_sha256,
    seal_weight_receipt,
    validate_parity_receipt_pair,
    verify_rtt_boundary,
    verify_rtt_checkout,
)

ROOT = Path(__file__).resolve().parents[1]
RTT_ROOT = ROOT.parent / "Rubrics-To-Tokens"
TRANSACTION = "receipt-transaction"


def _weights() -> list[tuple[str, torch.Tensor]]:
    return [
        ("model.embed_tokens.weight", torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)),
        ("model.norm.weight", torch.tensor([5, 6], dtype=torch.bfloat16)),
    ]


def _receipt(side: str, rank: int, paired_rank: int, weights: list[tuple[str, torch.Tensor]] | None = None) -> dict:
    receipt = TensorStreamReceipt(TRANSACTION, side, rank, paired_rank, accelerator_name="NVIDIA A100-SXM4-80GB")
    receipt.record_batch(weights or _weights())
    receipt.finish_stream()
    if side == "infer":
        receipt.set_internal_before(_weights())
        receipt.mark_loader_success()
        receipt.set_internal_after(_weights())
    return receipt.snapshot()


def _artifact(actors: list[dict] | None = None, infers: list[dict] | None = None, **kwargs: Any) -> dict:
    resolved_config_sha256 = kwargs.pop("resolved_config_sha256", "d" * 64)
    rtt_revision = kwargs.pop("rtt_revision", RTT_REVISION)
    rtt_boundary_sha256 = kwargs.pop("rtt_boundary_sha256", RTT_BOUNDARY_SHA256)
    return build_weight_receipt_artifact(
        actors or [_receipt("actor", 0, 0), _receipt("actor", 1, 1)],
        infers or [_receipt("infer", 0, 0), _receipt("infer", 1, 1)],
        model_identity={
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": "model-revision",
            "snapshot_sha256": "a" * 64,
            "tokenizer_files_sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
        },
        resolved_config_sha256=resolved_config_sha256,
        rtt_revision=rtt_revision,
        rtt_boundary_sha256=rtt_boundary_sha256,
        transaction_id=TRANSACTION,
        **kwargs,
    )


def _checks(artifact: dict) -> set[tuple[str, str | None]]:
    return {(failure["check"], failure.get("field")) for failure in artifact["failures"]}


def test_exact_receipt_passes_with_complete_redacted_identity() -> None:
    artifact = _artifact()
    assert artifact["status"] == "receipt_passed"
    assert artifact["optimizer_updates"] == 0
    assert artifact["generation_started_before_seal"] is False
    assert artifact["topology"]["pairs"] == [
        {"actor_rank": 0, "infer_rank": 0},
        {"actor_rank": 1, "infer_rank": 1},
    ]
    assert artifact["runtime"]["rtt_revision"] == RTT_REVISION
    assert artifact["runtime"]["rtt_boundary_sha256"] == RTT_BOUNDARY_SHA256
    assert artifact["runtime"]["resolved_config_sha256"] == "d" * 64
    assert len(artifact["receipt_manifest_sha256"]) == 64
    assert all(receipt["loader"]["loaded"] for receipt in artifact["infer_receipts"])
    assert artifact["claim"] == RECEIPT_CLAIM
    assert artifact["non_claim"] == RECEIPT_NON_CLAIM
    serialized = json.dumps(artifact, sort_keys=True)
    assert all(field not in serialized for field in ("prompt", "response", "secret", "credential", "environment"))


def test_one_bit_bf16_mutation_fails_digest_and_cross_replica() -> None:
    mutated = _weights()
    raw = mutated[0][1].view(torch.int16)
    raw[0, 0] ^= 1
    artifact = _artifact(infers=[_receipt("infer", 0, 0, mutated), _receipt("infer", 1, 1)])
    checks = _checks(artifact)
    assert ("transport", "sha256") in checks
    assert ("infer_cross_replica", "sha256") in checks


@pytest.mark.parametrize("field", ["name", "shape", "dtype", "nbytes", "sha256"])
def test_transport_metadata_mismatch_fails_closed(field: str) -> None:
    infer = _receipt("infer", 0, 0)
    replacements = {
        "name": "changed.weight",
        "shape": [4],
        "dtype": "torch.float16",
        "nbytes": 7,
        "sha256": "0" * 64,
    }
    infer["items"][0][field] = replacements[field]
    artifact = _artifact(infers=[infer, _receipt("infer", 1, 1)])
    assert ("transport", field) in _checks(artifact)


def test_missing_duplicate_and_order_fail_closed() -> None:
    missing = _receipt("infer", 0, 0)
    missing["items"].pop()
    duplicate = _receipt("infer", 1, 1)
    duplicate["items"][1]["name"] = duplicate["items"][0]["name"]
    artifact = _artifact(infers=[missing, duplicate])
    assert ("transport", "count") in _checks(artifact)
    assert ("duplicate", None) in _checks(artifact)

    reordered = _receipt("infer", 0, 0)
    reordered["items"].reverse()
    artifact = _artifact(infers=[reordered, _receipt("infer", 1, 1)])
    assert ("transport", "order") in _checks(artifact)


def test_missing_rank_pair_swap_and_cross_replica_fail_closed() -> None:
    missing = _artifact(actors=[_receipt("actor", 0, 0)])
    assert ("missing_rank", None) in _checks(missing)

    swapped = _artifact(
        actors=[_receipt("actor", 0, 1), _receipt("actor", 1, 0)],
        infers=[_receipt("infer", 0, 1), _receipt("infer", 1, 0)],
    )
    assert ("pair", None) in _checks(swapped)

    changed = _weights()
    changed[1][1][0] = 9
    cross_replica = _artifact(
        actors=[_receipt("actor", 0, 0), _receipt("actor", 1, 1, changed)],
        infers=[_receipt("infer", 0, 0), _receipt("infer", 1, 1, changed)],
    )
    assert ("actor_cross_replica", "sha256") in _checks(cross_replica)

    duplicate_rank = _artifact(actors=[_receipt("actor", 0, 0), _receipt("actor", 0, 0)])
    assert ("duplicate_rank", None) in _checks(duplicate_rank)


def test_internal_pre_post_and_cross_replica_are_independent_fail_closed_checks() -> None:
    infer = _receipt("infer", 0, 0)
    infer["internal_parameters"]["after"]["items"][0]["sha256"] = "0" * 64
    artifact = _artifact(infers=[infer, _receipt("infer", 1, 1)])
    checks = _checks(artifact)
    assert ("internal_pre_post", "sha256") in checks
    assert ("internal_after_cross_replica", "sha256") in checks
    assert artifact["non_claim"] == RECEIPT_NON_CLAIM


def test_receipt_pairing_rejects_parent_config_mutation_and_cross_run(
    tmp_path: Path,
) -> None:
    resolved_config_sha256 = canonical_sha256({"defaults": ["base"], "base": {"max_steps": 20}})
    mutated_config_sha256 = canonical_sha256({"defaults": ["base"], "base": {"max_steps": 21}})
    receipt_path = tmp_path / "receipt.json"
    seal_weight_receipt(receipt_path, _artifact(resolved_config_sha256=resolved_config_sha256))
    link = build_receipt_link(receipt_path, resolved_config_sha256)
    parity = {
        "runtime_backend": {"resolved_config_sha256": resolved_config_sha256},
        "weight_receipt": link,
    }
    validate_parity_receipt_pair(parity, receipt_path)

    inherited_parent_mutation = {
        **parity,
        "runtime_backend": {"resolved_config_sha256": mutated_config_sha256},
        "weight_receipt": {**link, "resolved_config_sha256": mutated_config_sha256},
    }
    with pytest.raises(WeightReceiptError, match="identity is invalid"):
        validate_parity_receipt_pair(inherited_parent_mutation, receipt_path)

    other_receipt_path = tmp_path / "other-receipt.json"
    other = _artifact(resolved_config_sha256=resolved_config_sha256)
    other["transaction_id"] = "other-transaction"
    for worker_receipt in other["actor_receipts"] + other["infer_receipts"]:
        worker_receipt["transaction_id"] = "other-transaction"
    other["receipt_manifest_sha256"] = canonical_sha256(
        {"actor_receipts": other["actor_receipts"], "infer_receipts": other["infer_receipts"]}
    )
    seal_weight_receipt(other_receipt_path, other)
    with pytest.raises(WeightReceiptError, match="immutable run linkage"):
        validate_parity_receipt_pair(parity, other_receipt_path)


def test_revision_config_boundary_and_a100_identity_are_gated() -> None:
    wrong_revision = _artifact(rtt_revision="0" * 40)
    assert ("rtt_revision", None) in _checks(wrong_revision)
    wrong_boundary = _artifact(rtt_boundary_sha256={})
    assert ("rtt_boundary_sha256", None) in _checks(wrong_boundary)
    wrong_config = _artifact(resolved_config_sha256="not-a-sha")
    assert ("resolved_config_sha256", None) in _checks(wrong_config)
    actor = _receipt("actor", 0, 0)
    actor["accelerator_name"] = "NVIDIA H100"
    wrong_accelerator = _artifact(actors=[actor, _receipt("actor", 1, 1)])
    assert ("accelerator", None) in _checks(wrong_accelerator)


def test_generator_is_lazy_and_consumed_once() -> None:
    events: list[str] = []

    def weights():
        events.append("produced")
        yield _weights()[0]

    receipt = TensorStreamReceipt(TRANSACTION, "actor", 0, 0, accelerator_name="NVIDIA A100-SXM4-80GB")
    wrapped = receipt.wrap(weights())
    assert events == []
    assert list(wrapped)[0][0] == "model.embed_tokens.weight"
    assert events == ["produced"]
    assert receipt.stream_complete is True
    with pytest.raises(WeightReceiptError, match="more than once"):
        list(receipt.wrap(_weights()))


def _load_roll_hook(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, types.ModuleType]:
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = REQUIRED_VLLM_VERSION
    fake_model_update = types.ModuleType("roll.third_party.megatron.model_update")
    fake_model_update.gather_all_hf_weights = lambda *args, **kwargs: iter((_weights(),))
    fake_worker = types.ModuleType("roll.third_party.vllm.worker")

    class WorkerV1:
        def custom_init_worker(self, *args: Any, **kwargs: Any) -> None:
            pass

        def reload_model(self) -> None:
            pass

        def load_weights(self, weights: Any) -> None:
            self.model_runner.model.load_weights(weights=weights)

    fake_worker.WorkerV1 = WorkerV1
    modules = {
        "vllm": fake_vllm,
        "roll": types.ModuleType("roll"),
        "roll.third_party": types.ModuleType("roll.third_party"),
        "roll.third_party.megatron": types.ModuleType("roll.third_party.megatron"),
        "roll.third_party.megatron.model_update": fake_model_update,
        "roll.third_party.vllm": types.ModuleType("roll.third_party.vllm"),
        "roll.third_party.vllm.worker": fake_worker,
    }
    modules["roll.third_party.megatron"].model_update = fake_model_update
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/rdan_grpo/roll_weight_receipt.py"
    spec = importlib.util.spec_from_file_location("test_roll_weight_receipt", path)
    assert spec is not None and spec.loader is not None
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    return hook, fake_model_update


def test_loader_exception_cannot_claim_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    hook, _ = _load_roll_hook(monkeypatch)

    class Model:
        def named_parameters(self):
            return iter(_weights())

        def load_weights(self, weights: Any) -> None:
            next(iter(weights))
            raise RuntimeError("loader failed")

    worker = hook.ReceiptWorkerV1()
    worker.model_runner = SimpleNamespace(model=Model())
    worker.custom_init_worker()
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA A100-SXM4-80GB")
    worker.rdan_begin_weight_receipt(TRANSACTION, 0, 0)
    with pytest.raises(RuntimeError, match="loader failed"):
        worker.load_weights(iter(_weights()))
    snapshot = hook._required_receipt(worker).snapshot()
    assert snapshot["loader"]["failed"] is True
    assert snapshot["loader"]["loaded"] is False
    assert snapshot["stream_complete"] is False


def test_consuming_noop_loader_keeps_artifact_claim_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    hook, _ = _load_roll_hook(monkeypatch)

    class Model:
        def named_parameters(self):
            return iter(_weights())

        def load_weights(self, weights: Any) -> None:
            list(weights)

    worker = hook.ReceiptWorkerV1()
    worker.model_runner = SimpleNamespace(model=Model())
    worker.custom_init_worker()
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA A100-SXM4-80GB")
    worker.rdan_begin_weight_receipt(TRANSACTION, 0, 0)
    worker.load_weights(iter(_weights()))
    infer = worker.rdan_get_weight_receipt()
    artifact = _artifact(infers=[infer, _receipt("infer", 1, 1)])

    assert artifact["status"] == "receipt_passed"
    assert artifact["claim"] == RECEIPT_CLAIM
    assert artifact["non_claim"] == RECEIPT_NON_CLAIM
    assert "applied" not in artifact["claim"]
    assert "does not prove that a loader applied" in artifact["non_claim"]


@pytest.mark.parametrize("raises", [False, True])
def test_rtt_generator_is_restored_after_success_and_failure(monkeypatch: pytest.MonkeyPatch, raises: bool) -> None:
    hook, model_update = _load_roll_hook(monkeypatch)
    original = model_update.gather_all_hf_weights
    worker = SimpleNamespace(rank=0)
    updater = SimpleNamespace(model_update_name="update")
    setattr(worker, hook._UPDATER_ATTR, updater)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA A100-SXM4-80GB")
    hook.begin_actor_weight_receipt(worker, TRANSACTION, 0)

    def update() -> None:
        list(model_update.gather_all_hf_weights())
        if raises:
            raise RuntimeError("update failed")

    if raises:
        with pytest.raises(RuntimeError, match="update failed"):
            hook.run_receipted_actor_update(worker, "update", update)
    else:
        hook.run_receipted_actor_update(worker, "update", update)
    assert model_update.gather_all_hf_weights is original


def test_artifact_write_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    artifact = _artifact()
    seal_weight_receipt(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    with pytest.raises(FileExistsError):
        seal_weight_receipt(output, artifact)


def test_failure_is_sealed_before_exception(tmp_path: Path) -> None:
    output = tmp_path / "receipt-failure.json"
    artifact = _artifact(update_error="RuntimeError")
    with pytest.raises(WeightReceiptError, match="model_update"):
        seal_weight_receipt(output, artifact)
    sealed = json.loads(output.read_text(encoding="utf-8"))
    assert sealed["status"] == "receipt_failed"
    assert sealed["optimizer_updates"] == 0
    assert sealed["generation_started_before_seal"] is False
    assert sealed["claim"] is None


def test_live_runner_does_not_generate_or_write_logprob_artifacts_after_receipt_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location("receipt_run_roll_parity", ROOT / "scripts/run_roll_parity.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    success = tmp_path / "success.json"
    failure = tmp_path / "failure.json"
    receipt = tmp_path / "receipt.json"
    snapshot = tmp_path / "model-revision"
    snapshot.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("diagnostic: true\n", encoding="utf-8")
    actor = SimpleNamespace(model_args=SimpleNamespace(model_name_or_path=str(snapshot)))
    infer = SimpleNamespace(
        model_args=SimpleNamespace(model_name_or_path=str(snapshot)),
        strategy_args=SimpleNamespace(
            strategy_config={"worker_extension_cls": "rdan_grpo.roll_weight_receipt.ReceiptWorkerV1"}
        ),
    )
    pipeline_config = SimpleNamespace(
        actor_train=actor,
        actor_infer=infer,
        global_template="qwen3_nothinking",
        track_with=None,
        tracker_kwargs=None,
    )
    args = SimpleNamespace(
        output=success,
        failure_output=failure,
        weight_receipt_output=receipt,
        rtt_root=tmp_path,
        config=config,
        snapshot=snapshot,
        responses=32,
    )
    events: list[str] = []

    class Pipeline:
        def __init__(self, config: Any) -> None:
            self.tokenizer = object()

        def seal_weight_receipt(self, output: Path, **kwargs: Any) -> None:
            seal_weight_receipt(output, _artifact(update_error="RuntimeError"))

    fake_ray = types.ModuleType("ray")
    fake_ray.shutdown = lambda: events.append("shutdown")
    fake_dacite = types.ModuleType("dacite")
    fake_dacite.from_dict = lambda **kwargs: pipeline_config
    fake_hydra = types.ModuleType("hydra")
    fake_hydra.compose = lambda **kwargs: {"diagnostic": True}
    fake_hydra.initialize_config_dir = lambda **kwargs: nullcontext()
    fake_omega = types.ModuleType("omegaconf")
    fake_omega.OmegaConf = SimpleNamespace(to_container=lambda config, resolve: config)
    fake_chat = types.ModuleType("roll.datasets.chat_template")
    fake_chat.register_chat_template = lambda name: lambda function: function
    fake_initialize = types.ModuleType("roll.distributed.scheduler.initialize")
    fake_initialize.init = lambda: events.append("init")
    fake_rubric = types.ModuleType("roll.pipeline.rlvr.rubric_config")
    fake_rubric.RLVRConfig = object
    fake_live = types.ModuleType("rdan_grpo.roll_live")
    fake_live.ObservedActorWorker = type("ObservedActorWorker", (), {})
    fake_live.ObservedLogprobInferWorker = type("ObservedLogprobInferWorker", (), {})
    fake_live.RuntimeParityPipeline = Pipeline
    for name, module in {
        "ray": fake_ray,
        "dacite": fake_dacite,
        "hydra": fake_hydra,
        "omegaconf": fake_omega,
        "roll": types.ModuleType("roll"),
        "roll.datasets": types.ModuleType("roll.datasets"),
        "roll.datasets.chat_template": fake_chat,
        "roll.distributed": types.ModuleType("roll.distributed"),
        "roll.distributed.scheduler": types.ModuleType("roll.distributed.scheduler"),
        "roll.distributed.scheduler.initialize": fake_initialize,
        "roll.pipeline": types.ModuleType("roll.pipeline"),
        "roll.pipeline.rlvr": types.ModuleType("roll.pipeline.rlvr"),
        "roll.pipeline.rlvr.rubric_config": fake_rubric,
        "rdan_grpo.roll_live": fake_live,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    import rdan_grpo.roll_compat as compatibility

    monkeypatch.setattr(script, "_parse_args", lambda: args)
    monkeypatch.setattr(script, "verify_rtt_checkout", lambda root: RTT_BOUNDARY_SHA256)
    monkeypatch.setattr(compatibility, "install_rtt_compat", lambda root: None)
    monkeypatch.setattr(script, "build_runtime_identity", lambda *args, **kwargs: {"model": "identity"})
    monkeypatch.setattr(script, "run_runtime_parity", lambda *args, **kwargs: events.append("generated"))

    with pytest.raises(WeightReceiptError, match="model_update"):
        script.main()
    assert receipt.is_file()
    assert not success.exists()
    assert not failure.exists()
    assert "generated" not in events
    assert events == ["init", "shutdown"]


@pytest.mark.skipif(not RTT_ROOT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_rtt_boundary_revision_bytes_are_exact() -> None:
    assert verify_rtt_checkout(RTT_ROOT) == RTT_BOUNDARY_SHA256
    assert verify_rtt_boundary(RTT_ROOT) == RTT_BOUNDARY_SHA256


@pytest.mark.skipif(not RTT_ROOT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_dirty_rtt_checkout_is_rejected_before_boundary_evidence(tmp_path: Path) -> None:
    clone = tmp_path / "Rubrics-To-Tokens"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-local", str(RTT_ROOT), str(clone)],
        check=True,
    )
    subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", RTT_REVISION], check=True)
    target = clone / "roll/pipeline/base_pipeline.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(WeightReceiptError, match="checkout is dirty"):
        verify_rtt_checkout(clone)
