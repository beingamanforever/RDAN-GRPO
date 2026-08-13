from __future__ import annotations

import hashlib
import json
import os
import runpy
import struct
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rdan_grpo.response_publish import (
    HF_REPO_ID,
    PublishError,
    load_publish_identity,
    publish_response_model,
)
from rdan_grpo.roll_response_checkpoint import (
    ArtifactIdentity,
    CheckpointError,
    CheckpointIdentity,
    CheckpointState,
    create_checkpoint_stage,
    promote_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
HubUploader = runpy.run_path(ROOT / "scripts/publish_response_model.py")["HubUploader"]


def _identity(*, horizon: int = 2, method: str = "rdan_scalar") -> CheckpointIdentity:
    return CheckpointIdentity(
        planned_horizon=horizon,
        method=method,
        method_weight=0.5,
        resolved_config_sha256="a" * 64,
        certificate=ArtifactIdentity(id="preflight-v1", sha256="b" * 64),
        data=ArtifactIdentity(id="merged-v1", sha256="c" * 64),
        revisions={"code": "d" * 40, "rtt": "e" * 40, "model": "f" * 40},
        base_checkpoint_sha256="1" * 64,
        wandb={
            "entity": "RDAN-GRPO",
            "project": "rdan-grpo-qwen3-4b",
            "run_id": "qwen-rdan-scalar-s42",
            "name": "qwen-rdan-scalar-s42",
            "group": "qwen-rdan-scalar-train",
        },
    )


def _state(step: int) -> CheckpointState:
    updates = step * 2
    return CheckpointState(
        completed_step=step,
        optimizer_counters={0: updates, 1: updates},
        scheduler_counters={0: updates, 1: updates},
        scheduler_state={"last_epoch": step},
        rng_artifacts={
            "driver": "rng/driver.pt",
            "infer_0": "rng/infer-rank-0.pt",
            "infer_1": "rng/infer-rank-1.pt",
        },
        metrics={"actor/total_loss": 0.4, "actor/grad_norm": 1.2},
        peak_memory={0: 73_000_000_000, 1: 74_000_000_000},
        reward_variance=0.25,
        group_diagnostics={
            "group_count": 10,
            "response_active_group_count": 8,
            "response_active_group_rate": 0.8,
            "quality_active_group_count": 2,
            "quality_active_group_rate": 0.2,
            "selected_reward_variance_mean": 0.25,
        },
        clipping_fraction=0.1,
        receipt_links={"initial": "receipts/initial.json", "post_update": "receipts/post.json"},
    )


def _write_actor(actor: Path, *, indexed: bool = False, missing: str | None = None, unknown: bool = False) -> None:
    actor.mkdir(parents=True)
    values = {
        "config.json": {"model_type": "qwen3"},
        "generation_config.json": {"do_sample": True},
        "tokenizer.json": {"version": "1.0"},
        "tokenizer_config.json": {"chat_template": "{{ messages }}"},
    }
    for name, value in values.items():
        if name != missing:
            _write_json(actor / name, value)
    (actor / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (actor / "vocab.json").write_text("{}\n", encoding="utf-8")
    if indexed:
        _write_safetensors(actor / "model-00001-of-00002.safetensors", {"model.a": b"ab"})
        _write_safetensors(actor / "model-00002-of-00002.safetensors", {"model.b": b"cd"})
        _write_json(
            actor / "model.safetensors.index.json",
            {
                "metadata": {"total_size": 32},
                "weight_map": {
                    "model.a": "model-00001-of-00002.safetensors",
                    "model.b": "model-00002-of-00002.safetensors",
                },
            },
        )
    else:
        _write_safetensors(actor / "model.safetensors", {"model.weight": b"abcd"})
    (actor / "dcp").mkdir()
    (actor / "dcp/__0_0.distcp").write_bytes(b"dcp")
    for rank in range(2):
        _write_json(actor / f"rdan-response-counters-rank-{rank}.json", {"rank": rank})
    if unknown:
        (actor / "optimizer.pt").write_bytes(b"must-not-upload")


def _checkpoint(
    tmp_path: Path,
    *,
    step: int = 2,
    identity: CheckpointIdentity | None = None,
    indexed: bool = False,
    missing: str | None = None,
    unknown: bool = False,
    missing_operational: str | None = None,
) -> tuple[Path, CheckpointIdentity]:
    identity = identity or _identity()
    stage = create_checkpoint_stage(tmp_path / "checkpoints", step)
    _write_actor(stage / "actor", indexed=indexed, missing=missing, unknown=unknown)
    if missing_operational == "dcp":
        (stage / "actor/dcp/__0_0.distcp").unlink()
        (stage / "actor/dcp").rmdir()
    elif missing_operational == "counter":
        (stage / "actor/rdan-response-counters-rank-1.json").unlink()
    files = {
        "rng/driver.pt": b"driver",
        "rng/infer-rank-0.pt": b"infer-0",
        "rng/infer-rank-1.pt": b"infer-1",
        "receipts/initial.json": b"{}\n",
        "receipts/post.json": b"{}\n",
        "scheduler/state.json": b"{}\n",
    }
    for relative, body in files.items():
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    artifacts = [path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()]
    return promote_checkpoint(stage, identity=identity, state=_state(step), artifacts=artifacts), identity


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    offset = 0
    header: dict[str, Any] = {}
    for name, body in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(body)], "data_offsets": [offset, offset + len(body)]}
        offset += len(body)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(tensors.values()))


@pytest.mark.parametrize("indexed", [False, True])
def test_final_checkpoint_publishes_only_allowlisted_hf_files(tmp_path: Path, indexed: bool) -> None:
    checkpoint, identity = _checkpoint(tmp_path, indexed=indexed)
    receipt_path = tmp_path / "artifacts/publish.json"
    calls: list[dict[str, Any]] = []

    def upload(**kwargs: Any) -> str:
        calls.append(kwargs)
        assert not receipt_path.exists()
        return "2" * 40

    receipt = publish_response_model(
        checkpoint,
        identity=identity,
        receipt_path=receipt_path,
        uploader=upload,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == HF_REPO_ID
    assert call["revision"] == "rdan-scalar"
    assert call["method"] == "rdan_scalar"
    staged = {file.path_in_repo for file in call["files"]}
    assert {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"} <= staged
    assert not any("dcp" in name or "counter" in name or "manifest" in name for name in staged)
    assert all(file.local_path.parent != checkpoint / "actor" for file in call["files"])
    assert all(not file.local_path.exists() for file in call["files"])
    assert [entry["path"] for entry in receipt["files"]] == sorted(staged)
    assert (
        receipt["checkpoint_manifest_sha256"]
        == hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()
    )
    assert receipt_path.read_bytes() == (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert "token" not in receipt and "hf_token" not in receipt


def test_nonfinal_wrong_destination_and_revision_are_rejected_before_upload(tmp_path: Path) -> None:
    identity = _identity(horizon=2)
    checkpoint, _ = _checkpoint(tmp_path, step=1, identity=identity)
    called = False

    def upload(**kwargs: Any) -> str:
        nonlocal called
        called = True
        return "2" * 40

    with pytest.raises(PublishError, match="planned horizon"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=tmp_path / "nonfinal.json",
            uploader=upload,
        )
    with pytest.raises(PublishError, match="must publish"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=tmp_path / "repo.json",
            uploader=upload,
            repo_id="other/model",
        )
    with pytest.raises(PublishError, match="revision"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=tmp_path / "revision.json",
            uploader=upload,
            revision="main",
        )
    assert called is False


@pytest.mark.parametrize(
    ("missing", "unknown", "message"),
    [("tokenizer.json", False, "missing"), (None, True, "unknown")],
)
def test_missing_or_unknown_actor_files_are_rejected(
    tmp_path: Path, missing: str | None, unknown: bool, message: str
) -> None:
    checkpoint, identity = _checkpoint(tmp_path, missing=missing, unknown=unknown)

    with pytest.raises(PublishError, match=message):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=tmp_path / "publish.json",
            uploader=lambda **kwargs: "2" * 40,
        )


@pytest.mark.parametrize("missing_operational", ["dcp", "counter"])
def test_missing_dcp_or_rank_evidence_is_rejected(tmp_path: Path, missing_operational: str) -> None:
    checkpoint, identity = _checkpoint(tmp_path, missing_operational=missing_operational)

    with pytest.raises(PublishError, match="DCP or rank evidence"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=tmp_path / "publish.json",
            uploader=lambda **kwargs: "2" * 40,
        )


@pytest.mark.parametrize("mutation", ["corrupt", "symlink"])
def test_corrupt_or_symlinked_checkpoint_is_rejected(tmp_path: Path, mutation: str) -> None:
    checkpoint, identity = _checkpoint(tmp_path)
    target = checkpoint / "actor/config.json"
    if mutation == "corrupt":
        target.write_text("{}\n", encoding="utf-8")
    else:
        target.unlink()
        os.symlink(checkpoint / "actor/tokenizer_config.json", target)

    with pytest.raises(CheckpointError, match="corrupt|symlink"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=tmp_path / "publish.json",
            uploader=lambda **kwargs: "2" * 40,
        )


def test_invalid_commit_or_upload_error_never_seals_receipt(tmp_path: Path) -> None:
    checkpoint, identity = _checkpoint(tmp_path)
    receipt = tmp_path / "publish.json"

    with pytest.raises(PublishError, match="commit hash"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=receipt,
            uploader=lambda **kwargs: "not-a-commit",
        )
    assert not receipt.exists()

    def fail(**kwargs: Any) -> str:
        raise ConnectionError("hub unavailable")

    with pytest.raises(ConnectionError, match="hub unavailable"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=receipt,
            uploader=fail,
        )
    assert not receipt.exists()


def test_mutated_staged_bytes_block_receipt(tmp_path: Path) -> None:
    checkpoint, identity = _checkpoint(tmp_path)
    receipt = tmp_path / "publish.json"

    def mutate(**kwargs: Any) -> str:
        target = next(file.local_path for file in kwargs["files"] if file.path_in_repo == "config.json")
        target.chmod(0o600)
        target.write_text('{"model_type":"other"}\n', encoding="utf-8")
        return "2" * 40

    with pytest.raises(PublishError, match="changed during publication"):
        publish_response_model(
            checkpoint,
            identity=identity,
            receipt_path=receipt,
            uploader=mutate,
        )
    assert not receipt.exists()
    assert json.loads((checkpoint / "actor/config.json").read_text(encoding="utf-8"))["model_type"] == "qwen3"


def test_hub_uploader_reconciles_revision_to_exact_file_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[Any] = []

    class FakeApi:
        def __init__(self, token: str) -> None:
            events.append(("init", token))

        def create_branch(self, *args: Any, **kwargs: Any) -> None:
            events.append(("branch", args, kwargs))

        def model_info(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            events.append(("info", args, kwargs))
            return SimpleNamespace(sha="4" * 40)

        def list_repo_files(self, *args: Any, **kwargs: Any) -> list[str]:
            events.append(("list", args, kwargs))
            return ["config.json", "model.safetensors.index.json", "old.safetensors", "dcp/optimizer"]

        def create_commit(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            events.append(("commit", args, kwargs))
            return SimpleNamespace(oid="3" * 40)

    fake_hub = SimpleNamespace(
        __version__="0.36.2",
        HfApi=FakeApi,
        CommitOperationAdd=lambda **kwargs: ("add", kwargs),
        CommitOperationDelete=lambda **kwargs: ("delete", kwargs),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    uploader = HubUploader("secret")

    commit = uploader(
        repo_id=HF_REPO_ID,
        revision="rdan-scalar",
        method="rdan_scalar",
        completed_step=2,
        files=(
            SimpleNamespace(path_in_repo="config.json", local_path=config),
            SimpleNamespace(path_in_repo="model.safetensors", local_path=config),
        ),
    )

    assert commit == "3" * 40
    operations = next(event[2]["operations"] for event in events if event[0] == "commit")
    commit_call = next(event for event in events if event[0] == "commit")
    list_call = next(event for event in events if event[0] == "list")
    assert list_call[2]["revision"] == "4" * 40
    assert commit_call[2]["parent_commit"] == "4" * 40
    assert operations[:3] == [
        ("delete", {"path_in_repo": "dcp/optimizer"}),
        ("delete", {"path_in_repo": "model.safetensors.index.json"}),
        ("delete", {"path_in_repo": "old.safetensors"}),
    ]
    assert {operation[1]["path_in_repo"] for operation in operations[3:]} == {
        "config.json",
        "model.safetensors",
    }


def test_hub_uploader_propagates_parent_conflict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class ConflictApi:
        def __init__(self, token: str) -> None:
            pass

        def create_branch(self, *args: Any, **kwargs: Any) -> None:
            pass

        def model_info(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(sha="4" * 40)

        def list_repo_files(self, *args: Any, **kwargs: Any) -> list[str]:
            return []

        def create_commit(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            raise RuntimeError("parent commit conflict")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            __version__="0.36.2",
            HfApi=ConflictApi,
            CommitOperationAdd=lambda **kwargs: ("add", kwargs),
            CommitOperationDelete=lambda **kwargs: ("delete", kwargs),
        ),
    )
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="parent commit conflict"):
        HubUploader("secret")(
            repo_id=HF_REPO_ID,
            revision="rdan-scalar",
            method="rdan_scalar",
            completed_step=2,
            files=(SimpleNamespace(path_in_repo=model.name, local_path=model),),
        )


def test_identity_loader_requires_canonical_regular_exact_schema(tmp_path: Path) -> None:
    identity = _identity()
    path = tmp_path / "identity.json"
    value = asdict(identity)
    _write_json(path, value)
    assert load_publish_identity(path) == identity

    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(PublishError, match="canonical"):
        load_publish_identity(path)

    target = tmp_path / "target.json"
    _write_json(target, asdict(replace(identity, method="rl_aon")))
    path.unlink()
    os.symlink(target, path)
    with pytest.raises(PublishError, match="regular"):
        load_publish_identity(path)
