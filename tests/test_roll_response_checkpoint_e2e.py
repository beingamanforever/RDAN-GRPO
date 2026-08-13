from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rdan_grpo.roll_response_checkpoint import (
    ArtifactIdentity,
    CheckpointError,
    CheckpointIdentity,
    CheckpointState,
    create_checkpoint_stage,
    load_checkpoint,
    promote_checkpoint,
)


def _identity(*, horizon: int = 20, method: str = "rdan_scalar") -> CheckpointIdentity:
    return CheckpointIdentity(
        planned_horizon=horizon,
        method=method,
        method_weight=0.5,
        resolved_config_sha256="a" * 64,
        certificate=ArtifactIdentity(id="scalar-preflight-v1", sha256="b" * 64),
        data=ArtifactIdentity(id="merged-response-data-v1", sha256="f" * 64),
        revisions={"code": "c" * 40, "rtt": "d" * 40, "model": "e" * 40},
        base_checkpoint_sha256="1" * 64,
        wandb={
            "entity": "RDAN-GRPO",
            "project": "rdan-grpo-qwen3-4b",
            "run_id": "qwen-rdan-scalar-s42",
            "name": "qwen-rdan-scalar-s42",
            "group": "qwen-rdan-scalar-train",
        },
    )


def _state(step: int = 1, updates: int | None = None) -> CheckpointState:
    updates = step * 2 if updates is None else updates
    return CheckpointState(
        completed_step=step,
        optimizer_counters={0: updates, 1: updates},
        scheduler_counters={0: updates, 1: updates},
        scheduler_state={"last_epoch": step, "base_lrs": [1e-6, 1e-6]},
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


def _write_artifacts(stage: Path) -> list[str]:
    files = {
        "model/rank-0/model.distcp": b"model-0",
        "model/rank-1/model.distcp": b"model-1",
        "optimizer/rank-0/optimizer.distcp": b"optimizer-0",
        "optimizer/rank-1/optimizer.distcp": b"optimizer-1",
        "rng/driver.pt": b"rng-driver",
        "rng/infer-rank-0.pt": b"rng-infer-0",
        "rng/infer-rank-1.pt": b"rng-infer-1",
        "receipts/initial.json": b'{"status":"receipt_passed"}\n',
        "receipts/post.json": b'{"status":"receipt_passed"}\n',
        "scheduler/state.json": b'{"last_epoch":1}\n',
    }
    for relative, body in files.items():
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return list(files)


def _promoted(tmp_path: Path, *, step: int = 1) -> tuple[Path, CheckpointIdentity]:
    identity = _identity()
    stage = create_checkpoint_stage(tmp_path / "run" / "checkpoints", step)
    artifacts = _write_artifacts(stage)
    return promote_checkpoint(stage, identity=identity, state=_state(step), artifacts=artifacts), identity


def _rewrite_manifest(path: Path, update: Any) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    update(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_checkpoint_promotion_and_load_seal_complete_inventory(tmp_path: Path) -> None:
    checkpoint, identity = _promoted(tmp_path)

    assert checkpoint.name == "step-000001"
    assert not (checkpoint.parent / ".incomplete-step-000001").exists()
    manifest = load_checkpoint(checkpoint, identity=identity)
    assert manifest["status"] == "promoted"
    assert (manifest["completed_step"], manifest["next_step"], manifest["planned_horizon"]) == (1, 2, 20)
    assert manifest["optimizer_counters"] == {"0": 2, "1": 2}
    assert manifest["scheduler_counters"] == {"0": 2, "1": 2}
    assert manifest["rng_artifacts"]["driver"]["path"] == "rng/driver.pt"
    assert manifest["rng_artifacts"]["infer_0"]["path"] == "rng/infer-rank-0.pt"
    assert manifest["wandb"]["group"] == "qwen-rdan-scalar-train"
    assert manifest["receipt_links"]["post_update"]["path"] == "receipts/post.json"
    assert [entry["path"] for entry in manifest["inventory"]] == sorted(_write_artifacts_names())
    body = (checkpoint / "manifest.json").read_bytes()
    assert body == json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def test_one_pipeline_transaction_rejects_wrong_optimizer_cadence(tmp_path: Path) -> None:
    identity = _identity()
    stage = create_checkpoint_stage(tmp_path / "run" / "checkpoints", 1)
    artifacts = _write_artifacts(stage)

    with pytest.raises(CheckpointError, match="inconsistent with completed step"):
        promote_checkpoint(stage, identity=identity, state=_state(step=1, updates=32), artifacts=artifacts)


def _write_artifacts_names() -> list[str]:
    return [
        "model/rank-0/model.distcp",
        "model/rank-1/model.distcp",
        "optimizer/rank-0/optimizer.distcp",
        "optimizer/rank-1/optimizer.distcp",
        "rng/driver.pt",
        "rng/infer-rank-0.pt",
        "rng/infer-rank-1.pt",
        "receipts/initial.json",
        "receipts/post.json",
        "scheduler/state.json",
    ]


@pytest.mark.parametrize("mutation", ["corrupt", "missing", "extra"])
def test_loader_rejects_inventory_byte_and_membership_drift(tmp_path: Path, mutation: str) -> None:
    checkpoint, identity = _promoted(tmp_path)
    target = checkpoint / "model/rank-0/model.distcp"
    if mutation == "corrupt":
        target.write_bytes(b"changed")
    elif mutation == "missing":
        target.unlink()
    else:
        (checkpoint / "unowned.bin").write_bytes(b"extra")

    with pytest.raises(CheckpointError, match="corrupt|missing or extra"):
        load_checkpoint(checkpoint, identity=identity)


def test_symlinks_are_rejected_during_promotion_and_resume(tmp_path: Path) -> None:
    root = tmp_path / "run" / "checkpoints"
    stage = create_checkpoint_stage(root, 1)
    artifacts = _write_artifacts(stage)
    os.symlink(stage / "rng/driver.pt", stage / "rng/link.pt")
    artifacts.append("rng/link.pt")
    with pytest.raises(CheckpointError, match="symlink"):
        promote_checkpoint(stage, identity=_identity(), state=_state(), artifacts=artifacts)

    checkpoint, identity = _promoted(tmp_path / "second")
    target = checkpoint / "rng/driver.pt"
    target.unlink()
    os.symlink(checkpoint / "rng/infer-rank-1.pt", target)
    with pytest.raises(CheckpointError, match="symlink"):
        load_checkpoint(checkpoint, identity=identity)


def test_incomplete_stage_and_identity_or_horizon_drift_are_rejected(tmp_path: Path) -> None:
    stage = create_checkpoint_stage(tmp_path / "incomplete", 1)
    _write_artifacts(stage)
    with pytest.raises(CheckpointError, match="incomplete"):
        load_checkpoint(stage, identity=_identity())

    checkpoint, identity = _promoted(tmp_path / "promoted")
    with pytest.raises(CheckpointError, match="identity|horizon"):
        load_checkpoint(checkpoint, identity=replace(identity, method="rl_aon"))
    with pytest.raises(CheckpointError, match="identity|horizon"):
        load_checkpoint(checkpoint, identity=replace(identity, planned_horizon=21))
    drifted_wandb = {**identity.wandb, "group": "wrong-group"}
    with pytest.raises(CheckpointError, match="identity|horizon"):
        load_checkpoint(checkpoint, identity=replace(identity, wandb=drifted_wandb))


def test_promotion_never_overwrites_existing_destination(tmp_path: Path) -> None:
    root = tmp_path / "run" / "checkpoints"
    stage = create_checkpoint_stage(root, 1)
    artifacts = _write_artifacts(stage)
    destination = root / "step-000001"
    destination.mkdir()
    marker = destination / "marker"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(CheckpointError, match="already exists"):
        promote_checkpoint(stage, identity=_identity(), state=_state(), artifacts=artifacts)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert stage.exists()


def test_retry_quarantines_stale_incomplete_bytes_before_new_stage(tmp_path: Path) -> None:
    root = tmp_path / "run" / "checkpoints"
    stale = create_checkpoint_stage(root, 1)
    (stale / "partial.bin").write_bytes(b"partial-dcp")

    fresh = create_checkpoint_stage(root, 1)

    quarantine = root / ".quarantined-step-000001-attempt-000001"
    assert fresh == root.resolve() / ".incomplete-step-000001"
    assert (quarantine / "partial.bin").read_bytes() == b"partial-dcp"
    with pytest.raises(CheckpointError, match="directory name|promoted"):
        load_checkpoint(quarantine, identity=_identity())


@pytest.mark.parametrize(
    ("state", "match"),
    [
        (replace(_state(), reward_variance=0.0), "nonzero within-group"),
        (replace(_state(), clipping_fraction=1.0), "below 1"),
        (
            replace(
                _state(),
                group_diagnostics={
                    **_state().group_diagnostics,
                    "quality_active_group_rate": 0.0,
                    "quality_active_group_count": 0,
                },
            ),
            "quality active group rate",
        ),
    ],
)
def test_promotion_rejects_useless_reward_and_clipping_diagnostics(
    tmp_path: Path, state: CheckpointState, match: str
) -> None:
    stage = create_checkpoint_stage(tmp_path / "run" / "checkpoints", 1)
    artifacts = _write_artifacts(stage)

    with pytest.raises(CheckpointError, match=match):
        promote_checkpoint(stage, identity=_identity(), state=state, artifacts=artifacts)


def test_hybrid_checkpoint_requires_quality_activity(tmp_path: Path) -> None:
    diagnostics = {
        **_state().group_diagnostics,
        "quality_active_group_rate": 0.0,
        "quality_active_group_count": 0,
    }
    stage = create_checkpoint_stage(tmp_path / "run" / "checkpoints", 1)
    artifacts = _write_artifacts(stage)

    with pytest.raises(CheckpointError, match="quality active group rate"):
        promote_checkpoint(
            stage,
            identity=_identity(method="rtt_papo_response"),
            state=replace(_state(), group_diagnostics=diagnostics),
            artifacts=artifacts,
        )


@pytest.mark.parametrize(
    ("optimizer", "scheduler", "match"),
    [
        ({0: 1}, {0: 1}, "exact DP2"),
        ({0: 1, 2: 1}, {0: 1, 2: 1}, "exact DP2"),
        ({0: 1, 1: 2}, {0: 1, 1: 2}, "optimizer counters differ"),
        ({0: 1, 1: 1}, {0: 1, 1: 2}, "scheduler counters differ"),
        ({0: 1, 1: 1}, {0: 0, 1: 0}, "optimizer and scheduler counters differ"),
        ({0: 1, 1: 1}, {0: 2, 1: 2}, "optimizer and scheduler counters differ"),
    ],
)
def test_promotion_rejects_rank_and_training_counter_drift(
    tmp_path: Path,
    optimizer: dict[int, int],
    scheduler: dict[int, int],
    match: str,
) -> None:
    stage = create_checkpoint_stage(tmp_path / "run" / "checkpoints", 1)
    artifacts = _write_artifacts(stage)
    state = replace(_state(), optimizer_counters=optimizer, scheduler_counters=scheduler)

    with pytest.raises(CheckpointError, match=match):
        promote_checkpoint(stage, identity=_identity(), state=state, artifacts=artifacts)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("optimizer_counters", {"0": 1}, "exact DP2"),
        ("optimizer_counters", {"0": 1, "1": 2}, "optimizer counters differ"),
        ("scheduler_counters", {"0": 1, "1": 2}, "scheduler counters differ"),
        ("optimizer_counters", {"0": 3, "1": 3}, "optimizer and scheduler counters differ"),
    ],
)
def test_loader_rejects_rank_and_training_counter_drift(
    tmp_path: Path,
    field: str,
    value: dict[str, int],
    match: str,
) -> None:
    checkpoint, identity = _promoted(tmp_path)
    _rewrite_manifest(checkpoint, lambda manifest: manifest.__setitem__(field, value))

    with pytest.raises(CheckpointError, match=match):
        load_checkpoint(checkpoint, identity=identity)


def test_loader_rejects_counters_inconsistent_with_completed_step(tmp_path: Path) -> None:
    checkpoint, identity = _promoted(tmp_path)

    def drift_counters(manifest: dict[str, Any]) -> None:
        manifest["optimizer_counters"] = {"0": 0, "1": 0}
        manifest["scheduler_counters"] = {"0": 0, "1": 0}

    _rewrite_manifest(checkpoint, drift_counters)

    with pytest.raises(CheckpointError, match="completed step"):
        load_checkpoint(checkpoint, identity=identity)

    checkpoint, identity = _promoted(tmp_path / "zero", step=0)

    def add_updates_without_transaction(manifest: dict[str, Any]) -> None:
        manifest["optimizer_counters"] = {"0": 1, "1": 1}
        manifest["scheduler_counters"] = {"0": 1, "1": 1}

    _rewrite_manifest(checkpoint, add_updates_without_transaction)

    with pytest.raises(CheckpointError, match="completed step"):
        load_checkpoint(checkpoint, identity=identity)


@pytest.mark.parametrize(
    ("step", "updates"),
    [(0, 1), (1, 0), (2, 1)],
)
def test_promotion_rejects_pipeline_and_optimizer_count_unit_drift(
    tmp_path: Path,
    step: int,
    updates: int,
) -> None:
    stage = create_checkpoint_stage(tmp_path / "run" / "checkpoints", step)
    artifacts = _write_artifacts(stage)

    with pytest.raises(CheckpointError, match="completed step"):
        promote_checkpoint(stage, identity=_identity(), state=_state(step, updates), artifacts=artifacts)


def test_artifact_paths_cannot_escape_the_stage(tmp_path: Path) -> None:
    root = tmp_path / "run" / "checkpoints"
    stage = create_checkpoint_stage(root, 1)
    _write_artifacts(stage)
    (stage.parent / "outside.bin").write_bytes(b"outside")

    with pytest.raises(CheckpointError, match="relative"):
        promote_checkpoint(stage, identity=_identity(), state=_state(), artifacts=["../outside.bin"])


def test_loader_rejects_non_promoted_status_and_invalid_step_linkage(tmp_path: Path) -> None:
    checkpoint, identity = _promoted(tmp_path / "status")
    _rewrite_manifest(checkpoint, lambda manifest: manifest.__setitem__("status", "incomplete"))
    with pytest.raises(CheckpointError, match="promoted"):
        load_checkpoint(checkpoint, identity=identity)

    checkpoint, identity = _promoted(tmp_path / "linkage")
    _rewrite_manifest(checkpoint, lambda manifest: manifest.__setitem__("next_step", 4))
    with pytest.raises(CheckpointError, match="linkage"):
        load_checkpoint(checkpoint, identity=identity)
