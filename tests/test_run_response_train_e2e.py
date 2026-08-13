from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from rdan_grpo import response_identity

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_response_train.py"
    spec = importlib.util.spec_from_file_location("test_run_response_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_separate_one_step_and_fresh_resume_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    base = [
        "run_response_train.py",
        "--config",
        "train.yaml",
        "--rtt-root",
        "rtt",
        "--snapshot",
        "snapshot",
        "--certificate",
        "certificate.json",
        "--preflight-config",
        "preflight.yaml",
        "--evaluator-rows",
        "rows.jsonl",
        "--program",
        "program.json",
        "--runtime-parity",
        "parity.json",
        "--checkpoint-root",
        "checkpoints",
        "--run-dir",
        "run",
        "--method",
        "rdan_scalar",
        "--quality-weight",
        "0.5",
        "--planned-horizon",
        "500",
        "--stage",
        "pilot",
    ]
    monkeypatch.setattr(sys, "argv", [*base, "--stop-after-step", "1"])
    one_step = module._parse_args()
    assert (one_step.planned_horizon, one_step.stop_after_step, one_step.resume_checkpoint) == (500, 1, None)

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--stop-after-step", "2", "--resume-checkpoint", "checkpoints/step-000001"],
    )
    resumed = module._parse_args()
    assert resumed.planned_horizon == 500
    assert resumed.stop_after_step == 2
    assert resumed.resume_checkpoint == Path("checkpoints/step-000001")


def test_runtime_parity_must_link_exact_production_config(tmp_path: Path) -> None:
    module = _module()
    artifact = {
        "status": "parity_passed",
        "runtime_backend": {
            "actor_train_strategy": "fsdp2_train",
            "actor_infer_strategy": "hf_infer",
            "transformer_impl": "huggingface",
            "rtt_revision": module.RTT_REVISION,
            "production_train_config_sha256": "a" * 64,
        },
    }
    path = tmp_path / "parity.json"
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")
    assert module._runtime_parity(path, "a" * 64) == artifact
    with pytest.raises(ValueError, match="different production config"):
        module._runtime_parity(path, "b" * 64)


def test_tracking_run_id_is_identical_across_fresh_resume(tmp_path: Path) -> None:
    module = _module()
    config = type("Config", (), {"seed": 240520})()
    response = module.ResponseConfig("rdan_scalar", 0.5, None, "a" * 64)
    certificate = {"source_sha256": {"dataset": "b" * 64}}
    model = {"snapshot_sha256": "c" * 64}
    original = module._git_revision
    module._git_revision = lambda path: "d" * 40
    try:
        first, first_identity = module._tracking(
            config, response, certificate, model, "e" * 64, "d" * 40, tmp_path, False, "pilot"
        )
        resumed, resumed_identity = module._tracking(
            config, response, certificate, model, "e" * 64, "d" * 40, tmp_path, True, "pilot"
        )
    finally:
        module._git_revision = original

    assert first_identity == resumed_identity
    assert first["id"] == resumed["id"]
    assert first["resume"] == "allow"
    assert resumed["resume"] == "must"
    assert resumed["metadata"]["stage"] == "pilot"


def test_tracking_accepts_rtt_papo_response(tmp_path: Path) -> None:
    module = _module()
    response = module.ResponseConfig("rtt_papo_response", 0.5, None, "a" * 64)
    config = type("Config", (), {"seed": 240520})()
    certificate = {"source_sha256": {"train_config": "b" * 64}}
    model = {"snapshot_sha256": "c" * 64}

    kwargs, identity = module._tracking(
        config,
        response,
        certificate,
        model,
        "d" * 64,
        "e" * 40,
        tmp_path,
        False,
        "pilot",
    )

    assert kwargs["metadata"]["method"] == "rtt-papo-response"
    assert identity["name"] == "qwen-rtt-papo-response-pilot-s240520"


def test_current_launch_gate_rejects_later_methods_explicitly() -> None:
    module = _module()
    module._require_current_launch_method("rtt_papo_response")
    for method in ("rl_csr", "rl_aon", "rl_mix"):
        with pytest.raises(ValueError, match="later method-scoped lifecycle freeze"):
            module._require_current_launch_method(method)


@pytest.mark.parametrize(
    ("stage", "stop", "message"),
    [("pilot", None, "pilot"), ("pilot", 21, "pilot"), ("train", 20, "complete")],
)
def test_stage_rejects_ambiguous_or_partial_runs(stage: str, stop: int | None, message: str) -> None:
    module = _module()
    args = type("Args", (), {"stage": stage, "stop_after_step": stop})()
    with pytest.raises(ValueError, match=message):
        module._validate_stage(args)

    module._validate_stage(type("Args", (), {"stage": "pilot", "stop_after_step": 20})())
    module._validate_stage(type("Args", (), {"stage": "train", "stop_after_step": None})())


def test_checkpoint_identity_is_immutable_across_resume(tmp_path: Path) -> None:
    module = _module()
    identity = module.CheckpointIdentity(
        planned_horizon=500,
        method="rdan_scalar",
        method_weight=0.5,
        resolved_config_sha256="a" * 64,
        certificate=module.ArtifactIdentity("cert", "b" * 64),
        data=module.ArtifactIdentity("data", "c" * 64),
        revisions={"code": "d" * 40, "rtt": "e" * 40, "model": "f" * 40},
        base_checkpoint_sha256="1" * 64,
        wandb={"entity": "RDAN-GRPO", "project": "rdan-grpo-qwen3-4b", "run_id": "run", "name": "n", "group": "g"},
    )
    path = module._bind_checkpoint_identity(tmp_path, identity, False)
    assert json.loads(path.read_text(encoding="utf-8")) == asdict(identity)
    assert module._bind_checkpoint_identity(tmp_path, identity, True) == path

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        module._bind_checkpoint_identity(tmp_path, identity, True)


def test_git_revision_rejects_dirty_worktree(tmp_path: Path) -> None:
    module = _module()
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "initial"], check=True, capture_output=True)
    assert len(module._git_revision(tmp_path)) == 40

    untracked = tmp_path / "untracked.txt"
    untracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git worktree"):
        module._git_revision(tmp_path)
    untracked.unlink()

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git worktree"):
        module._git_revision(tmp_path)


def test_certificate_generator_and_training_share_one_source_identity_function() -> None:
    train = _module()
    preflight_path = ROOT / "scripts/run_roll_preflight.py"
    spec = importlib.util.spec_from_file_location("test_run_roll_preflight_identity", preflight_path)
    assert spec is not None and spec.loader is not None
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    assert preflight.response_source_hashes is train.response_source_hashes
    assert train.response_source_hashes.__module__ == "rdan_grpo.response_identity"


@pytest.mark.parametrize(
    ("relative", "key"),
    [
        ("src/rdan_grpo/roll_response_config.py", "response_config"),
        ("src/rdan_grpo/wandb_tracking.py", "wandb_tracking"),
        ("scripts/run_response_train.py", "train_cli"),
        ("scripts/run_roll_preflight.py", "preflight_cli"),
    ],
)
def test_response_identity_changes_with_every_launch_semantic_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    key: str,
) -> None:
    source = tmp_path / "src/rdan_grpo"
    shutil.copytree(ROOT / "src/rdan_grpo", source)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_response_train.py", "run_roll_preflight.py"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    evaluator = tmp_path / "evaluator.jsonl"
    train = tmp_path / "train.yaml"
    preflight = tmp_path / "preflight.yaml"
    for path in (evaluator, train, preflight):
        path.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(response_identity, "lifecycle_source_hashes", lambda _: {"lifecycle": "f" * 64})

    before = response_identity.response_source_hashes(
        source,
        evaluator_rows=evaluator,
        train_config=train,
        preflight_config=preflight,
        program=tmp_path / "program.json",
    )
    target = tmp_path / relative
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = response_identity.response_source_hashes(
        source,
        evaluator_rows=evaluator,
        train_config=train,
        preflight_config=preflight,
        program=tmp_path / "program.json",
    )

    assert before[key] != after[key]
