from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from rdan_grpo import response_identity
from rdan_grpo.response_readiness import ResponseReadinessError
from rdan_grpo.roll_bridge import require_train_certificate

ROOT = Path(__file__).resolve().parents[1]


def _module() -> types.ModuleType:
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
        "--readiness-receipt",
        "readiness.json",
        "--readiness-bootstrap",
        "bootstrap.json",
        "--readiness-evidence",
        "judge.json",
        "--readiness-evidence",
        "parity.json",
        "--readiness-evidence",
        "certificate.json",
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
        "recovery",
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


def test_cli_requires_response_readiness_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    values = {
        "--config": "train.yaml",
        "--rtt-root": "rtt",
        "--snapshot": "snapshot",
        "--certificate": "certificate.json",
        "--preflight-config": "preflight.yaml",
        "--evaluator-rows": "rows.jsonl",
        "--program": "program.json",
        "--runtime-parity": "parity.json",
        "--readiness-bootstrap": "bootstrap.json",
        "--checkpoint-root": "checkpoints",
        "--run-dir": "run",
        "--method": "rdan_scalar",
        "--quality-weight": "0.5",
        "--planned-horizon": "500",
        "--stage": "recovery",
        "--stop-after-step": "1",
    }
    argv = ["run_response_train.py", *(item for pair in values.items() for item in pair)]
    for name in ("judge.json", "parity.json", "certificate.json"):
        argv.extend(("--readiness-evidence", name))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        module._parse_args()


@pytest.mark.parametrize("message", ["differs", "stale"])
def test_invalid_response_readiness_fails_at_runner_boundary(
    message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    files = {name: tmp_path / name for name in ("program.json", "readiness.json", "bootstrap.json", "evidence.json")}
    for path in files.values():
        path.write_text("{}\n", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "readiness_receipt": files["readiness.json"],
            "readiness_bootstrap": files["bootstrap.json"],
            "readiness_evidence": [files["evidence.json"]] * 3,
        },
    )()
    called = False

    def reject(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise ResponseReadinessError(message)

    monkeypatch.setattr(module, "validate_response_readiness", reject)
    with pytest.raises(ResponseReadinessError, match=message):
        module._require_response_readiness(args, files["program.json"], "a" * 40)
    assert called is True


def test_valid_response_readiness_passes_before_pipeline_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    paths = [
        tmp_path / name
        for name in ("program.json", "readiness.json", "bootstrap.json", "judge.json", "parity.json", "no-update.json")
    ]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    program, receipt, bootstrap, *evidence = paths
    args = type(
        "Args",
        (),
        {"readiness_receipt": receipt, "readiness_bootstrap": bootstrap, "readiness_evidence": evidence},
    )()
    observed = []
    monkeypatch.setattr(
        module,
        "validate_response_readiness",
        lambda *values, **kwargs: observed.append((values, kwargs)) or {"status": "ready"},
    )

    assert module._require_response_readiness(args, program, "a" * 40) == {"status": "ready"}
    assert observed[0][0] == (receipt, program, bootstrap, tuple(evidence))
    source = (ROOT / "scripts/run_response_train.py").read_text(encoding="utf-8")
    assert source.index("_require_response_readiness(args") < source.index("sys.path.insert(0, str(rtt_root))")
    assert source.index("_require_response_readiness(args") < source.index("build_response_training_pipeline(")


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
            "production_resolved_config_sha256": "c" * 64,
            "preflight_train_config_sha256": "d" * 64,
            "preflight_resolved_config_sha256": "e" * 64,
        },
    }
    path = tmp_path / "parity.json"
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")
    assert module._runtime_parity(path, "a" * 64, "c" * 64, "d" * 64, "e" * 64) == artifact
    with pytest.raises(ValueError, match="different composed launch configs"):
        module._runtime_parity(path, "a" * 64, "f" * 64, "d" * 64, "e" * 64)


def test_refreshed_parent_pin_cannot_reuse_old_composed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    for name in (
        "qwen_scalar_train.yaml",
        "qwen_rtt_papo_response_train.yaml",
        "qwen_rtt_papo_response_preflight.yaml",
    ):
        shutil.copy2(ROOT / "configs/roll" / name, tmp_path / name)
    parent = tmp_path / "qwen_scalar_train.yaml"
    production = tmp_path / "qwen_rtt_papo_response_train.yaml"
    preflight = tmp_path / "qwen_rtt_papo_response_preflight.yaml"
    monkeypatch.setenv("RDAN_MODEL_SNAPSHOT", "/models/qwen")
    monkeypatch.setattr(module, "_compose_config", _compose_test_config)
    program = {"launch_train_config": {"hydra_parent": {"sha256": response_identity.file_sha256(parent)}}}
    old_production = response_identity.canonical_resolved_config_sha256(module._compose_config(production))
    old_preflight = response_identity.canonical_resolved_config_sha256(module._compose_config(preflight))
    parent.write_text(parent.read_text(encoding="utf-8").replace("seed: 240520", "seed: 240521"), encoding="utf-8")
    program["launch_train_config"]["hydra_parent"]["sha256"] = response_identity.file_sha256(parent)
    new_production = response_identity.canonical_resolved_config_sha256(module._compose_config(production))
    new_preflight = response_identity.canonical_resolved_config_sha256(module._compose_config(preflight))
    assert old_production != new_production
    assert old_preflight != new_preflight

    parity = {
        "status": "parity_passed",
        "runtime_backend": {
            "actor_train_strategy": "fsdp2_train",
            "actor_infer_strategy": "hf_infer",
            "transformer_impl": "huggingface",
            "rtt_revision": module.RTT_REVISION,
            "production_train_config_sha256": "a" * 64,
            "production_resolved_config_sha256": old_production,
            "preflight_train_config_sha256": "b" * 64,
            "preflight_resolved_config_sha256": old_preflight,
        },
    }
    parity_path = tmp_path / "parity.json"
    parity_path.write_text(json.dumps(parity), encoding="utf-8")
    with pytest.raises(ValueError, match="different composed launch configs"):
        module._runtime_parity(parity_path, "a" * 64, new_production, "b" * 64, new_preflight)

    body = {
        "schema_version": 2,
        "ready": True,
        "method": "rtt_papo_response",
        "quality_weight": 0.5,
        "config_sha256": "b" * 64,
        "source_sha256": {
            "train_resolved_config": old_production,
            "preflight_resolved_config": old_preflight,
        },
        "metrics": {},
        "reasons": [],
    }
    certificate = {
        "certificate_id": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest(),
        **body,
    }
    certificate_path = tmp_path / "no-update.json"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(ValueError, match="source hashes do not match"):
        require_train_certificate(
            certificate_path,
            method="rtt_papo_response",
            quality_weight=0.5,
            source_sha256={
                "train_resolved_config": new_production,
                "preflight_resolved_config": new_preflight,
            },
        )


def _compose_test_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = payload.pop("defaults", [])
    merged: dict = {}
    for entry in defaults:
        if entry == "_self_":
            continue
        _merge_config(merged, _compose_test_config(path.with_name(f"{entry}.yaml")))
    _merge_config(merged, payload)
    return merged


def _merge_config(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_config(target[key], value)
        else:
            target[key] = value


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
            config, response, certificate, model, "e" * 64, "d" * 40, tmp_path, False, "recovery"
        )
        resumed, resumed_identity = module._tracking(
            config, response, certificate, model, "e" * 64, "d" * 40, tmp_path, True, "recovery"
        )
    finally:
        module._git_revision = original

    assert first_identity == resumed_identity
    assert first["id"] == resumed["id"]
    assert first["resume"] == "allow"
    assert resumed["resume"] == "must"
    assert resumed["metadata"]["stage"] == "resume"

    pilot, pilot_identity = module._tracking(
        config, response, certificate, model, "e" * 64, "d" * 40, tmp_path, False, "pilot"
    )
    assert pilot_identity != first_identity
    assert pilot["id"] != first["id"]


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
    [
        ("recovery", None, "recovery"),
        ("recovery", 20, "recovery"),
        ("pilot", None, "pilot"),
        ("pilot", 2, "pilot"),
        ("train", 20, "complete"),
    ],
)
def test_stage_rejects_ambiguous_or_partial_runs(stage: str, stop: int | None, message: str) -> None:
    module = _module()
    args = type("Args", (), {"stage": stage, "stop_after_step": stop})()
    with pytest.raises(ValueError, match=message):
        module._validate_stage(args)

    module._validate_stage(type("Args", (), {"stage": "recovery", "stop_after_step": 1})())
    module._validate_stage(type("Args", (), {"stage": "recovery", "stop_after_step": 2})())
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


def test_response_evidence_revision_rejects_dirty_or_mismatched_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "initial"], check=True, capture_output=True)
    revision = response_identity.clean_repository_revision(tmp_path)
    assert len(revision) == 40
    assert response_identity.clean_repository_revision(tracked) == revision
    with pytest.raises(response_identity.ResponseIdentityError, match="differs"):
        response_identity.clean_repository_revision(tmp_path, "f" * 40)

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(response_identity.ResponseIdentityError, match="clean Git worktree"):
        response_identity.clean_repository_revision(tmp_path)


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
        ("src/rdan_grpo/evaluator_cert.py", "evaluator_cert"),
        ("src/rdan_grpo/hir.py", "hir"),
        ("src/rdan_grpo/judge.py", "judge"),
        ("src/rdan_grpo/program.py", "program"),
        ("src/rdan_grpo/roll_compat.py", "roll_compat"),
        ("src/rdan_grpo/roll_response_config.py", "response_config"),
        ("src/rdan_grpo/response_readiness.py", "response_readiness"),
        ("src/rdan_grpo/roll_same_backend.py", "roll_same_backend"),
        ("src/rdan_grpo/roll_same_backend_live.py", "roll_same_backend_live"),
        ("src/rdan_grpo/roll_weight_receipt.py", "roll_weight_receipt"),
        ("src/rdan_grpo/runtime_parity.py", "runtime_parity"),
        ("src/rdan_grpo/rubrichub_rules.py", "rubrichub_rules"),
        ("src/rdan_grpo/safe_rule.py", "safe_rule"),
        ("src/rdan_grpo/scalar_data.py", "scalar_data"),
        ("src/rdan_grpo/wandb_tracking.py", "wandb_tracking"),
        ("src/rdan_grpo/weight_receipt.py", "weight_receipt"),
        ("scripts/run_response_train.py", "train_cli"),
        ("scripts/run_roll_preflight.py", "preflight_cli"),
        ("scripts/run_response_readiness.py", "readiness_cli"),
        ("scripts/run_same_backend_parity.py", "same_backend_cli"),
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
    for name in (
        "run_response_train.py",
        "run_roll_preflight.py",
        "run_response_readiness.py",
        "run_same_backend_parity.py",
    ):
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
        train_resolved_config_sha256="a" * 64,
        preflight_resolved_config_sha256="b" * 64,
    )
    target = tmp_path / relative
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = response_identity.response_source_hashes(
        source,
        evaluator_rows=evaluator,
        train_config=train,
        preflight_config=preflight,
        program=tmp_path / "program.json",
        train_resolved_config_sha256="a" * 64,
        preflight_resolved_config_sha256="b" * 64,
    )

    assert before[key] != after[key]
