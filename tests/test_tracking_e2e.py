from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import rdan_grpo.tracking as tracking
from rdan_grpo.tracking import (
    RdanWandbTracker,
    WandbTrackingError,
    canonical_config_sha256,
    deterministic_run_id,
    register_wandb_tracker,
    verify_rtt_tracking,
)

ROOT = Path(__file__).resolve().parents[1]
RTT_ROOT = ROOT.parent / "Rubrics-To-Tokens"


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, Any], int | None, dict[str, Any]]] = []
        self.artifacts: list[tuple[Any, list[str]]] = []
        self.finished = False

    def log(self, values: dict[str, Any], step: int | None, **kwargs: Any) -> None:
        self.logs.append((values, step, kwargs))

    def log_artifact(self, artifact: Any, aliases: list[str]) -> Any:
        self.artifacts.append((artifact, aliases))
        return artifact

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []
        self.dirs: list[str] = []
        self.waited = False

    def add_file(self, path: str) -> None:
        self.files.append(path)

    def add_dir(self, path: str) -> None:
        self.dirs.append(path)

    def wait(self) -> None:
        self.waited = True


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("wandb")
    module.calls = []
    module.runs = []
    module.Artifact = FakeArtifact

    def init(**kwargs: Any) -> FakeRun:
        run = FakeRun()
        module.calls.append(kwargs)
        module.runs.append(run)
        return run

    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "test-key-present-only")
    return module


def _metadata(kind: str = "train", config: dict[str, Any] | None = None) -> dict[str, Any]:
    identity = {
        "resolved_config_sha256": canonical_config_sha256({} if config is None else config),
        "model_revision": "b" * 40,
        "data_sha256": "c" * 64,
        "code_revision": "d" * 40,
        "checkpoint_sha256": "e" * 64,
    }
    if kind == "train":
        return {"kind": "train", "method": "rdan-scalar", "stage": "pilot", "seed": 240520, **identity}
    return {"kind": "eval", "method": "base", "benchmark": "ifeval", "seed": 42, **identity}


def _kwargs(run_dir: Path, metadata: dict[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
    metadata = metadata or _metadata()
    kind = metadata["kind"]
    if kind == "train":
        group = f"qwen-{metadata['method']}"
        name = f"{group}-{metadata['stage']}-s{metadata['seed']}"
    else:
        group = f"qwen-{metadata['method']}-eval"
        name = f"{group}-{metadata['benchmark']}-s{metadata['seed']}"
    values = {
        "entity": "RDAN-GRPO",
        "project": "rdan-grpo-qwen3-4b",
        "group": group,
        "name": name,
        "job_type": kind,
        "id": deterministic_run_id(metadata),
        "resume": "allow",
        "tags": [metadata["method"], kind, f"seed-{metadata['seed']}"],
        "notes": "Pinned Qwen experiment",
        "log_dir": str(run_dir),
        "settings": {"console": "off", "silent": True},
        "metadata": metadata,
    }
    values.update(changes)
    return values


@pytest.mark.parametrize("resume", ["allow", "must"])
def test_all_init_fields_and_resume_survive_with_exact_config(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
    resume: str,
) -> None:
    config = {
        "learning_rate": 1e-6,
        "provider": {"max_tokens": 4096},
        "nested": [{"dtype": "bf16"}],
    }
    metadata = _metadata(config=config)
    tracker = RdanWandbTracker(config, **_kwargs(tmp_path, metadata, resume=resume))
    call = fake_wandb.calls[0]
    expected = _kwargs(tmp_path, metadata, resume=resume)
    for key in ("entity", "project", "group", "name", "job_type", "id", "resume", "tags", "notes", "settings"):
        assert call[key] == expected[key]
    assert call["dir"] == str(tmp_path)
    assert call["config"] == {
        "learning_rate": 1e-6,
        "provider": {"max_tokens": 4096},
        "nested": [{"dtype": "bf16"}],
        "rdan_identity": {
            key: expected["metadata"][key]
            for key in (
                "checkpoint_sha256",
                "code_revision",
                "data_sha256",
                "model_revision",
                "resolved_config_sha256",
            )
        },
    }

    tracker.log({"loss": 0.25, "auth_token": "sk-or-v1-metric-secret"}, 3, commit=False)
    tracker.finish()
    assert fake_wandb.runs[0].logs == [({"loss": 0.25, "auth_token": "[REDACTED]"}, 3, {"commit": False})]
    assert fake_wandb.runs[0].finished is True
    serialized = json.dumps(call, sort_keys=True) + (tmp_path / "rdan-events.jsonl").read_text(encoding="utf-8")
    assert "metric-secret" not in serialized
    assert "test-key-present-only" not in serialized
    header = json.loads((tmp_path / "rdan-events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert header["identity"] == call["config"]["rdan_identity"]
    assert header["identity"]["resolved_config_sha256"] == canonical_config_sha256(config)


def test_changed_config_with_stale_digest_fails_before_any_write(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = {"learning_rate": 1e-6, "nested": {"dtype": "bf16"}}
    metadata = _metadata(config=sealed)
    monkeypatch.setattr(
        tracking,
        "_append_jsonl",
        lambda *args, **kwargs: pytest.fail("stale config must fail before local write"),
    )
    with pytest.raises(WandbTrackingError, match="resolved_config_sha256"):
        RdanWandbTracker({**sealed, "learning_rate": 2e-6}, **_kwargs(tmp_path, metadata))
    assert fake_wandb.calls == []
    assert not (tmp_path / "rdan-events.jsonl").exists()


@pytest.mark.parametrize(
    "config",
    [
        {"bad": float("nan")},
        {"bad": (1, 2)},
        {1: "non-string-key"},
        {"openrouter_api_key": "sk-or-v1-never-serialize"},
        {"source": "hf_" + "a" * 30},
    ],
)
def test_noncanonical_or_secret_config_fails_before_any_write(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[Any, Any],
) -> None:
    monkeypatch.setattr(
        tracking,
        "_append_jsonl",
        lambda *args, **kwargs: pytest.fail("invalid config must fail before local write"),
    )
    with pytest.raises(WandbTrackingError):
        RdanWandbTracker(config, **_kwargs(tmp_path))
    assert fake_wandb.calls == []


def test_canonical_config_accepts_nonsecret_hf_strategy_name() -> None:
    config = {"actor_infer": {"strategy_name": "hf_infer"}}
    assert (
        canonical_config_sha256(config) == hashlib.sha256(b'{"actor_infer":{"strategy_name":"hf_infer"}}').hexdigest()
    )


def test_canonical_config_accepts_an_unset_credential_field() -> None:
    config = {"rewards": {"llm_judge": {"judge_api_key": None, "judge_model_type": "api"}}}
    assert canonical_config_sha256(config)


def test_canonical_config_rejects_a_populated_credential_field() -> None:
    config = {"rewards": {"llm_judge": {"judge_api_key": "sk-or-v1-" + "a" * 32}}}
    with pytest.raises(WandbTrackingError, match="resolved credentials"):
        canonical_config_sha256(config)


def test_semantic_config_hash_excludes_only_tracker_kwargs() -> None:
    config = {
        "track_with": "rdan_wandb",
        "max_steps": 20,
        "tracker_kwargs": {"metadata": {"resolved_config_sha256": "pending"}},
    }
    changed_tracker = {**config, "tracker_kwargs": {"metadata": {"resolved_config_sha256": "changed"}}}
    changed_steps = {**config, "max_steps": 21}
    assert canonical_config_sha256(config) == canonical_config_sha256(changed_tracker)
    assert canonical_config_sha256(config) != canonical_config_sha256(changed_steps)


def test_runtime_timestamps_do_not_change_resume_identity() -> None:
    first = {
        "profiler_output_dir": "/runs/profiler/qwen/20260813-010203",
        "length_profiler_dir": "/runs/length/qwen/20260813-010203",
        "checkpoint_config": {"output_dir": "/runs/checkpoints/qwen/20260813-010203"},
    }
    resumed = {
        "profiler_output_dir": "/runs/profiler/qwen/20260814-111213",
        "length_profiler_dir": "/runs/length/qwen/20260814-111213",
        "checkpoint_config": {"output_dir": "/runs/checkpoints/qwen/20260814-111213"},
    }
    changed = {**resumed, "profiler_output_dir": "/runs/profiler/other/20260814-111213"}

    assert canonical_config_sha256(first) == canonical_config_sha256(resumed)
    assert canonical_config_sha256(first) != canonical_config_sha256(changed)


@pytest.mark.parametrize(
    "method",
    ["rdan-scalar", "rl-csr", "rl-aon", "rl-mix", "rtt-aon", "rtt-csr", "rdan-full", "sft", "dpo"],
)
def test_training_method_names_are_complete_and_deterministic(method: str) -> None:
    metadata = {**_metadata(), "method": method}
    assert deterministic_run_id(metadata) == deterministic_run_id(dict(metadata))


def test_deterministic_id_binds_every_sealed_identity_field(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
) -> None:
    metadata = _metadata()
    run_id = deterministic_run_id(metadata)
    assert deterministic_run_id(dict(metadata)) == run_id
    for field in (
        "resolved_config_sha256",
        "model_revision",
        "data_sha256",
        "code_revision",
        "checkpoint_sha256",
    ):
        changed = {**metadata, field: ("f" if metadata[field][0] != "f" else "0") + metadata[field][1:]}
        assert deterministic_run_id(changed) != run_id
        message = "resolved_config_sha256" if field == "resolved_config_sha256" else "deterministic safe run id"
        with pytest.raises(WandbTrackingError, match=message):
            RdanWandbTracker({}, **_kwargs(tmp_path, changed, id=run_id, resume="must"))
    assert fake_wandb.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resolved_config_sha256", "a" * 63),
        ("model_revision", "B" * 40),
        ("data_sha256", "c" * 63),
        ("code_revision", 1),
        ("checkpoint_sha256", "not-a-hash"),
    ],
)
def test_malformed_sealed_identity_fails_before_wandb(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
    field: str,
    value: Any,
) -> None:
    metadata = {**_metadata(), field: value}
    with pytest.raises(WandbTrackingError, match="full lowercase hashes"):
        RdanWandbTracker({}, **_kwargs(tmp_path, metadata, id="rdan-invalid"))
    assert fake_wandb.calls == []


def test_missing_or_extra_sealed_identity_field_is_rejected() -> None:
    missing = _metadata()
    missing.pop("checkpoint_sha256")
    with pytest.raises(WandbTrackingError, match="fields do not match"):
        deterministic_run_id(missing)
    with pytest.raises(WandbTrackingError, match="fields do not match"):
        deterministic_run_id({**_metadata(), "extra_revision": "f" * 40})


def test_evaluation_identity_uses_explicit_metadata(tmp_path: Path, fake_wandb: types.ModuleType) -> None:
    metadata = _metadata("eval")
    values = _kwargs(tmp_path, metadata)
    del values["entity"]
    del values["project"]
    RdanWandbTracker({}, **values)
    call = fake_wandb.calls[0]
    assert call["entity"] == "RDAN-GRPO"
    assert call["project"] == "rdan-grpo-qwen3-4b"
    assert call["group"] == "qwen-base-eval"
    assert call["name"] == "qwen-base-eval-ifeval-s42"
    assert call["job_type"] == "eval"


def test_bad_name_unknown_metadata_and_nondeterministic_id_fail_closed(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
) -> None:
    with pytest.raises(WandbTrackingError, match="group or name"):
        RdanWandbTracker({}, **_kwargs(tmp_path, name="qwen-rdan-scalar-other-s240520"))
    with pytest.raises(WandbTrackingError, match="unknown training stage"):
        RdanWandbTracker({}, **_kwargs(tmp_path, {**_metadata(), "stage": "other"}))
    with pytest.raises(WandbTrackingError, match="deterministic safe run id"):
        RdanWandbTracker({}, **_kwargs(tmp_path, id="user-selected-id"))
    assert fake_wandb.calls == []


def test_missing_environment_key_and_credential_kwarg_fail_before_wandb(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY")
    with pytest.raises(WandbTrackingError, match="WANDB_API_KEY"):
        RdanWandbTracker({}, **_kwargs(tmp_path))
    monkeypatch.setenv("WANDB_API_KEY", "present")
    with pytest.raises(WandbTrackingError, match="credential options"):
        RdanWandbTracker({}, **_kwargs(tmp_path), api_key="wandb_v1_forbidden")
    assert fake_wandb.calls == []


def test_artifact_logging_is_run_scoped_and_rejects_unsafe_paths(
    tmp_path: Path,
    fake_wandb: types.ModuleType,
) -> None:
    tracker = RdanWandbTracker({}, **_kwargs(tmp_path))
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text('{"score": 1}\n', encoding="utf-8")
    tracker.log_artifact(
        artifact_path,
        name="qwen-rdan-scalar-metrics",
        artifact_type="evaluation",
        aliases=("latest",),
        metadata={"provider_token": "sk-or-v1-artifact-secret"},
    )
    artifact, aliases = fake_wandb.runs[0].artifacts[0]
    assert artifact.files == [str(artifact_path)]
    assert artifact.metadata == {"provider_token": "[REDACTED]"}
    assert artifact.waited is True
    assert aliases == ["latest"]

    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(WandbTrackingError, match="contained"):
        tracker.log_artifact(outside, name="outside-artifact", artifact_type="evaluation")
    link = tmp_path / "linked.json"
    link.symlink_to(artifact_path)
    with pytest.raises(WandbTrackingError, match="canonical"):
        tracker.log_artifact(link, name="linked-artifact", artifact_type="evaluation")


@pytest.mark.skipif(not RTT_ROOT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_exact_pinned_rtt_tracker_registers_without_modifying_rtt() -> None:
    registry: dict[str, Any] = {}
    assert verify_rtt_tracking(RTT_ROOT).read_bytes() == (RTT_ROOT / tracking.RTT_TRACKING_PATH).read_bytes()
    assert register_wandb_tracker(RTT_ROOT, registry) is RdanWandbTracker
    assert registry == {"rdan_wandb": RdanWandbTracker}
    with pytest.raises(WandbTrackingError, match="already contains"):
        register_wandb_tracker(RTT_ROOT, registry)


def test_wrong_revision_dirty_checkout_and_bad_digest_fail_before_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "rtt"
    source = checkout / tracking.RTT_TRACKING_PATH
    source.parent.mkdir(parents=True)
    source.write_text("tracker_registry = {}\n", encoding="utf-8")
    _git(checkout, "init")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")
    revision = _git(checkout, "rev-parse", "HEAD")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(WandbTrackingError, match="unexpected RTT revision"):
        register_wandb_tracker(checkout, {})
    monkeypatch.setattr(tracking, "RTT_REVISION", revision)
    monkeypatch.setattr(tracking, "RTT_TRACKING_SHA256", digest)
    source.write_text("tracker_registry = {'changed': object()}\n", encoding="utf-8")
    with pytest.raises(WandbTrackingError, match="dirty"):
        register_wandb_tracker(checkout, {})
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "changed")
    monkeypatch.setattr(tracking, "RTT_REVISION", _git(checkout, "rev-parse", "HEAD"))
    with pytest.raises(WandbTrackingError, match="tracking digest"):
        register_wandb_tracker(checkout, {})


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()
