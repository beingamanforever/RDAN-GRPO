from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rdan_grpo import scalar_data
from rdan_grpo.program import ProgramContractError, check_program

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _render_tmp_data_paths_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    original = scalar_data._display_path

    def display(path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT):
            parts = resolved.parts
            if "data" in parts:
                return Path(*parts[parts.index("data") :]).as_posix()
        return original(path)

    monkeypatch.setattr(scalar_data, "_display_path", display)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _freezer():
    return _load("freeze_response_lifecycle", ROOT / "scripts/freeze_response_lifecycle.py")


def _transitions():
    return _load("freeze_response_transition_fixtures", ROOT / "tests/test_program_transitions.py")


def _write_lifecycle_inputs(tmp_path: Path) -> tuple[object, Path]:
    fixtures = _transitions()
    program_path = fixtures._contract_copy(tmp_path)
    config_root = program_path.parent.parent
    judge = fixtures._calibration_artifact(config_root)
    fixtures._write(config_root / "artifacts/qwen_judge_calibration.json", judge)
    program = fixtures._read(program_path)
    parity = fixtures._parity_artifact(tmp_path, program["same_backend_configs"]["production"]["sha256"])
    fixtures._write(config_root / "artifacts/qwen_runtime_parity.json", parity)
    vllm = fixtures._vllm_parity_artifact(tmp_path, program["same_backend_configs"]["production"]["sha256"])
    fixtures._write(config_root / "artifacts/qwen_vllm_runtime_parity.json", vllm)
    return fixtures, program_path


@pytest.mark.parametrize(
    "order",
    [
        ("judge", "parity", "vllm-parity"),
        ("vllm-parity", "parity", "judge"),
    ],
)
def test_freezes_launch_parity_gates_in_any_order_with_exact_pins(tmp_path: Path, order: tuple[str, ...]) -> None:
    _, program_path = _write_lifecycle_inputs(tmp_path)
    freezer = _freezer()

    for stage in order:
        reference = freezer.freeze_stage(program_path, stage)
        artifact = program_path.parent.parent / Path(reference["path"]).relative_to("configs")
        key = {
            "judge": "judge_calibration",
            "parity": "runtime_parity",
            "vllm-parity": "vllm_runtime_parity",
        }[stage]
        body = json.loads(artifact.read_text(encoding="utf-8"))
        program = json.loads(program_path.read_text(encoding="utf-8"))
        assert reference == program["lifecycle_artifacts"][key]
        assert reference["artifact_id"] == body["id"]
        assert reference["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert program["readiness"]["launch"] == "blocked_until_all_launch_artifacts_frozen"

    bundle = check_program(program_path)
    assert bundle.program["readiness"]["judge"] == "ready"
    assert set(bundle.lifecycle_artifacts) >= {"judge_calibration", "runtime_parity", "vllm_runtime_parity"}


def test_no_update_requires_all_launch_gates_and_alone_unlocks_launch(tmp_path: Path) -> None:
    fixtures, program_path = _write_lifecycle_inputs(tmp_path)
    freezer = _freezer()
    original = program_path.read_bytes()
    no_update_path = program_path.parent.parent / "artifacts/qwen_no_update_certificate.json"
    fixtures._write(no_update_path, {"certificate_id": "a" * 64, "ready": True})

    with pytest.raises(freezer.LifecycleFreezeError, match="requires frozen lifecycle evidence"):
        freezer.freeze_stage(program_path, "no-update")
    assert program_path.read_bytes() == original

    freezer.freeze_stage(program_path, "parity")
    freezer.freeze_stage(program_path, "judge")
    with pytest.raises(freezer.LifecycleFreezeError, match="vllm_runtime_parity"):
        freezer.freeze_stage(program_path, "no-update")
    freezer.freeze_stage(program_path, "vllm-parity")
    artifact, _ = fixtures._no_update_artifact(check_program(program_path))
    fixtures._write(no_update_path, artifact)

    reference = freezer.freeze_stage(program_path, "no-update")
    bundle = check_program(program_path)
    assert reference["artifact_id"] == artifact["certificate_id"]
    assert reference["sha256"] == hashlib.sha256(no_update_path.read_bytes()).hexdigest()
    assert bundle.program["readiness"]["launch"] == "ready"


def test_deep_validation_failure_preserves_program_atomically(tmp_path: Path) -> None:
    fixtures, program_path = _write_lifecycle_inputs(tmp_path)
    freezer = _freezer()
    parity_path = program_path.parent.parent / "artifacts/qwen_runtime_parity.json"
    parity = fixtures._read(parity_path)
    parity["status"] = "pending"
    fixtures._write(parity_path, parity)
    original = program_path.read_bytes()

    with pytest.raises(ProgramContractError, match="parity artifact is invalid"):
        freezer.freeze_stage(program_path, "parity")

    assert program_path.read_bytes() == original
    assert not list(program_path.parent.glob(".qwen_first.*.json"))


def test_rejects_artifact_that_is_not_at_the_exact_configured_path(tmp_path: Path) -> None:
    fixtures = _transitions()
    program_path = fixtures._contract_copy(tmp_path / "repo")
    config_root = program_path.parent.parent
    outside = tmp_path / "judge.json"
    fixtures._write(outside, fixtures._calibration_artifact(config_root))
    artifact_path = config_root / "artifacts/qwen_judge_calibration.json"
    artifact_path.symlink_to(outside)
    original = program_path.read_bytes()
    freezer = _freezer()

    with pytest.raises(freezer.LifecycleFreezeError, match="resolve exactly"):
        freezer.freeze_stage(program_path, "judge")

    assert program_path.read_bytes() == original
