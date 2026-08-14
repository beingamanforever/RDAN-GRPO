from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from rdan_grpo import response_readiness as readiness_module
from rdan_grpo.program import MODEL_NAME, MODEL_REVISION, RTT_REVISION, ProgramContractError
from rdan_grpo.response_readiness import (
    EVIDENCE_ORDER,
    ResponseReadinessError,
    issue_response_readiness,
    validate_response_readiness,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
RUNTIME = readiness_module._load_runtime_expectation()


@pytest.fixture(scope="module")
def ready_inputs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("response-readiness")
    program_path = root / "configs/program/qwen_first.json"
    artifact_root = root / "configs/artifacts"
    program_path.parent.mkdir(parents=True)
    artifact_root.mkdir()
    judge_path = artifact_root / "qwen_judge_calibration.json"
    parity_path = artifact_root / "qwen_runtime_parity.json"
    no_update_path = artifact_root / "qwen_no_update_certificate.json"
    artifacts = {
        "judge_calibration": {"id": "judge-v1", "status": "calibrated"},
        "runtime_parity": {"id": "parity-v1", "status": "parity_passed"},
    }
    for path, name in ((judge_path, "judge_calibration"), (parity_path, "runtime_parity")):
        path.write_text(json.dumps(artifacts[name], sort_keys=True) + "\n", encoding="utf-8")
    refs = {
        name: {
            "status": "frozen",
            "path": f"configs/artifacts/{path.name}",
            "artifact_id": artifacts[name]["id"],
            "sha256": _file_sha256(path),
        }
        for name, path in (("judge_calibration", judge_path), ("runtime_parity", parity_path))
    }
    artifacts["no_update"] = {
        "certificate_id": "no-update-v1",
        "ready": True,
        "source_sha256": {
            "judge_calibration": refs["judge_calibration"]["sha256"],
            "runtime_parity": refs["runtime_parity"]["sha256"],
        },
    }
    no_update_path.write_text(json.dumps(artifacts["no_update"], sort_keys=True) + "\n", encoding="utf-8")
    refs["no_update"] = {
        "status": "frozen",
        "path": "configs/artifacts/qwen_no_update_certificate.json",
        "artifact_id": "no-update-v1",
        "sha256": _file_sha256(no_update_path),
    }
    program = {
        "id": "qwen_first_v1",
        "readiness": {"launch": "ready", "scalar_training": "ready"},
        "lifecycle_artifacts": refs,
    }
    program_path.write_text(json.dumps(program, sort_keys=True) + "\n", encoding="utf-8")
    bootstrap_path = root / "run/a100-response-bootstrap.json"
    bootstrap_path.parent.mkdir()
    bootstrap_path.write_text(json.dumps(_bootstrap_report(), sort_keys=True, separators=(",", ":")) + "\n")
    original_check = readiness_module.check_program

    def check_program(path: Path) -> Any:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded_artifacts = {}
        for name, reference in loaded["lifecycle_artifacts"].items():
            artifact_path = root / reference["path"]
            if _file_sha256(artifact_path) != reference["sha256"]:
                raise ProgramContractError(f"lifecycle {name} hash mismatch")
            loaded_artifacts[name] = json.loads(artifact_path.read_text(encoding="utf-8"))
        return SimpleNamespace(
            program=loaded,
            lifecycle_artifacts=loaded_artifacts,
            repo_root=root,
        )

    readiness_module.check_program = check_program
    try:
        yield {
            "program": program_path,
            "bootstrap": bootstrap_path,
            "evidence": (judge_path, parity_path, no_update_path),
        }
    finally:
        readiness_module.check_program = original_check


def test_cli_issues_canonical_receipt_and_check_is_nonmutating(
    ready_inputs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load("run_response_readiness_test", ROOT / "scripts/run_response_readiness.py")
    monkeypatch.setattr(script, "clean_repository_revision", lambda path, expected=None: expected or REVISION)
    output = tmp_path / "response-readiness.json"
    arguments = _arguments(ready_inputs, output)

    assert script.main(arguments) == 0
    before = output.read_bytes()
    before_stat = output.stat()
    receipt = json.loads(before)
    assert receipt["status"] == "ready"
    assert receipt["schema_version"] == 2
    assert receipt["compute_contract_sha256"] == RUNTIME.compute_sha256
    assert receipt["evidence_order"] == list(EVIDENCE_ORDER)
    assert set(receipt["evidence"]) == set(EVIDENCE_ORDER)
    assert "secret" not in before.decode("utf-8").lower()

    assert script.main([*arguments, "--check"]) == 0
    assert output.read_bytes() == before
    assert output.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_missing_evidence_fails_closed(ready_inputs: dict[str, Any], tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    evidence = (*ready_inputs["evidence"][:2], missing)
    with pytest.raises(ResponseReadinessError, match="regular non-symlink"):
        issue_response_readiness(
            ready_inputs["program"],
            ready_inputs["bootstrap"],
            evidence,
            tmp_path / "receipt.json",
            rdan_revision=REVISION,
        )


def test_minimal_or_mismatched_bootstrap_fails_closed(ready_inputs: dict[str, Any], tmp_path: Path) -> None:
    minimal = tmp_path / "minimal.json"
    minimal.write_text('{"schema_version":1,"status":"passed"}\n', encoding="utf-8")
    with pytest.raises(ResponseReadinessError, match="minimal or incomplete"):
        issue_response_readiness(
            ready_inputs["program"],
            minimal,
            ready_inputs["evidence"],
            tmp_path / "minimal-receipt.json",
            rdan_revision=REVISION,
        )

    mismatched = tmp_path / "mismatched.json"
    report = _bootstrap_report()
    report["model"]["revision"] = "b" * 40
    mismatched.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ResponseReadinessError, match="model identity"):
        issue_response_readiness(
            ready_inputs["program"],
            mismatched,
            ready_inputs["evidence"],
            tmp_path / "mismatched-receipt.json",
            rdan_revision=REVISION,
        )


def test_stale_bootstrap_revision_fails_closed(ready_inputs: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ResponseReadinessError, match="stale or mismatched"):
        issue_response_readiness(
            ready_inputs["program"],
            ready_inputs["bootstrap"],
            ready_inputs["evidence"],
            tmp_path / "receipt.json",
            rdan_revision="b" * 40,
        )


def test_compute_contract_drift_invalidates_bootstrap_evidence(
    ready_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = json.loads(readiness_module.COMPUTE_CONTRACT.read_text(encoding="utf-8"))
    runtime = payload["response_runtime"]
    runtime["host"]["minimum_ram_gib"] += 1
    runtime["platform"]["container_contract"] = str(ROOT / "requirements/a100-response-container.json")
    runtime["packages"]["contracts"] = [
        str(ROOT / "requirements/a100-response-linux-py312.lock"),
        str(ROOT / "requirements/a100-response-flash.txt"),
    ]
    drifted = tmp_path / "compute.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(readiness_module, "COMPUTE_CONTRACT", drifted)

    with pytest.raises(ResponseReadinessError, match="did not pass"):
        issue_response_readiness(
            ready_inputs["program"],
            ready_inputs["bootstrap"],
            ready_inputs["evidence"],
            tmp_path / "receipt.json",
            rdan_revision=REVISION,
        )


def test_reordered_evidence_fails_closed(ready_inputs: dict[str, Any], tmp_path: Path) -> None:
    judge, parity, no_update = ready_inputs["evidence"]
    with pytest.raises(ResponseReadinessError, match="judge calibration, runtime parity, no-update order"):
        issue_response_readiness(
            ready_inputs["program"],
            ready_inputs["bootstrap"],
            (parity, judge, no_update),
            tmp_path / "receipt.json",
            rdan_revision=REVISION,
        )


def test_duplicate_publication_never_replaces_receipt(ready_inputs: dict[str, Any], tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    issue_response_readiness(
        ready_inputs["program"],
        ready_inputs["bootstrap"],
        ready_inputs["evidence"],
        output,
        rdan_revision=REVISION,
    )
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        issue_response_readiness(
            ready_inputs["program"],
            ready_inputs["bootstrap"],
            ready_inputs["evidence"],
            output,
            rdan_revision=REVISION,
        )
    assert output.read_bytes() == original


def test_tampered_lifecycle_evidence_and_receipt_fail_closed(ready_inputs: dict[str, Any], tmp_path: Path) -> None:
    parity = ready_inputs["evidence"][1]
    original = parity.read_bytes()
    try:
        payload = json.loads(original)
        payload["status"] = "tampered"
        parity.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ProgramContractError, match="hash mismatch"):
            issue_response_readiness(
                ready_inputs["program"],
                ready_inputs["bootstrap"],
                ready_inputs["evidence"],
                tmp_path / "tampered-evidence.json",
                rdan_revision=REVISION,
            )
    finally:
        parity.write_bytes(original)

    output = tmp_path / "receipt.json"
    issue_response_readiness(
        ready_inputs["program"],
        ready_inputs["bootstrap"],
        ready_inputs["evidence"],
        output,
        rdan_revision=REVISION,
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    receipt["status"] = "tampered"
    output.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ResponseReadinessError, match="differs"):
        validate_response_readiness(
            output,
            ready_inputs["program"],
            ready_inputs["bootstrap"],
            ready_inputs["evidence"],
            rdan_revision=REVISION,
        )


def test_secret_material_in_bootstrap_is_rejected(ready_inputs: dict[str, Any], tmp_path: Path) -> None:
    bootstrap = tmp_path / "secret-bootstrap.json"
    report = _bootstrap_report()
    report["runtime_imports"]["api_key"] = "sk-or-v1-abcdefghijklmnopqrstuvwxyz"
    bootstrap.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ResponseReadinessError, match="credential value"):
        issue_response_readiness(
            ready_inputs["program"],
            bootstrap,
            ready_inputs["evidence"],
            tmp_path / "receipt.json",
            rdan_revision=REVISION,
        )


def _arguments(inputs: dict[str, Any], output: Path) -> list[str]:
    args = [
        "--program",
        str(inputs["program"]),
        "--bootstrap",
        str(inputs["bootstrap"]),
    ]
    for path in inputs["evidence"]:
        args.extend(("--evidence", str(path)))
    args.extend(("--output", str(output), "--rdan-revision", REVISION))
    return args


def _bootstrap_report() -> dict[str, Any]:
    sha = "f" * 64
    packages = dict(RUNTIME.packages)
    packages["torch"] = f"{packages['torch']}+{RUNTIME.allowed_local_suffixes['torch']}"
    container = RUNTIME.container
    return {
        "schema_version": 2,
        "status": "passed",
        "profile": RUNTIME.profile,
        "python": RUNTIME.python_receipt,
        "platform": RUNTIME.platform_receipt,
        "compute_contract_sha256": RUNTIME.compute_sha256,
        "container": {
            "image": container["image"],
            "index_digest": container["manifest_digest"],
            "amd64_digest": container["linux_amd64_digest"],
            "cuda": container["cuda"],
            "release": container["nvidia_pytorch_release"],
            "identity_source": "docker-image-inspect",
            "image_id": "sha256:" + "e" * 64,
        },
        "repositories": {"rdan": REVISION, "rtt": RTT_REVISION},
        "model": {
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "file_sha256": {"config.json": sha},
            "tokenizer_sha256": {"tokenizer.json": sha},
        },
        "storage": {
            "cache": {"path": "/cache", "free_bytes": 1},
            "run": {"path": "/run", "free_bytes": 1},
        },
        "gpu": {
            "count": RUNTIME.gpu_count,
            "cuda": RUNTIME.cuda_runtime,
            "driver": "575.57.08",
            "nccl": RUNTIME.packages[RUNTIME.nccl_package],
            "topology": {"gpu0_to_gpu1": "NV12", "gpu1_to_gpu0": "NV12"},
            "devices": [
                {
                    "index": 0,
                    "uuid": "GPU-0000",
                    "name": "NVIDIA A100-SXM4-80GB",
                    "memory_mib": 81_920,
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                },
                {
                    "index": 1,
                    "uuid": "GPU-1111",
                    "name": "NVIDIA A100-SXM4-80GB",
                    "memory_mib": 81_920,
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                },
            ],
        },
        "requirements": dict(RUNTIME.artifact_hashes),
        "packages": packages,
        "runtime_imports": {
            "fsdp2": "FSDP2TrainStrategy",
            "hf": "HfInferStrategy",
            "infer_worker": "ResponseInferWorker",
            "pipeline": "ResponseTrainingPipeline",
            "reward_worker": "RTTCompatibleRubricRewardWorker",
            "train_worker": "ResponseActorWorker",
            "sdpa": True,
        },
        "data_preparation": {},
        "data_runtime": {},
        "venv": "/cache/a100-response-venv",
        "host_readiness": {
            "ram": {
                "total_bytes": RUNTIME.minimum_ram_bytes,
                "minimum_bytes": RUNTIME.minimum_ram_bytes,
            },
            "disk": {
                "cache": {
                    "available_bytes": RUNTIME.minimum_free_disk_bytes,
                    "minimum_bytes": RUNTIME.minimum_free_disk_bytes,
                },
                "run": {
                    "available_bytes": RUNTIME.minimum_free_disk_bytes,
                    "minimum_bytes": RUNTIME.minimum_free_disk_bytes,
                },
            },
            "gpu": {
                "count": 2,
                "driver": "575.57.08",
                "devices": [
                    {
                        "index": 0,
                        "uuid": "GPU-0000",
                        "name": "NVIDIA A100-SXM4-80GB",
                        "memory_mib": 81_920,
                        "memory_used_mib": 0,
                        "utilization_percent": 0,
                    },
                    {
                        "index": 1,
                        "uuid": "GPU-1111",
                        "name": "NVIDIA A100-SXM4-80GB",
                        "memory_mib": 81_920,
                        "memory_used_mib": 0,
                        "utilization_percent": 0,
                    },
                ],
                "compute_process_count": 0,
                "topology": {"gpu0_to_gpu1": "NV12", "gpu1_to_gpu0": "NV12"},
            },
        },
        "capabilities": {
            "judge_access": True,
            "tracking_access": True,
            "model_publish_access": True,
        },
    }


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
