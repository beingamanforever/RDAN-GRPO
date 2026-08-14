#!/usr/bin/env python3
"""Atomically freeze validated response-training lifecycle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.program import ProgramContractError, check_program  # noqa: E402

STAGES = {
    "judge": ("judge_calibration", "configs/artifacts/qwen_judge_calibration.json", "id"),
    "parity": ("runtime_parity", "configs/artifacts/qwen_runtime_parity.json", "id"),
    "vllm-parity": ("vllm_runtime_parity", "configs/artifacts/qwen_vllm_runtime_parity.json", "id"),
    "no-update": ("no_update", "configs/artifacts/qwen_no_update_certificate.json", "certificate_id"),
}


class LifecycleFreezeError(ValueError):
    """Raised when lifecycle evidence cannot be frozen safely."""


def freeze_stage(program_path: Path, stage: str) -> dict[str, Any]:
    """Validate and atomically pin one immutable lifecycle artifact."""

    if stage not in STAGES:
        raise LifecycleFreezeError(f"unknown lifecycle stage: {stage}")
    program_path = program_path.absolute()
    _require_program_path(program_path)
    original = program_path.read_bytes()
    bundle = check_program(program_path)
    key, expected_path, id_key = STAGES[stage]
    refs = bundle.program["lifecycle_artifacts"]
    reference = refs[key]
    if reference != {
        "status": "pending",
        "path": expected_path,
        "artifact_id": "pending",
        "sha256": "pending",
    }:
        raise LifecycleFreezeError(f"{stage} lifecycle reference is not pending")
    if stage == "no-update":
        pending = [
            name
            for name in ("judge_calibration", "runtime_parity", "vllm_runtime_parity")
            if refs[name]["status"] != "frozen"
        ]
        if pending:
            raise LifecycleFreezeError(f"no-update requires frozen lifecycle evidence: {', '.join(pending)}")

    repo_root = program_path.parent.parent.parent
    artifact_path = repo_root / expected_path
    _require_artifact_path(artifact_path, repo_root, reference["path"], expected_path)
    artifact = _load_json(artifact_path)
    artifact_id = artifact.get(id_key)
    if not isinstance(artifact_id, str) or artifact_id in {"", "pending"}:
        raise LifecycleFreezeError(f"{stage} artifact has no immutable {id_key}")
    if stage == "no-update" and artifact.get("ready") is not True:
        raise LifecycleFreezeError("no-update artifact is not ready")
    artifact_sha256 = _sha256(artifact_path)

    program = deepcopy(bundle.program)
    program["lifecycle_artifacts"][key] = {
        "status": "frozen",
        "path": expected_path,
        "artifact_id": artifact_id,
        "sha256": artifact_sha256,
    }
    _set_readiness(program)
    candidate = _write_candidate(program_path, program)
    try:
        check_program(candidate)
        if program_path.read_bytes() != original:
            raise LifecycleFreezeError("program changed while lifecycle evidence was being validated")
        if _sha256(artifact_path) != artifact_sha256:
            raise LifecycleFreezeError("lifecycle artifact changed while it was being validated")
        os.replace(candidate, program_path)
        _fsync_dir(program_path.parent)
    finally:
        candidate.unlink(missing_ok=True)
    return program["lifecycle_artifacts"][key]


def _set_readiness(program: dict[str, Any]) -> None:
    refs = program["lifecycle_artifacts"]
    readiness = program["readiness"]
    readiness["judge"] = (
        "ready" if refs["judge_calibration"]["status"] == "frozen" else "blocked_until_frozen_calibration"
    )
    required = (
        "scalar_data",
        "response_data",
        "judge_calibration",
        "runtime_parity",
        "vllm_runtime_parity",
        "no_update",
    )
    launch_ready = readiness["scalar_training"] == "ready" and all(
        refs[name]["status"] == "frozen" for name in required
    )
    readiness["launch"] = "ready" if launch_ready else "blocked_until_all_launch_artifacts_frozen"


def _require_program_path(path: Path) -> None:
    if path.parent.name != "program" or path.parent.parent.name != "configs":
        raise LifecycleFreezeError("program must be inside configs/program")
    if not path.is_file() or path.is_symlink():
        raise LifecycleFreezeError("program must be a regular non-symlink file")


def _require_artifact_path(path: Path, root: Path, configured: Any, expected: str) -> None:
    if configured != expected:
        raise LifecycleFreezeError(f"lifecycle artifact path must be {expected}")
    expected_path = (root / expected).absolute()
    if path.absolute() != expected_path or path.resolve() != expected_path:
        raise LifecycleFreezeError(f"lifecycle artifact must resolve exactly to {expected}")
    if not path.is_file() or path.is_symlink():
        raise LifecycleFreezeError("lifecycle artifact must be a regular non-symlink file")


def _load_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise LifecycleFreezeError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise LifecycleFreezeError("lifecycle artifact must contain a JSON object")
    return value


def _write_candidate(program_path: Path, program: dict[str, Any]) -> Path:
    body = (json.dumps(program, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{program_path.stem}.", suffix=".json", dir=program_path.parent)
    candidate = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    """Freeze one response lifecycle stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--program", type=Path, default=ROOT / "configs/program/qwen_first.json")
    args = parser.parse_args()
    reference = freeze_stage(args.program, args.stage)
    print(json.dumps({"stage": args.stage, **reference}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (LifecycleFreezeError, ProgramContractError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"response lifecycle freeze blocked: {error}") from error
