#!/usr/bin/env python3
"""Capture a pinned local vLLM server identity manifest from Linux procfs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo import baseline  # noqa: E402

PROC_ROOT = Path("/proc")
PACKAGES = ("vllm", "transformers", "torch")


def _read_argv(pid: int, proc_root: Path) -> list[str]:
    if pid <= 0:
        raise baseline.EvaluationError("pid must be positive")
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
        argv = [value.decode() for value in raw.split(b"\0") if value]
    except (OSError, UnicodeDecodeError) as error:
        raise baseline.EvaluationError(f"cannot read command line for pid {pid}") from error
    if not argv:
        raise baseline.EvaluationError(f"process {pid} has an empty command line")
    baseline._reject_sensitive_argv(argv)
    return argv


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise baseline.EvaluationError(f"required server package is not installed: {name}") from error
    return versions


def _roles(path: Path) -> set[str]:
    roles = baseline._required_snapshot_roles(path)
    if path.name == "tokenizer_config.json":
        try:
            tokenizer_config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise baseline.EvaluationError("tokenizer_config.json is invalid") from error
        if isinstance(tokenizer_config, dict) and isinstance(tokenizer_config.get("chat_template"), str):
            roles.add("chat_template")
    return roles


def _snapshot_files(snapshot: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError) as error:
                raise baseline.EvaluationError(f"snapshot symlink is invalid: {path.relative_to(snapshot)}") from error
            if not resolved.is_relative_to(snapshot):
                raise baseline.EvaluationError(f"snapshot symlink escapes the snapshot: {path.relative_to(snapshot)}")
        if not path.is_file():
            continue
        roles = _roles(path)
        if not roles:
            continue
        try:
            size, digest = baseline._sha256(path)
        except OSError as error:
            raise baseline.EvaluationError(f"cannot hash snapshot identity: {path.relative_to(snapshot)}") from error
        files.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "roles": sorted(roles),
                "bytes": size,
                "sha256": digest,
            }
        )
    return files


def _write_atomic(output: Path, manifest: dict[str, Any], expected_model: dict[str, Any]) -> Path:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise baseline.EvaluationError(f"output already exists: {output}")
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as error:
        raise baseline.EvaluationError(f"cannot create manifest beside output: {output}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        baseline._load_server_manifest(temporary, expected_model)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise baseline.EvaluationError(f"output already exists: {output}") from error
        except OSError as error:
            raise baseline.EvaluationError(f"cannot publish server identity manifest: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return output


def capture_server_identity(pid: int, snapshot_path: Path, output: Path, proc_root: Path = PROC_ROOT) -> Path:
    snapshot = snapshot_path.absolute()
    if not snapshot.is_dir() or snapshot.resolve() != snapshot or snapshot.name != baseline.PINNED_MODEL_REVISION:
        raise baseline.EvaluationError("snapshot path must be the resolved pinned model revision directory")
    output_path = output.absolute()
    if output_path.exists() or output_path.is_symlink():
        raise baseline.EvaluationError(f"output already exists: {output_path}")
    argv = _read_argv(pid, proc_root)
    expected_model = baseline._load_config(baseline.CONFIG)["model"]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": expected_model["name"],
            "revision": expected_model["revision"],
            "snapshot_commit": expected_model["revision"],
            "snapshot_path": str(snapshot),
        },
        "files": _snapshot_files(snapshot),
        "server": {"argv": argv, "packages": _package_versions()},
    }
    return _write_atomic(output, manifest, expected_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = capture_server_identity(args.pid, args.snapshot_path, args.output)
    except baseline.EvaluationError as error:
        parser.exit(1, f"server identity capture failed: {error}\n")
    print(f"captured server identity: {output}")


if __name__ == "__main__":
    main()
