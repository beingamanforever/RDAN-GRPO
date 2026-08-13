from __future__ import annotations

import json
import stat
import sys
from datetime import datetime
from pathlib import Path

import pytest

from rdan_grpo import baseline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import capture_server_identity as capture  # noqa: E402


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshots" / baseline.PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    files = {
        "config.json": "{}",
        "generation_config.json": "{}",
        "model-00001-of-00002.safetensors": "shard-one",
        "model-00002-of-00002.safetensors": "shard-two",
        "model.safetensors.index.json": "{}",
        "tokenizer.json": "{}",
        "tokenizer_config.json": '{"chat_template":"{{ messages }}"}',
        "README.md": "not an identity file",
        "weights.gguf": "unknown identity type",
    }
    for name, content in files.items():
        (snapshot / name).write_text(content, encoding="utf-8")
    return snapshot


def _proc(proc_root: Path, pid: int, snapshot: Path, extra: list[str] | None = None) -> list[str]:
    argv = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(snapshot),
        "--served-model-name",
        "qwen3-4b-instruct-2507",
        *(extra or []),
    ]
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
    return argv


@pytest.fixture
def versions(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    value = {"vllm": "0.10.2", "transformers": "4.55.2", "torch": "2.7.1+cu128"}
    monkeypatch.setattr(capture, "_package_versions", lambda: value)
    return value


def test_captures_all_shards_and_embedded_template(tmp_path: Path, versions: dict[str, str]) -> None:
    snapshot = _snapshot(tmp_path)
    proc_root = tmp_path / "proc"
    argv = _proc(proc_root, 123, snapshot)
    output = tmp_path / "artifacts/server.json"

    assert capture.capture_server_identity(123, snapshot, output, proc_root) == output.absolute()

    manifest = json.loads(output.read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert set(entries) == {
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert "weights.gguf" not in entries
    assert entries["tokenizer_config.json"]["roles"] == ["chat_template", "tokenizer"]
    assert manifest["server"] == {"argv": argv, "packages": versions}
    assert datetime.fromisoformat(manifest["created_at"]).utcoffset().total_seconds() == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))
    baseline._load_server_manifest(
        output,
        {
            "name": baseline.PINNED_MODEL,
            "revision": baseline.PINNED_MODEL_REVISION,
            "served_name": "qwen3-4b-instruct-2507",
        },
    )


def test_rejects_omitted_recognized_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, versions: dict[str, str]
) -> None:
    snapshot = _snapshot(tmp_path)
    proc_root = tmp_path / "proc"
    _proc(proc_root, 123, snapshot)
    complete = capture._snapshot_files(snapshot)
    monkeypatch.setattr(
        capture,
        "_snapshot_files",
        lambda _snapshot: [entry for entry in complete if entry["path"] != "model-00002-of-00002.safetensors"],
    )
    output = tmp_path / "server.json"

    with pytest.raises(baseline.EvaluationError, match="omits required snapshot file"):
        capture.capture_server_identity(123, snapshot, output, proc_root)
    assert not output.exists()


@pytest.mark.parametrize("extra", [["--api-key", "secret-value"], ["--host", "0.0.0.0"], ["--foo", "10.0.0.7"]])
def test_rejects_sensitive_or_host_ip_argv(tmp_path: Path, versions: dict[str, str], extra: list[str]) -> None:
    snapshot = _snapshot(tmp_path)
    proc_root = tmp_path / "proc"
    _proc(proc_root, 123, snapshot, extra)
    with pytest.raises(baseline.EvaluationError, match="credentials|server host|host IP"):
        capture.capture_server_identity(123, snapshot, tmp_path / "server.json", proc_root)


def test_accepts_localhost_bind_without_host_identity(tmp_path: Path, versions: dict[str, str]) -> None:
    snapshot = _snapshot(tmp_path)
    proc_root = tmp_path / "proc"
    _proc(proc_root, 123, snapshot, ["--host", "localhost"])
    output = tmp_path / "server.json"

    capture.capture_server_identity(123, snapshot, output, proc_root)

    assert json.loads(output.read_text(encoding="utf-8"))["server"]["argv"][-2:] == ["--host", "localhost"]


def test_rejects_wrong_pid_and_snapshot(tmp_path: Path, versions: dict[str, str]) -> None:
    snapshot = _snapshot(tmp_path)
    with pytest.raises(baseline.EvaluationError, match="cannot read command line"):
        capture.capture_server_identity(999, snapshot, tmp_path / "server.json", tmp_path / "proc")

    wrong_snapshot = tmp_path / "snapshots/wrong-revision"
    wrong_snapshot.mkdir()
    with pytest.raises(baseline.EvaluationError, match="pinned model revision"):
        capture.capture_server_identity(123, wrong_snapshot, tmp_path / "server.json", tmp_path / "proc")


def test_rejects_escaping_snapshot_symlink(tmp_path: Path, versions: dict[str, str]) -> None:
    snapshot = _snapshot(tmp_path)
    proc_root = tmp_path / "proc"
    _proc(proc_root, 123, snapshot)
    outside = tmp_path / "outside.safetensors"
    outside.write_text("outside", encoding="utf-8")
    (snapshot / "escaped.safetensors").symlink_to(outside)
    with pytest.raises(baseline.EvaluationError, match="symlink escapes"):
        capture.capture_server_identity(123, snapshot, tmp_path / "server.json", proc_root)


def test_existing_output_is_not_overwritten(tmp_path: Path, versions: dict[str, str]) -> None:
    snapshot = _snapshot(tmp_path)
    output = tmp_path / "server.json"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(baseline.EvaluationError, match="output already exists"):
        capture.capture_server_identity(123, snapshot, output, tmp_path / "missing-proc")
    assert output.read_text(encoding="utf-8") == "preserve"
