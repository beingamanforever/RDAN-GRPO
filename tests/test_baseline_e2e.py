from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from rdan_grpo import baseline

ROOT = Path(__file__).resolve().parents[1]


class _ServerState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.block_prompt: str | None = None
        self.release = threading.Event()


@contextmanager
def _fake_server(state: _ServerState) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._send({"data": [{"id": "qwen3-4b-instruct-2507", "root": str(state.root)}]})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            prompt = request["messages"][-1]["content"]
            state.calls[prompt] += 1
            if prompt == state.block_prompt:
                state.release.wait(10)
            if state.errors[prompt]:
                state.errors[prompt] -= 1
                self.send_response(500)
                self.end_headers()
                return
            self._send({"choices": [{"message": {"content": f"response-{prompt}"}}]})

        def _send(self, value: dict[str, Any]) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        state.release.set()
        server.shutdown()
        thread.join()


def _sha_entry(snapshot: Path, name: str, roles: list[str]) -> dict[str, Any]:
    size, digest = baseline._sha256(snapshot / name)
    return {"path": name, "roles": roles, "bytes": size, "sha256": digest}


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    rtt = tmp_path / "rtt"
    input_path = rtt / "Benchmark/instruction_following_eval/data/input_data.jsonl"
    input_path.parent.mkdir(parents=True)
    records = [
        {"key": "zero", "prompt": "prompt-zero", "instruction_id_list": ["test"]},
        {"key": "one", "prompt": "prompt-one", "instruction_id_list": ["test"]},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    _, input_hash = baseline._sha256(input_path)

    config = json.loads(baseline.CONFIG.read_text(encoding="utf-8"))
    config["benchmarks"]["ifeval"].update({"records": 2, "sha256": input_hash})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    snapshot = tmp_path / "snapshot" / baseline.PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"model")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": "2026-08-13T00:00:00+00:00",
        "model": {
            "name": baseline.PINNED_MODEL,
            "revision": baseline.PINNED_MODEL_REVISION,
            "snapshot_commit": baseline.PINNED_MODEL_REVISION,
            "snapshot_path": str(snapshot),
        },
        "files": [
            _sha_entry(snapshot, "model.safetensors", ["model"]),
            _sha_entry(snapshot, "tokenizer.json", ["tokenizer"]),
            _sha_entry(snapshot, "tokenizer_config.json", ["tokenizer", "chat_template"]),
        ],
        "server": {
            "argv": [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(snapshot),
                "--served-model-name",
                "qwen3-4b-instruct-2507",
            ],
            "packages": {"vllm": "0.10.2", "transformers": "4.55.2", "torch": "2.7.1+cu128"},
        },
    }
    manifest_path = tmp_path / "server.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        baseline,
        "_verify_rtt",
        lambda _root, revision: {"revision": revision, "tree": "0" * 40},
    )

    def scorer(
        _benchmark: str, rtt_root: Path, generation_path: Path, output: Path, _timeout: int
    ) -> tuple[list[str], Path]:
        generated = [json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()]
        native = output / "native"
        native.mkdir()
        rows = [
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "instruction_id_list": ["test"],
                "follow_instruction_list": [True],
                "follow_all_instructions": True,
            }
            for row in generated
        ]
        baseline._write_jsonl(native / "eval_results_strict.jsonl", rows)
        baseline._write_jsonl(native / "eval_results_loose.jsonl", rows)
        (output / "scorer_stdout.txt").write_text("", encoding="utf-8")
        (output / "scorer_stderr.txt").write_text("", encoding="utf-8")
        baseline._write_json(output / "scorer.json", {"command": ["fake-scorer"], "exit_code": 0})
        return ["fake-scorer"], rtt_root

    monkeypatch.setattr(baseline, "_run_scorer", scorer)
    return {"rtt": rtt, "config": config_path, "snapshot": snapshot, "manifest": manifest_path}


def _run(paths: dict[str, Path], api_base: str, output: Path) -> Path:
    return baseline.run_evaluation(
        "ifeval",
        paths["rtt"],
        api_base,
        paths["manifest"],
        output,
        concurrency=1,
        config_path=paths["config"],
        argv=["run_base_eval.py", "--api-base", api_base],
    )


def test_rejects_correct_alias_for_wrong_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    wrong_snapshot = tmp_path / "wrong-snapshot"
    wrong_snapshot.mkdir()
    state = _ServerState(wrong_snapshot)
    with _fake_server(state) as api_base:
        with pytest.raises(baseline.EvaluationError, match="checkpoint does not match"):
            _run(paths, api_base, tmp_path / "output")
    assert not state.calls


def test_killed_run_reuses_valid_partial_records_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "output"
    state = _ServerState(paths["snapshot"])
    state.block_prompt = "prompt-one"
    with _fake_server(state) as api_base:
        child = """
import sys
from pathlib import Path
from rdan_grpo import baseline
baseline._verify_rtt = lambda _root, revision: {"revision": revision, "tree": "0" * 40}
baseline.run_evaluation(
    "ifeval", Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]),
    concurrency=1, config_path=Path(sys.argv[5]), argv=["run_base_eval.py", "--api-base", sys.argv[2]],
)
"""
        environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                child,
                str(paths["rtt"]),
                api_base,
                str(paths["manifest"]),
                str(output),
                str(paths["config"]),
            ],
            cwd=ROOT,
            env=environment,
        )
        partial_generation = tmp_path / ".output.partial/partial_generation.jsonl"
        deadline = time.monotonic() + 10
        while (not partial_generation.exists() or state.calls["prompt-one"] != 1) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert partial_generation.exists()
        assert len(partial_generation.read_text(encoding="utf-8").splitlines()) == 1
        process.kill()
        process.wait(5)
        state.block_prompt = None
        state.release.set()
        _run(paths, api_base, output)

    assert state.calls == Counter({"prompt-one": 2, "prompt-zero": 1})
    assert len((output / "generation.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    artifact_text = "\n".join(path.read_text(errors="ignore") for path in output.rglob("*") if path.is_file())
    assert "127.0.0.1" not in artifact_text


def test_reclaims_stale_dead_owner_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "output"
    lock = tmp_path / ".output.lock"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": 999_999_999,
                "process_start": "stale",
                "created_at": "2026-08-13T00:00:00+00:00",
                "owner_token": "stale",
            }
        ),
        encoding="utf-8",
    )
    state = _ServerState(paths["snapshot"])
    with _fake_server(state) as api_base:
        _run(paths, api_base, output)
    assert output.is_dir()
    assert not lock.exists()


def test_live_owner_lock_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "output"
    lock = tmp_path / ".output.lock"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": baseline.os.getpid(),
                "process_start": baseline._process_start(baseline.os.getpid()),
                "created_at": "2026-08-13T00:00:00+00:00",
                "owner_token": "live",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(baseline.EvaluationError, match="live process owns"):
        _run(paths, "http://127.0.0.1:1/v1", output)


def test_rejects_partial_from_different_server_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "output"
    state = _ServerState(paths["snapshot"])
    state.errors["prompt-one"] = 1
    with _fake_server(state) as api_base:
        with pytest.raises(baseline.EvaluationError, match="generation failed"):
            _run(paths, api_base, output)
        calls_before = copy.copy(state.calls)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["created_at"] = "2026-08-13T00:00:01+00:00"
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(baseline.EvaluationError, match="partial run identity does not match"):
            _run(paths, api_base, output)
    assert state.calls == calls_before
    partial = tmp_path / ".output.partial"
    assert len((partial / "partial_generation.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len(list((partial / "failures").glob("*.json"))) == 2
