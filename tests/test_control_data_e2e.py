from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import threading
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from rdan_grpo import baseline, control_data

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/HIR_trainv1.jsonl"


class _State:
    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self.teacher_calls: Counter[int] = Counter()
        self.candidate_calls: Counter[int] = Counter()
        self.teacher_failures = Counter({10613: 3})
        self.candidate_failures = Counter({10613: 1})
        self.teacher_requests: list[dict[str, Any]] = []
        self.candidate_requests: list[dict[str, Any]] = []
        self.authorization: list[str | None] = []
        self.candidate_finish_reason = "stop"
        self.catalog_slug = "openai/gpt-5.6-luna-20260709"


def _row_id(prompt: str) -> int:
    return 10611 if "Boundary Waters" in prompt else 10613


@contextmanager
def _fake_servers(state: _State) -> Iterator[tuple[str, str]]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/v1/models":
                self._send(
                    {
                        "data": [
                            {
                                "id": "openai/gpt-5.6-luna",
                                "canonical_slug": state.catalog_slug,
                                "supported_parameters": ["max_tokens", "reasoning", "seed"],
                            }
                        ]
                    }
                )
            else:
                self._send({"data": [{"id": "qwen3-4b-instruct-2507", "root": str(state.snapshot)}]})

        def do_POST(self) -> None:  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            row_id = _row_id(request["messages"][0]["content"])
            if self.path == "/api/v1/chat/completions":
                state.teacher_calls[row_id] += 1
                state.teacher_requests.append(request)
                state.authorization.append(self.headers.get("Authorization"))
                if state.teacher_failures[row_id]:
                    state.teacher_failures[row_id] -= 1
                    self.send_response(500)
                    self.end_headers()
                    return
                self._send(
                    {
                        "model": "openai/gpt-5.6-luna-20260709",
                        "openrouter_metadata": {
                            "requested": "openai/gpt-5.6-luna-20260709",
                            "attempt": 1,
                            "endpoints": {
                                "available": [
                                    {
                                        "provider": "OpenAI",
                                        "model": "openai/gpt-5.6-luna-20260709",
                                        "selected": True,
                                    }
                                ]
                            },
                        },
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "content": f"Teacher response for source row {row_id}.",
                                    "reasoning": "hidden-teacher-reasoning",
                                },
                            }
                        ],
                    }
                )
                return
            state.candidate_calls[row_id] += 1
            state.candidate_requests.append(request)
            if state.candidate_failures[row_id]:
                state.candidate_failures[row_id] -= 1
                self.send_response(500)
                self.end_headers()
                return
            self._send(
                {
                    "model": "qwen3-4b-instruct-2507",
                    "choices": [
                        {
                            "index": index,
                            "finish_reason": state.candidate_finish_reason,
                            "message": {"content": f"Qwen candidate {index} for source row {row_id}."},
                        }
                        for index in range(8)
                    ],
                }
            )

        def _send(self, value: dict[str, Any]) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield f"{base}/api/v1/chat/completions", f"{base}/v1"
    finally:
        server.shutdown()
        thread.join()


def _sha_entry(snapshot: Path, name: str, roles: list[str]) -> dict[str, Any]:
    size, digest = baseline._sha256(snapshot / name)
    return {"path": name, "roles": roles, "bytes": size, "sha256": digest}


def _server_manifest(tmp_path: Path) -> tuple[Path, Path]:
    snapshot = tmp_path / "snapshot" / control_data.PINNED_QWEN_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"model")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": "2026-08-13T00:00:00+00:00",
        "model": {
            "name": control_data.PINNED_QWEN_MODEL,
            "revision": control_data.PINNED_QWEN_REVISION,
            "snapshot_commit": control_data.PINNED_QWEN_REVISION,
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
    path = tmp_path / "server.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot, path


def _latest(path: Path) -> dict[int, dict[str, Any]]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        values[row["row_id"]] = row
    return values


def _test_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    paths = [
        "configs/baselines/qwen_control_data.json",
        "configs/data/hir.json",
        "configs/artifacts/hir_evaluator_certificate.json",
        "configs/artifacts/hir_route_implementation.json",
        "src/rdan_grpo/control_data.py",
    ]
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    judge = root / "configs/artifacts/qwen_judge_calibration.json"
    judge.write_text(
        json.dumps({"schema_version": 1, "id": "qwen_judge_calibration_v1", "status": "calibrated"}),
        encoding="utf-8",
    )
    dev = {
        "schema_version": 1,
        "id": "qwen_dev_split_v1",
        "split": "dev",
        "source_data_sha256": control_data.PINNED_SOURCE["sha256"],
        "taxonomy_sha256": control_data.program_contract.HIR_TAXONOMY_SHA256,
        "records": 1,
        "row_ids": [10612],
        "row_ids_sha256": control_data.program_contract._json_sha256({"row_ids": [10612]}),
    }
    dev_path = root / "configs/artifacts/qwen_dev_split.json"
    dev_path.write_text(json.dumps(dev, sort_keys=True) + "\n", encoding="utf-8")
    return root, root / paths[0], root / paths[1]


def _evidence(teacher: Path, candidates: Path, repo: Path, output: Path) -> tuple[Path, list[dict[str, Any]]]:
    source_rows, _ = control_data._load_source(SOURCE, control_data.PINNED_SOURCE)
    source_by_id = {row.row_id: row for row in source_rows}
    rows = []
    for row_id, teacher_row in _latest(teacher).items():
        source = source_by_id[row_id]
        hard_count = sum(source.hard_mask)
        soft_count = len(source.hard_mask) - hard_count
        outputs = [
            (teacher_row["content_sha256"], 5.0, True),
            *[
                (
                    candidate["content_sha256"],
                    20.0 if index == 1 else (-1.0 if index in {6, 7} else 10.0 - index),
                    index not in {0, 6, 7},
                )
                for index, candidate in enumerate(_latest(candidates)[row_id]["candidates"])
            ],
        ]
        for output_sha256, quality, hard_pass in outputs:
            rows.append(
                {
                    "row_id": row_id,
                    "source_sha256": source.digest,
                    "output_sha256": output_sha256,
                    "rubrics": control_data._rubric_identities(source),
                    "hard": {"pass": hard_pass, "rubric_passes": [hard_pass] * hard_count},
                    "soft": {"valid": True, "quality": quality, "rubric_scores": [quality] * soft_count},
                    "leakage": {"prompt": False, "reference": False},
                }
            )
    data_path = output.with_name("evidence.jsonl")
    _write_jsonl(data_path, rows)
    dev_path = repo / "configs/artifacts/qwen_dev_split.json"
    judge_path = repo / "configs/artifacts/qwen_judge_calibration.json"
    leakage_path = repo / "src/rdan_grpo/control_data.py"
    selected_ids = list(_latest(teacher))
    manifest = {
        "schema_version": 1,
        "id": "control_evidence_v1",
        "status": "frozen",
        "source": {
            "sha256": control_data.PINNED_SOURCE["sha256"],
            "row_ids": selected_ids,
            "row_ids_sha256": control_data.program_contract._json_sha256({"row_ids": selected_ids}),
        },
        "certificates": {
            "authoritative_evaluator": json.loads(
                (repo / "configs/baselines/qwen_control_data.json").read_text(encoding="utf-8")
            )["evidence"]["authoritative_evaluator"],
            "evaluator_implementation": json.loads(
                (repo / "configs/baselines/qwen_control_data.json").read_text(encoding="utf-8")
            )["evidence"]["evaluator_implementation"],
            "judge_calibration": {
                "path": "configs/artifacts/qwen_judge_calibration.json",
                "id": "qwen_judge_calibration_v1",
                "sha256": hashlib.sha256(judge_path.read_bytes()).hexdigest(),
            },
            "leakage_detector": {
                "path": "src/rdan_grpo/control_data.py",
                "id": "control_data_exact_leakage_v1",
                "sha256": hashlib.sha256(leakage_path.read_bytes()).hexdigest(),
            },
        },
        "dev_split": {
            "path": "configs/artifacts/qwen_dev_split.json",
            "id": "qwen_dev_split_v1",
            "sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
            "row_ids": [10612],
        },
        "soft_quality": {"minimum": -1.0, "maximum": 20.0, "higher_is_better": True},
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "records": len(rows),
        },
    }
    output.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return output, rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def test_control_data_pipeline_is_resumable_fail_closed_and_byte_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, server_manifest = _server_manifest(tmp_path)
    repo, config_path, hir_manifest_path = _test_repo(tmp_path)
    judge_validations = []

    def validate_judge(artifact: dict[str, Any], reference: dict[str, Any], _root: Path) -> None:
        expected = {
            "schema_version": 1,
            "id": "qwen_judge_calibration_v1",
            "status": "calibrated",
        }
        if artifact != expected or reference["artifact_id"] != artifact.get("id"):
            raise control_data.program_contract.ProgramContractError("invalid test judge certificate")
        judge_validations.append(reference["sha256"])

    monkeypatch.setattr(control_data.program_contract, "_validate_judge_calibration", validate_judge)

    def select_fixture_rows(
        rows: list[control_data._SourceRow],
        requested: Any,
        _config: dict[str, Any],
        _root: Path,
    ) -> list[control_data._SourceRow]:
        selected = control_data._select_rows(rows, requested)
        unsupported = [row.row_id for row in selected if row.row_id not in {10611, 10613}]
        if unsupported:
            raise control_data.ControlDataError(f"selected rows have unsupported hard routes: {unsupported}")
        return selected

    original_verified_json = control_data._verified_json

    def verify_fixture_certificate(path: Path, expected_sha256: str, expected_id: str) -> dict[str, Any]:
        if expected_id in {"hir_evaluator_certificate_v1", "hir_route_implementation_v1"}:
            value = json.loads(path.read_text(encoding="utf-8"))
            assert value["schema_version"] == 1
            assert value["id"] == expected_id
            return value
        return original_verified_json(path, expected_sha256, expected_id)

    monkeypatch.setattr(control_data, "_certified_selection", select_fixture_rows)
    monkeypatch.setattr(control_data, "_verified_json", verify_fixture_certificate)
    state = _State(snapshot)
    teacher_raw = tmp_path / "teacher.jsonl"
    candidate_raw = tmp_path / "candidates.jsonl"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(control_data.ControlDataError, match="required in the environment"):
        control_data.run_teacher_stage(
            SOURCE,
            teacher_raw,
            row_ids=[10611],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )
    assert not teacher_raw.exists()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")

    extra_config = repo / "configs/baselines/extra-config.json"
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    config_value["unexpected"] = True
    extra_config.write_text(json.dumps(config_value), encoding="utf-8")
    with pytest.raises(control_data.ControlDataError, match="pinned contract"):
        control_data.run_teacher_stage(
            SOURCE,
            tmp_path / "extra-config.jsonl",
            row_ids=[10611],
            config_path=extra_config,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )
    missing_config = repo / "configs/baselines/missing-config.json"
    del config_value["unexpected"]
    del config_value["teacher"]["timeout_seconds"]
    missing_config.write_text(json.dumps(config_value), encoding="utf-8")
    with pytest.raises(control_data.ControlDataError, match="pinned contract"):
        control_data.run_teacher_stage(
            SOURCE,
            tmp_path / "missing-config.jsonl",
            row_ids=[10611],
            config_path=missing_config,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )
    duplicate_config = repo / "configs/baselines/duplicate-config.json"
    duplicate_config.write_text(
        config_path.read_text(encoding="utf-8").replace(
            '"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'
        ),
        encoding="utf-8",
    )
    with pytest.raises(control_data.ControlDataError, match="duplicate JSON key"):
        control_data.run_teacher_stage(
            SOURCE,
            tmp_path / "duplicate-config.jsonl",
            row_ids=[10611],
            config_path=duplicate_config,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )

    with _fake_servers(state) as (teacher_endpoint, qwen_api):
        calls_before = sum(state.teacher_calls.values())
        with pytest.raises(control_data.ControlDataError, match="source bytes"):
            bad_source = tmp_path / "tampered-source.jsonl"
            bad_source.write_text("{}\n", encoding="utf-8")
            control_data.run_teacher_stage(
                bad_source,
                tmp_path / "unused.jsonl",
                row_ids=[10611],
                config_path=config_path,
                hir_manifest_path=hir_manifest_path,
                endpoint=teacher_endpoint,
                command=["test"],
                sleep=lambda _: None,
            )
        assert sum(state.teacher_calls.values()) == calls_before

        state.catalog_slug = "openai/gpt-5.6-luna-20990101"
        with pytest.raises(control_data.ControlDataError, match="does not support"):
            control_data.run_teacher_stage(
                SOURCE,
                tmp_path / "alias-drift.jsonl",
                row_ids=[10611],
                config_path=config_path,
                hir_manifest_path=hir_manifest_path,
                endpoint=teacher_endpoint,
                command=["test"],
                sleep=lambda _: None,
            )
        assert sum(state.teacher_calls.values()) == calls_before
        state.catalog_slug = "openai/gpt-5.6-luna-20260709"
        with pytest.raises(control_data.ControlDataError, match="unsupported hard routes"):
            control_data.run_teacher_stage(
                SOURCE,
                tmp_path / "unsupported-row.jsonl",
                row_ids=[1],
                config_path=config_path,
                hir_manifest_path=hir_manifest_path,
                endpoint=teacher_endpoint,
                command=["test"],
                sleep=lambda _: None,
            )

        first = control_data.run_teacher_stage(
            SOURCE,
            teacher_raw,
            row_ids=[10613, 10611],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            endpoint=teacher_endpoint,
            command=["test"],
            sleep=lambda _: None,
        )
        assert (first.successful_rows, first.failed_rows, first.recorded_failures) == (1, 1, 1)
        resumed = control_data.run_teacher_stage(
            SOURCE,
            teacher_raw,
            row_ids=[10611, 10613],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            endpoint=teacher_endpoint,
            command=["test"],
            sleep=lambda _: None,
        )
        assert (resumed.successful_rows, resumed.failed_rows, resumed.recorded_failures) == (2, 0, 1)
        assert state.teacher_calls == Counter({10613: 4, 10611: 1})
        assert all(value == "Bearer test-only-secret" for value in state.authorization)
        request = state.teacher_requests[0]
        assert request["model"] == "openai/gpt-5.6-luna-20260709"
        assert "temperature" not in request
        assert request["reasoning"] == {"effort": "medium", "exclude": True}
        assert request["provider"] == {
            "order": ["openai"],
            "only": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": False,
        }
        assert "hidden-teacher-reasoning" not in teacher_raw.read_text(encoding="utf-8")
        assert "test-only-secret" not in "".join(
            path.read_text(encoding="utf-8") for path in tmp_path.glob("teacher.jsonl*") if path.is_file()
        )

        calls_before = sum(state.teacher_calls.values())
        control_data.run_teacher_stage(
            SOURCE,
            teacher_raw,
            row_ids=[10611, 10613],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            endpoint=teacher_endpoint,
            command=["test"],
            sleep=lambda _: None,
        )
        assert sum(state.teacher_calls.values()) == calls_before
        with pytest.raises(control_data.ControlDataError, match="identity does not match"):
            control_data.run_teacher_stage(
                SOURCE,
                teacher_raw,
                row_ids=[10611, 10613],
                config_path=config_path,
                hir_manifest_path=hir_manifest_path,
                endpoint=teacher_endpoint,
                command=["different"],
                sleep=lambda _: None,
            )

        lock_path = teacher_raw.with_name(f"{teacher_raw.name}.lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(control_data.ControlDataError, match="owns the stage lock"):
                control_data.run_teacher_stage(
                    SOURCE,
                    teacher_raw,
                    row_ids=[10611, 10613],
                    config_path=config_path,
                    hir_manifest_path=hir_manifest_path,
                    endpoint=teacher_endpoint,
                    command=["test"],
                    sleep=lambda _: None,
                )

        wrong_manifest = tmp_path / "wrong-server.json"
        wrong = json.loads(server_manifest.read_text(encoding="utf-8"))
        wrong["model"]["revision"] = "0" * 40
        wrong_manifest.write_text(json.dumps(wrong), encoding="utf-8")
        with pytest.raises(control_data.ControlDataError, match="server identity failed"):
            control_data.run_candidate_stage(
                SOURCE,
                tmp_path / "wrong-candidates.jsonl",
                qwen_api,
                wrong_manifest,
                row_ids=[10611],
                config_path=config_path,
                hir_manifest_path=hir_manifest_path,
                command=["test"],
            )

        first = control_data.run_candidate_stage(
            SOURCE,
            candidate_raw,
            qwen_api,
            server_manifest,
            row_ids=[10611, 10613],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )
        assert (first.successful_rows, first.failed_rows, first.recorded_failures) == (1, 1, 1)
        resumed = control_data.run_candidate_stage(
            SOURCE,
            candidate_raw,
            qwen_api,
            server_manifest,
            row_ids=[10613, 10611],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )
        assert (resumed.successful_rows, resumed.failed_rows, resumed.recorded_failures) == (2, 0, 1)
        assert state.candidate_calls == Counter({10613: 2, 10611: 1})
        qwen_request = state.candidate_requests[0]
        assert {key: qwen_request[key] for key in ("temperature", "top_p", "top_k", "max_tokens", "seed", "n")} == {
            "temperature": 0.99,
            "top_p": 0.99,
            "top_k": 100,
            "max_tokens": 4096,
            "seed": 240520,
            "n": 8,
        }
        state.candidate_finish_reason = "length"
        incomplete = tmp_path / "incomplete-candidates.jsonl"
        incomplete_result = control_data.run_candidate_stage(
            SOURCE,
            incomplete,
            qwen_api,
            server_manifest,
            row_ids=[10611],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )
        assert (incomplete_result.successful_rows, incomplete_result.failed_rows) == (0, 1)
        assert _latest(incomplete)[10611]["error"] == "candidate_incomplete"

    with pytest.raises(control_data.ControlDataError, match="outside Git"):
        control_data.run_teacher_stage(
            SOURCE,
            ROOT / "raw-control-test.jsonl",
            row_ids=[10611],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )
    raw_symlink = tmp_path / "raw-symlink.jsonl"
    raw_symlink.symlink_to(ROOT / "configs/data/hir.json")
    with pytest.raises(control_data.ControlDataError, match="outside Git"):
        control_data.run_teacher_stage(
            SOURCE,
            raw_symlink,
            row_ids=[10611],
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["test"],
        )

    short_write_path = tmp_path / "short-write.jsonl"
    original_write = control_data.os.write
    write_calls = 0

    def short_write(descriptor: int, value: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return original_write(descriptor, value[:3])
        return original_write(descriptor, value)

    with monkeypatch.context() as patch:
        patch.setattr(control_data.os, "write", short_write)
        control_data._append_jsonl(short_write_path, {"row_id": 1})
    assert write_calls == 2
    assert short_write_path.read_bytes().endswith(b"\n")
    assert json.loads(short_write_path.read_text(encoding="utf-8")) == {"row_id": 1}

    interrupted_write_path = tmp_path / "interrupted-write.jsonl"
    interrupted_write_path.write_bytes(b'{"row_id":0}\n')
    write_calls = 0

    def interrupt_write(descriptor: int, value: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return original_write(descriptor, value[:3])
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(control_data.os, "write", interrupt_write)
        with pytest.raises(KeyboardInterrupt):
            control_data._append_jsonl(interrupted_write_path, {"row_id": 1})
    assert interrupted_write_path.read_bytes() == b'{"row_id":0}\n'

    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            duplicate_outputs = [repo / f"data/output-{index}" for index in range(4)]
            duplicate_outputs[second_index] = duplicate_outputs[first_index]
            with pytest.raises(control_data.ControlDataError, match="paths must be distinct"):
                control_data.freeze_control_data(
                    SOURCE,
                    teacher_raw,
                    candidate_raw,
                    tmp_path / "not-read.json",
                    *duplicate_outputs,
                    config_path=config_path,
                    hir_manifest_path=hir_manifest_path,
                    command=["freeze-test"],
                )

    teacher_records = [json.loads(line) for line in teacher_raw.read_text(encoding="utf-8").splitlines()]
    first_teacher = next(row for row in teacher_records if row["row_id"] == 10611 and row["status"] == "ok")
    second_teacher = next(row for row in teacher_records if row["row_id"] == 10613 and row["status"] == "ok")
    second_teacher["content"] = first_teacher["content"]
    second_teacher["content_sha256"] = first_teacher["content_sha256"]
    _write_jsonl(teacher_raw, teacher_records)

    original_link = control_data.os.link
    for failure in (OSError("injected publish failure"), KeyboardInterrupt()):
        expected = control_data.ControlDataError if isinstance(failure, OSError) else KeyboardInterrupt
        for fail_at in range(1, 5):
            published = [tmp_path / f"publish-{type(failure).__name__}-{fail_at}-{index}" for index in range(4)]
            calls = 0

            def fail_link(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == fail_at:
                    raise failure
                original_link(source, destination)

            with monkeypatch.context() as patch:
                patch.setattr(control_data.os, "link", fail_link)
                with pytest.raises(expected, match="publish" if expected is control_data.ControlDataError else None):
                    control_data._publish_immutable(dict.fromkeys(published, b"content"))
            assert not any(path.exists() for path in published)

    evidence, evidence_rows = _evidence(teacher_raw, candidate_raw, repo, tmp_path / "evidence-manifest.json")
    judge_path = repo / "configs/artifacts/qwen_judge_calibration.json"
    valid_judge_bytes = judge_path.read_bytes()
    malformed_judge = json.loads(valid_judge_bytes)
    malformed_judge["status"] = "forged"
    judge_path.write_text(json.dumps(malformed_judge), encoding="utf-8")
    malformed_manifest = tmp_path / "malformed-judge-manifest.json"
    malformed_value = json.loads(evidence.read_text(encoding="utf-8"))
    malformed_value["certificates"]["judge_calibration"]["sha256"] = hashlib.sha256(
        judge_path.read_bytes()
    ).hexdigest()
    malformed_manifest.write_text(json.dumps(malformed_value), encoding="utf-8")
    with pytest.raises(control_data.ControlDataError, match="judge calibration certificate is invalid"):
        control_data.freeze_control_data(
            SOURCE,
            teacher_raw,
            candidate_raw,
            malformed_manifest,
            repo / "data/malformed-sft.jsonl",
            repo / "data/malformed-dpo.jsonl",
            repo / "configs/artifacts/malformed-sft.json",
            repo / "configs/artifacts/malformed-dpo.json",
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["freeze-test"],
        )
    judge_path.write_bytes(valid_judge_bytes)
    missing = tmp_path / "missing-manifest.json"
    missing_value = json.loads(evidence.read_text(encoding="utf-8"))
    missing_value["source"]["row_ids"] = missing_value["source"]["row_ids"][:-1]
    missing.write_text(json.dumps(missing_value), encoding="utf-8")
    with pytest.raises(control_data.ControlDataError, match="forged or reordered"):
        control_data.freeze_control_data(
            SOURCE,
            teacher_raw,
            candidate_raw,
            missing,
            tmp_path / "missing-sft.jsonl",
            tmp_path / "missing-dpo.jsonl",
            tmp_path / "missing-sft.json",
            tmp_path / "missing-dpo.json",
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["freeze-test"],
        )

    forged = tmp_path / "forged-manifest.json"
    forged_value = json.loads(evidence.read_text(encoding="utf-8"))
    forged_value["certificates"]["authoritative_evaluator"]["sha256"] = "0" * 64
    forged.write_text(json.dumps(forged_value), encoding="utf-8")
    with pytest.raises(control_data.ControlDataError, match="forged"):
        control_data.freeze_control_data(
            SOURCE,
            teacher_raw,
            candidate_raw,
            forged,
            tmp_path / "unsupported-sft.jsonl",
            tmp_path / "unsupported-dpo.jsonl",
            tmp_path / "unsupported-sft.json",
            tmp_path / "unsupported-dpo.json",
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["freeze-test"],
        )

    mixed_scale = tmp_path / "mixed-scale-manifest.json"
    mixed_value = json.loads(evidence.read_text(encoding="utf-8"))
    mixed_value["soft_quality"]["maximum"] = 10.0
    mixed_scale.write_text(json.dumps(mixed_value), encoding="utf-8")
    with pytest.raises(control_data.ControlDataError, match="incomplete evidence"):
        control_data.freeze_control_data(
            SOURCE,
            teacher_raw,
            candidate_raw,
            mixed_scale,
            tmp_path / "mixed-sft.jsonl",
            tmp_path / "mixed-dpo.jsonl",
            tmp_path / "mixed-sft.json",
            tmp_path / "mixed-dpo.json",
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["freeze-test"],
        )

    dev_leak = tmp_path / "dev-leak-manifest.json"
    dev_value = json.loads(evidence.read_text(encoding="utf-8"))
    dev_artifact_path = repo / "configs/artifacts/qwen_dev_split_leak.json"
    dev_artifact = json.loads((repo / "configs/artifacts/qwen_dev_split.json").read_text(encoding="utf-8"))
    dev_artifact["id"] = "qwen_dev_split_leak_v1"
    dev_artifact["row_ids"] = [10611]
    dev_artifact["row_ids_sha256"] = control_data.program_contract._json_sha256({"row_ids": [10611]})
    dev_artifact_path.write_text(json.dumps(dev_artifact, sort_keys=True) + "\n", encoding="utf-8")
    dev_value["dev_split"] = {
        "path": "configs/artifacts/qwen_dev_split_leak.json",
        "id": dev_artifact["id"],
        "sha256": hashlib.sha256(dev_artifact_path.read_bytes()).hexdigest(),
        "row_ids": [10611],
    }
    dev_leak.write_text(json.dumps(dev_value), encoding="utf-8")
    with pytest.raises(control_data.ControlDataError, match="leak into"):
        control_data.freeze_control_data(
            SOURCE,
            teacher_raw,
            candidate_raw,
            dev_leak,
            tmp_path / "leak-sft.jsonl",
            tmp_path / "leak-dpo.jsonl",
            tmp_path / "leak-sft.json",
            tmp_path / "leak-dpo.json",
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["freeze-test"],
        )

    duplicate_raw = tmp_path / "duplicate-teacher.jsonl"
    duplicate_raw.write_bytes(teacher_raw.read_bytes() + teacher_raw.read_bytes().splitlines(keepends=True)[0])
    duplicate_identity = duplicate_raw.with_name(f"{duplicate_raw.name}.identity.json")
    duplicate_identity.write_bytes(teacher_raw.with_name(f"{teacher_raw.name}.identity.json").read_bytes())
    with pytest.raises(control_data.ControlDataError, match="duplicate"):
        control_data.freeze_control_data(
            SOURCE,
            duplicate_raw,
            candidate_raw,
            evidence,
            tmp_path / "duplicate-sft.jsonl",
            tmp_path / "duplicate-dpo.jsonl",
            tmp_path / "duplicate-sft.json",
            tmp_path / "duplicate-dpo.json",
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["freeze-test"],
        )

    sft = repo / "data/sft.jsonl"
    dpo = repo / "data/dpo.jsonl"
    sft_manifest_path = repo / "configs/artifacts/sft-manifest.json"
    dpo_manifest_path = repo / "configs/artifacts/dpo-manifest.json"
    sft_manifest, dpo_manifest = control_data.freeze_control_data(
        SOURCE,
        teacher_raw,
        candidate_raw,
        evidence,
        sft,
        dpo,
        sft_manifest_path,
        dpo_manifest_path,
        config_path=config_path,
        hir_manifest_path=hir_manifest_path,
        command=["freeze-test"],
    )
    sft_rows = [json.loads(line) for line in sft.read_text(encoding="utf-8").splitlines()]
    dpo_rows = [json.loads(line) for line in dpo.read_text(encoding="utf-8").splitlines()]
    assert len(sft_rows) == 1
    assert len(dpo_rows) == 2
    assert all(set(row) == {"row_id", "prompt", "output"} for row in sft_rows)
    assert all(set(row) == {"row_id", "prompt", "chosen", "rejected"} for row in dpo_rows)
    assert [row["chosen"] for row in dpo_rows] == [
        "Qwen candidate 1 for source row 10611.",
        "Qwen candidate 1 for source row 10613.",
    ]
    assert sft_manifest["source_sha256"] == json.loads(config_path.read_text())["source_sha256"]
    assert sft_manifest["data"]["sha256"] == hashlib.sha256(sft.read_bytes()).hexdigest()
    assert dpo_manifest["data"]["sha256"] == hashlib.sha256(dpo.read_bytes()).hexdigest()
    control_data.program_contract._validate_dataset_manifest(sft_manifest, "sft", repo)
    control_data.program_contract._validate_dataset_manifest(dpo_manifest, "dpo", repo)
    assert judge_validations
    for name, manifest in (("sft", sft_manifest), ("dpo", dpo_manifest)):
        pinned = {
            "id": f"{name}_reconstructed",
            "data": {
                "manifest_id": manifest["id"],
                "teacher": {
                    "model_id": "openai/gpt-5.6-luna-20260709",
                    "revision": "openai/gpt-5.6-luna-20260709",
                },
            },
        }
        control_data.program_contract._validate_baseline_artifact(name, pinned, manifest, repo)

    with pytest.raises(control_data.ControlDataError, match="already exists"):
        control_data.freeze_control_data(
            SOURCE,
            teacher_raw,
            candidate_raw,
            evidence,
            sft,
            dpo,
            sft_manifest_path,
            dpo_manifest_path,
            config_path=config_path,
            hir_manifest_path=hir_manifest_path,
            command=["freeze-test"],
        )
