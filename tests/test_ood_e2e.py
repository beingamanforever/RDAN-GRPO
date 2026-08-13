from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from rdan_grpo import baseline, ood


class _State:
    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self.requests: list[dict[str, Any]] = []
        self.calls: Counter[int] = Counter()


@contextmanager
def _server(state: _State) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._send({"data": [{"id": "checkpoint", "root": str(state.snapshot)}]})

        def do_POST(self) -> None:  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(request)
            seed = request["seed"]
            state.calls[seed] += 1
            answer = "41" if seed == 1701 else "42"
            self._send(
                {
                    "choices": [
                        {
                            "message": {"content": f"Reasoning\n\\boxed{{{answer}}}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 4},
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
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()


def _entry(snapshot: Path, name: str, roles: list[str]) -> dict[str, Any]:
    size, digest = baseline._sha256(snapshot / name)
    return {"path": name, "roles": roles, "bytes": size, "sha256": digest}


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data = tmp_path / "math.jsonl"
    rows = [{"problem": "What is 40 + 2?", "answer": "42"}, {"problem": "Six times seven?", "answer": "42"}]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data_hash = ood._sha256(data)
    config = json.loads(ood.CONFIG.read_text(encoding="utf-8"))
    config["benchmarks"]["math_500"].update({"records": 2, "sha256": data_hash})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setitem(ood.EXPECTED, "math_500", (2, ood.EXPECTED["math_500"][1], data_hash))
    monkeypatch.setattr(
        ood,
        "_verify_math_runtime",
        lambda _root, scorer: {"revision": scorer["scorer_revision"], "antlr_version": scorer["antlr_version"]},
    )
    monkeypatch.setattr(
        ood,
        "_score_math",
        lambda gold, response: (
            ood._last_boxed(response) == gold,
            "parsed" if ood._last_boxed(response) else "missing_boxed",
        ),
    )

    revision = "a" * 40
    snapshot = tmp_path / "snapshot" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"model")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": "2026-08-13T00:00:00+00:00",
        "model": {
            "name": "test/checkpoint",
            "revision": revision,
            "snapshot_commit": revision,
            "snapshot_path": str(snapshot),
        },
        "files": [
            _entry(snapshot, "model.safetensors", ["model"]),
            _entry(snapshot, "tokenizer.json", ["tokenizer"]),
            _entry(snapshot, "tokenizer_config.json", ["tokenizer", "chat_template"]),
        ],
        "server": {
            "argv": [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(snapshot),
                "--served-model-name",
                "checkpoint",
            ],
            "packages": {"vllm": "0.10.2", "transformers": "4.57.0", "torch": "2.8.0"},
        },
    }
    manifest_path = tmp_path / "server.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"data": data, "config": config_path, "snapshot": snapshot, "manifest": manifest_path}


def _run(paths: dict[str, Path], api_base: str, output: Path) -> Path:
    return ood.run_evaluation(
        benchmark="math_500",
        data_path=paths["data"],
        api_base=api_base,
        server_manifest=paths["manifest"],
        output_dir=output,
        concurrency=2,
        config_path=paths["config"],
        math_verify_root=output.parent,
        argv=["run_ood_eval.py", "--api-base", api_base],
    )


def test_fake_server_e2e_seals_scores_and_reuses_fixed_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "outside" / "run"
    state = _State(paths["snapshot"])
    with _server(state) as api_base:
        _run(paths, api_base, output)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["correct"] == 8
    assert metrics["total"] == 10
    assert metrics["micro_accuracy"] == 0.8
    assert [row["accuracy"] for row in metrics["per_completion"]] == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert state.calls == Counter({seed: 2 for seed in (1701, 1702, 1703, 1704, 1705)})
    assert all("chat_template_kwargs" not in request for request in state.requests)
    assert all(request["temperature"] == 0.7 and request["top_p"] == 0.8 for request in state.requests)
    assert {request["top_k"] for request in state.requests} == {20}
    assert "\\boxed{}" in state.requests[0]["messages"][0]["content"]
    compact = (output / "metrics.json").read_text(encoding="utf-8")
    assert "prompt" not in compact and "response" not in compact
    manifest = json.loads((output / "sha256_manifest.json").read_text(encoding="utf-8"))
    assert {item["path"] for item in manifest["files"]} == {
        "command.json",
        "metrics.json",
        "records.jsonl",
        "resume_identity.json",
    }
    assert "127.0.0.1" not in (output / "command.json").read_text(encoding="utf-8")


def test_sealed_run_is_verified_and_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "outside" / "run"
    state = _State(paths["snapshot"])
    with _server(state) as api_base:
        _run(paths, api_base, output)
        before = copy.copy(state.calls)
        _run(paths, api_base, output)
    assert state.calls == before


def test_math_rescore_reuses_sealed_generations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    source = tmp_path / "outside" / "source"
    state = _State(paths["snapshot"])
    with _server(state) as api_base:
        _run(paths, api_base, source)
    source_manifest = ood._sha256(source / "sha256_manifest.json")
    source_records = ood._sha256(source / "records.jsonl")
    monkeypatch.setattr(ood, "_score_math", lambda _gold, _response: (True, "parsed"))

    output = ood.rescore_math_artifact(
        source_dir=source,
        output_dir=tmp_path / "outside" / "rescored",
        data_path=paths["data"],
        math_verify_root=tmp_path,
        config_path=paths["config"],
        argv=["rescore_math_artifact.py"],
    )

    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))["correct"] == 10
    provenance = json.loads((output / "rescore.json").read_text(encoding="utf-8"))
    assert provenance["source_manifest_sha256"] == source_manifest
    assert provenance["source_records_sha256"] == source_records
    assert provenance["changed_rows"] == 2
    assert ood._sha256(source / "records.jsonl") == source_records
    ood._verify_seal(output)

    other = tmp_path / "outside" / "other-source"
    shutil.copytree(source, other)
    (other / "command.json").write_text('{"argv":["different"]}\n', encoding="utf-8")
    ood._seal(other, {".lock", "sha256_manifest.json"})
    with pytest.raises(ood.OODError, match="provenance differs"):
        ood.rescore_math_artifact(
            source_dir=other,
            output_dir=output,
            data_path=paths["data"],
            math_verify_root=tmp_path,
            config_path=paths["config"],
        )


def test_valid_partial_resume_does_not_repeat_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "outside" / "run"
    state = _State(paths["snapshot"])
    with _server(state) as api_base:
        _run(paths, api_base, output)
        rows = (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
        (output / "partial_records.jsonl").write_text(rows[0] + "\n", encoding="utf-8")
        (output / "records.jsonl").unlink()
        (output / "metrics.json").unlink()
        (output / "command.json").unlink()
        (output / "sha256_manifest.json").unlink()
        before = copy.copy(state.calls)
        request_count = len(state.requests)
        _run(paths, api_base, output)
    first = json.loads(rows[0])
    assert state.calls[first["request_seed"]] == before[first["request_seed"]] + 1
    assert sum(state.calls.values()) == sum(before.values()) + 9
    assert not any(
        request["seed"] == first["request_seed"] and request["messages"][0]["content"] == first["prompt"]
        for request in state.requests[request_count:]
    )


def test_resume_rejects_server_identity_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "outside" / "run"
    state = _State(paths["snapshot"])
    with _server(state) as api_base:
        _run(paths, api_base, output)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["created_at"] = "2026-08-13T00:00:01+00:00"
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ood.OODError, match="resume identity"):
            _run(paths, api_base, output)


def test_gpqa_is_blocked_before_data_or_server_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ood, "_load_server", lambda *_args: pytest.fail("server accessed"))
    monkeypatch.setattr(ood, "_load_items", lambda *_args: pytest.fail("data accessed"))
    with pytest.raises(ood.OODError, match="blocked until authenticated"):
        ood.run_evaluation(
            benchmark="gpqa",
            data_path=tmp_path / "missing.csv",
            api_base="http://127.0.0.1:1/v1",
            server_manifest=tmp_path / "missing.json",
            output_dir=tmp_path / "outside",
            concurrency=1,
        )


def test_gpqa_runs_only_when_config_matches_frozen_access_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = {
        "dataset": "Idavidrein/gpqa",
        "config": "gpqa_main",
        "split": "train",
        "revision": "1" * 40,
        "records": 448,
        "filename": "gpqa_main.csv",
        "sha256": "2" * 64,
    }
    access = {
        "schema_version": 1,
        "id": "gpqa_access_v1",
        "status": "frozen",
        "dataset": dataset,
        "access": {"provider": "huggingface", "gated": True, "verified": True},
    }
    access_path = tmp_path / "gpqa_access.json"
    access_path.write_text(json.dumps(access), encoding="utf-8")
    config = json.loads(ood.CONFIG.read_text(encoding="utf-8"))
    config["benchmarks"]["gpqa"] = {
        **dataset,
        "shuffle_revision": "56686c06f5e19865c153de0fdb11be3890014df7",
        "shuffle_seed": 0,
        "execution_state": "runnable_frozen_access",
    }
    config_path = tmp_path / "ood.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(ood, "_load_server", lambda *_args: (_ for _ in ()).throw(ood.OODError("server reached")))
    with pytest.raises(ood.OODError, match="server reached"):
        ood.run_evaluation(
            benchmark="gpqa",
            data_path=tmp_path / "gpqa_main.csv",
            api_base="http://127.0.0.1:1/v1",
            server_manifest=tmp_path / "server.json",
            output_dir=tmp_path / "outside",
            concurrency=1,
            config_path=config_path,
            gpqa_access_path=access_path,
        )

    config["benchmarks"]["gpqa"]["sha256"] = "3" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ood.OODError, match="does not match"):
        ood._load_config(config_path, access_path)


def test_config_rejects_prompt_drift(tmp_path: Path) -> None:
    config = json.loads(ood.CONFIG.read_text(encoding="utf-8"))
    config["prompts"]["math_500"] += " changed"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ood.OODError, match="prompts differ"):
        ood._load_config(path)


@pytest.mark.parametrize(
    ("response", "benchmark", "expected"),
    [
        ("reason\nAnswer: A", "gpqa", "A"),
        ("reason\nanswer: A", "gpqa", None),
        ("reason\nAnswer: A extra", "gpqa", None),
        ("Answer: J", "mmlu_pro", "J"),
        ("Answer: K", "mmlu_pro", None),
    ],
)
def test_strict_last_line_scoring(response: str, benchmark: str, expected: str | None) -> None:
    assert ood._last_line_answer(response, benchmark) == expected


def test_gpqa_permutation_is_fixed_and_mmlu_removes_only_literal_na() -> None:
    records = [
        {
            "Question": "Question one?",
            "Correct Answer": "right",
            "Incorrect Answer 1": "wrong one",
            "Incorrect Answer 2": "wrong two",
            "Incorrect Answer 3": "wrong three",
        },
        {
            "Question": "Question two?",
            "Correct Answer": "yes",
            "Incorrect Answer 1": "no one",
            "Incorrect Answer 2": "no two",
            "Incorrect Answer 3": "no three",
        },
    ]
    template = "Question: {}\nAnswer instructions"
    first = ood._gpqa_items(records, template, 0)
    second = ood._gpqa_items(records, template, 0)
    assert [item.permutation for item in first] == [item.permutation for item in second]
    assert [item.permutation for item in first] == [(2, 0, 1, 3), (0, 1, 3, 2)]
    assert [item.answer for item in first] == ["D", "C"]
    item = ood._mmlu_item({"question": "Q?", "options": ["x", "N/A", "y"], "answer_index": 2}, 0, template)
    assert "A. x\nB. y" in item.prompt
    assert item.answer == "B"


def test_last_boxed_handles_nested_and_requires_closed_box() -> None:
    assert ood._last_boxed(r"first \boxed{1}, final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert ood._last_boxed(r"\boxed{unfinished") is None


def test_math_scorer_wraps_selected_latex_in_box(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Config:
        def __init__(self, *, boxed_match_priority: int) -> None:
            assert boxed_match_priority == 0

    def parse(value: str, *, extraction_config: list[Config], fallback_mode: str) -> list[str]:
        calls.append((value, extraction_config, fallback_mode))
        return [value]

    module = SimpleNamespace(LatexExtractionConfig=Config, parse=parse, verify=lambda gold, pred: gold == pred)
    monkeypatch.setitem(sys.modules, "math_verify", module)
    assert ood._score_math("p-q", r"work \boxed{p-q}") == (True, "parsed")
    assert [call[0] for call in calls] == [r"\boxed{p-q}", r"\boxed{p-q}"]
    assert all(call[2] == "no_fallback" for call in calls)


def test_pinned_math_scorer_handles_tuple_and_symbolic_answers() -> None:
    root = os.environ.get("RDAN_MATH_VERIFY_ROOT")
    if root is None:
        pytest.skip("set RDAN_MATH_VERIFY_ROOT to run the pinned parser regression")
    config = ood._load_config(ood.CONFIG)["benchmarks"]["math_500"]
    ood._verify_math_runtime(Path(root), config)
    assert ood._score_math(r"\left( 3, \frac{\pi}{2} \right)", r"\boxed{(3, \frac{\pi}{2})}") == (
        True,
        "parsed",
    )
    assert ood._score_math("p - q", r"\boxed{p-q}") == (True, "parsed")


def test_math_runtime_has_no_unpinned_fallback() -> None:
    with pytest.raises(ood.OODError, match="pinned Math-Verify checkout"):
        ood._verify_math_runtime(None, {"scorer_revision": "a" * 40, "antlr_version": "4.9.3"})


def test_raw_output_inside_repository_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ood.OODError, match="outside the Git checkout"):
        ood.run_evaluation(
            benchmark="math_500",
            data_path=tmp_path / "missing",
            api_base="http://127.0.0.1:1/v1",
            server_manifest=tmp_path / "missing",
            output_dir=ood.ROOT / "results/ood-raw",
            concurrency=1,
        )


@pytest.mark.parametrize("mutation", ["added_file", "added_dir", "removed", "renamed"])
def test_seal_rejects_any_unmanifested_directory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    paths = _setup(tmp_path, monkeypatch)
    output = tmp_path / "outside" / "run"
    state = _State(paths["snapshot"])
    with _server(state) as api_base:
        _run(paths, api_base, output)
        if mutation == "added_file":
            (output / "extra.txt").write_text("extra", encoding="utf-8")
        elif mutation == "added_dir":
            (output / "extra").mkdir()
        elif mutation == "removed":
            (output / "metrics.json").unlink()
        else:
            (output / "metrics.json").rename(output / "renamed.json")
        with pytest.raises(ood.OODError, match="sealed OOD"):
            _run(paths, api_base, output)
