"""Pinned, resumable OOD evaluation for frozen Qwen checkpoints."""

from __future__ import annotations

import concurrent.futures
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdan_grpo import baseline

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/eval/ood_qwen.json"
GPQA_ACCESS = ROOT / "configs/artifacts/gpqa_access.json"
EXPECTED = {
    "math_500": (
        500,
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    ),
    "gpqa": (448, "pending_gated_access", "pending_gated_access"),
    "mmlu_pro": (
        12032,
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    ),
}
EXPECTED_PROMPTS = {
    "math_500": "Question: {}\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "multiple_choice": (
        "Question: {}\nAnswer the multiple choice question. The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of choices. "
        "Think step by step before answering."
    ),
}
MC_PATTERN = {"gpqa": re.compile(r"^Answer:\s*([A-D])\s*$"), "mmlu_pro": re.compile(r"^Answer:\s*([A-J])\s*$")}
RAW_KEYS = {
    "resume_identity",
    "benchmark",
    "item_hash",
    "completion_index",
    "request_seed",
    "prompt",
    "response",
    "response_sha256",
    "finish_reason",
    "completion_tokens",
    "prompt_tokens",
    "latency_seconds",
    "parser_state",
    "parsed_answer",
    "correct",
    "permutation",
}


class OODError(RuntimeError):
    """Raised when OOD evidence violates the frozen protocol."""


@dataclass(frozen=True)
class Item:
    item_hash: str
    prompt: str
    answer: str
    permutation: tuple[int, ...] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_config(path: Path, gpqa_access_path: Path = GPQA_ACCESS) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        protocol = config["protocol"]
        generation = protocol["generation"]
        if config["schema_version"] != 1 or set(config["benchmarks"]) != set(EXPECTED):
            raise OODError("unsupported OOD configuration")
        if config["prompts"] != EXPECTED_PROMPTS:
            raise OODError("OOD prompts differ from RTT Appendix B")
        if protocol["name"] != "rdan-rtt-appendix-b-v1":
            raise OODError("unsupported OOD protocol identity")
        if protocol["selection_use"] != "evaluation_only_after_checkpoint_freeze":
            raise OODError("OOD data must be evaluation only")
        if protocol["completion_seeds"] != [1701, 1702, 1703, 1704, 1705] or generation != {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 4096,
            "n": 1,
            "request_timeout_seconds": 600,
        }:
            raise OODError("generation protocol differs from the frozen contract")
        if protocol["chat_template"] != "checkpoint_default":
            raise OODError("OOD evaluation must use the checkpoint default chat template")
        for name, expected in EXPECTED.items():
            benchmark = config["benchmarks"][name]
            if name == "gpqa":
                continue
            if (benchmark["records"], benchmark["revision"], benchmark["sha256"]) != expected:
                raise OODError(f"{name} dataset pin differs from the frozen contract")
        if config["benchmarks"]["math_500"]["scorer_revision"] != "ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b":
            raise OODError("Math-Verify revision differs from the frozen contract")
        if config["benchmarks"]["math_500"]["antlr_version"] != "4.9.3":
            raise OODError("ANTLR revision differs from the frozen contract")
        gpqa = config["benchmarks"]["gpqa"]
        if gpqa["shuffle_revision"] != "56686c06f5e19865c153de0fdb11be3890014df7" or gpqa["shuffle_seed"] != 0:
            raise OODError("GPQA option permutation differs from the frozen contract")
        pending_gpqa = (
            (gpqa["records"], gpqa["revision"], gpqa["sha256"]) == EXPECTED["gpqa"]
            and gpqa["filename"] == "pending_gated_access"
            and gpqa["execution_state"] == "blocked_gated_access"
        )
        if not pending_gpqa:
            access = _load_gpqa_access(gpqa_access_path)
            if gpqa != {
                **access["dataset"],
                "shuffle_revision": gpqa["shuffle_revision"],
                "shuffle_seed": 0,
                "execution_state": "runnable_frozen_access",
            }:
                raise OODError("GPQA config does not match the frozen access artifact")
        mmlu = config["benchmarks"]["mmlu_pro"]
        if mmlu["remove_option"] != "N/A" or mmlu["reshuffle"] is not False:
            raise OODError("MMLU-Pro option handling differs from the frozen contract")
        return config
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        raise OODError(f"invalid OOD configuration: {error}") from error


def _load_gpqa_access(path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OODError(f"invalid GPQA access artifact: {error}") from error
    expected_keys = {"schema_version", "id", "status", "dataset", "access"}
    dataset_keys = {"dataset", "config", "split", "revision", "records", "filename", "sha256"}
    if not isinstance(artifact, dict) or set(artifact) != expected_keys:
        raise OODError("invalid GPQA access artifact")
    dataset = artifact.get("dataset")
    access = artifact.get("access")
    if (
        artifact.get("schema_version") != 1
        or not isinstance(artifact.get("id"), str)
        or artifact.get("id") in {"", "pending"}
        or artifact.get("status") != "frozen"
        or not isinstance(dataset, dict)
        or set(dataset) != dataset_keys
        or dataset.get("dataset") != "Idavidrein/gpqa"
        or dataset.get("config") != "gpqa_main"
        or dataset.get("split") != "train"
        or dataset.get("records") != 448
        or not isinstance(dataset.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", dataset["revision"]) is None
        or not isinstance(dataset.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset["sha256"]) is None
        or not isinstance(dataset.get("filename"), str)
        or not dataset["filename"]
        or Path(dataset["filename"]).is_absolute()
        or ".." in Path(dataset["filename"]).parts
        or access != {"provider": "huggingface", "gated": True, "verified": True}
    ):
        raise OODError("invalid GPQA access artifact")
    return artifact


def _load_server(path: Path, api_base: str, timeout: int) -> tuple[dict[str, Any], str, str]:
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        model = candidate["model"]
        argv = candidate["server"]["argv"]
        served_name = baseline._argument(argv, "--served-model-name")
        expected = {"name": model["name"], "revision": model["revision"], "served_name": served_name}
    except (KeyError, TypeError, json.JSONDecodeError, OSError, baseline.EvaluationError) as error:
        raise OODError(f"invalid server identity: {error}") from error
    try:
        manifest, digest = baseline._load_server_manifest(path, expected)
        baseline._verify_model(api_base, served_name, Path(model["snapshot_path"]), timeout)
    except baseline.EvaluationError as error:
        raise OODError(str(error)) from error
    return manifest, digest, served_name


def _load_items(name: str, path: Path, config: Mapping[str, Any], prompts: Mapping[str, str]) -> list[Item]:
    expected_hash = config["sha256"]
    if expected_hash == "pending_gated_access":
        raise OODError("GPQA is blocked until authenticated bytes, revision, and SHA-256 are frozen")
    if not path.is_file() or _sha256(path) != expected_hash:
        raise OODError(f"{name} input does not match the pinned SHA-256")
    if name == "math_500":
        records = _jsonl(path)
        items = [_math_item(record, index, prompts["math_500"]) for index, record in enumerate(records)]
    elif name == "gpqa":
        records = _csv(path)
        items = _gpqa_items(records, prompts["multiple_choice"], config["shuffle_seed"])
    else:
        records = _parquet(path)
        items = [_mmlu_item(record, index, prompts["multiple_choice"]) for index, record in enumerate(records)]
    if len(items) != config["records"] or len({item.item_hash for item in items}) != len(items):
        raise OODError(f"{name} prepared count or stable identity is invalid")
    return items


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise OODError(f"invalid JSONL input: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise OODError("JSONL rows must be objects")
    return rows


def _csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise OODError(f"invalid CSV input: {error}") from error


def _parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise OODError("MMLU-Pro requires pyarrow to read the pinned parquet file") from error
    try:
        return parquet.read_table(path).to_pylist()
    except Exception as error:
        raise OODError(f"invalid MMLU-Pro parquet input: {error}") from error


def _text(record: Mapping[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OODError(f"{label} has invalid {key}")
    return value


def _item_hash(name: str, index: int, question: str) -> str:
    return hashlib.sha256(f"{name}\0{index}\0{question}".encode()).hexdigest()


def _math_item(record: Mapping[str, Any], index: int, template: str) -> Item:
    question = _text(record, "problem", f"MATH-500 row {index}")
    answer = _text(record, "answer", f"MATH-500 row {index}")
    return Item(_item_hash("math_500", index, question), template.replace("{}", question, 1), answer)


def _gpqa_items(records: Sequence[Mapping[str, Any]], template: str, seed: int) -> list[Item]:
    rng = random.Random(seed)
    items = []
    for index, record in enumerate(records):
        question = _text(record, "Question", f"GPQA row {index}")
        choices = [
            *[_text(record, f"Incorrect Answer {number}", f"GPQA row {index}") for number in range(1, 4)],
            _text(record, "Correct Answer", f"GPQA row {index}"),
        ]
        permutation = list(range(4))
        rng.shuffle(permutation)
        ordered = [choices[position] for position in permutation]
        answer = chr(65 + permutation.index(3))
        question_with_choices = (
            question
            + "\nChoices:\n"
            + "\n".join(f"({chr(65 + position)}) {choice}" for position, choice in enumerate(ordered))
        )
        items.append(
            Item(
                _item_hash("gpqa", index, question),
                template.replace("{}", question_with_choices, 1),
                answer,
                tuple(permutation),
            )
        )
    return items


def _mmlu_item(record: Mapping[str, Any], index: int, template: str) -> Item:
    question = _text(record, "question", f"MMLU-Pro row {index}")
    options = record.get("options")
    answer_index = record.get("answer_index")
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise OODError(f"MMLU-Pro row {index} has invalid options")
    if isinstance(answer_index, bool) or not isinstance(answer_index, int) or not 0 <= answer_index < len(options):
        raise OODError(f"MMLU-Pro row {index} has invalid answer_index")
    correct = options[answer_index]
    filtered = [option for option in options if option != "N/A"]
    if correct == "N/A" or correct not in filtered or not 2 <= len(filtered) <= 10:
        raise OODError(f"MMLU-Pro row {index} has invalid filtered options")
    answer = chr(65 + filtered.index(correct))
    question_with_options = (
        question
        + "\nOptions:\n"
        + "\n".join(f"{chr(65 + position)}. {option}" for position, option in enumerate(filtered))
    )
    return Item(_item_hash("mmlu_pro", index, question), template.replace("{}", question_with_options, 1), answer)


def _last_line_answer(response: str, benchmark: str) -> str | None:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    match = MC_PATTERN[benchmark].fullmatch(lines[-1]) if lines else None
    return match.group(1) if match else None


def _last_boxed(response: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", response))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[start:index]
    return None


def _verify_math_runtime(root: Path | None, config: Mapping[str, Any]) -> dict[str, str]:
    if root is None or not root.is_dir():
        raise OODError("MATH-500 requires the pinned Math-Verify checkout")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode or status.returncode or result.stdout.strip() != config["scorer_revision"] or status.stdout:
        raise OODError("Math-Verify checkout identity is not pinned and clean")
    try:
        antlr = importlib.metadata.version("antlr4-python3-runtime")
    except importlib.metadata.PackageNotFoundError as error:
        raise OODError("ANTLR runtime is not installed") from error
    if antlr != config["antlr_version"]:
        raise OODError(f"ANTLR runtime must equal {config['antlr_version']}")
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        import math_verify
    except ImportError as error:
        raise OODError("cannot import Math-Verify from the pinned checkout") from error
    module_path = Path(math_verify.__file__).resolve()
    if not module_path.is_relative_to(root.resolve()):
        raise OODError("Math-Verify import does not resolve to the pinned checkout")
    return {"revision": result.stdout.strip(), "antlr_version": antlr, "module": str(module_path)}


def _score_math(gold: str, response: str) -> tuple[bool, str]:
    extracted = _last_boxed(response)
    if extracted is None:
        return False, "missing_boxed"
    try:
        from math_verify import LatexExtractionConfig, parse, verify

        config = [LatexExtractionConfig(boxed_match_priority=0)]
        gold_parsed = parse(f"\\boxed{{{gold}}}", extraction_config=config, fallback_mode="no_fallback")
        predicted = parse(f"\\boxed{{{extracted}}}", extraction_config=config, fallback_mode="no_fallback")
        if not gold_parsed or not predicted:
            return False, "parse_failed"
        return bool(verify(gold_parsed, predicted)), "parsed"
    except Exception:
        return False, "verifier_error"


def _request(api_base: str, served_name: str, item: Item, seed: int, generation: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content": item.prompt}],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "max_tokens": generation["max_tokens"],
        "seed": seed,
        "n": 1,
    }
    started = time.monotonic()
    try:
        response = baseline._http_json(f"{api_base}/chat/completions", generation["request_timeout_seconds"], payload)
        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise OODError("completion response must contain exactly one choice")
        choice = choices[0]
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise OODError("completion content is empty")
        usage = response.get("usage", {})
        finish_reason = choice.get("finish_reason", "unknown")
        if not isinstance(usage, Mapping) or not isinstance(finish_reason, str):
            raise OODError("completion metadata is invalid")
        for name in ("completion_tokens", "prompt_tokens"):
            value = usage.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise OODError(f"completion {name} is invalid")
        return {
            "response": content,
            "finish_reason": finish_reason,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "latency_seconds": time.monotonic() - started,
        }
    except (KeyError, IndexError, TypeError, baseline.EvaluationError) as error:
        raise OODError(f"invalid completion response: {error}") from error


def _resume(
    path: Path,
    identity: str,
    benchmark: str,
    tasks: Mapping[tuple[str, int], tuple[Item, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        path.write_bytes(data)
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            row = json.loads(line)
            key = (row["item_hash"], row["completion_index"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise OODError(f"invalid partial record at line {line_number}") from error
        expected = tasks.get(key)
        if (
            set(row) != RAW_KEYS
            or row.get("resume_identity") != identity
            or row.get("benchmark") != benchmark
            or expected is None
            or key in completed
        ):
            raise OODError(f"partial identity mismatch at line {line_number}")
        item, seed = expected
        response = row.get("response")
        if not isinstance(response, str):
            raise OODError(f"partial evidence mismatch at line {line_number}")
        correct, parser_state, parsed = _score_row(benchmark, item, row)
        if (
            row.get("request_seed") != seed
            or row.get("prompt") != item.prompt
            or row.get("permutation") != (list(item.permutation) if item.permutation else None)
            or hashlib.sha256(response.encode()).hexdigest() != row.get("response_sha256")
            or row.get("correct") is not correct
            or row.get("parser_state") != parser_state
            or row.get("parsed_answer") != parsed
        ):
            raise OODError(f"partial evidence mismatch at line {line_number}")
        completed[key] = row
    return completed


def _score_row(name: str, item: Item, generated: Mapping[str, Any]) -> tuple[bool, str, str | None]:
    response = generated["response"]
    if name == "math_500":
        correct, state = _score_math(item.answer, response)
        return correct, state, None
    parsed = _last_line_answer(response, name)
    return parsed == item.answer, "parsed" if parsed else "malformed", parsed


def _run_generations(
    name: str,
    items: Sequence[Item],
    seeds: Sequence[int],
    api_base: str,
    served_name: str,
    generation: Mapping[str, Any],
    partial: Path,
    resume_identity: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    tasks = [(item, index, seed) for item in items for index, seed in enumerate(seeds)]
    expected = {(item.item_hash, index): (item, seed) for item, index, seed in tasks}
    completed = _resume(partial, resume_identity, name, expected)
    remaining = iter(task for task in tasks if (task[0].item_hash, task[1]) not in completed)
    with partial.open("a", encoding="utf-8") as handle, concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        active: dict[concurrent.futures.Future[dict[str, Any]], tuple[Item, int, int]] = {}
        for _ in range(concurrency):
            try:
                item, index, seed = next(remaining)
            except StopIteration:
                break
            active[pool.submit(_request, api_base, served_name, item, seed, generation)] = (item, index, seed)
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                item, index, seed = active.pop(future)
                generated = future.result()
                correct, parser_state, parsed = _score_row(name, item, generated)
                row = {
                    "resume_identity": resume_identity,
                    "benchmark": name,
                    "item_hash": item.item_hash,
                    "completion_index": index,
                    "request_seed": seed,
                    "prompt": item.prompt,
                    "response": generated["response"],
                    "response_sha256": hashlib.sha256(generated["response"].encode()).hexdigest(),
                    "finish_reason": generated["finish_reason"],
                    "completion_tokens": generated["completion_tokens"],
                    "prompt_tokens": generated["prompt_tokens"],
                    "latency_seconds": generated["latency_seconds"],
                    "parser_state": parser_state,
                    "parsed_answer": parsed,
                    "correct": correct,
                    "permutation": item.permutation,
                }
                completed[(item.item_hash, index)] = row
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                try:
                    next_item, next_index, next_seed = next(remaining)
                except StopIteration:
                    continue
                active[pool.submit(_request, api_base, served_name, next_item, next_seed, generation)] = (
                    next_item,
                    next_index,
                    next_seed,
                )
    if len(completed) != len(tasks):
        raise OODError("completion set is incomplete")
    return [completed[(item.item_hash, index)] for item in items for index in range(len(seeds))]


def _metrics(name: str, rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    by_completion = []
    for index, seed in enumerate(seeds):
        selected = [row for row in rows if row["completion_index"] == index]
        value = sum(bool(row["correct"]) for row in selected)
        by_completion.append(
            {
                "completion_index": index,
                "seed": seed,
                "correct": value,
                "total": len(selected),
                "accuracy": value / len(selected),
            }
        )
    return {
        "schema_version": 1,
        "benchmark": name,
        "correct": correct,
        "total": total,
        "micro_accuracy": correct / total,
        "per_completion": by_completion,
        "malformed": sum(row["parser_state"] != "parsed" for row in rows),
        "truncated": sum(row["finish_reason"] == "length" for row in rows),
    }


def _seal(output: Path, excluded: set[str]) -> dict[str, Any]:
    files = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in excluded:
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {"schema_version": 1, "algorithm": "sha256", "files": files}
    _write_json(output / "sha256_manifest.json", manifest)
    return manifest


def _verify_seal(output: Path) -> None:
    try:
        manifest = json.loads((output / "sha256_manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise OODError(f"invalid sealed OOD manifest: {error}") from error
    if (
        manifest.get("schema_version") != 1
        or manifest.get("algorithm") != "sha256"
        or not isinstance(files, list)
        or len(files) not in {4, 5}
    ):
        raise OODError("invalid sealed OOD manifest")
    required = {"command.json", "metrics.json", "records.jsonl", "resume_identity.json"}
    expected = {entry.get("path") for entry in files if isinstance(entry, dict)}
    if (
        any(
            not isinstance(entry, dict)
            or set(entry) != {"path", "bytes", "sha256"}
            or not isinstance(entry["bytes"], int)
            or isinstance(entry["bytes"], bool)
            or entry["bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            for entry in files
        )
        or not required.issubset(expected)
        or expected - required not in (set(), {"rescore.json"})
    ):
        raise OODError("sealed OOD file set is incomplete")
    allowed = expected | {"sha256_manifest.json"}
    entries = list(output.iterdir())
    actual = {path.name for path in entries}
    if actual not in (allowed, allowed | {".lock"}) or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise OODError("sealed OOD directory contents differ from the manifest")
    for entry in files:
        path = output / entry["path"]
        if not path.is_file() or path.stat().st_size != entry.get("bytes") or _sha256(path) != entry.get("sha256"):
            raise OODError(f"sealed OOD artifact mismatch: {entry.get('path')}")


def _safe_argv(argv: Sequence[str]) -> list[str]:
    safe = list(argv)
    redact_next = False
    for index, argument in enumerate(safe):
        if redact_next:
            safe[index] = "<redacted>"
            redact_next = False
        elif argument in {"--api-base", "--api-key", "--token", "--password"}:
            redact_next = True
        elif any(argument.startswith(f"{flag}=") for flag in ("--api-base", "--api-key", "--token", "--password")):
            safe[index] = argument.split("=", 1)[0] + "=<redacted>"
    return safe


def _rescore_rows(
    source: Path,
    items: Sequence[Item],
    seeds: Sequence[int],
    identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    expected = [(item, index, seed) for item in items for index, seed in enumerate(seeds)]
    rows = _jsonl(source / "records.jsonl")
    if len(rows) != len(expected):
        raise OODError("source MATH-500 completion set is incomplete")
    identity_hash = identity.get("sha256")
    rescored = []
    changed = 0
    for line_number, (row, task) in enumerate(zip(rows, expected, strict=True), 1):
        item, index, seed = task
        response = row.get("response")
        if (
            set(row) != RAW_KEYS
            or row.get("resume_identity") != identity_hash
            or row.get("benchmark") != "math_500"
            or row.get("item_hash") != item.item_hash
            or row.get("completion_index") != index
            or row.get("request_seed") != seed
            or row.get("prompt") != item.prompt
            or row.get("permutation") is not None
            or not isinstance(response, str)
            or hashlib.sha256(response.encode()).hexdigest() != row.get("response_sha256")
        ):
            raise OODError(f"source MATH-500 identity mismatch at line {line_number}")
        correct, state = _score_math(item.answer, response)
        updated = dict(row)
        updated.update({"correct": correct, "parser_state": state, "parsed_answer": None})
        changed += updated != row
        rescored.append(updated)
    return rescored, changed


def _verify_rescore_provenance(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OODError(f"invalid MATH-500 rescore provenance: {error}") from error
    keys = {*expected, "argv"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or any(value.get(key) != item for key, item in expected.items())
        or not isinstance(value.get("argv"), list)
        or not all(isinstance(item, str) for item in value["argv"])
    ):
        raise OODError("MATH-500 rescore provenance differs from the current source or scorer")


def rescore_math_artifact(
    *,
    source_dir: Path,
    output_dir: Path,
    data_path: Path,
    math_verify_root: Path,
    config_path: Path = CONFIG,
    gpqa_access_path: Path = GPQA_ACCESS,
    argv: Sequence[str] = (),
) -> Path:
    """Rescore a sealed MATH-500 artifact without issuing model requests."""

    source = source_dir.resolve()
    output = output_dir.resolve()
    if source == output or output.is_relative_to(ROOT.resolve()):
        raise OODError("rescored OOD output must use a new directory outside the Git checkout")
    _verify_seal(source)
    config = _load_config(config_path, gpqa_access_path)
    benchmark = config["benchmarks"]["math_500"]
    scorer = _verify_math_runtime(math_verify_root, benchmark)
    items = _load_items("math_500", data_path, benchmark, config["prompts"])
    try:
        identity = json.loads((source / "resume_identity.json").read_text(encoding="utf-8"))
        identity_value = identity["identity"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise OODError(f"invalid source MATH-500 identity: {error}") from error
    if (
        identity_value.get("benchmark") != "math_500"
        or identity_value.get("dataset_revision") != benchmark["revision"]
        or identity_value.get("dataset_sha256") != benchmark["sha256"]
        or identity_value.get("protocol_sha256") != _json_hash(config["protocol"])
    ):
        raise OODError("source MATH-500 identity differs from the frozen protocol")
    rows, changed = _rescore_rows(source, items, config["protocol"]["completion_seeds"], identity)
    source_manifest_hash = _sha256(source / "sha256_manifest.json")
    source_records_hash = _sha256(source / "records.jsonl")
    provenance = {
        "schema_version": 1,
        "correction": "boxed_latex_parse_v1",
        "source_manifest_sha256": source_manifest_hash,
        "source_records_sha256": source_records_hash,
        "dataset_sha256": benchmark["sha256"],
        "scorer": scorer,
        "harness_sha256": _sha256(Path(__file__).resolve()),
        "changed_rows": changed,
        "total": len(rows),
    }

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OODError("another process owns the rescored OOD directory") from error
        if (output / "sha256_manifest.json").exists():
            _verify_seal(output)
            _verify_rescore_provenance(output / "rescore.json", provenance)
            return output
        if set(path.name for path in output.iterdir()) != {".lock"}:
            raise OODError("rescored OOD output directory must be empty")
        shutil.copyfile(source / "command.json", output / "command.json")
        shutil.copyfile(source / "resume_identity.json", output / "resume_identity.json")
        (output / "records.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_json(output / "metrics.json", _metrics("math_500", rows, config["protocol"]["completion_seeds"]))
        _write_json(
            output / "rescore.json",
            {
                **provenance,
                "argv": _safe_argv(argv),
            },
        )
        _seal(output, {".lock", "sha256_manifest.json"})
        _verify_seal(output)
    return output


def run_evaluation(
    *,
    benchmark: str,
    data_path: Path,
    api_base: str,
    server_manifest: Path,
    output_dir: Path,
    concurrency: int,
    config_path: Path = CONFIG,
    math_verify_root: Path | None = None,
    gpqa_access_path: Path = GPQA_ACCESS,
    argv: Sequence[str] = (),
) -> Path:
    """Run or resume one sealed OOD benchmark evaluation."""

    if benchmark not in EXPECTED or concurrency < 1:
        raise OODError("invalid benchmark or concurrency")
    output = output_dir.resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise OODError("raw OOD outputs must remain outside the Git checkout")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OODError("another process owns the OOD output directory") from error
        config = _load_config(config_path, gpqa_access_path)
        benchmark_config = config["benchmarks"][benchmark]
        if benchmark == "gpqa" and benchmark_config["execution_state"] != "runnable_frozen_access":
            raise OODError("GPQA is blocked until authenticated bytes, revision, and SHA-256 are frozen")
        api = baseline._api_base(api_base)
        server, server_hash, served_name = _load_server(
            server_manifest, api, config["protocol"]["generation"]["request_timeout_seconds"]
        )
        items = _load_items(benchmark, data_path, benchmark_config, config["prompts"])
        scorer = (
            _verify_math_runtime(math_verify_root, benchmark_config)
            if benchmark == "math_500"
            else {
                "name": "strict_last_nonempty_line",
                "pattern": MC_PATTERN[benchmark].pattern,
            }
        )
        harness = {path.name: _sha256(path) for path in (Path(__file__).resolve(), ROOT / "scripts/run_ood_eval.py")}
        identity_value = {
            "benchmark": benchmark,
            "dataset_revision": benchmark_config["revision"],
            "dataset_sha256": benchmark_config["sha256"],
            "server_sha256": server_hash,
            "model": server["model"],
            "protocol_sha256": _json_hash(config["protocol"]),
            "config_sha256": _sha256(config_path),
            "renderer_sha256": _json_hash(config["prompts"]),
            "harness": harness,
            "scorer": scorer,
        }
        resume_identity = _json_hash(identity_value)
        identity_path = output / "resume_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != {
            "sha256": resume_identity,
            "identity": identity_value,
        }:
            raise OODError("output resume identity does not match this run")
        _write_json(identity_path, {"sha256": resume_identity, "identity": identity_value})
        if (output / "sha256_manifest.json").exists():
            _verify_seal(output)
            return output
        partial = output / "partial_records.jsonl"
        rows = _run_generations(
            benchmark,
            items,
            config["protocol"]["completion_seeds"],
            api,
            served_name,
            config["protocol"]["generation"],
            partial,
            resume_identity,
            concurrency,
        )
        records = output / "records.jsonl"
        records.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        compact = _metrics(benchmark, rows, config["protocol"]["completion_seeds"])
        _write_json(output / "metrics.json", compact)
        _write_json(output / "command.json", {"argv": _safe_argv(argv)})
        partial.unlink()
        _seal(output, {".lock", "sha256_manifest.json"})
        _verify_seal(output)
        return output
