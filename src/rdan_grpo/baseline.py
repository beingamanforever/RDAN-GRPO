"""Fail-closed base-model evaluation through vLLM and native RTT scorers."""

from __future__ import annotations

import concurrent.futures
import copy
import fcntl
import hashlib
import importlib.metadata
import ipaddress
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from rdan_grpo import baseline_models

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "configs").is_dir():
    ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/baselines/qwen_base.json"
MAX_HTTP_BYTES = 16 * 1024 * 1024
PINNED_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PINNED_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PINNED_RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
MULDIMIF_PER_EXAMPLE_CODE = """
import json
import sys

from evaluation.evaluation import check, pre_process
from utils.data_utils import load_data

rows = check(pre_process(load_data(sys.argv[1]), "auto"))
with open(sys.argv[2], "w", encoding="utf-8") as output:
    for index, row in enumerate(rows):
        judges = row["judges"]
        value = {"index": index, "id": row["id"], "judges": judges, "passed": all(judges)}
        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\\n")
"""


class EvaluationError(RuntimeError):
    """Raised when an evaluation invariant fails."""


@dataclass(frozen=True)
class Item:
    identity: str
    prompt: str
    messages: list[dict[str, str]]
    source: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
        if config["schema_version"] != 1 or set(config["benchmarks"]) != {"ifeval", "ifbench", "muldimif"}:
            raise EvaluationError("unsupported baseline configuration")
        if config["rtt"]["revision"] != PINNED_RTT_REVISION:
            raise EvaluationError("model or RTT pin does not match the base-evaluation contract")
        baseline_models.load_model_contract(config)
        generation = config["generation"]
        expected = {
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "max_tokens": 4096,
            "seed": 42,
            "n": 1,
        }
        if any(generation.get(key) != value for key, value in expected.items()):
            raise EvaluationError("generation settings do not match the pinned greedy contract")
        return config
    except (KeyError, TypeError, json.JSONDecodeError, baseline_models.ModelContractError) as error:
        raise EvaluationError(f"invalid baseline configuration: {error}") from error


def _run_git(rtt_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(rtt_root), *args], capture_output=True, text=True, check=False, timeout=30
    )
    if result.returncode:
        raise EvaluationError(f"RTT git verification failed: git {' '.join(args)} exited {result.returncode}")
    return result.stdout.strip()


def _verify_rtt(rtt_root: Path, expected_revision: str) -> dict[str, str]:
    root = rtt_root.resolve()
    if not root.is_dir():
        raise EvaluationError(f"RTT root is not a directory: {root}")
    revision = _run_git(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise EvaluationError(f"RTT revision mismatch: expected {expected_revision}, found {revision}")
    if _run_git(root, "status", "--porcelain"):
        raise EvaluationError("RTT checkout is not clean")
    return {"revision": revision, "tree": _run_git(root, "rev-parse", "HEAD^{tree}")}


def _require_string(record: dict[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{label} has invalid {key}")
    return value


def _load_items(benchmark: str, path: Path, expected_records: int, expected_hash: str) -> tuple[list[Item], int]:
    size, actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise EvaluationError(f"benchmark input hash mismatch: expected {expected_hash}, found {actual_hash}")
    try:
        if benchmark == "muldimif":
            with path.open(encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                raise EvaluationError("MulDimIF input must be a JSON list")
        else:
            records = []
            with path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        raise EvaluationError(f"blank benchmark record at line {line_number}")
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot parse benchmark input: {error}") from error
    if len(records) != expected_records:
        raise EvaluationError(f"benchmark record mismatch: expected {expected_records}, found {len(records)}")

    items: list[Item] = []
    identities: set[str] = set()
    prompts: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise EvaluationError(f"benchmark record {index} is not an object")
        identity_key = "id" if benchmark == "muldimif" else "key"
        if identity_key not in record or isinstance(record[identity_key], (dict, list, bool)):
            raise EvaluationError(f"benchmark record {index} has invalid {identity_key}")
        identity = json.dumps(record[identity_key], ensure_ascii=False, sort_keys=True)
        if identity in identities:
            raise EvaluationError(f"duplicate prompt identity: {identity}")
        identities.add(identity)

        if benchmark == "muldimif":
            conversations = record.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                raise EvaluationError(f"MulDimIF record {index} has invalid conversations")
            messages = []
            for message in conversations:
                if not isinstance(message, dict):
                    raise EvaluationError(f"MulDimIF record {index} has invalid conversation")
                role = _require_string(message, "role", f"MulDimIF record {index}")
                content = _require_string(message, "content", f"MulDimIF record {index}")
                messages.append({"role": role, "content": content})
            if messages[-1]["role"] != "user":
                raise EvaluationError(f"MulDimIF record {index} does not end with a user message")
            prompt = messages[-1]["content"]
        else:
            prompt = _require_string(record, "prompt", f"benchmark record {index}")
            messages = [{"role": "user", "content": prompt}]
        if prompt in prompts:
            raise EvaluationError(f"duplicate prompt text at record {index}")
        prompts.add(prompt)
        items.append(Item(identity=identity, prompt=prompt, messages=messages, source=record))
    return items, size


def _api_base(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise EvaluationError("api base must be an HTTP URL without credentials")
    if parts.query or parts.fragment or parts.path.rstrip("/") not in {"", "/v1"}:
        raise EvaluationError("api base path must be empty or /v1 and cannot contain a query or fragment")
    return urlunsplit((parts.scheme, parts.netloc, "/v1", "", ""))


def _http_json(url: str, timeout: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST"
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise EvaluationError(f"HTTP request failed with status {response.status}")
            body = response.read(MAX_HTTP_BYTES + 1)
    except HTTPError as error:
        raise EvaluationError(f"HTTP request failed with status {error.code}") from error
    except (TimeoutError, URLError, OSError) as error:
        raise EvaluationError(f"HTTP request failed: {error}") from error
    if len(body) > MAX_HTTP_BYTES:
        raise EvaluationError("HTTP response exceeded the size limit")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError("HTTP response is not valid JSON") from error
    if not isinstance(value, dict):
        raise EvaluationError("HTTP response is not a JSON object")
    return value


def _argument(argv: list[str], name: str) -> str:
    values = []
    for index, argument in enumerate(argv):
        if argument == name and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif argument.startswith(f"{name}="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise EvaluationError(f"server argv must contain exactly one {name}")
    return values[0]


def _reject_sensitive_argv(argv: list[str]) -> None:
    secret_flags = ("--api-key", "--hf-token", "--password", "--ssl-keyfile")
    for index, argument in enumerate(argv):
        if argument == "--host":
            if index + 1 >= len(argv) or argv[index + 1] != "localhost":
                raise EvaluationError("server host must be localhost")
        elif argument.startswith("--host=") and argument != "--host=localhost":
            raise EvaluationError("server host must be localhost")
        if any(argument == flag or argument.startswith(f"{flag}=") for flag in secret_flags):
            raise EvaluationError("server argv must not contain credentials or private key paths")
        candidate = argument.split("=", 1)[-1]
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            parts = urlsplit(candidate)
            if parts.hostname:
                try:
                    ipaddress.ip_address(parts.hostname)
                except ValueError:
                    continue
                raise EvaluationError(f"server argv contains a host IP at argument {index}")
        else:
            raise EvaluationError(f"server argv contains a host IP at argument {index}")


def _required_snapshot_roles(path: Path) -> set[str]:
    name = path.name.lower()
    roles = set()
    if name in {"config.json", "generation_config.json"} or name.endswith(
        (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json", ".pt", ".pth")
    ):
        roles.add("model")
    if (
        name.startswith(("tokenizer", "vocab"))
        or name in {"merges.txt", "special_tokens_map.json", "added_tokens.json"}
        or name.endswith(".model")
    ):
        roles.add("tokenizer")
    if "chat_template" in name or name.endswith(".jinja"):
        roles.add("chat_template")
    return roles


def _resolve_snapshot_file(path: Path, snapshot: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        cache_root = snapshot.parent.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise EvaluationError(f"snapshot symlink is invalid: {path.relative_to(snapshot)}") from error
    if path.is_symlink() and not resolved.is_relative_to(cache_root):
        raise EvaluationError(f"snapshot symlink escapes the model cache: {path.relative_to(snapshot)}")
    return resolved


def _load_server_manifest(path: Path, expected_model: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid server identity manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "created_at", "model", "files", "server"}
        or manifest.get("schema_version") != 1
    ):
        raise EvaluationError("unsupported server identity manifest")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"])
        model = manifest["model"]
        snapshot_path = Path(model["snapshot_path"])
        files = manifest["files"]
        server = manifest["server"]
        argv = server["argv"]
        packages = server["packages"]
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"invalid server identity manifest: {error}") from error
    if created_at.tzinfo is None:
        raise EvaluationError("server identity creation timestamp must include a timezone")
    if (
        not isinstance(model, dict)
        or set(model) != {"name", "revision", "snapshot_commit", "snapshot_path"}
        or model.get("name") != expected_model["name"]
        or model.get("revision") != expected_model["revision"]
        or model.get("snapshot_commit") != expected_model["revision"]
        or not snapshot_path.is_absolute()
        or snapshot_path.resolve() != snapshot_path
        or snapshot_path.name != expected_model["revision"]
        or not snapshot_path.is_dir()
    ):
        raise EvaluationError("server model snapshot does not match the pinned checkpoint")
    if not isinstance(server, dict) or set(server) != {"argv", "packages"}:
        raise EvaluationError("server identity fields are invalid")
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) and value for value in argv):
        raise EvaluationError("server argv must be a non-empty string list")
    _reject_sensitive_argv(argv)
    if any(argument == "--chat-template" or argument.startswith("--chat-template=") for argument in argv):
        raise EvaluationError("server argv must use the pinned tokenizer chat template")
    if any(argument == "--tokenizer" or argument.startswith("--tokenizer=") for argument in argv):
        raise EvaluationError("server argv must not override the pinned tokenizer")
    model_argument = Path(_argument(argv, "--model"))
    if not model_argument.is_absolute() or model_argument.resolve() != snapshot_path:
        raise EvaluationError("server argv model path does not match the resolved snapshot")
    if _argument(argv, "--served-model-name") != expected_model["served_name"]:
        raise EvaluationError("server argv served name does not match the pinned alias")
    if (
        not isinstance(packages, dict)
        or set(packages) != {"vllm", "transformers", "torch"}
        or any(
            not isinstance(packages.get(name), str) or not packages[name] for name in ("vllm", "transformers", "torch")
        )
    ):
        raise EvaluationError("server package identities are incomplete")
    if not isinstance(files, list) or not files:
        raise EvaluationError("server model file identities are missing")
    covered_roles: set[str] = set()
    seen_paths: set[str] = set()
    declared_roles: dict[str, set[str]] = {}
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise EvaluationError(f"invalid server model file identity at index {index}")
        relative = entry.get("path")
        roles = entry.get("roles")
        if (
            set(entry) != {"path", "roles", "bytes", "sha256"}
            or not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative in seen_paths
            or not isinstance(roles, list)
            or not roles
            or not all(isinstance(role, str) for role in roles)
            or not set(roles) <= {"model", "tokenizer", "chat_template"}
            or not isinstance(entry.get("bytes"), int)
            or isinstance(entry.get("bytes"), bool)
            or entry["bytes"] < 0
            or not isinstance(entry.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise EvaluationError(f"invalid server model file identity at index {index}")
        seen_paths.add(relative)
        declared_roles[relative] = set(roles)
        file_path = snapshot_path.joinpath(*PurePosixPath(relative).parts)
        if not file_path.exists() and not file_path.is_symlink():
            raise EvaluationError(f"server model file is missing: {relative}")
        resolved_file = _resolve_snapshot_file(file_path, snapshot_path)
        if not resolved_file.is_file():
            raise EvaluationError(f"server model file is missing: {relative}")
        size, digest = _sha256(file_path)
        if entry.get("bytes") != size or entry.get("sha256") != digest:
            raise EvaluationError(f"server model file identity mismatch: {relative}")
        if "chat_template" in roles and "chat_template" not in file_path.name and file_path.suffix != ".jinja":
            try:
                template = json.loads(file_path.read_text(encoding="utf-8")).get("chat_template")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
                raise EvaluationError(f"invalid chat template identity: {relative}") from error
            if not isinstance(template, str) or not template:
                raise EvaluationError(f"invalid chat template identity: {relative}")
        covered_roles.update(roles)
    if covered_roles != {"model", "tokenizer", "chat_template"}:
        raise EvaluationError("server identities must cover model, tokenizer, and chat template files")
    for file_path in snapshot_path.rglob("*"):
        if file_path.is_symlink():
            _resolve_snapshot_file(file_path, snapshot_path)
        if file_path.is_file():
            relative = file_path.relative_to(snapshot_path).as_posix()
            required_roles = _required_snapshot_roles(file_path)
            if required_roles and not required_roles <= declared_roles.get(relative, set()):
                raise EvaluationError(f"server identity omits required snapshot file roles: {relative}")
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def _verify_model(api_base: str, served_name: str, snapshot_path: Path, timeout: int) -> None:
    response = _http_json(f"{api_base}/models", timeout)
    models = response.get("data")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise EvaluationError("/v1/models must expose exactly one model")
    if models[0].get("id") != served_name:
        raise EvaluationError(f"served model mismatch: expected {served_name}, found {models[0].get('id')!r}")
    root = models[0].get("root")
    if not isinstance(root, str) or not Path(root).is_absolute() or Path(root).resolve() != snapshot_path:
        raise EvaluationError("served model checkpoint does not match the server identity manifest")


def _generate_one(item: Item, api_base: str, model: str, generation: dict[str, Any]) -> str:
    payload = {
        "model": model,
        "messages": item.messages,
        **{key: generation[key] for key in ("temperature", "top_p", "top_k", "max_tokens", "seed", "n")},
        "chat_template_kwargs": generation["chat_template_kwargs"],
    }
    response = _http_json(f"{api_base}/chat/completions", generation["request_timeout_seconds"], payload)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise EvaluationError("completion response must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise EvaluationError("completion choice has no message object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise EvaluationError("completion response is empty")
    return content


def _generate(
    items: list[Item],
    api_base: str,
    model: str,
    generation: dict[str, Any],
    concurrency: int,
    partial_path: Path,
    completed: dict[int, str],
) -> list[str]:
    responses = dict(completed)
    remaining = iter(index for index in range(len(items)) if index not in responses)
    failures: list[tuple[int, BaseException]] = []
    with (
        partial_path.open("a", encoding="utf-8") as partial,
        concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        active: dict[concurrent.futures.Future[str], int] = {}
        for index in itertools.islice(remaining, concurrency):
            active[executor.submit(_generate_one, items[index], api_base, model, generation)] = index
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                try:
                    response = future.result()
                except BaseException as error:
                    failures.append((index, error))
                    continue
                responses[index] = response
                partial.write(
                    json.dumps(
                        {"index": index, "identity": items[index].identity, "response": response},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                partial.flush()
                os.fsync(partial.fileno())
            if not failures:
                while len(active) < concurrency:
                    try:
                        index = next(remaining)
                    except StopIteration:
                        break
                    active[executor.submit(_generate_one, items[index], api_base, model, generation)] = index
    if failures:
        index, error = min(failures, key=lambda value: value[0])
        raise EvaluationError(f"generation failed at record {index}: {error}") from error
    if len(responses) != len(items):
        raise EvaluationError(f"generation count mismatch: expected {len(items)}, found {len(responses)}")
    return [responses[index] for index in range(len(items))]


def _load_completed(path: Path, items: list[Item]) -> dict[int, str]:
    if not path.exists():
        return {}
    completed: dict[int, str] = {}
    try:
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
            path.write_bytes(data)
        for line_number, encoded in enumerate(data.splitlines(), 1):
            line = encoded.decode()
            if not line.strip():
                raise EvaluationError(f"blank partial generation record at line {line_number}")
            row = json.loads(line)
            index = row.get("index") if isinstance(row, dict) else None
            if (
                not isinstance(row, dict)
                or set(row) != {"index", "identity", "response"}
                or not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(items)
                or index in completed
                or row.get("identity") != items[index].identity
                or not isinstance(row.get("response"), str)
                or not row["response"].strip()
            ):
                raise EvaluationError(f"invalid partial generation record at line {line_number}")
            completed[index] = row["response"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid partial generation file: {error}") from error
    return completed


def _generation_rows(benchmark: str, items: list[Item], responses: list[str]) -> list[dict[str, Any]]:
    rows = []
    for item, response in zip(items, responses):
        if benchmark == "muldimif":
            row = copy.deepcopy(item.source)
            row["conversations"] = [*row["conversations"], {"role": "assistant", "content": response}]
        else:
            row = {"prompt": item.prompt, "response": response}
        rows.append(row)
    return rows


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(value for value in (str(path.resolve()), inherited) if value)


def _scorer_command(
    benchmark: str, rtt_root: Path, generation_path: Path, native_dir: Path
) -> tuple[list[str], Path, dict[str, str]]:
    input_path = (
        rtt_root
        / {
            "ifeval": "Benchmark/instruction_following_eval/data/input_data.jsonl",
            "ifbench": "Benchmark/IFBench/data/IFBench_test.jsonl",
            "muldimif": "Benchmark/MulDimIF/Data/test.json",
        }[benchmark]
    )
    env = os.environ.copy()
    if benchmark == "ifeval":
        cwd = rtt_root / "Benchmark"
        command = [
            sys.executable,
            "-m",
            "instruction_following_eval.evaluation_main",
            f"--input_data={input_path}",
            f"--input_response_data={generation_path}",
            f"--output_dir={native_dir}",
        ]
    elif benchmark == "ifbench":
        cwd = rtt_root / "Benchmark/IFBench"
        command = [
            sys.executable,
            "run_eval.py",
            f"--input_data={input_path}",
            f"--input_response_data={generation_path}",
            f"--output_dir={native_dir}",
        ]
    else:
        cwd = rtt_root / "Benchmark/MulDimIF"
        _prepend_pythonpath(env, cwd / "Code")
        command = [
            sys.executable,
            "Code/evaluation/evaluation.py",
            f"--file_path={generation_path}",
            f"--save_path={native_dir / 'breakdown.json'}",
        ]
    return command, cwd, env


def _run_scorer(
    benchmark: str, rtt_root: Path, generation_path: Path, output: Path, timeout: int
) -> tuple[list[str], Path]:
    native_dir = output / "native"
    native_dir.mkdir()
    command, cwd, env = _scorer_command(benchmark, rtt_root, generation_path, native_dir)
    timed_out = False
    try:
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=timeout
        )
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
        exit_code = None
    (output / "scorer_stdout.txt").write_text(stdout, encoding="utf-8")
    (output / "scorer_stderr.txt").write_text(stderr, encoding="utf-8")
    _write_json(
        output / "scorer.json",
        {"command": command, "cwd": str(cwd), "exit_code": exit_code, "timed_out": timed_out},
    )
    if timed_out:
        raise EvaluationError(f"native {benchmark} scorer timed out")
    if exit_code:
        raise EvaluationError(f"native {benchmark} scorer exited {exit_code}")
    return command, cwd


def _derive_muldimif_per_example(rtt_root: Path, generation_path: Path, output: Path, timeout: int) -> None:
    native_path = output / "native/per_example.jsonl"
    env = os.environ.copy()
    _prepend_pythonpath(env, rtt_root / "Benchmark/MulDimIF/Code")
    command = [sys.executable, "-c", MULDIMIF_PER_EXAMPLE_CODE, str(generation_path), str(native_path)]
    try:
        result = subprocess.run(
            command,
            cwd=rtt_root / "Benchmark/MulDimIF",
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise EvaluationError("MulDimIF per-example derivation timed out") from error
    if result.returncode:
        raise EvaluationError(f"MulDimIF per-example derivation exited {result.returncode}: {result.stderr.strip()}")


def _read_jsonl(path: Path, expected_records: int) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as file:
            rows = []
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    raise EvaluationError(f"blank native scorer result at line {line_number} in {path.name}")
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid native scorer output {path.name}: {error}") from error
    if len(rows) != expected_records or not all(isinstance(row, dict) for row in rows):
        raise EvaluationError(f"invalid native scorer record count in {path.name}: {len(rows)}")
    return rows


def _if_metrics(output: Path, items: list[Item], responses: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"schema_version": 1, "records": len(items)}
    for mode in ("strict", "loose"):
        rows = _read_jsonl(output / "native" / f"eval_results_{mode}.jsonl", len(items))
        instruction_correct = 0
        rubric_correct = 0
        rubric_total = 0
        for index, (row, item, response) in enumerate(zip(rows, items, responses)):
            followed = row.get("follow_instruction_list")
            instruction_ids = row.get("instruction_id_list")
            if (
                row.get("prompt") != item.prompt
                or row.get("response") != response
                or not isinstance(followed, list)
                or not followed
                or not all(isinstance(value, bool) for value in followed)
                or not isinstance(instruction_ids, list)
                or instruction_ids != item.source.get("instruction_id_list")
                or len(followed) != len(instruction_ids)
                or row.get("follow_all_instructions") != all(followed)
            ):
                raise EvaluationError(f"invalid native {mode} result at record {index}")
            instruction_correct += int(all(followed))
            rubric_correct += sum(followed)
            rubric_total += len(followed)
        metrics[mode] = {
            "instruction_level": {
                "correct": instruction_correct,
                "total": len(items),
                "accuracy": instruction_correct / len(items),
            },
            "rubric_level": {
                "correct": rubric_correct,
                "total": rubric_total,
                "accuracy": rubric_correct / rubric_total,
            },
        }
    return metrics


def _muldimif_metrics(output: Path, items: list[Item]) -> dict[str, Any]:
    expected_records = len(items)
    path = output / "native/breakdown.json"
    try:
        with path.open(encoding="utf-8") as file:
            breakdown = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid MulDimIF breakdown: {error}") from error
    if not isinstance(breakdown, dict) or not isinstance(breakdown.get("Overall"), str):
        raise EvaluationError("MulDimIF breakdown has no Overall result")
    match = re.fullmatch(r"(\d+)/(\d+)=([0-9]+(?:\.[0-9]+)?)", breakdown["Overall"])
    if not match:
        raise EvaluationError("MulDimIF Overall result has an invalid schema")
    correct, total, accuracy = int(match[1]), int(match[2]), float(match[3])
    if total != expected_records or correct > total or abs(correct / total - accuracy) > 1e-9:
        raise EvaluationError("MulDimIF Overall result is inconsistent")
    per_example = _read_jsonl(output / "native/per_example.jsonl", expected_records)
    derived_correct = 0
    for index, (row, item) in enumerate(zip(per_example, items)):
        judges = row.get("judges")
        if (
            set(row) != {"index", "id", "judges", "passed"}
            or row.get("index") != index
            or row.get("id") != item.source.get("id")
            or not isinstance(judges, list)
            or not judges
            or not all(isinstance(judge, int) and not isinstance(judge, bool) and judge in {0, 1} for judge in judges)
            or not isinstance(row.get("passed"), bool)
            or row["passed"] != all(judges)
        ):
            raise EvaluationError(f"invalid MulDimIF per-example result at record {index}")
        derived_correct += int(row["passed"])
    if derived_correct != correct:
        raise EvaluationError(
            f"MulDimIF per-example aggregate disagrees with Overall: expected {correct}, found {derived_correct}"
        )
    return {
        "schema_version": 1,
        "records": expected_records,
        "overall": {"correct": correct, "total": total, "accuracy": accuracy, "native": breakdown["Overall"]},
    }


def _harness_identity() -> dict[str, Any]:
    files = []
    for path in (
        Path(__file__).resolve(),
        Path(baseline_models.__file__).resolve(),
        (ROOT / "scripts/run_base_eval.py").resolve(),
    ):
        size, digest = _sha256(path)
        files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": size, "sha256": digest})
    return {"algorithm": "sha256", "files": files, "sha256": _json_hash(files)}


def _environment(server_manifest_hash: str, harness: dict[str, Any]) -> dict[str, Any]:
    packages = {}
    for name in (
        "absl-py",
        "emoji",
        "immutabledict",
        "langdetect",
        "nltk",
        "pandas",
        "spacy",
        "torch",
        "transformers",
        "vllm",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        gpu = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        gpu = {"command": command, "available": False, "error": type(error).__name__}
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "gpu": gpu,
        "harness": harness,
        "server_manifest": {"sha256": server_manifest_hash},
    }


def _process_start(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, check=False, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def _owner_is_live(owner: Any) -> bool:
    if not isinstance(owner, dict) or owner.get("schema_version") != 1:
        return True
    pid = owner.get("pid")
    expected_start = owner.get("process_start")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not isinstance(expected_start, str):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    actual_start = _process_start(pid)
    return actual_start is None or actual_start == expected_start


def _acquire_lock(path: Path) -> tuple[int, str]:
    token = uuid.uuid4().hex
    owner = {
        "schema_version": 1,
        "pid": os.getpid(),
        "process_start": _process_start(os.getpid()) or "unavailable",
        "created_at": _now(),
        "owner_token": token,
    }
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                file = path.open("r+", encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError as error:
                raise EvaluationError(f"output lock is unreadable and cannot be reclaimed: {path}") from error
            try:
                try:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise EvaluationError(f"another live process owns output lock: {path}") from error
                try:
                    existing = json.load(file)
                    existing_stat = os.fstat(file.fileno())
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise EvaluationError(f"output lock is unreadable and cannot be reclaimed: {path}") from error
                if _owner_is_live(existing):
                    raise EvaluationError(f"another live process owns output lock: {path}")
                try:
                    current_stat = path.stat()
                except FileNotFoundError:
                    continue
                if (current_stat.st_dev, current_stat.st_ino) == (existing_stat.st_dev, existing_stat.st_ino):
                    path.unlink(missing_ok=True)
            finally:
                file.close()
            continue
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(descriptor, (json.dumps(owner, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
        return descriptor, token


def _release_lock(path: Path, descriptor: int, token: str) -> None:
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    else:
        if isinstance(owner, dict) and owner.get("owner_token") == token:
            path.unlink(missing_ok=True)
    finally:
        os.close(descriptor)


def _safe_argv(argv: list[str] | None) -> list[str]:
    safe = []
    redact_next = False
    for argument in argv or []:
        if redact_next:
            safe.append("<local-endpoint>")
            redact_next = False
        elif argument == "--api-base":
            safe.append(argument)
            redact_next = True
        elif argument.startswith("--api-base="):
            safe.append("--api-base=<local-endpoint>")
        else:
            safe.append(argument)
    return safe


def _record_failure(partial: Path, error: BaseException) -> None:
    failures = partial / "failures"
    failures.mkdir(exist_ok=True)
    number = len(list(failures.glob("*.json"))) + 1
    record = {"failed_at": _now(), "error_type": type(error).__name__, "message": str(error)}
    _write_json(failures / f"{number:06d}.json", record)
    _write_json(partial / "failure.json", record)


def _archive_scorer_attempt(partial: Path) -> None:
    paths = [
        partial / name
        for name in ("native", "scorer.json", "scorer_stdout.txt", "scorer_stderr.txt")
        if (partial / name).exists()
    ]
    if not paths:
        return
    attempts = partial / "failed_scorer_attempts"
    attempts.mkdir(exist_ok=True)
    target = attempts / f"{len(list(attempts.iterdir())) + 1:06d}"
    target.mkdir()
    for path in paths:
        path.rename(target / path.name)


def _manifest(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise EvaluationError(f"artifact symlink is forbidden: {path.relative_to(output)}")
        if path.is_file() and path.name != "sha256_manifest.json":
            size, digest = _sha256(path)
            entries.append({"path": path.relative_to(output).as_posix(), "bytes": size, "sha256": digest})
    _write_json(
        output / "sha256_manifest.json",
        {"algorithm": "sha256", "scope": "all regular final files except sha256_manifest.json", "files": entries},
    )


def run_evaluation(
    benchmark: str,
    rtt_root: Path,
    api_base: str,
    server_manifest_path: Path,
    output_dir: Path,
    concurrency: int = 8,
    config_path: Path = CONFIG,
    argv: list[str] | None = None,
) -> Path:
    """Run one sealed base evaluation and return its final artifact directory."""
    if benchmark not in {"ifeval", "ifbench", "muldimif"}:
        raise EvaluationError(f"unsupported benchmark: {benchmark}")
    if not 1 <= concurrency <= 128:
        raise EvaluationError("concurrency must be between 1 and 128")
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.parent / f".{output_dir.name}.lock"
    lock_fd, lock_token = _acquire_lock(lock)
    partial = output_dir.parent / f".{output_dir.name}.partial"
    try:
        if output_dir.exists():
            raise EvaluationError(f"output directory already exists: {output_dir}")
        started_at = _now()
        config = _load_config(config_path)
        config_bytes, config_hash = _sha256(config_path)
        benchmark_config = config["benchmarks"][benchmark]
        normalized_api = _api_base(api_base)
        rtt_root = rtt_root.resolve()
        rtt = _verify_rtt(rtt_root, config["rtt"]["revision"])
        input_path = (rtt_root / benchmark_config["input_path"]).resolve()
        if not input_path.is_relative_to(rtt_root):
            raise EvaluationError("benchmark input escapes RTT root")
        items, input_bytes = _load_items(
            benchmark, input_path, benchmark_config["records"], benchmark_config["sha256"]
        )
        server_manifest, server_manifest_hash = _load_server_manifest(server_manifest_path, config["model"])
        snapshot_path = Path(server_manifest["model"]["snapshot_path"])
        harness = _harness_identity()
        resume_identity = {
            "schema_version": 1,
            "benchmark": benchmark,
            "concurrency": concurrency,
            "config": {
                "path": str(config_path.resolve()),
                "bytes": config_bytes,
                "sha256": config_hash,
                "resolved_sha256": _json_hash(config),
            },
            "input": {
                "path": str(input_path),
                "bytes": input_bytes,
                "records": len(items),
                "sha256": benchmark_config["sha256"],
            },
            "rtt": {"root": str(rtt_root), **rtt},
            "server_manifest": {"sha256": server_manifest_hash},
            "harness": {"sha256": harness["sha256"]},
        }
        if partial.exists():
            try:
                prior_identity = json.loads((partial / "resume_identity.json").read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvaluationError("existing partial run has no valid resume identity") from error
            if prior_identity != resume_identity:
                raise EvaluationError("existing partial run identity does not match this invocation")
        else:
            partial.mkdir(mode=0o700)
            _write_json(partial / "command.json", {"argv": _safe_argv(argv), "cwd": str(Path.cwd())})
            _write_json(partial / "resume_identity.json", resume_identity)
            _write_json(partial / "server_identity.json", server_manifest)
        resolved = copy.deepcopy(config)
        resolved["runtime"] = {
            "benchmark": benchmark,
            "concurrency": concurrency,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "rtt_root": str(rtt_root),
            "started_at": started_at,
        }
        if not (partial / "resolved_config.json").exists():
            _write_json(partial / "resolved_config.json", resolved)
            _write_json(partial / "environment.json", _environment(server_manifest_hash, harness))
        generation = config["generation"]
        _verify_model(
            normalized_api,
            config["model"]["served_name"],
            snapshot_path,
            generation["request_timeout_seconds"],
        )
        partial_generation = partial / "partial_generation.jsonl"
        completed = _load_completed(partial_generation, items)
        responses = _generate(
            items,
            normalized_api,
            config["model"]["served_name"],
            generation,
            concurrency,
            partial_generation,
            completed,
        )
        generation_path = partial / "generation.jsonl"
        _write_jsonl(generation_path, _generation_rows(benchmark, items, responses))
        _archive_scorer_attempt(partial)
        scorer_command, scorer_cwd = _run_scorer(
            benchmark, rtt_root, generation_path, partial, generation["scorer_timeout_seconds"]
        )
        if benchmark == "muldimif":
            _derive_muldimif_per_example(rtt_root, generation_path, partial, generation["scorer_timeout_seconds"])
        metrics = (
            _muldimif_metrics(partial, items) if benchmark == "muldimif" else _if_metrics(partial, items, responses)
        )
        metrics["benchmark"] = benchmark
        _write_json(partial / "metrics.json", metrics)
        generation_bytes, generation_hash = _sha256(generation_path)
        _write_json(
            partial / "provenance.json",
            {
                "benchmark": benchmark,
                "completed_at": _now(),
                "data": {"path": str(input_path), "bytes": input_bytes, "sha256": benchmark_config["sha256"]},
                "generation": {"bytes": generation_bytes, "sha256": generation_hash},
                "harness": harness,
                "model": config["model"],
                "rtt": {**config["rtt"], **rtt},
                "scorer": {"command": scorer_command, "cwd": str(scorer_cwd)},
                "server_manifest": {"sha256": server_manifest_hash},
            },
        )
        partial_generation.unlink()
        (partial / "failure.json").unlink(missing_ok=True)
        _manifest(partial)
        partial.rename(output_dir)
        return output_dir
    except BaseException as error:
        if partial.exists():
            (partial / "sha256_manifest.json").unlink(missing_ok=True)
            _record_failure(partial, error)
            if isinstance(error, EvaluationError):
                error.add_note(f"partial artifacts: {partial}")
        raise
    finally:
        _release_lock(lock, lock_fd, lock_token)
