"""Fail-closed publishing for final response-only checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from rdan_grpo.roll_response_checkpoint import ArtifactIdentity, CheckpointIdentity, load_checkpoint

HF_REPO_ID = "beingamanforever/RDAN-GRPO-Qwen3-4B"
RECEIPT_SCHEMA_VERSION = 1

_COMMIT = re.compile(r"[0-9a-f]{40}")
_METHOD = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SHARD = re.compile(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors")
_MAX_SAFETENSORS_HEADER = 100_000_000
_REQUIRED = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}
_OPTIONAL = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "vocab.json",
}
_OPERATIONAL_FILES = {"rdan-response-counters-rank-0.json", "rdan-response-counters-rank-1.json"}
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_IDENTITY_KEYS = {
    "planned_horizon",
    "method",
    "method_weight",
    "resolved_config_sha256",
    "certificate",
    "data",
    "revisions",
    "base_checkpoint_sha256",
    "wandb",
}


class PublishError(ValueError):
    """Raised when a checkpoint or publication receipt cannot be trusted."""


@dataclass(frozen=True)
class PublishFile:
    """One validated local file and its destination path in the model repository."""

    path_in_repo: str
    local_path: Path
    size: int
    sha256: str


Uploader = Callable[..., str]


def method_revision(method: str) -> str:
    """Return the documented Hub revision reserved for one response method."""

    if not isinstance(method, str) or not _METHOD.fullmatch(method):
        raise PublishError("checkpoint method cannot identify a safe Hub revision")
    return method.replace("_", "-")


def load_publish_identity(path: str | Path) -> CheckpointIdentity:
    """Load a canonical, symlink-free CheckpointIdentity JSON artifact."""

    source = Path(path)
    try:
        mode = source.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise PublishError("checkpoint identity must be a regular file")
        body = source.read_bytes()
        value = json.loads(body)
    except PublishError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublishError(f"cannot read checkpoint identity: {error}") from error
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS or body != _canonical_json(value):
        raise PublishError("checkpoint identity artifact is not canonical or has an invalid schema")
    certificate = _artifact_identity(value["certificate"], "certificate")
    data = _artifact_identity(value["data"], "data")
    try:
        return CheckpointIdentity(
            planned_horizon=value["planned_horizon"],
            method=value["method"],
            method_weight=value["method_weight"],
            resolved_config_sha256=value["resolved_config_sha256"],
            certificate=certificate,
            data=data,
            revisions=value["revisions"],
            base_checkpoint_sha256=value["base_checkpoint_sha256"],
            wandb=value["wandb"],
        )
    except (KeyError, TypeError) as error:
        raise PublishError("checkpoint identity artifact is invalid") from error


def publish_response_model(
    checkpoint: str | Path,
    *,
    identity: CheckpointIdentity,
    receipt_path: str | Path,
    uploader: Uploader,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
) -> dict[str, Any]:
    """Validate and upload a final HF actor, then atomically seal its receipt."""

    if repo_id != HF_REPO_ID:
        raise PublishError(f"response models must publish to {HF_REPO_ID}")
    expected_revision = method_revision(identity.method)
    if revision is None:
        revision = expected_revision
    if revision != expected_revision:
        raise PublishError("Hub revision must match the checkpoint method")
    receipt = Path(receipt_path)
    with _receipt_lock(receipt):
        manifest = load_checkpoint(checkpoint, identity=identity)
        completed = manifest["completed_step"]
        if completed != manifest["planned_horizon"] or completed != identity.planned_horizon:
            raise PublishError("only the completed planned horizon can be published")
        checkpoint_path = Path(checkpoint)
        files = _publish_files(checkpoint_path / "actor", manifest["inventory"])
        with _staged_files(files, receipt.parent) as staged:
            commit = uploader(
                repo_id=repo_id,
                revision=revision,
                method=identity.method,
                completed_step=completed,
                files=staged,
            )
            _validate_staged(staged)
            if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
                raise PublishError("Hub uploader returned an invalid commit hash")
        value = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "published",
            "repo_id": repo_id,
            "revision": revision,
            "commit": commit,
            "method": identity.method,
            "completed_step": completed,
            "planned_horizon": identity.planned_horizon,
            "checkpoint_manifest_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
            "files": [{"path": file.path_in_repo, "size": file.size, "sha256": file.sha256} for file in files],
        }
        _seal_receipt(receipt, value)
    return value


def _publish_files(actor: Path, inventory_value: Any) -> tuple[PublishFile, ...]:
    inventory = _inventory(inventory_value)
    try:
        if actor.is_symlink() or not actor.is_dir():
            raise PublishError("checkpoint actor directory is missing or unsafe")
        entries = sorted(os.scandir(actor), key=lambda entry: entry.name)
    except PublishError:
        raise
    except OSError as error:
        raise PublishError(f"cannot inspect checkpoint actor: {error}") from error
    names: set[str] = set()
    paths: dict[str, Path] = {}
    operational_files: set[str] = set()
    has_dcp = False
    for entry in entries:
        if entry.is_symlink():
            raise PublishError(f"checkpoint actor contains a symlink: {entry.name}")
        if entry.name == "dcp" and entry.is_dir(follow_symlinks=False):
            has_dcp = True
            continue
        if entry.name in _OPERATIONAL_FILES and entry.is_file(follow_symlinks=False):
            operational_files.add(entry.name)
            continue
        if not entry.is_file(follow_symlinks=False):
            raise PublishError(f"checkpoint actor contains an unknown entry: {entry.name}")
        names.add(entry.name)
        paths[entry.name] = Path(entry.path)
    if (
        not has_dcp
        or operational_files != _OPERATIONAL_FILES
        or not any(path.startswith("actor/dcp/") for path in inventory)
    ):
        raise PublishError("checkpoint actor is missing complete DCP or rank evidence")
    allowed = _hf_names(paths)
    unknown = names - allowed
    if unknown:
        raise PublishError(f"checkpoint actor contains unknown HF files: {', '.join(sorted(unknown))}")
    if names != allowed:
        raise PublishError("checkpoint actor is missing required HF files")
    _validate_json_files(paths)
    _validate_safetensors(paths, allowed)
    result: list[PublishFile] = []
    for name in sorted(allowed):
        relative = f"actor/{name}"
        entry = inventory.get(relative)
        if entry is None:
            raise PublishError(f"HF file is absent from checkpoint inventory: {name}")
        result.append(PublishFile(name, paths[name], entry["size"], entry["sha256"]))
    return tuple(result)


@contextmanager
def _staged_files(files: tuple[PublishFile, ...], parent: Path) -> Iterator[tuple[PublishFile, ...]]:
    try:
        temporary = tempfile.TemporaryDirectory(prefix=".response-publish-", dir=parent)
    except OSError as error:
        raise PublishError(f"cannot create publication staging directory: {error}") from error
    with temporary as name:
        root = Path(name)
        staged: list[PublishFile] = []
        for file in files:
            destination = root / file.path_in_repo
            digest = hashlib.sha256()
            size = 0
            try:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                source_descriptor = os.open(file.local_path, flags)
                with os.fdopen(source_descriptor, "rb") as source, destination.open("xb") as target:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise PublishError("publication source changed into a non-regular file")
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                destination.chmod(0o400)
            except PublishError:
                raise
            except OSError as error:
                raise PublishError(f"cannot stage publication file {file.path_in_repo}: {error}") from error
            if (size, digest.hexdigest()) != (file.size, file.sha256):
                raise PublishError(f"publication source changed after checkpoint validation: {file.path_in_repo}")
            staged.append(PublishFile(file.path_in_repo, destination, size, digest.hexdigest()))
        _fsync_directory(root)
        yield tuple(staged)


def _validate_staged(files: tuple[PublishFile, ...]) -> None:
    for file in files:
        try:
            mode = file.local_path.lstat().st_mode
        except OSError as error:
            raise PublishError(f"cannot revalidate uploaded file {file.path_in_repo}: {error}") from error
        if not stat.S_ISREG(mode) or (file.local_path.stat().st_size, _sha256(file.local_path)) != (
            file.size,
            file.sha256,
        ):
            raise PublishError(f"uploaded file changed during publication: {file.path_in_repo}")


def _hf_names(paths: Mapping[str, Path]) -> set[str]:
    names = set(paths)
    if not _REQUIRED <= names:
        raise PublishError("checkpoint actor is missing required config or tokenizer files")
    allowed = set(_REQUIRED) | (names & _OPTIONAL)
    monolithic = "model.safetensors" in names
    indexed = "model.safetensors.index.json" in names
    if monolithic == indexed:
        raise PublishError("checkpoint actor requires exactly one complete safetensors layout")
    if monolithic:
        return allowed | {"model.safetensors"}
    index = _json_object(paths["model.safetensors.index.json"], "model index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PublishError("model index has no weight map")
    shards = set(weight_map.values())
    if any(not isinstance(name, str) or not _SHARD.fullmatch(name) for name in shards):
        raise PublishError("model index contains an unsafe shard path")
    if any(not isinstance(key, str) or not key for key in weight_map):
        raise PublishError("model index contains an invalid tensor name")
    if not shards <= names:
        raise PublishError("checkpoint actor is missing a referenced model shard")
    totals = {int(_SHARD.fullmatch(name).group(2)) for name in shards}
    total = next(iter(totals)) if len(totals) == 1 else 0
    if total != len(shards) or {int(_SHARD.fullmatch(name).group(1)) for name in shards} != set(range(1, total + 1)):
        raise PublishError("model index shard sequence is incomplete")
    return allowed | {"model.safetensors.index.json"} | shards


def _validate_json_files(paths: Mapping[str, Path]) -> None:
    config = _json_object(paths["config.json"], "model config")
    if config.get("model_type") != "qwen3":
        raise PublishError("checkpoint actor is not a Qwen3 model")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        _json_object(paths[name], name)


def _validate_safetensors(paths: Mapping[str, Path], allowed: set[str]) -> None:
    if "model.safetensors" in allowed:
        expected: dict[str, str] | None = None
        shards = ("model.safetensors",)
    else:
        index = _json_object(paths["model.safetensors.index.json"], "model index")
        expected = index["weight_map"]
        shards = tuple(sorted(set(expected.values())))
    observed: dict[str, str] = {}
    for name in shards:
        for key in _safetensor_keys(paths[name]):
            if key in observed:
                raise PublishError("model safetensors contain duplicate tensor names")
            observed[key] = name
    if not observed or (expected is not None and observed != expected):
        raise PublishError("model safetensors do not match the model index")


def _safetensor_keys(path: Path) -> tuple[str, ...]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise PublishError("safetensors header is truncated")
            header_length = struct.unpack("<Q", raw_length)[0]
            if (
                header_length == 0
                or header_length % 8
                or header_length > _MAX_SAFETENSORS_HEADER
                or header_length > size - 8
            ):
                raise PublishError("safetensors header length is invalid")
            header = json.loads(stream.read(header_length))
    except PublishError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as error:
        raise PublishError(f"cannot read safetensors header: {error}") from error
    if not isinstance(header, dict):
        raise PublishError("safetensors header must be a JSON object")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise PublishError("safetensors metadata is invalid")
    data_size = size - 8 - header_length
    ranges: list[tuple[int, int]] = []
    for key, tensor in header.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, dict):
            raise PublishError("safetensors tensor entry is invalid")
        if set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise PublishError("safetensors tensor entry schema is invalid")
        width = _DTYPE_BYTES.get(tensor["dtype"])
        shape = tensor["shape"]
        offsets = tensor["data_offsets"]
        if width is None or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise PublishError("safetensors tensor metadata is invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape + offsets):
            raise PublishError("safetensors tensor dimensions or offsets are invalid")
        start, end = offsets
        elements = 1
        for dimension in shape:
            elements *= dimension
        if start > end or end > data_size or end - start != elements * width:
            raise PublishError("safetensors tensor byte range is invalid")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise PublishError("safetensors tensor byte ranges are not contiguous")
        cursor = end
    if not ranges or cursor != data_size:
        raise PublishError("safetensors data buffer is incomplete")
    return tuple(header)


def _inventory(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise PublishError("checkpoint inventory is invalid")
    result: dict[str, dict[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise PublishError("checkpoint inventory is invalid")
        result[entry["path"]] = entry
    return result


def _artifact_identity(value: Any, name: str) -> ArtifactIdentity:
    if not isinstance(value, dict) or set(value) != {"id", "sha256"}:
        raise PublishError(f"checkpoint {name} identity is invalid")
    return ArtifactIdentity(id=value["id"], sha256=value["sha256"])


def _json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublishError(f"cannot read {name}: {error}") from error
    if not isinstance(value, dict):
        raise PublishError(f"{name} must be a JSON object")
    return value


@contextmanager
def _receipt_lock(receipt: Path) -> Iterator[None]:
    try:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        if receipt.parent.is_symlink() or not receipt.parent.is_dir():
            raise PublishError("publication receipt parent must be a real directory")
        if receipt.exists() or receipt.is_symlink():
            raise PublishError("publication receipt already exists")
        lock = receipt.with_name(f".{receipt.name}.lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    except PublishError:
        raise
    except OSError as error:
        raise PublishError(f"cannot acquire publication receipt lock: {error}") from error
    try:
        yield
    finally:
        try:
            lock.unlink()
        except OSError as error:
            raise PublishError(f"cannot release publication receipt lock: {error}") from error


def _seal_receipt(path: Path, value: Mapping[str, Any]) -> None:
    body = _canonical_json(value)
    temporary = path.with_name(f".{path.name}.incomplete-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise PublishError(f"cannot seal publication receipt: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PublishError(f"cannot hash publication artifact: {error}") from error
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise PublishError(f"publication artifact is not canonical JSON: {error}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
