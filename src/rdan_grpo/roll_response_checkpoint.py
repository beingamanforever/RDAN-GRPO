"""Atomic local checkpoints for response-only training runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from rdan_grpo.roll_response_config import UPDATES_PER_STEP
from rdan_grpo.roll_scalar import QUALITY_METHODS

SCHEMA_VERSION = 3
MANIFEST_NAME = "manifest.json"
_INCOMPLETE_PREFIX = ".incomplete-step-"
_PROMOTED_PREFIX = "step-"
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
_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "completed_step",
    "next_step",
    *_IDENTITY_KEYS,
    "optimizer_counters",
    "scheduler_counters",
    "scheduler_state",
    "rng_artifacts",
    "metrics",
    "peak_memory",
    "reward_variance",
    "group_diagnostics",
    "clipping_fraction",
    "receipt_links",
    "inventory",
}
_WANDB_KEYS = {"entity", "project", "run_id", "name", "group"}
_REVISION_KEYS = {"code", "rtt", "model"}
_GROUP_KEYS = {
    "group_count",
    "response_active_group_count",
    "response_active_group_rate",
    "quality_active_group_count",
    "quality_active_group_rate",
    "selected_reward_variance_mean",
}


class CheckpointError(ValueError):
    """Raised when checkpoint bytes or identity cannot be trusted."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable identity for a certified input artifact."""

    id: str
    sha256: str


@dataclass(frozen=True)
class CheckpointIdentity:
    """Immutable run identity required both at promotion and resume."""

    planned_horizon: int
    method: str
    method_weight: float | None
    resolved_config_sha256: str
    certificate: ArtifactIdentity
    data: ArtifactIdentity
    revisions: Mapping[str, str]
    base_checkpoint_sha256: str
    wandb: Mapping[str, str]


@dataclass(frozen=True)
class CheckpointState:
    """State observed after one fully completed optimizer transaction."""

    completed_step: int
    optimizer_counters: Mapping[int | str, int]
    scheduler_counters: Mapping[int | str, int]
    scheduler_state: Mapping[str, Any]
    rng_artifacts: Mapping[int | str, str | Path]
    metrics: Mapping[str, int | float]
    peak_memory: Mapping[int | str, int | float]
    reward_variance: float
    group_diagnostics: Mapping[str, int | float]
    clipping_fraction: float
    receipt_links: Mapping[str, str | Path]


def create_checkpoint_stage(root: str | Path, completed_step: int) -> Path:
    """Create the deterministic incomplete directory for one checkpoint step."""

    step = _nonnegative_int(completed_step, "completed_step")
    checkpoint_root = _checkpoint_root(root)
    stage = checkpoint_root / f"{_INCOMPLETE_PREFIX}{step:06d}"
    destination = checkpoint_root / f"{_PROMOTED_PREFIX}{step:06d}"
    if destination.exists() or destination.is_symlink():
        raise CheckpointError(f"checkpoint step {step} already exists")
    try:
        if stage.is_symlink():
            raise CheckpointError("checkpoint stage must not be a symlink")
        if stage.exists():
            _quarantine_stage(stage, step)
        stage.mkdir(mode=0o700)
        _fsync_directory(checkpoint_root)
    except OSError as error:
        raise CheckpointError(f"cannot create checkpoint stage: {error}") from error
    return stage


def _quarantine_stage(stage: Path, step: int) -> Path:
    if not stage.is_dir():
        raise CheckpointError("stale checkpoint stage must be a real directory")
    attempt = 1
    while True:
        target = stage.parent / f".quarantined-step-{step:06d}-attempt-{attempt:06d}"
        if not target.exists() and not target.is_symlink():
            break
        attempt += 1
    try:
        os.rename(stage, target)
        _fsync_directory(stage.parent)
    except OSError as error:
        raise CheckpointError(f"cannot quarantine stale checkpoint stage: {error}") from error
    return target


def promote_checkpoint(
    stage: str | Path,
    *,
    identity: CheckpointIdentity,
    state: CheckpointState,
    artifacts: Sequence[str | Path],
) -> Path:
    """Seal a complete inventory and atomically promote an incomplete checkpoint."""

    stage_path, stage_step = _stage_path(stage)
    identity_value = _identity_value(identity)
    state_value = _state_value(state, identity_value["planned_horizon"], identity_value["method"])
    if state_value["completed_step"] != stage_step:
        raise CheckpointError("checkpoint stage and completed step do not match")
    expected_files = _artifact_paths(artifacts)
    observed_files = _scan_files(stage_path, include_manifest=False)
    if set(observed_files) != expected_files:
        raise CheckpointError("checkpoint stage contains missing or unowned artifact files")
    inventory = [_inventory_entry(path, observed_files[path]) for path in sorted(observed_files)]
    inventory_by_path = {entry["path"]: entry for entry in inventory}
    rng = _artifact_links(state.rng_artifacts, inventory_by_path, "rng_artifacts")
    receipts = _artifact_links(state.receipt_links, inventory_by_path, "receipt_links")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "promoted",
        "completed_step": stage_step,
        "next_step": stage_step + 1,
        **identity_value,
        "optimizer_counters": state_value["optimizer_counters"],
        "scheduler_counters": state_value["scheduler_counters"],
        "scheduler_state": state_value["scheduler_state"],
        "rng_artifacts": rng,
        "metrics": state_value["metrics"],
        "peak_memory": state_value["peak_memory"],
        "reward_variance": state_value["reward_variance"],
        "group_diagnostics": state_value["group_diagnostics"],
        "clipping_fraction": state_value["clipping_fraction"],
        "receipt_links": receipts,
        "inventory": inventory,
    }
    body = _canonical_json(manifest)
    manifest_path = stage_path / MANIFEST_NAME
    try:
        descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_tree(stage_path)
        _verify_inventory(stage_path, inventory_by_path)
    except OSError as error:
        raise CheckpointError(f"cannot seal checkpoint manifest: {error}") from error

    destination = stage_path.parent / f"{_PROMOTED_PREFIX}{stage_step:06d}"
    with _step_lock(destination):
        if destination.exists() or destination.is_symlink():
            raise CheckpointError(f"promoted checkpoint already exists: {destination.name}")
        try:
            os.rename(stage_path, destination)
            _fsync_directory(destination.parent)
        except OSError as error:
            raise CheckpointError(f"cannot promote checkpoint: {error}") from error
    return destination


def load_checkpoint(path: str | Path, *, identity: CheckpointIdentity) -> dict[str, Any]:
    """Load a promoted checkpoint after revalidating every retained byte."""

    checkpoint, path_step = _promoted_path(path)
    manifest_path = checkpoint / MANIFEST_NAME
    try:
        mode = manifest_path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise CheckpointError("checkpoint manifest is not a regular file")
        body = manifest_path.read_bytes()
        manifest = json.loads(body)
    except CheckpointError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read checkpoint manifest: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise CheckpointError("checkpoint manifest schema is invalid")
    if body != _canonical_json(manifest):
        raise CheckpointError("checkpoint manifest is not canonical JSON")
    _validate_manifest_linkage(manifest, path_step)
    expected_identity = _identity_value(identity)
    if {key: manifest[key] for key in _IDENTITY_KEYS} != expected_identity:
        raise CheckpointError("checkpoint run identity or planned horizon drifted")
    inventory = _validate_inventory(manifest["inventory"])
    _verify_inventory(checkpoint, inventory)
    _validate_links(manifest["rng_artifacts"], inventory, "rng_artifacts")
    _validate_links(manifest["receipt_links"], inventory, "receipt_links")
    _validate_loaded_state(manifest)
    return manifest


def _checkpoint_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise CheckpointError("checkpoint root must be a real directory")
        return path.resolve(strict=True)
    except CheckpointError:
        raise
    except OSError as error:
        raise CheckpointError(f"cannot prepare checkpoint root: {error}") from error


def _stage_path(value: str | Path) -> tuple[Path, int]:
    path = Path(value).expanduser()
    if path.name.startswith(_PROMOTED_PREFIX):
        raise CheckpointError("checkpoint promotion requires an incomplete stage")
    step = _step_from_name(path.name, _INCOMPLETE_PREFIX)
    try:
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path.absolute():
            raise CheckpointError("checkpoint stage must be a canonical real directory")
    except OSError as error:
        raise CheckpointError(f"cannot inspect checkpoint stage: {error}") from error
    return path, step


def _promoted_path(value: str | Path) -> tuple[Path, int]:
    path = Path(value).expanduser()
    if path.name.startswith(_INCOMPLETE_PREFIX):
        raise CheckpointError("incomplete checkpoint stages cannot be loaded")
    step = _step_from_name(path.name, _PROMOTED_PREFIX)
    try:
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path.absolute():
            raise CheckpointError("checkpoint path must be a canonical real directory")
    except OSError as error:
        raise CheckpointError(f"cannot inspect checkpoint path: {error}") from error
    return path, step


def _step_from_name(name: str, prefix: str) -> int:
    suffix = name.removeprefix(prefix)
    if not name.startswith(prefix) or len(suffix) != 6 or not suffix.isascii() or not suffix.isdigit():
        raise CheckpointError("checkpoint directory name is invalid")
    return int(suffix)


def _identity_value(identity: CheckpointIdentity) -> dict[str, Any]:
    if not isinstance(identity, CheckpointIdentity):
        raise CheckpointError("checkpoint identity is invalid")
    horizon = _positive_int(identity.planned_horizon, "planned_horizon")
    if not isinstance(identity.method, str) or not identity.method:
        raise CheckpointError("checkpoint method must be non-empty")
    weight = identity.method_weight
    if weight is not None and (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or not 0 <= float(weight) <= 1
    ):
        raise CheckpointError("checkpoint method weight must be finite in [0, 1]")
    if not _sha256(identity.resolved_config_sha256):
        raise CheckpointError("resolved config hash is invalid")
    if not isinstance(identity.certificate, ArtifactIdentity):
        raise CheckpointError("checkpoint certificate identity is invalid")
    certificate = {"id": identity.certificate.id, "sha256": identity.certificate.sha256}
    if not certificate["id"] or not _sha256(certificate["sha256"]):
        raise CheckpointError("checkpoint certificate identity is invalid")
    if not isinstance(identity.data, ArtifactIdentity):
        raise CheckpointError("checkpoint data identity is invalid")
    data = {"id": identity.data.id, "sha256": identity.data.sha256}
    if not data["id"] or not _sha256(data["sha256"]):
        raise CheckpointError("checkpoint data identity is invalid")
    revisions = _string_mapping(identity.revisions, "revisions")
    if set(revisions) != _REVISION_KEYS:
        raise CheckpointError("checkpoint revisions must identify code, RTT, and model")
    if not _sha256(identity.base_checkpoint_sha256):
        raise CheckpointError("base checkpoint hash is invalid")
    wandb = _string_mapping(identity.wandb, "wandb")
    if set(wandb) != _WANDB_KEYS:
        raise CheckpointError("W&B identity schema is invalid")
    return {
        "planned_horizon": horizon,
        "method": identity.method,
        "method_weight": None if weight is None else float(weight),
        "resolved_config_sha256": identity.resolved_config_sha256,
        "certificate": certificate,
        "data": data,
        "revisions": revisions,
        "base_checkpoint_sha256": identity.base_checkpoint_sha256,
        "wandb": wandb,
    }


def _state_value(state: CheckpointState, planned_horizon: int, method: str) -> dict[str, Any]:
    if not isinstance(state, CheckpointState):
        raise CheckpointError("checkpoint state is invalid")
    completed = _nonnegative_int(state.completed_step, "completed_step")
    if completed > planned_horizon:
        raise CheckpointError("completed step exceeds the planned horizon")
    optimizer = _rank_counters(state.optimizer_counters, "optimizer_counters")
    scheduler = _rank_counters(state.scheduler_counters, "scheduler_counters")
    _validate_training_counters(optimizer, scheduler, completed)
    scheduler_state = _json_safe(state.scheduler_state, "scheduler_state")
    if not isinstance(scheduler_state, dict):
        raise CheckpointError("scheduler state must be a JSON object")
    metrics = _number_mapping(state.metrics, "metrics")
    peak = _number_mapping(state.peak_memory, "peak_memory", rank_keys=True, nonnegative=True)
    if set(peak) != set(optimizer):
        raise CheckpointError("peak memory ranks differ from optimizer ranks")
    variance = _finite_number(state.reward_variance, "reward_variance", minimum=0)
    diagnostics = _group_diagnostics(state.group_diagnostics, method)
    if variance <= 0 or diagnostics["selected_reward_variance_mean"] != variance:
        raise CheckpointError("checkpoint requires nonzero within-group reward variance")
    clipping = _finite_number(state.clipping_fraction, "clipping_fraction", minimum=0, maximum=1)
    if clipping >= 1:
        raise CheckpointError("clipping_fraction must be below 1")
    return {
        "completed_step": completed,
        "optimizer_counters": optimizer,
        "scheduler_counters": scheduler,
        "scheduler_state": scheduler_state,
        "metrics": metrics,
        "peak_memory": peak,
        "reward_variance": variance,
        "group_diagnostics": diagnostics,
        "clipping_fraction": clipping,
    }


def _artifact_paths(values: Sequence[str | Path]) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise CheckpointError("checkpoint artifacts must be a non-empty sequence")
    paths = [_relative_path(value) for value in values]
    if len(paths) != len(set(paths)) or MANIFEST_NAME in paths:
        raise CheckpointError("checkpoint artifact paths are duplicated or reserved")
    return set(paths)


def _relative_path(value: str | Path) -> str:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise CheckpointError("checkpoint artifact path is invalid")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CheckpointError("checkpoint artifact path must be normalized and relative")
    return path.as_posix()


def _scan_files(root: Path, *, include_manifest: bool) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise CheckpointError(f"cannot scan checkpoint directory: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                raise CheckpointError(f"checkpoint contains a symlink: {relative}")
            if entry.is_dir(follow_symlinks=False):
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                if include_manifest or relative != MANIFEST_NAME:
                    files[relative] = path
            else:
                raise CheckpointError(f"checkpoint contains a non-regular artifact: {relative}")

    visit(root)
    return files


def _inventory_entry(relative: str, path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            os.fsync(stream.fileno())
    except OSError as error:
        raise CheckpointError(f"cannot hash checkpoint artifact {relative}: {error}") from error
    return {"path": relative, "size": size, "sha256": digest.hexdigest()}


def _artifact_links(
    values: Mapping[int | str, str | Path],
    inventory: Mapping[str, Mapping[str, Any]],
    name: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping) or not values:
        raise CheckpointError(f"{name} must be a non-empty mapping")
    links: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        label = _label(key, name)
        relative = _relative_path(value)
        if relative not in inventory:
            raise CheckpointError(f"{name} references an unowned artifact")
        entry = inventory[relative]
        links[label] = {"path": relative, "size": entry["size"], "sha256": entry["sha256"]}
    if len(links) != len(values):
        raise CheckpointError(f"{name} has duplicate labels")
    return dict(sorted(links.items()))


def _validate_manifest_linkage(manifest: Mapping[str, Any], path_step: int) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "promoted":
        raise CheckpointError("checkpoint is not a promoted supported artifact")
    completed = _nonnegative_int(manifest.get("completed_step"), "completed_step")
    next_step = _nonnegative_int(manifest.get("next_step"), "next_step")
    horizon = _positive_int(manifest.get("planned_horizon"), "planned_horizon")
    if completed != path_step or next_step != completed + 1 or completed > horizon:
        raise CheckpointError("checkpoint step linkage is invalid")


def _validate_inventory(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CheckpointError("checkpoint inventory is invalid")
    inventory: dict[str, dict[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise CheckpointError("checkpoint inventory entry is invalid")
        relative = _relative_path(entry["path"])
        size = _nonnegative_int(entry["size"], "inventory size")
        if relative == MANIFEST_NAME or not _sha256(entry["sha256"]) or relative in inventory:
            raise CheckpointError("checkpoint inventory entry is invalid")
        inventory[relative] = {"path": relative, "size": size, "sha256": entry["sha256"]}
    if [entry["path"] for entry in value] != sorted(inventory):
        raise CheckpointError("checkpoint inventory order is invalid")
    return inventory


def _validate_links(value: Any, inventory: Mapping[str, Mapping[str, Any]], name: str) -> None:
    if not isinstance(value, dict) or not value:
        raise CheckpointError(f"checkpoint {name} is invalid")
    for label, link in value.items():
        _label(label, name)
        if not isinstance(link, dict) or set(link) != {"path", "size", "sha256"}:
            raise CheckpointError(f"checkpoint {name} link is invalid")
        relative = _relative_path(link["path"])
        if inventory.get(relative) != {"path": relative, "size": link["size"], "sha256": link["sha256"]}:
            raise CheckpointError(f"checkpoint {name} link does not match inventory")


def _validate_loaded_state(manifest: Mapping[str, Any]) -> None:
    optimizer = _rank_counters(manifest["optimizer_counters"], "optimizer_counters")
    scheduler = _rank_counters(manifest["scheduler_counters"], "scheduler_counters")
    _validate_training_counters(optimizer, scheduler, manifest["completed_step"])
    state = _json_safe(manifest["scheduler_state"], "scheduler_state")
    if not isinstance(state, dict):
        raise CheckpointError("checkpoint scheduler state is invalid")
    _number_mapping(manifest["metrics"], "metrics")
    peak = _number_mapping(manifest["peak_memory"], "peak_memory", rank_keys=True, nonnegative=True)
    if set(peak) != set(optimizer):
        raise CheckpointError("checkpoint peak memory ranks differ")
    variance = _finite_number(manifest["reward_variance"], "reward_variance", minimum=0)
    diagnostics = _group_diagnostics(manifest["group_diagnostics"], str(manifest["method"]))
    if variance <= 0 or diagnostics["selected_reward_variance_mean"] != variance:
        raise CheckpointError("checkpoint requires nonzero within-group reward variance")
    clipping = _finite_number(manifest["clipping_fraction"], "clipping_fraction", minimum=0, maximum=1)
    if clipping >= 1:
        raise CheckpointError("checkpoint clipping_fraction must be below 1")


def _group_diagnostics(value: Any, method: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or set(value) != _GROUP_KEYS:
        raise CheckpointError("group_diagnostics schema is invalid")
    group_count = _positive_int(value["group_count"], "group_count")
    response_count = _nonnegative_int(value["response_active_group_count"], "response_active_group_count")
    quality_count = _nonnegative_int(value["quality_active_group_count"], "quality_active_group_count")
    if response_count > group_count or quality_count > group_count or response_count == 0:
        raise CheckpointError("group_diagnostics counts are invalid")
    response_rate = _finite_number(
        value["response_active_group_rate"], "response_active_group_rate", minimum=0, maximum=1
    )
    quality_rate = _finite_number(
        value["quality_active_group_rate"], "quality_active_group_rate", minimum=0, maximum=1
    )
    variance = _finite_number(value["selected_reward_variance_mean"], "selected_reward_variance_mean", minimum=0)
    if (
        not math.isclose(float(response_rate), response_count / group_count, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(float(quality_rate), quality_count / group_count, rel_tol=0, abs_tol=1e-12)
        or variance <= 0
    ):
        raise CheckpointError("group_diagnostics rates or variance are invalid")
    if method in QUALITY_METHODS and quality_rate < 0.1:
        raise CheckpointError("quality method requires quality active group rate at least 0.1")
    return {
        "group_count": group_count,
        "response_active_group_count": response_count,
        "response_active_group_rate": response_rate,
        "quality_active_group_count": quality_count,
        "quality_active_group_rate": quality_rate,
        "selected_reward_variance_mean": variance,
    }


def _rank_counters(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise CheckpointError(f"{name} must be a non-empty per-rank mapping")
    result: dict[str, int] = {}
    for key, counter in value.items():
        rank = _rank(key, name)
        result[rank] = _nonnegative_int(counter, name)
    if len(result) != len(value):
        raise CheckpointError(f"{name} contains duplicate ranks")
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _validate_training_counters(
    optimizer: Mapping[str, int],
    scheduler: Mapping[str, int],
    completed_step: int,
) -> None:
    if set(optimizer) != {"0", "1"} or set(scheduler) != {"0", "1"}:
        raise CheckpointError("checkpoint training counters require exact DP2 ranks 0 and 1")
    if optimizer["0"] != optimizer["1"]:
        raise CheckpointError("checkpoint optimizer counters differ across replicas")
    if scheduler["0"] != scheduler["1"]:
        raise CheckpointError("checkpoint scheduler counters differ across replicas")
    if optimizer["0"] != scheduler["0"]:
        raise CheckpointError("checkpoint optimizer and scheduler counters differ")
    updates = optimizer["0"]
    if updates != completed_step * UPDATES_PER_STEP:
        raise CheckpointError("checkpoint training counters are inconsistent with completed step")


def _number_mapping(
    value: Any,
    name: str,
    *,
    rank_keys: bool = False,
    nonnegative: bool = False,
) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or not value:
        raise CheckpointError(f"{name} must be a non-empty mapping")
    result: dict[str, int | float] = {}
    for key, number in value.items():
        label = _rank(key, name) if rank_keys else _label(key, name)
        result[label] = _finite_number(number, name, nonnegative=nonnegative)
    if len(result) != len(value):
        raise CheckpointError(f"{name} contains duplicate keys")
    return dict(sorted(result.items(), key=(lambda item: int(item[0])) if rank_keys else None))


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise CheckpointError(f"{name} must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        label = _label(key, name)
        if not isinstance(item, str) or not item:
            raise CheckpointError(f"{name} values must be non-empty strings")
        result[label] = item
    if len(result) != len(value):
        raise CheckpointError(f"{name} contains duplicate keys")
    return dict(sorted(result.items()))


def _json_safe(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            label = _label(key, name)
            if label in result:
                raise CheckpointError(f"{name} contains duplicate keys")
            result[label] = _json_safe(item, name)
        return dict(sorted(result.items()))
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, name) for item in value]
    raise CheckpointError(f"{name} is not canonical JSON data")


def _rank(value: Any, name: str) -> str:
    if isinstance(value, bool):
        raise CheckpointError(f"{name} rank is invalid")
    if isinstance(value, int):
        rank = value
    elif isinstance(value, str) and value.isascii() and value.isdigit() and str(int(value)) == value:
        rank = int(value)
    else:
        raise CheckpointError(f"{name} rank is invalid")
    if rank < 0:
        raise CheckpointError(f"{name} rank is invalid")
    return str(rank)


def _label(value: Any, name: str) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value):
        raise CheckpointError(f"{name} key is invalid")
    return str(value)


def _finite_number(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CheckpointError(f"{name} must contain finite numbers")
    if nonnegative and value < 0 or minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise CheckpointError(f"{name} number is outside its valid range")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise CheckpointError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"checkpoint manifest is not canonical JSON: {error}") from error


def _verify_inventory(root: Path, inventory: Mapping[str, Mapping[str, Any]]) -> None:
    observed = _scan_files(root, include_manifest=False)
    if set(observed) != set(inventory):
        raise CheckpointError("checkpoint inventory has missing or extra files")
    for relative, entry in inventory.items():
        if _inventory_entry(relative, observed[relative]) != entry:
            raise CheckpointError(f"checkpoint artifact is corrupt: {relative}")


def _fsync_tree(root: Path) -> None:
    directories = []
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        current = Path(directory)
        for name in names:
            path = current / name
            if path.is_symlink():
                raise CheckpointError(f"checkpoint contains a symlink: {path.relative_to(root)}")
        for name in files:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise CheckpointError(f"checkpoint contains a non-regular artifact: {path.relative_to(root)}")
        directories.append(current)
    for directory in dict.fromkeys(directories):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _step_lock(destination: Path) -> Iterator[None]:
    lock = destination.parent / f".{destination.name}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as error:
        message = (
            f"checkpoint promotion is already in progress: {destination.name}"
            if isinstance(error, FileExistsError)
            else f"cannot acquire checkpoint promotion lock: {error}"
        )
        raise CheckpointError(message) from error
    os.close(descriptor)
    try:
        yield
    finally:
        try:
            lock.unlink()
            _fsync_directory(destination.parent)
        except OSError as error:
            raise CheckpointError(f"cannot release checkpoint promotion lock: {error}") from error
