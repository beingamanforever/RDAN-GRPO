"""Fail-closed W&B tracking for the pinned RTT runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
RTT_TRACKING_PATH = "roll/utils/tracking.py"
RTT_TRACKING_SHA256 = "58923728093768f2ab6a20cb3f5627ae417c48aedb5dc768abebe4cd690744ea"
TRACKER_NAME = "rdan_wandb"
WANDB_ENTITY = "RDAN-GRPO"
WANDB_PROJECT = "rdan-grpo-qwen3-4b"

METHODS = frozenset(
    {
        "base",
        "sft",
        "dpo",
        "rtt-papo-response",
        "rdan-scalar",
        "rl-csr",
        "rl-aon",
        "rl-mix",
        "rtt-aon",
        "rtt-csr",
        "rdan-full",
    }
)
TRAIN_STAGES = frozenset({"pilot", "confirm", "train", "resume"})
BENCHMARKS = frozenset({"ifeval", "ifbench", "muldimif", "advancedif", "math-500", "gpqa", "mmlu-pro"})

_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{2,127}\Z")
_GIT_HASH = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_FIELDS = frozenset(
    {"resolved_config_sha256", "model_revision", "data_sha256", "code_revision", "checkpoint_sha256"}
)
_SECRET_VALUE = re.compile(r"(?:sk-or-v1-[A-Za-z0-9_-]{20,}|wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})")
_SECRET_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "credential", "password", "refresh_token", "secret", "token"}
)
_REDACTED = "[REDACTED]"


class WandbTrackingError(ValueError):
    """Raised when the W&B tracking contract cannot be proven safe."""


def register_wandb_tracker(
    rtt_root: str | Path,
    registry: MutableMapping[str, Any] | None = None,
) -> type[RdanWandbTracker]:
    """Register the RDAN tracker after verifying the exact clean RTT checkout."""

    tracking_path = verify_rtt_tracking(rtt_root)
    target = registry if registry is not None else _load_registry(tracking_path)
    if TRACKER_NAME in target:
        raise WandbTrackingError(f"RTT tracker registry already contains {TRACKER_NAME}")
    target[TRACKER_NAME] = RdanWandbTracker
    return RdanWandbTracker


def verify_rtt_tracking(rtt_root: str | Path) -> Path:
    """Verify the RTT revision, clean worktree, and exact tracker source bytes."""

    root = Path(rtt_root).resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise WandbTrackingError(f"RTT path is not the checkout root: {root}")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != RTT_REVISION:
        raise WandbTrackingError(f"unexpected RTT revision: {revision}")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise WandbTrackingError("RTT checkout is dirty")
    path = root / RTT_TRACKING_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != RTT_TRACKING_SHA256:
        raise WandbTrackingError(f"unexpected RTT tracking digest: {digest}")
    return path


def deterministic_run_id(metadata: Mapping[str, Any]) -> str:
    """Build the stable W&B run ID for validated explicit run metadata."""

    normalized = _validate_metadata(metadata)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return f"rdan-{hashlib.sha256(payload).hexdigest()[:20]}"


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the exact semantic config without self-referential tracker options."""

    normalized = _canonical_config(config)
    normalized.pop("tracker_kwargs", None)
    _normalize_runtime_paths(normalized)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RdanWandbTracker:
    """Mirror metrics to W&B while retaining a redacted local JSONL authority."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        values = _init_values(kwargs)
        metadata = _validate_metadata(values.pop("metadata"))
        safe_config = _canonical_config(config)
        if canonical_config_sha256(safe_config) != metadata["resolved_config_sha256"]:
            raise WandbTrackingError("config does not match metadata.resolved_config_sha256")
        _validate_identity(values, metadata)
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise WandbTrackingError("WANDB_API_KEY is required in the environment")

        run_dir = _safe_run_dir(values.pop("log_dir"))
        settings = values["settings"]
        if "rdan_identity" in safe_config:
            raise WandbTrackingError("config contains reserved rdan_identity metadata")
        identity = {key: metadata[key] for key in sorted(_IDENTITY_FIELDS)}
        wandb_config = {**safe_config, "rdan_identity": identity}
        init_record = {
            "type": "init",
            "metadata": metadata,
            "identity": identity,
            "config": wandb_config,
            "wandb": {**values, "dir": str(run_dir), "settings": _json_safe(settings)},
        }
        self._events = run_dir / "rdan-events.jsonl"
        _append_jsonl(self._events, init_record)

        wandb = importlib.import_module("wandb")
        self._wandb = wandb
        self._run_dir = run_dir
        self.run = wandb.init(
            entity=values["entity"],
            project=values["project"],
            group=values["group"],
            name=values["name"],
            job_type=values["job_type"],
            id=values["id"],
            resume=values["resume"],
            tags=values["tags"],
            notes=values["notes"],
            dir=str(run_dir),
            settings=settings,
            config=wandb_config,
        )

    def log(self, values: dict[str, Any], step: int | None, **kwargs: Any) -> None:
        """Append redacted metrics locally before mirroring the same values to W&B."""

        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise WandbTrackingError("step must be a non-negative integer or None")
        unknown = set(kwargs) - {"commit"}
        if unknown or ("commit" in kwargs and not isinstance(kwargs["commit"], bool)):
            raise WandbTrackingError(f"unsupported W&B log options: {sorted(unknown)}")
        safe_values = _json_safe(redact_secrets(values))
        record = {"type": "metrics", "step": step, "metrics": safe_values}
        if "commit" in kwargs:
            record["commit"] = kwargs["commit"]
        _append_jsonl(self._events, record)
        self.run.log(safe_values, step=step, **kwargs)

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        aliases: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Upload a run-owned file or directory after rejecting unsafe paths and names."""

        artifact_path = _safe_artifact_path(path, self._run_dir)
        _require_safe_name(name, "artifact name")
        _require_safe_name(artifact_type, "artifact type")
        safe_aliases = [_safe_alias(alias) for alias in aliases]
        safe_metadata = _json_safe(redact_secrets(metadata or {}))
        artifact = self._wandb.Artifact(name=name, type=artifact_type, metadata=safe_metadata)
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))
        relative = artifact_path.relative_to(self._run_dir).as_posix()
        _append_jsonl(
            self._events,
            {
                "type": "artifact",
                "path": relative,
                "name": name,
                "artifact_type": artifact_type,
                "aliases": safe_aliases,
                "metadata": safe_metadata,
            },
        )
        logged = self.run.log_artifact(artifact, aliases=safe_aliases)
        wait = getattr(logged, "wait", None)
        if not callable(wait):
            raise WandbTrackingError("W&B artifact upload does not expose a completion boundary")
        wait()
        return logged

    def finish(self) -> None:
        """Seal the local event stream and finish the W&B run."""

        _append_jsonl(self._events, {"type": "finish"})
        self.run.finish()


def redact_secrets(value: Any, key: str | None = None) -> Any:
    """Return a recursive copy with credential-like keys and values redacted."""

    if key is not None and _secret_key(key):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(item_key): redact_secrets(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        return _REDACTED
    return value


def _canonical_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise WandbTrackingError("W&B config must be a JSON object")
    if _contains_secret(config):
        raise WandbTrackingError("W&B config must not contain resolved credentials")
    normalized = _strict_json(config)
    if not isinstance(normalized, dict):
        raise WandbTrackingError("W&B config must be a JSON object")
    return normalized


def _strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WandbTrackingError("W&B config values must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise WandbTrackingError("W&B config object keys must be strings")
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    raise WandbTrackingError(f"W&B config value is not exact JSON: {type(value).__name__}")


def _init_values(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = sorted(key for key in kwargs if _secret_key(key))
    if forbidden:
        raise WandbTrackingError(f"credential options are forbidden: {forbidden}")
    allowed = {
        "entity",
        "project",
        "group",
        "name",
        "job_type",
        "id",
        "resume",
        "tags",
        "notes",
        "log_dir",
        "settings",
        "metadata",
    }
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise WandbTrackingError(f"unsupported W&B tracker options: {unknown}")
    required = allowed - {"entity", "project"}
    missing = sorted(required - set(kwargs))
    if missing:
        raise WandbTrackingError(f"missing W&B tracker options: {missing}")
    values = dict(kwargs)
    values.setdefault("entity", WANDB_ENTITY)
    values.setdefault("project", WANDB_PROJECT)
    if _contains_secret(values):
        raise WandbTrackingError("W&B tracker options contain a credential-like value")
    return values


def _validate_identity(values: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    if values["entity"] != WANDB_ENTITY or values["project"] != WANDB_PROJECT:
        raise WandbTrackingError("W&B entity or project does not match the frozen experiment")
    expected_group, expected_name = _expected_names(metadata)
    if values["group"] != expected_group or values["name"] != expected_name:
        raise WandbTrackingError("W&B group or name does not match explicit run metadata")
    if values["job_type"] != metadata["kind"]:
        raise WandbTrackingError("W&B job_type does not match explicit run metadata")
    expected_id = deterministic_run_id(metadata)
    if values["id"] != expected_id or not _SAFE_NAME.fullmatch(values["id"]):
        raise WandbTrackingError("W&B run id is not the deterministic safe run id")
    if values["resume"] not in {"allow", "must"}:
        raise WandbTrackingError("W&B resume must be allow or must")
    tags = values["tags"]
    if not isinstance(tags, (list, tuple)) or not tags or any(not _safe_text(tag) for tag in tags):
        raise WandbTrackingError("W&B tags must be a non-empty sequence of safe strings")
    if not _safe_text(values["notes"]):
        raise WandbTrackingError("W&B notes must be a non-empty safe string")
    settings = values["settings"]
    if not isinstance(settings, Mapping) or settings.get("console") != "off" or _contains_secret(settings):
        raise WandbTrackingError("W&B settings must be a safe mapping with console off")


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise WandbTrackingError("run metadata must be a mapping")
    kind = metadata.get("kind")
    expected = (
        {"kind", "method", "seed", "stage"} if kind == "train" else {"kind", "method", "seed", "benchmark"}
    ) | _IDENTITY_FIELDS
    if set(metadata) != expected:
        raise WandbTrackingError("run metadata fields do not match its kind")
    if not all(
        _full_hash(metadata[field], pattern)
        for field, pattern in (
            ("resolved_config_sha256", _SHA256),
            ("model_revision", _GIT_HASH),
            ("data_sha256", _SHA256),
            ("code_revision", _GIT_HASH),
            ("checkpoint_sha256", _SHA256),
        )
    ):
        raise WandbTrackingError("run metadata sealed identity fields must be full lowercase hashes")
    method = metadata.get("method")
    seed = metadata.get("seed")
    if method not in METHODS or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
        raise WandbTrackingError("run metadata has an unknown method or invalid seed")
    if kind == "train" and metadata.get("stage") not in TRAIN_STAGES:
        raise WandbTrackingError("run metadata has an unknown training stage")
    if kind == "eval" and metadata.get("benchmark") not in BENCHMARKS:
        raise WandbTrackingError("run metadata has an unknown benchmark")
    if kind not in {"train", "eval"}:
        raise WandbTrackingError("run metadata has an unknown kind")
    return {key: metadata[key] for key in sorted(metadata)}


def _normalize_runtime_paths(config: dict[str, Any]) -> None:
    timestamp = re.compile(r"/\d{8}-\d{6}\Z")
    for name in ("profiler_output_dir", "length_profiler_dir"):
        value = config.get(name)
        if isinstance(value, str):
            config[name] = timestamp.sub("/<runtime-timestamp>", value)
    checkpoint = config.get("checkpoint_config")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("output_dir"), str):
        checkpoint["output_dir"] = timestamp.sub("/<runtime-timestamp>", checkpoint["output_dir"])


def _expected_names(metadata: Mapping[str, Any]) -> tuple[str, str]:
    method = metadata["method"]
    seed = metadata["seed"]
    if metadata["kind"] == "train":
        return f"qwen-{method}", f"qwen-{method}-{metadata['stage']}-s{seed}"
    group = f"qwen-{method}-eval"
    return group, f"{group}-{metadata['benchmark']}-s{seed}"


def _full_hash(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _safe_run_dir(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise WandbTrackingError("W&B run directory must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or not path.is_dir() or path.is_symlink():
        raise WandbTrackingError("W&B run directory must be an existing canonical directory")
    return path


def _safe_artifact_path(value: str | Path, run_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or not path.exists() or path.is_symlink():
        raise WandbTrackingError("artifact path must be an existing canonical absolute path")
    try:
        path.relative_to(run_dir)
    except ValueError as error:
        raise WandbTrackingError("artifact path must be contained by the W&B run directory") from error
    if path.is_dir() and any(child.is_symlink() for child in path.rglob("*")):
        raise WandbTrackingError("artifact directories cannot contain symlinks")
    if not path.is_file() and not path.is_dir():
        raise WandbTrackingError("artifact path must be a regular file or directory")
    return path


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise WandbTrackingError("local event log is not a regular file")
    line = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{line}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WandbTrackingError("W&B values must be finite")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (RuntimeError, TypeError, ValueError) as error:
            raise WandbTrackingError(f"W&B value is not a JSON scalar: {type(value).__name__}") from error
        if scalar is not value:
            return _json_safe(scalar)
    raise WandbTrackingError(f"W&B value is not JSON serializable: {type(value).__name__}")


def _secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized in _SECRET_KEYS or any(
        normalized.endswith(suffix) for suffix in ("_api_key", "_credential", "_password", "_secret", "_token")
    )


def _contains_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        # An unset credential field carries nothing to leak, so only a populated one counts.
        return any(
            (_secret_key(str(key)) and item is not None) or _contains_secret(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item) for item in value)
    return isinstance(value, str) and _SECRET_VALUE.search(value) is not None


def _safe_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not _contains_secret(value) and "\n" not in value


def _require_safe_name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or _contains_secret(value):
        raise WandbTrackingError(f"{label} is unsafe")


def _safe_alias(value: Any) -> str:
    _require_safe_name(value, "artifact alias")
    return value


def _load_registry(tracking_path: Path) -> MutableMapping[str, Any]:
    module = importlib.import_module("roll.utils.tracking")
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != tracking_path:
        raise WandbTrackingError(f"loaded RTT tracker from an unexpected path: {module_path}")
    registry = getattr(module, "tracker_registry", None)
    if not isinstance(registry, MutableMapping):
        raise WandbTrackingError("RTT tracker registry is unavailable")
    return registry


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise WandbTrackingError(f"cannot verify RTT checkout: {type(error).__name__}") from error
    return result.stdout.strip()
