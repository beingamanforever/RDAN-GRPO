"""Immutable launch-readiness evidence for response-level RL."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdan_grpo.program import MODEL_NAME, MODEL_REVISION, RTT_REVISION, ProgramBundle, check_program

READINESS_ID = "qwen_response_readiness_v1"
EVIDENCE_ORDER = ("judge_calibration", "runtime_parity", "vllm_runtime_parity", "no_update")
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_CONTRACT = REPO_ROOT / "configs/compute/qwen_a100_2x.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s\\]+)")
_SECRET = re.compile(r"(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9]{20,})")
_CREDENTIAL_KEY = re.compile(r"(?:^|_)(?:api_key|password|secret|token|credential)s?$", re.IGNORECASE)


class ResponseReadinessError(ValueError):
    """Raised when Phase 4 is not authorized by exact Phase 3 evidence."""


@dataclass(frozen=True)
class RuntimeExpectation:
    """Runtime identities required by the canonical compute contract."""

    profile: str
    minimum_ram_bytes: int
    minimum_free_disk_bytes: int
    platform_receipt: str
    python_receipt: str
    container: Mapping[str, Any]
    gpu_model: str
    gpu_count: int
    minimum_gpu_memory_mib: int
    minimum_driver: tuple[int, ...]
    cuda_runtime: str
    nccl_package: str
    supported_links: tuple[str, ...]
    packages: Mapping[str, str]
    allowed_local_suffixes: Mapping[str, str]
    artifact_hashes: Mapping[str, str]
    compute_sha256: str


def build_response_readiness(
    program_path: str | Path,
    bootstrap_path: str | Path,
    evidence_paths: Sequence[str | Path],
    *,
    rdan_revision: str,
) -> dict[str, Any]:
    """Validate the complete evidence chain and return its canonical receipt."""

    if not _REVISION.fullmatch(rdan_revision):
        raise ResponseReadinessError("RDAN revision must be an exact 40-character Git revision")
    runtime = _load_runtime_expectation()
    program_file = _regular_file(program_path, "program")
    bootstrap_file = _regular_file(bootstrap_path, "bootstrap marker")
    if len(evidence_paths) != len(EVIDENCE_ORDER):
        raise ResponseReadinessError(f"evidence must contain exactly {len(EVIDENCE_ORDER)} ordered artifacts")
    evidence_files = tuple(_regular_file(path, "lifecycle evidence") for path in evidence_paths)
    snapshots = {path: path.read_bytes() for path in (program_file, bootstrap_file, *evidence_files)}

    bootstrap = _load_json_bytes(snapshots[bootstrap_file], "bootstrap marker")
    _validate_bootstrap(bootstrap, rdan_revision, runtime)
    bundle = check_program(program_file)
    _validate_program_state(bundle)
    _validate_evidence_files(bundle, evidence_files, snapshots)
    _reject_secrets(bootstrap, "bootstrap marker")
    _verify_unchanged(snapshots)
    return _build_receipt(bundle, snapshots, program_file, bootstrap_file, evidence_files, rdan_revision, runtime)


def _validate_evidence_files(
    bundle: ProgramBundle,
    evidence_files: tuple[Path, ...],
    snapshots: Mapping[Path, bytes],
) -> None:
    expected_paths = tuple(_lifecycle_path(bundle, name) for name in EVIDENCE_ORDER)
    if evidence_files != expected_paths:
        raise ResponseReadinessError(
            "lifecycle evidence must be supplied in judge calibration, HF parity, vLLM parity, no-update order"
        )
    for name, path in zip(EVIDENCE_ORDER, evidence_files, strict=True):
        payload = _load_json_bytes(snapshots[path], name)
        if payload != bundle.lifecycle_artifacts[name]:
            raise ResponseReadinessError(f"{name} bytes differ from the validated program artifact")

    refs = bundle.program["lifecycle_artifacts"]
    no_update = bundle.lifecycle_artifacts["no_update"]
    source_hashes = no_update["source_sha256"]
    if source_hashes.get("judge_calibration") != refs["judge_calibration"]["sha256"]:
        raise ResponseReadinessError("no-update evidence does not bind the judge calibration")
    if source_hashes.get("runtime_parity") != refs["runtime_parity"]["sha256"]:
        raise ResponseReadinessError("no-update evidence does not bind runtime parity")


def _verify_unchanged(snapshots: Mapping[Path, bytes]) -> None:
    for path, original in snapshots.items():
        if path.read_bytes() != original:
            raise ResponseReadinessError(f"evidence changed during validation: {path.name}")


def _build_receipt(
    bundle: ProgramBundle,
    snapshots: Mapping[Path, bytes],
    program_file: Path,
    bootstrap_file: Path,
    evidence_files: tuple[Path, ...],
    rdan_revision: str,
    runtime: RuntimeExpectation,
) -> dict[str, Any]:
    refs = bundle.program["lifecycle_artifacts"]
    body: dict[str, Any] = {
        "schema_version": 2,
        "id": READINESS_ID,
        "status": "ready",
        "compute_contract_sha256": runtime.compute_sha256,
        "program": {
            "id": bundle.program["id"],
            "sha256": _bytes_sha256(snapshots[program_file]),
            "rdan_revision": rdan_revision,
        },
        "bootstrap": {
            "sha256": _bytes_sha256(snapshots[bootstrap_file]),
            "profile": runtime.profile,
            "rdan_revision": rdan_revision,
            "rtt_revision": RTT_REVISION,
            "model_revision": MODEL_REVISION,
            "container_digest": runtime.container["linux_amd64_digest"],
        },
        "evidence_order": list(EVIDENCE_ORDER),
        "evidence": {
            name: {
                "artifact_id": refs[name]["artifact_id"],
                "sha256": _bytes_sha256(snapshots[path]),
            }
            for name, path in zip(EVIDENCE_ORDER, evidence_files, strict=True)
        },
    }
    return {"receipt_id": _json_sha256(body), **body}


def issue_response_readiness(
    program_path: str | Path,
    bootstrap_path: str | Path,
    evidence_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    rdan_revision: str,
) -> dict[str, Any]:
    """Validate and atomically publish one no-clobber readiness receipt."""

    receipt = build_response_readiness(
        program_path,
        bootstrap_path,
        evidence_paths,
        rdan_revision=rdan_revision,
    )
    _write_once(Path(output_path), receipt)
    return receipt


def validate_response_readiness(
    receipt_path: str | Path,
    program_path: str | Path,
    bootstrap_path: str | Path,
    evidence_paths: Sequence[str | Path],
    *,
    rdan_revision: str,
) -> dict[str, Any]:
    """Check an existing receipt against current evidence without writing."""

    receipt_file = _regular_file(receipt_path, "readiness receipt")
    actual = _load_json_bytes(receipt_file.read_bytes(), "readiness receipt")
    expected = build_response_readiness(
        program_path,
        bootstrap_path,
        evidence_paths,
        rdan_revision=rdan_revision,
    )
    if actual != expected:
        raise ResponseReadinessError("readiness receipt differs from the current validated evidence")
    body = {key: value for key, value in actual.items() if key != "receipt_id"}
    if actual.get("receipt_id") != _json_sha256(body):
        raise ResponseReadinessError("readiness receipt ID is invalid")
    return actual


def _validate_program_state(bundle: ProgramBundle) -> None:
    refs = bundle.program["lifecycle_artifacts"]
    if tuple(name for name in EVIDENCE_ORDER if refs[name].get("status") != "frozen"):
        raise ResponseReadinessError(
            "judge calibration, HF parity, vLLM parity, and no-update evidence must be frozen"
        )
    if bundle.program["readiness"].get("launch") != "ready":
        raise ResponseReadinessError("validated program launch readiness is not ready")
    if bundle.program["readiness"].get("scalar_training") != "ready":
        raise ResponseReadinessError("validated scalar training readiness is not ready")


def _validate_bootstrap(
    report: Mapping[str, Any],
    rdan_revision: str,
    runtime: RuntimeExpectation,
) -> None:
    required = {
        "schema_version",
        "status",
        "profile",
        "python",
        "platform",
        "container",
        "repositories",
        "model",
        "storage",
        "gpu",
        "requirements",
        "packages",
        "runtime_imports",
        "data_preparation",
        "data_runtime",
        "venv",
        "host_readiness",
        "capabilities",
        "compute_contract_sha256",
    }
    if not required.issubset(report):
        raise ResponseReadinessError("bootstrap marker is minimal or incomplete")
    if (
        report.get("schema_version") != 2
        or report.get("status") != "passed"
        or report.get("profile") != runtime.profile
        or report.get("compute_contract_sha256") != runtime.compute_sha256
    ):
        raise ResponseReadinessError("bootstrap marker did not pass the response A100 profile")
    if report.get("python") != runtime.python_receipt or report.get("platform") != runtime.platform_receipt:
        raise ResponseReadinessError("bootstrap Python or platform identity is invalid")
    _validate_bootstrap_container(report, runtime)
    _validate_bootstrap_sources(report, rdan_revision, runtime)
    _validate_bootstrap_gpu(report, runtime)
    _validate_bootstrap_packages(report, runtime)
    _validate_bootstrap_runtime(report)
    _validate_host_readiness(report, runtime)
    if report.get("capabilities") != {
        "judge_access": True,
        "tracking_access": True,
        "model_publish_access": True,
    }:
        raise ResponseReadinessError("bootstrap capability evidence is incomplete")


def _validate_bootstrap_container(report: Mapping[str, Any], runtime: RuntimeExpectation) -> None:
    expected = runtime.container
    container = _mapping(report.get("container"), "bootstrap container")
    required = {
        "image": expected["image"],
        "index_digest": expected["manifest_digest"],
        "amd64_digest": expected["linux_amd64_digest"],
        "cuda": expected["cuda"],
        "release": expected["nvidia_pytorch_release"],
        "identity_source": "docker-image-inspect",
    }
    if any(container.get(key) != value for key, value in required.items()):
        raise ResponseReadinessError("bootstrap container identity is invalid")
    if not _DIGEST.fullmatch(str(container.get("image_id", ""))):
        raise ResponseReadinessError("bootstrap container content identity is invalid")


def _validate_bootstrap_sources(
    report: Mapping[str, Any],
    rdan_revision: str,
    runtime: RuntimeExpectation,
) -> None:
    repositories = _mapping(report.get("repositories"), "bootstrap repositories")
    if repositories != {"rdan": rdan_revision, "rtt": RTT_REVISION}:
        raise ResponseReadinessError("bootstrap repository revisions are stale or mismatched")
    model = _mapping(report.get("model"), "bootstrap model")
    if model.get("model") != MODEL_NAME or model.get("revision") != MODEL_REVISION:
        raise ResponseReadinessError("bootstrap model identity is invalid")
    if not isinstance(model.get("file_sha256"), Mapping) or not isinstance(model.get("tokenizer_sha256"), Mapping):
        raise ResponseReadinessError("bootstrap model hashes are incomplete")
    requirements = _mapping(report.get("requirements"), "bootstrap requirements")
    if any(requirements.get(name) != digest for name, digest in runtime.artifact_hashes.items()):
        raise ResponseReadinessError("bootstrap runtime contract hashes are stale or incomplete")


def _validate_bootstrap_gpu(report: Mapping[str, Any], runtime: RuntimeExpectation) -> None:
    gpu = _mapping(report.get("gpu"), "bootstrap GPU")
    devices = gpu.get("devices")
    topology = _mapping(gpu.get("topology"), "bootstrap GPU topology")
    nccl = runtime.packages[runtime.nccl_package]
    if (
        gpu.get("count") != runtime.gpu_count
        or gpu.get("cuda") != runtime.cuda_runtime
        or gpu.get("nccl") != nccl
        or _version_tuple(str(gpu.get("driver", ""))) < runtime.minimum_driver
        or not isinstance(devices, list)
        or len(devices) != runtime.gpu_count
        or any(not _valid_gpu(device, index, runtime) for index, device in enumerate(devices))
        or not _valid_topology(topology, runtime.supported_links)
    ):
        raise ResponseReadinessError("bootstrap GPU identity is invalid")


def _validate_bootstrap_packages(report: Mapping[str, Any], runtime: RuntimeExpectation) -> None:
    packages = _mapping(report.get("packages"), "bootstrap packages")
    if set(packages) != set(runtime.packages):
        raise ResponseReadinessError("bootstrap package set is invalid")
    if any(not _package_version_allowed(name, str(packages[name]), runtime) for name in runtime.packages):
        raise ResponseReadinessError("bootstrap package identity is invalid")


def _validate_bootstrap_runtime(report: Mapping[str, Any]) -> None:
    storage = _mapping(report.get("storage"), "bootstrap storage")
    if set(storage) != {"cache", "run"} or any(not _valid_storage(storage[name]) for name in storage):
        raise ResponseReadinessError("bootstrap storage evidence is invalid")
    imports = _mapping(report.get("runtime_imports"), "bootstrap runtime imports")
    expected_imports = {
        "fsdp2": "FSDP2TrainStrategy",
        "hf": "HfInferStrategy",
        "infer_worker": "ResponseInferWorker",
        "pipeline": "ResponseTrainingPipeline",
        "reward_worker": "RTTCompatibleRubricRewardWorker",
        "train_worker": "ResponseActorWorker",
    }
    if imports.get("sdpa") is not True or any(imports.get(key) != value for key, value in expected_imports.items()):
        raise ResponseReadinessError("bootstrap runtime import evidence is invalid")
    if not isinstance(report.get("data_preparation"), Mapping) or not isinstance(report.get("data_runtime"), Mapping):
        raise ResponseReadinessError("bootstrap data runtime evidence is invalid")


def _validate_host_readiness(report: Mapping[str, Any], runtime: RuntimeExpectation) -> None:
    readiness = _mapping(report.get("host_readiness"), "bootstrap host readiness")
    ram = _mapping(readiness.get("ram"), "bootstrap host RAM")
    if not _threshold_passes(ram, "total_bytes", runtime.minimum_ram_bytes):
        raise ResponseReadinessError("bootstrap host RAM evidence is invalid")
    disk = _mapping(readiness.get("disk"), "bootstrap host disk")
    if set(disk) != {"cache", "run"} or any(
        not _threshold_passes(
            _mapping(value, f"bootstrap {name} disk"),
            "available_bytes",
            runtime.minimum_free_disk_bytes,
        )
        for name, value in disk.items()
    ):
        raise ResponseReadinessError("bootstrap host disk evidence is invalid")
    gpu = _mapping(readiness.get("gpu"), "bootstrap host GPU readiness")
    topology = _mapping(gpu.get("topology"), "bootstrap host GPU topology")
    if (
        gpu.get("count") != runtime.gpu_count
        or gpu.get("compute_process_count") != 0
        or gpu.get("driver") != report["gpu"].get("driver")
        or not isinstance(gpu.get("devices"), list)
        or gpu["devices"] != report["gpu"].get("devices")
        or topology != report["gpu"].get("topology")
        or not _valid_topology(topology, runtime.supported_links)
    ):
        raise ResponseReadinessError("bootstrap host GPU readiness is invalid")


def _threshold_passes(value: Mapping[str, Any], observed: str, expected_minimum: int) -> bool:
    actual = value.get(observed)
    required = value.get("minimum_bytes")
    return (
        isinstance(actual, int)
        and not isinstance(actual, bool)
        and isinstance(required, int)
        and not isinstance(required, bool)
        and required == expected_minimum
        and actual >= required
    )


def _valid_gpu(value: Any, index: int, runtime: RuntimeExpectation) -> bool:
    if not isinstance(value, Mapping):
        return False
    memory = value.get("memory_mib")
    return (
        value.get("index") == index
        and isinstance(value.get("uuid"), str)
        and value["uuid"].startswith("GPU-")
        and runtime.gpu_model.upper() in str(value.get("name", "")).upper()
        and isinstance(memory, int)
        and not isinstance(memory, bool)
        and memory >= runtime.minimum_gpu_memory_mib
        and value.get("memory_used_mib") == 0
        and value.get("utilization_percent") == 0
    )


def _valid_topology(value: Mapping[str, Any], supported_links: Sequence[str]) -> bool:
    forward = value.get("gpu0_to_gpu1")
    reverse = value.get("gpu1_to_gpu0")
    return (
        set(value) == {"gpu0_to_gpu1", "gpu1_to_gpu0"}
        and forward == reverse
        and _supported_topology_link(str(forward), supported_links)
    )


def _valid_storage(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and isinstance(value.get("free_bytes"), int)
        and not isinstance(value.get("free_bytes"), bool)
        and value["free_bytes"] > 0
    )


def _load_runtime_expectation() -> RuntimeExpectation:
    compute = _load_json_bytes(COMPUTE_CONTRACT.read_bytes(), "compute contract")
    if compute.get("schema_version") != 4 or compute.get("id") != "qwen_a100_2x":
        raise ResponseReadinessError("compute contract identity is invalid")
    runtime = _mapping(compute.get("response_runtime"), "response runtime")
    target = _mapping(compute.get("target_topology"), "target topology")
    host = _mapping(runtime.get("host"), "response runtime host")
    platform_contract = _mapping(runtime.get("platform"), "response runtime platform")
    gpu = _mapping(runtime.get("gpu"), "response runtime GPU")
    package_contract = _mapping(runtime.get("packages"), "response runtime packages")
    _validate_runtime_contract_shape(runtime, host, platform_contract, gpu, package_contract)
    container_path = _runtime_path(platform_contract.get("container_contract"), "container contract")
    package_paths = tuple(
        _runtime_path(value, "package contract") for value in _required_string_list(package_contract, "contracts")
    )
    container = _load_json_bytes(container_path.read_bytes(), "container contract")
    return _make_runtime_expectation(
        runtime,
        target,
        host,
        platform_contract,
        gpu,
        package_contract,
        container,
        package_paths,
    )


def _make_runtime_expectation(
    runtime: Mapping[str, Any],
    target: Mapping[str, Any],
    host: Mapping[str, Any],
    platform_contract: Mapping[str, Any],
    gpu: Mapping[str, Any],
    package_contract: Mapping[str, Any],
    container: Mapping[str, Any],
    package_paths: tuple[Path, ...],
) -> RuntimeExpectation:
    packages = _load_package_contracts(package_paths)
    nccl_package = _required_string(gpu, "nccl_package")
    if gpu.get("topology_contract") != "target_topology" or gpu.get("require_idle") is not True:
        raise ResponseReadinessError("response runtime GPU contract is invalid")
    if nccl_package not in packages:
        raise ResponseReadinessError("response runtime package contracts omit NCCL")
    python = _mapping(container.get("python"), "container Python")
    operating_system = _mapping(container.get("os"), "container operating system")
    if container.get("schema_version") != 1 or operating_system.get("id") != "ubuntu":
        raise ResponseReadinessError("container contract identity is invalid")
    artifacts = (COMPUTE_CONTRACT, _runtime_path(platform_contract["container_contract"], "container"), *package_paths)
    return RuntimeExpectation(
        profile=_required_string(runtime, "profile"),
        minimum_ram_bytes=_positive_int(host, "minimum_ram_gib") * 2**30,
        minimum_free_disk_bytes=_positive_int(host, "minimum_free_disk_gib") * 2**30,
        platform_receipt=_required_string(platform_contract, "receipt"),
        python_receipt=f"{_positive_int(python, 'major')}.{_positive_int(python, 'minor')}.x",
        container=container,
        gpu_model=_required_string(target, "model"),
        gpu_count=_positive_int(target, "count"),
        minimum_gpu_memory_mib=_positive_int(target, "minimum_memory_gib_each") * 1024,
        minimum_driver=_version_tuple(_required_string(gpu, "minimum_driver")),
        cuda_runtime=_required_string(gpu, "cuda_runtime"),
        nccl_package=nccl_package,
        supported_links=tuple(_required_string_list(gpu, "supported_links")),
        packages=packages,
        allowed_local_suffixes=_required_string_mapping(package_contract, "allowed_local_suffixes"),
        artifact_hashes={path.name: _bytes_sha256(path.read_bytes()) for path in artifacts},
        compute_sha256=_bytes_sha256(COMPUTE_CONTRACT.read_bytes()),
    )


def _validate_runtime_contract_shape(
    runtime: Mapping[str, Any],
    host: Mapping[str, Any],
    platform_contract: Mapping[str, Any],
    gpu: Mapping[str, Any],
    packages: Mapping[str, Any],
) -> None:
    _require_exact_keys(runtime, {"profile", "host", "platform", "gpu", "packages"}, "response runtime")
    _require_exact_keys(host, {"minimum_ram_gib", "minimum_free_disk_gib"}, "response runtime host")
    _require_exact_keys(
        platform_contract,
        {"system", "machines", "receipt", "container_contract"},
        "response runtime platform",
    )
    _require_exact_keys(
        gpu,
        {
            "topology_contract",
            "minimum_driver",
            "cuda_runtime",
            "nccl_package",
            "require_idle",
            "supported_links",
        },
        "response runtime GPU",
    )
    _require_exact_keys(
        packages,
        {"index_url", "torch_index_url", "torch_backend", "contracts", "allowed_local_suffixes"},
        "response runtime packages",
    )


def _load_package_contracts(paths: Sequence[Path]) -> dict[str, str]:
    packages: dict[str, str] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _PIN.match(line)
            if match is None:
                continue
            name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            version = match.group(2)
            if name in packages and packages[name] != version:
                raise ResponseReadinessError(f"conflicting runtime package identity: {name}")
            packages[name] = version
    if not packages:
        raise ResponseReadinessError("runtime package contracts contain no exact pins")
    return packages


def _runtime_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ResponseReadinessError(f"{name} path is invalid")
    path = (COMPUTE_CONTRACT.parent / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ResponseReadinessError(f"{name} must stay inside the repository") from error
    if path.is_symlink() or not path.is_file():
        raise ResponseReadinessError(f"{name} must be a regular file")
    return path


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ResponseReadinessError(f"runtime contract {key} must be a non-empty string")
    return item


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise ResponseReadinessError(f"runtime contract {key} must be a positive integer")
    return item


def _required_string_list(value: Mapping[str, Any], key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list) or not item or any(not isinstance(entry, str) or not entry for entry in item):
        raise ResponseReadinessError(f"runtime contract {key} must be a non-empty string list")
    return item


def _required_string_mapping(value: Mapping[str, Any], key: str) -> dict[str, str]:
    item = value.get(key)
    invalid = isinstance(item, Mapping) and any(
        not isinstance(name, str) or not isinstance(pin, str) for name, pin in item.items()
    )
    if not isinstance(item, Mapping) or invalid:
        raise ResponseReadinessError(f"runtime contract {key} must map strings to strings")
    return dict(item)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ResponseReadinessError(f"{name} fields differ from the supported contract")


def _package_version_allowed(name: str, actual: str, runtime: RuntimeExpectation) -> bool:
    expected = runtime.packages[name]
    suffix = runtime.allowed_local_suffixes.get(name)
    return actual == expected or (suffix is not None and actual == f"{expected}+{suffix}")


def _supported_topology_link(value: str, prefixes: Sequence[str]) -> bool:
    return any(value == prefix or (prefix == "NV" and re.fullmatch(r"NV\d+", value)) for prefix in prefixes)


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(map(int, match.group(1).split("."))) if match else ()


def _lifecycle_path(bundle: ProgramBundle, name: str) -> Path:
    reference = bundle.program["lifecycle_artifacts"][name]
    path = (bundle.repo_root / reference["path"]).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ResponseReadinessError(f"{name} must be the exact regular program artifact")
    return path


def _regular_file(path: str | Path, name: str) -> Path:
    target = Path(path).absolute()
    if target.is_symlink() or not target.is_file() or target.resolve() != target:
        raise ResponseReadinessError(f"{name} must be a regular non-symlink file")
    return target


def _load_json_bytes(raw: bytes, name: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ResponseReadinessError(f"{name} contains duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseReadinessError(f"{name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ResponseReadinessError(f"{name} must contain a JSON object")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseReadinessError(f"{name} must be an object")
    return value


def _reject_secrets(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _CREDENTIAL_KEY.search(str(key)) and not isinstance(item, bool):
                raise ResponseReadinessError(f"{path} contains a credential value")
            _reject_secrets(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")
        return
    if isinstance(value, str) and _SECRET.search(value):
        raise ResponseReadinessError(f"{path} contains secret material")


def _write_once(path: Path, receipt: Mapping[str, Any]) -> None:
    target = path.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(target)
    payload = _canonical_json(receipt) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_symlink_ancestors(path: Path) -> None:
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ResponseReadinessError(f"output parent must not be a symlink: {current}")
        current = current.parent


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
