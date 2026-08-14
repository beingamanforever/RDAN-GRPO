#!/usr/bin/env python3
"""Install or verify the exact two-A100 response-training runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = (
    REPO_ROOT / "requirements/a100-response-torch.txt",
    REPO_ROOT / "requirements/a100-response-direct.txt",
    REPO_ROOT / "requirements/a100-response-flash.txt",
)
LOCK = REPO_ROOT / "requirements/a100-response-linux-py312.lock"
LOCK_INPUT = REPO_ROOT / "requirements/a100-response.in"
BOOTSTRAP_LOCK = REPO_ROOT / "requirements/a100-response-bootstrap-linux-py312.lock"
BOOTSTRAP_INPUT = REPO_ROOT / "requirements/a100-response-bootstrap.in"
SNAPSHOT_MANIFEST = REPO_ROOT / "requirements/qwen3-4b-instruct-2507-snapshot.json"
COMPUTE_CONTRACT = REPO_ROOT / "configs/compute/qwen_a100_2x.json"
DATA_REQUIREMENTS = REPO_ROOT / "requirements/data-prep-py311-direct.txt"
DATA_LOCK = REPO_ROOT / "requirements/data-prep-linux-py311.lock"
TORCH_REQUIREMENTS = REQUIREMENTS[0]
DIRECT_REQUIREMENTS = REQUIREMENTS[1]
FLASH_REQUIREMENTS = REQUIREMENTS[2]

DATA_PYTHON_VERSION = "3.11.15"
DATA_ENV_NAME = "data-prep-py311"
DATA_PYTHON_BIN = "data-python"
DATA_PACKAGE_PINS = {
    "absl-py": "2.5.0",
    "fasttext-wheel": "0.9.2",
    "huggingface-hub": "0.36.2",
    "immutabledict": "4.3.1",
    "langdetect": "1.0.9",
    "nltk": "3.10.0",
    "numpy": "1.26.4",
    "pyarrow": "25.0.1",
    "transformers": "4.57.0",
}
RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOKENIZER_HASHES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
}
SNAPSHOT_HASHES = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "8e3dd0c3b5b11897cc71092ccfe517bb7a9783479baa3665aad73c8d1a2041cd",
    "config.json": "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
    "generation_config.json": "835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00003.safetensors": "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6",
    "model-00002-of-00003.safetensors": "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1",
    "model-00003-of-00003.safetensors": "7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d",
    "model.safetensors.index.json": "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SNAPSHOT_SIZES = {
    ".gitattributes": 1_570,
    "LICENSE": 11_343,
    "README.md": 8_168,
    "config.json": 727,
    "generation_config.json": 238,
    "merges.txt": 1_671_839,
    "model-00001-of-00003.safetensors": 3_957_900_840,
    "model-00002-of-00003.safetensors": 3_987_450_520,
    "model-00003-of-00003.safetensors": 99_630_640,
    "model.safetensors.index.json": 32_819,
    "tokenizer.json": 11_422_654,
    "tokenizer_config.json": 9_377,
    "vocab.json": 2_776_833,
}
SECRET_ENV_NAMES = ("OPENROUTER_API_KEY", "WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
PRIVATE_ENV_NAMES = (*SECRET_ENV_NAMES, "OPENAI_API_KEY")
INSTALL_OVERRIDE_ENV_NAMES = (
    "PIP_BUILD_CONSTRAINT",
    "PIP_CONFIG_FILE",
    "PIP_CONSTRAINT",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_NO_INDEX",
    "PIP_REQUIRE_VIRTUALENV",
    "PIP_TARGET",
    "PIP_TRUSTED_HOST",
    "PIP_USER",
    "UV_BUILD_CONSTRAINT",
    "UV_CONSTRAINT",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_INDEX",
    "UV_INDEX_URL",
    "UV_NO_INDEX",
    "UV_PYTHON_DOWNLOADS_JSON_URL",
    "UV_PYTHON_INSTALL_MIRROR",
)
IDENTITY_RECEIPT = Path("/run/rdan/a100-image-identity.json")
IDENTITY_NAME = "a100-image-identity.json"

_PIN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s\\]+)")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_AMBIENT_PROBE = r"""
import importlib.util
import json
spec = importlib.util.find_spec("roll")
origin = None if spec is None else (spec.origin or "namespace")
print("RDAN_AMBIENT=" + json.dumps({"origin": origin}, sort_keys=True))
"""

_PACKAGE_PROBE = r"""
import importlib.metadata
import json
versions = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    if name:
        versions[name] = dist.version
print("RDAN_PACKAGES=" + json.dumps(versions, sort_keys=True))
"""

_DATA_RUNTIME_PROBE = r"""
import importlib.metadata
import json
import platform
versions = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    if name:
        versions[name] = dist.version
print("RDAN_DATA_RUNTIME=" + json.dumps({"packages": versions, "python": platform.python_version()}, sort_keys=True))
"""

_CUDA_PROBE = r"""
import json
import torch
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name, "memory_mib": props.total_memory // 2**20})
payload = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": devices,
    "nccl_available": torch.distributed.is_nccl_available(),
    "nccl_version": torch.cuda.nccl.version() if torch.distributed.is_nccl_available() else None,
}
print("RDAN_CUDA=" + json.dumps(payload, sort_keys=True))
"""

_IMPORT_PROBE = r"""
import json
import os
import pathlib
import sys
rtt = pathlib.Path(sys.argv[1]).resolve()
rdan = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(rtt))
sys.path.insert(0, str(rdan / "src"))
from rdan_grpo.roll_compat import install_rtt_compat
install_rtt_compat(rtt)
import accelerate
import absl
import codetiming
import dacite
import datasets
import deepspeed
import fasttext
import flash_attn.bert_padding
import hydra
import huggingface_hub
import langdetect
import latex2sympy2
import latex2sympy2_extended
import math_verify
import more_itertools
import numpy
import omegaconf
import openai
import peft
import psutil
import pybase64
import pyarrow
import ray
import requests
import tensordict
import torch
import transformers
import wandb
import yaml
from rdan_grpo.roll_response_pipeline import ResponseTrainingPipeline
from rdan_grpo.roll_response_workers import ResponseActorWorker, ResponseInferWorker
from rdan_grpo.roll_reward import RTTCompatibleRubricRewardWorker
from roll.distributed.strategy.fsdp2_strategy import FSDP2TrainStrategy
from roll.distributed.strategy.hf_strategy import HfInferStrategy
import rdan_grpo
import roll
payload = {
    "rdan_origin": pathlib.Path(rdan_grpo.__file__).resolve().as_posix(),
    "roll_origin": pathlib.Path(roll.__file__).resolve().as_posix(),
    "fsdp2": FSDP2TrainStrategy.__name__,
    "hf": HfInferStrategy.__name__,
    "pipeline": ResponseTrainingPipeline.__name__,
    "reward_worker": RTTCompatibleRubricRewardWorker.__name__,
    "train_worker": ResponseActorWorker.__name__,
    "infer_worker": ResponseInferWorker.__name__,
    "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
}
print("RDAN_IMPORTS=" + json.dumps(payload, sort_keys=True))
"""


class BootstrapError(RuntimeError):
    """Raised when the runtime cannot be certified exactly."""


@dataclass(frozen=True)
class RuntimeContract:
    """Canonical response runtime values loaded from the compute contract."""

    profile: str
    minimum_ram_bytes: int
    minimum_free_disk_bytes: int
    system: str
    machines: tuple[str, ...]
    platform_receipt: str
    python_version: tuple[int, int]
    container_image: str
    container_index_digest: str
    container_amd64_digest: str
    container_cuda: str
    container_os: str
    container_release: str
    gpu_model: str
    gpu_count: int
    minimum_gpu_memory_mib: int
    minimum_driver: tuple[int, ...]
    cuda_runtime: str
    nccl_package: str
    require_idle: bool
    supported_links: tuple[str, ...]
    package_contracts: tuple[Path, ...]
    allowed_local_suffixes: Mapping[str, str]
    index_url: str
    torch_index_url: str
    torch_backend: str
    sha256: str

    @property
    def container_ref(self) -> str:
        """Return the immutable Linux AMD64 container reference."""

        repository = self.container_image.split(":", 1)[0]
        return f"{repository}@{self.container_amd64_digest}"


@dataclass(frozen=True)
class HostInfo:
    """Host facts that do not require a subprocess."""

    system: str
    machine: str
    implementation: str
    python: tuple[int, ...]
    os_release: Mapping[str, str]
    container: bool
    cuda: str | None
    container_release: str | None
    image_digest: str | None
    image_id: str | None
    identity_source: str | None


@dataclass(frozen=True)
class Inputs:
    """Immutable paths and revisions supplied by the operator."""

    rtt_root: Path
    rdan_root: Path
    rdan_revision: str
    snapshot: Path
    cache_root: Path
    run_root: Path


@dataclass
class _SetupState:
    stage: Path
    data_root: Path | None
    published: list[tuple[Path, Path]]


def _load_runtime_contract() -> RuntimeContract:
    compute = _load_json_object(COMPUTE_CONTRACT, "compute contract")
    if compute.get("schema_version") != 4 or compute.get("id") != "qwen_a100_2x":
        raise BootstrapError("compute contract identity is invalid")
    runtime = _required_mapping(compute, "response_runtime", "compute contract")
    target = _required_mapping(compute, "target_topology", "compute contract")
    host = _required_mapping(runtime, "host", "response runtime")
    platform_contract = _required_mapping(runtime, "platform", "response runtime")
    gpu = _required_mapping(runtime, "gpu", "response runtime")
    packages = _required_mapping(runtime, "packages", "response runtime")
    container_path = _contract_path(platform_contract.get("container_contract"), "container contract")
    container = _load_json_object(container_path, "container contract")
    package_paths = tuple(_contract_path(value, "package contract") for value in _string_list(packages, "contracts"))
    return _build_runtime_contract(
        compute,
        runtime,
        target,
        host,
        platform_contract,
        gpu,
        packages,
        container,
        package_paths,
    )


def _build_runtime_contract(
    compute: Mapping[str, Any],
    runtime: Mapping[str, Any],
    target: Mapping[str, Any],
    host: Mapping[str, Any],
    platform_contract: Mapping[str, Any],
    gpu: Mapping[str, Any],
    packages: Mapping[str, Any],
    container: Mapping[str, Any],
    package_paths: tuple[Path, ...],
) -> RuntimeContract:
    _validate_runtime_shape(runtime, host, platform_contract, gpu, packages)
    _validate_container_shape(container)
    _validate_target_reference(target, gpu)
    suffixes = _string_mapping(packages, "allowed_local_suffixes")
    contract = RuntimeContract(
        profile=_required_string(runtime, "profile"),
        minimum_ram_bytes=_positive_int(host, "minimum_ram_gib") * 2**30,
        minimum_free_disk_bytes=_positive_int(host, "minimum_free_disk_gib") * 2**30,
        system=_required_string(platform_contract, "system"),
        machines=tuple(_string_list(platform_contract, "machines")),
        platform_receipt=_required_string(platform_contract, "receipt"),
        python_version=_container_python(container),
        container_image=_required_string(container, "image"),
        container_index_digest=_required_digest(container, "manifest_digest"),
        container_amd64_digest=_required_digest(container, "linux_amd64_digest"),
        container_cuda=_required_string(container, "cuda"),
        container_os=_container_os(container),
        container_release=_required_string(container, "nvidia_pytorch_release"),
        gpu_model=_required_string(target, "model"),
        gpu_count=_positive_int(target, "count"),
        minimum_gpu_memory_mib=_positive_int(target, "minimum_memory_gib_each") * 1024,
        minimum_driver=_required_version(gpu, "minimum_driver"),
        cuda_runtime=_required_string(gpu, "cuda_runtime"),
        nccl_package=_required_string(gpu, "nccl_package"),
        require_idle=gpu.get("require_idle") is True,
        supported_links=tuple(_string_list(gpu, "supported_links")),
        package_contracts=package_paths,
        allowed_local_suffixes=suffixes,
        index_url=_required_url(packages, "index_url"),
        torch_index_url=_required_url(packages, "torch_index_url"),
        torch_backend=_required_string(packages, "torch_backend"),
        sha256=hashlib.sha256(COMPUTE_CONTRACT.read_bytes()).hexdigest(),
    )
    _validate_package_contracts(contract)
    return contract


def _validate_runtime_shape(
    runtime: Mapping[str, Any],
    host: Mapping[str, Any],
    platform_contract: Mapping[str, Any],
    gpu: Mapping[str, Any],
    packages: Mapping[str, Any],
) -> None:
    _exact_keys(runtime, {"profile", "host", "platform", "gpu", "packages"}, "response runtime")
    _exact_keys(host, {"minimum_ram_gib", "minimum_free_disk_gib"}, "response runtime host")
    _exact_keys(
        platform_contract,
        {"system", "machines", "receipt", "container_contract"},
        "response runtime platform",
    )
    _exact_keys(
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
    _exact_keys(
        packages,
        {"index_url", "torch_index_url", "torch_backend", "contracts", "allowed_local_suffixes"},
        "response runtime packages",
    )


def _validate_container_shape(container: Mapping[str, Any]) -> None:
    _exact_keys(
        container,
        {
            "cuda",
            "image",
            "linux_amd64_digest",
            "manifest_digest",
            "nvidia_pytorch_release",
            "os",
            "python",
            "schema_version",
        },
        "container contract",
    )
    if container.get("schema_version") != 1:
        raise BootstrapError("container contract schema is invalid")


def _validate_target_reference(target: Mapping[str, Any], gpu: Mapping[str, Any]) -> None:
    if gpu.get("topology_contract") != "target_topology":
        raise BootstrapError("response runtime must reference the target topology")
    if not gpu.get("require_idle"):
        raise BootstrapError("response runtime must require idle GPUs")
    if _positive_int(target, "count") != 2:
        raise BootstrapError("response runtime requires exactly two GPUs")


def _validate_package_contracts(contract: RuntimeContract) -> None:
    expected = _runtime_expected_packages(contract)
    if not expected or contract.nccl_package not in expected:
        raise BootstrapError("response runtime package contracts omit the NCCL identity")
    for name, suffix in contract.allowed_local_suffixes.items():
        if name not in expected or not re.fullmatch(r"[a-z0-9.]+", suffix):
            raise BootstrapError("response runtime local package suffixes are invalid")


class Runner:
    """Run bounded commands without invoking a shell."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Return stdout or raise a redacted command failure."""

        source_env = os.environ if env is None else env
        process_env = dict(source_env)
        allowed_secrets = _docker_secret_env_names(args)
        for name in PRIVATE_ENV_NAMES:
            if name not in allowed_secrets:
                process_env.pop(name, None)
        for name in INSTALL_OVERRIDE_ENV_NAMES:
            process_env.pop(name, None)
        result = subprocess.run(
            list(args),
            cwd=cwd,
            env=process_env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            command = Path(args[0]).name
            detail = (result.stderr or result.stdout).strip()[-2000:]
            for name in PRIVATE_ENV_NAMES:
                value = source_env.get(name, "")
                if value:
                    detail = detail.replace(value, "[REDACTED]")
            raise BootstrapError(f"{command} failed with exit {result.returncode}: {detail}")
        return result.stdout


def main() -> int:
    """Check the current runtime or create a new environment explicitly."""

    args = _parse_args()
    runner = Runner()
    env = dict(os.environ)
    try:
        if args.launch_setup or args.launch_check:
            inputs = _inputs(args)
            report = launch_environment(
                inputs,
                Path(args.venv),
                runner,
                env,
                setup=args.launch_setup,
                max_build_jobs=args.max_build_jobs,
            )
        elif args.resolve_lock:
            report = resolve_lock(runner, env)
        else:
            inputs = _inputs(args)
            if args.setup:
                report = setup_environment(inputs, Path(args.venv), runner, env, args.max_build_jobs)
            else:
                python = _existing_venv_python(Path(args.venv), inputs)
                report = check_environment(inputs, runner, env, python=python)
                data_python = _existing_data_python(inputs)
                data = _verify_data_runtime(data_python, runner, inputs.rdan_root, env)
                prepared = _prepare_data(python, data_python, inputs, runner, env, check=True)
                report = report | {"data_preparation": prepared, "data_runtime": data}
    except BootstrapError as error:
        print(f"A100 response bootstrap failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


def check_environment(
    inputs: Inputs,
    runner: Runner,
    env: Mapping[str, str],
    *,
    host: HostInfo | None = None,
    python: Path | None = None,
    include_packages: bool = True,
) -> dict[str, Any]:
    """Return a secret-free report only after every requested check passes."""

    contract = _load_runtime_contract()
    credentials = _verify_credentials(env)
    env = _subprocess_env(env)
    host = host or _host_info(env)
    python = (python or Path(sys.executable)).resolve()
    _verify_host(host, contract)
    _verify_inputs(inputs, contract)
    revisions = _verify_repositories(inputs, runner)
    snapshot = _verify_snapshot(inputs.snapshot)
    roots = _verify_roots(inputs)
    readiness = _verify_static_readiness(inputs, runner, env, contract)
    _verify_ambient_roll(python, runner, inputs.rdan_root, env)
    gpu = _verify_gpu(
        python,
        runner,
        inputs.rdan_root,
        env,
        exact_torch=include_packages,
        static=readiness["gpu"],
        contract=contract,
    )
    report = _base_report(contract, host, revisions, snapshot, roots, readiness, credentials, gpu)
    if include_packages:
        report["packages"] = _verify_packages(python, runner, inputs.rdan_root, env, contract)
        report["runtime_imports"] = _verify_runtime_imports(python, runner, inputs, env)
    return report


def _base_report(
    contract: RuntimeContract,
    host: HostInfo,
    revisions: Mapping[str, str],
    snapshot: Mapping[str, Any],
    roots: Mapping[str, Any],
    readiness: Mapping[str, Any],
    credentials: Mapping[str, bool],
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "passed",
        "profile": contract.profile,
        "python": f"{contract.python_version[0]}.{contract.python_version[1]}.x",
        "platform": contract.platform_receipt,
        "compute_contract_sha256": contract.sha256,
        "container": {
            "image": contract.container_image,
            "index_digest": contract.container_index_digest,
            "amd64_digest": contract.container_amd64_digest,
            "cuda": contract.container_cuda,
            "release": contract.container_release,
            "image_id": host.image_id,
            "identity_source": host.identity_source,
        },
        "repositories": revisions,
        "model": snapshot,
        "storage": roots,
        "host_readiness": readiness,
        "capabilities": credentials,
        "gpu": gpu,
        "requirements": {
            path.name: _sha256(path)
            for path in (
                *REQUIREMENTS,
                LOCK,
                LOCK_INPUT,
                BOOTSTRAP_LOCK,
                BOOTSTRAP_INPUT,
                _container_contract_path(),
                DATA_LOCK,
                DATA_REQUIREMENTS,
                SNAPSHOT_MANIFEST,
                COMPUTE_CONTRACT,
                *contract.package_contracts,
            )
        },
    }


def setup_environment(
    inputs: Inputs,
    venv: Path,
    runner: Runner,
    env: Mapping[str, str],
    max_build_jobs: int,
    *,
    host: HostInfo | None = None,
) -> dict[str, Any]:
    """Create a new environment and atomically seal its passing report."""

    contract = _load_runtime_contract()
    credential_env = env
    env = _subprocess_env(credential_env)
    check_environment(inputs, runner, credential_env, host=host, include_packages=False)
    venv = _alias_path(venv)
    data_venv = inputs.cache_root / DATA_ENV_NAME
    marker = inputs.run_root / "a100-response-bootstrap.json"
    _verify_setup_paths(venv, marker, inputs)
    _verify_data_setup_path(data_venv, inputs)
    state = _SetupState(
        stage=Path(tempfile.mkdtemp(prefix=f".{venv.name}.", dir=venv.parent)),
        data_root=None,
        published=[],
    )
    try:
        python = _install_production_environment(state.stage, inputs, runner, env, contract, max_build_jobs)
        state.data_root, data_stage, data_python = _create_data_environment(python, inputs, runner, env)
        data = _verify_data_runtime(data_python, runner, inputs.rdan_root, env)
        prepared = _prepare_data(python, data_python, inputs, runner, env, check=False)
        report = check_environment(inputs, runner, credential_env, host=host, python=python)
        sealed = _publish_setup(state, venv, data_venv, data_stage, report, prepared, data)
        _write_marker(marker, sealed)
    except BaseException:
        _cleanup_setup(state)
        raise
    return sealed | {"marker": str(marker)}


def _install_production_environment(
    stage: Path,
    inputs: Inputs,
    runner: Runner,
    env: Mapping[str, str],
    contract: RuntimeContract,
    max_build_jobs: int,
) -> Path:
    runner.run([sys.executable, "-m", "venv", str(stage)], cwd=inputs.rdan_root, env=env)
    python = stage / "bin/python"
    if not python.is_file():
        raise BootstrapError("venv creation did not produce bin/python")
    build_env = dict(env) | {"MAX_JOBS": str(max_build_jobs)}
    _install_pip_contract(python, BOOTSTRAP_LOCK, runner, env, contract, build=False, dependencies=True)
    _install_pip_contract(python, LOCK, runner, build_env, contract, build=True, dependencies=True)
    _install_pip_contract(python, FLASH_REQUIREMENTS, runner, build_env, contract, build=True, dependencies=False)
    runner.run([str(python), "-m", "pip", "--isolated", "check"], env=env)
    return python


def _install_pip_contract(
    python: Path,
    requirements: Path,
    runner: Runner,
    env: Mapping[str, str],
    contract: RuntimeContract,
    *,
    build: bool,
    dependencies: bool,
) -> None:
    command = [str(python), "-m", "pip", "--isolated", "install", "--index-url", contract.index_url]
    if requirements != FLASH_REQUIREMENTS:
        command.extend(["--extra-index-url", contract.torch_index_url])
    if build:
        command.append("--no-build-isolation")
    if not dependencies:
        command.append("--no-deps")
    command.extend(["--require-hashes", "-r", str(requirements)])
    runner.run(command, env=env)


def _publish_setup(
    state: _SetupState,
    venv: Path,
    data_venv: Path,
    data_stage: Path,
    report: Mapping[str, Any],
    prepared: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    _publish_environment(state.stage, venv)
    state.published.append((venv, state.stage))
    _publish_environment(data_stage, data_venv)
    if state.data_root is None:
        raise BootstrapError("data environment stage is missing")
    state.published.append((data_venv, state.data_root))
    data_python = str(data_venv / "bin" / DATA_PYTHON_BIN)
    return report | {
        "data_preparation": dict(prepared) | {"data_python": data_python},
        "data_runtime": dict(data) | {"python": data_python},
        "venv": str(venv),
    }


def _cleanup_setup(state: _SetupState) -> None:
    for alias, target in reversed(state.published):
        alias.unlink(missing_ok=True)
        if target.exists():
            shutil.rmtree(target)
    if not state.published and state.stage.exists():
        shutil.rmtree(state.stage)
    if state.data_root is not None and state.data_root.exists():
        shutil.rmtree(state.data_root)


def resolve_lock(runner: Runner, env: Mapping[str, str]) -> dict[str, Any]:
    """Regenerate the exact target lock without modifying an environment."""

    contract = _load_runtime_contract()
    env = _subprocess_env(env)
    _verify_host(_host_info(env), contract)
    pairs = (
        (LOCK_INPUT, LOCK, "3.12", True),
        (BOOTSTRAP_INPUT, BOOTSTRAP_LOCK, "3.12", True),
        (DATA_REQUIREMENTS, DATA_LOCK, "3.11", False),
    )
    candidates = [
        (source, target.with_suffix(".lock.new"), version, torch) for source, target, version, torch in pairs
    ]
    if any(output.exists() for _, output, _, _ in candidates):
        raise BootstrapError("refusing to overwrite a pending lock candidate")
    results = []
    for source, output, version, torch in candidates:
        command = [
            "uv",
            "pip",
            "compile",
            str(source),
            "--python-platform",
            "x86_64-manylinux_2_28",
            "--python-version",
            version,
            "--generate-hashes",
            "--output-file",
            str(output),
        ]
        if torch:
            command[command.index("--generate-hashes") : command.index("--generate-hashes")] = [
                "--torch-backend",
                contract.torch_backend,
            ]
        runner.run(command, cwd=REPO_ROOT, env=env)
        results.append({"path": str(output), "sha256": _sha256(output)})
    return {"schema_version": 1, "status": "candidates_generated", "candidates": results}


def launch_environment(
    inputs: Inputs,
    venv: Path,
    runner: Runner,
    env: Mapping[str, str],
    *,
    setup: bool,
    max_build_jobs: int,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, Any]:
    """Inspect the pinned image on the host and run the internal contract."""

    contract = _load_runtime_contract()
    _verify_credentials(env)
    docker_env = _docker_environment(env)
    env = _subprocess_env(env)
    _verify_launch_platform(system or platform.system(), machine or platform.machine(), contract)
    _verify_inputs(inputs, contract)
    _verify_roots(inputs)
    _verify_static_readiness(inputs, runner, env, contract)
    venv = _alias_path(venv)
    marker = inputs.run_root / "a100-response-bootstrap.json"
    if setup:
        _verify_setup_paths(venv, marker, inputs)
    else:
        _existing_venv_python(venv, inputs)
    for path in (inputs.rdan_root, inputs.rtt_root, inputs.snapshot, inputs.cache_root, inputs.run_root, venv):
        if "," in str(path):
            raise BootstrapError("container bind paths cannot contain commas")

    if setup:
        runner.run(["docker", "pull", "--platform", "linux/amd64", contract.container_ref], env=env)
    identity = _inspect_launch_image(runner, env, contract)
    receipt = inputs.run_root / IDENTITY_NAME
    if setup:
        _seal_identity(receipt, identity)
    elif _read_identity(receipt, contract) != identity:
        raise BootstrapError("existing external image identity receipt differs from Docker inspection")
    credential_names = tuple(name for name in SECRET_ENV_NAMES if name in docker_env)
    command = _container_command(inputs, venv, receipt, setup, max_build_jobs, credential_names, contract)
    output = runner.run(command, cwd=inputs.rdan_root, env=docker_env)
    report = _last_json_object(output)
    if (
        report.get("schema_version") != 2
        or report.get("status") != "passed"
        or report.get("container", {}).get("image_id") != identity["image_id"]
    ):
        raise BootstrapError("container bootstrap did not return the inspected image identity")
    return report | {"external_identity_receipt": str(receipt)}


def _verify_launch_platform(system: str, machine: str, contract: RuntimeContract) -> None:
    if system != contract.system or machine not in contract.machines:
        raise BootstrapError("A100 container launch requires a Linux x86_64 host")


def _inspect_launch_image(
    runner: Runner,
    env: Mapping[str, str],
    contract: RuntimeContract,
) -> dict[str, Any]:
    inspected = runner.run(["docker", "image", "inspect", contract.container_ref], env=env)
    return _image_identity(inspected, contract)


def _image_identity(raw: str, contract: RuntimeContract | None = None) -> dict[str, Any]:
    contract = contract or _load_runtime_contract()
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BootstrapError("docker image inspect returned invalid JSON") from error
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise BootstrapError("docker image inspect must return exactly one image")
    image = records[0]
    digests = image.get("RepoDigests")
    expected = contract.container_ref
    if not isinstance(digests, list) or expected not in digests:
        raise BootstrapError("docker inspected image does not have the pinned repository digest")
    image_id = image.get("Id")
    if image.get("Architecture") != "amd64" or image.get("Os") != "linux" or not _DIGEST.fullmatch(str(image_id)):
        raise BootstrapError("docker inspected image is not Linux AMD64 with a content ID")
    return {
        "architecture": "amd64",
        "image_id": image_id,
        "os": "linux",
        "repo_digest": contract.container_amd64_digest,
        "requested_ref": contract.container_ref,
        "schema_version": 1,
        "source": "docker-image-inspect",
    }


def _seal_identity(path: Path, identity: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_identity(path) != identity:
            raise BootstrapError("existing external image identity receipt differs from Docker inspection")
        return
    _write_marker(path, identity)


def _read_identity(path: Path, contract: RuntimeContract | None = None) -> dict[str, Any]:
    contract = contract or _load_runtime_contract()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid external image identity receipt: {error}") from error
    expected = {
        "architecture": "amd64",
        "os": "linux",
        "repo_digest": contract.container_amd64_digest,
        "requested_ref": contract.container_ref,
        "schema_version": 1,
        "source": "docker-image-inspect",
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise BootstrapError("external image identity receipt differs from the pinned image")
    if set(payload) != {*expected, "image_id"} or not _DIGEST.fullmatch(str(payload.get("image_id", ""))):
        raise BootstrapError("external image identity receipt has an invalid image content ID")
    return payload


def _container_command(
    inputs: Inputs,
    venv: Path,
    receipt: Path,
    setup: bool,
    max_build_jobs: int,
    credential_names: Sequence[str],
    contract: RuntimeContract,
) -> list[str]:
    command = _container_command_prefix(inputs)
    for name in credential_names:
        command.extend(["--env", name])
    for source, target, readonly in _container_mounts(inputs, receipt):
        mount = f"type=bind,src={source},dst={target}"
        command.extend(["--mount", f"{mount},readonly" if readonly else mount])
    command.extend(_container_bootstrap_args(inputs, venv, setup, max_build_jobs, contract))
    return command


def _container_mounts(inputs: Inputs, receipt: Path) -> tuple[tuple[Path, Path, bool], ...]:
    return (
        (inputs.rdan_root, inputs.rdan_root, False),
        (inputs.rtt_root, inputs.rtt_root, True),
        (inputs.snapshot, inputs.snapshot, True),
        (inputs.cache_root, inputs.cache_root, False),
        (inputs.run_root, inputs.run_root, False),
        (receipt, IDENTITY_RECEIPT, True),
    )


def _container_command_prefix(inputs: Inputs) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--workdir",
        str(inputs.rdan_root),
        "--env",
        "PYTHONNOUSERSITE=1",
    ]


def _container_bootstrap_args(
    inputs: Inputs,
    venv: Path,
    setup: bool,
    max_build_jobs: int,
    contract: RuntimeContract,
) -> list[str]:
    return [
        contract.container_ref,
        "python3",
        str(inputs.rdan_root / "scripts/bootstrap_a100_response.py"),
        "--setup" if setup else "--check",
        "--rtt-root",
        str(inputs.rtt_root),
        "--rdan-root",
        str(inputs.rdan_root),
        "--rdan-revision",
        inputs.rdan_revision,
        "--snapshot",
        str(inputs.snapshot),
        "--cache-root",
        str(inputs.cache_root),
        "--run-root",
        str(inputs.run_root),
        "--venv",
        str(venv),
        "--max-build-jobs",
        str(max_build_jobs),
    ]


def _last_json_object(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise BootstrapError("container bootstrap did not return a JSON report")


def _create_data_environment(
    production_python: Path,
    inputs: Inputs,
    runner: Runner,
    env: Mapping[str, str],
) -> tuple[Path, Path, Path]:
    root = Path(tempfile.mkdtemp(prefix=f".{DATA_ENV_NAME}.", dir=inputs.cache_root))
    try:
        data_venv, data_python = _populate_data_environment(production_python, root, inputs, runner, env)
    except BaseException:
        if root.exists():
            shutil.rmtree(root)
        raise
    return root, data_venv, data_python


def _populate_data_environment(
    production_python: Path,
    root: Path,
    inputs: Inputs,
    runner: Runner,
    env: Mapping[str, str],
) -> tuple[Path, Path]:
    contract = _load_runtime_contract()
    uv = production_python.parent / "uv"
    if not uv.is_file():
        raise BootstrapError("production environment did not install the pinned uv executable")
    data_venv = root / "venv"
    managed = root / "managed-python"
    cache = root / "uv-cache"
    managed.mkdir(mode=0o755, exist_ok=True)
    uv_env = _data_uv_environment(env, cache, managed)
    _install_data_python(uv, managed, runner, inputs, uv_env)
    _create_data_venv(uv, data_venv, runner, inputs, uv_env)
    data_python = data_venv / "bin/python"
    if not data_python.is_file():
        raise BootstrapError("data-preparation venv creation did not produce bin/python")
    _install_data_packages(uv, data_python, runner, inputs, uv_env, contract)
    data_python = _write_data_python_launcher(data_venv)
    if cache.exists():
        shutil.rmtree(cache)
    return data_venv, data_python


def _data_uv_environment(env: Mapping[str, str], cache: Path, managed: Path) -> dict[str, str]:
    return _subprocess_env(env) | {
        "UV_CACHE_DIR": str(cache),
        "UV_LINK_MODE": "copy",
        "UV_PYTHON_INSTALL_DIR": str(managed),
    }


def _install_data_python(
    uv: Path,
    managed: Path,
    runner: Runner,
    inputs: Inputs,
    env: Mapping[str, str],
) -> None:
    runner.run(
        [
            str(uv),
            "python",
            "install",
            DATA_PYTHON_VERSION,
            "--install-dir",
            str(managed),
            "--managed-python",
            "--no-bin",
            "--no-config",
        ],
        cwd=inputs.rdan_root,
        env=env,
    )


def _create_data_venv(
    uv: Path,
    data_venv: Path,
    runner: Runner,
    inputs: Inputs,
    env: Mapping[str, str],
) -> None:
    runner.run(
        [
            str(uv),
            "venv",
            str(data_venv),
            "--python",
            DATA_PYTHON_VERSION,
            "--managed-python",
            "--no-python-downloads",
            "--relocatable",
            "--no-project",
            "--no-config",
        ],
        cwd=inputs.rdan_root,
        env=env,
    )


def _install_data_packages(
    uv: Path,
    data_python: Path,
    runner: Runner,
    inputs: Inputs,
    env: Mapping[str, str],
    contract: RuntimeContract,
) -> None:
    runner.run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(data_python),
            "--default-index",
            contract.index_url,
            "--require-hashes",
            "--no-config",
            "-r",
            str(DATA_LOCK),
        ],
        cwd=inputs.rdan_root,
        env=env,
    )
    runner.run(
        [str(uv), "pip", "check", "--python", str(data_python), "--no-config"],
        cwd=inputs.rdan_root,
        env=env,
    )


def _write_data_python_launcher(venv: Path) -> Path:
    launcher = venv / "bin" / DATA_PYTHON_BIN
    if launcher.exists() or launcher.is_symlink():
        raise BootstrapError("refusing to overwrite the data Python launcher")
    content = '#!/bin/sh\nexec "$(dirname "$0")/python" "$@"\n'
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=launcher.parent, delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o755)
        os.replace(temporary, launcher)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return launcher


def _verify_data_runtime(
    python: Path,
    runner: Runner,
    cwd: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    output = runner.run([str(python), "-c", _DATA_RUNTIME_PROBE], cwd=cwd, env=_subprocess_env(env))
    payload = _probe_json(output, "RDAN_DATA_RUNTIME=")
    expected = _locked_packages(DATA_LOCK)
    packages = payload.get("packages")
    observed = (
        {_normalize_name(name): version for name, version in packages.items()} if isinstance(packages, dict) else {}
    )
    if payload.get("python") != DATA_PYTHON_VERSION or observed != expected:
        raise BootstrapError("data-preparation runtime differs from the Python 3.11 lock")
    return {"packages": {name: observed[name] for name in sorted(observed)}, "python_version": DATA_PYTHON_VERSION}


def _prepare_data(
    production_python: Path,
    data_python: Path,
    inputs: Inputs,
    runner: Runner,
    env: Mapping[str, str],
    *,
    check: bool,
) -> dict[str, Any]:
    command = [
        str(production_python),
        str(inputs.rdan_root / "scripts/prepare_a100_response_data.py"),
        "--data-python",
        str(data_python),
        "--rtt-root",
        str(inputs.rtt_root),
        "--snapshot",
        str(inputs.snapshot),
    ]
    if check:
        command.append("--check")
    output = runner.run(command, cwd=inputs.rdan_root, env=_subprocess_env(env))
    report = _last_json_object(output)
    if report.get("status") != "passed" or report.get("checked_only") is not check:
        raise BootstrapError("response data preparation did not return a passing report")
    return report | {"data_python": str(data_python)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--setup", action="store_true")
    mode.add_argument("--launch-check", action="store_true")
    mode.add_argument("--launch-setup", action="store_true")
    mode.add_argument("--resolve-lock", action="store_true")
    parser.add_argument("--rtt-root", type=Path)
    parser.add_argument("--rdan-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--rdan-revision")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--max-build-jobs", type=int, default=4)
    args = parser.parse_args()
    if (args.check or args.setup or args.launch_check or args.launch_setup) and args.venv is None:
        parser.error("runtime checks and setup require --venv")
    if not 1 <= args.max_build_jobs <= 16:
        parser.error("--max-build-jobs must be between 1 and 16")
    runtime_args = (args.rtt_root, args.rdan_revision, args.snapshot, args.cache_root, args.run_root)
    if not args.resolve_lock and any(value is None for value in runtime_args):
        parser.error("runtime checks and setup require RTT, revision, snapshot, cache, and run paths")
    if args.resolve_lock and any(value is not None for value in (*runtime_args, args.venv)):
        parser.error("--resolve-lock does not accept runtime paths")
    return args


def _inputs(args: argparse.Namespace) -> Inputs:
    if None in (args.rtt_root, args.rdan_revision, args.snapshot, args.cache_root, args.run_root):
        raise BootstrapError("runtime inputs are incomplete")
    return Inputs(
        rtt_root=args.rtt_root.expanduser().resolve(),
        rdan_root=args.rdan_root.expanduser().resolve(),
        rdan_revision=args.rdan_revision,
        snapshot=args.snapshot.expanduser().resolve(),
        cache_root=args.cache_root.expanduser().resolve(),
        run_root=args.run_root.expanduser().resolve(),
    )


def _host_info(env: Mapping[str, str]) -> HostInfo:
    os_release = _os_release(Path("/etc/os-release"))
    identity = _read_identity(IDENTITY_RECEIPT)
    return HostInfo(
        system=platform.system(),
        machine=platform.machine(),
        implementation=platform.python_implementation(),
        python=tuple(sys.version_info[:3]),
        os_release=os_release,
        container=Path("/.dockerenv").is_file() or _containerized(Path("/proc/1/cgroup")),
        cuda=env.get("CUDA_VERSION"),
        container_release=env.get("NVIDIA_PYTORCH_VERSION"),
        image_digest=identity.get("repo_digest"),
        image_id=identity.get("image_id"),
        identity_source=identity.get("source"),
    )


def _verify_host(host: HostInfo, contract: RuntimeContract) -> None:
    if host.system != contract.system or host.machine not in contract.machines:
        raise BootstrapError("host must be Linux x86_64")
    if host.implementation != "CPython" or host.python[:2] != contract.python_version:
        expected = ".".join(map(str, contract.python_version)) + ".x"
        observed = ".".join(map(str, host.python))
        raise BootstrapError(f"Python must be exact CPython {expected}, got {host.implementation} {observed}")
    if host.os_release.get("ID") != "ubuntu" or host.os_release.get("VERSION_ID") != contract.container_os:
        raise BootstrapError("host must be Ubuntu 24.04")
    if not host.container:
        raise BootstrapError("runtime must execute inside the pinned NGC container")
    if host.cuda != contract.container_cuda or host.container_release != contract.container_release:
        raise BootstrapError("container identity does not match NVIDIA PyTorch 25.06")
    if host.image_digest != contract.container_amd64_digest:
        raise BootstrapError("container image digest does not match the pinned Linux AMD64 manifest")
    if host.identity_source != "docker-image-inspect" or not _DIGEST.fullmatch(host.image_id or ""):
        raise BootstrapError("container image identity was not externally inspected")


def _os_release(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BootstrapError(f"cannot read container operating system identity: {error}") from error
    values: dict[str, str] = {}
    for line in lines:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _containerized(path: Path) -> bool:
    try:
        value = path.read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return any(marker in value for marker in ("docker", "containerd", "kubepods"))


def _verify_inputs(inputs: Inputs, contract: RuntimeContract) -> None:
    if inputs.rdan_root != REPO_ROOT:
        raise BootstrapError("--rdan-root must identify the checkout containing this bootstrap")
    if not _REVISION.fullmatch(inputs.rdan_revision):
        raise BootstrapError("--rdan-revision must be a full lowercase Git commit")
    contracts = (
        *REQUIREMENTS,
        LOCK,
        LOCK_INPUT,
        BOOTSTRAP_LOCK,
        BOOTSTRAP_INPUT,
        _container_contract_path(),
        DATA_LOCK,
        DATA_REQUIREMENTS,
        SNAPSHOT_MANIFEST,
        COMPUTE_CONTRACT,
        *contract.package_contracts,
    )
    for path in contracts:
        if not path.is_file():
            raise BootstrapError(f"missing requirement contract: {path.name}")
    expected = _expected_packages()
    if not expected or len(expected) != sum(
        1 for path in REQUIREMENTS for line in path.read_text().splitlines() if _PIN.match(line)
    ):
        raise BootstrapError("requirement contracts contain duplicate or invalid exact pins")
    _verify_lock(expected, contract)
    _verify_bootstrap_lock()
    _verify_container_contract(contract)
    _verify_data_lock()
    _verify_snapshot_manifest()


def _verify_repositories(inputs: Inputs, runner: Runner) -> dict[str, str]:
    rtt = _verify_repo(inputs.rtt_root, RTT_REVISION, "RTT", runner)
    rdan = _verify_repo(inputs.rdan_root, inputs.rdan_revision, "RDAN", runner)
    return {"rdan": rdan, "rtt": rtt}


def _verify_repo(root: Path, revision: str, name: str, runner: Runner) -> str:
    if not root.is_dir():
        raise BootstrapError(f"{name} checkout is missing")
    top = Path(runner.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"]).strip()).resolve()
    if top != root:
        raise BootstrapError(f"{name} root is not the exact Git top level")
    observed = runner.run(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    if observed != revision:
        raise BootstrapError(f"{name} revision mismatch")
    status = runner.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"])
    if status.strip():
        raise BootstrapError(f"{name} checkout is dirty")
    return observed


def _verify_snapshot(snapshot: Path) -> dict[str, Any]:
    if not snapshot.is_dir() or snapshot.name != MODEL_REVISION:
        raise BootstrapError("model snapshot path must end in the pinned model revision")
    required = ("config.json", "generation_config.json", *TOKENIZER_HASHES)
    if any(not (snapshot / name).is_file() for name in required):
        raise BootstrapError("model snapshot is incomplete")
    _verify_snapshot_weights(snapshot)
    hashes = SNAPSHOT_HASHES | TOKENIZER_HASHES
    if set(hashes) != set(SNAPSHOT_SIZES) or any(not (snapshot / name).is_file() for name in hashes):
        raise BootstrapError("model snapshot is incomplete")
    observed = {path.name for path in snapshot.iterdir() if path.is_file() or path.is_symlink()}
    if observed != set(hashes):
        raise BootstrapError("model snapshot top-level file inventory differs from the pinned revision")
    for name, expected in hashes.items():
        if (snapshot / name).stat().st_size != SNAPSHOT_SIZES[name]:
            raise BootstrapError(f"model snapshot {name} size mismatch")
        if _sha256(snapshot / name) != expected:
            raise BootstrapError(f"model snapshot {name} hash mismatch")
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid model config: {error}") from error
    if not isinstance(config, dict) or config.get("model_type") != "qwen3":
        raise BootstrapError("model snapshot is not the pinned Qwen3 architecture")
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "snapshot": str(snapshot),
        "file_sha256": dict(hashes),
        "tokenizer_sha256": dict(TOKENIZER_HASHES),
    }


def _verify_snapshot_weights(snapshot: Path) -> None:
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        if not (snapshot / "model.safetensors").is_file():
            raise BootstrapError("model snapshot has no complete safetensors weight set")
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid model safetensors index: {error}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise BootstrapError("model safetensors index has no weight map")
    names = list(weight_map.values())
    if any(
        not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") for name in names
    ):
        raise BootstrapError("model safetensors index contains an invalid shard path")
    shards = set(names)
    if any(not (snapshot / name).is_file() for name in shards):
        raise BootstrapError("model snapshot is missing a referenced safetensors shard")
    expected = {name for name in SNAPSHOT_HASHES if name.endswith(".safetensors")}
    if shards != expected:
        raise BootstrapError("model safetensors index differs from the pinned shard inventory")


def _verify_roots(inputs: Inputs) -> dict[str, Any]:
    roots = {"cache": inputs.cache_root, "run": inputs.run_root}
    if _is_within(inputs.cache_root, inputs.run_root) or _is_within(inputs.run_root, inputs.cache_root):
        raise BootstrapError("cache and run roots must be distinct and non-overlapping")
    forbidden = (inputs.rdan_root, inputs.rtt_root, inputs.snapshot)
    report: dict[str, Any] = {}
    for name, root in roots.items():
        if not root.is_dir():
            raise BootstrapError(f"{name} root must already exist")
        if any(_is_within(root, path) or _is_within(path, root) for path in forbidden):
            raise BootstrapError(f"{name} root must be outside repositories and the model snapshot")
        mode = stat.S_IMODE(root.stat().st_mode)
        if not mode & 0o222 or not os.access(root, os.W_OK | os.X_OK):
            raise BootstrapError(f"{name} root is not writable")
        usage = os.statvfs(root)
        report[name] = {"path": str(root), "free_bytes": usage.f_bavail * usage.f_frsize}
    return report


def _verify_ambient_roll(python: Path, runner: Runner, cwd: Path, env: Mapping[str, str]) -> None:
    payload = _probe_json(runner.run([str(python), "-c", _AMBIENT_PROBE], cwd=cwd, env=env), "RDAN_AMBIENT=")
    if payload.get("origin") is not None:
        raise BootstrapError("ambient roll module conflicts with the pinned source checkout")


def _verify_credentials(env: Mapping[str, str]) -> dict[str, bool]:
    present = {
        "judge_access": bool(env.get("OPENROUTER_API_KEY", "").strip()),
        "tracking_access": bool(env.get("WANDB_API_KEY", "").strip()),
        "model_publish_access": any(env.get(name, "").strip() for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")),
    }
    missing = [name for name, value in present.items() if not value]
    if missing:
        raise BootstrapError(f"required service credential capabilities are absent: {missing}")
    return present


def _verify_static_readiness(
    inputs: Inputs,
    runner: Runner,
    env: Mapping[str, str],
    contract: RuntimeContract,
) -> dict[str, Any]:
    memory = runner.run(["free", "--bytes"], cwd=inputs.rdan_root, env=env)
    total_ram = _parse_total_ram(memory)
    if total_ram < contract.minimum_ram_bytes:
        minimum_gib = contract.minimum_ram_bytes // 2**30
        raise BootstrapError(f"host RAM must be at least {minimum_gib} GiB")
    disk: dict[str, dict[str, int]] = {}
    for name, root in (("cache", inputs.cache_root), ("run", inputs.run_root)):
        output = runner.run(
            ["df", "--block-size=1", "--output=avail", str(root)],
            cwd=inputs.rdan_root,
            env=env,
        )
        available = _parse_available_disk(output)
        if available < contract.minimum_free_disk_bytes:
            minimum_gib = contract.minimum_free_disk_bytes // 2**30
            raise BootstrapError(f"{name} filesystem must have at least {minimum_gib} GiB free")
        disk[name] = {"available_bytes": available, "minimum_bytes": contract.minimum_free_disk_bytes}
    return {
        "ram": {"total_bytes": total_ram, "minimum_bytes": contract.minimum_ram_bytes},
        "disk": disk,
        "gpu": _verify_static_gpu(runner, inputs.rdan_root, env, contract),
    }


def _parse_total_ram(output: str) -> int:
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0].rstrip(":") == "Mem" and len(fields) >= 2:
            try:
                return int(fields[1])
            except ValueError as error:
                raise BootstrapError("free returned invalid host RAM evidence") from error
    raise BootstrapError("free did not return host RAM evidence")


def _parse_available_disk(output: str) -> int:
    values = [line.strip() for line in output.splitlines()[1:] if line.strip()]
    if len(values) != 1:
        raise BootstrapError("df must return exactly one filesystem availability value")
    try:
        return int(values[0])
    except ValueError as error:
        raise BootstrapError("df returned invalid filesystem availability evidence") from error


def _verify_static_gpu(
    runner: Runner,
    cwd: Path,
    env: Mapping[str, str],
    contract: RuntimeContract,
) -> dict[str, Any]:
    smi = runner.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=cwd,
        env=env,
    )
    devices = [_parse_smi(line) for line in smi.splitlines() if line.strip()]
    _verify_devices(devices, "nvidia-smi", contract)
    uuids = [device["uuid"] for device in devices]
    if len(set(uuids)) != len(uuids) or any(re.fullmatch(r"GPU-[A-Za-z0-9-]+", value) is None for value in uuids):
        raise BootstrapError("nvidia-smi must report one distinct valid UUID per A100 GPU")
    if contract.require_idle and any(
        device["memory_used_mib"] != 0 or device["utilization_percent"] != 0 for device in devices
    ):
        raise BootstrapError("both A100 GPUs must be idle with zero memory use and zero utilization")
    processes = runner.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        cwd=cwd,
        env=env,
    )
    if processes.strip():
        raise BootstrapError("A100 GPUs must have no active compute processes")
    drivers = {device["driver"] for device in devices}
    if len(drivers) != 1 or _version_tuple(next(iter(drivers))) < contract.minimum_driver:
        raise BootstrapError("NVIDIA driver is below the CUDA 12.9.1 requirement")
    topology = _parse_topology(
        runner.run(["nvidia-smi", "topo", "-m"], cwd=cwd, env=env),
        contract,
    )
    return _static_gpu_report(devices, next(iter(drivers)), topology)


def _static_gpu_report(
    devices: Sequence[Mapping[str, Any]],
    driver: str,
    topology: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "count": len(devices),
        "driver": driver,
        "devices": [
            {
                "index": item["index"],
                "uuid": item["uuid"],
                "name": item["name"],
                "memory_mib": item["memory_mib"],
                "memory_used_mib": item["memory_used_mib"],
                "utilization_percent": item["utilization_percent"],
            }
            for item in devices
        ],
        "compute_process_count": 0,
        "topology": topology,
    }


def _parse_topology(output: str, contract: RuntimeContract) -> dict[str, str]:
    lines = [line.split() for line in output.splitlines() if line.strip()]
    header = next((fields for fields in lines if fields and fields[0] == "GPU0"), None)
    gpu_columns = [] if header is None else [field for field in header if re.fullmatch(r"GPU\d+", field)]
    rows = {fields[0]: fields for fields in lines if fields and re.fullmatch(r"GPU\d+", fields[0])}
    if gpu_columns != ["GPU0", "GPU1"] or set(rows) != {"GPU0", "GPU1"}:
        raise BootstrapError("nvidia-smi topology must contain exactly GPU0 and GPU1")
    if len(rows["GPU0"]) < 3 or len(rows["GPU1"]) < 3:
        raise BootstrapError("nvidia-smi topology matrix is incomplete")
    forward, reverse = rows["GPU0"][2], rows["GPU1"][1]
    if forward != reverse or not _supported_topology_link(forward, contract.supported_links):
        raise BootstrapError("GPU0 and GPU1 must have one exact reciprocal supported topology link")
    if rows["GPU0"][1] != "X" or rows["GPU1"][2] != "X":
        raise BootstrapError("nvidia-smi topology diagonal is invalid")
    return {"gpu0_to_gpu1": forward, "gpu1_to_gpu0": reverse}


def _verify_gpu(
    python: Path,
    runner: Runner,
    cwd: Path,
    env: Mapping[str, str],
    *,
    exact_torch: bool,
    static: Mapping[str, Any],
    contract: RuntimeContract,
) -> dict[str, Any]:
    nvcc = runner.run(["nvcc", "--version"], cwd=cwd, env=env)
    match = re.search(r"release\s+(\d+\.\d+)", nvcc)
    if match is None or match.group(1) != contract.cuda_runtime:
        raise BootstrapError(f"nvcc must be exact CUDA {contract.cuda_runtime}")
    torch = _probe_json(runner.run([str(python), "-c", _CUDA_PROBE], cwd=cwd, env=env), "RDAN_CUDA=")
    if torch.get("cuda") != contract.cuda_runtime:
        raise BootstrapError(f"PyTorch must report CUDA {contract.cuda_runtime}")
    expected_packages = _runtime_expected_packages(contract)
    if exact_torch and not _version_allowed("torch", str(torch.get("torch")), expected_packages, contract):
        raise BootstrapError("PyTorch differs from the response runtime package contract")
    if torch.get("cuda_available") is not True or torch.get("nccl_available") is not True:
        raise BootstrapError("PyTorch CUDA and NCCL runtimes must both be available")
    nccl_version = _required_nccl_version(contract)
    if tuple(torch.get("nccl_version") or ()) != nccl_version:
        expected_nccl = ".".join(map(str, nccl_version))
        raise BootstrapError(f"PyTorch NCCL runtime must be exact {expected_nccl}")
    torch_devices = torch.get("devices")
    if not isinstance(torch_devices, list):
        raise BootstrapError("invalid PyTorch CUDA device report")
    normalized = [
        {"index": item.get("index"), "name": item.get("name"), "memory_mib": item.get("memory_mib")}
        for item in torch_devices
        if isinstance(item, dict)
    ]
    _verify_devices(normalized, "PyTorch", contract)
    return {
        "count": static["count"],
        "cuda": contract.cuda_runtime,
        "driver": static["driver"],
        "devices": static["devices"],
        "nccl": ".".join(map(str, nccl_version)),
        "topology": static["topology"],
        "torch": torch["torch"],
    }


def _parse_smi(line: str) -> dict[str, Any]:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 7:
        raise BootstrapError("invalid nvidia-smi device report")
    try:
        return {
            "index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_mib": int(fields[3]),
            "memory_used_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
            "driver": fields[6],
        }
    except ValueError as error:
        raise BootstrapError("invalid numeric field in nvidia-smi report") from error


def _verify_devices(
    devices: Sequence[Mapping[str, Any]],
    source: str,
    contract: RuntimeContract,
) -> None:
    if len(devices) != contract.gpu_count:
        raise BootstrapError(f"{source} must expose exactly two GPUs")
    for index, device in enumerate(devices):
        if device.get("index") != index:
            raise BootstrapError(f"{source} GPU indexes must be contiguous 0 and 1")
        if contract.gpu_model.upper() not in str(device.get("name", "")).upper():
            raise BootstrapError(f"{source} must expose only A100 GPUs")
        memory = device.get("memory_mib")
        if not isinstance(memory, int) or memory < contract.minimum_gpu_memory_mib:
            minimum_gib = contract.minimum_gpu_memory_mib // 1024
            raise BootstrapError(f"{source} A100 memory must be at least {minimum_gib} GiB per device")


def _required_nccl_version(contract: RuntimeContract) -> tuple[int, ...]:
    version = _runtime_expected_packages(contract).get(contract.nccl_package, "")
    parsed = _version_tuple(version)
    if len(parsed) != 3:
        raise BootstrapError("resolved lock has no exact three-part NCCL runtime identity")
    return parsed


def _verify_packages(
    python: Path,
    runner: Runner,
    cwd: Path,
    env: Mapping[str, str],
    contract: RuntimeContract,
) -> dict[str, str]:
    expected = _runtime_expected_packages(contract)
    runner.run([str(python), "-m", "pip", "--isolated", "check"], cwd=cwd, env=env)
    output = runner.run([str(python), "-c", _PACKAGE_PROBE], cwd=cwd, env=env)
    raw = _probe_json(output, "RDAN_PACKAGES=")
    observed = {_normalize_name(name): value for name, value in raw.items()}
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise BootstrapError(f"installed distribution set mismatch: missing={missing}, extra={extra}")
    for name in expected:
        if not _version_allowed(name, str(observed.get(name)), expected, contract):
            raise BootstrapError(f"installed distribution mismatch for {name}")
    return {name: str(observed[name]) for name in sorted(expected)}


def _verify_runtime_imports(
    python: Path,
    runner: Runner,
    inputs: Inputs,
    env: Mapping[str, str],
) -> dict[str, Any]:
    clean_env = dict(env)
    clean_env.pop("PYTHONPATH", None)
    output = runner.run(
        [str(python), "-c", _IMPORT_PROBE, str(inputs.rtt_root), str(inputs.rdan_root)],
        cwd=inputs.rdan_root,
        env=clean_env,
    )
    payload = _probe_json(output, "RDAN_IMPORTS=")
    roll = Path(str(payload.get("roll_origin", ""))).resolve()
    rdan = Path(str(payload.get("rdan_origin", ""))).resolve()
    if not _is_within(roll, inputs.rtt_root) or not _is_within(rdan, inputs.rdan_root / "src"):
        raise BootstrapError("production imports did not resolve to the pinned source checkouts")
    if payload.get("sdpa") is not True:
        raise BootstrapError("PyTorch scaled dot product attention is unavailable")
    expected = {
        "fsdp2": "FSDP2TrainStrategy",
        "hf": "HfInferStrategy",
        "infer_worker": "ResponseInferWorker",
        "pipeline": "ResponseTrainingPipeline",
        "reward_worker": "RTTCompatibleRubricRewardWorker",
        "train_worker": "ResponseActorWorker",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise BootstrapError("production FSDP2/HF response modules failed to import exactly")
    return payload


def _verify_setup_paths(venv: Path, marker: Path, inputs: Inputs) -> None:
    if venv.exists() or venv.is_symlink():
        raise BootstrapError("refusing to overwrite an existing virtual environment")
    if marker.exists():
        raise BootstrapError("refusing to overwrite an existing bootstrap marker")
    if not venv.parent.is_dir() or not _is_within(venv, inputs.cache_root):
        raise BootstrapError("venv parent must exist under the selected cache root")
    if marker.parent.resolve() != inputs.run_root:
        raise BootstrapError("bootstrap marker must live directly under the run root")


def _verify_data_setup_path(venv: Path, inputs: Inputs) -> None:
    if venv.exists() or venv.is_symlink():
        raise BootstrapError("refusing to overwrite an existing data-preparation environment")
    if venv.parent.resolve() != inputs.cache_root:
        raise BootstrapError("data-preparation environment must live directly under the cache root")


def _publish_environment(stage: Path, alias: Path) -> None:
    if alias.exists() or alias.is_symlink():
        raise BootstrapError("refusing to overwrite an environment during atomic publication")
    with tempfile.NamedTemporaryFile(dir=alias.parent, prefix=f".{alias.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    temporary.unlink()
    try:
        os.symlink(stage, temporary, target_is_directory=True)
        os.replace(temporary, alias)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_venv_python(venv: Path, inputs: Inputs) -> Path:
    alias = _alias_path(venv)
    resolved = alias.resolve()
    python = resolved / "bin/python"
    if not alias.is_symlink() or not _is_within(resolved, inputs.cache_root) or not python.is_file():
        raise BootstrapError("existing venv must contain bin/python under the selected cache root")
    return python


def _existing_data_python(inputs: Inputs) -> Path:
    alias = _alias_path(inputs.cache_root / DATA_ENV_NAME)
    resolved = alias.resolve()
    python = resolved / "bin" / DATA_PYTHON_BIN
    if (
        not alias.is_symlink()
        or not _is_within(resolved, inputs.cache_root)
        or python.is_symlink()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise BootstrapError("existing data environment must contain a regular executable launcher")
    return python


def _subprocess_env(env: Mapping[str, str]) -> dict[str, str]:
    clean = dict(env)
    for name in (*PRIVATE_ENV_NAMES, *INSTALL_OVERRIDE_ENV_NAMES):
        clean.pop(name, None)
    return clean


def _docker_environment(env: Mapping[str, str]) -> dict[str, str]:
    clean = _subprocess_env(env)
    for name in SECRET_ENV_NAMES:
        if env.get(name, "").strip():
            clean[name] = env[name]
    return clean


def _docker_secret_env_names(args: Sequence[str]) -> set[str]:
    if list(args[:2]) != ["docker", "run"]:
        return set()
    return {
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "--env" and args[index + 1] in SECRET_ENV_NAMES
    }


def _alias_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.parent.resolve() / expanded.name


def _write_marker(path: Path, report: Mapping[str, Any]) -> None:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _expected_packages() -> dict[str, str]:
    expected: dict[str, str] = {}
    for path in REQUIREMENTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _PIN.match(line)
            if match is None:
                continue
            name = _normalize_name(match.group(1))
            if name in expected:
                raise BootstrapError(f"duplicate exact requirement pin: {name}")
            expected[name] = match.group(2)
    return expected


def _verify_lock(expected: Mapping[str, str], contract: RuntimeContract) -> None:
    locked = _locked_packages()
    for name in expected:
        if name == "flash-attn":
            continue
        if not _version_allowed(name, str(locked.get(name)), expected, contract):
            raise BootstrapError(f"resolved lock differs from direct contract for {name}")


def _locked_packages(path: Path = LOCK) -> dict[str, str]:
    locked: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match is not None:
            locked[_normalize_name(match.group(1))] = match.group(2)
    return locked


def _verify_bootstrap_lock() -> None:
    main = _locked_packages()
    bootstrap = _locked_packages(BOOTSTRAP_LOCK)
    if not bootstrap:
        raise BootstrapError("bootstrap lock contains no packages")
    drift = sorted(name for name, version in bootstrap.items() if main.get(name) != version)
    if drift:
        raise BootstrapError(f"bootstrap lock differs from the full lock: {drift}")


def _verify_data_lock() -> None:
    direct = _packages_from(DATA_REQUIREMENTS)
    locked = _locked_packages(DATA_LOCK)
    if direct != DATA_PACKAGE_PINS:
        raise BootstrapError("data-preparation direct contract differs from the frozen detector runtime")
    if not locked:
        raise BootstrapError("data-preparation lock contains no packages")
    drift = sorted(name for name, version in direct.items() if locked.get(name) != version)
    if drift:
        raise BootstrapError(f"data-preparation lock differs from its direct contract: {drift}")


def _packages_from(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match is not None:
            packages[_normalize_name(match.group(1))] = match.group(2)
    return packages


def _verify_container_contract(contract: RuntimeContract) -> None:
    payload = _load_json_object(_container_contract_path(), "container contract")
    expected = {
        "cuda": contract.container_cuda,
        "image": contract.container_image,
        "linux_amd64_digest": contract.container_amd64_digest,
        "manifest_digest": contract.container_index_digest,
        "nvidia_pytorch_release": contract.container_release,
        "os": {"id": "ubuntu", "version": contract.container_os},
        "python": {"major": contract.python_version[0], "minor": contract.python_version[1]},
        "schema_version": 1,
    }
    if payload != expected:
        raise BootstrapError("container contract differs from the bootstrap runtime")


def _verify_snapshot_manifest() -> None:
    try:
        payload = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid model snapshot manifest: {error}") from error
    hashes = SNAPSHOT_HASHES | TOKENIZER_HASHES
    expected = {
        "files": {name: {"sha256": hashes[name], "size": SNAPSHOT_SIZES[name]} for name in sorted(hashes)},
        "model": MODEL,
        "revision": MODEL_REVISION,
        "schema_version": 1,
    }
    if payload != expected:
        raise BootstrapError("model snapshot manifest differs from the pinned revision contract")


def _probe_json(output: str, prefix: str) -> dict[str, Any]:
    lines = [line[len(prefix) :] for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise BootstrapError(f"runtime probe did not emit exactly one {prefix[:-1]} record")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise BootstrapError(f"runtime probe emitted invalid {prefix[:-1]} JSON") from error
    if not isinstance(payload, dict):
        raise BootstrapError(f"runtime probe {prefix[:-1]} record must be an object")
    return payload


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid {name}: {error}") from error
    if not isinstance(payload, dict):
        raise BootstrapError(f"{name} must contain one JSON object")
    return payload


def _required_mapping(value: Mapping[str, Any], key: str, name: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise BootstrapError(f"{name}.{key} must be an object")
    return item


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise BootstrapError(f"runtime contract {key} must be a non-empty string")
    return item


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise BootstrapError(f"runtime contract {key} must be a positive integer")
    return item


def _required_digest(value: Mapping[str, Any], key: str) -> str:
    item = _required_string(value, key)
    if not _DIGEST.fullmatch(item):
        raise BootstrapError(f"runtime contract {key} must be a SHA256 digest")
    return item


def _required_url(value: Mapping[str, Any], key: str) -> str:
    item = _required_string(value, key)
    if re.fullmatch(r"https://[A-Za-z0-9./_-]+", item) is None:
        raise BootstrapError(f"runtime contract {key} must be an HTTPS URL")
    return item


def _required_version(value: Mapping[str, Any], key: str) -> tuple[int, ...]:
    item = _required_string(value, key)
    parsed = _version_tuple(item)
    if len(parsed) != 3 or re.fullmatch(r"\d+\.\d+\.\d+", item) is None:
        raise BootstrapError(f"runtime contract {key} must be a three-part version")
    return parsed


def _string_list(value: Mapping[str, Any], key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list) or not item or any(not isinstance(entry, str) or not entry for entry in item):
        raise BootstrapError(f"runtime contract {key} must be a non-empty string list")
    return item


def _string_mapping(value: Mapping[str, Any], key: str) -> dict[str, str]:
    item = value.get(key)
    invalid = isinstance(item, Mapping) and any(
        not isinstance(name, str) or not isinstance(pin, str) for name, pin in item.items()
    )
    if not isinstance(item, Mapping) or invalid:
        raise BootstrapError(f"runtime contract {key} must map strings to strings")
    return dict(item)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise BootstrapError(f"{name} fields differ from the supported contract")


def _contract_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BootstrapError(f"{name} path must be a non-empty string")
    path = (COMPUTE_CONTRACT.parent / value).resolve()
    if not _is_within(path, REPO_ROOT) or not path.is_file() or path.is_symlink():
        raise BootstrapError(f"{name} must be a regular file inside the repository")
    return path


def _container_contract_path() -> Path:
    compute = _load_json_object(COMPUTE_CONTRACT, "compute contract")
    runtime = _required_mapping(compute, "response_runtime", "compute contract")
    platform_contract = _required_mapping(runtime, "platform", "response runtime")
    return _contract_path(platform_contract.get("container_contract"), "container contract")


def _container_python(container: Mapping[str, Any]) -> tuple[int, int]:
    python = _required_mapping(container, "python", "container contract")
    _exact_keys(python, {"major", "minor"}, "container Python")
    return (_positive_int(python, "major"), _positive_int(python, "minor"))


def _container_os(container: Mapping[str, Any]) -> str:
    operating_system = _required_mapping(container, "os", "container contract")
    _exact_keys(operating_system, {"id", "version"}, "container operating system")
    if operating_system.get("id") != "ubuntu":
        raise BootstrapError("response runtime requires Ubuntu")
    return _required_string(operating_system, "version")


def _runtime_expected_packages(contract: RuntimeContract) -> dict[str, str]:
    expected: dict[str, str] = {}
    for path in contract.package_contracts:
        for name, version in _packages_from(path).items():
            if name in expected and expected[name] != version:
                raise BootstrapError(f"conflicting response runtime package identity: {name}")
            expected[name] = version
    return expected


def _version_allowed(
    name: str,
    actual: str,
    expected: Mapping[str, str],
    contract: RuntimeContract,
) -> bool:
    version = expected.get(name)
    if version is None:
        return False
    allowed = {version}
    suffix = contract.allowed_local_suffixes.get(name)
    if suffix:
        allowed.add(f"{version}+{suffix}")
    return actual in allowed


def _supported_topology_link(value: str, prefixes: Sequence[str]) -> bool:
    return any(value == prefix or (prefix == "NV" and re.fullmatch(r"NV\d+", value)) for prefix in prefixes)


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(map(int, match.group(1).split("."))) if match else ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if not _SHA.fullmatch(value):
        raise BootstrapError(f"invalid SHA256 for {path.name}")
    return value


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
