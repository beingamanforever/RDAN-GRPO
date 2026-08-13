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
CONTAINER_CONTRACT = REPO_ROOT / "requirements/a100-response-container.json"
SNAPSHOT_MANIFEST = REPO_ROOT / "requirements/qwen3-4b-instruct-2507-snapshot.json"
DATA_REQUIREMENTS = REPO_ROOT / "requirements/data-prep-py311-direct.txt"
DATA_LOCK = REPO_ROOT / "requirements/data-prep-linux-py311.lock"
TORCH_REQUIREMENTS = REQUIREMENTS[0]
DIRECT_REQUIREMENTS = REQUIREMENTS[1]
FLASH_REQUIREMENTS = REQUIREMENTS[2]

PYTHON_VERSION = (3, 12)
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
SECRET_ENV_NAMES = ("OPENROUTER_API_KEY", "WANDB_API_KEY")
PRIVATE_ENV_NAMES = (*SECRET_ENV_NAMES, "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY")
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
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
PYPI_INDEX = "https://pypi.org/simple"
CUDA_VERSION = "12.9"
MIN_DRIVER = (575, 57, 8)
MIN_GPU_MIB = 79 * 1024
GPU_COUNT = 2
CONTAINER_IMAGE = "nvcr.io/nvidia/pytorch:25.06-py3"
CONTAINER_INDEX_DIGEST = "sha256:025d9b102b5436d4af8af58f12c6a46b7e5d16f19543b1d2cc4446bf2650b4f1"
CONTAINER_AMD64_DIGEST = "sha256:3cb18e2c438db8af2d3a659ca27fac5da328640261c38c48a34edcd223c38af9"
CONTAINER_REF = f"nvcr.io/nvidia/pytorch@{CONTAINER_AMD64_DIGEST}"
CONTAINER_CUDA = "12.9.1.010"
CONTAINER_OS = "24.04"
CONTAINER_RELEASE = "25.06"
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
        for name in (*PRIVATE_ENV_NAMES, *INSTALL_OVERRIDE_ENV_NAMES):
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

    env = _subprocess_env(env)
    host = host or _host_info(env)
    python = (python or Path(sys.executable)).resolve()
    _verify_host(host)
    _verify_inputs(inputs)
    revisions = _verify_repositories(inputs, runner)
    snapshot = _verify_snapshot(inputs.snapshot)
    roots = _verify_roots(inputs)
    _verify_ambient_roll(python, runner, inputs.rdan_root, env)
    gpu = _verify_gpu(python, runner, inputs.rdan_root, env, exact_torch=include_packages)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "profile": "fsdp2-hf-sdpa-2xa100",
        "python": f"{PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}.x",
        "platform": "Ubuntu-24.04-x86_64",
        "container": {
            "image": CONTAINER_IMAGE,
            "index_digest": CONTAINER_INDEX_DIGEST,
            "amd64_digest": CONTAINER_AMD64_DIGEST,
            "cuda": CONTAINER_CUDA,
            "release": CONTAINER_RELEASE,
            "image_id": host.image_id,
            "identity_source": host.identity_source,
        },
        "repositories": revisions,
        "model": snapshot,
        "storage": roots,
        "gpu": gpu,
        "requirements": {
            path.name: _sha256(path)
            for path in (
                *REQUIREMENTS,
                LOCK,
                LOCK_INPUT,
                BOOTSTRAP_LOCK,
                BOOTSTRAP_INPUT,
                CONTAINER_CONTRACT,
                DATA_LOCK,
                DATA_REQUIREMENTS,
                SNAPSHOT_MANIFEST,
            )
        },
    }
    if include_packages:
        packages = _verify_packages(python, runner, inputs.rdan_root, env)
        imports = _verify_runtime_imports(python, runner, inputs, env)
        report["packages"] = packages
        report["runtime_imports"] = imports
    return report


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

    env = _subprocess_env(env)
    check_environment(inputs, runner, env, host=host, include_packages=False)
    venv = _alias_path(venv)
    data_venv = inputs.cache_root / DATA_ENV_NAME
    marker = inputs.run_root / "a100-response-bootstrap.json"
    _verify_setup_paths(venv, marker, inputs)
    _verify_data_setup_path(data_venv, inputs)
    stage = Path(tempfile.mkdtemp(prefix=f".{venv.name}.", dir=venv.parent))
    data_stage_root: Path | None = None
    published: list[tuple[Path, Path]] = []
    try:
        runner.run([sys.executable, "-m", "venv", str(stage)], cwd=inputs.rdan_root, env=env)
        python = stage / "bin/python"
        if not python.is_file():
            raise BootstrapError("venv creation did not produce bin/python")
        runner.run(
            [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--index-url",
                PYPI_INDEX,
                "--extra-index-url",
                TORCH_INDEX,
                "--require-hashes",
                "-r",
                str(BOOTSTRAP_LOCK),
            ],
            env=env,
        )
        runner.run(
            [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--index-url",
                PYPI_INDEX,
                "--extra-index-url",
                TORCH_INDEX,
                "--no-build-isolation",
                "--require-hashes",
                "-r",
                str(LOCK),
            ],
            env=dict(env) | {"MAX_JOBS": str(max_build_jobs)},
        )
        runner.run(
            [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--index-url",
                PYPI_INDEX,
                "--no-build-isolation",
                "--no-deps",
                "--require-hashes",
                "-r",
                str(FLASH_REQUIREMENTS),
            ],
            env=dict(env) | {"MAX_JOBS": str(max_build_jobs)},
        )
        runner.run([str(python), "-m", "pip", "--isolated", "check"], env=env)
        data_stage_root, data_stage, data_python = _create_data_environment(python, inputs, runner, env)
        data = _verify_data_runtime(data_python, runner, inputs.rdan_root, env)
        prepared = _prepare_data(python, data_python, inputs, runner, env, check=False)
        report = check_environment(inputs, runner, env, host=host, python=python)
        _publish_environment(stage, venv)
        published.append((venv, stage))
        _publish_environment(data_stage, data_venv)
        published.append((data_venv, data_stage_root))
        sealed = report | {
            "data_preparation": prepared | {"data_python": str(data_venv / "bin" / DATA_PYTHON_BIN)},
            "data_runtime": data | {"python": str(data_venv / "bin" / DATA_PYTHON_BIN)},
            "venv": str(venv),
        }
        _write_marker(marker, sealed)
    except BaseException:
        for alias, target in reversed(published):
            alias.unlink(missing_ok=True)
            if target.exists():
                shutil.rmtree(target)
        if not published and stage.exists():
            shutil.rmtree(stage)
        if data_stage_root is not None and data_stage_root.exists():
            shutil.rmtree(data_stage_root)
        raise
    return sealed | {"marker": str(marker)}


def resolve_lock(runner: Runner, env: Mapping[str, str]) -> dict[str, Any]:
    """Regenerate the exact target lock without modifying an environment."""

    env = _subprocess_env(env)
    _verify_host(_host_info(env))
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
                "cu129",
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

    env = _subprocess_env(env)
    system = system or platform.system()
    machine = machine or platform.machine()
    if system != "Linux" or machine not in {"x86_64", "AMD64"}:
        raise BootstrapError("A100 container launch requires a Linux x86_64 host")
    _verify_inputs(inputs)
    _verify_roots(inputs)
    venv = _alias_path(venv)
    marker = inputs.run_root / "a100-response-bootstrap.json"
    if setup:
        _verify_setup_paths(venv, marker, inputs)
    else:
        _existing_venv_python(venv, inputs)
    for path in (inputs.rdan_root, inputs.rtt_root, inputs.snapshot, inputs.cache_root, inputs.run_root, venv):
        if "," in str(path):
            raise BootstrapError("container bind paths cannot contain commas")

    runner.run(["docker", "pull", "--platform", "linux/amd64", CONTAINER_REF], env=env)
    inspected = runner.run(["docker", "image", "inspect", CONTAINER_REF], env=env)
    identity = _image_identity(inspected)
    receipt = inputs.run_root / IDENTITY_NAME
    _seal_identity(receipt, identity)
    command = _container_command(inputs, venv, receipt, setup, max_build_jobs)
    output = runner.run(command, cwd=inputs.rdan_root, env=env)
    report = _last_json_object(output)
    if report.get("status") != "passed" or report.get("container", {}).get("image_id") != identity["image_id"]:
        raise BootstrapError("container bootstrap did not return the inspected image identity")
    return report | {"external_identity_receipt": str(receipt)}


def _image_identity(raw: str) -> dict[str, Any]:
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BootstrapError("docker image inspect returned invalid JSON") from error
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise BootstrapError("docker image inspect must return exactly one image")
    image = records[0]
    digests = image.get("RepoDigests")
    expected = f"nvcr.io/nvidia/pytorch@{CONTAINER_AMD64_DIGEST}"
    if not isinstance(digests, list) or expected not in digests:
        raise BootstrapError("docker inspected image does not have the pinned repository digest")
    image_id = image.get("Id")
    if image.get("Architecture") != "amd64" or image.get("Os") != "linux" or not _DIGEST.fullmatch(str(image_id)):
        raise BootstrapError("docker inspected image is not Linux AMD64 with a content ID")
    return {
        "architecture": "amd64",
        "image_id": image_id,
        "os": "linux",
        "repo_digest": CONTAINER_AMD64_DIGEST,
        "requested_ref": CONTAINER_REF,
        "schema_version": 1,
        "source": "docker-image-inspect",
    }


def _seal_identity(path: Path, identity: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_identity(path) != identity:
            raise BootstrapError("existing external image identity receipt differs from Docker inspection")
        return
    _write_marker(path, identity)


def _read_identity(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid external image identity receipt: {error}") from error
    expected = {
        "architecture": "amd64",
        "os": "linux",
        "repo_digest": CONTAINER_AMD64_DIGEST,
        "requested_ref": CONTAINER_REF,
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
) -> list[str]:
    mounts = (
        (inputs.rdan_root, inputs.rdan_root, False),
        (inputs.rtt_root, inputs.rtt_root, True),
        (inputs.snapshot, inputs.snapshot, True),
        (inputs.cache_root, inputs.cache_root, False),
        (inputs.run_root, inputs.run_root, False),
        (receipt, IDENTITY_RECEIPT, True),
    )
    command = [
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
    for source, target, readonly in mounts:
        mount = f"type=bind,src={source},dst={target}"
        command.extend(["--mount", f"{mount},readonly" if readonly else mount])
    command.extend(
        [
            CONTAINER_REF,
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
    )
    return command


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
    uv = production_python.parent / "uv"
    if not uv.is_file():
        raise BootstrapError("production environment did not install the pinned uv executable")
    data_venv = root / "venv"
    managed = root / "managed-python"
    cache = root / "uv-cache"
    managed.mkdir(mode=0o755, exist_ok=True)
    uv_env = _subprocess_env(env) | {
        "UV_CACHE_DIR": str(cache),
        "UV_LINK_MODE": "copy",
        "UV_PYTHON_INSTALL_DIR": str(managed),
    }
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
        env=uv_env,
    )
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
        env=uv_env,
    )
    data_python = data_venv / "bin/python"
    if not data_python.is_file():
        raise BootstrapError("data-preparation venv creation did not produce bin/python")
    runner.run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(data_python),
            "--default-index",
            PYPI_INDEX,
            "--require-hashes",
            "--no-config",
            "-r",
            str(DATA_LOCK),
        ],
        cwd=inputs.rdan_root,
        env=uv_env,
    )
    runner.run(
        [str(uv), "pip", "check", "--python", str(data_python), "--no-config"],
        cwd=inputs.rdan_root,
        env=uv_env,
    )
    data_python = _write_data_python_launcher(data_venv)
    if cache.exists():
        shutil.rmtree(cache)
    return data_venv, data_python


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


def _verify_host(host: HostInfo) -> None:
    if host.system != "Linux" or host.machine not in {"x86_64", "AMD64"}:
        raise BootstrapError("host must be Linux x86_64")
    if host.implementation != "CPython" or host.python[:2] != PYTHON_VERSION:
        expected = ".".join(map(str, PYTHON_VERSION)) + ".x"
        observed = ".".join(map(str, host.python))
        raise BootstrapError(f"Python must be exact CPython {expected}, got {host.implementation} {observed}")
    if host.os_release.get("ID") != "ubuntu" or host.os_release.get("VERSION_ID") != CONTAINER_OS:
        raise BootstrapError("host must be Ubuntu 24.04")
    if not host.container:
        raise BootstrapError("runtime must execute inside the pinned NGC container")
    if host.cuda != CONTAINER_CUDA or host.container_release != CONTAINER_RELEASE:
        raise BootstrapError("container identity does not match NVIDIA PyTorch 25.06")
    if host.image_digest != CONTAINER_AMD64_DIGEST:
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


def _verify_inputs(inputs: Inputs) -> None:
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
        CONTAINER_CONTRACT,
        DATA_LOCK,
        DATA_REQUIREMENTS,
        SNAPSHOT_MANIFEST,
    )
    for path in contracts:
        if not path.is_file():
            raise BootstrapError(f"missing requirement contract: {path.name}")
    expected = _expected_packages()
    if not expected or len(expected) != sum(
        1 for path in REQUIREMENTS for line in path.read_text().splitlines() if _PIN.match(line)
    ):
        raise BootstrapError("requirement contracts contain duplicate or invalid exact pins")
    _verify_lock(expected)
    _verify_bootstrap_lock()
    _verify_container_contract()
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


def _verify_gpu(
    python: Path,
    runner: Runner,
    cwd: Path,
    env: Mapping[str, str],
    *,
    exact_torch: bool,
) -> dict[str, Any]:
    smi = runner.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=cwd,
        env=env,
    )
    devices = [_parse_smi(line) for line in smi.splitlines() if line.strip()]
    _verify_devices(devices, "nvidia-smi")
    drivers = {device["driver"] for device in devices}
    if len(drivers) != 1 or _version_tuple(next(iter(drivers))) < MIN_DRIVER:
        raise BootstrapError("NVIDIA driver is below the CUDA 12.9.1 requirement")
    nvcc = runner.run(["nvcc", "--version"], cwd=cwd, env=env)
    match = re.search(r"release\s+(\d+\.\d+)", nvcc)
    if match is None or match.group(1) != CUDA_VERSION:
        raise BootstrapError("nvcc must be exact CUDA 12.9")
    torch = _probe_json(runner.run([str(python), "-c", _CUDA_PROBE], cwd=cwd, env=env), "RDAN_CUDA=")
    if torch.get("cuda") != CUDA_VERSION:
        raise BootstrapError("PyTorch must report CUDA 12.9")
    if exact_torch and torch.get("torch") != "2.8.0+cu129":
        raise BootstrapError("PyTorch must be exact 2.8.0+cu129")
    if torch.get("cuda_available") is not True or torch.get("nccl_available") is not True:
        raise BootstrapError("PyTorch CUDA and NCCL runtimes must both be available")
    torch_devices = torch.get("devices")
    if not isinstance(torch_devices, list):
        raise BootstrapError("invalid PyTorch CUDA device report")
    normalized = [
        {"index": item.get("index"), "name": item.get("name"), "memory_mib": item.get("memory_mib")}
        for item in torch_devices
        if isinstance(item, dict)
    ]
    _verify_devices(normalized, "PyTorch")
    return {
        "count": GPU_COUNT,
        "cuda": CUDA_VERSION,
        "driver": next(iter(drivers)),
        "devices": [
            {"index": item["index"], "name": item["name"], "memory_mib": item["memory_mib"]} for item in devices
        ],
        "torch": torch["torch"],
    }


def _parse_smi(line: str) -> dict[str, Any]:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4:
        raise BootstrapError("invalid nvidia-smi device report")
    try:
        return {"index": int(fields[0]), "name": fields[1], "memory_mib": int(fields[2]), "driver": fields[3]}
    except ValueError as error:
        raise BootstrapError("invalid numeric field in nvidia-smi report") from error


def _verify_devices(devices: Sequence[Mapping[str, Any]], source: str) -> None:
    if len(devices) != GPU_COUNT:
        raise BootstrapError(f"{source} must expose exactly two GPUs")
    for index, device in enumerate(devices):
        if device.get("index") != index:
            raise BootstrapError(f"{source} GPU indexes must be contiguous 0 and 1")
        if "A100" not in str(device.get("name", "")).upper():
            raise BootstrapError(f"{source} must expose only A100 GPUs")
        memory = device.get("memory_mib")
        if not isinstance(memory, int) or memory < MIN_GPU_MIB:
            raise BootstrapError(f"{source} A100 memory must be at least 79 GiB per device")


def _verify_packages(python: Path, runner: Runner, cwd: Path, env: Mapping[str, str]) -> dict[str, str]:
    expected = _locked_packages() | {"flash-attn": _expected_packages()["flash-attn"]}
    runner.run([str(python), "-m", "pip", "--isolated", "check"], cwd=cwd, env=env)
    output = runner.run([str(python), "-c", _PACKAGE_PROBE], cwd=cwd, env=env)
    raw = _probe_json(output, "RDAN_PACKAGES=")
    observed = {_normalize_name(name): value for name, value in raw.items()}
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise BootstrapError(f"installed distribution set mismatch: missing={missing}, extra={extra}")
    for name, version in expected.items():
        actual = observed.get(name)
        allowed = {version}
        if name in {"torch", "torchvision", "torchaudio"}:
            allowed.add(f"{version}+cu129")
        if actual not in allowed:
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


def _verify_lock(expected: Mapping[str, str]) -> None:
    locked = _locked_packages()
    for name, version in expected.items():
        if name == "flash-attn":
            continue
        allowed = {version}
        if name in {"torch", "torchvision", "torchaudio"}:
            allowed.add(f"{version}+cu129")
        if locked.get(name) not in allowed:
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


def _verify_container_contract() -> None:
    try:
        payload = json.loads(CONTAINER_CONTRACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid container contract: {error}") from error
    expected = {
        "cuda": CONTAINER_CUDA,
        "image": CONTAINER_IMAGE,
        "linux_amd64_digest": CONTAINER_AMD64_DIGEST,
        "manifest_digest": CONTAINER_INDEX_DIGEST,
        "nvidia_pytorch_release": CONTAINER_RELEASE,
        "os": {"id": "ubuntu", "version": CONTAINER_OS},
        "python": {"major": PYTHON_VERSION[0], "minor": PYTHON_VERSION[1]},
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
