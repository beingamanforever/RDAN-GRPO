from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bootstrap_a100_response.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bootstrap_a100_response", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BOOTSTRAP = _load_module()
RUNTIME = BOOTSTRAP._load_runtime_contract()


class FakeRunner:
    def __init__(self, inputs: Any) -> None:
        self.inputs = inputs
        self.revisions = {
            inputs.rtt_root: BOOTSTRAP.RTT_REVISION,
            inputs.rdan_root: inputs.rdan_revision,
        }
        self.dirty: Path | None = None
        self.gpu_name = "NVIDIA A100-SXM4-80GB"
        self.gpu_count = 2
        self.gpu_memory_mib = 81920
        self.gpu_memory_used_mib = 0
        self.gpu_utilization_percent = 0
        self.compute_processes = ""
        self.topology_forward = "PIX"
        self.topology_reverse = "PIX"
        self.driver = "575.57.08"
        self.cuda = "12.9"
        self.nccl_version = [2, 27, 3]
        self.host_ram_bytes = 210 * 2**30
        self.free_disk_bytes = {inputs.cache_root: 2_700 * 2**30, inputs.run_root: 2_700 * 2**30}
        self.package_overrides: dict[str, str | None] = {}
        self.ambient_origin: str | None = None
        self.fail_next_install = False
        self.commands: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        del cwd
        command = list(args)
        self.commands.append(command)
        self.envs.append(dict(env or {}))
        if command[0] == "git":
            return self._git(command)
        if command[:2] == ["free", "--bytes"]:
            return f"              total used free\nMem: {self.host_ram_bytes} 0 {self.host_ram_bytes}\n"
        if command[:3] == ["df", "--block-size=1", "--output=avail"]:
            return f"Avail\n{self.free_disk_bytes[Path(command[-1])]}\n"
        if command[:2] == [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        ]:
            return "\n".join(
                f"{index}, GPU-{index}, {self.gpu_name}, {self.gpu_memory_mib}, "
                f"{self.gpu_memory_used_mib}, {self.gpu_utilization_percent}, {self.driver}"
                for index in range(self.gpu_count)
            )
        if command[:2] == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name"]:
            return self.compute_processes
        if command == ["nvidia-smi", "topo", "-m"]:
            return (
                "        GPU0 GPU1 CPU Affinity\n"
                f"GPU0    X    {self.topology_forward}   0-31\n"
                f"GPU1    {self.topology_reverse}  X     0-31\n"
            )
        if command[0] == "nvcc":
            return f"Cuda compilation tools, release {self.cuda}, V{self.cuda}.41\n"
        if len(command) >= 3 and command[1:3] == ["-m", "venv"]:
            python = Path(command[3]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            return ""
        if len(command) >= 3 and command[1:3] == ["-m", "pip"]:
            if "install" in command and self.fail_next_install:
                self.fail_next_install = False
                raise BOOTSTRAP.BootstrapError("injected install failure")
            if "install" in command and str(BOOTSTRAP.LOCK) in command:
                (Path(command[0]).parent / "uv").write_text("", encoding="utf-8")
            return ""
        if Path(command[0]).name == "uv" and command[1:3] == ["python", "install"]:
            return ""
        if Path(command[0]).name == "uv" and command[1] == "venv":
            python = Path(command[2]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            return ""
        if Path(command[0]).name == "uv" and command[1:3] in (["pip", "install"], ["pip", "check"]):
            return ""
        if command[:3] == ["uv", "pip", "compile"]:
            output = Path(command[command.index("--output-file") + 1])
            output.write_text("locked==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
            return ""
        if "RDAN_AMBIENT=" in command[2]:
            return f"RDAN_AMBIENT={json.dumps({'origin': self.ambient_origin})}\n"
        if "RDAN_CUDA=" in command[2]:
            return "RDAN_CUDA=" + json.dumps(self._cuda()) + "\n"
        if "RDAN_PACKAGES=" in command[2]:
            versions = BOOTSTRAP._locked_packages() | {"flash-attn": BOOTSTRAP._expected_packages()["flash-attn"]}
            versions.update(self.package_overrides)
            return "RDAN_PACKAGES=" + json.dumps(versions, sort_keys=True) + "\n"
        if "RDAN_DATA_RUNTIME=" in command[2]:
            payload = {"packages": BOOTSTRAP._locked_packages(BOOTSTRAP.DATA_LOCK), "python": "3.11.15"}
            return "RDAN_DATA_RUNTIME=" + json.dumps(payload, sort_keys=True) + "\n"
        if "RDAN_IMPORTS=" in command[2]:
            payload = {
                "rdan_origin": str(self.inputs.rdan_root / "src/rdan_grpo/__init__.py"),
                "roll_origin": str(self.inputs.rtt_root / "roll/__init__.py"),
                "fsdp2": "FSDP2TrainStrategy",
                "hf": "HfInferStrategy",
                "infer_worker": "ResponseInferWorker",
                "pipeline": "ResponseTrainingPipeline",
                "reward_worker": "RTTCompatibleRubricRewardWorker",
                "train_worker": "ResponseActorWorker",
                "sdpa": True,
            }
            return "noise from a dependency\nRDAN_IMPORTS=" + json.dumps(payload, sort_keys=True) + "\n"
        if len(command) > 1 and command[1].endswith("prepare_a100_response_data.py"):
            return json.dumps({"checked_only": "--check" in command, "status": "passed"}) + "\n"
        raise AssertionError(f"unexpected command: {command}")

    def _git(self, command: list[str]) -> str:
        root = Path(command[2])
        operation = command[3:]
        if operation == ["rev-parse", "--show-toplevel"]:
            return f"{root}\n"
        if operation == ["rev-parse", "HEAD"]:
            return f"{self.revisions[root]}\n"
        if operation == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return " M tracked.py\n" if root == self.dirty else ""
        raise AssertionError(f"unexpected git command: {command}")

    def _cuda(self) -> dict[str, Any]:
        return {
            "torch": "2.8.0+cu129",
            "cuda": self.cuda,
            "cuda_available": True,
            "device_count": 2,
            "devices": [{"index": index, "name": self.gpu_name, "memory_mib": 81920} for index in range(2)],
            "nccl_available": True,
            "nccl_version": self.nccl_version,
        }


class LaunchRunner:
    def __init__(self, *, repo_digest: str = RUNTIME.container_amd64_digest) -> None:
        self.repo_digest = repo_digest
        self.image_id = "sha256:" + "2" * 64
        self.commands: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.static = FakeRunner.__new__(FakeRunner)
        self.static.host_ram_bytes = 210 * 2**30
        self.static.free_disk_bytes = 2_700 * 2**30
        self.static.gpu_count = 2
        self.static.gpu_name = "NVIDIA A100-SXM4-80GB"
        self.static.gpu_memory_mib = 81920
        self.static.gpu_memory_used_mib = 0
        self.static.gpu_utilization_percent = 0
        self.static.compute_processes = ""
        self.static.topology_forward = "PIX"
        self.static.topology_reverse = "PIX"
        self.static.driver = "575.57.08"

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        del cwd
        command = list(args)
        self.commands.append(command)
        self.envs.append(dict(env or {}))
        if command[:2] == ["free", "--bytes"]:
            return f"Mem: {self.static.host_ram_bytes} 0 0\n"
        if command[:3] == ["df", "--block-size=1", "--output=avail"]:
            return f"Avail\n{self.static.free_disk_bytes}\n"
        if command[:2] == [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        ]:
            return "\n".join(
                f"{index}, GPU-{index}, {self.static.gpu_name}, {self.static.gpu_memory_mib}, "
                f"{self.static.gpu_memory_used_mib}, {self.static.gpu_utilization_percent}, {self.static.driver}"
                for index in range(self.static.gpu_count)
            )
        if command[:2] == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name"]:
            return self.static.compute_processes
        if command == ["nvidia-smi", "topo", "-m"]:
            return (
                "        GPU0 GPU1 CPU Affinity\n"
                f"GPU0    X    {self.static.topology_forward}   0-31\n"
                f"GPU1    {self.static.topology_reverse}  X     0-31\n"
            )
        if command[:2] == ["docker", "pull"]:
            return ""
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Architecture": "amd64",
                        "Id": self.image_id,
                        "Os": "linux",
                        "RepoDigests": [f"nvcr.io/nvidia/pytorch@{self.repo_digest}"],
                    }
                ]
            )
        if command[:2] == ["docker", "run"]:
            return "NGC banner\n" + json.dumps(
                {"container": {"image_id": self.image_id}, "schema_version": 2, "status": "passed"}
            )
        raise AssertionError(f"unexpected launch command: {command}")


@pytest.fixture
def contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, FakeRunner, dict[str, str]]:
    rtt = tmp_path / "rtt"
    snapshot = tmp_path / "snapshots" / BOOTSTRAP.MODEL_REVISION
    cache = tmp_path / "cache"
    run = tmp_path / "runs"
    for path in (rtt, snapshot, cache, run):
        path.mkdir(parents=True)
    tokenizer = {"tokenizer.json": b"tokenizer", "tokenizer_config.json": b"tokenizer config"}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in tokenizer.items()}
    monkeypatch.setattr(BOOTSTRAP, "TOKENIZER_HASHES", hashes)
    for name, content in tokenizer.items():
        (snapshot / name).write_bytes(content)
    (snapshot / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (snapshot / "generation_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"weights-2")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    model_files = (
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    )
    monkeypatch.setattr(
        BOOTSTRAP,
        "SNAPSHOT_HASHES",
        {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in model_files},
    )
    monkeypatch.setattr(
        BOOTSTRAP,
        "SNAPSHOT_SIZES",
        {name: (snapshot / name).stat().st_size for name in (*model_files, *tokenizer)},
    )
    snapshot_manifest = tmp_path / "snapshot-manifest.json"
    all_hashes = BOOTSTRAP.SNAPSHOT_HASHES | BOOTSTRAP.TOKENIZER_HASHES
    snapshot_manifest.write_text(
        json.dumps(
            {
                "files": {
                    name: {"sha256": all_hashes[name], "size": BOOTSTRAP.SNAPSHOT_SIZES[name]}
                    for name in sorted(all_hashes)
                },
                "model": BOOTSTRAP.MODEL,
                "revision": BOOTSTRAP.MODEL_REVISION,
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(BOOTSTRAP, "SNAPSHOT_MANIFEST", snapshot_manifest)
    inputs = BOOTSTRAP.Inputs(
        rtt_root=rtt,
        rdan_root=BOOTSTRAP.REPO_ROOT,
        rdan_revision="a" * 40,
        snapshot=snapshot,
        cache_root=cache,
        run_root=run,
    )
    env = {
        "HF_TOKEN": "hugging-face-secret",
        "OPENROUTER_API_KEY": "openrouter-secret",
        "PIP_CONSTRAINT": "/etc/pip/constraint.txt",
        "WANDB_API_KEY": "wandb-secret",
    }
    return inputs, FakeRunner(inputs), env


def test_check_passes_without_exposing_secrets(contract: tuple[Any, FakeRunner, dict[str, str]]) -> None:
    inputs, runner, env = contract
    report = BOOTSTRAP.check_environment(inputs, runner, env, host=_host())

    encoded = json.dumps(report, sort_keys=True)
    assert report["profile"] == "fsdp2-hf-sdpa-2xa100"
    assert report["gpu"]["count"] == 2
    assert report["gpu"]["nccl"] == "2.27.3"
    assert report["host_readiness"]["gpu"]["topology"] == {
        "gpu0_to_gpu1": "PIX",
        "gpu1_to_gpu0": "PIX",
    }
    assert report["capabilities"] == {
        "judge_access": True,
        "model_publish_access": True,
        "tracking_access": True,
    }
    assert report["packages"]["transformers"] == "4.57.0"
    assert report["runtime_imports"]["pipeline"] == "ResponseTrainingPipeline"
    assert "openrouter-secret" not in encoded
    assert "wandb-secret" not in encoded
    assert "hugging-face-secret" not in encoded
    assert all(
        "OPENROUTER_API_KEY" not in item and "WANDB_API_KEY" not in item and "HF_TOKEN" not in item
        for item in runner.envs
    )
    assert all("PIP_CONSTRAINT" not in item for item in runner.envs)


def test_parse_topology_strips_ansi_header_and_accepts_nvlink() -> None:
    output = (
        "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
        "GPU0\t X \t NV12\t0-31\t0\tN/A\n"
        "GPU1\tNV12\t X \t0-31\t0\tN/A\n"
    )

    assert BOOTSTRAP._parse_topology(output, RUNTIME) == {
        "gpu0_to_gpu1": "NV12",
        "gpu1_to_gpu0": "NV12",
    }


@pytest.mark.parametrize(
    ("missing", "capability"),
    [
        ("OPENROUTER_API_KEY", "judge_access"),
        ("WANDB_API_KEY", "tracking_access"),
        ("HF_TOKEN", "model_publish_access"),
    ],
)
def test_check_requires_service_credentials(
    contract: tuple[Any, FakeRunner, dict[str, str]], missing: str, capability: str
) -> None:
    inputs, runner, _ = contract
    env = contract[2] | {missing: ""}

    with pytest.raises(BOOTSTRAP.BootstrapError, match=capability):
        BOOTSTRAP.check_environment(inputs, runner, env, host=_host())


def test_check_rejects_presence_flags_without_raw_credentials(
    contract: tuple[Any, FakeRunner, dict[str, str]],
) -> None:
    inputs, runner, _ = contract
    env = {
        "RDAN_JUDGE_CREDENTIAL_PRESENT": "1",
        "RDAN_MODEL_PUBLISH_CREDENTIAL_PRESENT": "1",
        "RDAN_TRACKING_CREDENTIAL_PRESENT": "1",
    }

    with pytest.raises(BOOTSTRAP.BootstrapError, match="required service credential capabilities are absent"):
        BOOTSTRAP.check_environment(inputs, runner, env, host=_host())


def test_check_accepts_hugging_face_hub_token_without_naming_it_in_receipt(
    contract: tuple[Any, FakeRunner, dict[str, str]],
) -> None:
    inputs, runner, env = contract
    env = env | {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "alternate-private-value"}

    report = BOOTSTRAP.check_environment(inputs, runner, env, host=_host())

    encoded = json.dumps(report, sort_keys=True)
    assert report["capabilities"]["model_publish_access"] is True
    assert "HF_TOKEN" not in encoded
    assert "HUGGING_FACE_HUB_TOKEN" not in encoded
    assert "alternate-private-value" not in encoded


def test_check_consumes_drifted_compute_threshold_exactly(
    contract: tuple[Any, FakeRunner, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs, runner, env = contract
    source = BOOTSTRAP.COMPUTE_CONTRACT
    payload = json.loads(source.read_text(encoding="utf-8"))
    runtime = payload["response_runtime"]
    runtime["host"]["minimum_ram_gib"] = 211
    runtime["platform"]["container_contract"] = str(
        RUNTIME.package_contracts[0].parent / "a100-response-container.json"
    )
    runtime["packages"]["contracts"] = [str(path) for path in RUNTIME.package_contracts]
    drifted = tmp_path / "compute.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(BOOTSTRAP, "COMPUTE_CONTRACT", drifted)

    loaded = BOOTSTRAP._load_runtime_contract()

    assert loaded.minimum_ram_bytes == 211 * 2**30
    with pytest.raises(BOOTSTRAP.BootstrapError, match="at least 211 GiB"):
        BOOTSTRAP.check_environment(inputs, runner, env, host=_host())


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("python", "Python must be exact"),
        ("platform", "host must be Linux x86_64"),
        ("container", "inside the pinned NGC container"),
        ("image", "container identity does not match"),
        ("digest", "container image digest does not match"),
        ("identity", "not externally inspected"),
        ("ram", "host RAM must be at least 192 GiB"),
        ("cache-disk", "cache filesystem must have at least 512 GiB free"),
        ("run-disk", "run filesystem must have at least 512 GiB free"),
        ("gpu", "must expose only A100"),
        ("gpu-count", "must expose exactly two GPUs"),
        ("gpu-memory", "memory must be at least 79 GiB"),
        ("gpu-memory-used", "must be idle"),
        ("gpu-utilization", "must be idle"),
        ("gpu-process", "no active compute processes"),
        ("gpu-topology", "exact reciprocal supported topology link"),
        ("driver", "driver is below"),
        ("cuda", "nvcc must be exact CUDA 12.9"),
        ("nccl", "NCCL runtime must be exact 2.27.3"),
        ("rtt", "RTT revision mismatch"),
        ("dirty", "RDAN checkout is dirty"),
        ("package", "installed distribution mismatch"),
        ("package-extra", "installed distribution set mismatch"),
        ("snapshot", "tokenizer.json size mismatch"),
        ("config-byte", "config.json hash mismatch"),
        ("weights-byte", "model-00001-of-00002.safetensors hash mismatch"),
        ("weights", "missing a referenced safetensors shard"),
        ("root-ancestor", "outside repositories"),
        ("ambient", "ambient roll module conflicts"),
    ],
)
def test_check_fails_closed(
    contract: tuple[Any, FakeRunner, dict[str, str]],
    case: str,
    message: str,
) -> None:
    inputs, runner, env = contract
    host = _host()
    if case == "python":
        host = _host(python=(3, 11, 14))
    elif case == "platform":
        host = _host(system="Darwin", machine="arm64")
    elif case == "container":
        host = _host(container=False)
    elif case == "image":
        host = _host(container_release="25.05")
    elif case == "digest":
        host = _host(image_digest="sha256:" + "0" * 64)
    elif case == "identity":
        host = _host(identity_source="operator-claim")
    elif case == "ram":
        runner.host_ram_bytes = RUNTIME.minimum_ram_bytes - 1
    elif case == "cache-disk":
        runner.free_disk_bytes[inputs.cache_root] = RUNTIME.minimum_free_disk_bytes - 1
    elif case == "run-disk":
        runner.free_disk_bytes[inputs.run_root] = RUNTIME.minimum_free_disk_bytes - 1
    elif case == "gpu":
        runner.gpu_name = "NVIDIA L40S"
    elif case == "gpu-count":
        runner.gpu_count = 1
    elif case == "gpu-memory":
        runner.gpu_memory_mib = 79 * 1024 - 1
    elif case == "gpu-memory-used":
        runner.gpu_memory_used_mib = 1
    elif case == "gpu-utilization":
        runner.gpu_utilization_percent = 1
    elif case == "gpu-process":
        runner.compute_processes = "GPU-0, 42, python\n"
    elif case == "gpu-topology":
        runner.topology_reverse = "PHB"
    elif case == "driver":
        runner.driver = "570.00.00"
    elif case == "cuda":
        runner.cuda = "12.8"
    elif case == "nccl":
        runner.nccl_version = [2, 26, 2]
    elif case == "rtt":
        runner.revisions[inputs.rtt_root] = "b" * 40
    elif case == "dirty":
        runner.dirty = inputs.rdan_root
    elif case == "package":
        runner.package_overrides["transformers"] = "4.56.0"
    elif case == "package-extra":
        runner.package_overrides["unplanned-package"] = "1.0"
    elif case == "snapshot":
        (inputs.snapshot / "tokenizer.json").write_text("changed", encoding="utf-8")
    elif case == "config-byte":
        (inputs.snapshot / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    elif case == "weights-byte":
        (inputs.snapshot / "model-00001-of-00002.safetensors").write_bytes(b"Weights-1")
    elif case == "weights":
        (inputs.snapshot / "model-00002-of-00002.safetensors").unlink()
    elif case == "root-ancestor":
        inputs = BOOTSTRAP.Inputs(
            rtt_root=inputs.rtt_root,
            rdan_root=inputs.rdan_root,
            rdan_revision=inputs.rdan_revision,
            snapshot=inputs.snapshot,
            cache_root=inputs.rtt_root,
            run_root=inputs.run_root,
        )
    elif case == "ambient":
        runner.ambient_origin = "/tmp/site-packages/roll/__init__.py"

    with pytest.raises(BOOTSTRAP.BootstrapError, match=message):
        BOOTSTRAP.check_environment(inputs, runner, env, host=host)


def test_setup_seals_marker_and_refuses_overwrite(
    contract: tuple[Any, FakeRunner, dict[str, str]],
) -> None:
    inputs, runner, env = contract
    venv = inputs.cache_root / "a100-response-venv"

    report = BOOTSTRAP.setup_environment(inputs, venv, runner, env, 4, host=_host())
    marker = inputs.run_root / "a100-response-bootstrap.json"
    sealed = json.loads(marker.read_text(encoding="utf-8"))

    assert report["marker"] == str(marker)
    assert sealed["status"] == "passed"
    assert all(isinstance(command, list) for command in runner.commands)
    assert any("--no-build-isolation" in command for command in runner.commands)
    assert any("--require-hashes" in command for command in runner.commands)
    assert any(str(BOOTSTRAP.LOCK) in command for command in runner.commands)
    assert any(str(BOOTSTRAP.BOOTSTRAP_LOCK) in command for command in runner.commands)
    assert any(str(BOOTSTRAP.DATA_LOCK) in command for command in runner.commands)
    prepare = next(
        command
        for command in runner.commands
        if len(command) > 1 and command[1].endswith("prepare_a100_response_data.py")
    )
    data_python = Path(prepare[prepare.index("--data-python") + 1])
    assert data_python.name == BOOTSTRAP.DATA_PYTHON_BIN
    assert data_python.is_file()
    assert not data_python.is_symlink()
    assert data_python.stat().st_mode & 0o111
    assert report["data_runtime"]["python_version"] == "3.11.15"
    assert all(
        "OPENROUTER_API_KEY" not in item and "PIP_CONSTRAINT" not in item and "WANDB_API_KEY" not in item
        for item in runner.envs
    )
    assert any(
        command[-1] == "check" for command in runner.commands if len(command) > 2 and command[1:3] == ["-m", "pip"]
    )
    with pytest.raises(BOOTSTRAP.BootstrapError, match="refusing to overwrite"):
        BOOTSTRAP.setup_environment(inputs, venv, runner, env, 4, host=_host())


def test_setup_cleans_failed_stage_and_retries(contract: tuple[Any, FakeRunner, dict[str, str]]) -> None:
    inputs, runner, env = contract
    venv = inputs.cache_root / "a100-response-venv"
    runner.fail_next_install = True

    with pytest.raises(BOOTSTRAP.BootstrapError, match="injected install failure"):
        BOOTSTRAP.setup_environment(inputs, venv, runner, env, 4, host=_host())

    assert not venv.exists()
    assert not list(inputs.cache_root.glob(f".{venv.name}.*"))
    report = BOOTSTRAP.setup_environment(inputs, venv, runner, env, 4, host=_host())
    assert report["status"] == "passed"
    assert venv.is_dir()


def test_host_launcher_inspects_digest_and_mounts_receipt(
    contract: tuple[Any, FakeRunner, dict[str, str]],
) -> None:
    inputs, _, env = contract
    runner = LaunchRunner()
    venv = inputs.cache_root / "a100-response-venv"

    report = BOOTSTRAP.launch_environment(
        inputs,
        venv,
        runner,
        env,
        setup=True,
        max_build_jobs=4,
        system="Linux",
        machine="x86_64",
    )

    receipt = inputs.run_root / BOOTSTRAP.IDENTITY_NAME
    docker_run = next(command for command in runner.commands if command[:2] == ["docker", "run"])
    assert report["status"] == "passed"
    assert json.loads(receipt.read_text(encoding="utf-8"))["repo_digest"] == RUNTIME.container_amd64_digest
    assert RUNTIME.container_ref in docker_run
    assert any(value.endswith(f"dst={BOOTSTRAP.IDENTITY_RECEIPT},readonly") for value in docker_run)
    assert "openrouter-secret" not in " ".join(docker_run)
    assert "wandb-secret" not in " ".join(docker_run)
    assert "hugging-face-secret" not in " ".join(docker_run)
    assert all(
        any(docker_run[index : index + 2] == ["--env", name] for index in range(len(docker_run) - 1))
        for name in ("OPENROUTER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")
    )
    assert not any("RDAN_" in item and "CREDENTIAL_PRESENT" in item for item in docker_run)
    assert "--env-file" not in docker_run
    assert docker_run.count("--workdir") == 1
    assert not any(value.startswith(f"type=bind,src={venv},") for value in docker_run)
    docker_env = next(
        process_env
        for command, process_env in zip(runner.commands, runner.envs, strict=True)
        if command[:2] == ["docker", "run"]
    )
    assert {name: docker_env.get(name) for name in ("OPENROUTER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")} == {
        "OPENROUTER_API_KEY": "openrouter-secret",
        "WANDB_API_KEY": "wandb-secret",
        "HF_TOKEN": "hugging-face-secret",
    }
    assert "HUGGING_FACE_HUB_TOKEN" not in docker_env
    assert "PIP_CONSTRAINT" not in docker_env
    assert all(
        not any(name in process_env for name in BOOTSTRAP.SECRET_ENV_NAMES)
        for command, process_env in zip(runner.commands, runner.envs, strict=True)
        if command[:2] != ["docker", "run"]
    )


def test_host_launcher_rejects_wrong_inspected_digest(
    contract: tuple[Any, FakeRunner, dict[str, str]],
) -> None:
    inputs, _, env = contract

    with pytest.raises(BOOTSTRAP.BootstrapError, match="pinned repository digest"):
        BOOTSTRAP.launch_environment(
            inputs,
            inputs.cache_root / "a100-response-venv",
            LaunchRunner(repo_digest="sha256:" + "0" * 64),
            env,
            setup=True,
            max_build_jobs=4,
            system="Linux",
            machine="x86_64",
        )


def test_launch_check_is_read_only_and_does_not_pull(
    contract: tuple[Any, FakeRunner, dict[str, str]],
) -> None:
    inputs, _, env = contract
    runner = LaunchRunner()
    target = inputs.cache_root / "sealed-env"
    (target / "bin").mkdir(parents=True)
    (target / "bin/python").write_text("", encoding="utf-8")
    venv = inputs.cache_root / "a100-response-venv"
    venv.symlink_to(target, target_is_directory=True)
    identity = BOOTSTRAP._image_identity(
        json.dumps(
            [
                {
                    "Architecture": "amd64",
                    "Id": runner.image_id,
                    "Os": "linux",
                    "RepoDigests": [RUNTIME.container_ref],
                }
            ]
        )
    )
    receipt = inputs.run_root / BOOTSTRAP.IDENTITY_NAME
    BOOTSTRAP._seal_identity(receipt, identity)
    before = receipt.read_bytes()

    report = BOOTSTRAP.launch_environment(
        inputs,
        venv,
        runner,
        env,
        setup=False,
        max_build_jobs=4,
        system="Linux",
        machine="x86_64",
    )

    assert report["status"] == "passed"
    assert receipt.read_bytes() == before
    assert not any(command[:2] == ["docker", "pull"] for command in runner.commands)


def test_cli_help_uses_a_real_subprocess() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=False)

    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "--setup" in result.stdout


def test_runner_redacts_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    result = subprocess.CompletedProcess(["tool"], 1, "", "failed with private-value another-value hugging-face-value")
    observed: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args
        observed.update(kwargs)
        return result

    monkeypatch.setattr(BOOTSTRAP.subprocess, "run", fake_run)

    with pytest.raises(BOOTSTRAP.BootstrapError) as caught:
        BOOTSTRAP.Runner().run(
            ["tool"],
            env={
                "OPENROUTER_API_KEY": "private-value",
                "HF_TOKEN": "hugging-face-value",
                "PIP_CONSTRAINT": "/etc/pip/constraint.txt",
                "WANDB_API_KEY": "another-value",
            },
        )

    assert "private-value" not in str(caught.value)
    assert "another-value" not in str(caught.value)
    assert "hugging-face-value" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "OPENROUTER_API_KEY" not in observed["env"]
    assert "PIP_CONSTRAINT" not in observed["env"]
    assert "WANDB_API_KEY" not in observed["env"]
    assert "HF_TOKEN" not in observed["env"]


def test_runner_forwards_only_named_secrets_to_docker_run(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(BOOTSTRAP.subprocess, "run", fake_run)
    command = [
        "docker",
        "run",
        "--env",
        "OPENROUTER_API_KEY",
        "--env",
        "WANDB_API_KEY",
        "--env",
        "HF_TOKEN",
        "image",
    ]
    BOOTSTRAP.Runner().run(
        command,
        env={
            "HF_TOKEN": "hugging-face-secret",
            "HUGGING_FACE_HUB_TOKEN": "unused-alternate-secret",
            "OPENAI_API_KEY": "unused-openai-secret",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "PIP_CONSTRAINT": "/etc/pip/constraint.txt",
            "WANDB_API_KEY": "wandb-secret",
        },
    )

    assert observed["env"] == {
        "HF_TOKEN": "hugging-face-secret",
        "OPENROUTER_API_KEY": "openrouter-secret",
        "WANDB_API_KEY": "wandb-secret",
    }
    assert not any("secret" in item for item in command)


def test_data_python_launcher_is_regular_and_executes(tmp_path: Path) -> None:
    venv = tmp_path / "data-venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)

    launcher = BOOTSTRAP._write_data_python_launcher(venv)
    result = subprocess.run(
        [str(launcher), "-c", "print('launcher-passed')"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "launcher-passed"
    assert launcher.is_file()
    assert not launcher.is_symlink()


def test_resolve_lock_writes_only_a_candidate(
    contract: tuple[Any, FakeRunner, dict[str, str]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, runner, env = contract
    lock = tmp_path / "a100.lock"
    bootstrap_lock = tmp_path / "bootstrap.lock"
    data_lock = tmp_path / "data.lock"
    source = tmp_path / "a100.in"
    bootstrap_source = tmp_path / "bootstrap.in"
    data_source = tmp_path / "data.in"
    source.write_text("torch==2.8.0\n", encoding="utf-8")
    bootstrap_source.write_text("pip==25.1.1\n", encoding="utf-8")
    data_source.write_text("numpy==1.26.4\n", encoding="utf-8")
    monkeypatch.setattr(BOOTSTRAP, "LOCK", lock)
    monkeypatch.setattr(BOOTSTRAP, "LOCK_INPUT", source)
    monkeypatch.setattr(BOOTSTRAP, "BOOTSTRAP_LOCK", bootstrap_lock)
    monkeypatch.setattr(BOOTSTRAP, "BOOTSTRAP_INPUT", bootstrap_source)
    monkeypatch.setattr(BOOTSTRAP, "DATA_LOCK", data_lock)
    monkeypatch.setattr(BOOTSTRAP, "DATA_REQUIREMENTS", data_source)
    monkeypatch.setattr(BOOTSTRAP, "_host_info", lambda env: _host())

    report = BOOTSTRAP.resolve_lock(runner, env)

    candidates = (
        lock.with_suffix(".lock.new"),
        bootstrap_lock.with_suffix(".lock.new"),
        data_lock.with_suffix(".lock.new"),
    )
    assert report["status"] == "candidates_generated"
    assert {Path(item["path"]) for item in report["candidates"]} == set(candidates)
    assert all(candidate.is_file() for candidate in candidates)
    assert not lock.exists()
    assert not bootstrap_lock.exists()
    assert not data_lock.exists()


def _host(
    *,
    system: str = "Linux",
    machine: str = "x86_64",
    python: tuple[int, ...] = (3, 12, 9),
    container: bool = True,
    container_release: str = RUNTIME.container_release,
    image_digest: str = RUNTIME.container_amd64_digest,
    identity_source: str = "docker-image-inspect",
) -> Any:
    return BOOTSTRAP.HostInfo(
        system=system,
        machine=machine,
        implementation="CPython",
        python=python,
        os_release={"ID": "ubuntu", "VERSION_ID": "24.04"},
        container=container,
        cuda=RUNTIME.container_cuda,
        container_release=container_release,
        image_digest=image_digest,
        image_id="sha256:" + "1" * 64,
        identity_source=identity_source,
    )
