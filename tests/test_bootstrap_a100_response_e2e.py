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


class FakeRunner:
    def __init__(self, inputs: Any) -> None:
        self.inputs = inputs
        self.revisions = {
            inputs.rtt_root: BOOTSTRAP.RTT_REVISION,
            inputs.rdan_root: inputs.rdan_revision,
        }
        self.dirty: Path | None = None
        self.gpu_name = "NVIDIA A100-SXM4-80GB"
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
        if command[0] == "nvidia-smi":
            return "\n".join(f"{index}, {self.gpu_name}, 81920, 575.57.08" for index in range(2))
        if command[0] == "nvcc":
            return "Cuda compilation tools, release 12.9, V12.9.41\n"
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
            "cuda": "12.9",
            "cuda_available": True,
            "device_count": 2,
            "devices": [{"index": index, "name": self.gpu_name, "memory_mib": 81920} for index in range(2)],
            "nccl_available": True,
        }


class LaunchRunner:
    def __init__(self, *, repo_digest: str = BOOTSTRAP.CONTAINER_AMD64_DIGEST) -> None:
        self.repo_digest = repo_digest
        self.image_id = "sha256:" + "2" * 64
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
                {"container": {"image_id": self.image_id}, "schema_version": 1, "status": "passed"}
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
    assert report["packages"]["transformers"] == "4.57.0"
    assert report["runtime_imports"]["pipeline"] == "ResponseTrainingPipeline"
    assert "openrouter-secret" not in encoded
    assert "wandb-secret" not in encoded
    assert all("OPENROUTER_API_KEY" not in item and "WANDB_API_KEY" not in item for item in runner.envs)
    assert all("PIP_CONSTRAINT" not in item for item in runner.envs)


def test_check_does_not_require_service_credentials(contract: tuple[Any, FakeRunner, dict[str, str]]) -> None:
    inputs, runner, _ = contract

    report = BOOTSTRAP.check_environment(inputs, runner, {}, host=_host())

    assert report["status"] == "passed"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("python", "Python must be exact"),
        ("platform", "host must be Linux x86_64"),
        ("container", "inside the pinned NGC container"),
        ("image", "container identity does not match"),
        ("digest", "container image digest does not match"),
        ("identity", "not externally inspected"),
        ("gpu", "must expose only A100"),
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
    elif case == "gpu":
        runner.gpu_name = "NVIDIA L40S"
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
    assert json.loads(receipt.read_text(encoding="utf-8"))["repo_digest"] == BOOTSTRAP.CONTAINER_AMD64_DIGEST
    assert BOOTSTRAP.CONTAINER_REF in docker_run
    assert any(value.endswith(f"dst={BOOTSTRAP.IDENTITY_RECEIPT},readonly") for value in docker_run)
    assert "openrouter-secret" not in " ".join(docker_run)
    assert "wandb-secret" not in " ".join(docker_run)
    assert "--env-file" not in docker_run
    assert docker_run.count("--workdir") == 1
    assert not any(value.startswith(f"type=bind,src={venv},") for value in docker_run)
    assert all("OPENROUTER_API_KEY" not in item and "WANDB_API_KEY" not in item for item in runner.envs)


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


def test_cli_help_uses_a_real_subprocess() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=False)

    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "--setup" in result.stdout


def test_runner_redacts_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    result = subprocess.CompletedProcess(["tool"], 1, "", "failed with private-value")
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
                "PIP_CONSTRAINT": "/etc/pip/constraint.txt",
                "WANDB_API_KEY": "another-value",
            },
        )

    assert "private-value" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "OPENROUTER_API_KEY" not in observed["env"]
    assert "PIP_CONSTRAINT" not in observed["env"]
    assert "WANDB_API_KEY" not in observed["env"]


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
    container_release: str = BOOTSTRAP.CONTAINER_RELEASE,
    image_digest: str = BOOTSTRAP.CONTAINER_AMD64_DIGEST,
    identity_source: str = "docker-image-inspect",
) -> Any:
    return BOOTSTRAP.HostInfo(
        system=system,
        machine=machine,
        implementation="CPython",
        python=python,
        os_release={"ID": "ubuntu", "VERSION_ID": "24.04"},
        container=container,
        cuda=BOOTSTRAP.CONTAINER_CUDA,
        container_release=container_release,
        image_digest=image_digest,
        image_id="sha256:" + "1" * 64,
        identity_source=identity_source,
    )
