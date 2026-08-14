from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import SavePlan

import rdan_grpo.roll_compat as compat
from rdan_grpo.roll_compat import (
    RTT_BASE_CONFIG_SHA256,
    RTT_MEGATRON_SHA256,
    RTT_REVISION,
    RTT_UTILS_SHA256,
    dump_batch_to_reward_system,
    install_rtt_compat,
    load_sync_hf_rlvr_config,
    patch_torch_find_nd_overlapping_shards,
    patch_torch_validate_global_plan,
)

ROOT = Path(__file__).resolve().parents[2]
RTT = ROOT / "Rubrics-To-Tokens"


@pytest.fixture(autouse=True)
def restore_import_state() -> object:
    path = list(sys.path)
    missing = object()
    patcher = sys.modules.get("mcore_adapter.patcher", missing)
    parent = sys.modules.get("mcore_adapter", missing)
    yield
    sys.path[:] = path
    for name, value in (("mcore_adapter.patcher", patcher), ("mcore_adapter", parent)):
        if value is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def test_pinned_rtt_revision_and_utils_digest() -> None:
    if not RTT.is_dir():
        pytest.skip("RTT reference checkout is absent")
    revision = subprocess.run(
        ["git", "-C", str(RTT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert revision == RTT_REVISION
    base_config = RTT / "roll/configs/base_config.py"
    assert hashlib.sha256(base_config.read_bytes()).hexdigest() == RTT_BASE_CONFIG_SHA256
    assert hashlib.sha256((RTT / "roll/pipeline/rlvr/utils.py").read_bytes()).hexdigest() == RTT_UTILS_SHA256
    strategy = RTT / "roll/distributed/strategy/megatron_strategy.py"
    assert hashlib.sha256(strategy.read_bytes()).hexdigest() == RTT_MEGATRON_SHA256


@pytest.mark.skipif(not RTT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_pinned_install_rtt_compat_succeeds_in_fresh_process() -> None:
    for dependency in ("megatron", "ray"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"ROLL {dependency} dependency is unavailable")
    code = f"""
import sys
from pathlib import Path
rtt = Path({str(RTT)!r})
repo = Path({str(ROOT / "RDAN-GRPO")!r})
sys.path[:0] = [str(repo / "src"), str(rtt), str(rtt / "mcore_adapter/src")]
from rdan_grpo.roll_compat import dump_batch_to_reward_system, install_rtt_compat
install_rtt_compat(rtt)
from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy
from roll.pipeline.rlvr import utils
assert utils.dump_batch_to_reward_system is dump_batch_to_reward_system
assert MegatronInferStrategy.inner_forward_step.__rdan_compat_owner__ == "rdan-grpo:local-qwen3-mask"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not RTT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_pinned_install_rtt_compat_without_optional_megatron(tmp_path: Path) -> None:
    try:
        megatron_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        megatron_spec = None
    if megatron_spec is not None:
        pytest.skip("optional Megatron dependency is installed")
    code = f"""
import sys
from pathlib import Path
from types import ModuleType
rtt = Path({str(RTT)!r})
repo = Path({str(ROOT / "RDAN-GRPO")!r})
sys.path[:0] = [str(repo / "src"), str(rtt), str(rtt / "mcore_adapter/src")]
codetiming = ModuleType("codetiming")
codetiming.Timer = type("Timer", (), {{}})
protocol = ModuleType("roll.distributed.scheduler.protocol")
protocol.DataProto = type("DataProto", (), {{}})
sys.modules["codetiming"] = codetiming
sys.modules["roll.distributed.scheduler.protocol"] = protocol
from rdan_grpo.roll_compat import dump_batch_to_reward_system, install_rtt_compat
install_rtt_compat(rtt)
from roll.pipeline.rlvr import utils
assert Path(utils.__file__).resolve() == (rtt / "roll/pipeline/rlvr/utils.py").resolve()
assert utils.dump_batch_to_reward_system is dump_batch_to_reward_system
assert "roll.distributed.strategy.megatron_strategy" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "", "ROLL_LOG_DIR": str(tmp_path / "roll-logs")},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_load_sync_hf_config_uses_local_surrogate_and_restores_result(monkeypatch, tmp_path) -> None:
    payload = _sync_hf_payload()
    original = deepcopy(payload)
    monkeypatch.setattr(compat, "_verify_sync_hf_rtt", lambda root: None)

    def construct(config_cls, surrogate):
        assert config_cls is object
        assert surrogate is not payload
        assert surrogate["actor_infer"]["strategy_args"]["strategy_name"] == "vllm"
        assert surrogate["actor_train"]["device_mapping"] == "[0, 1]"
        assert surrogate["actor_infer"]["device_mapping"] == "[0, 1]"
        assert surrogate["rewards"] == {}
        return _constructed_sync_config(max_concurrency=1000, infer_strategy="vllm")

    monkeypatch.setattr(compat, "_construct_config", construct)
    config = load_sync_hf_rlvr_config(tmp_path, object, payload)

    assert config.actor_infer.strategy_args.strategy_name == "hf_infer"
    assert config.actor_infer.max_concurrency == 1
    assert config.actor_train.worker_cls == compat._ACTOR_WORKER
    assert config.actor_infer.worker_cls == compat._INFER_WORKER
    assert config.rewards == config.domain_2_tag == config.tag_2_domain == {}
    assert payload == original


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("actor_train", "strategy_args", "strategy_name"), "megatron_train", "fsdp2_train"),
        (("actor_infer", "strategy_args", "strategy_name"), "vllm", "hf_infer"),
        (("actor_train", "worker_cls"), "wrong.Actor", "actor_train.worker_cls"),
        (("actor_infer", "worker_cls"), "wrong.Infer", "actor_infer.worker_cls"),
        (("actor_train", "device_mapping"), "[0]", "device mappings"),
        (("actor_infer", "device_mapping"), "[1, 0]", "device mappings"),
        (("actor_train", "num_gpus_per_worker"), 2, "num_gpus_per_worker=1"),
        (("actor_infer", "num_gpus_per_worker"), 2, "num_gpus_per_worker=1"),
        (("async_generation_ratio",), 1, "async_generation_ratio=0"),
        (("generate_opt_level",), 1, "generate_opt_level=0"),
        (("rewards",), {}, "rewards=null"),
        (("actor_infer", "max_concurrency"), 2, "actor_infer.max_concurrency=1"),
    ],
)
def test_load_sync_hf_config_rejects_unsupported_profiles(monkeypatch, tmp_path, path, value, message) -> None:
    payload = _sync_hf_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    monkeypatch.setattr(
        compat,
        "_construct_config",
        lambda *args, **kwargs: pytest.fail("invalid profiles must fail before construction"),
    )
    with pytest.raises(ValueError, match=message):
        load_sync_hf_rlvr_config(tmp_path, object, payload)


def test_load_sync_hf_config_rejects_base_config_byte_drift(monkeypatch, tmp_path) -> None:
    path = tmp_path / "roll/configs/base_config.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"modified")
    monkeypatch.setattr(
        compat,
        "_run_git",
        lambda root, *args: RTT_REVISION if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        compat,
        "_construct_config",
        lambda *args, **kwargs: pytest.fail("byte drift must fail before construction"),
    )

    with pytest.raises(RuntimeError, match="unexpected RTT base config digest"):
        load_sync_hf_rlvr_config(tmp_path, object, _sync_hf_payload())


def test_load_sync_hf_config_preserves_input_when_construction_fails(monkeypatch, tmp_path) -> None:
    payload = _sync_hf_payload()
    original = deepcopy(payload)
    monkeypatch.setattr(compat, "_verify_sync_hf_rtt", lambda root: None)

    def fail(config_cls, surrogate):
        surrogate["actor_infer"]["max_concurrency"] = 999
        raise LookupError("construction failed")

    monkeypatch.setattr(compat, "_construct_config", fail)
    with pytest.raises(LookupError, match="construction failed"):
        load_sync_hf_rlvr_config(tmp_path, object, payload)
    assert payload == original


def test_load_sync_hf_config_rejects_failed_strategy_restoration(monkeypatch, tmp_path) -> None:
    payload = _sync_hf_payload()
    monkeypatch.setattr(compat, "_verify_sync_hf_rtt", lambda root: None)

    class StickyStrategy:
        @property
        def strategy_name(self):
            return "vllm"

        @strategy_name.setter
        def strategy_name(self, value):
            pass

    config = _constructed_sync_config(max_concurrency=1000, infer_strategy="vllm")
    config.actor_infer.strategy_args = StickyStrategy()
    monkeypatch.setattr(compat, "_construct_config", lambda config_cls, surrogate: config)

    with pytest.raises(RuntimeError, match="did not restore actor_infer strategy hf_infer"):
        load_sync_hf_rlvr_config(tmp_path, object, payload)


@pytest.mark.skipif(not RTT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_pinned_rlvr_config_preserves_same_backend_cluster_contract() -> None:
    pytest.importorskip("dacite")
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    code = f"""
import os
import sys
from pathlib import Path
rtt = Path({str(RTT)!r})
repo = Path({str(ROOT / "RDAN-GRPO")!r})
sys.path[:0] = [str(repo / "src"), str(rtt), str(rtt / 'mcore_adapter/src')]
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdan_grpo.roll_compat import load_sync_hf_rlvr_config
from roll.pipeline.rlvr.rubric_config import RLVRConfig
os.environ["RDAN_MODEL_SNAPSHOT"] = "/models/qwen"
with initialize_config_dir(version_base=None, config_dir=str(repo / "configs/roll")):
    composed = compose(config_name="qwen_scalar_same_backend_parity")
payload = OmegaConf.to_container(composed, resolve=True)
assert isinstance(payload, dict)
config = load_sync_hf_rlvr_config(rtt, RLVRConfig, payload)
assert config.actor_train.worker_cls == {compat._ACTOR_WORKER!r}
assert config.actor_infer.worker_cls == {compat._INFER_WORKER!r}
assert config.actor_train.world_size == config.actor_infer.world_size == 2
assert config.actor_infer.max_concurrency == 1
assert config.generate_opt_level == 0 and config.async_generation_ratio == 0
assert config.rewards == config.domain_2_tag == config.tag_2_domain == {{}}
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not RTT.is_dir(), reason="pinned RTT checkout is unavailable")
def test_pinned_cluster_caller_resolves_exact_worker_classes() -> None:
    for dependency in ("dacite", "hydra", "omegaconf", "ray"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"ROLL {dependency} dependency is unavailable")
    code = f"""
import os
import sys
from pathlib import Path
rtt = Path({str(RTT)!r})
repo = Path({str(ROOT / "RDAN-GRPO")!r})
sys.path[:0] = [str(repo / "src"), str(rtt), str(rtt / 'mcore_adapter/src')]
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdan_grpo.roll_compat import install_rtt_compat, load_sync_hf_rlvr_config
from roll.pipeline.rlvr.rubric_config import RLVRConfig
from roll.utils.import_utils import safe_import_class
install_rtt_compat(rtt)
os.environ["RDAN_MODEL_SNAPSHOT"] = "/models/qwen"
with initialize_config_dir(version_base=None, config_dir=str(repo / "configs/roll")):
    composed = compose(config_name="qwen_scalar_same_backend_parity")
payload = OmegaConf.to_container(composed, resolve=True)
assert isinstance(payload, dict)
config = load_sync_hf_rlvr_config(rtt, RLVRConfig, payload)
assert config.rewards == config.domain_2_tag == config.tag_2_domain == {{}}
for path in (config.actor_train.worker_cls, config.actor_infer.worker_cls):
    worker = safe_import_class(path)
    assert worker is not None
    assert f"{{worker.__module__}}.{{worker.__name__}}" == path
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unexpected_revision_does_not_mutate_import_state(monkeypatch, tmp_path) -> None:
    before = list(sys.path)

    def run(command, **kwargs):
        del kwargs
        output = str(tmp_path) if command[-2:] == ["rev-parse", "--show-toplevel"] else "0" * 40
        return SimpleNamespace(stdout=output + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="unexpected RTT revision"):
        install_rtt_compat(tmp_path)
    assert sys.path == before
    assert "mcore_adapter.patcher" not in sys.modules


def test_missing_utils_digest_does_not_mutate_import_state(monkeypatch, tmp_path) -> None:
    path = tmp_path / "roll/pipeline/rlvr"
    path.mkdir(parents=True)
    (path / "utils.py").write_text("unexpected", encoding="utf-8")

    def run(command, **kwargs):
        del kwargs
        if command[-2:] == ["rev-parse", "--show-toplevel"]:
            output = str(tmp_path)
        elif command[-3:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
            output = ""
        else:
            output = RTT_REVISION
        return SimpleNamespace(stdout=output + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    before = list(sys.path)
    with pytest.raises(RuntimeError, match="unexpected RTT utils digest"):
        install_rtt_compat(tmp_path)
    assert sys.path == before
    assert "mcore_adapter.patcher" not in sys.modules


def test_dirty_checkout_fails_before_import_or_mutation(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    before = list(sys.path)

    def run(command, **kwargs):
        del kwargs
        args = command[3:]
        outputs = {
            ("rev-parse", "--show-toplevel"): str(root),
            ("rev-parse", "HEAD"): RTT_REVISION,
            ("status", "--porcelain=v1", "--untracked-files=all"): "?? foreign.py",
        }
        return SimpleNamespace(stdout=outputs[tuple(args)] + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: pytest.fail(f"dirty checkout imported {name}"),
    )
    with pytest.raises(RuntimeError, match="must be clean"):
        install_rtt_compat(root)
    assert sys.path == before
    assert "mcore_adapter.patcher" not in sys.modules


@pytest.mark.parametrize(
    "foreign_name",
    ["roll.pipeline.rlvr.utils", "roll.distributed.strategy.megatron_strategy"],
)
def test_foreign_roll_module_fails_before_roll_function_mutation(monkeypatch, tmp_path, foreign_name) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    foreign = SimpleNamespace(__file__=str(tmp_path / "foreign/utils.py"))
    modules = {
        "roll.pipeline.rlvr.utils": SimpleNamespace(__file__=str(root / "roll/pipeline/rlvr/utils.py")),
        "roll.distributed.strategy.megatron_strategy": SimpleNamespace(
            __file__=str(root / "roll/distributed/strategy/megatron_strategy.py"),
            MegatronInferStrategy=_strategy_class(),
        ),
    }
    monkeypatch.setattr(importlib, "import_module", lambda name: foreign if name == foreign_name else modules[name])
    with pytest.raises(RuntimeError, match="unexpected path"):
        install_rtt_compat(root)
    assert str(root / "mcore_adapter/src") == sys.path[0]
    assert sys.modules["mcore_adapter.patcher"].__rdan_compat_owner__ == "rdan-grpo:rtt-b1ab2fb"
    assert not hasattr(modules["roll.pipeline.rlvr.utils"], "dump_batch_to_reward_system")
    assert not hasattr(
        modules["roll.distributed.strategy.megatron_strategy"].MegatronInferStrategy.inner_forward_step,
        "__rdan_compat_owner__",
    )


@pytest.mark.parametrize(
    "foreign_name",
    ["roll.pipeline.rlvr.utils", "roll.distributed.strategy.megatron_strategy"],
)
def test_preloaded_foreign_roll_module_fails_before_import(monkeypatch, tmp_path, foreign_name) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    before = list(sys.path)
    foreign = ModuleType(foreign_name)
    foreign.__file__ = str(tmp_path / "foreign.py")
    monkeypatch.setitem(sys.modules, foreign_name, foreign)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: pytest.fail(f"preloaded conflict imported {name}"),
    )
    with pytest.raises(RuntimeError, match="unexpected path"):
        install_rtt_compat(root)
    assert sys.path == before
    assert "mcore_adapter.patcher" not in sys.modules


def test_foreign_roll_resolution_fails_before_import_or_mutation(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    before = list(sys.path)
    monkeypatch.setattr(compat, "_module_spec_path", lambda name: (tmp_path / "foreign.py").resolve())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: pytest.fail(f"foreign resolution imported {name}"),
    )
    with pytest.raises(RuntimeError, match="resolved .* unexpected path"):
        install_rtt_compat(root)
    assert sys.path == before
    assert "mcore_adapter.patcher" not in sys.modules


def test_installer_exposes_only_pinned_patcher_api(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    utils = SimpleNamespace()
    _mock_roll_imports(monkeypatch, utils, root)
    install_rtt_compat(root)

    patcher = sys.modules["mcore_adapter.patcher"]
    public = {name for name in vars(patcher) if not name.startswith("__")}
    assert public == {"patch_torch_find_nd_overlapping_shards", "patch_torch_validate_global_plan"}
    assert str(root / "mcore_adapter/src") == sys.path[0]
    assert utils.dump_batch_to_reward_system is dump_batch_to_reward_system
    install_rtt_compat(root)


def test_installer_skips_optional_megatron_when_dependency_is_absent(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    utils = SimpleNamespace(__file__=str(root / "roll/pipeline/rlvr/utils.py"))
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        if name != "roll.pipeline.rlvr.utils":
            pytest.fail(f"optional Megatron module was imported: {name}")
        return utils

    monkeypatch.setattr(compat, "_megatron_core_available", lambda: False)
    monkeypatch.setattr(importlib, "import_module", import_module)
    install_rtt_compat(root)

    assert imported == ["roll.pipeline.rlvr.utils"]
    assert utils.dump_batch_to_reward_system is dump_batch_to_reward_system


def test_installer_rejects_loaded_conflicting_patcher(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    _mock_roll_imports(monkeypatch, root=root)
    monkeypatch.setitem(sys.modules, "mcore_adapter.patcher", ModuleType("mcore_adapter.patcher"))
    with pytest.raises(RuntimeError, match="conflicting module is already loaded"):
        install_rtt_compat(root)
    assert str(root / "mcore_adapter/src") not in sys.path


def test_installer_rejects_forged_compat_owner(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    _mock_roll_imports(monkeypatch, root=root)
    patcher = ModuleType("mcore_adapter.patcher")
    patcher.__rdan_compat_owner__ = "rdan-grpo:rtt-b1ab2fb"
    monkeypatch.setitem(sys.modules, "mcore_adapter.patcher", patcher)
    with pytest.raises(RuntimeError, match="conflicting module is already loaded"):
        install_rtt_compat(root)


def test_installer_exposes_patcher_on_loaded_pinned_package(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path)
    parent = ModuleType("mcore_adapter")
    parent.__path__ = [str(root / "mcore_adapter/src/mcore_adapter")]
    monkeypatch.setitem(sys.modules, "mcore_adapter", parent)
    _mock_roll_imports(monkeypatch, root=root)
    install_rtt_compat(root)
    assert parent.patcher is sys.modules["mcore_adapter.patcher"]


def test_installer_accepts_byte_identical_installed_package(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path / "rtt")
    installed = tmp_path / "site-packages/mcore_adapter"
    shutil.copytree(root / "mcore_adapter/src/mcore_adapter", installed)
    parent = ModuleType("mcore_adapter")
    parent.__path__ = [str(installed)]
    monkeypatch.setitem(sys.modules, "mcore_adapter", parent)
    _mock_roll_imports(monkeypatch, root=root)
    install_rtt_compat(root)
    assert parent.patcher is sys.modules["mcore_adapter.patcher"]


def test_installer_rejects_modified_installed_package(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path / "rtt")
    _mock_roll_imports(monkeypatch, root=root)
    installed = tmp_path / "site-packages/mcore_adapter"
    shutil.copytree(root / "mcore_adapter/src/mcore_adapter", installed)
    (installed / "__init__.py").write_text("modified", encoding="utf-8")
    parent = ModuleType("mcore_adapter")
    parent.__path__ = [str(installed)]
    monkeypatch.setitem(sys.modules, "mcore_adapter", parent)
    with pytest.raises(RuntimeError, match="conflicting mcore_adapter package"):
        install_rtt_compat(root)
    assert str(root / "mcore_adapter/src") not in sys.path


def test_installer_rejects_real_conflicting_patcher(monkeypatch, tmp_path) -> None:
    root = _fake_rtt(monkeypatch, tmp_path / "rtt")
    _mock_roll_imports(monkeypatch, root=root)
    conflict = tmp_path / "conflict/mcore_adapter"
    conflict.mkdir(parents=True)
    (conflict / "__init__.py").write_text("", encoding="utf-8")
    (conflict / "patcher.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(conflict.parent))
    with pytest.raises(RuntimeError, match="conflicting real module exists"):
        install_rtt_compat(root)
    assert str(root / "mcore_adapter/src") not in sys.path


def test_overlap_patch_detects_overlap_and_shared_boundary() -> None:
    original = torch.distributed._shard.sharding_spec._internals._find_nd_overlapping_shards
    try:
        patch_torch_find_nd_overlapping_shards()
        check = torch.distributed._shard.sharding_spec._internals._find_nd_overlapping_shards
        overlap = [_shard([0, 0], [4, 4]), _shard([2, 1], [4, 2])]
        boundary = [_shard([0, 0], [4, 4]), _shard([4, 0], [4, 4])]
        assert check(overlap, [0, 1]) == (0, 1)
        assert check(boundary, [0, 1]) is None
    finally:
        torch.distributed._shard.sharding_spec._internals._find_nd_overlapping_shards = original


def test_global_plan_patch_rejects_overlap_and_accepts_partition() -> None:
    original = torch.distributed.checkpoint.default_planner._validate_global_plan
    try:
        patch_torch_validate_global_plan()
        check = torch.distributed.checkpoint.default_planner._validate_global_plan
        overlap = _metadata([([0, 0], [3, 4]), ([2, 0], [2, 4])])
        partition = _metadata([([0, 0], [2, 4]), ([2, 0], [2, 4])])
        plans = [SavePlan([]), SavePlan([])]
        assert check(plans, overlap) is False
        assert check(plans, partition) is True
    finally:
        torch.distributed.checkpoint.default_planner._validate_global_plan = original


def test_missing_reward_hook_is_noop_only_when_disabled() -> None:
    dump_batch_to_reward_system(SimpleNamespace(meta_info={}), None)
    dump_batch_to_reward_system(SimpleNamespace(meta_info={"reward_system_config": {}}), None)
    with pytest.raises(RuntimeError, match="cannot be emulated"):
        dump_batch_to_reward_system(SimpleNamespace(meta_info={"reward_system_config": {"sink": "enabled"}}), None)


def test_local_dense_qwen_uses_internal_causal_mask_without_mutating_batch(monkeypatch) -> None:
    strategy = _patched_strategy(monkeypatch)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    batch = {"input_ids": torch.tensor([[3, 4, 0], [5, 6, 7]]), "attention_mask": mask}
    data = SimpleNamespace(batch=batch, non_tensor_batch={})
    seen = []

    def model(**kwargs):
        seen.append(kwargs["attention_mask"])
        return "output"

    model.config = _model_config()
    instance = _strategy_instance()
    original_mask = mask.clone()
    output = strategy.inner_forward_step(instance, None, iter((data,)), model)

    assert output == "output"
    assert seen == [None]
    assert batch["attention_mask"] is mask
    assert torch.equal(mask, original_mask)


def test_transformer_engine_path_is_untouched(monkeypatch) -> None:
    strategy = _patched_strategy(monkeypatch)
    mask = torch.tensor([[1, 1, 0]])
    data = SimpleNamespace(batch={"input_ids": torch.tensor([[3, 4, 0]]), "attention_mask": mask})
    seen = []

    def model(**kwargs):
        seen.append(kwargs["attention_mask"])

    model.config = _model_config(transformer_impl="transformer_engine")
    instance = _strategy_instance(transformer_impl="transformer_engine")
    strategy.inner_forward_step(instance, None, iter((data,)), model)
    assert seen == [mask]


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([1, 1, 0]),
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[0, 1, 1]]),
        torch.tensor([[0, 0, 0]]),
    ],
)
def test_local_dense_qwen_rejects_unsupported_masks(monkeypatch, mask) -> None:
    strategy = _patched_strategy(monkeypatch)
    input_ids = torch.zeros(mask.shape, dtype=torch.long)
    data = SimpleNamespace(batch={"input_ids": input_ids, "attention_mask": mask}, non_tensor_batch={})

    def model(**kwargs):
        raise AssertionError("invalid batches must not reach the model")

    model.config = _model_config()
    with pytest.raises(RuntimeError, match="local Qwen3 compatibility requires"):
        strategy.inner_forward_step(_strategy_instance(), None, iter((data,)), model)


@pytest.mark.parametrize(
    "mode",
    ["packing", "moe", "multimodal"],
)
def test_unsupported_model_modes_are_untouched(monkeypatch, mode) -> None:
    strategy = _patched_strategy(monkeypatch)
    instance = _strategy_instance(use_sequence_packing=mode == "packing")
    config = _model_config(hf_model_type="qwen3_moe", num_moe_experts=8) if mode == "moe" else _model_config()
    non_tensor_batch = {"multi_modal_inputs": [object()]} if mode == "multimodal" else {}
    mask = torch.tensor([[1, 1, 0]])
    data = SimpleNamespace(
        batch={"input_ids": torch.tensor([[3, 4, 0]]), "attention_mask": mask},
        non_tensor_batch=non_tensor_batch,
    )
    seen = []

    def model(**kwargs):
        seen.append(kwargs["attention_mask"])

    model.config = config
    strategy.inner_forward_step(instance, None, iter((data,)), model)
    assert seen == [mask]


def _fake_rtt(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    utils = root / "roll/pipeline/rlvr/utils.py"
    utils.parent.mkdir(parents=True)
    utils.write_bytes(b"pinned")
    strategy = root / "roll/distributed/strategy/megatron_strategy.py"
    strategy.parent.mkdir(parents=True)
    strategy.write_bytes(b"pinned strategy")
    package = root / "mcore_adapter/src/mcore_adapter"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    def run(command, **kwargs):
        del kwargs
        args = command[3:]
        if args == ["rev-parse", "--show-toplevel"]:
            output = str(root)
        elif args == ["status", "--porcelain=v1", "--untracked-files=all"]:
            output = ""
        else:
            output = RTT_REVISION
        return SimpleNamespace(stdout=output + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(compat, "RTT_UTILS_SHA256", hashlib.sha256(b"pinned").hexdigest())
    monkeypatch.setattr(compat, "RTT_MEGATRON_SHA256", hashlib.sha256(b"pinned strategy").hexdigest())
    monkeypatch.setattr(compat, "_megatron_core_available", lambda: True)
    paths = {
        "roll.pipeline.rlvr.utils": root / "roll/pipeline/rlvr/utils.py",
        "roll.distributed.strategy.megatron_strategy": root / "roll/distributed/strategy/megatron_strategy.py",
    }
    monkeypatch.setattr(compat, "_module_spec_path", lambda name: paths[name].resolve())
    return root


def _sync_hf_payload() -> dict:
    return {
        "pretrain": "/models/qwen",
        "response_length": 16,
        "rewards": None,
        "async_generation_ratio": 0,
        "generate_opt_level": 0,
        "actor_train": {
            "worker_cls": compat._ACTOR_WORKER,
            "strategy_args": {"strategy_name": "fsdp2_train", "strategy_config": {}},
            "device_mapping": "list(range(0, 2))",
            "num_gpus_per_worker": 1,
        },
        "actor_infer": {
            "worker_cls": compat._INFER_WORKER,
            "generating_args": {"num_return_sequences": 1},
            "strategy_args": {"strategy_name": "hf_infer", "strategy_config": {}},
            "device_mapping": [0, 1],
            "num_gpus_per_worker": 1,
            "max_concurrency": 1,
        },
    }


def _constructed_sync_config(max_concurrency: int, infer_strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        async_generation_ratio=0,
        async_pipeline=False,
        generate_opt_level=0,
        rewards={"stale": object()},
        domain_2_tag={"stale": {"tag"}},
        tag_2_domain={"tag": "stale"},
        actor_train=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name="fsdp2_train"),
            worker_cls=compat._ACTOR_WORKER,
            device_mapping=[0, 1],
            num_gpus_per_worker=1,
            world_size=2,
        ),
        actor_infer=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name=infer_strategy),
            worker_cls=compat._INFER_WORKER,
            device_mapping=[0, 1],
            num_gpus_per_worker=1,
            world_size=2,
            max_concurrency=max_concurrency,
        ),
    )


def _mock_roll_imports(
    monkeypatch: pytest.MonkeyPatch,
    utils: object | None = None,
    root: Path = RTT,
) -> type:
    strategy = _strategy_class()
    utils_module = utils or SimpleNamespace()
    utils_module.__file__ = str(root / "roll/pipeline/rlvr/utils.py")
    strategy_module = SimpleNamespace(
        MegatronInferStrategy=strategy,
        __file__=str(root / "roll/distributed/strategy/megatron_strategy.py"),
    )
    modules = {
        "roll.distributed.strategy.megatron_strategy": strategy_module,
        "roll.pipeline.rlvr.utils": utils_module,
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    return strategy


def _patched_strategy(monkeypatch: pytest.MonkeyPatch) -> type:
    strategy = _mock_roll_imports(monkeypatch)
    compat._install_local_qwen_mask_patch()
    return strategy


def _strategy_class() -> type:
    class Strategy:
        def inner_forward_step(self, loss_func, data_iterator, model):
            del loss_func
            data = next(data_iterator)
            return model(
                input_ids=data.batch["input_ids"],
                attention_mask=data.batch["attention_mask"],
                position_ids=None,
            )

    return Strategy


def _strategy_instance(transformer_impl: str = "local", use_sequence_packing: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        megatron_train_args=SimpleNamespace(transformer_impl=transformer_impl),
        use_sequence_packing=use_sequence_packing,
    )


def _model_config(
    transformer_impl: str = "local", hf_model_type: str = "qwen3", num_moe_experts: int | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        transformer_impl=transformer_impl,
        hf_model_type=hf_model_type,
        num_moe_experts=num_moe_experts,
    )


def _shard(offsets: list[int], sizes: list[int]) -> ShardMetadata:
    return ShardMetadata(shard_offsets=offsets, shard_sizes=sizes, placement="rank:0/cpu")


def _metadata(chunks: list[tuple[list[int], list[int]]]) -> Metadata:
    storage = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size([4, 4]),
        chunks=[ChunkStorageMetadata(offsets=torch.Size(offset), sizes=torch.Size(size)) for offset, size in chunks],
    )
    return Metadata(state_dict_metadata={"weight": storage})
