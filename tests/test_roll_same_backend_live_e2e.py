from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY

ROOT = Path(__file__).resolve().parents[1]


class FakeData:
    def __init__(self, batch: dict[str, torch.Tensor], meta_info: dict[str, Any] | None = None) -> None:
        self.batch = batch
        self.meta_info = meta_info or {}

    def __len__(self) -> int:
        return len(next(iter(self.batch.values())))

    def clone(self) -> FakeData:
        return FakeData({name: tensor.clone() for name, tensor in self.batch.items()}, dict(self.meta_info))

    @classmethod
    def from_single_dict(cls, value: dict[str, torch.Tensor]) -> FakeData:
        return cls(value)


class FakeRemote:
    def __init__(self, result: Any = None, event: str | None = None, events: list[Any] | None = None) -> None:
        self.result = result
        self.event = event
        self.events = events

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        if self.events is not None:
            self.events.append((self.event, args, kwargs))
        return self.result


def _load_live(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    ray = types.ModuleType("ray")
    ray.get = lambda value: value
    collator = types.ModuleType("roll.datasets.collator")
    collator.DataCollatorWithPaddingForPaddedKeys = object
    dataset = types.ModuleType("roll.datasets.dataset")
    dataset.get_dataset = lambda _: []
    cluster = types.ModuleType("roll.distributed.executor.cluster")
    cluster.Cluster = object
    decorator = types.ModuleType("roll.distributed.scheduler.decorator")
    decorator.Dispatch = SimpleNamespace(ONE_TO_ALL=1)
    decorator.register = lambda **_kwargs: lambda function: function
    protocol = types.ModuleType("roll.distributed.scheduler.protocol")
    protocol.DataProto = FakeData
    providers = types.ModuleType("roll.models.model_providers")
    providers.default_tokenizer_provider = lambda **_kwargs: object()

    class BasePipeline:
        def __init__(self, config: Any) -> None:
            self.pipeline_config = config
            self.resource_manager = object()
            self.state = SimpleNamespace(step=0)

        def download_models(self, *clusters: Any) -> None:
            del clusters

        def set_model_update_pair(self, **kwargs: Any) -> None:
            del kwargs

    base = types.ModuleType("roll.pipeline.base_pipeline")
    base.BasePipeline = BasePipeline
    rubric = types.ModuleType("roll.pipeline.rlvr.rubircs_pipeline")
    rubric.get_encode_function = lambda *args: object()
    rubric.preprocess_dataset = lambda data, *args: data
    rubric.update_dataset_domain = lambda mapping, row: row
    receipt_hook = types.ModuleType("rdan_grpo.roll_fsdp_hf_receipt")
    for name in (
        "begin_fsdp_hf_receipt",
        "begin_hf_infer_receipt",
        "finish_hf_infer_receipt",
        "get_fsdp_actor_receipt",
        "run_receipted_fsdp_hf_update",
    ):
        setattr(receipt_hook, name, lambda *args, **kwargs: None)

    class Actor:
        def start_model_update(self, name: str) -> FakeData:
            return FakeData({}, {"name": name})

    class Infer:
        pass

    same = types.ModuleType("rdan_grpo.roll_same_backend")
    same.ObservedFSDP2ActorWorker = Actor
    same.SynchronousHFInferWorker = Infer
    modules = {
        "ray": ray,
        "roll": types.ModuleType("roll"),
        "roll.datasets": types.ModuleType("roll.datasets"),
        "roll.datasets.collator": collator,
        "roll.datasets.dataset": dataset,
        "roll.distributed": types.ModuleType("roll.distributed"),
        "roll.distributed.executor": types.ModuleType("roll.distributed.executor"),
        "roll.distributed.executor.cluster": cluster,
        "roll.distributed.scheduler": types.ModuleType("roll.distributed.scheduler"),
        "roll.distributed.scheduler.decorator": decorator,
        "roll.distributed.scheduler.protocol": protocol,
        "roll.models": types.ModuleType("roll.models"),
        "roll.models.model_providers": providers,
        "roll.pipeline": types.ModuleType("roll.pipeline"),
        "roll.pipeline.base_pipeline": base,
        "roll.pipeline.rlvr": types.ModuleType("roll.pipeline.rlvr"),
        "roll.pipeline.rlvr.rubircs_pipeline": rubric,
        "rdan_grpo.roll_fsdp_hf_receipt": receipt_hook,
        "rdan_grpo.roll_same_backend": same,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/rdan_grpo/roll_same_backend_live.py"
    spec = importlib.util.spec_from_file_location("test_roll_same_backend_live", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _config(module: types.ModuleType, worker_paths: bool = False) -> SimpleNamespace:
    actor_cls = module.ACTOR_WORKER_PATH if worker_paths else module.ReceiptedFSDP2ActorWorker
    infer_cls = module.INFER_WORKER_PATH if worker_paths else module.ReceiptedSynchronousHFInferWorker
    actor_train = SimpleNamespace(
        worker_cls=actor_cls,
        strategy_args=SimpleNamespace(strategy_name="fsdp2_train"),
        device_mapping=[0, 1],
        num_gpus_per_worker=1,
        world_size=2,
    )
    actor_infer = SimpleNamespace(
        worker_cls=infer_cls,
        strategy_args=SimpleNamespace(strategy_name="hf_infer"),
        device_mapping=[0, 1],
        num_gpus_per_worker=1,
        world_size=2,
        max_concurrency=1,
    )
    return SimpleNamespace(
        actor_train=actor_train,
        actor_infer=actor_infer,
        async_pipeline=False,
        async_generation_ratio=0,
        generate_opt_level=0,
        global_template="qwen3_nothinking",
        rewards={},
        track_with="stdout",
        tracker_kwargs={},
        max_steps=1,
    )


def test_pinned_worker_paths_resolve_at_the_real_pipeline_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_live(monkeypatch)
    config = _config(module, worker_paths=True)
    constructed: list[Any] = []
    monkeypatch.setattr(module, "SameBackendParityPipeline", lambda value: constructed.append(value) or "pipeline")

    assert module.build_same_backend_pipeline(config) == "pipeline"
    assert constructed == [config]
    assert config.actor_train.worker_cls is module.ReceiptedFSDP2ActorWorker
    assert config.actor_infer.worker_cls is module.ReceiptedSynchronousHFInferWorker
    tampered = _config(module, worker_paths=True)
    tampered.actor_infer.worker_cls = "rdan_grpo.bad.Worker"
    with pytest.raises(ValueError, match="actor_infer.worker_cls"):
        module.build_same_backend_pipeline(tampered)


def test_topology_sampling_and_rewardless_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_live(monkeypatch)
    config = _config(module)
    module._validate_pipeline_config(config)
    config.actor_infer.max_concurrency = 2
    with pytest.raises(ValueError, match="max_concurrency=1"):
        module._validate_pipeline_config(config)
    config.actor_infer.max_concurrency = 1
    config.rewards = {"judge": object()}
    with pytest.raises(ValueError, match="reward workers"):
        module._validate_pipeline_config(config)

    payload = (ROOT / "configs/roll/qwen_scalar_same_backend_parity.yaml").read_text(encoding="utf-8")
    assert "do_sample: true" in payload
    assert "temperature: 1.0" in payload
    assert "top_p: 1.0" in payload
    assert "top_k: 0" in payload
    assert "num_beams: 1" in payload
    assert "max_concurrency: 1" in payload
    assert "rewards: null" in payload


def test_same_backend_config_resolves_the_real_rtt_loader_path(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = yaml.safe_load((ROOT / "configs/roll/qwen_scalar_train.yaml").read_text(encoding="utf-8"))
    child = yaml.safe_load((ROOT / "configs/roll/qwen_scalar_same_backend_parity.yaml").read_text(encoding="utf-8"))
    assert "data_args" not in child["actor_train"]
    parent_data = parent["actor_train"]["data_args"]
    calls: list[tuple[str, list[str]]] = []
    datasets = types.ModuleType("datasets")
    datasets.Dataset = type("Dataset", (), {})
    datasets.IterableDataset = type("IterableDataset", (), {})

    def load_dataset(kind: str, data_files: list[str], **kwargs: Any) -> dict[str, str]:
        del kwargs
        calls.append((kind, data_files))
        return {"train": "loaded"}

    datasets.load_dataset = load_dataset
    configs = types.ModuleType("roll.configs")
    configs.DataArguments = object
    data_args_module = types.ModuleType("roll.configs.data_args")
    data_args_module.DataArguments = object
    logging = types.ModuleType("roll.utils.logging")
    logging.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)
    for name, module in {
        "datasets": datasets,
        "roll": types.ModuleType("roll"),
        "roll.configs": configs,
        "roll.configs.data_args": data_args_module,
        "roll.utils": types.ModuleType("roll.utils"),
        "roll.utils.logging": logging,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT.parent / "Rubrics-To-Tokens/roll/datasets/dataset.py"
    spec = importlib.util.spec_from_file_location("test_pinned_rtt_dataset", path)
    assert spec is not None and spec.loader is not None
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    monkeypatch.chdir(ROOT)

    result = loader.get_dataset(
        SimpleNamespace(
            file_name=parent_data["file_name"],
            dataset_dir=parent_data["dataset_dir"],
            dataset_type="json",
            prompt=None,
            response="solution",
        )
    )

    assert result == "loaded"
    assert calls == [("json", ["./data/HIR_trainv1_rdan_scalar_certified.jsonl"])]


def test_receipt_seals_before_generation_and_preserves_zero_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_live(monkeypatch)
    pipeline = module.SameBackendParityPipeline.__new__(module.SameBackendParityPipeline)
    pipeline._receipt_passed = False
    pipeline._generation_started = False
    pipeline._optimizer_updates = 0
    pipeline._pipeline_steps = 0
    pipeline.state = SimpleNamespace(step=0)
    pipeline.pipeline_config = SimpleNamespace(prompt_length=4)
    pipeline.dataset = [object()] * 4
    pipeline.tokenizer = object()
    events: list[Any] = []
    actor_receipts = [{"side": "actor", "rank": rank} for rank in range(2)]
    infer_receipts = [{"side": "infer", "rank": rank} for rank in range(2)]
    actor_workers = [
        SimpleNamespace(
            rdan_begin_fsdp_hf_receipt=FakeRemote(event=f"actor-begin-{rank}", events=events),
            rdan_get_fsdp_hf_receipt=FakeRemote(actor_receipts[rank], f"actor-get-{rank}", events),
        )
        for rank in range(2)
    ]
    infer_workers = [
        SimpleNamespace(
            rdan_begin_fsdp_hf_receipt=FakeRemote(event=f"infer-begin-{rank}", events=events),
            rdan_finish_fsdp_hf_receipt=FakeRemote(infer_receipts[rank], f"infer-finish-{rank}", events),
        )
        for rank in range(2)
    ]
    pipeline.actor_train = SimpleNamespace(workers=actor_workers)
    pipeline.actor_infer = SimpleNamespace(workers=infer_workers)
    pipeline.model_update = lambda step: events.append(("update", step))
    built: dict[str, Any] = {}

    def build(actors: Any, infers: Any, **kwargs: Any) -> dict[str, Any]:
        built.update(actors=actors, infers=infers, kwargs=kwargs)
        return {"status": "receipt_passed", "transaction_id": kwargs["transaction_id"]}

    monkeypatch.setattr(module, "build_fsdp_hf_receipt_artifact", build)
    monkeypatch.setattr(module, "seal_fsdp_hf_receipt", lambda path, artifact: events.append(("seal", path, artifact)))
    output = tmp_path / "receipt.json"
    artifact = pipeline.seal_weight_receipt(
        output,
        model_identity=SimpleNamespace(),
        resolved_config_sha256="a" * 64,
        rtt_revision="b" * 40,
        rtt_boundary_sha256={},
        generation_source_identity=GENERATION_SOURCE_IDENTITY,
    )

    labels = [event[0] if isinstance(event, tuple) else event for event in events]
    assert labels.index("update") > labels.index("infer-begin-1")
    assert labels.index("seal") > labels.index("infer-finish-1")
    assert pipeline._receipt_passed is True
    assert built["kwargs"]["optimizer_updates"] == built["kwargs"]["pipeline_steps"] == 0
    assert built["kwargs"]["generation_started_before_seal"] is False
    assert artifact["transaction_id"]

    blocked = module.SameBackendParityPipeline.__new__(module.SameBackendParityPipeline)
    blocked._receipt_passed = False
    blocked._generation_started = False
    with pytest.raises(RuntimeError, match="before generation"):
        blocked.collect_parity(32, {"num_return_sequences": 8})


def test_receipt_link_rebuilds_exact_pairing_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_live(monkeypatch)
    receipt = {
        "id": "qwen_a100_fsdp2_hf_weight_receipt_v1",
        "status": "receipt_passed",
        "transaction_id": "tx",
        "runtime": {
            "resolved_config_sha256": "a" * 64,
            "rtt_revision": "b" * 40,
            "rtt_boundary_sha256": {},
            **GENERATION_SOURCE_IDENTITY,
        },
        "model": {"model": "qwen"},
        "actor_receipts": [{"rank": 0}],
        "infer_receipts": [{"rank": 0}],
        "optimizer_updates": 0,
        "pipeline_steps": 0,
        "generation_started_before_seal": False,
    }
    rebuilt = dict(receipt)
    captured: list[dict[str, str]] = []

    def rebuild(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        captured.append(kwargs["generation_source_identity"])
        return dict(rebuilt)

    monkeypatch.setattr(module, "build_fsdp_hf_receipt_artifact", rebuild)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    link = module.build_fsdp_hf_receipt_link(path, "a" * 64)
    assert link["transaction_id"] == "tx"
    assert len(link["artifact_sha256"]) == 64
    assert captured == [GENERATION_SOURCE_IDENTITY]
    rebuilt["transaction_id"] = "different"
    with pytest.raises(module.FSDPHFReceiptError, match="evidence is invalid"):
        module.build_fsdp_hf_receipt_link(path, "a" * 64)
    receipt["runtime"].pop("generation_sample_sha256")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(module.FSDPHFReceiptError, match="malformed"):
        module.build_fsdp_hf_receipt_link(path, "a" * 64)


def test_cli_fails_before_rtt_import_ray_or_artifacts_on_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_same_backend_parity",
        ROOT / "scripts/run_same_backend_parity.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    output = tmp_path / "success.json"
    output.write_text("preserve\n", encoding="utf-8")
    checked = []
    monkeypatch.setattr(script, "verify_fsdp_hf_checkout", lambda root: checked.append(root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_same_backend_parity.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--production-config",
            str(tmp_path / "production.yaml"),
            "--preflight-config",
            str(tmp_path / "preflight.yaml"),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--output",
            str(output),
            "--failure-output",
            str(tmp_path / "failure.json"),
            "--weight-receipt-output",
            str(tmp_path / "receipt.json"),
            "--rtt-root",
            str(tmp_path / "rtt"),
        ],
    )

    with pytest.raises(FileExistsError, match="success.json"):
        script.main()
    assert output.read_text(encoding="utf-8") == "preserve\n"
    assert checked == []
    assert not (tmp_path / "failure.json").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_cli_rejects_generation_source_drift_before_pipeline_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_same_backend_parity_generation_drift",
        ROOT / "scripts/run_same_backend_parity.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    snapshot = tmp_path / script.MODEL_REVISION
    snapshot.mkdir()
    checked: list[Path] = []
    monkeypatch.setattr(script, "verify_fsdp_hf_checkout", lambda root: checked.append(Path(root)) or {})
    monkeypatch.setattr(
        script,
        "verify_transformers_generation_boundary",
        lambda: (_ for _ in ()).throw(script.ParityError("generation source drift")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_same_backend_parity.py",
            "--config",
            str(config),
            "--production-config",
            str(tmp_path / "production.yaml"),
            "--preflight-config",
            str(tmp_path / "preflight.yaml"),
            "--snapshot",
            str(snapshot),
            "--output",
            str(tmp_path / "success.json"),
            "--failure-output",
            str(tmp_path / "failure.json"),
            "--weight-receipt-output",
            str(tmp_path / "receipt.json"),
            "--rtt-root",
            str(tmp_path / "rtt"),
        ],
    )

    with pytest.raises(script.ParityError, match="source drift"):
        script.main()

    assert checked == [tmp_path / "rtt"]
    assert not (tmp_path / "success.json").exists()
    assert not (tmp_path / "failure.json").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_production_config_requires_training_topology_and_rejects_diagnostic_workers(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_same_backend_parity_production_config",
        ROOT / "scripts/run_same_backend_parity.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    snapshot = tmp_path / script.MODEL_REVISION
    snapshot.mkdir()
    actor = {
        "worker_cls": "rdan_grpo.roll_same_backend_train.ActorWorker",
        "model_args": {"model_name_or_path": str(snapshot)},
        "strategy_args": {"strategy_name": "fsdp2_train", "strategy_config": {"transformer_impl": "huggingface"}},
        "device_mapping": [0, 1],
        "world_size": 2,
        "num_gpus_per_worker": 1,
    }
    infer = {
        **actor,
        "worker_cls": "rdan_grpo.roll_same_backend_train.InferWorker",
        "strategy_args": {"strategy_name": "hf_infer", "strategy_config": {"transformer_impl": "huggingface"}},
    }
    payload = {
        "actor_train": actor,
        "actor_infer": infer,
        "async_pipeline": False,
        "async_generation_ratio": 0,
        "generate_opt_level": 0,
        "max_steps": 20,
        "rewards": {"llm_judge": {}},
    }

    script._validate_production_config(payload, snapshot)

    payload["actor_train"]["worker_cls"] = "rdan_grpo.roll_same_backend_live.ReceiptedFSDP2ActorWorker"
    with pytest.raises(script.ParityError, match="diagnostic-only workers"):
        script._validate_production_config(payload, snapshot)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["actor_train"].update(device_mapping=[0]),
        lambda value: value["actor_infer"].update(world_size=1),
        lambda value: value.update(async_pipeline=True),
        lambda value: value.update(generate_opt_level=1),
        lambda value: value.update(max_steps=1),
        lambda value: value.update(rewards={}),
    ],
)
def test_production_config_rejects_training_invariant_drift(tmp_path: Path, mutate: object) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_same_backend_parity_production_drift",
        ROOT / "scripts/run_same_backend_parity.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    snapshot = tmp_path / script.MODEL_REVISION
    snapshot.mkdir()
    base = {
        "model_args": {"model_name_or_path": str(snapshot)},
        "strategy_args": {"strategy_config": {"transformer_impl": "huggingface"}},
        "device_mapping": [0, 1],
        "world_size": 2,
        "num_gpus_per_worker": 1,
    }
    payload = {
        "actor_train": {
            **base,
            "worker_cls": "rdan_grpo.roll_same_backend_train.ActorWorker",
            "strategy_args": {**base["strategy_args"], "strategy_name": "fsdp2_train"},
        },
        "actor_infer": {
            **base,
            "worker_cls": "rdan_grpo.roll_same_backend_train.InferWorker",
            "strategy_args": {**base["strategy_args"], "strategy_name": "hf_infer"},
        },
        "async_pipeline": False,
        "async_generation_ratio": 0,
        "generate_opt_level": 0,
        "max_steps": 20,
        "rewards": {"llm_judge": {}},
    }
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(script.ParityError, match="same-backend production config"):
        script._validate_production_config(payload, snapshot)


def test_cli_success_and_receipt_failure_are_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_same_backend_parity_outcomes",
        ROOT / "scripts/run_same_backend_parity.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(
        "rdan_grpo.runtime_parity.verify_transformers_generation_boundary",
        lambda: dict(GENERATION_SOURCE_IDENTITY),
    )
    identity = script.RuntimeIdentity(
        model="Qwen/Qwen3-4B-Instruct-2507",
        revision="c" * 40,
        snapshot_sha256="1" * 64,
        tokenizer_files_sha256="2" * 64,
        chat_template_sha256="3" * 64,
    )
    config = SimpleNamespace(
        async_pipeline=False,
        async_generation_ratio=0,
        generate_opt_level=0,
        actor_train=SimpleNamespace(
            strategy_args=SimpleNamespace(
                strategy_name="fsdp2_train",
                strategy_config={"transformer_impl": "huggingface"},
            )
        ),
        actor_infer=SimpleNamespace(
            strategy_args=SimpleNamespace(strategy_name="hf_infer"),
            generating_args=SimpleNamespace(to_dict=lambda: {"num_return_sequences": 8}),
        ),
    )
    success_dir = tmp_path / "success"
    success_dir.mkdir()
    success_args = SimpleNamespace(
        output=success_dir / "parity.json",
        failure_output=success_dir / "failure.json",
        weight_receipt_output=success_dir / "receipt.json",
    )
    success_args.weight_receipt_output.write_text("{}\n", encoding="utf-8")
    script._seal_success(success_args, {"status": "parity_passed"})
    assert success_args.output.is_file()
    assert success_args.weight_receipt_output.is_file()
    assert not success_args.failure_output.exists()

    failure_dir = tmp_path / "failure"
    failure_dir.mkdir()
    failure_args = SimpleNamespace(
        output=failure_dir / "parity.json",
        failure_output=failure_dir / "failure.json",
        weight_receipt_output=failure_dir / "receipt.json",
        responses=32,
    )
    failure_args.weight_receipt_output.write_text("{}\n", encoding="utf-8")
    production_config = failure_dir / "production.yaml"
    production_config.write_text("production\n", encoding="utf-8")
    preflight_config = failure_dir / "preflight.yaml"
    preflight_config.write_text("preflight\n", encoding="utf-8")
    script._seal_receipt_failure(
        config,
        identity,
        ROOT / "configs/roll/qwen_scalar_same_backend_parity.yaml",
        "a" * 64,
        {
            "transaction_id": "tx",
            "artifact_sha256": "b" * 64,
            "resolved_config_sha256": "a" * 64,
        },
        production_config,
        "c" * 64,
        preflight_config,
        "d" * 64,
        failure_args,
    )
    assert failure_args.failure_output.is_file()
    assert failure_args.weight_receipt_output.is_file()
    assert not failure_args.output.exists()
    failure = json.loads(failure_args.failure_output.read_text(encoding="utf-8"))
    assert failure["failure"]["code"] == "receipt_failed"
