from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rdan_grpo.checkpoint import ArtifactIdentity, CheckpointIdentity, load_checkpoint
from rdan_grpo.dataset import canonical_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/rdan_grpo/pipeline.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_one_step_calls_exact_transfer_train_checkpoint_order(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pipeline(monkeypatch)
    events: list[str] = []
    pipeline = module.ResponseTrainingPipeline.__new__(module.ResponseTrainingPipeline)
    pipeline._resume_manifest = None
    pipeline.completed_step = 0
    pipeline.stop_after_step = 1
    pipeline.pipeline_config = SimpleNamespace(
        num_return_sequences_in_group=8,
        actor_infer=SimpleNamespace(generating_args=SimpleNamespace(max_new_tokens=4096)),
    )
    pipeline.response_config = SimpleNamespace(method="rdan_scalar", quality_weight=0.5, mix_weight=None)
    pipeline.certificate = {"ready": True}
    pipeline.actor_train = SimpleNamespace(
        rdan_reset_cuda_peak=lambda **kwargs: events.append("reset_actor"),
    )
    pipeline.actor_infer = SimpleNamespace(
        rdan_reset_cuda_peak=lambda **kwargs: events.append("reset_infer"),
    )
    pipeline.scheduler = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: events.append("shutdown")))
    pipeline.tracker = SimpleNamespace(
        log=lambda **kwargs: events.append("log"),
        finish=lambda: events.append("finish"),
    )
    pipeline.state = SimpleNamespace(step=0, log_history=[])
    pipeline._transfer = lambda step: events.append(f"transfer:{step}")

    class FakeRewardedBatch:
        def __init__(self) -> None:
            selected = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            quality = torch.tensor([0.5] * 8)
            self.batch = {
                "rdan_selected_reward": selected,
                "rdan_response_advantage": selected,
                "rdan_raw_quality": quality,
                "rdan_quality_eligible": torch.ones(8, dtype=torch.bool),
                "rdan_quality_advantage": quality,
                "rdan_scalar_advantage": selected,
                "rdan_response_valid": torch.ones(8, dtype=torch.bool),
                "response_mask": torch.ones((8, 4), dtype=torch.bool),
            }

    pipeline._generate = lambda step: events.append(f"generate:{step}") or FakeRewardedBatch()
    pipeline._save_step = lambda *args: events.append("checkpoint") or Path("step-000001")

    def train(**kwargs: Any) -> SimpleNamespace:
        events.append("train")
        kwargs["observe_training_state"]()
        kwargs["observe_post_transaction_memory"]()
        return SimpleNamespace(
            promotion_ready=True,
            metrics={"rdan/response_token_clipfrac": 0.0},
            peak_memory_fraction=0.5,
        )

    pipeline.actor_train.rdan_training_state = lambda **kwargs: events.append("state") or []
    pipeline.actor_train.rdan_cuda_memory = lambda **kwargs: events.append("memory") or []
    monkeypatch.setattr(module, "run_response_train_step", train)
    monkeypatch.setattr(
        module,
        "_group_diagnostics",
        lambda *args: {
            "selected_reward_variance_mean": 0.25,
            "response_active_group_rate": 1.0,
            "quality_active_group_rate": 0.5,
        },
    )

    completed = pipeline.run()
    assert completed.completed_step == 1
    assert completed.checkpoints == (Path("step-000001"),)
    assert events == [
        "transfer:0",
        "reset_actor",
        "reset_infer",
        "generate:1",
        "train",
        "state",
        "memory",
        "transfer:1",
        "checkpoint",
        "log",
        "shutdown",
        "finish",
    ]


def test_cleanup_attempts_both_boundaries_and_preserves_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pipeline(monkeypatch)
    events: list[str] = []

    class Remote:
        def remote(self) -> None:
            events.append("shutdown")
            raise RuntimeError("shutdown failed")

    tracker = SimpleNamespace(finish=lambda: events.append("finish"))
    error = module._cleanup(SimpleNamespace(shutdown=Remote()), tracker)

    assert isinstance(error, RuntimeError)
    assert events == ["shutdown", "finish"]

    pipeline = module.ResponseTrainingPipeline.__new__(module.ResponseTrainingPipeline)
    pipeline._resume_manifest = None
    pipeline.completed_step = 0
    pipeline.stop_after_step = 1
    pipeline.actor_train = SimpleNamespace(rdan_reset_cuda_peak=lambda **kwargs: None)
    pipeline.actor_infer = SimpleNamespace(rdan_reset_cuda_peak=lambda **kwargs: None)
    pipeline.scheduler = SimpleNamespace(shutdown=Remote())
    pipeline.tracker = SimpleNamespace(finish=lambda: events.append("finish-primary"))
    pipeline._transfer = lambda step: None

    def fail_generation(step: int) -> None:
        raise ValueError("primary generation failure")

    pipeline._generate = fail_generation
    with pytest.raises(ValueError, match="primary generation failure"):
        pipeline.run()
    assert events[-2:] == ["shutdown", "finish-primary"]


def test_pipeline_uses_custom_transaction_and_never_stock_checkpoint_upload() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert "run_response_train_step" in calls
    assert "create_checkpoint_stage" in calls
    assert "promote_checkpoint" in calls
    assert "load_checkpoint" in calls
    assert any(name.endswith("get_batch_opt_level_0.remote") for name in calls)
    assert "do_checkpoint" not in calls
    assert not any(name.endswith(".upload") for name in calls)


def test_constructor_resets_shared_lists_before_base_pipeline_construction() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    constructor = _constructor(tree)
    statements = constructor.body
    reset_model = _statement_index(statements, "BasePipeline.model_update_groups")
    reset_checkpoint = _statement_index(statements, "BasePipeline.checkpoint_clusters")
    base_init = next(index for index, node in enumerate(statements) if _contains_call(node, "super.__init__"))
    assert reset_model < base_init
    assert reset_checkpoint < base_init


def test_resume_restores_dcp_driver_inference_rng_before_the_first_transfer() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    restore = source.index("self._restore_checkpoint()")
    run = source.index("def run(")
    run_steps = source.index("def _run_steps(", run)
    first_transfer = source.index("self._transfer(self.completed_step)", run_steps)
    load_dcp = source.index("rdan_load_dcp", restore)
    driver_rng = source.index("WorkerState.load_rng_state", restore)
    inference_rng = source.index("rdan_load_rng.remote", restore)
    assert restore < load_dcp < driver_rng < inference_rng
    assert restore < run < run_steps < first_transfer


def test_transfers_bracket_generation_training_and_atomic_promotion() -> None:
    run_steps = _method_source("_run_steps")
    run_step = _method_source("_run_step")
    save_checkpoint = _method_source("_save_checkpoint")
    assert run_steps.index("self._transfer(self.completed_step)") < run_steps.index("self._run_step")
    assert run_steps.index("self._run_step") < run_steps.index("self._save_step")
    assert run_step.index("self._generate(step)") < run_step.index("run_response_train_step")
    assert run_step.index("run_response_train_step") < run_step.index("self._transfer(step)")
    assert "promote_checkpoint" in save_checkpoint


def test_mixed_response_rows_reach_reward_contract_without_arrow_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_pipeline(monkeypatch)
    type1 = {
        "id": 1,
        "source": "type1",
        "prompt": "Type 1 prompt",
        "rubrics": [{"id": 1, "category": "", "description": "Use English.", "weight": 1}],
        "ground_truth": {"constraint_pattern": ["language:response_language"]},
    }
    type3 = {
        "id": "3",
        "source": "type3",
        "prompt": "Type 3 prompt",
        "rubrics": [{"id": 1, "category": "", "description": "Use JSON.", "weight": 1}],
        "ground_truth": {"constraint_pattern": "detectable_format:json_format"},
    }
    type4 = {
        "id": 4,
        "source": "type4",
        "prompt": "Type 4 prompt",
        "rubrics": [{"id": 1, "category": "", "description": "Use two paragraphs.", "weight": 1}],
        "ground_truth": {"checker": ["[rule] Paragraphs"], "functions": ["def check_following(): pass"]},
    }
    rubrichub_rubrics = [
        {
            "id": 1,
            "category": "rule",
            "description": "Use exactly two paragraphs.",
            "weight": 2,
            "verifier": "rule",
            "function": "ParagraphChecker",
            "parameters": {"num_paragraphs": 2.0},
        },
        {
            "id": 2,
            "category": "llm",
            "description": "The answer is clear.",
            "weight": 1,
            "verifier": "llm",
            "function": "",
            "parameters": {},
        },
    ]
    rubrichub_truth = {
        "hard_mask": [True, False],
        "rubric_routes": [
            {"rubric_index": 0, "function": "ParagraphChecker", "parameters": {"num_paragraphs": 2.0}},
            {"rubric_index": 1, "function": "", "parameters": {}},
        ],
    }
    rubrichub = {
        "id": "rubrichub-1",
        "source": "rubrichub_instruction_following",
        "prompt": "RubricHub prompt",
        "rubrics": rubrichub_rubrics,
        "ground_truth": rubrichub_truth,
    }
    path = tmp_path / "mixed.jsonl"
    rows = (type1, type3, type4, rubrichub)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = module._load_response_dataset(SimpleNamespace(file_name=str(path), dataset_dir="."))
    assert dataset[0]["id"] == "1"
    assert [dataset[index]["prompt"] for index in range(4)] == [row["prompt"] for row in rows]
    assert [dataset[index]["source"] for index in range(4)] == [row["source"] for row in rows]
    assert [json.loads(dataset[index]["ground_truth"]) for index in range(4)] == [row["ground_truth"] for row in rows]
    assert dataset[0]["ground_truth"] == canonical_json(type1["ground_truth"])
    assert dataset[1]["ground_truth"] == canonical_json(type3["ground_truth"])
    assert dataset[2]["ground_truth"] == canonical_json(type4["ground_truth"])
    assert dataset[3]["ground_truth"] == canonical_json(rubrichub_truth)
    assert dataset[0]["rubrics"] == canonical_json(type1["rubrics"])
    restored = dataset.with_transform(module._restore_rubrics)
    assert [restored[index]["rubrics"] for index in range(4)] == [row["rubrics"] for row in rows]


def test_builder_preserves_json_canonical_worker_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pipeline(monkeypatch)
    payload = {
        "actor_train": {"worker_cls": module.ACTOR_WORKER_PATH},
        "actor_infer": {"worker_cls": module.INFER_WORKER_PATH},
    }
    config = SimpleNamespace(
        actor_train=SimpleNamespace(
            worker_cls=module.ACTOR_WORKER_PATH,
            strategy_args=SimpleNamespace(strategy_name="fsdp2_train"),
        ),
        actor_infer=SimpleNamespace(
            worker_cls=module.INFER_WORKER_PATH,
            strategy_args=SimpleNamespace(strategy_name="vllm"),
        ),
        to_dict=lambda: payload,
    )
    before = canonical_json(config.to_dict())
    observed: dict[str, object] = {}

    def construct(value: object, **kwargs: object) -> object:
        observed.update(config=value, kwargs=kwargs)
        return value

    monkeypatch.setattr(module, "ResponseTrainingPipeline", construct)
    assert module.build_response_training_pipeline(config, marker=True) is config
    assert observed == {"config": config, "kwargs": {"marker": True}}
    assert canonical_json(config.to_dict()) == before
    assert config.actor_train.worker_cls == module.ACTOR_WORKER_PATH
    assert config.actor_infer.worker_cls == module.INFER_WORKER_PATH


def test_step_checkpoint_seals_promoted_manifest_and_defers_unscheduled_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_pipeline(monkeypatch)

    class Batch:
        def __init__(self) -> None:
            selected = torch.tensor([0.0, 1.0, 0.0, 1.0])
            quality = torch.tensor([0.0, 1.0, 1.0, 0.0])
            self.batch = {
                "rdan_selected_reward": selected,
                "rdan_raw_quality": quality,
                "rdan_quality_eligible": torch.ones(4, dtype=torch.bool),
            }
            self.non_tensor_batch = {"rdan_prompt_key": ["p0", "p0", "p1", "p1"]}
            self.meta_info = {}

        def __len__(self) -> int:
            return 4

    checkpoint_root = tmp_path / "checkpoints"
    pipeline = module.ResponseTrainingPipeline.__new__(module.ResponseTrainingPipeline)
    pipeline.checkpoint_root = checkpoint_root.resolve()
    pipeline.pipeline_config = SimpleNamespace(num_return_sequences_in_group=2, save_steps=20)
    pipeline.stop_after_step = 1
    pipeline._resume_path = None
    pipeline.response_config = SimpleNamespace(method="rdan_scalar")
    identity = CheckpointIdentity(
        planned_horizon=20,
        method="rdan_scalar",
        method_weight=0.5,
        resolved_config_sha256="a" * 64,
        certificate=ArtifactIdentity("cert", "b" * 64),
        data=ArtifactIdentity("merged-response-data-v1", "1" * 64),
        revisions={"code": "c" * 40, "rtt": "d" * 40, "model": "e" * 40},
        base_checkpoint_sha256="f" * 64,
        wandb={
            "entity": "RDAN-GRPO",
            "project": "rdan-grpo-qwen3-4b",
            "run_id": "run",
            "name": "name",
            "group": "group",
        },
    )
    pipeline.checkpoint_identity = identity

    def save_dcp(path: str, step: int, **kwargs: object) -> None:
        target = Path(path)
        target.mkdir(parents=True)
        (target / "rank.distcp").write_bytes(f"step-{step}".encode())

    pipeline.actor_train = SimpleNamespace(
        rdan_save_dcp=save_dcp,
        rdan_train_counters=lambda **kwargs: [
            {"rank": rank, "optimizer_steps": 2, "scheduler_steps": 2} for rank in range(2)
        ],
    )

    def remote() -> dict[str, torch.Tensor]:
        return {"cpu": torch.get_rng_state()}

    pipeline.actor_infer = SimpleNamespace(
        workers=[SimpleNamespace(rdan_save_rng=SimpleNamespace(remote=remote)) for _ in range(2)]
    )
    pipeline.scheduler = SimpleNamespace(get_scheduler_state=SimpleNamespace(remote=lambda: {"dataset_iter_count": 1}))
    monkeypatch.setattr(
        module,
        "WorkerState",
        SimpleNamespace(
            save_rng_state=lambda path, name: torch.save(torch.get_rng_state(), Path(path) / f"rng_state_{name}.pth")
        ),
    )
    result = SimpleNamespace(
        promotion_ready=True,
        metrics={"rdan/response_token_clipfrac": 0.2, "actor/grad_norm": 1.0},
        peak_memory_fraction=0.8,
    )
    observations = {
        "training_state": [{"rank": rank, "optimizer_steps": 2} for rank in range(2)],
        "memory": [{"rank": 0, "peak_bytes": 10}, {"rank": 1, "peak_bytes": 11}],
    }

    checkpoint = pipeline._save_step(1, Batch(), result, observations)

    assert checkpoint is not None and checkpoint.name == "step-000001"
    manifest = load_checkpoint(checkpoint, identity=identity)
    assert manifest["completed_step"] == 1
    assert manifest["group_diagnostics"]["response_active_group_rate"] == 1.0
    assert manifest["clipping_fraction"] == 0.2
    assert manifest["optimizer_counters"] == {"0": 2, "1": 2}
    assert manifest["peak_memory"] == {"0": 10, "1": 11}
    assert json.loads((checkpoint / module.TRAINING_STATE_ARTIFACT).read_text(encoding="utf-8")) == {
        "ranks": observations["training_state"]
    }

    pipeline.stop_after_step = 20
    deferred = pipeline._save_step(2, Batch(), result, observations)
    assert deferred is None
    assert not (checkpoint_root / "step-000002").exists()

    interleaved = Batch()
    interleaved.non_tensor_batch["rdan_prompt_key"] = ["p0", "p1", "p0", "p1"]
    with pytest.raises(RuntimeError, match="interleaved"):
        module._group_diagnostics(interleaved, 2)


def test_finish_failure_surfaces_without_repeating_optimizer_work(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pipeline(monkeypatch)
    pipeline = module.ResponseTrainingPipeline.__new__(module.ResponseTrainingPipeline)
    pipeline.completed_step = 1
    events: list[str] = []
    pipeline.scheduler = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: events.append("shutdown")))
    pipeline.tracker = SimpleNamespace(finish=lambda: (_ for _ in ()).throw(RuntimeError("finish failed")))
    pipeline._run_steps = lambda: [Path("step-000001")]

    with pytest.raises(RuntimeError, match="finish failed"):
        pipeline.run()
    assert events == ["shutdown"]

    pipeline.tracker = SimpleNamespace(finish=lambda: events.append("finish"))
    completed = pipeline.run()

    assert completed.completed_step == 1
    assert completed.checkpoints == (Path("step-000001"),)
    assert events == ["shutdown", "shutdown", "finish"]


def _constructor(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ResponseTrainingPipeline":
            return next(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"
            )
    raise AssertionError("ResponseTrainingPipeline.__init__ not found")


def _method_source(name: str) -> str:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ResponseTrainingPipeline":
            method = next(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == name)
            value = ast.get_source_segment(source, method)
            assert value is not None
            return value
    raise AssertionError(f"ResponseTrainingPipeline.{name} not found")


def _statement_index(statements: list[ast.stmt], target: str) -> int:
    for index, node in enumerate(statements):
        if isinstance(node, ast.Assign) and _name(node.targets[0]) == target:
            return index
    raise AssertionError(f"assignment not found: {target}")


def _contains_call(node: ast.AST, target: str) -> bool:
    return any(isinstance(child, ast.Call) and _name(child.func) == target for child in ast.walk(node))


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return ""


def _load_pipeline(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    ray = types.ModuleType("ray")
    ray.get = lambda value, **kwargs: value
    ray.remote = lambda value: value
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    scheduling.NodeAffinitySchedulingStrategy = object
    modules = {
        "ray": ray,
        "ray.util": types.ModuleType("ray.util"),
        "ray.util.scheduling_strategies": scheduling,
    }
    names = {
        "roll.datasets.collator": {"DataCollatorWithPaddingForPaddedKeys": object},
        "roll.datasets.dataset": {"get_dataset": lambda value: None},
        "roll.distributed.executor.cluster": {"Cluster": object},
        "roll.distributed.scheduler.generate_scheduler": {"DynamicSamplingScheduler": object},
        "roll.distributed.scheduler.protocol": {"DataProto": object},
        "roll.models.model_providers": {"default_tokenizer_provider": lambda **kwargs: None},
        "roll.pipeline.base_pipeline": {"BasePipeline": object},
        "roll.pipeline.rlvr.rlvr_pipeline": {
            "get_encode_function": lambda *args: None,
            "preprocess_dataset": lambda *args, **kwargs: None,
            "update_dataset_domain": lambda *args: None,
        },
        "roll.utils.worker_state": {"WorkerState": object},
    }
    for name, attributes in names.items():
        fake = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(fake, key, value)
        modules[name] = fake
    for name, fake in modules.items():
        monkeypatch.setitem(sys.modules, name, fake)
    path = ROOT / "src/rdan_grpo/pipeline.py"
    spec = importlib.util.spec_from_file_location("test_response_pipeline_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "torch", torch)
    return module
